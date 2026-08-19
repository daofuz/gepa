#!/usr/bin/env python3
"""Two-head soft-prompt router predicting nano and mini correctness."""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from run_sembench_accuracy_targeted_router import evaluate_scores, select_threshold
from run_sembench_qa50_softprompt_router import deduplicate
from run_sembench_train100_ratio_sweep import make_orders, select_train
from train_sembench_balanced_direct_router import evaluate, load_examples, text


TRAIN_COUNTS = {
    "both_correct": 50,
    "mini_only": 46,
    "nano_only": 3,
    "both_wrong": 1,
}


class CorrectnessDataset(Dataset):
    def __init__(self, examples: list[dict[str, Any]], tokenizer: Any, max_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        encoded = self.tokenizer(
            text(example),
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        return {
            **encoded,
            "nano_correct": float(example["nano_pred"] == example["gold"]),
            "mini_correct": float(example["mini_pred"] == example["gold"]),
        }


def collate(items: list[dict[str, Any]], pad_id: int) -> dict[str, torch.Tensor]:
    length = max(len(item["input_ids"]) for item in items)
    return {
        "input_ids": torch.tensor([
            item["input_ids"] + [pad_id] * (length - len(item["input_ids"]))
            for item in items
        ], dtype=torch.long),
        "attention_mask": torch.tensor([
            item["attention_mask"] + [0] * (length - len(item["attention_mask"]))
            for item in items
        ], dtype=torch.long),
        "labels": torch.tensor([
            [item["nano_correct"], item["mini_correct"]] for item in items
        ], dtype=torch.float),
    }


def loader(
    examples: list[dict[str, Any]],
    tokenizer: Any,
    max_length: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        CorrectnessDataset(examples, tokenizer, max_length),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda items: collate(items, tokenizer.pad_token_id),
    )


class TwoHeadCorrectnessRouter(nn.Module):
    def __init__(self, model_name: str, prompt_tokens: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=True)
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        hidden = int(self.encoder.config.hidden_size)
        self.prompt = nn.Parameter(torch.empty(prompt_tokens, hidden))
        nn.init.normal_(self.prompt, mean=0.0, std=0.02)
        self.correctness_heads = nn.Linear(hidden, 2)
        self.prompt_tokens = prompt_tokens

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        embeddings = self.encoder.get_input_embeddings()(input_ids)
        prompt = self.prompt.unsqueeze(0).expand(embeddings.shape[0], -1, -1)
        combined = torch.cat([prompt, embeddings], dim=1)
        prompt_mask = torch.ones(
            embeddings.shape[0],
            self.prompt_tokens,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        mask = torch.cat([prompt_mask, attention_mask], dim=1)
        outputs = self.encoder(inputs_embeds=combined, attention_mask=mask)
        cls_state = outputs.last_hidden_state[:, self.prompt_tokens, :]
        return self.correctness_heads(cls_state)


def correctness_probabilities(
    model: nn.Module, data_loader: DataLoader
) -> tuple[list[float], list[float]]:
    nano: list[float] = []
    mini: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            values = torch.sigmoid(
                model(batch["input_ids"], batch["attention_mask"])
            )
            nano.extend(values[:, 0].tolist())
            mini.extend(values[:, 1].tolist())
    return nano, mini


def score(nano: list[float], mini: list[float]) -> list[float]:
    return [p_mini - p_nano for p_nano, p_mini in zip(nano, mini)]


def train_one(
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    test: list[dict[str, Any]],
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, local_files_only=True)
    model = TwoHeadCorrectnessRouter(args.hf_model, args.prompt_tokens)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_loader = loader(
        train, tokenizer, args.max_length, args.batch_size, True
    )
    validation_loader = loader(
        validation, tokenizer, args.max_length, args.batch_size, False
    )
    test_loader = loader(
        test, tokenizer, args.max_length, args.batch_size, False
    )

    best: dict[str, dict[str, Any]] = {}
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = F.binary_cross_entropy_with_logits(logits, batch["labels"])
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch["labels"])

        q_nano, q_mini = correctness_probabilities(model, validation_loader)
        validation_score = score(q_nano, q_mini)
        calibrated = select_threshold(
            validation, validation_score, "max_accuracy"
        )
        direct_metrics = evaluate_scores(validation, validation_score, 0.0)
        candidates = {
            "direct_argmax": {
                "threshold": 0.0,
                "validation": direct_metrics,
            },
            "validation_calibrated": {
                "threshold": calibrated["threshold"],
                "validation": calibrated["validation_metrics"],
            },
        }
        trace = {
            "epoch": epoch,
            "train_bce": loss_sum / len(train),
            "policies": {},
        }
        for policy, candidate in candidates.items():
            metrics = candidate["validation"]
            key = (metrics["selected_accuracy"], -metrics["mini_calls"])
            trace["policies"][policy] = {
                "threshold": candidate["threshold"],
                "accuracy": metrics["selected_accuracy"],
                "mini_rate": metrics["mini_rate"],
            }
            if policy not in best or key > best[policy]["key"]:
                best[policy] = {
                    "key": key,
                    "epoch": epoch,
                    "threshold": candidate["threshold"],
                    "validation": metrics,
                    "state": copy.deepcopy({
                        name: parameter.detach().clone()
                        for name, parameter in model.named_parameters()
                        if parameter.requires_grad
                    }),
                }
        history.append(trace)

    policies = {}
    for policy, selected in best.items():
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in selected["state"]:
                    parameter.copy_(selected["state"][name])
        q_nano, q_mini = correctness_probabilities(model, test_loader)
        test_score = score(q_nano, q_mini)
        policies[policy] = {
            "best_epoch": selected["epoch"],
            "threshold": selected["threshold"],
            "validation": selected["validation"],
            "test": evaluate_scores(test, test_score, selected["threshold"]),
            "mean_q_nano": mean(q_nano),
            "mean_q_mini": mean(q_mini),
        }
    return {"seed": seed, "policies": policies, "history": history}


def aggregate(
    runs: list[dict[str, Any]], policy: str, field: str
) -> dict[str, float]:
    values = [float(run["policies"][policy]["test"][field]) for run in runs]
    return {"mean": mean(values), "std": pstdev(values)}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reviews",
        type=Path,
        default=root / "sembench/files/movie/data/sf_2000/Reviews.csv",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=root / "sembench_movie_router/model_outputs.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "sembench_movie_router/untouched_200_manifest.json",
    )
    parser.add_argument("--hf-model", default="distilbert-base-uncased")
    parser.add_argument("--prompt-tokens", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[505, 606, 707])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "sembench_movie_router/twohead_correctness_router.json",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    heldout_ids = set(manifest["review_ids"])
    examples = deduplicate(load_examples(args.reviews, args.cache))
    heldout = [example for example in examples if example["review_id"] in heldout_ids]
    historical = [example for example in examples if example["review_id"] not in heldout_ids]
    if len(heldout) != 200 or len(historical) != 812:
        raise AssertionError("Expected 812 historical and 200 held-out reviews")

    runs = []
    for seed in args.seeds:
        orders, validation, historical_test = make_orders(historical, seed)
        train = select_train(orders, TRAIN_COUNTS, seed)
        print(
            f"seed={seed}: two-head correctness BCE; "
            f"train={len(train)} validation={len(validation)} heldout={len(heldout)}",
            flush=True,
        )
        run = train_one(train, validation, heldout, seed, args)
        runs.append(run)
        for policy, values in run["policies"].items():
            metrics = values["test"]
            print(
                f"  {policy}: accuracy={metrics['selected_accuracy']:.4f} "
                f"saving={metrics['mini_saving_vs_all_mini']:.4f} "
                f"critical={metrics['mini_only_recall']:.4f} "
                f"both_wrong={metrics['both_wrong_mini_rate']:.4f}",
                flush=True,
            )

    fields = (
        "selected_accuracy",
        "accuracy_gap_vs_all_mini",
        "mini_rate",
        "mini_saving_vs_all_mini",
        "mini_only_recall",
        "both_wrong_mini_rate",
        "critical_misroutes",
        "unnecessary_mini",
    )
    policies = ("direct_argmax", "validation_calibrated")
    result = {
        "protocol": {
            "status": "post-hoc held-out comparison",
            "model": "two independent correctness heads",
            "outputs": ["P(nano correct)", "P(mini correct)"],
            "loss": "unweighted binary cross entropy, equal head weight",
            "train_counts": TRAIN_COUNTS,
            "heldout_used_for_training_or_selection": 0,
        },
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "heldout_counts": dict(Counter(example["bucket"] for example in heldout)),
        "baselines": {
            "all_nano": evaluate(heldout, [0] * len(heldout)),
            "all_mini": evaluate(heldout, [1] * len(heldout)),
        },
        "aggregate": {
            policy: {
                field: aggregate(runs, policy, field) for field in fields
            }
            for policy in policies
        },
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "baselines": result["baselines"],
        "aggregate": result["aggregate"],
    }, indent=2), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
