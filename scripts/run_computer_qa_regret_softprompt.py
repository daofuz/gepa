#!/usr/bin/env python3
"""Transfer the SemBench regret-weighted direct router to Computer QA."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from optimize_ollama_router_gepa import load_examples, outcome_bucket  # noqa: E402
from train_softprompt_router import (  # noqa: E402
    ROUTE_TO_LABEL,
    RoutingDataset,
    SoftPromptRouter,
    collate_batch,
    summarize_routes,
)


REGRET_WEIGHTS = {
    "mini_only": 6.0,
    "both_wrong": 1.8,
    "nano_only": 1.5,
    "both_correct": 0.5,
}


def resolve_ids(examples: list[Any], ids: list[int]) -> list[Any]:
    by_id = {example.question_id: example for example in examples}
    return [by_id[int(question_id)] for question_id in ids]


def make_loader(
    examples: list[Any], tokenizer: Any, args: argparse.Namespace, shuffle: bool
) -> DataLoader:
    dataset = RoutingDataset(
        examples,
        tokenizer,
        max_length=args.max_length,
        score_mode="risk_averse_utility",
        include_nano_answer=False,
        nano_response_max_chars=0,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_batch(batch, int(tokenizer.pad_token_id)),
    )


def probabilities(model: SoftPromptRouter, loader: DataLoader) -> list[float]:
    model.eval()
    values: list[float] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"], batch["attention_mask"])
            values.extend(torch.softmax(logits, dim=-1)[:, ROUTE_TO_LABEL["mini"]].tolist())
    return values


def routes(scores: list[float], threshold: float) -> list[str]:
    return ["mini" if score >= threshold else "nano" for score in scores]


def threshold_candidates(scores: list[float]) -> list[float]:
    unique = sorted(set(float(score) for score in scores))
    candidates = [0.0, 1.000001, *unique]
    candidates.extend((left + right) / 2.0 for left, right in zip(unique, unique[1:]))
    return sorted(set(candidates))


def select_max_accuracy_threshold(
    examples: list[Any], scores: list[float]
) -> tuple[float, dict[str, Any]]:
    rows = []
    for threshold in threshold_candidates(scores):
        summary = summarize_routes(examples, routes(scores, threshold), "risk_averse_utility")
        rows.append((threshold, summary))
    return max(
        rows,
        key=lambda row: (
            row[1]["final_accuracy"],
            -row[1]["mini_routes"],
            row[0],
        ),
    )


def compact_summary(examples: list[Any], route_names: list[str]) -> dict[str, Any]:
    summary = summarize_routes(examples, route_names, "risk_averse_utility")
    bucket_counts = Counter(outcome_bucket(example) for example in examples)
    critical_total = bucket_counts["mini_only"]
    both_wrong_total = bucket_counts["both_wrong"]
    correct_escalations = summary["failure_counts"].get("correct_escalation", 0)
    both_wrong_mini = summary["failure_counts"].get("both_failed_expensive", 0)
    return {
        "accuracy": summary["final_accuracy"],
        "cost_saving_rate_vs_all_mini": summary["cost_saving_rate_vs_all_mini"],
        "mini_rate": summary["mini_routes"] / max(len(examples), 1),
        "mini_routes": summary["mini_routes"],
        "nano_routes": summary["nano_routes"],
        "critical_recall": correct_escalations / max(critical_total, 1),
        "critical_misroutes": summary["failure_counts"].get("critical_misroute", 0),
        "both_wrong_mini_rate": both_wrong_mini / max(both_wrong_total, 1),
        "failure_counts": summary["failure_counts"],
    }


def train_once(
    train: list[Any],
    validation: list[Any],
    test: list[Any],
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_model, local_files_only=not args.allow_download
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    model = SoftPromptRouter(
        args.hf_model,
        soft_prompt_tokens=args.prompt_tokens,
        local_files_only=not args.allow_download,
        train_backbone=False,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_loader = make_loader(train, tokenizer, args, shuffle=True)
    validation_loader = make_loader(validation, tokenizer, args, shuffle=False)
    test_loader = make_loader(test, tokenizer, args, shuffle=False)

    best_key: tuple[float, int] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_threshold = 0.5
    best_validation: dict[str, Any] | None = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        item_count = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], batch["attention_mask"])
            labels = batch["labels"]
            per_example = F.cross_entropy(logits, labels, reduction="none")
            indices = batch["indices"].tolist()
            weights = torch.tensor(
                [REGRET_WEIGHTS[outcome_bucket(train[index])] for index in indices],
                dtype=per_example.dtype,
            )
            loss = (weights * per_example).sum() / weights.sum()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(labels)
            item_count += len(labels)

        validation_scores = probabilities(model, validation_loader)
        threshold, validation_summary = select_max_accuracy_threshold(
            validation, validation_scores
        )
        key = (validation_summary["final_accuracy"], -validation_summary["mini_routes"])
        history.append({
            "epoch": epoch,
            "train_loss": loss_sum / max(item_count, 1),
            "threshold": threshold,
            "validation_accuracy": validation_summary["final_accuracy"],
            "validation_mini_routes": validation_summary["mini_routes"],
        })
        if best_key is None or key > best_key:
            best_key = key
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_threshold = threshold
            best_validation = compact_summary(
                validation, routes(validation_scores, threshold)
            )

    assert best_state is not None and best_validation is not None
    model.load_state_dict(best_state)
    test_scores = probabilities(model, test_loader)
    return {
        "seed": seed,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "best_epoch": best_epoch,
        "threshold": best_threshold,
        "validation": best_validation,
        "test": compact_summary(test, routes(test_scores, best_threshold)),
        "history": history,
    }


def aggregate(runs: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    keys = [
        "accuracy",
        "cost_saving_rate_vs_all_mini",
        "mini_rate",
        "critical_recall",
        "critical_misroutes",
        "both_wrong_mini_rate",
    ]
    return {
        key: {
            "mean": mean(float(run["test"][key]) for run in runs),
            "std": pstdev(float(run["test"][key]) for run in runs),
        }
        for key in keys
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-file",
        type=Path,
        default=ROOT / "routing_split_softprompt_threshold_balanced16.json",
    )
    parser.add_argument("--mini-file", type=Path, default=ROOT / "computer science_result_mini.json")
    parser.add_argument("--nano-file", type=Path, default=ROOT / "computer science_result_nano.json")
    parser.add_argument("--hf-model", default="distilbert-base-uncased")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--prompt-tokens", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[505, 606, 707])
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/computer_qa_regret_softprompt_clean.json",
    )
    args = parser.parse_args()

    examples = load_examples(
        args.mini_file,
        args.nano_file,
        mini_cost_model="gpt-4.1-mini",
        nano_cost_model="gpt-4.1-nano",
    )
    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    train = resolve_ids(examples, split["train_ids"])
    validation = resolve_ids(examples, split["val_ids"])
    test = resolve_ids(examples, split["test_ids"])
    ids = [{example.question_id for example in part} for part in (train, validation, test)]
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
        raise AssertionError("Train, validation, and test IDs must be disjoint")

    runs = []
    for seed in args.seeds:
        print(f"seed={seed}: regret-weighted direct router", flush=True)
        run = train_once(train, validation, test, seed, args)
        runs.append(run)
        print(
            f"  accuracy={run['test']['accuracy']:.4f} "
            f"mini_rate={run['test']['mini_rate']:.4f} "
            f"critical_recall={run['test']['critical_recall']:.4f}",
            flush=True,
        )

    all_mini = compact_summary(test, ["mini"] * len(test))
    all_nano = compact_summary(test, ["nano"] * len(test))
    output = {
        "config": {
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "loss": "sum_i(weight[outcome_i] * CE_i) / sum_i(weight[outcome_i])",
            "regret_weights": REGRET_WEIGHTS,
            "threshold_policy": "validation max answer accuracy; ties use fewer mini calls",
            "router_input": "question, options, category, and source only",
            "nano_answer_exposed": False,
            "test_used_for_selection": False,
        },
        "split": {
            "train_count": len(train),
            "validation_count": len(validation),
            "test_count": len(test),
            "train_outcomes": dict(Counter(outcome_bucket(x) for x in train)),
            "validation_outcomes": dict(Counter(outcome_bucket(x) for x in validation)),
            "test_outcomes": dict(Counter(outcome_bucket(x) for x in test)),
        },
        "baselines": {"all_mini": all_mini, "all_nano": all_nano},
        "aggregate": aggregate(runs),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"baselines": output["baselines"], "aggregate": output["aggregate"]}, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
