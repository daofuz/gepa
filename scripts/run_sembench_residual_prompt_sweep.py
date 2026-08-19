#!/usr/bin/env python3
"""Evaluate residual soft-prompt reparameterization on historical splits only.

The data, loss, threshold selection, and frozen DistilBERT backbone are kept
identical to the current 8-token direct-router baseline.  Only the prompt
parameterization changes:

    effective_prompt = prompt + LayerNorm(up(ReLU(down(prompt))))

This follows Residual Prompt Tuning while testing bottleneck sizes appropriate
for the small (100-label) routing dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
from transformers import AutoModel

from run_sembench_accuracy_targeted_router import evaluate
from run_sembench_qa50_softprompt_router import deduplicate
from run_sembench_train100_ratio_sweep import make_orders, select_train
from train_sembench_balanced_direct_router import load_examples

import run_sembench_outcome_pairwise_router as experiment


TRAIN_COUNTS = {
    "both_correct": 50,
    "mini_only": 46,
    "nano_only": 3,
    "both_wrong": 1,
}

CONFIGS = {
    # Low-capacity residual adapter: useful when supervision is only 100 labels.
    "residual_8tok_m32_lr5e-3": {
        "prompt_tokens": 8,
        "bottleneck": 32,
        "learning_rate": 0.005,
        "weight_decay": 0.01,
    },
    # More expressive adapter, at the same optimization setting.
    "residual_8tok_m128_lr5e-3": {
        "prompt_tokens": 8,
        "bottleneck": 128,
        "learning_rate": 0.005,
        "weight_decay": 0.01,
    },
    # Check whether the larger residual adapter merely needs a gentler update.
    "residual_8tok_m128_lr1e-3": {
        "prompt_tokens": 8,
        "bottleneck": 128,
        "learning_rate": 0.001,
        "weight_decay": 0.01,
    },
}


class ResidualSoftPromptRouter(nn.Module):
    """Frozen encoder with a trainable residual-MLP prompt and linear head."""

    def __init__(
        self, model_name: str, prompt_tokens: int, bottleneck: int
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=True)
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        hidden = int(self.encoder.config.hidden_size)
        self.prompt = nn.Parameter(torch.empty(prompt_tokens, hidden))
        nn.init.normal_(self.prompt, mean=0.0, std=0.02)
        self.down = nn.Linear(hidden, bottleneck)
        self.up = nn.Linear(bottleneck, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.classifier = nn.Linear(hidden, 2)
        self.prompt_tokens = prompt_tokens

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        embeddings = self.encoder.get_input_embeddings()(input_ids)
        projected = self.up(torch.relu(self.down(self.prompt)))
        effective_prompt = self.prompt + self.norm(projected)
        prompt = effective_prompt.unsqueeze(0).expand(
            embeddings.shape[0], -1, -1
        )
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
        return self.classifier(cls_state)


def aggregate(runs: list[dict[str, Any]], field: str) -> dict[str, float]:
    values = [float(run["test"][field]) for run in runs]
    return {"mean": mean(values), "std": pstdev(values)}


def trainable_count(model_name: str, config: dict[str, Any]) -> int:
    model = ResidualSoftPromptRouter(
        model_name, config["prompt_tokens"], config["bottleneck"]
    )
    return sum(parameter.numel() for parameter in model.parameters()
               if parameter.requires_grad)


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
        "--untouched-manifest",
        type=Path,
        default=root / "sembench_movie_router/untouched_200_manifest.json",
    )
    parser.add_argument("--hf-model", default="distilbert-base-uncased")
    parser.add_argument("--seeds", type=int, nargs="+", default=[505, 606, 707])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "sembench_movie_router/residual_prompt_sweep.json",
    )
    args = parser.parse_args()

    manifest = json.loads(args.untouched_manifest.read_text(encoding="utf-8"))
    heldout_ids = set(manifest["review_ids"])
    examples = deduplicate(load_examples(args.reviews, args.cache))
    historical = [
        example for example in examples if example["review_id"] not in heldout_ids
    ]
    if len(historical) != 812:
        raise AssertionError(f"Expected 812 historical examples, found {len(historical)}")

    per_seed = {seed: make_orders(historical, seed) for seed in args.seeds}
    runs_by_config: dict[str, list[dict[str, Any]]] = {}
    counts: dict[str, int] = {}
    original_router = experiment.SoftPromptDirectRouter
    try:
        for name, config in CONFIGS.items():
            runs_by_config[name] = []
            counts[name] = trainable_count(args.hf_model, config)
            bottleneck = config["bottleneck"]
            experiment.SoftPromptDirectRouter = (
                lambda model_name, prompt_tokens, m=bottleneck:
                ResidualSoftPromptRouter(model_name, prompt_tokens, m)
            )
            run_args = SimpleNamespace(
                hf_model=args.hf_model,
                prompt_tokens=config["prompt_tokens"],
                learning_rate=config["learning_rate"],
                weight_decay=config["weight_decay"],
                epochs=args.epochs,
                batch_size=args.batch_size,
                max_length=args.max_length,
                margin=1.0,
            )
            for seed in args.seeds:
                orders, validation, historical_test = per_seed[seed]
                train = select_train(orders, TRAIN_COUNTS, seed)
                print(
                    f"config={name} seed={seed} bottleneck={bottleneck} "
                    f"lr={config['learning_rate']} "
                    f"trainable={counts[name]}",
                    flush=True,
                )
                run = experiment.run_one(
                    train,
                    validation,
                    historical_test,
                    seed,
                    pairwise_weight=0.0,
                    args=run_args,
                )
                runs_by_config[name].append(run)
                metrics = run["test"]
                print(
                    f"  accuracy={metrics['selected_accuracy']:.4f} "
                    f"saving={metrics['mini_saving_vs_all_mini']:.4f} "
                    f"critical={metrics['mini_only_recall']:.4f} "
                    f"both_wrong={metrics['both_wrong_mini_rate']:.4f}",
                    flush=True,
                )
    finally:
        experiment.SoftPromptDirectRouter = original_router

    fields = (
        "selected_accuracy",
        "accuracy_gap_vs_all_mini",
        "mini_saving_vs_all_mini",
        "mini_only_recall",
        "both_wrong_mini_rate",
    )
    summaries = {
        name: {field: aggregate(runs, field) for field in fields}
        for name, runs in runs_by_config.items()
    }
    ranking = sorted(
        [
            {
                "config": name,
                **CONFIGS[name],
                "trainable_parameters": counts[name],
                "accuracy": summary["selected_accuracy"]["mean"],
                "accuracy_std": summary["selected_accuracy"]["std"],
                "mini_saving": summary["mini_saving_vs_all_mini"]["mean"],
                "mini_saving_std": summary["mini_saving_vs_all_mini"]["std"],
                "critical_recall": summary["mini_only_recall"]["mean"],
                "both_wrong_mini_rate": summary["both_wrong_mini_rate"]["mean"],
            }
            for name, summary in summaries.items()
        ],
        key=lambda row: (
            row["accuracy"],
            row["critical_recall"],
            row["mini_saving"],
        ),
        reverse=True,
    )
    first_test = per_seed[args.seeds[0]][2]
    result = {
        "protocol": {
            "purpose": "development-only residual soft-prompt comparison",
            "heldout_200_used": False,
            "train_counts": TRAIN_COUNTS,
            "loss": "manual outcome-weighted cross entropy",
            "threshold": "validation maximum accuracy; ties choose fewer mini calls",
            "backbone": "frozen",
            "only_architecture_changed": True,
        },
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "baselines": {
            "all_nano": evaluate(first_test, [0] * len(first_test)),
            "all_mini": evaluate(first_test, [1] * len(first_test)),
        },
        "plain_8token_reference": {
            "source": "prompt_optimization_sweep.json",
            "accuracy": 0.9349593495934959,
            "mini_saving": 0.15853658536585366,
            "critical_recall": 0.9333333333333332,
            "both_wrong_mini_rate": 1.0,
            "trainable_parameters": 7682,
        },
        "ranking": ranking,
        "aggregate": summaries,
        "runs": runs_by_config,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"baselines": result["baselines"], "ranking": ranking}, indent=2))
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
