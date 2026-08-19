#!/usr/bin/env python3
"""Measure interaction between train-set size and plain soft-prompt length.

All runs use the same frozen DistilBERT, manual outcome-weighted CE,
optimization hyperparameters, historical validation/test rows, and nested
within-bucket training prefixes. The heldout-200 set is never used.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean, pstdev
from types import SimpleNamespace
from typing import Any

from run_sembench_accuracy_targeted_router import evaluate
from run_sembench_qa50_softprompt_router import deduplicate
from run_sembench_train100_ratio_sweep import make_orders
from train_sembench_balanced_direct_router import load_examples

import run_sembench_outcome_pairwise_router as experiment


SIZE_COUNTS = {
    50: {
        "both_correct": 25,
        "mini_only": 23,
        "nano_only": 1,
        "both_wrong": 1,
    },
    100: {
        "both_correct": 50,
        "mini_only": 46,
        "nano_only": 3,
        "both_wrong": 1,
    },
    200: {
        "both_correct": 133,
        "mini_only": 46,
        "nano_only": 7,
        "both_wrong": 14,
    },
}

PROMPT_TOKENS = (2, 4, 8, 16)


def select_nested(
    orders: dict[str, list[dict[str, Any]]],
    counts: dict[str, int],
    seed: int,
) -> list[dict[str, Any]]:
    train = []
    for bucket, count in counts.items():
        if len(orders[bucket]) < count:
            raise ValueError(
                f"Need {count} {bucket}, found {len(orders[bucket])}"
            )
        train.extend(orders[bucket][:count])
    random.Random(seed + 999983).shuffle(train)
    expected = sum(counts.values())
    if len(train) != expected:
        raise AssertionError(f"Expected {expected} rows, found {len(train)}")
    if len({row["review_id"] for row in train}) != expected:
        raise AssertionError("Training rows are not unique")
    return train


def aggregate(
    runs: list[dict[str, Any]], field: str, split: str = "test"
) -> dict[str, float]:
    values = [float(run[split][field]) for run in runs]
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
        "--untouched-manifest",
        type=Path,
        default=root / "sembench_movie_router/untouched_200_manifest.json",
    )
    parser.add_argument("--hf-model", default="distilbert-base-uncased")
    parser.add_argument("--seeds", type=int, nargs="+", default=[505, 606, 707])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "sembench_movie_router/size_prompt_grid.json",
    )
    args = parser.parse_args()

    heldout_ids = set(json.loads(
        args.untouched_manifest.read_text(encoding="utf-8")
    )["review_ids"])
    examples = deduplicate(load_examples(args.reviews, args.cache))
    historical = [
        row for row in examples if row["review_id"] not in heldout_ids
    ]
    if len(historical) != 812:
        raise AssertionError(f"Expected 812 historical rows, found {len(historical)}")
    per_seed = {seed: make_orders(historical, seed) for seed in args.seeds}

    runs_by_config: dict[str, list[dict[str, Any]]] = {}
    for train_size, counts in SIZE_COUNTS.items():
        for prompt_tokens in PROMPT_TOKENS:
            name = f"n{train_size}_p{prompt_tokens}"
            runs_by_config[name] = []
            run_args = SimpleNamespace(
                hf_model=args.hf_model,
                prompt_tokens=prompt_tokens,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                epochs=args.epochs,
                batch_size=args.batch_size,
                max_length=args.max_length,
                margin=1.0,
            )
            for seed in args.seeds:
                orders, validation, historical_test = per_seed[seed]
                train = select_nested(orders, counts, seed)
                print(
                    f"config={name} seed={seed} counts={counts}",
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

    rows = []
    for train_size, counts in SIZE_COUNTS.items():
        for prompt_tokens in PROMPT_TOKENS:
            name = f"n{train_size}_p{prompt_tokens}"
            runs = runs_by_config[name]
            prompt_parameters = prompt_tokens * 768
            trainable_parameters = prompt_parameters + 1538
            rows.append({
                "config": name,
                "train_size": train_size,
                "counts": counts,
                "critical_fraction": counts["mini_only"] / train_size,
                "prompt_tokens": prompt_tokens,
                "prompt_parameters": prompt_parameters,
                "trainable_parameters": trainable_parameters,
                "examples_per_prompt_token": train_size / prompt_tokens,
                "examples_per_trainable_parameter": (
                    train_size / trainable_parameters
                ),
                "validation_accuracy": aggregate(
                    runs, "selected_accuracy", "validation"
                ),
                "test_accuracy": aggregate(runs, "selected_accuracy"),
                "accuracy_gap_vs_all_mini": aggregate(
                    runs, "accuracy_gap_vs_all_mini"
                ),
                "mini_saving": aggregate(
                    runs, "mini_saving_vs_all_mini"
                ),
                "critical_recall": aggregate(runs, "mini_only_recall"),
                "both_wrong_mini_rate": aggregate(
                    runs, "both_wrong_mini_rate"
                ),
                "best_epoch": {
                    "mean": mean(float(run["best_epoch"]) for run in runs),
                    "std": pstdev(float(run["best_epoch"]) for run in runs),
                },
            })

    best_by_size = {}
    for train_size in SIZE_COUNTS:
        candidates = [row for row in rows if row["train_size"] == train_size]
        best_by_size[str(train_size)] = max(
            candidates,
            key=lambda row: (
                row["test_accuracy"]["mean"],
                row["critical_recall"]["mean"],
                row["mini_saving"]["mean"],
            ),
        )["config"]

    first_test = per_seed[args.seeds[0]][2]
    result = {
        "protocol": {
            "purpose": "train-size by prompt-length interaction",
            "heldout_200_used": False,
            "historical_validation_and_test_only": True,
            "within_bucket_training_prefixes_nested": True,
            "all_non_size_and_non_length_hyperparameters_fixed": True,
            "loss": "manual outcome-weighted cross entropy",
            "threshold": "validation maximum accuracy; ties choose fewer mini calls",
            "size_counts": SIZE_COUNTS,
            "prompt_tokens": PROMPT_TOKENS,
        },
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "baselines": {
            "all_nano": evaluate(first_test, [0] * len(first_test)),
            "all_mini": evaluate(first_test, [1] * len(first_test)),
        },
        "best_by_size": best_by_size,
        "grid": rows,
        "runs": runs_by_config,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "baselines": result["baselines"],
        "best_by_size": best_by_size,
        "grid": rows,
    }, indent=2), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
