#!/usr/bin/env python3
"""Sweep critical-focused 100-label mixtures without touching the fresh test.

All configurations share the same historical validation/test rows per seed.
Within each outcome bucket they also use prefixes of the same shuffled order,
so changes are driven as much as possible by the requested mixture rather than
by unrelated resampling.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from run_sembench_accuracy_targeted_router import evaluate
from run_sembench_expanded_outcome_router import make_expanded_split
from run_sembench_qa50_softprompt_router import deduplicate
from train_sembench_balanced_direct_router import load_examples

import run_sembench_outcome_pairwise_router as experiment


MIXTURES = {
    "scaled_current": {
        "both_correct": 67,
        "mini_only": 23,
        "nano_only": 3,
        "both_wrong": 7,
    },
    "critical30_bw5": {
        "both_correct": 62,
        "mini_only": 30,
        "nano_only": 3,
        "both_wrong": 5,
    },
    "critical40_bw3": {
        "both_correct": 54,
        "mini_only": 40,
        "nano_only": 3,
        "both_wrong": 3,
    },
    "critical46_bw5": {
        "both_correct": 46,
        "mini_only": 46,
        "nano_only": 3,
        "both_wrong": 5,
    },
    "critical46_bw1": {
        "both_correct": 50,
        "mini_only": 46,
        "nano_only": 3,
        "both_wrong": 1,
    },
}


def historical_examples(
    reviews: Path, cache: Path, untouched_manifest: Path
) -> list[dict[str, Any]]:
    examples = deduplicate(load_examples(reviews, cache))
    untouched_ids: set[str] = set()
    if untouched_manifest.exists():
        manifest = json.loads(untouched_manifest.read_text(encoding="utf-8"))
        untouched_ids = set(manifest["review_ids"])
    historical = [
        example for example in examples if example["review_id"] not in untouched_ids
    ]
    if len(historical) != 812:
        raise AssertionError(
            f"Expected 812 historical reviews after excluding untouched rows, "
            f"found {len(historical)}"
        )
    return historical


def make_orders(
    examples: list[dict[str, Any]], seed: int
) -> tuple[
    dict[str, list[dict[str, Any]]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    full_train, validation, test, _ = make_expanded_split(examples, seed)
    rng = random.Random(seed + 104729)
    orders = {
        bucket: [example for example in full_train if example["bucket"] == bucket]
        for bucket in ("both_correct", "mini_only", "nano_only", "both_wrong")
    }
    for values in orders.values():
        rng.shuffle(values)
    return orders, validation, test


def select_train(
    orders: dict[str, list[dict[str, Any]]], counts: dict[str, int], seed: int
) -> list[dict[str, Any]]:
    train = []
    for bucket, count in counts.items():
        if len(orders[bucket]) < count:
            raise ValueError(
                f"Need {count} {bucket} examples, found {len(orders[bucket])}"
            )
        train.extend(orders[bucket][:count])
    random.Random(seed + 999983).shuffle(train)
    if len(train) != 100 or len({example["review_id"] for example in train}) != 100:
        raise AssertionError("Each mixture must contain 100 unique reviews")
    return train


def aggregate(runs: list[dict[str, Any]], field: str) -> dict[str, float]:
    values = [float(run["test"][field]) for run in runs]
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
    parser.add_argument("--prompt-tokens", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[505, 606, 707])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "sembench_movie_router/train100_ratio_sweep.json",
    )
    args = parser.parse_args()

    examples = historical_examples(args.reviews, args.cache, args.untouched_manifest)
    per_seed = {seed: make_orders(examples, seed) for seed in args.seeds}
    runs_by_mixture: dict[str, list[dict[str, Any]]] = {}

    for name, counts in MIXTURES.items():
        runs_by_mixture[name] = []
        for seed in args.seeds:
            orders, validation, test = per_seed[seed]
            train = select_train(orders, counts, seed)
            train_ids = {example["review_id"] for example in train}
            validation_ids = {example["review_id"] for example in validation}
            test_ids = {example["review_id"] for example in test}
            if train_ids & validation_ids or train_ids & test_ids:
                raise AssertionError("Historical split leakage")
            print(f"mixture={name} seed={seed} counts={counts}", flush=True)
            run = experiment.run_one(
                train, validation, test, seed, pairwise_weight=0.0, args=args
            )
            runs_by_mixture[name].append(run)
            metrics = run["test"]
            print(
                f"  accuracy={metrics['selected_accuracy']:.4f} "
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
    summaries = {
        name: {field: aggregate(runs, field) for field in fields}
        for name, runs in runs_by_mixture.items()
    }
    ranking = sorted(
        [
            {
                "mixture": name,
                "counts": MIXTURES[name],
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
            "purpose": "exploratory 100-label outcome-mixture sweep",
            "untouched_200_used": False,
            "selection_data": "historical 40-row validation only",
            "evaluation_data": "historical 82-row test already used by prior work",
            "all_non_mixture_hyperparameters_fixed": True,
            "mixtures": MIXTURES,
        },
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "baselines": {
            "all_nano": evaluate(first_test, [0] * len(first_test)),
            "all_mini": evaluate(first_test, [1] * len(first_test)),
        },
        "ranking": ranking,
        "aggregate": summaries,
        "runs": runs_by_mixture,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"baselines": result["baselines"], "ranking": ranking}, indent=2))
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
