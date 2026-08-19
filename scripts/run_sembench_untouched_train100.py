#!/usr/bin/env python3
"""Evaluate the fixed 100-label 50/46/3/1 router on held-out-200 rows."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from run_sembench_accuracy_targeted_router import evaluate
from run_sembench_qa50_softprompt_router import deduplicate
from run_sembench_train100_ratio_sweep import make_orders, select_train
from train_sembench_balanced_direct_router import load_examples

import run_sembench_outcome_pairwise_router as experiment


LOCKED_COUNTS = {
    "both_correct": 50,
    "mini_only": 46,
    "nano_only": 3,
    "both_wrong": 1,
}


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
        "--manifest",
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
        default=root / "sembench_movie_router/untouched_200_train100_result.json",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    heldout_ids = set(manifest["review_ids"])
    examples = deduplicate(load_examples(args.reviews, args.cache))
    heldout = [example for example in examples if example["review_id"] in heldout_ids]
    historical = [example for example in examples if example["review_id"] not in heldout_ids]
    if len(heldout) != 200 or len(historical) != 812:
        raise AssertionError(
            f"Expected historical/heldout = 812/200, got "
            f"{len(historical)}/{len(heldout)}"
        )

    runs = []
    for seed in args.seeds:
        orders, validation, historical_test = make_orders(historical, seed)
        train = select_train(orders, LOCKED_COUNTS, seed)
        train_ids = {example["review_id"] for example in train}
        validation_ids = {example["review_id"] for example in validation}
        if train_ids & validation_ids or train_ids & heldout_ids:
            raise AssertionError("Leakage into train or validation")
        print(
            f"seed={seed}: train=100 validation=40 "
            f"ignored_historical_test={len(historical_test)} heldout=200",
            flush=True,
        )
        run = experiment.run_one(
            train, validation, heldout, seed, pairwise_weight=0.0, args=args
        )
        runs.append(run)
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
    result = {
        "protocol": {
            "status": "post-hoc held-out comparison; not a new untouched confirmation",
            "heldout_rows_used_for_training": 0,
            "heldout_rows_used_for_threshold": 0,
            "mixture_selected_on": "historical split only",
            "train_counts": LOCKED_COUNTS,
            "threshold": "old validation max accuracy",
        },
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "heldout": {
            "count": len(heldout),
            "bucket_counts": dict(Counter(example["bucket"] for example in heldout)),
        },
        "baselines": {
            "all_nano": evaluate(heldout, [0] * len(heldout)),
            "all_mini": evaluate(heldout, [1] * len(heldout)),
        },
        "aggregate": {field: aggregate(runs, field) for field in fields},
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "heldout": result["heldout"],
        "baselines": result["baselines"],
        "aggregate": result["aggregate"],
        "per_seed": [
            {
                "seed": run["seed"],
                "best_epoch": run["best_epoch"],
                "threshold": run["threshold"],
                "test": run["test"],
            }
            for run in runs
        ],
    }, indent=2), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
