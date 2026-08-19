#!/usr/bin/env python3
"""Evaluate the locked Movie router once on a newly collected untouched set.

Manifest IDs are excluded before reconstructing the historical training and
validation splits. Thresholds and best epochs are selected only on the old
validation data. Untouched labels are used only for final reporting.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from run_sembench_accuracy_targeted_router import evaluate
from run_sembench_qa50_softprompt_router import deduplicate
from train_sembench_balanced_direct_router import load_examples

import run_sembench_critical_enriched_outcome_router as split
import run_sembench_outcome_pairwise_router as experiment


LOCKED_TRAIN_COUNTS = {
    "both_correct": 133,
    "mini_only": 46,
    "nano_only": 7,
    "both_wrong": 14,
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
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "sembench_movie_router/untouched_200_router_result.json",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    untouched_ids = set(manifest["review_ids"])
    if len(untouched_ids) != int(manifest["count"]):
        raise AssertionError("Untouched manifest contains duplicate review IDs")

    examples = deduplicate(load_examples(args.reviews, args.cache))
    untouched = [example for example in examples if example["review_id"] in untouched_ids]
    historical = [example for example in examples if example["review_id"] not in untouched_ids]
    if len(untouched) != len(untouched_ids):
        found = {example["review_id"] for example in untouched}
        raise ValueError(f"Missing {len(untouched_ids - found)} untouched model-output pairs")
    if len(historical) != 812:
        raise AssertionError(
            f"Expected exactly 812 historical labeled reviews, found {len(historical)}"
        )

    split.TRAIN_COUNTS = dict(LOCKED_TRAIN_COUNTS)
    splits = {
        seed: split.make_critical_enriched_split(historical, seed)
        for seed in args.seeds
    }
    runs = []
    for seed in args.seeds:
        train, validation, historical_test, metadata = splits[seed]
        print(
            f"seed={seed}: locked train={len(train)} validation={len(validation)} "
            f"ignored_historical_test={len(historical_test)} untouched={len(untouched)}",
            flush=True,
        )
        run = experiment.run_one(
            train, validation, untouched, seed, pairwise_weight=0.0, args=args
        )
        run["split_metadata"] = metadata
        runs.append(run)
        metrics = run["test"]
        print(
            f"  untouched accuracy={metrics['selected_accuracy']:.4f} "
            f"mini_saving={metrics['mini_saving_vs_all_mini']:.4f} "
            f"critical_recall={metrics['mini_only_recall']:.4f} "
            f"both_wrong_to_mini={metrics['both_wrong_mini_rate']:.4f}",
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
            "status": "locked before untouched labels were collected",
            "method": "critical-focused outcome-only direct soft-prompt router",
            "train_counts": LOCKED_TRAIN_COUNTS,
            "validation_count": 40,
            "historical_test_rows_used_for_training_or_threshold": 0,
            "prompt_tokens": args.prompt_tokens,
            "loss": "regret-weighted cross entropy only",
            "regret_weights": {
                "both_correct": 0.5,
                "mini_only": 6.0,
                "nano_only": 1.5,
                "both_wrong": 1.8,
            },
            "threshold": "old validation max accuracy; tie breaks toward fewer mini calls",
            "seeds": args.seeds,
            "test_tuning": "none",
        },
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "untouched": {
            "count": len(untouched),
            "bucket_counts": dict(Counter(example["bucket"] for example in untouched)),
            "review_ids": manifest["review_ids"],
        },
        "baselines": {
            "all_nano": evaluate(untouched, [0] * len(untouched)),
            "all_mini": evaluate(untouched, [1] * len(untouched)),
            "routing_oracle": evaluate(
                untouched, [int(example["target"]) for example in untouched]
            ),
        },
        "aggregate": {field: aggregate(runs, field) for field in fields},
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    compact = {
        "untouched": result["untouched"],
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
    }
    print(json.dumps(compact, indent=2), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
