#!/usr/bin/env python3
"""Compare natural random 100-label training with critical-enriched selection.

The heldout-200 set is excluded.  For each seed, the random and enriched
experiments share the same historical pool, validation set, and test set.
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
from run_sembench_conservative_threshold_sweep import run_one
from run_sembench_qa50_softprompt_router import deduplicate
from run_sembench_train100_ratio_sweep import make_orders
from train_sembench_balanced_direct_router import load_examples


CONFIGS = {
    "random_plain_8tok": {
        "architecture": "plain",
        "prompt_tokens": 8,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "threshold_policy": "current",
    },
    "random_residual_8tok_m128": {
        "architecture": "residual",
        "prompt_tokens": 8,
        "bottleneck": 128,
        "learning_rate": 0.005,
        "weight_decay": 0.01,
        "threshold_policy": "conservative",
    },
}

ENRICHED_REFERENCES = {
    "plain_8tok": {
        "train_counts": {
            "both_correct": 50,
            "mini_only": 46,
            "nano_only": 3,
            "both_wrong": 1,
        },
        "accuracy": 0.9349593495934959,
        "mini_saving": 0.1585365853658537,
        "critical_recall": 0.9333333333333333,
        "both_wrong_mini_rate": 1.0,
        "source": "prompt_optimization_sweep.json",
    },
    "residual_8tok_m128_conservative": {
        "train_counts": {
            "both_correct": 50,
            "mini_only": 46,
            "nano_only": 3,
            "both_wrong": 1,
        },
        "accuracy": 0.9349593495934959,
        "mini_saving": 0.17073170731707318,
        "critical_recall": 0.9333333333333333,
        "both_wrong_mini_rate": 1.0,
        "source": "conservative_threshold_sweep.json",
    },
}


def random_train(
    orders: dict[str, list[dict[str, Any]]], seed: int
) -> list[dict[str, Any]]:
    pool = [example for values in orders.values() for example in values]
    train = random.Random(seed + 15485863).sample(pool, 100)
    if len({example["review_id"] for example in train}) != 100:
        raise AssertionError("Random training sample is not unique")
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
    parser.add_argument("--seeds", type=int, nargs="+", default=[505, 606, 707])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "sembench_movie_router/random100_selection.json",
    )
    args = parser.parse_args()

    heldout_ids = set(json.loads(
        args.untouched_manifest.read_text(encoding="utf-8")
    )["review_ids"])
    examples = deduplicate(load_examples(args.reviews, args.cache))
    historical = [
        example for example in examples if example["review_id"] not in heldout_ids
    ]
    if len(historical) != 812:
        raise AssertionError(f"Expected 812 historical examples, found {len(historical)}")
    per_seed = {seed: make_orders(historical, seed) for seed in args.seeds}
    random_sets = {
        seed: random_train(per_seed[seed][0], seed) for seed in args.seeds
    }
    random_counts = {
        str(seed): dict(Counter(
            example["bucket"] for example in random_sets[seed]
        ))
        for seed in args.seeds
    }

    runs_by_config: dict[str, list[dict[str, Any]]] = {}
    for name, config in CONFIGS.items():
        runs_by_config[name] = []
        for seed in args.seeds:
            _, validation, historical_test = per_seed[seed]
            train = random_sets[seed]
            train_ids = {example["review_id"] for example in train}
            validation_ids = {example["review_id"] for example in validation}
            test_ids = {example["review_id"] for example in historical_test}
            if train_ids & validation_ids or train_ids & test_ids:
                raise AssertionError("Historical split leakage")
            print(
                f"config={name} seed={seed} "
                f"counts={random_counts[str(seed)]}",
                flush=True,
            )
            run = run_one(
                train, validation, historical_test, seed, config, args
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

    fields = (
        "selected_accuracy",
        "accuracy_gap_vs_all_mini",
        "mini_saving_vs_all_mini",
        "mini_only_recall",
        "both_wrong_mini_rate",
        "critical_misroutes",
        "unnecessary_mini",
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
            "purpose": "isolate natural-random versus critical-enriched selection",
            "heldout_200_used": False,
            "random_train_size": 100,
            "random_sample_reused_across_architectures": True,
            "validation_and_test_matched_to_enriched_references": True,
            "loss": "manual outcome-weighted cross entropy",
        },
        "random_train_counts": random_counts,
        "enriched_references": ENRICHED_REFERENCES,
        "baselines": {
            "all_nano": evaluate(first_test, [0] * len(first_test)),
            "all_mini": evaluate(first_test, [1] * len(first_test)),
        },
        "ranking": ranking,
        "aggregate": summaries,
        "runs": runs_by_config,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "random_train_counts": random_counts,
        "baselines": result["baselines"],
        "enriched_references": ENRICHED_REFERENCES,
        "ranking": ranking,
    }, indent=2), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
