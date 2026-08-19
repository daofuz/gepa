#!/usr/bin/env python3
"""Post-hoc heldout-200 check for naturally random 100-label training."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from run_sembench_accuracy_targeted_router import evaluate
from run_sembench_conservative_threshold_sweep import run_one
from run_sembench_qa50_softprompt_router import deduplicate
from run_sembench_random100_selection import CONFIGS, random_train
from run_sembench_train100_ratio_sweep import make_orders
from train_sembench_balanced_direct_router import load_examples


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
    parser.add_argument("--seeds", type=int, nargs="+", default=[505, 606, 707])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "sembench_movie_router/random100_heldout200.json",
    )
    args = parser.parse_args()

    heldout_ids = set(json.loads(
        args.manifest.read_text(encoding="utf-8")
    )["review_ids"])
    examples = deduplicate(load_examples(args.reviews, args.cache))
    heldout = [example for example in examples if example["review_id"] in heldout_ids]
    historical = [
        example for example in examples if example["review_id"] not in heldout_ids
    ]
    if len(heldout) != 200 or len(historical) != 812:
        raise AssertionError(
            f"Expected historical/heldout = 812/200, got "
            f"{len(historical)}/{len(heldout)}"
        )
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
            if train_ids & validation_ids or train_ids & heldout_ids:
                raise AssertionError("Leakage into train, validation, or heldout")
            print(
                f"config={name} seed={seed} "
                f"counts={random_counts[str(seed)]} "
                f"ignored_historical_test={len(historical_test)}",
                flush=True,
            )
            run = run_one(train, validation, heldout, seed, config, args)
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
    result = {
        "protocol": {
            "status": "post-hoc heldout comparison; heldout already consumed",
            "heldout_rows_used_for_training": 0,
            "heldout_rows_used_for_threshold": 0,
            "random_train_size": 100,
            "random_sample_reused_across_architectures": True,
        },
        "random_train_counts": random_counts,
        "heldout": {
            "count": len(heldout),
            "bucket_counts": dict(Counter(
                example["bucket"] for example in heldout
            )),
        },
        "baselines": {
            "all_nano": evaluate(heldout, [0] * len(heldout)),
            "all_mini": evaluate(heldout, [1] * len(heldout)),
        },
        "ranking": ranking,
        "aggregate": summaries,
        "runs": runs_by_config,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "random_train_counts": random_counts,
        "heldout": result["heldout"],
        "baselines": result["baselines"],
        "ranking": ranking,
    }, indent=2), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
