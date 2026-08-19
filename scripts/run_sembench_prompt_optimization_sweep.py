#!/usr/bin/env python3
"""Develop soft-prompt length optimization on historical splits only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from types import SimpleNamespace
from typing import Any

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
    "8tok_lr1e-2": {
        "prompt_tokens": 8,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
    },
    "8tok_lr5e-3": {
        "prompt_tokens": 8,
        "learning_rate": 0.005,
        "weight_decay": 0.0,
    },
    "16tok_lr1e-2": {
        "prompt_tokens": 16,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
    },
    "16tok_lr5e-3": {
        "prompt_tokens": 16,
        "learning_rate": 0.005,
        "weight_decay": 0.0,
    },
    "16tok_lr2.5e-3": {
        "prompt_tokens": 16,
        "learning_rate": 0.0025,
        "weight_decay": 0.0,
    },
    "16tok_lr5e-3_wd1e-2": {
        "prompt_tokens": 16,
        "learning_rate": 0.005,
        "weight_decay": 0.01,
    },
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
        default=root / "sembench_movie_router/prompt_optimization_sweep.json",
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
    for name, config in CONFIGS.items():
        runs_by_config[name] = []
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
                f"config={name} seed={seed} "
                f"tokens={config['prompt_tokens']} lr={config['learning_rate']} "
                f"wd={config['weight_decay']}",
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
            "purpose": "development-only prompt length optimization",
            "heldout_200_used": False,
            "train_counts": TRAIN_COUNTS,
            "configs": CONFIGS,
            "all_other_settings_fixed": True,
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
        "runs": runs_by_config,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"baselines": result["baselines"], "ranking": ranking}, indent=2))
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
