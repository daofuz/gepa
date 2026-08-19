#!/usr/bin/env python3
"""Run a resumable fine-grained train-size x soft-prompt-length sweep.

The sweep preserves the outcome-balanced selection path used by the original
50/100/200 experiment. Training sizes increase by 10 and soft-prompt lengths
increase by 2. Identical runs from the coarse sweep are reused. The untouched
200-example test set is never used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from types import SimpleNamespace
from typing import Any

from run_sembench_accuracy_targeted_router import evaluate
from run_sembench_qa50_softprompt_router import deduplicate
from run_sembench_size_prompt_grid import aggregate, select_nested
from run_sembench_train100_ratio_sweep import make_orders
from train_sembench_balanced_direct_router import load_examples

import run_sembench_outcome_pairwise_router as experiment


def counts_for_size(size: int) -> dict[str, int]:
    """Interpolate the original nested 50/100/200 outcome compositions."""
    if not 50 <= size <= 200 or size % 10:
        raise ValueError("size must be a multiple of 10 from 50 through 200")

    mini_only = min(round(0.46 * size), 46)
    if size <= 100:
        nano_only = round(1 + (size - 50) * (2 / 50))
        both_wrong = 1
    else:
        nano_only = round(3 + (size - 100) * (4 / 100))
        both_wrong = round(1 + (size - 100) * (13 / 100))
    both_correct = size - mini_only - nano_only - both_wrong
    counts = {
        "both_correct": both_correct,
        "mini_only": mini_only,
        "nano_only": nano_only,
        "both_wrong": both_wrong,
    }
    if any(value < 0 for value in counts.values()):
        raise AssertionError(f"Invalid counts for size={size}: {counts}")
    return counts


def save_result(
    output: Path,
    args: argparse.Namespace,
    runs_by_config: dict[str, list[dict[str, Any]]],
    per_seed: dict[int, tuple[Any, Any, Any]],
) -> None:
    rows: list[dict[str, Any]] = []
    for size in args.sizes:
        counts = counts_for_size(size)
        for prompt_tokens in args.prompt_tokens:
            name = f"n{size}_p{prompt_tokens}"
            runs = runs_by_config.get(name, [])
            if not runs:
                continue
            rows.append({
                "config": name,
                "train_size": size,
                "counts": counts,
                "critical_fraction": counts["mini_only"] / size,
                "prompt_tokens": prompt_tokens,
                "examples_per_prompt_token": size / prompt_tokens,
                "completed_seeds": [int(run["seed"]) for run in runs],
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

    best_by_size: dict[str, str] = {}
    for size in args.sizes:
        candidates = [row for row in rows if row["train_size"] == size]
        if candidates:
            best_by_size[str(size)] = max(
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
            "purpose": "fine train-size by prompt-length interaction",
            "heldout_200_used": False,
            "historical_validation_and_test_only": True,
            "within_bucket_training_prefixes_nested": True,
            "size_step": 10,
            "prompt_token_step": 2,
            "size_counts": {
                str(size): counts_for_size(size) for size in args.sizes
            },
            "selection_note": (
                "Counts interpolate the original n50, n100, and n200 "
                "compositions; mini-only reaches the historical cap of 46."
            ),
            "loss": "manual outcome-weighted cross entropy",
            "threshold": (
                "validation maximum accuracy; ties choose fewer mini calls"
            ),
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
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")


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
    parser.add_argument(
        "--coarse-result",
        type=Path,
        default=root / "sembench_movie_router/size_prompt_grid.json",
    )
    parser.add_argument("--hf-model", default="distilbert-base-uncased")
    parser.add_argument("--seeds", type=int, nargs="+", default=[505])
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=list(range(50, 201, 10))
    )
    parser.add_argument(
        "--prompt-tokens", type=int, nargs="+", default=list(range(2, 17, 2))
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "sembench_movie_router/fine_size_prompt_grid.json",
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
        raise AssertionError(
            f"Expected 812 historical rows, found {len(historical)}"
        )
    per_seed = {seed: make_orders(historical, seed) for seed in args.seeds}

    runs_by_config: dict[str, list[dict[str, Any]]] = {}
    if args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        runs_by_config = previous.get("runs", {})

    if args.coarse_result.exists():
        coarse = json.loads(args.coarse_result.read_text(encoding="utf-8"))
        for name, coarse_runs in coarse.get("runs", {}).items():
            existing = {
                int(run["seed"]): run
                for run in runs_by_config.get(name, [])
            }
            for run in coarse_runs:
                seed = int(run["seed"])
                if seed in args.seeds:
                    existing.setdefault(seed, run)
            runs_by_config[name] = [
                existing[seed] for seed in args.seeds if seed in existing
            ]

    for size in args.sizes:
        counts = counts_for_size(size)
        for prompt_tokens in args.prompt_tokens:
            name = f"n{size}_p{prompt_tokens}"
            existing = {
                int(run["seed"]): run
                for run in runs_by_config.get(name, [])
            }
            for seed in args.seeds:
                if seed in existing:
                    print(f"reuse config={name} seed={seed}", flush=True)
                    continue
                orders, validation, historical_test = per_seed[seed]
                train = select_nested(orders, counts, seed)
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
                print(
                    f"train config={name} seed={seed} counts={counts}",
                    flush=True,
                )
                existing[seed] = experiment.run_one(
                    train,
                    validation,
                    historical_test,
                    seed,
                    pairwise_weight=0.0,
                    args=run_args,
                )
                runs_by_config[name] = [
                    existing[s] for s in args.seeds if s in existing
                ]
                save_result(args.output, args, runs_by_config, per_seed)
                metrics = existing[seed]["test"]
                print(
                    f"  accuracy={metrics['selected_accuracy']:.4f} "
                    f"saving={metrics['mini_saving_vs_all_mini']:.4f} "
                    f"critical={metrics['mini_only_recall']:.4f}",
                    flush=True,
                )

            runs_by_config[name] = [
                existing[s] for s in args.seeds if s in existing
            ]

    save_result(args.output, args, runs_by_config, per_seed)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
