#!/usr/bin/env python3
"""Check whether p16 improves from more examples or merely more update steps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from types import SimpleNamespace
from typing import Any

from run_sembench_qa50_softprompt_router import deduplicate
from run_sembench_size_prompt_grid import SIZE_COUNTS, select_nested
from run_sembench_train100_ratio_sweep import make_orders
from train_sembench_balanced_direct_router import load_examples

import run_sembench_outcome_pairwise_router as experiment


MATCHED_EPOCHS = {
    50: 29,   # ceil(50 / 8) * 29 = 203 updates
    100: 16,  # ceil(100 / 8) * 16 = 208 updates
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
    parser.add_argument(
        "--grid-result",
        type=Path,
        default=root / "sembench_movie_router/size_prompt_grid.json",
    )
    parser.add_argument("--hf-model", default="distilbert-base-uncased")
    parser.add_argument("--seeds", type=int, nargs="+", default=[505, 606, 707])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "sembench_movie_router/p16_matched_steps.json",
    )
    args = parser.parse_args()

    heldout_ids = set(json.loads(
        args.untouched_manifest.read_text(encoding="utf-8")
    )["review_ids"])
    examples = deduplicate(load_examples(args.reviews, args.cache))
    historical = [
        row for row in examples if row["review_id"] not in heldout_ids
    ]
    per_seed = {seed: make_orders(historical, seed) for seed in args.seeds}

    runs_by_size: dict[str, list[dict[str, Any]]] = {}
    for train_size, epochs in MATCHED_EPOCHS.items():
        key = str(train_size)
        runs_by_size[key] = []
        run_args = SimpleNamespace(
            hf_model=args.hf_model,
            prompt_tokens=16,
            learning_rate=args.learning_rate,
            weight_decay=0.0,
            epochs=epochs,
            batch_size=args.batch_size,
            max_length=args.max_length,
            margin=1.0,
        )
        for seed in args.seeds:
            orders, validation, historical_test = per_seed[seed]
            train = select_nested(orders, SIZE_COUNTS[train_size], seed)
            print(
                f"size={train_size} seed={seed} epochs={epochs}",
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
            runs_by_size[key].append(run)
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
        "mini_saving_vs_all_mini",
        "mini_only_recall",
        "both_wrong_mini_rate",
    )
    matched = {
        size: {field: aggregate(runs, field) for field in fields}
        for size, runs in runs_by_size.items()
    }
    grid = json.loads(args.grid_result.read_text(encoding="utf-8"))
    fixed_epoch_reference = {
        str(size): next(
            row for row in grid["grid"]
            if row["train_size"] == size and row["prompt_tokens"] == 16
        )
        for size in (50, 100, 200)
    }
    result = {
        "protocol": {
            "purpose": "separate example-count and optimizer-step effects for p16",
            "heldout_200_used": False,
            "prompt_tokens": 16,
            "target_updates": "approximately 200",
            "matched_epochs": MATCHED_EPOCHS,
            "batch_size": args.batch_size,
            "limitation": (
                "More epochs also create more validation checkpoint choices, "
                "so this is an exploratory rather than perfectly isolated test."
            ),
        },
        "fixed_8epoch_reference": fixed_epoch_reference,
        "matched_step_aggregate": matched,
        "runs": runs_by_size,
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "fixed_8epoch_reference": fixed_epoch_reference,
        "matched_step_aggregate": matched,
    }, indent=2), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
