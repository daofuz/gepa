#!/usr/bin/env python3
"""Merge independently completed fine-grid seed files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


FIELDS = (
    ("validation_accuracy", "validation", "selected_accuracy"),
    ("test_accuracy", "test", "selected_accuracy"),
    ("accuracy_gap_vs_all_mini", "test", "accuracy_gap_vs_all_mini"),
    ("mini_saving", "test", "mini_saving_vs_all_mini"),
    ("critical_recall", "test", "mini_only_recall"),
    ("both_wrong_mini_rate", "test", "both_wrong_mini_rate"),
)


def summary(values: list[float]) -> dict[str, float]:
    return {"mean": mean(values), "std": pstdev(values)}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="*",
        default=[
            root / "sembench_movie_router/fine_size_prompt_grid.json",
            root / "sembench_movie_router/fine_size_prompt_grid_seed606.json",
            root / "sembench_movie_router/fine_size_prompt_grid_seed707.json",
        ],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "sembench_movie_router/fine_size_prompt_grid_3seed.json",
    )
    args = parser.parse_args()

    sources = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    runs_by_config: dict[str, dict[int, dict[str, Any]]] = {}
    for source in sources:
        for config, runs in source["runs"].items():
            target = runs_by_config.setdefault(config, {})
            for run in runs:
                target[int(run["seed"])] = run

    rows = []
    for source_row in sources[0]["grid"]:
        config = source_row["config"]
        seed_runs = runs_by_config[config]
        if sorted(seed_runs) != [505, 606, 707]:
            raise AssertionError(f"{config} has seeds {sorted(seed_runs)}")
        runs = [seed_runs[seed] for seed in (505, 606, 707)]
        row = {
            key: source_row[key]
            for key in (
                "config",
                "train_size",
                "counts",
                "critical_fraction",
                "prompt_tokens",
                "examples_per_prompt_token",
            )
        }
        row["completed_seeds"] = [505, 606, 707]
        for output_name, split, field in FIELDS:
            row[output_name] = summary([
                float(run[split][field]) for run in runs
            ])
        row["best_epoch"] = summary([
            float(run["best_epoch"]) for run in runs
        ])
        rows.append(row)

    best_by_size = {}
    for size in range(50, 201, 10):
        candidates = [row for row in rows if row["train_size"] == size]
        best_by_size[str(size)] = max(
            candidates,
            key=lambda row: (
                row["test_accuracy"]["mean"],
                row["critical_recall"]["mean"],
                row["mini_saving"]["mean"],
            ),
        )["config"]

    result = {
        "protocol": {
            **sources[0]["protocol"],
            "seeds": [505, 606, 707],
            "merged_from": [str(path) for path in args.inputs],
        },
        "baselines": sources[0]["baselines"],
        "best_by_size": best_by_size,
        "grid": rows,
        "runs": {
            config: [by_seed[seed] for seed in (505, 606, 707)]
            for config, by_seed in runs_by_config.items()
        },
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {args.output} with {len(rows)} configurations")


if __name__ == "__main__":
    main()
