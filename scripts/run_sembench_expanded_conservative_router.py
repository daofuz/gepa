#!/usr/bin/env python3
"""Expanded outcome-only router with conservative validation calibration."""

from __future__ import annotations

from typing import Any

import run_sembench_outcome_pairwise_router as experiment
from run_sembench_accuracy_targeted_router import threshold_candidates
from run_sembench_expanded_outcome_router import make_expanded_split
from train_sembench_balanced_direct_router import evaluate


def select_conservative_threshold(
    examples: list[dict[str, Any]], scores: list[float], policy: str
) -> dict[str, Any]:
    del policy
    all_mini_accuracy = evaluate(examples, [1] * len(examples))["selected_accuracy"]
    rows = []
    for threshold in threshold_candidates(scores):
        predictions = [int(score >= threshold) for score in scores]
        rows.append({"threshold": threshold, **evaluate(examples, predictions)})
    feasible = [
        row for row in rows
        if row["mini_only_recall"] >= 1.0 - 1e-12
        and row["both_wrong_mini_rate"] >= 1.0 - 1e-12
    ]
    if not feasible:
        raise AssertionError("The all-mini threshold must satisfy conservative constraints")
    selected = max(
        feasible,
        key=lambda row: (
            row["selected_accuracy"],
            -row["mini_calls"],
            row["threshold"],
        ),
    )
    return {
        "threshold": selected["threshold"],
        "validation_metrics": {
            key: value for key, value in selected.items() if key != "threshold"
        },
        "validation_all_mini_accuracy": all_mini_accuracy,
    }


if __name__ == "__main__":
    experiment.make_split = make_expanded_split
    experiment.select_threshold = select_conservative_threshold
    experiment.main()
