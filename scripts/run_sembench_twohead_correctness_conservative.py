#!/usr/bin/env python3
"""Balanced two-head router with conservative validation calibration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import run_sembench_twohead_correctness_router as evaluation
import run_sembench_twohead_correctness_balanced as balanced
from train_sembench_balanced_direct_router import evaluate


def threshold_candidates(scores):
    unique = sorted(set(float(value) for value in scores))
    values = [min(unique) - 1e-6, max(unique) + 1e-6, 0.0]
    values.extend(unique)
    values.extend((a + b) / 2.0 for a, b in zip(unique, unique[1:]))
    return sorted(set(values))


def conservative_threshold(examples, scores, policy):
    rows = []
    for threshold in threshold_candidates(scores):
        metrics = evaluate(
            examples, [int(value >= threshold) for value in scores]
        )
        rows.append({"threshold": threshold, **metrics})
    feasible = [
        row for row in rows
        if row["mini_only_recall"] == 1.0
        and row["both_wrong_mini_rate"] == 1.0
    ]
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
        "validation_all_mini_accuracy": evaluate(
            examples, [1] * len(examples)
        )["selected_accuracy"],
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-tokens", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[505, 606, 707])
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            root
            / "sembench_movie_router"
            / "twohead_correctness_balanced_conservative.json"
        ),
    )
    args = parser.parse_args()

    evaluation.F.binary_cross_entropy_with_logits = balanced.balanced_bce
    evaluation.select_threshold = conservative_threshold
    sys.argv = [
        sys.argv[0],
        "--prompt-tokens",
        str(args.prompt_tokens),
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--output",
        str(args.output),
    ]
    evaluation.main()

    result = json.loads(args.output.read_text(encoding="utf-8"))
    result["protocol"]["loss"] = (
        "per-head inverse-frequency balanced binary cross entropy"
    )
    result["protocol"]["threshold"] = (
        "validation mini-only recall=100% and both-wrong-to-mini=100%; "
        "then max answer accuracy and fewer mini calls"
    )
    result["protocol"]["correctness_counts"] = balanced.COUNTS
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Annotated conservative two-head protocol in {args.output}", flush=True)


if __name__ == "__main__":
    main()
