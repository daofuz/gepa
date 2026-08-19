#!/usr/bin/env python3
"""Two-head correctness router with per-head inverse-frequency BCE."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import run_sembench_twohead_correctness_router as evaluation


COUNTS = {
    "nano": {"positive": 53, "negative": 47},
    "mini": {"positive": 96, "negative": 4},
}
POSITIVE_WEIGHTS = torch.tensor([
    100 / (2 * COUNTS["nano"]["positive"]),
    100 / (2 * COUNTS["mini"]["positive"]),
], dtype=torch.float)
NEGATIVE_WEIGHTS = torch.tensor([
    100 / (2 * COUNTS["nano"]["negative"]),
    100 / (2 * COUNTS["mini"]["negative"]),
], dtype=torch.float)
_ORIGINAL_BCE = evaluation.F.binary_cross_entropy_with_logits


def balanced_bce(logits, labels, *args, **kwargs):
    per_head = _ORIGINAL_BCE(logits, labels, reduction="none")
    weights = torch.where(labels > 0.5, POSITIVE_WEIGHTS, NEGATIVE_WEIGHTS)
    return (weights * per_head).sum() / weights.sum()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-tokens", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[505, 606, 707])
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "sembench_movie_router/twohead_correctness_balanced.json",
    )
    args = parser.parse_args()

    evaluation.F.binary_cross_entropy_with_logits = balanced_bce
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
    result["protocol"]["correctness_counts"] = COUNTS
    result["protocol"]["positive_weights"] = POSITIVE_WEIGHTS.tolist()
    result["protocol"]["negative_weights"] = NEGATIVE_WEIGHTS.tolist()
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Annotated balanced two-head loss in {args.output}", flush=True)


if __name__ == "__main__":
    main()
