#!/usr/bin/env python3
"""Run the train-100 held-out evaluation with inverse route-class frequency."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import run_sembench_untouched_train100 as evaluation


NANO_COUNT = 53
MINI_COUNT = 47
TOTAL = NANO_COUNT + MINI_COUNT
NANO_WEIGHT = TOTAL / (2.0 * NANO_COUNT)
MINI_WEIGHT = TOTAL / (2.0 * MINI_COUNT)
LOSS_WEIGHTS = {
    "both_correct": NANO_WEIGHT,
    "mini_only": MINI_WEIGHT,
    "nano_only": NANO_WEIGHT,
    "both_wrong": MINI_WEIGHT,
}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[505, 606, 707])
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    output = args.output or (
        root
        / "sembench_movie_router"
        / f"untouched_200_train100_{args.prompt_tokens}token_inverse_frequency.json"
    )

    evaluation.experiment.REGRET_WEIGHTS = dict(LOSS_WEIGHTS)
    sys.argv = [
        sys.argv[0],
        "--prompt-tokens",
        str(args.prompt_tokens),
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--output",
        str(output),
    ]
    evaluation.main()

    result = json.loads(output.read_text(encoding="utf-8"))
    result["protocol"]["loss"] = "inverse-frequency route-class weighted cross entropy"
    result["protocol"]["route_class_counts"] = {
        "nano": NANO_COUNT,
        "mini": MINI_COUNT,
    }
    result["protocol"]["loss_weights"] = {
        "nano": NANO_WEIGHT,
        "mini": MINI_WEIGHT,
        "outcome_mapping": LOSS_WEIGHTS,
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"Annotated inverse-frequency weights in {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
