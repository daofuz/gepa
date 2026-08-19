#!/usr/bin/env python3
"""200-row outcome router focused on mini-only rather than both-wrong noise."""

import run_sembench_critical_enriched_outcome_router as split
import run_sembench_outcome_pairwise_router as experiment


if __name__ == "__main__":
    split.TRAIN_COUNTS = {
        "both_correct": 133,
        "mini_only": 46,
        "nano_only": 7,
        "both_wrong": 14,
    }
    experiment.make_split = split.make_critical_enriched_split
    experiment.main()
