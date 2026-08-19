#!/usr/bin/env python3
"""Outcome-only router on a smaller, critical-enriched unique training set."""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

import run_sembench_outcome_pairwise_router as experiment
from run_sembench_expanded_outcome_router import make_expanded_split


TRAIN_COUNTS = {
    "both_correct": 110,
    "mini_only": 46,
    "nano_only": 7,
    "both_wrong": 37,
}


def make_critical_enriched_split(
    examples: list[dict[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    full_train, validation, test, _ = make_expanded_split(examples, seed)
    rng = random.Random(seed + 104729)
    by_bucket = {
        bucket: [example for example in full_train if example["bucket"] == bucket]
        for bucket in TRAIN_COUNTS
    }
    train: list[dict[str, Any]] = []
    for bucket, count in TRAIN_COUNTS.items():
        values = by_bucket[bucket]
        rng.shuffle(values)
        if len(values) < count:
            raise ValueError(f"Need {count} {bucket} examples, found {len(values)}")
        train.extend(values[:count])
    rng.shuffle(train)

    train_ids = {example["review_id"] for example in train}
    validation_ids = {example["review_id"] for example in validation}
    test_ids = {example["review_id"] for example in test}
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise AssertionError("Review leakage across critical-enriched split")
    if (len(train_ids), len(validation_ids), len(test_ids)) != (200, 40, 82):
        raise AssertionError("Expected 200/40/82 critical-enriched split")
    return train, validation, test, {
        "train_counts": dict(Counter(example["bucket"] for example in train)),
        "validation_counts": dict(Counter(example["bucket"] for example in validation)),
        "test_counts": dict(Counter(example["bucket"] for example in test)),
        "train_unique": len(train_ids),
        "validation_unique": len(validation_ids),
        "test_unique": len(test_ids),
        "unique_labeled_reviews": len(train_ids) + len(validation_ids),
        "critical_labeled_reviews": sum(
            example["bucket"] == "mini_only" for example in train + validation
        ),
    }


if __name__ == "__main__":
    experiment.make_split = make_critical_enriched_split
    experiment.main()
