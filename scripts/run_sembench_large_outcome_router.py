#!/usr/bin/env python3
"""Use every non-test Movie review for outcome-only direct routing."""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

import run_sembench_outcome_pairwise_router as experiment
from run_sembench_accuracy_targeted_router import VALIDATION_COUNTS
from run_sembench_critical_pairwise_router import critical_pairwise_loss
from run_sembench_qa50_softprompt_router import TEST_COUNTS
from train_sembench_balanced_direct_router import BUCKETS


def make_large_split(
    examples: list[dict[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Create 290 train + 40 validation + 82 test from all 412 unique rows."""
    rng = random.Random(seed)
    by_bucket = {
        bucket: [example for example in examples if example["bucket"] == bucket]
        for bucket in BUCKETS
    }
    for values in by_bucket.values():
        rng.shuffle(values)

    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for bucket in BUCKETS:
        values = by_bucket[bucket]
        test_count = TEST_COUNTS[bucket]
        validation_count = VALIDATION_COUNTS[bucket]
        test.extend(values[:test_count])
        validation.extend(values[test_count : test_count + validation_count])
        train.extend(values[test_count + validation_count :])

    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(test)
    train_ids = {example["review_id"] for example in train}
    validation_ids = {example["review_id"] for example in validation}
    test_ids = {example["review_id"] for example in test}
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise AssertionError("Review leakage across train, validation, and test")
    if (len(train_ids), len(validation_ids), len(test_ids)) != (290, 40, 82):
        raise AssertionError("Expected a complete 290/40/82 partition")
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
    experiment.make_split = make_large_split
    experiment.outcome_pairwise_loss = critical_pairwise_loss
    experiment.main()
