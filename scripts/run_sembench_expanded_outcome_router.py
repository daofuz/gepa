#!/usr/bin/env python3
"""Train outcome-only routing with 400 newly collected Movie reviews.

The original per-seed 82-review test and 40-review validation sets are rebuilt
from the old 412-review pool.  Every newly collected review is train-only, so
comparisons with the 200-label experiment use exactly the same held-out rows.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import run_sembench_outcome_pairwise_router as experiment
from run_sembench_accuracy_targeted_router import make_split as make_original_split


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "sembench_movie_router/additional_400_manifest.json"


def make_expanded_split(
    examples: list[dict[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    new_ids = set(manifest["review_ids"])
    old_examples = [example for example in examples if example["review_id"] not in new_ids]
    if len(old_examples) != 412:
        raise AssertionError(f"Expected original pool of 412, found {len(old_examples)}")
    _, validation, test, _ = make_original_split(old_examples, seed)
    validation_ids = {example["review_id"] for example in validation}
    test_ids = {example["review_id"] for example in test}
    train = [
        example for example in examples
        if example["review_id"] not in validation_ids | test_ids
    ]
    train_ids = {example["review_id"] for example in train}
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise AssertionError("Review leakage across expanded split")
    if (len(train_ids), len(validation_ids), len(test_ids)) != (690, 40, 82):
        raise AssertionError(
            f"Expected expanded 690/40/82 split, got "
            f"{len(train_ids)}/{len(validation_ids)}/{len(test_ids)}"
        )
    return train, validation, test, {
        "train_counts": dict(Counter(example["bucket"] for example in train)),
        "validation_counts": dict(Counter(example["bucket"] for example in validation)),
        "test_counts": dict(Counter(example["bucket"] for example in test)),
        "train_unique": len(train_ids),
        "validation_unique": len(validation_ids),
        "test_unique": len(test_ids),
        "unique_labeled_reviews": len(train_ids) + len(validation_ids),
        "new_train_only_reviews": len(new_ids),
        "critical_labeled_reviews": sum(
            example["bucket"] == "mini_only" for example in train + validation
        ),
    }


if __name__ == "__main__":
    experiment.make_split = make_expanded_split
    experiment.main()
