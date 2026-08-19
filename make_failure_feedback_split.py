#!/usr/bin/env python3
"""Create a fixed test split plus failure-feedback train/val split."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from optimize_ollama_router_gepa import load_examples, outcome_bucket


BUCKETS = ("mini_only", "nano_only", "both_correct", "both_wrong")


def parse_counts(raw: str) -> dict[str, int]:
    counts = {name: 0 for name in BUCKETS}
    for part in raw.split(","):
        if not part.strip():
            continue
        name, value = part.split("=", 1)
        name = name.strip()
        if name not in counts:
            raise ValueError(f"Unknown bucket {name!r}; expected one of {BUCKETS}.")
        counts[name] = int(value.strip())
    return counts


def take(ids: list[int], count: int, label: str) -> list[int]:
    if count > len(ids):
        raise ValueError(f"Requested {count} ids for {label}, but only {len(ids)} available.")
    selected = ids[:count]
    del ids[:count]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mini-file", default="computer science_result_mini.json")
    parser.add_argument("--nano-file", default="computer science_result_nano.json")
    parser.add_argument(
        "--failure-file",
        default="mini_correct_nano_wrong_routed_nano_testdata_pareto_ollama.json",
        help="JSON produced by filter_mini_correct_routed_nano.py --require-nano-wrong.",
    )
    parser.add_argument(
        "--test-counts",
        default="mini_only=10,nano_only=3,both_correct=50,both_wrong=20",
    )
    parser.add_argument(
        "--train-counts",
        default="mini_only=25,nano_only=10,both_correct=10,both_wrong=5",
    )
    parser.add_argument(
        "--val-counts",
        default="mini_only=10,nano_only=3,both_correct=17,both_wrong=20",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="routing_split_failure_feedback_fixed_test.json")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    examples = load_examples(Path(args.mini_file), Path(args.nano_file))
    example_by_id = {example.question_id: example for example in examples}

    buckets: dict[str, list[int]] = {name: [] for name in BUCKETS}
    for example in examples:
        buckets[outcome_bucket(example)].append(example.question_id)
    for ids in buckets.values():
        rng.shuffle(ids)

    failure_payload = json.loads(Path(args.failure_file).read_text(encoding="utf-8"))
    failure_ids = [
        int(row["question_id"])
        for row in failure_payload.get("rows", [])
        if int(row["question_id"]) in example_by_id
    ]
    failure_ids = sorted(set(failure_ids))
    failure_by_bucket: dict[str, list[int]] = {name: [] for name in BUCKETS}
    for question_id in failure_ids:
        failure_by_bucket[outcome_bucket(example_by_id[question_id])].append(question_id)

    test_counts = parse_counts(args.test_counts)
    train_counts = parse_counts(args.train_counts)
    val_counts = parse_counts(args.val_counts)

    # Hold failure-feedback ids out of the fixed test so they can be used for training.
    test_pool = {
        name: [question_id for question_id in ids if question_id not in failure_ids]
        for name, ids in buckets.items()
    }
    test_ids: list[int] = []
    for name, count in test_counts.items():
        test_ids.extend(take(test_pool[name], count, f"test/{name}"))

    excluded = set(test_ids)
    train_val_pool = {
        name: [question_id for question_id in ids if question_id not in excluded]
        for name, ids in buckets.items()
    }

    train_ids: list[int] = []
    mini_failures = [question_id for question_id in failure_ids if question_id in train_val_pool["mini_only"]]
    train_mini_target = train_counts["mini_only"]
    train_failure_ids = mini_failures[: min(train_mini_target, len(mini_failures))]
    train_ids.extend(train_failure_ids)
    train_val_pool["mini_only"] = [
        question_id for question_id in train_val_pool["mini_only"] if question_id not in train_failure_ids
    ]
    remaining_train_counts = dict(train_counts)
    remaining_train_counts["mini_only"] -= len(train_failure_ids)
    for name, count in remaining_train_counts.items():
        train_ids.extend(take(train_val_pool[name], count, f"train/{name}"))

    val_ids: list[int] = []
    for name, count in val_counts.items():
        val_ids.extend(take(train_val_pool[name], count, f"validation/{name}"))

    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    rng.shuffle(test_ids)

    def counts_for(ids: list[int]) -> dict[str, int]:
        counts = {name: 0 for name in BUCKETS}
        for question_id in ids:
            counts[outcome_bucket(example_by_id[question_id])] += 1
        return counts

    payload: dict[str, Any] = {
        "description": (
            "Fixed hard test split plus 50-example failure-feedback training split. "
            "The test set is selected first and is not the remainder after train/val."
        ),
        "seed": args.seed,
        "source_failure_file": args.failure_file,
        "requested_counts": {
            "test": test_counts,
            "train": train_counts,
            "validation": val_counts,
        },
        "failure_feedback_ids": failure_ids,
        "failure_feedback_ids_used_in_train": sorted(set(train_failure_ids)),
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "counts": {
            "train": counts_for(train_ids),
            "validation": counts_for(val_ids),
            "test": counts_for(test_ids),
        },
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k.endswith("ids") is False}, indent=2))


if __name__ == "__main__":
    main()
