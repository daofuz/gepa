#!/usr/bin/env python3
"""Double the QA-style SemBench Movie training set to 200 instances.

The doubled outcome composition is 106 both-correct, 52 mini-only, 8
nano-only, and 34 both-wrong.  Scarce outcomes are resampled only inside the
training fold; validation and test remain unique and review-disjoint.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from run_sembench_qa50_softprompt_router import TEST_COUNTS, deduplicate
from run_sembench_size100_expected_utility_sweep import (
    aggregate_runs,
    make_ranking,
    train_configuration,
    tune_thresholds,
)
from train_sembench_balanced_direct_router import BUCKETS, evaluate, load_examples


TRAIN_UNIQUE_COUNTS = {
    "both_correct": 74,
    "mini_only": 17,
    "nano_only": 3,
    "both_wrong": 14,
}
TRAIN_RESAMPLE_COUNTS = {
    "mini_only": 31,
    "nano_only": 4,
    "both_wrong": 17,
}
VAL_COUNTS = {
    "both_correct": 32,
    "mini_only": 4,
    "nano_only": 1,
    "both_wrong": 3,
}


def make_size200_split(
    examples: list[dict[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    by_bucket = {bucket: [x for x in examples if x["bucket"] == bucket] for bucket in BUCKETS}
    for values in by_bucket.values():
        rng.shuffle(values)

    train_unique: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for bucket in BUCKETS:
        values = by_bucket[bucket]
        required = TEST_COUNTS[bucket] + TRAIN_UNIQUE_COUNTS[bucket] + VAL_COUNTS[bucket]
        if len(values) < required:
            raise ValueError(f"Need {required} unique {bucket} examples, found {len(values)}")
        cursor = 0
        test.extend(values[cursor:cursor + TEST_COUNTS[bucket]])
        cursor += TEST_COUNTS[bucket]
        train_unique.extend(values[cursor:cursor + TRAIN_UNIQUE_COUNTS[bucket]])
        cursor += TRAIN_UNIQUE_COUNTS[bucket]
        validation.extend(values[cursor:cursor + VAL_COUNTS[bucket]])

    train = list(train_unique)
    resampled_ids: dict[str, list[str]] = {}
    for bucket, count in TRAIN_RESAMPLE_COUNTS.items():
        candidates = [x for x in train_unique if x["bucket"] == bucket]
        repeats = [rng.choice(candidates) for _ in range(count)]
        train.extend(repeats)
        resampled_ids[bucket] = [x["review_id"] for x in repeats]

    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(test)
    train_ids = {x["review_id"] for x in train}
    validation_ids = {x["review_id"] for x in validation}
    test_ids = {x["review_id"] for x in test}
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise AssertionError("Review leakage across train, validation, and test")
    all_instances = train + validation
    metadata = {
        "train_instance_counts": dict(Counter(x["bucket"] for x in train)),
        "train_unique_counts": dict(Counter(x["bucket"] for x in train_unique)),
        "validation_counts": dict(Counter(x["bucket"] for x in validation)),
        "test_counts": dict(Counter(x["bucket"] for x in test)),
        "train_instances": len(train),
        "train_unique_reviews": len(train_ids),
        "validation_unique_reviews": len(validation_ids),
        "total_label_instances": len(all_instances),
        "total_unique_labeled_reviews": len(train_ids) + len(validation_ids),
        "critical_instances": sum(x["bucket"] == "mini_only" for x in all_instances),
        "critical_percentage": sum(x["bucket"] == "mini_only" for x in all_instances)
        / len(all_instances),
        "resampled_train_counts": TRAIN_RESAMPLE_COUNTS,
        "resampled_train_review_ids": resampled_ids,
    }
    if metadata["total_label_instances"] != 200 or metadata["critical_instances"] != 52:
        raise AssertionError(f"Unexpected size-200 composition: {metadata}")
    return train, validation, test, metadata


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", type=Path, default=root / "sembench/files/movie/data/sf_2000/Reviews.csv")
    parser.add_argument("--cache", type=Path, default=root / "sembench_movie_router/model_outputs.json")
    parser.add_argument("--hf-model", default="distilbert-base-uncased")
    parser.add_argument("--prompt-tokens", type=int, nargs="+", default=[4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[505, 606, 707])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--decision-threshold", type=float, default=0.45)
    parser.add_argument("--target-mini-rate", type=float, default=0.25)
    parser.add_argument("--mini-rate-penalty", type=float, default=2.0)
    parser.add_argument("--mini-penalty", type=float, default=0.4)
    parser.add_argument("--selection-cost-weight", type=float, default=0.5)
    parser.add_argument(
        "--output", type=Path,
        default=root / "sembench_movie_router/size200_eu_4token_threshold_tuning_fair.json",
    )
    args = parser.parse_args()

    raw_examples = load_examples(args.reviews, args.cache)
    examples = deduplicate(raw_examples)
    available = Counter(example["bucket"] for example in examples)
    splits = {seed: make_size200_split(examples, seed) for seed in args.seeds}
    by_prompt: dict[int, list[dict[str, Any]]] = {}
    for prompt_tokens in args.prompt_tokens:
        by_prompt[prompt_tokens] = []
        for seed in args.seeds:
            train, validation, test, _ = splits[seed]
            print(f"size=200 prompt_tokens={prompt_tokens} seed={seed}: training", flush=True)
            run = train_configuration(
                train, validation, test, seed, prompt_tokens, args
            )
            by_prompt[prompt_tokens].append(run)
            metrics = run["test"]
            print(
                f"size=200 prompt_tokens={prompt_tokens} seed={seed}: "
                f"accuracy={metrics['selected_accuracy']:.4f} "
                f"utility={metrics['mean_utility']:.4f} "
                f"critical_recall={metrics['mini_only_recall']:.4f} "
                f"both_wrong_mini={metrics['both_wrong_mini_rate']:.4f} "
                f"mini_rate={metrics['mini_rate']:.4f}",
                flush=True,
            )

    first_test = splits[args.seeds[0]][2]
    output = {
        "config": {
            **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "loss": "-expected_utility + 0.4*p_mini + 2.0*(p_mini-0.25)^2",
            "both_wrong_target": "mini",
            "router_inputs": ["semantic_operator_description", "review_text"],
            "nano_output_exposed_to_router": False,
            "checkpoint_cost_saving_measure": "1 - validation_mini_rate",
        },
        "unique_reviews": len(examples),
        "available_bucket_counts": dict(available),
        "split_metadata": {str(seed): splits[seed][3] for seed in args.seeds},
        "baselines_on_seed_505_test": {
            "all_nano": evaluate(first_test, [0] * len(first_test)),
            "all_mini": evaluate(first_test, [1] * len(first_test)),
            "target_oracle": evaluate(first_test, [int(x["target"]) for x in first_test]),
        },
        "ranking_at_fixed_045": make_ranking(by_prompt),
        "threshold_tuning": tune_thresholds(by_prompt),
        "aggregate_at_fixed_045": {
            str(prompt_tokens): aggregate_runs(runs)
            for prompt_tokens, runs in by_prompt.items()
        },
        "runs": {str(prompt_tokens): runs for prompt_tokens, runs in by_prompt.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "split": output["split_metadata"][str(args.seeds[0])],
        "ranking_at_fixed_045": output["ranking_at_fixed_045"],
        "threshold_tuning": {
            key: {
                "selected_threshold": value["selected_threshold"],
                "selected_threshold_test_metrics": value["selected_threshold_test_metrics"],
            }
            for key, value in output["threshold_tuning"].items()
        },
        "baselines": output["baselines_on_seed_505_test"],
    }, indent=2), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
