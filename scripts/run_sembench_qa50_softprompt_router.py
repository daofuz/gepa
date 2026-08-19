#!/usr/bin/env python3
"""Run a leakage-safe SemBench Movie router matching the QA balanced-50 mix.

The computer-QA reference has 13 critical (mini-only) examples among 50
labels.  This experiment therefore fixes critical prevalence at 26%, uses a
40/10 train/validation split, and routes both mini-only and both-wrong cases
to mini.  Only the query/operator description and source review are visible
to the direct soft-prompt router.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from train_sembench_balanced_direct_router import (
    BUCKETS,
    UTILITY,
    RouterDataset,
    SoftPromptDirectRouter,
    collate,
    evaluate,
    load_examples,
)


# Exact critical prevalence of the QA balanced-50 reference: 13 / 50 = 26%.
# Movie has only four nano-only examples after the leakage-safe holdout, so
# the unavailable five nano-only slots are filled with both-correct examples.
TRAIN_COUNTS = {
    "both_correct": 15,
    "mini_only": 11,
    "nano_only": 3,
    "both_wrong": 11,
}
VAL_COUNTS = {
    "both_correct": 4,
    "mini_only": 2,
    "nano_only": 1,
    "both_wrong": 3,
}
LABEL_COUNTS = {
    bucket: TRAIN_COUNTS[bucket] + VAL_COUNTS[bucket] for bucket in BUCKETS
}
TEST_COUNTS = {
    "both_correct": 72,
    "mini_only": 5,
    "nano_only": 1,
    "both_wrong": 4,
}


def deduplicate(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one record per review ID and reject inconsistent duplicates."""
    unique: dict[str, dict[str, Any]] = {}
    for example in examples:
        review_id = example["review_id"]
        previous = unique.get(review_id)
        if previous is not None and previous["bucket"] != example["bucket"]:
            raise ValueError(f"Inconsistent duplicate review {review_id}")
        unique.setdefault(review_id, example)
    return list(unique.values())


def make_split(
    examples: list[dict[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    by_bucket = {bucket: [x for x in examples if x["bucket"] == bucket] for bucket in BUCKETS}
    for values in by_bucket.values():
        rng.shuffle(values)

    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for bucket in BUCKETS:
        values = by_bucket[bucket]
        required = TEST_COUNTS[bucket] + TRAIN_COUNTS[bucket] + VAL_COUNTS[bucket]
        if len(values) < required:
            raise ValueError(f"Need {required} unique {bucket} examples, found {len(values)}")
        cursor = 0
        test.extend(values[cursor:cursor + TEST_COUNTS[bucket]])
        cursor += TEST_COUNTS[bucket]
        train.extend(values[cursor:cursor + TRAIN_COUNTS[bucket]])
        cursor += TRAIN_COUNTS[bucket]
        validation.extend(values[cursor:cursor + VAL_COUNTS[bucket]])

    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(test)
    all_ids = [x["review_id"] for x in train + validation + test]
    if len(all_ids) != len(set(all_ids)):
        raise AssertionError("Review leakage across train, validation, and test")
    return train, validation, test


def make_loader(
    examples: list[dict[str, Any]], tokenizer: Any, args: argparse.Namespace, shuffle: bool
) -> DataLoader:
    return DataLoader(
        RouterDataset(examples, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=shuffle,
        collate_fn=lambda items: collate(items, tokenizer.pad_token_id),
    )


def probabilities(
    model: SoftPromptDirectRouter, loader: DataLoader
) -> list[float]:
    model.eval()
    result: list[float] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"], batch["attention_mask"])
            result.extend(torch.softmax(logits, dim=-1)[:, 1].tolist())
    return result


def population_weighted_utility(
    examples: list[dict[str, Any]], predictions: list[int], priors: dict[str, float]
) -> float:
    """Score a balanced validation set using the natural Movie prevalence."""
    bucket_scores: dict[str, list[float]] = {bucket: [] for bucket in BUCKETS}
    for example, prediction in zip(examples, predictions):
        route = "mini" if prediction else "nano"
        bucket_scores[example["bucket"]].append(UTILITY[(example["bucket"], route)])
    return sum(priors[bucket] * mean(bucket_scores[bucket]) for bucket in BUCKETS)


def select_threshold(
    examples: list[dict[str, Any]], probs: list[float], priors: dict[str, float]
) -> tuple[float, float, int]:
    best: tuple[float, int, float] | None = None
    for step in range(1, 100):
        threshold = step / 100
        predictions = [int(probability >= threshold) for probability in probs]
        utility = population_weighted_utility(examples, predictions, priors)
        candidate = (utility, -sum(predictions), -threshold)
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    utility, negative_calls, negative_threshold = best
    return -negative_threshold, utility, -negative_calls


def train_once(
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    test: list[dict[str, Any]],
    priors: dict[str, float],
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, local_files_only=True)
    model = SoftPromptDirectRouter(args.hf_model, args.prompt_tokens)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )
    criterion = nn.CrossEntropyLoss()
    train_loader = make_loader(train, tokenizer, args, shuffle=True)
    validation_loader = make_loader(validation, tokenizer, args, shuffle=False)
    test_loader = make_loader(test, tokenizer, args, shuffle=False)

    best_key: tuple[float, int] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_threshold = 0.5
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(batch["labels"])

        validation_probs = probabilities(model, validation_loader)
        threshold, weighted_utility, mini_calls = select_threshold(
            validation, validation_probs, priors
        )
        history.append({
            "epoch": epoch,
            "train_loss": loss_sum / len(train),
            "validation_threshold": threshold,
            "validation_population_weighted_utility": weighted_utility,
            "validation_mini_calls": mini_calls,
        })
        key = (weighted_utility, -mini_calls)
        if best_key is None or key > best_key:
            best_key = key
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_threshold = threshold

    assert best_state is not None
    model.load_state_dict(best_state)
    test_probs = probabilities(model, test_loader)
    calibrated_predictions = [int(value >= best_threshold) for value in test_probs]
    default_predictions = [int(value >= 0.5) for value in test_probs]
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "threshold": best_threshold,
        "train_counts": dict(Counter(x["bucket"] for x in train)),
        "validation_counts": dict(Counter(x["bucket"] for x in validation)),
        "test_counts": dict(Counter(x["bucket"] for x in test)),
        "train_review_ids": [x["review_id"] for x in train],
        "validation_review_ids": [x["review_id"] for x in validation],
        "test_review_ids": [x["review_id"] for x in test],
        "calibrated": evaluate(test, calibrated_predictions),
        "default_threshold": evaluate(test, default_predictions),
        "history": history,
    }


def aggregate(runs: list[dict[str, Any]], result_name: str) -> dict[str, dict[str, float]]:
    metrics = [
        "mini_rate",
        "route_accuracy",
        "selected_accuracy",
        "mean_utility",
        "mini_only_recall",
        "both_wrong_mini_rate",
        "critical_misroutes",
        "both_wrong_cheap_routes",
        "unnecessary_mini",
    ]
    summary: dict[str, dict[str, float]] = {}
    for metric in metrics:
        values = [float(run[result_name][metric]) for run in runs]
        summary[metric] = {"mean": mean(values), "std": pstdev(values)}
    return summary


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", type=Path, default=root / "sembench/files/movie/data/sf_2000/Reviews.csv")
    parser.add_argument("--cache", type=Path, default=root / "sembench_movie_router/model_outputs.json")
    parser.add_argument("--hf-model", default="distilbert-base-uncased")
    parser.add_argument("--prompt-tokens", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seeds", type=int, nargs="+", default=[505, 606, 707, 808, 909])
    parser.add_argument(
        "--output", type=Path,
        default=root / "sembench_movie_router/qa50_critical26_softprompt_direct.json",
    )
    args = parser.parse_args()

    raw_examples = load_examples(args.reviews, args.cache)
    examples = deduplicate(raw_examples)
    available_counts = Counter(x["bucket"] for x in examples)
    priors = {bucket: available_counts[bucket] / len(examples) for bucket in BUCKETS}
    runs = []
    for seed in args.seeds:
        train, validation, test = make_split(examples, seed)
        print(f"seed={seed}: training direct soft prompt", flush=True)
        runs.append(train_once(train, validation, test, priors, seed, args))
        metrics = runs[-1]["calibrated"]
        print(
            f"seed={seed}: accuracy={metrics['selected_accuracy']:.4f} "
            f"utility={metrics['mean_utility']:.4f} mini_rate={metrics['mini_rate']:.4f}",
            flush=True,
        )

    baseline_test = make_split(examples, args.seeds[0])[2]
    output = {
        "config": {
            **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "reference_split": "routing_split_half1_ablation50_balancedish_seed0.json",
            "reference_critical_count": 13,
            "reference_total_labels": 50,
            "critical_percentage": 0.26,
            "both_wrong_target": "mini",
            "router_inputs": ["semantic_operator_description", "review_text"],
            "nano_output_exposed_to_router": False,
        },
        "raw_rows": len(raw_examples),
        "unique_reviews": len(examples),
        "duplicates_removed": len(raw_examples) - len(examples),
        "available_bucket_counts": dict(available_counts),
        "natural_bucket_priors": priors,
        "requested_qa50_counts": {
            "both_correct": 14,
            "mini_only": 13,
            "nano_only": 9,
            "both_wrong": 14,
        },
        "movie_feasible_label_counts": LABEL_COUNTS,
        "movie_feasible_note": (
            "Critical remains exactly 13/50 and both-wrong remains 14/50. "
            "Only four unique nano-only examples remain after test holdout, so five "
            "nano-only slots are filled with both-correct examples without oversampling."
        ),
        "baselines_on_seed_505_test": {
            "all_nano": evaluate(baseline_test, [0] * len(baseline_test)),
            "all_mini": evaluate(baseline_test, [1] * len(baseline_test)),
            "target_oracle": evaluate(baseline_test, [int(x["target"]) for x in baseline_test]),
        },
        "aggregate_calibrated": aggregate(runs, "calibrated"),
        "aggregate_default_threshold": aggregate(runs, "default_threshold"),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "unique_reviews": output["unique_reviews"],
        "available_bucket_counts": output["available_bucket_counts"],
        "label_counts": output["movie_feasible_label_counts"],
        "aggregate_calibrated": output["aggregate_calibrated"],
        "baselines": output["baselines_on_seed_505_test"],
    }, indent=2), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
