#!/usr/bin/env python3
"""Sweep direct soft-prompt length with 100 balanced training instances.

The training distribution keeps the computer-QA critical prevalence at 26%.
Because only 21 unique critical Movie examples remain after the fixed test
holdout, five training examples are resampled within the training fold.  No
review ID crosses train, validation, or test boundaries.
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

from run_sembench_qa50_softprompt_router import (
    TEST_COUNTS,
    aggregate,
    deduplicate,
    make_loader,
    population_weighted_utility,
    probabilities,
    select_threshold,
)
from train_sembench_balanced_direct_router import (
    BUCKETS,
    SoftPromptDirectRouter,
    evaluate,
    load_examples,
)


# 80 training instances + 20 validation examples = 100.  Critical is 26%.
# The five repeated critical examples occur only in the training loader.
TRAIN_UNIQUE_COUNTS = {
    "both_correct": 41,
    "mini_only": 17,
    "nano_only": 3,
    "both_wrong": 14,
}
TRAIN_RESAMPLE_COUNTS = {"mini_only": 5}
VAL_COUNTS = {
    "both_correct": 12,
    "mini_only": 4,
    "nano_only": 1,
    "both_wrong": 3,
}


def make_size100_split(
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
    resampled_ids: list[str] = []
    for bucket, count in TRAIN_RESAMPLE_COUNTS.items():
        candidates = [x for x in train_unique if x["bucket"] == bucket]
        repeats = rng.sample(candidates, count)
        train.extend(repeats)
        resampled_ids.extend(x["review_id"] for x in repeats)

    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(test)
    train_ids = {x["review_id"] for x in train}
    validation_ids = {x["review_id"] for x in validation}
    test_ids = {x["review_id"] for x in test}
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise AssertionError("Review leakage across train, validation, and test")
    metadata = {
        "train_instance_counts": dict(Counter(x["bucket"] for x in train)),
        "train_unique_counts": dict(Counter(x["bucket"] for x in train_unique)),
        "validation_counts": dict(Counter(x["bucket"] for x in validation)),
        "test_counts": dict(Counter(x["bucket"] for x in test)),
        "train_instances": len(train),
        "train_unique_reviews": len(train_ids),
        "validation_unique_reviews": len(validation_ids),
        "total_label_instances": len(train) + len(validation),
        "total_unique_labeled_reviews": len(train_ids) + len(validation_ids),
        "critical_instances": sum(x["bucket"] == "mini_only" for x in train + validation),
        "critical_percentage": sum(x["bucket"] == "mini_only" for x in train + validation)
        / (len(train) + len(validation)),
        "resampled_train_review_ids": resampled_ids,
    }
    return train, validation, test, metadata


def train_configuration(
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    test: list[dict[str, Any]],
    priors: dict[str, float],
    seed: int,
    prompt_tokens: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, local_files_only=True)
    model = SoftPromptDirectRouter(args.hf_model, prompt_tokens)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )
    criterion = nn.CrossEntropyLoss()
    local_args = copy.copy(args)
    local_args.prompt_tokens = prompt_tokens
    train_loader = make_loader(train, tokenizer, local_args, shuffle=True)
    validation_loader = make_loader(validation, tokenizer, local_args, shuffle=False)
    test_loader = make_loader(test, tokenizer, local_args, shuffle=False)

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
        key = (weighted_utility, -mini_calls)
        history.append({
            "epoch": epoch,
            "train_loss": loss_sum / len(train),
            "validation_threshold": threshold,
            "validation_population_weighted_utility": weighted_utility,
            "validation_mini_calls": mini_calls,
        })
        if best_key is None or key > best_key:
            best_key = key
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_threshold = threshold

    assert best_state is not None
    model.load_state_dict(best_state)
    test_probs = probabilities(model, test_loader)
    calibrated = [int(value >= best_threshold) for value in test_probs]
    default = [int(value >= 0.5) for value in test_probs]
    return {
        "seed": seed,
        "prompt_tokens": prompt_tokens,
        "best_epoch": best_epoch,
        "threshold": best_threshold,
        "calibrated": evaluate(test, calibrated),
        "default_threshold": evaluate(test, default),
        "history": history,
    }


def ranking_summary(by_prompt: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for prompt_tokens, runs in by_prompt.items():
        calibrated = aggregate(runs, "calibrated")
        rows.append({
            "prompt_tokens": prompt_tokens,
            "mean_utility": calibrated["mean_utility"]["mean"],
            "std_utility": calibrated["mean_utility"]["std"],
            "mean_selected_accuracy": calibrated["selected_accuracy"]["mean"],
            "mean_mini_rate": calibrated["mini_rate"]["mean"],
            "mean_critical_recall": calibrated["mini_only_recall"]["mean"],
            "mean_both_wrong_mini_rate": calibrated["both_wrong_mini_rate"]["mean"],
        })
    return sorted(rows, key=lambda row: (row["mean_utility"], row["mean_selected_accuracy"]), reverse=True)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", type=Path, default=root / "sembench/files/movie/data/sf_2000/Reviews.csv")
    parser.add_argument("--cache", type=Path, default=root / "sembench_movie_router/model_outputs.json")
    parser.add_argument("--hf-model", default="distilbert-base-uncased")
    parser.add_argument("--prompt-tokens", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    parser.add_argument("--seeds", type=int, nargs="+", default=[505, 606, 707])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument(
        "--output", type=Path,
        default=root / "sembench_movie_router/size100_softprompt_token_sweep.json",
    )
    args = parser.parse_args()

    raw_examples = load_examples(args.reviews, args.cache)
    examples = deduplicate(raw_examples)
    available = Counter(x["bucket"] for x in examples)
    priors = {bucket: available[bucket] / len(examples) for bucket in BUCKETS}
    splits = {seed: make_size100_split(examples, seed) for seed in args.seeds}

    by_prompt: dict[int, list[dict[str, Any]]] = {}
    for prompt_tokens in args.prompt_tokens:
        by_prompt[prompt_tokens] = []
        for seed in args.seeds:
            train, validation, test, _ = splits[seed]
            print(f"prompt_tokens={prompt_tokens} seed={seed}: training", flush=True)
            run = train_configuration(
                train, validation, test, priors, seed, prompt_tokens, args
            )
            by_prompt[prompt_tokens].append(run)
            metrics = run["calibrated"]
            print(
                f"prompt_tokens={prompt_tokens} seed={seed}: "
                f"accuracy={metrics['selected_accuracy']:.4f} "
                f"utility={metrics['mean_utility']:.4f} "
                f"critical_recall={metrics['mini_only_recall']:.4f} "
                f"mini_rate={metrics['mini_rate']:.4f}",
                flush=True,
            )

    first_test = splits[args.seeds[0]][2]
    output = {
        "config": {
            **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "critical_target_percentage": 0.26,
            "both_wrong_target": "mini",
            "router_inputs": ["semantic_operator_description", "review_text"],
            "nano_output_exposed_to_router": False,
        },
        "raw_rows": len(raw_examples),
        "unique_reviews": len(examples),
        "available_bucket_counts": dict(available),
        "natural_bucket_priors": priors,
        "split_metadata": {str(seed): splits[seed][3] for seed in args.seeds},
        "baselines_on_seed_505_test": {
            "all_nano": evaluate(first_test, [0] * len(first_test)),
            "all_mini": evaluate(first_test, [1] * len(first_test)),
            "target_oracle": evaluate(first_test, [int(x["target"]) for x in first_test]),
        },
        "ranking": ranking_summary(by_prompt),
        "aggregate": {
            str(prompt_tokens): {
                "calibrated": aggregate(runs, "calibrated"),
                "default_threshold": aggregate(runs, "default_threshold"),
            }
            for prompt_tokens, runs in by_prompt.items()
        },
        "runs": {str(prompt_tokens): runs for prompt_tokens, runs in by_prompt.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "split": output["split_metadata"][str(args.seeds[0])],
        "ranking": output["ranking"],
        "baselines": output["baselines_on_seed_505_test"],
    }, indent=2), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
