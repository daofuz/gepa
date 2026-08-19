#!/usr/bin/env python3
"""Direct sentiment router with validation critical-recall constraints."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch
from transformers import AutoTokenizer

from run_sembench_accuracy_targeted_router import (
    capture_trainable,
    evaluate_scores,
    loader,
    make_split,
    probabilities,
    restore_trainable,
    sentiment_loss,
    threshold_candidates,
)
from run_sembench_qa50_softprompt_router import deduplicate
from train_sembench_balanced_direct_router import (
    SoftPromptDirectRouter,
    evaluate,
    load_examples,
)


POLICIES = ("critical_only", "critical_and_both_wrong")


def select_constrained_threshold(
    examples: list[dict[str, Any]], scores: list[float], policy: str
) -> dict[str, Any]:
    rows = []
    for threshold in threshold_candidates(scores):
        predictions = [int(score >= threshold) for score in scores]
        rows.append({"threshold": threshold, **evaluate(examples, predictions)})
    feasible = [row for row in rows if row["mini_only_recall"] >= 1.0 - 1e-12]
    if policy == "critical_and_both_wrong":
        feasible = [
            row for row in feasible
            if row["both_wrong_mini_rate"] >= 1.0 - 1e-12
        ]
    if not feasible:
        raise AssertionError("All-mini threshold should always satisfy the constraints")
    return max(
        feasible,
        key=lambda row: (-row["mini_calls"], row["selected_accuracy"], row["threshold"]),
    )


def aggregate(runs: list[dict[str, Any]], policy: str, field: str) -> dict[str, float]:
    values = [float(run["policies"][policy]["test"][field]) for run in runs]
    return {"mean": mean(values), "std": pstdev(values)}


def run_seed(
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    test: list[dict[str, Any]],
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
        weight_decay=args.weight_decay,
    )
    train_loader = loader(train, tokenizer, args, "sentiment", True)
    validation_loader = loader(validation, tokenizer, args, "sentiment", False)
    test_loader = loader(test, tokenizer, args, "sentiment", False)
    counts = Counter(int(example["gold"] == "POSITIVE") for example in train)
    class_weights = torch.tensor(
        [len(train) / (2.0 * max(counts[index], 1)) for index in (0, 1)],
        dtype=torch.float,
    )

    best: dict[str, dict[str, Any]] = {}
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = sentiment_loss(
                logits, batch["labels"], class_weights, "sentiment_ce"
            )
            loss.backward()
            optimizer.step()

        scores = probabilities(model, validation_loader)
        trace: dict[str, Any] = {"epoch": epoch}
        for policy in POLICIES:
            calibration = select_constrained_threshold(validation, scores, policy)
            key = (
                -calibration["mini_calls"],
                calibration["selected_accuracy"],
                calibration["threshold"],
            )
            trace[policy] = {
                "threshold": calibration["threshold"],
                "validation_accuracy": calibration["selected_accuracy"],
                "validation_mini_rate": calibration["mini_rate"],
            }
            if policy not in best or key > best[policy]["key"]:
                best[policy] = {
                    "key": key,
                    "epoch": epoch,
                    "state": capture_trainable(model),
                    "calibration": calibration,
                }
        history.append(trace)

    policies = {}
    for policy, selected in best.items():
        restore_trainable(model, selected["state"])
        test_scores = probabilities(model, test_loader)
        policies[policy] = {
            "best_epoch": selected["epoch"],
            "threshold": selected["calibration"]["threshold"],
            "validation": {
                key: value for key, value in selected["calibration"].items()
                if key != "threshold"
            },
            "test": evaluate_scores(test, test_scores, selected["calibration"]["threshold"]),
        }
    return {"seed": seed, "policies": policies, "history": history}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reviews", type=Path,
        default=root / "sembench/files/movie/data/sf_2000/Reviews.csv",
    )
    parser.add_argument(
        "--cache", type=Path,
        default=root / "sembench_movie_router/model_outputs.json",
    )
    parser.add_argument("--hf-model", default="distilbert-base-uncased")
    parser.add_argument("--prompt-tokens", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[505, 606, 707])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--output", type=Path,
        default=root / "sembench_movie_router/critical_constrained_direct.json",
    )
    args = parser.parse_args()

    examples = deduplicate(load_examples(args.reviews, args.cache))
    splits = {seed: make_split(examples, seed) for seed in args.seeds}
    runs = []
    for seed in args.seeds:
        train, validation, test, _ = splits[seed]
        print(f"seed={seed}: critical-constrained direct", flush=True)
        run = run_seed(train, validation, test, seed, args)
        runs.append(run)
        for policy in POLICIES:
            metrics = run["policies"][policy]["test"]
            print(
                f"  {policy}: accuracy={metrics['selected_accuracy']:.4f} "
                f"mini_rate={metrics['mini_rate']:.4f} "
                f"critical_recall={metrics['mini_only_recall']:.4f} "
                f"both_wrong={metrics['both_wrong_mini_rate']:.4f}",
                flush=True,
            )

    fields = (
        "selected_accuracy",
        "mini_rate",
        "mini_saving_vs_all_mini",
        "mini_only_recall",
        "both_wrong_mini_rate",
        "critical_misroutes",
    )
    first_test = splits[args.seeds[0]][2]
    result = {
        "config": {
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "openai_api_calls": 0,
            "protocol": "200 unique labels; direct pre-routing; no nano answer",
        },
        "baselines": {
            "all_nano": evaluate(first_test, [0] * len(first_test)),
            "all_mini": evaluate(first_test, [1] * len(first_test)),
        },
        "aggregate": {
            policy: {field: aggregate(runs, policy, field) for field in fields}
            for policy in POLICIES
        },
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
