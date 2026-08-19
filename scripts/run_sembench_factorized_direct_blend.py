#!/usr/bin/env python3
"""Blend semantic-label and outcome-risk soft-prompt heads for direct routing.

Both heads see only the review text.  The semantic head learns the operator
ground truth (positive/negative); the risk head learns the conservative route
target (mini for mini-only and both-wrong).  Validation chooses a small convex
blend grid and an exact empirical threshold by final answer accuracy.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch
from transformers import AutoTokenizer

from run_sembench_accuracy_targeted_router import (
    capture_trainable,
    direct_loss,
    evaluate_scores,
    loader,
    make_split,
    probabilities,
    restore_trainable,
    select_threshold,
    sentiment_loss,
)
from run_sembench_qa50_softprompt_router import deduplicate
from train_sembench_balanced_direct_router import (
    SoftPromptDirectRouter,
    evaluate,
    load_examples,
)


ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def blend_scores(
    sentiment_scores: list[float], route_scores: list[float], alpha: float
) -> list[float]:
    """Blend in log-odds space; alpha=1 is sentiment-only."""
    result = []
    for sentiment, route in zip(sentiment_scores, route_scores):
        sentiment = min(max(float(sentiment), 1e-6), 1.0 - 1e-6)
        route = min(max(float(route), 1e-6), 1.0 - 1e-6)
        sentiment_logit = math.log(sentiment / (1.0 - sentiment))
        route_logit = math.log(route / (1.0 - route))
        value = alpha * sentiment_logit + (1.0 - alpha) * route_logit
        result.append(1.0 / (1.0 + math.exp(-value)))
    return result


def aggregate(runs: list[dict[str, Any]], field: str) -> dict[str, float]:
    values = [float(run["test"][field]) for run in runs]
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

    sentiment_model = SoftPromptDirectRouter(args.hf_model, args.prompt_tokens)
    route_model = SoftPromptDirectRouter(args.hf_model, args.prompt_tokens)
    sentiment_optimizer = torch.optim.AdamW(
        [p for p in sentiment_model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    route_optimizer = torch.optim.AdamW(
        [p for p in route_model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    sentiment_train = loader(train, tokenizer, args, "sentiment", True)
    route_train = loader(train, tokenizer, args, "route", True)
    validation_loader = loader(validation, tokenizer, args, "sentiment", False)
    test_loader = loader(test, tokenizer, args, "sentiment", False)
    counts = Counter(int(x["gold"] == "POSITIVE") for x in train)
    class_weights = torch.tensor(
        [len(train) / (2.0 * max(counts[index], 1)) for index in (0, 1)],
        dtype=torch.float,
    )

    best: dict[str, Any] | None = None
    history = []
    for epoch in range(1, args.epochs + 1):
        sentiment_model.train()
        for batch in sentiment_train:
            sentiment_optimizer.zero_grad(set_to_none=True)
            logits = sentiment_model(batch["input_ids"], batch["attention_mask"])
            loss = sentiment_loss(
                logits, batch["labels"], class_weights, "sentiment_ce"
            )
            loss.backward()
            sentiment_optimizer.step()

        route_model.train()
        for batch in route_train:
            route_optimizer.zero_grad(set_to_none=True)
            logits = route_model(batch["input_ids"], batch["attention_mask"])
            loss = direct_loss(
                logits,
                batch["labels"],
                batch["buckets"],
                "regret_ce",
                epoch,
                args.epochs,
            )
            loss.backward()
            route_optimizer.step()

        sentiment_validation = probabilities(sentiment_model, validation_loader)
        route_validation = probabilities(route_model, validation_loader)
        epoch_best: dict[str, Any] | None = None
        for alpha in ALPHAS:
            scores = blend_scores(sentiment_validation, route_validation, alpha)
            calibration = select_threshold(validation, scores, "max_accuracy")
            metrics = calibration["validation_metrics"]
            key = (metrics["selected_accuracy"], -metrics["mini_calls"], alpha)
            if epoch_best is None or key > epoch_best["key"]:
                epoch_best = {
                    "key": key,
                    "alpha": alpha,
                    "calibration": calibration,
                }

        assert epoch_best is not None
        history.append({
            "epoch": epoch,
            "alpha": epoch_best["alpha"],
            "threshold": epoch_best["calibration"]["threshold"],
            "validation_accuracy": epoch_best["calibration"]["validation_metrics"][
                "selected_accuracy"
            ],
            "validation_mini_rate": epoch_best["calibration"]["validation_metrics"][
                "mini_rate"
            ],
        })
        if best is None or epoch_best["key"] > best["key"]:
            best = {
                **epoch_best,
                "epoch": epoch,
                "sentiment_state": copy.deepcopy(capture_trainable(sentiment_model)),
                "route_state": copy.deepcopy(capture_trainable(route_model)),
            }

    assert best is not None
    restore_trainable(sentiment_model, best["sentiment_state"])
    restore_trainable(route_model, best["route_state"])
    test_scores = blend_scores(
        probabilities(sentiment_model, test_loader),
        probabilities(route_model, test_loader),
        best["alpha"],
    )
    return {
        "seed": seed,
        "best_epoch": best["epoch"],
        "alpha_sentiment": best["alpha"],
        "threshold": best["calibration"]["threshold"],
        "validation": best["calibration"]["validation_metrics"],
        "test": evaluate_scores(test, test_scores, best["calibration"]["threshold"]),
        "history": history,
    }


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
        default=root / "sembench_movie_router/factorized_direct_blend.json",
    )
    args = parser.parse_args()

    examples = deduplicate(load_examples(args.reviews, args.cache))
    splits = {seed: make_split(examples, seed) for seed in args.seeds}
    runs = []
    for seed in args.seeds:
        train, validation, test, _ = splits[seed]
        print(f"seed={seed}: factorized direct blend", flush=True)
        run = run_seed(train, validation, test, seed, args)
        runs.append(run)
        print(
            f"  accuracy={run['test']['selected_accuracy']:.4f} "
            f"mini_rate={run['test']['mini_rate']:.4f} "
            f"critical_recall={run['test']['mini_only_recall']:.4f} "
            f"alpha={run['alpha_sentiment']:.2f}",
            flush=True,
        )

    first_test = splits[args.seeds[0]][2]
    fields = (
        "selected_accuracy",
        "mini_rate",
        "mini_saving_vs_all_mini",
        "mini_only_recall",
        "both_wrong_mini_rate",
        "critical_misroutes",
    )
    result = {
        "config": {
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "alpha_grid": ALPHAS,
            "openai_api_calls": 0,
            "protocol": "200 unique labels; direct pre-routing; no nano answer",
        },
        "baselines": {
            "all_nano": evaluate(first_test, [0] * len(first_test)),
            "all_mini": evaluate(first_test, [1] * len(first_test)),
        },
        "aggregate": {field: aggregate(runs, field) for field in fields},
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
