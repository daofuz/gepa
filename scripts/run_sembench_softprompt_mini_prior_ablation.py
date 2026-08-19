#!/usr/bin/env python3
"""Ablate a MAP-style Mini prior for the SemBench Movie soft-prompt router.

The experiment reuses the historical, leakage-safe validation/test construction
and the outcome-enriched 100-label training mixture.  Its hard route target is
the paper's cost-derived target: Mini only for mini-only rows; Nano for both
correct, nano-only, and both-wrong rows.  Cross entropy is weighted by the
inverse expected cost of the target route.  The optional MAP term regularizes
the classifier intercept difference toward a declared prior Mini probability.

No answer-model API calls are made; all outcomes and token counts are cached.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
from statistics import mean, pstdev
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from run_sembench_accuracy_targeted_router import (
    capture_trainable,
    evaluate_scores,
    loader,
    probabilities,
    restore_trainable,
    select_threshold,
)
from run_sembench_qa50_softprompt_router import deduplicate
from run_sembench_size_prompt_grid import select_nested
from run_sembench_train100_ratio_sweep import make_orders
from train_sembench_balanced_direct_router import (
    SoftPromptDirectRouter,
    evaluate,
    load_examples,
)


TRAIN_COUNTS = {
    "both_correct": 50,
    "mini_only": 46,
    "nano_only": 3,
    "both_wrong": 1,
}

# This is the paper's recorded 4:1 Mini:Nano price ratio.  Absolute currency
# scale cancels because each batch divides by the sum of example weights.
PRICING_PER_1M = {
    "nano": {"input": 0.10, "output": 0.40},
    "mini": {"input": 0.40, "output": 1.60},
}
MODEL_KEYS = {"nano": "gpt-5-nano", "mini": "gpt-5-mini"}


def cost_target(bucket: str) -> int:
    """Return 1 for Mini and 0 for Nano under the cost-derived tie-break."""
    return int(bucket == "mini_only")


def apply_cost_targets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**row, "target": cost_target(str(row["bucket"]))} for row in rows]


def realized_cost(cache: dict[str, Any], review_id: str, route: str) -> float:
    row = cache[f"{review_id}::{MODEL_KEYS[route]}"]
    price = PRICING_PER_1M[route]
    return (
        float(row["input_tokens"]) * price["input"]
        + float(row["output_tokens"]) * price["output"]
    ) / 1_000_000.0


def expected_costs(
    train: list[dict[str, Any]], cache: dict[str, Any]
) -> dict[str, float]:
    return {
        route: mean(realized_cost(cache, str(row["review_id"]), route) for row in train)
        for route in ("nano", "mini")
    }


def fixed_threshold_summary(
    rows: list[dict[str, Any]], scores: list[float], threshold: float = 0.5
) -> dict[str, Any]:
    return evaluate_scores(rows, scores, threshold)


def prior_loss(
    model: SoftPromptDirectRouter, prior_mini_probability: float, strength: float
) -> torch.Tensor:
    if strength == 0.0:
        return model.classifier.bias.sum() * 0.0
    target_log_odds = math.log(prior_mini_probability / (1.0 - prior_mini_probability))
    bias_log_odds = model.classifier.bias[1] - model.classifier.bias[0]
    return 0.5 * strength * (bias_log_odds - target_log_odds).pow(2)


def run_one(
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    test: list[dict[str, Any]],
    cache: dict[str, Any],
    seed: int,
    prior_mini_probability: float,
    prior_strength: float,
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
    train_loader = loader(train, tokenizer, args, "route", True)
    validation_loader = loader(validation, tokenizer, args, "route", False)
    test_loader = loader(test, tokenizer, args, "route", False)

    costs = expected_costs(train, cache)
    inverse_cost = torch.tensor(
        [1.0 / costs["nano"], 1.0 / costs["mini"]], dtype=torch.float
    )
    best: dict[str, dict[str, Any]] = {}
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_wce = 0.0
        total_prior = 0.0
        seen = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], batch["attention_mask"])
            labels = batch["labels"]
            ce = F.cross_entropy(logits, labels, reduction="none")
            weights = inverse_cost[labels]
            wce = (weights * ce).sum() / weights.sum()
            map_term = prior_loss(model, prior_mini_probability, prior_strength)
            loss = wce + map_term
            loss.backward()
            optimizer.step()
            count = len(labels)
            total_loss += float(loss.detach()) * count
            total_wce += float(wce.detach()) * count
            total_prior += float(map_term.detach()) * count
            seen += count

        validation_scores = probabilities(model, validation_loader)
        calibrated = select_threshold(validation, validation_scores, "max_accuracy")
        fixed = fixed_threshold_summary(validation, validation_scores)
        epoch_trace = {
            "epoch": epoch,
            "train_loss": total_loss / seen,
            "weighted_ce": total_wce / seen,
            "prior_term": total_prior / seen,
            "bias_log_odds": float(
                (model.classifier.bias[1] - model.classifier.bias[0]).detach()
            ),
            "calibrated_threshold": calibrated["threshold"],
            "calibrated_validation_accuracy": calibrated["validation_metrics"][
                "selected_accuracy"
            ],
            "fixed_validation_accuracy": fixed["selected_accuracy"],
            "fixed_validation_mini_rate": fixed["mini_rate"],
        }
        history.append(epoch_trace)

        candidates = {
            "calibrated": (
                (
                    calibrated["validation_metrics"]["selected_accuracy"],
                    -calibrated["validation_metrics"]["mini_calls"],
                ),
                calibrated["threshold"],
                calibrated["validation_metrics"],
            ),
            "fixed_0.5": (
                (fixed["selected_accuracy"], -fixed["mini_calls"]),
                0.5,
                fixed,
            ),
        }
        for policy, (key, threshold, metrics) in candidates.items():
            if policy not in best or key > best[policy]["key"]:
                best[policy] = {
                    "key": key,
                    "epoch": epoch,
                    "threshold": threshold,
                    "validation": metrics,
                    "state": copy.deepcopy(capture_trainable(model)),
                }

    policies: dict[str, Any] = {}
    for policy, selected in best.items():
        restore_trainable(model, selected["state"])
        test_scores = probabilities(model, test_loader)
        policies[policy] = {
            "best_epoch": selected["epoch"],
            "threshold": selected["threshold"],
            "validation": selected["validation"],
            "test": evaluate_scores(test, test_scores, selected["threshold"]),
            "final_bias_log_odds": float(
                (model.classifier.bias[1] - model.classifier.bias[0]).detach()
            ),
        }

    return {
        "seed": seed,
        "prior_mini_probability": prior_mini_probability,
        "prior_strength": prior_strength,
        "target_log_odds": math.log(
            prior_mini_probability / (1.0 - prior_mini_probability)
        ),
        "expected_costs": costs,
        "inverse_cost_ratio_mini_to_nano": costs["nano"] / costs["mini"],
        "policies": policies,
        "history": history,
    }


def aggregate(runs: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    fields = (
        "selected_accuracy",
        "mini_rate",
        "mini_saving_vs_all_mini",
        "mini_only_recall",
        "both_wrong_mini_rate",
        "critical_misroutes",
        "unnecessary_mini",
    )
    result: dict[str, Any] = {}
    for field in fields:
        values = [float(run["policies"][policy]["test"][field]) for run in runs]
        result[field] = {"mean": mean(values), "std": pstdev(values)}
    thresholds = [float(run["policies"][policy]["threshold"]) for run in runs]
    result["threshold"] = {"mean": mean(thresholds), "std": pstdev(thresholds)}
    biases = [float(run["policies"][policy]["final_bias_log_odds"]) for run in runs]
    result["bias_log_odds"] = {"mean": mean(biases), "std": pstdev(biases)}
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reviews",
        type=Path,
        default=root / "sembench/files/movie/data/sf_2000/Reviews.csv",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=root / "sembench_movie_router/model_outputs.json",
    )
    parser.add_argument(
        "--untouched-manifest",
        type=Path,
        default=root / "sembench_movie_router/untouched_200_manifest.json",
    )
    parser.add_argument("--hf-model", default="distilbert-base-uncased")
    parser.add_argument("--prompt-tokens", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[505, 606, 707])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--prior-mini-probability", type=float, default=0.75)
    parser.add_argument(
        "--prior-strengths", type=float, nargs="+", default=[0.0, 0.01, 0.1, 1.0]
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "sembench_movie_router/softprompt_mini_prior_ablation.json",
    )
    args = parser.parse_args()
    if not 0.5 < args.prior_mini_probability < 1.0:
        raise ValueError("--prior-mini-probability must be between 0.5 and 1")

    heldout_ids = set(
        json.loads(args.untouched_manifest.read_text(encoding="utf-8"))["review_ids"]
    )
    cache = json.loads(args.cache.read_text(encoding="utf-8"))
    examples = deduplicate(load_examples(args.reviews, args.cache))
    historical = [row for row in examples if row["review_id"] not in heldout_ids]
    if len(historical) != 812:
        raise AssertionError(f"Expected 812 historical rows, found {len(historical)}")
    per_seed = {seed: make_orders(historical, seed) for seed in args.seeds}

    run_args = SimpleNamespace(
        hf_model=args.hf_model,
        prompt_tokens=args.prompt_tokens,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    runs_by_strength: dict[str, list[dict[str, Any]]] = {}
    for strength in args.prior_strengths:
        key = f"{strength:g}"
        runs_by_strength[key] = []
        for seed in args.seeds:
            orders, validation_raw, test_raw = per_seed[seed]
            train_raw = select_nested(orders, TRAIN_COUNTS, seed)
            train = apply_cost_targets(train_raw)
            validation = apply_cost_targets(validation_raw)
            test = apply_cost_targets(test_raw)
            print(
                f"prior_strength={strength:g} prior_pi={args.prior_mini_probability:g} "
                f"seed={seed}",
                flush=True,
            )
            run = run_one(
                train,
                validation,
                test,
                cache,
                seed,
                args.prior_mini_probability,
                strength,
                run_args,
            )
            runs_by_strength[key].append(run)
            for policy in ("calibrated", "fixed_0.5"):
                metrics = run["policies"][policy]["test"]
                print(
                    f"  {policy}: accuracy={metrics['selected_accuracy']:.4f} "
                    f"saving={metrics['mini_saving_vs_all_mini']:.4f} "
                    f"critical={metrics['mini_only_recall']:.4f}",
                    flush=True,
                )

    summaries = {
        strength: {
            policy: aggregate(runs, policy)
            for policy in ("calibrated", "fixed_0.5")
        }
        for strength, runs in runs_by_strength.items()
    }
    first_test = apply_cost_targets(per_seed[args.seeds[0]][2])
    result = {
        "protocol": {
            "dataset": "SemBench Movie historical split; untouched 200 excluded",
            "train_counts": TRAIN_COUNTS,
            "loss": "inverse-expected-cost weighted cross entropy plus intercept MAP prior",
            "prior": (
                "lambda/2 * ((b_mini-b_nano)-logit(prior_mini_probability))^2"
            ),
            "target": "Mini for MO; Nano for BC, NO, and BW (Nano tie-break)",
            "pricing_per_1m": PRICING_PER_1M,
            "selection": "validation maximum accuracy, then fewer Mini calls",
            "openai_api_calls": 0,
        },
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "baselines": {
            "all_nano": evaluate(first_test, [0] * len(first_test)),
            "all_mini": evaluate(first_test, [1] * len(first_test)),
        },
        "aggregate": summaries,
        "runs": runs_by_strength,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"baselines": result["baselines"], "aggregate": summaries}, indent=2))
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
