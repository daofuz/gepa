#!/usr/bin/env python3
"""Two-head soft-prompt correctness router with explicit MAP base-rate priors.

The model predicts P(Nano correct | x) and P(Mini correct | x) separately.
Gaussian priors on the two correctness-head intercepts encode declared base
accuracy beliefs without imposing a prior on the final Mini routing rate.
Routing uses predicted accuracy uplift, P(Mini correct)-P(Nano correct), and a
validation-only threshold implements the accuracy/cost operating constraint.

No answer-model API calls are made.  The evaluation is post-hoc on the existing
SemBench Movie held-out set and must not be described as a fresh final test.
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
import torch.nn.functional as F
from transformers import AutoTokenizer

import run_sembench_twohead_correctness_router as base
from run_sembench_accuracy_targeted_router import evaluate_scores, select_threshold
from run_sembench_qa50_softprompt_router import deduplicate
from run_sembench_softprompt_mini_prior_ablation import expected_costs
from run_sembench_train100_ratio_sweep import make_orders, select_train
from train_sembench_balanced_direct_router import evaluate, load_examples


TRAIN_COUNTS = {
    "both_correct": 50,
    "mini_only": 46,
    "nano_only": 3,
    "both_wrong": 1,
}


def logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


def capture_trainable(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def restore_trainable(
    model: torch.nn.Module, state: dict[str, torch.Tensor]
) -> None:
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in state:
                parameter.copy_(state[name])


def map_prior(
    model: base.TwoHeadCorrectnessRouter,
    prior_accuracy_nano: float,
    prior_accuracy_mini: float,
    prior_strength: float,
    train_size: int,
) -> torch.Tensor:
    """Negative log Gaussian prior in mean-data-loss units.

    If k=prior_strength, the prior precision relative to the mean BCE is k/n.
    This makes k interpretable as an effective prior sample-size scale while
    retaining a proper Gaussian prior on each intercept log-odds.
    """
    biases = model.correctness_heads.bias
    centers = biases.new_tensor(
        [logit(prior_accuracy_nano), logit(prior_accuracy_mini)]
    )
    return (prior_strength / (2.0 * train_size)) * (biases - centers).pow(2).sum()


def policy_candidates(
    validation: list[dict[str, Any]], scores: list[float]
) -> dict[str, dict[str, Any]]:
    max_accuracy = select_threshold(validation, scores, "max_accuracy")
    accuracy_floor = select_threshold(validation, scores, "match_all_mini")
    return {
        "direct_uplift": {
            "threshold": 0.0,
            "validation": evaluate_scores(validation, scores, 0.0),
        },
        "validation_max_accuracy": {
            "threshold": max_accuracy["threshold"],
            "validation": max_accuracy["validation_metrics"],
        },
        "validation_accuracy_floor": {
            "threshold": accuracy_floor["threshold"],
            "validation": accuracy_floor["validation_metrics"],
        },
    }


def selection_key(policy: str, metrics: dict[str, Any]) -> tuple[float, float]:
    if policy == "validation_accuracy_floor":
        return (-float(metrics["mini_calls"]), float(metrics["selected_accuracy"]))
    return (float(metrics["selected_accuracy"]), -float(metrics["mini_calls"]))


def run_one(
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    test: list[dict[str, Any]],
    cache: dict[str, Any],
    seed: int,
    prior_strength: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, local_files_only=True)
    model = base.TwoHeadCorrectnessRouter(args.hf_model, args.prompt_tokens)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_loader = base.loader(
        train, tokenizer, args.max_length, args.batch_size, True
    )
    validation_loader = base.loader(
        validation, tokenizer, args.max_length, args.batch_size, False
    )
    test_loader = base.loader(
        test, tokenizer, args.max_length, args.batch_size, False
    )
    costs = expected_costs(train, cache)
    inverse_cost = torch.tensor(
        [1.0 / costs["nano"], 1.0 / costs["mini"]], dtype=torch.float
    )

    best: dict[str, dict[str, Any]] = {}
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_bce = 0.0
        total_prior = 0.0
        seen = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], batch["attention_mask"])
            elementwise_bce = F.binary_cross_entropy_with_logits(
                logits, batch["labels"], reduction="none"
            )
            # Each correctness likelihood is weighted by the inverse expected
            # cost of querying that model. Normalizing by the total head weight
            # keeps the loss scale stable when absolute token prices change.
            bce = (elementwise_bce * inverse_cost).sum() / (
                len(batch["labels"]) * inverse_cost.sum()
            )
            prior = map_prior(
                model,
                args.prior_accuracy_nano,
                args.prior_accuracy_mini,
                prior_strength,
                len(train),
            )
            loss = bce + prior
            loss.backward()
            optimizer.step()
            count = len(batch["labels"])
            total_loss += float(loss.detach()) * count
            total_bce += float(bce.detach()) * count
            total_prior += float(prior.detach()) * count
            seen += count

        q_nano, q_mini = base.correctness_probabilities(model, validation_loader)
        scores = base.score(q_nano, q_mini)
        candidates = policy_candidates(validation, scores)
        trace: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": total_loss / seen,
            "bce": total_bce / seen,
            "prior": total_prior / seen,
            "head_biases": model.correctness_heads.bias.detach().tolist(),
            "mean_validation_q_nano": mean(q_nano),
            "mean_validation_q_mini": mean(q_mini),
            "policies": {},
        }
        for policy, candidate in candidates.items():
            metrics = candidate["validation"]
            key = selection_key(policy, metrics)
            trace["policies"][policy] = {
                "threshold": candidate["threshold"],
                "accuracy": metrics["selected_accuracy"],
                "mini_rate": metrics["mini_rate"],
            }
            if policy not in best or key > best[policy]["key"]:
                best[policy] = {
                    "key": key,
                    "epoch": epoch,
                    "threshold": candidate["threshold"],
                    "validation": metrics,
                    "state": copy.deepcopy(capture_trainable(model)),
                }
        history.append(trace)

    policies: dict[str, Any] = {}
    for policy, selected in best.items():
        restore_trainable(model, selected["state"])
        q_nano, q_mini = base.correctness_probabilities(model, test_loader)
        scores = base.score(q_nano, q_mini)
        policies[policy] = {
            "best_epoch": selected["epoch"],
            "threshold": selected["threshold"],
            "validation": selected["validation"],
            "test": evaluate_scores(test, scores, selected["threshold"]),
            "mean_q_nano": mean(q_nano),
            "mean_q_mini": mean(q_mini),
            "mean_predicted_uplift": mean(scores),
            "head_biases": model.correctness_heads.bias.detach().tolist(),
        }
    return {
        "seed": seed,
        "prior_strength": prior_strength,
        "prior_centers": {
            "nano_accuracy": args.prior_accuracy_nano,
            "mini_accuracy": args.prior_accuracy_mini,
            "nano_log_odds": logit(args.prior_accuracy_nano),
            "mini_log_odds": logit(args.prior_accuracy_mini),
        },
        "expected_costs": costs,
        "normalized_head_weights": {
            "nano": float(inverse_cost[0] / inverse_cost.sum()),
            "mini": float(inverse_cost[1] / inverse_cost.sum()),
        },
        "policies": policies,
        "history": history,
    }


def aggregate(
    runs: list[dict[str, Any]], policy: str
) -> dict[str, dict[str, float]]:
    fields = (
        "selected_accuracy",
        "accuracy_gap_vs_all_mini",
        "mini_rate",
        "mini_saving_vs_all_mini",
        "mini_only_recall",
        "both_wrong_mini_rate",
        "critical_misroutes",
        "unnecessary_mini",
    )
    output: dict[str, dict[str, float]] = {}
    for field in fields:
        values = [float(run["policies"][policy]["test"][field]) for run in runs]
        output[field] = {"mean": mean(values), "std": pstdev(values)}
    for field in ("threshold", "mean_q_nano", "mean_q_mini", "mean_predicted_uplift"):
        values = [float(run["policies"][policy][field]) for run in runs]
        output[field] = {"mean": mean(values), "std": pstdev(values)}
    return output


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
        "--manifest",
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
    parser.add_argument("--prior-accuracy-nano", type=float, default=0.80)
    parser.add_argument("--prior-accuracy-mini", type=float, default=0.90)
    parser.add_argument(
        "--prior-strengths", type=float, nargs="+", default=[0.0, 5.0, 20.0, 100.0]
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "outputs/twohead_correctness_map_prior.json",
    )
    args = parser.parse_args()
    if not 0.0 < args.prior_accuracy_nano < args.prior_accuracy_mini < 1.0:
        raise ValueError("Require 0 < Nano prior accuracy < Mini prior accuracy < 1")

    heldout_ids = set(
        json.loads(args.manifest.read_text(encoding="utf-8"))["review_ids"]
    )
    cache = json.loads(args.cache.read_text(encoding="utf-8"))
    examples = deduplicate(load_examples(args.reviews, args.cache))
    heldout = [row for row in examples if row["review_id"] in heldout_ids]
    historical = [row for row in examples if row["review_id"] not in heldout_ids]
    if len(heldout) != 200 or len(historical) != 812:
        raise AssertionError("Expected 812 historical and 200 held-out reviews")
    per_seed = {seed: make_orders(historical, seed) for seed in args.seeds}

    runs_by_strength: dict[str, list[dict[str, Any]]] = {}
    for strength in args.prior_strengths:
        key = f"{strength:g}"
        runs_by_strength[key] = []
        for seed in args.seeds:
            orders, validation, _ = per_seed[seed]
            train = select_train(orders, TRAIN_COUNTS, seed)
            print(
                f"prior_strength={strength:g} seed={seed} "
                f"q_nano={args.prior_accuracy_nano:g} q_mini={args.prior_accuracy_mini:g}",
                flush=True,
            )
            run = run_one(train, validation, heldout, cache, seed, strength, args)
            runs_by_strength[key].append(run)
            for policy, values in run["policies"].items():
                metrics = values["test"]
                print(
                    f"  {policy}: accuracy={metrics['selected_accuracy']:.4f} "
                    f"saving={metrics['mini_saving_vs_all_mini']:.4f} "
                    f"critical={metrics['mini_only_recall']:.4f}",
                    flush=True,
                )

    policies = (
        "direct_uplift",
        "validation_max_accuracy",
        "validation_accuracy_floor",
    )
    aggregate_results = {
        strength: {policy: aggregate(runs, policy) for policy in policies}
        for strength, runs in runs_by_strength.items()
    }
    result = {
        "protocol": {
            "status": "post-hoc held-out comparison; not a fresh final test",
            "model": "shared frozen encoder with separate Nano/Mini correctness heads",
            "outputs": ["P(Nano correct|x)", "P(Mini correct|x)"],
            "data_loss": (
                "binary cross entropy over both correctness labels, with each "
                "head weighted by 1 / expected query cost and normalized per batch"
            ),
            "prior": (
                "independent Gaussian priors on head intercept log-odds; "
                "precision in mean-loss units is prior_strength/train_size"
            ),
            "routing_score": "P(Mini correct|x)-P(Nano correct|x)",
            "train_counts": TRAIN_COUNTS,
            "heldout_used_for_training_or_selection": 0,
            "openai_api_calls": 0,
        },
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "heldout_counts": dict(Counter(row["bucket"] for row in heldout)),
        "baselines": {
            "all_nano": evaluate(heldout, [0] * len(heldout)),
            "all_mini": evaluate(heldout, [1] * len(heldout)),
        },
        "aggregate": aggregate_results,
        "runs": runs_by_strength,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"baselines": result["baselines"], "aggregate": aggregate_results}, indent=2))
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
