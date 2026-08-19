#!/usr/bin/env python3
"""Random-100 soft-prompt routing with several Mini-better priors.

This controlled ablation uses the same naturally random training examples for
every prior configuration.  A two-head router predicts Nano and Mini
correctness separately.  The data term is inverse-expected-query-cost weighted
BCE.  Priors encode only the declared belief that Mini is usually more
accurate; validation-only thresholds determine the deployed Mini rate.

No answer-model API calls are made.  Evaluation on the existing held-out 200
is post-hoc and must not be described as a fresh final test.
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
from run_sembench_random100_selection import random_train
from run_sembench_softprompt_mini_prior_ablation import expected_costs
from run_sembench_train100_ratio_sweep import make_orders
from train_sembench_balanced_direct_router import evaluate, load_examples


def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def disagreement_train(
    orders: dict[str, list[dict[str, Any]]], seed: int
) -> list[dict[str, Any]]:
    """Select observable disagreements first, then random agreement coverage."""
    pool = [row for values in orders.values() for row in values]
    disagreements = [row for row in pool if row["nano_pred"] != row["mini_pred"]]
    agreements = [row for row in pool if row["nano_pred"] == row["mini_pred"]]
    rng = random.Random(seed + 32452843)
    rng.shuffle(disagreements)
    selected = disagreements[:100]
    if len(selected) < 100:
        selected.extend(rng.sample(agreements, 100 - len(selected)))
    rng.shuffle(selected)
    if len({row["review_id"] for row in selected}) != 100:
        raise AssertionError("Disagreement-selected training sample is not unique")
    return selected


def trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def restore(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in state:
                parameter.copy_(state[name])


def bernoulli_kl(target: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(predicted.dtype).eps
    predicted = predicted.clamp(eps, 1.0 - eps)
    return (
        target * (target.log() - predicted.log())
        + (1.0 - target)
        * ((1.0 - target).log() - (1.0 - predicted).log())
    )


def prior_loss(
    model: base.TwoHeadCorrectnessRouter,
    logits: torch.Tensor,
    method: str,
    strength: float,
    train_size: int,
    prior_nano: float,
    prior_mini: float,
    log_odds_margin: float,
) -> torch.Tensor:
    zero = logits.sum() * 0.0
    if method == "none" or strength == 0.0:
        return zero
    scale = strength / train_size
    if method == "bias_map":
        centers = logits.new_tensor([logit(prior_nano), logit(prior_mini)])
        return 0.5 * scale * (
            model.correctness_heads.bias - centers
        ).pow(2).sum()
    if method == "mean_accuracy_kl":
        targets = logits.new_tensor([prior_nano, prior_mini])
        means = torch.sigmoid(logits).mean(dim=0)
        return scale * bernoulli_kl(targets, means).sum()
    differences = logits[:, 1] - logits[:, 0]
    if method == "mean_dominance":
        return scale * F.softplus(log_odds_margin - differences.mean())
    if method == "pointwise_dominance":
        return scale * F.softplus(log_odds_margin - differences).mean()
    raise ValueError(f"Unknown prior method: {method}")


def policies(
    validation: list[dict[str, Any]], scores: list[float]
) -> dict[str, dict[str, Any]]:
    maximum = select_threshold(validation, scores, "max_accuracy")
    floor = select_threshold(validation, scores, "match_all_mini")
    return {
        "direct_uplift": {
            "threshold": 0.0,
            "validation": evaluate_scores(validation, scores, 0.0),
        },
        "validation_max_accuracy": {
            "threshold": maximum["threshold"],
            "validation": maximum["validation_metrics"],
        },
        "validation_accuracy_floor": {
            "threshold": floor["threshold"],
            "validation": floor["validation_metrics"],
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
    method: str,
    strength: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, local_files_only=True)
    model = base.TwoHeadCorrectnessRouter(args.hf_model, args.prompt_tokens)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_loader = base.loader(train, tokenizer, args.max_length, args.batch_size, True)
    validation_loader = base.loader(
        validation, tokenizer, args.max_length, args.batch_size, False
    )
    test_loader = base.loader(test, tokenizer, args.max_length, args.batch_size, False)
    costs = expected_costs(train, cache)
    inverse_cost = torch.tensor(
        [1.0 / costs["nano"], 1.0 / costs["mini"]], dtype=torch.float
    )

    best: dict[str, dict[str, Any]] = {}
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "w_bce": 0.0, "prior": 0.0, "count": 0}
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], batch["attention_mask"])
            raw = F.binary_cross_entropy_with_logits(
                logits, batch["labels"], reduction="none"
            )
            w_bce = (raw * inverse_cost).sum() / (
                len(batch["labels"]) * inverse_cost.sum()
            )
            prior = prior_loss(
                model,
                logits,
                method,
                strength,
                len(train),
                args.prior_accuracy_nano,
                args.prior_accuracy_mini,
                args.log_odds_margin,
            )
            loss = w_bce + prior
            loss.backward()
            optimizer.step()
            count = len(batch["labels"])
            totals["loss"] += float(loss.detach()) * count
            totals["w_bce"] += float(w_bce.detach()) * count
            totals["prior"] += float(prior.detach()) * count
            totals["count"] += count

        q_nano, q_mini = base.correctness_probabilities(model, validation_loader)
        scores = base.score(q_nano, q_mini)
        candidates = policies(validation, scores)
        trace = {
            "epoch": epoch,
            "train_loss": totals["loss"] / totals["count"],
            "weighted_bce": totals["w_bce"] / totals["count"],
            "prior": totals["prior"] / totals["count"],
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
                    "state": copy.deepcopy(trainable_state(model)),
                }
        history.append(trace)

    output: dict[str, Any] = {}
    for policy, chosen in best.items():
        restore(model, chosen["state"])
        q_nano, q_mini = base.correctness_probabilities(model, test_loader)
        scores = base.score(q_nano, q_mini)
        output[policy] = {
            "best_epoch": chosen["epoch"],
            "threshold": chosen["threshold"],
            "validation": chosen["validation"],
            "test": evaluate_scores(test, scores, chosen["threshold"]),
            "mean_q_nano": mean(q_nano),
            "mean_q_mini": mean(q_mini),
            "mean_predicted_uplift": mean(scores),
        }
    return {
        "seed": seed,
        "method": method,
        "strength": strength,
        "train_counts": dict(Counter(row["bucket"] for row in train)),
        "empirical_train_accuracy": {
            "nano": mean(float(row["nano_pred"] == row["gold"]) for row in train),
            "mini": mean(float(row["mini_pred"] == row["gold"]) for row in train),
        },
        "expected_costs": costs,
        "normalized_head_weights": {
            "nano": float(inverse_cost[0] / inverse_cost.sum()),
            "mini": float(inverse_cost[1] / inverse_cost.sum()),
        },
        "policies": output,
        "history": history,
    }


def aggregate(runs: list[dict[str, Any]], policy: str) -> dict[str, Any]:
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
    result: dict[str, Any] = {}
    for field in fields:
        values = [float(run["policies"][policy]["test"][field]) for run in runs]
        result[field] = {"mean": mean(values), "std": pstdev(values)}
    for field in ("threshold", "mean_q_nano", "mean_q_mini", "mean_predicted_uplift"):
        values = [float(run["policies"][policy][field]) for run in runs]
        result[field] = {"mean": mean(values), "std": pstdev(values)}
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
    parser.add_argument("--log-odds-margin", type=float, default=0.25)
    parser.add_argument(
        "--training-selection",
        choices=("random", "disagreement"),
        default="random",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["none", "bias_map", "mean_accuracy_kl", "mean_dominance", "pointwise_dominance"],
    )
    parser.add_argument("--strengths", type=float, nargs="+", default=[20.0, 100.0])
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "outputs/sembench_random100_prior_methods.json",
    )
    args = parser.parse_args()
    if not 0 < args.prior_accuracy_nano < args.prior_accuracy_mini < 1:
        raise ValueError("Require 0 < Nano prior accuracy < Mini prior accuracy < 1")

    heldout_ids = set(json.loads(args.manifest.read_text(encoding="utf-8"))["review_ids"])
    cache = json.loads(args.cache.read_text(encoding="utf-8"))
    examples = deduplicate(load_examples(args.reviews, args.cache))
    heldout = [row for row in examples if row["review_id"] in heldout_ids]
    historical = [row for row in examples if row["review_id"] not in heldout_ids]
    if len(heldout) != 200 or len(historical) != 812:
        raise AssertionError("Expected 812 historical and 200 held-out reviews")

    configurations: list[tuple[str, float]] = [("none", 0.0)]
    configurations.extend(
        (method, strength)
        for method in args.methods
        if method != "none"
        for strength in args.strengths
    )
    runs_by_config: dict[str, list[dict[str, Any]]] = {}
    for method, strength in configurations:
        key = f"{method}_k{strength:g}" if method != "none" else "none"
        runs_by_config[key] = []
        for seed in args.seeds:
            orders, validation, _ = make_orders(historical, seed)
            train = (
                random_train(orders, seed)
                if args.training_selection == "random"
                else disagreement_train(orders, seed)
            )
            if {r["review_id"] for r in train} & {r["review_id"] for r in validation}:
                raise AssertionError("Train/validation leakage")
            print(f"config={key} seed={seed} counts={dict(Counter(r['bucket'] for r in train))}", flush=True)
            run = run_one(train, validation, heldout, cache, seed, method, strength, args)
            runs_by_config[key].append(run)
            metrics = run["policies"]["validation_max_accuracy"]["test"]
            print(
                f"  maxacc: accuracy={metrics['selected_accuracy']:.4f} "
                f"saving={metrics['mini_saving_vs_all_mini']:.4f} "
                f"MO_recall={metrics['mini_only_recall']:.4f}",
                flush=True,
            )

    policy_names = ("direct_uplift", "validation_max_accuracy", "validation_accuracy_floor")
    summary = {
        key: {policy: aggregate(runs, policy) for policy in policy_names}
        for key, runs in runs_by_config.items()
    }
    result = {
        "protocol": {
            "status": "post-hoc held-out comparison; not a fresh final test",
            "training_selection": args.training_selection,
            "model": "frozen DistilBERT, 8 soft tokens, two correctness heads",
            "data_loss": "inverse-expected-query-cost weighted BCE over Nano/Mini correctness",
            "prior_belief": "Mini has higher base correctness than Nano",
            "prior_methods": {
                "bias_map": "Gaussian MAP on correctness-head intercept log-odds",
                "mean_accuracy_kl": "KL from declared correctness base rates to batch-mean predictions",
                "mean_dominance": "softplus penalty if mean Mini log-odds do not exceed Nano",
                "pointwise_dominance": "softplus penalty for each input whose Mini log-odds do not exceed Nano",
            },
            "heldout_used_for_training_or_selection": 0,
            "answer_model_api_calls": 0,
        },
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "heldout_counts": dict(Counter(row["bucket"] for row in heldout)),
        "baselines": {
            "all_nano": evaluate(heldout, [0] * len(heldout)),
            "all_mini": evaluate(heldout, [1] * len(heldout)),
        },
        "summary": summary,
        "runs": runs_by_config,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
