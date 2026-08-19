#!/usr/bin/env python3
"""Outcome-only direct router with regret CE plus pairwise ranking.

This experiment deliberately excludes semantic-label supervision and model
answers from the router.  The only target is the route outcome derived offline:
mini for mini-only/both-wrong and nano for both-correct/nano-only.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from run_sembench_accuracy_targeted_router import (
    REGRET_WEIGHTS,
    evaluate_scores,
    loader,
    make_split,
    probabilities,
    select_threshold,
)
from run_sembench_qa50_softprompt_router import deduplicate
from train_sembench_balanced_direct_router import (
    SoftPromptDirectRouter,
    evaluate,
    load_examples,
)


def outcome_pairwise_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    buckets: list[str],
    pairwise_weight: float,
    margin: float,
) -> tuple[torch.Tensor, float, float]:
    """Combine regret-weighted CE with weighted mini-needed-vs-safe ranking."""
    per_example = F.cross_entropy(logits, labels, reduction="none")
    weights = torch.tensor(
        [REGRET_WEIGHTS[bucket] for bucket in buckets], dtype=logits.dtype
    )
    regret_ce = (weights * per_example).sum() / weights.sum()

    mini_indices = torch.nonzero(labels == 1, as_tuple=False).flatten()
    nano_indices = torch.nonzero(labels == 0, as_tuple=False).flatten()
    if pairwise_weight <= 0.0 or not len(mini_indices) or not len(nano_indices):
        return regret_ce, float(regret_ce.detach()), 0.0

    mini_scores = logits[mini_indices, 1] - logits[mini_indices, 0]
    nano_scores = logits[nano_indices, 1] - logits[nano_indices, 0]
    score_gaps = mini_scores[:, None] - nano_scores[None, :]

    mini_weights = weights[mini_indices]
    nano_weights = weights[nano_indices]
    pair_weights = torch.sqrt(mini_weights[:, None] * nano_weights[None, :])
    pair_losses = F.softplus(margin - score_gaps)
    ranking = (pair_weights * pair_losses).sum() / pair_weights.sum()
    total = regret_ce + pairwise_weight * ranking
    return total, float(regret_ce.detach()), float(ranking.detach())


def capture_trainable(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def restore_trainable(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in state:
                parameter.copy_(state[name])


def run_one(
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    test: list[dict[str, Any]],
    seed: int,
    pairwise_weight: float,
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

    best: dict[str, Any] | None = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_ranking = 0.0
        seen = 0
        ranking_batches = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss, regret_ce, ranking = outcome_pairwise_loss(
                logits,
                batch["labels"],
                batch["buckets"],
                pairwise_weight,
                args.margin,
            )
            loss.backward()
            optimizer.step()
            count = len(batch["labels"])
            total_loss += float(loss.detach()) * count
            total_ce += regret_ce * count
            total_ranking += ranking
            seen += count
            ranking_batches += int(ranking > 0.0)

        validation_scores = probabilities(model, validation_loader)
        calibration = select_threshold(validation, validation_scores, "max_accuracy")
        metrics = calibration["validation_metrics"]
        key = (metrics["selected_accuracy"], -metrics["mini_calls"])
        history.append({
            "epoch": epoch,
            "train_loss": total_loss / max(seen, 1),
            "regret_ce": total_ce / max(seen, 1),
            "mean_active_batch_ranking": (
                total_ranking / ranking_batches if ranking_batches else 0.0
            ),
            "threshold": calibration["threshold"],
            "validation_accuracy": metrics["selected_accuracy"],
            "validation_mini_rate": metrics["mini_rate"],
            "validation_critical_recall": metrics["mini_only_recall"],
        })
        if best is None or key > best["key"]:
            best = {
                "key": key,
                "epoch": epoch,
                "state": copy.deepcopy(capture_trainable(model)),
                "calibration": calibration,
            }

    assert best is not None
    restore_trainable(model, best["state"])
    test_scores = probabilities(model, test_loader)
    return {
        "seed": seed,
        "pairwise_weight": pairwise_weight,
        "best_epoch": best["epoch"],
        "threshold": best["calibration"]["threshold"],
        "validation": best["calibration"]["validation_metrics"],
        "test": evaluate_scores(test, test_scores, best["calibration"]["threshold"]),
        "history": history,
    }


def aggregate(runs: list[dict[str, Any]], field: str) -> dict[str, float]:
    values = [float(run["test"][field]) for run in runs]
    return {"mean": mean(values), "std": pstdev(values)}


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
    parser.add_argument(
        "--pairwise-weights", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0]
    )
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--output", type=Path,
        default=root / "sembench_movie_router/outcome_pairwise_router.json",
    )
    args = parser.parse_args()

    examples = deduplicate(load_examples(args.reviews, args.cache))
    splits = {seed: make_split(examples, seed) for seed in args.seeds}
    by_weight: dict[str, list[dict[str, Any]]] = {}
    for pairwise_weight in args.pairwise_weights:
        key = str(pairwise_weight)
        by_weight[key] = []
        for seed in args.seeds:
            train, validation, test, _ = splits[seed]
            print(
                f"pairwise_weight={pairwise_weight:g} seed={seed}: outcome-only direct",
                flush=True,
            )
            run = run_one(
                train, validation, test, seed, pairwise_weight, args
            )
            by_weight[key].append(run)
            metrics = run["test"]
            print(
                f"  accuracy={metrics['selected_accuracy']:.4f} "
                f"mini_rate={metrics['mini_rate']:.4f} "
                f"critical_recall={metrics['mini_only_recall']:.4f} "
                f"both_wrong={metrics['both_wrong_mini_rate']:.4f}",
                flush=True,
            )

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
    summaries = {
        key: {field: aggregate(runs, field) for field in fields}
        for key, runs in by_weight.items()
    }
    ranking = sorted(
        [
            {
                "pairwise_weight": key,
                "accuracy": summary["selected_accuracy"]["mean"],
                "accuracy_std": summary["selected_accuracy"]["std"],
                "mini_rate": summary["mini_rate"]["mean"],
                "mini_saving": summary["mini_saving_vs_all_mini"]["mean"],
                "critical_recall": summary["mini_only_recall"]["mean"],
                "both_wrong_mini_rate": summary["both_wrong_mini_rate"]["mean"],
            }
            for key, summary in summaries.items()
        ],
        key=lambda row: (row["accuracy"], -row["mini_rate"]),
        reverse=True,
    )
    first_test = splits[args.seeds[0]][2]
    result = {
        "config": {
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "supervision": "outcome-only; no semantic-label target; no model answer input",
            "openai_api_calls": 0,
        },
        "baselines": {
            "all_nano": evaluate(first_test, [0] * len(first_test)),
            "all_mini": evaluate(first_test, [1] * len(first_test)),
        },
        "ranking": ranking,
        "aggregate": summaries,
        "runs": by_weight,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"baselines": result["baselines"], "ranking": ranking}, indent=2))
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
