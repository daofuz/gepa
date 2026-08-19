#!/usr/bin/env python3
"""Compare validation threshold tie-breaks without using held-out examples."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from statistics import mean, pstdev
from types import SimpleNamespace
from typing import Any

import torch
from transformers import AutoTokenizer

from run_sembench_accuracy_targeted_router import (
    evaluate,
    evaluate_scores,
    loader,
    probabilities,
    threshold_candidates,
)
from run_sembench_outcome_pairwise_router import (
    capture_trainable,
    outcome_pairwise_loss,
    restore_trainable,
)
from run_sembench_qa50_softprompt_router import deduplicate
from run_sembench_residual_prompt_sweep import ResidualSoftPromptRouter
from run_sembench_train100_ratio_sweep import make_orders, select_train
from train_sembench_balanced_direct_router import (
    SoftPromptDirectRouter,
    load_examples,
)


TRAIN_COUNTS = {
    "both_correct": 50,
    "mini_only": 46,
    "nano_only": 3,
    "both_wrong": 1,
}

CONFIGS = {
    "plain_8tok_current_tiebreak": {
        "architecture": "plain",
        "prompt_tokens": 8,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "threshold_policy": "current",
    },
    "plain_8tok_conservative_tiebreak": {
        "architecture": "plain",
        "prompt_tokens": 8,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "threshold_policy": "conservative",
    },
    "residual_8tok_m128_conservative_tiebreak": {
        "architecture": "residual",
        "prompt_tokens": 8,
        "bottleneck": 128,
        "learning_rate": 0.005,
        "weight_decay": 0.01,
        "threshold_policy": "conservative",
    },
}


def select_threshold(
    examples: list[dict[str, Any]],
    scores: list[float],
    policy: str,
) -> dict[str, Any]:
    rows = []
    for threshold in threshold_candidates(scores):
        predictions = [int(value >= threshold) for value in scores]
        rows.append({"threshold": threshold, **evaluate(examples, predictions)})
    if policy == "current":
        key = lambda row: (
            row["selected_accuracy"],
            -row["mini_calls"],
            row["threshold"],
        )
    elif policy == "conservative":
        # Accuracy remains the primary validation objective.  For exact ties,
        # prefer avoiding mini-only errors and routing both-wrong to mini before
        # considering cost.
        key = lambda row: (
            row["selected_accuracy"],
            row["mini_only_recall"],
            row["both_wrong_mini_rate"],
            -row["mini_calls"],
            row["threshold"],
        )
    else:
        raise ValueError(policy)
    selected = max(rows, key=key)
    return {
        "threshold": selected["threshold"],
        "validation_metrics": {
            name: value for name, value in selected.items() if name != "threshold"
        },
    }


def build_model(model_name: str, config: dict[str, Any]) -> torch.nn.Module:
    if config["architecture"] == "plain":
        return SoftPromptDirectRouter(model_name, config["prompt_tokens"])
    return ResidualSoftPromptRouter(
        model_name, config["prompt_tokens"], config["bottleneck"]
    )


def run_one(
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    test: list[dict[str, Any]],
    seed: int,
    config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, local_files_only=True)
    model = build_model(args.hf_model, config)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    run_args = SimpleNamespace(
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    train_loader = loader(train, tokenizer, run_args, "route", True)
    validation_loader = loader(validation, tokenizer, run_args, "route", False)
    test_loader = loader(test, tokenizer, run_args, "route", False)

    best: dict[str, Any] | None = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss, _, _ = outcome_pairwise_loss(
                logits,
                batch["labels"],
                batch["buckets"],
                pairwise_weight=0.0,
                margin=1.0,
            )
            loss.backward()
            optimizer.step()
            count = len(batch["labels"])
            total_loss += float(loss.detach()) * count
            seen += count

        validation_scores = probabilities(model, validation_loader)
        calibration = select_threshold(
            validation,
            validation_scores,
            config["threshold_policy"],
        )
        metrics = calibration["validation_metrics"]
        if config["threshold_policy"] == "conservative":
            selection_key = (
                metrics["selected_accuracy"],
                metrics["mini_only_recall"],
                metrics["both_wrong_mini_rate"],
                -metrics["mini_calls"],
            )
        else:
            selection_key = (
                metrics["selected_accuracy"],
                -metrics["mini_calls"],
            )
        history.append({
            "epoch": epoch,
            "train_loss": total_loss / max(seen, 1),
            "threshold": calibration["threshold"],
            "validation_accuracy": metrics["selected_accuracy"],
            "validation_mini_rate": metrics["mini_rate"],
            "validation_critical_recall": metrics["mini_only_recall"],
            "validation_both_wrong_mini_rate": metrics["both_wrong_mini_rate"],
        })
        if best is None or selection_key > best["key"]:
            best = {
                "key": selection_key,
                "epoch": epoch,
                "state": copy.deepcopy(capture_trainable(model)),
                "calibration": calibration,
            }

    assert best is not None
    restore_trainable(model, best["state"])
    test_scores = probabilities(model, test_loader)
    return {
        "seed": seed,
        "best_epoch": best["epoch"],
        "threshold": best["calibration"]["threshold"],
        "validation": best["calibration"]["validation_metrics"],
        "test": evaluate_scores(
            test, test_scores, best["calibration"]["threshold"]
        ),
        "history": history,
    }


def aggregate(runs: list[dict[str, Any]], field: str) -> dict[str, float]:
    values = [float(run["test"][field]) for run in runs]
    return {"mean": mean(values), "std": pstdev(values)}


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
    parser.add_argument("--seeds", type=int, nargs="+", default=[505, 606, 707])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "sembench_movie_router/conservative_threshold_sweep.json",
    )
    args = parser.parse_args()

    heldout_ids = set(json.loads(
        args.untouched_manifest.read_text(encoding="utf-8")
    )["review_ids"])
    examples = deduplicate(load_examples(args.reviews, args.cache))
    historical = [
        example for example in examples if example["review_id"] not in heldout_ids
    ]
    if len(historical) != 812:
        raise AssertionError(f"Expected 812 historical examples, found {len(historical)}")
    per_seed = {seed: make_orders(historical, seed) for seed in args.seeds}

    runs_by_config: dict[str, list[dict[str, Any]]] = {}
    for name, config in CONFIGS.items():
        runs_by_config[name] = []
        for seed in args.seeds:
            orders, validation, historical_test = per_seed[seed]
            train = select_train(orders, TRAIN_COUNTS, seed)
            print(f"config={name} seed={seed}", flush=True)
            run = run_one(train, validation, historical_test, seed, config, args)
            runs_by_config[name].append(run)
            metrics = run["test"]
            print(
                f"  accuracy={metrics['selected_accuracy']:.4f} "
                f"saving={metrics['mini_saving_vs_all_mini']:.4f} "
                f"critical={metrics['mini_only_recall']:.4f} "
                f"both_wrong={metrics['both_wrong_mini_rate']:.4f}",
                flush=True,
            )

    fields = (
        "selected_accuracy",
        "accuracy_gap_vs_all_mini",
        "mini_saving_vs_all_mini",
        "mini_only_recall",
        "both_wrong_mini_rate",
    )
    summaries = {
        name: {field: aggregate(runs, field) for field in fields}
        for name, runs in runs_by_config.items()
    }
    ranking = sorted(
        [
            {
                "config": name,
                **CONFIGS[name],
                "accuracy": summary["selected_accuracy"]["mean"],
                "accuracy_std": summary["selected_accuracy"]["std"],
                "mini_saving": summary["mini_saving_vs_all_mini"]["mean"],
                "mini_saving_std": summary["mini_saving_vs_all_mini"]["std"],
                "critical_recall": summary["mini_only_recall"]["mean"],
                "both_wrong_mini_rate": summary["both_wrong_mini_rate"]["mean"],
            }
            for name, summary in summaries.items()
        ],
        key=lambda row: (
            row["accuracy"],
            row["critical_recall"],
            row["mini_saving"],
        ),
        reverse=True,
    )
    first_test = per_seed[args.seeds[0]][2]
    result = {
        "protocol": {
            "heldout_200_used": False,
            "train_counts": TRAIN_COUNTS,
            "purpose": "isolate validation threshold tie-break behavior",
            "conservative_priority": [
                "validation accuracy",
                "mini-only recall",
                "both-wrong mini rate",
                "fewer mini calls",
            ],
        },
        "baselines": {
            "all_nano": evaluate(first_test, [0] * len(first_test)),
            "all_mini": evaluate(first_test, [1] * len(first_test)),
        },
        "ranking": ranking,
        "aggregate": summaries,
        "runs": runs_by_config,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"baselines": result["baselines"], "ranking": ranking}, indent=2))
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
