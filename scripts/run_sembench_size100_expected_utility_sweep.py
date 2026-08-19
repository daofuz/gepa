#!/usr/bin/env python3
"""QA-style expected-utility soft-prompt sweep on SemBench Movie.

This keeps the size-100 splits from the cross-entropy sweep fixed and changes
the learning objective to:

    -expected_utility + 0.4 * P(mini) + 2.0 * (P(mini) - 0.25)^2

The decision threshold is fixed at 0.45.  Checkpoints are selected using
validation utility plus 0.5 times the nano-route rate as a cost-saving proxy.
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
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from run_sembench_qa50_softprompt_router import deduplicate, make_loader, probabilities
from run_sembench_size100_prompt_sweep import make_size100_split
from train_sembench_balanced_direct_router import (
    BUCKETS,
    UTILITY,
    SoftPromptDirectRouter,
    evaluate,
    load_examples,
)


def route_score_tensor(
    examples: list[dict[str, Any]], device: torch.device | None = None
) -> torch.Tensor:
    return torch.tensor(
        [[UTILITY[(example["bucket"], "nano")], UTILITY[(example["bucket"], "mini")]]
         for example in examples],
        dtype=torch.float,
        device=device,
    )


def train_configuration(
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    test: list[dict[str, Any]],
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
        weight_decay=args.weight_decay,
    )
    local_args = copy.copy(args)
    local_args.prompt_tokens = prompt_tokens
    train_loader: DataLoader = make_loader(train, tokenizer, local_args, shuffle=True)
    validation_loader = make_loader(validation, tokenizer, local_args, shuffle=False)
    test_loader = make_loader(test, tokenizer, local_args, shuffle=False)

    # RouterDataset preserves the original example index for an unshuffled
    # dataset, but the train loader shuffles.  Construct scores from the binary
    # labels in each batch: label=mini corresponds to mini-only/both-wrong.
    # Those two buckets have different utilities, so attach scores by review ID
    # through a dataset wrapper rather than deriving them from labels.
    score_by_id = {
        example["review_id"]: [
            UTILITY[(example["bucket"], "nano")],
            UTILITY[(example["bucket"], "mini")],
        ]
        for example in train
    }

    # Rebuild the train loader with route scores included in each item.
    class UtilityDataset(torch.utils.data.Dataset):
        def __init__(self) -> None:
            from train_sembench_balanced_direct_router import RouterDataset

            self.base = RouterDataset(train, tokenizer, args.max_length)

        def __len__(self) -> int:
            return len(self.base)

        def __getitem__(self, index: int) -> dict[str, Any]:
            item = self.base[index]
            example = train[index]
            return {**item, "route_scores": score_by_id[example["review_id"]]}

    from train_sembench_balanced_direct_router import collate

    def utility_collate(items: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        base = collate(items, tokenizer.pad_token_id)
        base["route_scores"] = torch.tensor(
            [item["route_scores"] for item in items], dtype=torch.float
        )
        return base

    train_loader = DataLoader(
        UtilityDataset(),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=utility_collate,
    )

    best_key: tuple[float, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        expected_utility_sum = 0.0
        predicted_mini_sum = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], batch["attention_mask"])
            probs = torch.softmax(logits, dim=-1)
            route_scores = batch["route_scores"]
            expected_utility = (probs * route_scores).sum(dim=-1).mean()
            predicted_mini_rate = probs[:, 1].mean()
            loss = (
                -expected_utility
                + args.mini_penalty * predicted_mini_rate
                + args.mini_rate_penalty
                * (predicted_mini_rate - args.target_mini_rate).pow(2)
            )
            loss.backward()
            optimizer.step()
            batch_size = len(batch["labels"])
            loss_sum += float(loss.item()) * batch_size
            expected_utility_sum += float(expected_utility.item()) * batch_size
            predicted_mini_sum += float(predicted_mini_rate.item()) * batch_size

        validation_probs = probabilities(model, validation_loader)
        validation_predictions = [
            int(value >= args.decision_threshold) for value in validation_probs
        ]
        validation_metrics = evaluate(validation, validation_predictions)
        nano_route_saving_proxy = 1.0 - validation_metrics["mini_rate"]
        selection_score = (
            validation_metrics["mean_utility"]
            + args.selection_cost_weight * nano_route_saving_proxy
        )
        train_probs = probabilities(model, make_loader(train, tokenizer, local_args, shuffle=False))
        train_predictions = [int(value >= args.decision_threshold) for value in train_probs]
        train_metrics = evaluate(train, train_predictions)
        history.append({
            "epoch": epoch,
            "train_loss": loss_sum / len(train),
            "train_soft_expected_utility": expected_utility_sum / len(train),
            "train_predicted_mini_probability": predicted_mini_sum / len(train),
            "train_discrete_utility": train_metrics["mean_utility"],
            "validation_utility": validation_metrics["mean_utility"],
            "validation_selected_accuracy": validation_metrics["selected_accuracy"],
            "validation_mini_rate": validation_metrics["mini_rate"],
            "validation_nano_route_saving_proxy": nano_route_saving_proxy,
            "validation_selection_score": selection_score,
        })
        key = (selection_score, train_metrics["mean_utility"])
        if best_key is None or key > best_key:
            best_key = key
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

    assert best_state is not None and best_key is not None
    model.load_state_dict(best_state)
    validation_probs = probabilities(model, validation_loader)
    test_probs = probabilities(model, test_loader)
    test_predictions = [int(value >= args.decision_threshold) for value in test_probs]
    test_metrics = evaluate(test, test_predictions)

    validation_threshold_curve = []
    threshold_curve = []
    for step in range(0, 101):
        threshold = step / 100
        validation_metrics = evaluate(
            validation, [int(value >= threshold) for value in validation_probs]
        )
        validation_threshold_curve.append({
            "threshold": threshold,
            "selected_accuracy": validation_metrics["selected_accuracy"],
            "mean_utility": validation_metrics["mean_utility"],
            "mini_rate": validation_metrics["mini_rate"],
            "mini_only_recall": validation_metrics["mini_only_recall"],
            "both_wrong_mini_rate": validation_metrics["both_wrong_mini_rate"],
            "selection_score": validation_metrics["mean_utility"]
            + args.selection_cost_weight * (1.0 - validation_metrics["mini_rate"]),
        })
        metrics = evaluate(test, [int(value >= threshold) for value in test_probs])
        threshold_curve.append({
            "threshold": threshold,
            "selected_accuracy": metrics["selected_accuracy"],
            "mean_utility": metrics["mean_utility"],
            "mini_rate": metrics["mini_rate"],
            "mini_only_recall": metrics["mini_only_recall"],
            "both_wrong_mini_rate": metrics["both_wrong_mini_rate"],
        })
    return {
        "seed": seed,
        "prompt_tokens": prompt_tokens,
        "trainable_parameters": prompt_tokens * 768 + 768 * 2 + 2,
        "best_epoch": best_epoch,
        "best_validation_selection_score": best_key[0],
        "test": test_metrics,
        "history": history,
        "validation_threshold_curve": validation_threshold_curve,
        "test_threshold_curve": threshold_curve,
    }


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    metrics = [
        "selected_accuracy",
        "mean_utility",
        "mini_rate",
        "route_accuracy",
        "mini_only_recall",
        "both_wrong_mini_rate",
        "critical_misroutes",
        "both_wrong_cheap_routes",
        "unnecessary_mini",
    ]
    return {
        metric: {
            "mean": mean(float(run["test"][metric]) for run in runs),
            "std": pstdev(float(run["test"][metric]) for run in runs),
        }
        for metric in metrics
    }


def make_ranking(by_prompt: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    ranking = []
    for prompt_tokens, runs in by_prompt.items():
        summary = aggregate_runs(runs)
        ranking.append({
            "prompt_tokens": prompt_tokens,
            "trainable_parameters": prompt_tokens * 768 + 768 * 2 + 2,
            "mean_utility": summary["mean_utility"]["mean"],
            "std_utility": summary["mean_utility"]["std"],
            "mean_selected_accuracy": summary["selected_accuracy"]["mean"],
            "mean_mini_rate": summary["mini_rate"]["mean"],
            "mean_critical_recall": summary["mini_only_recall"]["mean"],
            "mean_both_wrong_mini_rate": summary["both_wrong_mini_rate"]["mean"],
        })
    return sorted(
        ranking,
        key=lambda row: (row["mean_utility"], row["mean_selected_accuracy"]),
        reverse=True,
    )


def tune_thresholds(by_prompt: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    """Choose one threshold per prompt length using validation only."""
    result: dict[str, Any] = {}
    metrics = [
        "selected_accuracy",
        "mean_utility",
        "mini_rate",
        "mini_only_recall",
        "both_wrong_mini_rate",
    ]
    for prompt_tokens, runs in by_prompt.items():
        rows = []
        for index in range(101):
            threshold = index / 100
            validation_scores = [
                run["validation_threshold_curve"][index]["selection_score"] for run in runs
            ]
            row: dict[str, Any] = {
                "threshold": threshold,
                "validation_selection_score_mean": mean(validation_scores),
                "validation_selection_score_std": pstdev(validation_scores),
            }
            for metric in metrics:
                values = [run["test_threshold_curve"][index][metric] for run in runs]
                row[f"test_{metric}_mean"] = mean(values)
                row[f"test_{metric}_std"] = pstdev(values)
            rows.append(row)
        selected = max(
            rows,
            key=lambda row: (row["validation_selection_score_mean"], row["threshold"]),
        )
        result[str(prompt_tokens)] = {
            "selection_rule": "maximize mean validation utility + 0.5 * validation nano rate",
            "selected_threshold": selected["threshold"],
            "selected_threshold_test_metrics": selected,
            "curve": rows,
        }
    return result

def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", type=Path, default=root / "sembench/files/movie/data/sf_2000/Reviews.csv")
    parser.add_argument("--cache", type=Path, default=root / "sembench_movie_router/model_outputs.json")
    parser.add_argument("--hf-model", default="distilbert-base-uncased")
    parser.add_argument("--prompt-tokens", type=int, nargs="+", default=[4, 8, 16, 32, 64])
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
        default=root / "sembench_movie_router/size100_expected_utility_prompt_sweep.json",
    )
    args = parser.parse_args()

    raw_examples = load_examples(args.reviews, args.cache)
    examples = deduplicate(raw_examples)
    available = Counter(example["bucket"] for example in examples)
    splits = {seed: make_size100_split(examples, seed) for seed in args.seeds}
    by_prompt: dict[int, list[dict[str, Any]]] = {}
    for prompt_tokens in args.prompt_tokens:
        by_prompt[prompt_tokens] = []
        for seed in args.seeds:
            train, validation, test, _ = splits[seed]
            print(f"prompt_tokens={prompt_tokens} seed={seed}: expected-utility training", flush=True)
            run = train_configuration(
                train, validation, test, seed, prompt_tokens, args
            )
            by_prompt[prompt_tokens].append(run)
            metrics = run["test"]
            print(
                f"prompt_tokens={prompt_tokens} seed={seed}: "
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
            "loss": "-expected_utility + mini_penalty*p_mini + mini_rate_penalty*(p_mini-target_mini_rate)^2",
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
        "ranking": make_ranking(by_prompt),
        "threshold_tuning": tune_thresholds(by_prompt),
        "aggregate": {
            str(prompt_tokens): aggregate_runs(runs)
            for prompt_tokens, runs in by_prompt.items()
        },
        "runs": {str(prompt_tokens): runs for prompt_tokens, runs in by_prompt.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "split": output["split_metadata"][str(args.seeds[0])],
        "ranking": output["ranking"],
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
