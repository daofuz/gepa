#!/usr/bin/env python3
"""Evaluate a Movie-trained direct router on Computer Science QA.

Two transfer settings are reported:
  * strict zero-shot: Movie weights and Movie validation threshold;
  * calibration-only: Movie weights, but threshold calibrated with 40 QA labels.

The router is never trained on QA in this script and never sees nano answers.
"""

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

from run_computer_qa_regret_softprompt import (
    compact_summary,
    make_loader as make_qa_loader,
    resolve_ids,
    routes,
    select_max_accuracy_threshold,
)
from run_sembench_accuracy_targeted_router import (
    capture_trainable,
    deduplicate,
    direct_loss,
    loader as make_movie_loader,
    make_split,
    probabilities as movie_probabilities,
    restore_trainable,
    select_threshold as select_movie_threshold,
)
from train_sembench_balanced_direct_router import (
    SoftPromptDirectRouter,
    load_examples as load_movie_examples,
)
from optimize_ollama_router_gepa import load_examples as load_qa_examples, outcome_bucket


ROOT = Path(__file__).resolve().parents[1]


def qa_probabilities(model: torch.nn.Module, data_loader: Any) -> list[float]:
    values: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            logits = model(batch["input_ids"], batch["attention_mask"])
            values.extend(torch.softmax(logits, dim=-1)[:, 1].tolist())
    return values


def train_movie_and_transfer(
    movie_examples: list[dict[str, Any]],
    qa_validation: list[Any],
    qa_test: list[Any],
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    train, validation, _, split_metadata = make_split(movie_examples, seed)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, local_files_only=True)
    model = SoftPromptDirectRouter(args.hf_model, args.prompt_tokens)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_loader = make_movie_loader(train, tokenizer, args, "route", True)
    validation_loader = make_movie_loader(validation, tokenizer, args, "route", False)

    best_key: tuple[float, int] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_threshold = 0.5
    best_epoch = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = direct_loss(
                logits,
                batch["labels"],
                batch["buckets"],
                "regret_ce",
                epoch,
                args.epochs,
            )
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(batch["labels"])
        scores = movie_probabilities(model, validation_loader)
        calibration = select_movie_threshold(validation, scores, "max_accuracy")
        metrics = calibration["validation_metrics"]
        key = (metrics["selected_accuracy"], -metrics["mini_calls"])
        history.append({
            "epoch": epoch,
            "train_loss": loss_sum / len(train),
            "movie_validation_accuracy": metrics["selected_accuracy"],
            "movie_validation_mini_rate": metrics["mini_rate"],
            "movie_threshold": calibration["threshold"],
        })
        if best_key is None or key > best_key:
            best_key = key
            best_state = capture_trainable(model)
            best_threshold = calibration["threshold"]
            best_epoch = epoch

    assert best_state is not None
    restore_trainable(model, best_state)
    qa_validation_loader = make_qa_loader(qa_validation, tokenizer, args, shuffle=False)
    qa_test_loader = make_qa_loader(qa_test, tokenizer, args, shuffle=False)
    validation_scores = qa_probabilities(model, qa_validation_loader)
    test_scores = qa_probabilities(model, qa_test_loader)

    _, qa_calibration_summary = select_max_accuracy_threshold(
        qa_validation, validation_scores
    )
    qa_threshold, _ = select_max_accuracy_threshold(qa_validation, validation_scores)
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "movie_threshold": best_threshold,
        "qa_calibrated_threshold": qa_threshold,
        "movie_split": split_metadata,
        "strict_zero_shot": compact_summary(
            qa_test, routes(test_scores, best_threshold)
        ),
        "qa_threshold_calibration_only": compact_summary(
            qa_test, routes(test_scores, qa_threshold)
        ),
        "qa_validation_calibration": compact_summary(
            qa_validation, routes(validation_scores, qa_threshold)
        ),
        "history": history,
    }


def aggregate(runs: list[dict[str, Any]], key: str) -> dict[str, dict[str, float]]:
    metrics = [
        "accuracy",
        "mini_rate",
        "cost_saving_rate_vs_all_mini",
        "critical_recall",
        "both_wrong_mini_rate",
    ]
    return {
        metric: {
            "mean": mean(float(run[key][metric]) for run in runs),
            "std": pstdev(float(run[key][metric]) for run in runs),
        }
        for metric in metrics
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--movie-reviews", type=Path,
        default=ROOT / "sembench/files/movie/data/sf_2000/Reviews.csv",
    )
    parser.add_argument(
        "--movie-cache", type=Path,
        default=ROOT / "sembench_movie_router/model_outputs.json",
    )
    parser.add_argument(
        "--qa-split", type=Path,
        default=ROOT / "routing_split_computer_qa_regret_train165_val40.json",
    )
    parser.add_argument(
        "--mini-file", type=Path, default=ROOT / "computer science_result_mini.json"
    )
    parser.add_argument(
        "--nano-file", type=Path, default=ROOT / "computer science_result_nano.json"
    )
    parser.add_argument("--hf-model", default="distilbert-base-uncased")
    parser.add_argument("--prompt-tokens", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[505, 606, 707])
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "outputs/movie_router_zero_shot_computer_qa.json",
    )
    args = parser.parse_args()

    movie_examples = deduplicate(
        load_movie_examples(args.movie_reviews, args.movie_cache)
    )
    qa_examples = load_qa_examples(
        args.mini_file,
        args.nano_file,
        mini_cost_model="gpt-4.1-mini",
        nano_cost_model="gpt-4.1-nano",
    )
    split = json.loads(args.qa_split.read_text(encoding="utf-8"))
    qa_validation = resolve_ids(qa_examples, split["val_ids"])
    qa_test = resolve_ids(qa_examples, split["test_ids"])

    runs = []
    for seed in args.seeds:
        print(f"seed={seed}: train on Movie, transfer to Computer QA", flush=True)
        run = train_movie_and_transfer(
            movie_examples, qa_validation, qa_test, seed, args
        )
        runs.append(run)
        print(
            f"  strict_zero_shot_accuracy={run['strict_zero_shot']['accuracy']:.4f} "
            f"strict_mini_rate={run['strict_zero_shot']['mini_rate']:.4f} "
            f"calibration_only_accuracy={run['qa_threshold_calibration_only']['accuracy']:.4f} "
            f"calibrated_mini_rate={run['qa_threshold_calibration_only']['mini_rate']:.4f}",
            flush=True,
        )

    result = {
        "config": {
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "training_task": "SemBench Movie positive/negative classification",
            "transfer_task": "Computer Science multiple-choice QA",
            "loss": "regret-weighted cross entropy",
            "nano_answer_exposed": False,
            "openai_api_calls": 0,
        },
        "qa_test_outcomes": dict(Counter(outcome_bucket(x) for x in qa_test)),
        "baselines": {
            "all_nano": compact_summary(qa_test, ["nano"] * len(qa_test)),
            "all_mini": compact_summary(qa_test, ["mini"] * len(qa_test)),
        },
        "aggregate": {
            "strict_zero_shot": aggregate(runs, "strict_zero_shot"),
            "qa_threshold_calibration_only": aggregate(
                runs, "qa_threshold_calibration_only"
            ),
        },
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "baselines": result["baselines"],
        "aggregate": result["aggregate"],
    }, indent=2), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
