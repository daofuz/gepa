#!/usr/bin/env python3
"""Five-fold OOF calibration and ensembling for the Computer QA router."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from optimize_ollama_router_gepa import load_examples, outcome_bucket  # noqa: E402
from train_softprompt_router import (  # noqa: E402
    ROUTE_TO_LABEL,
    RoutingDataset,
    SoftPromptRouter,
    collate_batch,
)
from scripts.run_computer_qa_regret_softprompt import (  # noqa: E402
    REGRET_WEIGHTS,
    compact_summary,
    routes,
    select_max_accuracy_threshold,
)


def stratified_folds(examples: list[Any], fold_count: int, seed: int) -> list[list[Any]]:
    rng = random.Random(seed)
    folds: list[list[Any]] = [[] for _ in range(fold_count)]
    buckets = sorted({outcome_bucket(example) for example in examples})
    for bucket in buckets:
        values = [example for example in examples if outcome_bucket(example) == bucket]
        rng.shuffle(values)
        for index, example in enumerate(values):
            folds[index % fold_count].append(example)
    for fold in folds:
        rng.shuffle(fold)
    return folds


def make_loader(
    examples: list[Any], tokenizer: Any, max_length: int, batch_size: int, shuffle: bool
) -> DataLoader:
    dataset = RoutingDataset(
        examples,
        tokenizer,
        max_length=max_length,
        score_mode="risk_averse_utility",
        include_nano_answer=False,
        nano_response_max_chars=0,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda items: collate_batch(items, int(tokenizer.pad_token_id)),
    )


def probabilities(model: SoftPromptRouter, loader: DataLoader) -> list[float]:
    values: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"], batch["attention_mask"])
            values.extend(torch.softmax(logits, dim=-1)[:, ROUTE_TO_LABEL["mini"]].tolist())
    return values


def train_fold(
    train: list[Any],
    validation: list[Any],
    test: list[Any],
    tokenizer: Any,
    fold_index: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    seed = args.seed + fold_index * 101
    random.seed(seed)
    torch.manual_seed(seed)
    model = SoftPromptRouter(
        args.hf_model,
        soft_prompt_tokens=args.prompt_tokens,
        local_files_only=True,
        train_backbone=False,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_loader = make_loader(train, tokenizer, args.max_length, args.batch_size, True)
    validation_loader = make_loader(
        validation, tokenizer, args.max_length, args.batch_size, False
    )
    test_loader = make_loader(test, tokenizer, args.max_length, args.batch_size, False)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        item_count = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], batch["attention_mask"])
            labels = batch["labels"]
            per_example = F.cross_entropy(logits, labels, reduction="none")
            indices = batch["indices"].tolist()
            weights = torch.tensor(
                [REGRET_WEIGHTS[outcome_bucket(train[index])] for index in indices],
                dtype=per_example.dtype,
            )
            loss = (weights * per_example).sum() / weights.sum()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(labels)
            item_count += len(labels)
        history.append({"epoch": epoch, "train_loss": loss_sum / max(item_count, 1)})

    validation_scores = probabilities(model, validation_loader)
    fold_threshold, _ = select_max_accuracy_threshold(validation, validation_scores)
    test_scores = probabilities(model, test_loader)
    return {
        "fold": fold_index,
        "seed": seed,
        "train_count": len(train),
        "validation_count": len(validation),
        "train_outcomes": dict(Counter(outcome_bucket(x) for x in train)),
        "validation_outcomes": dict(Counter(outcome_bucket(x) for x in validation)),
        "validation_ids": [x.question_id for x in validation],
        "validation_scores": validation_scores,
        "fold_threshold": fold_threshold,
        "fold_validation": compact_summary(
            validation, routes(validation_scores, fold_threshold)
        ),
        "test_scores": test_scores,
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mini-file", type=Path, default=ROOT / "computer science_result_mini.json")
    parser.add_argument("--nano-file", type=Path, default=ROOT / "computer science_result_nano.json")
    parser.add_argument("--hf-model", default="bert-base-uncased")
    parser.add_argument("--prompt-tokens", type=int, default=4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=505)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/computer_qa_regret_bert_base_oof5_ensemble.json",
    )
    args = parser.parse_args()

    all_examples = load_examples(
        args.mini_file,
        args.nano_file,
        mini_cost_model="gpt-4.1-mini",
        nano_cost_model="gpt-4.1-nano",
    )
    development = all_examples[:205]
    test = all_examples[205:]
    development_ids = {example.question_id for example in development}
    test_ids = {example.question_id for example in test}
    if development_ids & test_ids:
        raise AssertionError("Development and test IDs overlap")

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    folds = stratified_folds(development, args.folds, args.seed)
    fold_results = []
    for fold_index, validation in enumerate(folds):
        validation_ids = {example.question_id for example in validation}
        train = [example for example in development if example.question_id not in validation_ids]
        print(
            f"fold={fold_index + 1}/{args.folds} train={len(train)} "
            f"validation={len(validation)}",
            flush=True,
        )
        result = train_fold(train, validation, test, tokenizer, fold_index, args)
        fold_results.append(result)
        metrics = result["fold_validation"]
        print(
            f"  val_accuracy={metrics['accuracy']:.4f} "
            f"mini_rate={metrics['mini_rate']:.4f} "
            f"critical_recall={metrics['critical_recall']:.4f}",
            flush=True,
        )

    score_by_id: dict[int, float] = {}
    for validation, result in zip(folds, fold_results):
        score_by_id.update(
            {
                example.question_id: score
                for example, score in zip(validation, result["validation_scores"])
            }
        )
    if set(score_by_id) != development_ids:
        raise AssertionError("OOF scores do not cover the full first 205")
    oof_scores = [score_by_id[example.question_id] for example in development]
    global_threshold, _ = select_max_accuracy_threshold(development, oof_scores)
    oof_summary = compact_summary(development, routes(oof_scores, global_threshold))

    averaged_test_scores = [
        sum(result["test_scores"][index] for result in fold_results) / args.folds
        for index in range(len(test))
    ]
    global_test = compact_summary(test, routes(averaged_test_scores, global_threshold))

    mean_margin_routes = []
    majority_vote_routes = []
    for index in range(len(test)):
        margins = [
            result["test_scores"][index] - result["fold_threshold"]
            for result in fold_results
        ]
        votes = [margin >= 0 for margin in margins]
        mean_margin_routes.append("mini" if sum(margins) / args.folds >= 0 else "nano")
        majority_vote_routes.append("mini" if sum(votes) >= (args.folds // 2 + 1) else "nano")

    output = {
        "config": {
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "loss": "regret-weighted cross entropy",
            "regret_weights": REGRET_WEIGHTS,
            "router_input": "question, options, category, and source only",
            "nano_answer_exposed": False,
            "checkpoint_policy": "fixed epoch count; no fold early stopping",
            "threshold_policy": "max answer accuracy on all 205 OOF scores; ties use fewer mini calls",
        },
        "development_outcomes": dict(Counter(outcome_bucket(x) for x in development)),
        "test_outcomes": dict(Counter(outcome_bucket(x) for x in test)),
        "global_oof_threshold": global_threshold,
        "oof_summary": oof_summary,
        "test": {
            "mean_probability_global_threshold": global_test,
            "mean_fold_margin": compact_summary(test, mean_margin_routes),
            "majority_fold_vote": compact_summary(test, majority_vote_routes),
        },
        "baselines": {
            "all_mini": compact_summary(test, ["mini"] * len(test)),
            "all_nano": compact_summary(test, ["nano"] * len(test)),
        },
        "folds": fold_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "global_oof_threshold": global_threshold,
        "oof_summary": oof_summary,
        "test": output["test"],
        "baselines": output["baselines"],
    }, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
