#!/usr/bin/env python3
"""Accuracy-targeted direct routing and nano-first cascades on SemBench Movie.

The experiment uses 200 unique labeled reviews and an 82-review, review-disjoint
test set.  Thresholds are chosen from validation data only.  The main policy
minimizes mini calls subject to matching the validation accuracy of all-mini.

Compared methods:
  * direct soft prompt with ordinary cross entropy;
  * direct soft prompt with regret-weighted cross entropy;
  * direct soft prompt with regret-weighted focal loss;
  * direct soft prompt with CE pretraining and an accuracy-EU fine-tuning term;
  * TF-IDF sentiment verifier, used both as a direct router and nano-first cascade;
  * soft-prompt sentiment verifiers with weighted CE or focal loss, likewise used
    as direct routers and nano-first cascades.

No OpenAI calls are made.  Mini/nano outcomes come from the existing cache.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from run_sembench_qa50_softprompt_router import TEST_COUNTS, deduplicate
from train_sembench_balanced_direct_router import (
    BUCKETS,
    SoftPromptDirectRouter,
    evaluate,
    load_examples,
    text,
)


TRAIN_COUNTS = {
    "both_correct": 126,
    "mini_only": 17,
    "nano_only": 3,
    "both_wrong": 14,
}
VALIDATION_COUNTS = {
    "both_correct": 32,
    "mini_only": 4,
    "nano_only": 1,
    "both_wrong": 3,
}

# Absolute regret gaps from the latest mini-biased utility table.  Weighting,
# rather than duplicating scarce reviews, preserves a genuinely 200-unique-label
# experiment while making mini-only errors dominate the learning signal.
REGRET_WEIGHTS = {
    "both_correct": 0.5,
    "mini_only": 6.0,
    "nano_only": 1.5,
    "both_wrong": 1.8,
}

# Accuracy-aligned route rewards.  Both-wrong has no answer-accuracy difference;
# the small 0.1 preference retains the requested conservative mini tie-break.
ACCURACY_REWARDS = {
    "both_correct": (1.0, 1.0),
    "mini_only": (0.0, 1.0),
    "nano_only": (1.0, 0.0),
    "both_wrong": (0.0, 0.1),
}


def make_split(
    examples: list[dict[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Create 160 train + 40 validation + 82 test, all unique and disjoint."""
    rng = random.Random(seed)
    by_bucket = {bucket: [x for x in examples if x["bucket"] == bucket] for bucket in BUCKETS}
    for values in by_bucket.values():
        rng.shuffle(values)

    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for bucket in BUCKETS:
        values = by_bucket[bucket]
        required = TEST_COUNTS[bucket] + VALIDATION_COUNTS[bucket] + TRAIN_COUNTS[bucket]
        if len(values) < required:
            raise ValueError(f"Need {required} unique {bucket} reviews, found {len(values)}")
        cursor = 0
        test.extend(values[cursor : cursor + TEST_COUNTS[bucket]])
        cursor += TEST_COUNTS[bucket]
        validation.extend(values[cursor : cursor + VALIDATION_COUNTS[bucket]])
        cursor += VALIDATION_COUNTS[bucket]
        train.extend(values[cursor : cursor + TRAIN_COUNTS[bucket]])

    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(test)
    train_ids = {x["review_id"] for x in train}
    validation_ids = {x["review_id"] for x in validation}
    test_ids = {x["review_id"] for x in test}
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise AssertionError("Review leakage across train, validation, and test")
    if len(train_ids) != 160 or len(validation_ids) != 40 or len(test_ids) != 82:
        raise AssertionError("Split unexpectedly contains duplicates")
    return train, validation, test, {
        "train_counts": dict(Counter(x["bucket"] for x in train)),
        "validation_counts": dict(Counter(x["bucket"] for x in validation)),
        "test_counts": dict(Counter(x["bucket"] for x in test)),
        "train_unique": len(train_ids),
        "validation_unique": len(validation_ids),
        "test_unique": len(test_ids),
        "unique_labeled_reviews": len(train_ids) + len(validation_ids),
        "critical_labeled_reviews": sum(x["bucket"] == "mini_only" for x in train + validation),
    }


class TaskDataset(Dataset):
    def __init__(
        self,
        examples: list[dict[str, Any]],
        tokenizer: Any,
        max_length: int,
        task: str,
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task = task

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        encoded = self.tokenizer(
            text(example), truncation=True, max_length=self.max_length, padding=False
        )
        label = (
            int(example["target"])
            if self.task == "route"
            else int(example["gold"] == "POSITIVE")
        )
        return {
            **encoded,
            "label": label,
            "bucket": example["bucket"],
        }


def collate(items: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    length = max(len(item["input_ids"]) for item in items)
    return {
        "input_ids": torch.tensor(
            [item["input_ids"] + [pad_id] * (length - len(item["input_ids"])) for item in items],
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            [item["attention_mask"] + [0] * (length - len(item["attention_mask"])) for item in items],
            dtype=torch.long,
        ),
        "labels": torch.tensor([item["label"] for item in items], dtype=torch.long),
        "buckets": [item["bucket"] for item in items],
    }


def loader(
    examples: list[dict[str, Any]],
    tokenizer: Any,
    args: argparse.Namespace,
    task: str,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        TaskDataset(examples, tokenizer, args.max_length, task),
        batch_size=args.batch_size,
        shuffle=shuffle,
        collate_fn=lambda items: collate(items, tokenizer.pad_token_id),
    )


def probabilities(model: torch.nn.Module, data_loader: DataLoader) -> list[float]:
    values: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            logits = model(batch["input_ids"], batch["attention_mask"])
            values.extend(torch.softmax(logits, dim=-1)[:, 1].tolist())
    return values


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


def route_scores_from_probability(
    examples: list[dict[str, Any]], probabilities_: list[float], mode: str
) -> list[float]:
    """Convert model probabilities to P(mini-worthy) routing scores."""
    if mode == "direct_route":
        return probabilities_
    if mode == "direct_sentiment":
        # Mini is materially better almost exclusively on positive reviews.
        return probabilities_
    if mode == "nano_disagreement":
        return [
            p_positive if example["nano_pred"] == "NEGATIVE" else 1.0 - p_positive
            for example, p_positive in zip(examples, probabilities_)
        ]
    raise ValueError(mode)


def threshold_candidates(scores: list[float]) -> list[float]:
    # The exact empirical score boundaries give a strictly better calibration
    # search than a coarse fixed grid.  0 and >1 retain all-mini/all-nano.
    unique = sorted(set(float(value) for value in scores))
    candidates = [0.0, 1.000001]
    candidates.extend(unique)
    candidates.extend((left + right) / 2.0 for left, right in zip(unique, unique[1:]))
    return sorted(set(candidates))


def select_threshold(
    examples: list[dict[str, Any]], scores: list[float], policy: str
) -> dict[str, Any]:
    all_mini_accuracy = evaluate(examples, [1] * len(examples))["selected_accuracy"]
    rows = []
    for threshold in threshold_candidates(scores):
        predictions = [int(value >= threshold) for value in scores]
        metrics = evaluate(examples, predictions)
        rows.append({"threshold": threshold, **metrics})

    if policy == "match_all_mini":
        feasible = [row for row in rows if row["selected_accuracy"] + 1e-12 >= all_mini_accuracy]
        selected = max(
            feasible,
            key=lambda row: (-row["mini_calls"], row["selected_accuracy"], row["threshold"]),
        )
    elif policy == "max_accuracy":
        selected = max(
            rows,
            key=lambda row: (row["selected_accuracy"], -row["mini_calls"], row["threshold"]),
        )
    else:
        raise ValueError(policy)
    return {
        "threshold": selected["threshold"],
        "validation_metrics": {key: value for key, value in selected.items() if key != "threshold"},
        "validation_all_mini_accuracy": all_mini_accuracy,
    }


def evaluate_scores(
    examples: list[dict[str, Any]], scores: list[float], threshold: float
) -> dict[str, Any]:
    metrics = evaluate(examples, [int(value >= threshold) for value in scores])
    all_mini_accuracy = evaluate(examples, [1] * len(examples))["selected_accuracy"]
    return {
        **metrics,
        "accuracy_gap_vs_all_mini": metrics["selected_accuracy"] - all_mini_accuracy,
        "mini_saving_vs_all_mini": 1.0 - metrics["mini_rate"],
    }


def direct_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    buckets: list[str],
    method: str,
    epoch: int,
    epochs: int,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, labels, reduction="none")
    if method == "standard_ce":
        return ce.mean()
    weights = torch.tensor([REGRET_WEIGHTS[bucket] for bucket in buckets], dtype=torch.float)
    weighted_ce = (weights * ce).sum() / weights.sum()
    if method == "regret_ce":
        return weighted_ce
    if method == "regret_focal":
        correct_probability = torch.softmax(logits, dim=-1).gather(1, labels[:, None]).squeeze(1)
        focal = (1.0 - correct_probability).pow(2) * ce
        return (weights * focal).sum() / weights.sum()
    if method == "ce_then_accuracy_eu":
        if epoch <= max(1, epochs - 2):
            return weighted_ce
        rewards = torch.tensor([ACCURACY_REWARDS[bucket] for bucket in buckets], dtype=torch.float)
        expected_reward = (torch.softmax(logits, dim=-1) * rewards).sum(dim=-1).mean()
        return 0.7 * weighted_ce - 0.3 * expected_reward
    raise ValueError(method)


def train_direct(
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    test: list[dict[str, Any]],
    seed: int,
    method: str,
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

    best: dict[str, dict[str, Any]] = {}
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = direct_loss(
                logits, batch["labels"], batch["buckets"], method, epoch, args.epochs
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch["labels"])
        validation_scores = route_scores_from_probability(
            validation, probabilities(model, validation_loader), "direct_route"
        )
        epoch_trace: dict[str, Any] = {"epoch": epoch, "train_loss": total_loss / len(train)}
        for policy in ("match_all_mini", "max_accuracy"):
            calibration = select_threshold(validation, validation_scores, policy)
            metrics = calibration["validation_metrics"]
            key = (
                (-metrics["mini_calls"], metrics["selected_accuracy"])
                if policy == "match_all_mini"
                else (metrics["selected_accuracy"], -metrics["mini_calls"])
            )
            epoch_trace[policy] = {
                "threshold": calibration["threshold"],
                "selected_accuracy": metrics["selected_accuracy"],
                "mini_rate": metrics["mini_rate"],
            }
            if policy not in best or key > best[policy]["key"]:
                best[policy] = {
                    "key": key,
                    "epoch": epoch,
                    "state": capture_trainable(model),
                    "calibration": calibration,
                }
        history.append(epoch_trace)

    policies = {}
    for policy, selected in best.items():
        restore_trainable(model, selected["state"])
        test_scores = route_scores_from_probability(
            test, probabilities(model, test_loader), "direct_route"
        )
        policies[policy] = {
            "best_epoch": selected["epoch"],
            "threshold": selected["calibration"]["threshold"],
            "validation": selected["calibration"]["validation_metrics"],
            "test": evaluate_scores(test, test_scores, selected["calibration"]["threshold"]),
        }
    return {"method": method, "seed": seed, "policies": policies, "history": history}


def sentiment_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor,
    method: str,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, labels, weight=class_weights, reduction="none")
    if method == "sentiment_ce":
        return ce.mean()
    if method == "sentiment_focal":
        correct_probability = torch.softmax(logits, dim=-1).gather(1, labels[:, None]).squeeze(1)
        return ((1.0 - correct_probability).pow(2) * ce).mean()
    raise ValueError(method)


def train_sentiment_softprompt(
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    test: list[dict[str, Any]],
    seed: int,
    method: str,
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
    counts = Counter(int(x["gold"] == "POSITIVE") for x in train)
    class_weights = torch.tensor(
        [len(train) / (2.0 * max(counts[index], 1)) for index in (0, 1)], dtype=torch.float
    )

    # Keep independent best states for pre-routing and nano-first cascading.
    best: dict[str, dict[str, Any]] = {}
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = sentiment_loss(logits, batch["labels"], class_weights, method)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch["labels"])

        p_positive = probabilities(model, validation_loader)
        epoch_trace: dict[str, Any] = {"epoch": epoch, "train_loss": total_loss / len(train)}
        for routing_mode in ("direct_sentiment", "nano_disagreement"):
            scores = route_scores_from_probability(validation, p_positive, routing_mode)
            for policy in ("match_all_mini", "max_accuracy"):
                calibration = select_threshold(validation, scores, policy)
                metrics = calibration["validation_metrics"]
                key = (
                    (-metrics["mini_calls"], metrics["selected_accuracy"])
                    if policy == "match_all_mini"
                    else (metrics["selected_accuracy"], -metrics["mini_calls"])
                )
                name = f"{routing_mode}:{policy}"
                epoch_trace[name] = {
                    "threshold": calibration["threshold"],
                    "selected_accuracy": metrics["selected_accuracy"],
                    "mini_rate": metrics["mini_rate"],
                }
                if name not in best or key > best[name]["key"]:
                    best[name] = {
                        "key": key,
                        "epoch": epoch,
                        "state": capture_trainable(model),
                        "calibration": calibration,
                    }
        history.append(epoch_trace)

    routes: dict[str, Any] = {"direct_sentiment": {}, "nano_disagreement": {}}
    for name, selected in best.items():
        routing_mode, policy = name.split(":")
        restore_trainable(model, selected["state"])
        p_positive = probabilities(model, test_loader)
        test_scores = route_scores_from_probability(test, p_positive, routing_mode)
        routes[routing_mode][policy] = {
            "best_epoch": selected["epoch"],
            "threshold": selected["calibration"]["threshold"],
            "validation": selected["calibration"]["validation_metrics"],
            "test": evaluate_scores(test, test_scores, selected["calibration"]["threshold"]),
        }
    return {"method": method, "seed": seed, "routes": routes, "history": history}


def make_tfidf(seed: int) -> Pipeline:
    features = FeatureUnion([
        (
            "word",
            TfidfVectorizer(
                ngram_range=(1, 2), max_features=12000, sublinear_tf=True, min_df=1
            ),
        ),
        (
            "char",
            TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), max_features=12000,
                sublinear_tf=True, min_df=2,
            ),
        ),
    ])
    return Pipeline([
        ("features", features),
        (
            "classifier",
            LogisticRegression(
                C=2.0, max_iter=3000, class_weight="balanced", random_state=seed
            ),
        ),
    ])


def train_tfidf_sentiment(
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    test: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    model = make_tfidf(seed)
    model.fit([text(x) for x in train], [int(x["gold"] == "POSITIVE") for x in train])
    validation_positive = model.predict_proba([text(x) for x in validation])[:, 1].tolist()
    test_positive = model.predict_proba([text(x) for x in test])[:, 1].tolist()
    routes: dict[str, Any] = {"direct_sentiment": {}, "nano_disagreement": {}}
    for routing_mode in routes:
        validation_scores = route_scores_from_probability(
            validation, validation_positive, routing_mode
        )
        test_scores = route_scores_from_probability(test, test_positive, routing_mode)
        for policy in ("match_all_mini", "max_accuracy"):
            calibration = select_threshold(validation, validation_scores, policy)
            routes[routing_mode][policy] = {
                "threshold": calibration["threshold"],
                "validation": calibration["validation_metrics"],
                "test": evaluate_scores(test, test_scores, calibration["threshold"]),
            }
    return {"method": "tfidf_sentiment", "seed": seed, "routes": routes}


def get_test_metrics(run: dict[str, Any], method_key: str, policy: str) -> dict[str, Any]:
    if method_key.startswith("direct_"):
        return run["policies"][policy]["test"]
    routing_mode = method_key.split("/", 1)[1]
    return run["routes"][routing_mode][policy]["test"]


def aggregate_method(
    runs: list[dict[str, Any]], method_key: str, policy: str
) -> dict[str, dict[str, float]]:
    metrics = [
        "selected_accuracy",
        "accuracy_gap_vs_all_mini",
        "mini_rate",
        "mini_saving_vs_all_mini",
        "mini_only_recall",
        "both_wrong_mini_rate",
        "critical_misroutes",
        "unnecessary_mini",
    ]
    output = {}
    for metric in metrics:
        values = [float(get_test_metrics(run, method_key, policy)[metric]) for run in runs]
        output[metric] = {"mean": mean(values), "std": pstdev(values)}
    return output


def compact_row(
    name: str, runs: list[dict[str, Any]], method_key: str, policy: str
) -> dict[str, Any]:
    summary = aggregate_method(runs, method_key, policy)
    return {
        "method": name,
        "policy": policy,
        "accuracy": summary["selected_accuracy"]["mean"],
        "accuracy_std": summary["selected_accuracy"]["std"],
        "gap_vs_all_mini": summary["accuracy_gap_vs_all_mini"]["mean"],
        "mini_rate": summary["mini_rate"]["mean"],
        "mini_saving": summary["mini_saving_vs_all_mini"]["mean"],
        "critical_recall": summary["mini_only_recall"]["mean"],
        "both_wrong_mini_rate": summary["both_wrong_mini_rate"]["mean"],
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
        default=root / "sembench_movie_router/accuracy_targeted_router_comparison.json",
    )
    args = parser.parse_args()

    examples = deduplicate(load_examples(args.reviews, args.cache))
    available = Counter(x["bucket"] for x in examples)
    splits = {seed: make_split(examples, seed) for seed in args.seeds}
    direct_methods = ["standard_ce", "regret_ce", "regret_focal", "ce_then_accuracy_eu"]
    sentiment_methods = ["sentiment_ce", "sentiment_focal"]
    direct_runs: dict[str, list[dict[str, Any]]] = {name: [] for name in direct_methods}
    sentiment_runs: dict[str, list[dict[str, Any]]] = {name: [] for name in sentiment_methods}
    tfidf_runs: list[dict[str, Any]] = []

    for seed in args.seeds:
        train, validation, test, _ = splits[seed]
        print(f"seed={seed}: TF-IDF sentiment verifier", flush=True)
        tfidf_runs.append(train_tfidf_sentiment(train, validation, test, seed))
        for method in direct_methods:
            print(f"seed={seed}: direct soft prompt {method}", flush=True)
            run = train_direct(train, validation, test, seed, method, args)
            direct_runs[method].append(run)
            metrics = run["policies"]["match_all_mini"]["test"]
            print(
                f"  accuracy={metrics['selected_accuracy']:.4f} "
                f"mini_rate={metrics['mini_rate']:.4f} "
                f"critical_recall={metrics['mini_only_recall']:.4f}",
                flush=True,
            )
        for method in sentiment_methods:
            print(f"seed={seed}: sentiment soft prompt {method}", flush=True)
            run = train_sentiment_softprompt(train, validation, test, seed, method, args)
            sentiment_runs[method].append(run)
            metrics = run["routes"]["nano_disagreement"]["match_all_mini"]["test"]
            print(
                f"  cascade accuracy={metrics['selected_accuracy']:.4f} "
                f"mini_rate={metrics['mini_rate']:.4f} "
                f"critical_recall={metrics['mini_only_recall']:.4f}",
                flush=True,
            )

    first_test = splits[args.seeds[0]][2]
    summaries: dict[str, Any] = {}
    ranking = []
    for method, runs in direct_runs.items():
        key = f"direct_{method}"
        summaries[key] = {
            policy: aggregate_method(runs, key, policy)
            for policy in ("match_all_mini", "max_accuracy")
        }
        ranking.append(compact_row(key, runs, key, "match_all_mini"))
    for method, runs in sentiment_runs.items():
        for routing_mode in ("direct_sentiment", "nano_disagreement"):
            key = f"{method}/{routing_mode}"
            summaries[key] = {
                policy: aggregate_method(runs, key, policy)
                for policy in ("match_all_mini", "max_accuracy")
            }
            ranking.append(compact_row(key, runs, key, "match_all_mini"))
    for routing_mode in ("direct_sentiment", "nano_disagreement"):
        key = f"tfidf_sentiment/{routing_mode}"
        summaries[key] = {
            policy: aggregate_method(tfidf_runs, key, policy)
            for policy in ("match_all_mini", "max_accuracy")
        }
        ranking.append(compact_row(key, tfidf_runs, key, "match_all_mini"))

    ranking.sort(
        key=lambda row: (
            row["accuracy"] >= evaluate(first_test, [1] * len(first_test))["selected_accuracy"] - 0.01,
            row["accuracy"],
            row["mini_saving"],
        ),
        reverse=True,
    )
    result = {
        "config": {
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "label_protocol": "200 unique labeled reviews: 160 train + 40 validation",
            "threshold_protocol": (
                "validation only; minimize mini calls subject to validation accuracy >= all-mini"
            ),
            "both_wrong_target": "mini",
            "regret_weights": REGRET_WEIGHTS,
            "accuracy_rewards": ACCURACY_REWARDS,
            "nano_output_exposed_to_direct_router": False,
            "nano_output_exposed_to_nano_disagreement_cascade": True,
            "openai_api_calls": 0,
        },
        "unique_reviews": len(examples),
        "available_counts": dict(available),
        "split_metadata": {str(seed): splits[seed][3] for seed in args.seeds},
        "baselines": {
            "all_nano": evaluate(first_test, [0] * len(first_test)),
            "all_mini": evaluate(first_test, [1] * len(first_test)),
            "routing_oracle": evaluate(first_test, [int(x["target"]) for x in first_test]),
        },
        "ranking_match_all_mini": ranking,
        "aggregate": summaries,
        "runs": {
            "direct": direct_runs,
            "sentiment_softprompt": sentiment_runs,
            "tfidf_sentiment": tfidf_runs,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "baselines": result["baselines"],
        "ranking_match_all_mini": ranking,
    }, indent=2), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
