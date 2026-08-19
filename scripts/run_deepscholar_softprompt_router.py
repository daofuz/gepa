#!/usr/bin/env python3
"""Train the existing frozen-encoder soft-prompt router on DeepScholar-Bench.

By default, the offline experiment treats papers with at least the median
number of ground-truth citations as requiring the ``full`` DeepScholar route.
This is an evidence-burden proxy, not a claim that one generation system beat
another.  Passing ``--labels-csv`` switches to measured per-query system scores.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the exact frozen-encoder + learned prompt-token implementation used by
# the existing computer-QA routing experiments in this workspace.
from train_softprompt_router import SoftPromptRouter  # noqa: E402


ROUTES = ("cheap", "full")


@dataclass(frozen=True)
class Example:
    arxiv_id: str
    title: str
    abstract: str
    citation_count: int
    label: int
    cheap_score: float | None = None
    full_score: float | None = None

    @property
    def text(self) -> str:
        return f"Research paper: {self.title}\n\nAbstract:\n{self.abstract}"


class DeepScholarDataset(Dataset[dict[str, Any]]):
    def __init__(self, examples: list[Example], tokenizer: Any, max_length: int) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        encoded = self.tokenizer(
            self.examples[index].text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "label": self.examples[index].label,
        }


def collate(items: list[dict[str, Any]], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(item["input_ids"]) for item in items)
    return {
        "input_ids": torch.tensor(
            [
                item["input_ids"]
                + [pad_token_id] * (max_len - len(item["input_ids"]))
                for item in items
            ],
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            [
                item["attention_mask"]
                + [0] * (max_len - len(item["attention_mask"]))
                for item in items
            ],
            dtype=torch.long,
        ),
        "labels": torch.tensor([item["label"] for item in items], dtype=torch.long),
    }


def citation_counts(path: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            arxiv_id = (row.get("parent_paper_arxiv_id") or "").strip()
            if arxiv_id:
                counts[arxiv_id] += 1
    return dict(counts)


def measured_labels(path: Path, minimum_full_gain: float) -> dict[str, tuple[int, float, float]]:
    labels: dict[str, tuple[int, float, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"arxiv_id", "cheap_score", "full_score"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Measured labels CSV is missing columns: {sorted(missing)}")
        for row in reader:
            arxiv_id = row["arxiv_id"].strip()
            cheap_score = float(row["cheap_score"])
            full_score = float(row["full_score"])
            label = int(full_score - cheap_score >= minimum_full_gain)
            labels[arxiv_id] = (label, cheap_score, full_score)
    return labels


def load_examples(
    benchmark_path: Path,
    citations_path: Path,
    labels_path: Path | None,
    minimum_full_gain: float,
) -> tuple[list[Example], dict[str, Any]]:
    counts = citation_counts(citations_path)
    if not counts:
        raise ValueError(f"No parent-paper citation counts found in {citations_path}")

    score_labels = measured_labels(labels_path, minimum_full_gain) if labels_path else None
    burden_threshold = float(median(counts.values()))
    examples: list[Example] = []
    with benchmark_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            arxiv_id = (row.get("arxiv_id") or "").strip()
            title = (row.get("title") or "").strip()
            abstract = (row.get("abstract") or "").strip()
            if not arxiv_id or not abstract or arxiv_id not in counts:
                continue
            if score_labels is None:
                label = int(counts[arxiv_id] >= burden_threshold)
                cheap_score = None
                full_score = None
            else:
                if arxiv_id not in score_labels:
                    continue
                label, cheap_score, full_score = score_labels[arxiv_id]
            examples.append(
                Example(
                    arxiv_id=arxiv_id,
                    title=title,
                    abstract=abstract,
                    citation_count=counts[arxiv_id],
                    label=label,
                    cheap_score=cheap_score,
                    full_score=full_score,
                )
            )

    if len(examples) < 20:
        raise ValueError(f"Need at least 20 matched examples, found {len(examples)}")
    label_counts = Counter(example.label for example in examples)
    if len(label_counts) != 2:
        raise ValueError(f"Need both routing labels, found {dict(label_counts)}")

    metadata = {
        "label_source": "measured_system_scores" if labels_path else "ground_truth_citation_burden_proxy",
        "label_rule": (
            f"full_score - cheap_score >= {minimum_full_gain}"
            if labels_path
            else f"full when ground-truth citation count >= median ({burden_threshold:g})"
        ),
        "proxy_warning": None if labels_path else (
            "Citation burden is an offline proxy for required search depth; it is not a measured "
            "cheap-vs-full DeepScholar quality comparison."
        ),
        "citation_count": {
            "minimum": min(example.citation_count for example in examples),
            "median": median(example.citation_count for example in examples),
            "maximum": max(example.citation_count for example in examples),
            "mean": mean(example.citation_count for example in examples),
        },
        "label_counts": {ROUTES[label]: count for label, count in sorted(label_counts.items())},
    }
    return examples, metadata


def stratified_split(
    examples: list[Example], seed: int
) -> tuple[list[Example], list[Example], list[Example]]:
    rng = random.Random(seed)
    grouped = {label: [example for example in examples if example.label == label] for label in (0, 1)}
    train: list[Example] = []
    validation: list[Example] = []
    test: list[Example] = []
    for values in grouped.values():
        rng.shuffle(values)
        train_count = round(len(values) * 40 / 63)
        validation_count = round(len(values) * 10 / 63)
        train.extend(values[:train_count])
        validation.extend(values[train_count:train_count + validation_count])
        test.extend(values[train_count + validation_count:])
    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(test)
    all_ids = [example.arxiv_id for example in train + validation + test]
    if len(all_ids) != len(set(all_ids)):
        raise AssertionError("Query leakage across train, validation, and test")
    for name, split in (("train", train), ("validation", validation), ("test", test)):
        if len({example.label for example in split}) != 2:
            raise ValueError(f"{name} split does not contain both labels")
    return train, validation, test


def make_loader(
    examples: list[Example],
    tokenizer: Any,
    max_length: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = DeepScholarDataset(examples, tokenizer, max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda items: collate(items, int(tokenizer.pad_token_id)),
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def probabilities(
    model: SoftPromptRouter, loader: DataLoader, device: torch.device
) -> list[float]:
    model.eval()
    result: list[float] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            result.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().tolist())
    return result


def route_utility(label: int, prediction: int, overroute_utility: float) -> float:
    if label == prediction:
        return 1.0
    if label == 0 and prediction == 1:
        return overroute_utility
    return 0.0


def auc(labels: list[int], probs: list[float]) -> float:
    positive = [prob for label, prob in zip(labels, probs) if label == 1]
    negative = [prob for label, prob in zip(labels, probs) if label == 0]
    if not positive or not negative:
        return float("nan")
    favorable = 0.0
    for pos in positive:
        for neg in negative:
            favorable += 1.0 if pos > neg else (0.5 if pos == neg else 0.0)
    return favorable / (len(positive) * len(negative))


def evaluate(
    examples: list[Example],
    probs: list[float],
    threshold: float,
    overroute_utility: float,
    cheap_cost: float,
    full_cost: float,
) -> dict[str, Any]:
    labels = [example.label for example in examples]
    predictions = [int(probability >= threshold) for probability in probs]
    tp = sum(label == 1 and pred == 1 for label, pred in zip(labels, predictions))
    tn = sum(label == 0 and pred == 0 for label, pred in zip(labels, predictions))
    fp = sum(label == 0 and pred == 1 for label, pred in zip(labels, predictions))
    fn = sum(label == 1 and pred == 0 for label, pred in zip(labels, predictions))
    positive_count = tp + fn
    negative_count = tn + fp
    recall = tp / positive_count if positive_count else 0.0
    specificity = tn / negative_count if negative_count else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    mean_cost = mean(full_cost if pred else cheap_cost for pred in predictions)
    return {
        "threshold": threshold,
        "accuracy": (tp + tn) / len(labels),
        "balanced_accuracy": (recall + specificity) / 2,
        "full_precision": precision,
        "full_recall": recall,
        "cheap_recall": specificity,
        "auc": auc(labels, probs),
        "brier": mean((prob - label) ** 2 for label, prob in zip(labels, probs)),
        "full_rate": mean(predictions),
        "critical_misroutes": fn,
        "unnecessary_full": fp,
        "mean_utility": mean(
            route_utility(label, pred, overroute_utility)
            for label, pred in zip(labels, predictions)
        ),
        "mean_cost_units": mean_cost,
        "cost_vs_all_full": mean_cost / full_cost,
        "confusion": {"true_full": tp, "false_full": fp, "true_cheap": tn, "false_cheap": fn},
    }


def select_threshold(
    examples: list[Example], probs: list[float], overroute_utility: float
) -> tuple[float, tuple[float, float, int]]:
    candidates = sorted({0.0, 1.0, *probs})
    best_threshold = 0.5
    best_key: tuple[float, float, int] | None = None
    labels = [example.label for example in examples]
    for threshold in candidates:
        predictions = [int(probability >= threshold) for probability in probs]
        utility = mean(
            route_utility(label, pred, overroute_utility)
            for label, pred in zip(labels, predictions)
        )
        full_labels = sum(labels)
        recall = (
            sum(label == 1 and pred == 1 for label, pred in zip(labels, predictions))
            / full_labels
        )
        key = (utility, recall, -sum(predictions))
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = threshold
    assert best_key is not None
    return best_threshold, best_key


def train_once(
    examples: list[Example],
    tokenizer: Any,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], tuple[float, float, int]]:
    set_seed(seed)
    train, validation, test = stratified_split(examples, seed)
    train_loader = make_loader(train, tokenizer, args.max_length, args.batch_size, True)
    validation_loader = make_loader(validation, tokenizer, args.max_length, args.batch_size, False)
    test_loader = make_loader(test, tokenizer, args.max_length, args.batch_size, False)

    model = SoftPromptRouter(
        args.hf_model,
        soft_prompt_tokens=args.prompt_tokens,
        local_files_only=not args.allow_model_download,
        train_backbone=False,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    best_key: tuple[float, float, int] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_threshold = 0.5
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            labels = batch["labels"].to(device)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(labels)

        validation_probs = probabilities(model, validation_loader, device)
        threshold, key = select_threshold(validation, validation_probs, args.overroute_utility)
        metrics = evaluate(
            validation,
            validation_probs,
            threshold,
            args.overroute_utility,
            args.cheap_cost,
            args.full_cost,
        )
        history.append({"epoch": epoch, "train_loss": loss_sum / len(train), **metrics})
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            best_threshold = threshold
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
                if not name.startswith("encoder.")
            }

    assert best_key is not None and best_state is not None
    model.load_state_dict(best_state, strict=False)
    test_probs = probabilities(model, test_loader, device)
    test_metrics = evaluate(
        test,
        test_probs,
        best_threshold,
        args.overroute_utility,
        args.cheap_cost,
        args.full_cost,
    )
    predictions = [int(probability >= best_threshold) for probability in test_probs]
    baselines = {
        "all_cheap": evaluate(
            test, [0.0] * len(test), 0.5, args.overroute_utility, args.cheap_cost, args.full_cost
        ),
        "all_full": evaluate(
            test, [1.0] * len(test), 0.5, args.overroute_utility, args.cheap_cost, args.full_cost
        ),
        "oracle": evaluate(
            test,
            [float(example.label) for example in test],
            0.5,
            args.overroute_utility,
            args.cheap_cost,
            args.full_cost,
        ),
    }
    run = {
        "seed": seed,
        "split_sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
        "split_label_counts": {
            name: dict(Counter(ROUTES[example.label] for example in split))
            for name, split in (("train", train), ("validation", validation), ("test", test))
        },
        "best_epoch": best_epoch,
        "threshold": best_threshold,
        "validation_selection_key": list(best_key),
        "test": test_metrics,
        "baselines": baselines,
        "history": history,
        "test_traces": [
            {
                "arxiv_id": example.arxiv_id,
                "title": example.title,
                "citation_count": example.citation_count,
                "target_route": ROUTES[example.label],
                "full_probability": probability,
                "predicted_route": ROUTES[prediction],
            }
            for example, probability, prediction in zip(test, test_probs, predictions)
        ],
    }
    return run, best_state, best_key


def aggregate(runs: list[dict[str, Any]], path: tuple[str, ...]) -> dict[str, dict[str, float]]:
    metrics = (
        "accuracy",
        "balanced_accuracy",
        "full_precision",
        "full_recall",
        "cheap_recall",
        "auc",
        "brier",
        "full_rate",
        "critical_misroutes",
        "unnecessary_full",
        "mean_utility",
        "mean_cost_units",
        "cost_vs_all_full",
    )
    summaries: dict[str, dict[str, float]] = {}
    for metric in metrics:
        values = []
        for run in runs:
            current: Any = run
            for component in path:
                current = current[component]
            values.append(float(current[metric]))
        summaries[metric] = {"mean": mean(values), "std": pstdev(values)}
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "deepscholar/dataset/related_works_combined.csv",
    )
    parser.add_argument(
        "--citations",
        type=Path,
        default=ROOT / "deepscholar/dataset/citations.csv",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=None,
        help="Optional CSV with arxiv_id, cheap_score, and full_score columns.",
    )
    parser.add_argument("--minimum-full-gain", type=float, default=0.05)
    parser.add_argument("--hf-model", default="distilbert-base-uncased")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--prompt-tokens", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--overroute-utility", type=float, default=0.2)
    parser.add_argument("--cheap-cost", type=float, default=1.0)
    parser.add_argument("--full-cost", type=float, default=4.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[505, 606, 707, 808, 909])
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/deepscholar_softprompt_citation_burden.json",
    )
    parser.add_argument(
        "--save-model",
        type=Path,
        default=ROOT / "outputs/deepscholar_softprompt_citation_burden.pt",
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    examples, data_metadata = load_examples(
        args.benchmark,
        args.citations,
        args.labels_csv,
        args.minimum_full_gain,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_model,
        local_files_only=not args.allow_model_download,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    print(
        f"Loaded {len(examples)} DeepScholar queries on {device}; "
        f"labels={data_metadata['label_counts']}",
        flush=True,
    )
    runs: list[dict[str, Any]] = []
    selected_state: dict[str, torch.Tensor] | None = None
    selected_key: tuple[float, float, int] | None = None
    selected_seed = args.seeds[0]
    for seed in args.seeds:
        print(f"seed={seed}: training {args.prompt_tokens}-token soft prompt", flush=True)
        run, state, key = train_once(examples, tokenizer, seed, args, device)
        runs.append(run)
        if selected_key is None or key > selected_key:
            selected_key = key
            selected_state = copy.deepcopy(state)
            selected_seed = seed
        metrics = run["test"]
        print(
            f"seed={seed}: accuracy={metrics['accuracy']:.4f} "
            f"auc={metrics['auc']:.4f} utility={metrics['mean_utility']:.4f} "
            f"full_rate={metrics['full_rate']:.4f}",
            flush=True,
        )

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    output = {
        "method": "frozen_encoder_soft_prompt_router",
        "router_input": ["paper_title", "paper_abstract"],
        "data": data_metadata,
        "config": {**config, "device_used": str(device)},
        "aggregate_test": aggregate(runs, ("test",)),
        "aggregate_baselines": {
            name: aggregate(runs, ("baselines", name))
            for name in ("all_cheap", "all_full", "oracle")
        },
        "selected_checkpoint_seed": selected_seed,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    assert selected_state is not None
    args.save_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "method": output["method"],
            "hf_model": args.hf_model,
            "prompt_tokens": args.prompt_tokens,
            "selected_seed": selected_seed,
            "state_dict": selected_state,
            "label_metadata": data_metadata,
        },
        args.save_model,
    )
    print(json.dumps(output["aggregate_test"], indent=2), flush=True)
    print(f"Saved metrics to {args.output}", flush=True)
    print(f"Saved trainable soft-prompt state to {args.save_model}", flush=True)


if __name__ == "__main__":
    main()
