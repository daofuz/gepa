#!/usr/bin/env python3
"""Compare natural and outcome-balanced direct routing on SemBench Movie.

The risk-averse target sends ``mini_only`` and ``both_wrong`` examples to
mini, while ``both_correct`` and ``nano_only`` examples go to nano.  The
evaluation split is stratified and disjoint from both 100-example training
sets.  No nano response is exposed to the direct router.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


BUCKETS = ("both_correct", "mini_only", "nano_only", "both_wrong")
MINI_TARGETS = {"mini_only", "both_wrong"}
UTILITY = {
    ("both_correct", "nano"): 1.0,
    ("both_correct", "mini"): 0.2,
    ("mini_only", "nano"): -2.0,
    ("mini_only", "mini"): 2.0,
    ("nano_only", "nano"): 1.0,
    ("nano_only", "mini"): -1.0,
    ("both_wrong", "nano"): -0.5,
    ("both_wrong", "mini"): 0.2,
}


def load_examples(reviews: Path, cache_path: Path) -> list[dict[str, Any]]:
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    examples = []
    with reviews.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            review_id = row["reviewId"]
            mini_key = f"{review_id}::gpt-5-mini"
            nano_key = f"{review_id}::gpt-5-nano"
            if mini_key not in cache or nano_key not in cache:
                continue
            gold = row["scoreSentiment"].upper()
            mini_pred = cache[mini_key]["pred"]
            nano_pred = cache[nano_key]["pred"]
            mini_correct = mini_pred == gold
            nano_correct = nano_pred == gold
            bucket = (
                "both_correct" if mini_correct and nano_correct else
                "mini_only" if mini_correct else
                "nano_only" if nano_correct else
                "both_wrong"
            )
            examples.append({
                "review_id": review_id,
                "movie_id": row["id"],
                "text": row["reviewText"],
                "gold": gold,
                "mini_pred": mini_pred,
                "nano_pred": nano_pred,
                "bucket": bucket,
                "target": 1 if bucket in MINI_TARGETS else 0,
            })
    return examples


def make_splits(
    examples: list[dict[str, Any]], seed: int, train_size: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    by_bucket = {bucket: [x for x in examples if x["bucket"] == bucket] for bucket in BUCKETS}
    for values in by_bucket.values():
        rng.shuffle(values)

    # Hold out 20% of every outcome type, including the scarce nano-only type.
    test = []
    pool = []
    for bucket in BUCKETS:
        values = by_bucket[bucket]
        test_count = max(1, round(0.2 * len(values)))
        test.extend(values[:test_count])
        pool.extend(values[test_count:])
    rng.shuffle(test)

    # Enrich critical and both-wrong examples without duplicating annotations.
    remaining = {bucket: [x for x in pool if x["bucket"] == bucket] for bucket in BUCKETS}
    desired = {
        "mini_only": min(30, len(remaining["mini_only"])),
        "both_wrong": min(25, len(remaining["both_wrong"])),
        "nano_only": min(5, len(remaining["nano_only"])),
    }
    desired["both_correct"] = train_size - sum(desired.values())
    balanced = []
    for bucket in BUCKETS:
        balanced.extend(remaining[bucket][: desired[bucket]])
    if len(balanced) != train_size:
        raise ValueError(f"Could not construct {train_size} balanced unique examples")
    rng.shuffle(balanced)

    # Natural baseline is independently sampled from the same post-test pool.
    natural = rng.sample(pool, train_size)
    return natural, balanced, test


def text(example: dict[str, Any]) -> str:
    return (
        "Semantic operator: classify whether a movie review is positive or negative.\n"
        f"Review: {example['text']}"
    )


def evaluate(examples: list[dict[str, Any]], predictions: list[int]) -> dict[str, Any]:
    routes = ["mini" if value else "nano" for value in predictions]
    selected = [
        example["mini_pred"] if route == "mini" else example["nano_pred"]
        for example, route in zip(examples, routes)
    ]
    targets = [int(example["target"]) for example in examples]
    mini_only = [i for i, example in enumerate(examples) if example["bucket"] == "mini_only"]
    both_wrong = [i for i, example in enumerate(examples) if example["bucket"] == "both_wrong"]
    return {
        "count": len(examples),
        "bucket_counts": dict(Counter(example["bucket"] for example in examples)),
        "mini_calls": sum(predictions),
        "mini_rate": sum(predictions) / len(predictions),
        "route_accuracy": sum(a == b for a, b in zip(targets, predictions)) / len(targets),
        "selected_accuracy": sum(
            prediction == example["gold"] for prediction, example in zip(selected, examples)
        ) / len(examples),
        "mean_utility": sum(
            UTILITY[(example["bucket"], route)] for example, route in zip(examples, routes)
        ) / len(examples),
        "mini_only_recall": (
            sum(predictions[i] == 1 for i in mini_only) / len(mini_only) if mini_only else None
        ),
        "both_wrong_mini_rate": (
            sum(predictions[i] == 1 for i in both_wrong) / len(both_wrong) if both_wrong else None
        ),
        "critical_misroutes": sum(predictions[i] == 0 for i in mini_only),
        "both_wrong_cheap_routes": sum(predictions[i] == 0 for i in both_wrong),
        "unnecessary_mini": sum(
            prediction == 1 and example["bucket"] in {"both_correct", "nano_only"}
            for prediction, example in zip(predictions, examples)
        ),
    }


def train_tfidf(train: list[dict[str, Any]], test: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=8000, sublinear_tf=True)),
        ("classifier", LogisticRegression(C=1.0, max_iter=2000, random_state=seed)),
    ])
    model.fit([text(x) for x in train], [x["target"] for x in train])
    predictions = model.predict([text(x) for x in test]).tolist()
    return evaluate(test, predictions)


class RouterDataset(Dataset):
    def __init__(self, examples: list[dict[str, Any]], tokenizer: Any, max_length: int) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        encoded = self.tokenizer(
            text(example), truncation=True, max_length=self.max_length, padding=False
        )
        return {**encoded, "label": example["target"]}


def collate(items: list[dict[str, Any]], pad_id: int) -> dict[str, torch.Tensor]:
    length = max(len(item["input_ids"]) for item in items)
    return {
        "input_ids": torch.tensor([
            item["input_ids"] + [pad_id] * (length - len(item["input_ids"])) for item in items
        ], dtype=torch.long),
        "attention_mask": torch.tensor([
            item["attention_mask"] + [0] * (length - len(item["attention_mask"])) for item in items
        ], dtype=torch.long),
        "labels": torch.tensor([item["label"] for item in items], dtype=torch.long),
    }


class SoftPromptDirectRouter(nn.Module):
    def __init__(self, model_name: str, prompt_tokens: int) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=True)
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        hidden = int(self.encoder.config.hidden_size)
        self.prompt = nn.Parameter(torch.empty(prompt_tokens, hidden))
        nn.init.normal_(self.prompt, mean=0.0, std=0.02)
        self.classifier = nn.Linear(hidden, 2)
        self.prompt_tokens = prompt_tokens

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        embeddings = self.encoder.get_input_embeddings()(input_ids)
        prompt = self.prompt.unsqueeze(0).expand(embeddings.shape[0], -1, -1)
        combined = torch.cat([prompt, embeddings], dim=1)
        prompt_mask = torch.ones(
            embeddings.shape[0], self.prompt_tokens,
            dtype=attention_mask.dtype, device=attention_mask.device,
        )
        mask = torch.cat([prompt_mask, attention_mask], dim=1)
        outputs = self.encoder(inputs_embeds=combined, attention_mask=mask)
        return self.classifier(outputs.last_hidden_state[:, self.prompt_tokens, :])


def train_softprompt(
    train: list[dict[str, Any]], test: list[dict[str, Any]], args: argparse.Namespace
) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, local_files_only=True)
    model = SoftPromptDirectRouter(args.hf_model, args.prompt_tokens)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(
        RouterDataset(train, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda items: collate(items, tokenizer.pad_token_id),
    )
    model.train()
    history = []
    for epoch in range(1, args.epochs + 1):
        loss_sum = 0.0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["labels"])
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(batch["labels"])
        history.append({"epoch": epoch, "loss": loss_sum / len(train)})

    model.eval()
    predictions = []
    test_loader = DataLoader(
        RouterDataset(test, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda items: collate(items, tokenizer.pad_token_id),
    )
    with torch.no_grad():
        for batch in test_loader:
            predictions.extend(model(batch["input_ids"], batch["attention_mask"]).argmax(-1).tolist())
    return {**evaluate(test, predictions), "history": history}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", type=Path, default=root / "sembench/files/movie/data/sf_2000/Reviews.csv")
    parser.add_argument("--cache", type=Path, default=root / "sembench_movie_router/model_outputs.json")
    parser.add_argument("--train-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=404)
    parser.add_argument("--hf-model", default="distilbert-base-uncased")
    parser.add_argument("--prompt-tokens", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=root / "sembench_movie_router/balanced_direct_router.json")
    args = parser.parse_args()

    examples = load_examples(args.reviews, args.cache)
    natural, balanced, test = make_splits(examples, args.seed, args.train_size)
    result = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "available_bucket_counts": dict(Counter(x["bucket"] for x in examples)),
        "natural_train_counts": dict(Counter(x["bucket"] for x in natural)),
        "balanced_train_counts": dict(Counter(x["bucket"] for x in balanced)),
        "test_counts": dict(Counter(x["bucket"] for x in test)),
        "test_review_ids": [x["review_id"] for x in test],
        "results": {
            "natural_tfidf": train_tfidf(natural, test, args.seed),
            "balanced_tfidf": train_tfidf(balanced, test, args.seed),
            "natural_softprompt": train_softprompt(natural, test, args),
            "balanced_softprompt": train_softprompt(balanced, test, args),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    printable = {name: {k: v for k, v in value.items() if k != "history"} for name, value in result["results"].items()}
    print(json.dumps({
        "available_bucket_counts": result["available_bucket_counts"],
        "natural_train_counts": result["natural_train_counts"],
        "balanced_train_counts": result["balanced_train_counts"],
        "test_counts": result["test_counts"],
        "results": printable,
    }, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
