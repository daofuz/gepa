#!/usr/bin/env python3
"""Train a soft-prompt sentiment verifier for the SemBench Movie cascade."""

from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


class ReviewDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        encoded = self.tokenizer(
            row["reviewText"], truncation=True, max_length=self.max_length, padding=False
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "label": 1 if row["gold"] == "POSITIVE" else 0,
            "index": index,
        }


def collate(batch: list[dict[str, Any]], pad_id: int) -> dict[str, torch.Tensor]:
    size = max(len(x["input_ids"]) for x in batch)
    ids, masks = [], []
    for item in batch:
        padding = size - len(item["input_ids"])
        ids.append(item["input_ids"] + [pad_id] * padding)
        masks.append(item["attention_mask"] + [0] * padding)
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
        "labels": torch.tensor([x["label"] for x in batch], dtype=torch.long),
        "indices": torch.tensor([x["index"] for x in batch], dtype=torch.long),
    }


class SoftPromptVerifier(nn.Module):
    def __init__(self, model_name: str, prompt_tokens: int, train_backbone: bool) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=True)
        hidden = int(self.encoder.config.hidden_size)
        self.soft_prompt = nn.Parameter(torch.empty(prompt_tokens, hidden))
        nn.init.normal_(self.soft_prompt, mean=0.0, std=0.02)
        self.classifier = nn.Linear(hidden, 2)
        self.prompt_tokens = prompt_tokens
        if not train_backbone:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        embeddings = self.encoder.get_input_embeddings()(input_ids)
        prompt = self.soft_prompt.unsqueeze(0).expand(embeddings.shape[0], -1, -1)
        full_embeddings = torch.cat([prompt, embeddings], dim=1)
        prompt_mask = torch.ones(
            embeddings.shape[0], self.prompt_tokens,
            dtype=attention_mask.dtype, device=attention_mask.device,
        )
        full_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        state = self.encoder(inputs_embeds=full_embeddings, attention_mask=full_mask)
        cls = state.last_hidden_state[:, self.prompt_tokens, :]
        return self.classifier(cls)


def predict(
    model: nn.Module,
    rows: list[dict[str, Any]],
    tokenizer: Any,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> tuple[list[int], list[float]]:
    dataset = ReviewDataset(rows, tokenizer, max_length)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda items: collate(items, tokenizer.pad_token_id),
    )
    predictions: list[int] = []
    confidences: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            probs = torch.softmax(logits, dim=-1)
            confidence, labels = probs.max(dim=-1)
            predictions.extend(labels.cpu().tolist())
            confidences.extend(confidence.cpu().tolist())
    return predictions, confidences


def sentiment_metrics(rows: list[dict[str, Any]], predictions: list[int]) -> dict[str, float]:
    gold = [1 if x["gold"] == "POSITIVE" else 0 for x in rows]
    return {
        "accuracy": accuracy_score(gold, predictions),
        "f1": f1_score(gold, predictions, zero_division=0),
    }


def cascade_metrics(
    rows: list[dict[str, Any]], predictions: list[int], confidences: list[float], threshold: float
) -> dict[str, Any]:
    verifier = ["POSITIVE" if value else "NEGATIVE" for value in predictions]
    routes = [
        "mini" if pred != row["nano_pred"] and confidence >= threshold else "nano"
        for row, pred, confidence in zip(rows, verifier, confidences)
    ]
    selected = [
        row["mini_pred"] if route == "mini" else row["nano_pred"]
        for row, route in zip(rows, routes)
    ]
    critical = sum(
        row["target_route"] == "mini" and route != "mini" for row, route in zip(rows, routes)
    )
    unnecessary = sum(
        row["target_route"] == "nano" and route == "mini" for row, route in zip(rows, routes)
    )
    return {
        "threshold": threshold,
        "accuracy": accuracy_score([x["gold"] for x in rows], selected),
        "positive_f1": f1_score(
            [x["gold"] == "POSITIVE" for x in rows],
            [x == "POSITIVE" for x in selected],
            zero_division=0,
        ),
        "mini_calls": routes.count("mini"),
        "mini_rate": routes.count("mini") / len(routes),
        "critical_misroutes": critical,
        "unnecessary_mini": unnecessary,
        "routes": routes,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-results", type=Path, default=root / "sembench_movie_router" / "results.json")
    parser.add_argument("--model-outputs", type=Path, default=root / "sembench_movie_router" / "model_outputs.json")
    parser.add_argument("--reviews", type=Path, default=root / "sembench" / "files" / "movie" / "data" / "sf_2000" / "Reviews.csv")
    parser.add_argument("--hf-model", default="distilbert-base-uncased")
    parser.add_argument("--prompt-tokens", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, default=root / "sembench_movie_router" / "softprompt_verifier_results.json")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    router_result = json.loads(args.router_results.read_text(encoding="utf-8"))
    cache = json.loads(args.model_outputs.read_text(encoding="utf-8"))
    config = router_result["config"]

    # Reuse the canonical loader and preference construction from the runner.
    from run_sembench_movie_router import attach_outputs, load_rows

    total = int(config["train_size"]) + int(config["test_size"])
    rows = attach_outputs(
        load_rows(args.reviews, total), cache, config["mini_model"], config["nano_model"]
    )
    train_all = rows[: int(config["train_size"])]
    test = rows[int(config["train_size"]) :]
    indices = list(range(len(train_all)))
    random.shuffle(indices)
    val_count = max(20, len(train_all) // 5)
    val_ids = set(indices[:val_count])
    train = [row for i, row in enumerate(train_all) if i not in val_ids]
    val = [row for i, row in enumerate(train_all) if i in val_ids]

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, local_files_only=True)
    model = SoftPromptVerifier(args.hf_model, args.prompt_tokens, train_backbone=False)
    device = torch.device("cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.learning_rate
    )
    counts = [sum(row["gold"] == label for row in train) for label in ["NEGATIVE", "POSITIVE"]]
    weights = torch.tensor([len(train) / (2 * max(value, 1)) for value in counts], dtype=torch.float)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    dataset = ReviewDataset(train, tokenizer, args.max_length)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda items: collate(items, tokenizer.pad_token_id),
    )

    best_accuracy = -1.0
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            loss = criterion(logits, batch["labels"].to(device))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch["labels"])
        val_pred, _ = predict(model, val, tokenizer, args.max_length, args.batch_size, device)
        metrics = sentiment_metrics(val, val_pred)
        history.append({"epoch": epoch, "loss": total_loss / len(train), **metrics})
        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            best_state = deepcopy(model.state_dict())
        print(f"epoch={epoch} loss={total_loss / len(train):.4f} val_acc={metrics['accuracy']:.3f}")

    assert best_state is not None
    model.load_state_dict(best_state)
    train_pred, train_conf = predict(model, train_all, tokenizer, args.max_length, args.batch_size, device)
    test_pred, test_conf = predict(model, test, tokenizer, args.max_length, args.batch_size, device)

    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    cascades = [cascade_metrics(test, test_pred, test_conf, value) for value in thresholds]
    result = {
        "config": vars(args) | {
            "router_results": str(args.router_results), "model_outputs": str(args.model_outputs),
            "reviews": str(args.reviews), "output": str(args.output),
        },
        "best_val_accuracy": best_accuracy,
        "history": history,
        "train_sentiment": sentiment_metrics(train_all, train_pred),
        "test_sentiment": sentiment_metrics(test, test_pred),
        "cascades": cascades,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"config", "history"}}, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
