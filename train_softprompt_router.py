#!/usr/bin/env python3
"""Train a soft-prompt router for mini/nano answer selection.

This is a local baseline for testing continuous prompt routing. It freezes a
HuggingFace encoder, prepends trainable embeddings to each question, and trains
only those embeddings plus a small classifier head.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from optimize_ollama_router_gepa import (
    RoutingExample,
    classify_failure,
    count_outcomes,
    format_question,
    load_examples,
    outcome_bucket,
    parse_bucket_counts,
    select_examples,
    split_train_val,
    split_train_val_from_file,
    split_train_val_targeted,
)


ROUTES = ("nano", "mini")
ROUTE_TO_LABEL = {"nano": 0, "mini": 1}
LABEL_TO_ROUTE = {0: "nano", 1: "mini"}

RISK_AVERSE_UTILITY_RULE = {
    "both_correct_select_nano": 1.0,
    "both_correct_select_mini": 0.2,
    "mini_only_select_mini": 2.0,
    "mini_only_select_nano": -2.0,
    "nano_only_select_nano": 1.0,
    "nano_only_select_mini": -1.0,
    "both_wrong_select_nano": -0.5,
    "both_wrong_select_mini": 0.2,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def route_score(example: RoutingExample, route: str, score_mode: str) -> float:
    mini_correct = example.mini_pred == example.answer
    nano_correct = example.nano_pred == example.answer
    selected_correct = (route == "nano" and nano_correct) or (route == "mini" and mini_correct)
    selected_cost = example.nano_cost_usd if route == "nano" else example.mini_cost_usd

    if score_mode == "risk_averse_utility":
        return RISK_AVERSE_UTILITY_RULE[f"{outcome_bucket(example)}_select_{route}"]
    if score_mode == "balanced_utility":
        if mini_correct and nano_correct:
            return 1.0 if route == "nano" else 0.7
        if mini_correct:
            return 1.0 if route == "mini" else 0.0
        if nano_correct:
            return 1.0 if route == "nano" else 0.0
        return 0.3 if route == "nano" else 0.0
    if score_mode == "pareto_cost":
        return (1.0 / selected_cost) if selected_correct and selected_cost > 0 else 0.0
    if nano_correct:
        return 1.0 if route == "nano" else (0.85 if mini_correct else 0.0)
    if mini_correct:
        return 1.0 if route == "mini" else 0.0
    return 1.0 if route == "nano" else 0.85


def ideal_route(example: RoutingExample, score_mode: str) -> str:
    nano_score = route_score(example, "nano", score_mode)
    mini_score = route_score(example, "mini", score_mode)
    if nano_score == mini_score:
        return "nano"
    return "nano" if nano_score > mini_score else "mini"


def example_text(
    example: RoutingExample,
    include_nano_answer: bool,
    nano_response_max_chars: int,
) -> str:
    if include_nano_answer:
        return format_question(
            example,
            router_output_format="token",
            include_nano_answer=True,
            nano_response_max_chars=nano_response_max_chars,
        )
    option_lines = []
    for i, option in enumerate(example.options):
        option_lines.append(f"{chr(ord('A') + i)}. {option}")
    return (
        f"Category: {example.category}\n"
        f"Source: {example.src}\n\n"
        f"Question:\n{example.question}\n\n"
        f"Options:\n" + "\n".join(option_lines)
    )


class RoutingDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        examples: list[RoutingExample],
        tokenizer: Any,
        max_length: int,
        score_mode: str,
        include_nano_answer: bool,
        nano_response_max_chars: int,
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.score_mode = score_mode
        self.include_nano_answer = include_nano_answer
        self.nano_response_max_chars = nano_response_max_chars

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        encoded = self.tokenizer(
            example_text(example, self.include_nano_answer, self.nano_response_max_chars),
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "label": ROUTE_TO_LABEL[ideal_route(example, self.score_mode)],
            "route_scores": [
                route_score(example, "nano", self.score_mode),
                route_score(example, "mini", self.score_mode),
            ],
            "index": index,
        }


def collate_batch(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = []
    attention_mask = []
    labels = []
    route_scores = []
    indices = []
    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * pad_len)
        attention_mask.append(item["attention_mask"] + [0] * pad_len)
        labels.append(item["label"])
        route_scores.append(item["route_scores"])
        indices.append(item["index"])
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "route_scores": torch.tensor(route_scores, dtype=torch.float),
        "indices": torch.tensor(indices, dtype=torch.long),
    }


class SoftPromptRouter(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        soft_prompt_tokens: int,
        local_files_only: bool,
        train_backbone: bool,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
        )
        hidden_size = int(self.encoder.config.hidden_size)
        self.soft_prompt = nn.Parameter(torch.empty(soft_prompt_tokens, hidden_size))
        nn.init.normal_(self.soft_prompt, mean=0.0, std=0.02)
        self.classifier = nn.Linear(hidden_size, len(ROUTES))
        self.soft_prompt_tokens = soft_prompt_tokens

        if not train_backbone:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = self.encoder.get_input_embeddings()(input_ids)
        batch_size = token_embeddings.shape[0]
        prompt = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)
        inputs_embeds = torch.cat([prompt, token_embeddings], dim=1)
        prompt_mask = torch.ones(
            batch_size,
            self.soft_prompt_tokens,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        full_attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        outputs = self.encoder(inputs_embeds=inputs_embeds, attention_mask=full_attention_mask)
        cls_state = outputs.last_hidden_state[:, self.soft_prompt_tokens, :]
        return self.classifier(cls_state)


def summarize_routes(
    examples: list[RoutingExample],
    routes: list[str],
    score_mode: str,
) -> dict[str, Any]:
    total = len(examples) or 1
    traces = []
    route_counts = Counter(routes)
    failure_counts: Counter[str] = Counter()
    correct = 0
    score_sum = 0.0
    cost_used = 0.0
    total_mini_cost = 0.0

    for example, route in zip(examples, routes):
        mini_correct = example.mini_pred == example.answer
        nano_correct = example.nano_pred == example.answer
        selected_correct = (route == "nano" and nano_correct) or (route == "mini" and mini_correct)
        failure_type = classify_failure(route, nano_correct, mini_correct)
        score = route_score(example, route, score_mode)
        used_cost = example.nano_cost_usd if route == "nano" else example.mini_cost_usd
        correct += int(selected_correct)
        score_sum += score
        cost_used += used_cost
        total_mini_cost += example.mini_cost_usd
        failure_counts[failure_type] += 1
        traces.append(
            {
                "question_id": example.question_id,
                "route": route,
                "target_route": ideal_route(example, score_mode),
                "score": score,
                "selected_correct": selected_correct,
                "failure_type": failure_type,
                "mini_correct": mini_correct,
                "nano_correct": nano_correct,
            }
        )

    return {
        "count": len(examples),
        "score_mode": score_mode,
        "mean_score": score_sum / total,
        "final_accuracy": correct / total,
        "cost_saving_rate_vs_all_mini": 1.0 - (cost_used / total_mini_cost)
        if total_mini_cost
        else 0.0,
        "nano_routes": route_counts["nano"],
        "mini_routes": route_counts["mini"],
        "failure_counts": dict(failure_counts),
        "target_counts": dict(Counter(ideal_route(example, score_mode) for example in examples)),
        "traces": traces,
    }


def predict_routes(
    model: SoftPromptRouter,
    dataset: RoutingDataset,
    batch_size: int,
    pad_token_id: int,
    device: torch.device,
    decision_threshold: float = 0.5,
) -> tuple[list[str], list[list[float]]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, pad_token_id),
    )
    routes: list[str] = []
    probabilities: list[list[float]] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)
            mini_probs = probs[:, ROUTE_TO_LABEL["mini"]]
            labels = (mini_probs >= decision_threshold).long().cpu().tolist()
            routes.extend(LABEL_TO_ROUTE[int(label)] for label in labels)
            probabilities.extend(probs.cpu().tolist())
    return routes, probabilities


def routes_from_probabilities(
    probabilities: list[list[float]],
    decision_threshold: float,
) -> list[str]:
    return [
        "mini" if row[ROUTE_TO_LABEL["mini"]] >= decision_threshold else "nano"
        for row in probabilities
    ]


def threshold_summaries(
    examples: list[RoutingExample],
    probabilities: list[list[float]],
    score_mode: str,
    threshold_step: float = 0.05,
) -> list[dict[str, Any]]:
    summaries = []
    step_count = round(1.0 / threshold_step)
    if threshold_step <= 0 or abs(step_count * threshold_step - 1.0) > 1e-9:
        raise ValueError("threshold_step must be positive and divide 1.0 exactly")
    for step in range(0, step_count + 1):
        threshold = round(step * threshold_step, 10)
        routes = routes_from_probabilities(probabilities, threshold)
        summary = summarize_routes(examples, routes, score_mode)
        summaries.append(
            {
                "threshold": threshold,
                "mean_score": summary["mean_score"],
                "final_accuracy": summary["final_accuracy"],
                "cost_saving_rate_vs_all_mini": summary["cost_saving_rate_vs_all_mini"],
                "nano_routes": summary["nano_routes"],
                "mini_routes": summary["mini_routes"],
                "failure_counts": summary["failure_counts"],
            }
        )
    return summaries


def select_validation_threshold(
    examples: list[RoutingExample],
    probabilities: list[list[float]],
    score_mode: str,
    selection_cost_weight: float,
    threshold_step: float,
) -> tuple[float, dict[str, Any], float]:
    candidates = threshold_summaries(
        examples,
        probabilities,
        score_mode,
        threshold_step=threshold_step,
    )
    selected = max(
        candidates,
        key=lambda summary: (
            float(summary["mean_score"])
            + selection_cost_weight * float(summary["cost_saving_rate_vs_all_mini"]),
            -int(summary["failure_counts"].get("critical_misroute", 0)),
            float(summary["final_accuracy"]),
            float(summary["cost_saving_rate_vs_all_mini"]),
        ),
    )
    selection_score = float(selected["mean_score"]) + selection_cost_weight * float(
        selected["cost_saving_rate_vs_all_mini"]
    )
    return float(selected["threshold"]), selected, selection_score

def train_epoch(
    model: SoftPromptRouter,
    dataset: RoutingDataset,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    batch_size: int,
    pad_token_id: int,
    device: torch.device,
    loss_mode: str,
    target_mini_rate: float | None,
    mini_rate_penalty: float,
    mini_penalty: float,
) -> float:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, pad_token_id),
    )
    model.train()
    total_loss = 0.0
    total_items = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        route_scores = batch["route_scores"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=-1)
        if loss_mode == "expected_utility":
            expected_utility = (probs * route_scores).sum(dim=-1).mean()
            mini_rate = probs[:, ROUTE_TO_LABEL["mini"]].mean()
            loss = -expected_utility + mini_penalty * mini_rate
            if target_mini_rate is not None:
                target = torch.tensor(target_mini_rate, dtype=mini_rate.dtype, device=mini_rate.device)
                loss = loss + mini_rate_penalty * (mini_rate - target).pow(2)
        else:
            loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * labels.numel()
        total_items += int(labels.numel())
    return total_loss / max(total_items, 1)


def resolve_split_ids(
    all_examples: list[RoutingExample],
    split_path: Path,
    key: str,
) -> list[RoutingExample]:
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    ids = payload.get(key) or []
    by_id = {example.question_id: example for example in all_examples}
    resolved = []
    for raw_id in ids:
        question_id = int(raw_id)
        if question_id not in by_id:
            raise ValueError(f"{split_path} references unknown question_id={question_id}")
        resolved.append(by_id[question_id])
    return resolved


def load_split(
    args: argparse.Namespace,
) -> tuple[list[RoutingExample], list[RoutingExample], list[RoutingExample], list[RoutingExample]]:
    all_examples = load_examples(
        Path(args.mini_file),
        Path(args.nano_file),
        mini_cost_model=args.mini_cost_model,
        nano_cost_model=args.nano_cost_model,
    )
    testset: list[RoutingExample] = []
    if args.split_file:
        split_path = Path(args.split_file)
        trainset, valset = split_train_val_from_file(all_examples, split_path)
        testset = resolve_split_ids(all_examples, split_path, "test_ids")
        examples = trainset + valset
    elif args.sampling == "targeted":
        trainset, valset = split_train_val_targeted(
            all_examples,
            parse_bucket_counts(args.target_train_counts),
            parse_bucket_counts(args.target_val_counts),
            args.seed,
        )
        examples = trainset + valset
    else:
        examples = select_examples(all_examples, args.limit, args.sampling, args.seed)
        trainset, valset = split_train_val(examples, args.train_ratio, args.seed)
    return examples, trainset, valset, testset


def make_dataset(
    examples: list[RoutingExample],
    tokenizer: Any,
    args: argparse.Namespace,
) -> RoutingDataset:
    return RoutingDataset(
        examples,
        tokenizer,
        max_length=args.max_length,
        score_mode=args.score_mode,
        include_nano_answer=args.include_nano_answer,
        nano_response_max_chars=args.nano_response_max_chars,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mini-file", default="computer science_result_mini.json")
    parser.add_argument("--nano-file", default="computer science_result_nano.json")
    parser.add_argument("--mini-cost-model", default="gpt-4.1-mini")
    parser.add_argument("--nano-cost-model", default="gpt-4.1-nano")
    parser.add_argument("--hf-model", default="distilbert-base-uncased")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--sampling",
        choices=["balanced", "sequential", "random", "targeted"],
        default="balanced",
    )
    parser.add_argument(
        "--target-train-counts",
        default="mini_only=25,nano_only=10,both_correct=10,both_wrong=5",
    )
    parser.add_argument(
        "--target-val-counts",
        default="mini_only=10,nano_only=6,both_correct=17,both_wrong=17",
    )
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--score-mode",
        choices=["pareto_cost", "balanced_utility", "risk_averse_utility", "legacy"],
        default="risk_averse_utility",
    )
    parser.add_argument("--include-nano-answer", action="store_true")
    parser.add_argument("--nano-response-max-chars", type=int, default=1600)
    parser.add_argument("--soft-prompt-tokens", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--class-weight", action="store_true")
    parser.add_argument(
        "--loss-mode",
        choices=["cross_entropy", "expected_utility"],
        default="cross_entropy",
    )
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument(
        "--auto-select-threshold",
        action="store_true",
        help="Select the decision threshold on validation data for every checkpoint.",
    )
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--target-mini-rate", type=float, default=None)
    parser.add_argument("--mini-rate-penalty", type=float, default=0.0)
    parser.add_argument("--mini-penalty", type=float, default=0.0)
    parser.add_argument(
        "--selection-cost-weight",
        type=float,
        default=0.0,
        help="When validation scores tie, prefer checkpoints with more cost saving by this weight.",
    )
    parser.add_argument("--train-backbone", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="softprompt_router_result.json")
    parser.add_argument("--save-model", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )

    examples, trainset, valset, testset = load_split(args)
    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_model,
        local_files_only=not args.allow_download,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    pad_token_id = int(tokenizer.pad_token_id)

    train_dataset = make_dataset(trainset, tokenizer, args)
    val_dataset = make_dataset(valset, tokenizer, args)
    all_dataset = make_dataset(examples, tokenizer, args)
    test_dataset = make_dataset(testset, tokenizer, args) if testset else None

    model = SoftPromptRouter(
        args.hf_model,
        soft_prompt_tokens=args.soft_prompt_tokens,
        local_files_only=not args.allow_download,
        train_backbone=args.train_backbone,
    ).to(device)

    labels = [ROUTE_TO_LABEL[ideal_route(example, args.score_mode)] for example in trainset]
    if args.class_weight:
        counts = Counter(labels)
        weights = torch.tensor(
            [len(labels) / max(counts.get(label, 0), 1) for label in range(len(ROUTES))],
            dtype=torch.float,
            device=device,
        )
        criterion: nn.Module = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history = []
    best_val_score = float("-inf")
    best_selection_score = float("-inf")
    best_train_score_at_best_val = float("-inf")
    best_decision_threshold = args.decision_threshold
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            train_dataset,
            optimizer,
            criterion,
            args.batch_size,
            pad_token_id,
            device,
            args.loss_mode,
            args.target_mini_rate,
            args.mini_rate_penalty,
            args.mini_penalty,
        )
        _, train_probs = predict_routes(
            model, train_dataset, args.batch_size, pad_token_id, device, args.decision_threshold
        )
        _, val_probs = predict_routes(
            model, val_dataset, args.batch_size, pad_token_id, device, args.decision_threshold
        )
        if args.auto_select_threshold:
            epoch_threshold, _, val_selection_score = select_validation_threshold(
                valset,
                val_probs,
                args.score_mode,
                args.selection_cost_weight,
                args.threshold_step,
            )
        else:
            epoch_threshold = args.decision_threshold
            fixed_val_routes = routes_from_probabilities(val_probs, epoch_threshold)
            fixed_val_summary = summarize_routes(valset, fixed_val_routes, args.score_mode)
            val_selection_score = float(fixed_val_summary["mean_score"]) + (
                args.selection_cost_weight
                * float(fixed_val_summary["cost_saving_rate_vs_all_mini"])
            )
        train_routes = routes_from_probabilities(train_probs, epoch_threshold)
        val_routes = routes_from_probabilities(val_probs, epoch_threshold)
        train_summary = summarize_routes(trainset, train_routes, args.score_mode)
        val_summary = summarize_routes(valset, val_routes, args.score_mode)
        history.append(
            {
                "epoch": epoch,
                "decision_threshold": epoch_threshold,
                "train_loss": train_loss,
                "train_mean_score": train_summary["mean_score"],
                "train_accuracy": train_summary["final_accuracy"],
                "val_mean_score": val_summary["mean_score"],
                "val_accuracy": val_summary["final_accuracy"],
                "val_nano_routes": val_summary["nano_routes"],
                "val_mini_routes": val_summary["mini_routes"],
            }
        )
        val_score = float(val_summary["mean_score"])
        train_score = float(train_summary["mean_score"])
        if (
            val_selection_score > best_selection_score
            or (
                val_selection_score == best_selection_score
                and train_score > best_train_score_at_best_val
            )
        ):
            best_val_score = val_score
            best_selection_score = val_selection_score
            best_train_score_at_best_val = train_score
            best_decision_threshold = epoch_threshold
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"train_score={train_summary['mean_score']:.4f} "
            f"val_score={val_summary['mean_score']:.4f} "
            f"val_acc={val_summary['final_accuracy']:.3f} "
            f"threshold={epoch_threshold:.2f}"
        )
    if best_state is not None:
        model.load_state_dict(best_state)

    train_routes, train_probs = predict_routes(
        model, train_dataset, args.batch_size, pad_token_id, device, best_decision_threshold
    )
    val_routes, val_probs = predict_routes(
        model, val_dataset, args.batch_size, pad_token_id, device, best_decision_threshold
    )
    all_routes, all_probs = predict_routes(
        model, all_dataset, args.batch_size, pad_token_id, device, best_decision_threshold
    )
    test_routes: list[str] = []
    test_probs: list[list[float]] = []
    if test_dataset is not None:
        test_routes, test_probs = predict_routes(
            model, test_dataset, args.batch_size, pad_token_id, device, best_decision_threshold
        )

    train_summary = summarize_routes(trainset, train_routes, args.score_mode)
    val_summary = summarize_routes(valset, val_routes, args.score_mode)
    all_summary = summarize_routes(examples, all_routes, args.score_mode)
    test_summary = summarize_routes(testset, test_routes, args.score_mode) if testset else None
    summaries = [train_summary, val_summary, all_summary]
    if test_summary is not None:
        summaries.append(test_summary)
    for summary in summaries:
        summary["traces"] = summary["traces"][: args.limit if args.limit > 0 else len(summary["traces"])]

    payload = {
        "args": vars(args),
        "device": str(device),
        "selected_count": len(examples),
        "train_count": len(trainset),
        "val_count": len(valset),
        "test_count": len(testset),
        "selected_outcome_counts": count_outcomes(examples),
        "train_outcome_counts": count_outcomes(trainset),
        "val_outcome_counts": count_outcomes(valset),
        "test_outcome_counts": count_outcomes(testset) if testset else None,
        "history": history,
        "best_val_mean_score": best_val_score,
        "best_selection_score": best_selection_score,
        "best_train_mean_score_at_best_val": best_train_score_at_best_val,
        "selected_decision_threshold": best_decision_threshold,
        "train_summary": train_summary,
        "val_summary": val_summary,
        "all_summary": all_summary,
        "test_summary": test_summary,
        "train_threshold_summaries": threshold_summaries(trainset, train_probs, args.score_mode, args.threshold_step),
        "val_threshold_summaries": threshold_summaries(valset, val_probs, args.score_mode, args.threshold_step),
        "all_threshold_summaries": threshold_summaries(examples, all_probs, args.score_mode, args.threshold_step),
        "test_threshold_summaries": threshold_summaries(testset, test_probs, args.score_mode, args.threshold_step) if testset else None,
        "train_probabilities": train_probs,
        "val_probabilities": val_probs,
        "all_probabilities": all_probs,
        "test_probabilities": test_probs,
        "model_config": {
            "hf_model": args.hf_model,
            "soft_prompt_tokens": args.soft_prompt_tokens,
            "train_backbone": args.train_backbone,
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "total_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        },
    }

    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key.endswith("_summary")}, indent=2))
    print(f"Wrote {args.output}")

    if args.save_model:
        save_path = Path(args.save_model)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "routes": ROUTES,
                "selected_decision_threshold": best_decision_threshold,
            },
            save_path,
        )
        print(f"Wrote {save_path}")


if __name__ == "__main__":
    main()
