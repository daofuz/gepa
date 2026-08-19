#!/usr/bin/env python3
"""Evaluate a local Qwen scoring router on the clean Computer QA split."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from optimize_ollama_router_gepa import (  # noqa: E402
    OllamaChat,
    load_examples,
    outcome_bucket,
)
from scripts.run_computer_qa_regret_softprompt import (  # noqa: E402
    compact_summary,
    resolve_ids,
    routes,
    select_max_accuracy_threshold,
)

TARGETS = {
    "both_correct": "nano",
    "mini_only": "mini",
    "nano_only": "nano",
    "both_wrong": "mini",
}

SYSTEM_PROMPT = """You are a pre-routing model for multiple-choice computer science questions.
Choose whether a cheap model (nano) or a stronger model (mini) should answer.

Routing objective:
- Use nano when both models are likely correct.
- Use mini when mini is likely correct and nano is likely wrong.
- Use nano when nano is likely correct and mini is likely wrong.
- Use mini when both models are likely wrong.

You never see either model's answer. Infer routing risk only from the question, options,
category, and source. Produce a mini-needed score from 0 to 100:
- 0 means definitely nano.
- 100 means definitely mini.

Return compact JSON only: {"mini_score": <number>}"""


def question_text(example: Any, include_label: bool = False) -> str:
    options = "\n".join(
        f"{chr(ord('A') + index)}. {option}"
        for index, option in enumerate(example.options)
    )
    text = (
        f"Category: {example.category}\n"
        f"Source: {example.src}\n"
        f"Question: {example.question}\n"
        f"Options:\n{options}"
    )
    if include_label:
        text += f"\nRoute: {TARGETS[outcome_bucket(example)]}"
    return text


def demonstrations(train: list[Any]) -> list[Any]:
    selected = []
    for bucket in ("both_correct", "mini_only", "nano_only", "both_wrong"):
        values = [example for example in train if outcome_bucket(example) == bucket]
        if len(values) < 2:
            raise ValueError(f"Need at least two training examples for {bucket}")
        selected.extend(values[:2])
    return selected


def parse_score(raw: str) -> tuple[float, bool]:
    text = raw.strip()
    try:
        payload = json.loads(text)
        value = float(payload["mini_score"])
        return max(0.0, min(100.0, value)) / 100.0, False
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        match = re.search(r"mini_score[\"'\s:=>]+(-?\d+(?:\.\d+)?)", text, re.I)
        if match:
            value = float(match.group(1))
            return max(0.0, min(100.0, value)) / 100.0, True
    return 0.5, True


def load_cache(path: Path, model: str) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("model") == model:
            rows[int(row["question_id"])] = row
    return rows


def score_examples(
    examples: list[Any],
    demo_examples: list[Any],
    model: str,
    cache_path: Path,
    timeout: float,
) -> tuple[list[float], list[dict[str, Any]]]:
    cache = load_cache(cache_path, model)
    client = OllamaChat(
        model=model,
        temperature=0.0,
        timeout=timeout,
        num_predict=32,
        think=False,
        retry_empty_with_tokens=64,
    )
    demo_messages = []
    for example in demo_examples:
        demo_messages.extend(
            [
                {"role": "user", "content": question_text(example)},
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "mini_score": (
                                85
                                if TARGETS[outcome_bucket(example)] == "mini"
                                else 15
                            )
                        }
                    ),
                },
            ]
        )

    rows = []
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    for index, example in enumerate(examples, start=1):
        cached = cache.get(example.question_id)
        if cached is not None:
            rows.append(cached)
            continue
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *demo_messages,
            {
                "role": "user",
                "content": question_text(example)
                + '\nReturn only JSON: {"mini_score": <number>}',
            },
        ]
        started = time.perf_counter()
        raw = client(messages)
        elapsed = time.perf_counter() - started
        score, parse_fallback = parse_score(raw)
        row = {
            "question_id": example.question_id,
            "model": model,
            "score": score,
            "raw_response": raw,
            "parse_fallback": parse_fallback,
            "elapsed_seconds": elapsed,
        }
        with cache_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        cache[example.question_id] = row
        rows.append(row)
        if index % 20 == 0 or index == len(examples):
            print(f"scored={index}/{len(examples)}", flush=True)
    return [float(row["score"]) for row in rows], rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-file",
        type=Path,
        default=ROOT / "routing_split_computer_qa_balanced100_train80_val20_seed505.json",
    )
    parser.add_argument("--model", default="qwen3.5:2b")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "outputs/computer_qa_qwen35_2b_router_cache.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/computer_qa_qwen35_2b_local_router.json",
    )
    args = parser.parse_args()

    examples = load_examples(
        ROOT / "computer science_result_mini.json",
        ROOT / "computer science_result_nano.json",
        mini_cost_model="gpt-4.1-mini",
        nano_cost_model="gpt-4.1-nano",
    )
    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    train = resolve_ids(examples, split["train_ids"])
    validation = resolve_ids(examples, split["val_ids"])
    test = resolve_ids(examples, split["test_ids"])
    demo_examples = demonstrations(train)

    validation_scores, validation_rows = score_examples(
        validation, demo_examples, args.model, args.cache, args.timeout
    )
    threshold, _ = select_max_accuracy_threshold(validation, validation_scores)
    validation_routes = routes(validation_scores, threshold)
    test_scores, test_rows = score_examples(
        test, demo_examples, args.model, args.cache, args.timeout
    )
    test_routes = routes(test_scores, threshold)

    output = {
        "config": {
            "model": args.model,
            "split_file": str(args.split_file),
            "temperature": 0.0,
            "think": False,
            "demonstrations": 8,
            "demonstration_outcomes": dict(
                Counter(outcome_bucket(example) for example in demo_examples)
            ),
            "router_input": "question, options, category, and source only",
            "nano_answer_exposed": False,
            "threshold_policy": "validation max answer accuracy; ties use fewer mini calls",
            "test_used_for_selection": False,
        },
        "threshold": threshold,
        "validation": compact_summary(validation, validation_routes),
        "test": compact_summary(test, test_routes),
        "baselines": {
            "all_mini": compact_summary(test, ["mini"] * len(test)),
            "all_nano": compact_summary(test, ["nano"] * len(test)),
        },
        "invalid_or_fallback_outputs": sum(
            bool(row["parse_fallback"]) for row in validation_rows + test_rows
        ),
        "mean_latency_seconds": sum(
            float(row["elapsed_seconds"]) for row in validation_rows + test_rows
        )
        / (len(validation_rows) + len(test_rows)),
        "demonstration_ids": [example.question_id for example in demo_examples],
        "validation_scores": validation_rows,
        "test_scores": test_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "threshold": output["threshold"],
                "validation": output["validation"],
                "test": output["test"],
                "baselines": output["baselines"],
                "invalid_or_fallback_outputs": output[
                    "invalid_or_fallback_outputs"
                ],
                "mean_latency_seconds": output["mean_latency_seconds"],
            },
            indent=2,
        )
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
