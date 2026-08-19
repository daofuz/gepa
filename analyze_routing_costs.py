#!/usr/bin/env python3
"""Compute token and cost comparisons for saved mini/nano answer streams."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import tiktoken


PRICING_PER_1M = {
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
}


@dataclass(frozen=True)
class CountedRow:
    question_id: int
    answer: str
    pred: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float


def encoding_for(model: str) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("o200k_base")


def count_chat_tokens(messages: list[dict[str, Any]], encoding: tiktoken.Encoding) -> int:
    """Approximate modern OpenAI chat token accounting from message payloads."""
    tokens = 3
    for message in messages:
        tokens += 3
        for key, value in message.items():
            if value is None:
                continue
            tokens += len(encoding.encode(str(value)))
            if key == "name":
                tokens += 1
    return tokens


def count_prompt_tokens(row: dict[str, Any], encoding: tiktoken.Encoding) -> int:
    prompt = row.get("prompt")
    if isinstance(prompt, list):
        return count_chat_tokens(prompt, encoding)
    if isinstance(prompt, str):
        return len(encoding.encode(prompt))
    raise ValueError(f"Row {row.get('question_id')} has no usable prompt field.")


def count_rows(path: Path, model: str) -> dict[int, CountedRow]:
    data = json.loads(path.read_text(encoding="utf-8"))
    encoding = encoding_for(model)
    price = PRICING_PER_1M[model]
    rows: dict[int, CountedRow] = {}
    for row in data:
        input_tokens = count_prompt_tokens(row, encoding)
        output_tokens = len(encoding.encode(str(row.get("response", ""))))
        cost = (
            input_tokens * price["input"] / 1_000_000
            + output_tokens * price["output"] / 1_000_000
        )
        question_id = int(row["question_id"])
        rows[question_id] = CountedRow(
            question_id=question_id,
            answer=str(row["answer"]).strip().upper(),
            pred=str(row.get("pred", "")).strip().upper(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_usd=cost,
        )
    return rows


def route_for_oracle(mini: CountedRow, nano: CountedRow) -> str:
    mini_correct = mini.pred == mini.answer
    nano_correct = nano.pred == nano.answer
    if nano_correct:
        return "nano"
    if mini_correct:
        return "mini"
    return "nano"


def summarize_route(
    name: str,
    ids: Iterable[int],
    mini_rows: dict[int, CountedRow],
    nano_rows: dict[int, CountedRow],
    route_by_id: dict[int, str],
) -> dict[str, Any]:
    count = 0
    correct = 0
    mini_count = 0
    nano_count = 0
    input_tokens = 0
    output_tokens = 0
    cost = 0.0
    for question_id in ids:
        route = route_by_id[question_id]
        count += 1
        if route not in {"mini", "nano"}:
            continue
        row = mini_rows[question_id] if route == "mini" else nano_rows[question_id]
        mini_count += route == "mini"
        nano_count += route == "nano"
        correct += row.pred == row.answer
        input_tokens += row.input_tokens
        output_tokens += row.output_tokens
        cost += row.cost_usd
    return {
        "scenario": name,
        "questions": count,
        "accuracy": correct / count if count else 0.0,
        "correct": correct,
        "nano_routes": nano_count,
        "mini_routes": mini_count,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost_usd": cost,
        "cost_per_1000_questions": cost * 1000 / count if count else 0.0,
    }


def gepa_routes(path: Path, summary_name: str = "best_summary") -> dict[int, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    traces = payload[summary_name]["traces"]
    return {int(trace["question_id"]): str(trace["route"]).strip().lower() for trace in traces}


def add_deltas(rows: list[dict[str, Any]], baseline_name: str) -> list[dict[str, Any]]:
    baseline = next(row for row in rows if row["scenario"] == baseline_name)
    for row in rows:
        row["baseline"] = baseline_name
        row["cost_delta_vs_baseline"] = row["cost_usd"] - baseline["cost_usd"]
        row["cost_savings_vs_baseline"] = (
            1.0 - row["cost_usd"] / baseline["cost_usd"] if baseline["cost_usd"] else 0.0
        )
        row["accuracy_delta_vs_baseline"] = row["accuracy"] - baseline["accuracy"]
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "scenario",
        "questions",
        "accuracy",
        "correct",
        "nano_routes",
        "mini_routes",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cost_usd",
        "cost_per_1000_questions",
        "baseline",
        "cost_delta_vs_baseline",
        "cost_savings_vs_baseline",
        "accuracy_delta_vs_baseline",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mini-file", default="computer science_result_mini.json")
    parser.add_argument("--nano-file", default="computer science_result_nano.json")
    parser.add_argument("--gepa-result", default="gepa_router_prompt_result_50_openai_reflect.json")
    parser.add_argument("--output-json", default="routing_cost_comparison.json")
    parser.add_argument("--output-csv", default="routing_cost_comparison.csv")
    args = parser.parse_args()

    mini_rows = count_rows(Path(args.mini_file), "gpt-4.1-mini")
    nano_rows = count_rows(Path(args.nano_file), "gpt-4.1-nano")
    ids = sorted(set(mini_rows) & set(nano_rows))

    all_mini = {question_id: "mini" for question_id in ids}
    all_nano = {question_id: "nano" for question_id in ids}
    oracle = {question_id: route_for_oracle(mini_rows[question_id], nano_rows[question_id]) for question_id in ids}

    full_rows = [
        summarize_route("all_mini", ids, mini_rows, nano_rows, all_mini),
        summarize_route("all_nano", ids, mini_rows, nano_rows, all_nano),
        summarize_route("oracle_router", ids, mini_rows, nano_rows, oracle),
    ]
    add_deltas(full_rows, "all_mini")

    gepa_route_by_id = gepa_routes(Path(args.gepa_result))
    gepa_ids = sorted(gepa_route_by_id)
    gepa_rows = [
        summarize_route("all_mini_on_gepa50", gepa_ids, mini_rows, nano_rows, {i: "mini" for i in gepa_ids}),
        summarize_route("all_nano_on_gepa50", gepa_ids, mini_rows, nano_rows, {i: "nano" for i in gepa_ids}),
        summarize_route("oracle_on_gepa50", gepa_ids, mini_rows, nano_rows, {i: oracle[i] for i in gepa_ids}),
        summarize_route("gepa_best_on_gepa50", gepa_ids, mini_rows, nano_rows, gepa_route_by_id),
    ]
    add_deltas(gepa_rows, "all_mini_on_gepa50")

    payload = {
        "notes": [
            "Token counts use tiktoken with model-specific encodings and approximate modern chat message overhead.",
            "Costs use OpenAI public prices per 1M tokens: gpt-4.1-mini input $0.40/output $1.60; gpt-4.1-nano input $0.10/output $0.40.",
            "Router/optimizer overhead is excluded; this is answer-generation prompt plus completion cost.",
        ],
        "pricing_per_1m_tokens": PRICING_PER_1M,
        "full_dataset": full_rows,
        "gepa_50_subset": gepa_rows,
    }
    Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(Path(args.output_csv), full_rows + gepa_rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
