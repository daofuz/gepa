#!/usr/bin/env python3
"""Filter router traces where mini was correct but the router chose nano."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def normalize_answer(value: Any) -> str:
    return str(value).strip().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="routing_cost_comparison_testdata_pareto_ollama.json",
        help="Router evaluation JSON containing router_traces.",
    )
    parser.add_argument(
        "--output-json",
        default="mini_correct_routed_nano_testdata_pareto_ollama.json",
    )
    parser.add_argument(
        "--output-csv",
        default="mini_correct_routed_nano_testdata_pareto_ollama.csv",
    )
    parser.add_argument(
        "--require-nano-wrong",
        action="store_true",
        help="Keep only rows where nano_pred does not match the gold answer.",
    )
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    traces = payload.get("router_traces", [])
    filtered = []

    for trace in traces:
        gold = normalize_answer(trace.get("gold_answer"))
        mini_pred = normalize_answer(trace.get("mini_pred"))
        nano_pred = normalize_answer(trace.get("nano_pred"))
        route = str(trace.get("route", "")).strip().lower()

        nano_correct = nano_pred == gold
        if mini_pred == gold and route == "nano" and (
            not args.require_nano_wrong or not nano_correct
        ):
            filtered.append(
                {
                    "question_id": trace.get("question_id"),
                    "route": route,
                    "gold_answer": gold,
                    "mini_pred": mini_pred,
                    "nano_pred": nano_pred,
                    "nano_correct": nano_correct,
                    "mini_would_fix": not nano_correct,
                    "raw_response": trace.get("raw_response", ""),
                    "question": trace.get("question", ""),
                    "options": trace.get("options", []),
                }
            )

    summary = {
        "source": args.input,
        "total_traces": len(traces),
        "mini_correct_routed_nano": len(filtered),
        "mini_correct_routed_nano_where_nano_wrong": sum(
            1 for row in filtered if row["mini_would_fix"]
        ),
        "rows": filtered,
    }
    Path(args.output_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fieldnames = [
        "question_id",
        "route",
        "gold_answer",
        "mini_pred",
        "nano_pred",
        "nano_correct",
        "mini_would_fix",
        "raw_response",
        "question",
        "options",
    ]
    with Path(args.output_csv).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in filtered:
            csv_row = dict(row)
            csv_row["options"] = json.dumps(row["options"], ensure_ascii=False)
            writer.writerow(csv_row)

    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
