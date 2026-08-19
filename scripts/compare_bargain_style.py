#!/usr/bin/env python3
"""Compare routers against a BARGAIN-style oracle-agreement cascade.

The saved mini/nano answer streams do not contain token logprobs, so this script
cannot reproduce BARGAIN's thresholded proxy-confidence procedure exactly.
Instead, it adds the experiment-level comparator that BARGAIN is trying to
approximate when mini is the oracle: use nano only when nano's answer agrees
with mini, otherwise escalate to mini. This is a perfect oracle-agreement
cascade, not a deployable router, and it isolates the difference between
matching an oracle LLM distribution and optimizing ground-truth accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyze_routing_costs import add_deltas, count_rows, route_for_oracle, summarize_route


def load_split_ids(split_file: Path | None) -> tuple[list[int], list[int]]:
    if split_file is None:
        return [], []
    payload = json.loads(split_file.read_text(encoding="utf-8"))
    train_ids = [int(question_id) for question_id in payload.get("train_ids", [])]
    val_ids = [int(question_id) for question_id in payload.get("val_ids") or payload.get("validation_ids") or []]
    return train_ids, val_ids


def load_ids(split_file: Path | None, softprompt_result: Path | None, comparison_json: Path | None) -> list[int]:
    if split_file is not None:
        payload = json.loads(split_file.read_text(encoding="utf-8"))
        return [int(question_id) for question_id in payload["test_ids"]]
    if softprompt_result is not None:
        payload = json.loads(softprompt_result.read_text(encoding="utf-8"))
        traces = payload.get("test_summary", {}).get("traces") or []
        if traces:
            return [int(trace["question_id"]) for trace in traces]
    if comparison_json is not None:
        payload = json.loads(comparison_json.read_text(encoding="utf-8"))
        ids = payload.get("test_question_ids") or []
        if ids:
            return [int(question_id) for question_id in ids]
    raise ValueError("Provide --split-file, --softprompt-result with test traces, or --comparison-json.")


def load_softprompt_routes(path: Path | None) -> dict[int, str]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    traces = payload.get("test_summary", {}).get("traces") or []
    return {int(trace["question_id"]): str(trace["route"]).strip().lower() for trace in traces}


def perfect_oracle_agreement_routes(ids: Iterable[int], mini_rows: dict[int, Any], nano_rows: dict[int, Any]) -> dict[int, str]:
    return {
        question_id: "nano" if nano_rows[question_id].pred == mini_rows[question_id].pred else "mini"
        for question_id in ids
    }


def annotate_row(row: dict[str, Any], ids: Iterable[int], mini_rows: dict[int, Any], nano_rows: dict[int, Any], route_by_id: dict[int, str]) -> dict[str, Any]:
    failure_counts: Counter[str] = Counter()
    oracle_agree = 0
    nano_only_captured = 0
    critical_misroutes = 0
    agreement_nano_routes = 0
    disagreement_escalations = 0

    for question_id in ids:
        route = route_by_id[question_id]
        mini = mini_rows[question_id]
        nano = nano_rows[question_id]
        mini_correct = mini.pred == mini.answer
        nano_correct = nano.pred == nano.answer
        selected_pred = nano.pred if route == "nano" else mini.pred
        if selected_pred == mini.pred:
            oracle_agree += 1
        if route == "nano" and nano.pred == mini.pred:
            agreement_nano_routes += 1
        if route == "mini" and nano.pred != mini.pred:
            disagreement_escalations += 1
        if route == "nano" and nano_correct and not mini_correct:
            nano_only_captured += 1
        if route == "nano" and not nano_correct and mini_correct:
            critical_misroutes += 1
        if route == "nano" and nano_correct and mini_correct:
            failure_counts["successful_saving"] += 1
        elif route == "nano" and nano_correct and not mini_correct:
            failure_counts["nano_better_than_mini"] += 1
        elif route == "nano" and not nano_correct and mini_correct:
            failure_counts["critical_misroute"] += 1
        elif route == "nano":
            failure_counts["both_failed_but_cheap"] += 1
        elif route == "mini" and nano_correct:
            failure_counts["unnecessary_expensive_route"] += 1
        elif route == "mini" and mini_correct:
            failure_counts["correct_escalation"] += 1
        else:
            failure_counts["both_failed_expensive"] += 1

    count = row["questions"] or 1
    row["mini_oracle_agreement_rate"] = oracle_agree / count
    row["agreement_nano_routes"] = agreement_nano_routes
    row["disagreement_escalations"] = disagreement_escalations
    row["nano_only_captured"] = nano_only_captured
    row["critical_misroutes"] = critical_misroutes
    row["failure_counts"] = dict(failure_counts)
    return row


def summarize_named_route(name: str, ids: list[int], mini_rows: dict[int, Any], nano_rows: dict[int, Any], route_by_id: dict[int, str]) -> dict[str, Any]:
    row = summarize_route(name, ids, mini_rows, nano_rows, route_by_id)
    return annotate_row(row, ids, mini_rows, nano_rows, route_by_id)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "scenario",
        "questions",
        "accuracy",
        "correct",
        "nano_routes",
        "mini_routes",
        "cost_usd",
        "cost_per_1000_questions",
        "baseline",
        "cost_savings_vs_baseline",
        "accuracy_delta_vs_baseline",
        "mini_oracle_agreement_rate",
        "agreement_nano_routes",
        "disagreement_escalations",
        "nano_only_captured",
        "critical_misroutes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mini-file", default="computer science_result_mini.json")
    parser.add_argument("--nano-file", default="computer science_result_nano.json")
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--softprompt-result", default=None)
    parser.add_argument("--comparison-json", default=None)
    parser.add_argument("--output-json", default="bargain_style_comparison.json")
    parser.add_argument("--output-csv", default="bargain_style_comparison.csv")
    args = parser.parse_args()

    split_file = Path(args.split_file) if args.split_file else None
    softprompt_result = Path(args.softprompt_result) if args.softprompt_result else None
    comparison_json = Path(args.comparison_json) if args.comparison_json else None

    mini_rows = count_rows(Path(args.mini_file), "gpt-4.1-mini")
    nano_rows = count_rows(Path(args.nano_file), "gpt-4.1-nano")
    train_ids, val_ids = load_split_ids(split_file)
    calibration_ids = train_ids + val_ids
    ids = load_ids(split_file, softprompt_result, comparison_json)

    missing = [question_id for question_id in ids if question_id not in mini_rows or question_id not in nano_rows]
    if missing:
        raise ValueError(f"Missing mini/nano rows for {len(missing)} ids, first={missing[:5]}")

    all_mini = {question_id: "mini" for question_id in ids}
    all_nano = {question_id: "nano" for question_id in ids}
    ground_truth_oracle = {question_id: route_for_oracle(mini_rows[question_id], nano_rows[question_id]) for question_id in ids}
    bargain_perfect = perfect_oracle_agreement_routes(ids, mini_rows, nano_rows)
    softprompt_routes = load_softprompt_routes(softprompt_result)

    rows = [
        summarize_named_route("all_mini_on_testdata", ids, mini_rows, nano_rows, all_mini),
        summarize_named_route("all_nano_on_testdata", ids, mini_rows, nano_rows, all_nano),
        summarize_named_route("ground_truth_oracle_on_testdata", ids, mini_rows, nano_rows, ground_truth_oracle),
        summarize_named_route("bargain_perfect_mini_agreement_on_testdata", ids, mini_rows, nano_rows, bargain_perfect),
    ]
    if softprompt_routes:
        missing_softprompt = [question_id for question_id in ids if question_id not in softprompt_routes]
        if missing_softprompt:
            raise ValueError(f"Soft-prompt result is missing {len(missing_softprompt)} ids, first={missing_softprompt[:5]}")
        rows.append(summarize_named_route("softprompt_on_testdata", ids, mini_rows, nano_rows, softprompt_routes))

    add_deltas(rows, "all_mini_on_testdata")

    payload: dict[str, Any] = {
        "notes": [
            "BARGAIN uses proxy confidence/logprob thresholds to preserve agreement with an oracle LLM; the saved answer streams here do not include logprobs.",
            "bargain_perfect_mini_agreement_on_testdata is therefore an oracle-agreement ceiling: route to nano when nano and mini predict the same answer, otherwise route to mini.",
            "This row is not deployable without querying or predicting oracle agreement, but it isolates the objective mismatch: it can preserve mini's distribution while ignoring human ground-truth labels.",
        ],
        "mini_file": args.mini_file,
        "nano_file": args.nano_file,
        "split_file": args.split_file,
        "softprompt_result": args.softprompt_result,
        "comparison_json": args.comparison_json,
        "calibration_questions": len(calibration_ids),
        "train_questions": len(train_ids),
        "validation_questions": len(val_ids),
        "test_questions": len(ids),
        "calibration_question_ids": calibration_ids,
        "test_question_ids": ids,
        "comparison": rows,
    }

    if softprompt_result is not None:
        soft_payload = json.loads(softprompt_result.read_text(encoding="utf-8"))
        payload["softprompt_threshold_summaries"] = soft_payload.get("test_threshold_summaries")

    Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(Path(args.output_csv), rows)
    print(json.dumps({"calibration_questions": len(calibration_ids), "test_questions": len(ids), "comparison": rows}, indent=2))


if __name__ == "__main__":
    main()




