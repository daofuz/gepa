#!/usr/bin/env python3
"""Evaluate a recovered router prompt using a unique question_id trace cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from analyze_routing_costs import add_deltas, count_rows, route_for_oracle, summarize_route
from evaluate_router_testdata import (
    OllamaChat,
    classify_failure,
    format_question,
    load_examples,
    parse_router_output,
    score_route,
    write_csv,
)


def read_unique_traces(path: Path, allowed_ids: set[int]) -> dict[int, dict[str, Any]]:
    traces_by_id: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return traces_by_id
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        trace = json.loads(line)
        question_id = int(trace["question_id"])
        if question_id in allowed_ids:
            traces_by_id[question_id] = trace
    return traces_by_id


def write_unique_cache(path: Path, traces_by_id: dict[int, dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for question_id in sorted(traces_by_id):
            handle.write(json.dumps(traces_by_id[question_id], ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mini-file", default="computer science_result_mini.json")
    parser.add_argument("--nano-file", default="computer science_result_nano.json")
    parser.add_argument("--gepa-result", required=True)
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--ollama-timeout", type=float, default=90.0)
    parser.add_argument("--router-num-predict", type=int, default=16)
    parser.add_argument("--trace-cache", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    gepa_payload = json.loads(Path(args.gepa_result).read_text(encoding="utf-8"))
    held_in_ids = {
        int(trace["question_id"]) for trace in gepa_payload["best_summary"]["traces"]
    }
    best_prompt = gepa_payload["best_prompt"]
    router_model = gepa_payload.get("router_model") or gepa_payload.get("ollama_model") or "qwen3.5"
    router_output_format = gepa_payload.get("router_output_format") or gepa_payload.get(
        "scoring", {}
    ).get("router_output_format", "token")
    score_mode = gepa_payload.get("scoring", {}).get("score_mode", "accuracy_only")

    examples = load_examples(Path(args.mini_file), Path(args.nano_file))
    test_examples = [example for example in examples if example.question_id not in held_in_ids]
    test_ids = {example.question_id for example in test_examples}
    trace_cache_path = Path(args.trace_cache)
    traces_by_id = read_unique_traces(trace_cache_path, test_ids)
    write_unique_cache(trace_cache_path, traces_by_id)

    missing_examples = [example for example in test_examples if example.question_id not in traces_by_id]
    print(
        f"Loaded {len(traces_by_id)} unique cached routes; evaluating "
        f"{len(missing_examples)} remaining of {len(test_examples)} test questions.",
        flush=True,
    )

    router = OllamaChat(
        model=router_model,
        host=args.ollama_host,
        timeout=args.ollama_timeout,
        num_predict=args.router_num_predict,
        retry_empty_with_tokens=max(64, args.router_num_predict * 4),
    )

    with trace_cache_path.open("a", encoding="utf-8") as cache_handle:
        for index, example in enumerate(missing_examples, start=1):
            print(
                f"Routing {index}/{len(missing_examples)} "
                f"(question_id={example.question_id})...",
                flush=True,
            )
            messages = [
                {"role": "system", "content": best_prompt},
                {"role": "user", "content": format_question(example, router_output_format)},
            ]
            raw_response = router(messages)
            router_output = parse_router_output(raw_response, router_output_format)
            route = router_output["route"]
            score, selected_route = score_route(example, route, score_mode)
            mini_correct = example.mini_pred == example.answer
            nano_correct = example.nano_pred == example.answer
            trace = {
                "question_id": example.question_id,
                "question": example.question,
                "options": example.options,
                "gold_answer": example.answer,
                "mini_pred": example.mini_pred,
                "nano_pred": example.nano_pred,
                "mini_correct": mini_correct,
                "nano_correct": nano_correct,
                "route": route,
                "raw_response": raw_response,
                "router_output": router_output,
                "score": score,
                "selected_route": selected_route,
                "failure_type": classify_failure(route, nano_correct, mini_correct),
            }
            traces_by_id[example.question_id] = trace
            cache_handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
            cache_handle.flush()
            if index % args.progress_every == 0:
                print(
                    f"Completed {index}/{len(missing_examples)} new routes "
                    f"({len(traces_by_id)}/{len(test_examples)} total).",
                    flush=True,
                )

    write_unique_cache(trace_cache_path, traces_by_id)
    route_by_id = {question_id: trace["route"] for question_id, trace in traces_by_id.items()}
    invalid_routes = sum(1 for route in route_by_id.values() if route not in {"mini", "nano"})

    mini_rows = count_rows(Path(args.mini_file), "gpt-4.1-mini")
    nano_rows = count_rows(Path(args.nano_file), "gpt-4.1-nano")
    ordered_test_ids = [example.question_id for example in test_examples]
    all_mini = {question_id: "mini" for question_id in ordered_test_ids}
    all_nano = {question_id: "nano" for question_id in ordered_test_ids}
    oracle = {
        question_id: route_for_oracle(mini_rows[question_id], nano_rows[question_id])
        for question_id in ordered_test_ids
    }
    rows = [
        summarize_route("all_mini_on_testdata", ordered_test_ids, mini_rows, nano_rows, all_mini),
        summarize_route("all_nano_on_testdata", ordered_test_ids, mini_rows, nano_rows, all_nano),
        summarize_route("oracle_on_testdata", ordered_test_ids, mini_rows, nano_rows, oracle),
        summarize_route("gepa_best_on_testdata", ordered_test_ids, mini_rows, nano_rows, route_by_id),
    ]
    add_deltas(rows, "all_mini_on_testdata")
    payload = {
        "source_gepa_result": args.gepa_result,
        "test_questions": len(ordered_test_ids),
        "router_model": router_model,
        "router_output_format": router_output_format,
        "score_mode": score_mode,
        "invalid_router_outputs": invalid_routes,
        "invalid_json_outputs": sum(
            1 for trace in traces_by_id.values() if trace.get("router_output", {}).get("invalid_json")
        ),
        "test_question_ids": ordered_test_ids,
        "comparison": rows,
        "router_traces": [traces_by_id[question_id] for question_id in ordered_test_ids],
    }
    Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(Path(args.output_csv), rows)
    print(json.dumps({key: value for key, value in payload.items() if key != "router_traces"}, indent=2))


if __name__ == "__main__":
    main()
