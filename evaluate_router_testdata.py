#!/usr/bin/env python3
"""Evaluate a saved GEPA router prompt on examples held out from the GEPA subset."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import random
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from analyze_routing_costs import (
    add_deltas,
    count_rows,
    route_for_oracle,
    summarize_route,
)


RISK_AVERSE_UTILITY_RULE = {
    "both_correct_select_nano": 1.0,
    "both_correct_select_mini": 0.2,
    "mini_only_select_mini": 2.0,
    "mini_only_select_nano": -2.0,
    "nano_only_select_nano": 1.0,
    "nano_only_select_mini": -1.0,
    "both_wrong_select_nano": -0.5,
    "both_wrong_select_mini": 0.2,
    "invalid_route": -2.0,
}


@dataclass(frozen=True)
class RoutingExample:
    question_id: int
    question: str
    options: list[str]
    answer: str
    mini_pred: str
    nano_pred: str
    nano_response: str
    category: str
    src: str


def load_examples(mini_path: Path, nano_path: Path) -> list[RoutingExample]:
    mini_rows = json.loads(mini_path.read_text(encoding="utf-8"))
    nano_rows = json.loads(nano_path.read_text(encoding="utf-8"))
    nano_by_id = {row["question_id"]: row for row in nano_rows}
    examples: list[RoutingExample] = []
    for mini in mini_rows:
        nano = nano_by_id.get(mini["question_id"])
        if nano is None:
            continue
        if mini.get("answer") != nano.get("answer"):
            raise ValueError(f"Answer mismatch for question_id={mini['question_id']}")
        examples.append(
            RoutingExample(
                question_id=int(mini["question_id"]),
                question=mini["question"],
                options=list(mini["options"]),
                answer=mini["answer"],
                mini_pred=str(mini.get("pred", "")).strip().upper(),
                nano_pred=str(nano.get("pred", "")).strip().upper(),
                nano_response=str(nano.get("response", "")).strip(),
                category=mini.get("category", ""),
                src=mini.get("src", ""),
            )
        )
    return examples


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n[truncated]"


def format_question(
    example: RoutingExample,
    router_output_format: str = "token",
    include_nano_answer: bool = False,
    nano_response_max_chars: int = 4000,
) -> str:
    option_lines = []
    for i, option in enumerate(example.options):
        letter = chr(ord("A") + i)
        option_lines.append(f"{letter}. {option}")
    nano_answer_block = ""
    if include_nano_answer:
        nano_response = truncate_text(example.nano_response, nano_response_max_chars)
        nano_answer_block = (
            f"\n\nNano saved answer:\n"
            f"Nano predicted option: {example.nano_pred or '<missing>'}\n"
            f"Nano response:\n{nano_response or '<missing>'}\n"
        )
    if router_output_format == "json":
        if include_nano_answer:
            instruction = (
                "Decide whether to trust nano's saved answer or escalate to mini. "
                "Return compact JSON only with keys: route, difficulty, confidence, reason, "
                'risk_flags. route must be "nano" or "mini".'
            )
        else:
            instruction = (
                "Route this question to exactly one model. Return compact JSON only with keys: "
                'route, difficulty, confidence, reason, risk_flags. route must be "nano" or "mini".'
            )
    else:
        if include_nano_answer:
            instruction = (
                "Decide whether to trust nano's saved answer or escalate to mini. "
                "Return only: nano or mini."
            )
        else:
            instruction = "Route this question to exactly one model. Return only: nano or mini."
    return (
        f"Question ID: {example.question_id}\n"
        f"Category: {example.category}\n"
        f"Source: {example.src}\n\n"
        f"Question:\n{example.question}\n\n"
        f"Options:\n{chr(10).join(option_lines)}"
        f"{nano_answer_block}\n\n"
        f"{instruction}"
    )


def parse_route(text: str) -> str:
    matches = re.findall(r"\b(nano|mini)\b", text.strip().lower())
    if matches:
        return matches[-1]
    return "invalid"


def parse_router_output(text: str, router_output_format: str) -> dict[str, Any]:
    if router_output_format != "json":
        return {"route": parse_route(text), "invalid_json": False}
    raw = text.strip()
    invalid_json = False
    payload: dict[str, Any] = {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                payload = json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                invalid_json = True
        else:
            invalid_json = True
    route = parse_route(str(payload.get("route", ""))) if payload else parse_route(raw)
    return {
        "route": route,
        "difficulty": str(payload.get("difficulty", "")).strip().lower() if payload else "",
        "confidence": payload.get("confidence") if payload else None,
        "reason": str(payload.get("reason", "")).strip() if payload else "",
        "risk_flags": payload.get("risk_flags", []) if payload else [],
        "invalid_json": invalid_json,
    }


ALLOWED_EASY_ROUTER_BUILTINS = {
    "all": all,
    "any": any,
    "bool": bool,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
}


DISALLOWED_EASY_ROUTER_NODES = (
    ast.AsyncFunctionDef,
    ast.Await,
    ast.ClassDef,
    ast.Delete,
    ast.For,
    ast.Global,
    ast.Import,
    ast.ImportFrom,
    ast.Lambda,
    ast.Nonlocal,
    ast.Raise,
    ast.Try,
    ast.While,
    ast.With,
)


def extract_python_code(text: str) -> str:
    stripped = text.strip()
    fence_match = re.search(r"```(?:python)?\s*(.*?)```", stripped, re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()
    return stripped


def validate_easy_router_code(code: str) -> None:
    tree = ast.parse(code)
    if any(isinstance(node, DISALLOWED_EASY_ROUTER_NODES) for node in ast.walk(tree)):
        raise ValueError(
            "easy router code may not use imports, classes, loops, try/with, raise, "
            "lambda, async, global, or nonlocal statements."
        )
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1 or functions[0].name != "route_easy_case":
        raise ValueError("easy router code must define exactly one function named route_easy_case.")
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.Assign, ast.AnnAssign, ast.Expr)):
            raise ValueError("easy router code may only contain constants and route_easy_case().")


def compile_easy_router(code: str) -> tuple[Any | None, str | None]:
    code = extract_python_code(code)
    try:
        validate_easy_router_code(code)
        namespace: dict[str, Any] = {"__builtins__": ALLOWED_EASY_ROUTER_BUILTINS}
        exec(
            compile(code, "<easy_router_code>", "exec"),
            namespace,
            namespace,
        )
        fn = namespace.get("route_easy_case")
        if not callable(fn):
            raise ValueError("route_easy_case is not callable.")
        return fn, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def classify_failure(route: str, nano_correct: bool, mini_correct: bool) -> str:
    if route == "nano" and nano_correct and mini_correct:
        return "successful_saving"
    if route == "nano" and nano_correct and not mini_correct:
        return "nano_better_than_mini"
    if route == "nano" and not nano_correct and mini_correct:
        return "critical_misroute"
    if route == "nano" and not nano_correct and not mini_correct:
        return "both_failed_but_cheap"
    if route == "mini" and nano_correct:
        return "unnecessary_expensive_route"
    if route == "mini" and not nano_correct and mini_correct:
        return "correct_escalation"
    if route == "mini" and not nano_correct and not mini_correct:
        return "both_failed_expensive"
    return "unknown"


def outcome_bucket(example: RoutingExample) -> str:
    mini_correct = example.mini_pred == example.answer
    nano_correct = example.nano_pred == example.answer
    if mini_correct and nano_correct:
        return "both_correct"
    if mini_correct and not nano_correct:
        return "mini_only"
    if nano_correct and not mini_correct:
        return "nano_only"
    return "both_wrong"


class OllamaChat:
    def __init__(
        self,
        model: str,
        host: str,
        timeout: float,
        num_predict: int,
        retry_empty_with_tokens: int,
        request_retries: int = 3,
        retry_sleep_seconds: float = 2.0,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.num_predict = num_predict
        self.retry_empty_with_tokens = retry_empty_with_tokens
        self.request_retries = request_retries
        self.retry_sleep_seconds = retry_sleep_seconds

    def __call__(self, messages: list[dict[str, Any]]) -> str:
        content = self._chat_once(messages, self.num_predict)
        if content:
            return content
        retry_messages = list(messages)
        retry_messages.append(
            {
                "role": "user",
                "content": "Your previous response was empty. Reply now with the requested final answer only. /no_think",
            }
        )
        return self._chat_once(retry_messages, self.retry_empty_with_tokens)

    def _chat_once(self, messages: list[dict[str, Any]], num_predict: int) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"temperature": 0.0, "num_predict": num_predict},
        }
        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        attempts = max(1, self.request_retries + 1)
        for attempt in range(1, attempts + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                if exc.code < 500 or attempt >= attempts:
                    raise RuntimeError(
                        f"Could not call Ollama at {self.host}; is model '{self.model}' available?"
                    ) from exc
            except (TimeoutError, socket.timeout) as exc:
                if attempt >= attempts:
                    raise RuntimeError(
                        f"Ollama call to model '{self.model}' timed out after {self.timeout} seconds."
                    ) from exc
            except urllib.error.URLError as exc:
                if attempt >= attempts:
                    raise RuntimeError(
                        f"Could not call Ollama at {self.host}; is model '{self.model}' available?"
                    ) from exc
            time.sleep(self.retry_sleep_seconds * attempt)
        content = body.get("message", {}).get("content", "")
        return content.strip() if isinstance(content, str) else ""


def score_route(example: RoutingExample, route: str, score_mode: str) -> tuple[float, str]:
    mini_correct = example.mini_pred == example.answer
    nano_correct = example.nano_pred == example.answer
    if score_mode == "risk_averse_utility":
        if route not in {"nano", "mini"}:
            return RISK_AVERSE_UTILITY_RULE["invalid_route"], "invalid"
        return RISK_AVERSE_UTILITY_RULE[f"{outcome_bucket(example)}_select_{route}"], route
    if route == "nano":
        return (1.0 if nano_correct else 0.0), "nano"
    if route == "mini":
        return (1.0 if mini_correct else 0.0), "mini"
    return 0.0, "invalid"


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
    parser.add_argument(
        "--split-file",
        default=None,
        help="Optional split manifest with test_ids. When set, evaluates exactly those test IDs.",
    )
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--ollama-timeout", type=float, default=600.0)
    parser.add_argument("--router-num-predict", type=int, default=32)
    parser.add_argument(
        "--include-nano-answer",
        action="store_true",
        default=None,
        help="Include nano's saved prediction and response in the router input; defaults to the GEPA result setting.",
    )
    parser.add_argument(
        "--nano-response-max-chars",
        type=int,
        default=None,
        help="Maximum characters of nano's saved response to include; defaults to the GEPA result setting.",
    )
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0, help="Optional random test subset size; 0 uses all held-out examples.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trace-cache", default="routing_testdata_router_traces.jsonl")
    parser.add_argument("--output-json", default="routing_cost_comparison_testdata.json")
    parser.add_argument("--output-csv", default="routing_cost_comparison_testdata.csv")
    args = parser.parse_args()

    gepa_payload = json.loads(Path(args.gepa_result).read_text(encoding="utf-8"))
    held_in_ids = {int(trace["question_id"]) for trace in gepa_payload["best_summary"]["traces"]}
    best_prompt = gepa_payload["best_prompt"]
    best_easy_router_code = str(gepa_payload.get("best_easy_router_code") or "").strip()
    easy_router_fn = None
    easy_router_compile_error = None
    if best_easy_router_code:
        easy_router_fn, easy_router_compile_error = compile_easy_router(best_easy_router_code)
    router_model = gepa_payload.get("router_model") or gepa_payload.get("ollama_model") or "qwen3.5"
    router_output_format = gepa_payload.get("router_output_format") or gepa_payload.get(
        "scoring", {}
    ).get("router_output_format", "token")
    include_nano_answer = (
        bool(args.include_nano_answer)
        if args.include_nano_answer is not None
        else bool(gepa_payload.get("include_nano_answer", False))
    )
    nano_response_max_chars = (
        args.nano_response_max_chars
        if args.nano_response_max_chars is not None
        else int(gepa_payload.get("nano_response_max_chars", 4000))
    )
    score_mode = gepa_payload.get("scoring", {}).get("score_mode", "accuracy_only")

    examples = load_examples(Path(args.mini_file), Path(args.nano_file))
    if args.split_file:
        split_payload = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
        test_ids_from_split = {int(question_id) for question_id in split_payload["test_ids"]}
        test_examples = [example for example in examples if example.question_id in test_ids_from_split]
    else:
        test_examples = [example for example in examples if example.question_id not in held_in_ids]
    if args.limit > 0 and args.limit < len(test_examples):
        rng = random.Random(args.seed)
        test_examples = rng.sample(test_examples, args.limit)
        test_examples.sort(key=lambda example: example.question_id)

    router = OllamaChat(
        model=router_model,
        host=args.ollama_host,
        timeout=args.ollama_timeout,
        num_predict=args.router_num_predict,
        retry_empty_with_tokens=max(128, args.router_num_predict * 4),
    )
    traces: list[dict[str, Any]] = []
    trace_cache_path = Path(args.trace_cache)
    if trace_cache_path.exists():
        for line in trace_cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                trace = json.loads(line)
                traces.append(trace)

    route_by_id: dict[int, str] = {
        int(trace["question_id"]): str(trace["route"]).strip().lower()
        for trace in traces
    }
    remaining_examples = [example for example in test_examples if example.question_id not in route_by_id]
    print(
        f"Loaded {len(route_by_id)} cached routes; evaluating {len(remaining_examples)} remaining "
        f"of {len(test_examples)} test questions.",
        flush=True,
    )

    with trace_cache_path.open("a", encoding="utf-8") as cache_handle:
        for index, example in enumerate(remaining_examples, start=1):
            easy_router_source = "disabled"
            easy_router_error = None
            if best_easy_router_code and easy_router_compile_error:
                raw_response = f"easy_router_error: {easy_router_compile_error}"
                router_output = {"route": "invalid", "invalid_json": False}
                easy_router_source = "easy_router_error"
                easy_router_error = easy_router_compile_error
            elif easy_router_fn is not None:
                try:
                    decision = easy_router_fn(
                        example.question,
                        example.options,
                        example.category,
                        example.src,
                    )
                    easy_route = str(decision or "defer").strip().lower()
                    if easy_route in {"nano", "mini"}:
                        raw_response = f"easy_router:{easy_route}"
                        router_output = {"route": easy_route, "invalid_json": False}
                        easy_router_source = "easy_router"
                    elif easy_route in {"defer", "qwen", "llm", "none"}:
                        router_output = None
                        easy_router_source = "easy_router_defer"
                    else:
                        router_output = None
                        easy_router_source = "easy_router_invalid"
                        easy_router_error = f"route_easy_case returned {decision!r}"
                except Exception as exc:
                    raw_response = f"easy_router_error: {type(exc).__name__}: {exc}"
                    router_output = {"route": "invalid", "invalid_json": False}
                    easy_router_source = "easy_router_error"
                    easy_router_error = f"{type(exc).__name__}: {exc}"
            else:
                router_output = None

            if router_output is None:
                messages = [
                    {"role": "system", "content": best_prompt},
                    {
                        "role": "user",
                        "content": format_question(
                            example,
                            router_output_format,
                            include_nano_answer,
                            nano_response_max_chars,
                        ),
                    },
                ]
                raw_response = router(messages)
                router_output = parse_router_output(raw_response, router_output_format)
                if easy_router_source in {"disabled", "easy_router_defer"}:
                    easy_router_error = None
            route = router_output["route"]
            route_by_id[example.question_id] = route
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
                "nano_response": example.nano_response,
                "mini_correct": mini_correct,
                "nano_correct": nano_correct,
                "route": route,
                "raw_response": raw_response,
                "easy_router_source": easy_router_source,
                "easy_router_error": easy_router_error,
                "router_output": router_output,
                "score": score,
                "selected_route": selected_route,
                "failure_type": classify_failure(route, nano_correct, mini_correct),
            }
            traces.append(trace)
            cache_handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
            cache_handle.flush()
            if index == 1 or index % 25 == 0:
                print(
                    f"Evaluated {index}/{len(remaining_examples)} new routes "
                    f"({len(route_by_id)}/{len(test_examples)} total).",
                    flush=True,
                )
            if args.sleep_seconds:
                time.sleep(args.sleep_seconds)
    invalid_routes = sum(1 for route in route_by_id.values() if route not in {"mini", "nano"})

    mini_rows = count_rows(Path(args.mini_file), "gpt-4.1-mini")
    nano_rows = count_rows(Path(args.nano_file), "gpt-4.1-nano")
    test_ids = [example.question_id for example in test_examples]
    all_mini = {question_id: "mini" for question_id in test_ids}
    all_nano = {question_id: "nano" for question_id in test_ids}
    oracle = {question_id: route_for_oracle(mini_rows[question_id], nano_rows[question_id]) for question_id in test_ids}

    rows = [
        summarize_route("all_mini_on_testdata", test_ids, mini_rows, nano_rows, all_mini),
        summarize_route("all_nano_on_testdata", test_ids, mini_rows, nano_rows, all_nano),
        summarize_route("oracle_on_testdata", test_ids, mini_rows, nano_rows, oracle),
        summarize_route("gepa_best_on_testdata", test_ids, mini_rows, nano_rows, route_by_id),
    ]
    add_deltas(rows, "all_mini_on_testdata")

    payload = {
        "source_gepa_result": args.gepa_result,
        "split_file": args.split_file,
        "excluded_gepa_subset_questions": len(held_in_ids),
        "test_questions": len(test_ids),
        "router_model": router_model,
        "router_output_format": router_output_format,
        "include_nano_answer": include_nano_answer,
        "nano_response_max_chars": nano_response_max_chars,
        "hybrid_easy_router": bool(best_easy_router_code),
        "easy_router_compile_error": easy_router_compile_error,
        "score_mode": score_mode,
        "invalid_router_outputs": invalid_routes,
        "invalid_json_outputs": sum(
            1 for trace in traces if trace.get("router_output", {}).get("invalid_json")
        ),
        "test_question_ids": test_ids,
        "comparison": rows,
        "router_traces": traces,
    }
    Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(Path(args.output_csv), rows)
    print(json.dumps({k: v for k, v in payload.items() if k != "router_traces"}, indent=2))


if __name__ == "__main__":
    main()
