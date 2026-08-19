#!/usr/bin/env python3
"""
Optimize an Ollama/Qwen routing prompt with GEPA.

The router sees an MMLU-Pro question and chooses which saved answer stream to use:
`mini` for gpt-4.1-mini or `nano` for gpt-4.1-nano. The score strongly rewards
answer correctness and gives a smaller bonus for choosing the cheaper nano route.

Example:
    python optimize_ollama_router_gepa.py --ollama-model qwen3.5 --limit 10
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import gepa
import tiktoken
from gepa import EvaluationBatch, GEPAAdapter

from adaevolve_router_optimizer import (
    load_seed_candidates_from_results,
    run_adaevolve,
)


PRICING_PER_1M = {
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
}

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


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


DEFAULT_PROMPT = """You are a cost-aware model router for MMLU-Pro computer science questions.

Choose exactly one model:
- nano: cheaper and preferred whenever it is likely to preserve accuracy.
- mini: more capable and more expensive; choose it only when the question is likely to expose nano's weakness.

Routing rules:
1. Default to nano for straightforward definitions, standard facts, simple code tracing, and basic algorithm questions.
2. Choose mini for questions with multi-step calculation, tricky formal logic, subtle architecture/protocol distinctions, or dense reasoning where nano may confidently pick the wrong option.
3. If both models are likely to answer correctly, choose nano to save cost.
4. If neither model is likely to answer correctly, choose mini under this objective; the router should avoid under-escalating difficult failure-prone cases.

Examples:
- Basic definition or common concept -> nano
- Simple loop count or direct AP-style CS concept -> nano
- Cache simulation, matrix/rank trick, unification, or precise protocol edge case -> mini
- Ambiguous wording with a subtle NOT/EXCEPT distinction -> mini

Respond with exactly one token: nano or mini."""


DEFAULT_STRUCTURED_PROMPT = """You are a cost-aware model router for MMLU-Pro computer science questions.

Choose exactly one model:
- nano: cheaper and preferred whenever it is likely to preserve accuracy.
- mini: more capable and more expensive; choose it only when the question is likely to expose nano's weakness.

Routing rules:
1. Default to nano for straightforward definitions, standard facts, simple code tracing, and basic algorithm questions.
2. Choose mini for questions with multi-step calculation, tricky formal logic, subtle architecture/protocol distinctions, dense reasoning, or hidden traps where nano may confidently pick the wrong option.
3. If both models are likely to answer correctly, choose nano to save cost.
4. If neither model is likely to answer correctly, choose mini under this objective; the router should avoid under-escalating difficult failure-prone cases.

Return compact JSON only, with these keys:
- route: "nano" or "mini"
- difficulty: "easy", "medium", or "hard"
- confidence: number from 0.0 to 1.0
- reason: short explanation of why this route is appropriate
- risk_flags: short list of risk patterns, such as "multi_step", "not_except", "formal_logic", "protocol_edge_case", "calculation", "ambiguous_wording", "none"

Do not include markdown, prose outside JSON, or any extra keys."""


DEFAULT_EASY_ROUTER_CODE = '''def route_easy_case(question, options, category="", source=""):
    """Return 'nano' for obvious easy cases; return 'defer' for the Qwen router."""
    text = (question + "\\n" + "\\n".join(options)).lower()

    hard_markers = [
        "not", "except", "none of the above", "not enough information",
        "which of the following", "i.", "ii.", "iii.",
        "cache", "matrix", "rank", "unification", "boolean", "recurs",
        "capacity", "entropy", "half-life", "bayesian", "vertex cover",
        "maximum clique", "maximum flow", "chord", "dht", "nmap",
    ]
    if any(marker in text for marker in hard_markers):
        return "defer"

    easy_markers = [
        "what is", "what are", "define", "definition", "stands for",
        "abbreviation", "simple loop", "repeat", "standard fact",
    ]
    if len(question) < 180 and any(marker in text for marker in easy_markers):
        return "nano"

    return "defer"
'''


DEFAULT_NANO_ANSWER_PROMPT = """You are a risk-aware verifier/router for MMLU-Pro computer science questions.

You will see the question, answer options, and nano's saved answer. Choose exactly one model:
- nano: trust nano's saved answer when its reasoning is correct-looking, complete, internally consistent, and matches the selected option.
- mini: escalate when nano's saved answer is likely wrong and mini may recover the correct answer.

Your primary objective is to avoid critical misroutes: choosing nano when nano is wrong and mini is correct. This error is worse than paying extra for mini.

Choose mini when nano's saved answer:
1. Uses a questionable formula or skips a required calculation.
2. Ignores NOT, EXCEPT, ONLY, "cannot be determined", "none of the above", or similar qualifiers.
3. Contradicts itself, changes its conclusion, or its reasoning does not support its final option.
4. Treats an obscure protocol, tool, language, architecture, graph, probability, or formal-logic question as a routine fact.
5. Gives a plausible answer but does not rule out close distractors.

Choose nano when nano's saved answer is straightforward, well-supported, and the question is routine enough that mini is unlikely to improve accuracy. If both models are likely to fail, choose mini under this objective to avoid under-escalating difficult failure-prone cases.

Respond with exactly one token: nano or mini."""


DEFAULT_STRUCTURED_NANO_ANSWER_PROMPT = """You are a risk-aware verifier/router for MMLU-Pro computer science questions.

You will see the question, answer options, and nano's saved answer. Choose exactly one model:
- nano: trust nano's saved answer when its reasoning is correct-looking, complete, internally consistent, and matches the selected option.
- mini: escalate when nano's saved answer is likely wrong and mini may recover the correct answer.

Your primary objective is to avoid critical misroutes: choosing nano when nano is wrong and mini is correct. This error is worse than paying extra for mini.

Return compact JSON only, with these keys:
- route: "nano" or "mini"
- difficulty: "easy", "medium", or "hard"
- confidence: number from 0.0 to 1.0
- reason: short explanation of whether nano's saved answer should be trusted
- risk_flags: short list of nano-answer risk patterns, such as "wrong_formula", "unsupported_option", "missed_qualifier", "contradiction", "close_distractors", "obscure_fact", "none"

Do not include markdown, prose outside JSON, or any extra keys."""


@dataclass(frozen=True)
class RoutingExample:
    question_id: int
    question: str
    options: list[str]
    answer: str
    mini_pred: str
    nano_pred: str
    nano_response: str
    mini_cost_usd: float
    nano_cost_usd: float
    mini_input_tokens: int
    mini_output_tokens: int
    nano_input_tokens: int
    nano_output_tokens: int
    category: str
    src: str


@dataclass(frozen=True)
class RoutingOutput:
    raw_response: str
    route: str
    difficulty: str = ""
    confidence: float | None = None
    reason: str = ""
    risk_flags: list[str] | None = None
    invalid_json: bool = False


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


def encoding_for(model: str) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("o200k_base")


def count_chat_tokens(messages: list[dict[str, Any]], encoding: tiktoken.Encoding) -> int:
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


def count_answer_tokens(row: dict[str, Any], model: str) -> tuple[int, int, float]:
    if model not in PRICING_PER_1M:
        raise ValueError(
            f"No pricing configured for {model!r}. Add it to PRICING_PER_1M before using it."
        )
    encoding = encoding_for(model)
    input_tokens = count_prompt_tokens(row, encoding)
    output_tokens = len(encoding.encode(str(row.get("response", ""))))
    price = PRICING_PER_1M[model]
    cost_usd = (
        input_tokens * price["input"] / 1_000_000
        + output_tokens * price["output"] / 1_000_000
    )
    return input_tokens, output_tokens, cost_usd


def load_examples(
    mini_path: Path,
    nano_path: Path,
    mini_cost_model: str = "gpt-4.1-mini",
    nano_cost_model: str = "gpt-4.1-nano",
) -> list[RoutingExample]:
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

        mini_input_tokens, mini_output_tokens, mini_cost_usd = count_answer_tokens(
            mini, mini_cost_model
        )
        nano_input_tokens, nano_output_tokens, nano_cost_usd = count_answer_tokens(
            nano, nano_cost_model
        )

        examples.append(
            RoutingExample(
                question_id=int(mini["question_id"]),
                question=mini["question"],
                options=list(mini["options"]),
                answer=mini["answer"],
                mini_pred=str(mini.get("pred", "")).strip().upper(),
                nano_pred=str(nano.get("pred", "")).strip().upper(),
                nano_response=str(nano.get("response", "")).strip(),
                mini_cost_usd=mini_cost_usd,
                nano_cost_usd=nano_cost_usd,
                mini_input_tokens=mini_input_tokens,
                mini_output_tokens=mini_output_tokens,
                nano_input_tokens=nano_input_tokens,
                nano_output_tokens=nano_output_tokens,
                category=mini.get("category", ""),
                src=mini.get("src", ""),
            )
        )
    if not examples:
        raise ValueError("No joined examples were loaded.")
    return examples


def select_examples(
    examples: list[RoutingExample],
    limit: int,
    sampling: str,
    seed: int,
) -> list[RoutingExample]:
    if limit <= 0 or limit >= len(examples):
        return examples
    if sampling == "sequential":
        return examples[:limit]
    if sampling == "random":
        rng = random.Random(seed)
        return rng.sample(examples, limit)

    rng = random.Random(seed)
    buckets: dict[str, list[RoutingExample]] = {
        "mini_only": [],
        "nano_only": [],
        "both_correct": [],
        "both_wrong": [],
    }
    for example in examples:
        buckets[outcome_bucket(example)].append(example)
    for bucket_examples in buckets.values():
        rng.shuffle(bucket_examples)

    order = ["mini_only", "nano_only", "both_correct", "both_wrong"]
    selected: list[RoutingExample] = []
    while len(selected) < limit and any(buckets[name] for name in order):
        for name in order:
            if buckets[name] and len(selected) < limit:
                selected.append(buckets[name].pop())

    return selected


def parse_bucket_counts(raw: str) -> dict[str, int]:
    counts = {"mini_only": 0, "nano_only": 0, "both_correct": 0, "both_wrong": 0}
    if not raw.strip():
        return counts
    for part in raw.split(","):
        if "=" not in part:
            raise ValueError(f"Invalid bucket count {part!r}; expected name=count.")
        name, value = part.split("=", 1)
        name = name.strip()
        if name not in counts:
            raise ValueError(f"Unknown bucket {name!r}; expected one of {sorted(counts)}.")
        counts[name] = int(value.strip())
        if counts[name] < 0:
            raise ValueError(f"Bucket count for {name!r} must be non-negative.")
    return counts


def select_by_bucket_counts(
    buckets: dict[str, list[RoutingExample]],
    counts: dict[str, int],
    label: str,
) -> list[RoutingExample]:
    selected: list[RoutingExample] = []
    for name, count in counts.items():
        available = buckets[name]
        if count > len(available):
            raise ValueError(
                f"Requested {count} {name} examples for {label}, but only "
                f"{len(available)} are available."
            )
        selected.extend(available[:count])
        del available[:count]
    return selected


def split_train_val_targeted(
    examples: list[RoutingExample],
    train_counts: dict[str, int],
    val_counts: dict[str, int],
    seed: int,
) -> tuple[list[RoutingExample], list[RoutingExample]]:
    rng = random.Random(seed)
    buckets: dict[str, list[RoutingExample]] = {
        "mini_only": [],
        "nano_only": [],
        "both_correct": [],
        "both_wrong": [],
    }
    for example in examples:
        buckets[outcome_bucket(example)].append(example)
    for bucket_examples in buckets.values():
        rng.shuffle(bucket_examples)

    trainset = select_by_bucket_counts(buckets, train_counts, "train")
    valset = select_by_bucket_counts(buckets, val_counts, "validation")
    rng.shuffle(trainset)
    rng.shuffle(valset)
    return trainset, valset


def split_train_val_from_file(
    examples: list[RoutingExample],
    split_path: Path,
) -> tuple[list[RoutingExample], list[RoutingExample]]:
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    example_by_id = {example.question_id: example for example in examples}

    def resolve_ids(key: str) -> list[RoutingExample]:
        resolved = []
        for question_id in payload[key]:
            question_id = int(question_id)
            if question_id not in example_by_id:
                raise ValueError(f"{split_path} references unknown question_id={question_id}")
            resolved.append(example_by_id[question_id])
        return resolved

    return resolve_ids("train_ids"), resolve_ids("val_ids")


def split_train_val(
    examples: list[RoutingExample],
    train_ratio: float,
    seed: int,
) -> tuple[list[RoutingExample], list[RoutingExample]]:
    rng = random.Random(seed)
    buckets: dict[str, list[RoutingExample]] = {}
    for example in examples:
        buckets.setdefault(outcome_bucket(example), []).append(example)

    trainset: list[RoutingExample] = []
    valset: list[RoutingExample] = []
    for bucket_examples in buckets.values():
        rng.shuffle(bucket_examples)
        if len(bucket_examples) == 1:
            trainset.extend(bucket_examples)
            continue
        split_at = max(1, min(len(bucket_examples) - 1, int(len(bucket_examples) * train_ratio)))
        trainset.extend(bucket_examples[:split_at])
        valset.extend(bucket_examples[split_at:])

    if not valset and len(trainset) > 1:
        valset.append(trainset.pop())
    rng.shuffle(trainset)
    rng.shuffle(valset)
    return trainset, valset


def count_outcomes(examples: list[RoutingExample]) -> dict[str, int]:
    counts = {"mini_only": 0, "nano_only": 0, "both_correct": 0, "both_wrong": 0}
    for example in examples:
        counts[outcome_bucket(example)] += 1
    return counts


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
    options = "\n".join(option_lines)
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
        f"Options:\n{options}"
        f"{nano_answer_block}\n\n"
        f"{instruction}"
    )


def parse_route(text: str) -> str:
    normalized = text.strip().lower()
    matches = re.findall(r"\b(nano|mini)\b", normalized)
    if matches:
        return matches[-1]
    return "invalid"


def parse_router_output(text: str, router_output_format: str) -> RoutingOutput:
    if router_output_format != "json":
        return RoutingOutput(raw_response=text, route=parse_route(text))

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
    difficulty = str(payload.get("difficulty", "")).strip().lower() if payload else ""
    confidence_raw = payload.get("confidence") if payload else None
    try:
        confidence = float(confidence_raw) if confidence_raw is not None else None
    except (TypeError, ValueError):
        confidence = None
    if confidence is not None:
        confidence = max(0.0, min(1.0, confidence))
    reason = str(payload.get("reason", "")).strip() if payload else ""
    risk_flags_raw = payload.get("risk_flags", []) if payload else []
    if isinstance(risk_flags_raw, list):
        risk_flags = [str(flag).strip() for flag in risk_flags_raw if str(flag).strip()]
    elif risk_flags_raw:
        risk_flags = [str(risk_flags_raw).strip()]
    else:
        risk_flags = []
    return RoutingOutput(
        raw_response=text,
        route=route,
        difficulty=difficulty,
        confidence=confidence,
        reason=reason,
        risk_flags=risk_flags,
        invalid_json=invalid_json,
    )


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


def run_easy_router(
    code: str,
    example: RoutingExample,
) -> tuple[str, str, str | None]:
    fn, error = compile_easy_router(code)
    if fn is None:
        return "defer", "easy_router_error", error
    try:
        decision = fn(example.question, example.options, example.category, example.src)
    except Exception as exc:
        return "defer", "easy_router_error", f"{type(exc).__name__}: {exc}"
    route = str(decision or "defer").strip().lower()
    if route in {"nano", "mini"}:
        return route, "easy_router", None
    if route in {"defer", "qwen", "llm", "none"}:
        return "defer", "easy_router_defer", None
    return "defer", "easy_router_invalid", f"route_easy_case returned {decision!r}"


class OllamaChat:
    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        temperature: float = 0.0,
        timeout: float = 120.0,
        num_predict: int | None = None,
        think: bool | None = False,
        retry_empty_with_tokens: int | None = 256,
        request_retries: int = 3,
        retry_sleep_seconds: float = 2.0,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self.num_predict = num_predict
        self.think = think
        self.retry_empty_with_tokens = retry_empty_with_tokens
        self.request_retries = request_retries
        self.retry_sleep_seconds = retry_sleep_seconds

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt

        content = self._chat_once(messages, self.num_predict)
        if content or self.retry_empty_with_tokens is None:
            return content

        retry_messages = list(messages)
        retry_messages.append(
            {
                "role": "user",
                "content": "Your previous response was empty. Reply now with the requested final answer only. /no_think",
            }
        )
        return self._chat_once(retry_messages, self.retry_empty_with_tokens)

    def _chat_once(self, messages: list[dict[str, Any]], num_predict: int | None) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if num_predict is not None:
            payload["options"]["num_predict"] = num_predict
        if self.think is not None:
            payload["think"] = self.think

        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=data,
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
                        f"Could not call Ollama at {self.host}. Is Ollama running and is "
                        f"the model '{self.model}' pulled?"
                    ) from exc
            except (TimeoutError, socket.timeout) as exc:
                if attempt >= attempts:
                    raise RuntimeError(
                        f"Ollama call to model '{self.model}' timed out after {self.timeout} seconds. "
                        "Try increasing --ollama-timeout, or use a smaller/faster router model."
                    ) from exc
            except urllib.error.URLError as exc:
                if attempt >= attempts:
                    raise RuntimeError(
                        f"Could not call Ollama at {self.host}. Is Ollama running and is "
                        f"the model '{self.model}' pulled?"
                    ) from exc
            time.sleep(self.retry_sleep_seconds * attempt)
        message = body.get("message", {})
        content = message.get("content", "")
        return content.strip() if isinstance(content, str) else ""


class OpenAIChat:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.2,
        timeout: float = 120.0,
        max_tokens: int | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self.max_tokens = max_tokens
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required when using --reflection-provider openai.")

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens

        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI API request failed: HTTP {exc.code}: {detail}") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise RuntimeError(
                f"OpenAI reflection call to model '{self.model}' timed out after {self.timeout} seconds."
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError("Could not reach the OpenAI API for reflection.") from exc

        choices = body.get("choices", [])
        if not choices:
            return ""
        content = choices[0].get("message", {}).get("content", "")
        return content.strip() if isinstance(content, str) else ""


class RoutingAdapter(GEPAAdapter[RoutingExample, dict[str, Any], RoutingOutput]):
    def __init__(
        self,
        router_lm: OllamaChat,
        accuracy_weight: float = 0.85,
        cost_weight: float = 0.15,
        score_mode: str = "pareto_cost",
        router_output_format: str = "token",
        include_nano_answer: bool = False,
        nano_response_max_chars: int = 4000,
        sleep_seconds: float = 0.0,
    ) -> None:
        self.router_lm = router_lm
        self.accuracy_weight = accuracy_weight
        self.cost_weight = cost_weight
        self.score_mode = score_mode
        self.router_output_format = router_output_format
        self.include_nano_answer = include_nano_answer
        self.nano_response_max_chars = nano_response_max_chars
        self.sleep_seconds = sleep_seconds

    def evaluate(
        self,
        batch: list[RoutingExample],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[dict[str, Any], RoutingOutput]:
        router_prompt = candidate["router_prompt"]
        easy_router_code = candidate.get("easy_router_code", "").strip()
        easy_router_fn = None
        easy_router_compile_error = None
        if easy_router_code:
            easy_router_fn, easy_router_compile_error = compile_easy_router(easy_router_code)

        outputs: list[RoutingOutput] = []
        scores: list[float] = []
        trajectories: list[dict[str, Any]] | None = [] if capture_traces else None

        for example in batch:
            easy_router_source = "disabled"
            easy_router_error = easy_router_compile_error
            if easy_router_code and easy_router_compile_error:
                raw = f"easy_router_error: {easy_router_compile_error}"
                output = RoutingOutput(raw_response=raw, route="invalid")
                easy_router_source = "easy_router_error"
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
                        raw = f"easy_router:{easy_route}"
                        output = RoutingOutput(raw_response=raw, route=easy_route)
                        easy_router_source = "easy_router"
                        easy_router_error = None
                    elif easy_route in {"defer", "qwen", "llm", "none"}:
                        output = None
                        easy_router_source = "easy_router_defer"
                        easy_router_error = None
                    else:
                        output = None
                        easy_router_source = "easy_router_invalid"
                        easy_router_error = f"route_easy_case returned {decision!r}"
                except Exception as exc:
                    raw = f"easy_router_error: {type(exc).__name__}: {exc}"
                    output = RoutingOutput(raw_response=raw, route="invalid")
                    easy_router_source = "easy_router_error"
                    easy_router_error = f"{type(exc).__name__}: {exc}"
            else:
                output = None

            if output is None:
                messages = [
                    {"role": "system", "content": router_prompt},
                    {
                        "role": "user",
                        "content": format_question(
                            example,
                            self.router_output_format,
                            self.include_nano_answer,
                            self.nano_response_max_chars,
                        ),
                    },
                ]
                raw = self.router_lm(messages)
                output = parse_router_output(raw, self.router_output_format)
                if easy_router_source in {"disabled", "easy_router_defer"}:
                    easy_router_error = None
            route = output.route
            score, feedback = self.score_route(example, route)

            outputs.append(output)
            scores.append(score)

            if trajectories is not None:
                mini_correct = example.mini_pred == example.answer
                nano_correct = example.nano_pred == example.answer
                selected_correct = (
                    (route == "nano" and nano_correct)
                    or (route == "mini" and mini_correct)
                )
                if route == "nano":
                    cost_used = example.nano_cost_usd
                    oracle_route = "nano" if nano_correct else ("mini" if mini_correct else "nano")
                elif route == "mini":
                    cost_used = example.mini_cost_usd
                    oracle_route = "nano" if nano_correct else ("mini" if mini_correct else "nano")
                else:
                    cost_used = 0.0
                    oracle_route = "nano" if nano_correct else ("mini" if mini_correct else "nano")
                failure_type = classify_failure(route, nano_correct, mini_correct)
                trajectories.append(
                    {
                        "question_id": example.question_id,
                        "question": example.question,
                        "options": example.options,
                        "gold_answer": example.answer,
                        "mini_pred": example.mini_pred,
                        "nano_pred": example.nano_pred,
                        "nano_response": example.nano_response,
                        "mini_correct": mini_correct,
                        "nano_correct": nano_correct,
                        "mini_cost_usd": example.mini_cost_usd,
                        "nano_cost_usd": example.nano_cost_usd,
                        "mini_input_tokens": example.mini_input_tokens,
                        "mini_output_tokens": example.mini_output_tokens,
                        "nano_input_tokens": example.nano_input_tokens,
                        "nano_output_tokens": example.nano_output_tokens,
                        "route": route,
                        "raw_response": raw,
                        "easy_router_source": easy_router_source,
                        "easy_router_error": easy_router_error,
                        "router_output": {
                            "route": route,
                            "difficulty": output.difficulty,
                            "confidence": output.confidence,
                            "reason": output.reason,
                            "risk_flags": output.risk_flags or [],
                            "invalid_json": output.invalid_json,
                        },
                        "score": score,
                        "selected_correct": selected_correct,
                        "used_model": route,
                        "final_correct": selected_correct,
                        "cost_used": cost_used,
                        "oracle_route": oracle_route,
                        "failure_type": failure_type,
                        "feedback": feedback,
                    }
                )

            if self.sleep_seconds:
                time.sleep(self.sleep_seconds)

        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def score_route(self, example: RoutingExample, route: str) -> tuple[float, str]:
        mini_correct = example.mini_pred == example.answer
        nano_correct = example.nano_pred == example.answer

        if self.score_mode == "risk_averse_utility" and route not in {"nano", "mini"}:
            return (
                RISK_AVERSE_UTILITY_RULE["invalid_route"],
                f"Invalid route '{route}'. Output must be exactly nano or mini. "
                "Invalid routing is scored like a severe router failure.",
            )

        if route == "nano":
            selected_pred = example.nano_pred
            selected_correct = nano_correct
            selected_cost = example.nano_cost_usd
        elif route == "mini":
            selected_pred = example.mini_pred
            selected_correct = mini_correct
            selected_cost = example.mini_cost_usd
        else:
            return 0.0, f"Invalid route '{route}'. Output must be exactly nano or mini."

        if self.score_mode == "risk_averse_utility":
            failure_type = classify_failure(route, nano_correct, mini_correct)
            outcome = outcome_bucket(example)
            score = RISK_AVERSE_UTILITY_RULE[f"{outcome}_select_{route}"]
            if nano_correct:
                ideal_route = "nano"
            elif mini_correct:
                ideal_route = "mini"
            else:
                ideal_route = "mini"
        elif self.score_mode == "balanced_utility":
            if mini_correct and nano_correct:
                ideal_route = "nano"
                score = 1.0 if route == "nano" else 0.7
            elif mini_correct:
                ideal_route = "mini"
                score = 1.0 if route == "mini" else 0.0
            elif nano_correct:
                ideal_route = "nano"
                score = 1.0 if route == "nano" else 0.0
            else:
                ideal_route = "nano"
                score = 0.3 if route == "nano" else 0.0
        elif self.score_mode == "pareto_cost":
            score = (1.0 / selected_cost) if selected_correct and selected_cost > 0 else 0.0
            if mini_correct and nano_correct:
                ideal_route = "nano" if example.nano_cost_usd <= example.mini_cost_usd else "mini"
            elif nano_correct:
                ideal_route = "nano"
            elif mini_correct:
                ideal_route = "mini"
            else:
                ideal_route = "none"
        elif nano_correct:
            ideal_route = "nano"
            score = 1.0 if route == "nano" else (self.accuracy_weight if mini_correct else 0.0)
        elif mini_correct:
            ideal_route = "mini"
            score = 1.0 if route == "mini" else 0.0
        else:
            ideal_route = "nano"
            score = 1.0 if route == "nano" else self.accuracy_weight

        if selected_correct:
            feedback = (
                f"Route '{route}' was correct. Gold={example.answer}, selected_pred={selected_pred}. "
                f"mini_correct={mini_correct}, nano_correct={nano_correct}, ideal_route={ideal_route}. "
                f"mini_cost_usd={example.mini_cost_usd:.8f}, nano_cost_usd={example.nano_cost_usd:.8f}, "
                f"score_mode={self.score_mode}, score={score:.4f}. "
            )
            if route == "mini" and nano_correct:
                feedback += "However nano was also correct, so this spent extra cost."
            elif route == "nano":
                feedback += "Good cost-saving route."
            else:
                feedback += "Mini was needed because nano was not correct."
        else:
            feedback = (
                f"Route '{route}' was wrong. Gold={example.answer}, selected_pred={selected_pred}. "
                f"mini_correct={mini_correct}, nano_correct={nano_correct}, ideal_route={ideal_route}. "
                f"mini_cost_usd={example.mini_cost_usd:.8f}, nano_cost_usd={example.nano_cost_usd:.8f}, "
                f"score_mode={self.score_mode}, score={score:.4f}. "
            )
            if mini_correct and not nano_correct:
                if self.score_mode == "risk_averse_utility":
                    feedback += (
                        "This is a critical_misroute: nano was wrong while mini was correct, "
                        "so the objective applies the strongest penalty."
                    )
                else:
                    feedback += "This question needed mini."
            elif nano_correct and not mini_correct:
                feedback += "This question should have used nano."
            elif not mini_correct and not nano_correct:
                if self.score_mode == "risk_averse_utility":
                    feedback += (
                        "Neither model was correct; this objective now prefers mini in both-wrong "
                        "cases to avoid under-escalation bias."
                    )
                else:
                    feedback += "Neither model was correct; prefer nano because mini cannot recover accuracy."

        return score, feedback

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[dict[str, Any], RoutingOutput],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        allowed_components = {"router_prompt", "easy_router_code"}
        assert set(components_to_update).issubset(allowed_components)
        assert eval_batch.trajectories is not None

        traces = list(eval_batch.trajectories)
        total = len(traces) or 1
        final_accuracy = sum(1 for trace in traces if trace["final_correct"]) / total
        total_mini_cost = sum(trace["mini_cost_usd"] for trace in traces)
        cost_used = sum(trace["cost_used"] for trace in traces)
        cost_saving_rate = 1.0 - (cost_used / total_mini_cost) if total_mini_cost else 0.0
        critical_misroutes = [
            trace for trace in traces if trace["failure_type"] == "critical_misroute"
        ]
        unnecessary_mini = [
            trace for trace in traces if trace["failure_type"] == "unnecessary_expensive_route"
        ]
        invalid_json = [
            trace for trace in traces if trace.get("router_output", {}).get("invalid_json")
        ]

        def compact_example(trace: Mapping[str, Any]) -> dict[str, Any]:
            return {
                "question_id": trace["question_id"],
                "topic": trace.get("category", "computer science"),
                "question": trace["question"],
                "easy_router_source": trace.get("easy_router_source", "disabled"),
                "easy_router_error": trace.get("easy_router_error"),
                "router_output": trace["router_output"],
                "nano_saved_answer": {
                    "prediction": trace["nano_pred"],
                    "response": trace.get("nano_response", ""),
                },
                "actual": {
                    "nano_correct": trace["nano_correct"],
                    "mini_correct": trace["mini_correct"],
                    "oracle_route": trace["oracle_route"],
                    "failure_type": trace["failure_type"],
                },
            }

        batch_summary = {
            "final_accuracy": round(final_accuracy, 4),
            "cost_saving_rate_vs_all_mini": round(cost_saving_rate, 4),
            "critical_misroute_rate": round(len(critical_misroutes) / total, 4),
            "unnecessary_mini_rate": round(len(unnecessary_mini) / total, 4),
            "invalid_json_rate": round(len(invalid_json) / total, 4),
            "failure_type_counts": {
                name: sum(1 for trace in traces if trace["failure_type"] == name)
                for name in sorted({trace["failure_type"] for trace in traces})
            },
        }

        records = [
            {
                "Current router prompt": candidate["router_prompt"],
                "Current easy router Python code": candidate.get("easy_router_code", ""),
                "Batch result": batch_summary,
                "Critical misroute examples": [
                    compact_example(trace) for trace in critical_misroutes[:5]
                ],
                "Unnecessary expensive examples": [
                    compact_example(trace) for trace in unnecessary_mini[:5]
                ],
                "Reflection tasks": [
                    "What hidden failure patterns caused critical_misroute cases?",
                    "Which obvious easy cases can be safely handled by deterministic Python before calling Qwen?",
                    "Which cases must the Python function defer to Qwen because they are not truly easy?",
                    "Which routing rule should be added or strengthened to catch those cases?",
                    "Which routing rule should be relaxed to avoid unnecessary_expensive_route cases?",
                    "Produce a revised router policy and, if requested, revised Python easy-router code.",
                    "If JSON output is required, preserve the exact JSON schema.",
                ],
                "Output requirement": (
                    "For router_prompt, keep outputs parseable: route must be nano or mini; "
                    "for JSON mode include difficulty, confidence, reason, risk_flags. "
                    "For easy_router_code, output only valid Python defining route_easy_case("
                    "question, options, category='', source=''). The function must return 'nano', "
                    "'mini', or 'defer'. It should route only obvious easy cases directly and defer "
                    "uncertain cases to Qwen. Do not use imports, loops, classes, files, network, or side effects."
                ),
            }
        ]
        for trace in eval_batch.trajectories:
            options = "\n".join(
                f"{chr(ord('A') + i)}. {option}" for i, option in enumerate(trace["options"])
            )
            records.append(
                {
                    "Question": trace["question"],
                    "Options": options,
                    "Nano saved answer": {
                        "prediction": trace["nano_pred"],
                        "response": trace.get("nano_response", ""),
                    },
                    "Router output": trace["raw_response"],
                    "Easy router source": trace.get("easy_router_source", "disabled"),
                    "Easy router error": trace.get("easy_router_error"),
                    "Structured router output": trace["router_output"],
                    "Parsed route": trace["route"],
                    "Score": trace["score"],
                    "Failure type": trace["failure_type"],
                    "Feedback": trace["feedback"],
                    "Ground truth": {
                        "correct_answer": trace["gold_answer"],
                        "gpt-4.1-mini_prediction": trace["mini_pred"],
                        "gpt-4.1-nano_prediction": trace["nano_pred"],
                        "gpt-4.1-mini_correct": trace["mini_correct"],
                        "gpt-4.1-nano_correct": trace["nano_correct"],
                        "gpt-4.1-mini_cost_usd": trace["mini_cost_usd"],
                        "gpt-4.1-nano_cost_usd": trace["nano_cost_usd"],
                    },
                    "Current prompt": candidate["router_prompt"],
                    "Current easy router Python code": candidate.get("easy_router_code", ""),
                    "Prompt improvement goal": (
                        "Improve the hybrid router so deterministic Python handles only obvious easy "
                        "cases and Qwen handles the rest. Stay on the Pareto frontier: select the "
                        "cheapest model that is likely to answer correctly, and select mini only when "
                        "nano is likely to fail while mini is likely to succeed. The risk_averse_utility "
                        "objective strongly penalizes critical_misroute cases, where nano is selected "
                        "even though nano is wrong and mini is correct. The prompt must still force "
                        "exactly one token: nano or mini."
                    ),
                }
            )
        return {component: records for component in components_to_update}


def summarize(
    adapter: RoutingAdapter,
    examples: list[RoutingExample],
    candidate_or_prompt: dict[str, str] | str,
) -> dict[str, Any]:
    if isinstance(candidate_or_prompt, str):
        candidate = {"router_prompt": candidate_or_prompt}
    else:
        candidate = candidate_or_prompt
    batch = adapter.evaluate(examples, candidate, capture_traces=True)
    routes = [output.route for output in batch.outputs]
    correct = sum(1 for trace in (batch.trajectories or []) if trace["selected_correct"])
    traces = batch.trajectories or []
    failure_type_counts = {
        name: sum(1 for trace in traces if trace["failure_type"] == name)
        for name in sorted({trace["failure_type"] for trace in traces})
    }
    total = len(traces) or 1
    return {
        "mean_score": sum(batch.scores) / len(batch.scores),
        "accuracy": correct / len(batch.scores) if batch.scores else 0.0,
        "correct_routes": correct,
        "scores": batch.scores,
        "routes": routes,
        "nano_routes": routes.count("nano"),
        "mini_routes": routes.count("mini"),
        "invalid_routes": routes.count("invalid"),
        "failure_type_counts": failure_type_counts,
        "critical_misroute_rate": failure_type_counts.get("critical_misroute", 0) / total,
        "unnecessary_mini_rate": failure_type_counts.get("unnecessary_expensive_route", 0) / total,
        "invalid_json_rate": sum(
            1 for trace in traces if trace.get("router_output", {}).get("invalid_json")
        )
        / total,
        "traces": batch.trajectories,
    }


def select_candidate_index(
    result: Any,
    adapter: RoutingAdapter | None = None,
    valset: list[RoutingExample] | None = None,
) -> tuple[int, dict[int, dict[str, Any]]]:
    """Choose the best non-empty GEPA candidate with risk-aware tie-breaks."""
    scores = list(result.val_aggregate_scores)
    if not scores:
        raise ValueError("GEPA returned no validation scores.")
    valid_indices = [
        idx
        for idx, candidate in enumerate(result.candidates)
        if str(candidate.get("router_prompt", "")).strip()
    ]
    if not valid_indices:
        raise ValueError("GEPA returned no non-empty router prompts.")
    if adapter is None or valset is None:
        return max(valid_indices, key=lambda idx: (scores[idx], idx)), {}

    val_summaries: dict[int, dict[str, Any]] = {}
    for idx in valid_indices:
        val_summaries[idx] = summarize(adapter, valset, result.candidates[idx])

    def selection_key(idx: int) -> tuple[float, int, int, int, float, int]:
        summary = val_summaries[idx]
        failures = summary["failure_type_counts"]
        return (
            scores[idx],
            -failures.get("critical_misroute", 0),
            -summary["invalid_routes"],
            -failures.get("unnecessary_expensive_route", 0),
            summary["accuracy"],
            idx,
        )

    return max(valid_indices, key=selection_key), val_summaries


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--mini-file", default="computer science_result_mini.json")
    parser.add_argument("--nano-file", default="computer science_result_nano.json")
    parser.add_argument("--mini-cost-model", default="gpt-4.1-mini")
    parser.add_argument("--nano-cost-model", default="gpt-4.1-nano")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--ollama-model", default="qwen3.5")
    parser.add_argument("--router-model", default=None)
    parser.add_argument("--reflection-model", default=None)
    parser.add_argument("--reflection-provider", choices=["ollama", "openai"], default="ollama")
    parser.add_argument("--openai-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--openai-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--ollama-timeout", type=float, default=600.0)
    parser.add_argument("--openai-timeout", type=float, default=120.0)
    parser.add_argument("--router-num-predict", type=int, default=32)
    parser.add_argument(
        "--router-output-format",
        choices=["token", "json"],
        default="token",
        help="Use token for nano/mini only, or json for structured router traces.",
    )
    parser.add_argument(
        "--include-nano-answer",
        action="store_true",
        help="Include nano's saved prediction and response in the router input.",
    )
    parser.add_argument(
        "--hybrid-easy-router",
        action="store_true",
        help=(
            "Optimize a deterministic Python route_easy_case() program together with "
            "the Qwen router prompt. The Python program can route obvious easy cases "
            "directly and defer the rest to Qwen."
        ),
    )
    parser.add_argument(
        "--nano-response-max-chars",
        type=int,
        default=4000,
        help="Maximum characters of nano's saved response to include when --include-nano-answer is set; 0 disables truncation.",
    )
    parser.add_argument("--reflection-num-predict", type=int, default=2048)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Allow Ollama/Qwen thinking mode. By default the script disables it to avoid empty content.",
    )
    parser.add_argument("--max-metric-calls", type=int, default=30)
    parser.add_argument("--reflection-minibatch-size", type=int, default=4)
    parser.add_argument(
        "--sampling",
        choices=["balanced", "sequential", "random", "targeted"],
        default="balanced",
    )
    parser.add_argument(
        "--target-train-counts",
        default="mini_only=25,nano_only=10,both_correct=10,both_wrong=5",
        help="For --sampling targeted: comma-separated train bucket counts.",
    )
    parser.add_argument(
        "--target-val-counts",
        default="mini_only=10,nano_only=6,both_correct=17,both_wrong=17",
        help="For --sampling targeted: comma-separated validation bucket counts.",
    )
    parser.add_argument(
        "--split-file",
        default=None,
        help="JSON split manifest with train_ids and val_ids; overrides sampling for train/val.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--initial-prompt-file",
        default=None,
        help="Optional text file used as the GEPA seed router prompt.",
    )
    parser.add_argument("--accuracy-weight", type=float, default=0.85)
    parser.add_argument("--cost-weight", type=float, default=0.15)
    parser.add_argument(
        "--score-mode",
        choices=["pareto_cost", "balanced_utility", "risk_averse_utility", "legacy"],
        default="pareto_cost",
        help=(
            "pareto_cost scores a correct selected answer as 1 / selected_model_cost_usd "
            "and wrong routes as 0. balanced_utility uses a bounded routing utility table. "
            "risk_averse_utility heavily penalizes critical_misroute cases where nano is "
            "wrong but mini is correct. legacy uses the original bounded heuristic."
        ),
    )
    parser.add_argument("--output", default="gepa_router_prompt_result.json")
    parser.add_argument("--run-dir", default="gepa_router_run")
    parser.add_argument(
        "--optimizer",
        choices=["gepa", "adaevolve"],
        default="gepa",
        help="Use the original GEPA optimizer or the local AdaEvolve-style evolutionary optimizer.",
    )
    parser.add_argument(
        "--adaevolve-seed-result",
        action="append",
        default=[],
        help="Existing GEPA/AdaEvolve result JSON to seed AdaEvolve with. Can be repeated.",
    )
    parser.add_argument(
        "--adaevolve-eval-set",
        choices=["val", "train", "trainval"],
        default="val",
        help="Which split AdaEvolve uses as its evolutionary fitness evaluator.",
    )
    parser.add_argument(
        "--adaevolve-seed-candidates-per-result",
        type=int,
        default=1,
        help="How many candidates to import from each previous result; 1 means best only, 0 means all.",
    )
    parser.add_argument("--adaevolve-islands", type=int, default=4)
    parser.add_argument("--adaevolve-max-islands", type=int, default=8)
    parser.add_argument("--adaevolve-iterations", type=int, default=None)
    parser.add_argument("--adaevolve-ucb-c", type=float, default=0.8)
    parser.add_argument("--adaevolve-signal-decay", type=float, default=0.85)
    parser.add_argument("--adaevolve-bandit-decay", type=float, default=0.9)
    parser.add_argument("--adaevolve-min-explore-probability", type=float, default=0.15)
    parser.add_argument("--adaevolve-max-explore-probability", type=float, default=0.75)
    parser.add_argument("--adaevolve-signal-scale", type=float, default=6.0)
    parser.add_argument("--adaevolve-migration-interval", type=int, default=6)
    args = parser.parse_args()

    all_examples = load_examples(
        Path(args.mini_file),
        Path(args.nano_file),
        mini_cost_model=args.mini_cost_model,
        nano_cost_model=args.nano_cost_model,
    )
    target_train_counts = parse_bucket_counts(args.target_train_counts)
    target_val_counts = parse_bucket_counts(args.target_val_counts)
    if args.split_file:
        trainset, valset = split_train_val_from_file(all_examples, Path(args.split_file))
        examples = trainset + valset
    elif args.sampling == "targeted":
        trainset, valset = split_train_val_targeted(
            all_examples,
            target_train_counts,
            target_val_counts,
            args.seed,
        )
        examples = trainset + valset
    else:
        examples = select_examples(all_examples, args.limit, args.sampling, args.seed)
        trainset, valset = split_train_val(examples, args.train_ratio, args.seed)
    router_model = args.router_model or args.ollama_model
    reflection_model = args.reflection_model or (
        "gpt-4.1" if args.reflection_provider == "openai" else args.ollama_model
    )

    router_qwen = OllamaChat(
        model=router_model,
        host=args.ollama_host,
        timeout=args.ollama_timeout,
        num_predict=args.router_num_predict,
        think=True if args.enable_thinking else False,
        retry_empty_with_tokens=max(128, args.router_num_predict * 4),
    )
    if args.reflection_provider == "openai":
        reflection_lm = OpenAIChat(
            model=reflection_model,
            api_key=os.environ.get(args.openai_api_key_env),
            base_url=args.openai_base_url,
            timeout=args.openai_timeout,
            max_tokens=args.reflection_num_predict,
        )
    else:
        reflection_lm = OllamaChat(
            model=reflection_model,
            host=args.ollama_host,
            timeout=args.ollama_timeout,
            num_predict=args.reflection_num_predict,
            think=True if args.enable_thinking else False,
            retry_empty_with_tokens=max(512, args.reflection_num_predict),
        )
    adapter = RoutingAdapter(
        router_lm=router_qwen,
        accuracy_weight=args.accuracy_weight,
        cost_weight=args.cost_weight,
        score_mode=args.score_mode,
        router_output_format=args.router_output_format,
        include_nano_answer=args.include_nano_answer,
        nano_response_max_chars=args.nano_response_max_chars,
    )

    print(f"Loaded {len(examples)} examples: {len(trainset)} train, {len(valset)} validation")
    print(f"Sampling: {args.sampling}; selected outcome counts: {count_outcomes(examples)}")
    print(f"Train outcome counts: {count_outcomes(trainset)}")
    print(f"Validation outcome counts: {count_outcomes(valset)}")
    print(f"Using router model: {router_model} at {args.ollama_host}")
    print(f"Using reflection model: {reflection_model} via {args.reflection_provider}")
    print(f"Scoring mode: {args.score_mode}")
    print(f"Router output format: {args.router_output_format}")
    print(f"Include nano answer: {args.include_nano_answer}")
    print(f"Optimizer: {args.optimizer}")

    if args.initial_prompt_file:
        seed_prompt = Path(args.initial_prompt_file).read_text(encoding="utf-8").strip()
    elif args.include_nano_answer and args.router_output_format == "json":
        seed_prompt = DEFAULT_STRUCTURED_NANO_ANSWER_PROMPT
    elif args.include_nano_answer:
        seed_prompt = DEFAULT_NANO_ANSWER_PROMPT
    elif args.router_output_format == "json":
        seed_prompt = DEFAULT_STRUCTURED_PROMPT
    else:
        seed_prompt = DEFAULT_PROMPT
    seed_candidate = {"router_prompt": seed_prompt}
    if args.hybrid_easy_router:
        seed_candidate["easy_router_code"] = DEFAULT_EASY_ROUTER_CODE

    adaevolve_result = None
    result = None
    candidate_val_summaries: dict[int, dict[str, Any]] = {}
    selected_idx: int | None = None
    if args.optimizer == "adaevolve":
        seed_candidates = [seed_candidate] + load_seed_candidates_from_results(
            args.adaevolve_seed_result,
            args.hybrid_easy_router,
            args.adaevolve_seed_candidates_per_result,
        )
        adaevolve_result = run_adaevolve(
            seed_candidates=seed_candidates,
            trainset=trainset,
            valset=valset,
            adapter=adapter,
            reflection_lm=reflection_lm,
            summarize=summarize,
            compile_easy_router=compile_easy_router,
            max_metric_calls=args.max_metric_calls,
            run_dir=args.run_dir,
            seed=args.seed,
            hybrid_easy_router=args.hybrid_easy_router,
            router_output_format=args.router_output_format,
            include_nano_answer=args.include_nano_answer,
            eval_set_name=args.adaevolve_eval_set,
            num_islands=args.adaevolve_islands,
            max_islands=args.adaevolve_max_islands,
            ucb_c=args.adaevolve_ucb_c,
            signal_decay=args.adaevolve_signal_decay,
            bandit_decay=args.adaevolve_bandit_decay,
            min_explore_probability=args.adaevolve_min_explore_probability,
            max_explore_probability=args.adaevolve_max_explore_probability,
            signal_scale=args.adaevolve_signal_scale,
            migration_interval=args.adaevolve_migration_interval,
            iterations=args.adaevolve_iterations,
        )
        selected_idx = int(adaevolve_result["best_id"])
        selected_candidate = adaevolve_result["best_candidate"]
    else:
        result = gepa.optimize(
            seed_candidate=seed_candidate,
            trainset=trainset,
            valset=valset,
            adapter=adapter,
            reflection_lm=reflection_lm,
            max_metric_calls=args.max_metric_calls,
            reflection_minibatch_size=args.reflection_minibatch_size,
            run_dir=args.run_dir,
            display_progress_bar=True,
            seed=args.seed,
        )

        selected_idx, candidate_val_summaries = select_candidate_index(result, adapter, valset)
        selected_candidate = result.candidates[selected_idx]
    best_prompt = selected_candidate["router_prompt"]
    best_easy_router_code = selected_candidate.get("easy_router_code", "")
    initial_summary = summarize(adapter, examples, seed_candidate)
    best_summary = summarize(adapter, examples, selected_candidate)

    payload = {
        "ollama_model": args.ollama_model,
        "router_model": router_model,
        "reflection_model": reflection_model,
        "reflection_provider": args.reflection_provider,
        "optimizer": args.optimizer,
        "adaevolve": adaevolve_result,
        "mini_cost_model": args.mini_cost_model,
        "nano_cost_model": args.nano_cost_model,
        "router_output_format": args.router_output_format,
        "include_nano_answer": args.include_nano_answer,
        "nano_response_max_chars": args.nano_response_max_chars,
        "hybrid_easy_router": args.hybrid_easy_router,
        "limit": args.limit,
        "sampling": args.sampling,
        "split_file": args.split_file,
        "target_train_counts": target_train_counts if args.sampling == "targeted" else None,
        "target_val_counts": target_val_counts if args.sampling == "targeted" else None,
        "train_ratio": args.train_ratio,
        "initial_prompt_file": args.initial_prompt_file,
        "selected_outcome_counts": count_outcomes(examples),
        "train_outcome_counts": count_outcomes(trainset),
        "val_outcome_counts": count_outcomes(valset),
        "scoring": {
            "score_mode": args.score_mode,
            "router_output_format": args.router_output_format,
            "include_nano_answer": args.include_nano_answer,
            "pareto_cost_rule": (
                "score = 1.0 / selected_model_cost_usd when the selected model's saved "
                "answer is correct; otherwise score = 0.0"
            )
            if args.score_mode == "pareto_cost"
            else None,
            "balanced_utility_rule": {
                "both_correct_select_nano": 1.0,
                "both_correct_select_mini": 0.7,
                "mini_only_select_mini": 1.0,
                "mini_only_select_nano": 0.0,
                "nano_only_select_nano": 1.0,
                "nano_only_select_mini": 0.0,
                "both_wrong_select_nano": 0.3,
                "both_wrong_select_mini": 0.0,
            }
            if args.score_mode == "balanced_utility"
            else None,
            "risk_averse_utility_rule": RISK_AVERSE_UTILITY_RULE
            if args.score_mode == "risk_averse_utility"
            else None,
            "pricing_per_1m_tokens": PRICING_PER_1M,
            "legacy_accuracy_weight": args.accuracy_weight,
            "legacy_cost_weight": args.cost_weight,
        },
        "initial_prompt": seed_prompt,
        "initial_easy_router_code": seed_candidate.get("easy_router_code"),
        "best_prompt": best_prompt,
        "best_easy_router_code": best_easy_router_code,
        "selected_candidate_index": selected_idx,
        "gepa_default_best_index": result.best_idx if result is not None else None,
        "gepa_val_aggregate_scores": result.val_aggregate_scores if result is not None else None,
        "candidate_selection_rule": (
            "Maximize GEPA validation aggregate score, then prefer fewer critical_misroute "
            "cases, fewer invalid routes, fewer unnecessary expensive routes, higher "
            "validation accuracy, and finally newer candidate index."
        )
        if args.optimizer == "gepa"
        else (
            "Maximize AdaEvolve fitness on the configured eval set using adaptive island "
            "allocation, then evaluate the selected candidate on the full train+val examples."
        ),
        "candidate_val_summaries": candidate_val_summaries,
        "gepa_candidates": result.candidates if result is not None else None,
        "adaevolve_candidates": [
            record["candidate"] for record in (adaevolve_result or {}).get("records", [])
        ],
        "initial_summary": initial_summary,
        "best_summary": best_summary,
        "gepa_result_repr": repr(result) if result is not None else None,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nBest prompt:\n")
    print(best_prompt)
    if args.hybrid_easy_router:
        print("\nBest easy router code:\n")
        print(best_easy_router_code)
    print("\nInitial mean score:", initial_summary["mean_score"])
    print("Best mean score:", best_summary["mean_score"])
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
