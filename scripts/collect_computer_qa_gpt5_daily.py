#!/usr/bin/env python3
"""Collect GPT-5 and GPT-5 nano labels under daily token budgets.

The canonical 410-row Computer Science QA prompts are copied from the existing
GPT-4.1 mini/nano annotations.  Collection is sequential, cached by question
ID, atomically checkpointed after every successful API response, and capped in
two independent complimentary-token groups using UTC calendar days.

The local budget cannot see unrelated usage from other processes or projects.
Defaults therefore use 90% of the Tier 1-2 complimentary limits discussed for
this project: 225k tokens/day for GPT-5 and 2.25M for GPT-5 nano.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tiktoken
from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError


MODEL_CONFIG = {
    "gpt5": {
        "model": "gpt-5-2025-08-07",
        "group": "gpt5",
        "default_budget": 225_000,
        "output": "computer science_result_gpt5.json",
    },
    "gpt5_nano": {
        "model": "gpt-5-nano-2025-08-07",
        "group": "gpt5_nano",
        "default_budget": 2_250_000,
        "output": "computer science_result_gpt5_nano.json",
    },
}

ANSWER_PATTERNS = (
    re.compile(r"(?i)\b(?:the\s+)?answer\s+is\s*\(?\s*([A-J])\s*\)?"),
    re.compile(r"(?i)\banswer\s*[:=-]\s*\(?\s*([A-J])\s*\)?"),
    re.compile(r"(?i)\bfinal(?:\s+answer)?\s*[:=-]\s*\(?\s*([A-J])\s*\)?"),
    re.compile(r"(?i)\b(?:correct|best)\s+(?:option|choice)\s*(?:is|:)\s*\(?\s*([A-J])\s*\)?"),
    re.compile(r"(?i)\boption\s+([A-J])\b"),
    re.compile(r"\(([A-J])\)"),
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def save_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def validate_sources(mini_path: Path, nano_path: Path) -> list[dict[str, Any]]:
    mini = load_json(mini_path, [])
    nano = load_json(nano_path, [])
    if not isinstance(mini, list) or not isinstance(nano, list) or not mini:
        raise ValueError("Expected non-empty JSON lists for both source annotations")
    if len(mini) != len(nano):
        raise ValueError(f"Source length mismatch: {len(mini)} vs {len(nano)}")
    required = ("question_id", "question", "options", "answer", "prompt")
    for index, (left, right) in enumerate(zip(mini, nano)):
        if any(key not in left or key not in right for key in required):
            raise ValueError(f"Source row {index} is missing required fields")
        for key in required:
            if left[key] != right[key]:
                raise ValueError(f"Source mismatch at row {index}, field {key}")
    ids = [str(row["question_id"]) for row in mini]
    if len(ids) != len(set(ids)):
        raise ValueError("Source question IDs are not unique")
    return mini


def source_fingerprint(rows: list[dict[str, Any]]) -> str:
    minimal = [
        {
            "question_id": row["question_id"],
            "question": row["question"],
            "options": row["options"],
            "answer": row["answer"],
            "prompt": row["prompt"],
        }
        for row in rows
    ]
    encoded = json.dumps(minimal, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_answer(text: str, options: list[Any]) -> str | None:
    allowed = {chr(ord("A") + index) for index in range(len(options))}
    for pattern in ANSWER_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            candidate = matches[-1].upper()
            if candidate in allowed:
                return candidate
    stripped = text.strip().upper().strip("().[]{}:;,- ")
    if stripped in allowed:
        return stripped
    leading = re.match(r"^\s*([A-J])\s*[.):;-]", text, re.IGNORECASE)
    if leading and leading.group(1).upper() in allowed:
        return leading.group(1).upper()
    normalized = " ".join(text.strip().casefold().split())
    normalized = re.sub(r"^(?:answer|choice|option)\s*:\s*", "", normalized)
    for index, option in enumerate(options):
        candidate = " ".join(str(option).strip().casefold().split())
        if candidate and (
            normalized == candidate
            or normalized.startswith(candidate + ".")
            or normalized.startswith(candidate + " ")
        ):
            return chr(ord("A") + index)
    return None


def estimate_request_tokens(
    prompt: list[dict[str, Any]], encoder: Any, max_output_tokens: int
) -> int:
    # o200k_base plus deliberately generous per-message framing overhead.
    input_estimate = 16
    for message in prompt:
        input_estimate += 16
        input_estimate += len(encoder.encode(str(message.get("role", ""))))
        content = message.get("content", "")
        if isinstance(content, str):
            input_estimate += len(encoder.encode(content))
        else:
            input_estimate += len(
                encoder.encode(json.dumps(content, ensure_ascii=False, sort_keys=True))
            )
    return input_estimate + max_output_tokens


def normalize_existing(
    path: Path, model: str, source_ids: set[str]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = load_json(path, [])
    if not isinstance(rows, list):
        raise ValueError(f"Existing output must be a JSON list: {path}")
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        question_id = str(row.get("question_id", ""))
        if question_id not in source_ids:
            raise ValueError(f"Unknown question ID in {path}: {question_id}")
        if row.get("model") != model:
            raise ValueError(
                f"Model mismatch in {path}: {row.get('model')!r} != {model!r}"
            )
        if question_id in by_id:
            raise ValueError(f"Duplicate question ID in {path}: {question_id}")
        by_id[question_id] = row
    return rows, by_id


def result_row(
    source: dict[str, Any],
    model: str,
    response: Any,
    estimated_tokens: int,
) -> dict[str, Any]:
    output_text = response.output_text.strip()
    usage = response.usage
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
    if total_tokens <= 0:
        total_tokens = input_tokens + output_tokens
    prediction = parse_answer(output_text, source["options"])
    copied = {
        key: value
        for key, value in source.items()
        if key not in {"response", "pred", "model", "input_tokens", "output_tokens"}
    }
    return {
        **copied,
        "model": model,
        "response_id": getattr(response, "id", None),
        "response": output_text,
        "pred": prediction,
        "parse_valid": prediction is not None,
        "correct": prediction == str(source["answer"]).upper() if prediction else False,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "estimated_request_tokens": estimated_tokens,
        "collected_at": utc_now().isoformat(),
    }


def daily_usage_from_outputs(
    paths: list[Path], model_to_group: dict[str, str]
) -> dict[str, dict[str, int]]:
    usage: dict[str, dict[str, int]] = {}
    for path in paths:
        for row in load_json(path, []):
            model = row.get("model")
            group = model_to_group.get(str(model))
            attempts = [*row.get("prior_attempts", []), row]
            for attempt in attempts:
                timestamp = attempt.get("collected_at")
                if not group or not timestamp:
                    continue
                try:
                    day = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00")).astimezone(
                        timezone.utc
                    ).date().isoformat()
                except ValueError:
                    continue
                usage.setdefault(day, {}).setdefault(group, 0)
                usage[day][group] += int(attempt.get("total_tokens", 0) or 0)
    return usage


def merge_ledger(
    stored: dict[str, Any], reconstructed: dict[str, dict[str, int]]
) -> dict[str, Any]:
    ledger = stored if isinstance(stored, dict) else {}
    ledger.setdefault("days", {})
    for day, groups in reconstructed.items():
        ledger["days"].setdefault(day, {})
        for group, tokens in groups.items():
            ledger["days"][day][group] = max(
                int(ledger["days"][day].get(group, 0)), int(tokens)
            )
    return ledger


def call_model(
    client: OpenAI,
    model: str,
    prompt: list[dict[str, Any]],
    max_output_tokens: int,
    retries: int,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return client.responses.create(
                model=model,
                input=prompt,
                reasoning={"effort": "minimal"},
                text={"verbosity": "low"},
                max_output_tokens=max_output_tokens,
                store=False,
            )
        except (APIConnectionError, RateLimitError) as exc:
            last_error = exc
        except APIStatusError as exc:
            if exc.status_code < 500:
                raise
            last_error = exc
        if attempt < retries:
            time.sleep(min(2**attempt + random.random(), 15.0))
    raise RuntimeError(f"{model} failed after {retries + 1} attempts") from last_error


def ordered_output(
    source: list[dict[str, Any]], by_id: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    return [by_id[str(row["question_id"])] for row in source if str(row["question_id"]) in by_id]


def reparse_cached(
    source: list[dict[str, Any]], by_id: dict[str, dict[str, Any]]
) -> int:
    changed = 0
    source_by_id = {str(row["question_id"]): row for row in source}
    for question_id, cached in by_id.items():
        parsed = parse_answer(
            str(cached.get("response", "")),
            source_by_id[question_id]["options"],
        )
        if parsed != cached.get("pred") or bool(parsed) != bool(cached.get("parse_valid")):
            cached["pred"] = parsed
            cached["parse_valid"] = parsed is not None
            cached["correct"] = (
                parsed == str(source_by_id[question_id]["answer"]).upper()
                if parsed
                else False
            )
            changed += 1
    return changed


def compact_attempt(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row.get(key)
        for key in (
            "response_id", "response", "pred", "parse_valid", "input_tokens",
            "output_tokens", "total_tokens", "estimated_request_tokens", "collected_at",
        )
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-mini", type=Path, default=root / "computer science_result_mini.json"
    )
    parser.add_argument(
        "--source-nano", type=Path, default=root / "computer science_result_nano.json"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=root
    )
    parser.add_argument(
        "--ledger", type=Path, default=root / "outputs/gpt5_daily_collection/usage_ledger.json"
    )
    parser.add_argument(
        "--summary", type=Path, default=root / "outputs/gpt5_daily_collection/status.json"
    )
    parser.add_argument(
        "--models", nargs="+", choices=tuple(MODEL_CONFIG), default=list(MODEL_CONFIG)
    )
    parser.add_argument("--daily-gpt5-budget", type=int, default=225_000)
    parser.add_argument("--daily-nano-budget", type=int, default=2_250_000)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--retry-output-tokens", type=int, default=2048)
    parser.add_argument("--max-calls-per-model", type=int)
    parser.add_argument("--retry-invalid", action="store_true")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.daily_gpt5_budget <= 0 or args.daily_nano_budget <= 0:
        parser.error("Daily budgets must be positive")
    if args.max_calls_per_model is not None and args.max_calls_per_model <= 0:
        parser.error("--max-calls-per-model must be positive")

    load_dotenv(root / ".env")
    source = validate_sources(args.source_mini, args.source_nano)
    source_ids = {str(row["question_id"]) for row in source}
    fingerprint = source_fingerprint(source)
    encoder = tiktoken.get_encoding("o200k_base")
    output_paths = {
        name: args.output_dir / str(MODEL_CONFIG[name]["output"])
        for name in MODEL_CONFIG
    }
    model_to_group = {
        str(config["model"]): str(config["group"])
        for config in MODEL_CONFIG.values()
    }
    ledger = merge_ledger(
        load_json(args.ledger, {}),
        daily_usage_from_outputs(list(output_paths.values()), model_to_group),
    )
    ledger["source_fingerprint"] = fingerprint
    budgets = {
        "gpt5": args.daily_gpt5_budget,
        "gpt5_nano": args.daily_nano_budget,
    }
    client = None
    if not args.dry_run:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not configured")
        client = OpenAI(timeout=args.request_timeout, max_retries=0)

    run_stats: dict[str, Any] = {}
    for name in args.models:
        config = MODEL_CONFIG[name]
        model = str(config["model"])
        group = str(config["group"])
        path = output_paths[name]
        _, by_id = normalize_existing(path, model, source_ids)
        if reparse_cached(source, by_id):
            save_json_atomic(path, ordered_output(source, by_id))
        initial_count = len(by_id)
        calls = 0
        budget_stopped = False
        estimates_pending = []
        for row in source:
            question_id = str(row["question_id"])
            prior = by_id.get(question_id)
            if prior is not None and (prior.get("parse_valid") or not args.retry_invalid):
                continue
            output_cap = (
                args.retry_output_tokens
                if prior is not None and not prior.get("parse_valid")
                else args.max_output_tokens
            )
            estimate = estimate_request_tokens(
                row["prompt"], encoder, output_cap
            )
            estimates_pending.append(estimate)
            if args.dry_run:
                continue
            if args.max_calls_per_model is not None and calls >= args.max_calls_per_model:
                break
            day = utc_now().date().isoformat()
            ledger["days"].setdefault(day, {})
            used = int(ledger["days"][day].get(group, 0))
            if used + estimate > budgets[group]:
                budget_stopped = True
                break
            assert client is not None
            response = call_model(
                client,
                model,
                row["prompt"],
                output_cap,
                args.retries,
            )
            collected = result_row(row, model, response, estimate)
            if prior is not None:
                collected["prior_attempts"] = [
                    *prior.get("prior_attempts", []),
                    compact_attempt(prior),
                ]
            actual = int(collected["total_tokens"])
            ledger["days"][day][group] = used + actual
            ledger["updated_at"] = utc_now().isoformat()
            save_json_atomic(args.ledger, ledger)
            by_id[question_id] = collected
            save_json_atomic(path, ordered_output(source, by_id))
            calls += 1
            print(
                f"{name} {len(by_id)}/{len(source)} qid={question_id} "
                f"tokens={actual} day_used={used + actual}/{budgets[group]} "
                f"pred={collected['pred']}",
                flush=True,
            )

        invalid = sum(not bool(row.get("parse_valid")) for row in by_id.values())
        correct = sum(bool(row.get("correct")) for row in by_id.values())
        valid_count = len(by_id) - invalid
        today = utc_now().date().isoformat()
        run_stats[name] = {
            "model": model,
            "output": str(path),
            "source_rows": len(source),
            "collected": len(by_id),
            "remaining": len(source) - len(by_id),
            "valid_predictions": len(by_id) - invalid,
            "invalid_predictions": invalid,
            "accuracy_on_collected": correct / len(by_id) if by_id else None,
            "accuracy_on_valid_predictions": correct / valid_count if valid_count else None,
            "calls_this_run": calls,
            "new_rows_this_run": len(by_id) - initial_count,
            "today_used_tokens_local": int(
                ledger.get("days", {}).get(today, {}).get(group, 0)
            ),
            "daily_budget": budgets[group],
            "budget_stopped": budget_stopped,
            "pending_estimated_tokens": sum(estimates_pending),
        }

    summary = {
        "updated_at": utc_now().isoformat(),
        "status": "dry_run" if args.dry_run else "collected",
        "source": {
            "rows": len(source),
            "fingerprint_sha256": fingerprint,
            "mini_annotations": str(args.source_mini),
            "nano_annotations": str(args.source_nano),
        },
        "budget_policy": {
            "timezone": "UTC",
            "gpt5_daily_local_cap": args.daily_gpt5_budget,
            "gpt5_nano_daily_local_cap": args.daily_nano_budget,
            "safety_note": (
                "Local caps include only calls recorded by this collector; unrelated "
                "organization/project usage is not visible."
            ),
        },
        "models": run_stats,
    }
    save_json_atomic(args.summary, summary)
    if not args.dry_run:
        save_json_atomic(args.ledger, ledger)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
