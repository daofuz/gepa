from __future__ import annotations

import importlib.util
import inspect
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

from optimize_ollama_router_gepa import (
    DEFAULT_EASY_ROUTER_CODE,
    DEFAULT_PROMPT,
    OllamaChat,
    RoutingAdapter,
    compile_easy_router,
    count_outcomes,
    load_examples,
    split_train_val,
    split_train_val_from_file,
    summarize,
)


ROOT = Path(__file__).resolve().parent


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value not in {None, ""} else default


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value not in {None, ""} else default


def load_program(program_path: str) -> Any:
    path = Path(program_path).resolve()
    module_name = f"openevolve_router_candidate_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import candidate program: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def extract_candidate(module: Any) -> tuple[dict[str, str], str | None]:
    prompt = str(getattr(module, "ROUTER_PROMPT", "") or "").strip()
    if not prompt and hasattr(module, "get_router_prompt"):
        prompt = str(module.get_router_prompt() or "").strip()
    if not prompt:
        prompt = DEFAULT_PROMPT

    easy_router_code = ""
    if hasattr(module, "route_easy_case") and callable(module.route_easy_case):
        try:
            easy_router_code = inspect.getsource(module.route_easy_case).strip()
        except OSError:
            easy_router_code = ""
    if not easy_router_code and env_bool("OPENEVOLVE_ROUTER_FORCE_HYBRID", True):
        easy_router_code = DEFAULT_EASY_ROUTER_CODE

    candidate = {"router_prompt": prompt, "easy_router_code": easy_router_code}
    _, error = compile_easy_router(easy_router_code) if easy_router_code else (None, None)
    return candidate, error


def load_eval_examples() -> tuple[list[Any], list[Any], list[Any]]:
    mini_file = ROOT / os.environ.get("OPENEVOLVE_ROUTER_MINI_FILE", "computer science_result_mini.json")
    nano_file = ROOT / os.environ.get("OPENEVOLVE_ROUTER_NANO_FILE", "computer science_result_nano.json")
    examples = load_examples(mini_file, nano_file)

    split_file = os.environ.get("OPENEVOLVE_ROUTER_SPLIT_FILE")
    if split_file:
        trainset, valset = split_train_val_from_file(examples, ROOT / split_file)
        selected = trainset + valset
    else:
        rng = random.Random(env_int("OPENEVOLVE_ROUTER_SEED", 0))
        limit = env_int("OPENEVOLVE_ROUTER_LIMIT", 50)
        if 0 < limit < len(examples):
            selected = rng.sample(examples, limit)
        else:
            selected = examples
        trainset, valset = split_train_val(
            selected,
            env_float("OPENEVOLVE_ROUTER_TRAIN_RATIO", 0.8),
            env_int("OPENEVOLVE_ROUTER_SEED", 0),
        )

    eval_set = os.environ.get("OPENEVOLVE_ROUTER_EVAL_SET", "val").strip().lower()
    if eval_set == "train":
        eval_examples = trainset
    elif eval_set == "trainval":
        eval_examples = trainset + valset
    else:
        eval_examples = valset

    eval_limit = env_int("OPENEVOLVE_ROUTER_EVAL_LIMIT", 0)
    if eval_limit > 0 and eval_limit < len(eval_examples):
        eval_examples = eval_examples[:eval_limit]
    return selected, trainset, eval_examples


def make_adapter() -> RoutingAdapter:
    router = OllamaChat(
        model=os.environ.get("OPENEVOLVE_ROUTER_MODEL", "qwen3.5:latest"),
        host=os.environ.get("OPENEVOLVE_OLLAMA_HOST", "http://localhost:11434"),
        timeout=env_float("OPENEVOLVE_OLLAMA_TIMEOUT", 600.0),
        num_predict=env_int("OPENEVOLVE_ROUTER_NUM_PREDICT", 32),
        think=False,
        retry_empty_with_tokens=max(128, env_int("OPENEVOLVE_ROUTER_NUM_PREDICT", 32) * 4),
    )
    return RoutingAdapter(
        router_lm=router,
        score_mode=os.environ.get("OPENEVOLVE_ROUTER_SCORE_MODE", "risk_averse_utility"),
        router_output_format=os.environ.get("OPENEVOLVE_ROUTER_OUTPUT_FORMAT", "token"),
        include_nano_answer=env_bool("OPENEVOLVE_ROUTER_INCLUDE_NANO_ANSWER", False),
        nano_response_max_chars=env_int("OPENEVOLVE_ROUTER_NANO_RESPONSE_MAX_CHARS", 4000),
    )


def route_features(summary: dict[str, Any]) -> tuple[float, float, float]:
    total = max(1, len(summary.get("routes") or []))
    mini_route_rate = float(summary.get("mini_routes", 0)) / total
    critical_rate = float(summary.get("critical_misroute_rate", 0.0))
    unnecessary_rate = float(summary.get("unnecessary_mini_rate", 0.0))
    return mini_route_rate, critical_rate, unnecessary_rate


def score_components(summary: dict[str, Any]) -> dict[str, float]:
    total = max(1, len(summary.get("routes") or []))
    accuracy = float(summary.get("accuracy", 0.0))
    mean_score = float(summary.get("mean_score", 0.0))
    mini_rate = float(summary.get("mini_routes", 0)) / total
    nano_rate = float(summary.get("nano_routes", 0)) / total
    critical_rate = float(summary.get("critical_misroute_rate", 0.0))
    unnecessary_rate = float(summary.get("unnecessary_mini_rate", 0.0))
    invalid_rate = float(summary.get("invalid_routes", 0)) / total

    score_mode = os.environ.get("OPENEVOLVE_ROUTER_SCORE_MODE", "risk_averse_utility")
    if score_mode == "risk_averse_utility":
        # Diagnostic only. Fitness below uses raw GEPA mean_score exactly.
        normalized_mean_score = max(0.0, min(1.0, (mean_score + 2.0) / 4.0))
    else:
        normalized_mean_score = max(0.0, min(1.0, mean_score))

    return {
        "accuracy": accuracy,
        "mean_score": mean_score,
        "normalized_mean_score": normalized_mean_score,
        "mini_route_rate": mini_rate,
        "nano_route_rate": nano_rate,
        "critical_misroute_rate": critical_rate,
        "unnecessary_mini_rate": unnecessary_rate,
        "invalid_route_rate": invalid_rate,
    }


def score_for_accuracy(summary: dict[str, Any]) -> float:
    # Match GEPA's objective exactly: maximize the aggregate of RoutingAdapter scores.
    # For --score-mode risk_averse_utility, this is the average of the GEPA table:
    # mini_only+mini=2, mini_only+nano=-2, both_correct+nano=1, etc.
    return float(summary.get("mean_score", 0.0))

def maybe_record_best(program_path: str, candidate: dict[str, str], summary: dict[str, Any]) -> None:
    tracker = ROOT / os.environ.get("OPENEVOLVE_ROUTER_TRACKER", "openevolve_router_best.json")
    compact = {key: value for key, value in summary.items() if key != "traces"}
    compact["trace_count"] = len(summary.get("traces") or [])
    new_score = score_for_accuracy(summary)

    previous_score = -1.0
    if tracker.exists():
        try:
            previous = json.loads(tracker.read_text(encoding="utf-8"))
            previous_score = float(previous.get("combined_score", -1.0))
        except Exception:
            previous_score = -1.0

    if new_score >= previous_score:
        payload = {
            "program_path": str(Path(program_path).resolve()),
            "combined_score": new_score,
            "score_components": score_components(summary),
            "scoring_policy": {
                "objective": "gepa_mean_score",
                "score_mode": os.environ.get("OPENEVOLVE_ROUTER_SCORE_MODE", "risk_averse_utility"),
                "description": "OpenEvolve fitness is exactly summarize(...)[mean_score], the same aggregate RoutingAdapter score GEPA optimizes.",
            },
            "candidate": candidate,
            "summary": compact,
        }
        tracker.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate(program_path: str) -> dict[str, float | str]:
    try:
        module = load_program(program_path)
        candidate, easy_router_error = extract_candidate(module)
        if easy_router_error:
            return {
                "combined_score": 0.0,
                "accuracy": 0.0,
                "mean_score": -2.0,
                "normalized_mean_score": 0.0,
                "prompt_length": float(len(candidate["router_prompt"])),
                "mini_route_rate": 1.0,
                "nano_route_rate": 0.0,
                "critical_misroute_rate": 1.0,
                "unnecessary_mini_rate": 1.0,
                "invalid_routes": 1.0,
                "error": easy_router_error,
            }

        selected, trainset, eval_examples = load_eval_examples()
        adapter = make_adapter()
        summary = summarize(adapter, eval_examples, candidate)
        combined_score = score_for_accuracy(summary)
        mini_rate, critical_rate, unnecessary_rate = route_features(summary)
        maybe_record_best(program_path, candidate, summary)
        components = score_components(summary)
        return {
            "combined_score": combined_score,
            "accuracy": float(summary["accuracy"]),
            "mean_score": float(summary["mean_score"]),
            "normalized_mean_score": components["normalized_mean_score"],
            "prompt_length": float(len(candidate["router_prompt"])),
            "mini_route_rate": mini_rate,
            "nano_route_rate": components["nano_route_rate"],
            "critical_misroute_rate": critical_rate,
            "unnecessary_mini_rate": unnecessary_rate,
            "invalid_routes": float(summary.get("invalid_routes", 0)),
            "selected_examples": float(len(selected)),
            "train_examples": float(len(trainset)),
            "eval_examples": float(len(eval_examples)),
        }
    except Exception as exc:
        return {
            "combined_score": 0.0,
            "accuracy": 0.0,
            "mean_score": -2.0,
            "normalized_mean_score": 0.0,
            "prompt_length": 0.0,
            "mini_route_rate": 1.0,
            "nano_route_rate": 0.0,
            "critical_misroute_rate": 1.0,
            "unnecessary_mini_rate": 1.0,
            "invalid_routes": 1.0,
            "error": f"{type(exc).__name__}: {exc}",
        }


if __name__ == "__main__":
    result = evaluate(sys.argv[1] if len(sys.argv) > 1 else "openevolve_router_initial.py")
    print(json.dumps(result, ensure_ascii=False, indent=2))
