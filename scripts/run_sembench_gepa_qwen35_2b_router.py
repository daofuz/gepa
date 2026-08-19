#!/usr/bin/env python3
"""Optimize a local Qwen3.5-2B SemBench Movie router with GEPA.

The router sees only movie metadata and review text, never either answer
model's prediction. GEPA optimizes the routing instruction on the historical
100-label train split and selects a candidate on the historical 40-row
validation split. The historical 82-row test and the already-consumed
heldout-200 split are evaluation-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import gepa

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from optimize_ollama_router_gepa import (  # noqa: E402
    OllamaChat,
    RoutingAdapter,
    RoutingExample,
    select_candidate_index,
)
from run_sembench_accuracy_targeted_router import evaluate  # noqa: E402
from run_sembench_qa50_softprompt_router import deduplicate  # noqa: E402
from run_sembench_train100_ratio_sweep import make_orders, select_train  # noqa: E402
from train_sembench_balanced_direct_router import load_examples  # noqa: E402


TRAIN_COUNTS = {
    "both_correct": 50,
    "mini_only": 46,
    "nano_only": 3,
    "both_wrong": 1,
}

BASE_PROMPT = """You are a pre-routing model for an AI SQL semantic operator that
classifies movie-review sentiment. Decide whether the review should be sent to
the cheap nano model or the stronger mini model.

This is routing, not sentiment classification. Do not output POSITIVE or
NEGATIVE. Predict which answer model is needed:
- nano: use when the review is explicit, direct, and easy enough that nano is
  likely to classify it correctly.
- mini: use when nano is likely to fail but mini may succeed, including subtle
  implication, sarcasm, irony, mixed sentiment, contrast, negation, cultural
  references, compressed critic language, or ambiguous targets.
- If both models are likely correct, choose nano.
- If both models are likely wrong, choose mini as the conservative fallback.

Avoid shortcuts based on whether the review sounds positive or negative. Route
by linguistic difficulty and nano-failure risk, not by the sentiment label.

Respond with exactly one lowercase token: nano or mini."""


class CachedOllamaChat:
    """Persistent exact-request cache around the existing Ollama client."""

    def __init__(
        self,
        model: str,
        cache_path: Path,
        timeout: float,
        num_predict: int,
    ) -> None:
        self.model = model
        self.cache_path = cache_path
        self.client = OllamaChat(
            model=model,
            timeout=timeout,
            num_predict=num_predict,
            think=False,
            retry_empty_with_tokens=max(128, num_predict * 4),
        )
        self.cache: dict[str, str] = {}
        self.hits = 0
        self.misses = 0
        self.elapsed_seconds = 0.0
        if cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("model") == model:
                    self.cache[str(row["key"])] = str(row["response"])

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        canonical = json.dumps(prompt, ensure_ascii=False, sort_keys=True)
        key = hashlib.sha256(
            (self.model + "\n" + canonical).encode("utf-8")
        ).hexdigest()
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        started = time.perf_counter()
        response = self.client(prompt)
        elapsed = time.perf_counter() - started
        self.elapsed_seconds += elapsed
        self.misses += 1
        self.cache[key] = response
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "key": key,
                "model": self.model,
                "response": response,
                "elapsed_seconds": elapsed,
            }, ensure_ascii=True) + "\n")
        return response


def convert(example: dict[str, Any]) -> RoutingExample:
    return RoutingExample(
        question_id=int(example["review_id"]),
        question=example["text"],
        options=["POSITIVE", "NEGATIVE"],
        answer=example["gold"],
        mini_pred=example["mini_pred"],
        nano_pred=example["nano_pred"],
        nano_response="",
        mini_cost_usd=1.0,
        nano_cost_usd=0.25,
        mini_input_tokens=0,
        mini_output_tokens=0,
        nano_input_tokens=0,
        nano_output_tokens=0,
        category="movie-review sentiment",
        src=example["movie_id"],
    )


def demonstration_prompt(
    train: list[dict[str, Any]], base_prompt: str
) -> tuple[str, list[str]]:
    requested = {
        "both_correct": 2,
        "mini_only": 4,
        "nano_only": 2,
        "both_wrong": 1,
    }
    selected: list[dict[str, Any]] = []
    for bucket, count in requested.items():
        selected.extend(
            [row for row in train if row["bucket"] == bucket][:count]
        )
    lines = [
        "",
        "Historical routing demonstrations follow. They show routing outcomes,",
        "not sentiment labels. Infer general difficulty patterns rather than",
        "memorizing words, movie titles, or positive/negative polarity.",
    ]
    for index, row in enumerate(selected, start=1):
        route = "mini" if row["bucket"] in {"mini_only", "both_wrong"} else "nano"
        lines.extend([
            f"Example {index}:",
            f"Movie: {row['movie_id']}",
            f"Review: {row['text']}",
            f"Route: {route}",
        ])
    lines.append("For every new review, output exactly nano or mini.")
    return base_prompt + "\n" + "\n".join(lines), [
        row["review_id"] for row in selected
    ]


def candidate_metrics(
    adapter: RoutingAdapter,
    raw_examples: list[dict[str, Any]],
    converted: list[RoutingExample],
    candidate: dict[str, str],
) -> dict[str, Any]:
    batch = adapter.evaluate(converted, candidate, capture_traces=True)
    raw_routes = [output.route for output in batch.outputs]
    # Production-safe parsing fallback: malformed output escalates to mini.
    predictions = [0 if route == "nano" else 1 for route in raw_routes]
    metrics = evaluate(raw_examples, predictions)
    return {
        **metrics,
        "mini_saving_vs_all_mini": 1.0 - metrics["mini_rate"],
        "invalid_routes": raw_routes.count("invalid"),
        "raw_route_counts": dict(Counter(raw_routes)),
        "mean_gepa_score": (
            sum(batch.scores) / len(batch.scores) if batch.scores else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reviews",
        type=Path,
        default=ROOT / "sembench/files/movie/data/sf_2000/Reviews.csv",
    )
    parser.add_argument(
        "--model-cache",
        type=Path,
        default=ROOT / "sembench_movie_router/model_outputs.json",
    )
    parser.add_argument(
        "--heldout-manifest",
        type=Path,
        default=ROOT / "sembench_movie_router/untouched_200_manifest.json",
    )
    parser.add_argument("--seed", type=int, default=505)
    parser.add_argument("--router-model", default="qwen3.5:2b")
    parser.add_argument("--reflection-model", default="qwen3.5:4b")
    parser.add_argument("--max-metric-calls", type=int, default=20)
    parser.add_argument("--reflection-minibatch-size", type=int, default=4)
    parser.add_argument("--ollama-timeout", type=float, default=600.0)
    parser.add_argument(
        "--router-cache",
        type=Path,
        default=ROOT
        / "sembench_movie_router/gepa_qwen35_2b_router_cache.jsonl",
    )
    parser.add_argument(
        "--reflection-cache",
        type=Path,
        default=ROOT
        / "sembench_movie_router/gepa_qwen35_4b_reflection_cache.jsonl",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "sembench_movie_router/gepa_qwen35_2b_run",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "sembench_movie_router/gepa_qwen35_2b_router_result.json",
    )
    args = parser.parse_args()

    heldout_ids = set(json.loads(
        args.heldout_manifest.read_text(encoding="utf-8")
    )["review_ids"])
    all_rows = deduplicate(load_examples(args.reviews, args.model_cache))
    heldout_raw = [row for row in all_rows if row["review_id"] in heldout_ids]
    historical = [row for row in all_rows if row["review_id"] not in heldout_ids]
    if len(historical) != 812 or len(heldout_raw) != 200:
        raise AssertionError(
            f"Expected historical/heldout 812/200, got "
            f"{len(historical)}/{len(heldout_raw)}"
        )

    orders, validation_raw, historical_test_raw = make_orders(
        historical, args.seed
    )
    train_raw = select_train(orders, TRAIN_COUNTS, args.seed)
    train = [convert(row) for row in train_raw]
    validation = [convert(row) for row in validation_raw]
    historical_test = [convert(row) for row in historical_test_raw]
    heldout = [convert(row) for row in heldout_raw]

    seed_prompt, demonstration_ids = demonstration_prompt(train_raw, BASE_PROMPT)
    seed_candidate = {"router_prompt": seed_prompt}
    router_lm = CachedOllamaChat(
        args.router_model,
        args.router_cache,
        args.ollama_timeout,
        num_predict=16,
    )
    reflection_lm = CachedOllamaChat(
        args.reflection_model,
        args.reflection_cache,
        args.ollama_timeout,
        num_predict=1536,
    )
    adapter = RoutingAdapter(
        router_lm=router_lm,
        score_mode="risk_averse_utility",
        router_output_format="token",
        include_nano_answer=False,
    )

    print(
        f"train={len(train)} validation={len(validation)} "
        f"historical_test={len(historical_test)} heldout={len(heldout)}",
        flush=True,
    )
    print(
        f"router={args.router_model} reflection={args.reflection_model} "
        f"max_metric_calls={args.max_metric_calls}",
        flush=True,
    )
    print("Running GEPA optimization...", flush=True)
    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=train,
        valset=validation,
        adapter=adapter,
        reflection_lm=reflection_lm,
        max_metric_calls=args.max_metric_calls,
        reflection_minibatch_size=args.reflection_minibatch_size,
        run_dir=str(args.run_dir),
        display_progress_bar=True,
        seed=args.seed,
    )
    selected_index, candidate_val_summaries = select_candidate_index(
        result, adapter, validation
    )
    best_candidate = result.candidates[selected_index]
    print(
        f"GEPA candidates={len(result.candidates)} "
        f"selected={selected_index}",
        flush=True,
    )

    candidates = {
        "initial": seed_candidate,
        "gepa": best_candidate,
    }
    evaluations: dict[str, dict[str, Any]] = {}
    for name, candidate in candidates.items():
        print(f"Evaluating {name} candidate...", flush=True)
        evaluations[name] = {
            "validation": candidate_metrics(
                adapter, validation_raw, validation, candidate
            ),
            "historical_test": candidate_metrics(
                adapter, historical_test_raw, historical_test, candidate
            ),
            "heldout_200_posthoc": candidate_metrics(
                adapter, heldout_raw, heldout, candidate
            ),
        }
        print(json.dumps({
            "candidate": name,
            "historical_test": evaluations[name]["historical_test"],
            "heldout_200_posthoc": evaluations[name]["heldout_200_posthoc"],
        }, indent=2), flush=True)

    payload = {
        "protocol": {
            "router_input": "movie id and review text only",
            "answer_model_outputs_exposed": False,
            "train_counts": TRAIN_COUNTS,
            "validation_rows": len(validation),
            "historical_test_rows": len(historical_test),
            "heldout_rows": len(heldout),
            "heldout_status": (
                "post-hoc only; never used by GEPA or candidate selection, "
                "but previously consumed by other experiments"
            ),
            "invalid_output_fallback": "mini",
            "score_mode": "risk_averse_utility",
        },
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "demonstration_ids": demonstration_ids,
        "initial_prompt": seed_prompt,
        "best_prompt": best_candidate["router_prompt"],
        "selected_candidate_index": selected_index,
        "gepa_candidates": result.candidates,
        "gepa_val_aggregate_scores": list(result.val_aggregate_scores),
        "candidate_val_summaries": candidate_val_summaries,
        "evaluations": evaluations,
        "baselines": {
            "historical_test": {
                "all_nano": evaluate(
                    historical_test_raw, [0] * len(historical_test_raw)
                ),
                "all_mini": evaluate(
                    historical_test_raw, [1] * len(historical_test_raw)
                ),
            },
            "heldout_200": {
                "all_nano": evaluate(heldout_raw, [0] * len(heldout_raw)),
                "all_mini": evaluate(heldout_raw, [1] * len(heldout_raw)),
            },
        },
        "cache": {
            "router_hits": router_lm.hits,
            "router_misses": router_lm.misses,
            "router_uncached_elapsed_seconds": router_lm.elapsed_seconds,
            "reflection_hits": reflection_lm.hits,
            "reflection_misses": reflection_lm.misses,
            "reflection_uncached_elapsed_seconds": reflection_lm.elapsed_seconds,
        },
        "gepa_result_repr": repr(result),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "selected_candidate_index": selected_index,
        "gepa_val_aggregate_scores": payload["gepa_val_aggregate_scores"],
        "evaluations": evaluations,
        "baselines": payload["baselines"],
        "cache": payload["cache"],
    }, indent=2), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
