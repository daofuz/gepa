#!/usr/bin/env python3
"""Calibrate continuous scores from initial and GEPA-generated Qwen prompts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from run_computer_qa_qwen_local_router import parse_score
from run_sembench_accuracy_targeted_router import (
    evaluate,
    evaluate_scores,
    threshold_candidates,
)
from run_sembench_gepa_qwen35_2b_router import CachedOllamaChat
from run_sembench_qa50_softprompt_router import deduplicate
from run_sembench_train100_ratio_sweep import make_orders
from train_sembench_balanced_direct_router import load_examples


SCORE_OVERRIDE = """

CALIBRATION OUTPUT OVERRIDE:
For this experiment, do not make a hard route and ignore any earlier
instruction requiring a single nano/mini token. Estimate the risk that nano
will fail while mini is needed. Return compact JSON only:
{"mini_score": <number from 0 to 100>}

Use the full range. Low scores mean nano is safe; high scores mean mini is
needed. Do not predict POSITIVE or NEGATIVE and do not add explanations."""


def compact_question(row: dict[str, Any]) -> str:
    return (
        f"Movie: {row['movie_id']}\n"
        f"Review: {row['text']}\n"
        'Return only JSON: {"mini_score": <number>}'
    )


def score_rows(
    rows: list[dict[str, Any]],
    prompt: str,
    client: CachedOllamaChat,
) -> tuple[list[float], list[dict[str, Any]]]:
    scores = []
    details = []
    for index, row in enumerate(rows, start=1):
        raw = client([
            {"role": "system", "content": prompt + SCORE_OVERRIDE},
            {"role": "user", "content": compact_question(row)},
        ])
        score, fallback = parse_score(raw)
        scores.append(score)
        details.append({
            "review_id": row["review_id"],
            "score": score,
            "raw_response": raw,
            "parse_fallback": fallback,
        })
        if index % 20 == 0 or index == len(rows):
            print(f"scored={index}/{len(rows)}", flush=True)
    return scores, details


def calibrate(
    rows: list[dict[str, Any]], scores: list[float]
) -> dict[str, Any]:
    candidates = []
    for threshold in threshold_candidates(scores):
        predictions = [int(score >= threshold) for score in scores]
        metrics = evaluate(rows, predictions)
        candidates.append({"threshold": threshold, **metrics})
    selected = max(
        candidates,
        key=lambda item: (
            item["selected_accuracy"],
            item["mini_only_recall"],
            -item["mini_calls"],
            item["both_wrong_mini_rate"],
            item["threshold"],
        ),
    )
    return selected


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gepa-result",
        type=Path,
        default=root
        / "sembench_movie_router/gepa_qwen35_2b_router_result.json",
    )
    parser.add_argument(
        "--reviews",
        type=Path,
        default=root / "sembench/files/movie/data/sf_2000/Reviews.csv",
    )
    parser.add_argument(
        "--model-cache",
        type=Path,
        default=root / "sembench_movie_router/model_outputs.json",
    )
    parser.add_argument(
        "--heldout-manifest",
        type=Path,
        default=root / "sembench_movie_router/untouched_200_manifest.json",
    )
    parser.add_argument("--model", default="qwen3.5:2b")
    parser.add_argument("--seed", type=int, default=505)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--cache",
        type=Path,
        default=root
        / "sembench_movie_router/gepa_qwen35_2b_score_cache.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root
        / "sembench_movie_router/gepa_qwen35_2b_scored_result.json",
    )
    args = parser.parse_args()

    gepa_result = json.loads(args.gepa_result.read_text(encoding="utf-8"))
    prompts = {
        "initial": gepa_result["initial_prompt"],
        **{
            f"gepa_candidate_{index}": candidate["router_prompt"]
            for index, candidate in enumerate(gepa_result["gepa_candidates"])
            if index != 0
        },
    }
    heldout_ids = set(json.loads(
        args.heldout_manifest.read_text(encoding="utf-8")
    )["review_ids"])
    all_rows = deduplicate(load_examples(args.reviews, args.model_cache))
    historical = [row for row in all_rows if row["review_id"] not in heldout_ids]
    heldout = [row for row in all_rows if row["review_id"] in heldout_ids]
    _, validation, historical_test = make_orders(historical, args.seed)
    client = CachedOllamaChat(
        args.model, args.cache, args.timeout, num_predict=32
    )

    validation_results: dict[str, dict[str, Any]] = {}
    for name, prompt in prompts.items():
        print(f"Scoring validation with {name}...", flush=True)
        scores, details = score_rows(validation, prompt, client)
        calibration = calibrate(validation, scores)
        validation_results[name] = {
            "calibration": calibration,
            "scores": details,
        }
        print(json.dumps({
            "candidate": name,
            "calibration": calibration,
        }, indent=2), flush=True)

    selected_name = max(
        validation_results,
        key=lambda name: (
            validation_results[name]["calibration"]["selected_accuracy"],
            validation_results[name]["calibration"]["mini_only_recall"],
            -validation_results[name]["calibration"]["mini_calls"],
            validation_results[name]["calibration"]["both_wrong_mini_rate"],
            name,
        ),
    )
    selected_threshold = validation_results[selected_name]["calibration"][
        "threshold"
    ]
    selected_prompt = prompts[selected_name]
    print(
        f"Selected {selected_name} threshold={selected_threshold:.6f}",
        flush=True,
    )
    evaluations: dict[str, Any] = {}
    for split_name, rows in (
        ("historical_test", historical_test),
        ("heldout_200_posthoc", heldout),
    ):
        print(f"Scoring {split_name}...", flush=True)
        scores, details = score_rows(rows, selected_prompt, client)
        evaluations[split_name] = {
            "metrics": evaluate_scores(rows, scores, selected_threshold),
            "scores": details,
        }
        print(json.dumps({
            "split": split_name,
            "metrics": evaluations[split_name]["metrics"],
        }, indent=2), flush=True)

    result = {
        "protocol": {
            "gepa_prompt_candidates_selected_on": "historical validation only",
            "threshold_selected_on": "historical validation only",
            "selection_order": [
                "accuracy",
                "critical recall",
                "fewer mini calls",
                "both-wrong mini rate",
            ],
            "heldout_status": "post-hoc; never used for selection",
            "router_input": "movie id and review text only",
            "model_outputs_exposed": False,
        },
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "selected_candidate": selected_name,
        "threshold": selected_threshold,
        "validation_results": validation_results,
        "evaluations": evaluations,
        "baselines": {
            "historical_test": {
                "all_nano": evaluate(
                    historical_test, [0] * len(historical_test)
                ),
                "all_mini": evaluate(
                    historical_test, [1] * len(historical_test)
                ),
            },
            "heldout_200": {
                "all_nano": evaluate(heldout, [0] * len(heldout)),
                "all_mini": evaluate(heldout, [1] * len(heldout)),
            },
        },
        "cache": {
            "hits": client.hits,
            "misses": client.misses,
            "uncached_elapsed_seconds": client.elapsed_seconds,
        },
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "selected_candidate": selected_name,
        "threshold": selected_threshold,
        "validation": validation_results[selected_name]["calibration"],
        "evaluations": {
            name: value["metrics"] for name, value in evaluations.items()
        },
        "baselines": result["baselines"],
        "cache": result["cache"],
    }, indent=2), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
