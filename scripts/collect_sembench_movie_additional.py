#!/usr/bin/env python3
"""Collect mini/nano outputs for a fixed sample of new unique Movie reviews."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from run_sembench_movie_router import call_model, load_cache, save_json_atomic


def load_unique_rows(path: Path) -> list[dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for index, raw in enumerate(csv.DictReader(handle)):
            review_id = (raw.get("reviewId") or "").strip()
            review = (raw.get("reviewText") or "").strip()
            gold = (raw.get("scoreSentiment") or "").strip().upper()
            if not review_id or not review or gold not in {"POSITIVE", "NEGATIVE"}:
                continue
            rows.setdefault(
                review_id,
                {
                    "row_index": str(index),
                    "reviewId": review_id,
                    "reviewText": review,
                    "gold": gold,
                },
            )
    return list(rows.values())


def get_or_create_manifest(
    path: Path,
    rows: list[dict[str, str]],
    cache: dict[str, dict[str, Any]],
    count: int,
    seed: int,
    mini_model: str,
    nano_model: str,
) -> dict[str, Any]:
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if len(manifest["review_ids"]) != count:
            raise ValueError(
                f"Existing manifest has {len(manifest['review_ids'])} IDs, expected {count}"
            )
        return manifest

    eligible = [
        row for row in rows
        if f"{row['reviewId']}::{mini_model}" not in cache
        and f"{row['reviewId']}::{nano_model}" not in cache
    ]
    if len(eligible) < count:
        raise ValueError(f"Only {len(eligible)} fully uncached unique reviews; need {count}")
    selected = random.Random(seed).sample(eligible, count)
    manifest = {
        "seed": seed,
        "count": count,
        "mini_model": mini_model,
        "nano_model": nano_model,
        "selection": "uniform random sample from unique reviews missing both model outputs",
        "review_ids": [row["reviewId"] for row in selected],
        "row_indices": [int(row["row_index"]) for row in selected],
    }
    save_json_atomic(path, manifest)
    return manifest


def collect(
    selected: list[dict[str, str]],
    cache_path: Path,
    mini_model: str,
    nano_model: str,
    workers: int,
) -> tuple[dict[str, dict[str, Any]], int, list[str]]:
    cache = load_cache(cache_path)
    jobs = []
    for row in selected:
        for model in (mini_model, nano_model):
            key = f"{row['reviewId']}::{model}"
            if key not in cache:
                jobs.append((key, model, row["reviewText"]))
    if not jobs:
        return cache, 0, []

    print(f"Calling {len(jobs)} uncached row/model pairs with {workers} workers...", flush=True)
    client = OpenAI()
    completed = 0
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(call_model, client, model, review): key
            for key, model, review in jobs
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                cache[key] = future.result()
            except Exception as exc:
                errors.append(f"{key}: {type(exc).__name__}: {exc}")
            completed += 1
            if completed % 10 == 0 or completed == len(jobs):
                save_json_atomic(cache_path, cache)
                print(
                    f"  attempted={completed}/{len(jobs)} "
                    f"saved={completed - len(errors)} errors={len(errors)}",
                    flush=True,
                )
    return cache, len(jobs), errors


def summarize(
    selected: list[dict[str, str]],
    cache: dict[str, dict[str, Any]],
    mini_model: str,
    nano_model: str,
) -> dict[str, Any]:
    outcomes: Counter[str] = Counter()
    complete = 0
    input_tokens = 0
    output_tokens = 0
    for row in selected:
        mini = cache.get(f"{row['reviewId']}::{mini_model}")
        nano = cache.get(f"{row['reviewId']}::{nano_model}")
        if mini is None or nano is None:
            continue
        complete += 1
        mini_correct = mini["pred"] == row["gold"]
        nano_correct = nano["pred"] == row["gold"]
        outcome = (
            "both_correct" if mini_correct and nano_correct else
            "mini_only" if mini_correct else
            "nano_only" if nano_correct else
            "both_wrong"
        )
        outcomes[outcome] += 1
        input_tokens += int(mini.get("input_tokens", 0)) + int(nano.get("input_tokens", 0))
        output_tokens += int(mini.get("output_tokens", 0)) + int(nano.get("output_tokens", 0))
    return {
        "requested_unique_reviews": len(selected),
        "complete_unique_reviews": complete,
        "missing_unique_reviews": len(selected) - complete,
        "outcome_counts": dict(outcomes),
        "critical_mini_only": outcomes["mini_only"],
        "total_input_tokens": input_tokens,
        "total_output_tokens": output_tokens,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reviews", type=Path,
        default=root / "sembench/files/movie/data/sf_2000/Reviews.csv",
    )
    parser.add_argument(
        "--cache", type=Path,
        default=root / "sembench_movie_router/model_outputs.json",
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=root / "sembench_movie_router/additional_400_manifest.json",
    )
    parser.add_argument(
        "--summary", type=Path,
        default=root / "sembench_movie_router/additional_400_summary.json",
    )
    parser.add_argument("--count", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--mini-model", default="gpt-5-mini")
    parser.add_argument("--nano-model", default="gpt-5-nano")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_dotenv(root / ".env")
    rows = load_unique_rows(args.reviews)
    by_id = {row["reviewId"]: row for row in rows}
    cache = load_cache(args.cache)
    manifest = get_or_create_manifest(
        args.manifest,
        rows,
        cache,
        args.count,
        args.seed,
        args.mini_model,
        args.nano_model,
    )
    selected = [by_id[review_id] for review_id in manifest["review_ids"]]
    outstanding = sum(
        f"{row['reviewId']}::{model}" not in cache
        for row in selected
        for model in (args.mini_model, args.nano_model)
    )
    print(
        json.dumps({
            "unique_source_rows": len(rows),
            "selected_new_reviews": len(selected),
            "outstanding_api_calls": outstanding,
            "manifest": str(args.manifest),
            "dry_run": args.dry_run,
        }, indent=2),
        flush=True,
    )
    if args.dry_run:
        return

    cache, attempted, errors = collect(
        selected, args.cache, args.mini_model, args.nano_model, args.workers
    )
    summary = {
        "config": {
            "reviews": str(args.reviews),
            "cache": str(args.cache),
            "manifest": str(args.manifest),
            "count": args.count,
            "seed": args.seed,
            "mini_model": args.mini_model,
            "nano_model": args.nano_model,
            "workers": args.workers,
        },
        "api_pairs_attempted_this_run": attempted,
        "errors": errors,
        **summarize(selected, cache, args.mini_model, args.nano_model),
    }
    save_json_atomic(args.summary, summary)
    print(json.dumps(summary, indent=2), flush=True)
    if errors or summary["missing_unique_reviews"]:
        raise RuntimeError(
            f"Collection incomplete: {len(errors)} errors, "
            f"{summary['missing_unique_reviews']} reviews missing a pair"
        )


if __name__ == "__main__":
    main()
