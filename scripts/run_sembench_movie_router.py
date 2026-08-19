#!/usr/bin/env python3
"""Run a small, cached mini/nano routing experiment on SemBench Movie.

The first ``train_size`` reviews are treated as user-labeled examples.  The
following ``test_size`` reviews are held out.  Both models classify every
review, then lightweight pre-routing and nano-first cascade baselines learn
which rows should be escalated to mini.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline


LABEL_RE = re.compile(r"\b(POSITIVE|NEGATIVE)\b", re.IGNORECASE)
PROMPT = """Classify the sentiment expressed by this movie review.
Return exactly one label: POSITIVE or NEGATIVE.

Review:
{review}
"""


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reviews",
        type=Path,
        default=root / "sembench" / "files" / "movie" / "data" / "sf_2000" / "Reviews.csv",
    )
    parser.add_argument("--train-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--mini-model", default="gpt-5-mini")
    parser.add_argument("--nano-model", default="gpt-5-nano")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--cache",
        type=Path,
        default=root / "sembench_movie_router" / "model_outputs.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "sembench_movie_router" / "results.json",
    )
    return parser.parse_args()


def load_rows(path: Path, count: int) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for index, row in enumerate(reader):
            if index >= count:
                break
            review = (row.get("reviewText") or "").strip()
            gold = (row.get("scoreSentiment") or "").strip().upper()
            if review and gold in {"POSITIVE", "NEGATIVE"}:
                rows.append(
                    {
                        "row_index": str(index),
                        "id": row.get("id") or "",
                        "reviewId": row.get("reviewId") or str(index),
                        "reviewText": review,
                        "gold": gold,
                    }
                )
    if len(rows) < count:
        raise ValueError(f"Only found {len(rows)} usable rows in {path}; need {count}")
    return rows


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def call_model(client: OpenAI, model: str, review: str) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            response = client.responses.create(
                model=model,
                input=PROMPT.format(review=review),
                reasoning={"effort": "minimal"},
                text={"verbosity": "low"},
                max_output_tokens=64,
            )
            text = response.output_text.strip()
            match = LABEL_RE.search(text)
            if not match:
                raise ValueError(f"No sentiment label in response: {text!r}")
            usage = response.usage
            return {
                "pred": match.group(1).upper(),
                "response": text,
                "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
                "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
            }
        except Exception as exc:  # retries transient API and parse failures
            last_error = exc
            time.sleep(min(2**attempt, 12))
    raise RuntimeError(f"{model} failed after retries") from last_error


def collect_outputs(
    rows: list[dict[str, str]],
    models: list[str],
    cache_path: Path,
    workers: int,
) -> dict[str, dict[str, Any]]:
    cache = load_cache(cache_path)
    client = OpenAI()
    jobs = []
    for row in rows:
        for model in models:
            key = f"{row['reviewId']}::{model}"
            if key not in cache:
                jobs.append((key, model, row["reviewText"]))

    if jobs:
        print(f"Calling models for {len(jobs)} uncached row/model pairs...")
        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(call_model, client, model, review): key
                for key, model, review in jobs
            }
            for future in as_completed(futures):
                key = futures[future]
                cache[key] = future.result()
                completed += 1
                if completed % 10 == 0 or completed == len(jobs):
                    save_json_atomic(cache_path, cache)
                    print(f"  {completed}/{len(jobs)} complete")
    return cache


def attach_outputs(
    rows: list[dict[str, str]],
    cache: dict[str, dict[str, Any]],
    mini_model: str,
    nano_model: str,
) -> list[dict[str, Any]]:
    examples = []
    for row in rows:
        mini = cache[f"{row['reviewId']}::{mini_model}"]
        nano = cache[f"{row['reviewId']}::{nano_model}"]
        mini_correct = mini["pred"] == row["gold"]
        nano_correct = nano["pred"] == row["gold"]
        # Escalate only when mini repairs a nano error. Ties and both-wrong rows
        # prefer nano because mini adds cost without improving correctness.
        target_route = "mini" if mini_correct and not nano_correct else "nano"
        examples.append(
            {
                **row,
                "mini_pred": mini["pred"],
                "nano_pred": nano["pred"],
                "mini_correct": mini_correct,
                "nano_correct": nano_correct,
                "target_route": target_route,
                "outcome": (
                    "both_correct" if mini_correct and nano_correct else
                    "mini_only" if mini_correct else
                    "nano_only" if nano_correct else
                    "both_wrong"
                ),
            }
        )
    return examples


def fit_router(train: list[dict[str, Any]], nano_first: bool, seed: int) -> Pipeline:
    def text(example: dict[str, Any]) -> str:
        prefix = ""
        if nano_first:
            prefix = f"nano_prediction={example['nano_pred']} "
        return prefix + example["reviewText"]

    x_train = [text(example) for example in train]
    y_train = [1 if example["target_route"] == "mini" else 0 for example in train]
    if len(set(y_train)) < 2:
        raise ValueError("Training data contains only one routing class")
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=6000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=seed,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    return pipeline


def choose_threshold(
    model: Pipeline,
    train: list[dict[str, Any]],
    nano_first: bool,
) -> float:
    texts = [
        ((f"nano_prediction={x['nano_pred']} " if nano_first else "") + x["reviewText"])
        for x in train
    ]
    probabilities = model.predict_proba(texts)[:, 1]
    labels = [1 if x["target_route"] == "mini" else 0 for x in train]
    best = (float("-inf"), 0.5)
    for threshold in [i / 100 for i in range(5, 96, 5)]:
        predictions = [int(p >= threshold) for p in probabilities]
        # Missing a repairable nano failure is costlier than an unnecessary mini call.
        utility = sum(
            (2.0 if pred == 1 and gold == 1 else
             -2.0 if pred == 0 and gold == 1 else
             1.0 if pred == 0 and gold == 0 else
             0.2)
            for pred, gold in zip(predictions, labels)
        ) / len(labels)
        if utility > best[0]:
            best = (utility, threshold)
    return best[1]


def evaluate_routes(examples: list[dict[str, Any]], routes: list[str]) -> dict[str, Any]:
    gold = [x["gold"] for x in examples]
    selected = [x["mini_pred"] if route == "mini" else x["nano_pred"] for x, route in zip(examples, routes)]
    gold_positive = [label == "POSITIVE" for label in gold]
    selected_positive = [label == "POSITIVE" for label in selected]
    route_gold = [x["target_route"] == "mini" for x in examples]
    route_pred = [route == "mini" for route in routes]
    return {
        "accuracy": accuracy_score(gold, selected),
        "positive_f1": f1_score(gold_positive, selected_positive, zero_division=0),
        "positive_precision": precision_score(gold_positive, selected_positive, zero_division=0),
        "positive_recall": recall_score(gold_positive, selected_positive, zero_division=0),
        "router_accuracy": accuracy_score(route_gold, route_pred),
        "mini_calls": sum(route_pred),
        "mini_rate": sum(route_pred) / len(examples),
        "critical_misroutes": sum(gold_route and not pred for gold_route, pred in zip(route_gold, route_pred)),
        "unnecessary_mini": sum(not gold_route and pred for gold_route, pred in zip(route_gold, route_pred)),
    }


def model_routes(
    model: Pipeline,
    examples: list[dict[str, Any]],
    nano_first: bool,
    threshold: float,
) -> tuple[list[str], list[float]]:
    texts = [
        ((f"nano_prediction={x['nano_pred']} " if nano_first else "") + x["reviewText"])
        for x in examples
    ]
    probabilities = model.predict_proba(texts)[:, 1].tolist()
    return ["mini" if p >= threshold else "nano" for p in probabilities], probabilities


def main() -> None:
    args = parse_args()
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    random.seed(args.seed)
    count = args.train_size + args.test_size
    rows = load_rows(args.reviews, count)
    cache = collect_outputs(
        rows,
        [args.nano_model, args.mini_model],
        args.cache,
        args.workers,
    )
    examples = attach_outputs(rows, cache, args.mini_model, args.nano_model)
    train = examples[: args.train_size]
    test = examples[args.train_size :]

    results: dict[str, Any] = {
        "config": vars(args) | {"reviews": str(args.reviews), "cache": str(args.cache), "output": str(args.output)},
        "train_outcomes": dict(Counter(x["outcome"] for x in train)),
        "test_outcomes": dict(Counter(x["outcome"] for x in test)),
        "baselines": {
            "all_nano": evaluate_routes(test, ["nano"] * len(test)),
            "all_mini": evaluate_routes(test, ["mini"] * len(test)),
            "oracle": evaluate_routes(test, [x["target_route"] for x in test]),
        },
        "routers": {},
        "token_usage": {},
    }

    for nano_first, name in [(False, "tfidf_pre_router"), (True, "tfidf_nano_first")]:
        model = fit_router(train, nano_first, args.seed)
        threshold = choose_threshold(model, train, nano_first)
        routes, probabilities = model_routes(model, test, nano_first, threshold)
        results["routers"][name] = {
            "threshold": threshold,
            **evaluate_routes(test, routes),
            "traces": [
                {
                    "reviewId": x["reviewId"],
                    "gold": x["gold"],
                    "nano_pred": x["nano_pred"],
                    "mini_pred": x["mini_pred"],
                    "target_route": x["target_route"],
                    "route": route,
                    "p_mini": probability,
                }
                for x, route, probability in zip(test, routes, probabilities)
            ],
        }

    for model in [args.nano_model, args.mini_model]:
        values = [cache[f"{x['reviewId']}::{model}"] for x in rows]
        results["token_usage"][model] = {
            "calls": len(values),
            "input_tokens": sum(x["input_tokens"] for x in values),
            "output_tokens": sum(x["output_tokens"] for x in values),
        }

    save_json_atomic(args.output, results)
    print(json.dumps({k: v for k, v in results.items() if k != "routers"}, indent=2, default=str))
    print("Routers:")
    for name, value in results["routers"].items():
        print(name, {k: v for k, v in value.items() if k != "traces"})
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
