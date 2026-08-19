#!/usr/bin/env python3
"""Matched random-vs-observable active acquisition baseline at 50 labels."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import run_sembench_random100_prior_methods as experiment
from run_sembench_qa50_softprompt_router import deduplicate
from run_sembench_train100_ratio_sweep import make_orders
from train_sembench_balanced_direct_router import evaluate, load_examples


def pool_from_orders(orders: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [row for rows in orders.values() for row in rows]


def select_random(orders: dict[str, list[dict[str, Any]]], seed: int) -> list[dict[str, Any]]:
    return random.Random(seed + 49979687).sample(pool_from_orders(orders), 50)


def select_active(orders: dict[str, list[dict[str, Any]]], seed: int) -> list[dict[str, Any]]:
    pool = pool_from_orders(orders)
    disagreement = [row for row in pool if row["nano_pred"] != row["mini_pred"]]
    agreement = [row for row in pool if row["nano_pred"] == row["mini_pred"]]
    rng = random.Random(seed + 67867967)
    selected = rng.sample(disagreement, 25) + rng.sample(agreement, 25)
    rng.shuffle(selected)
    return selected


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ("selected_accuracy", "mini_saving_vs_all_mini", "mini_only_recall")
    output = {}
    for field in fields:
        values = [
            float(run["policies"]["validation_max_accuracy"]["test"][field])
            for run in runs
        ]
        output[field] = {"mean": mean(values), "std": pstdev(values), "values": values}
    return output


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", type=Path, default=root / "sembench/files/movie/data/sf_2000/Reviews.csv")
    parser.add_argument("--cache", type=Path, default=root / "sembench_movie_router/model_outputs.json")
    parser.add_argument("--manifest", type=Path, default=root / "sembench_movie_router/untouched_200_manifest.json")
    parser.add_argument("--hf-model", default="distilbert-base-uncased")
    parser.add_argument("--prompt-tokens", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[505, 606, 707])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--prior-accuracy-nano", type=float, default=0.80)
    parser.add_argument("--prior-accuracy-mini", type=float, default=0.90)
    parser.add_argument("--log-odds-margin", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=root / "outputs/sembench_active50_acquisition_baseline.json")
    args = parser.parse_args()

    ids = set(json.loads(args.manifest.read_text(encoding="utf-8"))["review_ids"])
    cache = json.loads(args.cache.read_text(encoding="utf-8"))
    examples = deduplicate(load_examples(args.reviews, args.cache))
    historical = [row for row in examples if row["review_id"] not in ids]
    heldout = [row for row in examples if row["review_id"] in ids]
    runs: dict[str, list[dict[str, Any]]] = {"random50": [], "active_25disagree_25agree": []}
    selectors = {"random50": select_random, "active_25disagree_25agree": select_active}
    for name, selector in selectors.items():
        for seed in args.seeds:
            orders, validation, _ = make_orders(historical, seed)
            train = selector(orders, seed)
            print(f"selection={name} seed={seed} counts={dict(Counter(r['bucket'] for r in train))}", flush=True)
            run = experiment.run_one(train, validation, heldout, cache, seed, "none", 0.0, args)
            runs[name].append(run)
            metrics = run["policies"]["validation_max_accuracy"]["test"]
            print(f"  accuracy={metrics['selected_accuracy']:.4f} saving={metrics['mini_saving_vs_all_mini']:.4f} MO_recall={metrics['mini_only_recall']:.4f}", flush=True)

    output = {
        "protocol": {
            "training_size": 50,
            "random": "uniform sample before labeling",
            "active": "25 observable Nano/Mini disagreements + 25 agreements before labeling",
            "gold_used_by_acquisition": False,
            "loss": "inverse-expected-query-cost weighted BCE; no prior",
            "test_status": "post-hoc held-out 200; not fresh",
        },
        "baselines": {
            "all_nano": evaluate(heldout, [0] * len(heldout)),
            "all_mini": evaluate(heldout, [1] * len(heldout)),
        },
        "summary": {name: aggregate(values) for name, values in runs.items()},
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
