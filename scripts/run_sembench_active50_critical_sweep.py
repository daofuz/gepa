#!/usr/bin/env python3
"""Step-1 composition sweep for a 50-label active-training set.

For each seed, increasing ``mini_only`` by one replaces exactly one
``both_correct`` row.  Two ``nano_only`` and one ``both_wrong`` control rows
remain fixed, so every training set has 50 examples.  Rows are nested prefixes
of fixed per-bucket orders; validation and the post-hoc held-out set are fixed.

This is a diagnostic *after annotation*: the observable acquisition signal is
Nano/Mini disagreement, while the exact MO/NO outcome is known only after gold
labels are obtained.  No answer-model API calls are made by this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import run_sembench_random100_prior_methods as experiment
from run_sembench_qa50_softprompt_router import deduplicate
from run_sembench_train100_ratio_sweep import make_orders
from train_sembench_balanced_direct_router import evaluate, load_examples


def select_nested_train(
    orders: dict[str, list[dict[str, Any]]], mini_only: int, seed: int
) -> list[dict[str, Any]]:
    counts = {
        "mini_only": mini_only,
        "nano_only": 2,
        "both_wrong": 1,
        "both_correct": 47 - mini_only,
    }
    if min(counts.values()) < 1 or sum(counts.values()) != 50:
        raise ValueError(f"Invalid 50-label mixture: {counts}")
    selected: list[dict[str, Any]] = []
    for bucket, count in counts.items():
        if len(orders[bucket]) < count:
            raise ValueError(f"Need {count} {bucket}, found {len(orders[bucket])}")
        selected.extend(orders[bucket][:count])
    random.Random(seed + 15485863).shuffle(selected)
    if len(selected) != 50 or len({row["review_id"] for row in selected}) != 50:
        raise AssertionError("Training set must contain 50 unique rows")
    return selected


def aggregate(runs: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    fields = (
        "selected_accuracy",
        "mini_rate",
        "mini_saving_vs_all_mini",
        "mini_only_recall",
        "both_wrong_mini_rate",
        "critical_misroutes",
        "unnecessary_mini",
    )
    output: dict[str, Any] = {}
    for field in fields:
        values = [float(run["policies"][policy]["test"][field]) for run in runs]
        output[field] = {"mean": mean(values), "std": pstdev(values), "values": values}
    return output


def write_csv(path: Path, summary: dict[str, Any], policy: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "mini_only", "both_correct", "nano_only", "both_wrong",
            "accuracy_mean", "accuracy_std", "mini_saving_mean",
            "mini_saving_std", "mo_recall_mean", "mo_recall_std",
        ])
        for key in sorted(summary, key=int):
            metrics = summary[key][policy]
            writer.writerow([
                int(key), 47 - int(key), 2, 1,
                metrics["selected_accuracy"]["mean"],
                metrics["selected_accuracy"]["std"],
                metrics["mini_saving_vs_all_mini"]["mean"],
                metrics["mini_saving_vs_all_mini"]["std"],
                metrics["mini_only_recall"]["mean"],
                metrics["mini_only_recall"]["std"],
            ])


def write_plot(path: Path, summary: dict[str, Any], policy: str) -> None:
    xs = sorted(int(key) for key in summary)
    accuracy = [summary[str(x)][policy]["selected_accuracy"]["mean"] * 100 for x in xs]
    accuracy_std = [summary[str(x)][policy]["selected_accuracy"]["std"] * 100 for x in xs]
    saving = [summary[str(x)][policy]["mini_saving_vs_all_mini"]["mean"] * 100 for x in xs]
    recall = [summary[str(x)][policy]["mini_only_recall"]["mean"] * 100 for x in xs]

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.6), sharex=True)
    axes[0].plot(xs, accuracy, color="#1f5a94", linewidth=2, label="Test accuracy")
    axes[0].fill_between(
        xs,
        [v - s for v, s in zip(accuracy, accuracy_std)],
        [v + s for v, s in zip(accuracy, accuracy_std)],
        color="#1f5a94", alpha=0.16, linewidth=0,
    )
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(xs, recall, color="#bd4b35", linewidth=2, label="MO recall")
    axes[1].plot(xs, saving, color="#428f5b", linewidth=2, label="Mini saving")
    axes[1].set_xlabel("Mini-only examples in 50-label training set")
    axes[1].set_ylabel("Rate (%)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, ncol=2)
    fig.suptitle("Critical-case composition sweep (BC = 47 - MO, NO = 2, BW = 1)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", type=Path, default=root / "sembench/files/movie/data/sf_2000/Reviews.csv")
    parser.add_argument("--cache", type=Path, default=root / "sembench_movie_router/model_outputs.json")
    parser.add_argument("--manifest", type=Path, default=root / "sembench_movie_router/untouched_200_manifest.json")
    parser.add_argument("--hf-model", default="distilbert-base-uncased")
    parser.add_argument("--prompt-tokens", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[505, 606, 707])
    parser.add_argument("--critical-min", type=int, default=1)
    parser.add_argument("--critical-max", type=int, default=46)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--prior-accuracy-nano", type=float, default=0.80)
    parser.add_argument("--prior-accuracy-mini", type=float, default=0.90)
    parser.add_argument("--log-odds-margin", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=root / "outputs/sembench_active50_critical_sweep.json")
    parser.add_argument("--csv-output", type=Path, default=root / "outputs/sembench_active50_critical_sweep.csv")
    parser.add_argument("--plot-output", type=Path, default=root / "outputs/sembench_active50_critical_sweep.png")
    args = parser.parse_args()
    if not (1 <= args.critical_min <= args.critical_max <= 46):
        raise ValueError("Require 1 <= critical-min <= critical-max <= 46")

    heldout_ids = set(json.loads(args.manifest.read_text(encoding="utf-8"))["review_ids"])
    cache = json.loads(args.cache.read_text(encoding="utf-8"))
    examples = deduplicate(load_examples(args.reviews, args.cache))
    heldout = [row for row in examples if row["review_id"] in heldout_ids]
    historical = [row for row in examples if row["review_id"] not in heldout_ids]
    if len(heldout) != 200 or len(historical) != 812:
        raise AssertionError("Expected 812 historical and 200 held-out rows")

    per_seed = {seed: make_orders(historical, seed) for seed in args.seeds}
    completed: dict[str, list[dict[str, Any]]] = {}
    if args.output.exists():
        old = json.loads(args.output.read_text(encoding="utf-8"))
        completed = old.get("runs", {})

    policies = ("direct_uplift", "validation_max_accuracy", "validation_accuracy_floor")
    for critical in range(args.critical_min, args.critical_max + 1):
        key = str(critical)
        old_by_seed = {int(run["seed"]): run for run in completed.get(key, [])}
        runs: list[dict[str, Any]] = []
        for seed in args.seeds:
            if seed in old_by_seed:
                runs.append(old_by_seed[seed])
                continue
            orders, validation, _ = per_seed[seed]
            train = select_nested_train(orders, critical, seed)
            if {r["review_id"] for r in train} & {r["review_id"] for r in validation}:
                raise AssertionError("Train/validation leakage")
            print(f"critical={critical:02d} seed={seed} counts={dict(Counter(r['bucket'] for r in train))}", flush=True)
            run = experiment.run_one(train, validation, heldout, cache, seed, "none", 0.0, args)
            runs.append(run)
            metrics = run["policies"]["validation_max_accuracy"]["test"]
            print(
                f"  accuracy={metrics['selected_accuracy']:.4f} "
                f"saving={metrics['mini_saving_vs_all_mini']:.4f} "
                f"MO_recall={metrics['mini_only_recall']:.4f}",
                flush=True,
            )
            completed[key] = runs
            checkpoint = {
                "protocol": {
                    "status": "post-hoc held-out composition diagnostic; not a fresh final test",
                    "training_size": 50,
                    "composition": "MO=k, BC=47-k, NO=2, BW=1",
                    "paired_step": "each +1 MO replaces exactly one BC within fixed bucket orders",
                    "selection_note": "exact outcome control is performed after run-both disagreement acquisition and gold annotation",
                    "model": "frozen DistilBERT, 8 soft tokens, two correctness heads",
                    "loss": "inverse-expected-query-cost weighted BCE; no prior loss",
                    "answer_model_api_calls": 0,
                },
                "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "heldout_counts": dict(Counter(row["bucket"] for row in heldout)),
                "baselines": {
                    "all_nano": evaluate(heldout, [0] * len(heldout)),
                    "all_mini": evaluate(heldout, [1] * len(heldout)),
                },
                "runs": completed,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
        completed[key] = sorted(runs, key=lambda row: int(row["seed"]))

    summary = {
        key: {policy: aggregate(runs, policy) for policy in policies}
        for key, runs in completed.items()
        if len({int(run["seed"]) for run in runs}) == len(args.seeds)
    }
    result = json.loads(args.output.read_text(encoding="utf-8"))
    result["runs"] = completed
    result["summary"] = summary
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_csv(args.csv_output, summary, "validation_max_accuracy")
    print(f"Saved {args.output} and {args.csv_output}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; completed runs remain checkpointed.", file=sys.stderr)
        raise
