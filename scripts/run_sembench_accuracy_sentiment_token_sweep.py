#!/usr/bin/env python3
"""Sweep soft-prompt length for the best accuracy-targeted nano-first cascade."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from run_sembench_accuracy_targeted_router import (
    aggregate_method,
    make_split,
    train_sentiment_softprompt,
)
from run_sembench_qa50_softprompt_router import deduplicate
from train_sembench_balanced_direct_router import evaluate, load_examples


def compact(runs: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    key = "sentiment_ce/nano_disagreement"
    summary = aggregate_method(runs, key, policy)
    return {
        "accuracy": summary["selected_accuracy"],
        "gap_vs_all_mini": summary["accuracy_gap_vs_all_mini"],
        "mini_rate": summary["mini_rate"],
        "mini_saving": summary["mini_saving_vs_all_mini"],
        "critical_recall": summary["mini_only_recall"],
        "both_wrong_mini_rate": summary["both_wrong_mini_rate"],
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
    parser.add_argument("--hf-model", default="distilbert-base-uncased")
    parser.add_argument("--prompt-tokens", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--seeds", type=int, nargs="+", default=[505, 606, 707])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--output", type=Path,
        default=root / "sembench_movie_router/accuracy_sentiment_token_sweep.json",
    )
    args = parser.parse_args()

    examples = deduplicate(load_examples(args.reviews, args.cache))
    splits = {seed: make_split(examples, seed) for seed in args.seeds}
    by_tokens: dict[int, list[dict[str, Any]]] = {}
    for prompt_tokens in args.prompt_tokens:
        local_args = copy.copy(args)
        local_args.prompt_tokens = prompt_tokens
        by_tokens[prompt_tokens] = []
        for seed in args.seeds:
            train, validation, test, _ = splits[seed]
            print(f"tokens={prompt_tokens} seed={seed}: sentiment CE cascade", flush=True)
            run = train_sentiment_softprompt(
                train, validation, test, seed, "sentiment_ce", local_args
            )
            by_tokens[prompt_tokens].append(run)
            metrics = run["routes"]["nano_disagreement"]["max_accuracy"]["test"]
            print(
                f"  accuracy={metrics['selected_accuracy']:.4f} "
                f"mini_rate={metrics['mini_rate']:.4f} "
                f"critical_recall={metrics['mini_only_recall']:.4f}",
                flush=True,
            )

    summaries = {
        str(tokens): {
            policy: compact(runs, policy)
            for policy in ("match_all_mini", "max_accuracy")
        }
        for tokens, runs in by_tokens.items()
    }
    ranking = sorted(
        [
            {
                "prompt_tokens": tokens,
                "accuracy": values["max_accuracy"]["accuracy"]["mean"],
                "accuracy_std": values["max_accuracy"]["accuracy"]["std"],
                "gap_vs_all_mini": values["max_accuracy"]["gap_vs_all_mini"]["mean"],
                "mini_rate": values["max_accuracy"]["mini_rate"]["mean"],
                "mini_rate_std": values["max_accuracy"]["mini_rate"]["std"],
                "critical_recall": values["max_accuracy"]["critical_recall"]["mean"],
            }
            for tokens, values in summaries.items()
        ],
        key=lambda row: (row["accuracy"], -row["mini_rate"]),
        reverse=True,
    )
    first_test = splits[args.seeds[0]][2]
    result = {
        "config": {
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "method": "sentiment_ce/nano_disagreement",
            "label_protocol": "200 unique labeled reviews",
            "openai_api_calls": 0,
        },
        "available_counts": dict(Counter(x["bucket"] for x in examples)),
        "baselines": {
            "all_nano": evaluate(first_test, [0] * len(first_test)),
            "all_mini": evaluate(first_test, [1] * len(first_test)),
        },
        "ranking_max_accuracy": ranking,
        "aggregate": summaries,
        "runs": {str(tokens): runs for tokens, runs in by_tokens.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "baselines": result["baselines"],
        "ranking_max_accuracy": ranking,
    }, indent=2), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
