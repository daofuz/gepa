#!/usr/bin/env python3
"""Recover a GEPA-style result JSON from a run_dir candidates.json file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import optimize_ollama_router_gepa as opt


def parse_val_scores_from_log_text(log_text: str, candidate_count: int) -> list[float | None]:
    scores: list[float | None] = [None] * candidate_count
    for raw_line in log_text.splitlines():
        line = raw_line.strip()
        if "Base program full valset score:" in line:
            try:
                scores[0] = float(line.split("Base program full valset score:", 1)[1].split()[0])
            except (IndexError, ValueError):
                pass
        if "Val aggregate for new program:" in line:
            try:
                score = float(line.rsplit(":", 1)[1].strip())
            except ValueError:
                continue
            # The previous "New program candidate index" line appears after this log line,
            # so assign scores to the next unseen non-seed candidate in creation order.
            for index in range(1, candidate_count):
                if scores[index] is None:
                    scores[index] = score
                    break
    return [score if score is not None else float("-inf") for score in scores]


def strip_trace_scores(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in summary.items()
        if key not in {"scores", "routes", "traces"}
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--sampling", default="balanced")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--score-mode", default="risk_averse_utility")
    parser.add_argument("--router-output-format", default="json")
    parser.add_argument("--router-model", default="qwen3.5")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--ollama-timeout", type=float, default=600.0)
    parser.add_argument("--router-num-predict", type=int, default=256)
    parser.add_argument("--mini-file", default="computer science_result_mini.json")
    parser.add_argument("--nano-file", default="computer science_result_nano.json")
    parser.add_argument("--mini-cost-model", default="gpt-4.1-mini")
    parser.add_argument("--nano-cost-model", default="gpt-4.1-nano")
    parser.add_argument("--reflection-provider", default="openai")
    parser.add_argument("--reflection-model", default="gpt-4.1")
    parser.add_argument(
        "--fast-log-only",
        action="store_true",
        help="Recover from candidates.json and run_log.txt without making any Ollama calls.",
    )
    args = parser.parse_args()

    opt.load_dotenv()
    run_dir = Path(args.run_dir)
    candidates = json.loads((run_dir / "candidates.json").read_text(encoding="utf-8"))
    all_examples = opt.load_examples(
        Path(args.mini_file),
        Path(args.nano_file),
        mini_cost_model=args.mini_cost_model,
        nano_cost_model=args.nano_cost_model,
    )
    examples = opt.select_examples(all_examples, args.limit, args.sampling, args.seed)
    trainset, valset = opt.split_train_val(examples, args.train_ratio, args.seed)

    if args.fast_log_only:
        log_text = (run_dir / "run_log.txt").read_text(encoding="utf-8", errors="replace")
        val_aggregate_scores = parse_val_scores_from_log_text(log_text, len(candidates))
        selected_idx = max(range(len(candidates)), key=lambda index: (val_aggregate_scores[index], index))
        seed_prompt = (
            opt.DEFAULT_STRUCTURED_PROMPT
            if args.router_output_format == "json"
            else opt.DEFAULT_PROMPT
        )
        payload = {
            "recovered_from_run_dir": args.run_dir,
            "recovery_mode": "fast_log_only",
            "ollama_model": args.router_model,
            "router_model": args.router_model,
            "reflection_model": args.reflection_model,
            "reflection_provider": args.reflection_provider,
            "mini_cost_model": args.mini_cost_model,
            "nano_cost_model": args.nano_cost_model,
            "router_output_format": args.router_output_format,
            "limit": args.limit,
            "sampling": args.sampling,
            "split_file": None,
            "target_train_counts": None,
            "target_val_counts": None,
            "train_ratio": args.train_ratio,
            "selected_outcome_counts": opt.count_outcomes(examples),
            "train_outcome_counts": opt.count_outcomes(trainset),
            "val_outcome_counts": opt.count_outcomes(valset),
            "scoring": {
                "score_mode": args.score_mode,
                "router_output_format": args.router_output_format,
                "risk_averse_utility_rule": opt.RISK_AVERSE_UTILITY_RULE,
                "pricing_per_1m_tokens": opt.PRICING_PER_1M,
            },
            "initial_prompt": seed_prompt,
            "best_prompt": candidates[selected_idx]["router_prompt"],
            "selected_candidate_index": selected_idx,
            "gepa_default_best_index": None,
            "gepa_val_aggregate_scores": val_aggregate_scores,
            "candidate_selection_rule": "Fast recovery uses GEPA validation aggregate scores from run_log.txt.",
            "candidate_val_summaries": {},
            "gepa_candidates": candidates,
            "initial_summary": {
                "traces": [{"question_id": example.question_id} for example in examples]
            },
            "best_summary": {
                "traces": [{"question_id": example.question_id} for example in examples]
            },
            "gepa_result_repr": "Recovered from candidates.json and run_log.txt after interrupted GEPA run.",
        }
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("selected", selected_idx, flush=True)
        print("val_scores", val_aggregate_scores, flush=True)
        print(f"Wrote {args.output}", flush=True)
        return

    router = opt.OllamaChat(
        model=args.router_model,
        host=args.ollama_host,
        timeout=args.ollama_timeout,
        num_predict=args.router_num_predict,
        think=False,
    )
    adapter = opt.RoutingAdapter(
        router,
        score_mode=args.score_mode,
        router_output_format=args.router_output_format,
    )

    val_aggregate_scores: list[float] = []
    for index, candidate in enumerate(candidates):
        summary = opt.summarize(adapter, valset, candidate["router_prompt"])
        val_aggregate_scores.append(summary["mean_score"])
        print(
            index,
            "val_score",
            summary["mean_score"],
            "accuracy",
            summary["accuracy"],
            "routes",
            summary["nano_routes"],
            summary["mini_routes"],
            "failures",
            summary["failure_type_counts"],
            flush=True,
        )

    result = SimpleNamespace(candidates=candidates, val_aggregate_scores=val_aggregate_scores)
    selected_idx, candidate_val_summaries = opt.select_candidate_index(result, adapter, valset)
    seed_prompt = (
        opt.DEFAULT_STRUCTURED_PROMPT
        if args.router_output_format == "json"
        else opt.DEFAULT_PROMPT
    )
    best_prompt = candidates[selected_idx]["router_prompt"]
    initial_summary = opt.summarize(adapter, examples, seed_prompt)
    best_summary = opt.summarize(adapter, examples, best_prompt)

    payload = {
        "recovered_from_run_dir": args.run_dir,
        "ollama_model": args.router_model,
        "router_model": args.router_model,
        "reflection_model": args.reflection_model,
        "reflection_provider": args.reflection_provider,
        "mini_cost_model": args.mini_cost_model,
        "nano_cost_model": args.nano_cost_model,
        "router_output_format": args.router_output_format,
        "limit": args.limit,
        "sampling": args.sampling,
        "split_file": None,
        "target_train_counts": None,
        "target_val_counts": None,
        "train_ratio": args.train_ratio,
        "selected_outcome_counts": opt.count_outcomes(examples),
        "train_outcome_counts": opt.count_outcomes(trainset),
        "val_outcome_counts": opt.count_outcomes(valset),
        "scoring": {
            "score_mode": args.score_mode,
            "router_output_format": args.router_output_format,
            "risk_averse_utility_rule": opt.RISK_AVERSE_UTILITY_RULE,
            "pricing_per_1m_tokens": opt.PRICING_PER_1M,
        },
        "initial_prompt": seed_prompt,
        "best_prompt": best_prompt,
        "selected_candidate_index": selected_idx,
        "gepa_default_best_index": None,
        "gepa_val_aggregate_scores": val_aggregate_scores,
        "candidate_selection_rule": (
            "Maximize validation aggregate score, then prefer fewer critical_misroute "
            "cases, fewer invalid routes, fewer unnecessary expensive routes, higher "
            "validation accuracy, and finally newer candidate index."
        ),
        "candidate_val_summaries": {
            str(index): strip_trace_scores(summary)
            for index, summary in candidate_val_summaries.items()
        },
        "gepa_candidates": candidates,
        "initial_summary": initial_summary,
        "best_summary": best_summary,
        "gepa_result_repr": "Recovered from candidates.json after interrupted GEPA run.",
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("selected", selected_idx, flush=True)
    print("initial_mean", initial_summary["mean_score"], flush=True)
    print("best_mean", best_summary["mean_score"], flush=True)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
