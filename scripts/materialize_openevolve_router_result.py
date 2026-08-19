from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optimize_ollama_router_gepa import (
    OllamaChat,
    RoutingAdapter,
    count_outcomes,
    load_examples,
    split_train_val_from_file,
    summarize,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker", default="openevolve_router_best.json")
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--mini-file", default="computer science_result_mini.json")
    parser.add_argument("--nano-file", default="computer science_result_nano.json")
    parser.add_argument("--router-model", default="qwen3.5:latest")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--ollama-timeout", type=float, default=600.0)
    parser.add_argument("--router-num-predict", type=int, default=32)
    parser.add_argument("--score-mode", default="risk_averse_utility")
    parser.add_argument("--router-output-format", default="token")
    parser.add_argument("--output", default="gepa_router_prompt_result_openevolve.json")
    args = parser.parse_args()

    tracker = json.loads(Path(args.tracker).read_text(encoding="utf-8"))
    candidate = tracker["candidate"]

    examples = load_examples(Path(args.mini_file), Path(args.nano_file))
    trainset, valset = split_train_val_from_file(examples, Path(args.split_file))
    selected = trainset + valset

    router = OllamaChat(
        model=args.router_model,
        host=args.ollama_host,
        timeout=args.ollama_timeout,
        num_predict=args.router_num_predict,
        think=False,
        retry_empty_with_tokens=max(128, args.router_num_predict * 4),
    )
    adapter = RoutingAdapter(
        router_lm=router,
        score_mode=args.score_mode,
        router_output_format=args.router_output_format,
        include_nano_answer=False,
    )

    best_summary = summarize(adapter, selected, candidate)
    payload = {
        "ollama_model": args.router_model,
        "router_model": args.router_model,
        "reflection_model": None,
        "reflection_provider": "openevolve",
        "optimizer": "openevolve",
        "openevolve": tracker,
        "router_output_format": args.router_output_format,
        "include_nano_answer": False,
        "nano_response_max_chars": 4000,
        "hybrid_easy_router": bool(candidate.get("easy_router_code")),
        "split_file": args.split_file,
        "selected_outcome_counts": count_outcomes(selected),
        "train_outcome_counts": count_outcomes(trainset),
        "val_outcome_counts": count_outcomes(valset),
        "scoring": {
            "score_mode": args.score_mode,
            "router_output_format": args.router_output_format,
            "include_nano_answer": False,
        },
        "initial_prompt": None,
        "initial_easy_router_code": None,
        "best_prompt": candidate["router_prompt"],
        "best_easy_router_code": candidate.get("easy_router_code", ""),
        "selected_candidate_index": tracker.get("program_path"),
        "candidate_selection_rule": (
            "OpenEvolve maximizes evaluator combined_score, which prioritizes routing "
            "accuracy and penalizes critical misroutes and invalid outputs."
        ),
        "gepa_candidates": None,
        "adaevolve_candidates": None,
        "openevolve_candidates": [candidate],
        "initial_summary": None,
        "best_summary": best_summary,
        "gepa_result_repr": None,
    }
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(
        "Best accuracy:",
        best_summary["accuracy"],
        "mean_score:",
        best_summary["mean_score"],
    )


if __name__ == "__main__":
    main()

