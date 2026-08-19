#!/usr/bin/env python3
"""Build real GPT-4.1 nano/mini routing labels on DeepScholar-Bench.

Both routes use the same DeepScholar pipeline configuration; only the OpenAI
model changes.  Per-query reference coverage is evaluated offline, then the
paired scores are written in the format consumed by the soft-prompt router.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import dotenv_values


WORKSPACE = Path(__file__).resolve().parents[1]
DEEPSCHOLAR = WORKSPACE / "deepscholar"
QUERIES = DEEPSCHOLAR / "dataset/queries.csv"
DATASET = DEEPSCHOLAR / "dataset/papers_with_related_works.csv"
IMPORTANT_CITATIONS = DEEPSCHOLAR / "dataset/important_citations.csv"
NO_YAML = "configs/no_yaml.yaml"
NANO_MODEL = "gpt-4.1-nano"
MINI_MODEL = "gpt-4.1-mini"


def load_environment() -> None:
    for path in (WORKSPACE / ".env", DEEPSCHOLAR / ".env"):
        if not path.exists():
            continue
        for key, value in dotenv_values(path).items():
            if value is not None:
                os.environ.setdefault(key, value)
    os.environ["PYTHONUTF8"] = "1"


def run(command: list[str], dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=DEEPSCHOLAR, check=True, env=os.environ.copy())


def ensure_queries() -> None:
    if QUERIES.exists():
        return
    sys.path.insert(0, str(DEEPSCHOLAR))
    from deepscholar_base.main import load_queries

    load_queries(str(QUERIES.relative_to(DEEPSCHOLAR)))


def generation_command(
    python: str,
    model: str,
    output_folder: Path,
    idx: int,
    args: argparse.Namespace,
) -> list[str]:
    return [
        python,
        "-m",
        "deepscholar_base.main",
        "--config-yaml",
        NO_YAML,
        "--queries-file",
        str(QUERIES.relative_to(DEEPSCHOLAR)),
        "--output-folder",
        str(output_folder.relative_to(DEEPSCHOLAR)),
        "--start-idx",
        str(idx),
        "--end-idx",
        str(idx + 1),
        "--model",
        model,
        "--search-mode",
        "recursive",
        "--no-web-search",
        "--num-search-steps",
        str(args.search_steps),
        "--num-search-queries-per-step",
        str(args.search_queries),
        "--per-query-max-search-results",
        str(args.search_results),
        "--final-max-results",
        str(args.final_results),
        "--use-structured-output",
        "--use-sem-filter",
        "--use-sem-topk",
        "--categorize-references",
        "--generate-category-summary",
        "--generate-insights",
    ]


def generation_complete(output_folder: Path, idx: int) -> bool:
    query_folder = output_folder / str(idx)
    required = ("final_report.md", "intro.md", "paper.csv", "stats.json")
    return all((query_folder / name).is_file() for name in required)


def summary_error(output_folder: Path) -> str:
    path = output_folder / "summary.json"
    if not path.exists():
        return "DeepScholar did not write summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload and payload[0].get("status") == "error":
        return str(payload[0].get("error", "unknown DeepScholar error"))
    return "DeepScholar output is incomplete"


def generate_route(
    python: str,
    model: str,
    output_folder: Path,
    ids: list[int],
    args: argparse.Namespace,
) -> None:
    for idx in ids:
        if generation_complete(output_folder, idx):
            print(f"{model} query {idx}: cached", flush=True)
            continue
        run(generation_command(python, model, output_folder, idx, args), args.dry_run)
        if not args.dry_run and not generation_complete(output_folder, idx):
            raise RuntimeError(f"{model} query {idx} failed: {summary_error(output_folder)}")


def evaluation_command(
    python: str,
    model: str,
    input_folder: Path,
    output_folder: Path,
    ids: list[int],
) -> list[str]:
    return [
        python,
        "-m",
        "eval.main",
        "--config-yaml",
        NO_YAML,
        "--modes",
        "deepscholar_base",
        "--evals",
        "reference_coverage",
        "--input-folder",
        str(input_folder.relative_to(DEEPSCHOLAR)),
        "--output-folder",
        str(output_folder.relative_to(DEEPSCHOLAR)),
        "--dataset-path",
        str(DATASET.relative_to(DEEPSCHOLAR)),
        "--important-citations-path",
        str(IMPORTANT_CITATIONS.relative_to(DEEPSCHOLAR)),
        "--model-name",
        model,
        "--file-id",
        *[str(idx) for idx in ids],
    ]


def score_file(eval_folder: Path) -> Path:
    return eval_folder / "reference_coverage/deepscholar_base.csv"


def load_scores(path: Path) -> dict[int, float]:
    scores: dict[int, float] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            idx = int(Path(row["folder_path"]).name)
            scores[idx] = float(row["reference_coverage"])
    return scores


def load_query_ids() -> dict[int, str]:
    result: dict[int, str] = {}
    with QUERIES.open("r", encoding="utf-8-sig", newline="") as handle:
        for idx, row in enumerate(csv.DictReader(handle)):
            result[idx] = row["arxiv_id"].strip()
    return result


def write_labels(
    path: Path,
    ids: list[int],
    nano_scores: dict[int, float],
    mini_scores: dict[int, float],
) -> int:
    query_ids = load_query_ids()
    matched = [idx for idx in ids if idx in nano_scores and idx in mini_scores]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["arxiv_id", "cheap_score", "full_score", "query_index"],
        )
        writer.writeheader()
        for idx in matched:
            writer.writerow(
                {
                    "arxiv_id": query_ids[idx],
                    "cheap_score": nano_scores[idx],
                    "full_score": mini_scores[idx],
                    "query_index": idx,
                }
            )
    return len(matched)


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    label_count: int,
    labels_path: Path,
) -> None:
    payload: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "models": {"cheap": NANO_MODEL, "full": MINI_MODEL},
        "controlled_difference": "Only the model name differs between routes.",
        "query_slice": {"start": args.start_idx, "end": args.end_idx},
        "pipeline": {
            "search_mode": "recursive",
            "web_search": False,
            "search_steps": args.search_steps,
            "search_queries_per_step": args.search_queries,
            "results_per_query": args.search_results,
            "final_results": args.final_results,
            "semantic_filter": True,
            "semantic_topk": True,
            "categorize_references": True,
            "generate_category_summary": True,
            "generate_insights": True,
        },
        "routing_score": "reference_coverage",
        "minimum_mini_gain": args.minimum_mini_gain,
        "label_count": label_count,
        "labels_csv": str(labels_path),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=20)
    parser.add_argument("--search-steps", type=int, default=1)
    parser.add_argument("--search-queries", type=int, default=1)
    parser.add_argument("--search-results", type=int, default=5)
    parser.add_argument("--final-results", type=int, default=10)
    parser.add_argument("--minimum-mini-gain", type=float, default=0.05)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEEPSCHOLAR / "outputs/gpt41_route",
    )
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--skip-softprompt", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.start_idx < 0 or args.end_idx <= args.start_idx:
        parser.error("Require 0 <= start-idx < end-idx")
    load_environment()
    ensure_queries()
    ids = list(range(args.start_idx, args.end_idx))
    python = sys.executable
    nano_folder = args.output_root / "nano"
    mini_folder = args.output_root / "mini"
    nano_eval = args.output_root / "eval_nano"
    mini_eval = args.output_root / "eval_mini"
    labels_path = args.output_root / "gpt41_nano_mini_labels.csv"

    if args.dry_run:
        print("Dry run: GPT-4.1 nano/mini use identical pipeline settings.", flush=True)
    elif not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing")

    if not args.skip_generation:
        generate_route(python, NANO_MODEL, nano_folder, ids, args)
        generate_route(python, MINI_MODEL, mini_folder, ids, args)

    if args.dry_run:
        run(evaluation_command(python, NANO_MODEL, nano_folder, nano_eval, ids), True)
        run(evaluation_command(python, MINI_MODEL, mini_folder, mini_eval, ids), True)
        return

    if not args.skip_evaluation:
        run(evaluation_command(python, NANO_MODEL, nano_folder, nano_eval, ids), False)
        run(evaluation_command(python, MINI_MODEL, mini_folder, mini_eval, ids), False)

    nano_score_path = score_file(nano_eval)
    mini_score_path = score_file(mini_eval)
    if not nano_score_path.exists() or not mini_score_path.exists():
        raise RuntimeError("Paired reference-coverage results are incomplete")
    label_count = write_labels(
        labels_path,
        ids,
        load_scores(nano_score_path),
        load_scores(mini_score_path),
    )
    write_manifest(args.output_root / "manifest.json", args, label_count, labels_path)
    print(f"Wrote {label_count} paired labels to {labels_path}", flush=True)

    if args.skip_softprompt:
        return
    if label_count < 20:
        print("Need at least 20 paired labels; softprompt training was not started.", flush=True)
        return
    softprompt_command = [
        "py",
        "-3.13",
        str(WORKSPACE / "scripts/run_deepscholar_softprompt_router.py"),
        "--labels-csv",
        str(labels_path),
        "--minimum-full-gain",
        str(args.minimum_mini_gain),
        "--output",
        str(WORKSPACE / "outputs/deepscholar_gpt41_nano_mini_softprompt.json"),
        "--save-model",
        str(WORKSPACE / "outputs/deepscholar_gpt41_nano_mini_softprompt.pt"),
    ]
    run(softprompt_command, False)


if __name__ == "__main__":
    main()
