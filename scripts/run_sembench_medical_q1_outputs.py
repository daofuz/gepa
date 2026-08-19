#!/usr/bin/env python3
"""Collect cached GPT-5 mini/nano outputs for SemBench Medical Q1.

Medical Q1 is a text semantic filter: determine whether a patient's symptom
description indicates an allergy.  The sampled dataset contains every positive
ground-truth row plus a reproducible sample of negative rows.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


LABEL_RE = re.compile(r"\b(NO_ALLERGY|ALLERGY)\b", re.IGNORECASE)
PROMPT = """Determine whether the patient's described symptoms indicate an allergy.
This is a benchmark classification task, not medical advice.
Return exactly one label: ALLERGY or NO_ALLERGY.

Symptoms:
{symptoms}
"""


def save_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def sample_rows(
    symptoms_path: Path, ground_truth_path: Path, sample_size: int, seed: int
) -> list[dict[str, Any]]:
    symptoms = pd.read_csv(symptoms_path)
    ground_truth = pd.read_csv(ground_truth_path)
    positive_ids = set(ground_truth["patient_id"].astype(int).tolist())
    positives = symptoms[symptoms["patient_id"].isin(positive_ids)]
    negatives = symptoms[~symptoms["patient_id"].isin(positive_ids)]
    if sample_size < len(positives):
        raise ValueError("sample_size must include every Medical Q1 positive")
    negative_count = sample_size - len(positives)
    sampled_negative = negatives.sample(n=negative_count, random_state=seed, replace=False)
    sampled = pd.concat([positives, sampled_negative], ignore_index=True).sample(
        frac=1.0, random_state=seed
    )
    rows = []
    for row in sampled.to_dict(orient="records"):
        patient_id = int(row["patient_id"])
        rows.append({
            "patient_id": str(patient_id),
            "symptom_id": str(int(row["symptom_id"])),
            "text": str(row["symptoms"]).strip(),
            "gold": "ALLERGY" if patient_id in positive_ids else "NO_ALLERGY",
        })
    if len({row["patient_id"] for row in rows}) != sample_size:
        raise AssertionError("Medical Q1 sample contains duplicate patients")
    return rows


def call_model(client: OpenAI, model: str, symptoms: str) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            response = client.responses.create(
                model=model,
                input=PROMPT.format(symptoms=symptoms),
                reasoning={"effort": "minimal"},
                text={"verbosity": "low"},
                max_output_tokens=64,
            )
            output_text = response.output_text.strip()
            match = LABEL_RE.search(output_text)
            if not match:
                raise ValueError(f"No allergy label in response: {output_text!r}")
            usage = response.usage
            return {
                "pred": match.group(1).upper(),
                "response": output_text,
                "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
                "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
            }
        except Exception as exc:
            last_error = exc
            time.sleep(min(2**attempt, 12))
    raise RuntimeError(f"{model} failed after retries") from last_error


def collect(
    rows: list[dict[str, Any]], models: list[str], cache_path: Path, workers: int
) -> dict[str, dict[str, Any]]:
    cache = load_cache(cache_path)
    client = OpenAI()
    jobs = []
    for row in rows:
        for model in models:
            key = f"{row['patient_id']}::{model}"
            if key not in cache:
                jobs.append((key, model, row["text"]))
    print(f"uncached row/model pairs: {len(jobs)}", flush=True)
    if jobs:
        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(call_model, client, model, symptoms): key
                for key, model, symptoms in jobs
            }
            for future in as_completed(futures):
                key = futures[future]
                cache[key] = future.result()
                completed += 1
                if completed % 20 == 0 or completed == len(jobs):
                    save_json_atomic(cache_path, cache)
                    print(f"  {completed}/{len(jobs)} complete", flush=True)
    return cache


def attach(
    rows: list[dict[str, Any]], cache: dict[str, dict[str, Any]], mini: str, nano: str
) -> list[dict[str, Any]]:
    examples = []
    for row in rows:
        mini_output = cache[f"{row['patient_id']}::{mini}"]
        nano_output = cache[f"{row['patient_id']}::{nano}"]
        mini_correct = mini_output["pred"] == row["gold"]
        nano_correct = nano_output["pred"] == row["gold"]
        bucket = (
            "both_correct" if mini_correct and nano_correct else
            "mini_only" if mini_correct else
            "nano_only" if nano_correct else
            "both_wrong"
        )
        examples.append({
            **row,
            "mini_pred": mini_output["pred"],
            "nano_pred": nano_output["pred"],
            "mini_correct": mini_correct,
            "nano_correct": nano_correct,
            "bucket": bucket,
            "target": 1 if bucket in {"mini_only", "both_wrong"} else 0,
        })
    return examples


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--symptoms", type=Path,
        default=root / "sembench/files/medical/data/text_symptoms_data.csv",
    )
    parser.add_argument(
        "--ground-truth", type=Path,
        default=root / "sembench/files/medical/raw_results/ground_truth/Q1.csv",
    )
    parser.add_argument("--sample-size", type=int, default=412)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--mini-model", default="gpt-5-mini")
    parser.add_argument("--nano-model", default="gpt-5-nano")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--cache", type=Path,
        default=root / "sembench_medical_router/model_outputs.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=root / "sembench_medical_router/medical_q1_examples.json",
    )
    args = parser.parse_args()

    load_dotenv(root / ".env")
    random.seed(args.seed)
    rows = sample_rows(args.symptoms, args.ground_truth, args.sample_size, args.seed)
    cache = collect(rows, [args.nano_model, args.mini_model], args.cache, args.workers)
    examples = attach(rows, cache, args.mini_model, args.nano_model)
    result = {
        "config": {
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "semantic_operator": "This patient has symptoms of an allergy",
        },
        "class_counts": dict(Counter(x["gold"] for x in examples)),
        "bucket_counts": dict(Counter(x["bucket"] for x in examples)),
        "all_nano_accuracy": mean(int(x["nano_correct"]) for x in examples),
        "all_mini_accuracy": mean(int(x["mini_correct"]) for x in examples),
        "token_usage": {
            model: {
                "input_tokens": sum(
                    cache[f"{row['patient_id']}::{model}"]["input_tokens"] for row in rows
                ),
                "output_tokens": sum(
                    cache[f"{row['patient_id']}::{model}"]["output_tokens"] for row in rows
                ),
            }
            for model in (args.nano_model, args.mini_model)
        },
        "examples": examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "class_counts": result["class_counts"],
        "bucket_counts": result["bucket_counts"],
        "all_nano_accuracy": result["all_nano_accuracy"],
        "all_mini_accuracy": result["all_mini_accuracy"],
        "token_usage": result["token_usage"],
    }, indent=2), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    from statistics import mean

    main()
