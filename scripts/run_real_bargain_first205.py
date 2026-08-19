#!/usr/bin/env python3
"""Run real BARGAIN-style calibration with proxy logprobs.

This script uses the installed BARGAIN thresholding logic, but separates
calibration and test data:
- calibrate threshold on first-205 IDs from a split file
- apply the learned score cutoff to held-out last-205 IDs

Proxy: fresh OpenAI call to gpt-4.1-nano with answer-letter logprob confidence.
Oracle: saved mini prediction from computer science_result_mini.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyze_routing_costs import count_rows, add_deltas, summarize_route
from BARGAIN.process.BARGAIN_A import test_if_true_mean_is_above_m, test_if_true_mean_is_below_m
from BARGAIN.sampler.wor_sampler import WoR_Sampler


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def option_prompt(row: dict[str, Any]) -> list[dict[str, str]]:
    options = "\n".join(f"{chr(65 + i)}. {option}" for i, option in enumerate(row["options"]))
    return [
        {"role": "system", "content": "Answer the multiple-choice question. Reply with only the option letter."},
        {"role": "user", "content": f"Question:\n{row['question']}\n\nOptions:\n{options}"},
    ]


def parse_letter(text: str, max_options: int) -> str:
    text = str(text or "").strip().upper()
    allowed = {chr(65 + i) for i in range(max_options)}
    for char in text:
        if char in allowed:
            return char
    match = re.search(r"\b([A-Z])\b", text)
    if match and match.group(1) in allowed:
        return match.group(1)
    return ""


def confidence_from_logprobs(choice: Any, pred: str) -> float:
    content = getattr(getattr(choice, "logprobs", None), "content", None) or []
    if not content:
        return float("-inf")
    # Prefer the generated answer-letter token. If tokenization includes spaces,
    # strip token text before comparing with the parsed letter.
    for tok in content:
        token = str(getattr(tok, "token", "")).strip().upper()
        if token == pred:
            return float(getattr(tok, "logprob", float("-inf")))
    # Fallback: use the first token logprob, which is usually the answer letter.
    return float(getattr(content[0], "logprob", float("-inf")))


def load_proxy_cache(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): v for k, v in data.items()}


def save_proxy_cache(path: Path, cache: dict[int, dict[str, Any]]) -> None:
    path.write_text(json.dumps({str(k): v for k, v in sorted(cache.items())}, indent=2), encoding="utf-8")


def collect_proxy_outputs(
    rows: list[dict[str, Any]],
    ids: list[int],
    client: OpenAI,
    model: str,
    cache_path: Path,
    sleep_seconds: float,
    limit: int = 0,
) -> dict[int, dict[str, Any]]:
    by_id = {int(row["question_id"]): row for row in rows}
    cache = load_proxy_cache(cache_path)
    todo = [qid for qid in ids if qid not in cache]
    if limit > 0:
        todo = todo[:limit]
    print(f"Proxy cache has {len(cache)} rows; collecting {len(todo)} missing rows.", flush=True)
    for idx, qid in enumerate(todo, start=1):
        row = by_id[qid]
        resp = client.chat.completions.create(
            model=model,
            messages=option_prompt(row),
            temperature=0,
            max_tokens=2,
            logprobs=True,
            top_logprobs=5,
        )
        choice = resp.choices[0]
        raw = choice.message.content or ""
        pred = parse_letter(raw, len(row["options"]))
        score = confidence_from_logprobs(choice, pred)
        cache[qid] = {
            "question_id": qid,
            "pred": pred,
            "raw_response": raw,
            "confidence_logprob": score,
            "confidence_probability": math.exp(score) if score > -100 else 0.0,
            "model": model,
            "usage": resp.usage.model_dump() if getattr(resp, "usage", None) else None,
        }
        if idx == 1 or idx % 25 == 0 or idx == len(todo):
            save_proxy_cache(cache_path, cache)
            print(f"Collected {idx}/{len(todo)} proxy logprob rows.", flush=True)
        if sleep_seconds:
            time.sleep(sleep_seconds)
    save_proxy_cache(cache_path, cache)
    return cache


def check_worth_trying(sample_indx: np.ndarray, sample_is_correct: np.ndarray, t: int, target: float) -> bool:
    if len(sample_indx) < 50:
        return True
    mask_at_t = sample_indx <= t
    samples_at_thresh = sample_is_correct[mask_at_t]
    if np.mean(samples_at_thresh) - np.std(samples_at_thresh) < target:
        return False
    return True


def sample_till_confident_above_target(
    sampler: WoR_Sampler,
    sorted_data_ids: np.ndarray,
    sorted_proxy_preds: np.ndarray,
    confidence: float,
    target: float,
    total_sampled: int,
    curr_thresh: int,
    oracle_pred_by_id: dict[int, str],
) -> tuple[bool, np.ndarray, int]:
    sample_step = 10
    sampled_is_correct = np.array([])
    sampled_index = np.array([], dtype=int)
    while check_worth_trying(sampled_index, sampled_is_correct, curr_thresh, target):
        sampled_indexes, budget_used, sampled_all = sampler.sample(curr_thresh, sample_step)
        sampled_data_ids = sorted_data_ids[sampled_indexes]
        proxy_preds = sorted_proxy_preds[sampled_indexes]
        correctness = np.array([
            str(proxy_pred).strip().upper() == oracle_pred_by_id[int(qid)]
            for qid, proxy_pred in zip(sampled_data_ids, proxy_preds)
        ])
        sampled_is_correct = np.concatenate([sampled_is_correct, correctness])
        sampled_index = np.concatenate([sampled_index, sampled_indexes])
        total_sampled += int(budget_used)
        if sampled_all:
            return not np.mean(sampled_is_correct) < target, sampled_index, total_sampled
        samples_at_thresh = sampled_is_correct[sampled_index <= curr_thresh]
        n_at_thresh = curr_thresh + 1
        if np.mean(samples_at_thresh) < target:
            conf_has_target = test_if_true_mean_is_below_m(
                samples_at_thresh,
                target,
                alpha=confidence,
                without_replacement=True,
                N=n_at_thresh,
                fixed_sample_size=False,
            )
            is_below_target = True
        else:
            conf_has_target = test_if_true_mean_is_above_m(
                samples_at_thresh,
                target,
                alpha=confidence,
                without_replacement=True,
                N=n_at_thresh,
                fixed_sample_size=False,
            )
            is_below_target = False
        if not conf_has_target:
            return not is_below_target, sampled_index, total_sampled
    return False, sampled_index, total_sampled


def calibrate_bargain_threshold(
    train_ids: list[int],
    proxy_cache: dict[int, dict[str, Any]],
    oracle_pred_by_id: dict[int, str],
    target: float,
    delta: float,
    m: int,
    seed: int,
) -> dict[str, Any]:
    np.random.seed(seed)
    data_ids = np.array(train_ids, dtype=int)
    proxy_preds = np.array([proxy_cache[int(qid)]["pred"] for qid in data_ids])
    proxy_scores = np.array([float(proxy_cache[int(qid)]["confidence_logprob"]) for qid in data_ids])
    sort_indx = np.argsort(proxy_scores)[::-1]
    sorted_data_ids = data_ids[sort_indx]
    sorted_proxy_preds = proxy_preds[sort_indx]
    sorted_proxy_scores = proxy_scores[sort_indx]
    sampler = WoR_Sampler(len(sorted_data_ids))
    thresh_step = max(len(sorted_data_ids) // m, 1)
    sample_indexes = np.array([])
    total_sampled = 0
    best_thresh = 0
    log: list[dict[str, Any]] = []
    for curr_thresh in range(thresh_step - 1, len(sorted_data_ids), thresh_step):
        if curr_thresh == len(sorted_data_ids) - 1:
            new_target = target
        else:
            n_from_proxy = curr_thresh + 1
            n_from_oracle = len(sorted_data_ids) - n_from_proxy
            new_target = (target * (n_from_oracle + n_from_proxy) - n_from_oracle) / n_from_proxy
            if new_target <= 0:
                log.append({"curr_thresh": curr_thresh, "new_target": new_target, "skipped": True})
                continue
        confident, sampled_index, total_sampled = sample_till_confident_above_target(
            sampler,
            sorted_data_ids,
            sorted_proxy_preds,
            delta,
            new_target,
            total_sampled,
            curr_thresh,
            oracle_pred_by_id,
        )
        sample_indexes = np.concatenate([sample_indexes, sampled_index])
        log.append({
            "curr_thresh": int(curr_thresh),
            "score_cutoff_at_thresh": float(sorted_proxy_scores[curr_thresh]),
            "new_target": float(new_target),
            "confident_above_target": bool(confident),
            "total_sampled": int(total_sampled),
        })
        if not confident:
            break
        best_thresh = curr_thresh
    proxy_sorted_positions = np.setdiff1d(np.arange(len(sorted_data_ids))[:best_thresh], np.array(sample_indexes).astype(int))
    proxy_ids = sorted_data_ids[proxy_sorted_positions]
    score_cutoff = float(sorted_proxy_scores[best_thresh]) if len(sorted_proxy_scores) else float("inf")
    # For held-out application, use the learned score cutoff. BARGAIN's in-sample
    # process excludes sampled calibration rows from proxy use; this exclusion has
    # no analogue on held-out test rows.
    return {
        "target": target,
        "delta": delta,
        "M": m,
        "seed": seed,
        "best_thresh_index": int(best_thresh),
        "score_cutoff": score_cutoff,
        "calibration_oracle_calls": int(total_sampled),
        "calibration_proxy_ids_in_in_sample_process": [int(x) for x in proxy_ids.tolist()],
        "calibration_proxy_fraction_in_in_sample_process": len(proxy_ids) / len(sorted_data_ids) if len(sorted_data_ids) else 0.0,
        "threshold_log": log,
    }


def route_from_cutoff(ids: list[int], proxy_cache: dict[int, dict[str, Any]], cutoff: float) -> dict[int, str]:
    return {
        int(qid): "nano" if float(proxy_cache[int(qid)]["confidence_logprob"]) >= cutoff else "mini"
        for qid in ids
    }


def summarize_cascade_cost(ids: list[int], route_by_id: dict[int, str], proxy_cache: dict[int, dict[str, Any]], mini_rows: dict[int, Any]) -> dict[str, Any]:
    proxy_prompt_tokens = 0
    proxy_completion_tokens = 0
    proxy_total_tokens = 0
    for qid in ids:
        usage = proxy_cache[int(qid)].get("usage") or {}
        proxy_prompt_tokens += int(usage.get("prompt_tokens") or 0)
        proxy_completion_tokens += int(usage.get("completion_tokens") or 0)
        proxy_total_tokens += int(usage.get("total_tokens") or 0)
    # gpt-4.1-nano pricing from analyze_routing_costs.py.
    proxy_cost = proxy_prompt_tokens * 0.10 / 1_000_000 + proxy_completion_tokens * 0.40 / 1_000_000
    oracle_cost = sum(mini_rows[int(qid)].cost_usd for qid in ids if route_by_id[int(qid)] == "mini")
    return {
        "cascade_proxy_prompt_tokens": proxy_prompt_tokens,
        "cascade_proxy_completion_tokens": proxy_completion_tokens,
        "cascade_proxy_total_tokens": proxy_total_tokens,
        "cascade_proxy_cost_usd": proxy_cost,
        "cascade_oracle_mini_cost_usd": oracle_cost,
        "cascade_total_cost_usd": proxy_cost + oracle_cost,
        "cascade_cost_per_1000_questions": (proxy_cost + oracle_cost) * 1000 / len(ids) if ids else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mini-file", default="computer science_result_mini.json")
    parser.add_argument("--nano-file", default="computer science_result_nano.json")
    parser.add_argument("--split-file", default="routing_split_first205_all_train_seed0.json")
    parser.add_argument("--proxy-model", default="gpt-4.1-nano")
    parser.add_argument("--target", type=float, default=0.9)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--M", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--proxy-cache", default="real_bargain_proxy_logprobs_gpt41nano.json")
    parser.add_argument("--output-json", default="real_bargain_first205_train_test205.json")
    parser.add_argument("--limit-proxy-calls", type=int, default=0)
    args = parser.parse_args()

    load_dotenv(Path(".env"))
    client = OpenAI()

    mini_data = json.loads(Path(args.mini_file).read_text(encoding="utf-8"))
    nano_data = json.loads(Path(args.nano_file).read_text(encoding="utf-8"))
    split = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
    train_ids = [int(x) for x in split["train_ids"] + (split.get("val_ids") or split.get("validation_ids") or [])]
    test_ids = [int(x) for x in split["test_ids"]]
    all_needed_ids = train_ids + test_ids

    proxy_cache = collect_proxy_outputs(
        nano_data,
        all_needed_ids,
        client,
        args.proxy_model,
        Path(args.proxy_cache),
        args.sleep_seconds,
        args.limit_proxy_calls,
    )
    missing = [qid for qid in all_needed_ids if qid not in proxy_cache]
    if missing:
        raise RuntimeError(f"Proxy cache still missing {len(missing)} rows; first={missing[:5]}")

    oracle_pred_by_id = {int(row["question_id"]): str(row.get("pred", "")).strip().upper() for row in mini_data}
    answer_by_id = {int(row["question_id"]): str(row.get("answer", "")).strip().upper() for row in mini_data}
    calibration = calibrate_bargain_threshold(
        train_ids,
        proxy_cache,
        oracle_pred_by_id,
        args.target,
        args.delta,
        args.M,
        args.seed,
    )
    bargain_routes = route_from_cutoff(test_ids, proxy_cache, float(calibration["score_cutoff"]))
    all_mini = {qid: "mini" for qid in test_ids}
    all_nano = {qid: "nano" for qid in test_ids}

    mini_rows = count_rows(Path(args.mini_file), "gpt-4.1-mini")
    nano_rows = count_rows(Path(args.nano_file), "gpt-4.1-nano")
    rows = [
        summarize_route("all_mini_on_testdata", test_ids, mini_rows, nano_rows, all_mini),
        summarize_route("all_saved_nano_on_testdata", test_ids, mini_rows, nano_rows, all_nano),
        summarize_route("real_bargain_on_testdata_selected_answer_cost", test_ids, mini_rows, nano_rows, bargain_routes),
    ]
    add_deltas(rows, "all_mini_on_testdata")
    cascade_cost = summarize_cascade_cost(test_ids, bargain_routes, proxy_cache, mini_rows)

    traces = []
    oracle_agree = 0
    correct = 0
    for qid in test_ids:
        route = bargain_routes[qid]
        pred = proxy_cache[qid]["pred"] if route == "nano" else oracle_pred_by_id[qid]
        selected_correct = pred == answer_by_id[qid]
        agrees_mini = pred == oracle_pred_by_id[qid]
        correct += int(selected_correct)
        oracle_agree += int(agrees_mini)
        traces.append({
            "question_id": qid,
            "route": route,
            "proxy_pred": proxy_cache[qid]["pred"],
            "oracle_mini_pred": oracle_pred_by_id[qid],
            "gold_answer": answer_by_id[qid],
            "confidence_logprob": proxy_cache[qid]["confidence_logprob"],
            "confidence_probability": proxy_cache[qid].get("confidence_probability"),
            "selected_pred": pred,
            "selected_correct": selected_correct,
            "agrees_with_mini_oracle": agrees_mini,
        })

    route_counts = Counter(bargain_routes.values())
    payload = {
        "notes": [
            "Real BARGAIN proxy confidence is collected from fresh gpt-4.1-nano answer-letter logprobs.",
            "Threshold is calibrated only on first-205 train/calibration IDs, using saved mini predictions as oracle outputs.",
            "The learned score cutoff is then applied to last-205 test IDs; test labels are used only for evaluation.",
            "selected_answer_cost uses existing saved answer costs for the selected model only; cascade_total_cost additionally counts fresh proxy calls for every test row plus mini cost on escalations.",
        ],
        "split_file": args.split_file,
        "proxy_model": args.proxy_model,
        "train_calibration_questions": len(train_ids),
        "test_questions": len(test_ids),
        "train_calibration_id_range": [min(train_ids), max(train_ids)],
        "test_id_range": [min(test_ids), max(test_ids)],
        "calibration": calibration,
        "route_counts": dict(route_counts),
        "test_accuracy_from_fresh_proxy_or_saved_mini": correct / len(test_ids),
        "test_mini_oracle_agreement_rate": oracle_agree / len(test_ids),
        "comparison_selected_answer_cost": rows,
        "cascade_cost_including_proxy_calls": cascade_cost,
        "test_traces": traces,
    }
    Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k not in {"test_traces"}}, indent=2))


if __name__ == "__main__":
    main()

