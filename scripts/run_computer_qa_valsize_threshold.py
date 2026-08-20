#!/usr/bin/env python3
"""How many validation labels does threshold search need? (Computer QA)

Follow-up to outputs/validation_acquisition_findings.md, which showed
validation-searched thresholds (maxacc/cap*) losing 4-6pt to pool-quantile
rules with 20 validation rows.  Open question: is that a verdict on
validation-based thresholds, or just on 20 rows?  The quantile rule adapts
to the pool's score distribution but cannot target accuracy; a labeled
validation set can, if it is large enough for the search not to overfit.

Design: per seed, train once (50 random rows, two-head router, unchanged
protocol), score all ~155 remaining pool rows once, then re-run operating
point selection on nested validation prefixes of size 10/20/50/100/all.
Two orderings of the same candidates: uniform random, and sequential kNN
disagreement propagation (the "pick hard questions for validation" idea) --
so at every size the kNN-acquired validation set is compared paired against
a random one.  Every arm sees the same model and the same score vectors;
the only variables are how many labeled rows the selector may look at and
which rows they are.  An oracle arm (select on the held-out test itself)
bounds what any selector could do.

Diagnostic, not budget-legal: validation rows beyond the 50-row training
budget are deliberately unpaid-for here; the question is statistical
sufficiency, not protocol legality.  Five seeds, held-out 205, no API calls.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from optimize_ollama_router_gepa import load_examples
from run_computer_qa_50label_budget import (
    build_pool_sample,
    pick_operating_points,
    train_and_score,
)
from run_computer_qa_prior_mo_random import make_orders, select_random
from run_computer_qa_knn_validation import (
    acquire_validation_knn,
    disagree_label,
    embed_rows,
    flat,
    train_rates,
)

N_TRAIN = 50
VAL_SIZES = (10, 20, 50, 100, None)  # None = all remaining pool rows
CRITERIA = ("maxacc", "cap50", "q50", "q70")


def strip_frontiers(criteria: dict) -> dict:
    return {
        name: {k: v for k, v in entry.items() if k != "frontier"}
        for name, entry in criteria.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mini-file", type=Path, default=ROOT / "computer science_result_mini.json")
    parser.add_argument("--nano-file", type=Path, default=ROOT / "computer science_result_nano.json")
    parser.add_argument("--split-file", type=Path,
                        default=ROOT / "routing_split_softprompt_threshold_balanced16.json")
    parser.add_argument("--hf-model", default="distilbert-base-uncased")
    parser.add_argument("--prompt-tokens", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[505, 606, 707, 808, 909])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/computer_qa_valsize_threshold.json")
    args = parser.parse_args()

    examples = load_examples(args.mini_file, args.nano_file)
    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    heldout_ids = {int(i) for i in split["test_ids"]}
    heldout = [e for e in examples if e.question_id in heldout_ids]
    pool = [e for e in examples if e.question_id not in heldout_ids]
    if len(heldout) != 205 or len(pool) != 205:
        raise AssertionError(f"Expected 205/205 split, got {len(heldout)}/{len(pool)}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, local_files_only=True)
    print("Embedding pool (raw text)...", flush=True)
    embeddings = embed_rows(pool, args, tokenizer)

    result = {}
    if args.output.exists():
        result = json.loads(args.output.read_text(encoding="utf-8"))
    result["protocol"] = {
        "design": "one model per seed (50 random train rows); operating point "
                  "selection re-run on nested random validation prefixes; "
                  "oracle selects on the held-out test itself",
        "val_sizes": [n if n else "all_pool_rest" for n in VAL_SIZES],
        "status": "diagnostic of validation-set sufficiency; rows beyond the "
                  "training 50 are not budget-accounted",
        "answer_model_api_calls": 0,
    }
    runs = result.setdefault("runs", {})

    for seed in args.seeds:
        key = str(seed)
        if key in runs:
            continue
        print(f"seed={seed}", flush=True)
        orders, standard_validation = make_orders(pool, seed)
        train = select_random(orders, seed, N_TRAIN)
        train_ids = {e.question_id for e in train}
        candidates = [e for e in flat(orders) if e.question_id not in train_ids]
        candidates += standard_validation
        rng = random.Random(seed + 33191)
        rng.shuffle(candidates)
        pool_sample = build_pool_sample(orders, train_ids, seed)
        vs, ps, ts = train_and_score(train, candidates, pool_sample, heldout,
                                     seed, {}, train_rates(train), args)
        knn_order = acquire_validation_knn(train, candidates[:], embeddings,
                                           seed, len(candidates))
        pos = {e.question_id: i for i, e in enumerate(candidates)}
        row = {"seed": seed, "n_candidates": len(candidates), "arms": {}}
        for size in VAL_SIZES:
            n = len(candidates) if size is None else size
            for order_name, ordered in (("random", candidates), ("knn", knn_order)):
                if size is None and order_name == "knn":
                    continue  # identical set at full size
                val = ordered[:n]
                idx = [pos[e.question_id] for e in val]
                criteria = pick_operating_points(
                    val, [[v[i] for i in idx] for v in vs], ps, heldout, ts)
                arm = (f"val{n}" if size is not None else f"val_all{n}")
                arm += "" if order_name == "random" else "_knn"
                entry = strip_frontiers(criteria)
                entry["validation_informative"] = int(
                    sum(disagree_label(e) for e in val))
                row["arms"][arm] = entry
                op = criteria["maxacc"]["operating_point"]
                print(f"  {arm:12s} n={n:3d} hard={entry['validation_informative']:3d} "
                      f"maxacc acc={op['accuracy']:.4f} rate={op['mini_rate']:.4f} "
                      f"q70acc={criteria['q70']['operating_point']['accuracy']:.4f}",
                      flush=True)
        oracle = pick_operating_points(heldout, ts, ps, heldout, ts)
        row["arms"]["oracle"] = strip_frontiers({"maxacc": oracle["maxacc"]})
        print(f"  oracle maxacc acc="
              f"{oracle['maxacc']['operating_point']['accuracy']:.4f}", flush=True)
        runs[key] = row
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    arm_names = sorted({a for r in runs.values() for a in r["arms"]},
                       key=lambda a: (a != "oracle", len(a), a))
    summary = {}
    for arm in arm_names:
        summary[arm] = {}
        for crit in CRITERIA + ("maxacc",):
            ops = [r["arms"][arm][crit]["operating_point"]
                   for r in runs.values() if crit in r["arms"][arm]]
            if not ops:
                continue
            summary[arm][crit] = {
                f: {"mean": mean(o[f] for o in ops), "std": pstdev(o[f] for o in ops)}
                for f in ("accuracy", "mini_rate", "mini_only_recall")
            }
    result["summary"] = summary
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
