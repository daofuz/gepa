#!/usr/bin/env python3
"""Can the validation set shrink to zero? (Computer QA)

The valsize sweep (outputs/valsize_threshold_findings.md) settled the
threshold: read it off the unlabeled pool's score quantile; labeled search
needs ~100 representative rows to catch up. That leaves the validation set
exactly one job — picking the epoch — and q70 accuracy was already flat from
10 validation rows up. This experiment asks whether even those 10 are needed.

Zero-extra-label epoch pickers, all evaluated at pool-quantile thresholds on
the same trained model per seed (paired):
  last_epoch    train the fixed 8 epochs, take the last — no selection at all
  train_as_val  pick the epoch scoring best on the 50 training rows (free:
                they are already paid for)
  cv5           5-fold CV inside the 50 training rows: retrain on 40, score
                the held 10, pick the epoch with best mean fold accuracy,
                then read that epoch off the full-50 model
  stability     unlabeled: pick the earliest epoch whose pool-sample routing
                agrees most with the previous epoch's (ties -> later)
References: val10/val20 (random labeled rows, epoch-only as in q*), and an
oracle epoch picked on the held-out test itself (selection ceiling).

Also train_maxacc: threshold searched on the training rows (zero-label
threshold search, expected to overfit) — to close that question too.

Five seeds, held-out 205, no API calls. Budget note: every arm here spends
0 labels beyond the 50 training rows except val10/val20, which are the
references being challenged.
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
    quantile_threshold,
    threshold_candidates,
    train_and_score,
)
from run_computer_qa_prior_mo_random import evaluate_at, make_orders, select_random
from run_computer_qa_knn_validation import flat, train_rates

N_TRAIN = 50
BUDGETS = (0.3, 0.5, 0.7)
CV_FOLDS = 5


def epoch_by_labeled(rows, row_scores, pool_scores, budget):
    """q*-style epoch pick: labeled rows score each epoch at that epoch's
    pool-quantile threshold."""
    best = None
    for e, scores in enumerate(row_scores):
        t = quantile_threshold(pool_scores[e], budget)
        m = evaluate_at(rows, scores, t)
        key = (m["accuracy"], -m["mini_calls"])
        if best is None or key > best[0]:
            best = (key, e)
    return best[1]


def epoch_by_stability(pool_scores, budget):
    routes = []
    for e, scores in enumerate(pool_scores):
        t = quantile_threshold(scores, budget)
        routes.append([s >= t for s in scores])
    best = None
    for e in range(1, len(routes)):
        agree = sum(a == b for a, b in zip(routes[e], routes[e - 1])) / len(routes[e])
        if best is None or agree >= best[0]:
            best = (agree, e)
    return best[1]


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
                        default=ROOT / "outputs/computer_qa_zero_validation.json")
    args = parser.parse_args()

    examples = load_examples(args.mini_file, args.nano_file)
    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    heldout_ids = {int(i) for i in split["test_ids"]}
    heldout = [e for e in examples if e.question_id in heldout_ids]
    pool = [e for e in examples if e.question_id not in heldout_ids]
    if len(heldout) != 205 or len(pool) != 205:
        raise AssertionError(f"Expected 205/205 split, got {len(heldout)}/{len(pool)}")

    result = {}
    if args.output.exists():
        result = json.loads(args.output.read_text(encoding="utf-8"))
    result["protocol"] = {
        "design": "one full-50 model per seed; epoch pickers compared paired "
                  "at pool-quantile thresholds; cv5 retrains 5 fold models "
                  "only to choose the epoch",
        "zero_label_arms": ["last_epoch", "train_as_val", "cv5", "stability",
                            "train_maxacc"],
        "reference_arms": ["val10", "val20", "oracle_epoch"],
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
        rest = [e for e in flat(orders) if e.question_id not in train_ids]
        rest += standard_validation
        rng = random.Random(seed + 90121)
        rng.shuffle(rest)
        pool_sample = build_pool_sample(orders, train_ids, seed)

        scored_rows = train + rest[:20]
        vs, ps, ts = train_and_score(train, scored_rows, pool_sample, heldout,
                                     seed, {}, train_rates(train), args)
        train_scores = [v[:N_TRAIN] for v in vs]
        ref_scores = [v[N_TRAIN:] for v in vs]

        fold_rng = random.Random(seed + 41387)
        shuffled = train[:]
        fold_rng.shuffle(shuffled)
        fold_acc = [[] for _ in range(args.epochs)]
        for f in range(CV_FOLDS):
            fold_val = shuffled[f::CV_FOLDS]
            fold_train = [e for e in shuffled if e not in fold_val]
            fvs, fps, _fts = train_and_score(
                fold_train, fold_val, pool_sample, heldout[:1],
                seed * 10 + f, {}, train_rates(fold_train), args)
            for e in range(args.epochs):
                t = quantile_threshold(fps[e], 0.5)
                fold_acc[e].append(evaluate_at(fold_val, fvs[e], t)["accuracy"])
        cv_epoch = max(range(args.epochs), key=lambda e: (mean(fold_acc[e]), e))

        epochs = {
            "last_epoch": args.epochs - 1,
            "train_as_val": epoch_by_labeled(train, train_scores, ps, 0.5),
            "cv5": cv_epoch,
            "stability": epoch_by_stability(ps, 0.5),
            "val10": epoch_by_labeled(rest[:10], [r[:10] for r in ref_scores],
                                      ps, 0.5),
            "val20": epoch_by_labeled(rest[:20], ref_scores, ps, 0.5),
            "oracle_epoch": epoch_by_labeled(heldout, ts, ps, 0.5),
        }

        row = {"seed": seed, "chosen_epochs": {a: e + 1 for a, e in epochs.items()},
               "arms": {}}
        for arm, e in epochs.items():
            entry = {}
            for budget in BUDGETS:
                t = quantile_threshold(ps[e], budget)
                entry[f"q{int(budget * 100)}"] = evaluate_at(heldout, ts[e], t)
            row["arms"][arm] = entry

        # zero-label threshold search: epoch and threshold both from train rows
        best = None
        for e, scores in enumerate(train_scores):
            for t in threshold_candidates(scores):
                m = evaluate_at(train, scores, t)
                k = (m["accuracy"], -m["mini_calls"])
                if best is None or k > best[0]:
                    best = (k, e, t)
        row["arms"]["train_maxacc"] = {
            "maxacc": evaluate_at(heldout, ts[best[1]], best[2])}
        row["chosen_epochs"]["train_maxacc"] = best[1] + 1

        for arm in ("last_epoch", "train_as_val", "cv5", "stability",
                    "val10", "val20", "oracle_epoch"):
            op = row["arms"][arm]["q70"]
            print(f"  {arm:12s} epoch={row['chosen_epochs'][arm]} "
                  f"q70 acc={op['accuracy']:.4f} rate={op['mini_rate']:.4f}",
                  flush=True)
        op = row["arms"]["train_maxacc"]["maxacc"]
        print(f"  train_maxacc epoch={row['chosen_epochs']['train_maxacc']} "
              f"acc={op['accuracy']:.4f} rate={op['mini_rate']:.4f}", flush=True)

        runs[key] = row
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    arm_names = ["last_epoch", "train_as_val", "cv5", "stability",
                 "val10", "val20", "oracle_epoch", "train_maxacc"]
    summary = {}
    for arm in arm_names:
        summary[arm] = {"chosen_epochs": [r["chosen_epochs"][arm]
                                          for r in runs.values()]}
        crits = ("maxacc",) if arm == "train_maxacc" else tuple(
            f"q{int(b * 100)}" for b in BUDGETS)
        for crit in crits:
            ops = [r["arms"][arm][crit] for r in runs.values()]
            summary[arm][crit] = {
                f: {"mean": mean(o[f] for o in ops), "std": pstdev(o[f] for o in ops)}
                for f in ("accuracy", "mini_rate", "mini_only_recall")
            }
    result["summary"] = summary
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
