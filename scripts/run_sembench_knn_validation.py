#!/usr/bin/env python3
"""Strict 50-touch budget: random train + kNN-acquired validation (Movie).

Stricter accounting than run_sembench_online_validation.py: there, Mini
probes were charged separately from the 50 gold labels (150-250 probes per
seed).  Here the budget is 50 ROWS TOTAL for train + validation combined --
a row either gets gold + model outputs (paid) or stays raw text forever.
No probing outside the budget; the oracle-built enriched validation
(needs pool-wide disagreement knowledge) and probe-built validation
(needs ~150 extra Mini calls) are both illegal under this accounting.

  Arms (5 seeds, held-out 200, no API calls):
    t25_v25random -- train 25 random (same salt as t25_v25) + validation
                     25 random.  The naive strict-budget protocol.
    t25_v25knn    -- same train 25; validation acquired one row at a time:
                     kNN disagreement propagation on frozen-DistilBERT
                     raw-text embeddings (no Nano answer -- unpaid rows
                     have none), seeded by the train rows' agree/disagree
                     bits (free once train is paid).  Each pick is paid
                     and becomes a validation row AND a new kNN seed.
    t50_trainval  -- all 50 in training (random 50); the train rows double
                     as "validation" for epoch/threshold selection
                     (in-budget reuse; maxacc = train-searched threshold).

  Criteria: maxacc / capB / qB exactly as run_sembench_50label_budget.py.
  Prior: none.  Rates passed to train_and_score are TRAIN-estimated with
  add-one smoothing (pool gold is out of budget; only feeds prior_gap,
  unused without a prior).

  References (budget-illegal): oracle enriched t25_v25 maxacc 90.8+-0.5,
  q70 89.6+-0.4, q50 88.2+-1.1, q30 86.3+-0.7; probe-built validation
  maxacc 91.4+-0.2.  All-Nano 85.0, all-Mini 91.5.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import torch
from transformers import AutoTokenizer

from run_sembench_50label_budget import (
    build_pool_sample,
    pick_operating_points,
    train_and_score,
)
from run_sembench_prior_mo_random import select_random
from run_sembench_qa50_softprompt_router import deduplicate
from run_sembench_train100_ratio_sweep import make_orders
from train_sembench_balanced_direct_router import evaluate, load_examples, text


N_TRAIN = 25
N_VAL = 25
KNN_K = 5
ARMS = ("t25_v25random", "t25_v25knn", "t50_trainval")


def disagree_label(e):
    return 1.0 if e["nano_pred"] != e["mini_pred"] else 0.0


def flat(orders):
    return [row for rows in orders.values() for row in rows]


def train_rates(train):
    """Base rates estimated from the paid train rows only, add-one smoothed
    (train_and_score applies logit() to them unconditionally)."""
    n = len(train)
    return {
        "p_nano": (sum(r["nano_pred"] == r["gold"] for r in train) + 1) / (n + 2),
        "p_mini": (sum(r["mini_pred"] == r["gold"] for r in train) + 1) / (n + 2),
        "pool_size": n,
    }


def embed_rows(rows, args, tokenizer):
    """Frozen-DistilBERT mean-pooled embeddings of the RAW text (no Nano
    answer -- unpaid rows have none), L2-normalized, computed once."""
    from transformers import AutoModel
    encoder = AutoModel.from_pretrained(args.hf_model, local_files_only=True)
    encoder.eval()
    out = {}
    with torch.no_grad():
        for start in range(0, len(rows), 16):
            chunk = rows[start:start + 16]
            enc = tokenizer([text(r) for r in chunk], truncation=True,
                            max_length=args.max_length, padding=True,
                            return_tensors="pt")
            hidden = encoder(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (hidden * mask).sum(1) / mask.sum(1)
            emb = torch.nn.functional.normalize(emb, dim=-1)
            for r, v in zip(chunk, emb):
                out[r["review_id"]] = v
    return out


def acquire_validation_knn(train, candidates, embeddings, seed):
    """Sequential in-budget acquisition: score every unpaid candidate by a
    distance-weighted kNN vote of the labeled rows' disagree bits, pay for
    the argmax, observe its bit, repeat.  Every pick IS a validation row."""
    rng = random.Random(seed + 60013)
    labeled = list(train)
    remaining = candidates[:]
    picked = []
    for _ in range(N_VAL):
        e_c = torch.stack([embeddings[r["review_id"]] for r in remaining])
        e_l = torch.stack([embeddings[r["review_id"]] for r in labeled])
        y = torch.tensor([disagree_label(r) for r in labeled])
        sims = e_c @ e_l.T
        k = min(KNN_K, len(labeled))
        top_sims, top_idx = sims.topk(k, dim=1)
        weights = top_sims.clamp(min=0.0) + 1e-6
        scores = (weights * y[top_idx]).sum(1) / weights.sum(1)
        jitter = torch.tensor([rng.random() * 1e-4 for _ in remaining])
        i = int((scores + jitter).argmax())
        row = remaining.pop(i)
        picked.append(row)
        labeled.append(row)
    return picked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", type=Path, default=ROOT / "sembench/files/movie/data/sf_2000/Reviews.csv")
    parser.add_argument("--cache", type=Path, default=ROOT / "sembench_movie_router/model_outputs.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "sembench_movie_router/untouched_200_manifest.json")
    parser.add_argument("--hf-model", default="distilbert-base-uncased")
    parser.add_argument("--prompt-tokens", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[505, 606, 707, 808, 909])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/sembench_knn_validation.json")
    args = parser.parse_args()

    heldout_ids = set(json.loads(args.manifest.read_text(encoding="utf-8"))["review_ids"])
    cache = json.loads(args.cache.read_text(encoding="utf-8"))
    examples = deduplicate(load_examples(args.reviews, args.cache))
    heldout = [r for r in examples if r["review_id"] in heldout_ids]
    historical = [r for r in examples if r["review_id"] not in heldout_ids]
    if len(heldout) != 200 or len(historical) != 812:
        raise AssertionError("Expected 812 historical and 200 held-out rows")
    per_seed = {seed: make_orders(historical, seed) for seed in args.seeds}
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, local_files_only=True)
    print("Embedding historical pool (raw text)...", flush=True)
    embeddings = embed_rows(historical, args, tokenizer)

    result = {}
    if args.output.exists():
        result = json.loads(args.output.read_text(encoding="utf-8"))
    result["protocol"] = {
        "status": "strict 50-touch budget: train + validation share 50 rows, everything else raw",
        "accounting": "a row is either paid (gold + model outputs) or raw text; no probes "
                      "outside the 50; embeddings and router scores on raw rows are free",
        "arms": {
            "t25_v25random": "train 25 random + validation 25 random",
            "t25_v25knn": f"train 25 random + validation 25 acquired sequentially by "
                          f"{KNN_K}-NN disagreement propagation on raw-text embeddings, "
                          "seeded by the train rows' free agree/disagree bits",
            "t50_trainval": "train 50 random; train rows reused as validation for "
                            "epoch/threshold selection (no separate validation)",
        },
        "prior": "none; rates train-estimated (add-one smoothed)",
        "references_budget_illegal": {
            "oracle_enriched_t25_v25": {"maxacc": 0.908, "q70": 0.896, "q50": 0.882, "q30": 0.863},
            "probe_built_validation_maxacc": 0.914,
        },
        "answer_model_api_calls": 0,
    }
    result["baselines"] = {
        "all_nano": evaluate(heldout, [0] * len(heldout)),
        "all_mini": evaluate(heldout, [1] * len(heldout)),
        "expected_disagreements_in_random_25": round(0.079 * N_VAL, 2),
    }
    runs = result.setdefault("runs", {})

    for arm in ARMS:
        done = {int(r["seed"]) for r in runs.get(arm, [])}
        for seed in args.seeds:
            if seed in done:
                continue
            orders, standard_validation, _ = per_seed[seed]
            print(f"{arm} seed={seed}", flush=True)
            if arm == "t50_trainval":
                train = select_random(orders, seed, N_TRAIN + N_VAL)
                validation = train
            else:
                train = select_random(orders, seed, N_TRAIN)
                train_ids = {r["review_id"] for r in train}
                candidates = [r for r in flat(orders) if r["review_id"] not in train_ids]
                candidates += standard_validation
                if arm == "t25_v25random":
                    rng = random.Random(seed + 71993)
                    validation = rng.sample(candidates, N_VAL)
                else:
                    validation = acquire_validation_knn(train, candidates, embeddings, seed)
            train_ids = {r["review_id"] for r in train}
            val_ids = {r["review_id"] for r in validation}
            if arm != "t50_trainval" and train_ids & val_ids:
                raise AssertionError("Train/validation overlap")
            pool_sample = build_pool_sample(orders, train_ids | val_ids, seed)
            vs, ps, ts = train_and_score(train, validation, pool_sample, heldout,
                                         cache, seed, {}, train_rates(train), args)
            criteria = pick_operating_points(validation, vs, ps, heldout, ts)
            run = {"seed": seed,
                   "validation_informative": sum(1 for r in validation if disagree_label(r)),
                   "criteria": criteria}
            if arm == "t25_v25knn":
                run["acquisition_trace"] = [int(disagree_label(r)) for r in validation]
            for name in ("maxacc", "q50"):
                op = criteria[name]["operating_point"]
                print(f"  informative={run['validation_informative']} {name:7s} "
                      f"acc={op['accuracy']:.4f} rate={op['mini_rate']:.4f}", flush=True)
            runs.setdefault(arm, []).append(run)
            runs[arm] = sorted(runs[arm], key=lambda r: int(r["seed"]))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    result["summary"] = {}
    criteria_names = ["maxacc", "q70", "q50", "q30"]
    for arm, rs in sorted(runs.items()):
        result["summary"][arm] = {
            "validation_informative": [r["validation_informative"] for r in rs],
        }
        for name in criteria_names:
            ops = [r["criteria"][name]["operating_point"] for r in rs]
            result["summary"][arm][name] = {
                f: {"mean": mean(o[f] for o in ops), "std": pstdev(o[f] for o in ops)}
                for f in ("accuracy", "mini_rate", "mini_only_recall")
            }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
