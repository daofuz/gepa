#!/usr/bin/env python3
"""Strict touch budget: random train + kNN-acquired validation (Computer QA).

Mirrors scripts/run_sembench_knn_validation.py.  Budget = --n-train + --n-val
rows total (default 25+25); a row is either paid (gold + model outputs) or
stays raw text.  Embeddings use the raw question text (no Nano answer --
unpaid rows have none).  Arms, named after the budget: tT_vVrandom, tT_vVknn
(sequential kNN disagreement propagation seeded by the train rows' free
agree/disagree bits, every pick paid and kept as a validation row), and
t(T+V)_trainval (whole budget in training, train reused as validation for
epoch/threshold selection).

References (budget-illegal): oracle enriched t25_v25 maxacc 78.6+-1.6,
q70 77.7+-1.8, q50 75.9+-1.8, q30 74.0+-1.9.  All-Nano 71.2, all-Mini 80.5;
pool disagreement rate ~22% (expected ~5.5 in a random 25).
Five seeds, held-out 205, no API calls.
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

import torch
from transformers import AutoTokenizer

from optimize_ollama_router_gepa import load_examples
from run_computer_qa_50label_budget import (
    build_pool_sample,
    pick_operating_points,
    train_and_score,
)
from run_computer_qa_prior_mo_random import (
    evaluate_at,
    make_orders,
    mini_ok,
    nano_ok,
    select_random,
)
from train_softprompt_router import example_text


KNN_K = 5


def arm_names(n_train, n_val):
    """Arm keys carry the budget so runs at different budgets never collide."""
    return (
        f"t{n_train}_v{n_val}random",
        f"t{n_train}_v{n_val}knn",
        f"t{n_train + n_val}_trainval",
    )


def raw_text(e):
    return example_text(e, include_nano_answer=False, nano_response_max_chars=0)


def disagree_label(e):
    return 1.0 if e.nano_pred != e.mini_pred else 0.0


def flat(orders):
    return [e for rows in orders.values() for e in rows]


def train_rates(train):
    """Base rates from the paid train rows only, add-one smoothed
    (train_and_score applies logit() to them unconditionally)."""
    n = len(train)
    return {
        "p_nano": (sum(nano_ok(e) for e in train) + 1) / (n + 2),
        "p_mini": (sum(mini_ok(e) for e in train) + 1) / (n + 2),
        "pool_size": n,
    }


def embed_rows(rows, args, tokenizer):
    """Frozen-DistilBERT mean-pooled embeddings of the raw question text,
    L2-normalized, computed once."""
    from transformers import AutoModel
    encoder = AutoModel.from_pretrained(args.hf_model, local_files_only=True)
    encoder.eval()
    out = {}
    with torch.no_grad():
        for start in range(0, len(rows), 16):
            chunk = rows[start:start + 16]
            enc = tokenizer([raw_text(e) for e in chunk], truncation=True,
                            max_length=args.max_length, padding=True,
                            return_tensors="pt")
            hidden = encoder(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (hidden * mask).sum(1) / mask.sum(1)
            emb = torch.nn.functional.normalize(emb, dim=-1)
            for e, v in zip(chunk, emb):
                out[e.question_id] = v
    return out


def acquire_validation_knn(train, candidates, embeddings, seed, n_val):
    """Sequential in-budget acquisition by kNN disagreement propagation;
    every pick is paid and kept as a validation row + new seed."""
    rng = random.Random(seed + 60013)
    labeled = list(train)
    remaining = candidates[:]
    picked = []
    for _ in range(n_val):
        e_c = torch.stack([embeddings[e.question_id] for e in remaining])
        e_l = torch.stack([embeddings[e.question_id] for e in labeled])
        y = torch.tensor([disagree_label(e) for e in labeled])
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
    parser.add_argument("--mini-file", type=Path, default=ROOT / "computer science_result_mini.json")
    parser.add_argument("--nano-file", type=Path, default=ROOT / "computer science_result_nano.json")
    parser.add_argument("--split-file", type=Path,
                        default=ROOT / "routing_split_softprompt_threshold_balanced16.json")
    parser.add_argument("--hf-model", default="distilbert-base-uncased")
    parser.add_argument("--n-train", type=int, default=25)
    parser.add_argument("--n-val", type=int, default=25)
    parser.add_argument("--prompt-tokens", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[505, 606, 707, 808, 909])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/computer_qa_knn_validation.json")
    args = parser.parse_args()

    examples = load_examples(args.mini_file, args.nano_file)
    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    heldout_ids = {int(i) for i in split["test_ids"]}
    heldout = [e for e in examples if e.question_id in heldout_ids]
    pool = [e for e in examples if e.question_id not in heldout_ids]
    if len(heldout) != 205 or len(pool) != 205:
        raise AssertionError(f"Expected 205/205 split, got {len(heldout)}/{len(pool)}")
    per_seed = {seed: make_orders(pool, seed) for seed in args.seeds}
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, local_files_only=True)
    print("Embedding pool (raw text)...", flush=True)
    embeddings = embed_rows(pool, args, tokenizer)

    result = {}
    if args.output.exists():
        result = json.loads(args.output.read_text(encoding="utf-8"))
    arm_random, arm_knn, arm_trainval = arm_names(args.n_train, args.n_val)
    budget = args.n_train + args.n_val
    result["protocol"] = {
        "status": f"strict {budget}-touch budget: train + validation share {budget} rows, "
                  "everything else raw",
        "accounting": "a row is either paid (gold + model outputs) or raw text; no probes "
                      f"outside the {budget}; embeddings and router scores on raw rows are free",
        "arms": {
            arm_random: f"train {args.n_train} random + validation {args.n_val} random",
            arm_knn: f"train {args.n_train} random + validation {args.n_val} acquired "
                     f"sequentially by {KNN_K}-NN disagreement propagation on raw-text "
                     "embeddings, seeded by the train rows' free agree/disagree bits",
            arm_trainval: f"train {budget} random; train rows reused as validation for "
                          "epoch/threshold selection (no separate validation)",
        },
        "prior": "none; rates train-estimated (add-one smoothed)",
        "references_budget_illegal": {
            "oracle_enriched_t25_v25": {"maxacc": 0.786, "q70": 0.777, "q50": 0.759, "q30": 0.740},
        },
        "answer_model_api_calls": 0,
    }
    result["baselines"] = {
        "all_nano": evaluate_at(heldout, [0.0] * len(heldout), 1.0),
        "all_mini": evaluate_at(heldout, [1.0] * len(heldout), 0.0),
        "expected_disagreements_in_random_validation": round(0.22 * args.n_val, 2),
    }
    runs = result.setdefault("runs", {})

    for arm in (arm_random, arm_knn, arm_trainval):
        done = {int(r["seed"]) for r in runs.get(arm, [])}
        for seed in args.seeds:
            if seed in done:
                continue
            orders, standard_validation = per_seed[seed]
            print(f"{arm} seed={seed}", flush=True)
            if arm == arm_trainval:
                train = select_random(orders, seed, budget)
                validation = train
            else:
                train = select_random(orders, seed, args.n_train)
                train_ids = {e.question_id for e in train}
                candidates = [e for e in flat(orders) if e.question_id not in train_ids]
                candidates += standard_validation
                if arm == arm_random:
                    rng = random.Random(seed + 71993)
                    validation = rng.sample(candidates, args.n_val)
                else:
                    validation = acquire_validation_knn(
                        train, candidates, embeddings, seed, args.n_val
                    )
            train_ids = {e.question_id for e in train}
            val_ids = {e.question_id for e in validation}
            if arm != arm_trainval and train_ids & val_ids:
                raise AssertionError("Train/validation overlap")
            pool_sample = build_pool_sample(orders, train_ids | val_ids, seed)
            vs, ps, ts = train_and_score(train, validation, pool_sample, heldout,
                                         seed, {}, train_rates(train), args)
            criteria = pick_operating_points(validation, vs, ps, heldout, ts)
            run = {"seed": seed,
                   "validation_informative": sum(1 for e in validation if disagree_label(e)),
                   "criteria": criteria}
            if arm == arm_knn:
                run["acquisition_trace"] = [int(disagree_label(e)) for e in validation]
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
