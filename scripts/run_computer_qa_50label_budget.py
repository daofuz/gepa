#!/usr/bin/env python3
"""Computer QA replication of the 50-gold-label + cost-controlled protocol.

Mirrors scripts/run_sembench_50label_budget.py.  Label splits (50 gold):
t25_v25 (train 25 random + validation 18 disagreements/7 agreements) and
t30_v20 (train 30 random + validation 15/5).  Criteria: maxacc (status quo),
capB (validation mini_rate <= B), qB (threshold = pool-sample score quantile,
gold never read), B in {0.3, 0.5, 0.7}.  Five seeds, held-out 205, no
answer-model API calls.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from optimize_ollama_router_gepa import load_examples, outcome_bucket
from run_computer_qa_prior_mo_random import (
    correctness_probabilities,
    evaluate_at,
    frontier,
    loader,
    logit,
    make_orders,
    pool_base_rates,
    select_random,
)
from run_sembench_twohead_correctness_router import TwoHeadCorrectnessRouter


PRIORS = [
    ("none", {}),
    ("mean_prob_l1", {"mean_prob": 1.0}),
]
SPLITS = [
    ("t25_v25", 25, 18, 7),
    ("t30_v20", 30, 15, 5),
]
BUDGETS = [0.3, 0.5, 0.7]
POOL_SAMPLE = 100


def flat(orders):
    return [e for rows in orders.values() for e in rows]


def build_validation(orders, standard_validation, train_ids, seed, n_dis, n_agr):
    candidates = [e for e in flat(orders) if e.question_id not in train_ids]
    candidates += standard_validation
    disagree = [e for e in candidates if e.nano_pred != e.mini_pred]
    agree = [e for e in candidates if e.nano_pred == e.mini_pred]
    rng = random.Random(seed + 51797)
    rows = rng.sample(disagree, min(n_dis, len(disagree)))
    rows += rng.sample(agree, min(n_agr, len(agree)))
    rng.shuffle(rows)
    return rows


def build_pool_sample(orders, used_ids, seed, n=POOL_SAMPLE):
    candidates = [e for e in flat(orders) if e.question_id not in used_ids]
    rng = random.Random(seed + 86243)
    return rng.sample(candidates, min(n, len(candidates)))


def threshold_candidates(scores):
    unique = sorted(set(scores))
    cands = [unique[0] - 1.0, unique[-1] + 1.0, *unique]
    cands += [(a + b) / 2.0 for a, b in zip(unique, unique[1:])]
    return sorted(set(cands))


def quantile_threshold(scores, budget):
    ordered = sorted(scores, reverse=True)
    k = round(budget * len(ordered))
    if k <= 0:
        return ordered[0] + 1.0
    if k >= len(ordered):
        return ordered[-1] - 1.0
    return (ordered[k - 1] + ordered[k]) / 2.0


def train_and_score(train, validation, pool_sample, test, seed, prior_cfg, rates, args):
    random.seed(seed)
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, local_files_only=True)
    model = TwoHeadCorrectnessRouter(args.hf_model, args.prompt_tokens)
    p_nano, p_mini = rates["p_nano"], rates["p_mini"]
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    tl = loader(train, tokenizer, args.max_length, args.batch_size, True)
    vl = loader(validation, tokenizer, args.max_length, args.batch_size, False)
    pl = loader(pool_sample, tokenizer, args.max_length, args.batch_size, False)
    xl = loader(test, tokenizer, args.max_length, args.batch_size, False)
    costs = {
        "nano": mean(e.nano_cost_usd for e in train),
        "mini": mean(e.mini_cost_usd for e in train),
    }
    inv = torch.tensor([1.0 / costs["nano"], 1.0 / costs["mini"]], dtype=torch.float)
    prior_gap = logit(p_mini) - logit(p_nano)

    val_scores, pool_scores, test_scores = [], [], []
    for _epoch in range(1, args.epochs + 1):
        model.train()
        for batch in tl:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], batch["attention_mask"])
            raw = F.binary_cross_entropy_with_logits(logits, batch["labels"], reduction="none")
            loss = (raw * inv).sum() / (len(batch["labels"]) * inv.sum())
            if "bias_map" in prior_cfg:
                bias = model.correctness_heads.bias
                loss = loss + 0.5 * prior_cfg["bias_map"] * (bias[1] - bias[0] - prior_gap).pow(2)
            if "mean_prob" in prior_cfg:
                q = torch.sigmoid(logits)
                loss = loss + prior_cfg["mean_prob"] * (
                    (q[:, 0].mean() - p_nano).pow(2) + (q[:, 1].mean() - p_mini).pow(2)
                )
            loss.backward()
            optimizer.step()
        for scores, dl in ((val_scores, vl), (pool_scores, pl), (test_scores, xl)):
            qn, qm = correctness_probabilities(model, dl)
            scores.append([m - n for n, m in zip(qn, qm)])
    return val_scores, pool_scores, test_scores


def pick_operating_points(validation, val_scores, pool_scores, test, test_scores):
    results = {}

    def val_key(scores, t):
        m = evaluate_at(validation, scores, t)
        return (m["accuracy"], -m["mini_calls"]), m

    def finish(name, epoch, t):
        op = evaluate_at(test, test_scores[epoch], t)
        results[name] = {
            "best_epoch": epoch + 1, "threshold": t,
            "operating_point": op,
            "frontier": frontier(test, test_scores[epoch]),
        }

    best = None
    for e, vs in enumerate(val_scores):
        for t in threshold_candidates(vs):
            key, _ = val_key(vs, t)
            if best is None or key > best[0]:
                best = (key, e, t)
    finish("maxacc", best[1], best[2])

    for budget in BUDGETS:
        best = None
        for e, vs in enumerate(val_scores):
            for t in threshold_candidates(vs):
                key, m = val_key(vs, t)
                if m["mini_rate"] > budget:
                    continue
                if best is None or key > best[0]:
                    best = (key, e, t)
        finish(f"cap{int(budget * 100)}", best[1], best[2])

    for budget in BUDGETS:
        best = None
        for e, vs in enumerate(val_scores):
            t = quantile_threshold(pool_scores[e], budget)
            key, _ = val_key(vs, t)
            if best is None or key > best[0]:
                best = (key, e, t)
        finish(f"q{int(budget * 100)}", best[1], best[2])
    return results


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
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/computer_qa_50label_budget.json")
    args = parser.parse_args()

    examples = load_examples(args.mini_file, args.nano_file)
    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    heldout_ids = {int(i) for i in split["test_ids"]}
    heldout = [e for e in examples if e.question_id in heldout_ids]
    pool = [e for e in examples if e.question_id not in heldout_ids]
    if len(heldout) != 205 or len(pool) != 205:
        raise AssertionError(f"Expected 205/205 split, got {len(heldout)}/{len(pool)}")
    per_seed = {seed: make_orders(pool, seed) for seed in args.seeds}
    rates_by_seed = {seed: pool_base_rates(per_seed[seed][0]) for seed in args.seeds}

    result = {}
    if args.output.exists():
        result = json.loads(args.output.read_text(encoding="utf-8"))
    result["protocol"] = {
        "status": "post-hoc held-out diagnostic on Computer QA: 50-gold-label + cost-controlled selection",
        "splits": {name: f"train {t} random + validation {d} disagreements/{a} agreements"
                   for name, t, d, a in SPLITS},
        "criteria": {
            "maxacc": "validation-max-accuracy (status quo)",
            "capB": "validation-max-accuracy s.t. validation mini_rate <= B",
            "qB": f"threshold = (1-B) quantile of scores on {POOL_SAMPLE} unlabeled pool rows; "
                  "epoch by validation accuracy under that threshold",
        },
        "gold_total": 50,
        "answer_model_api_calls": 0,
    }
    result["baselines"] = {
        "all_nano": evaluate_at(heldout, [0.0] * len(heldout), 1.0),
        "all_mini": evaluate_at(heldout, [1.0] * len(heldout), 0.0),
    }
    runs = result.setdefault("runs", {})

    for split_name, n_train, n_dis, n_agr in SPLITS:
        for prior_name, prior_cfg in PRIORS:
            key = f"{split_name}__{prior_name}"
            done = {int(r["seed"]) for r in runs.get(key, [])}
            for seed in args.seeds:
                if seed in done:
                    continue
                orders, standard_validation = per_seed[seed]
                train = select_random(orders, seed, n_train)
                train_ids = {e.question_id for e in train}
                validation = build_validation(orders, standard_validation, train_ids,
                                              seed, n_dis, n_agr)
                val_ids = {e.question_id for e in validation}
                if train_ids & val_ids:
                    raise AssertionError("Train/validation overlap")
                pool_sample = build_pool_sample(orders, train_ids | val_ids, seed)
                print(f"{key} seed={seed} train={len(train)} val={len(validation)} "
                      f"(informative={sum(1 for e in validation if e.nano_pred != e.mini_pred)})",
                      flush=True)
                vs, ps, ts = train_and_score(train, validation, pool_sample, heldout,
                                             seed, prior_cfg, rates_by_seed[seed], args)
                criteria = pick_operating_points(validation, vs, ps, heldout, ts)
                run = {"seed": seed,
                       "train_counts": dict(Counter(outcome_bucket(e) for e in train)),
                       "validation_size": len(validation),
                       "criteria": criteria}
                for name in ("maxacc", "cap50", "q50"):
                    op = criteria[name]["operating_point"]
                    print(f"  {name:7s} acc={op['accuracy']:.4f} rate={op['mini_rate']:.4f}",
                          flush=True)
                runs.setdefault(key, []).append(run)
                runs[key] = sorted(runs[key], key=lambda r: int(r["seed"]))
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    result["summary"] = {}
    criteria_names = ["maxacc"] + [f"cap{int(b*100)}" for b in BUDGETS] + [f"q{int(b*100)}" for b in BUDGETS]
    for key, rs in sorted(runs.items()):
        if not rs:
            continue
        result["summary"][key] = {}
        for name in criteria_names:
            ops = [r["criteria"][name]["operating_point"] for r in rs]
            result["summary"][key][name] = {
                f: {"mean": mean(o[f] for o in ops), "std": pstdev(o[f] for o in ops)}
                for f in ("accuracy", "mini_rate", "mini_saving_vs_all_mini", "mini_only_recall")
            }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
