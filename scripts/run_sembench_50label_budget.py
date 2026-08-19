#!/usr/bin/env python3
"""A 50-gold-label protocol (train + validation) with cost-controlled selection.

Follow-up to run_sembench_disagree_validation.py, which showed (a) a
disagreement-enriched validation stabilizes (epoch, threshold) selection and
(b) the validation-max-accuracy criterion then picks expensive operating
points (0.81-0.92 Mini rate).  This script asks whether BOTH the total gold
budget and the serving cost can come down together:

  Label splits (50 gold total; training labels are uniform random, following
  the scaling result that training does not benefit from curated
  disagreements):
    t25_v25 -- train 25 random, validation 18 disagreements + 7 agreements
    t30_v20 -- train 30 random, validation 15 disagreements + 5 agreements

  Operating-point criteria (post-processed from the same per-epoch scores,
  exactly paired):
    maxacc  -- validation-max-accuracy threshold+epoch (status quo)
    capB    -- validation-max-accuracy restricted to thresholds whose
               VALIDATION Mini rate <= B, for B in {0.3, 0.5, 0.7}
    qB      -- threshold = the (1-B) quantile of router scores on a 200-row
               UNLABELED pool sample (no gold read), so the realized Mini
               rate is pinned near B by construction; the epoch is chosen by
               validation accuracy under that threshold

Reference points: the 80-90-label protocols of
outputs/sembench_disagree_validation.json.  Five seeds, held-out 200, no
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
sys.path.insert(0, str(ROOT / "scripts"))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

import run_sembench_twohead_correctness_router as base
from run_sembench_accuracy_targeted_router import evaluate_scores
from run_sembench_mo_importance import frontier
from run_sembench_prior_mo_random import pool_base_rates, select_random
from run_sembench_qa50_softprompt_router import deduplicate
from run_sembench_softprompt_mini_prior_ablation import expected_costs
from run_sembench_train100_ratio_sweep import make_orders
from train_sembench_balanced_direct_router import evaluate, load_examples


PRIORS = [
    ("none", {}),
    ("mean_prob_l1", {"mean_prob": 1.0}),
]
SPLITS = [
    ("t25_v25", 25, 18, 7),
    ("t30_v20", 30, 15, 5),
]
BUDGETS = [0.3, 0.5, 0.7]
POOL_SAMPLE = 200


def logit(p):
    import math
    return math.log(p / (1.0 - p))


def flat(orders):
    return [row for rows in orders.values() for row in rows]


def build_validation(orders, standard_validation, train_ids, seed, n_dis, n_agr):
    candidates = [r for r in flat(orders) if r["review_id"] not in train_ids]
    candidates += standard_validation
    disagree = [r for r in candidates if r["nano_pred"] != r["mini_pred"]]
    agree = [r for r in candidates if r["nano_pred"] == r["mini_pred"]]
    rng = random.Random(seed + 51797)
    rows = rng.sample(disagree, min(n_dis, len(disagree)))
    rows += rng.sample(agree, min(n_agr, len(agree)))
    rng.shuffle(rows)
    return rows


def build_pool_sample(orders, used_ids, seed, n=POOL_SAMPLE):
    """Unlabeled rows for quantile thresholds: scores only, gold never read."""
    candidates = [r for r in flat(orders) if r["review_id"] not in used_ids]
    rng = random.Random(seed + 86243)
    return rng.sample(candidates, min(n, len(candidates)))


def threshold_candidates(scores):
    unique = sorted(set(scores))
    cands = [unique[0] - 1.0, unique[-1] + 1.0, *unique]
    cands += [(a + b) / 2.0 for a, b in zip(unique, unique[1:])]
    return sorted(set(cands))


def quantile_threshold(scores, budget):
    """Threshold routing ~budget fraction of rows (the highest scores) to Mini."""
    ordered = sorted(scores, reverse=True)
    k = round(budget * len(ordered))
    if k <= 0:
        return ordered[0] + 1.0
    if k >= len(ordered):
        return ordered[-1] - 1.0
    return (ordered[k - 1] + ordered[k]) / 2.0


def train_and_score(train, validation, pool_sample, test, cache, seed, prior_cfg, rates, args):
    random.seed(seed)
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, local_files_only=True)
    model = base.TwoHeadCorrectnessRouter(args.hf_model, args.prompt_tokens)
    p_nano, p_mini = rates["p_nano"], rates["p_mini"]
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    tl = base.loader(train, tokenizer, args.max_length, args.batch_size, True)
    vl = base.loader(validation, tokenizer, args.max_length, args.batch_size, False)
    pl = base.loader(pool_sample, tokenizer, args.max_length, args.batch_size, False)
    xl = base.loader(test, tokenizer, args.max_length, args.batch_size, False)
    costs = expected_costs(train, cache)
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
            qn, qm = base.correctness_probabilities(model, dl)
            scores.append(base.score(qn, qm))
    return val_scores, pool_scores, test_scores


def pick_operating_points(validation, val_scores, pool_scores, test, test_scores):
    """All criteria from the same per-epoch scores; returns per-criterion results."""
    results = {}

    def val_key(scores, t):
        m = evaluate_scores(validation, scores, t)
        return (m["selected_accuracy"], -m["mini_calls"]), m

    def finish(name, epoch, t):
        op = evaluate_scores(test, test_scores[epoch], t)
        results[name] = {
            "best_epoch": epoch + 1, "threshold": t,
            "operating_point": {
                "accuracy": op["selected_accuracy"],
                "mini_rate": op["mini_rate"],
                "mini_saving_vs_all_mini": op["mini_saving_vs_all_mini"],
                "mini_only_recall": op["mini_only_recall"],
            },
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
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/sembench_50label_budget.json")
    args = parser.parse_args()

    heldout_ids = set(json.loads(args.manifest.read_text(encoding="utf-8"))["review_ids"])
    cache = json.loads(args.cache.read_text(encoding="utf-8"))
    examples = deduplicate(load_examples(args.reviews, args.cache))
    heldout = [r for r in examples if r["review_id"] in heldout_ids]
    historical = [r for r in examples if r["review_id"] not in heldout_ids]
    if len(heldout) != 200 or len(historical) != 812:
        raise AssertionError("Expected 812 historical and 200 held-out rows")
    per_seed = {seed: make_orders(historical, seed) for seed in args.seeds}
    rates_by_seed = {seed: pool_base_rates(per_seed[seed][0]) for seed in args.seeds}

    result = {}
    if args.output.exists():
        result = json.loads(args.output.read_text(encoding="utf-8"))
    result["protocol"] = {
        "status": "post-hoc held-out diagnostic: 50-gold-label protocol with cost-controlled selection",
        "splits": {name: f"train {t} random + validation {d} disagreements/{a} agreements"
                   for name, t, d, a in SPLITS},
        "criteria": {
            "maxacc": "validation-max-accuracy (status quo)",
            "capB": "validation-max-accuracy s.t. validation mini_rate <= B",
            "qB": f"threshold = (1-B) quantile of scores on {POOL_SAMPLE} unlabeled pool rows "
                  "(gold never read); epoch by validation accuracy under that threshold",
        },
        "gold_total": 50,
        "answer_model_api_calls": 0,
    }
    result["baselines"] = {
        "all_nano": evaluate(heldout, [0] * len(heldout)),
        "all_mini": evaluate(heldout, [1] * len(heldout)),
    }
    runs = result.setdefault("runs", {})

    for split_name, n_train, n_dis, n_agr in SPLITS:
        for prior_name, prior_cfg in PRIORS:
            key = f"{split_name}__{prior_name}"
            done = {int(r["seed"]) for r in runs.get(key, [])}
            for seed in args.seeds:
                if seed in done:
                    continue
                orders, standard_validation, _ = per_seed[seed]
                train = select_random(orders, seed, n_train)
                train_ids = {r["review_id"] for r in train}
                validation = build_validation(orders, standard_validation, train_ids,
                                              seed, n_dis, n_agr)
                val_ids = {r["review_id"] for r in validation}
                if train_ids & val_ids:
                    raise AssertionError("Train/validation overlap")
                pool_sample = build_pool_sample(orders, train_ids | val_ids, seed)
                print(f"{key} seed={seed} train={len(train)} val={len(validation)} "
                      f"(informative={sum(1 for r in validation if r['nano_pred'] != r['mini_pred'])})",
                      flush=True)
                vs, ps, ts = train_and_score(train, validation, pool_sample, heldout,
                                             cache, seed, prior_cfg, rates_by_seed[seed], args)
                criteria = pick_operating_points(validation, vs, ps, heldout, ts)
                run = {"seed": seed,
                       "train_counts": dict(Counter(r["bucket"] for r in train)),
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
