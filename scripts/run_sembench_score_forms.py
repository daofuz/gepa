#!/usr/bin/env python3
"""Score read-out forms for the two-head router (Movie).

With calibrated heads, ranking by q_mini - q_nano is budget-optimal (it IS
the expected accuracy gain of escalation), so alternative forms can only win
through robustness to head noise/miscalibration -- which is plausible here:
25 training rows, and the Mini head has almost nothing to learn (Mini is
right 91% of the time), so its fluctuations may be pure noise inside the
difference.

Protocol = the minimal recipe: 25 random gold rows (same salt as t25_v25),
fixed 8 epochs, NO validation; the two heads' raw probabilities on the
held-out 200 are read out at the last epoch and re-scored under five forms:

  diff        q_mini - q_nano                  (baseline; Bayes-optimal if calibrated)
  neg_nano    -q_nano                          (ignore the Mini head entirely)
  logit_diff  logit(q_mini) - logit(q_nano)    (tail-emphasizing)
  product     q_mini * (1 - q_nano)            (P(recoverable error) under independence)
  rank_diff   pct-rank(q_mini) - pct-rank(q_nano)  (scale-free rank combination)

Same model, same rows -- the only variable is the formula.  Comparison:
frontier accuracy at fixed budgets (top-k routing).  Priors none and
mean_prob@1; five seeds; no API calls.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

import run_sembench_twohead_correctness_router as base
from run_sembench_prior_mo_random import pool_base_rates, select_random
from run_sembench_qa50_softprompt_router import deduplicate
from run_sembench_softprompt_mini_prior_ablation import expected_costs
from run_sembench_train100_ratio_sweep import make_orders
from train_sembench_balanced_direct_router import evaluate, load_examples


N_TRAIN = 25
BUDGETS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
PRIORS = [("none", {}), ("mean_prob_l1", {"mean_prob": 1.0})]


def logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1.0 - p))


def pct_rank(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    rank = [0.0] * len(values)
    for position, i in enumerate(order):
        rank[i] = position / (len(values) - 1)
    return rank


def score_forms(qn, qm):
    rn, rm = pct_rank(qn), pct_rank(qm)
    return {
        "diff": [m - n for n, m in zip(qn, qm)],
        "neg_nano": [-n for n in qn],
        "logit_diff": [logit(m) - logit(n) for n, m in zip(qn, qm)],
        "product": [m * (1 - n) for n, m in zip(qn, qm)],
        "rank_diff": [b - a for a, b in zip(rn, rm)],
    }


def frontier_accuracy(rows, scores, budget):
    order = sorted(range(len(rows)), key=lambda i: -scores[i])
    to_mini = set(order[:round(budget * len(rows))])
    return mean(
        (1.0 if ((rows[i]["mini_pred"] == rows[i]["gold"]) if i in to_mini
                 else (rows[i]["nano_pred"] == rows[i]["gold"])) else 0.0)
        for i in range(len(rows))
    )


def train_heads(train, test, cache, seed, prior_cfg, rates, args):
    """Standard two-head training, fixed epochs, no validation; returns the
    last epoch's head probabilities on the test set."""
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
    xl = base.loader(test, tokenizer, args.max_length, args.batch_size, False)
    costs = expected_costs(train, cache)
    inv = torch.tensor([1.0 / costs["nano"], 1.0 / costs["mini"]], dtype=torch.float)
    for _epoch in range(args.epochs):
        model.train()
        for batch in tl:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], batch["attention_mask"])
            raw = F.binary_cross_entropy_with_logits(logits, batch["labels"], reduction="none")
            loss = (raw * inv).sum() / (len(batch["labels"]) * inv.sum())
            if "mean_prob" in prior_cfg:
                q = torch.sigmoid(logits)
                loss = loss + prior_cfg["mean_prob"] * (
                    (q[:, 0].mean() - p_nano).pow(2) + (q[:, 1].mean() - p_mini).pow(2)
                )
            loss.backward()
            optimizer.step()
    return base.correctness_probabilities(model, xl)


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
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/sembench_score_forms.json")
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
        "status": "score read-out comparison on the minimal protocol (25 random gold, fixed 8 epochs, no validation)",
        "forms": ["diff", "neg_nano", "logit_diff", "product", "rank_diff"],
        "comparison": "same trained heads, same rows; only the read-out formula varies; "
                      "frontier accuracy at fixed budgets",
        "answer_model_api_calls": 0,
    }
    result["baselines"] = {
        "all_nano": evaluate(heldout, [0] * len(heldout))["selected_accuracy"],
        "all_mini": evaluate(heldout, [1] * len(heldout))["selected_accuracy"],
    }
    runs = result.setdefault("runs", {})

    for prior_name, prior_cfg in PRIORS:
        done = {int(r["seed"]) for r in runs.get(prior_name, [])}
        for seed in args.seeds:
            if seed in done:
                continue
            orders, _validation, _ = per_seed[seed]
            train = select_random(orders, seed, N_TRAIN)
            print(f"{prior_name} seed={seed}", flush=True)
            qn, qm = train_heads(train, heldout, cache, seed, prior_cfg,
                                 rates_by_seed[seed], args)
            forms = score_forms(qn, qm)
            run = {"seed": seed, "frontiers": {}}
            for name, scores in forms.items():
                run["frontiers"][name] = {
                    str(b): frontier_accuracy(heldout, scores, b) for b in BUDGETS
                }
            for name in ("diff", "neg_nano"):
                accs = run["frontiers"][name]
                print(f"  {name:10s} " + " ".join(f"{accs[str(b)]*100:.1f}" for b in (0.3, 0.5, 0.7)),
                      flush=True)
            runs.setdefault(prior_name, []).append(run)
            runs[prior_name] = sorted(runs[prior_name], key=lambda r: int(r["seed"]))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    result["summary"] = {}
    for prior_name, rs in sorted(runs.items()):
        result["summary"][prior_name] = {}
        for name in ("diff", "neg_nano", "logit_diff", "product", "rank_diff"):
            result["summary"][prior_name][name] = {
                str(b): {"mean": mean(r["frontiers"][name][str(b)] for r in rs),
                         "std": pstdev(r["frontiers"][name][str(b)] for r in rs)}
                for b in BUDGETS
            }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
