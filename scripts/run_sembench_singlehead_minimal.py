#!/usr/bin/env python3
"""The closing experiment: single-head router under the fully minimal protocol (Movie).

Everything the line converged to, end to end, trained from scratch:

  Architecture: ONE head predicting P(Nano correct) on the plain text
    (pre-route -- no Nano answer in the input, no cascade tax); route to
    Mini when the score -q is above threshold.
  Labels: 25 uniform random rows, gold check of Nano's answer only.
  Training: fixed 8 epochs.  NO validation set of any kind.
  Threshold: fixed from the start as a pool-score quantile at the chosen
    budget (200 unlabeled rows; gold never read).

  Prior variants (mean_prob anchors mean prediction to a Nano base rate):
    none        -- plain BCE
    prior_pool  -- base rate measured on the seed's selection pool (matches
                   every earlier prior run; mild protocol impurity)
    prior_train -- base rate estimated from the 25 training labels alone
                   (fully self-contained; closes the last impurity)

Reference: the two-head forms of outputs/sembench_score_forms.json (same
25-row training sets, same fixed epochs).  Metrics: frontier accuracy at
budgets, plus realized operating points under pool-sample quantile
thresholds.  Five seeds, held-out 200, no API calls.
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
import torch.nn.functional as F
from transformers import AutoTokenizer

from run_sembench_50label_budget import quantile_threshold
from run_sembench_external_baselines import SingleHeadRouter, make_loader, probabilities
from run_sembench_prior_mo_random import pool_base_rates, select_random
from run_sembench_qa50_softprompt_router import deduplicate
from run_sembench_train100_ratio_sweep import make_orders
from train_sembench_balanced_direct_router import load_examples, text


N_TRAIN = 25
BUDGETS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
OP_BUDGETS = [0.3, 0.5, 0.7]
POOL_SAMPLE = 200
VARIANTS = ("none", "prior_pool", "prior_train")


def plain_text(e):
    return text(e)


def nano_correct_label(e):
    return 1.0 if e["nano_pred"] == e["gold"] else 0.0


def flat(orders):
    return [row for rows in orders.values() for row in rows]


def frontier_accuracy(rows, scores, budget):
    order = sorted(range(len(rows)), key=lambda i: -scores[i])
    to_mini = set(order[:round(budget * len(rows))])
    return mean(
        (1.0 if ((rows[i]["mini_pred"] == rows[i]["gold"]) if i in to_mini
                 else (rows[i]["nano_pred"] == rows[i]["gold"])) else 0.0)
        for i in range(len(rows))
    )


def operating_point(rows, scores, threshold):
    to_mini = [s >= threshold for s in scores]
    correct = mean(
        (1.0 if ((r["mini_pred"] == r["gold"]) if m else (r["nano_pred"] == r["gold"])) else 0.0)
        for r, m in zip(rows, to_mini)
    )
    return {"accuracy": correct, "mini_rate": mean(1.0 if m else 0.0 for m in to_mini)}


def train_single_head(train, pool_sample, test, seed, variant, pool_rate, args):
    random.seed(seed)
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, local_files_only=True)
    model = SingleHeadRouter(args.hf_model, args.prompt_tokens)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    if variant == "prior_pool":
        anchor = pool_rate
    elif variant == "prior_train":
        anchor = mean(nano_correct_label(e) for e in train)
        anchor = min(max(anchor, 0.02), 0.98)
    else:
        anchor = None
    mk = lambda rows, sh: make_loader(rows, tokenizer, args.max_length, args.batch_size,
                                      sh, plain_text, nano_correct_label)
    tl, pl, xl = mk(train, True), mk(pool_sample, False), mk(test, False)
    for _epoch in range(args.epochs):
        model.train()
        for batch in tl:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = F.binary_cross_entropy_with_logits(logits, batch["labels"])
            if anchor is not None:
                q = torch.sigmoid(logits)
                loss = loss + args.prior_weight * (q.mean() - anchor).pow(2)
            loss.backward()
            optimizer.step()
    # score = -P(Nano correct): higher -> escalate
    pool_scores = [-p for p in probabilities(model, pl)]
    test_scores = [-p for p in probabilities(model, xl)]
    return pool_scores, test_scores


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
    parser.add_argument("--prior-weight", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/sembench_singlehead_minimal.json")
    args = parser.parse_args()

    heldout_ids = set(json.loads(args.manifest.read_text(encoding="utf-8"))["review_ids"])
    examples = deduplicate(load_examples(args.reviews, args.cache))
    heldout = [r for r in examples if r["review_id"] in heldout_ids]
    historical = [r for r in examples if r["review_id"] not in heldout_ids]
    if len(heldout) != 200 or len(historical) != 812:
        raise AssertionError("Expected 812 historical and 200 held-out rows")
    per_seed = {seed: make_orders(historical, seed) for seed in args.seeds}
    rates_by_seed = {seed: pool_base_rates(per_seed[seed][0])["p_nano"] for seed in args.seeds}

    result = {}
    if args.output.exists():
        result = json.loads(args.output.read_text(encoding="utf-8"))
    result["protocol"] = {
        "status": "single-head minimal protocol, end to end (closing experiment)",
        "architecture": "one head, P(Nano correct) on plain text (pre-route); score = -q",
        "labels": f"{N_TRAIN} uniform random rows, Nano gold checks only",
        "training": "fixed 8 epochs; no validation set of any kind",
        "threshold": f"pool-score quantile on {POOL_SAMPLE} unlabeled rows, fixed per budget",
        "variants": {
            "none": "plain BCE",
            "prior_pool": "mean anchored to pool Nano rate (matches earlier prior runs)",
            "prior_train": "mean anchored to the 25 training labels' own rate (fully self-contained)",
        },
        "answer_model_api_calls": 0,
    }
    result["baselines"] = {
        "all_nano": mean(1.0 if r["nano_pred"] == r["gold"] else 0.0 for r in heldout),
        "all_mini": mean(1.0 if r["mini_pred"] == r["gold"] else 0.0 for r in heldout),
    }
    runs = result.setdefault("runs", {})

    for variant in VARIANTS:
        done = {int(r["seed"]) for r in runs.get(variant, [])}
        for seed in args.seeds:
            if seed in done:
                continue
            orders, _validation, _ = per_seed[seed]
            train = select_random(orders, seed, N_TRAIN)
            train_ids = {r["review_id"] for r in train}
            rng = random.Random(seed + 86243)
            pool_sample = rng.sample(
                [r for r in flat(orders) if r["review_id"] not in train_ids], POOL_SAMPLE)
            print(f"{variant} seed={seed}", flush=True)
            pool_scores, test_scores = train_single_head(
                train, pool_sample, heldout, seed, variant, rates_by_seed[seed], args)
            run = {"seed": seed,
                   "frontier": {str(b): frontier_accuracy(heldout, test_scores, b)
                                for b in BUDGETS},
                   "quantile_ops": {}}
            for b in OP_BUDGETS:
                t = quantile_threshold(pool_scores, b)
                run["quantile_ops"][str(b)] = operating_point(heldout, test_scores, t)
            op = run["quantile_ops"]["0.5"]
            print(f"  q50 acc={op['accuracy']:.4f} rate={op['mini_rate']:.4f}", flush=True)
            runs.setdefault(variant, []).append(run)
            runs[variant] = sorted(runs[variant], key=lambda r: int(r["seed"]))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    result["summary"] = {}
    for variant, rs in sorted(runs.items()):
        result["summary"][variant] = {
            "frontier": {str(b): {"mean": mean(r["frontier"][str(b)] for r in rs),
                                  "std": pstdev(r["frontier"][str(b)] for r in rs)}
                         for b in BUDGETS},
            "quantile_ops": {str(b): {
                f: {"mean": mean(r["quantile_ops"][str(b)][f] for r in rs),
                    "std": pstdev(r["quantile_ops"][str(b)][f] for r in rs)}
                for f in ("accuracy", "mini_rate")}
                for b in OP_BUDGETS},
        }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
