#!/usr/bin/env python3
"""Computer QA closing experiment: single-head router, fully minimal protocol.

Mirrors scripts/run_sembench_singlehead_minimal.py: one head predicting
P(Nano correct) on the plain question text (pre-route), 25 uniform random
rows with Nano gold checks only, fixed 8 epochs, no validation of any kind,
threshold fixed as a pool-score quantile.  Prior variants: none /
prior_pool / prior_train.  Five seeds, held-out 205, no API calls.
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
import torch.nn.functional as F
from transformers import AutoTokenizer

from optimize_ollama_router_gepa import load_examples
from run_computer_qa_50label_budget import quantile_threshold
from run_computer_qa_external_baselines import make_loader, probabilities
from run_computer_qa_prior_mo_random import (
    make_orders,
    mini_ok,
    nano_ok,
    pool_base_rates,
    select_random,
)
from run_sembench_external_baselines import SingleHeadRouter
from train_softprompt_router import example_text


N_TRAIN = 25
BUDGETS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
OP_BUDGETS = [0.3, 0.5, 0.7]
POOL_SAMPLE = 100
VARIANTS = ("none", "prior_pool", "prior_train")


def plain_text(e):
    return example_text(e, include_nano_answer=False, nano_response_max_chars=0)


def nano_correct_label(e):
    return 1.0 if nano_ok(e) else 0.0


def flat(orders):
    return [e for rows in orders.values() for e in rows]


def frontier_accuracy(rows, scores, budget):
    order = sorted(range(len(rows)), key=lambda i: -scores[i])
    to_mini = set(order[:round(budget * len(rows))])
    return mean(
        (1.0 if (mini_ok(rows[i]) if i in to_mini else nano_ok(rows[i])) else 0.0)
        for i in range(len(rows))
    )


def operating_point(rows, scores, threshold):
    to_mini = [s >= threshold for s in scores]
    correct = mean(
        (1.0 if (mini_ok(r) if m else nano_ok(r)) else 0.0)
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
    pool_scores = [-p for p in probabilities(model, pl)]
    test_scores = [-p for p in probabilities(model, xl)]
    return pool_scores, test_scores


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
    parser.add_argument("--prior-weight", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/computer_qa_singlehead_minimal.json")
    args = parser.parse_args()

    examples = load_examples(args.mini_file, args.nano_file)
    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    heldout_ids = {int(i) for i in split["test_ids"]}
    heldout = [e for e in examples if e.question_id in heldout_ids]
    pool = [e for e in examples if e.question_id not in heldout_ids]
    if len(heldout) != 205 or len(pool) != 205:
        raise AssertionError(f"Expected 205/205 split, got {len(heldout)}/{len(pool)}")
    per_seed = {seed: make_orders(pool, seed) for seed in args.seeds}
    rates_by_seed = {seed: pool_base_rates(per_seed[seed][0])["p_nano"] for seed in args.seeds}

    result = {}
    if args.output.exists():
        result = json.loads(args.output.read_text(encoding="utf-8"))
    result["protocol"] = {
        "status": "single-head minimal protocol on Computer QA (closing experiment)",
        "architecture": "one head, P(Nano correct) on plain text (pre-route); score = -q",
        "labels": f"{N_TRAIN} uniform random rows, Nano gold checks only",
        "training": "fixed 8 epochs; no validation set of any kind",
        "threshold": f"pool-score quantile on {POOL_SAMPLE} unlabeled rows, fixed per budget",
        "variants": {
            "none": "plain BCE",
            "prior_pool": "mean anchored to pool Nano rate",
            "prior_train": "mean anchored to the 25 training labels' own rate",
        },
        "answer_model_api_calls": 0,
    }
    result["baselines"] = {
        "all_nano": mean(1.0 if nano_ok(e) else 0.0 for e in heldout),
        "all_mini": mean(1.0 if mini_ok(e) else 0.0 for e in heldout),
    }
    runs = result.setdefault("runs", {})

    for variant in VARIANTS:
        done = {int(r["seed"]) for r in runs.get(variant, [])}
        for seed in args.seeds:
            if seed in done:
                continue
            orders, _validation = per_seed[seed]
            train = select_random(orders, seed, N_TRAIN)
            train_ids = {e.question_id for e in train}
            rng = random.Random(seed + 86243)
            rest = [e for e in flat(orders) if e.question_id not in train_ids]
            pool_sample = rng.sample(rest, min(POOL_SAMPLE, len(rest)))
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
