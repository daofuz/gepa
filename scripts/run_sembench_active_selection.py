#!/usr/bin/env python3
"""Active acquisition of the 50-example training set: which selector, and does
a prior let weak selectors catch up?

Fixed labeling budget: 50 fully-labeled examples (a label = knowing both
Nano and Mini outcomes for that example).  Selectors, ordered by how much
observation they need BEFORE spending the label budget:

  random50        -- floor: uniform 50 from the selection pool.
                     Queries: 50 nano + 50 mini.
  iterative       -- honest budgeted active learning: 20 random seed labels,
                     then 3 rounds of +10 picked by router uncertainty
                     (smallest |q_mini - q_nano| on the unlabeled pool; the
                     acquisition router is retrained on labels so far, no
                     prior).  Queries: 50 nano + 50 mini.
  nano_first      -- run Nano on the WHOLE pool first (cheap; gold answers are
                     known for historical traffic, so nano-correct is then
                     observable pool-wide).  Spend 30 labels on nano-wrong
                     examples (MO or BW; which one is only revealed by the
                     Mini query) and 20 on nano-right.
                     Queries: |pool| nano + 50 mini.
  disagree_oracle -- 25 Nano/Mini disagreements + 25 agreements (the old
                     active-50 recipe).  Needs BOTH models on the whole pool:
                     a budget-violating upper bound, kept as reference.

Every selected 50 then trains the standard two-head router under three prior
conditions (none, bias_map@1, mean_prob@1) with the usual protocol
(validation-max-accuracy checkpoint, held-out 200, ranking frontier).
Compare with bucket50_mo8 (curated oracle mixture) from
outputs/sembench_prior_mo_random.json.  No answer-model API calls.
"""
from __future__ import annotations

import argparse
import copy
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
from run_sembench_mo_importance import agg_frontier
from run_sembench_prior_mo_random import pool_base_rates, select_random, train_router
from run_sembench_qa50_softprompt_router import deduplicate
from run_sembench_softprompt_mini_prior_ablation import expected_costs
from run_sembench_train100_ratio_sweep import make_orders
from train_sembench_balanced_direct_router import evaluate, load_examples


PRIORS = [
    ("none", {}),
    ("bias_map_l1", {"bias_map": 1.0}),
    ("mean_prob_l1", {"mean_prob": 1.0}),
]
NANO_WRONG = ("mini_only", "both_wrong")


def flat(orders):
    return [row for rows in orders.values() for row in rows]


def select_nano_first(orders, seed, wrong_budget=30, total=50):
    pool = flat(orders)
    wrong = [r for r in pool if r["bucket"] in NANO_WRONG]
    right = [r for r in pool if r["bucket"] not in NANO_WRONG]
    rng = random.Random(seed + 86028121)
    w = rng.sample(wrong, min(wrong_budget, len(wrong)))
    selected = w + rng.sample(right, total - len(w))
    rng.shuffle(selected)
    return selected


def select_nano_first_bwfilter(orders, seed, wrong_budget=30, total=50, keep_bw=2):
    """nano_first, but after labeling, drop most both-wrong rows from training.

    The 50 labels are still spent; the BW rows (identified only after their
    Mini query) are excluded from the training set instead of poisoning it.
    """
    selected = select_nano_first(orders, seed, wrong_budget, total)
    bw = [r for r in selected if r["bucket"] == "both_wrong"]
    keep = set(id(r) for r in bw[:keep_bw])
    return [r for r in selected if r["bucket"] != "both_wrong" or id(r) in keep]


def select_disagree_oracle(orders, seed, total=50):
    pool = flat(orders)
    disagree = [r for r in pool if r["nano_pred"] != r["mini_pred"]]
    agree = [r for r in pool if r["nano_pred"] == r["mini_pred"]]
    rng = random.Random(seed + 67867967)
    d = rng.sample(disagree, min(25, len(disagree)))
    selected = d + rng.sample(agree, total - len(d))
    rng.shuffle(selected)
    return selected


def _uncertainty_scores(train, validation, rows, cache, seed, args):
    """Train the no-prior router on the labels so far; score |q_mini - q_nano|."""
    random.seed(seed)
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, local_files_only=True)
    model = base.TwoHeadCorrectnessRouter(args.hf_model, args.prompt_tokens)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    tl = base.loader(train, tokenizer, args.max_length, args.batch_size, True)
    vl = base.loader(validation, tokenizer, args.max_length, args.batch_size, False)
    pl = base.loader(rows, tokenizer, args.max_length, args.batch_size, False)
    costs = expected_costs(train, cache)
    inv = torch.tensor([1.0 / costs["nano"], 1.0 / costs["mini"]], dtype=torch.float)
    best = None
    for _epoch in range(1, args.epochs + 1):
        model.train()
        for batch in tl:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], batch["attention_mask"])
            raw = F.binary_cross_entropy_with_logits(logits, batch["labels"], reduction="none")
            loss = (raw * inv).sum() / (len(batch["labels"]) * inv.sum())
            loss.backward()
            optimizer.step()
        qn, qm = base.correctness_probabilities(model, vl)
        vacc = mean(
            1.0 if (r["mini_pred"] if m >= n else r["nano_pred"]) == r["gold"] else 0.0
            for r, n, m in zip(validation, qn, qm)
        )
        state = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        if best is None or vacc > best[0]:
            best = (vacc, copy.deepcopy(state))
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in best[1]:
                p.copy_(best[1][n])
    qn, qm = base.correctness_probabilities(model, pl)
    return [abs(m - n) for n, m in zip(qn, qm)]


def select_iterative(orders, validation, cache, seed, args, seed_n=20, batch=10, rounds=3):
    pool = flat(orders)
    rng = random.Random(seed + 104395301)
    labeled = rng.sample(pool, seed_n)
    labeled_ids = {r["review_id"] for r in labeled}
    for round_index in range(rounds):
        unlabeled = [r for r in pool if r["review_id"] not in labeled_ids]
        scores = _uncertainty_scores(labeled, validation, unlabeled, cache,
                                     seed + round_index, args)
        picked = sorted(range(len(unlabeled)), key=lambda i: scores[i])[:batch]
        for i in picked:
            labeled.append(unlabeled[i])
            labeled_ids.add(unlabeled[i]["review_id"])
        print(f"  iterative round {round_index + 1}: "
              f"counts={dict(Counter(r['bucket'] for r in labeled))}", flush=True)
    rng.shuffle(labeled)
    return labeled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", type=Path, default=ROOT / "sembench/files/movie/data/sf_2000/Reviews.csv")
    parser.add_argument("--cache", type=Path, default=ROOT / "sembench_movie_router/model_outputs.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "sembench_movie_router/untouched_200_manifest.json")
    parser.add_argument("--hf-model", default="distilbert-base-uncased")
    parser.add_argument("--prompt-tokens", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[505, 606, 707])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--reuse-random50", type=Path,
                        default=ROOT / "outputs/sembench_prior_mo_random.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/sembench_active_selection.json")
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
    result.setdefault("protocol", {
        "status": "post-hoc held-out acquisition-strategy diagnostic; not a fresh final test",
        "budget": "50 fully-labeled examples; mini queries are the binding cost (4x nano price)",
        "selectors": {
            "random50": "uniform; 50 nano + 50 mini queries",
            "iterative": "20 random seed + 3x10 by router uncertainty; 50 nano + 50 mini",
            "nano_first": "nano on whole pool, then 30 labels on nano-wrong + 20 on nano-right; |pool| nano + 50 mini",
            "disagree_oracle": "25 disagreements + 25 agreements; needs both models pool-wide (upper bound)",
        },
        "priors": {name: repr(cfg) for name, cfg in PRIORS},
        "router": "frozen DistilBERT, 8 soft tokens, two correctness heads, inverse-cost weighted BCE",
        "operating_point": "validation-max-accuracy checkpoint and threshold; held-out 200",
        "answer_model_api_calls": 0,
    })
    result.setdefault("baselines", {
        "all_nano": evaluate(heldout, [0] * len(heldout)),
        "all_mini": evaluate(heldout, [1] * len(heldout)),
    })
    selections = result.setdefault("selections", {})
    runs = result.setdefault("runs", {})

    # Pre-seed random50 runs from the prior x data sweep (identical selector,
    # seeds, and training code path), so they are not retrained.
    if args.reuse_random50.exists():
        prior_result = json.loads(args.reuse_random50.read_text(encoding="utf-8"))
        for prior_name, _cfg in PRIORS:
            key = f"random50__{prior_name}"
            if key not in runs and key in prior_result.get("runs", {}):
                runs[key] = prior_result["runs"][key]

    def get_selection(name, seed):
        cached = selections.get(name, {}).get(str(seed))
        if cached is not None:
            by_id = {r["review_id"]: r for r in flat(per_seed[seed][0])}
            return [by_id[i] for i in cached]
        orders, validation, _ = per_seed[seed]
        if name == "random50":
            train = select_random(orders, seed, 50)
        elif name == "nano_first":
            train = select_nano_first(orders, seed)
        elif name == "nano_first_bwfilter":
            train = select_nano_first_bwfilter(orders, seed)
        elif name == "disagree_oracle":
            train = select_disagree_oracle(orders, seed)
        elif name == "iterative":
            train = select_iterative(orders, validation, cache, seed, args)
        else:
            raise ValueError(name)
        selections.setdefault(name, {})[str(seed)] = [r["review_id"] for r in train]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return train

    for selector in ("random50", "nano_first", "nano_first_bwfilter", "disagree_oracle", "iterative"):
        for prior_name, prior_cfg in PRIORS:
            key = f"{selector}__{prior_name}"
            done = {int(r["seed"]): r for r in runs.get(key, [])}
            collected = []
            for seed in args.seeds:
                if seed in done:
                    collected.append(done[seed]); continue
                orders, validation, _ = per_seed[seed]
                train = get_selection(selector, seed)
                if {r["review_id"] for r in train} & {r["review_id"] for r in validation}:
                    raise AssertionError("Train/validation leakage")
                print(f"{key} seed={seed} counts={dict(Counter(r['bucket'] for r in train))}",
                      flush=True)
                run = train_router(train, validation, heldout, cache, seed,
                                   prior_cfg, rates_by_seed[seed], args)
                op = run["operating_point"]
                print(f"  acc={op['accuracy']:.4f} rate={op['mini_rate']:.4f} "
                      f"save={op['mini_saving_vs_all_mini']:.4f} MOrec={op['mini_only_recall']:.4f}",
                      flush=True)
                collected.append(run)
                runs[key] = collected
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
            runs[key] = sorted(collected, key=lambda r: int(r["seed"]))

    result["summary"] = {}
    for key, rs in runs.items():
        if len(rs) != len(args.seeds):
            continue
        result["summary"][key] = {
            "mo_in_train": [r["train_counts"].get("mini_only", 0) for r in rs],
            "train_counts": [r["train_counts"] for r in rs],
            "mo_precision_mean": mean(r["train_counts"].get("mini_only", 0) / r["train_size"] for r in rs),
            "operating_point": {
                f: {"mean": mean(r["operating_point"][f] for r in rs),
                    "std": pstdev(r["operating_point"][f] for r in rs)}
                for f in ("accuracy", "mini_rate", "mini_saving_vs_all_mini", "mini_only_recall")
            },
            "frontier": agg_frontier(rs),
        }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
