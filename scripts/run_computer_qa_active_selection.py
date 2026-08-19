#!/usr/bin/env python3
"""Computer QA replication of the active-acquisition study.

Mirrors scripts/run_sembench_active_selection.py: fixed budget of 50
fully-labeled examples, four selectors (random50 floor, honest iterative
uncertainty, nano_first cascade, disagree_oracle upper bound), each crossed
with priors (none, bias_map@1, mean_prob@1).  Standard two-head router
protocol; held-out = the 205 test_ids; ranking frontier.  Compare with
bucket50_mo8 in outputs/computer_qa_prior_mo_random.json.
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from optimize_ollama_router_gepa import load_examples, outcome_bucket
from run_sembench_twohead_correctness_router import TwoHeadCorrectnessRouter
from run_computer_qa_prior_mo_random import (
    agg_frontier,
    correctness_probabilities,
    evaluate_at,
    loader,
    make_orders,
    mini_ok,
    nano_ok,
    pool_base_rates,
    select_random,
    train_router,
)


PRIORS = [
    ("none", {}),
    ("bias_map_l1", {"bias_map": 1.0}),
    ("mean_prob_l1", {"mean_prob": 1.0}),
]
NANO_WRONG = ("mini_only", "both_wrong")


def flat(orders):
    return [e for rows in orders.values() for e in rows]


def select_nano_first(orders, seed, wrong_budget=30, total=50):
    pool = flat(orders)
    wrong = [e for e in pool if outcome_bucket(e) in NANO_WRONG]
    right = [e for e in pool if outcome_bucket(e) not in NANO_WRONG]
    rng = random.Random(seed + 86028121)
    w = rng.sample(wrong, min(wrong_budget, len(wrong)))
    selected = w + rng.sample(right, total - len(w))
    rng.shuffle(selected)
    return selected


def select_nano_first_bwfilter(orders, seed, wrong_budget=30, total=50, keep_bw=2):
    """nano_first, but after labeling, drop most both-wrong rows from training."""
    selected = select_nano_first(orders, seed, wrong_budget, total)
    bw = [e for e in selected if outcome_bucket(e) == "both_wrong"]
    keep = {e.question_id for e in bw[:keep_bw]}
    return [e for e in selected
            if outcome_bucket(e) != "both_wrong" or e.question_id in keep]


def select_disagree_oracle(orders, seed, total=50):
    pool = flat(orders)
    disagree = [e for e in pool if e.nano_pred != e.mini_pred]
    agree = [e for e in pool if e.nano_pred == e.mini_pred]
    rng = random.Random(seed + 67867967)
    d = rng.sample(disagree, min(25, len(disagree)))
    selected = d + rng.sample(agree, total - len(d))
    rng.shuffle(selected)
    return selected


def _uncertainty_scores(train, validation, rows, seed, args):
    random.seed(seed)
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, local_files_only=True)
    model = TwoHeadCorrectnessRouter(args.hf_model, args.prompt_tokens)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    tl = loader(train, tokenizer, args.max_length, args.batch_size, True)
    vl = loader(validation, tokenizer, args.max_length, args.batch_size, False)
    pl = loader(rows, tokenizer, args.max_length, args.batch_size, False)
    inv = torch.tensor([
        1.0 / mean(e.nano_cost_usd for e in train),
        1.0 / mean(e.mini_cost_usd for e in train),
    ], dtype=torch.float)
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
        qn, qm = correctness_probabilities(model, vl)
        vacc = mean(
            1.0 if (mini_ok(e) if m >= n else nano_ok(e)) else 0.0
            for e, n, m in zip(validation, qn, qm)
        )
        state = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        if best is None or vacc > best[0]:
            best = (vacc, copy.deepcopy(state))
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in best[1]:
                p.copy_(best[1][n])
    qn, qm = correctness_probabilities(model, pl)
    return [abs(m - n) for n, m in zip(qn, qm)]


def select_iterative(orders, validation, seed, args, seed_n=20, batch=10, rounds=3):
    pool = flat(orders)
    rng = random.Random(seed + 104395301)
    labeled = rng.sample(pool, seed_n)
    labeled_ids = {e.question_id for e in labeled}
    for round_index in range(rounds):
        unlabeled = [e for e in pool if e.question_id not in labeled_ids]
        scores = _uncertainty_scores(labeled, validation, unlabeled, seed + round_index, args)
        picked = sorted(range(len(unlabeled)), key=lambda i: scores[i])[:batch]
        for i in picked:
            labeled.append(unlabeled[i])
            labeled_ids.add(unlabeled[i].question_id)
        print(f"  iterative round {round_index + 1}: "
              f"counts={dict(Counter(outcome_bucket(e) for e in labeled))}", flush=True)
    rng.shuffle(labeled)
    return labeled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mini-file", type=Path, default=ROOT / "computer science_result_mini.json")
    parser.add_argument("--nano-file", type=Path, default=ROOT / "computer science_result_nano.json")
    parser.add_argument("--split-file", type=Path,
                        default=ROOT / "routing_split_softprompt_threshold_balanced16.json")
    parser.add_argument("--hf-model", default="distilbert-base-uncased")
    parser.add_argument("--prompt-tokens", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[505, 606, 707])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--reuse-random50", type=Path,
                        default=ROOT / "outputs/computer_qa_prior_mo_random.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/computer_qa_active_selection.json")
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
    result.setdefault("protocol", {
        "status": "post-hoc held-out acquisition-strategy diagnostic on Computer QA",
        "budget": "50 fully-labeled examples; mini queries are the binding cost",
        "selectors": {
            "random50": "uniform; 50 nano + 50 mini queries",
            "iterative": "20 random seed + 3x10 by router uncertainty; 50 nano + 50 mini",
            "nano_first": "nano on whole pool, then 30 labels on nano-wrong + 20 on nano-right; |pool| nano + 50 mini",
            "disagree_oracle": "25 disagreements + 25 agreements; needs both models pool-wide (upper bound)",
        },
        "priors": {name: repr(cfg) for name, cfg in PRIORS},
        "router": "frozen DistilBERT, 8 soft tokens, two correctness heads, inverse-cost weighted BCE",
        "operating_point": "validation-max-accuracy checkpoint and threshold; held-out 205",
        "answer_model_api_calls": 0,
    })
    result.setdefault("baselines", {
        "all_nano": evaluate_at(heldout, [0.0] * len(heldout), 1.0),
        "all_mini": evaluate_at(heldout, [1.0] * len(heldout), 0.0),
    })
    selections = result.setdefault("selections", {})
    runs = result.setdefault("runs", {})

    if args.reuse_random50.exists():
        prior_result = json.loads(args.reuse_random50.read_text(encoding="utf-8"))
        for prior_name, _cfg in PRIORS:
            key = f"random50__{prior_name}"
            if key not in runs and key in prior_result.get("runs", {}):
                runs[key] = prior_result["runs"][key]

    def get_selection(name, seed):
        cached = selections.get(name, {}).get(str(seed))
        if cached is not None:
            by_id = {e.question_id: e for e in flat(per_seed[seed][0])}
            return [by_id[i] for i in cached]
        orders, validation = per_seed[seed]
        if name == "random50":
            train = select_random(orders, seed, 50)
        elif name == "nano_first":
            train = select_nano_first(orders, seed)
        elif name == "nano_first_bwfilter":
            train = select_nano_first_bwfilter(orders, seed)
        elif name == "disagree_oracle":
            train = select_disagree_oracle(orders, seed)
        elif name == "iterative":
            train = select_iterative(orders, validation, seed, args)
        else:
            raise ValueError(name)
        selections.setdefault(name, {})[str(seed)] = [e.question_id for e in train]
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
                orders, validation = per_seed[seed]
                train = get_selection(selector, seed)
                if {e.question_id for e in train} & {e.question_id for e in validation}:
                    raise AssertionError("Train/validation leakage")
                print(f"{key} seed={seed} counts={dict(Counter(outcome_bucket(e) for e in train))}",
                      flush=True)
                run = train_router(train, validation, heldout, seed,
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
