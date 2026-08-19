#!/usr/bin/env python3
"""Score read-out forms for the two-head router (Computer QA).

Mirrors scripts/run_sembench_score_forms.py: minimal protocol (25 random
gold rows, fixed 8 epochs, no validation), last-epoch head probabilities on
the held-out 205 re-scored under five forms (diff / neg_nano / logit_diff /
product / rank_diff), frontier accuracy at fixed budgets.  Priors none and
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from optimize_ollama_router_gepa import load_examples
from run_computer_qa_prior_mo_random import (
    correctness_probabilities,
    loader,
    make_orders,
    mini_ok,
    nano_ok,
    pool_base_rates,
    select_random,
)
from run_sembench_twohead_correctness_router import TwoHeadCorrectnessRouter


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
        (1.0 if (mini_ok(rows[i]) if i in to_mini else nano_ok(rows[i])) else 0.0)
        for i in range(len(rows))
    )


def train_heads(train, test, seed, prior_cfg, rates, args):
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
    xl = loader(test, tokenizer, args.max_length, args.batch_size, False)
    costs = {
        "nano": mean(e.nano_cost_usd for e in train),
        "mini": mean(e.mini_cost_usd for e in train),
    }
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
    return correctness_probabilities(model, xl)


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
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/computer_qa_score_forms.json")
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
        "status": "score read-out comparison on the minimal protocol (Computer QA)",
        "forms": ["diff", "neg_nano", "logit_diff", "product", "rank_diff"],
        "comparison": "same trained heads, same rows; only the read-out formula varies",
        "answer_model_api_calls": 0,
    }
    result["baselines"] = {
        "all_nano": mean(1.0 if nano_ok(e) else 0.0 for e in heldout),
        "all_mini": mean(1.0 if mini_ok(e) else 0.0 for e in heldout),
    }
    runs = result.setdefault("runs", {})

    for prior_name, prior_cfg in PRIORS:
        done = {int(r["seed"]) for r in runs.get(prior_name, [])}
        for seed in args.seeds:
            if seed in done:
                continue
            orders, _validation = per_seed[seed]
            train = select_random(orders, seed, N_TRAIN)
            print(f"{prior_name} seed={seed}", flush=True)
            qn, qm = train_heads(train, heldout, seed, prior_cfg, rates_by_seed[seed], args)
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
