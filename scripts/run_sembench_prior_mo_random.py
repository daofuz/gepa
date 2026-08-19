#!/usr/bin/env python3
"""Does a Mini>Nano accuracy prior in the loss help, and is curated MO needed?

Crossed study on the 8-token two-head soft-prompt router (frozen DistilBERT,
inverse-expected-cost weighted BCE, validation-max-accuracy checkpoint,
held-out 200 + ranking frontier -- identical protocol to
run_sembench_mo_importance.py so results are directly comparable).

  DATA conditions:
    bucket50_mo8  -- curated fixed-50 mixture MO=8, BC=39, NO=2, BW=1
                     (the additive/fixed sweet spot from the MO study)
    random50      -- uniform random 50 from the training pool (no curation)
    random100     -- uniform random 100 (does doubling random data catch up?)

  PRIOR implementations (belief: Mini base correctness > Nano):
    none            -- baseline weighted BCE
    bias_init       -- head biases initialised at pool base-rate log-odds;
                       no extra loss term (prior as initialisation)
    bias_map@L      -- Gaussian MAP: L/2 * ((b_mini-b_nano) - prior_gap)^2
                       pulling the intercept gap toward the pool log-odds gap
    mean_prob@L     -- L * ((mean q_nano - p_nano)^2 + (mean q_mini - p_mini)^2)
                       pulling batch-mean predictions toward pool base rates
    label_smooth@E  -- targets (1-E)*y + E*(p_nano, p_mini): every example
                       carries a little of the prior

Prior base rates p_nano/p_mini are measured on the seed's training POOL (never
validation/test/held-out).  No answer-model API calls.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
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
from run_sembench_accuracy_targeted_router import evaluate_scores, select_threshold
from run_sembench_mo_importance import agg_frontier, frontier, select_fixed50
from run_sembench_qa50_softprompt_router import deduplicate
from run_sembench_softprompt_mini_prior_ablation import expected_costs
from run_sembench_train100_ratio_sweep import make_orders
from train_sembench_balanced_direct_router import evaluate, load_examples


PRIORS = [
    ("none", {}),
    ("bias_init", {"bias_init": True}),
    ("bias_map_l0.1", {"bias_map": 0.1}),
    ("bias_map_l1", {"bias_map": 1.0}),
    ("mean_prob_l0.1", {"mean_prob": 0.1}),
    ("mean_prob_l1", {"mean_prob": 1.0}),
    ("label_smooth_e0.1", {"label_smooth": 0.1}),
]


def logit(p):
    return math.log(p / (1.0 - p))


def pool_base_rates(orders):
    pool = [row for rows in orders.values() for row in rows]
    return {
        "p_nano": mean(float(r["nano_pred"] == r["gold"]) for r in pool),
        "p_mini": mean(float(r["mini_pred"] == r["gold"]) for r in pool),
        "pool_size": len(pool),
    }


def select_bucket50(orders, seed):
    return select_fixed50(orders, 8, seed)


def select_random(orders, seed, n):
    pool = [row for rows in orders.values() for row in rows]
    rng = random.Random(seed + 49979687 + n)
    selected = rng.sample(pool, n)
    ids = {row["review_id"] for row in selected}
    if len(ids) != n:
        raise AssertionError(f"Expected {n} unique rows, got {len(ids)}")
    return selected


def train_router(train, validation, test, cache, seed, prior_cfg, rates, args):
    random.seed(seed)
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, local_files_only=True)
    model = base.TwoHeadCorrectnessRouter(args.hf_model, args.prompt_tokens)
    p_nano, p_mini = rates["p_nano"], rates["p_mini"]
    if prior_cfg.get("bias_init"):
        with torch.no_grad():
            model.correctness_heads.bias.copy_(
                torch.tensor([logit(p_nano), logit(p_mini)])
            )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    tl = base.loader(train, tokenizer, args.max_length, args.batch_size, True)
    vl = base.loader(validation, tokenizer, args.max_length, args.batch_size, False)
    xl = base.loader(test, tokenizer, args.max_length, args.batch_size, False)
    costs = expected_costs(train, cache)
    inv = torch.tensor([1.0 / costs["nano"], 1.0 / costs["mini"]], dtype=torch.float)
    prior_gap = logit(p_mini) - logit(p_nano)
    prior_probs = torch.tensor([p_nano, p_mini], dtype=torch.float)
    smooth = prior_cfg.get("label_smooth", 0.0)

    best = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in tl:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], batch["attention_mask"])
            labels = batch["labels"]
            if smooth:
                labels = (1.0 - smooth) * labels + smooth * prior_probs
            raw = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            loss = (raw * inv).sum() / (len(labels) * inv.sum())
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
        qn, qm = base.correctness_probabilities(model, vl)
        vscores = base.score(qn, qm)
        cal = select_threshold(validation, vscores, "max_accuracy")
        vmetrics = cal["validation_metrics"]
        key = (vmetrics["selected_accuracy"], -vmetrics["mini_calls"])
        if best is None or key > best["key"]:
            best = {
                "key": key, "epoch": epoch, "threshold": cal["threshold"],
                "state": copy.deepcopy({n: p.detach().clone()
                                        for n, p in model.named_parameters() if p.requires_grad}),
            }

    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in best["state"]:
                p.copy_(best["state"][n])
    qn, qm = base.correctness_probabilities(model, xl)
    tscores = base.score(qn, qm)
    op = evaluate_scores(test, tscores, best["threshold"])
    bias = model.correctness_heads.bias.detach()
    return {
        "seed": seed,
        "train_counts": dict(Counter(r["bucket"] for r in train)),
        "train_size": len(train),
        "best_epoch": best["epoch"],
        "final_bias_gap": float(bias[1] - bias[0]),
        "operating_point": {
            "threshold": best["threshold"],
            "accuracy": op["selected_accuracy"],
            "mini_rate": op["mini_rate"],
            "mini_saving_vs_all_mini": op["mini_saving_vs_all_mini"],
            "mini_only_recall": op["mini_only_recall"],
        },
        "frontier": frontier(test, tscores),
        "mean_q_nano": mean(qn),
        "mean_q_mini": mean(qm),
    }


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
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/sembench_prior_mo_random.json")
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
        "status": "post-hoc held-out prior x data-selection diagnostic; not a fresh final test",
        "model": "frozen DistilBERT, 8 soft tokens, two correctness heads",
        "loss": "inverse-expected-query-cost weighted BCE plus optional prior term",
        "data_conditions": {
            "bucket50_mo8": "curated MO=8, BC=39, NO=2, BW=1 (total 50)",
            "random50": "uniform random 50 from training pool",
            "random100": "uniform random 100 from training pool",
        },
        "prior_conditions": {name: repr(cfg) for name, cfg in PRIORS},
        "prior_base_rates": "measured on each seed's training pool, never on validation/test/held-out",
        "operating_point": "validation-max-accuracy checkpoint and threshold",
        "frontier": "route top-scoring examples to Mini at each budget",
        "answer_model_api_calls": 0,
    })
    result.setdefault("baselines", {
        "all_nano": evaluate(heldout, [0] * len(heldout)),
        "all_mini": evaluate(heldout, [1] * len(heldout)),
    })
    result["pool_base_rates"] = {str(s): rates_by_seed[s] for s in args.seeds}
    runs = result.setdefault("runs", {})

    selectors = [
        ("bucket50_mo8", lambda o, s: select_bucket50(o, s)),
        ("random50", lambda o, s: select_random(o, s, 50)),
        ("random100", lambda o, s: select_random(o, s, 100)),
    ]
    for data_name, selector in selectors:
        for prior_name, prior_cfg in PRIORS:
            key = f"{data_name}__{prior_name}"
            done = {int(r["seed"]): r for r in runs.get(key, [])}
            collected = []
            for seed in args.seeds:
                if seed in done:
                    collected.append(done[seed]); continue
                orders, validation, _ = per_seed[seed]
                train = selector(orders, seed)
                if {r["review_id"] for r in train} & {r["review_id"] for r in validation}:
                    raise AssertionError("Train/validation leakage")
                print(f"{key} seed={seed} counts={dict(Counter(r['bucket'] for r in train))}", flush=True)
                run = train_router(train, validation, heldout, cache, seed,
                                   prior_cfg, rates_by_seed[seed], args)
                op = run["operating_point"]
                print(f"  acc={op['accuracy']:.4f} rate={op['mini_rate']:.4f} "
                      f"save={op['mini_saving_vs_all_mini']:.4f} MOrec={op['mini_only_recall']:.4f} "
                      f"bias_gap={run['final_bias_gap']:.3f}", flush=True)
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
            "train_size_mean": mean(r["train_size"] for r in rs),
            "mo_in_train": [r["train_counts"].get("mini_only", 0) for r in rs],
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
