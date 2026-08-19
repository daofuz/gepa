#!/usr/bin/env python3
"""Idea: spend annotated disagreements on the VALIDATION set, not training.

Diagnosis motivating this: threshold search only reacts to validation rows
where the two cached predictions differ (on agreements the routed answer is
identical either way), and the standard stratified 40-row Movie validation
contains exactly 5 such rows per seed.  The operating-point instability that
dominates all recent sweeps (mini_rate std 0.2-0.3) is threshold selection
fit on 5 effective examples.

Design: train the standard two-head router with the same training path as
run_sembench_disagree_scale_augment.py, score every epoch on two alternative
validation sets, and pick (epoch, threshold) per design.  NOTE: every
DataLoader iteration draws one value from the global torch RNG (even with
shuffle=False), so the extra per-epoch evaluations shift the training batch
order from epoch 2 on -- these runs are a fresh randomness realization of the
same protocol, NOT bit-identical to the scale/augment runs (their
standard-design aggregates match those runs within seed noise).  Within a
run, both designs score the SAME per-epoch models, so the design comparison
is exactly paired:

  standard      -- the stratified 40-row validation (5 informative rows;
                   40 gold rows).  Must reproduce the existing runs exactly.
  disagree_val  -- 20 disagreements + 10 agreements drawn from unused pool
                   rows plus the standard validation rows (30 gold rows,
                   ~20 informative).  Less gold, 4x threshold signal.

Training configs: random40 (current best) and d25_a25 (continuity), priors
none and mean_prob@1, seeds 505-909.  Held-out 200; per-epoch held-out score
vectors are stored so alternative operating-point policies can be evaluated
post hoc without retraining.  No answer-model API calls.
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
from run_sembench_disagree_scale_augment import select_disagree_scale
from run_sembench_mo_importance import agg_frontier, frontier
from run_sembench_prior_mo_random import pool_base_rates, select_random
from run_sembench_qa50_softprompt_router import deduplicate
from run_sembench_softprompt_mini_prior_ablation import expected_costs
from run_sembench_train100_ratio_sweep import make_orders
from train_sembench_balanced_direct_router import evaluate, load_examples


PRIORS = [
    ("none", {}),
    ("mean_prob_l1", {"mean_prob": 1.0}),
]
CONFIGS = [
    ("random40", lambda orders, seed: select_random(orders, seed, 40)),
    ("d25_a25", lambda orders, seed: select_disagree_scale(orders, seed, 25)),
]
VAL_DISAGREE = 20
VAL_AGREE = 10


def logit(p):
    return math.log(p / (1.0 - p))


def flat(orders):
    return [row for rows in orders.values() for row in rows]


def build_disagree_validation(orders, standard_validation, train_ids, seed):
    """20 disagreements + 10 agreements from rows not used in training:
    unused selection-pool rows plus the standard validation rows."""
    candidates = [r for r in flat(orders) if r["review_id"] not in train_ids]
    candidates += standard_validation
    disagree = [r for r in candidates if r["nano_pred"] != r["mini_pred"]]
    agree = [r for r in candidates if r["nano_pred"] == r["mini_pred"]]
    rng = random.Random(seed + 51797)
    d = rng.sample(disagree, min(VAL_DISAGREE, len(disagree)))
    a = rng.sample(agree, min(VAL_AGREE, len(agree)))
    rows = d + a
    rng.shuffle(rows)
    return rows


def informative(rows):
    return sum(1 for r in rows if r["nano_pred"] != r["mini_pred"])


def train_with_designs(train, designs, test, cache, seed, prior_cfg, rates, args):
    """Standard training loop (identical RNG stream to the scale/augment runs);
    per-epoch scores are recorded for every validation design and the test set,
    and (epoch, threshold) is selected per design afterwards."""
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
    design_loaders = {
        name: base.loader(rows, tokenizer, args.max_length, args.batch_size, False)
        for name, rows in designs.items()
    }
    xl = base.loader(test, tokenizer, args.max_length, args.batch_size, False)
    costs = expected_costs(train, cache)
    inv = torch.tensor([1.0 / costs["nano"], 1.0 / costs["mini"]], dtype=torch.float)
    prior_gap = logit(p_mini) - logit(p_nano)
    smooth = prior_cfg.get("label_smooth", 0.0)
    prior_probs = torch.tensor([p_nano, p_mini], dtype=torch.float)

    epoch_scores = {name: [] for name in designs}
    epoch_test_scores = []
    for _epoch in range(1, args.epochs + 1):
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
        # Inference only below: consumes no RNG, so the training trajectory
        # stays identical no matter how many designs are evaluated.
        for name, dl in design_loaders.items():
            qn, qm = base.correctness_probabilities(model, dl)
            epoch_scores[name].append(base.score(qn, qm))
        qn, qm = base.correctness_probabilities(model, xl)
        epoch_test_scores.append(base.score(qn, qm))

    out = {"seed": seed, "train_counts": dict(Counter(r["bucket"] for r in train)),
           "train_size": len(train), "designs": {},
           "epoch_test_scores": epoch_test_scores}
    for name, rows in designs.items():
        best = None
        for epoch_index, vscores in enumerate(epoch_scores[name]):
            cal = select_threshold(rows, vscores, "max_accuracy")
            vmetrics = cal["validation_metrics"]
            key = (vmetrics["selected_accuracy"], -vmetrics["mini_calls"])
            if best is None or key > best["key"]:
                best = {"key": key, "epoch": epoch_index + 1, "threshold": cal["threshold"]}
        tscores = epoch_test_scores[best["epoch"] - 1]
        op = evaluate_scores(test, tscores, best["threshold"])
        out["designs"][name] = {
            "validation_size": len(rows),
            "informative_rows": informative(rows),
            "best_epoch": best["epoch"],
            "threshold": best["threshold"],
            "operating_point": {
                "accuracy": op["selected_accuracy"],
                "mini_rate": op["mini_rate"],
                "mini_saving_vs_all_mini": op["mini_saving_vs_all_mini"],
                "mini_only_recall": op["mini_only_recall"],
            },
            "frontier": frontier(test, tscores),
        }
    return out


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
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/sembench_disagree_validation.json")
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
        "status": "post-hoc held-out operating-point-policy diagnostic",
        "question": "does a disagreement-enriched validation set stabilize threshold+checkpoint selection?",
        "designs": {
            "standard": "stratified 40-row validation (existing protocol; ~5 informative rows; 40 gold)",
            "disagree_val": f"{VAL_DISAGREE} disagreements + {VAL_AGREE} agreements from unused pool rows "
                            "plus standard-validation rows (30 gold; ~20 informative)",
        },
        "training": "identical path and RNG stream as run_sembench_disagree_scale_augment.py; "
                    "standard-design results must reproduce that file's runs exactly",
        "answer_model_api_calls": 0,
    }
    result["baselines"] = {
        "all_nano": evaluate(heldout, [0] * len(heldout)),
        "all_mini": evaluate(heldout, [1] * len(heldout)),
    }
    runs = result.setdefault("runs", {})

    for config_name, selector in CONFIGS:
        for prior_name, prior_cfg in PRIORS:
            key = f"{config_name}__{prior_name}"
            done = {int(r["seed"]) for r in runs.get(key, [])}
            for seed in args.seeds:
                if seed in done:
                    continue
                orders, standard_validation, _ = per_seed[seed]
                train = selector(orders, seed)
                train_ids = {r["review_id"] for r in train}
                if train_ids & {r["review_id"] for r in standard_validation}:
                    raise AssertionError("Train/validation leakage")
                dval = build_disagree_validation(orders, standard_validation, train_ids, seed)
                if {r["review_id"] for r in dval} & train_ids:
                    raise AssertionError("Disagree-validation overlaps train")
                designs = {"standard": standard_validation, "disagree_val": dval}
                print(f"{key} seed={seed} informative: standard={informative(standard_validation)}"
                      f"/{len(standard_validation)} disagree_val={informative(dval)}/{len(dval)}",
                      flush=True)
                run = train_with_designs(train, designs, heldout, cache, seed,
                                         prior_cfg, rates_by_seed[seed], args)
                for name in ("standard", "disagree_val"):
                    op = run["designs"][name]["operating_point"]
                    print(f"  {name:12s} acc={op['accuracy']:.4f} rate={op['mini_rate']:.4f} "
                          f"MOrec={op['mini_only_recall']:.4f} "
                          f"epoch={run['designs'][name]['best_epoch']}", flush=True)
                runs.setdefault(key, []).append(run)
                runs[key] = sorted(runs[key], key=lambda r: int(r["seed"]))
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    result["summary"] = {}
    for key, rs in sorted(runs.items()):
        if not rs:
            continue
        result["summary"][key] = {}
        for name in ("standard", "disagree_val"):
            ds = [r["designs"][name] for r in rs]
            result["summary"][key][name] = {
                "seeds": [int(r["seed"]) for r in rs],
                "informative_rows": [d["informative_rows"] for d in ds],
                "operating_point": {
                    f: {"mean": mean(d["operating_point"][f] for d in ds),
                        "std": pstdev(d["operating_point"][f] for d in ds)}
                    for f in ("accuracy", "mini_rate", "mini_saving_vs_all_mini", "mini_only_recall")
                },
                "frontier": agg_frontier(ds),
            }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
