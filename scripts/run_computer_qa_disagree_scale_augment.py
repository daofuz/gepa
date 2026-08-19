#!/usr/bin/env python3
"""Computer QA replication of the disagreement-scaling / free-agreement study.

Mirrors scripts/run_sembench_disagree_scale_augment.py on the Computer QA
nano/mini routing task (same router, loss, checkpointing, held-out 205 and
ranking frontier as run_computer_qa_prior_mo_random.py):

  1. DISAGREEMENT-ANNOTATION SCALING.  D gold disagreements + 25 gold
     agreements, D in {5, 15, 25}; the pool only holds ~24-26 disagreements,
     so D=25 (clamped) is the exhaustion point and equals the old
     disagree_oracle recipe (same sampling salt).  Matched-budget uniform
     random controls at the same totals.

  2. FREE-AGREEMENT AUGMENTATION.  Unlabeled agreement rows added to the
     d25_a25 gold base with a consistency loss (q_nano - q_mini)^2 -- on an
     agreement both heads must predict the same correctness probability.
     Gold labels of augmented rows are never read.  N in {50, all(~92)}.

Priors: none and mean_prob@1.  Reuses cached runs for random50
(outputs/computer_qa_prior_mo_random.json) and d25_a25 (disagree_oracle in
outputs/computer_qa_active_selection.json).  No answer-model API calls.
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from optimize_ollama_router_gepa import load_examples, outcome_bucket
from run_computer_qa_prior_mo_random import (
    agg_frontier,
    correctness_probabilities,
    evaluate_at,
    frontier,
    loader,
    logit,
    make_orders,
    mini_ok,
    nano_ok,
    pool_base_rates,
    select_random,
    select_threshold,
)
from run_sembench_twohead_correctness_router import TwoHeadCorrectnessRouter


PRIORS = [
    ("none", {}),
    ("mean_prob_l1", {"mean_prob": 1.0}),
]
DISAGREE_SWEEP = [5, 15, 25]
AGREE_FIXED = 25
AUG_SIZES = [50, "all"]
AUG_BASE_DISAGREE = 25


def flat(orders):
    return [e for rows in orders.values() for e in rows]


def select_disagree_scale(orders, seed, n_disagree, n_agree=AGREE_FIXED):
    """Salt matches select_disagree_oracle so (25, 25) reproduces it exactly."""
    pool = flat(orders)
    disagree = [e for e in pool if e.nano_pred != e.mini_pred]
    agree = [e for e in pool if e.nano_pred == e.mini_pred]
    take = len(disagree) if n_disagree == "all" else min(n_disagree, len(disagree))
    rng = random.Random(seed + 67867967)
    d = rng.sample(disagree, take)
    selected = d + rng.sample(agree, n_agree)
    rng.shuffle(selected)
    return selected


def select_agree_augment(orders, seed, train_ids, n):
    pool = flat(orders)
    agree = [e for e in pool
             if e.nano_pred == e.mini_pred and e.question_id not in train_ids]
    rng = random.Random(seed + 27644437)
    take = len(agree) if n == "all" else min(n, len(agree))
    return rng.sample(agree, take)


def train_router_aug(train, aug, validation, test, seed, prior_cfg, rates, args):
    """run_computer_qa_prior_mo_random.train_router plus optional consistency
    term over unlabeled agreement rows; identical path when aug is empty."""
    random.seed(seed)
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, local_files_only=True)
    model = TwoHeadCorrectnessRouter(args.hf_model, args.prompt_tokens)
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
    tl = loader(train, tokenizer, args.max_length, args.batch_size, True)
    vl = loader(validation, tokenizer, args.max_length, args.batch_size, False)
    xl = loader(test, tokenizer, args.max_length, args.batch_size, False)
    al = loader(aug, tokenizer, args.max_length, args.aug_batch_size, True) if aug else None
    costs = {
        "nano": mean(e.nano_cost_usd for e in train),
        "mini": mean(e.mini_cost_usd for e in train),
    }
    inv = torch.tensor([1.0 / costs["nano"], 1.0 / costs["mini"]], dtype=torch.float)
    prior_gap = logit(p_mini) - logit(p_nano)
    prior_probs = torch.tensor([p_nano, p_mini], dtype=torch.float)
    smooth = prior_cfg.get("label_smooth", 0.0)

    best = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        aug_iter = iter(al) if al is not None else None
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
            if aug_iter is not None:
                try:
                    ab = next(aug_iter)
                except StopIteration:
                    aug_iter = iter(al)
                    ab = next(aug_iter)
                # Consistency only: labels of augmented rows are never used.
                aq = torch.sigmoid(model(ab["input_ids"], ab["attention_mask"]))
                loss = loss + args.aug_weight * (aq[:, 0] - aq[:, 1]).pow(2).mean()
            loss.backward()
            optimizer.step()
        qn, qm = correctness_probabilities(model, vl)
        vscores = [m - n for n, m in zip(qn, qm)]
        threshold, vmetrics = select_threshold(validation, vscores)
        key = (vmetrics["accuracy"], -vmetrics["mini_calls"])
        if best is None or key > best["key"]:
            best = {
                "key": key, "epoch": epoch, "threshold": threshold,
                "state": copy.deepcopy({n: p.detach().clone()
                                        for n, p in model.named_parameters() if p.requires_grad}),
            }

    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in best["state"]:
                p.copy_(best["state"][n])
    qn, qm = correctness_probabilities(model, xl)
    tscores = [m - n for n, m in zip(qn, qm)]
    op = evaluate_at(test, tscores, best["threshold"])
    bias = model.correctness_heads.bias.detach()
    return {
        "seed": seed,
        "train_counts": dict(Counter(outcome_bucket(e) for e in train)),
        "train_size": len(train),
        "aug_size": len(aug),
        "best_epoch": best["epoch"],
        "final_bias_gap": float(bias[1] - bias[0]),
        "operating_point": {"threshold": best["threshold"], **op},
        "frontier": frontier(test, tscores),
        "mean_q_nano": mean(qn),
        "mean_q_mini": mean(qm),
        "score_stats": {"mean": mean(tscores), "std": pstdev(tscores)},
    }


def reuse_runs(runs, mapping, seeds):
    for target_key, (path, source_key) in mapping.items():
        if not path.exists():
            continue
        source = json.loads(path.read_text(encoding="utf-8")).get("runs", {}).get(source_key, [])
        have = {int(r["seed"]) for r in runs.get(target_key, [])}
        for run in source:
            if int(run["seed"]) in set(seeds) - have:
                runs.setdefault(target_key, []).append(run)
        if target_key in runs:
            runs[target_key] = sorted(runs[target_key], key=lambda r: int(r["seed"]))


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
    parser.add_argument("--aug-weight", type=float, default=1.0)
    parser.add_argument("--aug-batch-size", type=int, default=32)
    parser.add_argument("--reuse-random50", type=Path, default=ROOT / "outputs/computer_qa_prior_mo_random.json")
    parser.add_argument("--reuse-active", type=Path, default=ROOT / "outputs/computer_qa_active_selection.json")
    parser.add_argument("--smoke", action="store_true", help="1 seed, 2 epochs, one config per family")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/computer_qa_disagree_scale_augment.json")
    args = parser.parse_args()

    disagree_sweep, aug_sizes = DISAGREE_SWEEP, AUG_SIZES
    if args.smoke:
        args.seeds, args.epochs = [505], 2
        disagree_sweep, aug_sizes = [5], [50]
        if args.output == parser.get_default("output"):
            args.output = ROOT / "outputs/computer_qa_disagree_scale_augment_smoke.json"

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
        "status": "post-hoc held-out diagnostic on Computer QA",
        "router": "frozen DistilBERT, 8 soft tokens, two correctness heads, inverse-cost weighted BCE",
        "operating_point": "validation-max-accuracy checkpoint and threshold; held-out 205",
        "questions": {
            "scaling": f"D gold disagreements + {AGREE_FIXED} gold agreements, D in {DISAGREE_SWEEP} "
                       "(pool holds ~24-26 disagreements, so D=25 is exhaustion); "
                       "random controls at matched totals",
            "augment": f"d{AUG_BASE_DISAGREE}_a{AGREE_FIXED} gold base + N unlabeled agreement rows with "
                       f"consistency loss (q_nano - q_mini)^2, weight {args.aug_weight}, N in {AUG_SIZES}",
        },
        "augment_labels": "gold of augmented rows never used; agreement bit is observable from cached preds",
        "priors": {name: repr(cfg) for name, cfg in PRIORS},
        "reuse": "random50 from computer_qa_prior_mo_random.json; d25_a25 from "
                 "computer_qa_active_selection.json (disagree_oracle; identical salt and training path)",
        "answer_model_api_calls": 0,
    }
    oracle_scores = [1.0 if outcome_bucket(e) == "mini_only" else 0.0 for e in heldout]
    result["baselines"] = {
        "all_nano": evaluate_at(heldout, [0.0] * len(heldout), 1.0),
        "all_mini": evaluate_at(heldout, [1.0] * len(heldout), 0.0),
        "oracle": evaluate_at(heldout, oracle_scores, 0.5),
    }
    result["disagree_available"] = {
        str(seed): sum(1 for e in flat(per_seed[seed][0]) if e.nano_pred != e.mini_pred)
        for seed in args.seeds
    }
    runs = result.setdefault("runs", {})

    if not args.smoke:
        mapping = {}
        for prior_name, _cfg in PRIORS:
            mapping[f"d25_a{AGREE_FIXED}__{prior_name}"] = (args.reuse_active, f"disagree_oracle__{prior_name}")
            mapping[f"random50__{prior_name}"] = (args.reuse_random50, f"random50__{prior_name}")
        reuse_runs(runs, mapping, args.seeds)

    conditions = []
    for d in disagree_sweep:
        conditions.append((f"d{d}_a{AGREE_FIXED}",
                           lambda o, s, d=d: (select_disagree_scale(o, s, d), [])))
        total = d + AGREE_FIXED
        conditions.append((f"random{total}",
                           lambda o, s, n=total: (select_random(o, s, n), [])))
    weight_tag = "" if args.aug_weight == 1.0 else f"_w{args.aug_weight}"
    for n in aug_sizes:
        def make_aug(o, s, n=n):
            train = select_disagree_scale(o, s, AUG_BASE_DISAGREE)
            aug = select_agree_augment(o, s, {e.question_id for e in train}, n)
            return train, aug
        conditions.append((f"d{AUG_BASE_DISAGREE}_a{AGREE_FIXED}_aug{n}{weight_tag}", make_aug))

    for cond_name, build in conditions:
        for prior_name, prior_cfg in PRIORS:
            key = f"{cond_name}__{prior_name}"
            done = {int(r["seed"]) for r in runs.get(key, [])}
            for seed in args.seeds:
                if seed in done:
                    continue
                orders, validation = per_seed[seed]
                train, aug = build(orders, seed)
                train_ids = {e.question_id for e in train}
                if train_ids & {e.question_id for e in validation}:
                    raise AssertionError("Train/validation leakage")
                if {e.question_id for e in aug} & (train_ids | {e.question_id for e in validation}):
                    raise AssertionError("Augment overlaps train or validation")
                print(f"{key} seed={seed} counts={dict(Counter(outcome_bucket(e) for e in train))} "
                      f"aug={len(aug)}", flush=True)
                run = train_router_aug(train, aug, validation, heldout, seed,
                                       prior_cfg, rates_by_seed[seed], args)
                op = run["operating_point"]
                print(f"  acc={op['accuracy']:.4f} rate={op['mini_rate']:.4f} "
                      f"save={op['mini_saving_vs_all_mini']:.4f} MOrec={op['mini_only_recall']:.4f}",
                      flush=True)
                runs.setdefault(key, []).append(run)
                runs[key] = sorted(runs[key], key=lambda r: int(r["seed"]))
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    result["summary"] = {}
    for key, rs in sorted(runs.items()):
        if not rs:
            continue
        result["summary"][key] = {
            "seeds": [int(r["seed"]) for r in rs],
            "train_size_mean": mean(r["train_size"] for r in rs),
            "aug_size_mean": mean(r.get("aug_size", 0) for r in rs),
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
