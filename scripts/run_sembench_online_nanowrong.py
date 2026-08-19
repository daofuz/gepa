#!/usr/bin/env python3
"""Online active acquisition for a "will Nano get this wrong?" router (Movie).

Idea (2026-08-15): routing both-wrong rows to Mini changes accuracy by
exactly zero, so the router target can be "Nano wrong" (escalate) instead of
"which model is right".  Under that target the both-wrong ballast that
poisoned hard-example selectors (nano_first 87.0% < random) becomes CORRECT
positive labels, positives are 2x denser (15% vs 6.8%), a training label only
needs a gold check of Nano's answer (no Mini call), and perfect Nano-wrong
routing still reaches full oracle accuracy (93.0% held-out) at the
Nano-wrong rate (15%) -- the only price is wasted Mini spend on both-wrong
escalations.

Simulated online loop, all from cache: scorer = frozen DistilBERT + 8 soft
tokens + 1 head on [text + Nano answer] predicting P(Nano wrong).  25
training labels: 5 random seed labels, then 4 rounds of 5 picked by the arm's
policy (retraining the scorer between rounds):

  random25          -- all 25 uniform random (control)
  online_top        -- label the highest-scoring (hardest-looking) unlabeled rows
  online_eps        -- online_top with 30% random exploration
  online_uncertain  -- label rows with scores closest to 0.5

Hypothesis under test: with the Nano-wrong target, select-hard acquisition
flips from harmful to helpful vs random.  Per seed the enriched validation
(18 disagreements + 7 agreements) is built BEFORE acquisition and shared by
all arms.  Waste metric: both-wrong share of escalated traffic at each
operating point.  Five seeds, held-out 200, no API calls.
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

from run_sembench_50label_budget import (
    build_validation,
    quantile_threshold,
    threshold_candidates,
)
from run_sembench_accuracy_targeted_router import evaluate_scores
from run_sembench_external_baselines import SingleHeadRouter, make_loader, probabilities
from run_sembench_mo_importance import frontier
from run_sembench_qa50_softprompt_router import deduplicate
from run_sembench_train100_ratio_sweep import make_orders
from train_sembench_balanced_direct_router import evaluate, load_examples, text


BUDGETS = [0.3, 0.5, 0.7]
SEED_LABELS = 5
ROUNDS = 4
ROUND_BATCH = 5
EPSILON = 0.3
VAL_DISAGREE, VAL_AGREE = 18, 7
ARMS = ("random25", "online_top", "online_eps", "online_uncertain")


def nanowrong_text(e):
    return f"{text(e)} [nano answer] {e['nano_pred']}"


def nanowrong_label(e):
    return 1.0 if e["nano_pred"] != e["gold"] else 0.0


def flat(orders):
    return [row for rows in orders.values() for row in rows]


def fit_scorer(labeled, seed, args, tokenizer):
    random.seed(seed)
    torch.manual_seed(seed)
    model = SingleHeadRouter(args.hf_model, args.prompt_tokens)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    tl = make_loader(labeled, tokenizer, args.max_length, args.batch_size, True,
                     nanowrong_text, nanowrong_label)
    for _epoch in range(args.epochs):
        model.train()
        for batch in tl:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = F.binary_cross_entropy_with_logits(logits, batch["labels"])
            loss.backward()
            optimizer.step()
    return model


ARM_SALTS = {"random25": 11, "online_top": 22, "online_eps": 33, "online_uncertain": 44}


def acquire(arm, candidates, seed, args, tokenizer):
    """Return the 25 labeled rows chosen by the arm's policy."""
    rng = random.Random(seed + 7529 + ARM_SALTS[arm])
    if arm == "random25":
        return rng.sample(candidates, SEED_LABELS + ROUNDS * ROUND_BATCH)
    labeled = rng.sample(candidates, SEED_LABELS)
    labeled_ids = {r["review_id"] for r in labeled}
    for round_index in range(ROUNDS):
        unlabeled = [r for r in candidates if r["review_id"] not in labeled_ids]
        model = fit_scorer(labeled, seed + round_index, args, tokenizer)
        dl = make_loader(unlabeled, tokenizer, args.max_length, args.batch_size, False,
                         nanowrong_text, nanowrong_label)
        scores = probabilities(model, dl)
        if arm == "online_top":
            order = sorted(range(len(unlabeled)), key=lambda i: -scores[i])
            picks = order[:ROUND_BATCH]
        elif arm == "online_uncertain":
            order = sorted(range(len(unlabeled)), key=lambda i: abs(scores[i] - 0.5))
            picks = order[:ROUND_BATCH]
        elif arm == "online_eps":
            order = sorted(range(len(unlabeled)), key=lambda i: -scores[i])
            picks, cursor = [], 0
            remaining = set(range(len(unlabeled)))
            for _ in range(ROUND_BATCH):
                if rng.random() < EPSILON:
                    i = rng.choice(sorted(remaining))
                else:
                    while order[cursor] not in remaining:
                        cursor += 1
                    i = order[cursor]
                picks.append(i)
                remaining.discard(i)
        else:
            raise ValueError(arm)
        for i in picks:
            labeled.append(unlabeled[i])
            labeled_ids.add(unlabeled[i]["review_id"])
        print(f"    round {round_index + 1}: labels={len(labeled)} "
              f"nano_wrong={sum(1 for r in labeled if nanowrong_label(r))}", flush=True)
    return labeled


def final_train(train, validation, pool_sample, test, seed, args):
    random.seed(seed)
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, local_files_only=True)
    model = SingleHeadRouter(args.hf_model, args.prompt_tokens)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    mk = lambda rows, sh: make_loader(rows, tokenizer, args.max_length, args.batch_size,
                                      sh, nanowrong_text, nanowrong_label)
    tl, vl, pl, xl = mk(train, True), mk(validation, False), mk(pool_sample, False), mk(test, False)
    val_scores, pool_scores, test_scores = [], [], []
    for _epoch in range(args.epochs):
        model.train()
        for batch in tl:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = F.binary_cross_entropy_with_logits(logits, batch["labels"])
            loss.backward()
            optimizer.step()
        for store, dl in ((val_scores, vl), (pool_scores, pl), (test_scores, xl)):
            store.append(probabilities(model, dl))
    return val_scores, pool_scores, test_scores


def escalation_waste(test, scores, threshold):
    escalated = [r for r, s in zip(test, scores) if s >= threshold]
    if not escalated:
        return 0.0
    return sum(1 for r in escalated if r["bucket"] == "both_wrong") / len(escalated)


def pick_operating_points(validation, val_scores, pool_scores, test, test_scores):
    results = {}

    def val_key(scores, t):
        m = evaluate_scores(validation, scores, t)
        return (m["selected_accuracy"], -m["mini_calls"])

    def finish(name, epoch, t):
        op = evaluate_scores(test, test_scores[epoch], t)
        results[name] = {
            "best_epoch": epoch + 1, "threshold": t,
            "operating_point": {
                "accuracy": op["selected_accuracy"],
                "mini_rate": op["mini_rate"],
                "mini_only_recall": op["mini_only_recall"],
                "bw_share_of_escalations": escalation_waste(test, test_scores[epoch], t),
            },
            "frontier": frontier(test, test_scores[epoch]),
        }

    best = None
    for e, vs in enumerate(val_scores):
        for t in threshold_candidates(vs):
            key = val_key(vs, t)
            if best is None or key > best[0]:
                best = (key, e, t)
    finish("maxacc", best[1], best[2])

    for budget in BUDGETS:
        best = None
        for e, vs in enumerate(val_scores):
            t = quantile_threshold(pool_scores[e], budget)
            key = val_key(vs, t)
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
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/sembench_online_nanowrong.json")
    args = parser.parse_args()

    heldout_ids = set(json.loads(args.manifest.read_text(encoding="utf-8"))["review_ids"])
    examples = deduplicate(load_examples(args.reviews, args.cache))
    heldout = [r for r in examples if r["review_id"] in heldout_ids]
    historical = [r for r in examples if r["review_id"] not in heldout_ids]
    if len(heldout) != 200 or len(historical) != 812:
        raise AssertionError("Expected 812 historical and 200 held-out rows")
    per_seed = {seed: make_orders(historical, seed) for seed in args.seeds}
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, local_files_only=True)

    result = {}
    if args.output.exists():
        result = json.loads(args.output.read_text(encoding="utf-8"))
    result["protocol"] = {
        "status": "online active acquisition for the Nano-wrong routing target (cache simulation)",
        "target": "P(Nano wrong) on [text + Nano answer]; escalate high scores; both-wrong rows are "
                  "legitimate positives (routing them to Mini costs money, never accuracy)",
        "budget": f"25 training gold checks of Nano's answer only ({SEED_LABELS} random seed + "
                  f"{ROUNDS}x{ROUND_BATCH} by policy); enriched validation "
                  f"({VAL_DISAGREE} disagreements + {VAL_AGREE} agreements) fixed per seed "
                  "BEFORE acquisition and shared by all arms",
        "arms": list(ARMS),
        "waste_metric": "both-wrong share of escalated traffic at each operating point",
        "reference": "perfect Nano-wrong routing = oracle accuracy 93.0% at 15% Mini rate (held-out)",
        "answer_model_api_calls": 0,
    }
    result["baselines"] = {
        "all_nano": evaluate(heldout, [0] * len(heldout)),
        "all_mini": evaluate(heldout, [1] * len(heldout)),
        "perfect_nano_wrong": evaluate(heldout, [1 if r["nano_pred"] != r["gold"] else 0
                                                 for r in heldout]),
    }
    runs = result.setdefault("runs", {})

    for arm in ARMS:
        done = {int(r["seed"]) for r in runs.get(arm, [])}
        for seed in args.seeds:
            if seed in done:
                continue
            orders, standard_validation, _ = per_seed[seed]
            validation = build_validation(orders, standard_validation, set(), seed,
                                          VAL_DISAGREE, VAL_AGREE)
            val_ids = {r["review_id"] for r in validation}
            candidates = [r for r in flat(orders) if r["review_id"] not in val_ids]
            print(f"{arm} seed={seed}", flush=True)
            train = acquire(arm, candidates, seed, args, tokenizer)
            train_ids = {r["review_id"] for r in train}
            if train_ids & val_ids:
                raise AssertionError("Train/validation overlap")
            pool_rng = random.Random(seed + 86243)
            pool_sample = pool_rng.sample(
                [r for r in candidates if r["review_id"] not in train_ids], 200)
            vs, ps, ts = final_train(train, validation, pool_sample, heldout, seed, args)
            criteria = pick_operating_points(validation, vs, ps, heldout, ts)
            run = {"seed": seed,
                   "train_counts": dict(Counter(r["bucket"] for r in train)),
                   "nano_wrong_in_train": sum(1 for r in train if nanowrong_label(r)),
                   "criteria": criteria}
            for name in ("maxacc", "q30"):
                op = criteria[name]["operating_point"]
                print(f"  {name:7s} acc={op['accuracy']:.4f} rate={op['mini_rate']:.4f} "
                      f"bw_waste={op['bw_share_of_escalations']:.2f}", flush=True)
            runs.setdefault(arm, []).append(run)
            runs[arm] = sorted(runs[arm], key=lambda r: int(r["seed"]))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    result["summary"] = {}
    criteria_names = ["maxacc"] + [f"q{int(b*100)}" for b in BUDGETS]
    for arm, rs in sorted(runs.items()):
        result["summary"][arm] = {
            "nano_wrong_in_train": [r["nano_wrong_in_train"] for r in rs],
        }
        for name in criteria_names:
            ops = [r["criteria"][name]["operating_point"] for r in rs]
            result["summary"][arm][name] = {
                f: {"mean": mean(o[f] for o in ops), "std": pstdev(o[f] for o in ops)}
                for f in ("accuracy", "mini_rate", "mini_only_recall", "bw_share_of_escalations")
            }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
