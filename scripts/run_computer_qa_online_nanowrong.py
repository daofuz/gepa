#!/usr/bin/env python3
"""Computer QA replication of the online Nano-wrong acquisition study.

Mirrors scripts/run_sembench_online_nanowrong.py: single-head P(Nano wrong)
scorer on [question + Nano answer], 25 training gold checks of Nano's answer
(5 random seed + 4 rounds x 5 by policy), enriched validation (18/7) fixed
per seed before acquisition and shared across arms, waste metric = both-wrong
share of escalated traffic.  Note the CQA reference: perfect Nano-wrong
routing = oracle accuracy 83.9% at the Nano-wrong rate (~29%).  Five seeds,
held-out 205, no API calls.
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from optimize_ollama_router_gepa import load_examples, outcome_bucket
from run_computer_qa_50label_budget import (
    build_validation,
    quantile_threshold,
    threshold_candidates,
)
from run_computer_qa_external_baselines import make_loader, probabilities
from run_computer_qa_prior_mo_random import evaluate_at, frontier, make_orders
from run_sembench_external_baselines import SingleHeadRouter
from train_softprompt_router import example_text


BUDGETS = [0.3, 0.5, 0.7]
SEED_LABELS = 5
ROUNDS = 4
ROUND_BATCH = 5
EPSILON = 0.3
VAL_DISAGREE, VAL_AGREE = 18, 7
ARMS = ("random25", "online_top", "online_eps", "online_uncertain")
ARM_SALTS = {"random25": 11, "online_top": 22, "online_eps": 33, "online_uncertain": 44}


def nanowrong_text(e):
    return example_text(e, include_nano_answer=True, nano_response_max_chars=0)


def nanowrong_label(e):
    return 1.0 if e.nano_pred != e.answer else 0.0


def flat(orders):
    return [e for rows in orders.values() for e in rows]


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


def acquire(arm, candidates, seed, args, tokenizer):
    rng = random.Random(seed + 7529 + ARM_SALTS[arm])
    if arm == "random25":
        return rng.sample(candidates, SEED_LABELS + ROUNDS * ROUND_BATCH)
    labeled = rng.sample(candidates, SEED_LABELS)
    labeled_ids = {e.question_id for e in labeled}
    for round_index in range(ROUNDS):
        unlabeled = [e for e in candidates if e.question_id not in labeled_ids]
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
            labeled_ids.add(unlabeled[i].question_id)
        print(f"    round {round_index + 1}: labels={len(labeled)} "
              f"nano_wrong={sum(1 for e in labeled if nanowrong_label(e))}", flush=True)
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
    escalated = [e for e, s in zip(test, scores) if s >= threshold]
    if not escalated:
        return 0.0
    return sum(1 for e in escalated if outcome_bucket(e) == "both_wrong") / len(escalated)


def pick_operating_points(validation, val_scores, pool_scores, test, test_scores):
    results = {}

    def val_key(scores, t):
        m = evaluate_at(validation, scores, t)
        return (m["accuracy"], -m["mini_calls"])

    def finish(name, epoch, t):
        op = evaluate_at(test, test_scores[epoch], t)
        results[name] = {
            "best_epoch": epoch + 1, "threshold": t,
            "operating_point": {
                **op,
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
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/computer_qa_online_nanowrong.json")
    args = parser.parse_args()

    examples = load_examples(args.mini_file, args.nano_file)
    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    heldout_ids = {int(i) for i in split["test_ids"]}
    heldout = [e for e in examples if e.question_id in heldout_ids]
    pool = [e for e in examples if e.question_id not in heldout_ids]
    if len(heldout) != 205 or len(pool) != 205:
        raise AssertionError(f"Expected 205/205 split, got {len(heldout)}/{len(pool)}")
    per_seed = {seed: make_orders(pool, seed) for seed in args.seeds}
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, local_files_only=True)

    result = {}
    if args.output.exists():
        result = json.loads(args.output.read_text(encoding="utf-8"))
    result["protocol"] = {
        "status": "online active acquisition for the Nano-wrong routing target on Computer QA",
        "target": "P(Nano wrong) on [question + Nano answer]; both-wrong rows are legitimate positives",
        "budget": f"25 training gold checks of Nano's answer ({SEED_LABELS} random + "
                  f"{ROUNDS}x{ROUND_BATCH} by policy); enriched validation "
                  f"({VAL_DISAGREE}/{VAL_AGREE}) fixed per seed before acquisition, shared by arms",
        "arms": list(ARMS),
        "waste_metric": "both-wrong share of escalated traffic",
        "answer_model_api_calls": 0,
    }
    result["baselines"] = {
        "all_nano": evaluate_at(heldout, [0.0] * len(heldout), 1.0),
        "all_mini": evaluate_at(heldout, [1.0] * len(heldout), 0.0),
        "perfect_nano_wrong": evaluate_at(
            heldout, [1.0 if e.nano_pred != e.answer else 0.0 for e in heldout], 0.5),
    }
    runs = result.setdefault("runs", {})

    for arm in ARMS:
        done = {int(r["seed"]) for r in runs.get(arm, [])}
        for seed in args.seeds:
            if seed in done:
                continue
            orders, standard_validation = per_seed[seed]
            validation = build_validation(orders, standard_validation, set(), seed,
                                          VAL_DISAGREE, VAL_AGREE)
            val_ids = {e.question_id for e in validation}
            candidates = [e for e in flat(orders) if e.question_id not in val_ids]
            print(f"{arm} seed={seed}", flush=True)
            train = acquire(arm, candidates, seed, args, tokenizer)
            train_ids = {e.question_id for e in train}
            if train_ids & val_ids:
                raise AssertionError("Train/validation overlap")
            pool_rng = random.Random(seed + 86243)
            rest = [e for e in candidates if e.question_id not in train_ids]
            pool_sample = pool_rng.sample(rest, min(100, len(rest)))
            vs, ps, ts = final_train(train, validation, pool_sample, heldout, seed, args)
            criteria = pick_operating_points(validation, vs, ps, heldout, ts)
            run = {"seed": seed,
                   "train_counts": dict(Counter(outcome_bucket(e) for e in train)),
                   "nano_wrong_in_train": sum(1 for e in train if nanowrong_label(e)),
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
