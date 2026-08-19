#!/usr/bin/env python3
"""Replicate the prior x data-selection study on Computer QA.

Mirrors scripts/run_sembench_prior_mo_random.py (SemBench Movie) on the
Computer QA nano/mini routing task: 8-token two-head soft-prompt router
(frozen DistilBERT), inverse-expected-cost weighted BCE, validation-max-
accuracy checkpoint+threshold, fixed held-out test, ranking frontier.

  Splits: held-out = the 205 test_ids of
    routing_split_softprompt_threshold_balanced16.json (untouched);
    the other 205 examples form the historical pool.  Per seed a stratified
    30% of each bucket becomes validation; the rest is the selection pool.

  DATA conditions (same as Movie):
    bucket50_mo8  -- curated MO=8, BC=39, NO=2, BW=1 (total 50)
    random50      -- uniform random 50 from the selection pool
    random100     -- uniform random 100 from the selection pool

  PRIOR implementations (belief: Mini base correctness > Nano; base rates
  measured on the seed's selection pool only):
    none, bias_init, bias_map@{0.1,1}, mean_prob@{0.1,1}, label_smooth@0.1

No answer-model API calls; all outcomes come from the cached result files.
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
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from optimize_ollama_router_gepa import load_examples, outcome_bucket
from run_sembench_twohead_correctness_router import TwoHeadCorrectnessRouter
from train_softprompt_router import example_text


BUDGETS = [round(0.05 * i, 2) for i in range(0, 21)]
BUCKETS = ("both_correct", "mini_only", "nano_only", "both_wrong")
CURATED_COUNTS = {"mini_only": 8, "both_correct": 39, "nano_only": 2, "both_wrong": 1}
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


def nano_ok(e):
    return e.nano_pred == e.answer


def mini_ok(e):
    return e.mini_pred == e.answer


class CorrectnessDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        e = self.examples[index]
        encoded = self.tokenizer(
            example_text(e, include_nano_answer=False, nano_response_max_chars=0),
            truncation=True, max_length=self.max_length, padding=False,
        )
        return {**encoded, "nano_correct": float(nano_ok(e)), "mini_correct": float(mini_ok(e))}


def collate(items, pad_id):
    length = max(len(item["input_ids"]) for item in items)
    return {
        "input_ids": torch.tensor(
            [item["input_ids"] + [pad_id] * (length - len(item["input_ids"])) for item in items],
            dtype=torch.long),
        "attention_mask": torch.tensor(
            [item["attention_mask"] + [0] * (length - len(item["attention_mask"])) for item in items],
            dtype=torch.long),
        "labels": torch.tensor(
            [[item["nano_correct"], item["mini_correct"]] for item in items], dtype=torch.float),
    }


def loader(examples, tokenizer, max_length, batch_size, shuffle):
    return DataLoader(
        CorrectnessDataset(examples, tokenizer, max_length),
        batch_size=batch_size, shuffle=shuffle,
        collate_fn=lambda items: collate(items, tokenizer.pad_token_id),
    )


def correctness_probabilities(model, data_loader):
    nano, mini = [], []
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            values = torch.sigmoid(model(batch["input_ids"], batch["attention_mask"]))
            nano.extend(values[:, 0].tolist())
            mini.extend(values[:, 1].tolist())
    return nano, mini


def evaluate_at(test, scores, threshold):
    """Route to Mini when score >= threshold; report accuracy/cost metrics."""
    n = len(test)
    mini_calls = correct = mo_hit = 0
    mo_total = sum(1 for e in test if outcome_bucket(e) == "mini_only")
    for e, s in zip(test, scores):
        to_mini = s >= threshold
        mini_calls += int(to_mini)
        correct += int(mini_ok(e) if to_mini else nano_ok(e))
        if to_mini and outcome_bucket(e) == "mini_only":
            mo_hit += 1
    return {
        "accuracy": correct / n,
        "mini_rate": mini_calls / n,
        "mini_calls": mini_calls,
        "mini_saving_vs_all_mini": 1.0 - mini_calls / n,
        "mini_only_recall": (mo_hit / mo_total) if mo_total else 0.0,
    }


def select_threshold(validation, scores):
    unique = sorted(set(scores))
    candidates = [unique[0] - 1.0, unique[-1] + 1.0, *unique]
    candidates += [(a + b) / 2.0 for a, b in zip(unique, unique[1:])]
    best = None
    for t in sorted(set(candidates)):
        m = evaluate_at(validation, scores, t)
        key = (m["accuracy"], -m["mini_calls"], t)
        if best is None or key > best[0]:
            best = (key, t, m)
    return best[1], best[2]


def frontier(test, scores):
    order = sorted(range(len(test)), key=lambda i: scores[i], reverse=True)
    n = len(test)
    mo_ids = [i for i in range(n) if outcome_bucket(test[i]) == "mini_only"]
    out = []
    for b in BUDGETS:
        m = round(b * n)
        to_mini = set(order[:m])
        correct = sum(
            int(mini_ok(e) if i in to_mini else nano_ok(e)) for i, e in enumerate(test)
        )
        out.append({
            "budget": b,
            "mini_rate": m / n,
            "accuracy": correct / n,
            "mo_recall": (sum(1 for i in mo_ids if i in to_mini) / len(mo_ids)) if mo_ids else 0.0,
        })
    return out


def agg_frontier(runs):
    out = []
    for i, b in enumerate(BUDGETS):
        accs = [r["frontier"][i]["accuracy"] for r in runs]
        recs = [r["frontier"][i]["mo_recall"] for r in runs]
        out.append({
            "budget": b,
            "accuracy_mean": mean(accs), "accuracy_std": pstdev(accs),
            "mo_recall_mean": mean(recs),
        })
    return out


def make_orders(pool, seed):
    """Stratified per-bucket 30% validation; the rest is the selection pool."""
    rng = random.Random(seed + 7919)
    orders, validation = {}, []
    for bucket in BUCKETS:
        rows = [e for e in pool if outcome_bucket(e) == bucket]
        rng.shuffle(rows)
        val_n = max(1, round(0.3 * len(rows)))
        validation.extend(rows[:val_n])
        orders[bucket] = rows[val_n:]
    rng.shuffle(validation)
    return orders, validation


def select_curated(orders, seed):
    selected = []
    for bucket, count in CURATED_COUNTS.items():
        if len(orders[bucket]) < count:
            raise ValueError(f"Need {count} {bucket}, found {len(orders[bucket])}")
        selected.extend(orders[bucket][:count])
    random.Random(seed + 15485863).shuffle(selected)
    if len({e.question_id for e in selected}) != 50:
        raise AssertionError("Expected 50 unique curated rows")
    return selected


def select_random(orders, seed, n):
    pool = [e for rows in orders.values() for e in rows]
    rng = random.Random(seed + 49979687 + n)
    return rng.sample(pool, n)


def pool_base_rates(orders):
    pool = [e for rows in orders.values() for e in rows]
    return {
        "p_nano": mean(float(nano_ok(e)) for e in pool),
        "p_mini": mean(float(mini_ok(e)) for e in pool),
        "pool_size": len(pool),
    }


def train_router(train, validation, test, seed, prior_cfg, rates, args):
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
        "best_epoch": best["epoch"],
        "final_bias_gap": float(bias[1] - bias[0]),
        "operating_point": {"threshold": best["threshold"], **op},
        "frontier": frontier(test, tscores),
        "mean_q_nano": mean(qn),
        "mean_q_mini": mean(qm),
    }


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
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/computer_qa_prior_mo_random.json")
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
        "status": "post-hoc held-out prior x data-selection diagnostic on Computer QA",
        "model": "frozen DistilBERT, 8 soft tokens, two correctness heads",
        "loss": "inverse-expected-query-cost weighted BCE plus optional prior term",
        "heldout": "205 test_ids of routing_split_softprompt_threshold_balanced16.json",
        "validation": "stratified 30% of each bucket from the non-test pool, per seed",
        "data_conditions": {
            "bucket50_mo8": "curated MO=8, BC=39, NO=2, BW=1 (total 50)",
            "random50": "uniform random 50 from selection pool",
            "random100": "uniform random 100 from selection pool",
        },
        "prior_conditions": {name: repr(cfg) for name, cfg in PRIORS},
        "prior_base_rates": "measured on each seed's selection pool only",
        "operating_point": "validation-max-accuracy checkpoint and threshold",
        "frontier": "route top-scoring examples to Mini at each budget",
        "answer_model_api_calls": 0,
    })
    result.setdefault("baselines", {
        "all_nano": evaluate_at(heldout, [0.0] * len(heldout), 1.0),
        "all_mini": evaluate_at(heldout, [1.0] * len(heldout), 0.0),
    })
    result.setdefault("heldout_counts", dict(Counter(outcome_bucket(e) for e in heldout)))
    result["pool_base_rates"] = {str(s): rates_by_seed[s] for s in args.seeds}
    runs = result.setdefault("runs", {})

    selectors = [
        ("bucket50_mo8", lambda o, s: select_curated(o, s)),
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
                orders, validation = per_seed[seed]
                train = selector(orders, seed)
                if {e.question_id for e in train} & {e.question_id for e in validation}:
                    raise AssertionError("Train/validation leakage")
                print(f"{key} seed={seed} counts={dict(Counter(outcome_bucket(e) for e in train))}",
                      flush=True)
                run = train_router(train, validation, heldout, seed,
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
