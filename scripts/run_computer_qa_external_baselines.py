#!/usr/bin/env python3
"""External routing baselines under the unified 50-label protocol (Computer QA).

Mirrors scripts/run_sembench_external_baselines.py: RouteLLM-style pre-route
(single strong-wins head, plain BCE, quantile cost calibration) and
FrugalGPT-style cascade (scorer on question + Nano answer predicting P(Nano
correct), escalate below threshold; cost adds the always-paid Nano call).
BARGAIN: no token logprobs in the cache, so its proxy-confidence procedure is
not reproducible; the oracle-agreement cascade it approximates is reported as
a non-deployable reference.  Same splits, seeds, 50-gold budget (train 25
random + validation 18 disagreements/7 agreements) and cost model as
outputs/computer_qa_50label_budget.json.  Held-out 205, no API calls.
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
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from optimize_ollama_router_gepa import load_examples, outcome_bucket
from run_computer_qa_50label_budget import (
    build_pool_sample,
    build_validation,
    quantile_threshold,
    threshold_candidates,
)
from run_computer_qa_prior_mo_random import (
    evaluate_at,
    frontier,
    make_orders,
    select_random,
)
from run_sembench_external_baselines import SingleHeadRouter
from train_softprompt_router import example_text


BUDGETS = [0.3, 0.5, 0.7]
SPLIT = (25, 18, 7)


def routellm_text(e):
    return example_text(e, include_nano_answer=False, nano_response_max_chars=0)


def frugalgpt_text(e):
    return example_text(e, include_nano_answer=True, nano_response_max_chars=0)


def routellm_label(e):
    mini_ok = e.mini_pred == e.answer
    nano_ok = e.nano_pred == e.answer
    return 1.0 if (mini_ok and not nano_ok) else 0.0


def frugalgpt_label(e):
    return 1.0 if e.nano_pred == e.answer else 0.0


class TextDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length, text_fn, label_fn):
        self.examples, self.tokenizer = examples, tokenizer
        self.max_length, self.text_fn, self.label_fn = max_length, text_fn, label_fn

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        e = self.examples[i]
        enc = self.tokenizer(self.text_fn(e), truncation=True,
                             max_length=self.max_length, padding=False)
        return {**enc, "label": self.label_fn(e)}


def make_loader(examples, tokenizer, max_length, batch_size, shuffle, text_fn, label_fn):
    def collate(items):
        length = max(len(it["input_ids"]) for it in items)
        pad = tokenizer.pad_token_id
        return {
            "input_ids": torch.tensor(
                [it["input_ids"] + [pad] * (length - len(it["input_ids"])) for it in items],
                dtype=torch.long),
            "attention_mask": torch.tensor(
                [it["attention_mask"] + [0] * (length - len(it["attention_mask"])) for it in items],
                dtype=torch.long),
            "labels": torch.tensor([it["label"] for it in items], dtype=torch.float),
        }
    return DataLoader(TextDataset(examples, tokenizer, max_length, text_fn, label_fn),
                      batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


def probabilities(model, dl):
    out = []
    model.eval()
    with torch.no_grad():
        for batch in dl:
            out.extend(torch.sigmoid(model(batch["input_ids"], batch["attention_mask"])).tolist())
    return out


def train_baseline(method, train, validation, pool_sample, test, seed, args):
    text_fn = routellm_text if method == "routellm" else frugalgpt_text
    label_fn = routellm_label if method == "routellm" else frugalgpt_label
    to_score = (lambda p: p) if method == "routellm" else (lambda p: -p)

    random.seed(seed)
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, local_files_only=True)
    model = SingleHeadRouter(args.hf_model, args.prompt_tokens)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    mk = lambda rows, shuffle: make_loader(rows, tokenizer, args.max_length,
                                           args.batch_size, shuffle, text_fn, label_fn)
    tl, vl, pl, xl = mk(train, True), mk(validation, False), mk(pool_sample, False), mk(test, False)

    val_scores, pool_scores, test_scores = [], [], []
    for _epoch in range(1, args.epochs + 1):
        model.train()
        for batch in tl:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = F.binary_cross_entropy_with_logits(logits, batch["labels"])
            loss.backward()
            optimizer.step()
        for store, dl in ((val_scores, vl), (pool_scores, pl), (test_scores, xl)):
            store.append([to_score(p) for p in probabilities(model, dl)])
    return val_scores, pool_scores, test_scores


def pick_operating_points(validation, val_scores, pool_scores, test, test_scores):
    results = {}

    def val_key(scores, t):
        m = evaluate_at(validation, scores, t)
        return (m["accuracy"], -m["mini_calls"])

    def finish(name, epoch, t):
        op = evaluate_at(test, test_scores[epoch], t)
        results[name] = {
            "best_epoch": epoch + 1, "threshold": t,
            "operating_point": op,
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
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/computer_qa_external_baselines.json")
    args = parser.parse_args()

    examples = load_examples(args.mini_file, args.nano_file)
    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    heldout_ids = {int(i) for i in split["test_ids"]}
    heldout = [e for e in examples if e.question_id in heldout_ids]
    pool = [e for e in examples if e.question_id not in heldout_ids]
    if len(heldout) != 205 or len(pool) != 205:
        raise AssertionError(f"Expected 205/205 split, got {len(heldout)}/{len(pool)}")
    per_seed = {seed: make_orders(pool, seed) for seed in args.seeds}
    n_train, n_dis, n_agr = SPLIT

    result = {}
    if args.output.exists():
        result = json.loads(args.output.read_text(encoding="utf-8"))
    result["protocol"] = {
        "status": "external baselines under the unified 50-label protocol on Computer QA",
        "parity": "same model pair, splits, seeds, 50-gold budget (t25_v25), and cost model "
                  "as outputs/computer_qa_50label_budget.json",
        "methods": {
            "routellm": "pre-route single strong-wins head, plain BCE, quantile cost calibration",
            "frugalgpt": "cascade scorer on question + Nano answer predicting P(Nano correct); "
                         "cost adds the always-paid Nano call (0.25 + mini_rate mini-equivalents)",
            "bargain": "logprob-free cache -> proxy-confidence procedure not reproducible; "
                       "oracle-agreement cascade reported as its non-deployable reference",
        },
        "answer_model_api_calls": 0,
    }
    disagree_scores = [1.0 if e.nano_pred != e.mini_pred else 0.0 for e in heldout]
    result["baselines"] = {
        "all_nano": evaluate_at(heldout, [0.0] * len(heldout), 1.0),
        "all_mini": evaluate_at(heldout, [1.0] * len(heldout), 0.0),
        "bargain_oracle_agreement": {
            **evaluate_at(heldout, disagree_scores, 0.5),
            "note": "non-deployable: requires both answers on every row",
        },
    }
    runs = result.setdefault("runs", {})

    for method in ("routellm", "frugalgpt"):
        done = {int(r["seed"]) for r in runs.get(method, [])}
        for seed in args.seeds:
            if seed in done:
                continue
            orders, standard_validation = per_seed[seed]
            train = select_random(orders, seed, n_train)
            train_ids = {e.question_id for e in train}
            validation = build_validation(orders, standard_validation, train_ids,
                                          seed, n_dis, n_agr)
            val_ids = {e.question_id for e in validation}
            if train_ids & val_ids:
                raise AssertionError("Train/validation overlap")
            pool_sample = build_pool_sample(orders, train_ids | val_ids, seed)
            print(f"{method} seed={seed} train={len(train)} val={len(validation)}", flush=True)
            vs, ps, ts = train_baseline(method, train, validation, pool_sample,
                                        heldout, seed, args)
            criteria = pick_operating_points(validation, vs, ps, heldout, ts)
            run = {"seed": seed,
                   "train_counts": dict(Counter(outcome_bucket(e) for e in train)),
                   "criteria": criteria}
            for name in ("maxacc", "q50"):
                op = criteria[name]["operating_point"]
                print(f"  {name:7s} acc={op['accuracy']:.4f} rate={op['mini_rate']:.4f}", flush=True)
            runs.setdefault(method, []).append(run)
            runs[method] = sorted(runs[method], key=lambda r: int(r["seed"]))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    result["summary"] = {}
    criteria_names = ["maxacc"] + [f"q{int(b*100)}" for b in BUDGETS]
    for method, rs in sorted(runs.items()):
        result["summary"][method] = {}
        for name in criteria_names:
            ops = [r["criteria"][name]["operating_point"] for r in rs]
            result["summary"][method][name] = {
                f: {"mean": mean(o[f] for o in ops), "std": pstdev(o[f] for o in ops)}
                for f in ("accuracy", "mini_rate", "mini_only_recall")
            }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
