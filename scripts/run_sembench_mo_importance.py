#!/usr/bin/env python3
"""How important are Mini-only (MO) training examples for the router?

Two controlled designs, both mirroring the existing 8-token two-head
soft-prompt router (frozen DistilBERT, inverse-expected-cost weighted BCE,
NO prior loss, validation-max-accuracy checkpoint):

  DESIGN A -- fixed-50 (extends the critical sweep DOWN to MO=0):
      MO = k, BC = 47-k, NO = 2, BW = 1   (total 50)
      k in {0, 1, 4, 8, 22}.  k=0 is the key control the old sweep could not
      run (it required every bucket >= 1); k=1 and k=22 are calibration anchors
      that must reproduce the old sweep (87.83% / 90.00%).

  DESIGN B -- additive (isolates MO from the BC trade-off):
      BC = 40, NO = 2, BW = 1 held FIXED; MO = m added on top (total 43+m).
      m in {0, 2, 4, 8, 16, 24}.  Because BC/NO/BW are constant, any change is
      attributable to adding Mini-only examples, not to removing both-correct.

For every trained router we also record the RAW test scores and build a
ranking-based cost-accuracy frontier: at each Mini budget fraction we route the
highest-scoring examples to Mini and measure accuracy + MO recall.  Comparing
frontiers isolates *targeting quality* from the deployed escalation rate, which
the raw single-threshold sweep confounds.

No answer-model API calls.  Held-out 200 is used only for post-hoc reporting.
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
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

import run_sembench_twohead_correctness_router as base
from run_sembench_accuracy_targeted_router import evaluate_scores, select_threshold
from run_sembench_qa50_softprompt_router import deduplicate
from run_sembench_softprompt_mini_prior_ablation import expected_costs
from run_sembench_train100_ratio_sweep import make_orders
from train_sembench_balanced_direct_router import evaluate, load_examples


BUDGETS = [round(0.05 * i, 2) for i in range(0, 21)]  # 0.00 .. 1.00


def select_fixed50(orders, mo, seed):
    counts = {"mini_only": mo, "nano_only": 2, "both_wrong": 1, "both_correct": 47 - mo}
    return _assemble(orders, counts, seed, expected=50)


def select_additive(orders, mo, seed, bc=40):
    counts = {"mini_only": mo, "nano_only": 2, "both_wrong": 1, "both_correct": bc}
    return _assemble(orders, counts, seed, expected=bc + 3 + mo)


def _assemble(orders, counts, seed, expected):
    selected = []
    for bucket, count in counts.items():
        if count < 0:
            raise ValueError(f"Negative count for {bucket}: {count}")
        if len(orders[bucket]) < count:
            raise ValueError(f"Need {count} {bucket}, found {len(orders[bucket])}")
        selected.extend(orders[bucket][:count])
    random.Random(seed + 15485863).shuffle(selected)
    ids = {row["review_id"] for row in selected}
    if len(selected) != expected or len(ids) != expected:
        raise AssertionError(f"Expected {expected} unique rows, got {len(selected)}/{len(ids)}")
    return selected


def frontier(test, scores):
    """Ranking-based cost-accuracy frontier.

    Route the highest-scoring examples to Mini first.  At each budget fraction
    b, exactly round(b*N) examples go to Mini.  Returns accuracy and MO recall
    per budget, independent of any learned threshold.
    """
    order = sorted(range(len(test)), key=lambda i: scores[i], reverse=True)
    n = len(test)
    mo_ids = [i for i in range(n) if test[i]["bucket"] == "mini_only"]
    out = []
    for b in BUDGETS:
        m = round(b * n)
        to_mini = set(order[:m])
        correct = 0
        for i, row in enumerate(test):
            pred = row["mini_pred"] if i in to_mini else row["nano_pred"]
            correct += int(pred == row["gold"])
        mo_routed = sum(1 for i in mo_ids if i in to_mini)
        out.append({
            "budget": b,
            "mini_rate": m / n,
            "accuracy": correct / n,
            "mo_recall": (mo_routed / len(mo_ids)) if mo_ids else 0.0,
        })
    return out


def train_router(train, validation, test, cache, seed, args):
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
    xl = base.loader(test, tokenizer, args.max_length, args.batch_size, False)
    costs = expected_costs(train, cache)
    inv = torch.tensor([1.0 / costs["nano"], 1.0 / costs["mini"]], dtype=torch.float)

    best = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in tl:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], batch["attention_mask"])
            raw = F.binary_cross_entropy_with_logits(logits, batch["labels"], reduction="none")
            loss = (raw * inv).sum() / (len(batch["labels"]) * inv.sum())
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
    return {
        "seed": seed,
        "train_counts": dict(Counter(r["bucket"] for r in train)),
        "train_size": len(train),
        "best_epoch": best["epoch"],
        "operating_point": {  # validation-max-accuracy deployment
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


def agg_frontier(runs):
    """Average the ranking frontier across seeds at each budget."""
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
    parser.add_argument("--fixed50-mo", type=int, nargs="+", default=[0, 1, 4, 8, 22])
    parser.add_argument("--additive-mo", type=int, nargs="+", default=[0, 2, 4, 8, 16, 24])
    parser.add_argument("--additive-bc", type=int, default=40)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/sembench_mo_importance.json")
    args = parser.parse_args()

    heldout_ids = set(json.loads(args.manifest.read_text(encoding="utf-8"))["review_ids"])
    cache = json.loads(args.cache.read_text(encoding="utf-8"))
    examples = deduplicate(load_examples(args.reviews, args.cache))
    heldout = [r for r in examples if r["review_id"] in heldout_ids]
    historical = [r for r in examples if r["review_id"] not in heldout_ids]
    if len(heldout) != 200 or len(historical) != 812:
        raise AssertionError("Expected 812 historical and 200 held-out rows")
    per_seed = {seed: make_orders(historical, seed) for seed in args.seeds}

    result = {}
    if args.output.exists():
        result = json.loads(args.output.read_text(encoding="utf-8"))
    result.setdefault("protocol", {
        "status": "post-hoc held-out MO-importance diagnostic; not a fresh final test",
        "model": "frozen DistilBERT, 8 soft tokens, two correctness heads",
        "loss": "inverse-expected-query-cost weighted BCE; no prior",
        "designs": {
            "fixed50": "MO=k, BC=47-k, NO=2, BW=1 (total 50); k=0 is the key control",
            "additive": f"BC={args.additive_bc}, NO=2, BW=1 fixed; MO added on top",
        },
        "frontier": "route top-scoring examples to Mini at each budget; isolates targeting from escalation rate",
        "answer_model_api_calls": 0,
    })
    result.setdefault("baselines", {
        "all_nano": evaluate(heldout, [0] * len(heldout)),
        "all_mini": evaluate(heldout, [1] * len(heldout)),
    })
    result.setdefault("heldout_counts", dict(Counter(r["bucket"] for r in heldout)))
    runs = result.setdefault("runs", {})

    plan = [("fixed50", mo, select_fixed50) for mo in args.fixed50_mo]
    plan += [("additive", mo, lambda o, m, s: select_additive(o, m, s, args.additive_bc))
             for mo in args.additive_mo]

    for design, mo, builder in plan:
        key = f"{design}_mo{mo}"
        done = {int(r["seed"]): r for r in runs.get(key, [])}
        collected = []
        for seed in args.seeds:
            if seed in done:
                collected.append(done[seed]); continue
            orders, validation, _ = per_seed[seed]
            train = builder(orders, mo, seed)
            if {r["review_id"] for r in train} & {r["review_id"] for r in validation}:
                raise AssertionError("Train/validation leakage")
            print(f"{key} seed={seed} counts={dict(Counter(r['bucket'] for r in train))}", flush=True)
            run = train_router(train, validation, heldout, cache, seed, args)
            op = run["operating_point"]
            print(f"  acc={op['accuracy']:.4f} rate={op['mini_rate']:.4f} "
                  f"save={op['mini_saving_vs_all_mini']:.4f} MOrec={op['mini_only_recall']:.4f}", flush=True)
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
            "train_size": rs[0]["train_size"],
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
