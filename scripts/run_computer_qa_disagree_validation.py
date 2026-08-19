#!/usr/bin/env python3
"""Computer QA replication of the disagreement-enriched validation study.

Mirrors scripts/run_sembench_disagree_validation.py: same training path as
run_computer_qa_disagree_scale_augment.py (a fresh randomness realization,
not bit-identical -- per-epoch evaluation shifts DataLoader RNG draws), two
validation designs scored every epoch against the same per-epoch models
(exactly paired comparison), (epoch, threshold) picked per design:

  standard      -- stratified 62-row validation (existing protocol;
                   ~11-14 informative rows; 62 gold rows)
  disagree_val  -- up to 25 disagreements + 10 agreements from unused pool
                   rows plus the standard validation rows (35 gold rows).
                   When the training config already consumes the pool's
                   disagreements (d25_a25), the supply clamps to ~12-15 --
                   on a small pool, training and validation genuinely compete
                   for the same disagreement rows.

Training configs: random50 (the CQA winner) and d25_a25, priors none and
mean_prob@1, seeds 505-909, held-out 205.  Per-epoch held-out score vectors
are stored for post-hoc policy analysis.  No answer-model API calls.
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
from run_computer_qa_disagree_scale_augment import select_disagree_scale
from run_computer_qa_prior_mo_random import (
    agg_frontier,
    correctness_probabilities,
    evaluate_at,
    frontier,
    loader,
    logit,
    make_orders,
    pool_base_rates,
    select_random,
    select_threshold,
)
from run_sembench_twohead_correctness_router import TwoHeadCorrectnessRouter


PRIORS = [
    ("none", {}),
    ("mean_prob_l1", {"mean_prob": 1.0}),
]
CONFIGS = [
    ("random50", lambda orders, seed: select_random(orders, seed, 50)),
    ("d25_a25", lambda orders, seed: select_disagree_scale(orders, seed, 25)),
]
VAL_DISAGREE = 25
VAL_AGREE = 10


def flat(orders):
    return [e for rows in orders.values() for e in rows]


def build_disagree_validation(orders, standard_validation, train_ids, seed):
    candidates = [e for e in flat(orders) if e.question_id not in train_ids]
    candidates += standard_validation
    disagree = [e for e in candidates if e.nano_pred != e.mini_pred]
    agree = [e for e in candidates if e.nano_pred == e.mini_pred]
    rng = random.Random(seed + 51797)
    d = rng.sample(disagree, min(VAL_DISAGREE, len(disagree)))
    a = rng.sample(agree, min(VAL_AGREE, len(agree)))
    rows = d + a
    rng.shuffle(rows)
    return rows


def informative(rows):
    return sum(1 for e in rows if e.nano_pred != e.mini_pred)


def train_with_designs(train, designs, test, seed, prior_cfg, rates, args):
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
    design_loaders = {
        name: loader(rows, tokenizer, args.max_length, args.batch_size, False)
        for name, rows in designs.items()
    }
    xl = loader(test, tokenizer, args.max_length, args.batch_size, False)
    costs = {
        "nano": mean(e.nano_cost_usd for e in train),
        "mini": mean(e.mini_cost_usd for e in train),
    }
    inv = torch.tensor([1.0 / costs["nano"], 1.0 / costs["mini"]], dtype=torch.float)
    prior_gap = logit(p_mini) - logit(p_nano)
    prior_probs = torch.tensor([p_nano, p_mini], dtype=torch.float)
    smooth = prior_cfg.get("label_smooth", 0.0)

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
        # Inference only: no RNG consumed, trajectory unchanged.
        for name, dl in design_loaders.items():
            qn, qm = correctness_probabilities(model, dl)
            epoch_scores[name].append([m - n for n, m in zip(qn, qm)])
        qn, qm = correctness_probabilities(model, xl)
        epoch_test_scores.append([m - n for n, m in zip(qn, qm)])

    out = {"seed": seed, "train_counts": dict(Counter(outcome_bucket(e) for e in train)),
           "train_size": len(train), "designs": {},
           "epoch_test_scores": epoch_test_scores}
    for name, rows in designs.items():
        best = None
        for epoch_index, vscores in enumerate(epoch_scores[name]):
            threshold, vmetrics = select_threshold(rows, vscores)
            key = (vmetrics["accuracy"], -vmetrics["mini_calls"])
            if best is None or key > best["key"]:
                best = {"key": key, "epoch": epoch_index + 1, "threshold": threshold}
        tscores = epoch_test_scores[best["epoch"] - 1]
        op = evaluate_at(test, tscores, best["threshold"])
        out["designs"][name] = {
            "validation_size": len(rows),
            "informative_rows": informative(rows),
            "best_epoch": best["epoch"],
            "threshold": best["threshold"],
            "operating_point": op,
            "frontier": frontier(test, tscores),
        }
    return out


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
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/computer_qa_disagree_validation.json")
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
        "status": "post-hoc held-out operating-point-policy diagnostic on Computer QA",
        "question": "does a disagreement-enriched validation set stabilize threshold+checkpoint selection?",
        "designs": {
            "standard": "stratified 62-row validation (existing protocol; ~11-14 informative rows; 62 gold)",
            "disagree_val": f"up to {VAL_DISAGREE} disagreements + {VAL_AGREE} agreements from unused pool "
                            "rows plus standard-validation rows (35 gold; clamps when training "
                            "consumed the pool's disagreements)",
        },
        "training": "identical path and RNG stream as run_computer_qa_disagree_scale_augment.py",
        "answer_model_api_calls": 0,
    }
    result["baselines"] = {
        "all_nano": evaluate_at(heldout, [0.0] * len(heldout), 1.0),
        "all_mini": evaluate_at(heldout, [1.0] * len(heldout), 0.0),
    }
    runs = result.setdefault("runs", {})

    for config_name, selector in CONFIGS:
        for prior_name, prior_cfg in PRIORS:
            key = f"{config_name}__{prior_name}"
            done = {int(r["seed"]) for r in runs.get(key, [])}
            for seed in args.seeds:
                if seed in done:
                    continue
                orders, standard_validation = per_seed[seed]
                train = selector(orders, seed)
                train_ids = {e.question_id for e in train}
                if train_ids & {e.question_id for e in standard_validation}:
                    raise AssertionError("Train/validation leakage")
                dval = build_disagree_validation(orders, standard_validation, train_ids, seed)
                if {e.question_id for e in dval} & train_ids:
                    raise AssertionError("Disagree-validation overlaps train")
                designs = {"standard": standard_validation, "disagree_val": dval}
                print(f"{key} seed={seed} informative: standard={informative(standard_validation)}"
                      f"/{len(standard_validation)} disagree_val={informative(dval)}/{len(dval)}",
                      flush=True)
                run = train_with_designs(train, designs, heldout, seed,
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
