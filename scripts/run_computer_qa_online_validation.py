#!/usr/bin/env python3
"""Computer QA replication of the online-learned validation acquisition study.

Mirrors scripts/run_sembench_online_validation.py: probe scorer on
[question + Nano answer] predicting P(disagree), self-labeled probes (a probe
= one Mini call; the agree/disagree bit is then free), 10 random seed probes
+ rounds of 20 by policy, stop at 18 disagreements or 150 probes (the CQA
candidate pool is only ~180 rows; random probing expects ~82 probes at the
22% base rate).  Downstream: train = same random 25 as t25_v25, two-head
router, no prior; reference = oracle enriched validation in
outputs/computer_qa_50label_budget.json (maxacc 78.6+-1.6, q70 77.7+-1.8,
q50 75.9+-1.8, q30 74.0+-1.9).  Five seeds, held-out 205, no API calls.
"""
from __future__ import annotations

import argparse
import json
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
from run_computer_qa_50label_budget import (
    build_pool_sample,
    pick_operating_points,
    train_and_score,
)
from run_computer_qa_external_baselines import make_loader, probabilities
from run_computer_qa_prior_mo_random import (
    evaluate_at,
    make_orders,
    pool_base_rates,
    select_random,
)
from run_sembench_external_baselines import SingleHeadRouter
from train_softprompt_router import example_text


N_TRAIN = 25
VAL_DISAGREE, VAL_AGREE = 18, 7
SEED_PROBES = 10
PROBE_BATCH = 20
PROBE_CAP = 150
SCORER_EPOCHS = 4
ARMS = ("probe_random", "probe_online")
ARM_SALTS = {"probe_random": 111, "probe_online": 222}


def probe_text(e):
    return example_text(e, include_nano_answer=True, nano_response_max_chars=0)


def disagree_label(e):
    return 1.0 if e.nano_pred != e.mini_pred else 0.0


def flat(orders):
    return [e for rows in orders.values() for e in rows]


def fit_probe_scorer(probed, seed, args, tokenizer):
    random.seed(seed)
    torch.manual_seed(seed)
    model = SingleHeadRouter(args.hf_model, args.prompt_tokens)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    tl = make_loader(probed, tokenizer, args.max_length, args.batch_size, True,
                     probe_text, disagree_label)
    for _epoch in range(SCORER_EPOCHS):
        model.train()
        for batch in tl:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = F.binary_cross_entropy_with_logits(logits, batch["labels"])
            loss.backward()
            optimizer.step()
    return model


def probe_for_disagreements(arm, candidates, seed, args, tokenizer):
    rng = random.Random(seed + ARM_SALTS[arm])
    order = candidates[:]
    rng.shuffle(order)
    probed = order[:SEED_PROBES]
    probed_ids = {e.question_id for e in probed}
    found = sum(1 for e in probed if disagree_label(e))
    round_index = 0
    while found < VAL_DISAGREE and len(probed) < PROBE_CAP:
        batch = min(PROBE_BATCH, PROBE_CAP - len(probed))
        unprobed = [e for e in candidates if e.question_id not in probed_ids]
        if not unprobed:
            break
        if arm == "probe_random":
            picks = rng.sample(unprobed, min(batch, len(unprobed)))
        else:
            model = fit_probe_scorer(probed, seed + round_index, args, tokenizer)
            dl = make_loader(unprobed, tokenizer, args.max_length, args.batch_size,
                             False, probe_text, disagree_label)
            scores = probabilities(model, dl)
            ranked = sorted(range(len(unprobed)), key=lambda i: -scores[i])
            picks = [unprobed[i] for i in ranked[:batch]]
        for e in picks:
            probed.append(e)
            probed_ids.add(e.question_id)
        found = sum(1 for e in probed if disagree_label(e))
        round_index += 1
        print(f"    probes={len(probed)} disagreements={found}", flush=True)
    return probed, len(probed)


def build_validation_from_probes(probed, seed):
    disagree = [e for e in probed if disagree_label(e)]
    agree = [e for e in probed if not disagree_label(e)]
    rng = random.Random(seed + 33301)
    d = disagree if len(disagree) <= VAL_DISAGREE else rng.sample(disagree, VAL_DISAGREE)
    a = rng.sample(agree, min(VAL_AGREE, len(agree)))
    rows = d + a
    rng.shuffle(rows)
    return rows


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
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/computer_qa_online_validation.json")
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
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, local_files_only=True)

    result = {}
    if args.output.exists():
        result = json.loads(args.output.read_text(encoding="utf-8"))
    result["protocol"] = {
        "status": "online-learned validation acquisition on Computer QA (cache simulation)",
        "probing": f"scorer on [question + Nano answer] -> P(disagree); {SEED_PROBES} random seed "
                   f"probes, rounds of {PROBE_BATCH} by policy, self-labeled; stop at "
                   f"{VAL_DISAGREE} disagreements or {PROBE_CAP} probes",
        "arms": list(ARMS),
        "downstream": f"train = same random {N_TRAIN} as t25_v25; two-head router, no prior; "
                      "reference = oracle enriched validation in computer_qa_50label_budget.json",
        "gold_budget": f"{N_TRAIN} train + {VAL_DISAGREE + VAL_AGREE} validation = 50 gold; "
                       "Mini probes measured per arm",
        "answer_model_api_calls": 0,
    }
    result["baselines"] = {
        "all_nano": evaluate_at(heldout, [0.0] * len(heldout), 1.0),
        "all_mini": evaluate_at(heldout, [1.0] * len(heldout), 0.0),
        "random_probe_expected": VAL_DISAGREE / 0.22,
    }
    runs = result.setdefault("runs", {})

    for arm in ARMS:
        done = {int(r["seed"]) for r in runs.get(arm, [])}
        for seed in args.seeds:
            if seed in done:
                continue
            orders, standard_validation = per_seed[seed]
            train = select_random(orders, seed, N_TRAIN)
            train_ids = {e.question_id for e in train}
            candidates = [e for e in flat(orders) if e.question_id not in train_ids]
            candidates += standard_validation
            print(f"{arm} seed={seed}", flush=True)
            probed, probes_used = probe_for_disagreements(arm, candidates, seed, args, tokenizer)
            validation = build_validation_from_probes(probed, seed)
            val_ids = {e.question_id for e in validation}
            if train_ids & val_ids:
                raise AssertionError("Train/validation overlap")
            pool_sample = build_pool_sample(orders, train_ids | val_ids, seed)
            vs, ps, ts = train_and_score(train, validation, pool_sample, heldout,
                                         seed, {}, rates_by_seed[seed], args)
            criteria = pick_operating_points(validation, vs, ps, heldout, ts)
            run = {"seed": seed,
                   "probes_used": probes_used,
                   "disagreements_found": sum(1 for e in probed if disagree_label(e)),
                   "validation_informative": sum(1 for e in validation if disagree_label(e)),
                   "criteria": criteria}
            for name in ("maxacc", "q50"):
                op = criteria[name]["operating_point"]
                print(f"  probes={probes_used} {name:7s} acc={op['accuracy']:.4f} "
                      f"rate={op['mini_rate']:.4f}", flush=True)
            runs.setdefault(arm, []).append(run)
            runs[arm] = sorted(runs[arm], key=lambda r: int(r["seed"]))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    result["summary"] = {}
    criteria_names = ["maxacc", "q70", "q50", "q30"]
    for arm, rs in sorted(runs.items()):
        result["summary"][arm] = {
            "probes_used": {"mean": mean(r["probes_used"] for r in rs),
                            "std": pstdev(r["probes_used"] for r in rs),
                            "values": [r["probes_used"] for r in rs]},
            "validation_informative": [r["validation_informative"] for r in rs],
        }
        for name in criteria_names:
            ops = [r["criteria"][name]["operating_point"] for r in rs]
            result["summary"][arm][name] = {
                f: {"mean": mean(o[f] for o in ops), "std": pstdev(o[f] for o in ops)}
                for f in ("accuracy", "mini_rate", "mini_only_recall")
            }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
