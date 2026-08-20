#!/usr/bin/env python3
"""How should the paid validation rows be chosen?  (Computer QA, paired design)

Budget = --n-train paid training rows + --n-val paid validation rows.  Every
arm trains the SAME model on the SAME training rows with the SAME seed; the
only variable is which candidates become the validation set, so the comparison
is exactly paired and one training run serves every arm.

Acquisition strategies (all pay for exactly --n-val rows):
  random     uniform random -- the incumbent
  knn        k-NN disagreement propagation on raw-text embeddings, seeded by
             the training rows' free agree/disagree bits (hard-question
             enrichment in EMBEDDING space)
  boundary   the rows whose router score sits closest to the pool median --
             hard-question enrichment in SCORE space, where the threshold
             actually lives
  spread     rows taken at evenly spaced score quantiles, so the whole
             threshold range is covered rather than one neighbourhood

Router scores on unlabeled rows are free, so boundary/spread stay inside the
budget: they read the score column, then pay for the rows they pick.

Validation drives the threshold for the maxacc/cap criteria and only the epoch
for the q criteria, so an enrichment effect should show up on maxacc/cap first.
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

from optimize_ollama_router_gepa import outcome_bucket, load_examples
from run_computer_qa_50label_budget import pick_operating_points, train_and_score
from run_computer_qa_prior_mo_random import (
    evaluate_at,
    make_orders,
    mini_ok,
    nano_ok,
    select_random,
)
from run_computer_qa_knn_validation import (
    acquire_validation_knn,
    disagree_label,
    embed_rows,
    flat,
    train_rates,
)
from transformers import AutoTokenizer

STRATEGIES = ("random", "knn", "boundary", "spread")
CRITERIA = ("maxacc", "cap30", "cap50", "cap70", "q30", "q50", "q70")


def pick_random(candidates, scores, n_val, seed, **_):
    return random.Random(seed + 71993).sample(range(len(candidates)), n_val)


def pick_knn(candidates, scores, n_val, seed, train=None, embeddings=None, **_):
    chosen = acquire_validation_knn(train, candidates, embeddings, seed, n_val)
    index = {id(e): i for i, e in enumerate(candidates)}
    return [index[id(e)] for e in chosen]


def pick_boundary(candidates, scores, n_val, seed, **_):
    """Closest to the pool median score: the rows a threshold has to separate."""
    ordered = sorted(range(len(scores)), key=lambda i: scores[i])
    median = scores[ordered[len(ordered) // 2]]
    return sorted(range(len(scores)), key=lambda i: abs(scores[i] - median))[:n_val]


def pick_spread(candidates, scores, n_val, seed, **_):
    """Evenly spaced score quantiles, covering the whole threshold range."""
    ordered = sorted(range(len(scores)), key=lambda i: scores[i])
    step = len(ordered) / n_val
    return [ordered[min(len(ordered) - 1, int((k + 0.5) * step))] for k in range(n_val)]


PICKERS = {
    "random": pick_random,
    "knn": pick_knn,
    "boundary": pick_boundary,
    "spread": pick_spread,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mini-file", type=Path, default=ROOT / "computer science_result_mini.json")
    parser.add_argument("--nano-file", type=Path, default=ROOT / "computer science_result_nano.json")
    parser.add_argument("--split-file", type=Path,
                        default=ROOT / "routing_split_softprompt_threshold_balanced16.json")
    parser.add_argument("--hf-model", default="distilbert-base-uncased")
    parser.add_argument("--n-train", type=int, default=50)
    parser.add_argument("--n-val", type=int, default=20)
    parser.add_argument("--prompt-tokens", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[505, 606, 707, 808, 909])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/computer_qa_validation_acquisition.json")
    args = parser.parse_args()

    examples = load_examples(args.mini_file, args.nano_file)
    heldout_ids = {int(i) for i in json.loads(args.split_file.read_text(encoding="utf-8"))["test_ids"]}
    heldout = [e for e in examples if e.question_id in heldout_ids]
    pool = [e for e in examples if e.question_id not in heldout_ids]
    if len(heldout) != 205 or len(pool) != 205:
        raise AssertionError(f"Expected 205/205 split, got {len(heldout)}/{len(pool)}")

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, local_files_only=True)
    print("Embedding pool (raw text)...", flush=True)
    embeddings = embed_rows(pool, args, tokenizer)

    result = {
        "protocol": {
            "status": f"{args.n_train} paid train + {args.n_val} paid validation rows; "
                      "arms share the trained model and differ only in which rows "
                      "become the validation set (exactly paired)",
            "accounting": "router scores and embeddings on unpaid rows are free; each arm "
                          f"pays for the {args.n_val} rows it picks",
            "strategies": {
                "random": "uniform random",
                "knn": "k-NN disagreement propagation on raw-text embeddings",
                "boundary": "nearest the pool median router score",
                "spread": "evenly spaced score quantiles",
            },
            "validation_role": "threshold + epoch for maxacc/cap criteria, epoch only for "
                               "q criteria (those thresholds come from pool quantiles)",
            "answer_model_api_calls": 0,
        },
        "baselines": {
            "all_nano": evaluate_at(heldout, [0.0] * len(heldout), 1.0),
            "all_mini": evaluate_at(heldout, [1.0] * len(heldout), 0.0),
            "pool_disagreement_rate": sum(disagree_label(e) for e in pool) / len(pool),
        },
        "runs": {},
    }
    if args.output.exists():
        stored = json.loads(args.output.read_text(encoding="utf-8"))
        result["runs"] = stored.get("runs", {})

    for seed in args.seeds:
        orders, standard_validation = make_orders(pool, seed)
        train = select_random(orders, seed, args.n_train)
        train_ids = {e.question_id for e in train}
        candidates = [e for e in flat(orders) if e.question_id not in train_ids]
        candidates += standard_validation

        print(f"seed={seed}: training on {len(train)} rows, "
              f"{len(candidates)} candidates", flush=True)
        # candidates double as the "pool" so every candidate gets a free score
        # each epoch; validation passed here is unused for arm selection.
        _, cand_scores, test_scores = train_and_score(
            train, candidates[:1], candidates, heldout, seed, {}, train_rates(train), args
        )

        for strategy in STRATEGIES:
            if any(r["seed"] == seed for r in result["runs"].get(strategy, [])):
                continue
            idx = PICKERS[strategy](
                candidates, cand_scores[-1], args.n_val, seed,
                train=train, embeddings=embeddings,
            )
            validation = [candidates[i] for i in idx]
            val_scores = [[epoch[i] for i in idx] for epoch in cand_scores]
            criteria = pick_operating_points(
                validation, val_scores, cand_scores, heldout, test_scores
            )
            buckets = {}
            for e in validation:
                buckets[outcome_bucket(e)] = buckets.get(outcome_bucket(e), 0) + 1
            run = {
                "seed": seed,
                "validation_informative": int(sum(disagree_label(e) for e in validation)),
                "validation_buckets": buckets,
                "criteria": {k: criteria[k] for k in CRITERIA if k in criteria},
            }
            op = run["criteria"]["maxacc"]["operating_point"]
            print(f"  {strategy:9s} informative={run['validation_informative']:2d}/{args.n_val} "
                  f"maxacc acc={op['accuracy']:.4f} rate={op['mini_rate']:.3f}", flush=True)
            result["runs"].setdefault(strategy, []).append(run)
            result["runs"][strategy].sort(key=lambda r: r["seed"])
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    result["summary"] = {}
    for strategy, runs in sorted(result["runs"].items()):
        entry = {"validation_informative": [r["validation_informative"] for r in runs]}
        for name in CRITERIA:
            ops = [r["criteria"][name]["operating_point"] for r in runs if name in r["criteria"]]
            if not ops:
                continue
            entry[name] = {
                field: {"mean": mean(o[field] for o in ops), "std": pstdev(o[field] for o in ops)}
                for field in ("accuracy", "mini_rate", "mini_only_recall")
            }
        result["summary"][strategy] = entry
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
