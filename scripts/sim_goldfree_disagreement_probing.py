#!/usr/bin/env python3
"""Gold-free probe acquisition: can a disagreement model cut Mini pool coverage?

Setting: raw pool, no gold labels at all.  Nano is run over the whole pool
(cheap).  Every Mini call is a probe that reveals ONE observable bit --
whether the two models disagree -- and disagreement needs no gold.  On a
binary task disagreement is exactly MO union NO, so probes that land on
disagreements are the ones that harvest critical examples.

Loop: seed with a random probe batch, fit a TF-IDF logistic model on
(text, nano_pred) -> disagree using ONLY probed rows, probe the top-scored
unprobed rows, repeat.  Compare with uniform random probing at matched
Mini-call budgets.  Gold is used only to score MO afterwards, never to select.
No answer-model API calls.
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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from run_sembench_qa50_softprompt_router import deduplicate
from train_sembench_balanced_direct_router import load_examples


def probe_loop(pool, seed, seed_n, batch, rounds, strategy):
    """Return cumulative (probes, mo_found, disagree_found) after each round.

    Strategies prefixed with "class_" spend the budget only on rows whose Nano
    prediction is the minority class -- a free, gold-free, Mini-free filter.
    """
    rng = random.Random(seed)
    order = list(range(len(pool)))
    rng.shuffle(order)
    if strategy.startswith("class_"):
        minority = minority_class(pool)
        inside = [i for i in order if pool[i]["nano_pred"] == minority]
        outside = [i for i in order if pool[i]["nano_pred"] != minority]
        order = inside + outside
    probed = order[:seed_n]
    trace = [snapshot(pool, probed)]
    for _ in range(rounds):
        unprobed = [i for i in order if i not in set(probed)]
        if not unprobed:
            break
        if strategy in ("random", "class_random"):
            picked = unprobed[:batch]
        else:
            picked = rank_by_disagreement(pool, probed, unprobed, seed)[:batch]
        probed = probed + picked
        trace.append(snapshot(pool, probed))
    return trace


def snapshot(pool, probed):
    rows = [pool[i] for i in probed]
    return {
        "probes": len(probed),
        "mo": sum(r["bucket"] == "mini_only" for r in rows),
        "disagree": sum(r["nano_pred"] != r["mini_pred"] for r in rows),
    }


def rank_by_disagreement(pool, probed, unprobed, seed):
    """Fit disagree-vs-agree on probed rows only; score the unprobed rows."""
    train_text = [feature_text(pool[i]) for i in probed]
    train_y = [int(pool[i]["nano_pred"] != pool[i]["mini_pred"]) for i in probed]
    if len(set(train_y)) < 2:
        return unprobed
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    x = vec.fit_transform(train_text)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0,
                             random_state=seed)
    clf.fit(x, train_y)
    scores = clf.predict_proba(vec.transform([feature_text(pool[i]) for i in unprobed]))[:, 1]
    return [i for _, i in sorted(zip(-scores, unprobed))]


def feature_text(row):
    return f"NANO={row['nano_pred']} {row['text']}"


def minority_class(pool):
    counts = {}
    for row in pool:
        counts[row["nano_pred"]] = counts.get(row["nano_pred"], 0) + 1
    return min(counts, key=counts.get)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", type=Path, default=ROOT / "sembench/files/movie/data/sf_2000/Reviews.csv")
    parser.add_argument("--cache", type=Path, default=ROOT / "sembench_movie_router/model_outputs.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "sembench_movie_router/untouched_200_manifest.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[505, 606, 707, 808, 909])
    parser.add_argument("--seed-n", type=int, default=50)
    parser.add_argument("--batch", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/goldfree_disagreement_probing.json")
    args = parser.parse_args()

    heldout_ids = set(json.loads(args.manifest.read_text(encoding="utf-8"))["review_ids"])
    examples = deduplicate(load_examples(args.reviews, args.cache))
    pool = [r for r in examples if r["review_id"] not in heldout_ids]
    total_mo = sum(r["bucket"] == "mini_only" for r in pool)
    total_dis = sum(r["nano_pred"] != r["mini_pred"] for r in pool)
    print(f"pool={len(pool)} MO={total_mo} disagree={total_dis}", flush=True)

    result = {
        "protocol": {
            "setting": "no gold labels; nano run pool-wide; each mini call is a probe",
            "signal": "observable nano/mini disagreement (gold-free); MO scored post hoc only",
            "model": "TF-IDF(1,2) + balanced logistic on probed rows only",
            "answer_model_api_calls": 0,
        },
        "pool": {"size": len(pool), "mo": total_mo, "disagree": total_dis},
        "traces": {},
    }
    result["nano_class_rates"] = {
        pred: {
            "n": sum(r["nano_pred"] == pred for r in pool),
            "disagree_rate": mean(float(r["nano_pred"] != r["mini_pred"])
                                  for r in pool if r["nano_pred"] == pred),
        }
        for pred in sorted({r["nano_pred"] for r in pool})
    }
    for strategy in ("random", "active", "class_random", "class_active"):
        per_seed = [probe_loop(pool, s, args.seed_n, args.batch, args.rounds, strategy)
                    for s in args.seeds]
        agg = []
        for step in range(len(per_seed[0])):
            mo = [t[step]["mo"] for t in per_seed]
            dis = [t[step]["disagree"] for t in per_seed]
            agg.append({
                "probes": per_seed[0][step]["probes"],
                "mo_mean": mean(mo), "mo_std": pstdev(mo),
                "mo_recall": mean(mo) / total_mo,
                "disagree_mean": mean(dis),
                "disagree_recall": mean(dis) / total_dis,
            })
            print(f"{strategy:7s} probes={agg[-1]['probes']:4d} "
                  f"MO={agg[-1]['mo_mean']:5.1f}+-{agg[-1]['mo_std']:.1f} "
                  f"({100 * agg[-1]['mo_recall']:.0f}% recall) "
                  f"dis={agg[-1]['disagree_mean']:5.1f} "
                  f"({100 * agg[-1]['disagree_recall']:.0f}%)", flush=True)
        result["traces"][strategy] = agg
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
