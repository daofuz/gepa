#!/usr/bin/env python3
"""True cold start: raw pool only, no gold, no historical runs, no known rule.

Everything the previous simulation assumed as known -- that Nano's minority
class is where the disagreements live -- has to be discovered here, on the
same Mini budget that is being spent to collect them.

Free at cold start: run Nano over the pool once, then stratify on Nano's own
observable outputs (predicted class, input length).  Each Mini probe reveals
one self-labeling bit: agree or disagree.  Allocation over strata is Thompson
sampling on Beta posteriors, so the cost of *learning* the rule is paid out of
the same budget.  Gold is never used for selection, only to score MO after.

Strategies
  random         uniform over the pool (floor)
  bandit_class   Thompson over Nano-predicted-class strata
  bandit_grid    Thompson over class x length-tercile strata (finer, slower)
  oracle_class   probe the hot class directly (upper bound; needs the rule)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_sembench_qa50_softprompt_router import deduplicate
from train_sembench_balanced_direct_router import load_examples

TARGET_DISAGREEMENTS = 25


def strata_of(pool, kind):
    """Assign every row a stratum id from Nano-side observables only."""
    if kind == "class":
        return [r["nano_pred"] for r in pool]
    lengths = sorted(len(r["text"]) for r in pool)
    cut1, cut2 = lengths[len(lengths) // 3], lengths[2 * len(lengths) // 3]
    out = []
    for r in pool:
        n = len(r["text"])
        band = "S" if n <= cut1 else ("M" if n <= cut2 else "L")
        out.append(f"{r['nano_pred']}|{band}")
    return out


def run(pool, seed, budget, strategy):
    rng = random.Random(seed)
    if strategy == "random":
        strata = ["all"] * len(pool)
    elif strategy == "oracle_class":
        rates = defaultdict(lambda: [0, 0])
        for r in pool:
            cell = rates[r["nano_pred"]]
            cell[0] += r["nano_pred"] != r["mini_pred"]
            cell[1] += 1
        hot = max(rates, key=lambda k: rates[k][0] / rates[k][1])
        strata = ["hot" if r["nano_pred"] == hot else "cold" for r in pool]
    else:
        strata = strata_of(pool, "class" if strategy == "bandit_class" else "grid")

    remaining = defaultdict(list)
    for i, s in enumerate(strata):
        remaining[s].append(i)
    for s in remaining:
        rng.shuffle(remaining[s])
    posterior = {s: [1.0, 1.0] for s in remaining}
    if strategy == "oracle_class":
        posterior = {"hot": [1e6, 1.0], "cold": [1.0, 1e6]}

    mo = dis = 0
    trace, first_25 = [], None
    for step in range(1, budget + 1):
        live = [s for s in remaining if remaining[s]]
        if not live:
            break
        if strategy == "random":
            pick = live[0]
        else:
            pick = max(live, key=lambda s: rng.betavariate(*posterior[s]))
        row = pool[remaining[pick].pop()]
        hit = row["nano_pred"] != row["mini_pred"]
        posterior[pick][0 if hit else 1] += 1
        dis += hit
        mo += row["bucket"] == "mini_only"
        if dis >= TARGET_DISAGREEMENTS and first_25 is None:
            first_25 = step
        if step % 50 == 0:
            trace.append({"probes": step, "mo": mo, "disagree": dis})
    return trace, first_25


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", type=Path, default=ROOT / "sembench/files/movie/data/sf_2000/Reviews.csv")
    parser.add_argument("--cache", type=Path, default=ROOT / "sembench_movie_router/model_outputs.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "sembench_movie_router/untouched_200_manifest.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(505, 525)))
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/coldstart_stratified_probing.json")
    args = parser.parse_args()

    heldout_ids = set(json.loads(args.manifest.read_text(encoding="utf-8"))["review_ids"])
    examples = deduplicate(load_examples(args.reviews, args.cache))
    pool = [r for r in examples if r["review_id"] not in heldout_ids]
    total_mo = sum(r["bucket"] == "mini_only" for r in pool)
    print(f"pool={len(pool)} MO={total_mo} "
          f"disagree={sum(r['nano_pred'] != r['mini_pred'] for r in pool)} "
          f"seeds={len(args.seeds)}", flush=True)

    result = {
        "protocol": {
            "setting": "raw pool only: no gold, no historical runs, no known selection rule",
            "free": "one Nano pass over the pool; strata from Nano's own outputs",
            "probe": "each Mini call returns agree/disagree, which is self-labeling",
            "allocation": "Thompson sampling on Beta posteriors per stratum",
            "target": f"{TARGET_DISAGREEMENTS} observed disagreements (disagree_oracle input)",
            "answer_model_api_calls": 0,
        },
        "pool": {"size": len(pool), "mo": total_mo},
        "strategies": {},
    }
    for strategy in ("random", "bandit_class", "bandit_grid", "oracle_class"):
        traces, firsts = [], []
        for seed in args.seeds:
            trace, first = run(pool, seed, args.budget, strategy)
            traces.append(trace)
            firsts.append(first if first is not None else args.budget)
        agg = []
        for step in range(min(len(t) for t in traces)):
            agg.append({
                "probes": traces[0][step]["probes"],
                "mo_mean": mean(t[step]["mo"] for t in traces),
                "mo_std": pstdev(t[step]["mo"] for t in traces),
                "disagree_mean": mean(t[step]["disagree"] for t in traces),
            })
        result["strategies"][strategy] = {
            "probes_to_25_disagreements": {"mean": mean(firsts), "std": pstdev(firsts),
                                           "max": max(firsts)},
            "trace": agg,
        }
        line = "  ".join(f"{a['probes']}:{a['mo_mean']:.1f}" for a in agg
                         if a["probes"] in (100, 200, 300, 400, 500))
        print(f"{strategy:13s} probes_to_25dis={mean(firsts):6.1f}+-{pstdev(firsts):5.1f} "
              f"(worst {max(firsts)})   MO@ {line}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
