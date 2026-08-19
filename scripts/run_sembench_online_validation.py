#!/usr/bin/env python3
"""Random training set + ONLINE-LEARNED validation acquisition (Movie).

The enriched validation (18 disagreements + 7 agreements) is the one place
where disagreement annotation is irreplaceable, and finding those 18
disagreements is its only expensive part (a Mini probe per candidate).  The
earlier gold-free probing study used a dataset-specific class rule to
concentrate probes; per project discipline that is a baseline, not a method.
This script tests the GENERAL learned version, end to end:

  Probing loop (self-labeling, no gold): scorer = frozen DistilBERT + 8 soft
  tokens + 1 head on [text + Nano answer] predicting P(disagree).  Seed with
  10 random probes; then rounds of 20 probes picked by policy, retraining the
  scorer on all probed rows (label = the observed agree/disagree bit) between
  rounds; stop at 18 disagreements or the probe cap (250).

  Arms: probe_random (uniform probing, control) and probe_online (scorer-
  ranked probing).  Metric 1 (efficiency): Mini probes spent.

  End-to-end: validation = the 18 found disagreements + 7 agreements sampled
  from probe misses (free byproducts); train = the SAME random 25 rows as
  t25_v25 (identical salt); two-head router, no prior; quantile + max-
  accuracy criteria.  Metric 2 (effectiveness): held-out operating points vs
  the oracle-built enriched validation of outputs/sembench_50label_budget.json
  (maxacc 90.8+-0.5, q70 89.6+-0.4, q50 88.2+-1.1, q30 86.3+-0.7).

Quantile pool-samples include probed-but-unselected rows (excluding only
train/validation) to avoid the acquisition-skew artifact found in the online
nano-wrong study.  Five seeds, held-out 200, no API calls.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from run_sembench_50label_budget import (
    build_pool_sample,
    pick_operating_points,
    train_and_score,
)
from run_sembench_external_baselines import SingleHeadRouter, make_loader, probabilities
from run_sembench_prior_mo_random import pool_base_rates, select_random
from run_sembench_qa50_softprompt_router import deduplicate
from run_sembench_train100_ratio_sweep import make_orders
from train_sembench_balanced_direct_router import evaluate, load_examples, text


N_TRAIN = 25
VAL_DISAGREE, VAL_AGREE = 18, 7
SEED_PROBES = 10
PROBE_CAP = 250
SCORER_EPOCHS = 4
KNN_K = 5
KMEANS_K = 8
ARMS = ("probe_random", "probe_online", "probe_knn", "probe_cluster")
ARM_SALTS = {"probe_random": 111, "probe_online": 222, "probe_knn": 333, "probe_cluster": 444}
# Neural-scorer retraining is expensive, so it probes in batches of 20; the
# embedding methods update at negligible cost and react every 5 probes.
ARM_BATCH = {"probe_random": 20, "probe_online": 20, "probe_knn": 5, "probe_cluster": 5}


def probe_text(e):
    return f"{text(e)} [nano answer] {e['nano_pred']}"


def disagree_label(e):
    return 1.0 if e["nano_pred"] != e["mini_pred"] else 0.0


def flat(orders):
    return [row for rows in orders.values() for row in rows]


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


def embed_rows(rows, args, tokenizer):
    """Frozen-DistilBERT mean-pooled embeddings of [text + nano answer],
    L2-normalized.  Model-only, no training, computed once."""
    from transformers import AutoModel
    encoder = AutoModel.from_pretrained(args.hf_model, local_files_only=True)
    encoder.eval()
    out = {}
    with torch.no_grad():
        for start in range(0, len(rows), 16):
            chunk = rows[start:start + 16]
            enc = tokenizer([probe_text(r) for r in chunk], truncation=True,
                            max_length=args.max_length, padding=True,
                            return_tensors="pt")
            hidden = encoder(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (hidden * mask).sum(1) / mask.sum(1)
            emb = torch.nn.functional.normalize(emb, dim=-1)
            for r, v in zip(chunk, emb):
                out[r["review_id"]] = v
    return out


def kmeans_assign(embeddings_by_id, k, iterations=20, seed=1234):
    """Plain k-means on the embedding matrix; returns review_id -> cluster."""
    ids = sorted(embeddings_by_id)
    matrix = torch.stack([embeddings_by_id[i] for i in ids])
    gen = torch.Generator().manual_seed(seed)
    centers = matrix[torch.randperm(len(ids), generator=gen)[:k]].clone()
    for _ in range(iterations):
        assign = (matrix @ centers.T).argmax(dim=1)
        for c in range(k):
            members = matrix[assign == c]
            if len(members):
                centers[c] = torch.nn.functional.normalize(members.mean(0), dim=-1)
    assign = (matrix @ centers.T).argmax(dim=1)
    return {i: int(c) for i, c in zip(ids, assign)}


def probe_for_disagreements(arm, candidates, seed, args, tokenizer, embeddings, clusters):
    """Probe until VAL_DISAGREE disagreements are found or PROBE_CAP spent.
    A probe = paying one Mini call on the row; the agree/disagree bit is then
    observable for free (self-labeling).  Returns (probed_rows, probes_used)."""
    rng = random.Random(seed + ARM_SALTS[arm])
    order = candidates[:]
    rng.shuffle(order)
    probed = order[:SEED_PROBES]
    probed_ids = {r["review_id"] for r in probed}
    found = sum(1 for r in probed if disagree_label(r))
    round_index = 0
    while found < VAL_DISAGREE and len(probed) < PROBE_CAP:
        batch = min(ARM_BATCH[arm], PROBE_CAP - len(probed))
        unprobed = [r for r in candidates if r["review_id"] not in probed_ids]
        if not unprobed:
            break
        if arm == "probe_random":
            picks = rng.sample(unprobed, min(batch, len(unprobed)))
        elif arm == "probe_online":
            model = fit_probe_scorer(probed, seed + round_index, args, tokenizer)
            dl = make_loader(unprobed, tokenizer, args.max_length, args.batch_size,
                             False, probe_text, disagree_label)
            scores = probabilities(model, dl)
            ranked = sorted(range(len(unprobed)), key=lambda i: -scores[i])
            picks = [unprobed[i] for i in ranked[:batch]]
        elif arm == "probe_knn":
            e_un = torch.stack([embeddings[r["review_id"]] for r in unprobed])
            e_pr = torch.stack([embeddings[r["review_id"]] for r in probed])
            y = torch.tensor([disagree_label(r) for r in probed])
            sims = e_un @ e_pr.T
            k = min(KNN_K, len(probed))
            top_sims, top_idx = sims.topk(k, dim=1)
            weights = top_sims.clamp(min=0.0) + 1e-6
            scores = (weights * y[top_idx]).sum(1) / weights.sum(1)
            jitter = torch.tensor([rng.random() * 1e-4 for _ in unprobed])
            ranked = (scores + jitter).argsort(descending=True).tolist()
            picks = [unprobed[i] for i in ranked[:batch]]
        elif arm == "probe_cluster":
            succ = {c: 0 for c in set(clusters.values())}
            fail = {c: 0 for c in set(clusters.values())}
            for r in probed:
                c = clusters[r["review_id"]]
                if disagree_label(r):
                    succ[c] += 1
                else:
                    fail[c] += 1
            by_cluster = {}
            for r in unprobed:
                by_cluster.setdefault(clusters[r["review_id"]], []).append(r)
            picks = []
            for _ in range(batch):
                live = [c for c in by_cluster if by_cluster[c]]
                if not live:
                    break
                draw = {c: rng.betavariate(1 + succ[c], 1 + fail[c]) for c in live}
                c = max(draw, key=draw.get)
                r = by_cluster[c].pop(rng.randrange(len(by_cluster[c])))
                picks.append(r)
                # Cache replay reveals the outcome immediately; update the arm.
                if disagree_label(r):
                    succ[c] += 1
                else:
                    fail[c] += 1
        else:
            raise ValueError(arm)
        for r in picks:
            probed.append(r)
            probed_ids.add(r["review_id"])
        found = sum(1 for r in probed if disagree_label(r))
        round_index += 1
        print(f"    probes={len(probed)} disagreements={found}", flush=True)
    return probed, len(probed)


def build_validation_from_probes(probed, seed):
    disagree = [r for r in probed if disagree_label(r)]
    agree = [r for r in probed if not disagree_label(r)]
    rng = random.Random(seed + 33301)
    d = disagree if len(disagree) <= VAL_DISAGREE else rng.sample(disagree, VAL_DISAGREE)
    a = rng.sample(agree, min(VAL_AGREE, len(agree)))
    rows = d + a
    rng.shuffle(rows)
    return rows


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
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/sembench_online_validation.json")
    args = parser.parse_args()

    heldout_ids = set(json.loads(args.manifest.read_text(encoding="utf-8"))["review_ids"])
    cache = json.loads(args.cache.read_text(encoding="utf-8"))
    examples = deduplicate(load_examples(args.reviews, args.cache))
    heldout = [r for r in examples if r["review_id"] in heldout_ids]
    historical = [r for r in examples if r["review_id"] not in heldout_ids]
    if len(heldout) != 200 or len(historical) != 812:
        raise AssertionError("Expected 812 historical and 200 held-out rows")
    per_seed = {seed: make_orders(historical, seed) for seed in args.seeds}
    rates_by_seed = {seed: pool_base_rates(per_seed[seed][0]) for seed in args.seeds}
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, local_files_only=True)
    print("Embedding historical pool for knn/cluster arms...", flush=True)
    embeddings = embed_rows(historical, args, tokenizer)
    clusters = kmeans_assign(embeddings, KMEANS_K)

    result = {}
    if args.output.exists():
        result = json.loads(args.output.read_text(encoding="utf-8"))
    result["protocol"] = {
        "status": "online-learned validation acquisition, end to end (cache simulation)",
        "question": "can a learned probe scorer replace the oracle/class-rule construction of the "
                    "enriched validation, and does the resulting validation still deliver its gains?",
        "probing": f"{SEED_PROBES} random seed probes, then batches by policy (neural scorer 20/round; "
                   f"embedding knn/cluster 5/round, updates are near-free), self-labeled (no gold); "
                   f"stop at {VAL_DISAGREE} disagreements or {PROBE_CAP} probes",
        "arms": list(ARMS),
        "embedding_arms": f"frozen-DistilBERT mean-pooled embeddings of [text + Nano answer]; "
                          f"probe_knn = distance-weighted vote of the {KNN_K} nearest probed rows; "
                          f"probe_cluster = Thompson sampling over {KMEANS_K} k-means clusters "
                          "(per-probe Beta posterior updates)",
        "downstream": f"train = same random {N_TRAIN} as t25_v25; two-head router, no prior; "
                      "reference = oracle enriched validation in sembench_50label_budget.json",
        "gold_budget": f"{N_TRAIN} train + {VAL_DISAGREE + VAL_AGREE} validation = 50 gold; "
                       "Mini probes measured per arm",
        "answer_model_api_calls": 0,
    }
    result["baselines"] = {
        "all_nano": evaluate(heldout, [0] * len(heldout)),
        "all_mini": evaluate(heldout, [1] * len(heldout)),
        "random_probe_expected": VAL_DISAGREE / 0.079,
    }
    runs = result.setdefault("runs", {})

    for arm in ARMS:
        done = {int(r["seed"]) for r in runs.get(arm, [])}
        for seed in args.seeds:
            if seed in done:
                continue
            orders, standard_validation, _ = per_seed[seed]
            train = select_random(orders, seed, N_TRAIN)
            train_ids = {r["review_id"] for r in train}
            candidates = [r for r in flat(orders) if r["review_id"] not in train_ids]
            candidates += standard_validation
            print(f"{arm} seed={seed}", flush=True)
            probed, probes_used = probe_for_disagreements(arm, candidates, seed, args,
                                                          tokenizer, embeddings, clusters)
            validation = build_validation_from_probes(probed, seed)
            val_ids = {r["review_id"] for r in validation}
            if train_ids & val_ids:
                raise AssertionError("Train/validation overlap")
            pool_sample = build_pool_sample(orders, train_ids | val_ids, seed)
            vs, ps, ts = train_and_score(train, validation, pool_sample, heldout,
                                         cache, seed, {}, rates_by_seed[seed], args)
            criteria = pick_operating_points(validation, vs, ps, heldout, ts)
            run = {"seed": seed,
                   "probes_used": probes_used,
                   "disagreements_found": sum(1 for r in probed if disagree_label(r)),
                   "validation_informative": sum(1 for r in validation if disagree_label(r)),
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
