#!/usr/bin/env python3
"""External routing baselines under the unified 50-label protocol (Movie).

For the paper's baseline section: RouteLLM, FrugalGPT, and BARGAIN compared
against the two-head router using the SAME model pair (gpt-5-nano/mini), the
SAME splits and seeds, the SAME 50-gold-label budget (train 25 random +
validation 18 disagreements/7 agreements, as run_sembench_50label_budget.py),
and the SAME cost model (Nano = 1/4 Mini per call).

  routellm  -- pre-route, single "strong-model-wins" head on the frozen
               DistilBERT + 8 soft tokens.  Training label from the same 25
               gold rows: 1 iff Mini correct AND Nano wrong; ties go to the
               cheap model.  Plain BCE (no cost weighting).  Thresholds via
               validation-max-accuracy and pool-score quantiles -- the
               quantile calibration mirrors RouteLLM's own cost-threshold
               calibration.
  frugalgpt -- cascade scorer: input = review text + Nano's answer, single
               head predicts P(Nano correct); escalate to Mini when the
               score is below threshold.  Same labels/validation.  Cost
               accounting must add the always-paid Nano call
               (mini-equivalents = 0.25 + mini_rate).
  bargain   -- the cached streams hold no token logprobs, so BARGAIN's
               thresholded proxy-confidence procedure cannot be reproduced
               (same conclusion as scripts/compare_bargain_style.py).  We
               report the oracle-agreement cascade it approximates (route to
               Nano iff the two predictions agree): by the routing identity
               this equals all-Mini accuracy at the disagreement rate.
               Non-deployable reference, computed directly from the cache.

Five seeds, held-out 200, no answer-model API calls.
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
sys.path.insert(0, str(ROOT / "scripts"))

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from run_sembench_50label_budget import (
    build_pool_sample,
    build_validation,
    quantile_threshold,
    threshold_candidates,
)
from run_sembench_accuracy_targeted_router import evaluate_scores
from run_sembench_mo_importance import frontier
from run_sembench_prior_mo_random import select_random
from run_sembench_qa50_softprompt_router import deduplicate
from run_sembench_train100_ratio_sweep import make_orders
from train_sembench_balanced_direct_router import evaluate, load_examples, text


BUDGETS = [0.3, 0.5, 0.7]
SPLIT = (25, 18, 7)  # train n, validation disagreements, validation agreements


class SingleHeadRouter(nn.Module):
    def __init__(self, model_name, prompt_tokens):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=True)
        for p in self.encoder.parameters():
            p.requires_grad = False
        hidden = int(self.encoder.config.hidden_size)
        self.prompt = nn.Parameter(torch.empty(prompt_tokens, hidden))
        nn.init.normal_(self.prompt, mean=0.0, std=0.02)
        self.head = nn.Linear(hidden, 1)
        self.prompt_tokens = prompt_tokens

    def forward(self, input_ids, attention_mask):
        emb = self.encoder.get_input_embeddings()(input_ids)
        prompt = self.prompt.unsqueeze(0).expand(emb.shape[0], -1, -1)
        combined = torch.cat([prompt, emb], dim=1)
        pmask = torch.ones(emb.shape[0], self.prompt_tokens,
                           dtype=attention_mask.dtype, device=attention_mask.device)
        mask = torch.cat([pmask, attention_mask], dim=1)
        out = self.encoder(inputs_embeds=combined, attention_mask=mask)
        return self.head(out.last_hidden_state[:, self.prompt_tokens, :]).squeeze(-1)


def routellm_text(example):
    return text(example)


def frugalgpt_text(example):
    return f"{text(example)} [nano answer] {example['nano_pred']}"


def routellm_label(example):
    mini_ok = example["mini_pred"] == example["gold"]
    nano_ok = example["nano_pred"] == example["gold"]
    return 1.0 if (mini_ok and not nano_ok) else 0.0


def frugalgpt_label(example):
    return 1.0 if example["nano_pred"] == example["gold"] else 0.0


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
    """Returns per-epoch route scores (higher = route to Mini) for val/pool/test."""
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
        m = evaluate_scores(validation, scores, t)
        return (m["selected_accuracy"], -m["mini_calls"])

    def finish(name, epoch, t):
        op = evaluate_scores(test, test_scores[epoch], t)
        results[name] = {
            "best_epoch": epoch + 1, "threshold": t,
            "operating_point": {
                "accuracy": op["selected_accuracy"],
                "mini_rate": op["mini_rate"],
                "mini_only_recall": op["mini_only_recall"],
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
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/sembench_external_baselines.json")
    args = parser.parse_args()

    heldout_ids = set(json.loads(args.manifest.read_text(encoding="utf-8"))["review_ids"])
    examples = deduplicate(load_examples(args.reviews, args.cache))
    heldout = [r for r in examples if r["review_id"] in heldout_ids]
    historical = [r for r in examples if r["review_id"] not in heldout_ids]
    if len(heldout) != 200 or len(historical) != 812:
        raise AssertionError("Expected 812 historical and 200 held-out rows")
    per_seed = {seed: make_orders(historical, seed) for seed in args.seeds}
    n_train, n_dis, n_agr = SPLIT

    result = {}
    if args.output.exists():
        result = json.loads(args.output.read_text(encoding="utf-8"))
    result["protocol"] = {
        "status": "external baselines under the unified 50-label protocol (paper baseline section)",
        "parity": "same model pair, splits, seeds, 50-gold budget (t25_v25), and cost model "
                  "as outputs/sembench_50label_budget.json",
        "methods": {
            "routellm": "pre-route single strong-wins head, plain BCE, quantile cost calibration",
            "frugalgpt": "cascade scorer on text + Nano answer predicting P(Nano correct); "
                         "cost adds the always-paid Nano call (0.25 + mini_rate mini-equivalents)",
            "bargain": "logprob-free cache -> proxy-confidence procedure not reproducible; "
                       "oracle-agreement cascade reported as its non-deployable reference",
        },
        "answer_model_api_calls": 0,
    }
    disagree = [r for r in heldout if r["nano_pred"] != r["mini_pred"]]
    result["baselines"] = {
        "all_nano": evaluate(heldout, [0] * len(heldout)),
        "all_mini": evaluate(heldout, [1] * len(heldout)),
        "bargain_oracle_agreement": {
            "accuracy": evaluate(heldout, [1 if r["nano_pred"] != r["mini_pred"] else 0
                                           for r in heldout])["selected_accuracy"],
            "mini_rate": len(disagree) / len(heldout),
            "note": "non-deployable: requires both answers on every row",
        },
    }
    runs = result.setdefault("runs", {})

    for method in ("routellm", "frugalgpt"):
        done = {int(r["seed"]) for r in runs.get(method, [])}
        for seed in args.seeds:
            if seed in done:
                continue
            orders, standard_validation, _ = per_seed[seed]
            train = select_random(orders, seed, n_train)
            train_ids = {r["review_id"] for r in train}
            validation = build_validation(orders, standard_validation, train_ids,
                                          seed, n_dis, n_agr)
            val_ids = {r["review_id"] for r in validation}
            if train_ids & val_ids:
                raise AssertionError("Train/validation overlap")
            pool_sample = build_pool_sample(orders, train_ids | val_ids, seed)
            print(f"{method} seed={seed} train={len(train)} val={len(validation)}", flush=True)
            vs, ps, ts = train_baseline(method, train, validation, pool_sample,
                                        heldout, seed, args)
            criteria = pick_operating_points(validation, vs, ps, heldout, ts)
            run = {"seed": seed,
                   "train_counts": dict(Counter(r["bucket"] for r in train)),
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
