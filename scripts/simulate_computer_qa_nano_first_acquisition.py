#!/usr/bin/env python3
"""Compare full-pair and Nano-first acquisition on labeled CS-QA data.

The simulation treats the first ``pool_size`` questions as an already labeled
acquisition pool.  Full-pair acquisition reveals Mini for the whole pool before
choosing a route-target-balanced subset.  Nano-first acquisition uses only the
gold label and Nano prediction to choose the same subset, and reveals Mini only
for selected questions.  Because the legacy balanced target is Mini iff Nano is
wrong, the selected training data can be identical while Mini acquisition cost
changes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from optimize_ollama_router_gepa import load_examples, outcome_bucket  # noqa: E402
BUCKETS = ("both_correct", "mini_only", "nano_only", "both_wrong")


def mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
    }


def nano_correct(example: Any) -> bool:
    return example.nano_pred == example.answer


def balanced_ids(pool: list[Any], size: int, seed: int) -> list[int]:
    if size % 2:
        raise ValueError("Balanced subset size must be even")
    rng = random.Random(seed)
    correct = [example for example in pool if nano_correct(example)]
    wrong = [example for example in pool if not nano_correct(example)]
    rng.shuffle(correct)
    rng.shuffle(wrong)
    per_target = size // 2
    if per_target > min(len(correct), len(wrong)):
        raise ValueError(
            f"size={size} needs {per_target} examples per target, but pool has "
            f"Nano-correct={len(correct)} and Nano-wrong={len(wrong)}"
        )
    return sorted(
        [example.question_id for example in correct[:per_target] + wrong[:per_target]]
    )


def count_buckets(examples: list[Any]) -> dict[str, int]:
    counts = Counter(outcome_bucket(example) for example in examples)
    return {bucket: counts.get(bucket, 0) for bucket in BUCKETS}


def sha256_ids(ids: list[int]) -> str:
    payload = ",".join(str(value) for value in sorted(ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def outcome_stratified_split(
    examples: list[Any], validation_size: int, seed: int
) -> tuple[list[int], list[int]]:
    """Split selected pairs by four-way outcome using largest remainders."""
    rng = random.Random(seed)
    by_bucket = {bucket: [] for bucket in BUCKETS}
    for example in examples:
        by_bucket[outcome_bucket(example)].append(example.question_id)
    for ids in by_bucket.values():
        rng.shuffle(ids)

    raw = {
        bucket: len(ids) * validation_size / len(examples)
        for bucket, ids in by_bucket.items()
    }
    allocation = {bucket: int(raw[bucket]) for bucket in BUCKETS}
    remaining = validation_size - sum(allocation.values())
    order = sorted(
        BUCKETS,
        key=lambda bucket: (raw[bucket] - allocation[bucket], len(by_bucket[bucket])),
        reverse=True,
    )
    for bucket in order:
        if remaining == 0:
            break
        if allocation[bucket] < len(by_bucket[bucket]):
            allocation[bucket] += 1
            remaining -= 1
    if remaining:
        raise ValueError("Could not allocate the requested validation size")

    train_ids = []
    validation_ids = []
    for bucket in BUCKETS:
        count = allocation[bucket]
        validation_ids.extend(by_bucket[bucket][:count])
        train_ids.extend(by_bucket[bucket][count:])
    return sorted(train_ids), sorted(validation_ids)


def summarize(rows: list[dict[str, Any]], sizes: list[int]) -> list[dict[str, Any]]:
    output = []
    for size in sizes:
        group = [row for row in rows if row["selected_size"] == size]
        item: dict[str, Any] = {
            "selected_size": size,
            "selected_fraction": size / group[0]["pool_size"],
            "full_pair_mini_calls": group[0]["pool_size"],
            "nano_first_mini_calls": size,
            "mini_call_saving": 1.0 - size / group[0]["pool_size"],
        }
        for key in (
            "nano_first_mini_cost_usd",
            "mini_dollar_saving",
            "total_answer_cost_saving",
            "random_reveal_balanced_yield",
            "mini_only",
            "nano_only",
            "both_wrong",
            "both_correct",
        ):
            item[key] = mean_std([float(row[key]) for row in group])
        output.append(item)
    return output


def write_csv(path: Path, summary: list[dict[str, Any]]) -> None:
    fields = [
        "selected_size",
        "selected_fraction",
        "full_pair_mini_calls",
        "nano_first_mini_calls",
        "mini_call_saving",
        "mini_cost_mean_usd",
        "mini_dollar_saving_mean",
        "total_answer_cost_saving_mean",
        "random_reveal_balanced_yield_mean",
        "both_correct_mean",
        "mini_only_mean",
        "nano_only_mean",
        "both_wrong_mean",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in summary:
            writer.writerow({
                "selected_size": item["selected_size"],
                "selected_fraction": item["selected_fraction"],
                "full_pair_mini_calls": item["full_pair_mini_calls"],
                "nano_first_mini_calls": item["nano_first_mini_calls"],
                "mini_call_saving": item["mini_call_saving"],
                "mini_cost_mean_usd": item["nano_first_mini_cost_usd"]["mean"],
                "mini_dollar_saving_mean": item["mini_dollar_saving"]["mean"],
                "total_answer_cost_saving_mean": item["total_answer_cost_saving"]["mean"],
                "random_reveal_balanced_yield_mean": item["random_reveal_balanced_yield"]["mean"],
                "both_correct_mean": item["both_correct"]["mean"],
                "mini_only_mean": item["mini_only"]["mean"],
                "nano_only_mean": item["nano_only"]["mean"],
                "both_wrong_mean": item["both_wrong"]["mean"],
            })


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# CS-QA Nano-first acquisition simulation",
        "",
        "Full-pair reveals Mini for every pool item before selecting a balanced subset. "
        "Nano-first selects the same 50/50 Nano-correct/Nano-wrong subset from existing "
        "gold labels and Nano outputs, then reveals Mini only for selected items.",
        "",
        f"Pool: {payload['pool']['size']} questions; outcomes "
        f"{payload['pool']['outcomes']}; Nano cost ${payload['costs']['pool_nano_usd']:.6f}; "
        f"Mini cost ${payload['costs']['pool_mini_usd']:.6f}.",
        "",
        "| Final balanced data | Mini calls: full | Mini calls: Nano-first | Mini-call saving | Mini-$ saving | Total answer-cost saving | Random reveal balanced yield | Selected BC/MO/NO/BW |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in payload["summary"]:
        counts = "/".join(
            f"{item[key]['mean']:.1f}" for key in
            ("both_correct", "mini_only", "nano_only", "both_wrong")
        )
        lines.append(
            f"| {item['selected_size']} ({100*item['selected_fraction']:.0f}%) "
            f"| {item['full_pair_mini_calls']} | {item['nano_first_mini_calls']} "
            f"| {100*item['mini_call_saving']:.1f}% "
            f"| {100*item['mini_dollar_saving']['mean']:.1f}% "
            f"| {100*item['total_answer_cost_saving']['mean']:.1f}% "
            f"| {item['random_reveal_balanced_yield']['mean']:.1f} "
            f"| {counts} |"
        )
    reference = payload["existing_balanced100_counterfactual"]
    lines.extend([
        "",
        "## Existing balanced-100 counterfactual",
        "",
        f"The existing reported split selects all {reference['pool_nano_wrong']} "
        f"Nano-wrong questions and {reference['selected_nano_correct']} Nano-correct "
        f"questions from its {reference['pool_size']}-question pool. It could therefore "
        "have been selected before revealing Mini. Revealing Mini only for its "
        f"{reference['selected_size']} selected questions saves "
        f"{100*reference['mini_call_saving']:.2f}% of Mini calls, "
        f"{100*reference['mini_dollar_saving']:.2f}% of Mini dollars, and "
        f"{100*reference['total_answer_cost_saving']:.2f}% of total Nano+Mini answer "
        "cost while preserving the exact reported training IDs and downstream result.",
        "",
        "The two acquisition methods use identical selected IDs for a given seed, so "
        "their downstream router data and performance are identical by construction. "
        "Only the acquisition-time Mini cost differs. Random reveal is included to show "
        "how much balanced data is recoverable without Nano-correctness stratification.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mini", type=Path, default=ROOT / "computer science_result_mini.json")
    parser.add_argument("--nano", type=Path, default=ROOT / "computer science_result_nano.json")
    parser.add_argument("--pool-size", type=int, default=200)
    parser.add_argument("--sizes", type=int, nargs="+", default=[20, 40, 60, 80, 100])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(100)))
    parser.add_argument("--split-seed", type=int, default=505)
    parser.add_argument(
        "--reference-split",
        type=Path,
        default=ROOT / "routing_split_computer_qa_balanced100_train80_val20_seed505.json",
    )
    parser.add_argument(
        "--reference-result",
        type=Path,
        default=ROOT / "outputs/computer_qa_regret_balanced100_bert_base.json",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/computer_qa_nano_first_acquisition.json")
    args = parser.parse_args()

    examples = sorted(load_examples(args.mini, args.nano), key=lambda example: example.question_id)
    pool = examples[: args.pool_size]
    by_id = {example.question_id: example for example in pool}
    pool_nano_cost = sum(example.nano_cost_usd for example in pool)
    pool_mini_cost = sum(example.mini_cost_usd for example in pool)
    full_answer_cost = pool_nano_cost + pool_mini_cost

    rows = []
    for size in args.sizes:
        for seed in args.seeds:
            ids = balanced_ids(pool, size, seed)
            selected = [by_id[value] for value in ids]
            selected_mini_cost = sum(example.mini_cost_usd for example in selected)
            counts = count_buckets(selected)

            rng = random.Random(seed)
            random_revealed = rng.sample(pool, size)
            random_correct = sum(nano_correct(example) for example in random_revealed)
            random_wrong = size - random_correct
            random_balanced_yield = 2 * min(random_correct, random_wrong)

            rows.append({
                "seed": seed,
                "pool_size": args.pool_size,
                "selected_size": size,
                "selected_ids_sha256": sha256_ids(ids),
                "identical_selected_data": True,
                "nano_first_mini_cost_usd": selected_mini_cost,
                "mini_dollar_saving": 1.0 - selected_mini_cost / pool_mini_cost,
                "total_answer_cost_saving": 1.0 - (pool_nano_cost + selected_mini_cost) / full_answer_cost,
                "random_reveal_balanced_yield": random_balanced_yield,
                **counts,
            })

    summary = summarize(rows, args.sizes)

    reference_split = json.loads(args.reference_split.read_text(encoding="utf-8"))
    reference_ids = set(reference_split["train_ids"] + reference_split["val_ids"])
    reference_pool = examples[:205]
    reference_by_id = {example.question_id: example for example in reference_pool}
    reference_selected = [reference_by_id[value] for value in reference_ids]
    reference_wrong_ids = {
        example.question_id for example in reference_pool if not nano_correct(example)
    }
    reference_selected_mini_cost = sum(
        example.mini_cost_usd for example in reference_selected
    )
    reference_pool_mini_cost = sum(example.mini_cost_usd for example in reference_pool)
    reference_pool_nano_cost = sum(example.nano_cost_usd for example in reference_pool)
    reference_result = json.loads(args.reference_result.read_text(encoding="utf-8"))
    existing_counterfactual = {
        "split_file": str(args.reference_split),
        "router_result_file": str(args.reference_result),
        "pool_size": len(reference_pool),
        "selected_size": len(reference_ids),
        "pool_nano_wrong": len(reference_wrong_ids),
        "selected_nano_wrong": len(reference_ids & reference_wrong_ids),
        "selected_nano_correct": len(reference_ids - reference_wrong_ids),
        "captures_all_nano_wrong": reference_wrong_ids <= reference_ids,
        "selected_outcomes": count_buckets(reference_selected),
        "full_pair_mini_calls": len(reference_pool),
        "nano_first_mini_calls": len(reference_ids),
        "mini_call_saving": 1.0 - len(reference_ids) / len(reference_pool),
        "full_pair_mini_cost_usd": reference_pool_mini_cost,
        "nano_first_mini_cost_usd": reference_selected_mini_cost,
        "mini_dollar_saving": 1.0 - reference_selected_mini_cost / reference_pool_mini_cost,
        "total_answer_cost_saving": 1.0
        - (reference_pool_nano_cost + reference_selected_mini_cost)
        / (reference_pool_nano_cost + reference_pool_mini_cost),
        "reported_downstream_aggregate": reference_result["aggregate"],
        "selected_ids_sha256": sha256_ids(sorted(reference_ids)),
    }

    split_ids = balanced_ids(pool, max(args.sizes), args.split_seed)
    split_examples = [by_id[value] for value in split_ids]
    train_ids, val_ids = outcome_stratified_split(
        split_examples, validation_size=20, seed=args.split_seed
    )
    fixed_test_ids = [example.question_id for example in examples[205:]]
    split = {
        "description": "Nano-first target-balanced CS-QA split; first 200 acquisition pool, five unused first-half IDs, final 205 fixed test",
        "seed": args.split_seed,
        "selection_target": "mini iff Nano is wrong; Nano otherwise",
        "pool_count": len(pool),
        "mini_calls_full_pair": len(pool),
        "mini_calls_nano_first": len(split_ids),
        "train_count": len(train_ids),
        "val_count": len(val_ids),
        "test_count": len(fixed_test_ids),
        "train_ids": train_ids,
        "val_ids": val_ids,
        "validation_ids": val_ids,
        "test_ids": fixed_test_ids,
        "selected_ids_sha256": sha256_ids(split_ids),
        "selected_outcomes": count_buckets(split_examples),
        "train_outcomes": count_buckets([by_id[value] for value in train_ids]),
        "validation_outcomes": count_buckets([by_id[value] for value in val_ids]),
    }

    payload = {
        "protocol": {
            "gold_available_before_mini": True,
            "full_pair": "reveal Mini for all pool items, then select 50/50 Nano-correct/Nano-wrong",
            "nano_first": "select the identical balanced IDs from gold+Nano, then reveal Mini only for selected items",
            "balance_definition": "legacy route target: Mini iff Nano is wrong",
            "seeds": args.seeds,
        },
        "pool": {
            "size": len(pool),
            "question_id_range": [pool[0].question_id, pool[-1].question_id],
            "nano_correct": sum(nano_correct(example) for example in pool),
            "nano_wrong": sum(not nano_correct(example) for example in pool),
            "outcomes": count_buckets(pool),
        },
        "costs": {
            "pool_nano_usd": pool_nano_cost,
            "pool_mini_usd": pool_mini_cost,
            "full_pair_answer_usd": full_answer_cost,
        },
        "summary": summary,
        "runs": rows,
        "existing_balanced100_counterfactual": existing_counterfactual,
        "recommended_split": split,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(args.output.with_suffix(".csv"), summary)
    write_markdown(args.output.with_suffix(".md"), payload)
    split_path = ROOT / "routing_split_computer_qa_nano_first_balanced100_train80_val20_seed505.json"
    split_path.write_text(json.dumps(split, indent=2), encoding="utf-8")
    print(json.dumps({"pool": payload["pool"], "costs": payload["costs"], "summary": summary}, indent=2))
    print(f"Saved {args.output}")
    print(f"Saved {split_path}")


if __name__ == "__main__":
    main()
