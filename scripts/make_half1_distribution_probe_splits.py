import argparse
import json
import random
from collections import Counter
from pathlib import Path


def norm(value):
    if value is None:
        return ""
    return str(value).strip().lower()


def is_correct(row):
    pred = norm(row.get("pred"))
    answer = norm(row.get("answer"))
    answer_index = norm(row.get("answer_index"))
    return bool(pred) and (pred == answer or pred == answer_index)


def sortable_qid(qid):
    text = str(qid)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def bucket_for(mini_ok, nano_ok):
    if mini_ok and nano_ok:
        return "both_correct"
    if mini_ok and not nano_ok:
        return "critical"
    if not mini_ok and nano_ok:
        return "nano_only"
    return "both_wrong"


def counts(ids, buckets):
    return dict(Counter(buckets[qid] for qid in ids))


# These are 50-example probes designed from the existing ablations:
# - keep many critical/mini_only cases to preserve accuracy
# - add nano_only and both_correct cases to prevent all-mini collapse
# - keep both_wrong moderate so the router does not over-learn difficult => mini
SCENARIOS = {
    "crit19_nano6_bc20_bw5": {
        "critical": 19,
        "nano_only": 6,
        "both_correct": 20,
        "both_wrong": 5,
    },
    "crit18_nano7_bc20_bw5": {
        "critical": 18,
        "nano_only": 7,
        "both_correct": 20,
        "both_wrong": 5,
    },
    "crit17_nano8_bc20_bw5": {
        "critical": 17,
        "nano_only": 8,
        "both_correct": 20,
        "both_wrong": 5,
    },
    "crit17_nano9_bc18_bw6": {
        "critical": 17,
        "nano_only": 9,
        "both_correct": 18,
        "both_wrong": 6,
    },
    "crit19_nano5_bc18_bw8": {
        "critical": 19,
        "nano_only": 5,
        "both_correct": 18,
        "both_wrong": 8,
    },
    "crit16_nano9_bc20_bw5": {
        "critical": 16,
        "nano_only": 9,
        "both_correct": 20,
        "both_wrong": 5,
    },
}


def allocate_validation(target, val_count):
    raw = {bucket: target[bucket] * val_count / sum(target.values()) for bucket in target}
    val_alloc = {bucket: int(raw[bucket]) for bucket in target}
    remaining = val_count - sum(val_alloc.values())
    fractional = sorted(
        target,
        key=lambda bucket: (raw[bucket] - val_alloc[bucket], target[bucket]),
        reverse=True,
    )
    for bucket in fractional:
        if remaining <= 0:
            break
        if val_alloc[bucket] < target[bucket]:
            val_alloc[bucket] += 1
            remaining -= 1
    return val_alloc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mini", default="computer science_result_mini.json")
    parser.add_argument("--nano", default="computer science_result_nano.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-count", type=int, default=10)
    parser.add_argument("--prefix", default="routing_split_half1_probe50")
    args = parser.parse_args()

    mini_rows = json.loads(Path(args.mini).read_text(encoding="utf-8"))
    nano_rows = json.loads(Path(args.nano).read_text(encoding="utf-8"))
    mini_by_id = {row["question_id"]: row for row in mini_rows}
    nano_by_id = {row["question_id"]: row for row in nano_rows}

    qids = sorted(mini_by_id, key=sortable_qid)
    midpoint = len(qids) // 2
    first_half_ids = qids[:midpoint]
    second_half_ids = qids[midpoint:]

    buckets = {}
    by_bucket = {name: [] for name in ["critical", "nano_only", "both_correct", "both_wrong"]}
    for qid in qids:
        b = bucket_for(is_correct(mini_by_id[qid]), is_correct(nano_by_id[qid]))
        buckets[qid] = b
        if qid in first_half_ids:
            by_bucket[b].append(qid)

    rng = random.Random(args.seed)
    for ids in by_bucket.values():
        rng.shuffle(ids)

    print("first_half_counts", {key: len(value) for key, value in by_bucket.items()})

    for scenario_name, target in SCENARIOS.items():
        if sum(target.values()) != 50:
            raise ValueError(f"{scenario_name} does not sum to 50")
        for bucket, count in target.items():
            if count > len(by_bucket[bucket]):
                raise ValueError(
                    f"{scenario_name} asks for {count} {bucket}, only {len(by_bucket[bucket])} available"
                )

        selected = {
            bucket: sorted(by_bucket[bucket][:count], key=sortable_qid)
            for bucket, count in target.items()
        }
        val_alloc = allocate_validation(target, args.val_count)

        train_ids = []
        val_ids = []
        for bucket, ids in selected.items():
            val_part = ids[: val_alloc[bucket]]
            train_part = ids[val_alloc[bucket] :]
            train_ids.extend(train_part)
            val_ids.extend(val_part)

        train_ids = sorted(train_ids, key=sortable_qid)
        val_ids = sorted(val_ids, key=sortable_qid)
        test_ids = sorted(second_half_ids, key=sortable_qid)

        payload = {
            "description": (
                "Probe split for 50-example router optimization. Designed to test whether "
                "critical-heavy but nano-aware train/val distributions improve held-out "
                "accuracy while lowering mini-route cost."
            ),
            "seed": args.seed,
            "scenario": scenario_name,
            "target_train_val_counts": target,
            "total_examples": len(qids),
            "first_half_count": len(first_half_ids),
            "second_half_count": len(second_half_ids),
            "train_val_count": 50,
            "train_count": len(train_ids),
            "val_count": len(val_ids),
            "test_count": len(test_ids),
            "train_ids": train_ids,
            "val_ids": val_ids,
            "validation_ids": val_ids,
            "test_ids": test_ids,
            "bucket_counts": {
                "train": counts(train_ids, buckets),
                "validation": counts(val_ids, buckets),
                "train_val": counts(train_ids + val_ids, buckets),
                "test": counts(test_ids, buckets),
            },
        }
        output = Path(f"{args.prefix}_{scenario_name}_seed{args.seed}.json")
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print()
        print(output)
        print("target", target)
        print("train", payload["bucket_counts"]["train"])
        print("validation", payload["bucket_counts"]["validation"])
        print("train_val", payload["bucket_counts"]["train_val"])


if __name__ == "__main__":
    main()
