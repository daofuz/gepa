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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mini", default="computer science_result_mini.json")
    parser.add_argument("--nano", default="computer science_result_nano.json")
    parser.add_argument("--sizes", default="50,100,150,200")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefix", default="routing_split_half1_random_control_trainval")
    args = parser.parse_args()

    mini_rows = json.loads(Path(args.mini).read_text(encoding="utf-8"))
    nano_rows = json.loads(Path(args.nano).read_text(encoding="utf-8"))
    mini_by_id = {row["question_id"]: row for row in mini_rows}
    nano_by_id = {row["question_id"]: row for row in nano_rows}

    qids = sorted(mini_by_id, key=sortable_qid)
    missing = [qid for qid in qids if qid not in nano_by_id]
    if missing:
        raise ValueError(f"nano file is missing {len(missing)} question ids")

    buckets = {}
    for qid in qids:
        buckets[qid] = bucket_for(is_correct(mini_by_id[qid]), is_correct(nano_by_id[qid]))

    midpoint = len(qids) // 2
    first_half_ids = qids[:midpoint]
    second_half_ids = qids[midpoint:]

    print(f"total_examples {len(qids)}")
    print(f"first_half {len(first_half_ids)}")
    print(f"second_half {len(second_half_ids)}")
    print(f"first_half_counts {counts(first_half_ids, buckets)}")
    print(f"test_counts {counts(second_half_ids, buckets)}")

    for size_text in args.sizes.split(","):
        total = int(size_text.strip())
        rng = random.Random(args.seed + total)
        selected = list(first_half_ids)
        rng.shuffle(selected)
        selected = selected[:total]

        val_total = round(total * 0.2)
        val_ids = selected[:val_total]
        train_ids = selected[val_total:]

        train_ids = sorted(train_ids, key=sortable_qid)
        val_ids = sorted(val_ids, key=sortable_qid)
        test_ids = sorted(second_half_ids, key=sortable_qid)

        payload = {
            "description": "Random-control split: train/validation sampled uniformly from first half; second half is test. Critical cases are not forced.",
            "seed": args.seed,
            "sampling_seed": args.seed + total,
            "total_examples": len(qids),
            "first_half_count": len(first_half_ids),
            "second_half_count": len(second_half_ids),
            "train_val_count": total,
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

        output = Path(f"{args.prefix}{total}_seed{args.seed}.json")
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print()
        print(output)
        print(f"train_count {len(train_ids)}")
        print(f"val_count {len(val_ids)}")
        print(f"test_count {len(test_ids)}")
        print(f"train {payload['bucket_counts']['train']}")
        print(f"validation {payload['bucket_counts']['validation']}")
        print(f"train_val {payload['bucket_counts']['train_val']}")
        print(f"test {payload['bucket_counts']['test']}")


if __name__ == "__main__":
    main()
