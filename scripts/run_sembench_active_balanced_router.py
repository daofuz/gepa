#!/usr/bin/env python3
"""Simulate active 100-label acquisition for direct SemBench Movie routing.

The acquisition policy only uses information available before each annotation:
mini/nano predictions, review text, and a provisional sentiment model trained on
labels already acquired. Ground truth is used only after an example is selected.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from train_sembench_balanced_direct_router import (
    BUCKETS,
    UTILITY,
    evaluate,
    load_examples,
    text,
)


TEST_COUNTS = {
    "both_correct": 72,
    "mini_only": 5,
    "nano_only": 1,
    "both_wrong": 4,
}
ORACLE_BALANCED_COUNTS = {
    "both_correct": 60,
    "mini_only": 20,
    "nano_only": 4,
    "both_wrong": 16,
}


def make_model(seed: int) -> Pipeline:
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2), max_features=8000, sublinear_tf=True
            ),
        ),
        (
            "classifier",
            LogisticRegression(C=1.0, max_iter=2000, random_state=seed),
        ),
    ])


def split_test(
    examples: list[dict[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    test = []
    test_ids = set()
    for bucket in BUCKETS:
        values = [x for x in examples if x["bucket"] == bucket]
        rng.shuffle(values)
        chosen = values[: TEST_COUNTS[bucket]]
        test.extend(chosen)
        test_ids.update(x["review_id"] for x in chosen)
    rng.shuffle(test)
    pool = [x for x in examples if x["review_id"] not in test_ids]
    return pool, test


def random_acquisition(
    pool: list[dict[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    selected = rng.sample(pool, 100)
    return selected, {"method": "random"}


def oracle_balanced_acquisition(
    pool: list[dict[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    selected = []
    for bucket, count in ORACLE_BALANCED_COUNTS.items():
        values = [x for x in pool if x["bucket"] == bucket]
        selected.extend(rng.sample(values, count))
    rng.shuffle(selected)
    return selected, {"method": "oracle_balanced", "requested_counts": ORACLE_BALANCED_COUNTS}


def active_acquisition(
    pool: list[dict[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    disagreements = [x for x in pool if x["mini_pred"] != x["nano_pred"]]
    agreements = [x for x in pool if x["mini_pred"] == x["nano_pred"]]
    rng.shuffle(disagreements)
    rng.shuffle(agreements)

    disagreement_labels = disagreements[:30]
    agreement_seed = agreements[:20]
    initially_labeled = disagreement_labels + agreement_seed
    initially_labeled_ids = {x["review_id"] for x in initially_labeled}

    # Learn the latent semantic label, then inspect agreements that contradict
    # the provisional classifier. These are candidates for both-model failure.
    sentiment_model = make_model(seed)
    sentiment_model.fit(
        [x["text"] for x in initially_labeled],
        [int(x["gold"] == "POSITIVE") for x in initially_labeled],
    )
    remaining_agreements = [
        x for x in agreements if x["review_id"] not in initially_labeled_ids
    ]
    positive_probability = sentiment_model.predict_proba(
        [x["text"] for x in remaining_agreements]
    )[:, 1]
    scored = []
    for example, p_positive in zip(remaining_agreements, positive_probability):
        agreed_positive = example["mini_pred"] == "POSITIVE"
        p_agreed = p_positive if agreed_positive else 1.0 - p_positive
        scored.append((float(p_agreed), rng.random(), example))
    scored.sort(key=lambda value: (value[0], value[1]))
    hard_agreements = [value[2] for value in scored[:30]]

    selected_ids = initially_labeled_ids | {x["review_id"] for x in hard_agreements}
    coverage_pool = [x for x in agreements if x["review_id"] not in selected_ids]
    coverage_count = 100 - len(disagreement_labels) - len(agreement_seed) - len(hard_agreements)
    random_coverage = rng.sample(coverage_pool, coverage_count)
    selected = disagreement_labels + agreement_seed + hard_agreements + random_coverage
    rng.shuffle(selected)
    return selected, {
        "method": "observable_active",
        "disagreements": len(disagreement_labels),
        "agreement_seed": len(agreement_seed),
        "hard_agreements": len(hard_agreements),
        "random_coverage": len(random_coverage),
    }


def route_weight(example: dict[str, Any]) -> float:
    nano = UTILITY[(example["bucket"], "nano")]
    mini = UTILITY[(example["bucket"], "mini")]
    return abs(mini - nano)


def weighted_validation_utility(
    examples: list[dict[str, Any]],
    predictions: list[int],
    population_prior: dict[str, float],
) -> float:
    values = {bucket: [] for bucket in BUCKETS}
    for example, prediction in zip(examples, predictions):
        route = "mini" if prediction else "nano"
        values[example["bucket"]].append(UTILITY[(example["bucket"], route)])
    # If a scarce bucket is absent from acquired labels, it cannot influence
    # calibration; renormalize over represented buckets.
    represented = [bucket for bucket in BUCKETS if values[bucket]]
    normalizer = sum(population_prior[bucket] for bucket in represented)
    return sum(
        population_prior[bucket]
        * statistics.mean(values[bucket])
        / normalizer
        for bucket in represented
    )


def fit_direct_router(
    acquired: list[dict[str, Any]],
    test: list[dict[str, Any]],
    population_prior: dict[str, float],
    seed: int,
    utility_weighted: bool,
) -> dict[str, Any]:
    targets = np.asarray([x["target"] for x in acquired])
    bucket_labels = [x["bucket"] for x in acquired]
    folds = min(5, min(Counter(targets.tolist()).values()))
    if folds < 2:
        raise ValueError("Acquired set does not cover every outcome bucket twice")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    out_of_fold = np.zeros(len(acquired), dtype=float)
    fold_models = []

    for train_indices, validation_indices in splitter.split(acquired, targets):
        train = [acquired[index] for index in train_indices]
        validation = [acquired[index] for index in validation_indices]
        model = make_model(seed)
        fit_kwargs = {}
        if utility_weighted:
            fit_kwargs["classifier__sample_weight"] = [route_weight(x) for x in train]
        model.fit([text(x) for x in train], [x["target"] for x in train], **fit_kwargs)
        fold_models.append(model)
        out_of_fold[validation_indices] = model.predict_proba(
            [text(x) for x in validation]
        )[:, 1]

    candidates = []
    for threshold in [value / 100 for value in range(5, 96, 5)]:
        predictions = [int(value >= threshold) for value in out_of_fold]
        utility = weighted_validation_utility(acquired, predictions, population_prior)
        candidates.append((utility, -sum(predictions), threshold))
    threshold = max(candidates)[2]

    probabilities = np.mean(
        [model.predict_proba([text(x) for x in test])[:, 1] for model in fold_models],
        axis=0,
    )
    metrics = evaluate(test, [int(value >= threshold) for value in probabilities])
    return {
        "threshold": threshold,
        "utility_weighted_loss": utility_weighted,
        **metrics,
    }


def run_seed(examples: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    pool, test = split_test(examples, seed)
    population_counts = Counter(x["bucket"] for x in examples)
    population_prior = {
        bucket: population_counts[bucket] / len(examples) for bucket in BUCKETS
    }
    result = {"seed": seed, "test_counts": dict(Counter(x["bucket"] for x in test)), "methods": {}}
    acquisitions = {
        "random": random_acquisition(pool, seed),
        "observable_active": active_acquisition(pool, seed),
        "oracle_balanced": oracle_balanced_acquisition(pool, seed),
    }
    for acquisition_name, (acquired, acquisition_trace) in acquisitions.items():
        entry = {
            "acquisition": acquisition_trace,
            "acquired_counts": dict(Counter(x["bucket"] for x in acquired)),
        }
        for utility_weighted in (False, True):
            name = "utility_weighted" if utility_weighted else "cross_entropy"
            entry[name] = fit_direct_router(
                acquired, test, population_prior, seed, utility_weighted
            )
        result["methods"][acquisition_name] = entry
    return result


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [
        "mini_calls",
        "selected_accuracy",
        "mean_utility",
        "mini_only_recall",
        "both_wrong_mini_rate",
        "unnecessary_mini",
        "critical_misroutes",
    ]
    output = {}
    for acquisition in ("random", "observable_active", "oracle_balanced"):
        output[acquisition] = {}
        for loss in ("cross_entropy", "utility_weighted"):
            values = [run["methods"][acquisition][loss] for run in runs]
            output[acquisition][loss] = {
                metric: statistics.mean(value[metric] for value in values)
                for metric in metrics
            }
            output[acquisition][loss]["threshold"] = statistics.mean(
                value["threshold"] for value in values
            )
        acquired = [run["methods"][acquisition]["acquired_counts"] for run in runs]
        output[acquisition]["mean_acquired_counts"] = {
            bucket: statistics.mean(value.get(bucket, 0) for value in acquired)
            for bucket in BUCKETS
        }
    return output


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reviews",
        type=Path,
        default=root / "sembench/files/movie/data/sf_2000/Reviews.csv",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=root / "sembench_movie_router/model_outputs.json",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[505, 606, 707, 808, 909])
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "sembench_movie_router/active_balanced_direct_router.json",
    )
    args = parser.parse_args()
    loaded_examples = load_examples(args.reviews, args.cache)
    examples = list({x["review_id"]: x for x in loaded_examples}.values())
    runs = []
    for seed in args.seeds:
        run = run_seed(examples, seed)
        runs.append(run)
        print(f"seed={seed} active_counts={run['methods']['observable_active']['acquired_counts']}")
    result = {
        "config": {
            "reviews": str(args.reviews),
            "cache": str(args.cache),
            "seeds": args.seeds,
            "label_budget": 100,
            "both_wrong_target": "mini",
        },
        "available_counts": dict(Counter(x["bucket"] for x in examples)),
        "aggregate": aggregate(runs),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()





