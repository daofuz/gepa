#!/usr/bin/env python3
"""Replay active routing-data acquisition when pool gold is initially hidden.

Selection may use question text, options, source metadata, and the cached Nano
response. Gold remains hidden until final selection. Most strategies reveal Mini
only for selected items; disagreement strategies first reveal Mini on a declared
partial-probe budget. Hidden gold is used only to evaluate the resulting sample.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from optimize_ollama_router_gepa import load_examples, outcome_bucket  # noqa: E402


BUCKETS = ("both_correct", "mini_only", "nano_only", "both_wrong")


def visible_text(row: dict[str, Any], include_nano: bool) -> str:
    options = row.get("options", [])
    if not isinstance(options, list):
        options = [str(options)]
    pieces = [
        f"source {row.get('src', '')}",
        f"category {row.get('category', '')}",
        f"question {row.get('question', '')}",
        " options ".join(str(value) for value in options),
    ]
    if include_nano:
        pieces.extend(
            [
                f"nano prediction {row.get('pred', '')}",
                f"nano response {row.get('response', '')}",
            ]
        )
    return " ".join(pieces)


def build_features(rows: list[dict[str, Any]], include_nano: bool) -> Any:
    texts = [visible_text(row, include_nano) for row in rows]
    word = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=8000,
        sublinear_tf=True,
    ).fit_transform(texts)
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=6000,
        sublinear_tf=True,
    ).fit_transform(texts)
    numeric = []
    for row in rows:
        question = str(row.get("question", ""))
        response = str(row.get("response", ""))
        options = row.get("options", [])
        option_count = len(options) if isinstance(options, list) else 0
        lower = response.lower()
        numeric.append(
            [
                np.log1p(len(question)),
                np.log1p(len(response)),
                np.log1p(len(response.split())),
                option_count / 10.0,
                float("the answer is" in lower),
                float(any(token in lower for token in ("not sure", "uncertain", "maybe", "cannot"))),
            ]
        )
    return hstack([word, char, np.asarray(numeric, dtype=np.float64)], format="csr")


def fit_error_probabilities(
    features: Any,
    selected: list[int],
    labels: np.ndarray,
    seed: int,
) -> np.ndarray:
    observed = labels[selected]
    if len(set(observed.tolist())) < 2:
        base = float(observed[0]) if len(observed) else 0.5
        return np.full(features.shape[0], base, dtype=np.float64)
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        solver="liblinear",
        random_state=seed,
    )
    model.fit(features[selected], observed)
    return model.predict_proba(features)[:, 1]


def choose_by_quota(
    probabilities: np.ndarray,
    available: list[int],
    error_need: int,
    correct_need: int,
    count: int,
    exploration: float,
    rng: random.Random,
) -> list[int]:
    count = min(count, len(available))
    explore_count = min(count, int(round(count * exploration)))
    exploit_count = count - explore_count
    denominator = max(error_need + correct_need, 1)
    high_count = int(round(exploit_count * max(error_need, 0) / denominator))
    high_count = min(high_count, exploit_count)
    low_count = exploit_count - high_count

    ranked = sorted(available, key=lambda index: (probabilities[index], index))
    chosen = list(reversed(ranked[-high_count:])) if high_count else []
    chosen_set = set(chosen)
    for index in ranked:
        if len(chosen) >= high_count + low_count:
            break
        if index not in chosen_set:
            chosen.append(index)
            chosen_set.add(index)
    remaining = [index for index in available if index not in chosen_set]
    rng.shuffle(remaining)
    chosen.extend(remaining[:explore_count])
    return chosen


def seeded_select(
    features: Any,
    labels: np.ndarray,
    final_size: int,
    seed_size: int,
    batch_size: int,
    seed: int,
    iterative: bool,
    exploration: float,
) -> tuple[list[int], float | None]:
    rng = random.Random(seed)
    indices = list(range(len(labels)))
    rng.shuffle(indices)
    selected = indices[:seed_size]
    target_error = final_size // 2
    target_correct = final_size - target_error

    initial_probabilities = fit_error_probabilities(features, selected, labels, seed)
    unselected = [index for index in range(len(labels)) if index not in set(selected)]
    initial_auc = None
    if len(set(labels[unselected].tolist())) == 2:
        initial_auc = float(roc_auc_score(labels[unselected], initial_probabilities[unselected]))

    if not iterative:
        observed_error = int(labels[selected].sum())
        observed_correct = len(selected) - observed_error
        chosen = choose_by_quota(
            initial_probabilities,
            unselected,
            target_error - observed_error,
            target_correct - observed_correct,
            final_size - len(selected),
            exploration,
            rng,
        )
        return selected + chosen, initial_auc

    while len(selected) < final_size:
        probabilities = fit_error_probabilities(features, selected, labels, seed)
        selected_set = set(selected)
        available = [index for index in range(len(labels)) if index not in selected_set]
        observed_error = int(labels[selected].sum())
        observed_correct = len(selected) - observed_error
        chosen = choose_by_quota(
            probabilities,
            available,
            target_error - observed_error,
            target_correct - observed_correct,
            min(batch_size, final_size - len(selected)),
            exploration,
            rng,
        )
        selected.extend(chosen)
    return selected, initial_auc


def oracle_select(labels: np.ndarray, final_size: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    wrong = [index for index, value in enumerate(labels) if value == 1]
    correct = [index for index, value in enumerate(labels) if value == 0]
    rng.shuffle(wrong)
    rng.shuffle(correct)
    per_class = final_size // 2
    return wrong[:per_class] + correct[: final_size - per_class]


def random_select(pool_size: int, final_size: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return rng.sample(range(pool_size), final_size)


def disagreement_probe_select(
    examples: list[Any], probe_size: int, final_size: int, seed: int
) -> tuple[list[int], list[int]]:
    """Run Mini on a random probe, then prioritize observable disagreements."""
    rng = random.Random(seed)
    probe = rng.sample(range(len(examples)), probe_size)
    disagreements = [
        index for index in probe if examples[index].nano_pred != examples[index].mini_pred
    ]
    agreements = [
        index for index in probe if examples[index].nano_pred == examples[index].mini_pred
    ]
    rng.shuffle(disagreements)
    rng.shuffle(agreements)
    disagreement_budget = min(final_size // 2, len(disagreements))
    selected = disagreements[:disagreement_budget]
    selected.extend(agreements[: final_size - len(selected)])
    if len(selected) < final_size:
        selected.extend(
            disagreements[
                disagreement_budget : disagreement_budget + final_size - len(selected)
            ]
        )
    return selected, probe


def evaluate_selection(
    examples: list[Any],
    selected: list[int],
    labels: np.ndarray,
    pool_nano_cost: float,
    pool_mini_cost: float,
    initial_auc: float | None,
    mini_revealed: list[int] | None = None,
) -> dict[str, Any]:
    chosen = [examples[index] for index in selected]
    counts = Counter(outcome_bucket(example) for example in chosen)
    pool_counts = Counter(outcome_bucket(example) for example in examples)
    wrong = int(labels[selected].sum())
    correct = len(selected) - wrong
    revealed = selected if mini_revealed is None else mini_revealed
    selected_mini_cost = sum(examples[index].mini_cost_usd for index in revealed)
    full_cost = pool_nano_cost + pool_mini_cost
    return {
        "selected": len(selected),
        "nano_wrong": wrong,
        "nano_correct": correct,
        "usable_balanced": 2 * min(wrong, correct),
        "balanced_fraction_of_target": 2 * min(wrong, correct) / max(len(selected), 1),
        "mini_only_recall": counts["mini_only"] / max(pool_counts["mini_only"], 1),
        "both_wrong_recall": counts["both_wrong"] / max(pool_counts["both_wrong"], 1),
        "nano_only_recall": counts["nano_only"] / max(pool_counts["nano_only"], 1),
        "outcomes": {bucket: counts[bucket] for bucket in BUCKETS},
        "gold_calls": len(selected),
        "mini_calls": len(revealed),
        "gold_call_saving_vs_full": 1.0 - len(selected) / len(examples),
        "mini_call_saving_vs_full": 1.0 - len(revealed) / len(examples),
        "mini_dollar_saving_vs_full": 1.0 - selected_mini_cost / pool_mini_cost,
        "total_answer_cost_saving_vs_full": 1.0
        - (pool_nano_cost + selected_mini_cost) / full_cost,
        "initial_selector_auc": initial_auc,
    }


def aggregate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    groups = sorted({(row["strategy"], row["final_size"]) for row in records})
    scalar_keys = (
        "mini_calls",
        "nano_wrong",
        "usable_balanced",
        "balanced_fraction_of_target",
        "mini_only_recall",
        "both_wrong_recall",
        "nano_only_recall",
        "gold_call_saving_vs_full",
        "mini_dollar_saving_vs_full",
        "total_answer_cost_saving_vs_full",
    )
    for strategy, final_size in groups:
        rows = [row for row in records if row["strategy"] == strategy and row["final_size"] == final_size]
        item: dict[str, Any] = {"strategy": strategy, "final_size": final_size, "seeds": len(rows)}
        for key in scalar_keys:
            values = [float(row[key]) for row in rows]
            item[key] = {"mean": statistics.fmean(values), "std": statistics.pstdev(values)}
        auc_values = [float(row["initial_selector_auc"]) for row in rows if row["initial_selector_auc"] is not None]
        item["initial_selector_auc"] = {
            "mean": statistics.fmean(auc_values) if auc_values else None,
            "std": statistics.pstdev(auc_values) if auc_values else None,
        }
        item["outcomes_mean"] = {
            bucket: statistics.fmean(float(row["outcomes"][bucket]) for row in rows)
            for bucket in BUCKETS
        }
        output.append(item)
    return output


def write_outputs(payload: dict[str, Any], output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rows = payload["summary"]
    with output_prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "strategy",
                "final_size",
                "mini_calls_mean",
                "nano_wrong_mean",
                "usable_balanced_mean",
                "critical_recall_mean",
                "bw_recall_mean",
                "initial_auc_mean",
                "mini_dollar_saving_mean",
                "total_cost_saving_mean",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["strategy"],
                    row["final_size"],
                    row["mini_calls"]["mean"],
                    row["nano_wrong"]["mean"],
                    row["usable_balanced"]["mean"],
                    row["mini_only_recall"]["mean"],
                    row["both_wrong_recall"]["mean"],
                    row["initial_selector_auc"]["mean"],
                    row["mini_dollar_saving_vs_full"]["mean"],
                    row["total_answer_cost_saving_vs_full"]["mean"],
                ]
            )

    lines = [
        "# CS-QA acquisition with initially hidden gold",
        "",
        "Gold is hidden until final selection. Mini is revealed only for selected items in random/seeded strategies, or on the stated partial probe in disagreement strategies.",
        "",
        f"Pool: {payload['pool']['size']} questions; outcomes {payload['pool']['outcomes']}.",
        "",
        "| Strategy | Final labels | Mini calls | Nano-wrong | Usable balanced | MO recall | BW recall | Initial AUC | Mini-$ saving | Total cost saving |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        auc = row["initial_selector_auc"]["mean"]
        lines.append(
            f"| {row['strategy']} | {row['final_size']} | {row['mini_calls']['mean']:.1f} | "
            f"{row['nano_wrong']['mean']:.1f} | {row['usable_balanced']['mean']:.1f} | "
            f"{100*row['mini_only_recall']['mean']:.1f}% | {100*row['both_wrong_recall']['mean']:.1f}% | "
            f"{auc:.3f} | " if auc is not None else
            f"| {row['strategy']} | {row['final_size']} | {row['mini_calls']['mean']:.1f} | "
            f"{row['nano_wrong']['mean']:.1f} | {row['usable_balanced']['mean']:.1f} | "
            f"{100*row['mini_only_recall']['mean']:.1f}% | {100*row['both_wrong_recall']['mean']:.1f}% | -- | "
        )
        lines[-1] += (
            f"{100*row['mini_dollar_saving_vs_full']['mean']:.1f}% | "
            f"{100*row['total_answer_cost_saving_vs_full']['mean']:.1f}% |"
        )
    output_prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mini", type=Path, default=ROOT / "computer science_result_mini.json")
    parser.add_argument("--nano", type=Path, default=ROOT / "computer science_result_nano.json")
    parser.add_argument("--pool-size", type=int, default=200)
    parser.add_argument("--final-sizes", type=int, nargs="+", default=[60, 80, 100])
    parser.add_argument("--seed-size", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "outputs" / "computer_qa_unlabeled_active_acquisition",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = sorted(load_examples(args.mini, args.nano), key=lambda example: example.question_id)[: args.pool_size]
    nano_rows = json.loads(args.nano.read_text(encoding="utf-8"))
    rows_by_id = {int(row["question_id"]): row for row in nano_rows}
    visible_rows = [rows_by_id[int(example.question_id)] for example in examples]
    labels = np.asarray([int(example.nano_pred != example.answer) for example in examples], dtype=np.int64)
    question_features = build_features(visible_rows, include_nano=False)
    nano_features = build_features(visible_rows, include_nano=True)
    pool_nano_cost = sum(example.nano_cost_usd for example in examples)
    pool_mini_cost = sum(example.mini_cost_usd for example in examples)
    pool_counts = Counter(outcome_bucket(example) for example in examples)

    records: list[dict[str, Any]] = []
    for final_size in args.final_sizes:
        if final_size < args.seed_size or final_size % 2:
            raise ValueError("final sizes must be even and at least seed-size")
        for seed in range(args.seeds):
            strategies: list[tuple[str, list[int], float | None, list[int]]] = []
            random_ids = random_select(len(examples), final_size, seed)
            oracle_ids = oracle_select(labels, final_size, seed)
            strategies.append(("random", random_ids, None, random_ids))
            strategies.append(("gold_oracle", oracle_ids, None, oracle_ids))
            prefix = f"seed{args.seed_size}"
            critical_labels = np.asarray(
                [int(outcome_bucket(example) == "mini_only") for example in examples],
                dtype=np.int64,
            )
            for name, features, fit_labels, iterative, exploration in (
                (f"{prefix}_question_error_one_shot", question_features, labels, False, 0.0),
                (f"{prefix}_nano_error_one_shot", nano_features, labels, False, 0.0),
                (f"{prefix}_nano_error_iterative", nano_features, labels, True, 0.0),
                (f"{prefix}_nano_error_iterative_explore20", nano_features, labels, True, 0.2),
                (f"{prefix}_nano_critical_iterative_explore20", nano_features, critical_labels, True, 0.2),
            ):
                selected, auc = seeded_select(
                    features,
                    fit_labels,
                    final_size,
                    args.seed_size,
                    args.batch_size,
                    seed,
                    iterative,
                    exploration,
                )
                strategies.append((name, selected, auc, selected))
            if final_size == 100:
                for probe_size in (120, 150, 200):
                    selected, probe = disagreement_probe_select(
                        examples, probe_size, final_size, seed
                    )
                    strategies.append(
                        (f"disagreement_probe{probe_size}", selected, None, probe)
                    )
            for strategy, selected, auc, mini_revealed in strategies:
                result = evaluate_selection(
                    examples,
                    selected,
                    labels,
                    pool_nano_cost,
                    pool_mini_cost,
                    auc,
                    mini_revealed,
                )
                records.append({"strategy": strategy, "final_size": final_size, "seed": seed, **result})

    payload = {
        "protocol": {
            "gold_hidden_until_selected": True,
            "mini_reveal": "selected-only or partial disagreement probe, by strategy",
            "nano_run_on_full_pool": True,
            "seed_size": args.seed_size,
            "batch_size": args.batch_size,
            "seeds": args.seeds,
            "features": "TF-IDF question/options/source plus optional Nano prediction/response",
        },
        "pool": {
            "size": len(examples),
            "nano_wrong": int(labels.sum()),
            "nano_correct": int(len(labels) - labels.sum()),
            "outcomes": {bucket: pool_counts[bucket] for bucket in BUCKETS},
            "nano_cost_usd": pool_nano_cost,
            "mini_cost_usd": pool_mini_cost,
        },
        "summary": aggregate(records),
        "records": records,
    }
    write_outputs(payload, args.output_prefix)
    print(args.output_prefix.with_suffix(".md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()


