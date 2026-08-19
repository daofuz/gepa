#!/usr/bin/env python3
"""Cost-aware calibration for saved Computer QA OOF ensemble scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from optimize_ollama_router_gepa import load_examples  # noqa: E402
from scripts.run_computer_qa_regret_softprompt import compact_summary, routes  # noqa: E402


def candidates(scores: list[float]) -> list[float]:
    values = sorted(set(scores))
    if not values:
        return [0.0]
    thresholds = [values[0] - 1e-9, values[-1] + 1e-9]
    thresholds.extend(values)
    thresholds.extend((left + right) / 2 for left, right in zip(values, values[1:]))
    return sorted(set(thresholds))


def select_cost_aware_threshold(
    examples: list[Any], scores: list[float], accuracy_floor: float
) -> tuple[float, dict[str, Any]]:
    feasible: list[tuple[float, float, float, dict[str, Any]]] = []
    for threshold in candidates(scores):
        summary = compact_summary(examples, routes(scores, threshold))
        if summary["accuracy"] + 1e-12 >= accuracy_floor:
            feasible.append(
                (summary["mini_rate"], -summary["accuracy"], threshold, summary)
            )
    if not feasible:
        raise RuntimeError(f"No threshold reaches accuracy floor {accuracy_floor:.6f}")
    _, _, threshold, summary = min(feasible)
    return threshold, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "outputs/computer_qa_regret_bert_base_oof5_ensemble.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/computer_qa_regret_bert_base_oof5_cost_curve.json",
    )
    parser.add_argument(
        "--budgets-pp", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0, 3.0]
    )
    args = parser.parse_args()

    saved = json.loads(args.input.read_text(encoding="utf-8"))
    config = saved["config"]
    all_examples = load_examples(
        Path(config["mini_file"]),
        Path(config["nano_file"]),
        mini_cost_model="gpt-4.1-mini",
        nano_cost_model="gpt-4.1-nano",
    )
    development = all_examples[:205]
    test = all_examples[205:]
    development_by_id = {example.question_id: example for example in development}

    centered_by_id: dict[int, float] = {}
    for fold in saved["folds"]:
        threshold = float(fold["fold_threshold"])
        for question_id, score in zip(fold["validation_ids"], fold["validation_scores"]):
            centered_by_id[int(question_id)] = float(score) - threshold
    centered_oof = [centered_by_id[example.question_id] for example in development]
    if len(centered_by_id) != len(development_by_id):
        raise AssertionError("Centered OOF scores do not cover development examples")

    centered_test = []
    for index in range(len(test)):
        margins = [
            float(fold["test_scores"][index]) - float(fold["fold_threshold"])
            for fold in saved["folds"]
        ]
        centered_test.append(sum(margins) / len(margins))

    all_mini_development = compact_summary(development, ["mini"] * len(development))
    policies = {}
    for budget_pp in args.budgets_pp:
        floor = all_mini_development["accuracy"] - budget_pp / 100.0
        threshold, oof_summary = select_cost_aware_threshold(
            development, centered_oof, floor
        )
        policies[f"budget_{budget_pp:g}pp"] = {
            "accuracy_floor": floor,
            "centered_margin_threshold": threshold,
            "oof": oof_summary,
            "test": compact_summary(test, routes(centered_test, threshold)),
        }

    output = {
        "source": str(args.input),
        "method": "fold-centered OOF calibration; minimize mini rate subject to an accuracy floor",
        "all_mini_development": all_mini_development,
        "all_mini_test": compact_summary(test, ["mini"] * len(test)),
        "all_nano_test": compact_summary(test, ["nano"] * len(test)),
        "policies": policies,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
