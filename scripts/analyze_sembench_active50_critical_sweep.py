#!/usr/bin/env python3
"""Analyze the completed active-50 critical-composition sweep."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs/sembench_active50_critical_sweep.json"
CSV_OUTPUT = ROOT / "outputs/sembench_active50_critical_sweep_analysis.csv"
MD_OUTPUT = ROOT / "outputs/sembench_active50_critical_sweep_findings.md"
SVG_OUTPUT = ROOT / "outputs/sembench_active50_critical_sweep.svg"
POLICY = "validation_max_accuracy"


def run_metric(run: dict[str, Any], field: str) -> float:
    return float(run["policies"][POLICY]["test"][field])


def ols_slope(xs: list[float], ys: list[float]) -> float:
    xbar, ybar = mean(xs), mean(ys)
    denominator = sum((x - xbar) ** 2 for x in xs)
    return sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / denominator


def pearson(xs: list[float], ys: list[float]) -> float:
    xbar, ybar = mean(xs), mean(ys)
    numerator = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - xbar) ** 2 for x in xs) * sum((y - ybar) ** 2 for y in ys)
    )
    return numerator / denominator


def svg_plot(rows: list[dict[str, float]]) -> None:
    width, height = 940, 680
    left, right = 78, 30
    top_a, top_b, panel_h = 78, 382, 220
    xs = [row["critical"] for row in rows]
    acc = [100 * row["accuracy_mean"] for row in rows]
    acc_sd = [100 * row["accuracy_std"] for row in rows]
    saving = [100 * row["saving_mean"] for row in rows]
    recall = [100 * row["recall_mean"] for row in rows]

    def xcoord(value: float) -> float:
        return left + (value - xs[0]) / (xs[-1] - xs[0]) * (width - left - right)

    acc_lo = math.floor((min(v - s for v, s in zip(acc, acc_sd)) - 0.5) / 2) * 2
    acc_hi = math.ceil((max(v + s for v, s in zip(acc, acc_sd)) + 0.5) / 2) * 2

    def ycoord(value: float, lo: float, hi: float, top: float) -> float:
        return top + panel_h - (value - lo) / (hi - lo) * panel_h

    def points(values: list[float], lo: float, hi: float, top: float) -> str:
        return " ".join(
            f"{xcoord(x):.1f},{ycoord(y, lo, hi, top):.1f}"
            for x, y in zip(xs, values)
        )

    upper = [(xcoord(x), ycoord(v + s, acc_lo, acc_hi, top_a)) for x, v, s in zip(xs, acc, acc_sd)]
    lower = [(xcoord(x), ycoord(v - s, acc_lo, acc_hi, top_a)) for x, v, s in reversed(list(zip(xs, acc, acc_sd)))]
    band = " ".join(f"{x:.1f},{y:.1f}" for x, y in upper + lower)
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.axis{stroke:#555}.grid{stroke:#ddd}.lab{font-size:14px}.title{font-size:19px;font-weight:600}.leg{font-size:13px}</style>',
        '<text x="470" y="30" text-anchor="middle" class="title">50-label critical-case composition sweep</text>',
        '<text x="470" y="52" text-anchor="middle" class="lab">MO=k, BC=47-k, NO=2, BW=1; mean of 3 seeds</text>',
    ]
    for top, lo, hi, ylabel in ((top_a, acc_lo, acc_hi, "Accuracy (%)"), (top_b, 0, 100, "Rate (%)")):
        for tick in range(6):
            value = lo + (hi - lo) * tick / 5
            y = ycoord(value, lo, hi, top)
            body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid"/>')
            body.append(f'<text x="{left-9}" y="{y+5:.1f}" text-anchor="end" class="lab">{value:.1f}</text>')
        body.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+panel_h}" class="axis"/>')
        body.append(f'<line x1="{left}" y1="{top+panel_h}" x2="{width-right}" y2="{top+panel_h}" class="axis"/>')
        body.append(f'<text x="20" y="{top+panel_h/2}" transform="rotate(-90 20 {top+panel_h/2})" text-anchor="middle" class="lab">{ylabel}</text>')
    for tick in range(1, 47, 5):
        x = xcoord(tick)
        body.append(f'<line x1="{x:.1f}" y1="{top_b+panel_h}" x2="{x:.1f}" y2="{top_b+panel_h+5}" class="axis"/>')
        body.append(f'<text x="{x:.1f}" y="{top_b+panel_h+22}" text-anchor="middle" class="lab">{tick}</text>')
    body.extend([
        f'<polygon points="{band}" fill="#1f5a94" opacity="0.15"/>',
        f'<polyline points="{points(acc, acc_lo, acc_hi, top_a)}" fill="none" stroke="#1f5a94" stroke-width="2.4"/>',
        f'<polyline points="{points(recall, 0, 100, top_b)}" fill="none" stroke="#bd4b35" stroke-width="2.4"/>',
        f'<polyline points="{points(saving, 0, 100, top_b)}" fill="none" stroke="#428f5b" stroke-width="2.4"/>',
        '<line x1="720" y1="67" x2="750" y2="67" stroke="#1f5a94" stroke-width="3"/><text x="758" y="72" class="leg">accuracy (±1 SD)</text>',
        '<line x1="675" y1="369" x2="705" y2="369" stroke="#bd4b35" stroke-width="3"/><text x="713" y="374" class="leg">MO recall</text>',
        '<line x1="795" y1="369" x2="825" y2="369" stroke="#428f5b" stroke-width="3"/><text x="833" y="374" class="leg">Mini saving</text>',
        '<text x="470" y="665" text-anchor="middle" class="lab">Mini-only examples among 50 training labels</text>',
        '</svg>',
    ])
    SVG_OUTPUT.write_text("\n".join(body), encoding="utf-8")


def main() -> None:
    data = json.loads(INPUT.read_text(encoding="utf-8"))
    rows: list[dict[str, float]] = []
    for critical in range(1, 47):
        runs = data["runs"][str(critical)]
        if len(runs) != 3:
            raise AssertionError(f"critical={critical} has {len(runs)} runs")
        values = {
            "accuracy": [run_metric(run, "selected_accuracy") for run in runs],
            "saving": [run_metric(run, "mini_saving_vs_all_mini") for run in runs],
            "recall": [run_metric(run, "mini_only_recall") for run in runs],
        }
        rows.append({
            "critical": float(critical),
            "both_correct": float(47 - critical),
            "accuracy_mean": mean(values["accuracy"]),
            "accuracy_std": pstdev(values["accuracy"]),
            "saving_mean": mean(values["saving"]),
            "saving_std": pstdev(values["saving"]),
            "recall_mean": mean(values["recall"]),
            "recall_std": pstdev(values["recall"]),
        })

    with CSV_OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    best_accuracy = max(rows, key=lambda row: (row["accuracy_mean"], row["saving_mean"]))
    balanced_best = max(
        (row for row in rows if row["saving_mean"] >= 0.20),
        key=lambda row: (row["accuracy_mean"], row["saving_mean"]),
    )
    windows = [(1, 10), (11, 20), (21, 30), (31, 39), (40, 46)]
    window_rows = []
    for lo, hi in windows:
        selected = [row for row in rows if lo <= row["critical"] <= hi]
        # First average each seed over the window, then aggregate over seeds.
        per_seed = []
        for seed_index in range(3):
            runs = [data["runs"][str(k)][seed_index] for k in range(lo, hi + 1)]
            per_seed.append({
                "accuracy": mean(run_metric(run, "selected_accuracy") for run in runs),
                "saving": mean(run_metric(run, "mini_saving_vs_all_mini") for run in runs),
                "recall": mean(run_metric(run, "mini_only_recall") for run in runs),
            })
        window_rows.append({
            "range": f"{lo}-{hi}",
            "accuracy": mean(value["accuracy"] for value in per_seed),
            "accuracy_std": pstdev(value["accuracy"] for value in per_seed),
            "saving": mean(value["saving"] for value in per_seed),
            "recall": mean(value["recall"] for value in per_seed),
            "slope_pp_per_mo": 100 * ols_slope(
                [row["critical"] for row in selected],
                [row["accuracy_mean"] for row in selected],
            ),
        })

    early = rows[:10]
    high = rows[39:]
    paired_deltas = []
    for critical in range(1, 46):
        for seed_index in range(3):
            before = data["runs"][str(critical)][seed_index]
            after = data["runs"][str(critical + 1)][seed_index]
            paired_deltas.append(run_metric(after, "selected_accuracy") - run_metric(before, "selected_accuracy"))
    positive = sum(delta > 0 for delta in paired_deltas)
    negative = sum(delta < 0 for delta in paired_deltas)
    unchanged = sum(delta == 0 for delta in paired_deltas)

    lines = [
        "# Active-50 critical-case composition sweep",
        "",
        "Protocol: fixed 50 labels; MO=k, BC=47-k, NO=2, BW=1; k=1..46 in step 1; three seeds; frozen DistilBERT with 8 soft tokens and two correctness heads; inverse-expected-cost weighted BCE; no prior; validation-max-accuracy threshold; post-hoc held-out 200.",
        "",
        "## Main findings",
        "",
        f"- Best exact point: MO={int(best_accuracy['critical'])}, BC={int(best_accuracy['both_correct'])}, accuracy={100*best_accuracy['accuracy_mean']:.2f}% ± {100*best_accuracy['accuracy_std']:.2f}, Mini saving={100*best_accuracy['saving_mean']:.2f}%, MO recall={100*best_accuracy['recall_mean']:.2f}%.",
        f"- Best point retaining at least 20% Mini saving: MO={int(balanced_best['critical'])}, accuracy={100*balanced_best['accuracy_mean']:.2f}%, saving={100*balanced_best['saving_mean']:.2f}%.",
        f"- Low-critical proxy MO=1: accuracy={100*rows[0]['accuracy_mean']:.2f}%, saving={100*rows[0]['saving_mean']:.2f}%, MO recall={100*rows[0]['recall_mean']:.2f}%.",
        f"- Old enriched-recipe neighborhood MO=22: accuracy={100*rows[21]['accuracy_mean']:.2f}%, saving={100*rows[21]['saving_mean']:.2f}%, MO recall={100*rows[21]['recall_mean']:.2f}%.",
        f"- Maximum enrichment MO=46: accuracy={100*rows[-1]['accuracy_mean']:.2f}%, saving={100*rows[-1]['saving_mean']:.2f}%, MO recall={100*rows[-1]['recall_mean']:.2f}%.",
        f"- Across all 135 adjacent seed-paired steps: {positive} increased accuracy, {negative} decreased it, and {unchanged} were unchanged. A single extra MO example is therefore not reliably monotonic at n=50.",
        f"- Correlation across exact points: MO count vs accuracy r={pearson([r['critical'] for r in rows], [r['accuracy_mean'] for r in rows]):.3f}; MO count vs Mini saving r={pearson([r['critical'] for r in rows], [r['saving_mean'] for r in rows]):.3f}; MO count vs MO recall r={pearson([r['critical'] for r in rows], [r['recall_mean'] for r in rows]):.3f}.",
        "",
        "## Windowed trend",
        "",
        "| MO range | Accuracy | Mini saving | MO recall | Accuracy slope (pp / +1 MO) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in window_rows:
        lines.append(
            f"| {row['range']} | {100*row['accuracy']:.2f}% ± {100*row['accuracy_std']:.2f} | "
            f"{100*row['saving']:.2f}% | {100*row['recall']:.2f}% | {row['slope_pp_per_mo']:+.3f} |"
        )
    lines.extend([
        "",
        "Interpretation: critical-case enrichment improves the router most clearly when moving away from very sparse MO coverage, but the raw step-1 curve is noisy and not monotone. Moderate enrichment preserves BC coverage; extreme enrichment can make threshold calibration and seed choice dominate, and it does not deliver a consistent accuracy gain.",
        "",
        "Caveat: exact MO counts are available only after human gold annotation. Run-both acquisition can select disagreement candidates before labeling, but this composition sweep is a post-annotation diagnostic, not an implementable pre-label oracle selector.",
    ])
    MD_OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    svg_plot(rows)
    print("\n".join(lines))
    print(f"Saved {CSV_OUTPUT}, {MD_OUTPUT}, and {SVG_OUTPUT}")


if __name__ == "__main__":
    main()
