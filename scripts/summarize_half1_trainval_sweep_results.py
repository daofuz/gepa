import argparse
import json
from pathlib import Path


def pct(value):
    if value is None:
        return ""
    return f"{100 * value:.2f}%"


def best_comparison_row(data):
    comparison = data.get("comparison") or []
    preferred = [
        "adaevolve_best_on_testdata",
        "gepa_best_on_testdata",
        "router_best_on_testdata",
    ]
    for scenario in preferred:
        for row in comparison:
            if row.get("scenario") == scenario:
                return row
    for row in comparison:
        if str(row.get("scenario", "")).endswith("_best_on_testdata"):
            return row
    return {}


def comparison_scenario(data, scenario):
    for row in data.get("comparison") or []:
        if row.get("scenario") == scenario:
            return row
    return {}


def get_metric(data, *names):
    if any(name in {"all_mini_accuracy", "mini_accuracy"} for name in names):
        row = comparison_scenario(data, "all_mini_on_testdata")
        if "accuracy" in row:
            return row["accuracy"]
    if any(name in {"all_nano_accuracy", "nano_accuracy"} for name in names):
        row = comparison_scenario(data, "all_nano_on_testdata")
        if "accuracy" in row:
            return row["accuracy"]

    row = best_comparison_row(data)
    aliases = {
        "cost_savings_vs_all_mini": "cost_savings_vs_baseline",
        "cost_savings": "cost_savings_vs_baseline",
    }
    for name in names:
        if name in row:
            return row[name]
        alias = aliases.get(name)
        if alias and alias in row:
            return row[alias]
    for name in names:
        if name in data:
            return data[name]
    metrics = data.get("metrics", {})
    for name in names:
        if name in metrics:
            return metrics[name]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="50,100,150,200")
    parser.add_argument("--prefix", default="routing_cost_comparison_testdata_hybrid_half1_trainval")
    parser.add_argument("--suffix", default="_openai.json")
    args = parser.parse_args()

    rows = []
    for size_text in args.sizes.split(","):
        size = size_text.strip()
        path = Path(f"{args.prefix}{size}{args.suffix}")
        if not path.exists():
            rows.append([size, "missing", "", "", "", "", "", ""])
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        router_acc = get_metric(data, "router_accuracy", "accuracy")
        all_mini = get_metric(data, "all_mini_accuracy", "mini_accuracy")
        all_nano = get_metric(data, "all_nano_accuracy", "nano_accuracy")
        cost_savings = get_metric(data, "cost_savings_vs_all_mini", "cost_savings")
        row = best_comparison_row(data)
        route_counts = data.get("route_counts", {})
        rows.append(
            [
                size,
                pct(router_acc),
                pct(all_mini),
                pct(all_nano),
                row.get("mini_routes", route_counts.get("mini", data.get("mini_routes", ""))),
                row.get("nano_routes", route_counts.get("nano", data.get("nano_routes", ""))),
                pct(cost_savings),
                str(path),
            ]
        )

    headers = ["train+val", "router_acc", "all_mini", "all_nano", "mini_routes", "nano_routes", "savings", "file"]
    widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))))


if __name__ == "__main__":
    main()
