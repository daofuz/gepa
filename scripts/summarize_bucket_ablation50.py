import json
from collections import Counter, defaultdict
from pathlib import Path


EXPERIMENTS = [
    ("forced_critical_existing", "routing_split_half1_sweep_trainval50_withcritical_seed0.json", "routing_cost_comparison_testdata_hybrid_half1_trainval50_openai.json", "gepa_router_prompt_result_hybrid_half1_trainval50_openai_reflect.json"),
    ("random_control_existing", "routing_split_half1_random_control_trainval50_seed0.json", "routing_cost_comparison_testdata_hybrid_half1_random_control_trainval50_openai.json", "gepa_router_prompt_result_hybrid_half1_random_control_trainval50_openai_reflect.json"),
    ("both_correct_heavy", "routing_split_half1_ablation50_both_correct_heavy_seed0.json", "routing_cost_comparison_testdata_hybrid_half1_ablation50_both_correct_heavy_openai.json", "gepa_router_prompt_result_hybrid_half1_ablation50_both_correct_heavy_openai_reflect.json"),
    ("both_wrong_heavy", "routing_split_half1_ablation50_both_wrong_heavy_seed0.json", "routing_cost_comparison_testdata_hybrid_half1_ablation50_both_wrong_heavy_openai.json", "gepa_router_prompt_result_hybrid_half1_ablation50_both_wrong_heavy_openai_reflect.json"),
    ("balancedish", "routing_split_half1_ablation50_balancedish_seed0.json", "routing_cost_comparison_testdata_hybrid_half1_ablation50_balancedish_openai.json", "gepa_router_prompt_result_hybrid_half1_ablation50_balancedish_openai_reflect.json"),
    ("critical_both_wrong", "routing_split_half1_ablation50_critical_both_wrong_seed0.json", "routing_cost_comparison_testdata_hybrid_half1_ablation50_critical_both_wrong_openai.json", "gepa_router_prompt_result_hybrid_half1_ablation50_critical_both_wrong_openai_reflect.json"),
    ("critical_both_correct", "routing_split_half1_ablation50_critical_both_correct_seed0.json", "routing_cost_comparison_testdata_hybrid_half1_ablation50_critical_both_correct_openai.json", "gepa_router_prompt_result_hybrid_half1_ablation50_critical_both_correct_openai_reflect.json"),
]


def scenario(rows, name):
    return next(row for row in rows if row["scenario"] == name)


def pct(value):
    return 100 * value


rows = []
for name, split_file, result_file, gepa_file in EXPERIMENTS:
    if not Path(split_file).exists() or not Path(result_file).exists():
        continue
    split = json.loads(Path(split_file).read_text(encoding="utf-8"))
    result = json.loads(Path(result_file).read_text(encoding="utf-8"))
    gepa = json.loads(Path(gepa_file).read_text(encoding="utf-8")) if Path(gepa_file).exists() else {}
    best = scenario(result["comparison"], "gepa_best_on_testdata")
    train_val = split["bucket_counts"].get("train_val")
    if train_val is None:
        train = Counter(split["bucket_counts"]["train"])
        val = Counter(split["bucket_counts"]["validation"])
        train_val = dict(train + val)

    buckets = defaultdict(Counter)
    for trace in result["router_traces"]:
        route = trace.get("selected_route") or trace.get("route")
        mini = trace["mini_correct"]
        nano = trace["nano_correct"]
        if mini and nano:
            bucket = "both_correct"
        elif mini and not nano:
            bucket = "critical"
        elif nano and not mini:
            bucket = "nano_only"
        else:
            bucket = "both_wrong"
        buckets[bucket]["total"] += 1
        buckets[bucket][route] += 1
        if (route == "mini" and mini) or (route == "nano" and nano):
            buckets[bucket]["correct"] += 1

    rows.append(
        {
            "name": name,
            "critical_trainval": train_val.get("critical", 0),
            "nano_only_trainval": train_val.get("nano_only", 0),
            "both_correct_trainval": train_val.get("both_correct", 0),
            "both_wrong_trainval": train_val.get("both_wrong", 0),
            "accuracy": best["accuracy"],
            "correct": best["correct"],
            "cost": best["cost_usd"],
            "savings": best["cost_savings_vs_baseline"],
            "mini_routes": best["mini_routes"],
            "nano_routes": best["nano_routes"],
            "critical_mini": buckets["critical"]["mini"],
            "critical_total": buckets["critical"]["total"],
            "both_wrong_mini": buckets["both_wrong"]["mini"],
            "both_wrong_total": buckets["both_wrong"]["total"],
            "selected_candidate": gepa.get("selected_candidate_index"),
            "train_best_accuracy": gepa.get("best_summary", {}).get("accuracy"),
            "train_best_mini_routes": gepa.get("best_summary", {}).get("mini_routes"),
        }
    )

rows.sort(key=lambda row: (-row["accuracy"], row["cost"]))

headers = [
    "name",
    "crit",
    "nano_only",
    "both_correct",
    "both_wrong",
    "acc",
    "correct",
    "cost",
    "saving",
    "mini_routes",
    "critical_mini",
    "both_wrong_mini",
]

print("\t".join(headers))
for row in rows:
    print(
        "\t".join(
            [
                row["name"],
                str(row["critical_trainval"]),
                str(row["nano_only_trainval"]),
                str(row["both_correct_trainval"]),
                str(row["both_wrong_trainval"]),
                f"{pct(row['accuracy']):.2f}%",
                f"{row['correct']}/205",
                f"${row['cost']:.4f}",
                f"{pct(row['savings']):.2f}%",
                str(row["mini_routes"]),
                f"{row['critical_mini']}/{row['critical_total']}",
                f"{row['both_wrong_mini']}/{row['both_wrong_total']}",
            ]
        )
    )

