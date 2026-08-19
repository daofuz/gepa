import csv
import json
from pathlib import Path


ROOT = Path.cwd()
OUT_DIR = ROOT / "outputs" / "router_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SIZES = [25, 50, 75, 100, 125, 150, 200]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def scenario(rows, name):
    for row in rows:
        if row["scenario"] == name:
            return row
    raise KeyError(name)


def pct(value):
    return value * 100


points = []
baseline = None
for size in SIZES:
    path = ROOT / f"routing_cost_comparison_testdata_hybrid_half1_trainval{size}_openai.json"
    if not path.exists():
        continue
    data = read_json(path)
    comparison = data["comparison"]
    best = scenario(comparison, "gepa_best_on_testdata")
    points.append(
        {
            "label": str(size),
            "train_val": size,
            "cost": best["cost_usd"],
            "accuracy": best["accuracy"],
            "correct": best["correct"],
            "mini_routes": best["mini_routes"],
            "nano_routes": best["nano_routes"],
            "saving": best["cost_savings_vs_baseline"],
        }
    )
    if baseline is None:
        baseline = {
            "all-mini": scenario(comparison, "all_mini_on_testdata"),
            "all-nano": scenario(comparison, "all_nano_on_testdata"),
            "oracle": scenario(comparison, "oracle_on_testdata"),
        }

if not points:
    raise SystemExit("No result files found.")

reference_points = [
    {
        "label": "all-nano",
        "cost": baseline["all-nano"]["cost_usd"],
        "accuracy": baseline["all-nano"]["accuracy"],
    },
    {
        "label": "oracle",
        "cost": baseline["oracle"]["cost_usd"],
        "accuracy": baseline["oracle"]["accuracy"],
    },
    {
        "label": "all-mini",
        "cost": baseline["all-mini"]["cost_usd"],
        "accuracy": baseline["all-mini"]["accuracy"],
    },
]

csv_path = OUT_DIR / "cost_accuracy_forced_qwen35_latest.csv"
with csv_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "train_val",
            "cost_usd",
            "accuracy",
            "correct",
            "mini_routes",
            "nano_routes",
            "cost_savings_vs_all_mini",
        ],
    )
    writer.writeheader()
    for p in points:
        writer.writerow(
            {
                "train_val": p["train_val"],
                "cost_usd": p["cost"],
                "accuracy": p["accuracy"],
                "correct": p["correct"],
                "mini_routes": p["mini_routes"],
                "nano_routes": p["nano_routes"],
                "cost_savings_vs_all_mini": p["saving"],
            }
        )

width = 980
height = 640
left = 96
right = 48
top = 64
bottom = 92
plot_w = width - left - right
plot_h = height - top - bottom

all_costs = [p["cost"] for p in points] + [p["cost"] for p in reference_points]
all_accs = [p["accuracy"] for p in points] + [p["accuracy"] for p in reference_points]
x_min = min(all_costs) * 0.92
x_max = max(all_costs) * 1.04
y_min = max(0.68, min(all_accs) - 0.015)
y_max = min(0.86, max(all_accs) + 0.015)


def sx(cost):
    return left + (cost - x_min) / (x_max - x_min) * plot_w


def sy(acc):
    return top + (y_max - acc) / (y_max - y_min) * plot_h


def esc(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def line(x1, y1, x2, y2, color="#D1D5DB", width_=1, dash=""):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width_}"{dash_attr}/>'


svg = []
svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
svg.append('<rect width="100%" height="100%" fill="#FFFFFF"/>')
svg.append(f'<text x="{left}" y="34" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">Cost vs Accuracy: qwen3.5:latest forced-critical router</text>')
svg.append(f'<text x="{left}" y="56" font-family="Arial, sans-serif" font-size="13" fill="#4B5563">Fixed test set: second half, 205 questions. Cost excludes router/GEPA/reflection overhead.</text>')

for i in range(6):
    t = i / 5
    acc = y_min + t * (y_max - y_min)
    y = sy(acc)
    svg.append(line(left, y, width - right, y, "#E5E7EB"))
    svg.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#374151">{pct(acc):.0f}%</text>')

x_ticks = 5
for i in range(x_ticks + 1):
    t = i / x_ticks
    cost = x_min + t * (x_max - x_min)
    x = sx(cost)
    svg.append(line(x, top, x, height - bottom, "#F3F4F6"))
    svg.append(f'<text x="{x:.1f}" y="{height - bottom + 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#374151">${cost:.3f}</text>')

svg.append(line(left, height - bottom, width - right, height - bottom, "#111827", 1.5))
svg.append(line(left, top, left, height - bottom, "#111827", 1.5))
svg.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 26}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#111827">Cost on 205-question test set (USD)</text>')
svg.append(f'<text transform="translate(24 {top + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#111827">Accuracy</text>')

points_sorted = sorted(points, key=lambda p: p["train_val"])
polyline = " ".join(f'{sx(p["cost"]):.1f},{sy(p["accuracy"]):.1f}' for p in points_sorted)
svg.append(f'<polyline points="{polyline}" fill="none" stroke="#2563EB" stroke-width="2.5"/>')

for p in points_sorted:
    x = sx(p["cost"])
    y = sy(p["accuracy"])
    svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="#2563EB" stroke="#FFFFFF" stroke-width="2"/>')
    label = f'{p["label"]}'
    svg.append(f'<text x="{x + 10:.1f}" y="{y - 10:.1f}" font-family="Arial, sans-serif" font-size="12" font-weight="700" fill="#1D4ED8">{esc(label)}</text>')
    svg.append(f'<text x="{x + 10:.1f}" y="{y + 5:.1f}" font-family="Arial, sans-serif" font-size="11" fill="#374151">{pct(p["accuracy"]):.1f}%, ${p["cost"]:.3f}</text>')

ref_styles = {
    "all-nano": ("#6B7280", 7),
    "oracle": ("#059669", 8),
    "all-mini": ("#DC2626", 8),
}
for ref in reference_points:
    color, radius = ref_styles[ref["label"]]
    x = sx(ref["cost"])
    y = sy(ref["accuracy"])
    svg.append(f'<rect x="{x - radius:.1f}" y="{y - radius:.1f}" width="{radius * 2}" height="{radius * 2}" fill="{color}" stroke="#FFFFFF" stroke-width="2"/>')
    svg.append(f'<text x="{x + 11:.1f}" y="{y - 2:.1f}" font-family="Arial, sans-serif" font-size="12" font-weight="700" fill="{color}">{esc(ref["label"])}</text>')
    svg.append(f'<text x="{x + 11:.1f}" y="{y + 13:.1f}" font-family="Arial, sans-serif" font-size="11" fill="#374151">{pct(ref["accuracy"]):.1f}%, ${ref["cost"]:.3f}</text>')

legend_x = width - right - 250
legend_y = top + 8
svg.append(f'<rect x="{legend_x}" y="{legend_y}" width="250" height="74" rx="6" fill="#F9FAFB" stroke="#E5E7EB"/>')
svg.append(f'<circle cx="{legend_x + 18}" cy="{legend_y + 22}" r="6" fill="#2563EB"/>')
svg.append(f'<text x="{legend_x + 34}" y="{legend_y + 26}" font-family="Arial, sans-serif" font-size="12" fill="#111827">GEPA router by train+val size</text>')
svg.append(f'<rect x="{legend_x + 12}" y="{legend_y + 39}" width="12" height="12" fill="#059669"/>')
svg.append(f'<text x="{legend_x + 34}" y="{legend_y + 50}" font-family="Arial, sans-serif" font-size="12" fill="#111827">Oracle reference</text>')
svg.append(f'<rect x="{legend_x + 12}" y="{legend_y + 56}" width="12" height="12" fill="#DC2626"/>')
svg.append(f'<text x="{legend_x + 34}" y="{legend_y + 67}" font-family="Arial, sans-serif" font-size="12" fill="#111827">All-mini / all-nano references</text>')

svg.append("</svg>")

svg_path = OUT_DIR / "cost_accuracy_forced_qwen35_latest.svg"
svg_path.write_text("\n".join(svg), encoding="utf-8")

print(svg_path)
print(csv_path)
for p in points_sorted:
    print(p["train_val"], f'cost={p["cost"]:.6f}', f'acc={p["accuracy"]:.6f}', f'correct={p["correct"]}', f'mini={p["mini_routes"]}', f'nano={p["nano_routes"]}')
