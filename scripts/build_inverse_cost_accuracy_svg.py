import csv
import json
from pathlib import Path


ROOT = Path.cwd()
OUT_DIR = ROOT / "outputs" / "router_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SIZES = [25, 50, 75, 100, 125, 150, 200]


def read_json(path):
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
    comp = data["comparison"]
    best = scenario(comp, "gepa_best_on_testdata")
    points.append(
        {
            "label": str(size),
            "train_val": size,
            "cost": best["cost_usd"],
            "inverse_cost": 1 / best["cost_usd"],
            "accuracy": best["accuracy"],
            "correct": best["correct"],
            "mini_routes": best["mini_routes"],
            "nano_routes": best["nano_routes"],
            "saving": best["cost_savings_vs_baseline"],
        }
    )
    if baseline is None:
        baseline = {
            "all-mini": scenario(comp, "all_mini_on_testdata"),
            "all-nano": scenario(comp, "all_nano_on_testdata"),
            "oracle": scenario(comp, "oracle_on_testdata"),
        }

refs = []
for label in ["all-nano", "oracle", "all-mini"]:
    row = baseline[label]
    refs.append(
        {
            "label": label,
            "cost": row["cost_usd"],
            "inverse_cost": 1 / row["cost_usd"],
            "accuracy": row["accuracy"],
        }
    )

csv_path = OUT_DIR / "inverse_cost_accuracy_forced_qwen35_latest.csv"
with csv_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "train_val",
            "inverse_cost",
            "cost_usd",
            "accuracy",
            "correct",
            "mini_routes",
            "nano_routes",
            "cost_savings_vs_all_mini",
        ],
    )
    writer.writeheader()
    for p in sorted(points, key=lambda x: x["train_val"]):
        writer.writerow(
            {
                "train_val": p["train_val"],
                "inverse_cost": p["inverse_cost"],
                "cost_usd": p["cost"],
                "accuracy": p["accuracy"],
                "correct": p["correct"],
                "mini_routes": p["mini_routes"],
                "nano_routes": p["nano_routes"],
                "cost_savings_vs_all_mini": p["saving"],
            }
        )

width, height = 980, 640
left, right, top, bottom = 108, 48, 64, 92
plot_w = width - left - right
plot_h = height - top - bottom

all_x = [p["inverse_cost"] for p in points] + [r["inverse_cost"] for r in refs]
all_y = [p["accuracy"] for p in points] + [r["accuracy"] for r in refs]
x_min = min(all_x) * 0.94
x_max = max(all_x) * 1.04
y_min = max(0.68, min(all_y) - 0.015)
y_max = min(0.86, max(all_y) + 0.015)


def sx(x):
    return left + (x - x_min) / (x_max - x_min) * plot_w


def sy(y):
    return top + (y_max - y) / (y_max - y_min) * plot_h


def line(x1, y1, x2, y2, color="#D1D5DB", width_=1):
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width_}"/>'


svg = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
    '<rect width="100%" height="100%" fill="#FFFFFF"/>',
    f'<text x="{left}" y="34" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">Accuracy vs 1 / Cost: qwen3.5:latest forced-critical router</text>',
    f'<text x="{left}" y="56" font-family="Arial, sans-serif" font-size="13" fill="#4B5563">Higher x means lower answer-generation cost on the fixed 205-question test set.</text>',
]

for i in range(6):
    t = i / 5
    yv = y_min + t * (y_max - y_min)
    y = sy(yv)
    svg.append(line(left, y, width - right, y, "#E5E7EB"))
    svg.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#374151">{pct(yv):.0f}%</text>')

for i in range(6):
    t = i / 5
    xv = x_min + t * (x_max - x_min)
    x = sx(xv)
    svg.append(line(x, top, x, height - bottom, "#F3F4F6"))
    svg.append(f'<text x="{x:.1f}" y="{height - bottom + 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#374151">{xv:.1f}</text>')

svg.append(line(left, height - bottom, width - right, height - bottom, "#111827", 1.5))
svg.append(line(left, top, left, height - bottom, "#111827", 1.5))
svg.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 26}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#111827">1 / cost_usd on 205-question test set</text>')
svg.append(f'<text transform="translate(24 {top + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#111827">Accuracy</text>')

ordered = sorted(points, key=lambda p: p["train_val"])
polyline = " ".join(f'{sx(p["inverse_cost"]):.1f},{sy(p["accuracy"]):.1f}' for p in ordered)
svg.append(f'<polyline points="{polyline}" fill="none" stroke="#2563EB" stroke-width="2.5"/>')

for p in ordered:
    x = sx(p["inverse_cost"])
    y = sy(p["accuracy"])
    svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="#2563EB" stroke="#FFFFFF" stroke-width="2"/>')
    svg.append(f'<text x="{x + 10:.1f}" y="{y - 10:.1f}" font-family="Arial, sans-serif" font-size="12" font-weight="700" fill="#1D4ED8">{p["label"]}</text>')
    svg.append(f'<text x="{x + 10:.1f}" y="{y + 5:.1f}" font-family="Arial, sans-serif" font-size="11" fill="#374151">{pct(p["accuracy"]):.1f}%, 1/cost={p["inverse_cost"]:.1f}</text>')

styles = {"all-nano": "#6B7280", "oracle": "#059669", "all-mini": "#DC2626"}
for r in refs:
    x = sx(r["inverse_cost"])
    y = sy(r["accuracy"])
    color = styles[r["label"]]
    svg.append(f'<rect x="{x - 7:.1f}" y="{y - 7:.1f}" width="14" height="14" fill="{color}" stroke="#FFFFFF" stroke-width="2"/>')
    svg.append(f'<text x="{x + 11:.1f}" y="{y - 2:.1f}" font-family="Arial, sans-serif" font-size="12" font-weight="700" fill="{color}">{r["label"]}</text>')
    svg.append(f'<text x="{x + 11:.1f}" y="{y + 13:.1f}" font-family="Arial, sans-serif" font-size="11" fill="#374151">{pct(r["accuracy"]):.1f}%, {r["inverse_cost"]:.1f}</text>')

svg.append("</svg>")

svg_path = OUT_DIR / "inverse_cost_accuracy_forced_qwen35_latest.svg"
svg_path.write_text("\n".join(svg), encoding="utf-8")

print(svg_path)
print(csv_path)
for p in ordered:
    print(p["train_val"], f'inv_cost={p["inverse_cost"]:.4f}', f'cost={p["cost"]:.6f}', f'acc={p["accuracy"]:.6f}')
