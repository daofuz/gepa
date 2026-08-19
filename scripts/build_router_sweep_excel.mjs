import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = process.cwd();
const outputDir = path.join(root, "outputs", "router_sweep");
const outputPath = path.join(outputDir, "half1_trainval_sweep_results.xlsx");

const sizes = [50, 100, 150, 200];

function pct(n, d) {
  return d ? n / d : 0;
}

function scenario(comparison, name) {
  const row = comparison.find((item) => item.scenario === name);
  if (!row) throw new Error(`Missing scenario ${name}`);
  return row;
}

async function readJson(file) {
  return JSON.parse(await fs.readFile(path.join(root, file), "utf8"));
}

function bucketName(trace) {
  if (trace.mini_correct && trace.nano_correct) return "both_correct";
  if (trace.mini_correct && !trace.nano_correct) return "critical_mini_only";
  if (!trace.mini_correct && trace.nano_correct) return "nano_only";
  return "both_wrong";
}

const rows = [];
const bucketRows = [];
const gepaRows = [];

for (const size of sizes) {
  const evalData = await readJson(`routing_cost_comparison_testdata_hybrid_half1_trainval${size}_openai.json`);
  const gepaData = await readJson(`gepa_router_prompt_result_hybrid_half1_trainval${size}_openai_reflect.json`);
  const allMini = scenario(evalData.comparison, "all_mini_on_testdata");
  const allNano = scenario(evalData.comparison, "all_nano_on_testdata");
  const oracle = scenario(evalData.comparison, "oracle_on_testdata");
  const best = scenario(evalData.comparison, "gepa_best_on_testdata");
  const traces = evalData.router_traces || [];

  const bucketCounts = new Map();
  for (const name of ["both_correct", "critical_mini_only", "nano_only", "both_wrong"]) {
    bucketCounts.set(name, { total: 0, mini: 0, nano: 0, routerCorrect: 0 });
  }

  const easyCounts = { direct: 0, defer: 0 };
  for (const trace of traces) {
    const bucket = bucketCounts.get(bucketName(trace));
    const route = trace.selected_route || trace.route;
    bucket.total += 1;
    if (route === "mini") bucket.mini += 1;
    if (route === "nano") bucket.nano += 1;
    if ((route === "mini" && trace.mini_correct) || (route === "nano" && trace.nano_correct)) {
      bucket.routerCorrect += 1;
    }
    if (trace.easy_router_source === "easy_router") easyCounts.direct += 1;
    if (trace.easy_router_source === "easy_router_defer") easyCounts.defer += 1;
  }

  const critical = bucketCounts.get("critical_mini_only");
  const bothWrong = bucketCounts.get("both_wrong");

  rows.push([
    size,
    best.questions,
    best.accuracy,
    best.correct,
    allMini.accuracy,
    allNano.accuracy,
    oracle.accuracy,
    best.accuracy_delta_vs_baseline,
    best.mini_routes,
    best.nano_routes,
    best.cost_usd,
    best.cost_per_1000_questions,
    best.cost_savings_vs_baseline,
    critical.mini,
    critical.total,
    pct(critical.mini, critical.total),
    bothWrong.mini,
    bothWrong.total,
    pct(bothWrong.mini, bothWrong.total),
    easyCounts.direct,
    easyCounts.defer,
  ]);

  for (const [name, counts] of bucketCounts.entries()) {
    bucketRows.push([
      size,
      name,
      counts.total,
      counts.mini,
      counts.nano,
      counts.routerCorrect,
      pct(counts.mini, counts.total),
      pct(counts.routerCorrect, counts.total),
    ]);
  }

  gepaRows.push([
    size,
    gepaData.selected_candidate_index,
    gepaData.gepa_default_best_index,
    (gepaData.gepa_candidates || []).length,
    gepaData.initial_summary?.mean_score ?? null,
    gepaData.initial_summary?.accuracy ?? null,
    gepaData.initial_summary?.mini_routes ?? null,
    gepaData.initial_summary?.nano_routes ?? null,
    gepaData.initial_summary?.failure_type_counts?.critical_misroute ?? null,
    gepaData.best_summary?.mean_score ?? null,
    gepaData.best_summary?.accuracy ?? null,
    gepaData.best_summary?.mini_routes ?? null,
    gepaData.best_summary?.nano_routes ?? null,
    gepaData.best_summary?.failure_type_counts?.critical_misroute ?? null,
  ]);
}

await fs.mkdir(outputDir, { recursive: true });

const workbook = Workbook.create();

const summary = workbook.worksheets.add("Summary");
summary.showGridLines = false;
summary.getRange("A1:U1").values = [[
  "Train+Val",
  "Test Questions",
  "Router Accuracy",
  "Correct",
  "All Mini Accuracy",
  "All Nano Accuracy",
  "Oracle Accuracy",
  "Accuracy Delta vs All Mini",
  "Mini Routes",
  "Nano Routes",
  "Cost USD",
  "Cost / 1K Questions",
  "Cost Savings vs All Mini",
  "Critical Routed Mini",
  "Critical Total",
  "Critical Mini Rate",
  "Both Wrong Routed Mini",
  "Both Wrong Total",
  "Both Wrong Mini Rate",
  "Easy Router Direct",
  "Easy Router Defer",
]];
summary.getRange(`A2:U${rows.length + 1}`).values = rows;
summary.tables.add(`A1:U${rows.length + 1}`, true, "SummaryTable");
summary.freezePanes.freezeRows(1);

summary.getRange("A1:U1").format = {
  fill: "#1F4E78",
  font: { bold: true, color: "#FFFFFF" },
  wrapText: true,
};
summary.getRange("A:U").format.font = { name: "Aptos", size: 10 };
summary.getRange("C:C").format.numberFormat = "0.00%";
summary.getRange("E:H").format.numberFormat = "0.00%";
summary.getRange("K:K").format.numberFormat = "$0.0000";
summary.getRange("L:L").format.numberFormat = "$0.000";
summary.getRange("M:M").format.numberFormat = "0.00%";
summary.getRange("P:P").format.numberFormat = "0.00%";
summary.getRange("S:S").format.numberFormat = "0.00%";
summary.getRange("A:U").format.autofitColumns();
summary.getRange("A1:U1").format.rowHeightPx = 44;

const bucket = workbook.worksheets.add("Bucket Breakdown");
bucket.showGridLines = false;
bucket.getRange("A1:H1").values = [[
  "Train+Val",
  "Bucket",
  "Total",
  "Mini Routes",
  "Nano Routes",
  "Router Correct",
  "Mini Route Rate",
  "Router Accuracy in Bucket",
]];
bucket.getRange(`A2:H${bucketRows.length + 1}`).values = bucketRows;
bucket.tables.add(`A1:H${bucketRows.length + 1}`, true, "BucketTable");
bucket.freezePanes.freezeRows(1);
bucket.getRange("A1:H1").format = {
  fill: "#548235",
  font: { bold: true, color: "#FFFFFF" },
  wrapText: true,
};
bucket.getRange("A:H").format.font = { name: "Aptos", size: 10 };
bucket.getRange("G:H").format.numberFormat = "0.00%";
bucket.getRange("A:H").format.autofitColumns();

const gepa = workbook.worksheets.add("GEPA Internal");
gepa.showGridLines = false;
gepa.getRange("A1:N1").values = [[
  "Train+Val",
  "Selected Candidate",
  "GEPA Default Best",
  "Candidates",
  "Initial Mean Score",
  "Initial Accuracy",
  "Initial Mini Routes",
  "Initial Nano Routes",
  "Initial Critical Misroutes",
  "Best Mean Score",
  "Best Accuracy",
  "Best Mini Routes",
  "Best Nano Routes",
  "Best Critical Misroutes",
]];
gepa.getRange(`A2:N${gepaRows.length + 1}`).values = gepaRows;
gepa.tables.add(`A1:N${gepaRows.length + 1}`, true, "GepaInternalTable");
gepa.freezePanes.freezeRows(1);
gepa.getRange("A1:N1").format = {
  fill: "#7030A0",
  font: { bold: true, color: "#FFFFFF" },
  wrapText: true,
};
gepa.getRange("A:N").format.font = { name: "Aptos", size: 10 };
gepa.getRange("E:F").format.numberFormat = "0.00";
gepa.getRange("F:F").format.numberFormat = "0.00%";
gepa.getRange("J:K").format.numberFormat = "0.00";
gepa.getRange("K:K").format.numberFormat = "0.00%";
gepa.getRange("A:N").format.autofitColumns();

const notes = workbook.worksheets.add("Notes");
notes.showGridLines = false;
notes.getRange("A1:B8").values = [
  ["Item", "Value"],
  ["Experiment", "Half1 train/val sweep, half2 fixed test"],
  ["Router", "Hybrid easy Python router + qwen3.5:latest"],
  ["Reflection", "OpenAI gpt-4.1"],
  ["Score mode", "risk_averse_utility"],
  ["Output format", "token"],
  ["Best test accuracy", "train+val=50, 80.00%"],
  ["Best cost saving", "train+val=150/200, 62.26%, but accuracy drops to 72.20%"],
];
notes.tables.add("A1:B8", true, "NotesTable");
notes.getRange("A1:B1").format = {
  fill: "#666666",
  font: { bold: true, color: "#FFFFFF" },
};
notes.getRange("A:B").format.font = { name: "Aptos", size: 10 };
notes.getRange("A:B").format.autofitColumns();

const chartData = workbook.worksheets.add("Chart Data");
chartData.showGridLines = false;
chartData.getRange("A1:C1").values = [["Train+Val", "Router Accuracy", "Cost Savings"]];
chartData.getRange("A2:C5").formulas = [
  ["=Summary!A2", "=Summary!C2", "=Summary!M2"],
  ["=Summary!A3", "=Summary!C3", "=Summary!M3"],
  ["=Summary!A4", "=Summary!C4", "=Summary!M4"],
  ["=Summary!A5", "=Summary!C5", "=Summary!M5"],
];
const chart = chartData.charts.add("line", chartData.getRange("A1:C5"));
chart.title = "Router Accuracy and Cost Savings";
chart.hasLegend = true;
chart.xAxis = { axisType: "textAxis" };
chart.yAxis = { numberFormatCode: "0%" };
chart.setPosition("E1", "L18");
chartData.getRange("A:C").format.autofitColumns();

await workbook.inspect({
  kind: "table",
  range: "Summary!A1:U5",
  include: "values,formulas",
  tableMaxRows: 8,
  tableMaxCols: 21,
});

await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 50 },
});

await workbook.render({ sheetName: "Summary", autoCrop: "all", scale: 1, format: "png" });
await workbook.render({ sheetName: "Bucket Breakdown", autoCrop: "all", scale: 1, format: "png" });
await workbook.render({ sheetName: "GEPA Internal", autoCrop: "all", scale: 1, format: "png" });

const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(outputPath);
console.log(outputPath);
