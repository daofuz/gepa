import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = process.cwd();
const outputDir = path.join(root, "outputs", "router_sweep");
const outputPath = path.join(outputDir, "half1_forced_vs_random_control_results.xlsx");
const sizes = [50, 100, 150, 200];

const experiments = [
  {
    label: "forced_critical",
    display: "Forced critical",
    resultPrefix: "routing_cost_comparison_testdata_hybrid_half1_trainval",
    splitPrefix: "routing_split_half1_sweep_trainval",
    splitSuffix: "_withcritical_seed0.json",
  },
  {
    label: "random_control",
    display: "Random control",
    resultPrefix: "routing_cost_comparison_testdata_hybrid_half1_random_control_trainval",
    splitPrefix: "routing_split_half1_random_control_trainval",
    splitSuffix: "_seed0.json",
  },
];

function scenario(comparison, name) {
  const row = comparison.find((item) => item.scenario === name);
  if (!row) throw new Error(`Missing scenario ${name}`);
  return row;
}

function bucketName(trace) {
  if (trace.mini_correct && trace.nano_correct) return "both_correct";
  if (trace.mini_correct && !trace.nano_correct) return "critical_mini_only";
  if (!trace.mini_correct && trace.nano_correct) return "nano_only";
  return "both_wrong";
}

function safeRate(num, den) {
  return den ? num / den : 0;
}

async function readJson(file) {
  return JSON.parse(await fs.readFile(path.join(root, file), "utf8"));
}

const summaryRows = [];
const bucketRows = [];

for (const exp of experiments) {
  for (const size of sizes) {
    const resultFile = `${exp.resultPrefix}${size}_openai.json`;
    const splitFile = `${exp.splitPrefix}${size}${exp.splitSuffix}`;
    const result = await readJson(resultFile);
    const split = await readJson(splitFile);

    const best = scenario(result.comparison, "gepa_best_on_testdata");
    const allMini = scenario(result.comparison, "all_mini_on_testdata");
    const allNano = scenario(result.comparison, "all_nano_on_testdata");
    const oracle = scenario(result.comparison, "oracle_on_testdata");
    const traces = result.router_traces || [];

    const buckets = new Map([
      ["both_correct", { total: 0, mini: 0, nano: 0, correct: 0 }],
      ["critical_mini_only", { total: 0, mini: 0, nano: 0, correct: 0 }],
      ["nano_only", { total: 0, mini: 0, nano: 0, correct: 0 }],
      ["both_wrong", { total: 0, mini: 0, nano: 0, correct: 0 }],
    ]);
    const easy = { direct: 0, defer: 0 };

    for (const trace of traces) {
      const route = trace.selected_route || trace.route;
      const bucket = buckets.get(bucketName(trace));
      bucket.total += 1;
      if (route === "mini") bucket.mini += 1;
      if (route === "nano") bucket.nano += 1;
      if ((route === "mini" && trace.mini_correct) || (route === "nano" && trace.nano_correct)) {
        bucket.correct += 1;
      }
      if (trace.easy_router_source === "easy_router") easy.direct += 1;
      if (trace.easy_router_source === "easy_router_defer") easy.defer += 1;
    }

    const trainValCounts = split.bucket_counts?.train_val || {};
    const critical = buckets.get("critical_mini_only");
    const nanoOnly = buckets.get("nano_only");
    const bothWrong = buckets.get("both_wrong");
    const bothCorrect = buckets.get("both_correct");

    summaryRows.push([
      exp.display,
      size,
      split.train_count,
      split.val_count,
      split.test_count,
      trainValCounts.critical || 0,
      trainValCounts.nano_only || 0,
      trainValCounts.both_correct || 0,
      trainValCounts.both_wrong || 0,
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
      safeRate(critical.mini, critical.total),
      nanoOnly.nano,
      nanoOnly.total,
      safeRate(nanoOnly.nano, nanoOnly.total),
      bothWrong.mini,
      bothWrong.total,
      safeRate(bothWrong.mini, bothWrong.total),
      bothCorrect.nano,
      bothCorrect.total,
      safeRate(bothCorrect.nano, bothCorrect.total),
      easy.direct,
      easy.defer,
    ]);

    for (const [bucketLabel, counts] of buckets.entries()) {
      bucketRows.push([
        exp.display,
        size,
        bucketLabel,
        counts.total,
        counts.mini,
        counts.nano,
        counts.correct,
        safeRate(counts.mini, counts.total),
        safeRate(counts.nano, counts.total),
        safeRate(counts.correct, counts.total),
      ]);
    }
  }
}

await fs.mkdir(outputDir, { recursive: true });

const workbook = Workbook.create();

const summary = workbook.worksheets.add("Unified Summary");
summary.showGridLines = false;
const summaryHeaders = [
  "Experiment",
  "Train+Val",
  "Train Count",
  "Val Count",
  "Test Count",
  "Train+Val Critical",
  "Train+Val Nano Only",
  "Train+Val Both Correct",
  "Train+Val Both Wrong",
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
  "Nano Only Routed Nano",
  "Nano Only Total",
  "Nano Only Nano Rate",
  "Both Wrong Routed Mini",
  "Both Wrong Total",
  "Both Wrong Mini Rate",
  "Both Correct Routed Nano",
  "Both Correct Total",
  "Both Correct Nano Rate",
  "Easy Router Direct",
  "Easy Router Defer",
];
summary.getRange("A1:AH1").values = [summaryHeaders];
summary.getRange(`A2:AH${summaryRows.length + 1}`).values = summaryRows;
summary.tables.add(`A1:AH${summaryRows.length + 1}`, true, "UnifiedSummaryTable");
summary.freezePanes.freezeRows(1);
summary.getRange("A1:AH1").format = {
  fill: "#1F4E78",
  font: { bold: true, color: "#FFFFFF" },
  wrapText: true,
};
summary.getRange("A:AH").format.font = { name: "Aptos", size: 10 };
summary.getRange("J:J").format.numberFormat = "0.00%";
summary.getRange("L:O").format.numberFormat = "0.00%";
summary.getRange("R:R").format.numberFormat = "$0.0000";
summary.getRange("S:S").format.numberFormat = "$0.000";
summary.getRange("T:T").format.numberFormat = "0.00%";
summary.getRange("W:W").format.numberFormat = "0.00%";
summary.getRange("Z:Z").format.numberFormat = "0.00%";
summary.getRange("AC:AC").format.numberFormat = "0.00%";
summary.getRange("AF:AF").format.numberFormat = "0.00%";
summary.getRange("A:AH").format.autofitColumns();
summary.getRange("A1:AH1").format.rowHeightPx = 48;

const bucket = workbook.worksheets.add("Unified Buckets");
bucket.showGridLines = false;
bucket.getRange("A1:J1").values = [[
  "Experiment",
  "Train+Val",
  "Bucket",
  "Total",
  "Mini Routes",
  "Nano Routes",
  "Router Correct",
  "Mini Route Rate",
  "Nano Route Rate",
  "Router Accuracy in Bucket",
]];
bucket.getRange(`A2:J${bucketRows.length + 1}`).values = bucketRows;
bucket.tables.add(`A1:J${bucketRows.length + 1}`, true, "UnifiedBucketTable");
bucket.freezePanes.freezeRows(1);
bucket.getRange("A1:J1").format = {
  fill: "#548235",
  font: { bold: true, color: "#FFFFFF" },
  wrapText: true,
};
bucket.getRange("A:J").format.font = { name: "Aptos", size: 10 };
bucket.getRange("H:J").format.numberFormat = "0.00%";
bucket.getRange("A:J").format.autofitColumns();

const compact = workbook.worksheets.add("Compact View");
compact.showGridLines = false;
compact.getRange("A1:I1").values = [[
  "Experiment",
  "Train+Val",
  "Router Accuracy",
  "Correct",
  "Mini Routes",
  "Nano Routes",
  "Cost Savings",
  "Critical Routed Mini",
  "Critical Total",
]];
const compactRows = summaryRows.map((row) => [
  row[0],
  row[1],
  row[9],
  row[10],
  row[15],
  row[16],
  row[19],
  row[20],
  row[21],
]);
compact.getRange(`A2:I${compactRows.length + 1}`).values = compactRows;
compact.tables.add(`A1:I${compactRows.length + 1}`, true, "CompactTable");
compact.freezePanes.freezeRows(1);
compact.getRange("A1:I1").format = {
  fill: "#7030A0",
  font: { bold: true, color: "#FFFFFF" },
  wrapText: true,
};
compact.getRange("A:I").format.font = { name: "Aptos", size: 10 };
compact.getRange("C:C").format.numberFormat = "0.00%";
compact.getRange("G:G").format.numberFormat = "0.00%";
compact.getRange("A:I").format.autofitColumns();

const notes = workbook.worksheets.add("Notes");
notes.showGridLines = false;
notes.getRange("A1:B7").values = [
  ["Item", "Value"],
  ["Test set", "Fixed second half, 205 questions"],
  ["Experiments", "Forced critical vs random control"],
  ["Router", "Hybrid easy Python router + qwen3.5:latest"],
  ["Reflection", "OpenAI gpt-4.1"],
  ["Score mode", "risk_averse_utility"],
  ["Purpose", "Both result tables now use the same columns and metrics"],
];
notes.tables.add("A1:B7", true, "NotesTable");
notes.getRange("A1:B1").format = {
  fill: "#666666",
  font: { bold: true, color: "#FFFFFF" },
};
notes.getRange("A:B").format.font = { name: "Aptos", size: 10 };
notes.getRange("A:B").format.autofitColumns();

await workbook.inspect({
  kind: "table",
  range: "Unified Summary!A1:AH9",
  include: "values,formulas",
  tableMaxRows: 10,
  tableMaxCols: 34,
});
await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 50 },
});
await workbook.render({ sheetName: "Unified Summary", autoCrop: "all", scale: 1, format: "png" });
await workbook.render({ sheetName: "Unified Buckets", autoCrop: "all", scale: 1, format: "png" });
await workbook.render({ sheetName: "Compact View", autoCrop: "all", scale: 1, format: "png" });

const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(outputPath);
console.log(outputPath);
