import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const OUT = "C:/Users/25875/gepa/outputs/half1_205_softprompt_summary_en_v10.pptx";
const QA = "C:/Users/25875/gepa/outputs/half1_205_softprompt_summary_en_v10_render";
const W = 1280;
const H = 720;
const C = {
  ink: "#000000",
  body: "#222222",
  muted: "#555555",
  rule: "#B8BCC4",
  panel: "#EDEDED",
  pale: "#F6F6F6",
  mini: "#000000",
  nano: "#2F80ED",
  both: "#B8BCC4",
  wrong: "#FF6B35",
  canvas: "#FFFFFF",
};

const deck = Presentation.create({ slideSize: { width: W, height: H } });

const sweep = [
  { size: 25, train: 20, val: 5, mini_only: 19, nano_only: 0, both_correct: 6, both_wrong: 0, acc: 78.5, savings: 18.6, routes: "64/141" },
  { size: 50, train: 40, val: 10, mini_only: 19, nano_only: 1, both_correct: 20, both_wrong: 10, acc: 80.0, savings: 12.6, routes: "45/160" },
  { size: 75, train: 60, val: 15, mini_only: 19, nano_only: 1, both_correct: 43, both_wrong: 12, acc: 77.6, savings: 31.2, routes: "109/96" },
  { size: 100, train: 80, val: 20, mini_only: 19, nano_only: 2, both_correct: 62, both_wrong: 17, acc: 76.1, savings: 43.4, routes: "146/59" },
  { size: 125, train: 100, val: 25, mini_only: 19, nano_only: 3, both_correct: 80, both_wrong: 23, acc: 72.2, savings: 67.5, routes: "191/14" },
  { size: 150, train: 120, val: 30, mini_only: 19, nano_only: 6, both_correct: 101, both_wrong: 24, acc: 72.2, savings: 62.3, routes: "193/12" },
  { size: 200, train: 160, val: 40, mini_only: 19, nano_only: 9, both_correct: 142, both_wrong: 30, acc: 72.2, savings: 62.3, routes: "193/12" },
];

const random = [
  { size: 50, mini_only: 6, nano_only: 2, both_correct: 37, both_wrong: 5, acc: 73.7, savings: 49.5, routes: "162/43" },
  { size: 100, mini_only: 10, nano_only: 4, both_correct: 72, both_wrong: 14, acc: 75.1, savings: 43.9, routes: "144/61" },
  { size: 150, mini_only: 14, nano_only: 6, both_correct: 106, both_wrong: 24, acc: 72.2, savings: 61.9, routes: "192/13" },
  { size: 200, mini_only: 19, nano_only: 9, both_correct: 142, both_wrong: 30, acc: 72.2, savings: 61.9, routes: "192/13" },
];

const ablations = [
  ["balancedish 50", "13/9/14/14", "26/18/28/28%", "79.0%", "24.3%", "82/123"],
  ["critical + both-correct", "19/0/31/0", "38/0/62/0%", "76.1%", "34.7%", "122/83"],
  ["critical + both-wrong", "19/0/0/31", "38/0/0/62%", "78.0%", "24.4%", "85/120"],
  ["both-correct heavy", "5/0/40/5", "10/0/80/10%", "72.2%", "59.9%", "185/20"],
  ["both-wrong heavy", "5/0/14/31", "10/0/28/62%", "74.6%", "47.3%", "157/48"],
];

const softprompt = [
  ["half1 trainval50 smoke", "40/10", "205", "80.0%", "5.3%", "14/191", "1"],
  ["balanced50 risk EU", "38/12", "205", "81.0%", "40.6%", "95/110", "6"],
  ["first205 random50 risk EU", "50/0", "205", "71.2%", "63.4%", "169/36", "24"],
];

function addSlide(title, kicker = "") {
  const s = deck.slides.add();
  s.background.fill = C.canvas;
  if (kicker) tx(s, kicker, 42, 34, 720, 24, { size: 15, color: C.muted, bold: true });
  tx(s, title, 42, 62, 1060, 78, { size: 39, bold: true, color: C.ink });
  box(s, 42, 146, 1196, 1, C.rule);
  return s;
}

function tx(s, value, x, y, w, h, o = {}) {
  const t = s.shapes.add({
    geometry: "textbox",
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  t.text = String(value);
  t.text.style = {
    fontFace: "Aptos",
    fontSize: o.size ?? 18,
    bold: o.bold ?? false,
    color: o.color ?? C.body,
    alignment: o.align ?? "left",
  };
  return t;
}

function box(s, x, y, w, h, fill, border = "none") {
  return s.shapes.add({
    geometry: "rect",
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: { style: "solid", fill: border, width: border === "none" ? 0 : 1 },
  });
}

function metric(s, x, y, w, label, value, note = "", color = C.ink) {
  box(s, x, y, w, 138, C.panel);
  tx(s, label, x + 18, y + 16, w - 36, 24, { size: 15, color: C.muted, bold: true });
  tx(s, value, x + 18, y + 46, w - 36, 50, { size: 39, color, bold: true });
  tx(s, note, x + 18, y + 102, w - 36, 26, { size: 16, color: C.body });
}

function footer(s, source) {
  tx(s, source, 42, 662, 1060, 24, { size: 12, color: C.muted });
  tx(s, String(deck.slides.items.length).padStart(2, "0"), 1186, 662, 52, 24, { size: 14, color: C.muted, align: "right" });
}

function table(s, x, y, cols, rows, opt = {}) {
  const headerH = opt.headerH ?? 42;
  const rowH = opt.rowH ?? 42;
  const totalW = cols.reduce((a, c) => a + c.w, 0);
  box(s, x, y, totalW, headerH, C.panel);
  let cx = x;
  cols.forEach((c) => {
    tx(s, c.label, cx + 8, y + 11, c.w - 16, 22, { size: opt.headerSize ?? 15, bold: true, align: c.align ?? "left" });
    cx += c.w;
  });
  rows.forEach((r, i) => {
    const yy = y + headerH + i * rowH;
    box(s, x, yy, totalW, 1, C.rule);
    cx = x;
    cols.forEach((c, ci) => {
      const val = Array.isArray(r) ? r[ci] : r[c.key];
      const hi = opt.highlightRows?.includes(i);
      tx(s, val, cx + 8, yy + 10, c.w - 16, rowH - 12, {
        size: opt.bodySize ?? 15,
        bold: hi,
        color: hi ? C.wrong : C.body,
        align: c.align ?? "left",
      });
      cx += c.w;
    });
  });
}

function stackedBar(s, x, y, w, h, row, labels = true) {
  const total = row.mini_only + row.nano_only + row.both_correct + row.both_wrong;
  const parts = [
    ["mini_only", C.mini],
    ["nano_only", C.nano],
    ["both_correct", C.both],
    ["both_wrong", C.wrong],
  ];
  let cx = x;
  parts.forEach(([k, color]) => {
    const ww = w * row[k] / total;
    if (ww > 0) box(s, cx, y, ww, h, color);
    if (labels && ww > 34) {
      tx(s, `${Math.round(row[k] / total * 100)}%`, cx + 3, y + 5, ww - 6, h - 8, {
        size: 11,
        color: color === C.both ? C.ink : "#FFFFFF",
        align: "center",
        bold: true,
      });
    }
    cx += ww;
  });
}

function legend(s, x, y) {
  const items = [
    [C.mini, "mini-only"],
    [C.nano, "nano-only"],
    [C.both, "both correct"],
    [C.wrong, "both wrong"],
  ];
  let cx = x;
  items.forEach(([color, label]) => {
    box(s, cx, y + 7, 18, 10, color);
    tx(s, label, cx + 24, y, 120, 24, { size: 13, color: C.body });
    cx += 142;
  });
}

function smallBars(s, x, y, rows, opt = {}) {
  const labelW = opt.labelW ?? 70;
  const plotW = opt.plotW ?? 300;
  rows.forEach((r, i) => {
    const yy = y + i * 42;
    tx(s, r.label, x, yy - 4, labelW, 24, { size: 14, color: C.body });
    box(s, x + labelW, yy, plotW, 9, C.pale);
    box(s, x + labelW, yy, plotW * r.acc / 90, 9, C.ink);
    box(s, x + labelW, yy + 16, plotW, 9, C.pale);
    box(s, x + labelW, yy + 16, plotW * r.savings / 80, 9, C.wrong);
    tx(s, `${r.acc.toFixed(1)}%`, x + labelW + plotW + 10, yy - 5, 55, 20, { size: 12, color: C.ink });
    tx(s, `${r.savings.toFixed(1)}%`, x + labelW + plotW + 10, yy + 10, 55, 20, { size: 12, color: C.wrong });
  });
}

{
  const s = deck.slides.add();
  s.background.fill = C.canvas;
  tx(s, "First-half 205 experiments", 42, 66, 930, 74, { size: 56, bold: true });
  tx(s, "Model routing on computer-science multiple-choice questions", 42, 154, 1000, 42, { size: 26, color: C.body });
  box(s, 42, 242, 1196, 1, C.ink);
  metric(s, 42, 304, 260, "Task", "routing", "select one model per question");
  metric(s, 326, 304, 260, "Dataset", "410 MCQs", "computer-science questions");
  metric(s, 610, 304, 260, "GPT-4o mini", "stronger", "higher accuracy, higher cost");
  metric(s, 894, 304, 260, "GPT-4o nano", "cheaper", "lower cost, lower accuracy", C.wrong);
  tx(s, "Goal: preserve answer accuracy while reducing cost. Train and validate on the first 205 questions, then evaluate every router on the fixed second-half 205-question test set.", 42, 518, 1160, 66, { size: 22, bold: true });
  footer(s, "Sources: routing_split_half1_*.json, gepa_router_prompt_result_hybrid_half1_*.json, routing_cost_comparison_testdata_hybrid_half1_*.csv, softprompt_router_*.json");
}

{
  const s = addSlide("The fixed test set offers substantial cost savings through routing", "205/205 framing");
  metric(s, 42, 184, 250, "test mini-only", "26", "12.7%; must escalate", C.wrong);
  metric(s, 316, 184, 250, "test both-correct", "139", "67.8%; can be cheap");
  metric(s, 590, 184, 250, "test both-wrong", "33", "16.1%; no route fixes it");
  metric(s, 864, 184, 250, "oracle", "83.9%", "179 nano / 26 mini", C.wrong);
  table(s, 42, 388, [
    { label: "Policy", w: 220 },
    { label: "Accuracy", w: 130, align: "right" },
    { label: "Cost / 1k", w: 140, align: "right" },
    { label: "Savings", w: 120, align: "right" },
    { label: "nano/mini", w: 140, align: "right" },
    { label: "Meaning", w: 380 },
  ], [
    ["all-mini", "80.5%", "$1.124", "0%", "0/205", "Accuracy baseline."],
    ["all-nano", "71.2%", "$0.291", "74.1%", "205/0", "Cheap, but loses 9.3 points."],
    ["oracle", "83.9%", "$0.402", "64.2%", "179/26", "Large theoretical room because most items are both-correct."],
  ], { rowH: 54, bodySize: 16, highlightRows: [2] });
  footer(s, "Fixed test outcomes: mini-only 26, nano-only 7, both-correct 139, both-wrong 33");
}

{
  const s = addSlide("Critical cases carry the routing signal", "Why active sampling");
  tx(s, "A critical case is one where the models disagree. Agreement cases reveal little about which route to choose; disagreement cases expose the decision boundary.", 42, 172, 1170, 48, { size: 20, bold: true });

  box(s, 42, 232, 552, 108, C.panel);
  tx(s, "MINI-ONLY", 60, 246, 170, 22, { size: 16, color: C.wrong, bold: true });
  tx(s, "mini correct / nano wrong", 60, 274, 490, 30, { size: 25, bold: true });
  tx(s, "Escalation is necessary to preserve accuracy.", 60, 310, 490, 22, { size: 16, color: C.body });

  box(s, 616, 232, 552, 108, C.pale, C.rule);
  tx(s, "NANO-ONLY", 634, 246, 170, 22, { size: 16, color: C.nano, bold: true });
  tx(s, "nano correct / mini wrong", 634, 274, 490, 30, { size: 25, bold: true });
  tx(s, "The cheaper route is also the better route.", 634, 310, 490, 22, { size: 16, color: C.body });

  table(s, 42, 370, [
    { label: "Size-50 sampling", w: 250 },
    { label: "mini-only", w: 120, align: "right" },
    { label: "nano-only", w: 120, align: "right" },
    { label: "critical share", w: 170, align: "right" },
    { label: "both-correct", w: 190, align: "right" },
    { label: "Test accuracy", w: 160, align: "right" },
    { label: "Savings", w: 186, align: "right" },
  ], [
    ["actively selected", "19", "1", "40%", "40%", "80.0%", "12.6%"],
    ["random sample", "6", "2", "16%", "74%", "73.7%", "49.5%"],
  ], { rowH: 54, bodySize: 16, headerSize: 15, highlightRows: [0] });

  tx(s, "Random sampling mostly selected easy both-correct cases, pushing the router toward nano and losing 6.3 accuracy points. We therefore oversampled critical cases deliberately.", 42, 558, 1160, 56, { size: 20, bold: true });
  footer(s, "Critical = mini-only + nano-only; sources: half1 trainval50 with-critical split and random-control trainval50 split");
}

{
  const s = addSlide("Larger samples were not automatically better", "Prompt-router size sweep");
  table(s, 42, 184, [
    { label: "Size", w: 80, align: "right" },
    { label: "M/N/B/W mix", w: 170 },
    { label: "Accuracy", w: 110, align: "right" },
    { label: "Savings", w: 110, align: "right" },
    { label: "test nano/mini", w: 130, align: "right" },
    { label: "Readout", w: 596 },
  ], sweep.map((r) => [
    r.size,
    `${Math.round(r.mini_only / r.size * 100)}/${Math.round(r.nano_only / r.size * 100)}/${Math.round(r.both_correct / r.size * 100)}/${Math.round(r.both_wrong / r.size * 100)}%`,
    `${r.acc.toFixed(1)}%`,
    `${r.savings.toFixed(1)}%`,
    r.routes,
    r.size === 50 ? "Best accuracy-preserving point: 80.0%, but only 12.6% savings." : (r.size >= 125 ? "The router became nano-heavy as both-correct dominated." : "Intermediate trade-off.")
  ]), { rowH: 46, bodySize: 16, highlightRows: [1] });
  tx(s, "Size 50 best preserved accuracy. At larger sizes, the falling critical share and rising both-correct share pushed the router toward nano-heavy policies.", 42, 574, 1160, 50, { size: 20, bold: true });
  footer(s, "Fixed test set: second-half 205; source: routing_cost_comparison_testdata_hybrid_half1_trainval*_openai.csv");
}

{
  const s = addSlide("Random controls showed that natural sampling pushed the router toward nano", "Random-control comparison");
  legend(s, 42, 172);
  random.forEach((r, i) => {
    const yy = 224 + i * 68;
    tx(s, `random ${r.size}`, 42, yy - 2, 100, 24, { size: 16, bold: true });
    stackedBar(s, 160, yy, 530, 28, r, true);
    tx(s, `${r.mini_only}/${r.nano_only}/${r.both_correct}/${r.both_wrong}`, 710, yy + 3, 170, 22, { size: 14 });
    tx(s, `acc ${r.acc.toFixed(1)}%`, 902, yy + 3, 90, 22, { size: 14, bold: true });
    tx(s, `save ${r.savings.toFixed(1)}%`, 1006, yy + 3, 110, 22, { size: 14, color: C.wrong, bold: true });
  });
  tx(s, "Random50 had only 12% mini-only and 74% both-correct. That is why the random-control router saved more money but lost more accuracy than the critical-heavy size-50 split.", 42, 538, 1120, 58, { size: 22, bold: true });
  footer(s, "Sources: routing_split_half1_random_control_trainval*.json and routing_cost_comparison_testdata_hybrid_half1_random_control_trainval*.csv");
}

{
  const s = addSlide("The size-50 ablations changed risk posture, not just sample count", "Bucket ablations");
  table(s, 42, 184, [
    { label: "Training mix", w: 245 },
    { label: "M/N/B/W count", w: 150 },
    { label: "M/N/B/W share", w: 145 },
    { label: "Accuracy", w: 90, align: "right" },
    { label: "Savings", w: 90, align: "right" },
    { label: "test nano/mini", w: 125, align: "right" },
    { label: "Takeaway", w: 310 },
  ], ablations.map((r) => [
    r[0], r[1], r[2], r[3], r[4], r[5],
    r[0] === "balancedish 50" ? "Most balanced: 79.0% / 24.3%." :
    r[0] === "both-correct heavy" ? "Too many cheap-safe examples pushed nano." :
    r[0] === "critical + both-wrong" ? "More conservative, but limited savings." :
    "Shifted risk posture."
  ]), { rowH: 58, bodySize: 14, headerSize: 15, highlightRows: [0] });
  tx(s, "M/N/B/W = mini-only / nano-only / both-correct / both-wrong. The important lever was bucket mix, not merely optimizer choice.", 42, 580, 1120, 44, { size: 21, bold: true });
  footer(s, "Sources: gepa_router_prompt_result_hybrid_half1_ablation50_*.json and corresponding testdata CSV files");
}

{
  const s = addSlide("The soft-prompt router turned routing into a small trainable classifier", "Soft-prompt workflow");
  const y = 210;
  const steps = [
    ["1", "Build routing examples", "Each question is paired with mini/nano correctness and cost, then bucketed as mini-only, nano-only, both-correct, or both-wrong."],
    ["2", "Format the input text", "The encoder sees category, source, question, and answer options. These runs did not include the nano answer text."],
    ["3", "Prepend soft tokens", "Sixteen learned embedding vectors are concatenated before the token embeddings; they are not readable words."],
    ["4", "Classify the route", "The frozen encoder produces a representation, and a small linear head predicts nano vs mini."],
    ["5", "Apply a threshold", "The mini probability is compared with a decision threshold; higher threshold routes more examples to nano."],
  ];
  steps.forEach((r, i) => {
    const xx = 42 + i * 232;
    box(s, xx, y, 192, 280, i === 2 ? "#F2F2F2" : C.panel);
    tx(s, r[0], xx + 18, y + 18, 50, 40, { size: 34, bold: true, color: i === 2 ? C.wrong : C.ink });
    tx(s, r[1], xx + 18, y + 72, 156, 48, { size: 20, bold: true });
    tx(s, r[2], xx + 18, y + 132, 156, 126, { size: 15, color: C.body });
  });
  tx(s, "The important shift from prompt GEPA: the policy is no longer a single text instruction. It becomes a learned probability score that can be calibrated after training.", 42, 548, 1120, 54, { size: 22, bold: true });
  footer(s, "Implementation: train_softprompt_router.py; SoftPromptRouter.forward prepends learned embeddings and thresholds the mini probability");
}

{
  const s = addSlide("Backbone and training setup were intentionally small", "Soft-prompt architecture");
  metric(s, 42, 184, 260, "Backbone", "DistilBERT", "distilbert-base-uncased", C.wrong);
  metric(s, 326, 184, 260, "Backbone update", "frozen", "train_backbone = false");
  metric(s, 610, 184, 260, "Soft tokens", "16", "learned prompt embeddings");
  metric(s, 894, 184, 260, "Trainable params", "13.8k", "out of 66.4M total", C.wrong);
  table(s, 42, 378, [
    { label: "Component", w: 230 },
    { label: "What it did", w: 420 },
    { label: "Key setting in best run", w: 480 },
  ], [
    ["Input encoder", "Tokenized question/options into a transformer sequence.", "max_length 384; no nano-answer text included."],
    ["Soft prompt", "Added task-specific learned vectors before the real tokens.", "16 vectors of DistilBERT hidden size."],
    ["Classifier head", "Mapped the encoder state to nano/mini logits.", "Linear layer over the first real token state after soft tokens."],
    ["Training objective", "Optimized routing utility rather than just accuracy.", "expected_utility loss; target mini rate 0.25; mini penalty 0.4."],
    ["Decision rule", "Converted mini probability into a discrete route.", "threshold 0.45 for the selected 205-test run."],
  ], { rowH: 48, bodySize: 16, headerSize: 16, highlightRows: [3] });
  footer(s, "Best-run args: hf_model distilbert-base-uncased, epochs 8, batch 8, learning_rate 0.01, train_backbone false");
}
{
  const s = addSlide("Soft prompt was added as a trainable router after the prompt sweeps", "Soft-prompt follow-up");
  metric(s, 42, 184, 260, "Model form", "frozen encoder", "trainable soft tokens + classifier");
  metric(s, 326, 184, 260, "Initial split", "40 / 10", "same half1 trainval50 split");
  metric(s, 610, 184, 260, "Best heldout run", "81.0%", "205-test accuracy", C.wrong);
  metric(s, 894, 184, 260, "Best savings", "40.6%", "95 nano / 110 mini", C.wrong);
  table(s, 42, 386, [
    { label: "Run", w: 300 },
    { label: "train/val", w: 95, align: "right" },
    { label: "test", w: 70, align: "right" },
    { label: "Accuracy", w: 95, align: "right" },
    { label: "Savings", w: 95, align: "right" },
    { label: "nano/mini", w: 105, align: "right" },
    { label: "critical", w: 85, align: "right" },
    { label: "Interpretation", w: 330 },
  ], softprompt.map((r) => [
    r[0], r[1], r[2], r[3], r[4], r[5], r[6],
    r[0].includes("balanced50") ? "Best usable trade-off." : (r[0].includes("random50") ? "Cheap but unsafe." : "Accuracy-preserving, too conservative.")
  ]), { rowH: 58, bodySize: 14, headerSize: 14, highlightRows: [1] });
  footer(s, "Sources: softprompt_router_half1_trainval50_test_smoke.json, softprompt_router_balanced50_risk_best_expected_utility_test205.json, softprompt_router_first205_random50_seed0_risk_expected_utility.json");
}

{
  const s = addSlide("The soft-prompt threshold exposed a smoother risk-cost frontier", "Threshold sweep");
  const thresholdRows = [
    { label: "0.00", acc: 80.5, savings: 0.0 },
    { label: "0.25", acc: 81.0, savings: 9.5 },
    { label: "0.30", acc: 82.0, savings: 16.8 },
    { label: "0.35", acc: 82.0, savings: 24.6 },
    { label: "0.45", acc: 81.0, savings: 40.6 },
  ];
  smallBars(s, 42, 205, thresholdRows, { labelW: 60, plotW: 330 });
  table(s, 548, 190, [
    { label: "Threshold", w: 95, align: "right" },
    { label: "Accuracy", w: 100, align: "right" },
    { label: "Savings", w: 100, align: "right" },
    { label: "nano/mini", w: 120, align: "right" },
    { label: "Mean score", w: 110, align: "right" },
  ], [
    ["0.00", "80.5%", "0.0%", "0/205", "0.387"],
    ["0.25", "81.0%", "9.5%", "29/176", "0.460"],
    ["0.30", "82.0%", "16.8%", "46/159", "0.505"],
    ["0.35", "82.0%", "24.6%", "66/139", "0.551"],
    ["0.45", "81.0%", "40.6%", "95/110", "0.556"],
  ], { rowH: 52, bodySize: 16, highlightRows: [4] });
  tx(s, "Unlike a single prompt, the soft-prompt router can move along a threshold curve. The selected 0.45 point traded one accuracy point for a much larger cost reduction.", 42, 548, 1110, 54, { size: 22, bold: true });
  footer(s, "Threshold rows from softprompt_router_balanced50_risk_best_expected_utility_test205.json");
}

{
  const s = addSlide("What the combined story says", "Takeaways");
  table(s, 42, 184, [
    { label: "Finding", w: 340 },
    { label: "Evidence", w: 360 },
    { label: "Implication", w: 430 },
  ], [
    ["Data mix dominated the prompt-router behavior", "mini-only share fell from 76.0% to 9.5% across the size sweep.", "Do not treat larger train/val size as automatically better."],
    ["Size 50 remained the clean prompt-router anchor", "80.0% accuracy, 12.6% savings, 45/160 routes.", "It preserved accuracy because critical examples were over-represented."],
    ["Soft prompt improved the trade-off frontier", "81.0% accuracy, 40.6% savings, 95/110 routes.", "Trainable thresholding was more useful than another static prompt variant."],
    ["Random soft prompt was unsafe", "71.2% accuracy, 63.4% savings, 24 critical misroutes.", "The training mix still matters even with a learned router."],
  ], { rowH: 78, bodySize: 17, headerSize: 16, highlightRows: [2] });
  tx(s, "Next experiment: keep the fixed 205-test set, sweep bucket mix and soft-prompt threshold together, and report accuracy loss, cost savings, and critical misroutes as a single scorecard.", 42, 610, 1130, 38, { size: 21, bold: true });
  footer(s, "Scope: first-half 205 train/validation experiments plus later soft-prompt follow-up");
}

{
  const s = addSlide("Next: compare with routing baselines", "Future work");
  tx(s, "Simple controls", 42, 188, 520, 34, { size: 24, bold: true });
  ["Always-small / Always-large", "Oracle router", "Random / cost-ratio router"].forEach((name, i) => {
    const yy = 252 + i * 92;
    tx(s, name, 42, yy, 520, 42, { size: 27, bold: true });
    box(s, 42, yy + 54, 520, 1, C.rule);
  });

  tx(s, "Related methods", 656, 188, 540, 34, { size: 24, bold: true });
  ["FrugalGPT", "RouteLLM", "Hybrid LLM"].forEach((name, i) => {
    const yy = 252 + i * 92;
    tx(s, name, 656, yy, 540, 42, { size: 27, bold: true });
    box(s, 656, yy + 54, 540, 1, C.rule);
  });

  tx(s, "Compare the accuracy-cost curve on the same fixed second-half 205 test set.", 42, 576, 1160, 48, { size: 22, bold: true });
  footer(s, "Refs: FrugalGPT 2305.05176; RouteLLM 2406.18665; Hybrid LLM 2404.14618");
}

await fs.mkdir(QA, { recursive: true });
for (const [i, s] of deck.slides.items.entries()) {
  const stem = `slide-${String(i + 1).padStart(2, "0")}`;
  const png = await deck.export({ slide: s, format: "png", scale: 1 });
  await fs.writeFile(path.join(QA, `${stem}.png`), new Uint8Array(await png.arrayBuffer()));
  const layout = await s.export({ format: "layout" });
  await fs.writeFile(path.join(QA, `${stem}.layout.json`), await layout.text(), "utf8");
}
const pptx = await PresentationFile.exportPptx(deck);
await pptx.save(OUT);
console.log(OUT);














