import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const OUT = "C:/Users/25875/gepa/outputs/router_attempts_summary.pptx";
const QA = "C:/Users/25875/gepa/outputs/attempts_summary_tmp/qa";

const W = 1280;
const H = 720;
const C = {
  ink: "#000000",
  body: "#222222",
  muted: "#555555",
  rule: "#B8BCC4",
  panel: "#EDEDED",
  canvas: "#FFFFFF",
  accent: "#FF6B35",
  accent2: "#2F80ED",
};

const deck = Presentation.create({ slideSize: { width: W, height: H } });

function slide(title, kicker = "") {
  const s = deck.slides.add();
  s.background.fill = C.canvas;
  if (kicker) text(s, kicker, 42, 36, 520, 24, { size: 15, color: C.muted, bold: true });
  text(s, title, 42, 64, 980, 78, { size: 39, bold: true, color: C.ink });
  line(s, 42, 146, 1196);
  return s;
}

function text(s, value, x, y, w, h, opts = {}) {
  const t = s.shapes.add({
    geometry: "textbox",
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  t.text = value;
  t.text.style = {
    fontSize: opts.size ?? 20,
    bold: opts.bold ?? false,
    color: opts.color ?? C.body,
    fontFace: opts.font ?? "Microsoft YaHei",
    alignment: opts.align ?? "left",
  };
  return t;
}

function rect(s, x, y, w, h, fill = C.panel, border = "none") {
  return s.shapes.add({
    geometry: "rect",
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: { style: "solid", fill: border, width: border === "none" ? 0 : 1 },
  });
}

function line(s, x, y, w, color = C.rule) {
  return s.shapes.add({
    geometry: "rect",
    position: { left: x, top: y, width: w, height: 1 },
    fill: color,
    line: { style: "solid", fill: "none", width: 0 },
  });
}

function metric(s, x, y, w, label, value, note = "", color = C.ink) {
  rect(s, x, y, w, 146, C.panel);
  text(s, label, x + 20, y + 18, w - 40, 26, { size: 16, color: C.muted, bold: true });
  text(s, value, x + 20, y + 48, w - 40, 54, { size: 43, color, bold: true });
  if (note) text(s, note, x + 20, y + 106, w - 40, 30, { size: 17, color: C.body });
}

function bullet(s, x, y, w, lines) {
  lines.forEach((b, i) => {
    text(s, "•", x, y + i * 40, 18, 24, { size: 20, bold: true, color: C.accent });
    text(s, b, x + 28, y + i * 40, w - 28, 32, { size: 19, color: C.body });
  });
}

function table(s, x, y, columns, rows, opts = {}) {
  const rowH = opts.rowH ?? 42;
  const headerH = opts.headerH ?? 44;
  const widths = columns.map((c) => c.w);
  const totalW = widths.reduce((a, b) => a + b, 0);
  rect(s, x, y, totalW, headerH, C.panel);
  let cx = x;
  columns.forEach((c) => {
    text(s, c.label, cx + 10, y + 12, c.w - 20, 22, { size: opts.headerSize ?? 15, bold: true, color: C.ink, align: c.align ?? "left" });
    cx += c.w;
  });
  rows.forEach((r, ri) => {
    const yy = y + headerH + ri * rowH;
    line(s, x, yy, totalW);
    cx = x;
    columns.forEach((c, ci) => {
      const value = Array.isArray(r) ? r[ci] : r[c.key];
      const color = opts.highlightRows?.includes(ri) ? C.accent : C.body;
      text(s, String(value), cx + 10, yy + 11, c.w - 20, rowH - 12, { size: opts.bodySize ?? 15, color, bold: opts.highlightRows?.includes(ri) ?? false, align: c.align ?? "left" });
      cx += c.w;
    });
  });
}

function groupedBars(s, x, y, w, h, rows, opts = {}) {
  const maxAcc = opts.maxAcc ?? 90;
  const maxSavings = opts.maxSavings ?? 80;
  const gap = 14;
  const labelW = opts.labelW ?? 180;
  const plotW = w - labelW - 84;
  rows.forEach((r, i) => {
    const yy = y + i * gap * 2.75;
    text(s, r.label, x, yy - 2, labelW, 24, { size: 14, color: C.body });
    rect(s, x + labelW, yy, plotW, 10, "#F3F3F3");
    rect(s, x + labelW, yy, plotW * r.acc / maxAcc, 10, C.ink);
    rect(s, x + labelW, yy + 16, plotW, 10, "#F3F3F3");
    rect(s, x + labelW, yy + 16, plotW * r.savings / maxSavings, 10, C.accent);
    text(s, `${r.acc.toFixed(1)}%`, x + labelW + plotW + 10, yy - 5, 58, 20, { size: 13, color: C.ink });
    text(s, `${r.savings.toFixed(1)}%`, x + labelW + plotW + 10, yy + 11, 58, 20, { size: 13, color: C.accent });
  });
  rect(s, x + labelW, y + rows.length * gap * 2.75 + 8, 18, 8, C.ink);
  text(s, "准确率", x + labelW + 24, y + rows.length * gap * 2.75 + 1, 80, 20, { size: 13 });
  rect(s, x + labelW + 110, y + rows.length * gap * 2.75 + 8, 18, 8, C.accent);
  text(s, "成本节省", x + labelW + 134, y + rows.length * gap * 2.75 + 1, 90, 20, { size: 13 });
}

function footer(s, source) {
  text(s, source, 42, 665, 980, 22, { size: 12, color: C.muted });
  text(s, String(deck.slides.items.length).padStart(2, "0"), 1188, 662, 50, 22, { size: 14, color: C.muted, align: "right" });
}

{
  const s = deck.slides.add();
  s.background.fill = C.canvas;
  text(s, "路由实验复盘", 42, 62, 900, 76, { size: 62, bold: true, color: C.ink });
  text(s, "从 prompt router 到 token/risk-averse 与 soft prompt", 42, 154, 920, 44, { size: 26, color: C.body });
  line(s, 42, 242, 1196, C.ink);
  metric(s, 42, 300, 270, "任务", "MMLU-Pro CS", "mini / nano 二选一");
  metric(s, 342, 300, 270, "最大样本池", "410 题", "first half 205 + test 205");
  metric(s, 642, 300, 270, "最好准确率", "85.5%", "310 题；-1.0pp vs all-mini", C.accent);
  metric(s, 942, 300, 270, "最高节省", "71.7%", "310 题；-2.9pp vs all-mini", C.accent);
  text(s, "主结论：最好的方向不是单纯追求更便宜，而是在“避免 critical misroute”和“保留 nano 的成本优势”之间设定风险姿态。", 42, 510, 1100, 56, { size: 23, color: C.body, bold: true });
  footer(s, "来源：routing_cost_comparison_*.csv、routing_split_*.json、softprompt_router_*.json");
}

{
  const s = slide("实验口径先固定，否则指标会互相误导", "数据与基线");
  metric(s, 42, 184, 250, "模型", "mini vs nano", "mini 更准更贵，nano 更便宜");
  metric(s, 316, 184, 250, "成本价", "4x", "mini input/output 均为 nano 的 4 倍");
  metric(s, 590, 184, 250, "主 split", "205 / 205", "first half 做 train/val，second half 做 test");
  metric(s, 864, 184, 250, "候选目标", "oracle", "已知每题两模型对错后的上界");
  table(s, 42, 390, [
    { label: "测试口径", w: 250 },
    { label: "all-mini", w: 145, align: "right" },
    { label: "all-nano", w: 145, align: "right" },
    { label: "oracle", w: 145, align: "right" },
    { label: "可省成本上界", w: 170, align: "right" },
  ], [
    ["360 题早期测试", "84.7%", "76.7%", "85.6%", "66.4%"],
    ["205 题 half1 test", "80.5%", "71.2%", "83.9%", "64.2%"],
    ["310 题 token/risk 口径", "86.5%", "83.2%", "86.5%", "71.4%"],
  ], { rowH: 50, bodySize: 18, headerSize: 17 });
  bullet(s, 945, 410, 260, [
    "不同测试集大小不同，绝对准确率不能直接横比。",
    "真正要看的是相对 all-mini 的准确率损失和成本节省。",
    "oracle 显示：理论上有很大 nano 空间。"
  ]);
  footer(s, "基线来自 routing_cost_comparison_testdata_pareto_ollama.csv、hybrid_half1_trainval50_openai.csv、token_targeted_risk_averse_openai.csv");
}

{
  const s = slide("早期 GEPA prompt router 证明可行，但 trade-off 还不稳", "第一阶段");
  table(s, 42, 184, [
    { label: "口径", w: 230 },
    { label: "策略", w: 150 },
    { label: "准确率", w: 110, align: "right" },
    { label: "节省", w: 100, align: "right" },
    { label: "nano/mini", w: 135, align: "right" },
    { label: "解释", w: 410 },
  ], [
    ["360 题", "GEPA Pareto", "79.2%", "55.1%", "298/62", "省钱明显，但比 all-mini 低 5.6pp。"],
    ["205 题", "GEPA trainval50", "80.0%", "12.6%", "45/160", "准确率几乎贴近 all-mini，但成本收益较小。"],
    ["205 题", "OpenEvolve trainval50", "80.5%", "2.5%", "11/194", "非常保守，几乎退回 all-mini。"],
    ["205 题", "AdaEvolve trainval50", "74.6%", "45.3%", "149/56", "更便宜，但 critical 风险偏高。"],
  ], { rowH: 64, bodySize: 16, headerSize: 16, highlightRows: [1] });
  groupedBars(s, 820, 202, 390, 270, [
    { label: "Pareto 360", acc: 79.2, savings: 55.1 },
    { label: "GEPA 50", acc: 80.0, savings: 12.6 },
    { label: "OpenEvolve", acc: 80.5, savings: 2.5 },
    { label: "AdaEvolve", acc: 74.6, savings: 45.3 },
  ], { labelW: 110, maxAcc: 90, maxSavings: 80 });
  text(s, "阶段性判断：GEPA 能学到“什么时候该省”，但如果 objective 没有强约束 critical misroute，会在准确率和成本之间摆动。", 42, 580, 1060, 44, { size: 22, bold: true });
  footer(s, "来源：routing_cost_comparison_testdata_pareto_ollama.csv、hybrid_half1_trainval50_openai.csv、openevolve/adaevolve 对照 CSV");
}

{
  const s = slide("增加 train/val 数量没有单调改善，50 题反而最稳", "GEPA sweep");
  table(s, 42, 184, [
    { label: "train/val", w: 120, align: "right" },
    { label: "准确率", w: 105, align: "right" },
    { label: "节省", w: 105, align: "right" },
    { label: "nano/mini", w: 120, align: "right" },
    { label: "观察", w: 285 },
  ], [
    ["25", "78.5%", "18.6%", "64/141", "保守程度适中，少量掉点。"],
    ["50", "80.0%", "12.6%", "45/160", "准确率最接近 all-mini。"],
    ["75", "77.6%", "31.2%", "109/96", "开始更激进。"],
    ["100", "76.1%", "43.4%", "146/59", "省钱增加，准确率下滑。"],
    ["125-200", "72.2%", "62-68%", "约 192/13", "几乎变成 nano-heavy。"],
  ], { rowH: 48, bodySize: 16, headerSize: 16, highlightRows: [1] });
  groupedBars(s, 815, 184, 395, 340, [
    { label: "25", acc: 78.5, savings: 18.6 },
    { label: "50", acc: 80.0, savings: 12.6 },
    { label: "75", acc: 77.6, savings: 31.2 },
    { label: "100", acc: 76.1, savings: 43.4 },
    { label: "150", acc: 72.2, savings: 62.3 },
    { label: "200", acc: 72.2, savings: 62.3 },
  ], { labelW: 48, maxAcc: 90, maxSavings: 80 });
  text(s, "随机对照也显示类似倾向：trainval100 random control 为 75.1% / 43.9% 节省，trainval150/200 为 72.2% / 61.9% 节省。", 42, 590, 1120, 38, { size: 20, color: C.body });
  footer(s, "来源：routing_cost_comparison_testdata_hybrid_half1_trainval*_openai.csv、random_control_trainval*_openai.csv");
}

{
  const s = slide("ablation 的信号：样本构成比样本量更关键", "分桶与反思模型");
  table(s, 42, 184, [
    { label: "ablation", w: 340 },
    { label: "准确率", w: 100, align: "right" },
    { label: "节省", w: 100, align: "right" },
    { label: "nano/mini", w: 125, align: "right" },
    { label: "含义", w: 460 },
  ], [
    ["balancedish openai", "79.0%", "24.3%", "82/123", "更平衡，保留了一些成本收益。"],
    ["critical + both_wrong", "78.0%", "24.4%", "85/120", "更关注风险样本，准确率接近但成本更高。"],
    ["critical + both_correct", "76.1%", "34.7%", "122/83", "省钱更多，但准确率损失扩大。"],
    ["both_correct heavy", "72.2%", "59.9%", "185/20", "过多 easy/cheap 信号会推向 nano-heavy。"],
    ["Qwen 0.8B local reflect", "71.2%", "74.1%", "205/0", "塌缩为全 nano，说明反思/路由能力不足。"],
    ["Qwen 4B local/openai", "75.6-76.1%", "47.4-47.5%", "约 155/50", "比 0.8B 稳，但仍偏省钱。"],
  ], { rowH: 54, bodySize: 15, headerSize: 16, highlightRows: [0] });
  text(s, "这里最有价值的发现不是某个 ablation 单点赢，而是：critical / mini-only 样本必须被足够代表；否则 objective 很容易学成“默认 nano”。", 58, 585, 1060, 52, { size: 22, bold: true });
  footer(s, "来源：routing_cost_comparison_testdata_hybrid_half1_ablation50_*.csv");
}

{
  const s = slide("token 与 risk-averse 版本把结果推到更可用的前沿", "目标函数与输出格式");
  groupedBars(s, 42, 184, 590, 350, [
    { label: "manual risk full", acc: 85.5, savings: 36.9 },
    { label: "hybrid easy openai", acc: 85.2, savings: 46.0 },
    { label: "train100 pareto", acc: 84.6, savings: 57.3 },
    { label: "balanced100 risk", acc: 84.2, savings: 47.8 },
    { label: "token target openai", acc: 83.5, savings: 71.7 },
    { label: "token target ollama", acc: 83.5, savings: 64.2 },
  ], { labelW: 190, maxAcc: 90, maxSavings: 80 });
  table(s, 685, 184, [
    { label: "方案", w: 250 },
    { label: "准确率", w: 90, align: "right" },
    { label: "节省", w: 90, align: "right" },
    { label: "nano/mini", w: 115, align: "right" },
  ], [
    ["manual risk full", "85.5%", "36.9%", "183/127"],
    ["hybrid easy openai", "85.2%", "46.0%", "222/88"],
    ["token target openai", "83.5%", "71.7%", "305/5"],
    ["all-mini baseline", "86.5%", "0%", "0/310"],
  ], { rowH: 56, bodySize: 16, headerSize: 16, highlightRows: [0, 1] });
  text(s, "可选结论：如果优先保准，manual risk full 最好；如果优先降本，token target openai 几乎达到 oracle 成本，同时只比 all-mini 低 2.9pp。", 685, 476, 500, 84, { size: 21, bold: true });
  footer(s, "来源：routing_cost_comparison_testdata_token_*.csv、hybrid_easy_py_qwen35_latest_token_*.csv、balanced100_structured_trace_*.csv");
}

{
  const s = slide("soft prompt 是值得继续的支线，但还需要校准阈值", "替代优化路线");
  metric(s, 42, 184, 260, "soft prompt", "81.0%", "205 题 test accuracy", C.accent);
  metric(s, 330, 184, 260, "成本节省", "40.6%", "95 nano / 110 mini", C.accent);
  metric(s, 618, 184, 260, "critical", "6 次", "nano 误路由且 mini 会对");
  metric(s, 906, 184, 260, "正确升级", "20 次", "mini 选对且 nano 会错");
  table(s, 42, 380, [
    { label: "路线", w: 240 },
    { label: "代表结果", w: 200 },
    { label: "优点", w: 320 },
    { label: "问题", w: 360 },
  ], [
    ["OpenEvolve", "80.5% / 2.5% 节省", "能保准确率", "太保守，成本收益很小"],
    ["AdaEvolve", "74.6% / 45.3% 节省", "能探索更省钱策略", "准确率损失偏大"],
    ["Soft prompt", "81.0% / 40.6% 节省", "明显优于早期 GEPA 的成本收益", "仍有 6 个 critical misroute"],
  ], { rowH: 62, bodySize: 16, headerSize: 16, highlightRows: [2] });
  text(s, "下一步不该只加 epoch，而应把阈值、class weight 和 critical penalty 一起扫；soft prompt 的价值在于可连续调风险，不只输出一段 prompt。", 42, 600, 1120, 42, { size: 20, bold: true });
  footer(s, "来源：softprompt_router_balanced50_risk_best_expected_utility_test205.json、OpenEvolve/AdaEvolve testdata CSV");
}

{
  const s = slide("下一轮实验应围绕风险姿态，而不是再堆更多变体", "建议");
  table(s, 42, 184, [
    { label: "选择", w: 250 },
    { label: "适用场景", w: 350 },
    { label: "当前最佳数据", w: 260 },
    { label: "下一步", w: 300 },
  ], [
    ["保准优先", "希望接近 all-mini", "85.5%，36.9% 节省", "以 manual risk full 为主线，降低 critical。"],
    ["平衡优先", "可接受约 1-2pp 损失", "85.2%，46.0% 节省", "继续 hybrid easy + openai reflect。"],
    ["降本优先", "愿意用 3pp 换 70%+ 节省", "83.5%，71.7% 节省", "验证 token target 是否泛化。"],
    ["可训练路由", "想要可调阈值", "81.0%，40.6% 节省", "soft prompt 做 threshold/class-weight sweep。"],
  ], { rowH: 70, bodySize: 17, headerSize: 16, highlightRows: [1] });
  bullet(s, 42, 548, 1080, [
    "统一评估集：后续所有候选都固定到同一 310 或 205 题 heldout，避免口径漂移。",
    "主指标改为三元组：准确率损失、成本节省、critical misroute 数。",
    "先定产品风险阈值，再选择 router；这比继续追单点最高 accuracy 更有用。"
  ]);
  footer(s, "汇总依据：本工作区已有 CSV/JSON 结果；未引入外部数据");
}

await fs.mkdir(QA, { recursive: true });
for (const [i, s] of deck.slides.items.entries()) {
  const stem = `slide-${String(i + 1).padStart(2, "0")}`;
  const png = await deck.export({ slide: s, format: "png", scale: 1 });
  await fs.writeFile(path.join(QA, `${stem}.png`), new Uint8Array(await png.arrayBuffer()));
  const layout = await s.export({ format: "layout" });
  await fs.writeFile(path.join(QA, `${stem}.layout.json`), await layout.text(), "utf8");
}

const montage = await deck.export({ format: "webp", montage: true, scale: 1 });
await fs.writeFile(path.join(QA, "montage.webp"), new Uint8Array(await montage.arrayBuffer()));

const pptx = await PresentationFile.exportPptx(deck);
await pptx.save(OUT);
console.log(OUT);
