import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const OUT = "C:/Users/25875/gepa/outputs/half1_205_attempts_summary.pptx";
const QA = "C:/Users/25875/gepa/outputs/half1_205_attempts_summary_render";
const W = 1280, H = 720;
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
  ["critical + both_correct", "19/0/31/0", "38/0/62/0%", "76.1%", "34.7%", "122/83"],
  ["critical + both_wrong", "19/0/0/31", "38/0/0/62%", "78.0%", "24.4%", "85/120"],
  ["both_correct heavy", "5/0/40/5", "10/0/80/10%", "72.2%", "59.9%", "185/20"],
  ["both_wrong heavy", "5/0/14/31", "10/0/28/62%", "74.6%", "47.3%", "157/48"],
];

function addSlide(title, kicker = "") {
  const s = deck.slides.add();
  s.background.fill = C.canvas;
  if (kicker) tx(s, kicker, 42, 34, 600, 24, { size: 15, color: C.muted, bold: true });
  tx(s, title, 42, 62, 1040, 78, { size: 39, bold: true, color: C.ink });
  shape(s, 42, 146, 1196, 1, C.rule);
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
    fontFace: "Microsoft YaHei",
    fontSize: o.size ?? 18,
    bold: o.bold ?? false,
    color: o.color ?? C.body,
    alignment: o.align ?? "left",
  };
  return t;
}

function shape(s, x, y, w, h, fill, border = "none") {
  return s.shapes.add({
    geometry: "rect",
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: { style: "solid", fill: border, width: border === "none" ? 0 : 1 },
  });
}

function metric(s, x, y, w, label, value, note = "", color = C.ink) {
  shape(s, x, y, w, 138, C.panel);
  tx(s, label, x + 18, y + 16, w - 36, 24, { size: 15, color: C.muted, bold: true });
  tx(s, value, x + 18, y + 46, w - 36, 50, { size: 39, color, bold: true });
  tx(s, note, x + 18, y + 102, w - 36, 26, { size: 16, color: C.body });
}

function footer(s, source) {
  tx(s, source, 42, 662, 1050, 24, { size: 12, color: C.muted });
  tx(s, String(deck.slides.items.length).padStart(2, "0"), 1186, 662, 52, 24, { size: 14, color: C.muted, align: "right" });
}

function table(s, x, y, cols, rows, opt = {}) {
  const headerH = opt.headerH ?? 42;
  const rowH = opt.rowH ?? 42;
  const totalW = cols.reduce((a, c) => a + c.w, 0);
  shape(s, x, y, totalW, headerH, C.panel);
  let cx = x;
  cols.forEach((c) => {
    tx(s, c.label, cx + 8, y + 11, c.w - 16, 22, { size: opt.headerSize ?? 15, bold: true, align: c.align ?? "left" });
    cx += c.w;
  });
  rows.forEach((r, i) => {
    const yy = y + headerH + i * rowH;
    shape(s, x, yy, totalW, 1, C.rule);
    cx = x;
    cols.forEach((c, ci) => {
      const val = Array.isArray(r) ? r[ci] : r[c.key];
      const hi = opt.highlightRows?.includes(i);
      tx(s, val, cx + 8, yy + 10, c.w - 16, rowH - 12, { size: opt.bodySize ?? 15, bold: hi, color: hi ? C.wrong : C.body, align: c.align ?? "left" });
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
    if (ww > 0) shape(s, cx, y, ww, h, color);
    if (labels && ww > 34) tx(s, `${Math.round(row[k] / total * 100)}%`, cx + 3, y + 5, ww - 6, h - 8, { size: 11, color: color === C.both ? C.ink : "#FFFFFF", align: "center", bold: true });
    cx += ww;
  });
}

function legend(s, x, y) {
  const items = [
    ["mini_only", C.mini, "mini 对 / nano 错"],
    ["nano_only", C.nano, "nano 对 / mini 错"],
    ["both_correct", C.both, "都对"],
    ["both_wrong", C.wrong, "都错"],
  ];
  let cx = x;
  items.forEach(([key, color, label]) => {
    shape(s, cx, y + 7, 18, 10, color);
    tx(s, label, cx + 24, y, 135, 24, { size: 13, color: C.body });
    cx += 155;
  });
}

function smallBars(s, x, y, rows, opt = {}) {
  const labelW = opt.labelW ?? 70;
  const plotW = opt.plotW ?? 300;
  rows.forEach((r, i) => {
    const yy = y + i * 42;
    tx(s, r.label, x, yy - 4, labelW, 24, { size: 14, color: C.body });
    shape(s, x + labelW, yy, plotW, 9, C.pale);
    shape(s, x + labelW, yy, plotW * r.acc / 90, 9, C.ink);
    shape(s, x + labelW, yy + 16, plotW, 9, C.pale);
    shape(s, x + labelW, yy + 16, plotW * r.savings / 80, 9, C.wrong);
    tx(s, `${r.acc.toFixed(1)}%`, x + labelW + plotW + 10, yy - 5, 55, 20, { size: 12, color: C.ink });
    tx(s, `${r.savings.toFixed(1)}%`, x + labelW + plotW + 10, yy + 10, 55, 20, { size: 12, color: C.wrong });
  });
}

{
  const s = deck.slides.add();
  s.background.fill = C.canvas;
  tx(s, "只看 first-half 205 时期", 42, 66, 920, 74, { size: 58, bold: true });
  tx(s, "训练/验证来自前 205 题，测试固定为后 205 题", 42, 154, 900, 42, { size: 26, color: C.body });
  shape(s, 42, 242, 1196, 1, C.ink);
  metric(s, 42, 304, 260, "总题数", "410", "前 205 + 后 205");
  metric(s, 326, 304, 260, "训练候选池", "205", "只从 first half 采样");
  metric(s, 610, 304, 260, "固定测试集", "205", "second half 不变");
  metric(s, 894, 304, 260, "size sweep", "25-200", "每个 size 内 80/20 切 train/val", C.wrong);
  tx(s, "这版只讲这个阶段：hybrid_half1_trainval size sweep、random control、size=50 ablation，以及对应的 205 题 testdata 结果。", 42, 524, 1100, 54, { size: 23, bold: true });
  footer(s, "来源：routing_split_half1_*.json、gepa_router_prompt_result_hybrid_half1_*.json、routing_cost_comparison_testdata_hybrid_half1_*.csv");
}

{
  const s = addSlide("固定测试集的结构说明了为什么路由有空间", "205/205 口径");
  metric(s, 42, 184, 250, "test mini_only", "26", "12.7%；必须升 mini", C.wrong);
  metric(s, 316, 184, 250, "test both_correct", "139", "67.8%；理论上可走 nano");
  metric(s, 590, 184, 250, "test both_wrong", "33", "16.1%；选谁都错");
  metric(s, 864, 184, 250, "oracle", "83.9%", "179 nano / 26 mini", C.wrong);
  table(s, 42, 388, [
    { label: "策略", w: 220 },
    { label: "准确率", w: 130, align: "right" },
    { label: "成本/千题", w: 140, align: "right" },
    { label: "节省", w: 120, align: "right" },
    { label: "路由", w: 140, align: "right" },
    { label: "含义", w: 380 },
  ], [
    ["all-mini", "80.5%", "$1.124", "0%", "0/205", "准确率基线。"],
    ["all-nano", "71.2%", "$0.291", "74.1%", "205/0", "省钱但丢 9.3pp。"],
    ["oracle", "83.9%", "$0.402", "64.2%", "179/26", "理论上 both_correct 很多，空间很大。"],
  ], { rowH: 54, bodySize: 16, highlightRows: [2] });
  footer(s, "固定 test 205 outcome：mini_only 26、nano_only 7、both_correct 139、both_wrong 33");
}

{
  const s = addSlide("size 越大，样本越接近自然分布，critical 比例被稀释", "不同 size 的数据比例");
  legend(s, 42, 172);
  sweep.forEach((r, i) => {
    const yy = 220 + i * 54;
    tx(s, String(r.size), 42, yy - 2, 54, 24, { size: 17, bold: true, align: "right" });
    stackedBar(s, 118, yy, 610, 28, r, true);
    tx(s, `train/val ${r.train}/${r.val}`, 748, yy + 3, 130, 22, { size: 14, color: C.muted });
    tx(s, `${r.mini_only}/${r.nano_only}/${r.both_correct}/${r.both_wrong}`, 888, yy + 3, 210, 22, { size: 14 });
    tx(s, `${(r.mini_only/r.size*100).toFixed(1)}% critical`, 1094, yy + 3, 130, 22, { size: 14, color: C.wrong, bold: true });
  });
  tx(s, "关键变化：mini_only 数量一直是 19，但 size 从 25 增到 200 后，占比从 76.0% 降到 9.5%；both_correct 从 24.0% 升到 71.0%。", 42, 610, 1130, 38, { size: 20, bold: true });
  footer(s, "格式：mini_only / nano_only / both_correct / both_wrong；来源为 selected_outcome_counts");
}

{
  const s = addSlide("比例变化直接影响测试表现：大 size 变得更省钱，也更容易掉准", "size sweep 结果");
  table(s, 42, 184, [
    { label: "size", w: 70, align: "right" },
    { label: "数据比例 M/N/B/W", w: 205 },
    { label: "准确率", w: 95, align: "right" },
    { label: "节省", w: 90, align: "right" },
    { label: "test nano/mini", w: 125, align: "right" },
    { label: "读法", w: 420 },
  ], sweep.map((r) => [
    r.size,
    `${Math.round(r.mini_only/r.size*100)}/${Math.round(r.nano_only/r.size*100)}/${Math.round(r.both_correct/r.size*100)}/${Math.round(r.both_wrong/r.size*100)}%`,
    `${r.acc.toFixed(1)}%`,
    `${r.savings.toFixed(1)}%`,
    r.routes,
    r.size === 50 ? "最好保准点：80.0%，但只省 12.6%。" : (r.size >= 125 ? "both_correct 占比高，学成 nano-heavy。" : "处在保准和省钱之间。")
  ]), { rowH: 46, bodySize: 15, highlightRows: [1] });
  smallBars(s, 820, 210, sweep.map(r => ({ label: String(r.size), acc: r.acc, savings: r.savings })), { labelW: 45, plotW: 255 });
  shape(s, 870, 548, 18, 8, C.ink);
  tx(s, "准确率", 894, 540, 70, 22, { size: 13 });
  shape(s, 970, 548, 18, 8, C.wrong);
  tx(s, "成本节省", 994, 540, 90, 22, { size: 13 });
  footer(s, "测试集固定为 second half 205；结果来自 routing_cost_comparison_testdata_hybrid_half1_trainval*_openai.csv");
}

{
  const s = addSlide("随机对照说明：自然比例本身会强烈推向 nano-heavy", "random control");
  legend(s, 42, 172);
  random.forEach((r, i) => {
    const yy = 224 + i * 68;
    tx(s, `random ${r.size}`, 42, yy - 2, 100, 24, { size: 16, bold: true });
    stackedBar(s, 160, yy, 530, 28, r, true);
    tx(s, `${r.mini_only}/${r.nano_only}/${r.both_correct}/${r.both_wrong}`, 710, yy + 3, 170, 22, { size: 14 });
    tx(s, `acc ${r.acc.toFixed(1)}%`, 902, yy + 3, 90, 22, { size: 14, bold: true });
    tx(s, `save ${r.savings.toFixed(1)}%`, 1006, yy + 3, 110, 22, { size: 14, color: C.wrong, bold: true });
  });
  tx(s, "对照组 50 题只有 12% mini_only、74% both_correct，所以 router 很自然会学到“多走 nano”。这解释了为什么有 critical oversampling 的 size=50 反而更保准。", 42, 538, 1120, 58, { size: 22, bold: true });
  footer(s, "random_control split 与结果：routing_split_half1_random_control_trainval*.json、routing_cost_comparison_testdata_hybrid_half1_random_control_trainval*.csv");
}

{
  const s = addSlide("size=50 ablation 主要是在改变四类样本的风险姿态", "ablation");
  table(s, 42, 184, [
    { label: "训练构成", w: 245 },
    { label: "M/N/B/W 数量", w: 150 },
    { label: "M/N/B/W 比例", w: 145 },
    { label: "准确率", w: 90, align: "right" },
    { label: "节省", w: 90, align: "right" },
    { label: "test nano/mini", w: 125, align: "right" },
    { label: "结论", w: 310 },
  ], ablations.map((r) => [
    r[0], r[1], r[2], r[3], r[4], r[5],
    r[0] === "balancedish 50" ? "最平衡，79.0% / 24.3%。" :
    r[0] === "both_correct heavy" ? "过多 both_correct 会明显推向 nano。" :
    r[0] === "critical + both_wrong" ? "更保守，但成本收益有限。" :
    "改变风险姿态，不是单纯提升。"
  ]), { rowH: 58, bodySize: 14, headerSize: 15, highlightRows: [0] });
  tx(s, "M/N/B/W = mini_only / nano_only / both_correct / both_wrong。这里的 takeaway 是：样本比例比反思模型细节更决定路由倾向。", 42, 580, 1120, 44, { size: 21, bold: true });
  footer(s, "来源：gepa_router_prompt_result_hybrid_half1_ablation50_*.json 与对应 testdata CSV");
}

{
  const s = addSlide("这批 205/205 实验的复盘结论", "结论");
  table(s, 42, 184, [
    { label: "观察", w: 340 },
    { label: "数据证据", w: 350 },
    { label: "含义", w: 430 },
  ], [
    ["size=50 是这一阶段最保准的点", "80.0% accuracy，12.6% saving，45/160 routes", "critical 比例够高，router 不会太快塌向 nano。"],
    ["size 变大不是稳定变好", "125/150/200 都是 72.2% accuracy", "both_correct 占比升到 64-71%，训练信号更省钱但风险更大。"],
    ["random control 解释了分布效应", "random50 是 74% both_correct，只 12% mini_only", "自然采样更像“省钱训练集”，不适合保准目标。"],
    ["ablation 证明要主动配比", "balancedish50：26/18/28/28%，79.0% / 24.3%", "下一步应先定 M/N/B/W 配比，再跑 optimizer。"],
  ], { rowH: 78, bodySize: 17, headerSize: 16, highlightRows: [0] });
  tx(s, "建议下一轮：固定 test 205 不变，围绕 M/N/B/W 配比做小网格；主指标同时看准确率、成本节省和 critical misroute。", 42, 610, 1120, 38, { size: 21, bold: true });
  footer(s, "只覆盖 first-half 205 train/val -> second-half 205 test 阶段");
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
