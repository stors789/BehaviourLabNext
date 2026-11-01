/* 改容差即旧评估整体失效：真实 landmarks.js 在假 DOM 上运行。
   验证完成后修改容差：统计表、预测图层与提示同步消失，人工标注保留。 */
"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const SOURCE = path.join(__dirname, "..", "..", "behaviourlab", "web",
  "static", "landmarks.js");

class FakeNode {
  constructor(tag, props = {}) {
    this.tagName = tag;
    this.id = props.id || "";
    this.dataset = {};
    this.children = [];
    this.listeners = {};
    this.value = props.value || "";
    this.textContent = props.textContent || "";
    this.hidden = Boolean(props.hidden);
    this.selectedOptions = props.selectedOptions || [];
    this.parent = null;
    this.style = {};
  }
  get parentElement() { return this.parent; }
  append(child) { child.parent = this; this.children.push(child); return child; }
  addEventListener(type, fn) {
    (this.listeners[type] = this.listeners[type] || []).push(fn);
  }
  dispatch(type, extra = {}) {
    const event = Object.assign({ type, target: this, preventDefault() {} },
      extra);
    (this.listeners[type] || []).forEach((fn) => fn(event));
  }
  getBoundingClientRect() {
    return { width: 400, height: 300, left: 0, top: 0 };
  }
  replaceChildren() { this.children = []; }
  scrollIntoView() {}
}

function buildDocument() {
  const byId = {};
  const ids = ["landmark-form", "landmark-unit", "landmark-frames",
    "landmark-tolerance", "landmark-workbench", "landmark-frame",
    "landmark-point", "landmark-canvas", "landmark-canvas-empty",
    "landmark-frame-note", "landmark-supported", "tool-tolerance",
    "tool-error", "tool-status", "evaluation-box", "evaluation-body",
    "evaluation-note", "tool-save", "tool-verify", "tool-visible",
    "tool-invisible", "tool-clear"];
  for (const id of ids) byId[id] = new FakeNode("div", { id });
  byId["landmark-workbench"].hidden = true;
  byId["evaluation-box"].hidden = true;

  const wrap = new FakeNode("div", { id: "canvas-wrap" });
  wrap.append(byId["landmark-canvas"]);

  const openButton = new FakeNode("button", { id: "open-session-1" });
  openButton.dataset.openSession = "s1";

  const document = {
    querySelector(selector) {
      if (selector.startsWith("#")) return byId[selector.slice(1)] || null;
      return null;
    },
    querySelectorAll(selector) {
      if (selector === "[data-open-session]") return [openButton];
      return [];
    },
    createElement(tag) { return new FakeNode(tag); },
  };
  return { document, byId, openButton };
}

const images = [];
class ImageShim {
  constructor() {
    this.naturalWidth = 100;
    this.naturalHeight = 100;
    images.push(this);
  }
}

const requests = [];
const BLApi = {
  request(url, options) {
    return new Promise((resolve, reject) => {
      requests.push({ url, options: options || {}, resolve, reject });
    });
  },
};

const context = { document: null };
const { document, byId, openButton } = buildDocument();
context.document = document;
context.window = {
  BLApi,
  devicePixelRatio: 1,
  addEventListener() {},
};
context.Image = ImageShim;
vm.createContext(context);
vm.runInContext(fs.readFileSync(SOURCE, "utf8"), context, { filename: "landmarks.js" });

// 记录画布调用：预测框 strokeRect / 预测标签 fillText / 人工标注 arc。
const drawLog = [];
function fakeContext2d() {
  const record = (name) => (...args) => drawLog.push({ name, args });
  return { setTransform: record("setTransform"), clearRect: record("clearRect"),
    drawImage: record("drawImage"), beginPath: record("beginPath"),
    arc: record("arc"), fill: record("fill"), stroke: record("stroke"),
    strokeRect: record("strokeRect"), fillText: record("fillText") };
}
byId["landmark-canvas"].getContext = () => fakeContext2d();

const SESSION = {
  id: "s1", assay: "nor", orientation: 0, asset_id: "asset1",
  frames: [0], tolerance_px: 11, records: [], evaluation: null,
};

async function flush() {
  for (let i = 0; i < 5; i += 1) {
    await new Promise((resolve) => setImmediate(resolve));
  }
}

(async () => {
  openButton.dispatch("click");
  await flush();
  requests.shift().resolve({ session: SESSION });
  await flush();
  images[0].onload();

  // 一次人工标注（body）：容差修改后必须保留。
  // contain 布局：图按 3 倍缩放居中，clientX 需先越过 50px 左留白。
  byId["landmark-canvas"].dispatch("click", { clientX: 140, clientY: 150 });

  // 保存并验证：评估渲染出预测框与统计表。
  byId["tool-verify"].dispatch("click");
  await flush();
  const put = requests.filter((item) => item.options.method === "PUT").pop();
  put.resolve({ session: { ...SESSION, records: JSON.parse(put.options.body).records } });
  await flush();
  const evaluate = requests.filter((item) => item.url.endsWith("/evaluate")).pop();
  const evaluation = {
    tolerance_px: 11, supported_points: ["body"], note: "", metrics: {},
    by_point: { body: { visible: 1, covered: 1, hit: 1,
      matched_error_median_px: 1.0, matched_error_p90_px: 1.0 } },
    records: [{ frame: 0, points: {
      body: { prediction: [60, 50], error_px: 1.0 } } }],
  };
  evaluate.resolve({
    session: { ...SESSION, records: JSON.parse(put.options.body).records,
      evaluation },
    evaluation,
  });
  await flush();
  assert.strictEqual(byId["evaluation-box"].hidden, false, "验证完成表可见");
  const predicted = drawLog.filter((call) => call.name === "strokeRect"
    || (call.name === "fillText" && String(call.args[0]).includes("预测")));
  assert.ok(predicted.length >= 2, "旧评估画出了预测框与预测标签");

  // 修改容差：统计表、预测图层与提示同步失效，人工标注保留。
  drawLog.length = 0;
  byId["tool-tolerance"].value = "12";
  byId["tool-tolerance"].dispatch("input");
  assert.strictEqual(byId["evaluation-box"].hidden, true, "统计表立即消失");
  assert.strictEqual(byId["tool-status"].textContent, "", "旧提示一并复位");
  const repainted = drawLog.some((call) => call.name === "clearRect")
    && drawLog.some((call) => call.name === "drawImage");
  assert.ok(repainted, "画布已同步重绘");
  assert.ok(!drawLog.some((call) => call.name === "strokeRect"),
    "旧预测框不再出现");
  assert.ok(!drawLog.some((call) => call.name === "fillText"
    && String(call.args[0]).includes("预测")), "旧预测标签不再出现");
  assert.ok(drawLog.some((call) => call.name === "arc"),
    "人工标注保留在画布上");

  console.log("landmarks tolerance reset: all assertions passed");
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
