/* 关键点会话配置继承：真实 landmarks.js 在假 DOM 上运行。
   场景：① 固定反例（90°/12.5fps/三层检测）payload 带完整有效检测与
   单位层原样；② 0°/null 默认例；③ 创建请求期间切走，迟到响应丢弃。 */
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

const requests = [];
const BLApi = {
  request(url, options) {
    return new Promise((resolve, reject) => {
      requests.push({ url, options: options || {}, resolve, reject });
    });
  },
};

class ImageShim {
  constructor() {
    this.naturalWidth = 100;
    this.naturalHeight = 100;
  }
}

function boot(option) {
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
  byId["landmark-unit"].selectedOptions = [option];
  byId["landmark-frames"].value = "0";
  byId["landmark-tolerance"].value = "10";

  const document = {
    querySelector(selector) {
      if (selector.startsWith("#")) return byId[selector.slice(1)] || null;
      return null;
    },
    querySelectorAll() { return []; },
    createElement(tag) { return new FakeNode(tag); },
  };
  const context = { document };
  context.window = { BLApi, devicePixelRatio: 1, addEventListener() {} };
  context.Image = ImageShim;
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(SOURCE, "utf8"), context,
    { filename: "landmarks.js" });
  byId["landmark-canvas"].getContext = () => ({ setTransform() {},
    clearRect() {}, drawImage() {}, beginPath() {}, arc() {}, fill() {},
    stroke() {}, strokeRect() {}, fillText() {} });
  return byId;
}

async function flush() {
  for (let i = 0; i < 5; i += 1) {
    await new Promise((resolve) => setImmediate(resolve));
  }
}

function optionWith(dataset) {
  const option = new FakeNode("option");
  Object.assign(option.dataset, dataset);
  option.value = dataset.value || "u1";
  return option;
}

async function extractPayload(dataset) {
  const byId = boot(optionWith(dataset));
  byId["landmark-form"].dispatch("submit");
  await flush();
  const create = requests.find((item) => item.url === "/api/landmarks"
    && item.options.method === "POST");
  if (!create) {
    console.error("no create request");
    process.exit(1);
  }
  process.stdout.write(create.options.body);
}

if (process.argv[2] === "extract") {
  // 端到端模式：stdin 是真实渲染页面的 option dataset（camelCase），
  // 输出真实 landmarks.js createSession 构造的 payload JSON。
  let raw = "";
  process.stdin.on("data", (chunk) => { raw += chunk; });
  process.stdin.on("end", () => {
    extractPayload(JSON.parse(raw)).catch((error) => {
      console.error(error);
      process.exit(1);
    });
  });
} else {
  (async () => {
  // ① 固定反例：真实页面选项承载 90°/12.5fps 与三层覆盖后的检测。
  const effective = { bg_method: "temporal", temporal_diff_threshold: 33,
    gray_threshold: 111, min_area: 200 };
  const byId = boot(optionWith({
    asset: "a1", assay: "oft", name: "小鼠 1",
    geometry: JSON.stringify({ zones: { arena: {
      type: "quad", points: [[20, 10], [80, 10], [80, 70], [20, 70]] } } }),
    options: JSON.stringify({ arena_width_cm: 50 }),
    detection: JSON.stringify(effective),
    unitDetection: JSON.stringify({ min_area: 200 }),
    orientation: "90", fpsOverride: "12.5",
  }));
  byId["landmark-form"].dispatch("submit");
  await flush();
  const create = requests.find((item) => item.url === "/api/landmarks"
    && item.options.method === "POST");
  assert.ok(create, "创建请求已发出");
  const payload = JSON.parse(create.options.body);
  assert.strictEqual(payload.orientation, 90, "方向继承自视频");
  assert.strictEqual(payload.fps_override, 12.5, "解释FPS继承自视频");
  assert.deepStrictEqual(payload.detection, effective,
    "会话层携带完整有效检测");
  assert.deepStrictEqual(payload.unit.detection, { min_area: 200 },
    "单位层保持自己的覆盖原样");
  assert.strictEqual(payload.unit.id, "u1");
  assert.deepStrictEqual(payload.unit.geometry, { zones: { arena: {
    type: "quad", points: [[20, 10], [80, 10], [80, 70], [20, 70]] } } });

  // ② 默认例：0° 与 null 原样传递，不再写死。
  requests.length = 0;
  const byId2 = boot(optionWith({
    asset: "a2", assay: "oft", name: "小鼠 2",
    geometry: JSON.stringify({ zones: {} }),
    options: "{}", detection: "{}", unitDetection: "{}",
    orientation: "0", fpsOverride: "null",
  }));
  byId2["landmark-form"].dispatch("submit");
  await flush();
  const plain = JSON.parse(
    requests.find((item) => item.options.method === "POST").options.body);
  assert.strictEqual(plain.orientation, 0);
  assert.strictEqual(plain.fps_override, null);

  // ③ 创建期间切走（与真实编辑/切会话同一 revision 机制）：
  // 迟到响应不得打开旧会话、不得覆盖当前选择。
  requests.length = 0;
  const byId3 = boot(optionWith({
    asset: "a3", assay: "oft", name: "小鼠 3",
    geometry: JSON.stringify({ zones: {} }),
    options: "{}", detection: "{}", unitDetection: "{}",
    orientation: "90", fpsOverride: "12.5",
  }));
  byId3["landmark-form"].dispatch("submit");
  await flush();
  const pending = requests.find((item) => item.options.method === "POST");
  assert.ok(pending, "创建请求挂起中");
  byId3["tool-tolerance"].dispatch("input");   // 模拟创建期间切走
  pending.resolve({ session: { id: "late", frames: [0], records: [] } });
  await flush();
  const opened = requests.some((item) =>
    item.url === "/api/landmarks/late" && item.options.method === "GET");
  assert.ok(!opened, "迟到创建响应不打开旧会话");

  console.log("landmarks config inheritance: all assertions passed");
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
}
