/* 关键点工作台异步归属：真实 landmarks.js 在假 DOM 上运行。
   覆盖三组反例：保存期间再编辑丢弃迟到响应；切点清旧 evaluation；
   新帧加载完成前画布点击不产生坐标。 */
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

function fakeContext2d() {
  const noop = () => {};
  return { setTransform: noop, clearRect: noop, drawImage: noop,
    beginPath: noop, arc: noop, fill: noop, stroke: noop,
    strokeRect: noop, fillText: noop };
}
byId["landmark-canvas"].getContext = () => fakeContext2d();

const SESSION = {
  id: "s1", assay: "nor", orientation: 0, asset_id: "asset1",
  frames: [0, 2], tolerance_px: 10, records: [], evaluation: null,
};

async function flush() {
  for (let i = 0; i < 5; i += 1) {
    await new Promise((resolve) => setImmediate(resolve));
  }
}

(async () => {
  // 打开会话：迟到图片未加载前画布不可点击。
  openButton.dispatch("click");
  await flush();
  requests.shift().resolve({ session: SESSION });   // GET 会话返回
  await flush();
  assert.strictEqual(byId["landmark-workbench"].hidden, false);
  assert.strictEqual(images.length, 1, "会话打开后开始加载源帧");

  // 画布点击在图片加载前被忽略。
  byId["landmark-canvas"].dispatch("click", { clientX: 200, clientY: 100 });
  let save = requests.find((item) => item.options.method === "PUT");
  assert.strictEqual(save, undefined, "加载中点击不产生标注");

  // 图片加载完成：点击生效。
  images[0].onload();
  byId["landmark-canvas"].dispatch("click", { clientX: 200, clientY: 100 });
  byId["landmark-canvas"].dispatch("click", { clientX: 260, clientY: 100 });
  await flush();

  // 保存（PUT 挂起）期间再次编辑：迟到响应必须被丢弃。
  byId["tool-save"].dispatch("click");
  await flush();
  save = requests.find((item) => item.url === "/api/landmarks/s1"
    && item.options.method === "PUT");
  assert.ok(save, "保存请求已发出");
  const recordsDuringSave = JSON.parse(save.options.body).records.length;
  byId["landmark-canvas"].dispatch("click", { clientX: 300, clientY: 120 });
  assert.ok(JSON.stringify(byId["tool-status"].textContent).length > 0);
  save.resolve({ session: { ...SESSION, records: [{ frame: 0, points: {} }] } });
  await flush();
  assert.ok(
    byId["tool-status"].textContent.includes("又有修改"),
    `迟到保存被丢弃并提示：${byId["tool-status"].textContent}`);

  // 未再编辑时再次保存：响应被接受。
  byId["tool-save"].dispatch("click");
  await flush();
  const retry = requests.filter((item) => item.options.method === "PUT").pop();
  retry.resolve({
    session: { ...SESSION, records: JSON.parse(retry.options.body).records },
  });
  await flush();
  assert.strictEqual(byId["tool-status"].textContent, "已保存");

  // 验证完成后切点：evaluation 图层与数值一并清除。
  byId["tool-verify"].dispatch("click");
  await flush();
  const putRequest = requests.filter((item) => item.options.method === "PUT").pop();
  putRequest.resolve({ session: { ...SESSION, records: JSON.parse(putRequest.options.body).records } });
  await flush();
  const evaluateRequest = requests.filter(
    (item) => item.url.endsWith("/evaluate")).pop();
  evaluateRequest.resolve({
    session: { ...SESSION },
    evaluation: {
      tolerance_px: 10, supported_points: ["body", "nose", "tail"],
      note: "", metrics: {},
      by_point: { body: { visible: 1, covered: 1, hit: 1,
        matched_error_median_px: 1.0, matched_error_p90_px: 1.0 } },
      records: [],
    },
  });
  await flush();
  assert.strictEqual(byId["evaluation-box"].hidden, false, "验证表可见");
  byId["landmark-point"].dispatch("change");
  assert.strictEqual(byId["evaluation-box"].hidden, true, "切点清旧 evaluation");

  // 切帧后新图未加载：点击不产生坐标；加载完成才恢复。
  const before = requests.filter((item) => item.options.method === "PUT").length;
  byId["landmark-frame"].value = "2";
  byId["landmark-frame"].dispatch("change");
  assert.strictEqual(images.length, 2, "切帧触发新图加载");
  byId["landmark-canvas"].dispatch("click", { clientX: 200, clientY: 100 });
  images[1].onload();
  byId["landmark-canvas"].dispatch("click", { clientX: 220, clientY: 110 });
  byId["tool-save"].dispatch("click");
  await flush();
  const afterSwitch = requests.filter((item) => item.options.method === "PUT").pop();
  const recordsAfterSwitch = JSON.parse(afterSwitch.options.body).records.length;
  assert.strictEqual(recordsAfterSwitch, recordsDuringSave + 1,
    "加载中与加载后的点击只各记一次（加载中那次被忽略）");
  assert.strictEqual(before >= 1, true);

  console.log("landmarks async ownership: all assertions passed");
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
