/* 选择器触发处与背景确认合同：真实 project.js/editor-core.js 在假 DOM 上运行。
   覆盖：未决双击只 1 次调用；取消不发请求/恢复控件；原生错误在触发处
   显示；坏背景不替换旧值；原生路径先经服务端解码确认。 */
"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const STATIC = path.join(__dirname, "..", "..", "behaviourlab", "web", "static");
const editorCore = require(path.join(STATIC, "editor-core.js"));

class FakeNode {
  constructor(tag, props = {}) {
    this.tagName = tag;
    this.id = props.id || "";
    this.dataset = props.dataset || {};
    this.children = [];
    this.listeners = {};
    this.value = props.value || "";
    this.textContent = props.textContent || "";
    this.hidden = Boolean(props.hidden);
    this.disabled = false;
    this.parent = null;
  }
  get parentElement() { return this.parent; }
  append(...children) {
    for (const child of children) {
      child.parent = this;
      this.children.push(child);
    }
    return children[children.length - 1];
  }
  addEventListener(type, fn) {
    (this.listeners[type] = this.listeners[type] || []).push(fn);
  }
  dispatch(type, extra = {}) {
    const event = Object.assign({ type, target: this, preventDefault() {} },
      extra);
    (this.listeners[type] || []).forEach((fn) => fn(event));
  }
  replaceChildren() { this.children = []; }
  remove() {}
}

function buildDocument() {
  const byId = {};
  for (const id of ["project-dropzone", "import-status", "import-errors",
    "media-assay", "project-pick-files", "project-pick-folder"]) {
    byId[id] = new FakeNode("div", { id });
  }
  byId["project-dropzone"].dataset.project = "p1";
  const document = {
    querySelector(selector) {
      if (selector.startsWith("#")) return byId[selector.slice(1)] || null;
      return null;
    },
    querySelectorAll() { return []; },
    createElement(tag) { return new FakeNode(tag); },
  };
  return { document, byId };
}

function runProjectPage(context) {
  const source = fs.readFileSync(path.join(STATIC, "project.js"), "utf8");
  vm.runInContext(source, context, { filename: "project.js" });
}

(async () => {
  // -- editor-core guardedPick / pickBackground 的直接合同 ----------------
  {
    const trigger = new FakeNode("button");
    let calls = 0;
    let release;
    const pending = new Promise((resolve) => { release = resolve; });
    const first = editorCore.guardedPick(trigger, () => {
      calls += 1;
      return pending;
    });
    const second = await editorCore.guardedPick(trigger, () => {
      calls += 1;
      return Promise.resolve("x");
    });
    assert.strictEqual(second.status, "busy", "未决期间再次触发返回 busy");
    assert.strictEqual(trigger.disabled, true, "等待期间禁用触发控件");
    release("done");
    const result = await first;
    assert.strictEqual(result.status, "done");
    assert.strictEqual(trigger.disabled, false, "完成后恢复原控件");
    assert.strictEqual(calls, 1, "两次未决点击只产生 1 次调用");
  }
  {
    const trigger = new FakeNode("button");
    const failed = await editorCore.guardedPick(
      trigger, async () => { throw new Error("窗口尚未准备好"); });
    assert.strictEqual(failed.status, "error");
    assert.strictEqual(failed.error.message, "窗口尚未准备好");
    assert.strictEqual(trigger.disabled, false, "失败同样恢复原控件");
  }
  {
    // 原生背景：先确认可解码，坏背景失败向上抛、取消不校验。
    const validations = [];
    const ok = await editorCore.pickBackground(
      { choose_image: async () => "C:/背景/场地.png" },
      { validate: async (p) => { validations.push(p); } });
    assert.deepStrictEqual(validations, ["C:/背景/场地.png"]);
    assert.strictEqual(ok.status, "ok");

    const cancelled = await editorCore.pickBackground(
      { choose_image: async () => null },
      { validate: async () => { throw new Error("不应校验"); } });
    assert.strictEqual(cancelled.status, "cancel");

    await assert.rejects(
      editorCore.pickBackground(
        { choose_image: async () => "C:/背景/坏图.png" },
        { validate: async () => { throw new Error("背景无法解码为彩色图像"); } }),
      /背景无法解码/);

    const uploaded = await editorCore.pickBackground(null, {
      picker: { pickFiles: async () => [{ name: "a.png" }] },
      upload: async () => ({ path: "C:/data/backgrounds/x.png" }),
    });
    assert.strictEqual(uploaded.status, "ok");

    const skip = await editorCore.pickBackground(null, {
      picker: { pickFiles: async () => [] },
      upload: async () => { throw new Error("不应上传"); },
    });
    assert.strictEqual(skip.status, "cancel");
  }

  // -- 真实 project.js 的触发接线 ----------------------------------------
  const { document, byId } = buildDocument();
  const requests = [];
  const BLApi = {
    request(url, options) {
      requests.push({ url, options: options || {} });
      return Promise.resolve({ assets: [], errors: [] });
    },
    showError() {},
  };
  let pickMediaCalls = 0;
  let failNext = null;
  let pendingResolve = null;
  const BLCore = {
    guardedPick: editorCore.guardedPick,
    pickMedia: async () => {
      pickMediaCalls += 1;
      if (failNext) throw failNext;
      if (pendingResolve) return new Promise((resolve) => { pendingResolve = resolve; });
      return "desktop";
    },
  };

  const context = {
    document,
    location: { hash: "" },
    history: { replaceState() {} },
    window: {
      BLApi,
      BLCore,
      BLPickers: {},
      addEventListener() {},
      location: { reload() {}, href: "", hash: "" },
    },
  };
  context.globalThis = context;
  vm.createContext(context);
  runProjectPage(context);

  const filesButton = byId["project-pick-files"];
  const errorList = byId["import-errors"];
  const statusNode = byId["import-status"];

  // 取消不发空请求、不清旧值、控件恢复。
  filesButton.dispatch("click");
  await new Promise((resolve) => setImmediate(resolve));
  assert.strictEqual(pickMediaCalls, 1);
  assert.strictEqual(filesButton.disabled, false, "取消后恢复原控件");
  assert.strictEqual(requests.length, 0, "取消不发空导入请求");

  // 原生调用失败：错误显示在原触发处的错误列表，控件恢复。
  failNext = new Error("窗口尚未准备好");
  filesButton.dispatch("click");
  await new Promise((resolve) => setImmediate(resolve));
  assert.strictEqual(filesButton.disabled, false);
  const rows = errorList.children;
  assert.ok(rows.length >= 1, "触发处显示可读错误");
  const rowTexts = rows.map((row) =>
    (row.children || []).map((cell) => cell.textContent).join("|")).join(";");
  assert.ok(rowTexts.includes("窗口尚未准备好"), `可读错误：${rowTexts}`);
  assert.strictEqual(statusNode.hidden, true, "失败不保留进行中状态");
  failNext = null;

  // 未决期间防重复点击：只 1 次选择器调用，完成后恢复。
  pendingResolve = () => {};
  filesButton.dispatch("click");
  filesButton.dispatch("click");
  await new Promise((resolve) => setImmediate(resolve));
  const callsWithPending = pickMediaCalls;
  assert.strictEqual(filesButton.disabled, true, "等待期间禁用");
  pendingResolve("desktop");
  await new Promise((resolve) => setImmediate(resolve));
  await new Promise((resolve) => setImmediate(resolve));
  assert.strictEqual(filesButton.disabled, false, "完成后恢复原控件");
  assert.ok(pickMediaCalls >= callsWithPending);

  console.log("pick triggers & background contract: all assertions passed");
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
