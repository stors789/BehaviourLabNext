/* 设置页异步归属：真实 settings.js 在假 DOM 上运行。
   覆盖两组反例：保存中再编辑不得清 dirty；迟到预览不得覆盖过时状态。 */
"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const SOURCE = path.join(__dirname, "..", "..", "behaviourlab", "web",
  "static", "settings.js");

class FakeNode {
  constructor(tag, props = {}) {
    this.tagName = tag;
    this.dataset = Object.assign({}, props.dataset || {});
    this.attrs = Object.assign({}, props.attrs || {});
    this.children = [];
    this.listeners = {};
    this.value = props.value || "";
    this.type = props.type || "";
    this.checked = Boolean(props.checked);
    this.textContent = props.textContent || "";
    this.hidden = Boolean(props.hidden);
    this.disabled = false;
    this.className = props.className || "";
    this.parent = null;
  }
  append(child) { child.parent = this; this.children.push(child); return child; }
  addEventListener(type, fn) {
    (this.listeners[type] = this.listeners[type] || []).push(fn);
  }
  dispatch(type) {
    const event = { type, target: this, preventDefault() {} };
    (this.listeners[type] || []).forEach((fn) => fn(event));
    // 真实 DOM 会冒泡：输入事件最终在表单上被处理。
    if (this.parent) this.parent.dispatch(type);
  }
  setAttribute(name, value) { this.attrs[name] = String(value); }
  getAttribute(name) {
    if (name === "step") return this.attrs.step === undefined ? "" : this.attrs.step;
    return this.attrs[name] === undefined ? null : this.attrs[name];
  }
  getAttributeNames() { return Object.keys(this.attrs); }
  matches(selector) {
    if (selector.startsWith("#")) return this.attrs.id === selector.slice(1);
    if (selector.startsWith(".")) return this.className.split(/\s+/).includes(selector.slice(1));
    const attr = selector.match(/^\[([a-z-]+)(?:="?([^"\]]*)"?)?\]$/);
    if (attr) {
      const key = attr[1].replace(/^data-/, "");
      const expected = attr[2];
      if (key === "type") {
        if (expected === undefined) return Boolean(this.type);
        return this.type === expected;
      }
      if (key === "submit") return this.tagName === "button" && this.type === "submit";
      const actual = this.dataset[key] !== undefined ? this.dataset[key]
        : (this.attrs[key] !== undefined ? this.attrs[key] : undefined);
      if (actual === undefined) return false;
      return expected === undefined ? true : String(actual) === expected;
    }
    const byAttr = selector.match(/^([a-z]+)\[([a-z-]+)\]$/);
    if (byAttr) return this.tagName === byAttr[1] && this.attrs[byAttr[2]] !== undefined;
    return false;
  }
  querySelectorAll(selector) {
    const found = [];
    const walk = (node) => {
      node.children.forEach((child) => {
        if (child.matches(selector)) found.push(child);
        walk(child);
      });
    };
    walk(this);
    return found;
  }
  querySelector(selector) {
    return this.querySelectorAll(selector)[0] || null;
  }
  closest(selector) {
    let node = this.parent;
    while (node) {
      if (node.matches(selector)) return node;
      node = node.parent;
    }
    return null;
  }
}

function field(labelText, input) {
  const wrap = new FakeNode("label", { className: "field" });
  const span = new FakeNode("span", { textContent: labelText });
  wrap.append(span);
  wrap.append(input);
  return wrap;
}

function form(section, fields, button) {
  const node = new FakeNode("form", { dataset: { settingsSection: section } });
  fields.forEach((input) => node.append(field(input.dataset.param || "", input)));
  const note = new FakeNode("span", { className: "form-note", textContent: "" });
  const error = new FakeNode("span", { className: "form-error", textContent: "" });
  node.append(note);
  node.append(error);
  node.append(button);
  node.note = note;
  node.errorBox = error;
  node.button = button;
  return node;
}

function buildDocument() {
  const outputCheck = new FakeNode("input", {
    type: "checkbox", dataset: { scope: "output", param: "annotated_video" },
    checked: true,
  });
  outputCheck.checked = true;
  const outputForm = form("output", [outputCheck],
    new FakeNode("button", { type: "submit" }));

  const dpi = new FakeNode("input", {
    type: "number", value: "125",
    attrs: { step: "any", min: "72", max: "600" },
    dataset: { scope: "style", param: "dpi" },
  });
  const styleForm = form("style", [dpi],
    new FakeNode("button", { type: "submit" }));
  const previewButton = new FakeNode("button", { attrs: { id: "style-preview" } });
  previewButton.type = "button";
  previewButton.setAttribute("id", "style-preview");
  const previewBox = new FakeNode("div", { attrs: { id: "style-preview-box" } });
  previewBox.hidden = true;
  const previewImage = new FakeNode("img", { attrs: { id: "style-preview-image" } });
  const previewNote = new FakeNode("p", { attrs: { id: "style-preview-note" } });

  const forms = [outputForm, styleForm];
  const all = [...forms, previewButton, previewBox, previewImage, previewNote];
  return {
    document: {
      querySelectorAll(selector) {
        if (selector === "[data-settings-section]") return forms;
        const found = [];
        all.forEach((node) => {
          if (node.matches && node.matches(selector)) found.push(node);
          node.children.forEach((child) => {
            if (child.matches && child.matches(selector)) found.push(child);
          });
        });
        return found;
      },
      querySelector(selector) {
        return this.querySelectorAll(selector)[0] || null;
      },
    },
    nodes: { outputForm, outputCheck, styleForm, dpi, previewButton,
      previewBox, previewImage, previewNote },
  };
}

function load() {
  const built = buildDocument();
  const pending = [];
  const calls = [];
  const request = (url, options) => new Promise((resolve, reject) => {
    calls.push({ url, options, resolve, reject });
  });
  const window = {
    addEventListener() {},
    BLApi: { request, showError: () => {} },
  };
  const context = {
    window,
    document: built.document,
    console,
    Date,
    Map,
    Number,
    JSON,
    Promise,
    setTimeout,
  };
  context.globalThis = context;
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(SOURCE, "utf8"), context,
    { filename: "settings.js" });
  return { window, nodes: built.nodes, calls, pending };
}

async function saveDuringEdit() {
  const { window, nodes, calls } = load();
  const form = nodes.outputForm;
  // 真实路径：先改值（dirty=1），再提交，提交在飞行中继续改值。
  nodes.outputCheck.checked = true;
  nodes.outputCheck.dispatch("input");
  assert.strictEqual(form.dataset.dirty, "1");
  form.dispatch("submit");
  assert.strictEqual(calls.length, 1, "应发出一次 PUT");

  nodes.outputCheck.checked = false;  // 请求返回前继续编辑
  nodes.outputCheck.dispatch("input");

  calls[0].resolve({ key: "output", value: { annotated_video: true } });
  await new Promise((resolve) => setImmediate(resolve));

  assert.strictEqual(form.dataset.dirty, "1",
    "更晚的编辑不能被旧请求的成功清除");
  assert.ok(form.note.textContent.includes("当前输入仍未保存"),
    `提示应说明当前输入仍未保存：${form.note.textContent}`);
  assert.strictEqual(window.BLSettings.dirty(), true, "离页仍应提醒");

  // 再点保存：这次提交 false，成功后才清 dirty。
  form.dispatch("submit");
  assert.strictEqual(calls.length, 2);
  assert.strictEqual(calls[1].options.body,
    JSON.stringify({ value: { annotated_video: false } }));
  calls[1].resolve({ key: "output", value: { annotated_video: false } });
  await new Promise((resolve) => setImmediate(resolve));
  assert.strictEqual(form.dataset.dirty, "0");
  assert.strictEqual(form.note.textContent, "已保存");
}

async function latePreview() {
  const { window, nodes, calls } = load();
  nodes.previewButton.dispatch("click");
  assert.strictEqual(calls.length, 1, "应发出一次预览请求");
  assert.strictEqual(JSON.parse(calls[0].options.body).value.dpi, 125);

  // 请求返回前把 DPI 改为 200：编辑已让预览过时。
  nodes.dpi.value = "200";
  nodes.dpi.dispatch("input");
  assert.strictEqual(nodes.previewBox.dataset.state, "stale");

  calls[0].resolve({ image_url: "/api/previews/x.png", style: { dpi: 240 } });
  await new Promise((resolve) => setImmediate(resolve));
  nodes.previewImage.onload();        // 迟到图片

  assert.strictEqual(nodes.previewBox.dataset.state, "stale",
    "迟到 onload 不能把过时预览改回 ready");
  assert.strictEqual(window.BLSettings.previewState().stale, true);
}

async function main() {
  await saveDuringEdit();
  await latePreview();
  console.log("settings-async OK");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
