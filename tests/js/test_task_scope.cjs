/* 项目页任务列表范围：真实 tasks.js 在假 DOM 上运行。
   输入来自真实渲染页面（Flask /projects/<id> 与 /tasks 的 data-project、
   #task-seed 与批次 JSON）。这里只验证列表范围与刷新 URL，不代表浏览器渲染。 */
"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const SOURCE = path.join(__dirname, "..", "..", "behaviourlab", "web",
  "static", "tasks.js");

class FakeNode {
  constructor(tag) {
    this.tagName = tag;
    this.id = "";
    this.dataset = {};
    this.children = [];
    this.listeners = {};
    this.textContent = "";
    this.hidden = false;
    this.style = {};
    this.className = "";
    this.parent = null;
    this.classList = { add() {} };
  }
  append(child) {
    child.parent = this;
    this.children.push(child);
    return child;
  }
  replaceChildren(...nodes) {
    this.children = nodes;
    nodes.forEach((node) => { if (node) node.parent = this; });
  }
  setAttribute(name, value) { this[name] = value; }
  removeAttribute(name) { delete this[name]; }
  addEventListener(type, fn) {
    (this.listeners[type] = this.listeners[type] || []).push(fn);
  }
  dispatch(type, extra = {}) {
    const event = Object.assign({ type, target: this, preventDefault() {} },
      extra);
    (this.listeners[type] || []).forEach((fn) => fn(event));
  }
  closest(selector) {
    let node = this;
    while (node) {
      if (selector === "[data-action]" && node.dataset.action) return node;
      if (selector === "[data-batch]" && node.dataset.batch) return node;
      node = node.parent;
    }
    return null;
  }
}

function walk(node, visit) {
  visit(node);
  node.children.forEach((child) => walk(child, visit));
}

function batchId(node) {
  return node.id.startsWith("job-") ? node.id.slice(4) : null;
}

function boot(payload) {
  const nodes = {};
  ["task-list", "task-search", "task-clear-search", "task-no-match",
   "task-empty", "task-read-error", "task-retry", "task-seed", "error-text"]
    .forEach((key) => { nodes[key] = new FakeNode("div"); nodes[key].id = key; });
  nodes["task-list"].dataset.project = payload.projectId || "";
  nodes["task-seed"].textContent = JSON.stringify(payload.seed || []);
  const count = new FakeNode("p");

  const document = {
    querySelector(selector) {
      if (selector === "#task-read-error [data-error-text]") {
        return nodes["error-text"];
      }
      if (selector === "[data-task-count]") return count;
      if (selector.startsWith("#")) return nodes[selector.slice(1)] || null;
      return null;
    },
    querySelectorAll() { return []; },
    createElement(tag) { return new FakeNode(tag); },
    getElementById(id) { return null; },
  };

  const calls = [];
  const timers = [];
  let scripted = payload.responses ? payload.responses.slice() : [];
  function fetchStub(url, options) {
    calls.push({ url: String(url), options: options || {} });
    const next = scripted.length ? scripted.shift() : { ok: true, batches: [] };
    if (next && next.fail) {
      return Promise.resolve({
        ok: false, status: next.status || 500,
        text: () => Promise.resolve(JSON.stringify({
          ok: false, error: { message: next.message || "读取失败" } })),
      });
    }
    return Promise.resolve({
      ok: true, status: 200,
      text: () => Promise.resolve(JSON.stringify({
        ok: true, data: { batches: (next && next.batches) || [] } })),
    });
  }

  const context = {
    document,
    window: {
      confirm() { return true; },
      addEventListener() {},
    },
    fetch: fetchStub,
    setTimeout(fn) { timers.push(fn); return timers.length; },
    clearTimeout() {},
    console,
  };
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(SOURCE, "utf8"), context,
    { filename: "tasks.js" });

  return {
    nodes, count, calls, timers,
    script(responses) { scripted = responses.slice(); },
    cards() {
      return nodes["task-list"].children
        .map((child) => batchId(child))
        .filter(Boolean);
    },
    findCard(id) {
      return nodes["task-list"].children
        .find((child) => batchId(child) === id) || null;
    },
    findAction(id, action) {
      let found = null;
      const card = this.findCard(id);
      if (!card) return null;
      walk(card, (node) => {
        if (node.dataset.action === action) found = node;
      });
      return found;
    },
    async runTimers(count = 1) {
      let done = 0;
      while (timers.length && done < count) {
        const fn = timers.shift();
        fn();
        done += 1;
        await flush();
      }
      return done;
    },
  };
}

async function flush() {
  for (let i = 0; i < 6; i += 1) {
    await new Promise((resolve) => setImmediate(resolve));
  }
}

function batch(id, projectId, status) {
  return {
    id, assay: "oft", assay_label: "旷场实验 / OFT", status: status || "queued",
    project_id: projectId || null,
    progress: { fraction: 0.2, message: "分析中" },
    updated_at: "2026-11-03T00:00:00Z",
    videos: [{ name: id + ".mp4", status: "queued", run_id: null }],
  };
}

async function run(input) {
  const A = input.projectId;
  const B = input.otherProjectId;
  const a1 = batch(input.aBatch, A, "running");
  const b1 = batch(input.bBatch, B, "running");
  const solo = batch(input.independentBatch, null, "completed");

  // ① 项目页初始：只显示本项目批次（种子来自真实页面）。
  let ui = boot({ projectId: A, seed: [a1] });
  assert.deepStrictEqual(ui.cards(), [a1.id], "项目页初始只显示本项目");
  assert.strictEqual(ui.nodes["task-empty"].hidden, true);

  // ② 轮询：请求带 project_id，响应里的别项目批次也不会被渲染。
  ui.script([{ batches: [a1, b1, solo] }]);
  await ui.runTimers();
  const listCalls = ui.calls.filter((item) => !item.url.includes("/api/jobs/"));
  assert.strictEqual(listCalls.length, 1, "轮询只发一次列表请求");
  assert.strictEqual(listCalls[0].url,
    "/api/jobs?project_id=" + encodeURIComponent(A), "轮询保留项目范围");
  assert.deepStrictEqual(ui.cards(), [a1.id], "轮询不引入别项目或独立任务");

  // ③ 动作后刷新：停止使用当前卡片 ID，刷新仍带范围。
  ui.calls.length = 0;
  ui.script([{ batches: [a1] }, { batches: [] }]);
  const stop = ui.findAction(a1.id, "cancel");
  assert.ok(stop, "停止按钮属于本项目卡片");
  ui.nodes["task-list"].dispatch("click", { target: stop });
  await flush();
  const stopCall = ui.calls.find((item) => item.options.method === "POST");
  assert.strictEqual(stopCall.url, "/api/jobs/" + a1.id + "/cancel",
    "动作请求的 ID 来自当前卡片");
  const afterAction = ui.calls.filter((item) => !item.url.includes("/api/jobs/"));
  assert.strictEqual(afterAction[afterAction.length - 1].url,
    "/api/jobs?project_id=" + encodeURIComponent(A), "动作后刷新保留范围");
  assert.deepStrictEqual(ui.cards(), [], "停止后列表按服务器返回更新");

  // ③b 已完成批次：删除按钮同理，ID 取自该卡片，刷新仍带范围。
  const done = batch(input.aBatch, A, "completed");
  ui = boot({ projectId: A, seed: [done] });
  ui.calls.length = 0;
  ui.script([{ deleted: true }, { batches: [] }]);
  const remove = ui.findAction(done.id, "delete");
  assert.ok(remove, "已完成批次带删除按钮");
  ui.nodes["task-list"].dispatch("click", { target: remove });
  await flush();
  assert.strictEqual(
    ui.calls.find((item) => item.options.method === "DELETE").url,
    "/api/jobs/" + done.id, "删除使用当前卡片 ID");
  const removeRefresh = ui.calls.filter(
    (item) => !item.url.includes("/api/jobs/"));
  assert.strictEqual(removeRefresh[removeRefresh.length - 1].url,
    "/api/jobs?project_id=" + encodeURIComponent(A), "删除后刷新保留范围");

  // ④ 读取失败：保留上一次真实列表并报错，不退成全局列表。
  ui = boot({ projectId: A, seed: [a1] });
  ui.script([{ fail: true, status: 503, message: "服务不可用" }]);
  await ui.runTimers();
  assert.deepStrictEqual(ui.cards(), [a1.id], "失败时保留上次真实列表");
  assert.strictEqual(ui.nodes["task-read-error"].hidden, false, "显示读取错误");
  assert.ok(ui.nodes["error-text"].textContent.includes("服务不可用"));
  assert.ok(ui.calls.every((item) => item.url.includes("project_id=" + A)
      || item.url.includes("/api/jobs/")), "失败路径也不发全局请求");

  // ⑤ 空项目：没有批次就没有卡片，也不会冒出别人的任务。
  ui = boot({ projectId: A, seed: [] });
  assert.deepStrictEqual(ui.cards(), [], "空项目不显示别人的任务");

  // ⑤b 服务端若误回别项目批次，页面也不渲染（属主不符一律挡掉）。
  ui = boot({ projectId: A, seed: [a1] });
  ui.script([{ batches: [b1, solo] }]);
  await ui.runTimers();
  assert.deepStrictEqual(ui.cards(), [], "范围外的卡片不渲染");

  // ⑥ 全局任务页：不带范围，三者都在。
  ui = boot({ projectId: "", seed: [a1, b1, solo] });
  assert.deepStrictEqual(ui.cards().sort(), [a1.id, b1.id, solo.id].sort(),
    "全局页显示全部任务");
  ui.script([{ batches: [a1, b1, solo] }]);
  await ui.runTimers();
  const globalCalls = ui.calls.filter((item) => !item.url.includes("/api/jobs/"));
  assert.strictEqual(globalCalls[globalCalls.length - 1].url, "/api/jobs",
    "全局页请求不带 project_id");

  // ⑦ 搜索只在当前范围里过滤，不会引入范围外卡片。
  ui.nodes["task-search"].value = "zzz";
  ui.nodes["task-search"].dispatch("input");
  assert.deepStrictEqual(ui.cards(), [], "搜索只过滤已有范围");
}

let raw = "";
process.stdin.on("data", (chunk) => { raw += chunk; });
process.stdin.on("end", () => {
  run(JSON.parse(raw)).then(() => {
    console.log("task-scope OK");
  }).catch((error) => {
    console.error(error);
    process.exit(1);
  });
});
