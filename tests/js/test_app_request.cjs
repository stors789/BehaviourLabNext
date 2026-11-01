/* 公共请求包装器：FormData 不得手工设置 Content-Type，JSON 仍要显式声明。 */
"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const SOURCE = path.join(__dirname, "..", "..", "behaviourlab", "web",
  "static", "app.js");

function load() {
  const calls = [];
  const fetch = (url, init) => {
    calls.push({ url, init });
    return Promise.resolve({
      ok: true,
      status: 200,
      json: async () => ({ ok: true, data: { accepted: true } }),
    });
  };
  const document = {
    querySelector: () => null,
    querySelectorAll: () => [],
  };
  const context = {
    window: { location: { href: "" } },
    document,
    fetch,
    FormData,
    console,
    Promise,
    JSON,
    Error,
    Array,
    String,
    URLSearchParams,
  };
  context.globalThis = context;
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(SOURCE, "utf8"), context, { filename: "app.js" });
  return { api: context.window.BLApi, calls };
}

async function main() {
  const { api, calls } = load();

  // 健康检查是 JSON 请求：仍有 application/json。
  assert.ok(calls.length >= 1, "加载时应发健康检查");
  const health = calls[0];
  assert.strictEqual(health.init.headers["Content-Type"], "application/json");

  // FormData：交给浏览器生成带 boundary 的 multipart 头。
  const form = new FormData();
  form.append("project_id", "p".repeat(32));
  form.append("files", new Blob(["x"]), "clip.mp4");
  await api.request("/api/assets/upload", { method: "POST", body: form });
  const upload = calls[calls.length - 1];
  assert.strictEqual(upload.init.body, form);
  const headers = upload.init.headers || {};
  assert.ok(!("Content-Type" in headers),
    "FormData 请求不能带手工 Content-Type：" + JSON.stringify(headers));

  // JSON 路径不回归：对象体仍按 JSON 发出并带头。
  await api.request("/api/projects", {
    method: "POST", body: JSON.stringify({ name: "实验" }),
  });
  const jsonCall = calls[calls.length - 1];
  assert.strictEqual(jsonCall.init.headers["Content-Type"], "application/json");
  assert.strictEqual(jsonCall.init.body, JSON.stringify({ name: "实验" }));

  // 项目页真实调用的形状：files 与 project_id 同时出现在同一份 FormData。
  const projectForm = new FormData();
  projectForm.append("files", new Blob(["y"]), "project.mp4");
  projectForm.append("project_id", "q".repeat(32));
  await api.request("/api/assets/upload", { method: "POST", body: projectForm });
  const projectUpload = calls[calls.length - 1];
  assert.strictEqual(projectUpload.init.body, projectForm);
  assert.ok(!("Content-Type" in (projectUpload.init.headers || {})));
  assert.deepStrictEqual(Array.from(projectUpload.init.body.keys()).sort(),
    ["files", "project_id"]);

  console.log("app-request OK");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
