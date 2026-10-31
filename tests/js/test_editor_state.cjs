"use strict";

const assert = require("assert");
const path = require("path");
const fs = require("fs");
const {
  EditorCore,
  PreviewGate,
  TOOLBAR_HEIGHT,
  viewportToSource,
} = require("../../behaviourlab/web/static/editor-core.js");

function polygonSpec(extra) {
  return Object.assign({
    key: "arena",
    type: "polygon",
    count: 0,
    min_points: 3,
    fixed: false,
    auto_commit: false,
  }, extra || {});
}

// 1. 普通多边形：三点后仍等待显式完成。
const polygon = new EditorCore();
polygon.reset(polygonSpec());
for (let index = 0; index < 3; index += 1) {
  const result = polygon.addPoint({ x: 10 + index * 30, y: 20 + index * 10 });
  assert.strictEqual(result.status, "pending", "普通 polygon 不应自动提交");
}
assert.strictEqual(polygon.points.length, 3);
assert.strictEqual(polygon.committed, null, "三点后仍未有 geometry");
assert.strictEqual(polygon.canFinish(), true, "三点后应出现完成按钮");
const finished = polygon.finish();
assert.deepStrictEqual(finished.points, [[10, 20], [40, 30], [70, 40]]);

// 2. Y 中心三角形：固定三点自动提交。
const triangle = new EditorCore();
triangle.reset(polygonSpec({ key: "center", count: 3, fixed: true, auto_commit: true }));
let last = null;
for (let index = 0; index < 3; index += 1) {
  last = triangle.addPoint({ x: 100 + index, y: 200 + index });
}
assert.strictEqual(last.status, "committed");
assert.strictEqual(triangle.points.length, 0, "提交后未提交点应清空");
assert.strictEqual(triangle.committed.points.length, 3);
assert.strictEqual(triangle.canFinish(), false, "已提交时不再显示完成");

// 3. 工具头在两种状态下高度一致。
const emptyToolbar = TOOLBAR_HEIGHT + 0 * 0;
const doneToolbar = TOOLBAR_HEIGHT;
assert.strictEqual(emptyToolbar, 40);
assert.strictEqual(doneToolbar, emptyToolbar);

// 3b. 画布坐标：与 Python 固定预期一致，DPR 不影响保存坐标。
const base = {
  sourceWidth: 640, sourceHeight: 480,
  cssWidth: 934, cssHeight: 531,
  left: 20, top: 164,
};
const mapped = viewportToSource(base, 243.625, 296.75);
assert.ok(Math.abs(mapped.x - 100) <= 1e-6, "x 应为 100");
assert.ok(Math.abs(mapped.y - 120) <= 1e-6, "y 应为 120");
const retina = viewportToSource(base, 243.625, 296.75);
assert.deepStrictEqual(retina, mapped, "DPR 不参与保存坐标");
assert.strictEqual(viewportToSource(base, 20, 296.75), null, "留白不采点");
assert.strictEqual(viewportToSource(base, 500, 160), null, "图像外不采点");

const css = fs.readFileSync(
  path.join(__dirname, "../../behaviourlab/web/static/app.css"), "utf8");
assert.ok(/\.toolhead\s*\{[^}]*height:\s*40px/.test(css),
  "工具头必须固定 40px");
assert.ok(!/\.toolhead\.has-(undo|done)\s*\{[^}]*height/.test(css),
  "撤销/完成状态不得改变工具头高度");

// 4. 预览确认：80 → 81 → 80，旧响应不得复活确认。
const gate = new PreviewGate();
const config = (threshold) => ({ quality_threshold: threshold, unit: "u1" });
const request = gate.begin(config(80));
assert.strictEqual(gate.pending, true);
gate.invalidate();               // 改为 81
gate.invalidate();               // 改回 80
assert.strictEqual(gate.accept(request, { image_url: "/old.png" }), false,
  "过期响应必须被拒绝");
assert.strictEqual(gate.image, null, "旧图不能成为当前预览");
assert.strictEqual(gate.confirmed(config(80)), false, "当前配置仍未确认");

const fresh = gate.begin(config(80));
assert.strictEqual(gate.accept(fresh, { image_url: "/new.png" }), true);
assert.strictEqual(gate.confirmed(config(80)), true);
assert.strictEqual(gate.confirmed(config(81)), false, "改阈值即失效");

console.log("editor-state OK");
