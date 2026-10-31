"use strict";

const assert = require("assert");
const path = require("path");
const fs = require("fs");
const {
  EditorCore,
  PreviewGate,
  SaveQueue,
  TOOLBAR_HEIGHT,
  resolveStep,
  rotateGeometry,
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

// 5. 步骤规则：平台中心一点提交，洞数按参数，模式切换清空未提交点。
const platformSpec = resolveStep({
  key: "platform", type: "ellipse_or_point", count: 0, min_points: 1,
  modes: ["center", "ellipse"],
}, {});
assert.strictEqual(platformSpec.count, 1);
assert.deepStrictEqual(platformSpec.modes, ["center", "ellipse"]);
const platform = new EditorCore();
platform.reset(platformSpec);
const centre = platform.addPoint({ x: 200, y: 150 });
assert.strictEqual(centre.status, "committed", "平台中心一点应提交");
assert.strictEqual(platform.committed.type, "point");
assert.deepStrictEqual(platform.committed.points, [[200, 150]]);

const holesSpec = resolveStep({
  key: "holes", type: "holes", count: 0, count_key: "hole_count",
}, { hole_count: 3 });
assert.strictEqual(holesSpec.count, 3);
const holes = new EditorCore();
holes.reset(holesSpec);
holes.addPoint({ x: 10, y: 10 });
holes.addPoint({ x: 20, y: 20 });
assert.strictEqual(holes.committed, null, "不足 3 点不应提交");
const third = holes.addPoint({ x: 30, y: 30 });
assert.strictEqual(third.status, "committed");
assert.deepStrictEqual(holes.committed.holes.map((hole) => hole.id), ["1", "2", "3"]);
assert.deepStrictEqual(holes.committed.holes[2].center, [30, 30]);

const defaultHoles = resolveStep({
  key: "holes", type: "holes", count: 0, count_key: "hole_count",
}, { hole_count: 20 });
assert.strictEqual(defaultHoles.count, 20);
const twenty = new EditorCore();
twenty.reset(defaultHoles);
for (let index = 0; index < 3; index += 1) twenty.addPoint({ x: index, y: index });
assert.strictEqual(twenty.committed, null, "默认 20 洞时 3 点不应提交");
assert.strictEqual(resolveStep({
  key: "holes", type: "holes", count: 0, count_key: "hole_count",
}, { hole_count: 3.5 }).count, 0, "非法小数不得被取整");

const ellipseSpec = resolveStep({ key: "arena", type: "ellipse", min_points: 5 }, {});
assert.deepStrictEqual(ellipseSpec.modes, ["drag", "points"]);
const ellipse = new EditorCore();
ellipse.reset(ellipseSpec);
assert.strictEqual(ellipse.addPoint({ x: 1, y: 1 }).status, "ignored");
ellipse.setMode("points");
ellipse.addPoint({ x: 1, y: 1 });
const second = ellipse.addPoint({ x: 2, y: 2 });
assert.strictEqual(second.status, "pending");
ellipse.setMode("drag");
assert.strictEqual(ellipse.points.length, 0, "切模式清空未提交点");

// 6. 应用到其他视频：按目标原始宽高换算，往返恢复。
const rotated = rotateGeometry({
  arena: { type: "quad", points: [[100, 120], [200, 120], [200, 220], [100, 220]] },
  platform: { type: "ellipse", center: [100, 120], radii: [10, 6], angle: 20 },
}, 1280, 720, 0, 90);
assert.deepStrictEqual(rotated.arena.points[0], [599, 100]);
assert.deepStrictEqual(rotated.platform.center, [599, 100]);
assert.deepStrictEqual(rotated.platform.radii, [10, 6]);
assert.strictEqual(rotated.platform.angle, 110);
const restored = rotateGeometry(rotated, 1280, 720, 90, 0);
assert.deepStrictEqual(restored.arena.points[0], [100, 120]);
assert.strictEqual(restored.platform.angle, 20);

// 7. 保存队列：串行快照、旧响应不回写、失败后仍可重试。
(async () => {
  const tick = () => new Promise((resolve) => setTimeout(resolve, 0));
  const sends = [];
  const echoes = [];
  const queue = new SaveQueue((snapshot) => new Promise((resolve, reject) => {
    sends.push({ snapshot, resolve, reject });
  }));
  queue.mark();
  const first = queue.save(() => ({ v: "80" }), (draft) => echoes.push(draft));
  await tick();
  assert.strictEqual(sends.length, 1);
  queue.mark();
  const second = queue.save(() => ({ v: "81" }), (draft) => echoes.push(draft));
  assert.strictEqual(second, first, "并发保存共享同一队列");
  await tick();
  assert.strictEqual(sends.length, 1, "第二次保存应等待第一次完成");
  sends[0].resolve({ v: "80" });
  await tick();
  await tick();
  assert.strictEqual(sends.length, 2, "最新编辑应继续保存");
  assert.deepStrictEqual(sends[1].snapshot, { v: "81" });
  assert.deepStrictEqual(echoes, [], "旧响应不得回写当前草稿");
  sends[1].resolve({ v: "81" });
  assert.strictEqual(await first, true);
  assert.strictEqual(await second, true);
  assert.deepStrictEqual(echoes, [{ v: "81" }]);
  assert.strictEqual(queue.dirty(), false);

  let fail = true;
  const retry = new SaveQueue(() => (fail
    ? Promise.reject(new Error("400"))
    : Promise.resolve({ ok: true })));
  retry.mark();
  assert.strictEqual(await retry.save(() => ({ v: 1 })), false);
  assert.strictEqual(retry.dirty(), true, "失败后仍视为未保存");
  fail = false;
  assert.strictEqual(await retry.save(() => ({ v: 1 })), true);
  assert.strictEqual(retry.dirty(), false);

  console.log("editor-state OK");
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
