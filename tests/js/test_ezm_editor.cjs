// EZM 场地编辑：外缘与起始边界落在 geometry.arena / geometry.open_boundary，
// 环上辅助线只由保存几何与有效尺寸算出。
const assert = require("assert");

const {
  EditorCore,
  resolveStep,
  stepShape,
  stepApply,
  stepClear,
  moveVertex,
  ezmGuides,
} = require("../../behaviourlab/web/static/editor-core.js");

const STEPS = {
  arena: { key: "arena", type: "ellipse", min_points: 5, count: 0 },
  open_boundary: { key: "open_boundary", type: "point", count: 1 },
  mask: { key: "mask", type: "polygon", min_points: 3, optional: true },
};
const OPTIONS = {
  outer_diameter_cm: 60,
  ring_width_cm: 5,
  boundary_margin_cm: 0.5,
  confirm_s: 0.12,
  quality_threshold: 0.9,
  bin_seconds: 3,
};

function close(actual, expected, tolerance) {
  const limit = tolerance == null ? 1e-6 : tolerance;
  assert.ok(Math.abs(actual - expected) <= limit,
    "expected " + expected + ", got " + actual);
}

function drawEllipse(geometry) {
  const spec = resolveStep(STEPS.arena, OPTIONS);
  const editor = new EditorCore();
  editor.reset(spec, stepShape("ezm", geometry, "arena"));
  assert.strictEqual(spec.modes.join(","), "drag,points");
  const box = { x1: 40, y1: 70, x2: 160, y2: 130 };
  stepApply("ezm", geometry, "arena", editor.commitEllipse(box));
}

function drawBoundary(geometry, points) {
  const spec = resolveStep(STEPS.open_boundary, OPTIONS);
  const editor = new EditorCore();
  editor.reset(spec, stepShape("ezm", geometry, "open_boundary"));
  let committed = null;
  for (const point of points) {
    const result = editor.addPoint({ x: point[0], y: point[1] });
    if (result.status === "committed") committed = result.geometry;
  }
  stepApply("ezm", geometry, "open_boundary", committed);
}

function geometryFlow() {
  const geometry = {};
  drawEllipse(geometry);
  assert.ok(geometry.arena, "外缘已保存");
  assert.strictEqual(geometry.arena.type, "ellipse");
  assert.deepStrictEqual(geometry.arena.center, [100, 100]);
  assert.deepStrictEqual(geometry.arena.radii, [60, 30]);
  assert.strictEqual(geometry.arena.angle, 0);

  drawBoundary(geometry, [[160, 100]]);
  assert.deepStrictEqual(Object.keys(geometry).sort(),
    ["arena", "open_boundary"]);
  assert.deepStrictEqual(geometry.open_boundary,
    { type: "point", points: [[160, 100]] });

  // 已绘制的单点再点一次是拖动：原地改手柄，不生成第二个点。
  stepApply("ezm", geometry, "open_boundary",
    moveVertex(stepShape("ezm", geometry, "open_boundary"), 0,
      { x: 150, y: 100 }));
  assert.deepStrictEqual(geometry.open_boundary.points, [[150, 100]]);

  // 拖动起始点只改方向，不写回任何显示坐标。
  stepApply("ezm", geometry, "open_boundary",
    moveVertex(stepShape("ezm", geometry, "open_boundary"), 0,
      { x: 100, y: 40 }));
  const rotated = ezmGuides(geometry.arena, geometry.open_boundary, 60, 5);
  close(rotated.boundary_angle_deg, 270);
  close(rotated.rays[0].to[0], 100);

  // 撤销与重画只清当前步骤。
  stepClear("ezm", geometry, "open_boundary");
  assert.strictEqual(stepShape("ezm", geometry, "open_boundary"), null);
  assert.ok(stepShape("ezm", geometry, "arena"));
  drawBoundary(geometry, [[160, 100]]);

  // 序列化往返仍保持同一结构。
  const reloaded = JSON.parse(JSON.stringify(geometry));
  assert.deepStrictEqual(Object.keys(reloaded).sort(),
    ["arena", "open_boundary"]);
  assert.strictEqual(reloaded.open_boundary.type, "point");
  return reloaded;
}

function guideFlow(geometry) {
  const guides = ezmGuides(geometry.arena, geometry.open_boundary, 60, 5);
  assert.ok(guides);
  close(guides.boundary_angle_deg, 0);
  assert.deepStrictEqual(guides.center, [100, 100]);
  assert.deepStrictEqual(guides.rings, { outer: 30, inner: 25 });
  // 内圈是同心等比椭圆：半径按 (R - w)/R 缩放。
  close(guides.inner_radii[0], 50);
  close(guides.inner_radii[1], 25);
  // 起始射线沿 +x，外端在椭圆右端点、内端在同一方向的 5/6 处。
  close(guides.rays[0].to[0], 160);
  close(guides.rays[0].to[1], 100);
  close(guides.rays[0].from[0], 150);
  close(guides.rays[0].from[1], 100);
  // 四个标签按顺时针 O1、C1、O2、C2，开放绿封闭蓝由 open 标记。
  assert.deepStrictEqual(guides.labels.map((item) => item.zone),
    ["O1", "C1", "O2", "C2"]);
  assert.deepStrictEqual(guides.labels.map((item) => item.open),
    [true, false, true, false]);
  // O1 标签落在环带中线、45° 方向：物理 (19.445,19.445) → 像素 (138.891,119.445)。
  close(guides.labels[0].point[0], 138.891, 1e-3);
  close(guides.labels[0].point[1], 119.445, 1e-3);
  // 四条分界线相差 90°。
  for (let index = 1; index < 4; index += 1) {
    close(guides.rays[index].angle_deg - guides.rays[index - 1].angle_deg, 90);
  }

  // 参数改变立即换算出新的内外圈与标签位置。
  const wider = ezmGuides(geometry.arena, geometry.open_boundary, 60, 10);
  assert.deepStrictEqual(wider.rings, { outer: 30, inner: 20 });
  // 中线 25 cm、45°：物理 (17.678,17.678) → 像素 (135.355,117.678)。
  close(wider.labels[0].point[0], 135.355, 1e-3);
  close(wider.labels[0].point[1], 117.678, 1e-3);
  const bigger = ezmGuides(geometry.arena, geometry.open_boundary, 120, 5);
  assert.deepStrictEqual(bigger.rings, { outer: 60, inner: 55 });

  // 退化条件不出辅助线：缺 arena、缺边界、边界落在圆心、跑道宽为负。
  assert.strictEqual(ezmGuides(null, geometry.open_boundary, 60, 5), null);
  assert.strictEqual(ezmGuides(geometry.arena, null, 60, 5), null);
  assert.strictEqual(ezmGuides(geometry.arena,
    { type: "point", points: [[100, 100]] }, 60, 5), null);
  assert.strictEqual(ezmGuides(geometry.arena, geometry.open_boundary, 60, -1),
    null);
  // ≥5 点拟合的 arena 没有中心与半轴：画布不猜椭圆，分析端仍会拟合。
  assert.strictEqual(ezmGuides(
    { type: "ellipse", points: [[160, 100], [100, 130], [40, 100],
      [100, 70], [130, 85]] }, geometry.open_boundary, 60, 5), null);

  // 近圆心判据固定 D=1：有效性不随实际外径改变，方向恒为 +x。
  const nearStart = { type: "point", points: [[100.0000024, 100]] };
  const tooClose = { type: "point", points: [[100.0000006, 100]] };
  for (const [diameter, width] of [[0.1, 1 / 120], [60, 5]]) {
    const kept = ezmGuides(geometry.arena, nearStart, diameter, width);
    assert.ok(kept, "D=1 范数约 2e-8 在任何外径下都合法");
    close(kept.boundary_angle_deg, 0, 1e-9);
    assert.strictEqual(ezmGuides(geometry.arena, tooClose, diameter, width),
      null, "D=1 范数约 5e-9 在任何外径下都拒绝");
  }
}

geometryFlow();
guideFlow(geometryFlow());
console.log("ezm-editor OK");
