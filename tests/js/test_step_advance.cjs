// 画完自动换步骤：用真实实验步骤定义 + 真实 EditorCore 状态机跑提交序列。
// 步骤 JSON 由 tests/test_step_advance.py 从 experiments.catalog() 导出。
const assert = require("assert");
const fs = require("fs");

const {
  EditorCore,
  resolveStep,
  stepShape,
  stepApply,
  stepClear,
  nextRequiredStep,
} = require("../../behaviourlab/web/static/editor-core.js");

const STEPS = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));

function stepsFor(assay) {
  const steps = STEPS[assay];
  assert.ok(Array.isArray(steps) && steps.length, assay + " 步骤缺失");
  return steps;
}

function stepOf(assay, key) {
  const step = stepsFor(assay).find((item) => item.key === key);
  assert.ok(step, assay + " 缺少步骤 " + key);
  return step;
}

// 与 workspace.js 的 selectStep 一致：换步骤 = 换标题 + 用该步骤自己的
// 已存几何重置同一个编辑器。
function makeFlow(assay, options) {
  const geometry = {};
  const editor = new EditorCore();
  const state = { stepKey: null, switched: [], titles: [] };
  const opts = options || {};

  function drawn(key) {
    return Boolean(stepShape(assay, geometry, key));
  }

  function selectStep(key) {
    const step = stepOf(assay, key);
    state.stepKey = key;
    editor.reset(resolveStep(step, opts), stepShape(assay, geometry, key));
    state.titles.push("绘制：" + step.label);
    return step;
  }

  function commitGeometry(shape) {
    const previous = state.stepKey;
    stepApply(assay, geometry, previous, shape);
    const next = nextRequiredStep(stepsFor(assay), drawn);
    if (next && next.key !== previous) {
      selectStep(next.key);
      state.switched.push({ from: previous, to: next.key });
      return next;
    }
    state.switched.push({ from: previous, to: null });
    return null;
  }

  return { geometry, editor, state, selectStep, commitGeometry };
}

function addPoints(flow, points) {
  for (const point of points) {
    const result = flow.editor.addPoint({ x: point[0], y: point[1] });
    if (result.status === "committed") return flow.commitGeometry(result.geometry);
  }
  return null;
}

function finishStep(flow) {
  const shape = flow.editor.finish();
  if (!shape) return null;
  return flow.commitGeometry(shape);
}

function quad(x, y, size) {
  return [[x, y], [x + size, y], [x + size, y + size], [x, y + size]];
}

function triangle(points) {
  return points;
}

// -- EPM：中心区完成后应落到下一尚缺的必需臂区 ------------------------
function epmFlow() {
  const flow = makeFlow("epm");
  const order = ["center", "open_1", "open_2", "closed_1", "closed_2"];
  const expected = ["open_1", "open_2", "closed_1", "closed_2", null];
  flow.selectStep("center");
  order.forEach((key, index) => {
    assert.strictEqual(flow.state.stepKey, key, "EPM 当前步骤应为 " + key);
    let next = null;
    if (index === 0) {
      next = addPoints(flow, quad(90, 90, 20));
    } else {
      // 每一步先自己选中（用户点步骤行），再画四角。
      flow.selectStep(key);
      next = addPoints(flow, quad(10 + index * 30, 10, 12));
    }
    assert.strictEqual(next ? next.key : null, expected[index],
      "EPM " + key + " 之后应推进到 " + expected[index]);
    if (next) {
      // 新步骤：标题、编辑器草稿与几何归属都换成新步骤。
      assert.strictEqual(flow.state.stepKey, next.key);
      assert.strictEqual(flow.editor.spec.key, next.key);
      assert.strictEqual(flow.editor.committed, null);
      assert.strictEqual(flow.editor.points.length, 0);
      assert.strictEqual(flow.editor.editable(), true);
      assert.strictEqual(stepShape("epm", flow.geometry, next.key), null);
      assert.strictEqual(flow.state.titles[flow.state.titles.length - 1],
        "绘制：" + next.label);
    }
  });
  // 五个臂区都在 zones 里，遮罩没有出现也不挡推进。
  assert.deepStrictEqual(Object.keys(flow.geometry.zones).sort(),
    ["center", "closed_1", "closed_2", "open_1", "open_2"]);
  assert.strictEqual(flow.geometry.mask, undefined);

  // 全部必需步骤完成后重画任一臂：仍停在原地，不自动进入分析。
  flow.selectStep("open_2");
  // 重画 = 先清该步几何再画（与工具条“重画”一致），提交后仍按同一规则。
  stepClear("epm", flow.geometry, "open_2");
  flow.editor.redraw();
  addPoints(flow, quad(0, 0, 10));
  assert.strictEqual(flow.state.stepKey, "open_2");
  assert.ok(stepShape("epm", flow.geometry, "open_2"), "重画覆盖了该臂几何");
  // 保持可编辑：已提交几何的顶点仍可被拖动，编辑器不会被锁死。
  const hit = flow.editor.hitHandle({ x: 0, y: 0 }, 1);
  assert.ok(hit && hit.kind === "vertex", "已提交几何的顶点仍可拖动");
}

// -- EZM：外缘椭圆完成后落到 O1 起始边界 ------------------------------
function ezmFlow() {
  const flow = makeFlow("ezm");
  flow.selectStep("arena");
  const box = { x1: 10, y1: 10, x2: 110, y2: 90 };
  const shape = flow.editor.commitEllipse(box);
  assert.ok(shape && shape.type === "ellipse");
  const next = flow.commitGeometry(shape);
  assert.ok(next, "外缘完成后必须换步骤");
  assert.strictEqual(next.key, "open_boundary");
  assert.strictEqual(flow.state.stepKey, "open_boundary");
  assert.strictEqual(flow.editor.canFinish(), false, "边界是点位步骤，无完成按钮");

  // O1 起始边界：单点即提交；此后没有剩余必需步骤。
  const result = flow.editor.addPoint({ x: 60, y: 10 });
  assert.strictEqual(result.status, "committed");
  const after = flow.commitGeometry(result.geometry);
  assert.strictEqual(after, null);
  assert.strictEqual(flow.state.stepKey, "open_boundary");
  assert.strictEqual(flow.geometry.mask, undefined, "可选遮罩不进入推进判定");
}

// -- TST：普通多边形第三点不自行结束，点“完成”才推进 ------------------
function tstFlow() {
  const flow = makeFlow("tst");
  flow.selectStep("arena");
  const trianglePoints = triangle([[10, 10], [90, 10], [50, 80]]);
  let committed = null;
  for (const point of trianglePoints) {
    const result = flow.editor.addPoint({ x: point[0], y: point[1] });
    assert.strictEqual(result.status, "pending", "第三点后仍应等待“完成”");
    assert.ok(!result.geometry);
    committed = result.geometry || committed;
  }
  assert.strictEqual(committed, null);
  assert.strictEqual(flow.editor.canFinish(), true, "三点后完成按钮可用");
  assert.strictEqual(flow.state.stepKey, "arena", "未点完成前不换步骤");
  assert.strictEqual(flow.geometry.arena, undefined, "未完成不写草稿");

  const next = finishStep(flow);
  assert.ok(next, "点完成后必须推进");
  assert.strictEqual(next.key, "body_line");
  assert.ok(stepShape("tst", flow.geometry, "arena"));

  // 内径线两点自动提交，之后没有必需步骤（遮罩可选）。
  const line = flow.editor.addPoint({ x: 20, y: 20 });
  assert.strictEqual(line.status, "pending");
  const second = flow.editor.addPoint({ x: 30, y: 20 });
  assert.strictEqual(second.status, "committed");
  const after = flow.commitGeometry(second.geometry);
  assert.strictEqual(after, null);
  assert.strictEqual(flow.state.stepKey, "body_line");
}

// -- Y 迷宫：固定三点自动结束，随即推进到 A 臂 ------------------------
function ymazeFlow() {
  const flow = makeFlow("ymaze");
  flow.selectStep("center");
  const points = [[40, 40], [60, 40], [50, 60]];
  let next = null;
  points.forEach((point, index) => {
    const result = flow.editor.addPoint({ x: point[0], y: point[1] });
    if (index < 2) {
      assert.strictEqual(result.status, "pending", "固定三点未满不提交");
    } else {
      assert.strictEqual(result.status, "committed", "第三点自动提交");
      next = flow.commitGeometry(result.geometry);
    }
  });
  assert.ok(next);
  assert.strictEqual(next.key, "A");
  assert.ok(flow.geometry.zones.center);
}

// -- OFT：只有必需场地四角时画完不自动分析，也不被可选遮罩挡住 --------
function oftFlow() {
  const flow = makeFlow("oft");
  flow.selectStep("arena");
  const next = addPoints(flow, quad(10, 10, 50));
  assert.strictEqual(next, null, "四角完成后没有下一个必需步骤");
  assert.strictEqual(flow.state.stepKey, "arena");
  assert.ok(stepShape("oft", flow.geometry, "arena"));

  // 可选遮罩：画它不改变当前步骤，也不会被它挡住已有推进。
  flow.selectStep("mask");
  triangle([[1, 1], [9, 1], [5, 9]]).forEach((point) => {
    flow.editor.addPoint({ x: point[0], y: point[1] });
  });
  assert.strictEqual(flow.editor.canFinish(), true);
  finishStep(flow);
  assert.ok(stepShape("oft", flow.geometry, "mask"));
  assert.strictEqual(flow.state.stepKey, "mask");
  assert.strictEqual(nextRequiredStep(stepsFor("oft"), (key) =>
    Boolean(stepShape("oft", flow.geometry, key))), null);
}

// -- Barnes：指定数量洞心画满才提交，随即推进 ------------------------
function barnesFlow() {
  const flow = makeFlow("barnes", { hole_count: 4 });
  flow.selectStep("arena");
  const arena = flow.editor.commitEllipse({ x1: 10, y1: 10, x2: 190, y2: 190 });
  assert.ok(arena);
  const next = flow.commitGeometry(arena);
  assert.ok(next, "圆盘外缘完成后必须换步骤");
  assert.strictEqual(next.key, "holes");

  const centers = [[100, 40], [160, 100], [100, 160], [40, 100]];
  let committed = null;
  centers.forEach((point, index) => {
    const result = flow.editor.addPoint({ x: point[0], y: point[1] });
    if (index < centers.length - 1) {
      assert.strictEqual(result.status, "pending", "洞数不足不提交");
    } else {
      assert.strictEqual(result.status, "committed", "洞数画满即提交");
      committed = result.geometry;
    }
  });
  assert.ok(committed);
  const after = flow.commitGeometry(committed);
  assert.strictEqual(after, null, "洞位完成后没有剩余必需步骤");
  assert.deepStrictEqual(flow.geometry.holes.map((hole) => hole.id),
    ["1", "2", "3", "4"]);
  assert.strictEqual(flow.state.stepKey, "holes");
}

// -- EZM：足够边缘点拟合椭圆，点“完成”才推进 --------------------------
function ellipsePointsFlow() {
  const flow = makeFlow("ezm");
  flow.selectStep("arena");
  assert.strictEqual(flow.editor.setMode("points"), true);
  const edge = [[100, 10], [160, 30], [190, 100], [160, 170],
                [100, 190], [30, 100]];
  edge.forEach((point, index) => {
    const result = flow.editor.addPoint({ x: point[0], y: point[1] });
    assert.strictEqual(result.status, "pending", "边缘点不自动提交");
    if (index === edge.length - 1) {
      assert.strictEqual(result.ready, true);
    }
  });
  assert.strictEqual(flow.state.stepKey, "arena");
  const next = finishStep(flow);
  assert.ok(next, "边缘点拟合椭圆后必须推进");
  assert.strictEqual(next.key, "open_boundary");
  assert.strictEqual(flow.geometry.arena.type, "ellipse");
  assert.strictEqual(flow.geometry.arena.points.length, edge.length);
}

// -- 负例：无效几何与撤销都不算完成 -----------------------------------
function negativeCases() {
  // 过小的椭圆不得提交，也不换步骤。
  const ezm = makeFlow("ezm");
  ezm.selectStep("arena");
  assert.strictEqual(ezm.editor.commitEllipse({ x1: 10, y1: 10, x2: 10.5, y2: 10.5 }),
    null, "过小椭圆不得提交");
  assert.strictEqual(ezm.state.stepKey, "arena");
  assert.strictEqual(ezm.geometry.arena, undefined);

  // 撤销已提交几何：步骤回到未绘制，下一次提交仍按同一规则推进。
  const tst = makeFlow("tst");
  tst.selectStep("arena");
  tst.editor.addPoint({ x: 10, y: 10 });
  tst.editor.addPoint({ x: 90, y: 10 });
  tst.editor.addPoint({ x: 50, y: 80 });
  tst.commitGeometry(tst.editor.finish());
  assert.strictEqual(tst.state.stepKey, "body_line");
  tst.selectStep("arena");
  const undone = tst.editor.undo();
  assert.strictEqual(undone, "geometry");
  stepClear("tst", tst.geometry, "arena");
  assert.strictEqual(tst.editor.committed, null);
  assert.strictEqual(stepShape("tst", tst.geometry, "arena"), null);
  assert.strictEqual(nextRequiredStep(stepsFor("tst"), (key) =>
    Boolean(stepShape("tst", tst.geometry, key))).key, "arena",
  "撤销后这一步重新成为待完成步骤");
}

epmFlow();
ezmFlow();
tstFlow();
ymazeFlow();
oftFlow();
barnesFlow();
ellipsePointsFlow();
negativeCases();
console.log("step-advance OK");
