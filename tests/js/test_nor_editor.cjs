// NOR 场地编辑：四角 → 物体中心 ×2 → 可选探索范围，落盘只有 geometry.objects。
const assert = require("assert");

const {
  EditorCore,
  resolveStep,
  norObjects,
  norShape,
  norApply,
  norClear,
  norSetRole,
  norNormalizeRoles,
  moveVertex,
} = require("../../behaviourlab/web/static/editor-core.js");

// 与 experiments/nor.py SPEC.geometry 对应；Python 侧测试守住同一顺序。
const STEPS = [
  { key: "arena", type: "quad", count: 4 },
  { key: "object1", type: "point", count: 1, role: "object" },
  { key: "object2", type: "point", count: 1, role: "object" },
  { key: "explore1", type: "polygon", min_points: 3, optional: true },
  { key: "explore2", type: "polygon", min_points: 3, optional: true },
  { key: "mask", type: "polygon", min_points: 3, optional: true },
];

function draw(geometry, key, points) {
  const step = STEPS.find((item) => item.key === key);
  const spec = resolveStep(step, {});
  const editor = new EditorCore();
  editor.reset(spec, norShape(geometry, key));
  let committed = null;
  const states = [];
  for (const point of points) {
    const result = editor.addPoint({ x: point[0], y: point[1] });
    states.push(result.status);
    if (result.status === "committed") committed = result.geometry;
  }
  if (!committed && editor.canFinish()) committed = editor.finish();
  norApply(geometry, key, committed);
  return { committed, states, editor };
}

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function objectFlow() {
  const geometry = {};
  draw(geometry, "arena", [[20, 20], [220, 20], [220, 220], [20, 220]]);
  assert.deepStrictEqual(Object.keys(geometry), ["arena"]);

  // 物体 1：单击一次即完成，且只写入 objects[0].center。
  const first = draw(geometry, "object1", [[120, 120]]);
  assert.deepStrictEqual(first.states, ["committed"]);
  assert.ok(norShape(geometry, "object1"));
  assert.strictEqual(norShape(geometry, "object2"), null);
  assert.deepStrictEqual(norObjects(geometry).map((item) => item.center),
    [[120, 120], null]);

  draw(geometry, "object2", [[200, 120]]);
  assert.deepStrictEqual(norObjects(geometry).map((item) => item.center),
    [[120, 120], [200, 120]]);

  // 探索范围：两点不提交，三点后显式完成才落盘。
  const partial = draw(geometry, "explore1", [[100, 100], [140, 100]]);
  assert.deepStrictEqual(partial.states, ["pending", "pending"]);
  assert.strictEqual(partial.committed, null);
  assert.strictEqual(norShape(geometry, "explore1"), null);
  const zone = draw(geometry, "explore1",
    [[100, 100], [140, 100], [140, 140], [100, 140]]);
  assert.strictEqual(zone.committed.type, "polygon");
  assert.strictEqual(zone.committed.points.length, 4);

  // 刷新等价的序列化往返后结构与角色保持，且没有平行步骤键。
  const reloaded = clone(geometry);
  assert.deepStrictEqual(Object.keys(reloaded).sort(), ["arena", "objects"]);
  const objects = norObjects(reloaded);
  assert.strictEqual(objects.length, 2);
  assert.deepStrictEqual(objects.map((item) => [item.id, item.name, item.role]),
    [["object_1", "物体 1", "familiar"], ["object_2", "物体 2", "novel"]]);
  assert.deepStrictEqual(objects.map((item) => item.center),
    [[120, 120], [200, 120]]);
  assert.strictEqual(objects[0].zone.points.length, 4);
  assert.strictEqual(objects[1].zone, null);

  // 拖点只改中心，不新建步骤键。
  const moved = norShape(reloaded, "object1");
  norApply(reloaded, "object1", moveVertex(moved, 0, { x: 125, y: 118 }));
  assert.deepStrictEqual(norObjects(reloaded)[0].center, [125, 118]);
  assert.deepStrictEqual(Object.keys(reloaded).sort(), ["arena", "objects"]);

  // 撤销/重画清掉单个中心或范围，其他字段保留。
  const cleared = clone(geometry);
  norClear(cleared, "object2");
  assert.strictEqual(norObjects(cleared)[1].center, null);
  assert.deepStrictEqual(norObjects(cleared)[1].role, "novel");
  norClear(cleared, "explore1");
  assert.strictEqual(norObjects(cleared)[0].zone, null);
  assert.deepStrictEqual(norObjects(cleared)[0].center, [120, 120]);
  return geometry;
}

function roleSwitching() {
  const geometry = {};
  draw(geometry, "object1", [[120, 120]]);
  draw(geometry, "object2", [[200, 120]]);

  // 识别期：改一个角色同时把另一项设成相反角色。
  norSetRole(geometry, 1, "novel", "recognition");
  assert.deepStrictEqual(norObjects(geometry).map((item) => item.role),
    ["familiar", "novel"]);
  norSetRole(geometry, 0, "novel", "recognition");
  assert.deepStrictEqual(norObjects(geometry).map((item) => item.role),
    ["novel", "familiar"]);

  // 熟悉期不强制对调。
  const familiar = clone(geometry);
  norSetRole(familiar, 0, "familiar", "familiarization");
  assert.deepStrictEqual(norObjects(familiar).map((item) => item.role),
    ["familiar", "familiar"]);

  // 切回识别期时归一，保证恰好一新奇一熟悉。
  norNormalizeRoles(familiar, "recognition");
  assert.deepStrictEqual(norObjects(familiar).map((item) => item.role),
    ["familiar", "novel"]);
  norNormalizeRoles(familiar, "familiarization");
  assert.deepStrictEqual(norObjects(familiar).map((item) => item.role),
    ["familiar", "novel"]);
}

function unitsStaySeparate() {
  const first = {};
  const second = {};
  draw(first, "object1", [[120, 120]]);
  assert.ok(norShape(first, "object1"));
  // 另一只动物/视频没有画过，不能借用上一个单位的对象。
  assert.strictEqual(norShape(second, "object1"), null);
  assert.strictEqual(norShape(second, "object2"), null);
  assert.deepStrictEqual(norObjects(second).map((item) => item.center),
    [null, null]);
}

function undoKeepsOtherFields() {
  const geometry = {};
  draw(geometry, "arena", [[20, 20], [220, 20], [220, 220], [20, 220]]);
  draw(geometry, "object1", [[120, 120]]);
  draw(geometry, "object2", [[200, 120]]);
  draw(geometry, "explore1", [[100, 100], [140, 100], [140, 140], [100, 140]]);
  norSetRole(geometry, 0, "novel", "recognition");

  // 撤销物体 1 中心：只清该中心，角色/另一对象/探索范围都保留。
  norClear(geometry, "object1");
  assert.strictEqual(norShape(geometry, "object1"), null);
  let objects = norObjects(geometry);
  assert.strictEqual(objects[0].center, null);
  assert.strictEqual(objects[0].role, "novel");
  assert.ok(objects[0].zone);
  assert.deepStrictEqual(objects[1].center, [200, 120]);
  assert.ok(objects[1].zone === null);

  // 重新单击中心即可恢复，其他字段不变。
  draw(geometry, "object1", [[118, 116]]);
  objects = norObjects(geometry);
  assert.deepStrictEqual(objects[0].center, [118, 116]);
  assert.strictEqual(objects[0].role, "novel");
  assert.strictEqual(objects[0].zone.points.length, 4);
  assert.deepStrictEqual(objects[1].center, [200, 120]);
  assert.deepStrictEqual(Object.keys(geometry).sort(), ["arena", "objects"]);
}

const sample = objectFlow();
roleSwitching();
unitsStaySeparate();
undoKeepsOtherFields();
// 供 Python 侧核对：编辑产出的配置必须能被后端完整校验。
console.log("nor-geometry " + JSON.stringify(sample));
console.log("nor-editor OK");
