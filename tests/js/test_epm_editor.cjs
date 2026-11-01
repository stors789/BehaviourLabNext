// EPM 场地编辑：五区四角步骤写入 geometry.zones，遮罩留在顶层。
const assert = require("assert");

const {
  EditorCore,
  resolveStep,
  stepShape,
  stepApply,
  stepClear,
  norShape,
  norApply,
} = require("../../behaviourlab/web/static/editor-core.js");

const STEPS = [
  { key: "center", type: "quad", count: 4 },
  { key: "open_1", type: "quad", count: 4 },
  { key: "open_2", type: "quad", count: 4 },
  { key: "closed_1", type: "quad", count: 4 },
  { key: "closed_2", type: "quad", count: 4 },
  { key: "mask", type: "polygon", min_points: 3, optional: true },
];

function draw(assay, geometry, key, points) {
  const step = STEPS.find((item) => item.key === key);
  const spec = resolveStep(step, {});
  const editor = new EditorCore();
  editor.reset(spec, stepShape(assay, geometry, key));
  let committed = null;
  for (const point of points) {
    const result = editor.addPoint({ x: point[0], y: point[1] });
    if (result.status === "committed") committed = result.geometry;
  }
  if (!committed && editor.canFinish()) committed = editor.finish();
  stepApply(assay, geometry, key, committed);
}

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function zoneFlow() {
  const geometry = {};
  draw("epm", geometry, "center",
    [[90, 90], [110, 90], [110, 110], [90, 110]]);
  draw("epm", geometry, "open_1",
    [[111, 90], [190, 90], [190, 110], [111, 110]]);
  draw("epm", geometry, "open_2",
    [[10, 90], [89, 90], [89, 110], [10, 110]]);
  draw("epm", geometry, "closed_1",
    [[90, 10], [110, 10], [110, 89], [90, 89]]);
  draw("epm", geometry, "closed_2",
    [[90, 111], [110, 111], [110, 190], [90, 190]]);
  draw("epm", geometry, "mask",
    [[40, 40], [60, 40], [60, 60], [40, 60]]);

  assert.deepStrictEqual(Object.keys(geometry).sort(), ["mask", "zones"]);
  assert.deepStrictEqual(Object.keys(geometry.zones).sort(),
    ["center", "closed_1", "closed_2", "open_1", "open_2"]);
  for (const key of ["center", "open_1", "open_2", "closed_1", "closed_2"]) {
    assert.ok(stepShape("epm", geometry, key), key + " 已绘制");
  }

  // 撤销一个区域只清该键，其他区域与遮罩保留。
  stepClear("epm", geometry, "open_2");
  assert.strictEqual(stepShape("epm", geometry, "open_2"), null);
  assert.ok(stepShape("epm", geometry, "open_1"));
  assert.ok(stepShape("epm", geometry, "mask"));
  draw("epm", geometry, "open_2",
    [[10, 90], [89, 90], [89, 110], [10, 110]]);
  assert.ok(stepShape("epm", geometry, "open_2"));

  // 序列化往返保持结构。
  const reloaded = clone(geometry);
  assert.deepStrictEqual(Object.keys(reloaded.zones).sort(),
    ["center", "closed_1", "closed_2", "open_1", "open_2"]);
  assert.strictEqual(reloaded.mask.type, "polygon");

  // NOR 步骤不受 EPM 分支影响。
  const norGeometry = {};
  norApply(norGeometry, "object1", { type: "point", points: [[5, 5]] });
  assert.deepStrictEqual(stepShape("nor", norGeometry, "object1"),
    norShape(norGeometry, "object1"));
}

zoneFlow();
console.log("epm-editor OK");
