// Barnes 场地编辑：圆盘椭圆 → 按洞数标洞，落盘是 geometry.holes 数组。
const assert = require("assert");

const {
  EditorCore,
  resolveStep,
  stepShape,
  stepApply,
  rotateGeometry,
  moveHole,
} = require("../../behaviourlab/web/static/editor-core.js");

// 与 experiments/barnes.py SPEC.geometry 对应；Python 侧守住同一顺序。
const STEPS = [
  { key: "arena", type: "ellipse", min_points: 5 },
  { key: "holes", type: "holes", count_key: "hole_count" },
  { key: "mask", type: "polygon", min_points: 3, optional: true },
];

const options = { hole_count: 3 };
const geometry = {};

// 1. 拖出圆盘：中心 (300,250)、半轴 100。
const arenaStep = resolveStep(STEPS[0], options);
assert.deepStrictEqual(arenaStep.modes, ["drag", "points"]);
const arenaEditor = new EditorCore();
arenaEditor.reset(arenaStep, stepShape("barnes", geometry, "arena"));
assert.strictEqual(arenaEditor.isEllipseDrag(), true);
const arena = arenaEditor.commitEllipse({ x1: 200, y1: 150, x2: 400, y2: 350 });
assert(arena && arena.type === "ellipse");
stepApply("barnes", geometry, "arena", arena);
assert.deepStrictEqual(geometry.arena.radii, [100, 100]);

// 2. 洞数来自参数：恰好三个点才提交，洞号按顺序 1…N。
const holeStep = resolveStep(STEPS[1], options);
assert.strictEqual(holeStep.count, 3);
const holeEditor = new EditorCore();
holeEditor.reset(holeStep, stepShape("barnes", geometry, "holes"));
const centers = [[420, 250], [300, 370], [180, 250]];
let holes = null;
centers.forEach((point) => {
  const result = holeEditor.addPoint({ x: point[0], y: point[1] });
  if (result.status === "committed") holes = result.geometry;
});
assert(holes && holes.type === "holes");
stepApply("barnes", geometry, "holes", holes);
assert(Array.isArray(geometry.holes));
assert.deepStrictEqual(geometry.holes.map((hole) => hole.id), ["1", "2", "3"]);
assert.deepStrictEqual(geometry.holes[0].center, [420, 250]);

// 3. 读回步骤形状：数组 ↔ 带 type 的画布形状，不改落盘结构。
const shape = stepShape("barnes", geometry, "holes");
assert.strictEqual(shape.type, "holes");
assert.strictEqual(shape.holes.length, 3);

// 4. 拖动一个洞：仍写回数组，其他洞顺序不变。
stepApply("barnes", geometry, "holes",
  moveHole(shape, "2", { x: 300, y: 390 }));
assert.deepStrictEqual(geometry.holes[1].center, [300, 390]);
assert.deepStrictEqual(geometry.holes.map((hole) => hole.id), ["1", "2", "3"]);

// 5. 旋转 90°：洞心随画面旋转，仍是数组。
const rotated = rotateGeometry(geometry, 640, 480, 0, 90);
assert(Array.isArray(rotated.holes));
assert.strictEqual(rotated.holes.length, 3);
assert.deepStrictEqual(rotated.holes[0].center, [480 - 1 - 250, 420]);
assert.deepStrictEqual(rotated.holes[0].id, "1");

// 6. 撤销洞位等于删键，不留空数组占位。
const cleared = JSON.parse(JSON.stringify(geometry));
stepApply("barnes", cleared, "holes", null);
assert.strictEqual("holes" in cleared, false);
assert.strictEqual("arena" in cleared, true);

console.log("barnes-editor OK");
console.log("barnes-geometry " + JSON.stringify(geometry));
