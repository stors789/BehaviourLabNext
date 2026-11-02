/* 新指定窗默认终点：真实 editor-core 的有效时长与窗口提交逻辑。
   输出交给 pytest 写入真实草稿再读回。 */
"use strict";

const assert = require("assert");
const {
  effectiveVideoDuration,
  windowFieldValue,
  assignWindow,
} = require("../../behaviourlab/web/static/editor-core.js");

// 验收固定例：120 帧 / 原 30 FPS / 原时长 4 秒，解释 20 FPS → 6 秒。
const META = { frame_count: 120, fps: 30, duration_s: 4 };
const OVERRIDE = { fps_override: 20 };

function defaultEnd(meta, video, current, typed) {
  const duration = effectiveVideoDuration(meta, video);
  const saved = current && current.end_s != null ? current.end_s : null;
  return windowFieldValue(typed == null ? null : typed, saved, duration);
}

// ① 默认值：解释帧率优先，原 duration 不再抢先。
assert.strictEqual(defaultEnd(META, OVERRIDE, null, null), 6);

// ② 已保存的显式终点不被改写。
assert.strictEqual(
  defaultEnd(META, OVERRIDE, { mode: "exact", start_s: 1, end_s: 5.5 }, null),
  5.5);

// ③ 用户当前输入优先。
assert.strictEqual(
  defaultEnd(META, OVERRIDE, { mode: "exact", start_s: 1, end_s: 5.5 }, "7.25"),
  "7.25");

// ④ 无覆盖对照：仍按原容器时长。
assert.strictEqual(defaultEnd(META, { fps_override: null }, null, null), 4);

// ⑤ 原 duration 缺失但帧数仍在：规则与后端一致。
assert.strictEqual(
  defaultEnd({ frame_count: 120, fps: 30, duration_s: null }, OVERRIDE, null, null),
  6);

// ⑥ 事实不足：留空，而不是拿当前帧当片尾。
assert.strictEqual(
  defaultEnd({ frame_count: null, fps: null, duration_s: 4 }, OVERRIDE, null, null),
  null);
assert.strictEqual(
  windowFieldValue(null, null, null), null);

// ⑦ 真实提交：模式 change 后走同一个 assignWindow。
const fresh = assignWindow(null, "exact", {
  start_s: "1", end_s: String(defaultEnd(META, OVERRIDE, null, null)),
});
assert.strictEqual(fresh.error, null);
assert.deepStrictEqual(fresh.window, { mode: "exact", start_s: 1, end_s: 6 });

const kept = assignWindow(null, "exact", {
  start_s: "1",
  end_s: String(defaultEnd(META, OVERRIDE,
    { mode: "exact", start_s: 1, end_s: 5.5 }, null)),
});
assert.deepStrictEqual(kept.window, { mode: "exact", start_s: 1, end_s: 5.5 });

const typed = assignWindow(null, "exact", {
  start_s: "1",
  end_s: String(defaultEnd(META, OVERRIDE,
    { mode: "exact", start_s: 1, end_s: 5.5 }, "7.25")),
});
assert.deepStrictEqual(typed.window, { mode: "exact", start_s: 1, end_s: 7.25 });

const blank = assignWindow(null, "exact", {
  start_s: "1",
  end_s: String(defaultEnd({ frame_count: null, fps: null, duration_s: 4 },
    OVERRIDE, null, null)),
});
assert.ok(blank.error, "无有效时长且未填终点时必须报错，而不是伪造终点");

// ⑧ 切视频/切动物：同一函数对不同对象各自给值，不串用另一对象时长。
const other = effectiveVideoDuration(
  { frame_count: 300, fps: 25, duration_s: 12 }, { fps_override: 25 });
assert.strictEqual(other, 12, "另一对象按自己的帧数/解释帧率");
assert.strictEqual(defaultEnd(META, OVERRIDE, null, null), 6,
  "回到原对象仍是 6 秒");

// ⑨ 与后端同一规则的对照表（pytest 逐项与 detection.interpreted_duration 比对）。
const durationCases = [
  [{ frame_count: 120, fps: 30, duration_s: 4 }, { fps_override: 20 }],
  [{ frame_count: 120, fps: 30, duration_s: 4 }, { fps_override: null }],
  [{ frame_count: null, fps: 30, duration_s: 4 }, { fps_override: 20 }],
  [{ frame_count: null, fps: null, duration_s: 4 }, { fps_override: 20 }],
  [{ frame_count: 120, fps: null, duration_s: null }, { fps_override: 20 }],
  [{ frame_count: null, fps: 30, duration_s: null }, { fps_override: 20 }],
  [{ frame_count: 0, fps: null, duration_s: null }, { fps_override: 20 }],
  [{ duration_s: 10 }, { fps_override: 12.5 }],
  [{ frame_count: 250, fps: 25, duration_s: 10 }, { fps_override: 12.5 }],
].map(([meta, video]) => ({
  meta, video, value: effectiveVideoDuration(meta, video),
}));

console.log(JSON.stringify({
  fresh: fresh.window,
  kept: kept.window,
  typed: typed.window,
  blankError: blank.error,
  durationCases,
}));
