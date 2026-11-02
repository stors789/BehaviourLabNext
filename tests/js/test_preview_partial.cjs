/* 未知时长时的预览范围文案：真实 editor-core 的 previewNoteFor。
   只验证文案分支，不代表浏览器渲染。 */
"use strict";

const assert = require("assert");
const core = require("../../behaviourlab/web/static/editor-core.js");

const partial = {
  mode: "partial", partial: true, unknown_end: true,
  start_s: 0, end_s: 14.96, duration_s: 14.96,
};
const configured = { mode: "configured", start_s: 0, end_s: 20 };

function behaviorPayload(scope, warnings) {
  return {
    time_s: 14.96,
    preview_scope: scope,
    warnings: warnings || [],
    summary: {
      kind: "behavior",
      detection_available: true,
      detection: { candidates: 1, missing_reason: null },
      behavior: {
        mode: "oft", total_time_s: 14.96, center_time_s: 1.0,
        center_percent: 10.0, center_entries: 2, total_distance_cm: 5.0,
        mean_speed_cm_s: 1.0, observed_time_s: 14.96, missing_time_s: 0.0,
      },
    },
  };
}

// ① 部分范围：明确写出只到当前帧、完整终点未知。
const note = core.previewNoteFor(behaviorPayload(partial,
  ["视频有效时长未知：本次行为预览只扫描 0.00–14.96 s（截至当前源帧的部分范围，不是完整配置范围）。"]));
assert.ok(note.text.includes("部分范围 0.00–14.96 s"), note.text);
assert.ok(note.text.includes("完整终点未知"), note.text);
assert.ok(!note.text.includes("完整配置范围 0.00–14.96"), note.text);

// ② 完整配置范围仍是原话术（已知帧数主例不回退）。
const full = core.previewNoteFor(behaviorPayload(configured));
assert.ok(full.text.includes("完整配置范围 0.00–20.00 s"), full.text);

// ③ 单帧检测预览不受影响。
const single = core.previewNoteFor({
  time_s: 3.0,
  preview_scope: { mode: "frame", start_s: 3.0, end_s: 3.0 },
  summary: { kind: "detection", detection_available: true,
    detection: { candidates: 2, missing_reason: null } },
});
assert.ok(single.text.includes("单帧检测预览"), single.text);

// ④ 事实不足时有效时长为 null（工作台据此显示“总时长未知”）。
assert.strictEqual(core.effectiveVideoDuration({}, { fps_override: 12.5 }), null);
assert.strictEqual(
  core.effectiveVideoDuration({ frame_count: 250, fps: 25 }, { fps_override: 12.5 }),
  20);

console.log("preview-partial OK");
