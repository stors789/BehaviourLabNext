"use strict";

const assert = require("assert");
const {
  saveThenPreview,
  previewNoteFor,
} = require("../../behaviourlab/web/static/editor-core.js");

const CONFIG = {
  kind: "detection",
  asset_id: "asset-1",
  frame_index: 3,
  unit: { id: "unit-1", detection: { gray_threshold: 50 } },
};

async function saveRunsFirst() {
  const calls = [];
  let dirty = true;
  const result = await saveThenPreview({
    dirty: () => dirty,
    save: async () => {
      calls.push("PUT 50");
      dirty = false;
      return true;
    },
    alive: () => true,
    config: () => CONFIG,
    send: async (config) => {
      calls.push("POST preview");
      assert.strictEqual(config.unit.detection.gray_threshold, 50,
        "POST 必须携带已保存的 50");
      return { image_url: "/api/previews/x.png", summary: {} };
    },
  });
  assert.deepStrictEqual(calls, ["PUT 50", "POST preview"], "必须先保存再预览");
  assert.strictEqual(result.status, "ok");
  assert.strictEqual(result.data.image_url, "/api/previews/x.png");
}

async function saveFailureStopsPreview() {
  const calls = [];
  const result = await saveThenPreview({
    dirty: () => true,
    save: async () => {
      calls.push("PUT");
      return false;
    },
    alive: () => true,
    config: () => CONFIG,
    send: async () => {
      calls.push("POST");
      return {};
    },
  });
  assert.strictEqual(result.status, "save_failed");
  assert.deepStrictEqual(calls, ["PUT"], "保存失败不得发起预览");
}

async function stalePageStopsPreview() {
  const calls = [];
  const result = await saveThenPreview({
    dirty: () => true,
    save: async () => {
      calls.push("PUT");
      return true;
    },
    alive: () => false,
    config: () => CONFIG,
    send: async () => {
      calls.push("POST");
      return {};
    },
  });
  assert.strictEqual(result.status, "stale");
  assert.deepStrictEqual(calls, ["PUT"], "页面已换小鼠时不发起预览");
}

async function cleanStateSkipsSave() {
  const calls = [];
  const result = await saveThenPreview({
    dirty: () => false,
    save: async () => {
      calls.push("PUT");
      return true;
    },
    alive: () => true,
    config: () => CONFIG,
    send: async () => {
      calls.push("POST");
      return { image_url: "/y.png" };
    },
  });
  assert.strictEqual(result.status, "ok");
  assert.deepStrictEqual(calls, ["POST"], "没有改动时不多发 PUT");
}

function notes() {
  const saturated = previewNoteFor({
    summary: {
      detection_available: true,
      detection: { candidates: 0, missing_reason: "foreground_saturated" },
    },
    warnings: ["前景覆盖过高，已拒绝整片掩膜。"],
  });
  assert.strictEqual(saturated.tone, "warn");
  assert.ok(saturated.text.includes("前景饱和"), saturated.text);
  assert.ok(saturated.text.includes("前景覆盖过高"), "必须显示后端 warnings");

  const empty = previewNoteFor({
    summary: {
      detection_available: true,
      detection: { candidates: 0, missing_reason: "no_candidate" },
    },
    warnings: ["没有符合面积与质量要求的候选。"],
  });
  assert.strictEqual(empty.tone, "warn");
  assert.ok(empty.text.includes("无候选"), empty.text);
  assert.ok(empty.text.includes("没有符合面积与质量要求的候选。"), empty.text);

  const manual = previewNoteFor({
    summary: { detection_available: false, detection: null },
    warnings: ["人工记录路径不运行自动检测。"],
  });
  assert.strictEqual(manual.tone, "muted");
  assert.ok(manual.text.includes("未运行"), manual.text);

  const found = previewNoteFor({
    summary: {
      detection_available: true,
      detection: { candidates: 2, missing_reason: null },
    },
    warnings: [],
  });
  assert.strictEqual(found.tone, "ok");
  assert.ok(found.text.includes("候选 2 个"), found.text);

  const sized = previewNoteFor({
    summary: {
      detection_available: true,
      detection: { candidates: 1, missing_reason: "inconsistent_candidate_size" },
    },
    warnings: [],
  });
  assert.strictEqual(sized.tone, "warn");
  assert.ok(sized.text.includes("体型不一致"), sized.text);
}

(async () => {
  await saveRunsFirst();
  await saveFailureStopsPreview();
  await stalePageStopsPreview();
  await cleanStateSkipsSave();
  notes();
  console.log("workspace-preview OK");
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
