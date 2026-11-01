"use strict";

const assert = require("assert");
const {
  saveThenPreview,
  previewNoteFor,
  issueText,
  setFieldLabels,
  mwmLines,
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

function fieldReasonsWin() {
  const width = issueText({
    message: "草稿未通过校验",
    fields: [
      { field: "videos[0].units[0].options.arena_width_cm",
        message: "场地宽度必须 > 0" },
      { field: "videos[0].units[0].options.bin_seconds",
        message: "时间箱长度必须 > 0" },
    ],
  }, "保存失败，未开始分析");
  assert.strictEqual(width, "场地宽度必须 > 0；时间箱长度必须 > 0",
    "字段原因不能被通用提示覆盖");

  const fieldOnly = issueText({
    message: "草稿未通过校验",
    fields: [{ field: "videos[0].units[0].geometry.arena" }],
  }, "保存失败");
  assert.strictEqual(fieldOnly, "videos[0].units[0].geometry.arena",
    "没有 message 时退回字段名");

  const plain = issueText({ message: "服务不可用" }, "保存失败");
  assert.strictEqual(plain, "服务不可用");

  const fallback = issueText(new Error(""), "保存失败，未生成预览");
  assert.strictEqual(fallback, "保存失败，未生成预览");
}

function fieldNamesFromDescriptors() {
  // 界面用参数描述符的 label 登记可读名称，错误提示不再只给开发字段。
  setFieldLabels({
    center_size_cm: "中心区边长",
    bin_seconds: "时间箱长度",
    arena_width_cm: "场地宽度",
  });

  const zero = issueText({
    message: "草稿未通过校验",
    fields: [{ field: "videos[0].units[0].options.center_size_cm",
      message: "必须大于 0" }],
  }, "保存失败");
  assert.strictEqual(zero, "中心区边长：必须大于 0", "必须补上可读字段名");
  assert.ok(zero.indexOf("options.") < 0, "不抛裸开发字段");

  const named = issueText({
    fields: [{ field: "videos[0].units[0].options.arena_width_cm",
      message: "场地宽度必须 > 0" }],
  }, "保存失败");
  assert.strictEqual(named, "场地宽度必须 > 0", "名称已在文案里时不重复前缀");

  const many = issueText({
    fields: [
      { field: "videos[0].units[0].options.center_size_cm", message: "必须大于 0" },
      { field: "videos[0].units[0].options.bin_seconds", message: "必须大于 0" },
    ],
  }, "保存失败");
  assert.strictEqual(many, "中心区边长：必须大于 0；时间箱长度：必须大于 0");

  const unknown = issueText({
    fields: [{ field: "videos[0].units[0].mystery", message: "必须是对象" }],
  }, "保存失败");
  assert.strictEqual(unknown, "必须是对象", "缺少名称时仍给原因");

  setFieldLabels(null);
  assert.strictEqual(issueText({
    fields: [{ field: "videos[0].units[0].options.center_size_cm",
      message: "必须大于 0" }],
  }, "保存失败"), "必须大于 0", "未登记名称时退回原有文案");
}

function behaviorNote() {
  const found = previewNoteFor({
    summary: {
      detection_available: true,
      detection: { candidates: 1, missing_reason: null },
      behavior: {
        total_time_s: 3.5, immobility_time_s: 3.4,
        struggling_time_s: 0.0, uncertain_time_s: 0.1,
      },
    },
    warnings: [],
  });
  assert.ok(found.text.includes("行为 3.50 s"), found.text);
  assert.ok(found.text.includes("非挣扎 3.40"), found.text);
  assert.ok(found.text.includes("不确定 0.10"), found.text);

  const oft = previewNoteFor({
    summary: {
      detection_available: true,
      detection: { candidates: 1, missing_reason: null },
      behavior: {
        mode: "oft", total_time_s: 4.0, center_percent: 12.5,
        center_entries: 2, total_distance_cm: 130.25,
        mean_speed_cm_s: 3.5,
      },
    },
    warnings: [],
  });
  assert.ok(oft.text.includes("中心 12.50%"), oft.text);
  assert.ok(oft.text.includes("进入 2 次"), oft.text);
  assert.ok(oft.text.includes("路程 130.25 cm"), oft.text);
  assert.ok(!oft.text.includes("非挣扎"), "OFT 不套用 TST 标签");

  const fst = previewNoteFor({
    summary: {
      detection_available: true,
      detection: { candidates: 1, missing_reason: null },
      behavior: {
        mode: "fst", total_time_s: 3.0, immobility_time_s: 1.2,
        active_time_s: 1.5, uncertain_time_s: 0.3,
      },
    },
    warnings: [],
  });
  assert.ok(fst.text.includes("不动 1.20"), fst.text);
  assert.ok(fst.text.includes("活动 1.50"), fst.text);
  assert.ok(!fst.text.includes("非挣扎"), "FST 不套用 TST 标签");

  const mwm = previewNoteFor({
    summary: {
      detection_available: true,
      detection: { candidates: 1, missing_reason: null },
      behavior: {
        mode: "mwm", total_time_s: 4.0, platform_entries: 1,
        platform_time_s: 2.0, total_distance_cm: 30.0,
      },
    },
    warnings: [],
  });
  assert.ok(mwm.text.includes("进入 1 次"), mwm.text);
  assert.ok(mwm.text.includes("平台 2.00 s"), mwm.text);
  assert.ok(mwm.text.includes("路径 30.00 cm"), mwm.text);
  assert.ok(!mwm.text.includes("不动 "), "MWM 不套用 FST 标签");

  const nor = previewNoteFor({
    summary: {
      detection_available: true,
      detection: { candidates: 1, missing_reason: null },
      behavior: {
        mode: "nor", total_time_s: 4.0, total_object_exploration_s: 2.5,
        object_1_exploration_s: 1.0, object_2_exploration_s: 1.5,
        object_1_zone_time_s: 1.5, object_2_zone_time_s: 1.5,
      },
    },
    warnings: [],
  });
  assert.ok(nor.text.includes("探索 2.50 s"), nor.text);
  assert.ok(nor.text.includes("1.00 / 1.50 s"), nor.text);
  assert.ok(nor.text.includes("接近 3.00 s"), nor.text);
  assert.ok(!nor.text.includes("平台"), "NOR 不套用 MWM 标签");

  const epm = previewNoteFor({
    summary: {
      detection_available: true,
      detection: { candidates: 1, missing_reason: null },
      behavior: {
        mode: "epm", total_time_s: 0.2, open_time_s: 0.0,
        closed_time_s: 0.1, center_time_s: 0.1,
        total_arm_entries: 1, open_entries: 0,
      },
    },
    warnings: [],
  });
  assert.ok(epm.text.includes("入臂 1 次（开放 0）"), epm.text);

  const epmZero = previewNoteFor({
    summary: {
      detection_available: true,
      detection: { candidates: 1, missing_reason: null },
      behavior: {
        mode: "epm", total_time_s: 0.2, open_time_s: 0.0,
        closed_time_s: 0.0, center_time_s: 0.2,
        total_arm_entries: 0, open_entries: 0,
      },
    },
    warnings: [],
  });
  assert.ok(epmZero.text.includes("入臂 0 次（开放 0）"), epmZero.text);

  const epmMissing = previewNoteFor({
    summary: {
      detection_available: true,
      detection: { candidates: 1, missing_reason: null },
      behavior: { mode: "epm", total_time_s: 0.2, open_time_s: 0.1 },
    },
    warnings: [],
  });
  assert.ok(epmMissing.text.includes("入臂 — 次"), "缺字段仍显示 —");
}

function mwmQuadrantGuides() {
  const arena = { type: "ellipse", center: [100, 100], radii: [60, 30], angle: 30 };
  const guides = mwmLines(arena, 120, 25);
  assert.ok(guides, "椭圆与旋转必须产出象限方向");
  const expected = [
    [0.8870492613930958, 0.461674785820022],
    [0.343250149967973, -0.9392440228965868],
  ];
  guides.directions.forEach(([x, y], index) => {
    assert.ok(Math.abs(x - expected[index][0]) < 1e-9, "x 方向与物理变换一致");
    assert.ok(Math.abs(y - expected[index][1]) < 1e-9, "y 方向与物理变换一致");
  });
  assert.strictEqual(mwmLines({ type: "line", points: [[0, 0], [1, 0]] }, 120, 0),
    null, "非椭圆不画象限线");
}

async function saveFailureCarriesFieldReason() {
  const widthError = new Error("草稿未通过校验");
  widthError.fields = [{
    field: "videos[0].units[0].options.arena_width_cm",
    message: "场地宽度必须 > 0",
  }];
  let posts = 0;
  const result = await saveThenPreview({
    dirty: () => true,
    save: async () => false,
    alive: () => true,
    config: () => CONFIG,
    send: async () => {
      posts += 1;
      return {};
    },
  });
  assert.strictEqual(result.status, "save_failed");
  assert.strictEqual(posts, 0, "保存失败不得发起预览或开始分析");
  assert.strictEqual(issueText(widthError, "保存失败，未生成预览"),
    "场地宽度必须 > 0", "阻断提示必须带具体字段原因");
}

(async () => {
  await saveRunsFirst();
  await saveFailureStopsPreview();
  await stalePageStopsPreview();
  await cleanStateSkipsSave();
  notes();
  fieldReasonsWin();
  fieldNamesFromDescriptors();
  behaviorNote();
  mwmQuadrantGuides();
  await saveFailureCarriesFieldReason();
  console.log("workspace-preview OK");
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
