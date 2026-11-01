"use strict";

const assert = require("assert");
const {
  saveThenPreview,
  previewNoteFor,
  parseManualVisits,
  previewActionLabel,
  manualFormatIssue,
  issueText,
  setFieldLabels,
  mwmLines,
  normalizeWindow,
  assignWindow,
  reconcileVideos,
  selectionNeedsSave,
  persistSelection,
  previewContextAlive,
  removeUnit,
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
    closed_2: "封闭臂 2",
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

  const zone = issueText({
    fields: [{ field: "videos[0].units[0].geometry.zones.closed_2",
      message: "缺少封闭臂 2" }],
  }, "保存失败");
  assert.strictEqual(zone, "缺少封闭臂 2", "名称在句中也保持原样");

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

// Barnes：范围说明跟随真实 preview_scope，configured 不能说成片段。
function barnesScopeNotes() {
  const range = previewNoteFor({
    time_s: 12.5,
    preview_scope: { mode: "range", start_s: 11.5, end_s: 15.5 },
    summary: {
      kind: "behavior",
      detection_available: true,
      detection: { candidates: 1, missing_reason: null },
      behavior: {
        mode: "barnes", total_time_s: 3.0,
        window_start_s: 11.5, window_end_s: 14.5,
        visits: 2, primary_latency_s: 1.2, primary_errors: 1,
        primary_censored: false, nose_coverage_percent: 96.5,
        visit_source: "automatic",
      },
    },
    warnings: [],
  });
  assert.ok(range.text.includes("行为 11.50–14.50 s"), range.text);
  assert.ok(range.text.includes("源帧 12.50 s"), range.text);
  assert.ok(range.text.includes("当前片段 11.50–15.50 s"), range.text);
  assert.ok(range.text.includes("片段估计，非完整行为结果"), range.text);

  const configured = previewNoteFor({
    time_s: 7.25,
    preview_scope: { mode: "configured", start_s: 5, end_s: 10 },
    summary: {
      kind: "behavior",
      detection_available: true,
      detection: { candidates: 1, missing_reason: null },
      behavior: {
        mode: "barnes", total_time_s: 2.0,
        window_start_s: 5.99997, window_end_s: 7.99997,
        visits: 1, primary_latency_s: 0.5, primary_errors: 0,
        primary_censored: false, nose_coverage_percent: 100.0,
        visit_source: "manual",
      },
    },
    warnings: [],
  });
  assert.ok(configured.text.includes("行为 6.00–8.00 s"), configured.text);
  assert.ok(configured.text.includes("完整配置范围 5.00–10.00 s"), configured.text);
  assert.ok(configured.text.includes("源帧 7.25 s"), configured.text);
  assert.ok(!configured.text.includes("片段"),
    "configured 范围不得声明为片段：" + configured.text);
  assert.ok(configured.text.includes("人工记录"), configured.text);

  const detection = previewNoteFor({
    time_s: 3.0,
    preview_scope: { mode: "frame", start_s: 3, end_s: 3, frame_count: 1 },
    summary: {
      kind: "detection",
      detection_available: true,
      detection: { candidates: 1, missing_reason: null },
    },
    warnings: [],
  });
  assert.ok(detection.text.includes("源帧 3.00 s"), detection.text);
  assert.ok(detection.text.includes("单帧"), detection.text);
  assert.ok(!detection.text.includes("片段估计"), detection.text);
}

// Barnes 人工洞序列文本：中文逗号、空行、缺字段与非法时间。
function manualVisitsParsing() {
  const parsed = parseManualVisits("0.4，2\n1.2, 2\n\n2.4,1\n");
  assert.deepStrictEqual(parsed.entries, [
    { time_s: 0.4, hole_id: "2" },
    { time_s: 1.2, hole_id: "2" },
    { time_s: 2.4, hole_id: "1" },
  ]);
  assert.deepStrictEqual(parseManualVisits("   ").entries, [],
    "全空白输入表示零访问");
  assert.deepStrictEqual(parseManualVisits(null).entries, []);
  assert.ok(parseManualVisits("0.4").error.includes("第 1 行"));
  assert.ok(parseManualVisits("0.4,2,3").error.includes("第 1 行"));
  assert.ok(parseManualVisits(",2").error.includes("第 1 行"));
  assert.ok(parseManualVisits("0.4,").error.includes("第 1 行"));
  assert.ok(parseManualVisits("-1,2").error.includes("非负"));
  assert.ok(parseManualVisits("abc,2").error.includes("非负"));
  assert.ok(parseManualVisits("1,2\n0.5,1").entries.length === 2,
    "解析保留输入顺序，排序留给最终校验");
}

// 主按钮文字与 kind 同源；人工格式检查覆盖整份草稿的所有动物。
function previewActionContract() {
  assert.strictEqual(previewActionLabel("behavior"), "行为预览");
  assert.strictEqual(previewActionLabel("detection"), "检测预览");
  assert.strictEqual(previewActionLabel(undefined), "检测预览");

  const units = [
    { id: "a", name: "A 鼠", manual: { hole_visits: [] } },
    { id: "b", name: "B 鼠", manual: null },
    { id: "c", name: "C 鼠", manual: { hole_visits: null } },
  ];
  const cache = new Map([["a", { text: "0.4,", error: "第 1 行应为“视频秒,洞号”" }]]);
  const issue = manualFormatIssue(units, (unit) => cache.get(unit.id));
  assert.ok(issue, "任一动物有格式错误都要拦下整批提交");
  assert.strictEqual(issue.unitId, "a");
  assert.ok(issue.error.includes("第 1 行"), issue.error);
  assert.strictEqual(issue.unitName, "A 鼠");

  // 关闭人工模式（hole_visits=null）后不再阻止，缓存错误不复活。
  assert.strictEqual(manualFormatIssue(
    [{ id: "a", name: "A 鼠", manual: { hole_visits: null } }],
    () => ({ error: "旧错误" })), null);
  assert.strictEqual(manualFormatIssue(units, () => null), null);
  assert.strictEqual(manualFormatIssue([], () => null), null);
}

// 动物时间窗：global 写 null，exact 校验起止，relative 保留负偏移；
// 一次只写当前动物，另一只小鼠的窗口保持不变。
function animalWindowValues() {
  const first = { id: "a", window: null };
  const second = { id: "b", window: { mode: "exact", start_s: 1, end_s: 2 } };

  assert.deepStrictEqual(assignWindow(first, "global", {}),
    { window: null, error: null, field: null });
  assert.strictEqual(first.window, null);

  assert.deepStrictEqual(assignWindow(first, "exact", { start_s: "1.5", end_s: "4" }),
    { window: { mode: "exact", start_s: 1.5, end_s: 4 }, error: null, field: null });
  assert.deepStrictEqual(first.window, { mode: "exact", start_s: 1.5, end_s: 4 });
  assert.strictEqual(normalizeWindow("exact", { start_s: "", end_s: 4 }).field, "start_s");
  assert.strictEqual(normalizeWindow("exact", { start_s: 4, end_s: 4 }).field, "end_s");
  assert.strictEqual(normalizeWindow("exact", { start_s: -1, end_s: 4 }).field, "start_s");

  // 负偏移合法，持续时间必须为正：-1 / 3 不得被控件改成 0。
  const relative = assignWindow(first, "relative",
    { offset_s: "-1", duration_s: "3" });
  assert.deepStrictEqual(relative.window,
    { mode: "relative", offset_s: -1, duration_s: 3 });
  assert.strictEqual(first.window.offset_s, -1);
  assert.strictEqual(first.window.duration_s, 3);
  assert.strictEqual(normalizeWindow("relative",
    { offset_s: "", duration_s: 3 }).field, "offset_s");
  // 固定例：relative offset=-1、duration 改成 0；exact 起点清空。
  const badRelative = normalizeWindow("relative", { offset_s: -1, duration_s: 0 });
  assert.strictEqual(badRelative.window, null, "非法值不能当合法窗口");
  assert.strictEqual(badRelative.field, "duration_s");
  const blankStart = normalizeWindow("exact", { start_s: "", end_s: 4 });
  assert.strictEqual(blankStart.window, null, "清空的起点不能回填成 0");
  assert.strictEqual(blankStart.field, "start_s");
  assert.strictEqual(normalizeWindow("relative",
    { offset_s: 0, duration_s: 0 }).field, "duration_s");
  assert.deepStrictEqual(normalizeWindow("relative", { offset_s: 0, duration_s: 2 }),
    { window: { mode: "relative", offset_s: 0, duration_s: 2 }, error: null,
      field: null });

  // 校验失败不写任何动物；写入只影响传入的那一只。
  assert.ok(assignWindow(first, "exact", { start_s: 3, end_s: 1 }).error);
  assert.deepStrictEqual(first.window,
    { mode: "relative", offset_s: -1, duration_s: 3 });
  assert.deepStrictEqual(second.window, { mode: "exact", start_s: 1, end_s: 2 });
}

// 选择页勾选与草稿对齐：移除只退出本草稿的选择，保留原视频的完整配置；
// 新增视频按初始化规则补在末尾，不覆盖已有单位/几何/参数/人工记录。
function selectionSync() {
  const videoA = {
    asset_id: "a",
    window: { start_s: 1, end_s: 2 },
    units: [{ id: "u1", geometry: { arena: { type: "quad" } },
              options: { bin_seconds: 30 }, manual: { hole_visits: [] } }],
  };
  const videoB = { asset_id: "b", units: [] };
  const assets = [{ id: "a" }, { id: "b" }, { id: "c" }];
  const make = (asset) => ({
    asset_id: asset.id,
    window: { start_s: 0, end_s: null },
    units: [{ id: "new-" + asset.id }],
  });

  // 已有 A、B，移除 B：只留 A，且 A 的单位/几何/参数/人工记录原样保留。
  const removed = reconcileVideos([videoA, videoB], ["a"], assets, make);
  assert.deepStrictEqual(removed.map((video) => video.asset_id), ["a"]);
  assert.strictEqual(removed[0], videoA);
  assert.deepStrictEqual(removed[0].units[0].geometry, { arena: { type: "quad" } });
  assert.deepStrictEqual(removed[0].units[0].manual, { hole_visits: [] });

  // 已有 A，再选 B：A 不动，B 按初始化规则补在末尾。
  const added = reconcileVideos([videoA], ["a", "b"], assets, make);
  assert.deepStrictEqual(added.map((video) => video.asset_id), ["a", "b"]);
  assert.strictEqual(added[0], videoA);
  assert.deepStrictEqual(added[1].window, { start_s: 0, end_s: null });
  assert.deepStrictEqual(added[1].units, [{ id: "new-b" }]);

  // 重复选择不重复添加；空选择得到空列表。
  assert.deepStrictEqual(
    reconcileVideos([videoA], ["a", "a", "b"], assets, make)
      .map((video) => video.asset_id), ["a", "b"]);
  assert.deepStrictEqual(reconcileVideos([videoA, videoB], [], assets, make), []);
}

// 保存失败后的重试：第二次点击即使没有新差异也要再保存一次，
// 只有真的落盘（集合已是 A）才允许进入下一步。
async function selectionRetryAndPreviewGuard() {
  const draft = { videos: [{ asset_id: "a", units: [{ id: "u1" }] }] };
  const attempts = [];
  let fail = true;
  let pending = true; // 第一次点击已把选择改为 A，但保存失败

  const click = async () => {
    const outcome = await persistSelection({
      changed: false, // 第二次点击时集合已无新差异
      pending,
      dirty: () => true,
      save: async () => {
        attempts.push(draft.videos.map((video) => video.asset_id).join("+"));
        return !fail;
      },
    });
    if (outcome.status === "saved") pending = false;
    return outcome.status;
  };

  // 连续两次失败：两次都尝试，且都不放行。
  assert.strictEqual(await click(), "failed");
  assert.strictEqual(await click(), "failed");
  assert.strictEqual(attempts.length, 2, "第二次点击必须真的重试保存");
  assert.deepStrictEqual(attempts, ["a", "a"], "每次保存的集合都是 A");

  // 第一次失败、第二次成功：两次尝试，成功后放行。
  attempts.length = 0;
  fail = true;
  pending = true;
  assert.strictEqual(await click(), "failed");
  fail = false;
  assert.strictEqual(await click(), "saved");
  assert.strictEqual(attempts.length, 2);
  assert.strictEqual(pending, false, "成功后才清除未保存标记");

  // 没有改动、也不脏：不必保存就能继续。
  let calls = 0;
  const clean = await persistSelection({
    changed: false,
    pending: false,
    dirty: () => false,
    save: async () => { calls += 1; return true; },
  });
  assert.strictEqual(clean.status, "clean");
  assert.strictEqual(calls, 0);
  assert.strictEqual(selectionNeedsSave(false, true, false), true);
  assert.strictEqual(selectionNeedsSave(false, false, true), true);
  assert.strictEqual(selectionNeedsSave(false, false, false), false);

  // 预览有效期：等待保存/加载期间的新编辑或未解决错误都要作废旧流程。
  const start = { page: "p", assetId: "a", unitId: "u", edit: 7 };
  const same = { page: "p", assetId: "a", unitId: "u", edit: 7,
                 blocked: false, closed: false };
  assert.strictEqual(previewContextAlive(start, same), true);
  assert.strictEqual(previewContextAlive(start, { ...same, edit: 8 }), false);
  assert.strictEqual(previewContextAlive(start, { ...same, blocked: true }), false);
  assert.strictEqual(previewContextAlive(start, { ...same, closed: true }), false);
  assert.strictEqual(previewContextAlive(start, { ...same, assetId: "b" }), false);
  assert.strictEqual(previewContextAlive(start, { ...same, unitId: "v" }), false);
  assert.strictEqual(previewContextAlive(start, { ...same, page: "q" }), false);
}

// 删除当前小鼠：只去掉这一只，剩一只时没有删除入口，输入数组不被改写。
function unitRemoval() {
  const units = [{ id: "a" }, { id: "b" }, { id: "c" }];
  const plan = removeUnit(units, 1);
  assert.deepStrictEqual(plan.units.map((item) => item.id), ["a", "c"]);
  assert.strictEqual(plan.removed.id, "b");
  assert.strictEqual(plan.index, 1);
  assert.deepStrictEqual(units.map((item) => item.id), ["a", "b", "c"]);

  const last = removeUnit(units, 2);
  assert.deepStrictEqual(last.units.map((item) => item.id), ["a", "b"]);
  assert.strictEqual(last.index, 1, "删到末尾时选中前一只");

  assert.strictEqual(removeUnit([{ id: "a" }], 0), null, "最后一只不能删除");
  assert.strictEqual(removeUnit(units, 9), null);
  assert.strictEqual(removeUnit(null, 0), null);
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
  barnesScopeNotes();
  manualVisitsParsing();
  animalWindowValues();
  selectionSync();
  await selectionRetryAndPreviewGuard();
  unitRemoval();
  previewActionContract();
  await saveFailureCarriesFieldReason();
  console.log("workspace-preview OK");
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
