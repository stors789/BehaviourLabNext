/* 检测配置批量应用：真实 editor-core 的合并语义与保存队列保护。
   合并语义必须与后端 engine.merge_layers 一致（test_detection_sync.py
   用同一批分层用例逐值比对）。 */
"use strict";

const assert = require("assert");
const {
  cloneConfig,
  mergeDetectionLayers,
  effectiveDetection,
  SaveQueue,
} = require("../../behaviourlab/web/static/editor-core.js");

// -- 一、深合并语义（与 engine.merge_two 逐条对应） ---------------------

// 工厂 → 草稿 → 视频 → 鼠：后层覆盖同键，未显示字段保留前层值。
assert.deepStrictEqual(
  mergeDetectionLayers(
    { bg_method: "threshold", gray_threshold: 127, erosion_kernel: 3 },
    { gray_threshold: 100 },
    {},
    { erosion_kernel: 5 }),
  { bg_method: "threshold", gray_threshold: 100, erosion_kernel: 5 });

// null 不是"没填"：显式 null 整体替换，不能被前层值顶回。
assert.deepStrictEqual(
  mergeDetectionLayers({ static_background: "/a.png" },
                       { static_background: null }),
  { static_background: null });

// 数组与标量整体替换，不递归。
assert.deepStrictEqual(
  mergeDetectionLayers({ bins: [1, 2, 3] }, { bins: [9] }),
  { bins: [9] });

// 跨层嵌套对象逐键合并（模拟任何嵌套配置字段）。
assert.deepStrictEqual(
  mergeDetectionLayers(
    { nested: { a: 1, b: { c: 2 } } },
    { nested: { b: { d: 3 }, e: 4 } }),
  { nested: { a: 1, b: { c: 2, d: 3 }, e: 4 } });

// 后层 null 覆盖嵌套对象：整体替换，不抛错。
assert.deepStrictEqual(
  mergeDetectionLayers({ nested: { a: 1 } }, { nested: null }),
  { nested: null });

// 空/缺失层跳过；全空给空对象。
assert.deepStrictEqual(mergeDetectionLayers(null, undefined, {}), {});

// -- 二、独立副本 -------------------------------------------------------

const factory = { bg_method: "temporal", temporal_bg_samples: 90,
                  erosion_kernel: 3 };
const draft = { temporal_bg_samples: 60 };
const video = { min_area: 120 };
const unit = { min_area: 200, bg_method: "threshold" };
const effective = effectiveDetection(factory, draft, video, unit);

// 结果等于逐层覆盖的最终值。
assert.deepStrictEqual(effective, {
  bg_method: "threshold", temporal_bg_samples: 60,
  erosion_kernel: 3, min_area: 200 });

// 每只鼠独立副本：改副本不影响源层，也不影响其他副本。
const copyA = cloneConfig(effective);
const copyB = cloneConfig(effective);
copyA.min_area = 999;
copyA.erosion_kernel = 9;
assert.strictEqual(copyB.min_area, 200);
assert.strictEqual(copyB.erosion_kernel, 3);
assert.strictEqual(effective.min_area, 200);
assert.strictEqual(unit.min_area, 200);
assert.strictEqual(video.min_area, 120);

// 副本再按层解析仍是它自己：unit 层最优先。
assert.deepStrictEqual(
  effectiveDetection(factory, draft, video, copyA),
  { bg_method: "threshold", temporal_bg_samples: 60,
    erosion_kernel: 9, min_area: 999 });

// -- 三、保存队列：同步只走既有 markDirty→SaveQueue，迟到回包不改新编辑 --
// （场景即"同步后、保存进行中又单独调整另一只"的时序。）

function tick() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

async function queueChecks() {
  const sends = [];
  const echoes = [];
  const queue = new SaveQueue((snapshot) => new Promise((resolve, reject) => {
    sends.push({ snapshot, resolve, reject });
  }));

  queue.mark();                       // ① 同步写完各鼠 detection（编辑 v1）
  const first = queue.save(() => ({ videos: "快照-同步后" }),
    (draft) => echoes.push(draft));
  await tick();
  assert.strictEqual(sends.length, 1);
  assert.deepStrictEqual(sends[0].snapshot, { videos: "快照-同步后" });

  queue.mark();                       // ② 保存进行中又单独调整了一只（v2）
  // ②' markDirty 定时器触发：再次 save 只是更新快照工厂并复用同一次排空。
  const second = queue.save(() => ({ videos: "快照-再编辑" }),
    (draft) => echoes.push(draft));
  sends[0].resolve({ ok: true });     // ③ 旧回包迟到
  await tick();
  // 排空循环发现发送版本落后于新编辑：回包不合入，立即用最新快照重发。
  assert.strictEqual(echoes.length, 0, "迟到回包不覆盖新编辑");
  assert.strictEqual(sends.length, 2);
  assert.deepStrictEqual(sends[1].snapshot, { videos: "快照-再编辑" });

  sends[1].resolve({ ok: true });     // ④ 第二次发送也落盘，排空才结束
  assert.strictEqual(await first, true, "全部编辑落盘后才报告保存成功");
  assert.strictEqual(await second, true);
  // 只有 v2 的回包被合入；v1 的迟到回包被跳过。
  assert.deepStrictEqual(echoes, [{ ok: true }]);
  assert.strictEqual(queue.dirty(), false);

  // 保存失败：不误报成功，错误留在队列上等待重试。
  const failing = new SaveQueue(() => Promise.reject(new Error("500")));
  failing.mark();
  assert.strictEqual(await failing.save(() => ({ v: 1 }), () => {}), false);
  assert.strictEqual(failing.dirty(), true);
  assert.strictEqual(failing.lastError.message, "500");
}

queueChecks().then(() => {
  console.log("detection-sync OK");
  // 远程求值：Python 把真实目录默认与草稿分层投喂进来，取回合并结果，
  // 与 engine.resolve_detection 逐值比对（见 test_detection_sync.py）。
  if (process.argv[2]) {
    const job = JSON.parse(require("fs").readFileSync(process.argv[2], "utf8"));
    process.stdout.write(JSON.stringify({
      effective: effectiveDetection(job.defaults, job.draft, job.video,
                                    job.unit),
    }));
  }
}, (error) => { console.error(error); process.exit(1); });
