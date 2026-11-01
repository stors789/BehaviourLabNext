const assert = require("assert");
const {
  frameDisplayDelay,
  frameStillCurrent,
} = require("../../behaviourlab/web/static/frameview.js");

/** 固定时序：[0, 1.0, 1.1, 2.0] 秒，每次取图解码 20ms，1× 播放。 */
function simulate(times, { rate = 1, decodeMs = 20 } = {}) {
  const frames = times.map((time, index) => ({ index, time }));
  const shown = [];
  let now = 0; // 播放从 0 开始，帧 0 已经显示在 0ms。
  let previous = frames[0].time;
  shown.push({ index: 0, at: 0 });
  for (let i = 1; i < frames.length; i += 1) {
    const started = now;
    const ready = started + decodeMs;
    const delay = frameDisplayDelay(previous, frames[i].time, rate,
      ready - started);
    const at = ready + delay;
    shown.push({ index: i, at });
    previous = frames[i].time;
    now = at;
  }
  return shown;
}

function main() {
  // 1. 显示时刻 = 上次显示 + 真实相邻时间差/倍率（解码时间已扣掉）。
  const shown = simulate([0.0, 1.0, 1.1, 2.0]);
  assert.deepStrictEqual(shown.map((item) => item.at), [0, 1000, 1100, 2000]);
  assert.deepStrictEqual(shown.map((item) => item.index), [0, 1, 2, 3]);

  // 2. 倍率按真实间隔缩放。
  const fast = simulate([0.0, 1.0, 1.1, 2.0], { rate: 2 });
  assert.deepStrictEqual(fast.map((item) => item.at), [0, 500, 550, 1000]);

  // 3. 解码慢于间隔时立即显示，不再叠加旧间隔。
  const slow = simulate([0.0, 0.02, 0.04], { decodeMs: 120 });
  assert.deepStrictEqual(slow.map((item) => item.at), [0, 120, 240]);
  assert.strictEqual(frameDisplayDelay(1.0, 1.0, 1, 20), 0, "同一时刻不倒扣");
  assert.strictEqual(frameDisplayDelay(2.0, 1.0, 1, 20), 0, "倒序不倒扣");
  assert.strictEqual(frameDisplayDelay(null, 1.0, 1, 20), 0, "无前值立即显示");
  assert.strictEqual(frameDisplayDelay(0, 1, 0, 0), 1000, "非法倍率按 1×");

  // 4. 提交守卫：请求代号或页面上下文一变，旧帧就不能回写。
  const context = { ticket: 3, page: "draft|arena|0|0" };
  assert.strictEqual(frameStillCurrent(context, 3, "draft|arena|0|0"), true);
  assert.strictEqual(frameStillCurrent(context, 4, "draft|arena|0|0"), false,
    "暂停后再释出的旧响应不能提交");
  assert.strictEqual(frameStillCurrent(context, 3, "draft|review|0|0"), false,
    "离开记录页后不能提交");
  assert.strictEqual(frameStillCurrent(context, 3, "draft|arena|0|1"), false);
  assert.strictEqual(frameStillCurrent(null, 3, "draft|arena|0|0"), false);

  console.log("frame-playback OK");
}

main();
