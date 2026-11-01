(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.BLFrameView = factory();
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  function numberHeader(headers, name, fallback) {
    const value = headers && headers.get ? headers.get(name) : null;
    if (value === null || value === undefined || value === "") {
      return fallback;
    }
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  }

  function frameDisplayDelay(previousTime, nextTime, rate, elapsedMs) {
    /* 距离“该显示下一帧”的剩余时间：真实相邻源时间差 ÷ 倍率 − 已耗时间。
       解码慢于间隔时返回 0，立即显示。 */
    if (previousTime === null || previousTime === undefined
        || nextTime === null || nextTime === undefined) {
      return 0;
    }
    const previous = Number(previousTime);
    const next = Number(nextTime);
    if (!Number.isFinite(previous) || !Number.isFinite(next) || next <= previous) {
      return 0;
    }
    const speed = Number(rate);
    const factor = Number.isFinite(speed) && speed > 0 ? speed : 1;
    const spent = Number(elapsedMs);
    const wait = ((next - previous) * 1000) / factor
      - (Number.isFinite(spent) ? spent : 0);
    return wait > 0 ? wait : 0;
  }

  function frameStillCurrent(context, ticket, page) {
    /* 提交画面前的双重校验：请求代号仍是当前一个，且页面上下文没变。 */
    return Boolean(context) && context.ticket === ticket
      && context.page === page;
  }

  function createFrameLoader(options) {
    const settings = options || {};
    const fetchImpl = settings.fetchImpl || fetch;
    const ImageCtor = settings.ImageCtor || Image;
    const URLImpl = settings.URLImpl || URL;
    let generation = 0;

    async function load(url, requestOptions) {
      const ticket = ++generation;
      const response = await fetchImpl(url, {
        cache: "no-store",
        ...(requestOptions || {}),
      });
      if (!response.ok) {
        throw new Error(`读取源帧失败（${response.status}）`);
      }

      const frame = numberHeader(response.headers, "X-Source-Frame", null);
      const time = numberHeader(response.headers, "X-Source-Time", null);
      const clockSource =
        (response.headers && response.headers.get
          ? response.headers.get("X-Clock-Source")
          : "") || "";

      const blob = await response.blob();
      if (ticket !== generation) {
        return null;
      }

      const objectUrl = URLImpl.createObjectURL(blob);
      try {
        const image = await new Promise((resolve, reject) => {
          const element = new ImageCtor();
          element.onload = () => resolve(element);
          element.onerror = () => reject(new Error("源帧图像无法解码"));
          element.src = objectUrl;
        });
        if (ticket !== generation) {
          return null;
        }
        return { image, frame, time, clockSource };
      } finally {
        URLImpl.revokeObjectURL(objectUrl);
      }
    }

    function invalidate() {
      generation += 1;
    }

    function currentGeneration() {
      return generation;
    }

    return { load, invalidate, currentGeneration };
  }

  return { createFrameLoader, frameDisplayDelay, frameStillCurrent };
});
