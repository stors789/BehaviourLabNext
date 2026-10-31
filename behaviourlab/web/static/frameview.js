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

  return { createFrameLoader };
});
