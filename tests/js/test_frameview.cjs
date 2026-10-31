const assert = require("assert");
const { createFrameLoader } = require("../../behaviourlab/web/static/frameview.js");

class FakeImage {
  constructor() {
    this.onload = null;
    this.onerror = null;
    this._src = "";
  }

  set src(value) {
    this._src = value;
    setTimeout(() => {
      if (this.onload) this.onload();
    }, 0);
  }

  get src() {
    return this._src;
  }
}

function response(headers, blob) {
  return {
    ok: true,
    status: 200,
    headers: { get: (name) => (name in headers ? headers[name] : null) },
    blob: async () => blob,
  };
}

async function main() {
  const urls = [];
  const revoked = [];
  const URLImpl = {
    createObjectURL: () => {
      const url = `blob:test-${urls.length}`;
      urls.push(url);
      return url;
    },
    revokeObjectURL: (url) => revoked.push(url),
  };
  let seenOptions = null;
  const loader = createFrameLoader({
    fetchImpl: async (url, options) => {
      seenOptions = options;
      return response(
        {
          "X-Source-Frame": "7",
          "X-Source-Time": "0.28",
          "X-Clock-Source": "pts",
        },
        new Blob(["jpeg-bytes"])
      );
    },
    ImageCtor: FakeImage,
    URLImpl,
  });

  const result = await loader.load("/api/assets/a/frame?index=7");
  assert.strictEqual(result.frame, 7);
  assert.ok(Math.abs(result.time - 0.28) < 1e-12);
  assert.strictEqual(result.clockSource, "pts");
  assert.strictEqual(seenOptions.cache, "no-store");
  assert.deepStrictEqual(revoked, urls);
  assert.strictEqual(urls.length, 1);

  let release;
  const pending = new Promise((resolve) => {
    release = () =>
      resolve(
        response(
          { "X-Source-Frame": "1", "X-Source-Time": "0.04" },
          new Blob(["stale"])
        )
      );
  });
  const loader2 = createFrameLoader({
    fetchImpl: () => pending,
    ImageCtor: FakeImage,
    URLImpl,
  });
  const stale = loader2.load("/api/assets/a/frame?index=1");
  loader2.invalidate();
  release();
  assert.strictEqual(await stale, null);
  assert.strictEqual(urls.length, 1);
  assert.strictEqual(revoked.length, 1);

  console.log("frameview-ok");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
