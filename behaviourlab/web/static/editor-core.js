/* 几何编辑状态机与预览确认闸门；浏览器与 Node 共用。 */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  } else {
    root.BLCore = api;
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const TOOLBAR_HEIGHT = 40;
  const HANDLE_HIT_PX = 11;

  function clone(value) {
    return value == null ? value : JSON.parse(JSON.stringify(value));
  }

  function EditorCore() {
    this.spec = null;
    this.points = [];
    this.committed = null;
    this.mode = null;
    this.drag = null;
  }

  EditorCore.prototype.reset = function (spec, geometry) {
    this.spec = spec || null;
    this.points = [];
    this.committed = geometry ? clone(geometry) : null;
    this.mode = spec && spec.modes && spec.modes.length ? spec.modes[0] : null;
    this.drag = null;
  };

  EditorCore.prototype.setMode = function (mode) {
    if (!this.spec || !this.spec.modes) return false;
    if (!this.spec.modes.includes(mode)) return false;
    if (this.mode === mode) return false;
    this.mode = mode;
    this.points = [];
    this.drag = null;
    return true;
  };

  EditorCore.prototype.editable = function () {
    return Boolean(this.spec) && !this.committed;
  };

  EditorCore.prototype.isEllipseDrag = function () {
    if (!this.spec) return false;
    if (this.spec.type === "ellipse") return this.mode !== "points";
    if (this.spec.type === "ellipse_or_point") return this.mode === "ellipse";
    return false;
  };

  EditorCore.prototype.addPoint = function (point) {
    const spec = this.spec;
    if (!spec || !point) return { status: "ignored" };
    if (this.committed) return { status: "ignored" };
    if (this.isEllipseDrag()) return { status: "ignored" };

    this.points.push([Number(point.x), Number(point.y)]);
    return this._settle();
  };

  EditorCore.prototype._settle = function () {
    const spec = this.spec;
    const count = Number(spec.count) || 0;
    const minimum = Math.max(3, Number(spec.min_points) || 3);

    switch (spec.type) {
      case "point":
        return { status: "committed", geometry: this._commit() };
      case "line":
        return this._atLeast(2);
      case "quad":
        return this._atLeast(4);
      case "holes":
        // 洞数来自实验参数；非整数或缺失时不自动提交。
        return count > 0 && this.points.length >= count
          ? { status: "committed", geometry: this._commit() }
          : { status: "pending", ready: false };
      case "polygon":
        if (spec.fixed && count > 0) return this._atLeast(count);
        return {
          status: "pending",
          ready: this.points.length >= minimum,
        };
      case "ellipse":
        if (this.mode === "points") {
          return {
            status: "pending",
            ready: this.points.length >= Math.max(5, Number(spec.min_points) || 5),
          };
        }
        return { status: "ignored" };
      case "ellipse_or_point":
        if (this.mode === "center") {
          return { status: "committed", geometry: this._commit() };
        }
        return { status: "ignored" };
      default:
        return { status: "pending", ready: false };
    }
  };

  EditorCore.prototype._atLeast = function (target) {
    return this.points.length >= target
      ? { status: "committed", geometry: this._commit() }
      : { status: "pending", ready: false };
  };

  EditorCore.prototype.canFinish = function () {
    const spec = this.spec;
    if (!spec || this.committed) return false;
    const minimum = Math.max(3, Number(spec.min_points) || 3);
    if (spec.type === "polygon") return this.points.length >= minimum;
    if (spec.type === "ellipse") {
      return this.mode === "points" && this.points.length >= 5;
    }
    return false;
  };

  EditorCore.prototype.finish = function () {
    if (!this.canFinish()) return null;
    return this._commit();
  };

  EditorCore.prototype._commit = function () {
    const spec = this.spec;
    const points = this.points.map((point) => [point[0], point[1]]);
    if (spec.type === "holes") {
      const holes = points.map((point, index) => ({
        id: String(index + 1),
        center: point,
      }));
      this.committed = { type: "holes", holes };
    } else if (spec.type === "ellipse" && this.mode === "points") {
      this.committed = { type: "ellipse", points };
    } else if (spec.type === "ellipse_or_point" && this.mode === "center") {
      this.committed = { type: "point", points: [points[0]] };
    } else {
      this.committed = { type: spec.type, points };
    }
    this.points = [];
    return clone(this.committed);
  };

  EditorCore.prototype.commitEllipse = function (box) {
    if (!this.spec || this.committed) return null;
    if (!box) return null;
    const width = Math.abs(box.x2 - box.x1);
    const height = Math.abs(box.y2 - box.y1);
    const rx = width / 2;
    const ry = height / 2;
    if (rx <= 1 || ry <= 1) return null;
    const center = [(box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2];
    this.committed = {
      type: "ellipse",
      center: [center[0], center[1]],
      radii: [rx, ry],
      angle: 0,
    };
    this.points = [];
    return clone(this.committed);
  };

  EditorCore.prototype.undo = function () {
    if (this.points.length) {
      this.points.pop();
      return "points";
    }
    if (this.committed) {
      this.committed = null;
      return "geometry";
    }
    return null;
  };

  EditorCore.prototype.redraw = function () {
    this.points = [];
    this.committed = null;
    this.drag = null;
  };

  EditorCore.prototype.hitHandle = function (point, scale) {
    const radius = HANDLE_HIT_PX / Math.max(scale, 1e-9);
    const candidates = this.points.length ? this.points : [];
    for (let index = 0; index < candidates.length; index += 1) {
      const dx = candidates[index][0] - point.x;
      const dy = candidates[index][1] - point.y;
      if (Math.hypot(dx, dy) <= radius) return { kind: "point", index };
    }
    if (!this.committed) return null;
    const shape = this.committed;
    if (shape.type === "ellipse" && shape.center) {
      if (Math.hypot(shape.center[0] - point.x, shape.center[1] - point.y) <= radius) {
        return { kind: "center" };
      }
      const handles = ellipseHandles(shape);
      for (const handle of handles) {
        if (Math.hypot(handle.point[0] - point.x, handle.point[1] - point.y) <= radius) {
          return { kind: "radius", axis: handle.axis };
        }
      }
      return null;
    }
    if (shape.type === "holes") {
      for (const hole of shape.holes || []) {
        if (Math.hypot(hole.center[0] - point.x, hole.center[1] - point.y) <= radius) {
          return { kind: "hole", id: hole.id };
        }
      }
      return null;
    }
    const points = shape.points || [];
    for (let index = 0; index < points.length; index += 1) {
      if (Math.hypot(points[index][0] - point.x, points[index][1] - point.y) <= radius) {
        return { kind: "vertex", index };
      }
    }
    return null;
  };

  function ellipseHandles(shape) {
    const angle = ((Number(shape.angle) || 0) * Math.PI) / 180;
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    const cx = shape.center[0];
    const cy = shape.center[1];
    const rx = shape.radii[0];
    const ry = shape.radii[1];
    return [
      { axis: "rx", point: [cx + rx * cos, cy + rx * sin] },
      { axis: "ry", point: [cx - ry * sin, cy + ry * cos] },
    ];
  }

  function moveVertex(shape, index, point) {
    const next = clone(shape);
    next.points[index] = [point.x, point.y];
    return next;
  }

  function moveHole(shape, id, point) {
    const next = clone(shape);
    for (const hole of next.holes || []) {
      if (hole.id === id) hole.center = [point.x, point.y];
    }
    return next;
  }

  function moveEllipseCenter(shape, point) {
    const next = clone(shape);
    next.center = [point.x, point.y];
    return next;
  }

  function moveRadius(shape, axis, point) {
    const next = clone(shape);
    const angle = ((Number(shape.angle) || 0) * Math.PI) / 180;
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    const dx = point.x - shape.center[0];
    const dy = point.y - shape.center[1];
    if (axis === "rx") {
      next.radii[0] = Math.max(1, Math.abs(dx * cos + dy * sin));
    } else {
      next.radii[1] = Math.max(1, Math.abs(-dx * sin + dy * cos));
    }
    return next;
  }

  function SaveQueue(send) {
    this.send = send;
    this.editVersion = 0;
    this.savedVersion = 0;
    this.chain = Promise.resolve();
    this.lastError = null;
    this._draining = false;
    this._drainPromise = null;
    this._snapshotFactory = null;
    this._onEcho = null;
  }

  SaveQueue.prototype.mark = function () {
    this.editVersion += 1;
  };

  SaveQueue.prototype.dirty = function () {
    return this.editVersion > this.savedVersion;
  };

  SaveQueue.prototype.reset = function () {
    this.editVersion = 0;
    this.savedVersion = 0;
    this.chain = Promise.resolve();
    this.lastError = null;
    this._draining = false;
    this._drainPromise = null;
    this._snapshotFactory = null;
    this._onEcho = null;
  };

  SaveQueue.prototype.save = function (snapshotFactory, onEcho) {
    this._snapshotFactory = snapshotFactory;
    this._onEcho = onEcho;
    if (this._draining) return this._drainPromise;
    this._draining = true;
    this._drainPromise = (async () => {
      try {
        while (this.dirty()) {
          const version = this.editVersion;
          // 每次发送都取最新快照，旧响应只在无新编辑时回写。
          const snapshot = this._snapshotFactory();
          let echo;
          try {
            echo = await this.send(snapshot);
          } catch (error) {
            this.lastError = error;
            return false;
          }
          this.savedVersion = Math.max(this.savedVersion, version);
          if (version === this.editVersion && this._onEcho) this._onEcho(echo);
        }
        this.lastError = null;
        return true;
      } finally {
        this._draining = false;
        this._drainPromise = null;
      }
    })();
    return this._drainPromise;
  };

  function rotateGeometry(geometry, width, height, from, to) {
    const delta = ((to - from) % 360 + 360) % 360;
    const point = (p) => {
      let x = p[0];
      let y = p[1];
      if (from === 90) { const nx = y; y = height - 1 - x; x = nx; }
      else if (from === 180) { x = width - 1 - x; y = height - 1 - y; }
      else if (from === 270) { const nx = width - 1 - y; y = x; x = nx; }
      if (to === 90) { const nx = height - 1 - y; y = x; x = nx; }
      else if (to === 180) { x = width - 1 - x; y = height - 1 - y; }
      else if (to === 270) { const nx = y; y = width - 1 - x; x = nx; }
      return [x, y];
    };
    const next = {};
    Object.entries(geometry || {}).forEach(([key, shape]) => {
      if (!shape || typeof shape !== "object") return;
      if (shape.type === "ellipse" && shape.center) {
        next[key] = {
          type: "ellipse",
          center: point(shape.center),
          radii: [shape.radii[0], shape.radii[1]],
          angle: ((Number(shape.angle) || 0) + delta) % 180,
        };
      } else if (shape.type === "holes") {
        next[key] = {
          type: "holes",
          holes: (shape.holes || []).map((hole) => ({ id: hole.id, center: point(hole.center) })),
        };
      } else if (shape.points) {
        next[key] = { type: shape.type, points: shape.points.map(point) };
      } else {
        next[key] = shape;
      }
    });
    return next;
  }

  function mergeEcho(target, incoming) {
    if (!target || !incoming) return incoming || target;
    const byAsset = new Map((incoming.videos || []).map((video) => [video.asset_id, video]));
    const videos = (target.videos || []).map((video) => {
      const fresh = byAsset.get(video.asset_id);
      if (!fresh) return video;
      const byUnit = new Map((fresh.units || []).map((unit) => [unit.id, unit]));
      const units = (video.units || []).map((unit) => {
        const next = byUnit.get(unit.id);
        if (!next) return unit;
        Object.assign(unit, next);
        return unit;
      });
      Object.assign(video, fresh, { units });
      return video;
    });
    Object.assign(target, incoming, { videos });
    return target;
  }

  function resolveStep(spec, context) {
    if (!spec) return null;
    const options = context || {};
    let count = Number(spec.count) || 0;
    if (spec.count_key) {
      const raw = Number(options[spec.count_key]);
      count = Number.isInteger(raw) && raw > 0 ? raw : 0;
    }
    if (spec.type === "ellipse_or_point" && count === 0) count = 1;
    const modes = spec.modes && spec.modes.length
      ? spec.modes.slice()
      : spec.type === "ellipse" ? ["drag", "points"]
        : spec.type === "ellipse_or_point" ? ["center", "ellipse"] : [];
    return Object.assign({}, spec, { count, modes });
  }

  async function pickMedia(kind, options) {
    const opts = options || {};
    const bridge = opts.bridge || null;
    const method = kind === "folder" ? "choose_folder" : "choose_videos";
    if (bridge && typeof bridge[method] === "function") {
      const result = await bridge[method]();
      if (kind === "folder") {
        if (typeof result === "string" && result) await opts.onPaths([result]);
      } else if (Array.isArray(result) && result.length) {
        const paths = result.filter((item) => typeof item === "string" && item);
        if (paths.length) await opts.onPaths(paths);
      }
      return "desktop";
    }
    const picker = opts.picker;
    if (!picker || typeof picker.pickFiles !== "function") return "none";
    const files = await picker.pickFiles({
      multiple: true,
      directory: kind === "folder",
    });
    if (Array.isArray(files) && files.length) await opts.onFiles(files);
    return "browser";
  }

  function viewportToSource(view, pageX, pageY) {
    const scale = Math.min(view.cssWidth / view.sourceWidth,
      view.cssHeight / view.sourceHeight);
    if (!(scale > 0)) return null;
    const ox = (view.cssWidth - view.sourceWidth * scale) / 2;
    const oy = (view.cssHeight - view.sourceHeight * scale) / 2;
    const x = (pageX - (view.left || 0) - ox) / scale;
    const y = (pageY - (view.top || 0) - oy) / scale;
    if (x < -1e-9 || y < -1e-9 ||
        x > view.sourceWidth + 1e-9 || y > view.sourceHeight + 1e-9) {
      return null;
    }
    return { x, y };
  }

  function stableStringify(value) {
    if (value === null || typeof value !== "object") return JSON.stringify(value);
    if (Array.isArray(value)) {
      return "[" + value.map(stableStringify).join(",") + "]";
    }
    const keys = Object.keys(value).sort();
    return "{" + keys.map((key) => JSON.stringify(key) + ":" +
      stableStringify(value[key])).join(",") + "}";
  }

  function previewSignature(config) {
    return stableStringify(config || {});
  }

  // 预览说明：把真实 missing_reason 与 warnings 显示出来，空图不算无说明成功。
  const MISSING_LABELS = {
    foreground_saturated: "前景饱和",
    no_candidate: "无候选",
    inconsistent_candidate_size: "候选尺寸与已接受体型不一致",
    no_nearby_candidate: "附近没有符合位置门限的候选",
  };

  function previewNoteFor(data) {
    const summary = (data && data.summary) || {};
    const detection = summary.detection || null;
    const warnings = (data && Array.isArray(data.warnings) ? data.warnings : [])
      .filter((text) => typeof text === "string" && text.trim());
    const lines = [];
    let tone = "ok";
    if (summary.detection_available === false) {
      lines.push("检测：未运行（人工记录路径）");
      tone = "muted";
    } else if (!detection) {
      lines.push("检测：结果缺失");
      tone = "error";
    } else if (detection.missing_reason) {
      const reason = MISSING_LABELS[detection.missing_reason] || detection.missing_reason;
      lines.push("检测：" + reason);
      tone = "warn";
    } else if (!detection.candidates) {
      lines.push("检测：无候选");
      tone = "warn";
    } else {
      lines.push("检测：候选 " + detection.candidates + " 个");
    }
    const behavior = summary.behavior;
    if (behavior && typeof behavior === "object") {
      const value = (number) => (typeof number === "number" && isFinite(number)
        ? number.toFixed(2) : "—");
      lines.push("行为 " + value(behavior.total_time_s) + " s：非挣扎 "
        + value(behavior.immobility_time_s) + " / 挣扎 "
        + value(behavior.struggling_time_s) + " / 不确定 "
        + value(behavior.uncertain_time_s) + " s");
    }
    warnings.forEach((text) => lines.push(text));
    return { tone, text: lines.join("\n") };
  }

  // 校验失败优先展示字段级原因；通用文案只在没有具体字段时兜底。
  function issueText(error, fallback) {
    const issues = error && Array.isArray(error.fields) ? error.fields : [];
    const messages = issues
      .map((issue) => (issue && (issue.message || issue.field)) || "")
      .filter((text) => text);
    if (messages.length) return messages.join("；");
    return (error && error.message) || fallback || "请求失败";
  }

  // 预览前先落盘：保存失败不发起请求，页面/配置失效也不发起。
  // task: { dirty(), save(), alive(), config(), send(config) }
  async function saveThenPreview(task) {
    const job = task || {};
    if (typeof job.dirty === "function" && job.dirty()) {
      const saved = await job.save();
      if (!saved) return { status: "save_failed" };
    }
    if (typeof job.alive === "function" && !job.alive()) {
      return { status: "stale" };
    }
    const config = job.config();
    const data = await job.send(config);
    return { status: "ok", config, data };
  }

  function PreviewGate() {
    this.token = 0;
    this.signature = null;
    this.image = null;
    this.pending = false;
  }

  PreviewGate.prototype.invalidate = function () {
    this.token += 1;
    this.signature = null;
    this.image = null;
    this.pending = false;
    return this.token;
  };

  PreviewGate.prototype.begin = function (config) {
    this.token += 1;
    this.pending = true;
    this.image = null;
    this.signature = null;
    return { token: this.token, config: JSON.parse(JSON.stringify(config)) };
  };

  PreviewGate.prototype.accept = function (request, response) {
    if (!request || request.token !== this.token || !this.pending) {
      return false;
    }
    this.pending = false;
    this.signature = previewSignature(request.config);
    this.image = response || null;
    return true;
  };

  PreviewGate.prototype.fail = function (request) {
    if (!request || request.token !== this.token) return false;
    this.pending = false;
    this.signature = null;
    return true;
  };

  PreviewGate.prototype.confirmed = function (config) {
    return Boolean(this.signature) && this.signature === previewSignature(config);
  };

  return {
    TOOLBAR_HEIGHT,
    HANDLE_HIT_PX,
    EditorCore,
    PreviewGate,
    SaveQueue,
    previewSignature,
    previewNoteFor,
    issueText,
    saveThenPreview,
    stableStringify,
    resolveStep,
    rotateGeometry,
    mergeEcho,
    pickMedia,
    viewportToSource,
    moveVertex,
    moveHole,
    moveEllipseCenter,
    moveRadius,
  };
});
