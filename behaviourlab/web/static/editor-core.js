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

  // MWM 象限辅助线：把物理 x/y 轴反变换回源像素方向，与 Python 端一致。
  function mwmLines(arena, diameterCm, rotationDeg) {
    if (!arena || arena.type !== "ellipse" || !arena.center || !arena.radii) {
      return null;
    }
    const rx = Number(arena.radii[0]);
    const ry = Number(arena.radii[1]);
    const diameter = Number(diameterCm);
    if (!(rx > 0) || !(ry > 0) || !(diameter > 0)) return null;
    const theta = (Number(arena.angle) || 0) * Math.PI / 180;
    const ct = Math.cos(theta), st = Math.sin(theta);
    const a = diameter / (2 * rx), b = diameter / (2 * ry);
    const m00 = a * ct * ct + b * st * st;
    const m01 = (a - b) * ct * st;
    const m11 = a * st * st + b * ct * ct;
    const alpha = (Number(rotationDeg) || 0) * Math.PI / 180;
    const ca = Math.cos(alpha), sa = Math.sin(alpha);
    const n00 = ca * m00 + sa * m01;
    const n01 = ca * m01 + sa * m11;
    const n10 = sa * m00 - ca * m01;
    const n11 = sa * m01 - ca * m11;
    const det = n00 * n11 - n01 * n10;
    if (Math.abs(det) < 1e-12) return null;
    const directions = [
      [n11 / det, -n10 / det],
      [-n01 / det, n00 / det],
    ].map(([x, y]) => {
      const length = Math.hypot(x, y) || 1;
      return [x / length, y / length];
    });
    return {
      center: [Number(arena.center[0]), Number(arena.center[1])],
      directions,
      length: Math.max(rx, ry),
    };
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
      if (behavior.mode === "oft") {
        const count = (number) => (typeof number === "number" && isFinite(number)
          ? String(number) : "—");
        lines.push("行为 " + value(behavior.total_time_s) + " s：中心 "
          + value(behavior.center_percent) + "% / 进入 "
          + count(behavior.center_entries) + " 次 / 路程 "
          + value(behavior.total_distance_cm) + " cm / 平均速度 "
          + value(behavior.mean_speed_cm_s) + " cm/s");
      } else if (behavior.mode === "fst") {
        lines.push("行为 " + value(behavior.total_time_s) + " s：不动 "
          + value(behavior.immobility_time_s) + " / 活动 "
          + value(behavior.active_time_s) + " / 不确定 "
          + value(behavior.uncertain_time_s) + " s");
      } else if (behavior.mode === "mwm") {
        const count = (number) => (typeof number === "number" && isFinite(number)
          ? String(number) : "—");
        lines.push("行为 " + value(behavior.total_time_s) + " s：进入 "
          + count(behavior.platform_entries) + " 次 / 平台 "
          + value(behavior.platform_time_s) + " s / 路径 "
          + value(behavior.total_distance_cm) + " cm");
      } else if (behavior.mode === "nor") {
        const count = (number) => (typeof number === "number" && isFinite(number)
          ? number.toFixed(2) : "—");
        lines.push("行为 " + value(behavior.total_time_s) + " s：探索 "
          + value(behavior.total_object_exploration_s) + " s（"
          + count(behavior.object_1_exploration_s) + " / "
          + count(behavior.object_2_exploration_s) + " s）· 接近 "
          + count((Number(behavior.object_1_zone_time_s) || 0)
            + (Number(behavior.object_2_zone_time_s) || 0)) + " s");
      } else if (behavior.mode === "epm") {
        const count = (number) => (typeof number === "number" && isFinite(number)
          ? String(number) : "—");
        lines.push("行为 " + value(behavior.total_time_s) + " s：开放 "
          + value(behavior.open_time_s) + " / 封闭 "
          + value(behavior.closed_time_s) + " / 中心 "
          + value(behavior.center_time_s) + " s · 入臂 "
          + count(behavior.total_arm_entries) + " 次（开放 "
          + count(behavior.open_entries) + "）");
      } else {
        lines.push("行为 " + value(behavior.total_time_s) + " s：非挣扎 "
          + value(behavior.immobility_time_s) + " / 挣扎 "
          + value(behavior.struggling_time_s) + " / 不确定 "
          + value(behavior.uncertain_time_s) + " s");
      }
      if (typeof behavior.threshold_note === "string" && behavior.threshold_note) {
        lines.push(behavior.threshold_note);
      }
    }
    warnings.forEach((text) => lines.push(text));
    return { tone, text: lines.join("\n") };
  }

  // 字段路径本身是开发用语，界面用描述符登记的名称（如“中心区边长”）。
  let FIELD_LABELS = null;

  function setFieldLabels(labels) {
    FIELD_LABELS = labels && typeof labels === "object" ? labels : null;
  }

  function fieldName(field) {
    const path = typeof field === "string" ? field : "";
    const key = path ? path.split(".").pop() : "";
    if (!key || !FIELD_LABELS) return "";
    if (!Object.prototype.hasOwnProperty.call(FIELD_LABELS, key)) return "";
    const label = FIELD_LABELS[key];
    return typeof label === "string" ? label : "";
  }

  function issueLine(issue) {
    if (!issue) return "";
    const message = typeof issue.message === "string" ? issue.message.trim() : "";
    const name = fieldName(issue.field);
    if (message) {
      // 后端文案已经带名称时不重复前缀，未带名称时补上“字段：原因”。
      if (!name || message.indexOf(name) === 0) return message;
      return name + "：" + message;
    }
    return name || (typeof issue.field === "string" ? issue.field : "");
  }

  // 校验失败优先展示字段级原因；通用文案只在没有具体字段时兜底。
  function issueText(error, fallback) {
    const issues = error && Array.isArray(error.fields) ? error.fields : [];
    const messages = issues.map(issueLine).filter((text) => text);
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

  // NOR 物体步骤 ↔ geometry.objects 的桥接：步骤键只用于画布，落盘仍是
  // 恰好两个对象的 {id,name,role,center,zone} 结构。
  const NOR_OBJECT_STEPS = {
    object1: { index: 0, kind: "center" },
    object2: { index: 1, kind: "center" },
    explore1: { index: 0, kind: "zone" },
    explore2: { index: 1, kind: "zone" },
  };
  const NOR_OBJECT_DEFAULTS = [
    { id: "object_1", name: "物体 1", role: "familiar" },
    { id: "object_2", name: "物体 2", role: "novel" },
  ];

  function norKeyShape(geometry, key) {
    if (!geometry || !Object.prototype.hasOwnProperty.call(geometry, key)) {
      return null;
    }
    const shape = geometry[key];
    return shape ? clone(shape) : null;
  }

  function norObjects(geometry) {
    const source = geometry ? geometry.objects : null;
    const list = [];
    for (let index = 0; index < 2; index += 1) {
      const fallback = NOR_OBJECT_DEFAULTS[index];
      let entry = null;
      if (Array.isArray(source)) entry = source[index] || null;
      else if (source && typeof source === "object") {
        // 兼容早期只存两点形状的草稿。
        const points = Array.isArray(source.points) ? source.points : [];
        entry = points.length > index ? { center: points[index] } : null;
      }
      const center = entry && Array.isArray(entry.center)
        && entry.center.length >= 2
        ? [Number(entry.center[0]), Number(entry.center[1])] : null;
      const valid = center && Number.isFinite(center[0]) && Number.isFinite(center[1]);
      list.push({
        id: entry && typeof entry.id === "string" && entry.id
          ? entry.id : fallback.id,
        name: entry && typeof entry.name === "string" && entry.name
          ? entry.name : fallback.name,
        role: entry && (entry.role === "novel" || entry.role === "familiar")
          ? entry.role : fallback.role,
        center: valid ? center : null,
        zone: entry && entry.zone && Array.isArray(entry.zone.points)
          && entry.zone.points.length >= 3 ? clone(entry.zone) : null,
      });
    }
    return list;
  }

  function norShape(geometry, key) {
    const step = NOR_OBJECT_STEPS[key];
    if (!step) return norKeyShape(geometry, key);
    const objects = norObjects(geometry);
    if (step.kind === "center") {
      const center = objects[step.index].center;
      return center ? { type: "point", points: [[center[0], center[1]]] } : null;
    }
    const zone = objects[step.index].zone;
    return zone ? clone(zone) : null;
  }

  function norApply(geometry, key, shape) {
    if (!geometry) return;
    const step = NOR_OBJECT_STEPS[key];
    if (!step) {
      // 普通步骤沿用原来的语义：清除即删除键，不留下 null 占位。
      if (shape) geometry[key] = shape;
      else delete geometry[key];
      return;
    }
    const objects = norObjects(geometry);
    if (step.kind === "center") {
      const point = shape && Array.isArray(shape.points) ? shape.points[0] : null;
      objects[step.index].center = point
        ? [Number(point[0]), Number(point[1])] : null;
    } else {
      objects[step.index].zone = shape ? clone(shape) : null;
    }
    geometry.objects = objects;
  }

  function norClear(geometry, key) {
    norApply(geometry, key, null);
  }

  function norSetRole(geometry, index, role, phase) {
    if (!geometry || (index !== 0 && index !== 1)) return norObjects(geometry);
    const objects = norObjects(geometry);
    const value = role === "novel" ? "novel" : "familiar";
    objects[index].role = value;
    if (phase !== "familiarization") {
      // 识别期：改一个角色同时把另一项设成相反角色。
      objects[1 - index].role = value === "novel" ? "familiar" : "novel";
    }
    geometry.objects = objects;
    return objects;
  }

  // EPM 的五区存成 geometry.zones.<key>；画布步骤键与区域键同名。
  const NESTED_ZONE_STEPS = {
    epm: ["center", "open_1", "open_2", "closed_1", "closed_2"],
  };

  function stepShape(assay, geometry, key) {
    if (!geometry) return null;
    if (assay === "nor") return norShape(geometry, key);
    const nested = NESTED_ZONE_STEPS[assay];
    if (nested && nested.includes(key)) {
      const zones = geometry.zones;
      const shape = zones && typeof zones === "object" ? zones[key] : null;
      return shape ? clone(shape) : null;
    }
    return norKeyShape(geometry, key);
  }

  function stepApply(assay, geometry, key, shape) {
    if (!geometry) return;
    if (assay === "nor") {
      norApply(geometry, key, shape);
      return;
    }
    const nested = NESTED_ZONE_STEPS[assay];
    if (nested && nested.includes(key)) {
      if (shape) {
        if (!geometry.zones || typeof geometry.zones !== "object") {
          geometry.zones = {};
        }
        geometry.zones[key] = clone(shape);
      } else if (geometry.zones && typeof geometry.zones === "object") {
        delete geometry.zones[key];
      }
      return;
    }
    if (shape) geometry[key] = shape;
    else delete geometry[key];
  }

  function stepClear(assay, geometry, key) {
    stepApply(assay, geometry, key, null);
  }

  function norNormalizeRoles(geometry, phase) {
    if (!geometry) return norObjects(geometry);
    const objects = norObjects(geometry);
    if (phase !== "familiarization" && objects[0].role === objects[1].role) {
      objects[1].role = objects[0].role === "novel" ? "familiar" : "novel";
    }
    geometry.objects = objects;
    return objects;
  }

  return {
    TOOLBAR_HEIGHT,
    HANDLE_HIT_PX,
    EditorCore,
    PreviewGate,
    SaveQueue,
    previewSignature,
    previewNoteFor,
    issueText,
    setFieldLabels,
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
    mwmLines,
    norObjects,
    norShape,
    norApply,
    norClear,
    norSetRole,
    norNormalizeRoles,
    stepShape,
    stepApply,
    stepClear,
  };
});
