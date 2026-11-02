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

  // 当前实验有效步骤顺序里第一个尚未绘制的必需步骤；可选步骤不挡住推进，
  // 全部必需步骤完成时返回 null（停在原地继续可编辑，不自动进入分析）。
  function nextRequiredStep(steps, isDrawn) {
    const list = Array.isArray(steps) ? steps : [];
    const drawn = typeof isDrawn === "function" ? isDrawn : () => false;
    for (const step of list) {
      if (!step || step.required !== true) continue;
      if (drawn(step.key)) continue;
      return step;
    }
    return null;
  }

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

  // EZM 环上方向：内圈、四条分界线与四个区名都由保存的 arena/open_boundary
  // 与当前有效外径、跑道宽算出；只读几何，不把显示坐标写回草稿。
  const EZM_ZONES = ["O1", "C1", "O2", "C2"];
  const EZM_OPEN = ["O1", "O2"];

  function ezmGuides(arena, boundary, diameterCm, ringWidthCm) {
    if (!arena || !Array.isArray(arena.center) || !Array.isArray(arena.radii)) {
      return null;
    }
    const rx = Number(arena.radii[0]);
    const ry = Number(arena.radii[1]);
    const diameter = Number(diameterCm);
    const width = Number(ringWidthCm);
    if (!(rx > 0) || !(ry > 0) || !(diameter > 0) || !(width >= 0)) return null;
    const angle = Number(arena.angle) || 0;
    const theta = (angle * Math.PI) / 180;
    const ct = Math.cos(theta);
    const st = Math.sin(theta);
    const a = diameter / (2 * rx);
    const b = diameter / (2 * ry);
    const m00 = a * ct * ct + b * st * st;
    const m01 = (a - b) * ct * st;
    const m11 = a * st * st + b * ct * ct;
    const det = m00 * m11 - m01 * m01;
    if (Math.abs(det) < 1e-12) return null;
    const i00 = m11 / det;
    const i01 = -m01 / det;
    const i11 = m00 / det;
    const center = [Number(arena.center[0]), Number(arena.center[1])];
    const point = boundary && Array.isArray(boundary.points) &&
      boundary.points.length === 1 ? boundary.points[0] : null;
    if (!point) return null;
    const dx = Number(point[0]) - center[0];
    const dy = Number(point[1]) - center[1];
    // 方向判据固定 D=1：有效性不随实际外径变化；A_D 只是 A_1 的 D 倍。
    const a1 = 1 / (2 * rx);
    const b1 = 1 / (2 * ry);
    const u00 = a1 * ct * ct + b1 * st * st;
    const u01 = (a1 - b1) * ct * st;
    const u11 = a1 * st * st + b1 * ct * ct;
    const vx = u00 * dx + u01 * dy;
    const vy = u01 * dx + u11 * dy;
    if (!Number.isFinite(vx) || !Number.isFinite(vy)) return null;
    if (Math.hypot(vx, vy) < 1e-8) return null;
    const start = Math.atan2(vy, vx);
    const radius = diameter / 2;
    const inner = Math.max(0, radius - width);
    const at = (unitAngle, distance) => {
      const ux = distance * Math.cos(unitAngle);
      const uy = distance * Math.sin(unitAngle);
      return [center[0] + i00 * ux + i01 * uy,
        center[1] + i01 * ux + i11 * uy];
    };
    const rays = [];
    const labels = [];
    for (let index = 0; index < 4; index += 1) {
      const rayAngle = start + (index * Math.PI) / 2;
      rays.push({
        angle_deg: ((rayAngle * 180) / Math.PI % 360 + 360) % 360,
        from: at(rayAngle, inner),
        to: at(rayAngle, radius),
      });
      const middle = rayAngle + Math.PI / 4;
      labels.push({
        zone: EZM_ZONES[index],
        open: EZM_OPEN.indexOf(EZM_ZONES[index]) >= 0,
        point: at(middle, inner + (radius - inner) / 2),
      });
    }
    return {
      center,
      radii: [rx, ry],
      angle_deg: angle,
      inner_radii: [rx * inner / radius, ry * inner / radius],
      rings: { outer: radius, inner },
      rays,
      labels,
      boundary_angle_deg: ((start * 180) / Math.PI % 360 + 360) % 360,
    };
  }

  // 场地模板：只保留场地几何与允许的尺寸；平台/角色等试验相关字段不进模板。
  const TEMPLATE_BLOCKED_KEYS = ["role", "target", "target_id", "target_hole",
    "target_hole_id", "target_quadrant", "platform", "animal_id", "manual",
    "window"];

  function stripTemplateKeys(value) {
    if (Array.isArray(value)) return value.map(stripTemplateKeys);
    if (value && typeof value === "object") {
      const result = {};
      Object.keys(value).forEach((key) => {
        if (TEMPLATE_BLOCKED_KEYS.indexOf(key) >= 0) return;
        result[key] = stripTemplateKeys(value[key]);
      });
      return result;
    }
    return value;
  }

  function templatePayload(unit, templateKeys) {
    const allowed = Array.isArray(templateKeys) ? templateKeys : [];
    const geometry = unit && unit.geometry && typeof unit.geometry === "object"
      ? unit.geometry : {};
    const options = unit && unit.options && typeof unit.options === "object"
      ? unit.options : {};
    const picked = {};
    allowed.forEach((key) => {
      if (Object.prototype.hasOwnProperty.call(options, key)) {
        picked[key] = clone(options[key]);
      }
    });
    return { geometry: stripTemplateKeys(clone(geometry)), options: picked };
  }

  function applyFieldTemplate(assay, template, unit, templateKeys) {
    const allowed = Array.isArray(templateKeys) ? templateKeys : [];
    const source = template && template.geometry &&
      typeof template.geometry === "object" ? template.geometry : {};
    const geometry = stripTemplateKeys(clone(source));
    const current = unit && unit.geometry && typeof unit.geometry === "object"
      ? unit.geometry : {};
    // 平台是当次试验的目标，模板里没有它：应用时保留当前值。
    if (!geometry.platform && current.platform) {
      geometry.platform = clone(current.platform);
    }
    if (assay === "nor" && Array.isArray(geometry.objects)) {
      // 角色被剥离：优先沿用当前单位同序号的角色，否则第一 familiar、第二 novel。
      const defaults = ["familiar", "novel"];
      const previous = Array.isArray(current.objects) ? current.objects : [];
      const roles = [];
      geometry.objects.forEach((item, index) => {
        if (!item || typeof item !== "object") return;
        let role = item.role;
        if (role !== "familiar" && role !== "novel") {
          role = previous[index] && previous[index].role;
        }
        if (role !== "familiar" && role !== "novel") {
          role = defaults[index % 2];
        }
        item.role = role;
        roles.push(role);
      });
      if (roles.length === 2 && roles[0] === roles[1]) {
        geometry.objects[1].role = roles[0] === "familiar" ? "novel" : "familiar";
      }
    }
    // 旧版本把几何塞进 options：这些键不能随模板回到动物身上。
    const legacyGeometry = ["arena", "body_line", "diameter_line", "zones",
      "objects", "platform"];
    const currentOptions = clone((unit && unit.options) || {});
    legacyGeometry.forEach((key) => { delete currentOptions[key]; });
    const options = Object.assign({}, currentOptions);
    const templateOptions = template && template.options &&
      typeof template.options === "object" ? template.options : {};
    allowed.forEach((key) => {
      if (Object.prototype.hasOwnProperty.call(templateOptions, key)) {
        options[key] = clone(templateOptions[key]);
      }
    });
    return { geometry, options };
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
      } else if (Array.isArray(shape)) {
        // Barnes 的洞位数组：逐洞旋转中心，保留洞号与顺序。
        next[key] = shape.map((item) => (item && Array.isArray(item.center)
          ? Object.assign({}, item, { center: point(item.center) })
          : item));
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

  // -- 选择器触发守卫 ---------------------------------------------------
  const pickBusy = new WeakSet();

  async function guardedPick(trigger, run) {
    /* 等待期间防重复点击；完成/取消/失败都恢复原控件。
       失败不在这里显示：由调用方在原触发处给出可读错误。 */
    if (typeof run !== "function") return { status: "none" };
    if (trigger) {
      if (pickBusy.has(trigger) || trigger.disabled) {
        return { status: "busy" };
      }
      pickBusy.add(trigger);
      const previous = trigger.disabled;
      trigger.disabled = true;
      try {
        return { status: "done", result: await run() };
      } catch (error) {
        return { status: "error", error };
      } finally {
        pickBusy.delete(trigger);
        trigger.disabled = previous;
      }
    }
    try {
      return { status: "done", result: await run() };
    } catch (error) {
      return { status: "error", error };
    }
  }

  async function pickBackground(bridge, options) {
    /* 背景图选择：原生路径先经服务端确认可解码，浏览器上传由服务端
       解码后返回专用路径。取消返回 cancel，失败向上抛错——调用方
       保持旧有效背景不变。 */
    const opts = options || {};
    if (bridge && typeof bridge.choose_image === "function") {
      const path = await bridge.choose_image();
      if (typeof path !== "string" || !path) return { status: "cancel" };
      if (typeof opts.validate !== "function") {
        throw new Error("背景校验不可用");
      }
      await opts.validate(path);
      return { status: "ok", path };
    }
    const picker = opts.picker;
    if (!picker || typeof picker.pickFiles !== "function") {
      return { status: "none" };
    }
    const files = await picker.pickFiles({
      accept: ".png,.jpg,.jpeg,.bmp",
    });
    if (!Array.isArray(files) || !files.length) return { status: "cancel" };
    if (typeof opts.upload !== "function") {
      throw new Error("背景上传不可用");
    }
    const data = await opts.upload(files[0]);
    return { status: "ok", path: data.path };
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
    no_candidate_within_gate: "门限内没有符合的候选",
  };

  // 主按钮文字与下一次请求的 kind 同源，避免失败/修正后文字和行为不一致。
  function previewActionLabel(kind) {
    return kind === "behavior" ? "行为预览" : "检测预览";
  }

  // 整份草稿的人工输入检查：任一动物有未解析错误就阻止提交。
  function manualFormatIssue(units, lookup) {
    for (const unit of units || []) {
      if (!unit || !unit.manual || !Array.isArray(unit.manual.hole_visits)) {
        continue;
      }
      const cached = lookup ? lookup(unit) : null;
      if (cached && cached.error) {
        return {
          unitId: unit.id || null,
          unitName: unit.name || unit.id || "",
          error: cached.error,
        };
      }
    }
    return null;
  }

  // Barnes 人工洞序列：一行“视频秒, 洞号”，兼容中文逗号；保留输入顺序。
  function parseManualVisits(text) {
    const lines = String(text == null ? "" : text).split(/\r?\n/);
    const entries = [];
    for (let index = 0; index < lines.length; index += 1) {
      const line = lines[index].trim();
      if (!line) continue;
      const parts = line.replace(/，/g, ",").split(",")
        .map((part) => part.trim());
      if (parts.length !== 2 || !parts[0] || !parts[1]) {
        return { error: "第 " + (index + 1) + " 行应为“视频秒,洞号”" };
      }
      const seconds = Number(parts[0]);
      if (!Number.isFinite(seconds) || seconds < 0) {
        return { error: "第 " + (index + 1) + " 行时间必须是非负秒数" };
      }
      entries.push({ time_s: seconds, hole_id: parts[1] });
    }
    return { entries };
  }

  // 预览范围说明：configured 是完整配置范围，普通自动预览才是片段估计。
  function previewFramePrefix(frameTime) {
    return (typeof frameTime === "number" && isFinite(frameTime))
      ? "源帧 " + frameTime.toFixed(2) + " s · " : "";
  }

  function previewScopeNote(scope) {
    const fixed = (input) => (typeof input === "number" && isFinite(input)
      ? input.toFixed(2) : "—");
    if (!scope || typeof scope !== "object") return "";
    const span = fixed(scope.start_s) + "–" + fixed(scope.end_s) + " s";
    if (scope.mode === "frame") return "单帧检测预览（不产生行为统计）";
    if (scope.mode === "configured") return "完整配置范围 " + span;
    return "当前片段 " + span + "（片段估计，非完整行为结果）";
  }

  function previewNoteFor(data) {
    const summary = (data && data.summary) || {};
    const detection = summary.detection || null;
    const scope = (data && data.preview_scope) || null;
    const frameTime = data && typeof data.time_s === "number" ? data.time_s : null;
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
      } else if (behavior.mode === "ldb") {
        const count = (number) => (typeof number === "number" && isFinite(number)
          ? String(number) : "—");
        const percent = (number) => (typeof number === "number" && isFinite(number)
          ? number.toFixed(0) + "%" : "—");
        lines.push("行为 " + value(behavior.total_time_s) + " s：明室 "
          + value(behavior.light_time_s) + "（"
          + percent(behavior.light_time_percent) + "）/ 暗室 "
          + value(behavior.dark_time_s) + "（"
          + percent(behavior.dark_time_percent) + "）/ 门区 "
          + value(behavior.doorway_time_s) + " s · 初始 "
          + (behavior.initial_chamber === "light" ? "明室"
            : behavior.initial_chamber === "dark" ? "暗室" : "—")
          + " · 转换 " + count(behavior.total_transitions) + " 次");
        if (behavior.dark_box_visible === false) {
          lines.push("不可见暗室：明室失检按推断暗室计");
        }
      } else if (behavior.mode === "ezm") {
        const count = (number) => (typeof number === "number" && isFinite(number)
          ? String(number) : "—");
        const entryLatency = behavior.first_open_entry_censored
          ? "未确认（删失）"
          : value(behavior.first_open_entry_latency_s) + " s";
        lines.push("行为 " + value(behavior.total_time_s) + " s：开放 "
          + value(behavior.open_time_s) + " / 封闭 "
          + value(behavior.closed_time_s) + " / 边界 "
          + value(behavior.boundary_time_s) + " / 区外 "
          + value(behavior.outside_time_s) + " / 未知 "
          + value(behavior.unknown_time_s) + " s");
        lines.push("进入 开放 " + count(behavior.open_entries) + " / 封闭 "
          + count(behavior.closed_entries) + " 次 · 首次观察开放 "
          + value(behavior.first_open_observation_latency_s) + " s · 首次确认进入开放 "
          + entryLatency);
        // 行为统计范围按实际请求范围标注：configured 为完整配置窗。
        lines.push(previewFramePrefix(frameTime) + previewScopeNote(scope));
      } else if (behavior.mode === "barnes") {
        const count = (number) => (typeof number === "number" && isFinite(number)
          ? String(number) : "—");
        const percent = (number) => (typeof number === "number" && isFinite(number)
          ? number.toFixed(1) + "%" : "—");
        const primary = behavior.primary_censored
          ? "未到达目标" : value(behavior.primary_latency_s) + " s";
        const window = (behavior.window_start_s !== null
          && behavior.window_start_s !== undefined
          && behavior.window_end_s !== null
          && behavior.window_end_s !== undefined)
          ? value(behavior.window_start_s) + "–" + value(behavior.window_end_s) + " s"
          : "未确定";
        lines.push("行为 " + window + "：访问 " + count(behavior.visits)
          + " 次 · 首次目标 " + primary
          + " · 错误 " + count(behavior.primary_errors)
          + " · 鼻尖覆盖 " + percent(behavior.nose_coverage_percent));
        lines.push(previewFramePrefix(frameTime) + previewScopeNote(scope));
        if (behavior.visit_source === "manual") {
          lines.push("访问序列来自人工记录，未使用自动鼻尖证据");
        }
      } else if (behavior.mode === "ymaze") {
        const count = (number) => (typeof number === "number" && isFinite(number)
          ? String(number) : "—");
        const percent = (number) => (typeof number === "number" && isFinite(number)
          ? number.toFixed(0) + "%" : "—");
        lines.push("行为 " + value(behavior.total_time_s) + " s：入臂 "
          + count(behavior.total_arm_entries) + " 次（A "
          + count(behavior.arm_A_entries) + " / B "
          + count(behavior.arm_B_entries) + " / C "
          + count(behavior.arm_C_entries) + "）· 交替 "
          + count(behavior.alternation_count) + " / "
          + count(behavior.alternation_opportunities) + "（"
          + percent(behavior.alternation_percent) + "）");
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
    if (summary.kind === "detection") {
      // 检测预览只核对单帧；范围说明与行为统计无关。
      const note = previewScopeNote(scope) || "单帧检测预览（不产生行为统计）";
      lines.push(previewFramePrefix(frameTime) + note);
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
      // 原因里已含可读字段名（句首或句中）就不再补前缀，例如“缺少封闭臂 2”。
      if (!name || message.indexOf(name) >= 0) return message;
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

  // EPM 的五个区与 Y 迷宫的中心/三臂都存成 geometry.zones.<key>；画布步骤键与
  // 区域键同名，保存时不另造平行字段。
  const NESTED_ZONE_STEPS = {
    epm: ["center", "open_1", "open_2", "closed_1", "closed_2"],
    ymaze: ["center", "A", "B", "C"],
    ldb: ["light", "dark", "doorway"],
  };

  // Barnes 的洞位落盘为 geometry.holes 数组（恰好 hole_count 项），步骤键只
  // 用于画布；读写都在这里转换，避免另存一份带 type 的平行形状。
  function barnesShape(geometry, key) {
    if (key !== "holes") return norKeyShape(geometry, key);
    const list = geometry && Array.isArray(geometry.holes) ? geometry.holes : null;
    if (!list || !list.length) return null;
    const holes = [];
    list.forEach((hole) => {
      const center = hole && Array.isArray(hole.center) ? hole.center : null;
      if (!center || center.length < 2) return;
      const x = Number(center[0]);
      const y = Number(center[1]);
      if (!Number.isFinite(x) || !Number.isFinite(y)) return;
      holes.push({ id: String(hole.id), center: [x, y] });
    });
    return holes.length ? { type: "holes", holes } : null;
  }

  function barnesApply(geometry, key, shape) {
    if (key !== "holes") {
      if (shape) geometry[key] = clone(shape);
      else delete geometry[key];
      return;
    }
    const holes = shape && Array.isArray(shape.holes) ? shape.holes : null;
    if (!holes || !holes.length) {
      delete geometry.holes;
      return;
    }
    geometry.holes = holes.map((hole, index) => ({
      id: String(hole && hole.id != null && hole.id !== "" ? hole.id : index + 1),
      center: [Number(hole.center[0]), Number(hole.center[1])],
    }));
  }

  function stepShape(assay, geometry, key) {
    if (!geometry) return null;
    if (assay === "nor") return norShape(geometry, key);
    if (assay === "barnes") return barnesShape(geometry, key);
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
    if (assay === "barnes") {
      barnesApply(geometry, key, shape);
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

  // 动物时间窗：界面值转配置值。relative 的首次检测由服务端计算，这里只
  // 校验“偏移可为负、持续时间必须为正”，不补默认值也不改窗口计算。
  function normalizeWindow(mode, values) {
    const source = values && typeof values === "object" ? values : {};
    const read = (value) => {
      if (value === null || value === undefined) return null;
      if (typeof value === "string" && value.trim() === "") return null;
      const number = Number(value);
      return Number.isFinite(number) ? number : null;
    };
    if (mode === "global") return { window: null, error: null, field: null };
    if (mode === "exact") {
      const start = read(source.start_s);
      const end = read(source.end_s);
      if (start === null || start < 0) {
        return { window: null, error: "动物时间窗起点必须是非负秒数",
                 field: "start_s" };
      }
      if (end === null || end <= start) {
        return { window: null, error: "动物时间窗终点必须大于起点",
                 field: "end_s" };
      }
      return {
        window: { mode: "exact", start_s: start, end_s: end },
        error: null,
        field: null,
      };
    }
    if (mode === "relative") {
      const offset = read(source.offset_s);
      const duration = read(source.duration_s);
      if (offset === null) {
        return { window: null, error: "首次检测后的偏移必须是数值（可为负）",
                 field: "offset_s" };
      }
      if (duration === null || duration <= 0) {
        return { window: null, error: "持续时间必须是大于 0 的数值",
                 field: "duration_s" };
      }
      return {
        window: { mode: "relative", offset_s: offset, duration_s: duration },
        error: null,
        field: null,
      };
    }
    return { window: null, error: "未知的动物时间窗模式", field: "mode" };
  }

  // 选择页勾选与草稿视频对齐：保留仍选中视频的完整配置（单位、几何、
  // 参数、人工记录），新增视频按调用方给的初始化规则补在末尾。
  function reconcileVideos(videos, selectedIds, assets, makeVideo) {
    const list = Array.isArray(videos) ? videos : [];
    const picked = new Set(selectedIds || []);
    const kept = list.filter((video) => video && picked.has(video.asset_id));
    const added = [];
    (assets || []).forEach((asset) => {
      if (!picked.has(asset.id)) return;
      if (kept.some((video) => video.asset_id === asset.id)) return;
      if (added.some((video) => video.asset_id === asset.id)) return;
      added.push(makeVideo(asset));
    });
    return kept.concat(added);
  }

  // 保存取舍：本次是否改变了选择、上次是否没存上、草稿是否还脏。
  // 只看“这次有没有差异”会让失败后的第二次点击跳过重试。
  function selectionNeedsSave(changed, pending, dirty) {
    return Boolean(changed || pending || dirty);
  }

  // 选择持久化：需要时保存并等待结果。失败返回 failed，调用方必须留在选择页。
  async function persistSelection(task) {
    const job = task || {};
    const dirty = typeof job.dirty === "function" ? job.dirty() : Boolean(job.dirty);
    if (!selectionNeedsSave(Boolean(job.changed), Boolean(job.pending), dirty)) {
      return { status: "clean" };
    }
    const ok = await job.save();
    return ok ? { status: "saved" } : { status: "failed" };
  }

  // 预览流程是否仍对应当前界面：页面/视频/动物一致、编辑代号没变、
  // 没有未解决的窗口错误、页面未销毁。
  function previewContextAlive(expected, actual) {
    if (!expected || !actual) return false;
    if (actual.closed || actual.blocked) return false;
    return expected.page === actual.page &&
      expected.assetId === actual.assetId &&
      expected.unitId === actual.unitId &&
      expected.edit === actual.edit;
  }

  // 从当前视频的单位列表里移除一只：只剩一只时不允许删除。
  // 返回新的列表与删除后的选中位置，调用方据此清理缓存与预览确认。
  function removeUnit(units, index) {
    if (!Array.isArray(units) || units.length < 2) return null;
    if (!Number.isInteger(index) || index < 0 || index >= units.length) return null;
    const rest = units.slice(0, index).concat(units.slice(index + 1));
    return {
      units: rest,
      removed: units[index],
      index: Math.min(index, rest.length - 1),
    };
  }

  // 只写传入的这一个动物：换鼠后其他动物的窗口不受影响。
  function assignWindow(unit, mode, values) {
    const result = normalizeWindow(mode, values);
    if (result.error) return result;
    if (unit && typeof unit === "object") unit.window = result.window;
    return result;
  }

  return {
    TOOLBAR_HEIGHT,
    HANDLE_HIT_PX,
    EditorCore,
    PreviewGate,
    SaveQueue,
    previewSignature,
    previewNoteFor,
    parseManualVisits,
    previewActionLabel,
    manualFormatIssue,
    issueText,
    setFieldLabels,
    saveThenPreview,
    stableStringify,
    normalizeWindow,
    assignWindow,
    reconcileVideos,
    selectionNeedsSave,
    persistSelection,
    previewContextAlive,
    removeUnit,
    resolveStep,
    nextRequiredStep,
    rotateGeometry,
    mergeEcho,
    pickMedia,
    guardedPick,
    pickBackground,
    viewportToSource,
    moveVertex,
    moveHole,
    moveEllipseCenter,
    moveRadius,
    mwmLines,
    ezmGuides,
    applyFieldTemplate,
    templatePayload,
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
