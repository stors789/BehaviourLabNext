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
    this.mode = spec && spec.modes ? spec.modes[0] : null;
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
    if (this.isEllipseDrag() && spec.type === "ellipse_or_point") {
      return { status: "ignored" };
    }
    if (spec.type === "ellipse" && this.mode !== "points") {
      return { status: "ignored" };
    }

    this.points.push([Number(point.x), Number(point.y)]);
    return this._settle();
  };

  EditorCore.prototype._settle = function () {
    const spec = this.spec;
    const count = Number(spec.count) || 0;
    const fixed = Boolean(spec.fixed);
    const auto = Boolean(spec.auto_commit) || spec.type === "point" ||
      spec.type === "line" || spec.type === "quad" || spec.type === "holes" ||
      (spec.type === "ellipse_or_point" && this.mode === "center");

    if (spec.type === "polygon" && !(fixed && auto)) {
      const ready = this.points.length >= Math.max(3, Number(spec.min_points) || 3);
      return { status: "pending", ready };
    }
    if (spec.type === "ellipse") {
      const ready = this.points.length >= Math.max(5, Number(spec.min_points) || 5);
      return { status: "pending", ready };
    }
    if (count > 0 && this.points.length >= count) {
      return { status: "committed", geometry: this._commit() };
    }
    if (count === 0 && spec.type === "point") {
      return { status: "committed", geometry: this._commit() };
    }
    return { status: "pending", ready: false };
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
    previewSignature,
    stableStringify,
    viewportToSource,
    moveVertex,
    moveHole,
    moveEllipseCenter,
    moveRadius,
  };
});
