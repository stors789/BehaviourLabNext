/* 经典关键点会话：新建、标注、保存与像素误差验证。
   画布按等比 contain 显示，点击逆映射到旋转后的源像素。 */
(function () {
  "use strict";

  const el = {
    form: document.querySelector("#landmark-form"),
    unit: document.querySelector("#landmark-unit"),
    frames: document.querySelector("#landmark-frames"),
    tolerance: document.querySelector("#landmark-tolerance"),
    newError: document.querySelector("[data-new-error]"),
    workbench: document.querySelector("#landmark-workbench"),
    frameSelect: document.querySelector("#landmark-frame"),
    pointSelect: document.querySelector("#landmark-point"),
    canvas: document.querySelector("#landmark-canvas"),
    canvasEmpty: document.querySelector("#landmark-canvas-empty"),
    frameNote: document.querySelector("#landmark-frame-note"),
    supported: document.querySelector("#landmark-supported"),
    toolTolerance: document.querySelector("#tool-tolerance"),
    toolError: document.querySelector("#tool-error"),
    status: document.querySelector("#tool-status"),
    evaluationBox: document.querySelector("#evaluation-box"),
    evaluationBody: document.querySelector("#evaluation-body"),
    evaluationNote: document.querySelector("#evaluation-note"),
  };
  if (!el.workbench) return;

  const POINT_LABELS = {
    body: "身体中心", nose: "鼻尖", tail: "尾根",
    left_ear: "左耳", right_ear: "右耳",
  };
  const state = {
    session: null,
    frame: null,
    point: "body",
    image: null,           // Image 对象
    imageToken: 0,
    imageReady: false,     // 新帧载入完成前，禁止向旧图产生点击坐标
    revision: 0,           // 编辑修订：保存/验证响应用它丢弃过期结果
    context: null,
  };

  function api() {
    return window.BLApi;
  }

  function status(text, tone) {
    if (!el.status) return;
    el.status.textContent = text || "";
    el.status.dataset.tone = tone || "";
  }

  function fail(message) {
    if (el.toolError) el.toolError.textContent = message || "";
  }

  // -- 会话 -------------------------------------------------------------
  function parseFrames(raw) {
    const text = String(raw || "").trim();
    if (!text) return { error: "源帧不能为空" };
    const parts = text.split(",").map((item) => item.trim()).filter(Boolean);
    if (!parts.length) return { error: "源帧不能为空" };
    const seen = new Set();
    const values = [];
    for (const part of parts) {
      if (!/^\d+$/.test(part)) return { error: `源帧必须是非负整数：${part}` };
      const value = Number(part);
      if (seen.has(value)) return { error: `源帧不能重复：${value}` };
      seen.add(value);
      values.push(value);
    }
    return { frames: values.sort((a, b) => a - b) };
  }

  async function createSession(event) {
    event.preventDefault();
    if (el.newError) el.newError.textContent = "";
    const option = el.unit && el.unit.selectedOptions[0];
    if (!option) {
      if (el.newError) el.newError.textContent = "先选择一个视频与动物";
      return;
    }
    const parsed = parseFrames(el.frames && el.frames.value);
    if (parsed.error) {
      if (el.newError) el.newError.textContent = parsed.error;
      return;
    }
    const tolerance = Number(el.tolerance && el.tolerance.value);
    if (!Number.isFinite(tolerance) || tolerance <= 0) {
      if (el.newError) el.newError.textContent = "容差必须是有限正数";
      return;
    }
    // 从选中的草稿/视频/动物条目继承同一份配置快照：视频方向、解释
    // FPS 与 draft→video→unit 覆盖后的完整有效检测（option dataset
    // 由模板按草稿上下文渲染）。会话层带完整有效检测冻结，单位层
    // 保持该单位自己的覆盖原样——与 _settings_of 的覆盖顺序一致。
    const payload = {
      asset_id: option.dataset.asset,
      assay: option.dataset.assay,
      orientation: JSON.parse(option.dataset.orientation || "0"),
      frames: parsed.frames,
      tolerance_px: tolerance,
      unit: {
        id: option.value,
        name: option.dataset.name,
        geometry: JSON.parse(option.dataset.geometry || "{}"),
        options: JSON.parse(option.dataset.options || "{}"),
        detection: JSON.parse(option.dataset.unitDetection || "{}"),
      },
      detection: JSON.parse(option.dataset.detection || "{}"),
      fps_override: JSON.parse(option.dataset.fpsOverride || "null"),
      records: [],
    };
    const revision = state.revision; // 创建期间切走即丢弃迟到响应
    try {
      const data = await api().request("/api/landmarks", {
        method: "POST", body: JSON.stringify(payload),
      });
      if (state.revision !== revision) {
        status("会话已创建；当前选择已变化，未自动打开", "muted");
        return;
      }
      await openSession(data.session);
    } catch (error) {
      if (state.revision !== revision) return;
      if (el.newError) el.newError.textContent = error.message;
    }
  }

  async function openSession(session) {
    state.revision += 1;          // 换会话后，旧会话的迟到响应一律丢弃
    state.session = session;
    state.frame = session.frames[0];
    state.imageReady = false;
    el.workbench.hidden = false;
    if (el.toolTolerance) el.toolTolerance.value = session.tolerance_px;
    renderFrames();
    renderSupported();
    clearEvaluation();
    renderFigure();
    loadFrame();
    status("已载入会话；点击画面标注当前点。");
    el.workbench.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  async function loadSession(id) {
    try {
      const data = await api().request("/api/landmarks/" + id);
      await openSession(data.session);
    } catch (error) {
      status("无法载入会话：" + error.message, "error");
    }
  }

  function renderFrames() {
    if (!el.frameSelect || !state.session) return;
    el.frameSelect.replaceChildren(...state.session.frames.map((frame) => {
      const option = document.createElement("option");
      option.value = String(frame);
      option.textContent = "帧 " + frame;
      if (frame === state.frame) option.selected = true;
      return option;
    }));
  }

  function renderSupported() {
    if (!el.supported || !state.session) return;
    const supported = Array.isArray(state.session.evaluation &&
      state.session.evaluation.supported_points)
      ? state.session.evaluation.supported_points : null;
    const list = supported || (state.session.assay === "nor"
      || state.session.assay === "barnes" ? ["body", "nose", "tail"] : ["body"]);
    el.supported.textContent = "预测支持：" + list.map(
      (name) => POINT_LABELS[name] || name).join("、")
      + "；耳点可标注但无预测。";
  }

  // -- 画布 -------------------------------------------------------------
  function recordOf(frame) {
    const records = (state.session && state.session.records) || [];
    return records.find((item) => item.frame === frame) || null;
  }

  function markOf(frame, point) {
    const record = recordOf(frame);
    return record && record.points ? record.points[point] || null : null;
  }

  function predictionOf(frame, point) {
    const evaluation = state.session && state.session.evaluation;
    if (!evaluation) return null;
    const row = (evaluation.records || []).find((item) => item.frame === frame);
    const entry = row && row.points ? row.points[point] : null;
    return entry && entry.prediction ? entry.prediction : null;
  }

  function setRecord(frame, point, value) {
    const session = state.session;
    state.revision += 1;          // 任何标记修改都使未完成的保存/验证过期
    let record = session.records.find((item) => item.frame === frame);
    if (!record) {
      record = { frame, points: {} };
      session.records.push(record);
      session.records.sort((a, b) => a.frame - b.frame);
    }
    if (value === null) delete record.points[point];
    else record.points[point] = value;
    if (!Object.keys(record.points).length) {
      session.records = session.records.filter((item) => item !== record);
    }
    clearEvaluation();
    renderFigure();
  }

  function renderFigure() {
    const canvas = el.canvas;
    const image = state.image;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!image) {
      canvas.width = 0;
      canvas.height = 0;
      if (el.canvasEmpty) el.canvasEmpty.hidden = false;
      return;
    }
    if (el.canvasEmpty) el.canvasEmpty.hidden = true;
    const dpr = window.devicePixelRatio || 1;
    const box = canvas.parentElement.getBoundingClientRect();
    const width = Math.max(1, box.width - 2);
    const ratio = image.naturalHeight / image.naturalWidth;
    const height = Math.min(Math.max(120, box.height - 2), width * ratio);
    const scale = Math.min(width / image.naturalWidth, height / image.naturalHeight);
    const drawWidth = image.naturalWidth * scale;
    const drawHeight = image.naturalHeight * scale;
    canvas.style.width = drawWidth + "px";
    canvas.style.height = drawHeight + "px";
    canvas.width = Math.round(drawWidth * dpr);
    canvas.height = Math.round(drawHeight * dpr);
    context.setTransform(dpr * scale, 0, 0, dpr * scale, 0, 0);
    context.clearRect(0, 0, image.naturalWidth, image.naturalHeight);
    context.drawImage(image, 0, 0);
    state.context = { scale, drawWidth, drawHeight };

    const frame = state.frame;
    const session = state.session;
    if (!session) return;
    for (const point of Object.keys(POINT_LABELS)) {
      const mark = markOf(frame, point);
      if (mark && mark.visible) {
        context.beginPath();
        context.arc(mark.x, mark.y, 6 / scale, 0, Math.PI * 2);
        context.fillStyle = "rgba(22,104,78,.85)";
        context.fill();
        context.lineWidth = 2 / scale;
        context.strokeStyle = "#ffffff";
        context.stroke();
        label(point, mark.x, mark.y);
      }
      const prediction = predictionOf(frame, point);
      if (prediction) {
        const size = 11 / scale;
        context.strokeStyle = "#1c5f9e";
        context.lineWidth = 2 / scale;
        context.strokeRect(prediction[0] - size / 2, prediction[1] - size / 2,
          size, size);
        label(point + " 预测", prediction[0], prediction[1]);
      }
    }
    function label(text, x, y) {
      context.font = `${12 / scale}px "Segoe UI", "Microsoft YaHei", sans-serif`;
      context.fillStyle = "#142c40";
      context.fillText(text, x + 8 / scale, y - 8 / scale);
    }
  }

  async function loadFrame() {
    if (!state.session) return;
    const token = ++state.imageToken;
    // 新帧在加载完成前保留旧图，但禁止把点击记录到旧帧坐标上。
    state.imageReady = false;
    const session = state.session;
    const url = `/api/assets/${session.asset_id}/frame?index=${state.frame}`
      + `&orientation=${session.orientation}`;
    const image = new Image();
    image.onload = () => {
      // 迟到图片不能覆盖新会话或新帧。
      if (token !== state.imageToken) return;
      state.image = image;
      state.imageReady = true;
      if (el.frameNote) {
        el.frameNote.textContent = `帧 ${state.frame} · ${image.naturalWidth}×${image.naturalHeight}`;
      }
      renderFigure();
    };
    image.onerror = () => {
      if (token !== state.imageToken) return;
      state.image = null;
      state.imageReady = false;
      renderFigure();
      status("这一帧无法载入；不能在这个状态下标注。", "error");
    };
    image.src = url + "&t=" + Date.now();
  }

  function pointFromEvent(event) {
    const canvas = el.canvas;
    const image = state.image;
    if (!canvas || !image) return null;
    const rect = canvas.getBoundingClientRect();
    const scale = Math.min(rect.width / image.naturalWidth,
      rect.height / image.naturalHeight);
    const offsetX = (rect.width - image.naturalWidth * scale) / 2;
    const offsetY = (rect.height - image.naturalHeight * scale) / 2;
    const x = (event.clientX - rect.left - offsetX) / scale;
    const y = (event.clientY - rect.top - offsetY) / scale;
    if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
    if (x < 0 || y < 0 || x >= image.naturalWidth || y >= image.naturalHeight) {
      return null;
    }
    return { x, y };
  }

  // -- 保存与验证 -------------------------------------------------------
  function toleranceValue() {
    const value = Number(el.toolTolerance && el.toolTolerance.value);
    if (!Number.isFinite(value) || value <= 0) return null;
    return value;
  }

  async function save(verify) {
    if (!state.session) return;
    fail("");
    const tolerance = toleranceValue();
    if (tolerance === null) {
      fail("容差必须是有限正数");
      return;
    }
    const sessionId = state.session.id;
    const revision = state.revision;
    const body = {
      records: state.session.records,
      tolerance_px: tolerance,
    };
    try {
      status(verify ? "保存并验证…" : "保存中…");
      const saved = await api().request(
        "/api/landmarks/" + sessionId,
        { method: "PUT", body: JSON.stringify(body) });
      // 迟到响应归属：会话已切换或保存期间又有编辑，一律丢弃，
      // 不覆盖当前会话或标注；失败与过期都保留可修正输入。
      if (!state.session || state.session.id !== sessionId) return;
      if (state.revision !== revision) {
        status("保存期间标注又有修改，已保留当前编辑；需要同步请再次保存。",
               "error");
        return;
      }
      state.session = saved.session;
      status("已保存", "ok");
      if (verify) await evaluate();
    } catch (error) {
      fail(error.message);
      status("保存失败，编辑仍保留在页面上", "error");
    }
  }

  async function evaluate() {
    if (!state.session) return;
    fail("");
    const sessionId = state.session.id;
    const revision = state.revision;
    status("验证中…");
    try {
      const data = await api().request(
        `/api/landmarks/${sessionId}/evaluate`, { method: "POST", body: "{}" });
      if (!state.session || state.session.id !== sessionId) return;
      if (state.revision !== revision) {
        status("验证期间标注已有修改，结果已丢弃；请重新验证。", "error");
        return;
      }
      state.session = data.session;
      renderEvaluation(data.evaluation);
      renderFigure();
      status("验证完成", "ok");
    } catch (error) {
      fail(error.message);
      status("无法验证：" + error.message, "error");
    }
  }

  function clearEvaluation() {
    if (!state.session) return;
    state.session.evaluation = null;
    if (el.evaluationBox) el.evaluationBox.hidden = true;
    if (el.evaluationBody) el.evaluationBody.replaceChildren();
    // 预测图层属于旧评估：清评估必须同步重绘画布，蓝色预测框与
    // “预测”标签立即消失；人工标注保留。
    renderFigure();
  }

  function num(value, digits) {
    if (value === null || value === undefined) return "—";
    if (typeof value !== "number") return String(value);
    return value.toFixed(digits === undefined ? 2 : digits);
  }

  function renderEvaluation(evaluation) {
    if (!evaluation || !el.evaluationBox || !el.evaluationBody) return;
    el.evaluationBox.hidden = false;
    const body = el.evaluationBody;
    body.replaceChildren();
    const points = ["body", "nose", "tail", "left_ear", "right_ear"];
    const supported = new Set(evaluation.supported_points || []);
    for (const point of points) {
      const stat = (evaluation.by_point || {})[point];
      if (!stat) continue;
      const row = document.createElement("tr");
      const cells = [
        POINT_LABELS[point] + (supported.has(point) ? "" : "（不支持）"),
        stat.visible,
        stat.covered,
        stat.visible ? num(stat.covered / stat.visible, 2) : "—",
        stat.visible ? num(stat.hit / stat.visible, 2) : "—",
        "—", "—",
      ];
      if (point === "body" || point === "nose" || point === "tail") {
        // 逐点误差分位来自该点自己的匹配误差；无匹配显示 null，不
        // 复用总体分位数，也不把数值填给没有标注的点。
        cells[5] = num(stat.matched_error_median_px, 2);
        cells[6] = num(stat.matched_error_p90_px, 2);
      }
      cells.forEach((text) => {
        const cell = document.createElement("td");
        cell.textContent = String(text);
        row.append(cell);
      });
      body.append(row);
    }
    if (el.evaluationNote) {
      el.evaluationNote.textContent = (evaluation.note || "")
        + ` 容差 ${num(evaluation.tolerance_px, 2)} px。`;
    }
  }

  // -- 事件 -------------------------------------------------------------
  if (el.form) el.form.addEventListener("submit", createSession);
  document.querySelectorAll("[data-open-session]").forEach((button) => {
    button.addEventListener("click", () => loadSession(button.dataset.openSession));
  });
  if (el.frameSelect) {
    el.frameSelect.addEventListener("change", () => {
      state.frame = Number(el.frameSelect.value);
      state.imageReady = false;   // 新帧加载完成前不记录点击
      clearEvaluation();
      renderFigure();
      loadFrame();
    });
  }
  if (el.pointSelect) {
    el.pointSelect.addEventListener("change", () => {
      state.point = el.pointSelect.value;
      // 切点同样清旧 evaluation 图层与数值，不留上一名的预测框。
      clearEvaluation();
      renderFigure();
    });
  }
  if (el.toolTolerance) {
    el.toolTolerance.addEventListener("input", () => {
      state.revision += 1;
      // 改容差即旧评估整体失效：表格、预测图层与“验证完成”提示
      // 同步复位，人工标注保留；不引入新的异步排列。
      clearEvaluation();
      status("");
    });
  }
  if (el.canvas) {
    el.canvas.addEventListener("click", (event) => {
      if (!state.imageReady) return;   // 画面未加载完成，不产生坐标
      const point = pointFromEvent(event);
      if (!point || !state.session) return;
      setRecord(state.frame, state.point,
        { visible: true, x: point.x, y: point.y });
    });
  }
  const bind = (id, handler) => {
    const node = document.querySelector(id);
    if (node) node.addEventListener("click", handler);
  };
  bind("#tool-visible", () => {
    if (!state.session) return;
    const mark = markOf(state.frame, state.point);
    if (!mark || !mark.visible) {
      fail("先用鼠标在画面上点出这个点");
      return;
    }
    setRecord(state.frame, state.point,
      { visible: true, x: mark.x, y: mark.y });
  });
  bind("#tool-invisible", () => {
    if (!state.session) return;
    setRecord(state.frame, state.point, { visible: false, x: null, y: null });
  });
  bind("#tool-clear", () => {
    if (!state.session) return;
    setRecord(state.frame, state.point, null);
  });
  bind("#tool-save", () => save(false));
  bind("#tool-verify", () => save(true));
  window.addEventListener("resize", () => renderFigure());
})();
