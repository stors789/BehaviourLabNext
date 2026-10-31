/* 分析工作台：选择 → 视频设置 → 场地/检测（或人工记录/复核）。 */
(function () {
  "use strict";

  const root = document.querySelector("[data-analyze]");
  if (!root || !window.BLCore) return;

  const { EditorCore, PreviewGate } = window.BLCore;
  const canvas = document.querySelector("#stage-canvas");
  const ctx = canvas.getContext("2d");

  const state = {
    mode: root.dataset.mode === "independent" ? "independent" : "project",
    projectId: root.dataset.project || "",
    draftId: root.dataset.draft || "",
    assay: root.dataset.assay || "",
    step: root.dataset.step || "select",
    view: root.dataset.view === "detection" ? "detection" : "geometry",
    catalog: [],
    detectionDefaults: {},
    assets: [],
    selected: new Set(),
    draft: null,
    videoIndex: 0,
    unitIndex: 0,
    frame: 0,
    lastFrame: 0,
    image: null,
    imageToken: 0,
    frameInfo: null,
    editor: new EditorCore(),
    stepKey: null,
    saver: null,
    saveTimer: null,
    savedSnapshot: null,
    previewGate: new PreviewGate(),
    loading: false,
    recordMode: false,
    canvasMessage: "正在读取源帧…",
    navToken: 0,
    pendingNav: null,
    positionTimer: null,
    applyingHistory: false,
    draftRequest: null,
  };

  const el = {
    panels: Array.from(document.querySelectorAll("[data-panel]")),
    tasknav: document.querySelector("#tasknav"),
    assay: document.querySelector("#assay-select"),
    project: document.querySelector("#project-select"),
    pickVideos: document.querySelector("#pick-videos"),
    pickFolder: document.querySelector("#pick-folder"),
    dropzone: document.querySelector("#dropzone"),
    importStatus: document.querySelector("#import-status"),
    importErrors: document.querySelector("#import-errors"),
    assetList: document.querySelector("#asset-list"),
    assetEmpty: document.querySelector("#asset-empty"),
    assetCount: document.querySelector("#asset-count"),
    selectedList: document.querySelector("#selected-list"),
    selectedEmpty: document.querySelector("#selected-empty"),
    selectedCount: document.querySelector("#selected-count"),
    setup: document.querySelector("#setup-videos"),
    rangeList: document.querySelector("#range-list"),
    rangeImage: document.querySelector("#range-image"),
    rangeEmpty: document.querySelector("#range-empty"),
    rangeForm: document.querySelector("#range-form"),
    windowStart: document.querySelector("#window-start"),
    windowEnd: document.querySelector("#window-end"),
    orientation: document.querySelector("#orientation"),
    fpsOverride: document.querySelector("#fps-override"),
    sampleFps: document.querySelector("#sample-fps"),
    applyAll: document.querySelector("#apply-all"),
    confirmRange: document.querySelector("#confirm-range"),
    videoSelect: document.querySelector("#video-select"),
    unitSelect: document.querySelector("#unit-select"),
    addUnit: document.querySelector("#add-unit"),
    toolUndo: document.querySelector("#tool-undo"),
    toolRedraw: document.querySelector("#tool-redraw"),
    toolFinish: document.querySelector("#tool-finish"),
    frameRange: document.querySelector("#frame-range"),
    frameNumber: document.querySelector("#frame-number"),
    timeLabel: document.querySelector("#time-label"),
    statusbar: document.querySelector("#statusbar"),
    inspectBody: document.querySelector("#inspect-body"),
    primary: document.querySelector("#primary-action"),
    secondary: document.querySelector("#secondary-action"),
    preview: document.querySelector("#preview-layer"),
    previewNote: document.querySelector("#preview-note"),
  };

  const STATUS = {
    loading: "正在读取源帧…",
  };

  // -- 基础请求 -------------------------------------------------------
  async function request(path, options) {
    const response = await fetch(path, Object.assign({ headers: {} }, options));
    let payload = null;
    try {
      payload = await response.json();
    } catch (error) {
      payload = null;
    }
    if (!response.ok || !payload || payload.ok !== true) {
      const failure = payload && payload.error ? payload.error : {};
      const error = new Error(failure.message || "请求失败（" + response.status + "）");
      error.fields = Array.isArray(failure.fields) ? failure.fields : [];
      error.status = response.status;
      throw error;
    }
    return payload.data;
  }

  const get = (path) => request(path, { method: "GET" });
  const send = (method, path, body) => request(path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  function status(message, tone) {
    el.statusbar.textContent = message || "";
    el.statusbar.dataset.tone = tone || "";
  }

  function clone(value) {
    return value == null ? value : JSON.parse(JSON.stringify(value));
  }

  // -- 草稿与保存 -----------------------------------------------------
  function currentVideo() {
    return state.draft ? state.draft.videos[state.videoIndex] : null;
  }

  function currentUnit() {
    const video = currentVideo();
    return video ? video.units[state.unitIndex] : null;
  }

  function assetById(id) {
    return state.assets.find((asset) => asset.id === id) || null;
  }

  function saver() {
    if (!state.saver) {
      state.saver = new window.BLCore.SaveQueue((snapshot) =>
        send("PUT", "/api/drafts/" + state.draftId, snapshot)
          .then((data) => data.draft));
    }
    return state.saver;
  }

  function markDirty() {
    saver().mark();
    clearTimeout(state.saveTimer);
    state.saveTimer = setTimeout(() => { flushSave(); }, 600);
  }

  function isDirty() {
    return Boolean(state.saver) && state.saver.dirty();
  }

  function invalidatePreview() {
    state.previewGate.invalidate();
    if (el.preview) {
      el.preview.hidden = true;
      el.preview.removeAttribute("src");
    }
    if (el.secondary) el.secondary.hidden = true;
    if (el.primary && state.view === "detection") el.primary.textContent = "检测预览";
    if (state.step === "review" && el.primary) el.primary.textContent = "行为预览";
    setPreviewNote(null);
  }

  // 预览说明：把真实 missing_reason 与 warnings 显示出来，空图不算无说明成功。
  function setPreviewNote(note) {
    if (!el.previewNote) return;
    if (!note) {
      el.previewNote.hidden = true;
      el.previewNote.textContent = "";
      el.previewNote.removeAttribute("data-tone");
      return;
    }
    el.previewNote.hidden = false;
    el.previewNote.dataset.tone = note.tone || "";
    el.previewNote.textContent = note.text;
  }

  function currentSnapshot() {
    return clone({
      videos: state.draft.videos,
      options: state.draft.options,
      detection: state.draft.detection,
      output: state.draft.output,
      style: state.draft.style,
    });
  }

  function flushSave() {
    if (!state.draftId || !state.draft) return Promise.resolve(true);
    return saver().save(currentSnapshot, (draft) => {
      // 原地合并，保持当前 unit 引用有效，避免控件绑定到被替换的旧对象。
      window.BLCore.mergeEcho(state.draft, draft);
      state.savedSnapshot = clone(state.draft);
    }).then((ok) => {
      if (!ok && state.saver && state.saver.lastError) {
        status(state.saver.lastError.message, "error");
      }
      return ok && !isDirty();
    });
  }

  async function saveThen() {
    clearTimeout(state.saveTimer);
    return flushSave();
  }

  // -- 步骤与历史 -----------------------------------------------------
  function stepLabels() {
    if (state.recordMode) return ["视频", "记录", "复核"];
    return ["视频", "场地", "检测"];
  }

  function renderTasknav() {
    if (!el.tasknav) return;
    const labels = stepLabels();
    const links = el.tasknav.querySelectorAll("[data-step-link]");
    const order = state.recordMode ? ["select", "arena", "review"] : ["select", "range", "arena"];
    let activeIndex = 0;
    if (state.step === "range") activeIndex = 1;
    else if (state.step === "arena") activeIndex = state.view === "detection" ? 2 : 1;
    else if (state.step === "review") activeIndex = 2;
    links.forEach((link, index) => {
      const step = order[index];
      link.textContent = labels[index];
      link.dataset.stepLink = step;
      if (index === activeIndex) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    });
  }

  function stepUrl(step, view) {
    const params = new URLSearchParams();
    params.set("mode", state.mode);
    if (state.mode === "project" && state.projectId) params.set("project", state.projectId);
    if (state.draftId) params.set("draft_id", state.draftId);
    params.set("step", step);
    if (step === "arena") params.set("view", view || state.view);
    return "/analyze?" + params.toString();
  }

  function analysisState() {
    return {
      draftId: state.draftId || null,
      step: state.step,
      video: state.videoIndex,
      unit: state.unitIndex,
      frame: state.frame,
      arenaView: state.view,
    };
  }

  function rememberPosition() {
    if (!state.draftId || state.applyingHistory) return;
    history.replaceState({ analysis: analysisState() }, "", location.href);
  }

  function schedulePosition() {
    if (state.applyingHistory) return;
    clearTimeout(state.positionTimer);
    state.positionTimer = setTimeout(rememberPosition, 250);
  }

  async function goStep(step, options) {
    const opts = options || {};
    const leavingEditor = state.step === "arena" || state.step === "review";
    if (leavingEditor) {
      const token = ++state.navToken;
      const ok = await saveThen();
      if (token !== state.navToken) return false;
      if (!ok) {
        status("保存失败，仍停留在当前页", "error");
        return false;
      }
    }
    state.step = step;
    root.dataset.step = step;
    if (opts.view) state.view = opts.view;
    showPanel();
    renderTasknav();
    const entry = { analysis: analysisState() };
    if (opts.replace) history.replaceState(entry, "", stepUrl(step));
    else history.pushState(entry, "", stepUrl(step));
    if (step === "range") renderRange();
    if (step === "arena" || step === "review") {
      ensureContext();
      loadFrame();
      renderInspector();
    }
    return true;
  }

  function showPanel() {
    const key = state.step === "review" ? "arena" : state.step;
    el.panels.forEach((panel) => {
      const match = panel.dataset.panel === key;
      panel.hidden = !match;
      if (match) panel.classList.add("is-active");
      else panel.classList.remove("is-active");
    });
    document.body.classList.toggle("is-workspace", state.step === "arena" || state.step === "review");
  }

  // -- 选择视频 -------------------------------------------------------
  function hasProject() {
    return state.mode !== "project" || Boolean(state.projectId);
  }

  async function loadAssets() {
    const query = state.projectId ? "?project_id=" + encodeURIComponent(state.projectId) : "";
    const data = await get("/api/assets" + query);
    state.assets = data.assets || [];
    if (state.draft) {
      state.draft.videos.forEach((video) => state.selected.add(video.asset_id));
    }
    renderAssets();
  }

  function metaText(asset) {
    const metadata = asset.metadata || {};
    const parts = [];
    if (metadata.width && metadata.height) parts.push(metadata.width + " × " + metadata.height);
    if (metadata.duration_s) parts.push(Math.round(metadata.duration_s) + " 秒");
    return parts.join(" · ");
  }

  function renderAssets() {
    el.assetList.innerHTML = "";
    el.assetCount.textContent = state.assets.length ? state.assets.length + " 个" : "";
    el.assetEmpty.hidden = state.assets.length > 0;
    state.assets.forEach((asset) => {
      const item = document.createElement("li");
      item.className = "asset-row";
      const label = document.createElement("label");
      label.className = "asset-label";
      const box = document.createElement("input");
      box.type = "checkbox";
      box.checked = state.selected.has(asset.id);
      box.disabled = Boolean(state.draft && state.draft.videos.some((v) => v.asset_id === asset.id));
      box.addEventListener("change", () => {
        if (box.checked) state.selected.add(asset.id);
        else state.selected.delete(asset.id);
        renderSelected();
      });
      const name = document.createElement("span");
      name.className = "asset-name";
      name.textContent = asset.name;
      label.append(box, name);
      const meta = document.createElement("span");
      meta.className = "asset-meta";
      meta.textContent = metaText(asset);
      item.append(label, meta);
      el.assetList.append(item);
    });
    renderSelected();
  }

  function renderSelected() {
    const chosen = state.assets.filter((asset) => state.selected.has(asset.id));
    el.selectedList.innerHTML = "";
    el.selectedCount.textContent = chosen.length ? chosen.length + " 个" : "";
    el.selectedEmpty.hidden = chosen.length > 0;
    chosen.forEach((asset) => {
      const item = document.createElement("li");
      item.className = "selected-row";
      const name = document.createElement("span");
      name.className = "selected-name";
      name.textContent = asset.name;
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "remove-button";
      remove.textContent = "移除";
      remove.addEventListener("click", () => {
        state.selected.delete(asset.id);
        renderAssets();
      });
      item.append(name, remove);
      el.selectedList.append(item);
    });
    el.setup.disabled = chosen.length === 0 || state.mode === "project" && !state.projectId;
  }

  function showErrors(errors) {
    el.importErrors.innerHTML = "";
    if (!errors || !errors.length) {
      el.importErrors.hidden = true;
      return;
    }
    errors.forEach((error) => {
      const item = document.createElement("li");
      const name = document.createElement("span");
      name.className = "error-name";
      name.textContent = error.name || "导入失败";
      const message = document.createElement("span");
      message.className = "error-message";
      message.textContent = error.message || "";
      item.append(name, message);
      el.importErrors.append(item);
    });
    el.importErrors.hidden = false;
  }

  function newUnit(name) {
    const bytes = new Uint8Array(16);
    crypto.getRandomValues(bytes);
    const id = Array.from(bytes, (value) => value.toString(16).padStart(2, "0")).join("");
    return {
      id,
      name,
      animal_id: null,
      window: null,
      geometry: {},
      options: {},
      detection: {},
      manual: null,
    };
  }

  function selectionKey() {
    return [state.assay, state.projectId,
      Array.from(state.selected).sort().join(",")].join("|");
  }

  function createDraftOnce() {
    const key = selectionKey();
    const chosen = state.assets.filter((asset) => state.selected.has(asset.id));
    if (!chosen.length) {
      return Promise.reject(new Error("没有选择视频"));
    }
    if (state.draftRequest && state.draftRequest.key === key) {
      return state.draftRequest.promise;
    }
    const videos = chosen.map((asset) => ({
      asset_id: asset.id,
      window: { start_s: 0, end_s: null },
      orientation: 0,
      fps_override: null,
      sample_fps: null,
      units: [newUnit("小鼠 1")],
    }));
    const promise = send("POST", "/api/drafts", {
      assay: state.assay || "tst",
      project_id: state.projectId || null,
      videos,
    }).then((data) => {
      // 只接纳与当前选择一致的创建结果；过期响应不绑定。
      if (state.draftRequest && state.draftRequest.key === key &&
          selectionKey() === key && !state.draftId) {
        state.draft = data.draft;
        state.draftId = data.draft.id;
        state.savedSnapshot = clone(data.draft);
        state.saver = null;
      }
      return data;
    }).finally(() => {
      if (state.draftRequest && state.draftRequest.key === key) {
        state.draftRequest = null;
      }
    });
    state.draftRequest = { key, promise };
    return promise;
  }

  async function setupVideos() {
    if (!state.draftId) {
      const key = selectionKey();
      try {
        await createDraftOnce();
      } catch (error) {
        showErrors(error.fields && error.fields.length
          ? error.fields
          : [{ name: "草稿", message: error.message }]);
        return;
      }
      if (selectionKey() !== key) {
        status("选择已改变，请重新设置视频", "warn");
        return;
      }
      if (!state.draftId) return;
    }
    await goStep("range", { replace: true });
  }

  // -- 视频设置 -------------------------------------------------------
  function renderRange() {
    const videos = state.draft ? state.draft.videos : [];
    el.rangeList.innerHTML = "";
    videos.forEach((video, index) => {
      const asset = assetById(video.asset_id) || { name: "视频 " + (index + 1), metadata: {} };
      const button = document.createElement("button");
      button.type = "button";
      button.className = "range-row" + (index === state.videoIndex ? " is-current" : "");
      const name = document.createElement("span");
      name.className = "range-name";
      name.textContent = asset.name;
      const meta = document.createElement("span");
      meta.className = "range-meta";
      meta.textContent = metaText(asset);
      button.append(name, meta);
      button.addEventListener("click", () => {
        state.videoIndex = index;
        state.unitIndex = 0;
        state.frame = 0;
        renderRange();
      });
      el.rangeList.append(button);
    });

    const video = currentVideo();
    if (!video) return;
    el.windowStart.value = video.window && video.window.start_s != null ? video.window.start_s : 0;
    el.windowEnd.value = video.window && video.window.end_s != null ? video.window.end_s : "";
    el.orientation.value = String(video.orientation || 0);
    el.fpsOverride.value = video.fps_override == null ? "" : video.fps_override;
    el.sampleFps.value = video.sample_fps == null ? "" : video.sample_fps;
    loadRangeFrame(video);
  }

  async function loadRangeFrame(video) {
    el.rangeImage.removeAttribute("src");
    if (el.rangeEmpty) el.rangeEmpty.hidden = false;
    try {
      const response = await fetch(framePath(video, 0));
      if (!response.ok) return;
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      el.rangeImage.onload = () => {
        URL.revokeObjectURL(url);
        if (el.rangeEmpty) el.rangeEmpty.hidden = true;
      };
      el.rangeImage.src = url;
    } catch (error) {
      /* 视频不可读时保留空画面，不显示破图 */
    }
  }

  function framePath(video, index) {
    const params = new URLSearchParams({ index: String(index) });
    params.set("orientation", String(video.orientation || 0));
    if (video.fps_override != null) params.set("fps_override", String(video.fps_override));
    return "/api/assets/" + video.asset_id + "/frame?" + params.toString();
  }

  function numberOrNull(value) {
    if (value === "" || value == null) return null;
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  }

  function readRangeForm() {
    const video = currentVideo();
    if (!video) return false;
    const start = Number(el.windowStart.value || 0);
    const endRaw = el.windowEnd.value;
    const end = endRaw === "" ? null : Number(endRaw);
    if (!Number.isFinite(start) || start < 0) {
      status("开始时间必须是非负数", "error");
      return false;
    }
    if (end !== null && (!Number.isFinite(end) || end <= start)) {
      status("结束时间必须大于开始时间", "error");
      return false;
    }
    const fps = numberOrNull(el.fpsOverride.value);
    if (fps !== null && fps <= 0) {
      status("解释帧率必须大于 0", "error");
      return false;
    }
    const sample = numberOrNull(el.sampleFps.value);
    if (sample !== null && sample <= 0) {
      status("采样帧率必须大于 0", "error");
      return false;
    }
    const previous = video.orientation || 0;
    const next = Number(el.orientation.value || 0);
    if (previous !== next) {
      rotateVideoGeometry(video, previous, next);
    }
    video.window = { start_s: start, end_s: end };
    video.orientation = next;
    video.fps_override = fps;
    video.sample_fps = sample;
    invalidatePreview();
    markDirty();
    return true;
  }

  function rotateVideoGeometry(video, from, to) {
    const metadata = (assetById(video.asset_id) || {}).metadata || {};
    const width = metadata.width || 0;
    const height = metadata.height || 0;
    if (!width || !height) return;
    (video.units || []).forEach((unit) => {
      unit.geometry = rotateGeometry(unit.geometry || {}, width, height, from, to);
    });
  }

  function rotateGeometry(geometry, width, height, from, to) {
    return window.BLCore.rotateGeometry(geometry, width, height, from, to);
  }

  function applyAll() {
    if (!readRangeForm()) return;
    const source = currentVideo();
    state.draft.videos.forEach((video) => {
      if (video === source) return;
      const from = video.orientation || 0;
      const to = source.orientation || 0;
      if (from !== to) {
        // 每个目标按自己的原始宽高换算，不复制当前视频的几何。
        rotateVideoGeometry(video, from, to);
      }
      video.window = clone(source.window);
      video.orientation = to;
      video.fps_override = source.fps_override;
      video.sample_fps = source.sample_fps;
    });
    invalidatePreview();
    markDirty();
    status("已应用到其他视频", "ok");
  }

  async function confirmRange() {
    if (!readRangeForm()) return;
    state.draft.videos.forEach((video, index) => {
      if (!video.units || !video.units.length) {
        video.units = [newUnit("小鼠 1")];
      }
      if (index === state.videoIndex) {
        state.unitIndex = 0;
      }
    });
    markDirty();
    clearTimeout(state.saveTimer);
    const ok = await saveThen();
    if (!ok) {
      status("保存失败，未进入场地", "error");
      return;
    }
    await goStep("arena", { view: "geometry" });
  }

  // -- 工作台上下文 ---------------------------------------------------
  function ensureContext() {
    if (!state.draft || !state.draft.videos.length) return;
    if (state.videoIndex >= state.draft.videos.length) state.videoIndex = 0;
    const video = currentVideo();
    if (!video.units || !video.units.length) video.units = [newUnit("小鼠 1")];
    if (state.unitIndex >= video.units.length) state.unitIndex = 0;
    el.videoSelect.innerHTML = "";
    state.draft.videos.forEach((item, index) => {
      const asset = assetById(item.asset_id);
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = asset ? asset.name : "视频 " + (index + 1);
      option.selected = index === state.videoIndex;
      el.videoSelect.append(option);
    });
    el.unitSelect.innerHTML = "";
    video.units.forEach((unit, index) => {
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = unit.name || "小鼠 " + (index + 1);
      option.selected = index === state.unitIndex;
      el.unitSelect.append(option);
    });
    const meta = (assetById(video.asset_id) || {}).metadata || {};
    const fps = video.fps_override || meta.fps || 25;
    const frames = meta.frame_count || Math.floor((meta.duration_s || 0) * fps) || 0;
    state.lastFrame = Math.max(0, Math.round(frames) - 1);
    el.frameRange.max = String(state.lastFrame);
    el.frameNumber.max = String(state.lastFrame);

    const spec = currentSpec();
    if (spec && state.stepKey !== spec.key) selectStep(spec.key);
    state.recordMode = isManualMode();
    renderTasknav();
  }

  function isManualMode() {
    if (state.assay !== "ymaze") return false;
    const unit = currentUnit();
    const video = currentVideo();
    const options = Object.assign({}, (video && video.options) || {}, (unit && unit.options) || {});
    return options.mode === "manual";
  }

  function assaySpec() {
    return state.catalog.find((item) => item.key === state.assay) || null;
  }

  function geometrySteps() {
    const spec = assaySpec();
    return spec ? spec.geometry || [] : [];
  }

  function currentSpec() {
    const steps = geometrySteps();
    if (!steps.length) return null;
    return steps.find((step) => step.key === state.stepKey) || steps[0];
  }

  function effectiveOptions(unit) {
    const target = unit || currentUnit();
    const video = currentVideo();
    return Object.assign({},
      (state.draft && state.draft.options) || {},
      (video && video.options) || {},
      (target && target.options) || {});
  }

  function selectStep(key) {
    const step = geometrySteps().find((item) => item.key === key);
    if (!step) return;
    state.stepKey = key;
    const unit = currentUnit();
    const resolved = window.BLCore.resolveStep(step, effectiveOptions(unit));
    state.editor.reset(resolved, unit && unit.geometry ? unit.geometry[key] : null);
    syncTools();
    renderInspector();
    draw();
  }

  function syncTools() {
    const unit = currentUnit();
    const hasGeometry = Boolean(unit && unit.geometry && unit.geometry[state.stepKey]);
    const hasPending = state.editor.points.length > 0;
    el.toolFinish.hidden = !state.editor.canFinish();
    el.toolUndo.disabled = !hasGeometry && !hasPending;
    el.toolRedraw.disabled = !hasGeometry && !hasPending;
  }

  function refreshEditorSpec() {
    const step = geometrySteps().find((item) => item.key === state.stepKey);
    if (!step) return;
    const unit = currentUnit();
    const resolved = window.BLCore.resolveStep(step, effectiveOptions(unit));
    state.editor.reset(resolved, unit && unit.geometry ? unit.geometry[state.stepKey] : null);
    syncTools();
  }

  // -- 源帧 -----------------------------------------------------------
  async function loadFrame() {
    const video = currentVideo();
    if (!video) return;
    const token = ++state.imageToken;
    state.loading = true;
    state.canvasMessage = "正在读取源帧…";
    status(STATUS.loading);
    try {
      const response = await fetch(framePath(video, state.frame), { cache: "no-store" });
      if (!response.ok) {
        const payload = await response.json().catch(() => null);
        throw new Error(payload && payload.error ? payload.error.message : "源帧不可读");
      }
      const time = Number(response.headers.get("X-Source-Time"));
      const frameIndex = Number(response.headers.get("X-Source-Frame"));
      const clock = response.headers.get("X-Clock-Source") || "";
      const blob = await response.blob();
      if (token !== state.imageToken) return;
      const url = URL.createObjectURL(blob);
      const image = new Image();
      await new Promise((resolve, reject) => {
        image.onload = resolve;
        image.onerror = reject;
        image.src = url;
      });
      URL.revokeObjectURL(url);
      if (token !== state.imageToken) return;
      state.image = image;
      state.canvasMessage = "";
      invalidatePreview();
      state.frameInfo = { time, frame: frameIndex, clock };
      if (Number.isFinite(frameIndex) && frameIndex !== state.frame) state.frame = frameIndex;
      el.frameRange.value = String(state.frame);
      el.frameNumber.value = String(state.frame);
      el.timeLabel.textContent = Number.isFinite(time)
        ? time.toFixed(3) + " 秒 · " + clock
        : "";
      state.loading = false;
      status("");
      schedulePosition();
      draw();
    } catch (error) {
      if (token !== state.imageToken) return;
      state.loading = false;
      state.image = null;
      state.canvasMessage = "源帧不可读";
      draw();
      const asset = assetById(video.asset_id);
      status("源帧不可读：" + (asset ? asset.name : "视频"), "error");
      el.statusbar.title = error.message;
    }
  }

  function viewport() {
    const rect = canvas.getBoundingClientRect();
    const image = state.image;
    const width = image ? image.naturalWidth : 0;
    const height = image ? image.naturalHeight : 0;
    if (!width || !height) return null;
    const scale = Math.min(rect.width / width, rect.height / height);
    return {
      rect,
      width,
      height,
      scale,
      ox: (rect.width - width * scale) / 2,
      oy: (rect.height - height * scale) / 2,
    };
  }

  function toSource(event) {
    const view = viewport();
    if (!view) return null;
    return window.BLCore.viewportToSource({
      sourceWidth: view.width,
      sourceHeight: view.height,
      cssWidth: view.rect.width,
      cssHeight: view.rect.height,
      left: view.rect.left,
      top: view.rect.top,
    }, event.clientX, event.clientY);
  }

  function resizeCanvas() {
    const rect = canvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const width = Math.max(1, Math.round(rect.width));
    const height = Math.max(1, Math.round(rect.height));
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    canvas.style.width = width + "px";
    canvas.style.height = height + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }

  function toCss(point) {
    const view = viewport();
    if (!view) return [0, 0];
    return [point[0] * view.scale + view.ox, point[1] * view.scale + view.oy];
  }

  function draw() {
    const rect = canvas.getBoundingClientRect();
    ctx.clearRect(0, 0, rect.width, rect.height);
    ctx.fillStyle = "#152432";
    ctx.fillRect(0, 0, rect.width, rect.height);
    const view = viewport();
    if (!view || !state.image) {
      ctx.save();
      ctx.fillStyle = "#93a8b8";
      ctx.font = "13px 'Segoe UI', 'Microsoft YaHei', sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(state.canvasMessage || "没有可显示的源帧", rect.width / 2,
        rect.height / 2);
      ctx.restore();
      return;
    }
    ctx.drawImage(state.image, view.ox, view.oy, view.width * view.scale,
      view.height * view.scale);

    const video = currentVideo();
    if (!video) return;
    video.units.forEach((unit, index) => {
      if (index === state.unitIndex) return;
      Object.values(unit.geometry || {}).forEach((shape) => drawShape(shape, "#8ed7c7", 1.2, false));
    });
    const unit = currentUnit();
    if (!unit) return;
    const shape = unit.geometry ? unit.geometry[state.stepKey] : null;
    if (shape) drawShape(shape, "#ffd26f", 2.8, true);
    drawEditor();
  }

  function drawShape(shape, color, width, handles) {
    if (!shape) return;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = width;
    if (shape.type === "ellipse" && shape.center && shape.radii) {
      const center = toCss(shape.center);
      const rx = shape.radii[0] * viewport().scale;
      const ry = shape.radii[1] * viewport().scale;
      ctx.beginPath();
      ctx.ellipse(center[0], center[1], rx, ry,
        (Number(shape.angle) || 0) * Math.PI / 180, 0, Math.PI * 2);
      ctx.stroke();
      if (handles) {
        handleDot(center);
        handleDot(toCss([shape.center[0] + shape.radii[0], shape.center[1]]));
        handleDot(toCss([shape.center[0], shape.center[1] + shape.radii[1]]));
      }
    } else if (shape.type === "holes") {
      (shape.holes || []).forEach((hole) => {
        const point = toCss(hole.center);
        ctx.beginPath();
        ctx.arc(point[0], point[1], 4, 0, Math.PI * 2);
        ctx.stroke();
        ctx.font = "11px 'Segoe UI', sans-serif";
        ctx.fillText(hole.id, point[0] + 6, point[1] - 4);
      });
    } else if (shape.points && shape.points.length) {
      const points = shape.points.map(toCss);
      ctx.beginPath();
      if (shape.type === "point") {
        ctx.arc(points[0][0], points[0][1], 4, 0, Math.PI * 2);
        ctx.stroke();
      } else {
        points.forEach((point, index) => {
          if (index === 0) ctx.moveTo(point[0], point[1]);
          else ctx.lineTo(point[0], point[1]);
        });
        if (shape.type !== "line") ctx.closePath();
        ctx.stroke();
      }
      if (handles) points.forEach(handleDot);
    }
    ctx.restore();
  }

  function handleDot(point) {
    ctx.beginPath();
    ctx.arc(point[0], point[1], 4, 0, Math.PI * 2);
    ctx.fillStyle = "#fff";
    ctx.fill();
    ctx.strokeStyle = "#ffd26f";
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  function drawEditor() {
    if (!state.editor || !state.editor.spec) return;
    const points = state.editor.points.map(toCss);
    ctx.save();
    ctx.strokeStyle = "#ff6e46";
    ctx.fillStyle = "#ff6e46";
    ctx.lineWidth = 2;
    if (points.length) {
      ctx.beginPath();
      points.forEach((point, index) => {
        if (index === 0) ctx.moveTo(point[0], point[1]);
        else ctx.lineTo(point[0], point[1]);
      });
      ctx.stroke();
      points.forEach((point) => {
        ctx.beginPath();
        ctx.arc(point[0], point[1], 3.5, 0, Math.PI * 2);
        ctx.fill();
      });
    }
    if (state.editor.drag && state.editor.drag.box) {
      const box = state.editor.drag.box;
      const a = toCss([box.x1, box.y1]);
      const b = toCss([box.x2, box.y2]);
      ctx.beginPath();
      ctx.ellipse((a[0] + b[0]) / 2, (a[1] + b[1]) / 2,
        Math.abs(b[0] - a[0]) / 2, Math.abs(b[1] - a[1]) / 2, 0, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.restore();
    syncTools();
  }

  // -- 编辑器交互 -----------------------------------------------------
  function commitGeometry(geometry) {
    const unit = currentUnit();
    if (!unit) return;
    if (!unit.geometry) unit.geometry = {};
    unit.geometry[state.stepKey] = geometry;
    markDirty();
    invalidatePreview();
    syncTools();
    renderInspector();
    draw();
  }

  function addPointAt(event) {
    if (state.view === "detection") return;
    if (state.recordMode) return;
    const spec = currentSpec();
    if (!spec) return;
    const point = toSource(event);
    if (!point) return;
    const unit = currentUnit();
    const existing = unit.geometry ? unit.geometry[state.stepKey] : null;
    if (existing) {
      const hit = state.editor.hitHandle(point, viewport().scale);
      if (!hit) {
        status("该步骤已绘制，拖动手柄或重画", "warn");
        return;
      }
      state.editor.drag = { hit, box: null };
      canvas.setPointerCapture(event.pointerId);
      return;
    }
    if (state.editor.isEllipseDrag()) {
      state.editor.drag = { hit: null, box: { x1: point.x, y1: point.y, x2: point.x, y2: point.y } };
      canvas.setPointerCapture(event.pointerId);
      return;
    }
    const result = state.editor.addPoint(point);
    if (result.status === "committed") {
      commitGeometry(result.geometry);
    } else {
      syncTools();
      renderInspector();
      draw();
    }
  }

  function onPointerMove(event) {
    const drag = state.editor.drag;
    if (!drag) return;
    const point = toSource(event);
    if (!point) return;
    if (drag.box) {
      drag.box.x2 = point.x;
      drag.box.y2 = point.y;
      draw();
      return;
    }
    const unit = currentUnit();
    if (!unit) return;
    const shape = unit.geometry[state.stepKey];
    if (!shape) return;
    const hit = drag.hit;
    if (hit.kind === "vertex") {
      unit.geometry[state.stepKey] = window.BLCore.moveVertex(shape, hit.index, point);
    } else if (hit.kind === "hole") {
      unit.geometry[state.stepKey] = window.BLCore.moveHole(shape, hit.id, point);
    } else if (hit.kind === "center") {
      unit.geometry[state.stepKey] = window.BLCore.moveEllipseCenter(shape, point);
    } else if (hit.kind === "radius") {
      unit.geometry[state.stepKey] = window.BLCore.moveRadius(shape, hit.axis, point);
    }
    draw();
  }

  function onPointerUp(event) {
    const drag = state.editor.drag;
    if (!drag) return;
    state.editor.drag = null;
    if (drag.box) {
      const geometry = state.editor.commitEllipse(drag.box);
      if (geometry) commitGeometry(geometry);
      else {
        status("椭圆过小，已忽略", "warn");
        draw();
      }
      return;
    }
    markDirty();
    state.previewGate.invalidate();
    renderInspector();
    draw();
  }

  // -- 检查器 ---------------------------------------------------------
  function renderInspector() {
    if (!state.draft || !currentVideo()) return;
    const unit = currentUnit();
    if (!unit) return;
    const body = el.inspectBody;
    body.innerHTML = "";
    if (state.recordMode) {
      if (state.step === "review") renderReviewInspector(body, unit);
      else renderRecordInspector(body, unit);
      el.toolFinish.hidden = true;
      return;
    }
    if (state.view === "detection") {
      renderDetectionInspector(body, unit);
      return;
    }

    const steps = geometrySteps();
    const required = steps.filter((step) => step.required);
    const optional = steps.filter((step) => !step.required);
    const drawn = (step) => Boolean(unit.geometry && unit.geometry[step.key]);
    const list = document.createElement("div");
    list.className = "step-list";
    required.forEach((step) => list.append(stepRow(step, drawn(step))));
    body.append(list);
    if (optional.length) {
      const fold = document.createElement("details");
      fold.className = "fold";
      const summary = document.createElement("summary");
      summary.textContent = "可选区域";
      fold.append(summary);
      const inner = document.createElement("div");
      inner.className = "step-list";
      optional.forEach((step) => inner.append(stepRow(step, drawn(step))));
      fold.append(inner);
      body.append(fold);
    }
    appendExperimentSettings(body, unit);
    appendModeControl(body);
    const missing = required.filter((step) => !drawn(step));
    el.primary.textContent = "下一步：检测";
    el.secondary.hidden = true;
    syncTools();
    status(missing.length ? "缺少：" + missing.map((step) => step.label).join("、") : "");
  }

  const MODE_LABELS = {
    center: "标记中心",
    ellipse: "拖出椭圆",
    drag: "拖出椭圆",
    points: "标记边缘点",
  };

  function appendModeControl(body) {
    const spec = state.editor.spec;
    if (!spec || !spec.modes || spec.modes.length < 2) return;
    const wrap = document.createElement("div");
    wrap.className = "mode-row";
    spec.modes.forEach((mode) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "mode-button" + (state.editor.mode === mode ? " is-current" : "");
      button.textContent = MODE_LABELS[mode] || mode;
      button.addEventListener("click", () => {
        if (!state.editor.setMode(mode)) return;
        syncTools();
        renderInspector();
        draw();
        canvas.focus();
      });
      wrap.append(button);
    });
    body.append(wrap);
  }

  function stepRow(step, done) {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "step-row" + (state.stepKey === step.key ? " is-current" : "") +
      (done ? " is-done" : "");
    const label = document.createElement("span");
    label.textContent = step.label;
    const mark = document.createElement("span");
    mark.className = "step-mark";
    mark.textContent = done ? "已绘制" : step.optional ? "可选" : "未绘制";
    row.append(label, mark);
    row.addEventListener("click", () => {
      selectStep(step.key);
      canvas.focus();
    });
    return row;
  }

  function renderDetectionInspector(body, unit) {
    const title = document.createElement("h3");
    title.textContent = "检测";
    body.append(title, detectionFields(unit));
    appendExperimentSettings(body, unit);

    const confirmed = state.previewGate.confirmed(previewConfig("detection"));
    el.primary.textContent = confirmed ? "检查并开始分析" : "检测预览";
    el.secondary.textContent = "更新检测预览";
    el.secondary.hidden = !confirmed;
    el.preview.hidden = !confirmed;
  }

  const BG_METHOD_LABELS = [
    ["threshold", "灰度阈值"],
    ["temporal", "时间中位数"],
    ["static", "静态背景"],
    ["mog2", "MOG2"],
    ["knn", "KNN"],
    ["local_otsu", "局部 Otsu"],
  ];
  const BG_TYPE_LABELS = [["light", "浅底深鼠"], ["dark", "深底浅鼠"]];
  const INTEGER_DETECTION = new Set(["erosion_kernel", "erosion_iter",
    "dilation_kernel", "dilation_iter", "temporal_bg_samples"]);

  function detectionValue(unit, key) {
    const layers = [state.detectionDefaults, state.draft && state.draft.detection,
      currentVideo() && currentVideo().detection, unit && unit.detection];
    let value;
    for (const layer of layers) {
      if (layer && Object.prototype.hasOwnProperty.call(layer, key)) value = layer[key];
    }
    return value;
  }

  function writeDetection(key, value) {
    const target = currentUnit();
    if (!target) return;
    target.detection = Object.assign({}, target.detection || {}, { [key]: value });
    invalidatePreview();
    markDirty();
    renderInspector();
  }

  function selectControl(labelText, key, options, value, onChange) {
    const label = document.createElement("label");
    label.className = "field";
    const caption = document.createElement("span");
    caption.textContent = labelText;
    const select = document.createElement("select");
    options.forEach(([optionValue, optionLabel]) => {
      const option = document.createElement("option");
      option.value = optionValue;
      option.textContent = optionLabel;
      select.append(option);
    });
    select.value = value == null ? options[0][0] : String(value);
    select.addEventListener("change", () => onChange(select.value));
    label.append(caption, select);
    return label;
  }

  function numberControl(labelText, key, value, onChange, options) {
    const config = options || {};
    const label = document.createElement("label");
    label.className = "field";
    const caption = document.createElement("span");
    caption.textContent = labelText;
    const input = document.createElement("input");
    input.type = "number";
    input.step = "any";
    if (config.min != null) input.min = String(config.min);
    if (config.max != null) input.max = String(config.max);
    input.value = value == null ? "" : String(value);
    input.addEventListener("change", () => {
      const number = Number(input.value);
      if (!Number.isFinite(number)) {
        status(labelText + " 必须是数值", "error");
        input.value = String(value);
        return;
      }
      if (config.integer && !Number.isInteger(number)) {
        status(labelText + " 必须是整数", "error");
        input.value = String(value);
        return;
      }
      onChange(number);
    });
    label.append(caption, input);
    return label;
  }

  function detectionFields(unit) {
    const wrap = document.createElement("div");
    wrap.className = "field-stack";
    const method = detectionValue(unit, "bg_method") || "threshold";
    wrap.append(
      selectControl("背景方法", "bg_method", BG_METHOD_LABELS, method,
        (value) => writeDetection("bg_method", value)),
      selectControl("明暗", "bg_type", BG_TYPE_LABELS,
        detectionValue(unit, "bg_type") || "light",
        (value) => writeDetection("bg_type", value)),
    );

    if (method === "threshold") {
      wrap.append(numberControl("灰度阈值", "gray_threshold",
        detectionValue(unit, "gray_threshold"), (value) =>
          writeDetection("gray_threshold", value), { min: 0, max: 255 }));
      wrap.append(numberControl("身体灰度阈值", "body_gray_threshold",
        detectionValue(unit, "body_gray_threshold"), (value) =>
          writeDetection("body_gray_threshold", value), { min: 0, max: 255 }));
    } else if (method === "temporal" || method === "static") {
      wrap.append(numberControl("差异阈值", "temporal_diff_threshold",
        detectionValue(unit, "temporal_diff_threshold"), (value) =>
          writeDetection("temporal_diff_threshold", value), { min: 0, max: 255 }));
      wrap.append(numberControl("身体灰度阈值", "body_gray_threshold",
        detectionValue(unit, "body_gray_threshold"), (value) =>
          writeDetection("body_gray_threshold", value), { min: 0, max: 255 }));
    }

    if (method === "static") {
      const row = document.createElement("div");
      row.className = "field-row-inline";
      const button = document.createElement("button");
      button.type = "button";
      button.className = "button small";
      button.textContent = "选择背景图";
      button.addEventListener("click", () => { pickStaticBackground(); });
      const stateLabel = document.createElement("span");
      stateLabel.className = "meta";
      stateLabel.textContent = detectionValue(unit, "static_background")
        ? "已设置" : "未设置";
      row.append(button, stateLabel);
      wrap.append(row);
    }

    const more = document.createElement("details");
    more.className = "fold";
    const summary = document.createElement("summary");
    summary.textContent = "更多检测参数";
    const body = document.createElement("div");
    body.className = "fold-body";
    const numeric = [
      ["腐蚀核", "erosion_kernel", 1, true],
      ["腐蚀次数", "erosion_iter", 0, true],
      ["膨胀核", "dilation_kernel", 1, true],
      ["膨胀次数", "dilation_iter", 0, true],
      ["最小面积", "min_area", 0, false],
      ["最大面积", "max_area", 0, false],
      ["背景样本数", "temporal_bg_samples", 1, true],
      ["质量阈值", "quality_threshold", 0, false],
    ];
    numeric.forEach(([labelText, key, minimum, integer]) => {
      body.append(numberControl(labelText, key, detectionValue(unit, key),
        (value) => writeDetection(key, value),
        { min: minimum, integer }));
    });
    more.append(summary, body);
    wrap.append(more);
    return wrap;
  }

  async function pickStaticBackground() {
    const bridge = desktopBridge();
    let path = null;
    if (bridge && typeof bridge.choose_image === "function") {
      path = await bridge.choose_image();
    } else if (window.BLPickers) {
      const files = await window.BLPickers.pickFiles({
        accept: ".png,.jpg,.jpeg,.bmp",
      });
      if (files && files.length) {
        const form = new FormData();
        form.append("file", files[0]);
        try {
          const data = await request("/api/backgrounds/upload",
            { method: "POST", body: form });
          path = data.path;
        } catch (error) {
          status(error.message, "error");
          return;
        }
      }
    }
    if (typeof path === "string" && path) {
      writeDetection("static_background", path);
    }
  }

  function renderReviewInspector(body, unit) {
    const title = document.createElement("h3");
    title.textContent = "复核";
    const note = document.createElement("p");
    note.className = "muted";
    note.textContent = "复核人工记录后生成行为预览。";
    body.append(title, note);
    appendPathMode(body);
    appendExperimentSettings(body, unit);

    const events = (unit.manual && unit.manual.events) || [];
    const summary = document.createElement("p");
    summary.className = "muted";
    summary.textContent = events.length ? "已记录 " + events.length + " 条事件" : "还没有记录。";
    body.append(summary);

    const confirmed = state.previewGate.confirmed(previewConfig("behavior"));
    el.primary.textContent = confirmed ? "检查并开始分析" : "行为预览";
    el.secondary.textContent = "更新行为预览";
    el.secondary.hidden = !confirmed;
    el.preview.hidden = !confirmed;
  }

  function appendExperimentSettings(body, unit) {
    const settings = document.createElement("details");
    settings.className = "fold";
    settings.open = false;
    const summary = document.createElement("summary");
    summary.textContent = "实验设置";
    const inner = document.createElement("div");
    inner.className = "fold-body";
    inner.append(parameterFields(unit));
    settings.append(summary, inner);
    body.append(settings);

    const animal = document.createElement("details");
    animal.className = "fold";
    const animalSummary = document.createElement("summary");
    animalSummary.textContent = "小鼠设置";
    const animalBody = document.createElement("div");
    animalBody.className = "fold-body";
    animalBody.append(nameField(unit));
    animal.append(animalSummary, animalBody);
    body.append(animal);
  }

  function appendPathMode(body) {
    if (state.assay !== "ymaze") return;
    const unit = currentUnit();
    const current = (unit && unit.options && unit.options.mode) === "manual"
      ? "manual" : "auto";
    const wrap = document.createElement("div");
    wrap.className = "mode-row";
    [["auto", "自动几何"], ["manual", "人工记录"]].forEach(([value, label]) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "mode-button" + (current === value ? " is-current" : "");
      button.textContent = label;
      button.addEventListener("click", () => { setPathMode(value); });
      wrap.append(button);
    });
    body.append(wrap);
  }

  async function setPathMode(mode) {
    const unit = currentUnit();
    if (!unit) return;
    unit.options = Object.assign({}, unit.options || {}, { mode });
    state.recordMode = mode === "manual";
    invalidatePreview();
    markDirty();
    renderTasknav();
    if (state.step === "review") {
      await goStep("arena", { view: "geometry" });
      return;
    }
    state.view = "geometry";
    renderInspector();
    draw();
  }

  function parameterFields(unit) {
    const wrap = document.createElement("div");
    const params = (assaySpec() || {}).parameters || [];
    if (!params.length) {
      const empty = document.createElement("p");
      empty.className = "muted";
      empty.textContent = "该实验没有额外参数。";
      wrap.append(empty);
      return wrap;
    }
    params.forEach((param) => {
      const label = document.createElement("label");
      label.className = "field";
      const caption = document.createElement("span");
      caption.textContent = param.label + (param.unit ? "（" + param.unit + "）" : "");
      let input;
      if (param.type === "select") {
        input = document.createElement("select");
        (param.choices || []).forEach((choice) => {
          const option = document.createElement("option");
          option.value = choice.value;
          option.textContent = choice.label;
          input.append(option);
        });
      } else if (param.type === "checkbox") {
        input = document.createElement("input");
        input.type = "checkbox";
      } else {
        input = document.createElement("input");
        input.type = "number";
        if (param.min != null) input.min = String(param.min);
        if (param.max != null) input.max = String(param.max);
        if (param.step != null) input.step = String(param.step);
      }
      const value = effectiveOption(unit, param.key, param.default);
      if (input.type === "checkbox") input.checked = Boolean(value);
      else input.value = value == null ? "" : String(value);
      input.addEventListener("change", () => {
        // 始终写入当前草稿的 unit：保存回写后旧闭包引用可能已过期。
        const target = currentUnit();
        if (!target) return;
        if (!target.options) target.options = {};
        let changed = null;
        if (input.type === "checkbox") {
          target.options[param.key] = input.checked;
          changed = input.checked;
        } else if (input.type === "number") {
          const number = Number(input.value);
          if (!Number.isFinite(number)) {
            input.value = String(effectiveOption(target, param.key, param.default));
            return;
          }
          if (param.step === 1 && !Number.isInteger(number)) {
            status(param.label + " 必须是整数", "error");
            input.value = String(effectiveOption(target, param.key, param.default));
            return;
          }
          target.options[param.key] = number;
          changed = number;
        } else {
          target.options[param.key] = input.value;
          changed = input.value;
        }
        invalidatePreview();
        markDirty();
        if (param.key === "mode") {
          state.recordMode = changed === "manual";
          if (state.recordMode) state.view = "geometry";
          renderTasknav();
        }
        refreshEditorSpec();
        renderInspector();
        draw();
      });
      label.append(caption, input);
      wrap.append(label);
    });
    return wrap;
  }

  function effectiveOption(unit, key, fallback) {
    if (unit.options && key in unit.options) return unit.options[key];
    const video = currentVideo();
    if (video && video.options && key in video.options) return video.options[key];
    if (state.draft.options && key in state.draft.options) return state.draft.options[key];
    return fallback;
  }

  function nameField(unit) {
    const label = document.createElement("label");
    label.className = "field";
    const caption = document.createElement("span");
    caption.textContent = "名称";
    const input = document.createElement("input");
    input.type = "text";
    input.value = unit.name || "";
    input.addEventListener("change", () => {
      const target = currentUnit();
      if (!target) return;
      target.name = input.value.trim() || "小鼠 " + (state.unitIndex + 1);
      invalidatePreview();
      markDirty();
      ensureContext();
    });
    label.append(caption, input);
    return label;
  }

  // -- 人工记录（Y 迷宫人工路径） --------------------------------------
  function renderRecordInspector(body, unit) {
    const title = document.createElement("h3");
    title.textContent = "人工记录";
    const note = document.createElement("p");
    note.className = "muted";
    note.textContent = "按当前帧记录事件；播放器在第 16 块接入。";
    body.append(title, note);
    appendPathMode(body);

    const record = document.createElement("button");
    record.className = "button full";
    record.type = "button";
    record.textContent = "记录当前帧事件";
    record.addEventListener("click", () => {
      const target = currentUnit();
      if (!target) return;
      if (!target.manual) target.manual = { events: [] };
      if (!Array.isArray(target.manual.events)) target.manual.events = [];
      const time = state.frameInfo && Number.isFinite(state.frameInfo.time)
        ? state.frameInfo.time : null;
      if (time == null) {
        status("当前帧时间不可用", "error");
        return;
      }
      target.manual.events.push({ time_s: time, frame: state.frame, kind: "event" });
      invalidatePreview();
      markDirty();
      renderInspector();
    });
    body.append(record);

    const list = document.createElement("ul");
    list.className = "event-list";
    const events = (unit.manual && unit.manual.events) || [];
    if (!events.length) {
      const empty = document.createElement("li");
      empty.className = "muted";
      empty.textContent = "还没有记录。";
      list.append(empty);
    }
    events.forEach((event, index) => {
      const item = document.createElement("li");
      const label = document.createElement("span");
      label.textContent = "帧 " + event.frame + " · " + Number(event.time_s).toFixed(2) + " 秒";
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "remove-button";
      remove.textContent = "移除";
      remove.addEventListener("click", () => {
        const target = currentUnit();
        const live = (target && target.manual && target.manual.events) || [];
        live.splice(index, 1);
        invalidatePreview();
        markDirty();
        renderInspector();
      });
      item.append(label, remove);
      list.append(item);
    });
    body.append(list);
    el.primary.textContent = "下一步：复核";
    el.secondary.hidden = true;
  }

  // -- 预览与校验 -----------------------------------------------------
  function previewConfig(kind) {
    const video = currentVideo();
    const unit = currentUnit();
    return {
      kind: kind || "detection",
      asset_id: video.asset_id,
      frame_index: state.frame,
      orientation: video.orientation || 0,
      fps_override: video.fps_override,
      sample_fps: video.sample_fps,
      window: clone(video.window),
      unit: clone(unit),
      draft_options: clone(state.draft.options || {}),
      draft_detection: clone(state.draft.detection || {}),
      video_options: clone(video.options || {}),
      video_detection: clone(video.detection || {}),
    };
  }

  async function runPreview(kind) {
    if (!state.draftId) return;
    // 只接受显式的阶段名；事件对象等非字符串一律按当前阶段处理。
    const type = typeof kind === "string"
      ? kind : (state.step === "review" ? "behavior" : "detection");
    const video = currentVideo();
    const unit = currentUnit();
    if (!video || !unit) return;
    el.primary.disabled = true;
    el.secondary.disabled = true;
    status(type === "behavior" ? "正在生成行为预览…" : "正在生成检测预览…");
    let request = null;
    try {
      // 先落盘，再核对页面仍指向同一个视频/小鼠，最后才请求预览。
      const result = await window.BLCore.saveThenPreview({
        dirty: isDirty,
        save: saveThen,
        alive: () => {
          const live = currentVideo();
          const liveUnit = currentUnit();
          return Boolean(live && liveUnit && live.asset_id === video.asset_id &&
            liveUnit.id === unit.id);
        },
        config: () => previewConfig(type),
        send: (config) => {
          request = state.previewGate.begin(config);
          return send("POST", "/api/drafts/" + state.draftId + "/preview", {
            asset_id: config.asset_id,
            unit_id: config.unit.id,
            frame_index: config.frame_index,
            kind: config.kind,
          });
        },
      });
      if (result.status === "save_failed") {
        state.previewGate.invalidate();
        status("保存失败，未生成预览", "error");
        setPreviewNote({ tone: "error", text: "保存失败，未生成预览。" });
        return;
      }
      if (result.status === "stale") {
        state.previewGate.invalidate();
        status("配置已改变，请重新预览", "warn");
        setPreviewNote({ tone: "warn", text: "配置已改变，请重新预览。" });
        return;
      }
      const data = result.data;
      if (!state.previewGate.accept(request, data)) return;
      el.preview.src = data.image_url;
      el.preview.hidden = false;
      setPreviewNote(window.BLCore.previewNoteFor(data));
      status((data.preview_scope && data.preview_scope.mode === "frame"
        ? "本帧预览" : "片段预览") + " · 帧 " + data.frame_index, "ok");
    } catch (error) {
      if (request) state.previewGate.fail(request);
      setPreviewNote({ tone: "error", text: "预览失败：" + error.message });
      status(error.message, "error");
    } finally {
      el.primary.disabled = false;
      el.secondary.disabled = false;
      renderInspector();
    }
  }

  async function primaryAction() {
    if (state.step === "review") {
      if (state.previewGate.confirmed(previewConfig("behavior"))) await startAnalysis();
      else await runPreview("behavior");
      return;
    }
    if (state.recordMode) {
      await goStep("review");
      return;
    }
    if (state.view === "geometry") {
      const unit = currentUnit();
      const missing = geometrySteps().filter((step) =>
        step.required && !(unit.geometry && unit.geometry[step.key]));
      if (missing.length) {
        selectStep(missing[0].key);
        status("缺少：" + missing[0].label, "warn");
        canvas.focus();
        return;
      }
      await goStep("arena", { view: "detection" });
      return;
    }
    if (state.previewGate.confirmed(previewConfig("detection"))) {
      await startAnalysis();
      return;
    }
    await runPreview("detection");
  }

  async function startAnalysis() {
    if (!state.draftId) return;
    const kind = state.step === "review" ? "behavior" : "detection";
    if (!state.previewGate.confirmed(previewConfig(kind))) {
      status("配置已改变，请重新预览", "warn");
      renderInspector();
      return;
    }
    const settled = await saveThen();
    if (!settled) {
      status("保存失败，未开始分析", "error");
      return;
    }
    try {
      const data = await get("/api/drafts/" + state.draftId + "/validate");
      const issues = data.issues || [];
      if (issues.length) {
        status(issues.map((issue) => issue.message || issue.field).join("；"), "error");
        return;
      }
      status("校验通过；分析执行将在后续块接入。", "ok");
    } catch (error) {
      status(error.message, "error");
    }
  }

  // -- 事件绑定 -------------------------------------------------------
  function desktopBridge() {
    return window.pywebview && window.pywebview.api ? window.pywebview.api : null;
  }

  function mergeAssets(assets, errors) {
    showErrors(errors || []);
    (assets || []).forEach((asset) => {
      if (!state.assets.some((item) => item.id === asset.id)) state.assets.push(asset);
      state.selected.add(asset.id);
    });
    if (!state.draft) renderAssets();
  }

  async function importPaths(paths) {
    const list = (paths || []).filter((item) => typeof item === "string" && item.trim());
    if (!list.length) return;
    el.importStatus.hidden = false;
    el.importStatus.textContent = "正在导入视频…";
    try {
      const data = await send("POST", "/api/assets/import", {
        paths: list,
        project_id: state.projectId || null,
      });
      mergeAssets(data.assets, data.errors);
    } catch (error) {
      showErrors([{ name: "导入失败", message: error.message }]);
    } finally {
      el.importStatus.hidden = true;
    }
  }

  async function pickVideos() {
    await window.BLCore.pickMedia("videos", {
      bridge: desktopBridge(),
      picker: window.BLPickers,
      onPaths: importPaths,
      onFiles: uploadFiles,
    });
  }

  async function pickFolder() {
    await window.BLCore.pickMedia("folder", {
      bridge: desktopBridge(),
      picker: window.BLPickers,
      onPaths: importPaths,
      onFiles: uploadFiles,
    });
  }

  function bind() {
    el.pickVideos.addEventListener("click", () => { pickVideos(); });
    el.pickFolder.addEventListener("click", () => { pickFolder(); });
    ["dragenter", "dragover"].forEach((name) => {
      el.dropzone.addEventListener(name, (event) => {
        event.preventDefault();
        el.dropzone.classList.add("is-dragover");
      });
    });
    ["dragleave", "drop"].forEach((name) => {
      el.dropzone.addEventListener(name, (event) => {
        event.preventDefault();
        if (name === "dragleave" && el.dropzone.contains(event.relatedTarget)) return;
        el.dropzone.classList.remove("is-dragover");
      });
    });
    el.dropzone.addEventListener("drop", async (event) => {
      const files = Array.from(event.dataTransfer.files || []);
      if (files.length) await uploadFiles(files);
    });

    if (el.assay) {
      el.assay.addEventListener("change", () => {
        state.assay = el.assay.value;
        if (state.draftId) {
          state.draftId = "";
          state.draft = null;
          state.selected.clear();
          history.replaceState({ step: "select" }, "", stepUrl("select"));
          status("已切换实验，需重新创建草稿", "warn");
        }
        loadAssets();
      });
    }
    if (el.project) {
      el.project.addEventListener("change", () => {
        state.projectId = el.project.value;
        loadAssets();
      });
    }
    el.setup.addEventListener("click", setupVideos);

    el.applyAll.addEventListener("click", applyAll);
    el.confirmRange.addEventListener("click", confirmRange);

    el.videoSelect.addEventListener("change", () => {
      state.videoIndex = Number(el.videoSelect.value);
      state.unitIndex = 0;
      state.frame = 0;
      state.stepKey = null;
      state.editor.reset(null, null);
      ensureContext();
      loadFrame();
      renderInspector();
    });
    el.unitSelect.addEventListener("change", () => {
      state.unitIndex = Number(el.unitSelect.value);
      state.stepKey = null;
      state.view = "geometry";
      invalidatePreview();
      ensureContext();
      renderTasknav();
      renderInspector();
      draw();
    });
    el.addUnit.addEventListener("click", () => {
      const video = currentVideo();
      if (!video) return;
      video.units.push(newUnit("小鼠 " + (video.units.length + 1)));
      state.unitIndex = video.units.length - 1;
      state.stepKey = null;
      state.view = "geometry";
      invalidatePreview();
      markDirty();
      ensureContext();
      renderTasknav();
      renderInspector();
      draw();
    });

    el.toolUndo.addEventListener("click", () => {
      const result = state.editor.undo();
      const unit = currentUnit();
      if (result === "geometry" && unit) {
        delete unit.geometry[state.stepKey];
        markDirty();
        state.previewGate.invalidate();
      } else if (result === "points") {
        renderInspector();
      }
      draw();
    });
    el.toolRedraw.addEventListener("click", () => {
      const unit = currentUnit();
      state.editor.redraw();
      if (unit && unit.geometry) {
        delete unit.geometry[state.stepKey];
        markDirty();
      }
      renderInspector();
      draw();
    });
    el.toolFinish.addEventListener("click", () => {
      const geometry = state.editor.finish();
      if (geometry) commitGeometry(geometry);
    });

    canvas.addEventListener("pointerdown", addPointAt);
    canvas.addEventListener("pointermove", onPointerMove);
    canvas.addEventListener("pointerup", onPointerUp);
    canvas.addEventListener("pointercancel", () => {
      state.editor.drag = null;
      draw();
    });

    el.frameRange.addEventListener("input", () => {
      state.frame = Number(el.frameRange.value);
      loadFrame();
    });
    el.frameRange.addEventListener("change", () => {
      state.frame = Number(el.frameRange.value);
      loadFrame();
      rememberPosition();
    });
    el.frameNumber.addEventListener("keydown", (event) => {
      if (event.key !== "Enter") return;
      event.preventDefault();
      commitFrameNumber();
      el.frameNumber.blur();
    });
    el.frameNumber.addEventListener("blur", commitFrameNumber);

    el.primary.addEventListener("click", primaryAction);
    el.secondary.addEventListener("click", () => {
      runPreview(state.step === "review" ? "behavior" : "detection");
    });

    if (el.tasknav) {
      el.tasknav.addEventListener("click", (event) => {
        const link = event.target.closest("[data-step-link]");
        if (!link) return;
        event.preventDefault();
        const links = Array.from(el.tasknav.querySelectorAll("[data-step-link]"));
        const index = links.indexOf(link);
        const step = link.dataset.stepLink;
        const targetView = step === "arena"
          ? (index === 2 ? "detection" : "geometry")
          : state.view;
        if (step === state.step && (step !== "arena" || targetView === state.view)) return;
        goStep(step, { view: targetView });
      });
    }

    window.addEventListener("popstate", async (event) => {
      const target = event.state && event.state.analysis;
      const token = ++state.navToken;
      const ok = await saveThen();
      if (token !== state.navToken) return;
      if (!ok) {
        history.pushState({ analysis: analysisState() }, "", location.href);
        status("保存失败，未切换历史记录", "error");
        return;
      }
      if (!target) return;
      if (target.draftId && target.draftId !== state.draftId) {
        location.reload();
        return;
      }
      state.applyingHistory = true;
      try {
        state.step = target.step || "select";
        root.dataset.step = state.step;
        state.videoIndex = Number.isInteger(target.video) ? target.video : 0;
        state.unitIndex = Number.isInteger(target.unit) ? target.unit : 0;
        state.frame = Number.isInteger(target.frame) ? target.frame : 0;
        state.view = target.arenaView === "detection" ? "detection" : "geometry";
        showPanel();
        renderTasknav();
        if (state.step === "range") renderRange();
        if (state.step === "arena" || state.step === "review") {
          ensureContext();
          loadFrame();
          renderInspector();
        }
      } finally {
        state.applyingHistory = false;
      }
    });

    document.addEventListener("click", async (event) => {
      if (event.defaultPrevented || event.button !== 0) return;
      if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      const link = event.target.closest("a[href]");
      if (!link || link.target === "_blank" || link.hasAttribute("download")) return;
      const url = new URL(link.getAttribute("href"), location.href);
      if (url.origin !== location.origin) return;
      if (url.pathname === location.pathname && url.search === location.search &&
          url.hash === location.hash) {
        return;
      }
      event.preventDefault();
      const token = ++state.navToken;
      state.pendingNav = { token, href: url.href };
      const ok = await saveThen();
      if (token !== state.navToken) return;
      state.pendingNav = null;
      if (!ok) {
        status("保存失败，仍留在当前页", "error");
        return;
      }
      location.href = url.href;
    });

    document.addEventListener("keydown", (event) => {
      if (event.key !== "Escape" || !state.pendingNav) return;
      state.navToken += 1;
      state.pendingNav = null;
      status("已取消跳转", "warn");
    });

    window.addEventListener("beforeunload", (event) => {
      rememberPosition();
      if (!isDirty()) return;
      event.preventDefault();
      event.returnValue = "";
    });
    window.addEventListener("resize", resizeCanvas);
    if (window.ResizeObserver) {
      new ResizeObserver(() => resizeCanvas()).observe(canvas.parentElement);
    }
  }

  function commitFrameNumber() {
    const raw = el.frameNumber.value.trim();
    if (raw === "") {
      el.frameNumber.value = String(state.frame);
      return;
    }
    const value = Number(raw);
    if (!Number.isFinite(value)) {
      el.frameNumber.value = String(state.frame);
      return;
    }
    const next = Math.min(state.lastFrame, Math.max(0, Math.round(value)));
    el.frameNumber.value = String(next);
    if (next === state.frame) return;
    state.frame = next;
    el.frameRange.value = String(next);
    loadFrame();
    rememberPosition();
  }

  async function uploadFiles(files) {
    if (!files.length) return;
    el.importStatus.hidden = false;
    el.importStatus.textContent = "正在导入视频…";
    el.pickVideos.disabled = true;
    el.pickFolder.disabled = true;
    const form = new FormData();
    files.forEach((file) => form.append("files", file));
    if (state.projectId) form.append("project_id", state.projectId);
    try {
      const data = await request("/api/assets/upload", { method: "POST", body: form });
      const failures = data.errors || [];
      showErrors(failures);
      const added = data.assets || [];
      added.forEach((asset) => {
        if (!state.assets.some((item) => item.id === asset.id)) state.assets.push(asset);
        state.selected.add(asset.id);
      });
      if (!state.draft) renderAssets();
    } catch (error) {
      showErrors([{ name: "导入失败", message: error.message }]);
    } finally {
      el.importStatus.hidden = true;
      el.pickVideos.disabled = false;
      el.pickFolder.disabled = false;
    }
  }

  // -- 启动 -----------------------------------------------------------
  async function boot() {
    bind();
    showPanel();
    renderTasknav();
    try {
      const catalog = await get("/api/catalog");
      state.catalog = catalog.assays || catalog.catalog || catalog;
      if (!Array.isArray(state.catalog)) state.catalog = [];
      state.detectionDefaults = catalog.detection_defaults || {};
    } catch (error) {
      state.catalog = [];
      state.detectionDefaults = {};
    }
    if (el.assay && !el.assay.value && state.catalog.length) {
      el.assay.value = state.catalog[0].key;
    }
    if (!state.assay) state.assay = el.assay ? el.assay.value : "tst";

    if (state.draftId) {
      try {
        const data = await get("/api/drafts/" + state.draftId);
        state.draft = data.draft;
        state.savedSnapshot = clone(data.draft);
        state.saver = null;
        state.assay = data.draft.assay;
        if (data.draft.project_id) state.projectId = data.draft.project_id;
        if (el.assay) el.assay.value = state.assay;
      } catch (error) {
        status(error.message, "error");
      }
    }
    const entry = history.state && history.state.analysis;
    if (entry && entry.draftId === state.draftId) {
      state.videoIndex = Number.isInteger(entry.video) ? entry.video : 0;
      state.unitIndex = Number.isInteger(entry.unit) ? entry.unit : 0;
      state.frame = Number.isInteger(entry.frame) ? entry.frame : 0;
      if (state.step === "arena") {
        state.view = entry.arenaView === "detection" ? "detection" : "geometry";
      }
    }
    await loadAssets();
    if (state.step === "range") renderRange();
    if (state.step === "arena" || state.step === "review") {
      ensureContext();
      resizeCanvas();
      loadFrame();
      renderInspector();
    }
  }

  boot();
})();
