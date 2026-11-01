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
    shownFrame: 0,
    // 视频设置页自己的源帧位置：与场地画布的当前帧互不干扰。
    rangeFrame: 0,
    rangeToken: 0,
    image: null,
    imageToken: 0,
    frameInfo: null,
    editor: new EditorCore(),
    stepKey: null,
    saver: null,
    saveTimer: null,
    savedSnapshot: null,
    previewGate: new PreviewGate(),
    previewKind: "detection",
    previewAwaiting: false,
    pageClosed: false,
    loading: false,
    recordMode: false,
    canvasMessage: "正在读取源帧…",
    navToken: 0,
    pendingNav: null,
    applyingHistory: false,
    draftRequest: null,
    playing: false,
    playToken: 0,
    playTimer: null,
    playWait: null,
    playLastTime: null,
    rate: 1,
    editCounter: 0,
    // 人工洞序列的编辑文本与格式错误按动物缓存，互不影响。
    manualCache: new Map(),
    // 动物时间窗的未解决编辑按“视频+动物”保存：模式、原始文本与字段错误。
    windowEdits: new Map(),
    // 状态栏里人工格式错误与窗口错误的归属，用于只清理自己那条旧提示。
    manualStatus: null,
    windowStatus: null,
  };

  const el = {
    panels: Array.from(document.querySelectorAll("[data-panel]")),
    topContext: document.querySelector(".top-context"),
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
    setupNote: document.querySelector("#setup-note"),
    selectStatus: document.querySelector("#select-status"),
    rangeStatus: document.querySelector("#range-status"),
    rangeList: document.querySelector("#range-list"),
    rangeImage: document.querySelector("#range-image"),
    rangeEmpty: document.querySelector("#range-empty"),
    rangeForm: document.querySelector("#range-form"),
    rangeVideoName: document.querySelector("#range-video-name"),
    rangeFpsNote: document.querySelector("#range-fps-note"),
    rangeSlide: document.querySelector("#range-frame-range"),
    rangeIndex: document.querySelector("#range-frame-number"),
    rangeTime: document.querySelector("#range-time-label"),
    backToSelect: document.querySelector("#back-to-select"),
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
    toolTitle: document.querySelector("#toolhead-title"),
    frameRange: document.querySelector("#frame-range"),
    frameNumber: document.querySelector("#frame-number"),
    timeLabel: document.querySelector("#time-label"),
    statusbar: document.querySelector("#statusbar"),
    toolhead: document.querySelector("#toolhead"),
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

  // 状态写到当前阶段真正可见的那一条：选择/视频设置页没有画布状态栏。
  function statusNode() {
    if (state.step === "select") return el.selectStatus || el.statusbar;
    if (state.step === "range") return el.rangeStatus || el.statusbar;
    return el.statusbar;
  }

  function status(message, tone) {
    const node = statusNode();
    node.textContent = message || "";
    node.dataset.tone = tone || "";
  }

  // 校验失败优先显示字段级原因；通用文案只在没有具体字段时兜底。
  function issueText(error, fallback) {
    return window.BLCore.issueText(error, fallback);
  }

  function lastSaveText(fallback) {
    return issueText(state.saver && state.saver.lastError, fallback);
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
    // 只有影响分析内容的改变才推进编辑代号：折叠/输出开关不使预览失效。
    state.editCounter += 1;
    state.previewGate.invalidate();
    state.previewAwaiting = false;
    if (el.preview) {
      el.preview.hidden = true;
      el.preview.removeAttribute("src");
    }
    if (el.secondary) el.secondary.hidden = true;
    if (el.primary && state.view === "detection") {
      // 文字必须跟随下一次真正会请求的 kind，不能回落到检测。
      el.primary.textContent = window.BLCore.previewActionLabel(state.previewKind);
    }
    if (state.step === "review" && el.primary) {
      el.primary.textContent = window.BLCore.previewActionLabel("behavior");
    }
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
        status(issueText(state.saver.lastError, "保存失败"), "error");
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

  // 顶部上下文不依赖整页重载：新建草稿后立即换成实验名 + 任务导航，
  // 切实验清掉草稿时再退回页面名。
  function ensureTopContext() {
    const wrap = el.topContext;
    if (!wrap) return;
    if (!state.draftId) {
      if (!wrap.querySelector(".page-context")) {
        wrap.innerHTML = "";
        const line = document.createElement("p");
        line.className = "page-context";
        line.textContent = state.mode === "independent" ? "独立分析" : "添加项目分析";
        wrap.append(line);
      }
      el.tasknav = null;
      return;
    }
    let task = wrap.querySelector(".top-task");
    if (!task) {
      wrap.innerHTML = "";
      task = document.createElement("div");
      task.className = "top-task";
      const assayNode = document.createElement("span");
      assayNode.className = "top-assay";
      const nav = document.createElement("nav");
      nav.className = "tasknav";
      nav.id = "tasknav";
      nav.setAttribute("aria-label", "分析步骤");
      ["视频", "场地", "检测"].forEach((label, index) => {
        if (index) {
          const sep = document.createElement("span");
          sep.className = "tasknav-sep";
          sep.setAttribute("aria-hidden", "true");
          sep.textContent = "›";
          nav.append(sep);
        }
        const link = document.createElement("a");
        link.href = "#";
        link.dataset.stepLink = "select";
        link.textContent = label;
        nav.append(link);
      });
      task.append(assayNode, nav);
      wrap.append(task);
    }
    el.tasknav = task.querySelector("#tasknav");
    const label = (state.catalog.find((item) => item.key === state.assay) || {}).label || "";
    const assayNode = task.querySelector(".top-assay");
    if (assayNode) {
      assayNode.textContent = label;
      assayNode.hidden = !label;
    }
  }

  // 任务导航的三段与真实阶段一致：选择与视频设置同属“视频”，
  // “场地”“检测”才进画布的两个视图。
  function tasknavEntries() {
    if (state.recordMode) {
      return [{ step: "select" }, { step: "arena" }, { step: "review" }];
    }
    return [
      { step: "select" },
      { step: "arena", view: "geometry" },
      { step: "arena", view: "detection" },
    ];
  }

  function renderTasknav() {
    if (!el.tasknav) return;
    const labels = stepLabels();
    const links = el.tasknav.querySelectorAll("[data-step-link]");
    const entries = tasknavEntries();
    let activeIndex = 0;
    if (state.step === "arena") activeIndex = state.view === "detection" ? 2 : 1;
    else if (state.step === "review") activeIndex = 2;
    links.forEach((link, index) => {
      const entry = entries[index] || entries[0];
      link.textContent = labels[index];
      link.dataset.stepLink = entry.step;
      link.dataset.stepView = entry.view || "";
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

  function pageKey() {
    // 页面上下文身份：离开这个组合后，旧帧与旧预览响应都不能再写状态。
    return [
      state.draftId || "",
      state.step || "",
      state.videoIndex,
      state.unitIndex,
    ].join("|");
  }

  function pageAlive(context) {
    // 已销毁的页面或已换页的上下文都不再接受异步结果。
    return Boolean(context) && !state.pageClosed
      && pageKey() === context.page;
  }

  function analysisState() {
    return {
      draftId: state.draftId || null,
      step: state.step,
      video: state.videoIndex,
      unit: state.unitIndex,
      // 记录实际显示过的帧；请求了但没显示出来的帧不写进历史。
      frame: Number.isInteger(state.shownFrame) ? state.shownFrame : state.frame,
      // 视频设置页有自己的一份已显示源帧，与画布帧分开保存。
      rangeFrame: Number.isInteger(state.rangeFrame) ? state.rangeFrame : 0,
      arenaView: state.view,
    };
  }

  function rememberPosition(step, view) {
    // 立即写当前历史项：没有延迟回调，后退/前进时不会被旧页写回。
    if (!state.draftId || state.applyingHistory) return;
    history.replaceState({ analysis: analysisState() }, "",
      step ? stepUrl(step, view) : location.href);
  }

  async function goStep(step, options) {
    const opts = options || {};
    // 未解决的窗口编辑不得被带进下一阶段；回视频/视频设置仍放行，
    // 否则用户无法去修另一个视频的字段。
    const advancing = step === "arena" || step === "review";
    if (advancing && blockOnWindowError()) return false;
    stopPlayback();
    const leavingEditor = state.step === "arena" || state.step === "review";
    if (leavingEditor) {
      // 先把这个画布最后实际显示的帧写回原历史项，再切换到目标阶段；
      // 连续播放时提交不写 history，这里补上，后退/刷新才不会回到旧帧。
      rememberPosition();
      const token = ++state.navToken;
      const ok = await saveThen();
      if (token !== state.navToken) return false;
      if (!ok) {
        status(lastSaveText("保存失败，仍停留在当前页"), "error");
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
      // 草稿里的视频只在“当前仍选中”时锁定勾选；已移除的可以重新勾回来，
      // 不给“移除后无法再选”的死角。
      box.disabled = Boolean(state.draft && state.selected.has(asset.id) &&
        state.draft.videos.some((v) => v.asset_id === asset.id));
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

  function assetThumbPath(asset) {
    return "/api/assets/" + asset.id + "/frame?index=0&orientation=0";
  }

  function renderSelected() {
    const chosen = state.assets.filter((asset) => state.selected.has(asset.id));
    el.selectedList.innerHTML = "";
    el.selectedCount.textContent = chosen.length ? chosen.length + " 个" : "";
    el.selectedEmpty.hidden = chosen.length > 0;
    chosen.forEach((asset) => {
      const item = document.createElement("li");
      item.className = "chosen-row";
      const thumb = document.createElement("span");
      thumb.className = "thumb";
      const image = document.createElement("img");
      image.alt = "";
      image.loading = "lazy";
      image.src = assetThumbPath(asset);
      // 缩略图读不到时留明确占位，不放破图。
      image.addEventListener("error", () => {
        image.remove();
        const empty = document.createElement("span");
        empty.className = "thumb-empty";
        empty.textContent = "无缩略图";
        thumb.append(empty);
      });
      thumb.append(image);
      const body = document.createElement("span");
      body.className = "chosen-body";
      const name = document.createElement("span");
      name.className = "chosen-name";
      name.textContent = asset.name;
      name.title = asset.name;
      const meta = document.createElement("span");
      meta.className = "chosen-meta";
      meta.textContent = metaText(asset) || "视频";
      body.append(name, meta);
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "remove-button";
      remove.textContent = "移除";
      remove.addEventListener("click", () => {
        state.selected.delete(asset.id);
        renderAssets();
      });
      item.append(thumb, body, remove);
      el.selectedList.append(item);
    });
    const reason = !chosen.length ? "先选择要分析的视频"
      : (state.mode === "project" && !state.projectId ? "先选择项目" : "");
    el.setup.disabled = Boolean(reason);
    if (el.setupNote) el.setupNote.textContent = reason;
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

  // 新增视频沿用新建草稿时的初始化规则，不继承别的视频的几何或参数。
  function newDraftVideo(asset) {
    return {
      asset_id: asset.id,
      window: { start_s: 0, end_s: null },
      orientation: 0,
      fps_override: null,
      sample_fps: null,
      units: [newUnit("小鼠 1")],
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
    const videos = chosen.map(newDraftVideo);
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

  // 选择页的勾选必须与草稿实际分析集合一致：留下仍选中视频的单位、几何、
  // 参数和人工记录，新增视频按初始化规则补上，移除只退出本草稿的选择。
  function syncDraftVideos() {
    const draft = state.draft;
    if (!draft) return false;
    const before = draft.videos || [];
    const previous = before[state.videoIndex] || null;
    const videos = window.BLCore.reconcileVideos(
      before, state.selected, state.assets, newDraftVideo);
    const changed = videos.length !== before.length ||
      videos.some((video, index) => video !== before[index]);
    if (!changed) return false;
    draft.videos = videos;
    if (videos.length) {
      // 用同步前的视频对象定位，不能拿新数组的旧下标去猜。
      const index = previous ? videos.indexOf(previous) : -1;
      state.videoIndex = index >= 0 ? index : 0;
      const units = videos[state.videoIndex].units || [];
      if (state.unitIndex >= units.length) state.unitIndex = 0;
    }
    pruneManualCache();
    return true;
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
      ensureTopContext();
      renderTasknav();
    }
    if (state.draft && syncDraftVideos()) {
      markDirty();
      const saved = await saveThen();
      if (!saved) {
        status(lastSaveText("保存失败，选择未更新"), "error");
        return;
      }
    }
    ensureTopContext();
    renderTasknav();
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
      name.title = asset.name;
      const meta = document.createElement("span");
      meta.className = "range-meta";
      const count = (video.units || []).length;
      meta.textContent = (count ? count + " 只小鼠" : "未设置小鼠") +
        (metaText(asset) ? " · " + metaText(asset) : "");
      button.append(name, meta);
      button.addEventListener("click", () => {
        state.videoIndex = index;
        state.unitIndex = 0;
        // 换视频等于换一套源帧：设置页与画布都回到第 0 帧。
        state.rangeFrame = 0;
        state.frame = 0;
        state.shownFrame = 0;
        state.lastFrame = 0;
        renderRange();
        rememberPosition();
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

    const asset = assetById(video.asset_id) || { name: "", metadata: {} };
    const meta = asset.metadata || {};
    if (el.rangeVideoName) {
      el.rangeVideoName.textContent = asset.name || "视频 " + (state.videoIndex + 1);
      el.rangeVideoName.title = asset.name || "";
    }
    if (el.rangeFpsNote) {
      const sourceFps = Number(meta.fps);
      const parts = [];
      parts.push(Number.isFinite(sourceFps) && sourceFps > 0
        ? "源帧率 " + (Math.round(sourceFps * 100) / 100) + " FPS"
        : "源帧率未知");
      if (Number(meta.frame_count) > 0) parts.push(meta.frame_count + " 帧");
      el.rangeFpsNote.textContent = parts.join(" · ");
    }

    const fps = video.fps_override || Number(meta.fps) || 25;
    const frames = Number(meta.frame_count) ||
      Math.floor((Number(meta.duration_s) || 0) * fps) || 0;
    state.lastFrame = Math.max(0, Math.round(frames) - 1);
    if (el.rangeSlide) {
      el.rangeSlide.max = String(state.lastFrame);
      el.rangeSlide.value = String(Math.min(state.lastFrame, state.rangeFrame || 0));
    }
    if (el.rangeIndex) {
      el.rangeIndex.max = String(state.lastFrame);
      el.rangeIndex.value = String(Math.min(state.lastFrame, state.rangeFrame || 0));
    }
    loadRangeFrame(video, state.rangeFrame);
  }

  async function loadRangeFrame(video, index) {
    const token = ++state.rangeToken;
    const target = Math.max(0, Math.min(state.lastFrame,
      Number.isInteger(index) ? index : 0));
    // 请求目标只用于本次加载；state.rangeFrame 等图真正显示后再提交。
    if (el.rangeSlide) el.rangeSlide.value = String(target);
    if (el.rangeIndex) el.rangeIndex.value = String(target);
    if (el.rangeTime) {
      el.rangeTime.textContent = "第 — 帧 · --:--.--- / " +
        timeText(videoDuration(), 1);
    }
    el.rangeImage.removeAttribute("src");
    if (el.rangeEmpty) el.rangeEmpty.hidden = false;
    try {
      const response = await fetch(framePath(video, target), { cache: "no-store" });
      if (!response.ok || token !== state.rangeToken) return;
      const frame = Number(response.headers.get("X-Source-Frame"));
      const time = Number(response.headers.get("X-Source-Time"));
      const blob = await response.blob();
      if (token !== state.rangeToken) return;
      const url = URL.createObjectURL(blob);
      el.rangeImage.onload = () => {
        URL.revokeObjectURL(url);
        if (token !== state.rangeToken) return;
        if (el.rangeEmpty) el.rangeEmpty.hidden = true;
        if (el.rangeIndex && Number.isFinite(frame)) {
          el.rangeIndex.value = String(frame);
        }
        if (el.rangeTime) {
          el.rangeTime.textContent = "第 " + (Number.isFinite(frame) ? frame : target) +
            " 帧 · " + timeText(time, 3) + " / " + timeText(videoDuration(), 1);
        }
        // 只有这次请求仍有效且图已显示成功，才提交“已显示”的源帧与位置。
        state.rangeFrame = Number.isFinite(frame) ? Math.round(frame) : target;
        rememberPosition();
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
    // 方向或解释帧率变化后，设置页的源帧要按新参数重读。
    if (el.rangeSlide) loadRangeFrame(video, state.rangeFrame);
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
      status(lastSaveText("保存失败，未进入场地"), "error");
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
    pruneManualCache();
    renderTasknav();
  }

  function isManualMode() {
    if (state.assay !== "ymaze") return false;
    const unit = currentUnit();
    return Boolean(unit && unit.manual && unit.manual.mode === "manual");
  }

  // -- 人工 Y：记录容器与时间显示 -------------------------------------
  function manualOf(unit) {
    if (!unit) return null;
    if (!unit.manual || typeof unit.manual !== "object") return null;
    if (!Array.isArray(unit.manual.entries)) return null;
    return unit.manual;
  }

  function ensureManual(unit) {
    if (!unit.manual || typeof unit.manual !== "object") {
      unit.manual = { mode: "manual", entries: [] };
    }
    if (!Array.isArray(unit.manual.entries)) unit.manual.entries = [];
    unit.manual.mode = "manual";
    return unit.manual;
  }

  function timeText(seconds, digits) {
    if (!Number.isFinite(seconds) || seconds < 0) {
      return digits === 3 ? "--:--.---" : "--:--.-";
    }
    const minutes = Math.floor(seconds / 60);
    const rest = seconds - minutes * 60;
    const width = digits === 3 ? 6 : 4;
    return String(minutes).padStart(2, "0") + ":" +
      rest.toFixed(digits).padStart(width, "0");
  }

  function videoDuration() {
    const video = currentVideo();
    const asset = video ? assetById(video.asset_id) : null;
    const meta = (asset && asset.metadata) || {};
    const declared = Number(meta.duration_s);
    if (Number.isFinite(declared) && declared > 0) return declared;
    const fps = (video && video.fps_override) || Number(meta.fps) || 25;
    return (state.lastFrame + 1) / fps;
  }

  function frameLabel(info) {
    const total = timeText(videoDuration(), 1);
    if (!info || !Number.isFinite(info.time)) {
      return "第 — 帧 · --:--.--- / " + total;
    }
    return "第 " + info.frame + " 帧 · " + timeText(info.time, 3) + " / " + total;
  }

  function assaySpec() {
    return state.catalog.find((item) => item.key === state.assay) || null;
  }

  // 参数与几何描述符的 label 是界面唯一可读名称来源；错误提示据此补字段名，
  // 用户看到“中心区边长：必须大于 0”而不是 options.center_size_cm。
  function registerFieldLabels() {
    const labels = {};
    const current = assaySpec();
    const ordered = current
      ? [current].concat(state.catalog.filter((item) => item !== current))
      : state.catalog;
    ordered.forEach((item) => {
      (item.parameters || []).forEach((param) => {
        if (!param || !param.key || !param.label) return;
        if (!(param.key in labels)) labels[param.key] = param.label;
      });
      (item.geometry || []).forEach((step) => {
        if (!step || !step.key || !step.label) return;
        if (step.key in labels) return;
        const name = step.label.replace(/(四角|范围)$/, "").trim();
        labels[step.key] = name || step.label;
      });
    });
    if (typeof window.BLCore.setFieldLabels === "function") {
      window.BLCore.setFieldLabels(labels);
    }
  }

  function geometrySteps() {
    const spec = assaySpec();
    const steps = spec ? spec.geometry || [] : [];
    if (state.assay !== "ldb") return steps;
    // 不可见暗室隐藏绘图步骤，但不删除已绘制的几何。
    const visible = effectiveOptions(currentUnit()).dark_box_visible;
    return visible === false
      ? steps.filter((step) => step.key !== "dark")
      : steps;
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

  // 画布步骤键与持久化结构不一定同名（NOR 的物体、EPM 的五区），
  // 统一通过 BLCore 的步骤桥接读写。
  function stepShape(unit, key) {
    return window.BLCore.stepShape(state.assay,
      unit ? unit.geometry : null, key);
  }

  function setStepShape(unit, key, shape) {
    if (!unit) return;
    if (!unit.geometry) unit.geometry = {};
    window.BLCore.stepApply(state.assay, unit.geometry, key, shape);
  }

  function selectStep(key) {
    const step = geometrySteps().find((item) => item.key === key);
    if (!step) return;
    state.stepKey = key;
    const unit = currentUnit();
    const resolved = window.BLCore.resolveStep(step, effectiveOptions(unit));
    state.editor.reset(resolved, stepShape(unit, key));
    syncTools();
    renderInspector();
    draw();
  }

  function syncToolTitle() {
    if (!el.toolTitle) return;
    if (state.recordMode) {
      el.toolTitle.textContent = "";
      return;
    }
    if (state.view === "detection") {
      el.toolTitle.textContent = "检测";
      return;
    }
    const step = currentSpec();
    el.toolTitle.textContent = step ? "绘制：" + step.label : "";
  }

  function syncTools() {
    // 只有场地阶段能编辑几何：检测与人工阶段不摆可编辑工具。
    const editable = !state.recordMode && state.view !== "detection";
    const unit = currentUnit();
    const hasGeometry = Boolean(stepShape(unit, state.stepKey));
    const hasPending = state.editor.points.length > 0;
    el.toolUndo.hidden = !editable;
    el.toolRedraw.hidden = !editable;
    el.toolFinish.hidden = !editable || !state.editor.canFinish();
    el.toolUndo.disabled = !hasGeometry && !hasPending;
    el.toolRedraw.disabled = !hasGeometry && !hasPending;
    syncToolTitle();
  }

  function refreshEditorSpec() {
    const step = geometrySteps().find((item) => item.key === state.stepKey);
    if (!step) return;
    const unit = currentUnit();
    const resolved = window.BLCore.resolveStep(step, effectiveOptions(unit));
    state.editor.reset(resolved, stepShape(unit, state.stepKey));
    syncTools();
  }

  // -- 源帧 -----------------------------------------------------------
  // 取帧与提交分开：播放时先解码，再等到真正该显示的时刻才画上新帧。
  function beginFrameLoad(index, options) {
    const video = currentVideo();
    if (!video) return null;
    const keepInfo = Boolean(options && options.keepInfo);
    if (Number.isInteger(index)) {
      state.frame = Math.max(0, Math.min(state.lastFrame, index));
      el.frameRange.value = String(state.frame);
      el.frameNumber.value = String(state.frame);
    }
    const context = { ticket: ++state.imageToken, page: pageKey() };
    state.loading = true;
    if (keepInfo) {
      // 播放时画面仍是上一帧：先保留它的时间，等新帧真的显示再换。
      return context;
    }
    // 跳转时图片成功前不保留上一帧时间：记录不能借用旧帧的时刻。
    state.frameInfo = null;
    el.timeLabel.textContent = frameLabel(null);
    state.canvasMessage = "正在读取源帧…";
    status(STATUS.loading);
    return context;
  }

  async function fetchFrame(video, context) {
    const response = await fetch(framePath(video, state.frame), { cache: "no-store" });
    if (!response.ok) {
      const payload = await response.json().catch(() => null);
      throw new Error(payload && payload.error ? payload.error.message : "源帧不可读");
    }
    const time = Number(response.headers.get("X-Source-Time"));
    const frameIndex = Number(response.headers.get("X-Source-Frame"));
    const clock = response.headers.get("X-Clock-Source") || "";
    const blob = await response.blob();
    if (context.ticket !== state.imageToken) return null;
    const url = URL.createObjectURL(blob);
    let image;
    try {
      image = new Image();
      await new Promise((resolve, reject) => {
        image.onload = resolve;
        image.onerror = reject;
        image.src = url;
      });
    } finally {
      URL.revokeObjectURL(url);
    }
    if (context.ticket !== state.imageToken) return null;
    return { image, time, frame: frameIndex, clock };
  }

  function frameStillCurrent(context) {
    // 提交画面前同时校验请求代号与页面上下文，不只是请求代号。
    const api = window.BLFrameView;
    if (api && typeof api.frameStillCurrent === "function") {
      return api.frameStillCurrent(context, state.imageToken, pageKey());
    }
    // 旧缓存页未加载 frameview 时退回同一条判定，播放器不至于整页崩掉。
    return Boolean(context) && context.ticket === state.imageToken
      && context.page === pageKey();
  }

  function commitFrame(payload, context) {
    if (!payload || !frameStillCurrent(context)) return null;
    state.image = payload.image;
    state.canvasMessage = "";
    invalidatePreview();
    state.frameInfo = { time: payload.time, frame: payload.frame, clock: payload.clock };
    if (Number.isFinite(payload.frame) && payload.frame !== state.frame) {
      state.frame = payload.frame;
    }
    if (Number.isInteger(payload.frame)) state.shownFrame = payload.frame;
    el.frameRange.value = String(state.frame);
    el.frameNumber.value = String(state.frame);
    el.timeLabel.textContent = frameLabel(state.frameInfo);
    syncRecordHint();
    state.loading = false;
    status("");
    // 连续播放每帧不改 history；跳转成功立刻记下实际显示的帧。
    if (!state.playing) rememberPosition();
    draw();
    return state.frameInfo;
  }

  async function loadFrame(index) {
    const video = currentVideo();
    if (!video) return null;
    const context = beginFrameLoad(index);
    if (!context) return null;
    try {
      const payload = await fetchFrame(video, context);
      return commitFrame(payload, context);
    } catch (error) {
      if (!frameStillCurrent(context)) return null;
      state.loading = false;
      state.image = null;
      state.frameInfo = null;
      state.canvasMessage = "源帧不可读";
      el.timeLabel.textContent = frameLabel(null);
      syncRecordHint();
      draw();
      const asset = assetById(video.asset_id);
      status("源帧不可读：" + (asset ? asset.name : "视频"), "error");
      el.statusbar.title = error.message;
      return null;
    }
  }

  // -- 源帧播放（记录页） ---------------------------------------------
  function playbackButton() {
    return document.querySelector("[data-play-toggle]");
  }

  function syncPlayButton() {
    const button = playbackButton();
    if (button) button.textContent = state.playing ? "暂停" : "播放";
  }

  function syncRecordHint() {
    const hint = document.querySelector("[data-record-hint]");
    if (!hint) return;
    hint.textContent = state.frameInfo
      ? "空格播放/暂停 · A/B/C 记录当前帧"
      : "正在读取源帧，稍后可记录";
  }

  function waitPlayback(ms, token) {
    // 可取消的等待：暂停、离页、销毁都会立刻让等待返回 false。
    return new Promise((resolve) => {
      const timer = setTimeout(() => {
        state.playWait = null;
        resolve(token === state.playToken && state.playing);
      }, Math.max(0, ms));
      state.playWait = { timer, resolve };
    });
  }

  function stopPlayback() {
    const wasPlaying = state.playing;
    state.playToken += 1;
    state.playing = false;
    state.playLastTime = null;
    if (state.playWait) {
      clearTimeout(state.playWait.timer);
      const release = state.playWait.resolve;
      state.playWait = null;
      release(false);
    }
    if (state.playTimer) {
      clearTimeout(state.playTimer);
      state.playTimer = null;
    }
    // 停播同时废弃尚未显示的帧请求：稍后释出的旧响应不能回写。
    state.imageToken += 1;
    if (wasPlaying && Number.isInteger(state.shownFrame)
        && state.frame !== state.shownFrame) {
      // 暂停时请求了但没显示的帧不算数：滑条、帧号与历史回到实际画面。
      // 跳转（本来就没在播放）保留目标帧，否则滑条一松手就被旧帧顶回去。
      state.frame = state.shownFrame;
      el.frameRange.value = String(state.frame);
      el.frameNumber.value = String(state.frame);
    }
    syncPlayButton();
  }

  async function playStep(token) {
    if (!state.playing || token !== state.playToken) return;
    const video = currentVideo();
    if (!video) {
      stopPlayback();
      return;
    }
    if (state.shownFrame >= state.lastFrame) {
      stopPlayback();
      rememberPosition();
      return;
    }
    const started = performance.now();
    // 逐个请求下一源帧：显示时刻由真实相邻时间差与倍率决定，不用名义 FPS。
    const context = beginFrameLoad(state.shownFrame + 1, { keepInfo: true });
    let payload;
    try {
      payload = await fetchFrame(video, context);
    } catch (error) {
      if (token === state.playToken && frameStillCurrent(context)) {
        state.loading = false;
        state.canvasMessage = "源帧不可读";
        el.timeLabel.textContent = frameLabel(null);
        syncRecordHint();
        draw();
        status("源帧不可读：" + ((assetById(video.asset_id) || {}).name || "视频"),
          "error");
        el.statusbar.title = error.message;
        stopPlayback();
      }
      return;
    }
    if (!state.playing || token !== state.playToken) return;
    if (!payload) return;
    const delay = window.BLFrameView.frameDisplayDelay(
      state.playLastTime, payload.time, state.rate,
      performance.now() - started);
    if (delay > 0) {
      const ready = await waitPlayback(delay, token);
      if (!ready) return;
    }
    if (!state.playing || token !== state.playToken) return;
    const info = commitFrame(payload, context);
    if (!info) return;
    state.playLastTime = info.time;
    state.playTimer = setTimeout(() => { playStep(token); }, 0);
  }

  function togglePlayback() {
    if (state.playing) {
      stopPlayback();
      rememberPosition();
      return;
    }
    if (!state.draft || !currentVideo()) return;
    if (state.shownFrame >= state.lastFrame) return;
    state.playing = true;
    state.playToken += 1;
    const token = state.playToken;
    state.playLastTime = state.frameInfo ? state.frameInfo.time : null;
    syncPlayButton();
    playStep(token);
  }

  function setRate(rate) {
    state.rate = rate;
    renderInspector();
    status("播放速度 " + rate + "×");
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

  // 当前实验允许显示的已保存几何：LDB 关闭暗室可见时不画暗室，其余原样。
  function unitGeometryShapes(unit) {
    if (!unit) return [];
    const spec = assaySpec();
    let steps = spec ? spec.geometry || [] : [];
    if (state.assay === "ldb" && effectiveOptions(unit).dark_box_visible === false) {
      steps = steps.filter((step) => step.key !== "dark");
    }
    return steps
      .map((step) => ({ step, shape: stepShape(unit, step.key) }))
      .filter((item) => Boolean(item.shape));
  }

  function shapeAnchor(shape) {
    if (!shape) return null;
    if (shape.type === "ellipse" && shape.center) return shape.center;
    if (shape.type === "holes") return null;
    const points = shape.points || [];
    if (!points.length) return null;
    const sum = points.reduce((acc, point) => [acc[0] + point[0], acc[1] + point[1]],
      [0, 0]);
    return [sum[0] / points.length, sum[1] / points.length];
  }

  function drawStepLabel(step, shape) {
    const anchor = shapeAnchor(shape);
    if (!anchor || !step || !step.label) return;
    const point = toCss(anchor);
    ctx.save();
    ctx.font = "11px 'Segoe UI', 'Microsoft YaHei', sans-serif";
    ctx.fillStyle = "rgba(255, 210, 111, 0.9)";
    ctx.fillText(step.label, point[0] + 8, point[1] - 6);
    ctx.restore();
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
    // 纯人工 Y 的记录/复核只看原始源帧：不叠自动 ROI、编辑手柄或检测结果。
    // 几何与检测配置原样保留，切回自动仍会重新绘制。
    if (state.recordMode) return;
    // 其他动物只是只读参照。
    video.units.forEach((unit, index) => {
      if (index === state.unitIndex) return;
      unitGeometryShapes(unit).forEach((item) => {
        drawShape(item.shape, "#8ed7c7", 1.2, false);
      });
    });
    const unit = currentUnit();
    if (!unit) return;
    // 当前动物：其余已保存几何用同色细线，只有当前步骤带可编辑手柄。
    unitGeometryShapes(unit).forEach((item) => {
      if (item.step.key === state.stepKey) return;
      drawShape(item.shape, "#ffd26f", 1.6, false);
      if (item.step.type !== "holes") drawStepLabel(item.step, item.shape);
    });
    if (state.assay === "barnes") drawBarnesTarget(unit);
    if (state.assay === "mwm") drawMwmGuides(unit);
    const shape = stepShape(unit, state.stepKey);
    if (shape) drawShape(shape, "#ffd26f", 2.8, true);
    drawEditor();
  }

  function drawBarnesTarget(unit) {
    const holes = stepShape(unit, "holes");
    if (!holes) return;
    const target = String(effectiveOption(unit, "target_hole", "1"));
    const hole = (holes.holes || []).find((item) => String(item.id) === target);
    if (!hole) return;
    const point = toCss(hole.center);
    ctx.save();
    ctx.strokeStyle = "#ff6e46";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(point[0], point[1], 7, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }

  function drawMwmGuides(unit) {
    if (!unit || !unit.geometry) return;
    const view = viewport();
    if (!view) return;
    const diameter = Number(effectiveOption(unit, "pool_diameter_cm", 0));
    const rotation = Number(effectiveOption(unit, "coordinate_rotation_deg", 0)) || 0;
    const guides = (typeof BLCore !== "undefined" && BLCore.mwmLines)
      ? BLCore.mwmLines(unit.geometry.arena, diameter, rotation) : null;
    if (!guides) return;
    const center = toCss(guides.center);
    const length = guides.length * view.scale;
    ctx.save();
    ctx.setLineDash([7, 5]);
    ctx.strokeStyle = "rgba(255, 210, 111, 0.65)";
    ctx.lineWidth = 1.2;
    guides.directions.forEach(([dx, dy]) => {
      ctx.beginPath();
      ctx.moveTo(center[0] - dx * length, center[1] - dy * length);
      ctx.lineTo(center[0] + dx * length, center[1] + dy * length);
      ctx.stroke();
    });
    ctx.restore();
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
    setStepShape(unit, state.stepKey, geometry);
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
    const existing = stepShape(unit, state.stepKey);
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
    const shape = stepShape(unit, state.stepKey);
    if (!shape) return;
    const hit = drag.hit;
    if (hit.kind === "vertex") {
      setStepShape(unit, state.stepKey, window.BLCore.moveVertex(shape, hit.index, point));
    } else if (hit.kind === "hole") {
      setStepShape(unit, state.stepKey, window.BLCore.moveHole(shape, hit.id, point));
    } else if (hit.kind === "center") {
      setStepShape(unit, state.stepKey, window.BLCore.moveEllipseCenter(shape, point));
    } else if (hit.kind === "radius") {
      setStepShape(unit, state.stepKey, window.BLCore.moveRadius(shape, hit.axis, point));
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
    // 人工记录/复核没有几何绘制，撤销与重画在这里不适用。
    if (el.toolhead) el.toolhead.hidden = Boolean(state.recordMode);
    syncToolTitle();
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
    if (!steps.some((step) => step.key === state.stepKey)) {
      // 当前步骤被隐藏（如 LDB 关闭暗室可见）时回到第一个步骤。
      state.stepKey = steps.length ? steps[0].key : null;
      const first = steps[0];
      if (first) {
        state.editor.reset(window.BLCore.resolveStep(first, effectiveOptions(unit)),
                           stepShape(unit, first.key));
        syncTools();
      }
    }
    const required = steps.filter((step) => step.required);
    const optional = steps.filter((step) => !step.required);
    const drawn = (step) => Boolean(stepShape(unit, step.key));
    const list = document.createElement("div");
    list.className = "step-list";
    required.forEach((step) => {
      list.append(stepRow(step, drawn(step)));
      if (state.assay === "nor" && step.role === "object") {
        list.append(norRoleRow(step.key === "object1" ? 0 : 1));
      }
    });
    body.append(list);
    appendModeControl(body);
    if (state.assay === "ymaze") {
      // 场地页选择记录方式：自动追踪 / 人工记录（人工切换需确认已有记录）。
      appendPathMode(body);
    }
    if (optional.length) {
      const fold = document.createElement("details");
      fold.className = "fold";
      const summary = document.createElement("summary");
      summary.textContent = "可选区域";
      fold.append(summary);
      const inner = document.createElement("div");
      inner.className = "fold-body";
      const innerList = document.createElement("div");
      innerList.className = "step-list";
      optional.forEach((step) => innerList.append(stepRow(step, drawn(step))));
      inner.append(innerList);
      fold.append(inner);
      body.append(fold);
    }
    appendExperimentSettings(body, unit);
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

  function norRoleRow(index) {
    const unit = currentUnit();
    const wrap = document.createElement("label");
    wrap.className = "field inline unit";
    const caption = document.createElement("span");
    caption.textContent = "物体 " + (index + 1) + " 角色";
    const select = document.createElement("select");
    [["familiar", "熟悉"], ["novel", "新奇"]].forEach(([value, text]) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = text;
      select.append(option);
    });
    const objects = window.BLCore.norObjects(unit ? unit.geometry : null);
    select.value = objects[index].role;
    select.addEventListener("change", () => {
      const target = currentUnit();
      if (!target) return;
      if (!target.geometry) target.geometry = {};
      window.BLCore.norSetRole(target.geometry, index, select.value,
        effectiveOption(target, "phase", "recognition"));
      invalidatePreview();
      markDirty();
      renderInspector();
      draw();
    });
    wrap.append(caption, select);
    return wrap;
  }

  function renderDetectionInspector(body, unit) {
    // 检测阶段只放检测方法、阈值、输出与设置；场地步骤不在这里出现。
    const methods = document.createElement("div");
    methods.className = "inspect-section";
    methods.append(detectionFields(unit));
    body.append(methods);

    appendPreviewModes(body);

    const output = document.createElement("details");
    output.className = "fold";
    const outputSummary = document.createElement("summary");
    outputSummary.textContent = "输出";
    const outputBody = document.createElement("div");
    outputBody.className = "fold-body";
    appendAnnotationToggle(outputBody);
    output.append(outputSummary, outputBody);
    body.append(output);

    appendExperimentSettings(body, unit);

    const kind = state.previewKind === "behavior" ? "behavior" : "detection";
    const confirmed = state.previewGate.confirmed(previewConfig(kind));
    el.primary.textContent = confirmed
      ? "检查并开始分析" : window.BLCore.previewActionLabel(kind);
    el.secondary.textContent = kind === "behavior" ? "更新行为预览" : "更新检测预览";
    el.secondary.hidden = !confirmed;
    el.preview.hidden = !confirmed && !state.previewAwaiting;
    syncTools();
  }

  // 原始画面 / 检测预览 / 行为预览：三者切换同一画布上的叠画。
  function appendPreviewModes(body) {
    const wrap = document.createElement("div");
    wrap.className = "mode-row";
    [["raw", "原始画面"], ["detection", "检测预览"],
     ["behavior", "行为预览"]].forEach(([value, label]) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "mode-button" +
        (state.previewKind === value ? " is-current" : "");
      button.textContent = label;
      button.addEventListener("click", () => { selectPreviewKind(value); });
      wrap.append(button);
    });
    body.append(wrap);
  }

  async function selectPreviewKind(kind) {
    state.previewKind = kind;
    if (kind === "raw") {
      el.preview.hidden = true;
      el.preview.removeAttribute("src");
      status("原始画面", "ok");
      renderInspector();
      return;
    }
    await runPreview(kind);
  }

  // 输出开关只影响导出，不进入预览签名，切换时不强制重算。
  function appendAnnotationToggle(body) {
    const label = document.createElement("label");
    label.className = "field";
    const caption = document.createElement("span");
    caption.textContent = "生成标注视频";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = Boolean((state.draft.output || {}).annotated_video);
    input.addEventListener("change", () => {
      if (!state.draft.output) state.draft.output = {};
      state.draft.output.annotated_video = input.checked;
      markDirty();
      status(input.checked ? "已开启标注视频" : "已关闭标注视频", "ok");
    });
    label.append(caption, input);
    body.append(label);
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
    el.preview.hidden = !confirmed && !state.previewAwaiting;
  }

  // -- 人工事件（Barnes）----------------------------------------------
  // hole_visits:null 表示自动探洞；[] 表示人工确认零访问，两者不能混。
  function manualCache() {
    if (!state.manualCache) state.manualCache = new Map();
    return state.manualCache;
  }

  function manualText(unit) {
    const key = unit && unit.id ? unit.id : null;
    if (key && manualCache().has(key)) return manualCache().get(key);
    const list = unit && unit.manual && Array.isArray(unit.manual.hole_visits)
      ? unit.manual.hole_visits : [];
    const text = list.map((item) => {
      if (!item || typeof item.time_s !== "number" || !Number.isFinite(item.time_s)) {
        return "";
      }
      const hole = item.hole_id == null ? "" : String(item.hole_id);
      return item.time_s + ", " + hole;
    }).filter(Boolean).join("\n");
    const entry = { text, error: null };
    if (key) manualCache().set(key, entry);
    return entry;
  }

  function parseManualVisits(text) {
    return window.BLCore.parseManualVisits(text);
  }

  function manualFormatError() {
    // 提交的是整批视频/动物：任一动物的人工缓存有未解析错误都要拦下，
    // 并带上视频位置，两个视频同名动物时也能落到正确那一个。
    const videos = (state.draft && state.draft.videos) || [];
    for (let index = 0; index < videos.length; index += 1) {
      const units = (videos[index].units || []).filter(Boolean);
      const issue = window.BLCore.manualFormatIssue(units, (unit) =>
        (unit.id ? manualCache().get(unit.id) : null));
      if (!issue) continue;
      const asset = assetById(videos[index].asset_id);
      const videoLabel = asset && asset.name ? asset.name : "视频 " + (index + 1);
      return {
        message: "视频「" + videoLabel + "」·「" +
          (issue.unitName || "未命名小鼠") + "」" + issue.error +
          "；请回到该动物修正，或关闭其人工洞序列",
        videoIndex: index,
        unitId: issue.unitId,
        unitName: issue.unitName,
      };
    }
    return null;
  }

  // 未解决的动物时间窗编辑：提示具体字段并定位，返回 true 表示已拦截。
  function blockOnWindowError() {
    const issue = pendingWindowError();
    if (!issue) return false;
    status(issue.message, "error");
    const video = state.draft.videos[issue.videoIndex] || {};
    const unit = (video.units || [])[issue.unitIndex] || {};
    state.windowStatus = {
      key: (video.asset_id || "") + "|" + (unit.id || ""),
      text: issue.message,
    };
    revealWindowError(issue);
    return true;
  }

  // 切到出错的那个视频/动物，展开人工事件并聚焦原始非法文本。
  function revealManualIssue(issue) {
    const videos = (state.draft && state.draft.videos) || [];
    for (let index = 0; index < videos.length; index += 1) {
      const units = videos[index].units || [];
      const unitIndex = units.findIndex((unit) => unit && unit.id === issue.unitId);
      if (unitIndex < 0) continue;
      stopPlayback();
      state.videoIndex = index;
      state.unitIndex = unitIndex;
      state.stepKey = null;
      ensureContext();
      renderInspector();
      loadFrame();
      draw();
      const area = el.inspectBody.querySelector("[data-manual-input]");
      if (area) {
        const fold = area.closest("details");
        if (fold) fold.open = true;
        area.focus();
        const end = area.value.length;
        area.setSelectionRange(end, end);
      }
      return true;
    }
    return false;
  }

  function pruneManualCache() {
    if (state.manualCache) {
      const alive = new Set();
      ((state.draft && state.draft.videos) || []).forEach((video) => {
        (video.units || []).forEach((unit) => {
          if (unit && unit.id) alive.add(unit.id);
        });
      });
      Array.from(manualCache().keys()).forEach((key) => {
        if (!alive.has(key)) manualCache().delete(key);
      });
    }
    pruneWindowEdits();
  }

  // -- 动物时间窗的未解决编辑 -----------------------------------------
  // 键按视频+动物：切视频、切动物、重渲染都保留原始非法或未填完的文本。
  function windowEditKey(unit) {
    const video = currentVideo();
    return (video ? video.asset_id : "") + "|" + (unit && unit.id ? unit.id : "");
  }

  function windowEditOf(unit) {
    if (!unit) return null;
    return state.windowEdits.get(windowEditKey(unit)) || null;
  }

  function pruneWindowEdits() {
    if (!state.windowEdits) return;
    const alive = new Set();
    ((state.draft && state.draft.videos) || []).forEach((video) => {
      (video.units || []).forEach((unit) => {
        if (unit) alive.add((video.asset_id || "") + "|" + (unit.id || ""));
      });
    });
    Array.from(state.windowEdits.keys()).forEach((key) => {
      if (!alive.has(key)) state.windowEdits.delete(key);
    });
  }

  // 未解决的窗口编辑必须拦住推进/预览/校验/开始，不能静默用最后一次合法值。
  function pendingWindowError() {
    const videos = (state.draft && state.draft.videos) || [];
    for (let v = 0; v < videos.length; v += 1) {
      const units = videos[v].units || [];
      for (let u = 0; u < units.length; u += 1) {
        const unit = units[u];
        if (!unit) continue;
        const key = (videos[v].asset_id || "") + "|" + (unit.id || "");
        const entry = state.windowEdits.get(key);
        if (!entry || !entry.error) continue;
        const asset = assetById(videos[v].asset_id);
        const videoLabel = asset && asset.name ? asset.name : "视频 " + (v + 1);
        return {
          message: "视频「" + videoLabel + "」·「" + (unit.name || "小鼠 " + (u + 1)) +
            "」动物时间窗：" + entry.error,
          videoIndex: v,
          unitIndex: u,
          field: entry.field,
        };
      }
    }
    return null;
  }

  function revealWindowError(issue) {
    if (!issue) return;
    const video = state.draft && state.draft.videos[issue.videoIndex];
    const unit = video && (video.units || [])[issue.unitIndex];
    if (!unit) return;
    const needStep = state.step !== "arena" || state.view !== "geometry";
    const needUnit = state.videoIndex !== issue.videoIndex ||
      state.unitIndex !== issue.unitIndex;
    if (needStep || needUnit) {
      stopPlayback();
      state.videoIndex = issue.videoIndex;
      state.unitIndex = issue.unitIndex;
      state.stepKey = null;
      if (needStep) {
        // 字段在场地阶段的“实验设置”里：先切回该阶段再聚焦。
        state.step = "arena";
        state.view = "geometry";
        root.dataset.step = "arena";
        showPanel();
        rememberPosition("arena", "geometry");
      }
      ensureContext();
      renderTasknav();
      renderInspector();
      if (needUnit) loadFrame();
      draw();
    }
    const input = el.inspectBody.querySelector(
      '[data-window-field="' + (issue.field || "") + '"]');
    if (input) {
      const fold = input.closest("details");
      if (fold) fold.open = true;
      input.focus();
    }
  }

  function appendManualEvents(body, unit) {
    if (state.assay !== "barnes" || !unit) return;
    const manual = unit.manual && typeof unit.manual === "object" ? unit.manual : null;
    const enabled = Boolean(manual && manual.hole_visits !== null
      && manual.hole_visits !== undefined);
    const probe = effectiveOption(unit, "trial_type", "training") === "probe";

    const fold = document.createElement("details");
    fold.className = "fold";
    const summary = document.createElement("summary");
    summary.textContent = "人工事件";
    const inner = document.createElement("div");
    inner.className = "fold-body";

    const toggle = document.createElement("label");
    toggle.className = "field inline";
    const caption = document.createElement("span");
    caption.textContent = "使用人工洞序列";
    const check = document.createElement("input");
    check.type = "checkbox";
    check.checked = enabled;
    check.addEventListener("change", () => {
      const target = currentUnit();
      if (!target) return;
      if (!target.manual || typeof target.manual !== "object") target.manual = {};
      // 切换即改模式：关闭写 null，重新打开从空人工列表开始。
      target.manual.hole_visits = check.checked ? [] : null;
      if (target.id) {
        manualCache().delete(target.id);
        if (state.manualStatus && state.manualStatus.unitId === target.id &&
            statusNode().textContent === state.manualStatus.text) {
          state.manualStatus = null;
          status("");
        }
      }
      invalidatePreview();
      markDirty();
      renderInspector();
    });
    toggle.append(caption, check);
    inner.append(toggle);

    if (enabled) {
      const area = document.createElement("textarea");
      area.rows = 4;
      area.className = "manual-visits";
      area.dataset.manualInput = "1";
      area.placeholder = "视频秒, 洞号";
      const entry = manualText(unit);
      area.value = entry.text;
      const hint = document.createElement("p");
      hint.className = "muted manual-hint";
      hint.textContent = entry.error || "每行一条；按输入顺序计分，不自动排序。";
      if (entry.error) hint.dataset.tone = "error";
      area.addEventListener("input", () => {
        const target = currentUnit();
        if (!target) return;
        const cached = manualText(target);
        cached.text = area.value;
        const parsed = parseManualVisits(area.value);
        if (parsed.error) {
          // 保留原始编辑文本与错误；不覆盖最后一份合法数组。
          cached.error = parsed.error;
          hint.textContent = parsed.error;
          hint.dataset.tone = "error";
          state.manualStatus = { unitId: target.id, text: parsed.error };
          status(parsed.error, "error");
          invalidatePreview();
          return;
        }
        cached.error = null;
        hint.textContent = "每行一条；按输入顺序计分，不自动排序。";
        delete hint.dataset.tone;
        // 只清掉属于这只动物的那条旧格式错误提示，别动其他单位的未解决错误。
        if (state.manualStatus && state.manualStatus.unitId === target.id &&
            statusNode().textContent === state.manualStatus.text) {
          state.manualStatus = null;
          status("");
        }
        if (!target.manual || typeof target.manual !== "object") target.manual = {};
        target.manual.hole_visits = parsed.entries;
        invalidatePreview();
        markDirty();
      });
      inner.append(area, hint);
    }

    if (!probe) {
      // 独立可空字段：label 独占一行，输入框不撑出检查器宽度。
      const escapeLabel = document.createElement("label");
      escapeLabel.className = "manual-field";
      const escapeCaption = document.createElement("span");
      escapeCaption.textContent = "确认入箱时间（视频秒）";
      const escapeInput = document.createElement("input");
      escapeInput.type = "number";
      escapeInput.step = "any";
      escapeInput.min = "0";
      escapeInput.placeholder = "未确认";
      const current = manual && manual.escape_time_s != null
        ? manual.escape_time_s : null;
      escapeInput.value = current == null ? "" : String(current);
      escapeInput.addEventListener("change", () => {
        const target = currentUnit();
        if (!target) return;
        if (!target.manual || typeof target.manual !== "object") target.manual = {};
        const raw = escapeInput.value.trim();
        if (!raw) {
          target.manual.escape_time_s = null;
        } else {
          const seconds = Number(raw);
          if (!Number.isFinite(seconds) || seconds < 0) {
            status("确认入箱时间必须是非负秒数", "error");
            escapeInput.value = target.manual.escape_time_s == null
              ? "" : String(target.manual.escape_time_s);
            return;
          }
          target.manual.escape_time_s = seconds;
        }
        invalidatePreview();
        markDirty();
      });
      escapeLabel.append(escapeCaption, escapeInput);
      inner.append(escapeLabel);
    }

    fold.append(summary, inner);
    body.append(fold);
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
    // 自动路径在这里给动物时间窗入口；人工 Y 用记录阶段那一份（只允许 global/exact）。
    if (!(state.assay === "ymaze" && isManualMode())) {
      inner.append(animalWindowField(unit, { allowRelative: true }));
    }
    settings.append(summary, inner);
    body.append(settings);
    appendManualEvents(body, unit);

    const animal = document.createElement("details");
    animal.className = "fold";
    const animalSummary = document.createElement("summary");
    animalSummary.textContent = "小鼠设置";
    const animalBody = document.createElement("div");
    animalBody.className = "fold-body";
    animalBody.append(nameField(unit));
    const remove = unitDeleteField(unit);
    if (remove) animalBody.append(remove);
    animal.append(animalSummary, animalBody);
    body.append(animal);
  }

  function unitDeleteField(unit) {
    const video = currentVideo();
    if (!video || (video.units || []).length < 2 || !unit) return null;
    const wrap = document.createElement("div");
    wrap.className = "unit-delete";
    const button = document.createElement("button");
    button.type = "button";
    button.className = "button small danger";
    button.textContent = "删除当前小鼠";
    button.addEventListener("click", () => { deleteCurrentUnit(); });
    const note = document.createElement("p");
    note.textContent = "只从当前视频移除该小鼠的几何、参数与人工记录。";
    wrap.append(button, note);
    return wrap;
  }

  async function deleteCurrentUnit() {
    const video = currentVideo();
    if (!video || (video.units || []).length < 2) return;
    const unit = currentUnit();
    if (!unit) return;
    const name = unit.name || "小鼠 " + (state.unitIndex + 1);
    const remaining = video.units.length - 1;
    if (!window.confirm("删除「" + name + "」？将移除该小鼠在当前视频的几何、参数和人工记录；"
        + "视频、其他小鼠与已完成结果不受影响。剩余 " + remaining + " 只。")) {
      return;
    }
    const plan = window.BLCore.removeUnit(video.units, video.units.indexOf(unit));
    if (!plan) return;
    video.units = plan.units;
    if (unit.id) manualCache().delete(unit.id);
    state.unitIndex = plan.index;
    state.stepKey = null;
    state.view = "geometry";
    stopPlayback();
    // 旧预览确认与在途画面属于已删除的小鼠。
    invalidatePreview();
    state.imageToken += 1;
    markDirty();
    ensureContext();
    renderTasknav();
    renderInspector();
    draw();
    const ok = await saveThen();
    if (!ok) {
      status(lastSaveText("删除未保存，请重试"), "error");
      return;
    }
    status("已删除「" + name + "」", "ok");
  }

  function appendPathMode(body) {
    if (state.assay !== "ymaze") return;
    const unit = currentUnit();
    const current = unit && unit.manual && unit.manual.mode === "manual"
      ? "manual" : "auto";
    const wrap = document.createElement("div");
    wrap.className = "mode-row";
    [["auto", "自动追踪"], ["manual", "人工记录"]].forEach(([value, label]) => {
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
    const current = unit.manual && unit.manual.mode === "manual" ? "manual" : "auto";
    if (current === mode) return;
    if (mode === "manual") {
      // 自动 → 人工：新建空记录；相对单位窗改为跟随视频，几何/检测值保留。
      unit.manual = { mode: "manual", entries: [] };
      if (unit.window && unit.window.mode === "relative") unit.window = null;
      if (!unit.options) unit.options = {};
      unit.options.mode = "manual";
    } else {
      const entries = unit.manual && Array.isArray(unit.manual.entries)
        ? unit.manual.entries : [];
      if (entries.length && !window.confirm(
          "已有 " + entries.length + " 条人工记录，切回自动后不再使用。确认切换？")) {
        return;
      }
      unit.manual = null;
      if (!unit.options) unit.options = {};
      unit.options.mode = "auto";
    }
    // 任何模式切换都废弃旧预览确认与在途画面请求。
    stopPlayback();
    invalidatePreview();
    state.imageToken += 1;
    state.recordMode = mode === "manual";
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

  // 目标洞只取 1…洞数 的洞号；洞数是参数，选择项随它变化。
  function targetHoleChoices(unit, param) {
    const raw = Number(effectiveOption(unit, "hole_count", 20));
    const count = Number.isInteger(raw) && raw >= 2 && raw <= 100 ? raw : 20;
    const choices = [];
    for (let index = 1; index <= count; index += 1) {
      choices.push({ value: String(index), label: "洞 " + index });
    }
    const current = String(effectiveOption(unit, param.key, param.default) ?? "");
    if (current && !choices.some((choice) => choice.value === current)) {
      choices.unshift({ value: current, label: "洞 " + current });
    }
    return choices;
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
    const groups = [];
    const bySection = new Map();
    params.forEach((param) => {
      if (!param.section) {
        groups.push({ title: null, params: [param] });
        return;
      }
      let group = bySection.get(param.section);
      if (!group) {
        group = { title: param.section, params: [] };
        bySection.set(param.section, group);
        groups.push(group);
      }
      group.params.push(param);
    });
    groups.forEach((group) => {
      const fields = group.params.map((param) => parameterField(unit, param));
      if (!group.title) {
        fields.forEach((field) => wrap.append(field));
        return;
      }
      // 按描述符的 section 折叠：默认收起，不挤占当前任务内容。
      const section = document.createElement("details");
      section.className = "fold";
      const summary = document.createElement("summary");
      summary.textContent = group.title;
      const inner = document.createElement("div");
      inner.className = "fold-body";
      fields.forEach((field) => inner.append(field));
      section.append(summary, inner);
      wrap.append(section);
    });
    return wrap;
  }

  function parameterField(unit, param) {

    const label = document.createElement("label");
    label.className = "field";
    const caption = document.createElement("span");
    caption.textContent = param.label + (param.unit ? "（" + param.unit + "）" : "");
    let input;
    if (param.type === "select") {
      input = document.createElement("select");
      const choices = (state.assay === "barnes" && param.key === "target_hole")
        ? targetHoleChoices(unit, param) : (param.choices || []);
      choices.forEach((choice) => {
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
      if (param.key === "phase" && state.assay === "nor") {
        if (!target.geometry) target.geometry = {};
        window.BLCore.norNormalizeRoles(target.geometry, changed);
      }
      if (param.key === "trial_type" && state.assay === "barnes") {
        // 探测试次没有入箱：切到 probe 立即清掉确认时间并保存；切回不再恢复。
        if (!target.manual || typeof target.manual !== "object") target.manual = {};
        if (changed === "probe") target.manual.escape_time_s = null;
      }
      invalidatePreview();
      markDirty();
      if (param.key === "mode") {
        state.recordMode = changed === "manual";
        if (state.recordMode) state.view = "geometry";
        renderTasknav();
      }
      if (param.key === "dark_box_visible" && state.assay === "ldb") {
        // 可见性切换立即改变几何步骤与检测区域，废弃预览并重选步骤。
        ensureContext();
        state.view = "geometry";
      }
      refreshEditorSpec();
      renderInspector();
      draw();
    });
    label.append(caption, input);
    return label;
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

  // -- 人工 Y：记录、撤销与只读复核 -----------------------------------
  const ARM_LABELS = { A: "A 臂", B: "B 臂", C: "C 臂" };

  function playbackRow(options) {
    const opts = options || {};
    const wrap = document.createElement("div");
    wrap.className = "mode-row record-row";
    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "button small";
    toggle.dataset.playToggle = "1";
    toggle.textContent = state.playing ? "暂停" : "播放";
    toggle.addEventListener("click", togglePlayback);
    wrap.append(toggle);
    if (opts.rates !== false) {
      [0.5, 1, 2].forEach((rate) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "mode-button" + (state.rate === rate ? " is-current" : "");
        button.textContent = rate + "×";
        button.addEventListener("click", () => { setRate(rate); });
        wrap.append(button);
      });
    }
    return wrap;
  }

  function armRow() {
    const wrap = document.createElement("div");
    wrap.className = "mode-row record-row";
    ["A", "B", "C"].forEach((arm) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "mode-button";
      button.textContent = ARM_LABELS[arm];
      button.addEventListener("click", () => { appendManualEntry(arm); });
      wrap.append(button);
    });
    return wrap;
  }

  function manualList(unit, readOnly) {
    const entries = unit && unit.manual && Array.isArray(unit.manual.entries)
      ? unit.manual.entries : [];
    const list = document.createElement("ul");
    list.className = "event-list";
    if (!entries.length) {
      const empty = document.createElement("li");
      empty.className = "muted";
      empty.textContent = "尚未记录入臂";
      list.append(empty);
      return list;
    }
    entries.forEach((entry) => {
      const item = document.createElement("li");
      const label = document.createElement("span");
      const moment = Number(entry.time_s);
      label.textContent = timeText(moment, 3) + " · " + (ARM_LABELS[entry.arm] || entry.arm);
      item.append(label);
      list.append(item);
    });
    if (!readOnly) {
      // 列表按录入顺序显示；新增后滚到末尾。
      requestAnimationFrame(() => { list.scrollTop = list.scrollHeight; });
    }
    return list;
  }

  function appendManualEntry(arm) {
    const unit = currentUnit();
    if (!unit || state.assay !== "ymaze") return;
    const info = state.frameInfo;
    if (!info || info.frame !== state.frame || !Number.isFinite(info.time)) {
      status("正在读取源帧，稍后再记录", "warn");
      return;
    }
    const manual = ensureManual(unit);
    const entries = manual.entries;
    const last = entries.length ? Number(entries[entries.length - 1].time_s) : null;
    if (last !== null && Number.isFinite(last) && info.time <= last + 1e-9) {
      status("入臂时间必须晚于上一条记录", "error");
      return;
    }
    // 时间只取当前可见帧响应头，不取滑条、请求值或名义帧率。
    entries.push({ time_s: info.time, arm });
    invalidatePreview();
    markDirty();
    renderInspector();
    rememberPosition();
    status("");
  }

  function undoManualEntry() {
    const unit = currentUnit();
    if (!unit || !unit.manual || !Array.isArray(unit.manual.entries)) return;
    if (!unit.manual.entries.length) return;
    unit.manual.entries.pop();
    invalidatePreview();
    markDirty();
    renderInspector();
    rememberPosition();
  }

  function clearManualEntries() {
    const unit = currentUnit();
    if (!unit || !unit.manual || !Array.isArray(unit.manual.entries)) return;
    if (!unit.manual.entries.length) return;
    unit.manual.entries = [];
    invalidatePreview();
    markDirty();
    renderInspector();
    rememberPosition();
  }

  function binSecondsField(unit) {
    const label = document.createElement("label");
    label.className = "field";
    const caption = document.createElement("span");
    caption.textContent = "时间箱长度（s）";
    const input = document.createElement("input");
    input.type = "number";
    input.min = "0";
    input.step = "any";
    const current = effectiveOption(unit, "bin_seconds", 60);
    input.value = current == null ? "" : String(current);
    input.addEventListener("change", () => {
      const target = currentUnit();
      if (!target) return;
      const value = Number(input.value);
      if (!Number.isFinite(value) || value <= 0) {
        status("时间箱长度必须大于 0", "error");
        input.value = String(effectiveOption(target, "bin_seconds", 60));
        return;
      }
      if (!target.options) target.options = {};
      target.options.bin_seconds = value;
      invalidatePreview();
      markDirty();
      renderInspector();
    });
    label.append(caption, input);
    return label;
  }

  // 动物时间窗：跟随视频 / 指定时间 / 首次检测后。编辑一开始就撤销旧确认；
  // 非法或未填完的值按“视频+动物”原样保留并阻断推进，不用最后一次合法值顶替。
  function animalWindowField(unit, options) {
    const opts = options || {};
    const allowRelative = opts.allowRelative !== false;
    const video = currentVideo();
    const duration = videoDuration();
    const edit = windowEditOf(unit);
    const current = unit.window && typeof unit.window === "object" ? unit.window : null;
    const mode = edit ? edit.mode
      : (current && (current.mode === "exact" ||
        (current.mode === "relative" && allowRelative)) ? current.mode : "global");
    const raw = edit ? edit.values : null;

    const wrap = document.createElement("div");
    wrap.className = "field-stack window-field";
    const selectLabel = document.createElement("label");
    selectLabel.className = "field";
    const caption = document.createElement("span");
    caption.textContent = "动物时间窗";
    const select = document.createElement("select");
    const choices = [["global", "跟随视频"], ["exact", "指定时间"]];
    if (allowRelative) choices.push(["relative", "首次检测后"]);
    choices.forEach(([value, text]) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = text;
      option.selected = mode === value;
      select.append(option);
    });
    selectLabel.append(caption, select);
    wrap.append(selectLabel);

    const fieldRow = (labels) => {
      const row = document.createElement("div");
      row.className = "field-row";
      labels.forEach(([text, input]) => {
        const label = document.createElement("label");
        label.className = "field";
        const span = document.createElement("span");
        span.textContent = text;
        label.append(span, input);
        row.append(label);
      });
      return row;
    };
    const numberInput = (field, fallback, minimum) => {
      const input = document.createElement("input");
      input.type = "number";
      input.step = "any";
      input.dataset.windowField = field;
      // 只有真正有下限的字段才设 min：相对偏移允许负值。
      if (minimum != null) input.min = String(minimum);
      const value = raw && raw[field] != null ? raw[field] : fallback;
      input.value = value == null ? "" : String(value);
      return input;
    };

    const startInput = numberInput("start_s",
      current && current.start_s != null ? current.start_s
        : (video && video.window && video.window.start_s) || 0, 0);
    const endInput = numberInput("end_s",
      current && current.end_s != null ? current.end_s : duration, 0);
    const offsetInput = numberInput("offset_s",
      current && current.offset_s != null ? current.offset_s : 0, null);
    const spanInput = numberInput("duration_s",
      current && current.duration_s != null ? current.duration_s : 2, 0);

    const exactRow = fieldRow([["起点（秒）", startInput], ["终点（秒）", endInput]]);
    const relativeRow = fieldRow([["偏移（秒）", offsetInput], ["持续时间（秒）", spanInput]]);
    const errorNode = document.createElement("p");
    errorNode.className = "field-error";
    errorNode.hidden = true;
    wrap.append(exactRow, relativeRow, errorNode);

    const inputs = {
      start_s: startInput, end_s: endInput,
      offset_s: offsetInput, duration_s: spanInput,
    };
    const markField = (field) => {
      Object.keys(inputs).forEach((key) => {
        if (field && key === field) inputs[key].setAttribute("aria-invalid", "true");
        else inputs[key].removeAttribute("aria-invalid");
      });
    };
    const updateRows = () => {
      exactRow.hidden = select.value !== "exact";
      relativeRow.hidden = select.value !== "relative";
    };
    updateRows();
    if (edit && edit.error) {
      markField(edit.field);
      errorNode.hidden = false;
      errorNode.textContent = edit.error;
    }

    const commit = () => {
      const target = currentUnit();
      if (!target) return false;
      const values = {
        start_s: startInput.value,
        end_s: endInput.value,
        offset_s: offsetInput.value,
        duration_s: spanInput.value,
      };
      const result = window.BLCore.assignWindow(null, select.value, values);
      // 用户一改，上一份确认就不再对应当前界面。
      invalidatePreview();
      if (result.error) {
        state.windowEdits.set(windowEditKey(target),
          { mode: select.value, values, error: result.error, field: result.field });
        markField(result.field);
        errorNode.hidden = false;
        errorNode.textContent = result.error;
        state.windowStatus = { key: windowEditKey(target), text: result.error };
        status(result.error, "error");
        return false;
      }
      const fixedKey = windowEditKey(target);
      if (state.windowStatus && state.windowStatus.key === fixedKey &&
          statusNode().textContent === state.windowStatus.text) {
        // 只清掉这一处还没解决的提示，别的单位的问题保留。
        status("");
      }
      if (state.windowStatus && state.windowStatus.key === fixedKey) {
        state.windowStatus = null;
      }
      state.windowEdits.delete(fixedKey);
      target.window = result.window;
      markField(null);
      errorNode.hidden = true;
      errorNode.textContent = "";
      markDirty();
      return true;
    };

    select.addEventListener("change", () => {
      // 校验失败时保留刚选的模式与已输入值，只报字段原因，不静默回退。
      updateRows();
      commit();
    });
    [startInput, endInput, offsetInput, spanInput].forEach((input) => {
      input.addEventListener("input", () => { commit(); });
      input.addEventListener("change", () => { commit(); });
    });
    return wrap;
  }

  function appendManualSettings(body, unit) {
    const settings = document.createElement("details");
    settings.className = "fold";
    const summary = document.createElement("summary");
    summary.textContent = "实验设置";
    const inner = document.createElement("div");
    inner.className = "fold-body";
    inner.append(binSecondsField(unit), animalWindowField(unit, { allowRelative: false }));
    settings.append(summary, inner);
    body.append(settings);

    const animal = document.createElement("details");
    animal.className = "fold";
    const animalSummary = document.createElement("summary");
    animalSummary.textContent = "小鼠设置";
    const animalBody = document.createElement("div");
    animalBody.className = "fold-body";
    animalBody.append(nameField(unit));
    const remove = unitDeleteField(unit);
    if (remove) animalBody.append(remove);
    animal.append(animalSummary, animalBody);
    body.append(animal);
  }

  function renderRecordInspector(body, unit) {
    const title = document.createElement("h3");
    title.textContent = "人工入臂记录";
    body.append(title);
    body.append(playbackRow());
    body.append(armRow());
    const hint = document.createElement("p");
    hint.className = "muted";
    hint.dataset.recordHint = "1";
    body.append(hint);
    body.append(manualList(unit, false));
    const actions = document.createElement("div");
    actions.className = "mode-row";
    const undo = document.createElement("button");
    undo.type = "button";
    undo.className = "button small";
    undo.textContent = "撤销";
    undo.addEventListener("click", undoManualEntry);
    const clear = document.createElement("button");
    clear.type = "button";
    clear.className = "button small";
    clear.textContent = "清空";
    clear.addEventListener("click", clearManualEntries);
    actions.append(undo, clear);
    body.append(actions);
    appendManualSettings(body, unit);
    appendPathMode(body);
    el.primary.textContent = "下一步：复核";
    el.secondary.hidden = true;
    el.preview.hidden = true;
    syncPlayButton();
    syncRecordHint();
  }

  function renderReviewInspector(body, unit) {
    const title = document.createElement("h3");
    title.textContent = "复核画面";
    body.append(title);
    body.append(playbackRow({ rates: false }));
    body.append(manualList(unit, true));
    appendAnnotationToggle(body);

    const confirmed = state.previewGate.confirmed(previewConfig("behavior"));
    el.primary.textContent = confirmed ? "检查并开始分析" : "行为预览";
    el.secondary.textContent = "更新行为预览";
    el.secondary.hidden = !confirmed;
    el.preview.hidden = !confirmed && !state.previewAwaiting;
    syncPlayButton();
  }

  // -- 预览与校验 -----------------------------------------------------
  function previewConfig(kind) {
    const video = currentVideo();
    const unit = currentUnit();
    return {
      kind: kind || "detection",
      assay: state.assay,
      asset_id: video.asset_id,
      // 绑定实际显示过的帧，而不是还在请求中的目标帧。
      frame_index: Number.isInteger(state.shownFrame) ? state.shownFrame : state.frame,
      orientation: video.orientation || 0,
      fps_override: video.fps_override,
      sample_fps: video.sample_fps,
      window: clone(video.window),
      unit: clone(unit),
      draft_options: clone(state.draft.options || {}),
      draft_detection: clone(state.draft.detection || {}),
      video_options: clone(video.options || {}),
      video_detection: clone(video.detection || {}),
      // 单调编辑代号：值改回原样也必须重新预览。
      edit: state.editCounter,
    };
  }

  async function runPreview(kind) {
    if (!state.draftId) return;
    if (blockOnWindowError()) return;
    const formatError = manualFormatError();
    if (formatError) {
      // 非法人工格式不落盘、不预览：先修好输入。
      status(formatError.message, "error");
      state.manualStatus = { unitId: formatError.unitId, text: formatError.message };
      revealManualIssue(formatError);
      return;
    }
    // 只接受显式的阶段名；事件对象等非字符串一律按当前阶段处理。
    const type = typeof kind === "string"
      ? kind : (state.step === "review" ? "behavior" : "detection");
    state.previewKind = type;
    const video = currentVideo();
    const unit = currentUnit();
    if (!video || !unit) return;
    el.primary.disabled = true;
    el.secondary.disabled = true;
    const extracting = type === "behavior"
      && currentUnit() && currentUnit().options
      && currentUnit().options.mode === "lis";
    status(extracting ? "提取运动特征中，状态待定"
      : (type === "behavior" ? "正在生成行为预览…" : "正在生成检测预览…"));
    let request = null;
    // 页面上下文：离开这个页/视频/单位后，旧预览响应不能再确认或改新页状态。
    const context = { page: pageKey(), edit: state.editCounter };
    try {
      // 先落盘，再核对页面仍指向同一个视频/小鼠，最后才请求预览。
      const result = await window.BLCore.saveThenPreview({
        dirty: isDirty,
        save: saveThen,
        alive: () => {
          const live = currentVideo();
          const liveUnit = currentUnit();
          return Boolean(live && liveUnit && live.asset_id === video.asset_id &&
            liveUnit.id === unit.id && pageAlive(context));
        },
        config: () => previewConfig(type),
        send: (config) => {
          request = state.previewGate.begin(config);
          const body = {
            asset_id: config.asset_id,
            unit_id: config.unit.id,
            frame_index: config.frame_index,
            kind: config.kind,
          };
          // time_s 取当前可见帧的实际响应值；服务端仍按 frame_index 重读源图。
          if (state.frameInfo && Number.isFinite(state.frameInfo.time)) {
            body.time_s = state.frameInfo.time;
          }
          return send("POST", "/api/drafts/" + state.draftId + "/preview", body);
        },
      });
      if (result.status === "save_failed") {
        state.previewGate.invalidate();
        const reason = lastSaveText("保存失败，未生成预览");
        status(reason, "error");
        setPreviewNote({ tone: "error", text: reason });
        return;
      }
      if (result.status === "stale") {
        state.previewGate.invalidate();
        status("配置已改变，请重新预览", "warn");
        setPreviewNote({ tone: "warn", text: "配置已改变，请重新预览。" });
        return;
      }
      const data = result.data;
      if (!pageAlive(context)) {
        // 已经离开或已销毁页面：旧响应不得确认，也不得把图挂到新页面。
        state.previewGate.invalidate();
        state.previewAwaiting = false;
        return;
      }
      // 图片真正加载成功后才确认，加载失败不留下已确认状态。
      state.previewAwaiting = true;
      el.preview.onload = () => {
        state.previewAwaiting = false;
        if (!pageAlive(context)) {
          el.preview.hidden = true;
          return;
        }
        if (!state.previewGate.accept(request, data)) {
          el.preview.hidden = true;
          return;
        }
        setPreviewNote(window.BLCore.previewNoteFor(data));
        const scopeMode = data.preview_scope && data.preview_scope.mode;
        const scopeLabel = scopeMode === "frame" ? "本帧预览"
          : scopeMode === "configured" ? "配置范围预览" : "片段预览";
        status(scopeLabel + " · 帧 " + data.frame_index, "ok");
        renderInspector();
      };
      el.preview.onerror = () => {
        state.previewAwaiting = false;
        if (!pageAlive(context)) {
          el.preview.hidden = true;
          return;
        }
        state.previewGate.fail(request);
        el.preview.hidden = true;
        setPreviewNote({ tone: "error", text: "预览图加载失败。" });
        status("预览图加载失败", "error");
        renderInspector();
      };
      el.preview.src = data.image_url;
      el.preview.hidden = false;
    } catch (error) {
      if (request) state.previewGate.fail(request);
      // 页面已销毁或已换页时，失败提示也不写到新页面上。
      if (state.pageClosed || pageKey() !== context.page) return;
      const reason = issueText(error, "预览失败");
      setPreviewNote({ tone: "error", text: reason });
      status(reason, "error");
    } finally {
      el.primary.disabled = false;
      el.secondary.disabled = false;
      if (!state.pageClosed) renderInspector();
    }
  }

  async function primaryAction() {
    if (blockOnWindowError()) return;
    const formatError = manualFormatError();
    if (formatError) {
      status(formatError.message, "error");
      state.manualStatus = { unitId: formatError.unitId, text: formatError.message };
      revealManualIssue(formatError);
      return;
    }
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
        step.required && !stepShape(unit, step.key));
      if (missing.length) {
        selectStep(missing[0].key);
        status("缺少：" + missing[0].label, "warn");
        canvas.focus();
        return;
      }
      await goStep("arena", { view: "detection" });
      return;
    }
    if (state.previewKind === "behavior") {
      if (state.previewGate.confirmed(previewConfig("behavior"))) {
        await startAnalysis();
      } else {
        await runPreview("behavior");
      }
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
    if (blockOnWindowError()) return;
    stopPlayback();
    const kind = state.step === "review" ? "behavior"
      : (state.previewKind === "behavior" ? "behavior" : "detection");
    const config = previewConfig(kind);
    if (!state.previewGate.confirmed(config)) {
      status("配置已改变，请重新预览", "warn");
      renderInspector();
      return;
    }
    // 第一次异步等待之前冻结提交上下文：页面、单位、编辑代号与确认凭据。
    const context = {
      navToken: state.navToken,
      draftId: state.draftId,
      step: state.step,
      videoIndex: state.videoIndex,
      unitIndex: state.unitIndex,
      edit: state.editCounter,
      config,
    };
    const submission = () => {
      if (state.navToken !== context.navToken
          || state.draftId !== context.draftId
          || state.step !== context.step
          || state.videoIndex !== context.videoIndex
          || state.unitIndex !== context.unitIndex) {
        return "page";
      }
      if (state.editCounter !== context.edit
          || !state.previewGate.confirmed(context.config)) {
        return "config";
      }
      return "ok";
    };
    const abort = () => {
      const stateNow = submission();
      if (stateNow === "config") {
        status("配置已改变，请重新预览", "warn");
        renderInspector();
      }
      return stateNow !== "ok";
    };

    const settled = await saveThen();
    if (!settled) {
      status(lastSaveText("保存失败，未开始分析"), "error");
      return;
    }
    if (abort()) return;
    el.primary.disabled = true;
    status("正在校验草稿…");
    try {
      // 先取最终校验，issues 为空才创建任务，不留半个作业。
      const checked = await send("POST", "/api/drafts/" + context.draftId + "/validate", {});
      if (abort()) return;
      const issues = (checked && checked.issues) || [];
      if (issues.length) {
        status(issueText({ fields: issues }, "草稿未通过校验"), "error");
        renderInspector();
        return;
      }
      // 校验返回后、提交前再核对一次，旧校验不能当新配置通过。
      if (abort()) return;
      status("正在分析…");
      const data = await send("POST", "/api/jobs", { draft_id: context.draftId });
      const failed = (data.jobs || []).find((job) => job.status !== "done");
      if (failed) {
        status(failed.error || "分析失败", "error");
        return;
      }
      status("分析完成", "ok");
      if (submission() !== "ok") {
        // 已离开或已改配置：保留用户当前页面，只提示任务已开始。
        status("任务已开始，可在任务页查看", "ok");
        return;
      }
      window.location.href = "/results/" + data.run_id;
    } catch (error) {
      // 校验失败时给出可定位原因，而不是只显示“草稿未通过校验”。
      status(issueText(error, "分析失败"), "error");
    } finally {
      el.primary.disabled = false;
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
        ensureTopContext();
        renderTasknav();
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
    if (el.backToSelect) {
      el.backToSelect.addEventListener("click", () => { goStep("select"); });
    }
    if (el.rangeSlide) {
      el.rangeSlide.addEventListener("input", () => {
        const video = currentVideo();
        if (video) loadRangeFrame(video, Number(el.rangeSlide.value));
      });
    }
    if (el.rangeIndex) {
      const commitRangeFrame = () => {
        const video = currentVideo();
        if (!video) return;
        const value = Number(el.rangeIndex.value);
        if (!Number.isFinite(value)) {
          el.rangeIndex.value = String(state.rangeFrame);
          return;
        }
        loadRangeFrame(video, Math.round(value));
      };
      el.rangeIndex.addEventListener("change", commitRangeFrame);
      el.rangeIndex.addEventListener("keydown", (event) => {
        if (event.key !== "Enter") return;
        event.preventDefault();
        commitRangeFrame();
        el.rangeIndex.blur();
      });
    }

    el.videoSelect.addEventListener("change", () => {
      stopPlayback();
      state.videoIndex = Number(el.videoSelect.value);
      state.unitIndex = 0;
      state.frame = 0;
      state.shownFrame = 0;
      state.stepKey = null;
      state.editor.reset(null, null);
      ensureContext();
      loadFrame();
      renderInspector();
    });
    el.unitSelect.addEventListener("change", () => {
      stopPlayback();
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
      if (result === "geometry" && unit && unit.geometry) {
        window.BLCore.stepClear(state.assay, unit.geometry, state.stepKey);
        markDirty();
        state.previewGate.invalidate();
      }
      // 撤销点或撤销整段几何都要立刻同步侧栏的已绘制/缺失状态。
      if (result) renderInspector();
      draw();
    });
    el.toolRedraw.addEventListener("click", () => {
      const unit = currentUnit();
      state.editor.redraw();
      if (unit && unit.geometry) {
        window.BLCore.stepClear(state.assay, unit.geometry, state.stepKey);
        markDirty();
        state.previewGate.invalidate();
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
      // 先记住用户选定的目标，再停播：暂停会把滑条写回最后可见帧，
      // 停播后读取就丢掉了这次明确跳转。
      const target = Number(el.frameRange.value);
      stopPlayback();
      loadFrame(target);
    });
    el.frameRange.addEventListener("change", () => {
      const target = Number(el.frameRange.value);
      stopPlayback();
      loadFrame(target);
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
      // 沿用当前预览类型：行为预览下点“更新行为预览”不能退回单帧检测。
      runPreview(state.step === "review" || state.previewKind === "behavior" ? "behavior" : "detection");
    });

    if (el.topContext) {
      el.topContext.addEventListener("click", (event) => {
        const link = event.target.closest("[data-step-link]");
        if (!link) return;
        event.preventDefault();
        const step = link.dataset.stepLink;
        const declared = link.dataset.stepView || "";
        const targetView = step === "arena"
          ? (declared === "detection" ? "detection" : "geometry")
          : state.view;
        if (step === state.step && (step !== "arena" || targetView === state.view)) return;
        // 还没有草稿时留在选视频；有草稿但没视频时回视频设置，不落到空画布。
        if (step === "arena" && !state.draftId) {
          if (state.step !== "select") goStep("select");
          status("先选择视频并设置范围", "warn");
          return;
        }
        if (step === "arena" && !currentVideo()) {
          goStep("range");
          return;
        }
        goStep(step, { view: targetView });
      });
    }

    window.addEventListener("popstate", async (event) => {
      const target = event.state && event.state.analysis;
      stopPlayback();
      const token = ++state.navToken;
      const ok = await saveThen();
      if (token !== state.navToken) return;
      if (!ok) {
        history.pushState({ analysis: analysisState() }, "", location.href);
        status(lastSaveText("保存失败，未切换历史记录"), "error");
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
        state.shownFrame = state.frame;
        state.rangeFrame = Number.isInteger(target.rangeFrame)
          ? target.rangeFrame : state.frame;
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
        status(lastSaveText("保存失败，仍留在当前页"), "error");
        return;
      }
      rememberPosition();
      location.href = url.href;
    });

    document.addEventListener("keydown", manualHotkeys);

    document.addEventListener("keydown", (event) => {
      if (event.key !== "Escape" || !state.pendingNav) return;
      state.navToken += 1;
      state.pendingNav = null;
      status("已取消跳转", "warn");
    });

    window.addEventListener("pagehide", teardownPage);
    window.addEventListener("beforeunload", (event) => {
      // 只记录位置并请求离页提醒；用户取消离页时页面仍然存活，
      // 不能在这里解绑快捷键或停掉播放器。
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
    stopPlayback();
    el.frameRange.value = String(next);
    // 实际显示成功后才更新历史帧。
    loadFrame(next);
  }

  function manualHotkeys(event) {
    if (event.defaultPrevented || event.metaKey || event.ctrlKey
        || event.altKey || event.shiftKey) {
      return;
    }
    const target = event.target;
    const tag = target && target.tagName ? target.tagName.toLowerCase() : "";
    if (tag === "input" || tag === "select" || tag === "textarea"
        || (target && target.isContentEditable)) {
      return;
    }
    if (!state.recordMode || state.assay !== "ymaze") return;
    if (state.step !== "arena" || !state.draft) return;
    if (event.key === " " || event.key === "Spacebar") {
      event.preventDefault();
      togglePlayback();
      return;
    }
    const arm = event.key ? event.key.toUpperCase() : "";
    if (arm === "A" || arm === "B" || arm === "C") {
      event.preventDefault();
      appendManualEntry(arm);
    }
  }

  // 页面销毁时停播并解除键盘监听，旧请求不再作用于新页。
  function teardownPage() {
    state.pageClosed = true;
    // 销毁时连同预览请求与确认一起废弃：稍后返回的 JSON/PNG 不能再确认。
    state.previewGate.invalidate();
    state.previewAwaiting = false;
    stopPlayback();
    document.removeEventListener("keydown", manualHotkeys);
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
    registerFieldLabels();
    ensureTopContext();
    renderTasknav();
    const entry = history.state && history.state.analysis;
    if (entry && entry.draftId === state.draftId) {
      state.videoIndex = Number.isInteger(entry.video) ? entry.video : 0;
      state.unitIndex = Number.isInteger(entry.unit) ? entry.unit : 0;
      state.frame = Number.isInteger(entry.frame) ? entry.frame : 0;
      // 恢复阶段还没显示新帧前，历史就按这个目标帧继续。
      state.shownFrame = state.frame;
      state.rangeFrame = Number.isInteger(entry.rangeFrame)
        ? entry.rangeFrame : state.frame;
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
