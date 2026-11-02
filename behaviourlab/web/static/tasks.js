/* 任务中心：真实批次状态、搜索、停止/重跑/删除与轮询。
   进度条只表示进度，状态文字始终以服务器返回为准。 */
(function () {
  "use strict";

  const root = document.querySelector("#task-list");
  if (!root) return;

  const STATUS_LABELS = {
    queued: "等待中", running: "分析中", completed: "已完成",
    failed: "失败", cancelled: "已停止", interrupted: "已中断",
    not_started: "未开始",
  };
  const POLL_MS = 1500;

  const el = {
    search: document.querySelector("#task-search"),
    clearSearch: document.querySelector("#task-clear-search"),
    noMatch: document.querySelector("#task-no-match"),
    empty: document.querySelector("#task-empty"),
    count: document.querySelector("[data-task-count]"),
    error: document.querySelector("#task-read-error"),
    errorText: document.querySelector("#task-read-error [data-error-text]"),
    retry: document.querySelector("#task-retry"),
  };

  const state = { batches: [], query: "", timer: null, pending: new Set(),
    errors: new Map() };

  // 列表范围：项目页把自己的 project_id 交给任务列表，之后所有刷新（轮询、
  // 重试、动作后）都用同一个 URL；全局任务页没有范围，照旧列出全部任务。
  const projectScope = (root.dataset.project || "").trim();

  function listUrl() {
    return projectScope
      ? "/api/jobs?project_id=" + encodeURIComponent(projectScope)
      : "/api/jobs";
  }

  function inScope(batch) {
    // 服务端已按项目过滤；这里再挡一次，属主不符的卡片一律不渲染。
    if (!projectScope) return true;
    if (!Object.prototype.hasOwnProperty.call(batch, "project_id")) return true;
    return batch.project_id === projectScope;
  }

  // -- 读取 -------------------------------------------------------------
  async function request(url, options) {
    let response;
    try {
      response = await fetch(url, options);
    } catch (error) {
      throw new Error("网络错误，未能连接本地服务");
    }
    let payload = null;
    const text = await response.text();
    if (text) {
      try {
        payload = JSON.parse(text);
      } catch (error) {
        throw new Error("服务返回了非 JSON 响应");
      }
    }
    if (!response.ok || !payload || payload.ok !== true) {
      const message = (payload && payload.error && payload.error.message)
        || ("请求失败（HTTP " + response.status + "）");
      const fields = (payload && payload.error && payload.error.fields) || [];
      const detail = fields.map((item) => item.message).filter(Boolean).join("；");
      const error = new Error(detail ? message + "：" + detail : message);
      error.fields = fields;
      throw error;
    }
    return payload.data;
  }

  function send(url, options) {
    return request(url, Object.assign({ headers: { "Content-Type": "application/json" } },
      options, { body: JSON.stringify(options && options.body ? options.body : {}) }));
  }

  async function refresh() {
    try {
      const data = await request(listUrl());
      state.batches = (data.batches || []).filter(inScope);
      if (el.error) el.error.hidden = true;
      render();
    } catch (error) {
      // 读取失败保留最后一次真实状态，不显示成零任务，也不退成全局列表。
      if (el.error) {
        el.error.hidden = false;
        el.errorText.textContent = "读取任务失败：" + error.message;
      }
    }
    schedule();
  }

  function schedule() {
    clearTimeout(state.timer);
    const active = state.batches.some((batch) =>
      batch.status === "queued" || batch.status === "running");
    state.timer = active ? setTimeout(refresh, POLL_MS) : null;
  }

  // -- 渲染 -------------------------------------------------------------
  function matches(batch) {
    if (!state.query) return true;
    const needle = state.query.toLowerCase();
    return (batch.videos || []).some((video) =>
      String(video.name || "").toLowerCase().includes(needle));
  }

  function statusOf(value) {
    return STATUS_LABELS[value] || value || "";
  }

  function percentOf(batch) {
    const fraction = (batch.progress && batch.progress.fraction) || 0;
    return Math.max(0, Math.min(100, Math.floor(fraction * 100)));
  }

  function renderBatch(batch) {
    const item = document.createElement("li");
    item.className = "batch";
    item.id = "job-" + batch.id;
    item.dataset.batch = batch.id;
    item.dataset.status = batch.status;

    const head = document.createElement("div");
    head.className = "batch-head";
    const title = document.createElement("div");
    title.className = "batch-title";
    const assay = document.createElement("span");
    assay.className = "batch-assay";
    assay.textContent = batch.assay_label || batch.assay || "";
    const names = document.createElement("span");
    names.className = "batch-videos";
    names.textContent = (batch.videos || []).map((video) => video.name || "视频")
      .join(" · ");
    title.append(assay, names);

    const meta = document.createElement("div");
    meta.className = "batch-meta";
    const pill = document.createElement("span");
    pill.className = "status-pill";
    pill.dataset.status = batch.status || "";
    pill.textContent = statusOf(batch.status);
    const when = document.createElement("span");
    when.className = "meta";
    when.textContent = String(batch.updated_at || "").slice(0, 16).replace("T", " ");
    meta.append(pill, when);
    head.append(title, meta);

    const percent = percentOf(batch);
    const bar = document.createElement("div");
    bar.className = "progress";
    bar.setAttribute("role", "progressbar");
    bar.setAttribute("aria-valuemin", "0");
    bar.setAttribute("aria-valuemax", "100");
    bar.setAttribute("aria-valuenow", String(percent));
    const fill = document.createElement("span");
    fill.style.width = percent + "%";
    bar.append(fill);

    const message = document.createElement("p");
    message.className = "batch-message";
    message.textContent = (batch.progress && batch.progress.message) || "";

    item.append(head, bar, message);

    if (batch.error && batch.status !== "completed") {
      const error = document.createElement("p");
      error.className = "batch-error";
      error.textContent = batch.error;
      item.append(error);
    }

    const list = document.createElement("ol");
    list.className = "video-list";
    (batch.videos || []).forEach((video, index) => {
      const row = document.createElement("li");
      row.className = "video-row";
      row.dataset.video = video.run_id || String(index);
      row.dataset.status = video.status || "";
      const name = document.createElement("span");
      name.className = "video-name";
      name.textContent = video.name || "视频";
      const status = document.createElement("span");
      status.className = "video-status";
      status.textContent = statusOf(video.status);
      row.append(name, status);
      if (video.error) {
        const error = document.createElement("span");
        error.className = "video-error";
        error.textContent = video.error;
        row.append(error);
      }
      if (video.status === "completed" && video.run_id) {
        const link = document.createElement("a");
        link.className = "link";
        link.href = "/results/" + video.run_id;
        link.textContent = "查看结果";
        row.append(link);
      }
      list.append(row);
    });
    item.append(list);

    const actions = document.createElement("div");
    actions.className = "batch-actions";
    const active = batch.status === "queued" || batch.status === "running";
    const buttons = active ? ["cancel"] : ["replay", "delete"];
    buttons.forEach((action) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "button small" + (action === "delete" ? " danger" : "");
      button.dataset.action = action;
      button.textContent = action === "cancel" ? "停止"
        : action === "replay" ? "按原配置重新分析" : "删除";
      if (state.pending.has(batch.id + ":" + action)) {
        button.disabled = true;
        if (action === "cancel") button.textContent = "正在停止";
      }
      actions.append(button);
    });
    const error = document.createElement("span");
    error.className = "form-error";
    error.setAttribute("role", "alert");
    error.dataset.batchError = "";
    error.textContent = state.errors.get(batch.id) || "";
    actions.append(error);
    item.append(actions);
    return item;
  }

  function render() {
    const visible = state.batches.filter(matches);
    root.replaceChildren(...visible.map(renderBatch));
    if (el.empty) el.empty.hidden = state.batches.length > 0;
    if (el.noMatch) {
      el.noMatch.hidden = !(state.batches.length > 0 && visible.length === 0);
    }
    if (el.count) {
      el.count.textContent = state.query
        ? visible.length + " / " + state.batches.length + " 个批次"
        : state.batches.length + " 个批次";
    }
    const selected = root.dataset.selected;
    if (selected) {
      const target = document.getElementById("job-" + selected);
      if (target) target.classList.add("is-selected");
    }
  }

  // -- 动作 -------------------------------------------------------------
  async function act(batch, action) {
    const key = batch.id + ":" + action;
    if (state.pending.has(key)) return;
    if (action === "replay"
        && !window.confirm("使用完整原配置重新分析全部视频？")) return;
    if (action === "delete") {
      const ok = window.confirm(
        "将删除该任务记录和它拥有的运行产物，不删除原片或项目动物。继续？");
      if (!ok) return;
    }
    state.pending.add(key);
    state.errors.delete(batch.id);
    render();
    try {
      if (action === "delete") {
        await request("/api/jobs/" + batch.id, { method: "DELETE" });
      } else {
        await send("/api/jobs/" + batch.id + "/" + action, { method: "POST", body: {} });
      }
      state.pending.delete(key);
      await refresh();
    } catch (error) {
      state.pending.delete(key);
      state.errors.set(batch.id, error.message);
      render();
    }
  }

  root.addEventListener("click", (event) => {
    const button = event.target.closest("[data-action]");
    if (!button) return;
    const item = button.closest("[data-batch]");
    if (!item) return;
    const batch = state.batches.find((entry) => String(entry.id) === item.dataset.batch);
    if (!batch) return;
    act(batch, button.dataset.action);
  });

  if (el.search) {
    el.search.addEventListener("input", () => {
      state.query = el.search.value.trim();
      render();
    });
  }
  if (el.clearSearch) {
    el.clearSearch.addEventListener("click", () => {
      state.query = "";
      if (el.search) el.search.value = "";
      render();
    });
  }
  if (el.retry) el.retry.addEventListener("click", () => refresh());

  window.addEventListener("pagehide", () => clearTimeout(state.timer));

  const seed = document.querySelector("#task-seed");
  if (seed) {
    try {
      const parsed = JSON.parse(seed.textContent || "[]");
      state.batches = (Array.isArray(parsed) ? parsed : []).filter(inScope);
    } catch (error) {
      state.batches = [];
    }
  }
  render();
  schedule();
})();
