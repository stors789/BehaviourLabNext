(() => {
  "use strict";

  const root = document.querySelector("[data-analyze]");
  if (!root) return;

  const mode = root.dataset.mode === "independent" ? "independent" : "project";
  const videoAccept =
    ".mp4,.avi,.mov,.mkv,.m4v,.wmv,.mpg,.mpeg,.mts,.m2ts,.webm,.flv";
  const videoExtensions = videoAccept.split(",");

  function isVideoFile(file) {
    const name = String(file.name || "").toLowerCase();
    return videoExtensions.some((extension) => name.endsWith(extension));
  }

  const state = {
    projectId: root.dataset.project || "",
    assets: [],
    selected: new Set(),
    busy: false,
  };

  const el = {
    project: document.querySelector("#project-select"),
    pickVideos: document.querySelector("#pick-videos"),
    pickFolder: document.querySelector("#pick-folder"),
    dropzone: document.querySelector("#dropzone"),
    status: document.querySelector("#import-status"),
    errors: document.querySelector("#import-errors"),
    assetList: document.querySelector("#asset-list"),
    assetEmpty: document.querySelector("#asset-empty"),
    assetCount: document.querySelector("#asset-count"),
    selectedList: document.querySelector("#selected-list"),
    selectedEmpty: document.querySelector("#selected-empty"),
    selectedCount: document.querySelector("#selected-count"),
    setup: document.querySelector("#setup-videos"),
    setupNote: document.querySelector("#setup-note"),
  };

  function hasProject() {
    return mode !== "project" || Boolean(state.projectId);
  }

  async function request(path, options = {}) {
    const response = await fetch(path, options);
    let payload = null;
    try {
      payload = await response.json();
    } catch (error) {
      payload = null;
    }
    if (!response.ok || !payload || payload.ok !== true) {
      const failure = payload && payload.error ? payload.error : {};
      const error = new Error(failure.message || `请求失败（${response.status}）`);
      error.fields = Array.isArray(failure.fields) ? failure.fields : [];
      throw error;
    }
    return payload.data;
  }

  function formatDuration(seconds) {
    const total = Math.round(Number(seconds));
    if (!Number.isFinite(total) || total <= 0) return "";
    const hours = Math.floor(total / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    const rest = total % 60;
    const pad = (value) => String(value).padStart(2, "0");
    return hours
      ? `${hours}:${pad(minutes)}:${pad(rest)}`
      : `${minutes}:${pad(rest)}`;
  }

  function metaText(asset) {
    const metadata = asset.metadata || {};
    const parts = [];
    if (metadata.width && metadata.height) {
      parts.push(`${metadata.width} × ${metadata.height}`);
    }
    const duration = formatDuration(metadata.duration_s);
    if (duration) parts.push(duration);
    return parts.join(" · ");
  }

  function setBusy(busy, message) {
    state.busy = busy;
    el.status.hidden = !busy;
    el.status.textContent = busy ? message || "正在导入视频…" : "";
    const disabled = busy || !hasProject();
    el.pickVideos.disabled = disabled;
    el.pickFolder.disabled = disabled;
    el.dropzone.classList.toggle("is-busy", busy);
    el.dropzone.classList.toggle("is-disabled", !hasProject());
  }

  function showErrors(errors) {
    el.errors.innerHTML = "";
    if (!Array.isArray(errors) || errors.length === 0) {
      el.errors.hidden = true;
      return;
    }
    const fragment = document.createDocumentFragment();
    for (const error of errors) {
      const item = document.createElement("li");
      const name = document.createElement("span");
      name.className = "error-name";
      name.textContent = error.name || "导入失败";
      const message = document.createElement("span");
      message.className = "error-message";
      message.textContent = error.message || "";
      item.append(name, message);
      fragment.append(item);
    }
    el.errors.append(fragment);
    el.errors.hidden = false;
  }

  function selectedAssets() {
    return state.assets.filter((asset) => state.selected.has(asset.id));
  }

  function updateSetup() {
    const count = selectedAssets().length;
    el.selectedCount.textContent = count ? `${count} 个` : "";
    el.setup.disabled = count === 0;
    if (count === 0 && el.setupNote) {
      el.setupNote.hidden = true;
    }
  }

  function toggle(assetId, checked) {
    if (checked) {
      state.selected.add(assetId);
    } else {
      state.selected.delete(assetId);
    }
    renderAssets();
    renderSelected();
  }

  function renderAssets() {
    el.assetList.innerHTML = "";
    el.assetEmpty.hidden = state.assets.length > 0;
    el.assetCount.textContent = state.assets.length ? `${state.assets.length} 个` : "";
    const fragment = document.createDocumentFragment();
    for (const asset of state.assets) {
      const item = document.createElement("li");
      item.className = "asset-row";
      const label = document.createElement("label");
      label.className = "asset-label";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.checked = state.selected.has(asset.id);
      checkbox.addEventListener("change", () => toggle(asset.id, checkbox.checked));
      const name = document.createElement("span");
      name.className = "asset-name";
      name.textContent = asset.name;
      label.append(checkbox, name);
      const meta = document.createElement("span");
      meta.className = "asset-meta";
      meta.textContent = metaText(asset);
      item.append(label, meta);
      fragment.append(item);
    }
    el.assetList.append(fragment);
  }

  function renderSelected() {
    el.selectedList.innerHTML = "";
    const assets = selectedAssets();
    el.selectedEmpty.hidden = assets.length > 0;
    const fragment = document.createDocumentFragment();
    for (const asset of assets) {
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
        renderSelected();
      });
      item.append(name, remove);
      fragment.append(item);
    }
    el.selectedList.append(fragment);
    updateSetup();
  }

  function render() {
    renderAssets();
    renderSelected();
  }

  function mergeImported(data) {
    const incoming = Array.isArray(data.assets) ? data.assets : [];
    const byId = new Map(state.assets.map((asset) => [asset.id, asset]));
    for (const asset of incoming) {
      if (!asset || !asset.id) continue;
      byId.set(asset.id, asset);
      state.selected.add(asset.id);
    }
    state.assets = Array.from(byId.values());
    if (mode === "independent") {
      state.assets = state.assets.filter((asset) => !asset.project_id);
    }
    showErrors(data.errors || []);
    render();
  }

  async function importPaths(paths) {
    if (!paths.length || state.busy) return;
    setBusy(true, "正在导入视频…");
    try {
      const data = await request("/api/assets/import", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          paths,
          project_id: state.projectId || null,
        }),
      });
      mergeImported(data);
    } catch (error) {
      showErrors([{ name: "导入失败", message: error.message }]);
    } finally {
      setBusy(false);
    }
  }

  async function parseUpload(response) {
    let payload = null;
    try {
      payload = await response.json();
    } catch (error) {
      payload = null;
    }
    if (!response.ok || !payload || payload.ok !== true) {
      const failure = payload && payload.error ? payload.error : {};
      const error = new Error(failure.message || `上传失败（${response.status}）`);
      error.fields = Array.isArray(failure.fields) ? failure.fields : [];
      throw error;
    }
    return payload.data;
  }

  async function uploadFiles(files) {
    if (!files.length || state.busy) return;
    setBusy(true, "正在导入视频…");
    try {
      const form = new FormData();
      for (const file of files) {
        form.append("files", file, file.name);
      }
      if (state.projectId) {
        form.append("project_id", state.projectId);
      }
      const response = await fetch("/api/assets/upload", {
        method: "POST",
        body: form,
      });
      mergeImported(await parseUpload(response));
    } catch (error) {
      showErrors([{ name: "导入失败", message: error.message }]);
    } finally {
      setBusy(false);
    }
  }

  async function chooseVideos() {
    if (state.busy || !hasProject()) return;
    if (window.pywebview) {
      const api = window.pywebview.api;
      if (!api || typeof api.choose_videos !== "function") {
        showErrors([{ name: "选择器", message: "窗口尚未准备好" }]);
        return;
      }
      try {
        const paths = await api.choose_videos();
        if (Array.isArray(paths) && paths.length) {
          await importPaths(paths);
        }
      } catch (error) {
        showErrors([{ name: "选择器", message: error.message || String(error) }]);
      }
      return;
    }
    const files = await window.BLPickers.pickFiles({
      multiple: true,
      accept: videoAccept,
    });
    if (files.length) await uploadFiles(files);
  }

  async function chooseFolder() {
    if (state.busy || !hasProject()) return;
    if (window.pywebview) {
      const api = window.pywebview.api;
      if (!api || typeof api.choose_folder !== "function") {
        showErrors([{ name: "选择器", message: "窗口尚未准备好" }]);
        return;
      }
      try {
        const path = await api.choose_folder();
        if (path) await importPaths([path]);
      } catch (error) {
        showErrors([{ name: "选择器", message: error.message || String(error) }]);
      }
      return;
    }
    const files = await window.BLPickers.pickFiles({
      multiple: true,
      directory: true,
    });
    const videos = files.filter(isVideoFile);
    if (videos.length) await uploadFiles(videos);
  }

  if (el.project) {
    el.project.addEventListener("change", () => {
      const url = new URL(window.location.href);
      if (el.project.value) {
        url.searchParams.set("project", el.project.value);
      } else {
        url.searchParams.delete("project");
      }
      window.location.href = url.toString();
    });
  }

  el.pickVideos.addEventListener("click", chooseVideos);
  el.pickFolder.addEventListener("click", chooseFolder);

  el.dropzone.addEventListener("dragover", (event) => {
    event.preventDefault();
    if (!state.busy && hasProject()) {
      el.dropzone.classList.add("is-dragover");
    }
  });

  el.dropzone.addEventListener("dragleave", () => {
    el.dropzone.classList.remove("is-dragover");
  });

  el.dropzone.addEventListener("drop", async (event) => {
    event.preventDefault();
    el.dropzone.classList.remove("is-dragover");
    if (state.busy || !hasProject()) return;
    const files = Array.from(event.dataTransfer.files || []).filter(isVideoFile);
    if (files.length) await uploadFiles(files);
  });

  el.setup.addEventListener("click", () => {
    if (el.setup.disabled) return;
    el.setupNote.hidden = false;
    el.setupNote.textContent = "视频参数设置将在下一块接入。";
  });

  async function loadAssets() {
    if (!hasProject()) {
      render();
      setBusy(false);
      return;
    }
    try {
      const path = state.projectId
        ? `/api/assets?project_id=${encodeURIComponent(state.projectId)}`
        : "/api/assets";
      const data = await request(path);
      const assets = Array.isArray(data.assets) ? data.assets : [];
      state.assets = mode === "independent"
        ? assets.filter((asset) => !asset.project_id)
        : assets;
      render();
    } catch (error) {
      showErrors([{ name: "资产列表", message: error.message }]);
    }
    setBusy(false);
  }

  setBusy(false);
  loadAssets();
})();
