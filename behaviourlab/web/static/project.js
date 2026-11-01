/* 项目页：三个分区切换、视频导入与直接进入分析。 */
(() => {
  "use strict";

  const api = window.BLApi;
  if (!api) return;
  const { request, showError } = api;

  // -- 分区切换：只改显示，不创建草稿 -------------------------------
  const tabs = Array.from(document.querySelectorAll("[data-tab]"));
  const panels = Array.from(document.querySelectorAll("[data-panel]"));

  function showTab(name) {
    const target = ["org", "media", "analysis"].includes(name) ? name : "org";
    tabs.forEach((tab) => {
      const current = tab.dataset.tab === target;
      tab.classList.toggle("is-current", current);
      tab.setAttribute("aria-selected", current ? "true" : "false");
    });
    panels.forEach((panel) => {
      panel.hidden = panel.dataset.panel !== target;
    });
    if (location.hash !== "#" + target) {
      history.replaceState(null, "", "#" + target);
    }
  }

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => showTab(tab.dataset.tab));
  });
  showTab((location.hash || "").replace("#", ""));
  window.addEventListener("hashchange", () => {
    showTab((location.hash || "").replace("#", ""));
  });

  // -- 视频导入 -------------------------------------------------------
  const dropzone = document.querySelector("#project-dropzone");
  const status = document.querySelector("#import-status");
  const errorList = document.querySelector("#import-errors");
  const projectId = dropzone ? dropzone.dataset.project : "";

  function setStatus(text) {
    if (!status) return;
    status.hidden = !text;
    status.textContent = text || "";
  }

  function showErrors(errors) {
    if (!errorList) return;
    errorList.innerHTML = "";
    const items = (errors || []).filter(Boolean);
    if (!items.length) {
      errorList.hidden = true;
      return;
    }
    items.forEach((item) => {
      const row = document.createElement("li");
      row.className = "error-row";
      const name = document.createElement("span");
      name.className = "error-name";
      name.textContent = item.name || "未命名文件";
      const message = document.createElement("span");
      message.className = "error-message";
      message.textContent = item.message || "";
      row.append(name, message);
      errorList.append(row);
    });
    errorList.hidden = false;
  }

  async function uploadFiles(files) {
    const list = Array.from(files || []);
    if (!list.length) return;
    setStatus("正在导入 " + list.length + " 个文件…");
    const form = new FormData();
    list.forEach((file) => form.append("files", file));
    if (projectId) form.append("project_id", projectId);
    try {
      const data = await request("/api/assets/upload", {
        method: "POST",
        body: form,
      });
      showErrors(data.errors);
      setStatus("已导入 " + (data.assets || []).length + " 个视频");
      if ((data.assets || []).length) window.location.reload();
    } catch (error) {
      showErrors([{ name: "导入失败", message: error.message }]);
      setStatus("");
    }
  }

  async function importPaths(paths) {
    const list = (paths || []).filter((item) => typeof item === "string" && item);
    if (!list.length) return;
    setStatus("正在导入…");
    try {
      const data = await request("/api/assets/import", {
        method: "POST",
        body: JSON.stringify({ paths: list, project_id: projectId || null }),
      });
      showErrors(data.errors);
      setStatus("已导入 " + (data.assets || []).length + " 个视频");
      if ((data.assets || []).length) window.location.reload();
    } catch (error) {
      showErrors([{ name: "导入失败", message: error.message }]);
      setStatus("");
    }
  }

  function desktopBridge() {
    return window.pywebview && window.pywebview.api ? window.pywebview.api : null;
  }

  async function pick(kind) {
    const core = window.BLCore;
    if (core && typeof core.pickMedia === "function") {
      await core.pickMedia(kind, {
        bridge: desktopBridge(),
        picker: window.BLPickers,
        onPaths: importPaths,
        onFiles: uploadFiles,
      });
      return;
    }
    if (!window.BLPickers) return;
    const files = await window.BLPickers.pickFiles({
      multiple: true,
      directory: kind === "folder",
    });
    if (files && files.length) await uploadFiles(files);
  }

  const pickFilesButton = document.querySelector("#project-pick-files");
  const pickFolderButton = document.querySelector("#project-pick-folder");
  if (pickFilesButton) {
    pickFilesButton.addEventListener("click", () => { pick("videos"); });
  }
  if (pickFolderButton) {
    pickFolderButton.addEventListener("click", () => { pick("folder"); });
  }

  if (dropzone) {
    ["dragenter", "dragover"].forEach((name) => {
      dropzone.addEventListener(name, (event) => {
        event.preventDefault();
        dropzone.classList.add("is-dragover");
      });
    });
    ["dragleave", "drop"].forEach((name) => {
      dropzone.addEventListener(name, (event) => {
        event.preventDefault();
        if (name === "dragleave" && dropzone.contains(event.relatedTarget)) return;
        dropzone.classList.remove("is-dragover");
      });
    });
    dropzone.addEventListener("drop", async (event) => {
      const files = Array.from(event.dataTransfer.files || []);
      if (files.length) await uploadFiles(files);
    });
  }

  // -- 选中视频进入分析 -----------------------------------------------
  const assaySelect = document.querySelector("#media-assay");

  async function analyzeAsset(assetId, button) {
    const assay = assaySelect ? assaySelect.value : "";
    if (!projectId || !assetId) return;
    if (button) button.disabled = true;
    setStatus("正在创建分析…");
    try {
      const data = await request("/api/drafts", {
        method: "POST",
        body: JSON.stringify({
          assay: assay || undefined,
          project_id: projectId,
          videos: [{ asset_id: assetId, units: [{}] }],
        }),
      });
      const draft = data.draft;
      const params = new URLSearchParams({
        mode: "project",
        project: projectId,
        draft_id: draft.id,
        step: "range",
      });
      window.location.href = "/analyze?" + params.toString();
    } catch (error) {
      showErrors([{ name: "无法创建分析", message: error.message }]);
      setStatus("");
      if (button) button.disabled = false;
    }
  }

  document.querySelectorAll("[data-analyze-asset]").forEach((button) => {
    button.addEventListener("click", () => {
      analyzeAsset(button.dataset.analyzeAsset, button);
    });
  });
})();
