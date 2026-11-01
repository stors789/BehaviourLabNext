(() => {
  "use strict";

  const health = document.querySelector("#health");

  function setHealth(state, label) {
    if (!health) return;
    health.dataset.state = state;
    const text = health.querySelector(".health-text");
    if (text && label) text.textContent = label;
  }

  async function request(path, options = {}) {
    const init = {
      ...options,
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    };
    const response = await fetch(path, init);
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

  function describe(error) {
    if (error.fields && error.fields.length) {
      const detail = error.fields
        .map((item) => (item.field ? `${item.field}：${item.message}` : item.message))
        .join("；");
      return `${error.message}（${detail}）`;
    }
    return error.message || "请求失败";
  }

  function showError(box, error) {
    if (box) box.textContent = describe(error);
  }

  // 其他页面脚本复用同一套请求与错误文案。
  window.BLApi = { request, describe, showError };

  request("/api/health")
    .then(() => setHealth("ok", "本地服务正常"))
    .catch(() => setHealth("off", "服务不可用"));

  // 提交中锁定按钮，失败保留已填内容并给出原因。
  function lockWhilePending(form, action) {
    const button = form.querySelector('button[type="submit"]');
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (form.dataset.submitting === "1") return;
      form.dataset.submitting = "1";
      if (button) button.disabled = true;
      try {
        await action();
      } finally {
        form.dataset.submitting = "0";
        if (button) button.disabled = false;
      }
    });
  }

  const toggle = document.querySelector("#new-project-toggle");
  const projectForm = document.querySelector("#project-form");
  if (toggle && projectForm) {
    toggle.addEventListener("click", () => {
      projectForm.classList.toggle("hidden");
      if (!projectForm.classList.contains("hidden")) {
        const input = projectForm.querySelector("input");
        if (input) input.focus();
      }
    });

    lockWhilePending(projectForm, async () => {
      const errorBox = projectForm.querySelector(".form-error");
      if (errorBox) errorBox.textContent = "";
      const data = new FormData(projectForm);
      try {
        const result = await request("/api/projects", {
          method: "POST",
          body: JSON.stringify({
            name: data.get("name"),
            description: data.get("description") || "",
          }),
        });
        window.location.href = `/projects/${result.project.id}`;
      } catch (error) {
        showError(errorBox, error);
      }
    });
  }

  document.querySelectorAll('[data-form="group"]').forEach((form) => {
    lockWhilePending(form, async () => {
      const errorBox = form.querySelector(".form-error");
      if (errorBox) errorBox.textContent = "";
      const data = new FormData(form);
      try {
        await request(`/api/projects/${form.dataset.project}/groups`, {
          method: "POST",
          body: JSON.stringify({ name: data.get("name") }),
        });
        window.location.reload();
      } catch (error) {
        showError(errorBox, error);
      }
    });
  });

  document.querySelectorAll('[data-form="animal"]').forEach((form) => {
    lockWhilePending(form, async () => {
      const errorBox = form.querySelector(".form-error");
      if (errorBox) errorBox.textContent = "";
      const data = new FormData(form);
      const payload = { name: data.get("name") };
      const code = (data.get("code") || "").trim();
      if (code) payload.code = code;
      const groupId = data.get("group_id");
      if (groupId) payload.group_id = groupId;
      try {
        await request(`/api/projects/${form.dataset.project}/animals`, {
          method: "POST",
          body: JSON.stringify(payload),
        });
        window.location.reload();
      } catch (error) {
        showError(errorBox, error);
      }
    });
  });
})();
