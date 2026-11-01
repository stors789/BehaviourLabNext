/* 设置页：四个分区各自保存，真实表单不要求编辑 JSON。 */
(() => {
  "use strict";

  const api = window.BLApi;
  if (!api) return;
  const { request, showError } = api;

  const forms = Array.from(document.querySelectorAll("[data-settings-section]"));
  if (!forms.length) return;

  const previewButton = document.querySelector("#style-preview");
  const previewBox = document.querySelector("#style-preview-box");
  const previewImage = document.querySelector("#style-preview-image");
  const previewNote = document.querySelector("#style-preview-note");
  let previewStale = false;
  let previewDpi = null;
  // 每次输入让该表单的版本 +1；异步回调只能确认它提交时的那一版。
  const versions = new Map();
  function version(form) {
    return versions.get(form) || 0;
  }
  function bump(form) {
    versions.set(form, version(form) + 1);
  }

  function note(form) {
    return form.querySelector(".form-note");
  }

  function errorBox(form) {
    return form.querySelector(".form-error");
  }

  function setNote(form, text) {
    const box = note(form);
    if (box) box.textContent = text || "";
  }

  function setDirty(form, dirty) {
    form.dataset.dirty = dirty ? "1" : "0";
  }

  function isDirty(form) {
    return form.dataset.dirty === "1";
  }

  function readField(input) {
    if (input.type === "checkbox") return { value: input.checked };
    if (input.type === "number") {
      const raw = input.value.trim();
      if (raw === "") return { error: "必须填写数值" };
      const number = Number(raw);
      if (!Number.isFinite(number)) return { error: "必须是数值" };
      if (input.min !== "" && number < Number(input.min)) {
        return { error: "不能小于 " + input.min };
      }
      if (input.max !== "" && number > Number(input.max)) {
        return { error: "不能大于 " + input.max };
      }
      if (input.getAttribute("step") === "1" && !Number.isInteger(number)) {
        return { error: "必须是整数" };
      }
      return { value: number };
    }
    if (input.type === "text") {
      const raw = input.value.trim();
      return { value: raw === "" ? null : raw };
    }
    return { value: input.value };
  }

  function collect(form) {
    const scope = form.dataset.settingsSection;
    const nested = scope === "defaults" || scope === "detection";
    const value = nested ? {} : {};
    const inputs = Array.from(form.querySelectorAll("[data-param]"));
    for (const input of inputs) {
      const param = input.dataset.param;
      const result = readField(input);
      if (result.error) {
        return {
          error: (input.closest(".field")?.querySelector("span")?.textContent
            || param) + "：" + result.error,
          input,
        };
      }
      if (nested) {
        const assay = input.dataset.assay;
        if (!assay) {
          // 字段缺少实验归属时不能静默塞进 undefined 分区。
          return { error: param + "：字段没有实验归属", input };
        }
        if (!value[assay]) value[assay] = {};
        value[assay][param] = result.value;
      } else {
        value[param] = result.value;
      }
    }
    return { value };
  }

  async function save(form) {
    const button = form.querySelector('button[type="submit"]');
    const box = errorBox(form);
    if (box) box.textContent = "";
    const collected = collect(form);
    if (collected.error) {
      if (box) box.textContent = collected.error;
      if (collected.input) collected.input.focus();
      return false;
    }
    // 冻结本次提交的快照版本：期间的编辑不属于这次请求。
    const stamp = version(form);
    if (button) button.disabled = true;
    setNote(form, "保存中…");
    try {
      await request(`/api/settings/${form.dataset.settingsSection}`, {
        method: "PUT",
        body: JSON.stringify({ value: collected.value }),
      });
      if (version(form) !== stamp) {
        setDirty(form, true);
        setNote(form, "已保存早先的修改；当前输入仍未保存");
        return false;
      }
      setDirty(form, false);
      setNote(form, "已保存");
      return true;
    } catch (error) {
      showError(box, error);
      setNote(form, "保存失败，修改仍在页面上");
      return false;
    } finally {
      if (button) button.disabled = false;
    }
  }

  function styleValues() {
    const form = forms.find((item) => item.dataset.settingsSection === "style");
    if (!form) return null;
    const collected = collect(form);
    return collected.error ? null : collected.value;
  }

  function markStale() {
    if (!previewBox || previewBox.hidden) return;
    previewStale = true;
    if (previewNote) previewNote.textContent = "样式已改变，请重新预览";
    previewBox.dataset.state = "stale";
  }

  async function preview() {
    if (!previewButton || !previewBox || !previewImage) return;
    const value = styleValues();
    const form = forms.find((item) => item.dataset.settingsSection === "style");
    const box = form ? errorBox(form) : null;
    if (box) box.textContent = "";
    if (!value) {
      if (box) box.textContent = "样式字段必须先填写合法值";
      return;
    }
    // 预览同样只对发起时的样式版本负责。
    const stamp = version(form);
    previewButton.disabled = true;
    previewBox.hidden = false;
    previewBox.dataset.state = "loading";
    if (previewNote) previewNote.textContent = "正在生成预览…";
    try {
      const data = await request("/api/settings/style/preview", {
        method: "POST",
        body: JSON.stringify({ value }),
      });
      previewDpi = data.style ? data.style.dpi : null;
      const stale = () => {
        previewStale = true;
        previewBox.dataset.state = "stale";
        if (previewNote) previewNote.textContent = "样式已改变，请重新预览";
      };
      previewImage.onload = () => {
        if (version(form) !== stamp) {
          stale();
          return;
        }
        previewStale = false;
        previewBox.dataset.state = "ready";
        if (previewNote) {
          previewNote.textContent = "预览 DPI 上限 240；正式图仍按保存的 DPI 输出";
        }
      };
      previewImage.onerror = () => {
        if (version(form) !== stamp) {
          stale();
          return;
        }
        previewBox.dataset.state = "error";
        if (previewNote) previewNote.textContent = "预览图加载失败";
      };
      previewImage.src = data.image_url + "?v=" + Date.now();
    } catch (error) {
      previewBox.dataset.state = "error";
      if (previewNote) previewNote.textContent = "无法生成预览";
      showError(box, error);
    } finally {
      previewButton.disabled = false;
    }
  }

  forms.forEach((form) => {
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      save(form);
    });
    form.addEventListener("input", () => {
      bump(form);
      setDirty(form, true);
      setNote(form, "未保存的修改");
      if (form.dataset.settingsSection === "style") markStale();
    });
    form.addEventListener("change", () => {
      bump(form);
      setDirty(form, true);
      setNote(form, "未保存的修改");
      if (form.dataset.settingsSection === "style") markStale();
    });
  });

  if (previewButton) previewButton.addEventListener("click", preview);

  window.addEventListener("beforeunload", (event) => {
    if (!forms.some(isDirty)) return;
    event.preventDefault();
    event.returnValue = "";
  });

  window.BLSettings = {
    dirty: () => forms.some(isDirty),
    saveAll: async () => {
      for (const form of forms) {
        if (!isDirty(form)) continue;
        const ok = await save(form);
        if (!ok) return false;
      }
      return true;
    },
    previewState: () => ({ stale: previewStale, dpi: previewDpi }),
  };
})();
