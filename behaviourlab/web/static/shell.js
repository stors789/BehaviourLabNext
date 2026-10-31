/* 顶栏项目切换浮层：Escape / 外部点击关闭，焦点回到摘要。 */
(function () {
  "use strict";

  const root = document.querySelector("[data-switcher]");
  if (!root) return;
  const summary = root.querySelector(".switch-summary");
  const panel = root.querySelector(".switch-panel");
  if (!summary || !panel) return;

  function open() {
    panel.hidden = false;
    summary.setAttribute("aria-expanded", "true");
    const first = panel.querySelector("a");
    if (first) first.focus();
  }

  function close(restoreFocus) {
    if (panel.hidden) return;
    panel.hidden = true;
    summary.setAttribute("aria-expanded", "false");
    if (restoreFocus) summary.focus();
  }

  summary.addEventListener("click", () => {
    if (panel.hidden) open();
    else close(true);
  });

  document.addEventListener("pointerdown", (event) => {
    if (!panel.hidden && !root.contains(event.target)) close(false);
  });

  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    if (!panel.hidden) {
      event.preventDefault();
      close(true);
    }
  });

  document.addEventListener("click", (event) => {
    const skip = event.target.closest(".skip");
    if (!skip) return;
    const main = document.querySelector("#main");
    if (main) main.focus();
  });
})();
