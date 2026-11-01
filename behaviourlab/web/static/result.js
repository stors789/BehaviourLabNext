/* 结果页：动物切换、图形加载失败空态与打开目录。
   只改显示，不改冻结数据；失败不显示虚假成功。 */
(function () {
  "use strict";

  function seed() {
    var node = document.getElementById("result-seed");
    if (!node) return {};
    try {
      return JSON.parse(node.textContent || "{}") || {};
    } catch (error) {
      return {};
    }
  }

  var data = seed();

  // 动物名称选择器：只切换当前动物的显示，切换不留上一动物的图或说明。
  var selector = document.getElementById("unit-switch");
  var blocks = Array.prototype.slice.call(
    document.querySelectorAll(".unit-block[data-unit-id]"));
  function showUnit(id) {
    blocks.forEach(function (block) {
      block.hidden = block.getAttribute("data-unit-id") !== String(id);
    });
  }
  if (selector && blocks.length) {
    selector.addEventListener("change", function () {
      showUnit(selector.value);
    });
    showUnit(selector.value);
  }

  // 图形加载失败：替换为明确空态，不留破图占位。
  Array.prototype.forEach.call(
    document.querySelectorAll("[data-figures] img"),
    function (img) {
      img.addEventListener("error", function () {
        var figure = img.closest("figure");
        if (!figure) {
          img.remove();
          return;
        }
        while (figure.firstChild) {
          figure.removeChild(figure.firstChild);
        }
        var note = document.createElement("p");
        note.className = "muted";
        note.setAttribute("data-empty", "figure-error");
        note.textContent = "图形加载失败。";
        figure.appendChild(note);
      });
    });

  // 打开目录：只请求打开本 run 的发布目录；失败时给出明确提示。
  var openButton = document.getElementById("open-run-dir");
  var note = document.getElementById("open-dir-note");
  if (openButton) {
    openButton.addEventListener("click", function () {
      openButton.disabled = true;
      fetch("/api/runs/" + encodeURIComponent(data.run_id || "") + "/open",
            { method: "POST" })
        .then(function (response) { return response.json(); })
        .then(function (payload) {
          var inner = (payload && payload.data) || {};
          if (note) {
            note.hidden = false;
            note.textContent = (payload && payload.ok && inner.path)
              ? "发布目录：" + inner.path
              : ((payload && payload.error && payload.error.message)
                 || "无法打开结果目录。");
          }
        })
        .catch(function () {
          if (note) {
            note.hidden = false;
            note.textContent = "无法打开结果目录。";
          }
        })
        .finally(function () { openButton.disabled = false; });
    });
  }
})();
