(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.BLPickers = factory();
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  function pickFiles(options) {
    const opts = options || {};
    const doc = opts.document || (typeof document !== "undefined" ? document : null);
    if (!doc) {
      return Promise.reject(new Error("没有可用的文档环境"));
    }

    return new Promise(function (resolve) {
      const input = doc.createElement("input");
      input.type = "file";
      if (opts.multiple) {
        input.multiple = true;
      }
      if (opts.directory) {
        input.webkitdirectory = true;
        input.directory = true;
      }
      if (opts.accept) {
        input.accept = opts.accept;
      }
      input.style.display = "none";

      let settled = false;

      function cleanup(files) {
        if (settled) {
          return;
        }
        settled = true;
        input.removeEventListener("change", onChange);
        input.removeEventListener("cancel", onCancel);
        if (input.parentNode) {
          input.parentNode.removeChild(input);
        } else if (typeof input.remove === "function") {
          input.remove();
        }
        resolve(files);
      }

      function onChange() {
        cleanup(Array.from(input.files || []));
      }

      function onCancel() {
        cleanup([]);
      }

      input.addEventListener("change", onChange);
      input.addEventListener("cancel", onCancel);
      doc.body.appendChild(input);
      input.click();
    });
  }

  return { pickFiles: pickFiles };
});
