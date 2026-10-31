const assert = require("assert");
const { pickFiles } = require("../../behaviourlab/web/static/pickers.js");

class FakeInput {
  constructor() {
    this.listeners = { change: [], cancel: [] };
    this.parentNode = null;
    this.files = [];
    this.clicked = false;
    this.style = {};
  }

  addEventListener(type, handler) {
    this.listeners[type].push(handler);
  }

  removeEventListener(type, handler) {
    this.listeners[type] = this.listeners[type].filter((item) => item !== handler);
  }

  click() {
    this.clicked = true;
  }

  dispatch(type) {
    for (const handler of [...this.listeners[type]]) {
      handler();
    }
  }
}

function fakeDocument() {
  const body = {
    children: [],
    appendChild(node) {
      node.parentNode = body;
      body.children.push(node);
    },
    removeChild(node) {
      body.children = body.children.filter((item) => item !== node);
      node.parentNode = null;
    },
  };
  const doc = {
    body,
    created: [],
    createElement() {
      const input = new FakeInput();
      doc.created.push(input);
      return input;
    },
  };
  return doc;
}

async function main() {
  const doc = fakeDocument();
  const promise = pickFiles({ multiple: true, accept: ".mp4", document: doc });
  assert.strictEqual(doc.created.length, 1);
  const input = doc.created[0];
  assert.ok(input.parentNode === doc.body, "input 必须先挂到 body");
  assert.ok(input.clicked, "挂载后才 click");
  assert.ok(doc.body.children.includes(input), "change 前节点保持连接");
  assert.strictEqual(input.multiple, true);
  assert.strictEqual(input.accept, ".mp4");

  input.files = ["file-a", "file-b"];
  input.dispatch("change");
  assert.deepStrictEqual(await promise, ["file-a", "file-b"]);
  assert.ok(!doc.body.children.includes(input), "change 后才移除节点");
  assert.strictEqual(input.listeners.change.length, 0);

  const cancelPromise = pickFiles({ multiple: true, document: doc });
  const cancelInput = doc.created[1];
  assert.ok(doc.body.children.includes(cancelInput), "cancel 前节点保持连接");
  cancelInput.dispatch("cancel");
  assert.deepStrictEqual(await cancelPromise, []);
  assert.ok(!doc.body.children.includes(cancelInput), "cancel 后才移除节点");

  const againPromise = pickFiles({ multiple: false, document: doc });
  const againInput = doc.created[2];
  assert.ok(doc.body.children.includes(againInput), "取消后可再次打开");
  againInput.files = ["file-c"];
  againInput.dispatch("change");
  assert.deepStrictEqual(await againPromise, ["file-c"]);

  const folderPromise = pickFiles({ multiple: true, directory: true, document: doc });
  const folderInput = doc.created[3];
  assert.strictEqual(folderInput.webkitdirectory, true);
  folderInput.dispatch("cancel");
  assert.deepStrictEqual(await folderPromise, []);

  console.log("pickers-ok");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
