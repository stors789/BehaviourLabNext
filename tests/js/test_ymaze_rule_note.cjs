/* Y 迷宫入臂规则说明：真实 editor-core 函数与假 DOM 元素构建。
   覆盖两种规则的说明内容、切换行为与“不生效”标注判定。 */
"use strict";

const assert = require("assert");
const {
  ymazeEntryRuleNote,
  ymazeParamNotApplicable,
} = require("../../behaviourlab/web/static/editor-core.js");

// -- 一、来源与边界声明（两种规则都要有，且不宣称认证） -----------------

for (const rule of ["depth", "center"]) {
  const note = ymazeEntryRuleNote(rule);
  assert.ok(note.includes("IMPC"), "说明标注 IMPC 来源");
  assert.ok(note.includes("中心点进入新区域"), "说明引用 IMPC 判定表述");
  assert.ok(note.includes("本软件约定"), "区分软件约定");
  assert.ok(note.includes("不宣称完整遵循或获得 IMPC 认证"), "不宣称认证");
}

// -- 二、中心点规则说明：行为 + 无防抖 + 失效参数 -----------------------

const center = ymazeEntryRuleNote("center");
assert.ok(center.includes("首个有效观测即记一次"), "首 Observation 即计");
assert.ok(center.includes("无防抖"), "无防抖边界写明");
assert.ok(center.includes("入臂深度比例") && center.includes("均不生效"),
  "四个不生效参数写明");
assert.ok(!center.includes("持续确认（原规则）"), "不混入深度规则行为");

// -- 三、深度规则说明：原规则行为，不含“不生效”清单 ---------------------

const depth = ymazeEntryRuleNote("depth");
assert.ok(depth.includes("进入指定深度并持续确认"), "原规则行为");
assert.ok(!depth.includes("均不生效"), "深度规则没有失效参数清单");

// -- 四、规则切换：说明随当前值变化 -------------------------------------

assert.notStrictEqual(ymazeEntryRuleNote("center"), ymazeEntryRuleNote("depth"));

// -- 五、“不生效”标注判定 -----------------------------------------------

for (const key of ["entry_depth_ratio", "entry_confirm_s",
                   "entry_reset_center_s", "require_center"]) {
  assert.strictEqual(ymazeParamNotApplicable("center", key), true, key);
  assert.strictEqual(ymazeParamNotApplicable("depth", key), false, key);
}
// 不影响本规则的参数不标注。
assert.strictEqual(ymazeParamNotApplicable("center", "bin_seconds"), false);
assert.strictEqual(ymazeParamNotApplicable("center", "arm_length_cm"), false);
// 缺字段/未知值按深度规则解释。
assert.strictEqual(ymazeParamNotApplicable(null, "entry_depth_ratio"), false);
assert.strictEqual(ymazeParamNotApplicable(undefined, "entry_confirm_s"), false);

// -- 六、假 DOM：说明能构建为元素（label 内的 meta span） ----------------

function fakeElement(tag) {
  return {
    tag,
    className: "",
    textContent: "",
    children: [],
    append(child) { this.children.push(child); },
  };
}
const label = fakeElement("label");
const hint = fakeElement("span");
hint.className = "meta";
hint.textContent = ymazeEntryRuleNote("center");
label.append(hint);
assert.strictEqual(label.children.length, 1);
assert.ok(label.children[0].textContent.includes("无防抖"));

console.log("ymaze-rule-note OK");
