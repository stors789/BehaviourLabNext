"""工作台预览接线（Node）与检测方法枚举的前后端一致性。"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from behaviourlab import detection

ROOT = Path(__file__).resolve().parents[1]


def test_workspace_preview_contract():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node 不可用")
    script = Path(__file__).parent / "js" / "test_workspace_preview.cjs"
    result = subprocess.run(
        [node, str(script)], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert "workspace-preview OK" in result.stdout


def test_ui_lists_every_background_method():
    source = (ROOT / "behaviourlab/web/static/workspace.js").read_text(
        encoding="utf-8")
    block = source.split("const BG_METHOD_LABELS = [", 1)[1].split("];", 1)[0]
    methods = re.findall(r'\["([a-z0-9_]+)"', block)
    assert tuple(methods) == detection.BG_METHODS


def test_analyze_page_exposes_preview_note():
    html = (ROOT / "behaviourlab/web/templates/analyze.html").read_text(
        encoding="utf-8")
    assert 'id="preview-note"' in html


def test_secondary_preview_passes_stage_kind():
    source = (ROOT / "behaviourlab/web/static/workspace.js").read_text(
        encoding="utf-8")
    # 事件对象不能直接当 kind 传给后端。
    assert 'el.secondary.addEventListener("click", runPreview)' not in source
    # 次级按钮沿用当前预览类型：行为预览下不能退回单帧检测。
    assert ('runPreview(state.step === "review" || '
            'state.previewKind === "behavior" ? "behavior" : "detection")'
            in source)
    assert 'runPreview(state.step === "review" ? "behavior" : "detection")' \
        not in source


def test_parameter_sections_render_as_folds():
    source = (ROOT / "behaviourlab/web/static/workspace.js").read_text(
        encoding="utf-8")
    # 描述符里的 section 必须被消费：分组后折叠，不再平铺成一个大表单。
    assert "param.section" in source
    assert "summary.textContent = group.title;" in source
    from behaviourlab import experiments

    entry = [item for item in experiments.catalog()
             if item["key"] == "oft"][0]
    assert entry["parameters"]
    assert all(param.get("section") == "统计"
               for param in entry["parameters"])


def test_save_failures_show_field_reasons():
    source = (ROOT / "behaviourlab/web/static/workspace.js").read_text(
        encoding="utf-8")
    # 保存失败与因此被阻断的路径都走字段级文案，不再被通用提示覆盖。
    for fallback in ("保存失败，仍停留在当前页", "保存失败，未进入场地",
                     "保存失败，未生成预览", "保存失败，未开始分析"):
        assert f'lastSaveText("{fallback}")' in source, fallback
    assert 'status("保存失败' not in source
    assert "issueText(state.saver.lastError" in source
    assert "window.BLCore.issueText(error, fallback)" in source

    core = (ROOT / "behaviourlab/web/static/editor-core.js").read_text(
        encoding="utf-8")
    assert "function issueText(error, fallback)" in core
    assert "    issueText," in core


def test_error_text_registers_readable_field_names():
    source = (ROOT / "behaviourlab/web/static/workspace.js").read_text(
        encoding="utf-8")
    # 描述符 label 登记到错误格式化层，折叠参数区外也能看见字段名。
    assert "registerFieldLabels()" in source
    assert "window.BLCore.setFieldLabels(labels)" in source
    assert "param.label" in source

    core = (ROOT / "behaviourlab/web/static/editor-core.js").read_text(
        encoding="utf-8")
    assert "function setFieldLabels(labels)" in core
    assert "setFieldLabels," in core
    assert "function issueLine(issue)" in core


def test_tst_preview_modes_and_output_toggle():
    source = (ROOT / "behaviourlab/web/static/workspace.js").read_text(
        encoding="utf-8")
    # 原始画面 / 检测预览 / 行为预览三个入口可切换。
    for label in ("原始画面", "检测预览", "行为预览"):
        assert '"' + label + '"' in source, label
    assert "state.previewKind" in source
    assert "appendPreviewModes" in source
    assert "selectPreviewKind" in source
    # 输出开关写入既有 output 配置，且不废弃预览。
    assert "生成标注视频" in source
    assert "state.draft.output.annotated_video" in source
    toggle = source.split("function appendAnnotationToggle", 1)[1].split(
        "\n  }", 1)[0]
    assert "markDirty()" in toggle
    assert "invalidatePreview" not in toggle
    # 图片加载成功后才确认预览。
    assert "el.preview.onload" in source
    assert "state.previewAwaiting" in source


def test_player_commits_only_after_viewport_checks():
    source = (ROOT / "behaviourlab/web/static/workspace.js").read_text(
        encoding="utf-8")
    # 提交画面前校验请求代号与页面上下文，暂停/离页都会废弃在途帧。
    assert "frameStillCurrent(context)" in source
    assert "window.BLFrameView.frameStillCurrent(" in source
    commit = source.split("function commitFrame(payload, context)", 1)[1].split(
        "\n  }", 1)[0]
    assert commit.index("frameStillCurrent(context)") < commit.index(
        "state.image = payload.image")
    # 显示时刻由相邻真实时间差决定，且先等到点再提交。
    assert "frameDisplayDelay(" in source
    play = source.split("async function playStep(token)", 1)[1].split(
        "\n  }", 1)[0]
    assert play.index("frameDisplayDelay(") < play.index("commitFrame(")
    assert "state.playWait" in source
    # 历史记录实际显示过的帧，而不是请求中的目标帧。
    analysis = source.split("function analysisState()", 1)[1].split(
        "\n  }", 1)[0]
    assert "state.shownFrame" in analysis
    assert "frame: state.frame," not in analysis
    # 位置立即写当前历史项：没有延迟回调，后退目标不会被旧页写回。
    assert "schedulePosition" not in source
    assert "setTimeout(rememberPosition" not in source
    assert source.count(
        "history.replaceState({ analysis: analysisState() }") == 1
    commit = source.split("function commitFrame(payload, context)", 1)[1].split(
        "\n  }", 1)[0]
    assert "if (!state.playing) rememberPosition();" in commit


def test_beforeunload_keeps_page_alive_after_cancel():
    source = (ROOT / "behaviourlab/web/static/workspace.js").read_text(
        encoding="utf-8")
    handler = source.split('window.addEventListener("beforeunload"', 1)[1].split(
        "});", 1)[0]
    assert "teardownPage()" not in handler, "取消离页后快捷键必须仍可用"
    assert "rememberPosition()" in handler
    teardown = source.split("function teardownPage()", 1)[1].split("\n  }", 1)[0]
    assert "stopPlayback()" in teardown
    assert 'removeEventListener("keydown", manualHotkeys)' in teardown


def test_analysis_submission_freezes_context_before_first_await():
    source = (ROOT / "behaviourlab/web/static/workspace.js").read_text(
        encoding="utf-8")
    body = source.split("async function startAnalysis()", 1)[1].split(
        "\n  // -- 事件绑定", 1)[0]
    assert body.index("const context = {") < body.index("await saveThen()")
    assert "navToken: state.navToken," in body
    assert "edit: state.editCounter," in body
    assert body.count("abort()") >= 3
    assert body.index("if (abort()) return;") < body.index("POST\", \"/api/jobs")
    # 离开页面时只提示任务已开始，不强制跳转。
    assert "任务已开始，可在任务页查看" in body
    assert body.index("submission() !== \"ok\"") < body.index(
        'window.location.href = "/results/"')
    # 旧预览响应不能在新页面确认。
    preview = source.split("async function runPreview(kind)", 1)[1].split(
        "\n  async function primaryAction", 1)[0]
    assert "const context = { page: pageKey(), edit: state.editCounter };" in preview
    assert preview.count("pageKey() !== context.page") >= 3
