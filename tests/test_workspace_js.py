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
