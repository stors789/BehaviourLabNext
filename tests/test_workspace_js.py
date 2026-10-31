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
    assert ('runPreview(state.step === "review" ? "behavior" : "detection")'
            in source)
