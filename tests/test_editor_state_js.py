"""编辑器状态机、工具头高度与预览确认闸门（Node 复用）。"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def test_editor_state_and_preview_confirm():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node 不可用")
    script = Path(__file__).parent / "js" / "test_editor_state.cjs"
    result = subprocess.run(
        [node, str(script)], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert "editor-state OK" in result.stdout
