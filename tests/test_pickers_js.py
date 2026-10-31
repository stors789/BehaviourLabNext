"""浏览器隐藏文件输入的生命周期检查（Node DOM 替身）。"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def test_hidden_input_waits_for_change_or_cancel():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node 不可用")
    script = Path(__file__).parent / "js" / "test_pickers.cjs"
    result = subprocess.run(
        [node, str(script)], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert "pickers-ok" in result.stdout
