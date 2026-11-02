"""选择器触发守卫与背景确认的前端检查（Node DOM 替身）。"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def test_pick_triggers_and_background_contract():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node 不可用")
    script = Path(__file__).parent / "js" / "test_pick_triggers.cjs"
    result = subprocess.run(
        [node, str(script)], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert "all assertions passed" in result.stdout
