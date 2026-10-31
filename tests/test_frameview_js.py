"""前端取帧器的世代号与 objectURL 生命周期检查（Node）。"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def test_frame_loader_discards_stale_and_revokes_url():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node 不可用")
    script = Path(__file__).parent / "js" / "test_frameview.cjs"
    result = subprocess.run(
        [node, str(script)], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert "frameview-ok" in result.stdout
