"""改容差清旧评估的前端检查（Node DOM 替身）。"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def test_tolerance_input_clears_prediction_layers():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node 不可用")
    script = Path(__file__).parent / "js" / "test_landmarks_tolerance_reset.cjs"
    result = subprocess.run(
        [node, str(script)], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert "tolerance reset" in result.stdout
