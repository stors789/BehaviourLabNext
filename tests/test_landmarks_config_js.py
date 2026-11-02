"""关键点会话配置继承的前端检查（Node DOM 替身 + 真实 landmarks.js）。"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def test_config_inheritance_payloads_and_late_response():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node 不可用")
    script = Path(__file__).parent / "js" / "test_landmarks_config.cjs"
    result = subprocess.run(
        [node, str(script)], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert "landmarks config inheritance" in result.stdout
