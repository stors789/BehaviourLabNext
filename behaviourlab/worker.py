"""工作进程边界：执行已冻结任务。

本块只声明接口；真实执行在分析块接入，避免返回假结果。
"""

from __future__ import annotations

from typing import Any, Mapping


def run_frozen_job(resolved: Mapping[str, Any]) -> Any:
    raise NotImplementedError("工作进程将在后续块接入")
