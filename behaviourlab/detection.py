"""检测候选合并与公共检测默认值。

具体检测算法在后续块接入；本模块只提供跨实验共享的默认值与纯函数，
不读取 HTTP、控件或数据库。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Mapping

from .schema import is_number

# 公共检测默认值由检测块填充；此处保持为空对象，避免伪造未定义参数。
COMMON_DETECTION_DEFAULTS: dict = {}


def common_detection_defaults() -> dict:
    return deepcopy(COMMON_DETECTION_DEFAULTS)


def select_candidate(candidates: Iterable[Mapping[str, Any]] | None) -> dict | None:
    """按 quality 选择最佳候选；空列表返回 None。"""

    best: dict | None = None
    best_score = float("-inf")
    for candidate in candidates or ():
        if not isinstance(candidate, Mapping):
            continue
        score = candidate.get("quality", 0)
        if not is_number(score):
            continue
        if best is None or float(score) > best_score:
            best = dict(candidate)
            best_score = float(score)
    return best
