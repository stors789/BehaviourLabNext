"""Barnes 迷宫：圆盘、洞位与人工入箱记录。"""

from __future__ import annotations

from typing import Any

from .base import ExperimentNotImplemented, geometry_field, parameter

DEFAULTS = {"hole_count": 20}
DETECTION_DEFAULTS: dict = {}

SPEC = {
    "key": "barnes",
    "label": "Barnes 迷宫",
    "short": "Barnes",
    "parameters": [
        parameter(
            "hole_count",
            "洞数",
            "number",
            20,
            unit="个",
            minimum=1,
            maximum=40,
            step=1,
            section="场地",
        )
    ],
    "geometry": [
        geometry_field("disk", "圆盘", "ellipse", min_points=5),
        geometry_field("holes", "洞位", "holes", count_key="hole_count"),
        geometry_field("mask", "遮挡区域", "polygon", min_points=3, optional=True),
    ],
    "manual": ["hole_sequence", "entry"],
    "metrics": [],
    "template_keys": [],
}


def validate(job: Any) -> list[dict]:
    return []


def analyze(context: Any):
    raise ExperimentNotImplemented("barnes", "Barnes 迷宫")
