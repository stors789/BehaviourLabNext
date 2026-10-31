"""EPM 高架十字迷宫。"""

from __future__ import annotations

from typing import Any

from .base import ExperimentNotImplemented, geometry_field

DEFAULTS: dict = {}
DETECTION_DEFAULTS: dict = {}

SPEC = {
    "key": "epm",
    "label": "高架十字迷宫",
    "short": "EPM",
    "parameters": [],
    "geometry": [
        geometry_field("center", "中心区", "quad", count=4),
        geometry_field("open_arm_1", "开放臂 1", "quad", count=4),
        geometry_field("open_arm_2", "开放臂 2", "quad", count=4),
        geometry_field("closed_arm_1", "封闭臂 1", "quad", count=4),
        geometry_field("closed_arm_2", "封闭臂 2", "quad", count=4),
        geometry_field("mask", "遮挡区域", "polygon", min_points=3, optional=True),
    ],
    "metrics": [],
    "template_keys": [],
}


def validate(job: Any) -> list[dict]:
    return []


def analyze(context: Any):
    raise ExperimentNotImplemented("epm", "高架十字迷宫")
