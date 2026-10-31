"""NOR 新物体识别。"""

from __future__ import annotations

from typing import Any

from .base import ExperimentNotImplemented, geometry_field

DEFAULTS: dict = {}
DETECTION_DEFAULTS: dict = {}

SPEC = {
    "key": "nor",
    "label": "新物体识别",
    "short": "NOR",
    "parameters": [],
    "geometry": [
        geometry_field("arena", "场地", "quad", count=4),
        geometry_field("object1", "物体 1", "point", count=1, role="object"),
        geometry_field("explore1", "物体 1 探索区", "polygon", min_points=3,
                       optional=True, role="explore"),
        geometry_field("object2", "物体 2", "point", count=1, role="object"),
        geometry_field("explore2", "物体 2 探索区", "polygon", min_points=3,
                       optional=True, role="explore"),
        geometry_field("mask", "遮挡区域", "polygon", min_points=3, optional=True),
    ],
    "metrics": [],
    "template_keys": [],
}


def validate(job: Any) -> list[dict]:
    return []


def analyze(context: Any):
    raise ExperimentNotImplemented("nor", "新物体识别")
