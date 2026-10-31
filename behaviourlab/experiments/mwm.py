"""MWM Morris 水迷宫。"""

from __future__ import annotations

from typing import Any

from .base import ExperimentNotImplemented, geometry_field, parameter

DEFAULTS = {"coordinate_rotation_deg": 0.0}
DETECTION_DEFAULTS: dict = {}

SPEC = {
    "key": "mwm",
    "label": "Morris 水迷宫",
    "short": "MWM",
    "parameters": [
        parameter(
            "coordinate_rotation_deg",
            "象限参考线旋转",
            "number",
            0.0,
            unit="°",
            minimum=0,
            maximum=360,
            step=1,
            section="场地",
        )
    ],
    "geometry": [
        geometry_field("arena", "水池", "ellipse", min_points=5),
        geometry_field("platform", "平台", "ellipse_or_point", min_points=1,
                       modes=["center", "ellipse"]),
        geometry_field("mask", "遮挡区域", "polygon", min_points=3, optional=True),
    ],
    "metrics": [],
    "template_keys": [],
}


def validate(job: Any) -> list[dict]:
    return []


def analyze(context: Any):
    raise ExperimentNotImplemented("mwm", "Morris 水迷宫")
