"""EZM 高架零迷宫。"""

from __future__ import annotations

from typing import Any

from .base import ExperimentNotImplemented, geometry_field, parameter

DEFAULTS = {"ring_width": 10.0, "outer_diameter": 105.0}
DETECTION_DEFAULTS: dict = {}

SPEC = {
    "key": "ezm",
    "label": "高架零迷宫",
    "short": "EZM",
    "parameters": [
        parameter(
            "ring_width",
            "环宽",
            "number",
            10.0,
            unit="cm",
            minimum=1,
            maximum=60,
            step=0.5,
            section="场地",
        ),
        parameter(
            "outer_diameter",
            "外径",
            "number",
            105.0,
            unit="cm",
            minimum=10,
            maximum=300,
            step=1,
            section="场地",
        ),
    ],
    "geometry": [
        geometry_field("outer", "外缘", "ellipse", min_points=5),
        geometry_field("open_start", "开放区起点", "point", count=1),
        geometry_field("mask", "遮挡区域", "polygon", min_points=3, optional=True),
    ],
    "metrics": [],
    "template_keys": [],
}


def validate(job: Any) -> list[dict]:
    return []


def analyze(context: Any):
    raise ExperimentNotImplemented("ezm", "高架零迷宫")
