"""LDB 明暗箱。"""

from __future__ import annotations

from typing import Any

from .base import ExperimentNotImplemented, geometry_field, parameter

DEFAULTS = {"dark_visible": True}
DETECTION_DEFAULTS: dict = {}

SPEC = {
    "key": "ldb",
    "label": "明暗箱",
    "short": "LDB",
    "parameters": [
        parameter(
            "dark_visible",
            "暗室可见",
            "checkbox",
            True,
            section="场地",
        )
    ],
    "geometry": [
        geometry_field("light", "明室", "polygon", min_points=3),
        geometry_field("dark", "暗室", "polygon", min_points=3),
        geometry_field("door", "门区", "polygon", min_points=3, optional=True),
        geometry_field("mask", "遮挡区域", "polygon", min_points=3, optional=True),
    ],
    "metrics": [],
    "template_keys": [],
}


def validate(job: Any) -> list[dict]:
    return []


def analyze(context: Any):
    raise ExperimentNotImplemented("ldb", "明暗箱")
