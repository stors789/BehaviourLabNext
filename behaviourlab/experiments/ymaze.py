"""Y 迷宫：自动几何与专用人工 Y 路径共用同一实验。"""

from __future__ import annotations

from typing import Any

from .base import ExperimentNotImplemented, geometry_field, parameter

DEFAULTS = {"mode": "auto"}
DETECTION_DEFAULTS: dict = {}

SPEC = {
    "key": "ymaze",
    "label": "Y 迷宫",
    "short": "Y 迷宫",
    "parameters": [
        parameter(
            "mode",
            "路径",
            "select",
            "auto",
            choices=[
                {"value": "auto", "label": "自动几何"},
                {"value": "manual", "label": "人工记录"},
            ],
            section="基础",
        )
    ],
    "geometry": [
        geometry_field("center", "中心区", "polygon", count=3, auto_commit=True,
                       fixed=True),
        geometry_field("arm_a", "A 臂", "quad", count=4),
        geometry_field("arm_b", "B 臂", "quad", count=4),
        geometry_field("arm_c", "C 臂", "quad", count=4),
        geometry_field("mask", "遮挡区域", "polygon", min_points=3, optional=True),
    ],
    "paths": {
        "auto": ["select", "range", "arena", "detection"],
        "manual": ["select", "range", "arena", "review"],
    },
    "metrics": [],
    "template_keys": [],
}


def validate(job: Any) -> list[dict]:
    return []


def analyze(context: Any):
    raise ExperimentNotImplemented("ymaze", "Y 迷宫")
