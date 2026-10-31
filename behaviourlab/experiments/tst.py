"""TST 悬尾实验：classic / lis 是同一实验内的两种模式。"""

from __future__ import annotations

from typing import Any

from .base import ExperimentNotImplemented, geometry_field, parameter

DEFAULTS = {"mode": "classic"}

DETECTION_DEFAULTS: dict = {}

# 首次检测必须落在场地内，才允许锚定相对窗口。
FIRST_SEEN = "arena"

SPEC = {
    "key": "tst",
    "label": "悬尾实验",
    "short": "TST",
    "parameters": [
        parameter(
            "mode",
            "模式",
            "select",
            "classic",
            choices=[
                {"value": "classic", "label": "经典"},
                {"value": "lis", "label": "LIS"},
            ],
            section="基础",
        )
    ],
    "geometry": [
        geometry_field("arena", "场地", "polygon", min_points=3),
        geometry_field("body_line", "体长线", "line", count=2),
        geometry_field("mask", "遮挡区域", "polygon", min_points=3, optional=True),
    ],
    "metrics": [],
    "template_keys": [],
}


def validate(job: Any) -> list[dict]:
    return []


def analyze(context: Any):
    raise ExperimentNotImplemented("tst", "悬尾实验")
