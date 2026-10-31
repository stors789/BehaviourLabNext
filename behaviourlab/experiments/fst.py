"""FST 强迫游泳实验。"""

from __future__ import annotations

from typing import Any

from .base import ExperimentNotImplemented, geometry_field

DEFAULTS: dict = {}
DETECTION_DEFAULTS: dict = {}

# 首次检测必须落在场地内，才允许锚定相对窗口。
FIRST_SEEN = "arena"

SPEC = {
    "key": "fst",
    "label": "强迫游泳",
    "short": "FST",
    "parameters": [],
    "geometry": [
        geometry_field("arena", "场地", "polygon", min_points=3),
        geometry_field("diameter_line", "直径线", "line", count=2),
        geometry_field("waterline", "水位线", "line", count=2, optional=True),
        geometry_field("mask", "遮挡区域", "polygon", min_points=3, optional=True),
    ],
    "metrics": [],
    "template_keys": [],
}


def validate(job: Any) -> list[dict]:
    return []


def analyze(context: Any):
    raise ExperimentNotImplemented("fst", "强迫游泳")
