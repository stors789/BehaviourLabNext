"""Y 迷宫：支持视觉检测与专用人工 Y 路径。"""

from __future__ import annotations

from typing import Any

from .base import ExperimentNotImplemented

DEFAULTS: dict = {}
DETECTION_DEFAULTS: dict = {}

SPEC = {
    "key": "ymaze",
    "label": "Y 迷宫",
    "short": "Y 迷宫",
    "parameters": [],
    "geometry": [],
    "metrics": [],
    "template_keys": [],
}


def validate(job: Any) -> list[dict]:
    return []


def analyze(context: Any):
    raise ExperimentNotImplemented("ymaze", "Y 迷宫")
