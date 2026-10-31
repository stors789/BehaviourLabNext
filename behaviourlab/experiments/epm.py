"""EPM 高架十字迷宫。"""

from __future__ import annotations

from typing import Any

from .base import ExperimentNotImplemented

DEFAULTS: dict = {}
DETECTION_DEFAULTS: dict = {}

SPEC = {
    "key": "epm",
    "label": "高架十字迷宫",
    "short": "EPM",
    "parameters": [],
    "geometry": [],
    "metrics": [],
    "template_keys": [],
}


def validate(job: Any) -> list[dict]:
    return []


def analyze(context: Any):
    raise ExperimentNotImplemented("epm", "高架十字迷宫")
