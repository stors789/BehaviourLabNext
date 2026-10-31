"""EZM 高架零迷宫。"""

from __future__ import annotations

from typing import Any

from .base import ExperimentNotImplemented

DEFAULTS: dict = {}
DETECTION_DEFAULTS: dict = {}

SPEC = {
    "key": "ezm",
    "label": "高架零迷宫",
    "short": "EZM",
    "parameters": [],
    "geometry": [],
    "metrics": [],
    "template_keys": [],
}


def validate(job: Any) -> list[dict]:
    return []


def analyze(context: Any):
    raise ExperimentNotImplemented("ezm", "高架零迷宫")
