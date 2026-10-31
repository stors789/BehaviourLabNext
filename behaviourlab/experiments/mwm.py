"""MWM Morris 水迷宫。"""

from __future__ import annotations

from typing import Any

from .base import ExperimentNotImplemented

DEFAULTS: dict = {}
DETECTION_DEFAULTS: dict = {}

SPEC = {
    "key": "mwm",
    "label": "Morris 水迷宫",
    "short": "MWM",
    "parameters": [],
    "geometry": [],
    "metrics": [],
    "template_keys": [],
}


def validate(job: Any) -> list[dict]:
    return []


def analyze(context: Any):
    raise ExperimentNotImplemented("mwm", "Morris 水迷宫")
