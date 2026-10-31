"""OFT 旷场实验。"""

from __future__ import annotations

from typing import Any

from .base import ExperimentNotImplemented

DEFAULTS: dict = {}
DETECTION_DEFAULTS: dict = {}

SPEC = {
    "key": "oft",
    "label": "旷场实验",
    "short": "OFT",
    "parameters": [],
    "geometry": [],
    "metrics": [],
    "template_keys": [],
}


def validate(job: Any) -> list[dict]:
    return []


def analyze(context: Any):
    raise ExperimentNotImplemented("oft", "旷场实验")
