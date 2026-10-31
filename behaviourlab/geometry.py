"""源像素几何与尺度换算。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from .schema import ValidationError, is_number

UNITS = ("px", "cm", "cm_estimated")


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class Scale:
    """像素到厘米的换算；未知尺度时为 None，不把估计值写成实测。"""

    px_per_cm: float | None
    unit: str = "px"

    def __post_init__(self) -> None:
        if self.px_per_cm is not None:
            if not is_number(self.px_per_cm) or self.px_per_cm <= 0:
                raise ValidationError("每厘米像素数必须为正数", "px_per_cm")
        if self.unit not in UNITS:
            raise ValidationError("未知坐标单位", "unit")

    def to_cm(self, pixels: float) -> float | None:
        if self.px_per_cm is None:
            return None
        return pixels / self.px_per_cm


def distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def polygon_area(points: Iterable[Sequence[float]]) -> float:
    """鞋带公式；少于三个点返回 0。"""

    vertices = [(float(p[0]), float(p[1])) for p in points]
    if len(vertices) < 3:
        return 0.0
    total = 0.0
    for index, (x1, y1) in enumerate(vertices):
        x2, y2 = vertices[(index + 1) % len(vertices)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def point_in_polygon(point: Sequence[float], polygon: Sequence[Sequence[float]]) -> bool:
    """射线法；边界点结果按内部处理。"""

    x, y = float(point[0]), float(point[1])
    vertices = [(float(p[0]), float(p[1])) for p in polygon]
    if len(vertices) < 3:
        return False
    inside = False
    for index, (x1, y1) in enumerate(vertices):
        x2, y2 = vertices[(index + 1) % len(vertices)]
        if (y1 > y) != (y2 > y):
            cross_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < cross_x:
                inside = not inside
    return inside
