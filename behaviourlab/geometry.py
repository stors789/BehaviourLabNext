"""源像素几何：视图映射、四边形单应、形状包含、方向变换与椭圆拟合。

保存坐标一律是当前 orientation 旋转后的原分辨率像素，不是 CSS 像素。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .schema import ValidationError, is_number

UNITS = ("px", "cm", "cm_estimated")
ORIENTATIONS = (0, 90, 180, 270)
ELLIPSE_EPS = 1e-9
HIT_TOLERANCE_PX = 0.5


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
    for index, (x1, y1) in enumerate(vertices):
        x2, y2 = vertices[(index + 1) % len(vertices)]
        if _on_segment((x, y), (x1, y1), (x2, y2)):
            return True
    inside = False
    for index, (x1, y1) in enumerate(vertices):
        x2, y2 = vertices[(index + 1) % len(vertices)]
        if (y1 > y) != (y2 > y):
            cross_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < cross_x:
                inside = not inside
    return inside


def _on_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> bool:
    px, py = point
    x1, y1 = start
    x2, y2 = end
    cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    if abs(cross) > 1e-7:
        return False
    return (
        min(x1, x2) - 1e-9 <= px <= max(x1, x2) + 1e-9
        and min(y1, y2) - 1e-9 <= py <= max(y1, y2) + 1e-9
    )


def distance_to_segment(
    point: Sequence[float],
    start: Sequence[float],
    end: Sequence[float],
) -> float:
    px, py = float(point[0]), float(point[1])
    x1, y1 = float(start[0]), float(start[1])
    x2, y2 = float(end[0]), float(end[1])
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


@dataclass(frozen=True)
class Viewport:
    """画布 CSS 尺寸与源图尺寸之间的映射；left/top 为画布在页面中的位置。"""

    source_width: int
    source_height: int
    css_width: float
    css_height: float
    dpr: float = 1.0
    left: float = 0.0
    top: float = 0.0

    def __post_init__(self) -> None:
        if self.source_width <= 0 or self.source_height <= 0:
            raise ValidationError("源图尺寸必须为正数", "source")
        if self.css_width <= 0 or self.css_height <= 0:
            raise ValidationError("画布尺寸必须为正数", "canvas")
        if not is_number(self.dpr) or self.dpr <= 0:
            raise ValidationError("DPR 必须为正数", "dpr")

    @property
    def scale(self) -> float:
        return min(self.css_width / self.source_width,
                   self.css_height / self.source_height)

    @property
    def offset(self) -> tuple[float, float]:
        scale = self.scale
        return (
            (self.css_width - self.source_width * scale) / 2.0,
            (self.css_height - self.source_height * scale) / 2.0,
        )

    @property
    def bitmap_size(self) -> tuple[int, int]:
        return (
            int(round(self.css_width * self.dpr)),
            int(round(self.css_height * self.dpr)),
        )

    def to_source(self, page_x: float, page_y: float) -> tuple[float, float] | None:
        """页面坐标 → 源像素；落在留白或图像边界之外返回 None。"""

        scale = self.scale
        ox, oy = self.offset
        css_x = float(page_x) - self.left - ox
        css_y = float(page_y) - self.top - oy
        if scale <= 0:
            return None
        source_x = css_x / scale
        source_y = css_y / scale
        if not (-1e-9 <= source_x <= self.source_width + 1e-9):
            return None
        if not (-1e-9 <= source_y <= self.source_height + 1e-9):
            return None
        return (source_x, source_y)

    def to_css(self, source_x: float, source_y: float) -> tuple[float, float]:
        scale = self.scale
        ox, oy = self.offset
        return (source_x * scale + ox, source_y * scale + oy)


def homography(
    quad: Sequence[Sequence[float]],
    width: float,
    height: float,
) -> list[float]:
    """四角 → (0,0),(w,0),(w,h),(0,h) 的单应矩阵（行优先 8 个未知数）。"""

    if len(quad) != 4:
        raise ValidationError("四边形需要四个角点", "quad")
    if width <= 0 or height <= 0:
        raise ValidationError("目标宽高必须为正数", "quad")
    src = [(float(p[0]), float(p[1])) for p in quad]
    dst = [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)]
    matrix: list[list[float]] = []
    for (x, y), (u, v) in zip(src, dst):
        matrix.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, u])
        matrix.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, v])
    solved = _solve(matrix)
    if solved is None:
        raise ValidationError("四边形退化，无法建立映射", "quad")
    return solved


def _solve(matrix: list[list[float]]) -> list[float] | None:
    """高斯消元；奇异矩阵返回 None。"""

    rows = [row[:] for row in matrix]
    size = len(rows)
    for column in range(size):
        pivot = max(range(column, size), key=lambda r: abs(rows[r][column]))
        if abs(rows[pivot][column]) < 1e-12:
            return None
        rows[column], rows[pivot] = rows[pivot], rows[column]
        divisor = rows[column][column]
        for index in range(column, size + 1):
            rows[column][index] /= divisor
        for other in range(size):
            if other == column:
                continue
            factor = rows[other][column]
            if factor == 0:
                continue
            for index in range(column, size + 1):
                rows[other][index] -= factor * rows[column][index]
    return [rows[index][size] for index in range(size)]


def apply_homography(matrix: Sequence[float], point: Sequence[float]) -> tuple[float, float]:
    x, y = float(point[0]), float(point[1])
    denominator = matrix[6] * x + matrix[7] * y + 1.0
    if abs(denominator) < 1e-12:
        raise ValidationError("点映射到无穷远", "quad")
    return (
        (matrix[0] * x + matrix[1] * y + matrix[2]) / denominator,
        (matrix[3] * x + matrix[4] * y + matrix[5]) / denominator,
    )


def quad_mapper(
    quad: Sequence[Sequence[float]],
    width: float,
    height: float,
):
    """返回 (矩阵, 是否交叉) 的映射器；交叉四边形标记为无效。"""

    matrix = homography(quad, width, height)
    crossed = quad_is_crossed(quad)
    if crossed:
        raise ValidationError("四边形边交叉", "quad")

    def mapper(point: Sequence[float]) -> tuple[float, float]:
        return apply_homography(matrix, point)

    return mapper


def quad_is_crossed(quad: Sequence[Sequence[float]]) -> bool:
    if len(quad) != 4:
        return True
    points = [(float(p[0]), float(p[1])) for p in quad]
    if polygon_area(points) < 1e-9:
        return True
    return _segments_cross(points[0], points[1], points[2], points[3]) or _segments_cross(
        points[1], points[2], points[3], points[0]
    )


def _segments_cross(a, b, c, d) -> bool:
    def orient(p, q, r):
        value = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        return 0 if abs(value) < 1e-12 else (1 if value > 0 else -1)

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    return o1 != o2 and o3 != o4


def segments_intersect(a: Sequence[float], b: Sequence[float],
                       c: Sequence[float], d: Sequence[float]) -> bool:
    """两条线段是否相交：含端点接触与共线重叠，共线但分离不算。"""

    def orient(p, q, r):
        value = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        return 0 if abs(value) < 1e-12 else (1 if value > 0 else -1)

    def between(p, q, r):
        return (min(p[0], q[0]) - 1e-12 <= r[0] <= max(p[0], q[0]) + 1e-12
                and min(p[1], q[1]) - 1e-12 <= r[1] <= max(p[1], q[1]) + 1e-12)

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and between(a, b, c):
        return True
    if o2 == 0 and between(a, b, d):
        return True
    if o3 == 0 and between(c, d, a):
        return True
    if o4 == 0 and between(c, d, b):
        return True
    return False


def polygon_self_intersects(points: Sequence[Sequence[float]]) -> bool:
    """简单多边形判定：任意非相邻边都不得相交。"""

    count = len(points)
    if count < 4:
        return False
    for first in range(count):
        a, b = points[first], points[(first + 1) % count]
        for second in range(first + 1, count):
            if second in ((first + 1) % count, (first - 1) % count):
                continue
            c, d = points[second], points[(second + 1) % count]
            if segments_intersect(a, b, c, d):
                return True
    return False


def ellipse_normalized(
    point: Sequence[float],
    center: Sequence[float],
    radii: Sequence[float],
    angle_deg: float = 0.0,
) -> tuple[float, float]:
    """点相对椭圆的归一化坐标；angle 为度数，半径是半轴。"""

    rx, ry = float(radii[0]), float(radii[1])
    if rx <= 0 or ry <= 0:
        raise ValidationError("椭圆半轴必须为正数", "radii")
    dx = float(point[0]) - float(center[0])
    dy = float(point[1]) - float(center[1])
    radians = math.radians(float(angle_deg))
    cos_a, sin_a = math.cos(radians), math.sin(radians)
    u = (dx * cos_a + dy * sin_a) / rx
    v = (-dx * sin_a + dy * cos_a) / ry
    return (u, v)


def ellipse_contains(
    point: Sequence[float],
    center: Sequence[float],
    radii: Sequence[float],
    angle_deg: float = 0.0,
) -> bool:
    u, v = ellipse_normalized(point, center, radii, angle_deg)
    return u * u + v * v <= 1.0 + ELLIPSE_EPS


def fit_ellipse(points: Sequence[Sequence[float]]) -> dict:
    """至少五点拟合椭圆；返回 {center,radii,angle}（半轴、度数）。"""

    if len(points) < 5:
        raise ValidationError("椭圆拟合至少需要五个点", "points")
    import cv2
    import numpy as np

    array = np.array([[float(p[0]), float(p[1])] for p in points], dtype=np.float32)
    (cx, cy), (major, minor), angle = cv2.fitEllipse(array)
    if major <= 0 or minor <= 0:
        raise ValidationError("椭圆拟合退化", "points")
    return {
        "center": [float(cx), float(cy)],
        "radii": [float(major) / 2.0, float(minor) / 2.0],
        "angle": float(angle) % 180.0,
    }


def shape_contains(
    shape: Mapping[str, Any] | None,
    point: Sequence[float],
    *,
    tolerance: float = HIT_TOLERANCE_PX,
) -> bool:
    """按形状类型判定包含；polygon/ellipse 含边界。"""

    if not isinstance(shape, Mapping):
        return False
    kind = shape.get("type")
    if kind in ("polygon", "quad"):
        return point_in_polygon(point, shape.get("points") or [])
    if kind == "line":
        points = shape.get("points") or []
        if len(points) < 2:
            return False
        return distance_to_segment(point, points[0], points[1]) <= tolerance
    if kind == "point":
        points = shape.get("points") or []
        if not points:
            return False
        return distance(point, points[0]) <= tolerance
    if kind == "ellipse":
        center = shape.get("center")
        if isinstance(center, Sequence) and shape.get("radii"):
            return ellipse_contains(
                point, center, shape["radii"], shape.get("angle", 0.0)
            )
        fitted = fit_ellipse(shape.get("points") or [])
        return ellipse_contains(
            point, fitted["center"], fitted["radii"], fitted["angle"]
        )
    if kind == "union":
        return any(shape_contains(part, point, tolerance=tolerance)
                   for part in shape.get("shapes") or [])
    if kind == "holes":
        return any(
            distance(point, hole.get("center") or [0, 0]) <= tolerance
            for hole in shape.get("holes") or []
        )
    return False


def rotate_point(
    point: Sequence[float],
    width: float,
    height: float,
    from_orientation: int,
    to_orientation: int,
) -> tuple[float, float]:
    """先把旧方向反变换回原片，再变换到新方向。"""

    if from_orientation not in ORIENTATIONS or to_orientation not in ORIENTATIONS:
        raise ValidationError("方向必须是 0/90/180/270", "orientation")
    x, y = _to_original(float(point[0]), float(point[1]),
                        width, height, from_orientation)
    return _from_original(x, y, width, height, to_orientation)


def _to_original(x: float, y: float, width: float, height: float,
                 orientation: int) -> tuple[float, float]:
    if orientation == 0:
        return (x, y)
    if orientation == 90:
        # 原 (x,y) → (H-1-y, x)；逆为 (Y, H-1-X)
        return (y, height - 1.0 - x)
    if orientation == 180:
        return (width - 1.0 - x, height - 1.0 - y)
    return (width - 1.0 - y, x)


def _from_original(x: float, y: float, width: float, height: float,
                   orientation: int) -> tuple[float, float]:
    if orientation == 0:
        return (x, y)
    if orientation == 90:
        return (height - 1.0 - y, x)
    if orientation == 180:
        return (width - 1.0 - x, height - 1.0 - y)
    return (y, width - 1.0 - x)


def rotate_ellipse(
    ellipse: Mapping[str, Any],
    width: float,
    height: float,
    from_orientation: int,
    to_orientation: int,
) -> dict:
    """椭圆中心按点变换，radii 不变，angle 增加方向差并归一到 [0,180)。"""

    center = rotate_point(
        ellipse.get("center") or [0, 0], width, height,
        from_orientation, to_orientation,
    )
    delta = (int(to_orientation) - int(from_orientation)) % 360
    angle = (float(ellipse.get("angle", 0.0)) + delta) % 180.0
    return {
        "type": "ellipse",
        "center": [center[0], center[1]],
        "radii": [float(ellipse["radii"][0]), float(ellipse["radii"][1])],
        "angle": angle,
    }


def transform_geometry(
    geometry: Mapping[str, Any],
    width: float,
    height: float,
    from_orientation: int,
    to_orientation: int,
) -> dict:
    """整份几何从旧方向换到新方向；只做一次，往返可恢复。"""

    result: dict[str, Any] = {}
    for key, value in (geometry or {}).items():
        result[key] = _transform_value(
            value, width, height, from_orientation, to_orientation
        )
    return result


def _transform_value(value, width, height, from_orientation, to_orientation):
    if isinstance(value, Mapping):
        if value.get("type") == "ellipse" and value.get("center"):
            if value.get("points") and not value.get("radii"):
                return {
                    **value,
                    "points": _transform_points(
                        value["points"], width, height,
                        from_orientation, to_orientation,
                    ),
                }
            return rotate_ellipse(value, width, height,
                                  from_orientation, to_orientation)
        if value.get("type") in ("point", "line", "polygon", "quad"):
            return {
                **value,
                "points": _transform_points(
                    value.get("points") or [], width, height,
                    from_orientation, to_orientation,
                ),
            }
        if value.get("type") == "holes":
            return {
                **value,
                "holes": [
                    {
                        **hole,
                        "center": list(
                            rotate_point(hole.get("center") or [0, 0], width, height,
                                         from_orientation, to_orientation)
                        ),
                    }
                    for hole in value.get("holes") or []
                ],
            }
        if value.get("type") == "union":
            return {
                **value,
                "shapes": [
                    _transform_value(part, width, height,
                                     from_orientation, to_orientation)
                    for part in value.get("shapes") or []
                ],
            }
        if value.get("center") and value.get("radii"):
            return rotate_ellipse(value, width, height,
                                  from_orientation, to_orientation)
        return {
            item_key: _transform_value(item, width, height,
                                       from_orientation, to_orientation)
            for item_key, item in value.items()
        }
    if isinstance(value, list):
        return [_transform_value(item, width, height,
                                 from_orientation, to_orientation)
                for item in value]
    return value


def _transform_points(
    points: Sequence[Sequence[float]],
    width: float,
    height: float,
    from_orientation: int,
    to_orientation: int,
) -> list[list[float]]:
    return [
        list(rotate_point(point, width, height, from_orientation, to_orientation))
        for point in points
    ]
