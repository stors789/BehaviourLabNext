"""EZM 高架零迷宫：四段交替跑道、环上占用与相邻区域进入确认。

外缘椭圆先按 ``A = Rα·diag(D/(2rx), D/(2ry))·Rαᵀ`` 映射到以场地中心为
原点、外径为 D 的物理厘米坐标（y 仍随图像向下）。O1 起始射线只由
``open_boundary`` 单点的方向定义；θ 从该射线起每 90° 顺时针依次是
O1、C1、O2、C2。进入只在出现真实更晚观测且确认时间足够、并且新旧区域
在环上相邻时成立；边界曝光保留已确认主区，区外与未知清空主区（重获
主区只是重新定位，不算进入）。
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .. import detection, geometry as geo, media, observation, timeline
from ..config import ALGORITHM_VERSION
from ..schema import AnalysisResult, ValidationError, is_number
from .base import geometry_field, metric, parameter

KEY = "ezm"

DEFAULTS = {
    "outer_diameter_cm": 60.0,
    "ring_width_cm": 5.0,
    "boundary_margin_cm": 0.5,
    "confirm_s": 0.12,
    "quality_threshold": 0.4,
    "bin_seconds": 60.0,
}

OPTION_KEYS = tuple(DEFAULTS)
POSITIVE_KEYS = ("outer_diameter_cm", "ring_width_cm", "confirm_s",
                 "bin_seconds")
NONNEGATIVE_KEYS = ("boundary_margin_cm",)

# 环上四段跑道；顺序即从起始射线顺时针的判定顺序。
ZONE_KEYS = ("O1", "C1", "O2", "C2")
OPEN_KEYS = ("O1", "O2")
CLOSED_KEYS = ("C1", "C2")
EXTRA_ZONES = ("boundary", "outside", "unknown")
ALL_ZONES = ZONE_KEYS + EXTRA_ZONES
ZONE_GROUPS = {"O1": "open", "O2": "open", "C1": "closed", "C2": "closed"}

HALF_PI = math.pi / 2.0
TAU = math.pi * 2.0
TIME_EPS = 1e-9
BOUNDARY_EPS = 1e-9
CONTINUITY_EPS = 1e-7
CENTER_EPS = 1e-8

# 首次检测必须落在场地内，才允许锚定相对窗口。
FIRST_SEEN = "arena"

# 检测侧默认值：背景方法与形态学参数按本实验契约固定，质量阈值仍与统计
# 身体质量阈值分开保存（一个在 detection，一个在 options）。
DETECTION_DEFAULTS: dict = {
    "bg_method": "local_otsu",
    "erosion_kernel": 3,
    "erosion_iter": 1,
    "quality_threshold": 0.4,
}

SPEC = {
    "key": KEY,
    "label": "高架零迷宫 / O迷宫",
    "short": "EZM",
    "parameters": [
        parameter("outer_diameter_cm", "外径", "number", 60.0, unit="cm",
                  minimum=0, step="any"),
        parameter("ring_width_cm", "跑道宽", "number", 5.0, unit="cm",
                  minimum=0, step="any"),
        parameter("boundary_margin_cm", "边界余量", "number", 0.5, unit="cm",
                  minimum=0, step="any", section="统计"),
        parameter("confirm_s", "进入确认时间", "number", 0.12, unit="s",
                  minimum=0, step="any", section="统计"),
        parameter("quality_threshold", "身体质量阈值", "number", 0.4,
                  minimum=0, maximum=1, step="any", section="统计"),
        parameter("bin_seconds", "时间箱长度", "number", 60.0, unit="s",
                  minimum=0, step="any", section="统计"),
    ],
    "geometry": [
        geometry_field("arena", "外缘", "ellipse", min_points=5),
        geometry_field("open_boundary", "O1 开放区起始边界", "point", count=1),
        geometry_field("mask", "检测遮罩", "polygon", min_points=3,
                       optional=True),
    ],
    "metrics": [
        metric("open_time_s", "开放区时间", "s"),
        metric("closed_time_s", "封闭区时间", "s"),
        metric("boundary_time_s", "边界时间", "s"),
        metric("outside_time_s", "区外时间", "s"),
        metric("unknown_time_s", "未知时间", "s"),
        metric("open_time_percent_total", "开放区时间占总时长", "%"),
        metric("open_time_percent_known", "开放区时间占开放+封闭", "%"),
        metric("open_entries", "开放区进入次数", "次"),
        metric("closed_entries", "封闭区进入次数", "次"),
        metric("first_open_observation_latency_s", "首次观察开放区潜伏期", "s"),
        metric("first_open_entry_latency_s", "首次确认进入开放区潜伏期", "s"),
        metric("total_distance_cm", "总距离", "cm"),
        metric("mean_speed_cm_s", "平均速度", "cm/s"),
    ],
    "template_keys": ["outer_diameter_cm", "ring_width_cm"],
}


# -- 数值、椭圆与点 -----------------------------------------------------

def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not is_number(value):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _point(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    x, y = _finite(value[0]), _finite(value[1])
    if x is None or y is None:
        return None
    return x, y


def _shape_points(shape: Any) -> list[tuple[float, float]] | None:
    if not isinstance(shape, Mapping):
        return None
    raw = shape.get("points")
    if not isinstance(raw, (list, tuple)):
        return None
    points: list[tuple[float, float]] = []
    for item in raw:
        point = _point(item)
        if point is None:
            return None
        points.append(point)
    return points


def _simple_polygon(shape: Any) -> list[tuple[float, float]] | None:
    """遮罩必须是 type=polygon 的简单多边形；其他类型一律不认。"""

    if not isinstance(shape, Mapping) or shape.get("type") != "polygon":
        return None
    points = _shape_points(shape)
    if points is None or len(points) < 3:
        return None
    if abs(geo.polygon_area(points)) <= 1e-9:
        return None
    if geo.polygon_self_intersects(points):
        return None
    return points


def arena_params(geometry: Mapping[str, Any] | None
                 ) -> tuple[list[float], list[float], float] | None:
    """外缘椭圆 (center, radii, angle)；中心与半轴，或至少五点拟合。

    类型必须是 ``ellipse``：错误的类型不会被就地改写，也不会当成椭圆解析。
    """

    arena = (geometry or {}).get("arena")
    if not isinstance(arena, Mapping) or arena.get("type") != "ellipse":
        return None
    center = arena.get("center")
    radii = arena.get("radii")
    if (isinstance(center, (list, tuple)) and len(center) >= 2
            and isinstance(radii, (list, tuple)) and len(radii) >= 2):
        cx, cy = _finite(center[0]), _finite(center[1])
        rx, ry = _finite(radii[0]), _finite(radii[1])
        angle = _finite(arena.get("angle", 0.0))
        if None in (cx, cy, rx, ry, angle) or rx <= 0 or ry <= 0:
            return None
        return [cx, cy], [rx, ry], angle
    points = _shape_points(arena)
    if points is None or len(points) < 5:
        return None
    try:
        fitted = geo.fit_ellipse(points)
    except ValidationError:
        return None
    rx, ry = float(fitted["radii"][0]), float(fitted["radii"][1])
    if rx <= 0 or ry <= 0:
        return None
    return ([float(fitted["center"][0]), float(fitted["center"][1])],
            [rx, ry], float(fitted["angle"]))


def boundary_point(geometry: Mapping[str, Any] | None
                   ) -> tuple[float, float] | None:
    """open_boundary 必须是一个 type=point 的有限源像素点。"""

    shape = (geometry or {}).get("open_boundary")
    if not isinstance(shape, Mapping) or shape.get("type") != "point":
        return None
    points = _shape_points(shape)
    if points is None or len(points) != 1:
        return None
    return points[0]


def affine_matrix(center: Sequence[float], radii: Sequence[float],
                  angle_deg: float, diameter_cm: float) -> tuple[float, ...]:
    """A = Rα·diag(D/(2rx), D/(2ry))·Rαᵀ；对称，返回 (m00, m01, m11)。"""

    rx, ry = float(radii[0]), float(radii[1])
    if rx <= 0 or ry <= 0:
        raise ValidationError("椭圆半轴必须为正数", "geometry.arena")
    if not math.isfinite(float(diameter_cm)) or float(diameter_cm) <= 0:
        raise ValidationError("外径必须大于 0", "options.outer_diameter_cm")
    a = float(diameter_cm) / (2.0 * rx)
    b = float(diameter_cm) / (2.0 * ry)
    theta = math.radians(float(angle_deg))
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    return (a * cos_t * cos_t + b * sin_t * sin_t,
            (a - b) * cos_t * sin_t,
            a * sin_t * sin_t + b * cos_t * cos_t)


def to_cm(point: Sequence[float], center: Sequence[float],
          matrix: Sequence[float]) -> tuple[float, float]:
    """源像素 → 物理厘米；原点为场地中心，y 仍随图像向下。"""

    dx = float(point[0]) - float(center[0])
    dy = float(point[1]) - float(center[1])
    return (matrix[0] * dx + matrix[1] * dy,
            matrix[1] * dx + matrix[2] * dy)


def boundary_angle(geometry: Mapping[str, Any] | None) -> float | None:
    """起始边界的方向：只用 D=1 映射判有效并取角。

    ``A_1`` 把相对 arena 中心的向量按半轴归一；范数 < 1e-8 即太接近圆心。
    这条判据与 ``outer_diameter_cm`` 无关，方向在任意 D>0 下都相同。
    """

    params = arena_params(geometry)
    point = boundary_point(geometry)
    if params is None or point is None:
        return None
    center, radii, angle = params
    vector = to_cm(point, center, affine_matrix(center, radii, angle, 1.0))
    if math.hypot(vector[0], vector[1]) < CENTER_EPS:
        return None
    return math.atan2(vector[1], vector[0])


def _inverse(matrix: Sequence[float]) -> tuple[float, ...] | None:
    m00, m01, m11 = float(matrix[0]), float(matrix[1]), float(matrix[2])
    det = m00 * m11 - m01 * m01
    if abs(det) <= 1e-12:
        return None
    return (m11 / det, -m01 / det, m00 / det)


def physical_to_px(vector: Sequence[float], center: Sequence[float],
                   inverse: Sequence[float]) -> tuple[float, float]:
    """物理厘米向量 → 源像素偏移（用于画内外圈之间的分界线）。"""

    ux, uy = float(vector[0]), float(vector[1])
    return (float(center[0]) + inverse[0] * ux + inverse[1] * uy,
            float(center[1]) + inverse[1] * ux + inverse[2] * uy)


# -- 参数与几何校验 -----------------------------------------------------

def validate_unit(options: Mapping[str, Any] | None,
                  geometry: Mapping[str, Any] | None,
                  provided: Iterable[str] | None = None, *,
                  complete: bool = False) -> list[dict]:
    """只校验显式参数与已绘几何；未完成草稿仍可保存。"""

    values = options if isinstance(options, Mapping) else {}
    provided = set(OPTION_KEYS if provided is None else provided)
    issues: list[dict] = []

    for key in OPTION_KEYS:
        value = _finite(values.get(key))
        if value is None:
            # 未显式给出且合并后仍缺失时只在草稿里静默，最终校验由工厂默认补足。
            if key in provided:
                issues.append({"field": f"options.{key}",
                               "message": "必须是有限数值"})
            continue
        if key in POSITIVE_KEYS and value <= 0:
            issues.append({"field": f"options.{key}", "message": "必须大于 0"})
        elif key in NONNEGATIVE_KEYS and value < 0:
            issues.append({"field": f"options.{key}", "message": "必须 ≥ 0"})
        elif key == "quality_threshold" and not 0 <= value <= 1:
            issues.append({"field": "options.quality_threshold",
                           "message": "身体质量阈值必须在 0 到 1 之间"})

    # 跑道宽必须小于半径；用最终有效值比较，默认值也不能绕过这条约束。
    diameter = _finite(values.get("outer_diameter_cm"))
    width = _finite(values.get("ring_width_cm"))
    if diameter is not None and diameter > 0 and width is not None and width > 0 \
            and width >= diameter / 2.0:
        issues.append({
            "field": "options.ring_width_cm",
            "message": f"跑道宽必须小于外径的一半（{diameter / 2.0:g} cm）",
        })

    issues.extend(validate_geometry(geometry, require=complete))
    return issues


def validate_geometry(geometry: Mapping[str, Any] | None,
                      *, require: bool = False) -> list[dict]:
    """外缘必须是 ellipse、起始边界是一个非圆心点、遮罩必须是 polygon。

    已提供但类型错误的几何一律报错，且不会被就地改写成合法类型；完全缺失
    的步骤只在 ``require`` 时报告，未完成草稿仍可保存。
    """

    geometry = geometry or {}
    issues: list[dict] = []
    raw_arena = geometry.get("arena")
    arena: tuple[list[float], list[float], float] | None = None
    if raw_arena is None:
        if require:
            issues.append({"field": "geometry.arena", "message": "缺少外缘"})
    elif not isinstance(raw_arena, Mapping) \
            or raw_arena.get("type") != "ellipse":
        issues.append({"field": "geometry.arena",
                       "message": "外缘必须是 ellipse 类型"})
    else:
        arena = arena_params(geometry)
        if arena is None:
            issues.append({"field": "geometry.arena",
                           "message": "外缘需要中心与两个正半轴，或至少 5 个有限边缘点"})

    shape = geometry.get("open_boundary")
    if shape is None:
        if require:
            issues.append({"field": "geometry.open_boundary",
                           "message": "缺少 O1 开放区起始边界"})
    else:
        point = boundary_point(geometry)
        if point is None:
            issues.append({"field": "geometry.open_boundary",
                           "message": "需要 1 个有限源像素点"})
        elif arena is not None:
            theta = boundary_angle(geometry)
            if theta is None:
                issues.append({"field": "geometry.open_boundary",
                               "message": "起始边界太接近圆心，无法定义方向"})

    mask = geometry.get("mask")
    if mask is not None:
        if not isinstance(mask, Mapping) or mask.get("type") != "polygon":
            issues.append({"field": "geometry.mask",
                           "message": "检测遮罩必须是 polygon 类型"})
        elif _simple_polygon(mask) is None:
            issues.append({"field": "geometry.mask",
                           "message": "检测遮罩需要至少 3 个有限点、正面积且不自交"})
    return issues


def step_drawn(geometry: Mapping[str, Any] | None, key: str) -> bool:
    """画布、预览与最终校验共用的“该步骤是否已绘制”判定。"""

    geometry = geometry or {}
    if key == "arena":
        return arena_params(geometry) is not None
    if key == "open_boundary":
        return boundary_point(geometry) is not None
    if key == "mask":
        # 类型错误的遮罩不算已绘制，避免它被当成检测区域。
        return _simple_polygon(geometry.get("mask")) is not None
    return geometry.get(key) is not None


def preview_geometry_issues(geometry: Mapping[str, Any] | None) -> list[dict]:
    """预览入口的最小几何检查：已提供但类型错误的几何不继续运行。"""

    return validate_geometry(geometry, require=False)


# -- 环上几何 -----------------------------------------------------------

class _Ring:
    """物理坐标下的环带与四段跑道判定。"""

    def __init__(self, geometry: Mapping[str, Any],
                 options: Mapping[str, Any]) -> None:
        params = arena_params(geometry)
        if params is None:
            raise ValidationError("外缘需要合法椭圆", "geometry.arena")
        start = boundary_point(geometry)
        if start is None:
            raise ValidationError("缺少 O1 开放区起始边界",
                                  "geometry.open_boundary")
        self.center, self.radii, self.angle = params
        self.diameter = float(options["outer_diameter_cm"])
        self.width = float(options["ring_width_cm"])
        self.margin = float(options["boundary_margin_cm"])
        # 身体坐标用实际 D；方向判据固定 D=1，两者不混用。
        self.matrix = affine_matrix(self.center, self.radii, self.angle,
                                   self.diameter)
        theta0 = boundary_angle(geometry)
        if theta0 is None:
            raise ValidationError("起始边界太接近圆心，无法定义方向",
                                  "geometry.open_boundary")
        self.theta0 = theta0
        self.radius = self.diameter / 2.0
        self.inner_radius = self.radius - self.width

    def map(self, point: Sequence[float]) -> tuple[float, float]:
        return to_cm(point, self.center, self.matrix)

    def zone_of(self, x: float, y: float) -> str:
        """环带外为 outside；靠近分界射线 margin 内为 boundary。"""

        rho = math.hypot(x, y)
        if rho < self.inner_radius - BOUNDARY_EPS \
                or rho > self.radius + BOUNDARY_EPS:
            return "outside"
        theta = (math.atan2(y, x) - self.theta0) % TAU
        if theta >= TAU - 1e-12:
            theta = 0.0
        delta = theta % HALF_PI
        distance = rho * math.sin(min(delta, HALF_PI - delta))
        if distance <= self.margin + BOUNDARY_EPS:
            return "boundary"
        quarter = int(math.floor(theta / HALF_PI)) % 4
        return ZONE_KEYS[quarter]


def boundary_guides(geometry: Mapping[str, Any] | None,
                    options: Mapping[str, Any] | None) -> dict | None:
    """预览画布用的像素射线：中心、四条分界线、四个标签锚点。"""

    values = options if isinstance(options, Mapping) else {}
    diameter = _finite(values.get("outer_diameter_cm"))
    width = _finite(values.get("ring_width_cm"))
    if diameter is None or diameter <= 0 or width is None or width < 0:
        return None
    try:
        ring = _Ring(geometry or {}, {"outer_diameter_cm": diameter,
                                      "ring_width_cm": width,
                                      "boundary_margin_cm": 0.0})
    except (ValidationError, KeyError):
        return None
    inverse = _inverse(ring.matrix)
    if inverse is None:
        return None
    rays = []
    labels = []
    for index in range(4):
        angle = ring.theta0 + index * HALF_PI
        unit = (math.cos(angle), math.sin(angle))
        rays.append({
            "angle_deg": math.degrees(angle) % 360.0,
            "from": list(physical_to_px(
                (ring.inner_radius * unit[0], ring.inner_radius * unit[1]),
                ring.center, inverse)),
            "to": list(physical_to_px(
                (ring.radius * unit[0], ring.radius * unit[1]),
                ring.center, inverse)),
            "zone": ZONE_KEYS[index],
        })
        middle = ring.theta0 + index * HALF_PI + HALF_PI / 2.0
        label_unit = (math.cos(middle), math.sin(middle))
        radius = ring.inner_radius + (ring.radius - ring.inner_radius) / 2.0
        labels.append({
            "zone": ZONE_KEYS[index],
            "open": ZONE_KEYS[index] in OPEN_KEYS,
            "point": list(physical_to_px(
                (radius * label_unit[0], radius * label_unit[1]),
                ring.center, inverse)),
        })
    return {
        "center": [float(ring.center[0]), float(ring.center[1])],
        "radii": [float(ring.radii[0]), float(ring.radii[1])],
        "angle_deg": float(ring.angle),
        "inner_scale": (ring.inner_radius / ring.radius
                        if ring.radius > 0 else 0.0),
        "rings": {"outer": ring.radius, "inner": ring.inner_radius},
        "rays": rays,
        "labels": labels,
        "boundary_angle_deg": math.degrees(ring.theta0) % 360.0,
    }


# -- 观测行 -------------------------------------------------------------

def _unit_dict(unit: Mapping[str, Any]) -> dict:
    return {"id": unit.get("id"), "name": unit.get("name"),
            "animal_id": unit.get("animal_id")}


def observation_row(unit: Mapping[str, Any], frame: Any,
                    result: Any) -> dict:
    return timeline.base_row(
        _unit_dict(unit), frame, result,
        extra={"detection_method": (unit.get("detection") or {}).get("bg_method")},
    )


def _region_for(unit: Mapping[str, Any], options: Mapping[str, Any],
                width: int, height: int) -> np.ndarray:
    region, _ = detection.geometry_region(unit.get("geometry") or {}, KEY,
                                          options, width, height)
    return region


def _annotate(row: dict, ring: _Ring, threshold: float) -> None:
    """一行的共同观测：物理坐标、身体质量、七类之一与运动坐标。"""

    row["coordinate_unit"] = "cm"
    row["body_qualified"] = False
    row["zone"] = "unknown"
    row["x_cm"] = None
    row["y_cm"] = None
    row["motion_x"] = None
    row["motion_y"] = None
    if not row.get("detected") or row.get("x_px") is None:
        return
    x_cm, y_cm = ring.map((row["x_px"], row["y_px"]))
    row["x_cm"] = x_cm
    row["y_cm"] = y_cm
    quality = float(row.get("quality") or 0.0)
    qualified = quality + TIME_EPS >= threshold
    row["body_qualified"] = qualified
    if not qualified:
        # 质量不足不是封闭区：保留 unknown，也不产生运动坐标。
        return
    zone = ring.zone_of(x_cm, y_cm)
    row["zone"] = zone
    if zone in ZONE_GROUPS or zone == "boundary":
        row["motion_x"] = x_cm
        row["motion_y"] = y_cm


# -- 进入状态机 ---------------------------------------------------------

def _clear(state: dict) -> None:
    state["candidate"] = None
    state["candidate_start"] = None
    state["candidate_end"] = None


def _track(rows: Sequence[Mapping[str, Any]], window: timeline.Window,
           options: Mapping[str, Any], unit: Mapping[str, Any]) -> dict:
    """主区占用：初始区、相邻确认进入、边界保留与区外重定位。"""

    events: list[dict] = []
    entries = {"open": 0, "closed": 0}
    result = {
        "initial_zone": None,
        "first_open_observation_latency_s": None,
        "first_open_entry_latency_s": None,
        "entries": entries,
        "events": events,
    }
    if not window.known:
        return result
    origin = float(window.start_s)
    confirm = float(options["confirm_s"])
    state = {"candidate": None, "candidate_start": None, "candidate_end": None,
             "accepted": None}

    for row in sorted((item for item in rows if item.get("in_window")),
                      key=lambda item: (item["time_s"], item["frame"])):
        moment = max(float(row["time_s"]), float(row["interval_start_s"]))
        zone = row.get("zone") or "unknown"
        row["zone_event"] = False
        if zone in OPEN_KEYS \
                and result["first_open_observation_latency_s"] is None:
            # 观察时刻可以等于窗起，与真正的进入潜伏期分开保存。
            result["first_open_observation_latency_s"] = moment - origin

        if zone in ZONE_GROUPS:
            if state["accepted"] is None:
                state["accepted"] = zone
                if result["initial_zone"] is None:
                    result["initial_zone"] = zone
                _clear(state)
                row["accepted_zone"] = state["accepted"]
                continue
            if zone == state["accepted"]:
                _clear(state)
                row["accepted_zone"] = state["accepted"]
                continue
            contiguous = (state["candidate_end"] is not None
                          and abs(float(state["candidate_end"])
                                  - float(row["interval_start_s"]))
                          <= CONTINUITY_EPS)
            if state["candidate"] != zone or not contiguous:
                # 候选之间曝光不接续就重新起算，不借用上一段的时长。
                state["candidate"] = zone
                state["candidate_start"] = moment
            state["candidate_end"] = float(row["interval_end_s"])
            start = float(state["candidate_start"])
            if moment > start and moment - start >= confirm - TIME_EPS:
                previous = state["accepted"]
                index = ZONE_KEYS.index(zone) - ZONE_KEYS.index(previous)
                adjacent = index % 4 in (1, 3)
                state["accepted"] = zone
                if adjacent:
                    group = ZONE_GROUPS[zone]
                    entries[group] += 1
                    row["zone_event"] = True
                    events.append({
                        "event_id": f"{unit.get('id')}:zone_entry:"
                                    f"{len(events) + 1}",
                        "unit_id": unit.get("id"),
                        "unit_name": unit.get("name"),
                        "kind": "zone_entry",
                        "time_s": start,
                        "confirmed_s": moment,
                        "target": zone,
                        "from_zone": previous,
                        "source": "automatic",
                        "frame": row["frame"],
                    })
                    if zone in OPEN_KEYS \
                            and result["first_open_entry_latency_s"] is None:
                        result["first_open_entry_latency_s"] = start - origin
                _clear(state)
        elif zone == "boundary":
            # 真正经过边界后仍能确认邻区：只清候选，保留已确认主区。
            _clear(state)
        else:
            # 区外与未知都不能证明转移：清空主区，重获主区只是重新定位。
            state["accepted"] = None
            _clear(state)
        row["accepted_zone"] = state["accepted"]

    return result


# -- 汇总、分箱与表格 ---------------------------------------------------

def _percent(numerator: float, denominator: float) -> float | None:
    return timeline.ratio(numerator * 100.0, denominator)


def _zone_seconds(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    seconds = {key: 0.0 for key in ALL_ZONES}
    for row in rows:
        if not row.get("in_window"):
            continue
        span = float(row.get("dt_s") or 0.0)
        if span <= 0:
            continue
        zone = row.get("zone") or "unknown"
        if zone not in seconds:
            zone = "unknown"
        seconds[zone] += span
    return seconds


def _piece_distance(segment: Mapping[str, Any], low: float,
                    high: float) -> float:
    start = float(segment["start_s"])
    end = float(segment["end_s"])
    span = end - start
    if span <= 0:
        return 0.0
    p0, p1 = segment["p0"], segment["p1"]
    ratio_low = (low - start) / span
    ratio_high = (high - start) / span
    x0 = float(p0[0]) + (float(p1[0]) - float(p0[0])) * ratio_low
    y0 = float(p0[1]) + (float(p1[1]) - float(p0[1])) * ratio_low
    x1 = float(p0[0]) + (float(p1[0]) - float(p0[0])) * ratio_high
    y1 = float(p0[1]) + (float(p1[1]) - float(p0[1])) * ratio_high
    return math.hypot(x1 - x0, y1 - y0)


def _bin_rows(rows: Sequence[Mapping[str, Any]],
              segments: Sequence[Mapping[str, Any]],
              window: timeline.Window, options: Mapping[str, Any],
              unit: Mapping[str, Any]) -> list[dict]:
    """曝光与路径分别按真实交叠比例分箱；箱间合计必须守恒。"""

    if not window.known or window.duration_s <= 0:
        return []
    width = float(options["bin_seconds"])
    origin = float(window.start_s)
    duration = window.duration_s
    count = max(1, int(math.ceil((duration - TIME_EPS) / width)))
    buckets = [{key: 0.0 for key in ALL_ZONES} for _ in range(count)]
    distance = [0.0] * count
    measurable = [0.0] * count

    for row in rows:
        if not row.get("in_window"):
            continue
        zone = row.get("zone") or "unknown"
        if zone not in ALL_ZONES:
            zone = "unknown"
        for index, low, high in timeline.split_interval(
                row["interval_start_s"], row["interval_end_s"], origin, width):
            if 0 <= index < count:
                buckets[index][zone] += high - low

    for segment in segments:
        span = float(segment["end_s"]) - float(segment["start_s"])
        if span <= 0:
            continue
        for index, low, high in timeline.split_interval(
                segment["start_s"], segment["end_s"], origin, width):
            if not 0 <= index < count:
                continue
            distance[index] += _piece_distance(segment, low, high)
            measurable[index] += high - low

    bins: list[dict] = []
    for index in range(count):
        start = index * width
        end = min(start + width, duration)
        bucket = buckets[index]
        opened = bucket["O1"] + bucket["O2"]
        closed = bucket["C1"] + bucket["C2"]
        bins.append({
            "unit_id": unit.get("id"),
            "unit_name": unit.get("name"),
            "bin_index": index + 1,
            "start_s": start,
            "end_s": end,
            "duration_s": end - start,
            "open_time_s": opened,
            "closed_time_s": closed,
            "boundary_time_s": bucket["boundary"],
            "outside_time_s": bucket["outside"],
            "unknown_time_s": bucket["unknown"],
            "distance_cm": distance[index],
            "measurable_motion_time_s": measurable[index],
            "mean_speed_cm_s": (distance[index] / measurable[index]
                                if measurable[index] > 0 else None),
        })
    return bins


def _zone_table(unit: Mapping[str, Any], seconds: Mapping[str, float],
                events: Sequence[Mapping[str, Any]]) -> list[dict]:
    """固定七行：每区累计时长与真正以该区为 target 的事件数。"""

    counts = {key: 0 for key in ALL_ZONES}
    for event in events:
        target = event.get("target")
        if target in counts:
            counts[target] += 1
    return [{
        "unit_id": unit.get("id"),
        "unit_name": unit.get("name"),
        "zone": key,
        "time_s": float(seconds.get(key, 0.0)),
        "entries": counts[key],
    } for key in ALL_ZONES]


def _summary(unit: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
             segments: Sequence[Mapping[str, Any]], window: timeline.Window,
             options: Mapping[str, Any], ring: _Ring,
             tracking: Mapping[str, Any]) -> dict:
    times = timeline.time_summary(rows)
    seconds = _zone_seconds(rows)
    opened = seconds["O1"] + seconds["O2"]
    closed = seconds["C1"] + seconds["C2"]
    known = opened + closed + seconds["boundary"] + seconds["outside"]

    distance = measurable = 0.0
    max_speed: float | None = None
    for segment in segments:
        distance += float(segment["distance"])
        span = float(segment["end_s"]) - float(segment["start_s"])
        measurable += span
        if span > 0:
            speed = float(segment["speed"])
            max_speed = speed if max_speed is None else max(max_speed, speed)

    detected_frames = window_frames = 0
    for row in rows:
        if row.get("in_window"):
            window_frames += 1
            if row.get("detected"):
                detected_frames += 1

    first_entry = tracking["first_open_entry_latency_s"]
    summary = {
        "unit_id": unit.get("id"),
        "unit_name": unit.get("name"),
        "mode": KEY,
        "coordinate_unit": "cm",
        "total_time_s": times["total_time_s"],
        "observed_time_s": times["observed_time_s"],
        "missing_time_s": times["missing_time_s"],
        "detected_frames": detected_frames,
        "window_frames": window_frames,
        "open_time_s": opened,
        "closed_time_s": closed,
        "boundary_time_s": seconds["boundary"],
        "outside_time_s": seconds["outside"],
        "unknown_time_s": seconds["unknown"],
        "known_zone_time_s": known,
        "zone_seconds": dict(seconds),
        "open_time_percent_total": _percent(opened, times["total_time_s"]),
        "open_time_percent_known": _percent(opened, opened + closed),
        "initial_zone": tracking["initial_zone"],
        "first_open_observation_latency_s":
            tracking["first_open_observation_latency_s"],
        "first_open_entry_latency_s": first_entry,
        "first_open_entry_censored": first_entry is None,
        "open_entries": int(tracking["entries"]["open"]),
        "closed_entries": int(tracking["entries"]["closed"]),
        "total_distance_cm": distance,
        "measurable_motion_time_s": measurable,
        "mean_speed_cm_s": distance / measurable if measurable > 0 else None,
        "max_speed_cm_s": max_speed,
        "analysis_window_start_s": window.start_s if window.known else None,
        "analysis_window_end_s": window.end_s if window.known else None,
        # 图形与结果页只读这些由保存几何与实际尺寸算出的量。
        "outer_diameter_cm": ring.diameter,
        "ring_width_cm": ring.width,
        "inner_radius_cm": ring.inner_radius,
        "boundary_margin_cm": ring.margin,
        "boundary_angle_deg": math.degrees(ring.theta0) % 360.0,
        "confirm_s": float(options["confirm_s"]),
        "quality_threshold": float(options["quality_threshold"]),
        "bin_seconds": float(options["bin_seconds"]),
    }
    return summary


def _metadata(unit: Mapping[str, Any], options: Mapping[str, Any],
              ring: _Ring, detection_settings: Mapping[str, Any]) -> dict:
    return {
        "mode": KEY,
        "coordinate_unit": "cm",
        "scale": "affine A = Rα·diag(D/(2rx), D/(2ry))·Rαᵀ about the arena "
                 "centre; D is the declared outer diameter and y still points "
                 "down the image",
        "arena": {"center": [float(ring.center[0]), float(ring.center[1])],
                  "radii": [float(ring.radii[0]), float(ring.radii[1])],
                  "angle": float(ring.angle)},
        "ring": {"outer_radius_cm": ring.radius,
                 "inner_radius_cm": ring.inner_radius,
                 "width_cm": ring.width,
                 "boundary_margin_cm": ring.margin},
        "boundary_direction": {
            "source": "geometry.open_boundary pixel point",
            "angle_deg": math.degrees(ring.theta0) % 360.0,
        },
        "zone_order": "clockwise from the open-boundary ray: O1, C1, O2, C2 "
                      "(image y points down, so increasing θ is clockwise)",
        "zone_order_keys": list(ZONE_KEYS),
        "occupancy_denominator": {
            "open_time_percent_total":
                "total_time_s (whole animal-window exposure, includes missing)",
            "open_time_percent_known": "open_time_s + closed_time_s",
            "known_zone_time_s":
                "open_time_s + closed_time_s + boundary_time_s + outside_time_s",
        },
        "entry_rule": "zone_entry needs a later real observation at least "
                      "confirm_s after the candidate start and only when the "
                      "old and new zones are adjacent on the ring",
        "initial_zone_rule": "the first valid main zone becomes accepted "
                            "without an entry event",
        "boundary_rule": "boundary exposure clears the candidate but keeps the "
                         "accepted zone",
        "outside_unknown_rule": "outside/unknown clears the accepted zone and "
                                "all candidates; regaining a main zone is a "
                                "relocation, not an entry",
        "unknown_meaning": "no qualified body (missing exposure or quality "
                           "below the statistics threshold); unknown is not "
                           "the closed arm",
        "missing_rule": "missing exposure is counted as unknown only; "
                        "observed/missing keep raw detection availability",
        "motion_rule": "motion uses only consecutive completed segments "
                       "between qualified in-ring positions "
                       "(open/closed/boundary)",
        "mean_speed_denominator": "measurable completed motion duration",
        "quality_threshold": float(options["quality_threshold"]),
        "detection_quality_threshold":
            (detection_settings or {}).get("quality_threshold"),
    }


# -- 单只动物 -----------------------------------------------------------

def process_unit(unit: Mapping[str, Any],
                 captured: Sequence[Mapping[str, Any]],
                 video_window: timeline.Window,
                 first_seen: float | None) -> dict:
    """环上占用、相邻确认进入、统一运动事实、汇总、分箱与七类表。"""

    options = unit["options"]
    geometry = unit.get("geometry") or {}
    ring = _Ring(geometry, options)
    threshold = float(options["quality_threshold"])

    window = timeline.resolve_unit_window(
        unit.get("window"), video_window, first_seen)
    rows = timeline.clip_rows(captured, window)
    method = (unit.get("detection") or {}).get("bg_method")
    for row in rows:
        row.setdefault("detection_method", method)
        _annotate(row, ring, threshold)

    tracking = _track(rows, window, options, unit)
    segments = timeline.motion_segments(rows, window, x_key="motion_x",
                                        y_key="motion_y",
                                        require_continuity=True)
    timeline.attach_motion(rows, segments)

    summary = _summary(unit, rows, segments, window, options, ring, tracking)
    seconds = summary["zone_seconds"]
    metadata = _metadata(unit, options, ring, unit.get("detection") or {})
    warning = None
    if not window.known:
        warning = ("未检测到动物，无法确定相对时间窗："
                   + str(unit.get("name") or unit.get("id")))
    return {
        "rows": rows,
        "segments": segments,
        "events": tracking["events"],
        "bins": _bin_rows(rows, segments, window, options, unit),
        "summary": summary,
        "metadata": metadata,
        "zones": _zone_table(unit, seconds, tracking["events"]),
        "warning": warning,
    }


# -- 捕获与分析入口 -----------------------------------------------------

def capture(context: Any, units: Sequence[Mapping[str, Any]]) -> dict:
    """逐帧检测：正式分析与行为预览共用同一条捕获路径。"""

    job = context.job
    asset = job.get("asset") or {}
    metadata = asset.get("metadata") or {}
    source = context.source
    if source is None:
        window = job.get("window") or {}
        source = media.VideoSource(
            asset.get("path"),
            start_s=float(window.get("start_s") or 0.0),
            end_s=window.get("end_s"),
            orientation=int(job.get("orientation", 0) or 0),
            fps_override=job.get("fps_override"),
            sample_fps=job.get("sample_fps"),
            source_fps=metadata.get("fps"),
            decode_images=True,
        )

    detectors: dict[Any, Any] = {}
    regions: dict[Any, np.ndarray] = {}
    backgrounds: dict[tuple, Any] = {}
    captured: dict[Any, list[dict]] = {unit["id"]: [] for unit in units}
    unit_ids = [unit["id"] for unit in units]
    run_key = (job.get("run_id"), job.get("job_id"), KEY)
    progress_span = float(metadata.get("duration_s") or 0.0)

    for frame in source.iter_frames():
        if context.cancelled():
            raise ValidationError("分析已取消", "job")
        image = frame.image
        if image is None:
            continue
        height, width = image.shape[:2]
        for unit in units:
            unit_id = unit["id"]
            if unit_id not in detectors:
                settings = unit.get("detection") or {}
                signature = tuple(settings.get(name) for name in
                                  detection.FOREGROUND_KEYS)
                if signature not in backgrounds:
                    backgrounds[signature] = detection.prepare_background(
                        settings, asset, job,
                        int(job.get("orientation", 0) or 0), width, height)
                detectors[unit_id] = detection.shared_detector(
                    run_key, settings, background=backgrounds[signature])
                regions[unit_id] = _region_for(unit, unit.get("options") or {},
                                               width, height)
            detector = detectors[unit_id]
            result = observation.track(
                run_key, unit_id,
                image=image, region=regions[unit_id],
                settings=unit["detection"],
                signature=observation.signature_of(
                    unit.get("geometry") or {}, unit["detection"], False),
                assay=KEY, frame_index=frame.index, time_s=frame.time_s,
                foreground=detector.foreground(image),
                retain_units=unit_ids,
            )
            captured[unit_id].append(observation_row(unit, frame, result))
        context.progress(min(0.9, 0.1 + 0.8 * max(0.0, float(frame.time_s)) /
                             max(1e-9, progress_span or 1.0)))
    return captured


def _video_metadata(job: Mapping[str, Any], asset: Mapping[str, Any],
                    units: int) -> dict:
    """引擎侧补充的视频事实，与实验 metadata 并列而不是覆盖单位记录。"""

    metadata = asset.get("metadata") or {}
    window = job.get("window") or {}
    return {
        "asset_id": asset.get("id"),
        "name": asset.get("name"),
        "duration_s": metadata.get("duration_s"),
        "fps": metadata.get("fps"),
        "units": units,
        "orientation": int(job.get("orientation", 0) or 0),
        "fps_override": job.get("fps_override"),
        "sample_fps": job.get("sample_fps"),
        "window": {"start_s": window.get("start_s"),
                   "end_s": window.get("end_s")},
    }


def analyze(context: Any) -> AnalysisResult:
    job = context.job
    units = list(job.get("units") or [])
    if not units:
        raise ValidationError("草稿还没有小鼠", "units")
    for unit in units:
        issues = validate_unit(unit.get("options") or {},
                               unit.get("geometry") or {}, complete=True)
        if issues:
            raise ValidationError(issues[0]["message"], issues[0]["field"])

    captured = capture(context, units)
    if context.cancelled():
        raise ValidationError("分析已取消", "job")
    asset = job.get("asset") or {}
    metadata = asset.get("metadata") or {}
    all_rows = [row for rows in captured.values() for row in rows]
    video_window = timeline.resolve_video_window(
        job.get("window"), all_rows, metadata.get("duration_s"))
    first_seen = {
        unit["id"]: timeline.first_seen_in_window(
            KEY, captured[unit["id"]], video_window, unit.get("geometry") or {})
        for unit in units
    }

    observations: list[dict] = []
    summary: list[dict] = []
    events: list[dict] = []
    bins: list[dict] = []
    zones: list[dict] = []
    warnings: list[str] = []
    unit_metadata: dict[str, dict] = {}
    for unit in units:
        if context.cancelled():
            raise ValidationError("分析已取消", "job")
        facts = process_unit(unit, captured[unit["id"]], video_window,
                             first_seen[unit["id"]])
        observations.extend(facts["rows"])
        summary.append(facts["summary"])
        events.extend(facts["events"])
        bins.extend(facts["bins"])
        zones.extend(facts["zones"])
        unit_metadata[unit["id"]] = facts["metadata"]
        if facts["warning"]:
            warnings.append(facts["warning"])

    total = sum(float(row.get("total_time_s") or 0.0) for row in summary)
    observed = sum(float(row.get("observed_time_s") or 0.0) for row in summary)
    return AnalysisResult(
        observations=timeline.sort_rows(observations),
        summary=summary,
        events=events,
        bins=bins,
        tables={"zones": zones},
        warnings=warnings,
        metadata={
            "assay": KEY,
            "coordinate_unit": "cm",
            "video": _video_metadata(job, asset, len(units)),
            "coverage": {
                "units": len(units),
                "total_time_s": total,
                "observed_time_s": observed,
                "missing_time_s": max(0.0, total - observed),
            },
            "algorithm_version": job.get("algorithm_version")
                                 or ALGORITHM_VERSION,
            # 每只动物保留自己的尺度、分母与来源，导出前不再整体替换。
            "units": unit_metadata,
            "warnings": warnings,
        },
    )


def preview_analysis(*, asset: Mapping[str, Any], video: Mapping[str, Any],
                     unit: Mapping[str, Any], start_s: float,
                     end_s: float, time_s: float | None = None) -> dict:
    """行为预览：对实际片段跑同一捕获→处理核心，返回片段汇总。"""

    from ..schema import AnalysisContext, new_id

    options = unit.get("options") or {}
    issues = validate_unit(options, unit.get("geometry") or {}, complete=True)
    if issues:
        raise ValidationError(issues[0]["message"], issues[0]["field"])

    metadata = asset.get("metadata") or {}
    window = {"start_s": float(start_s), "end_s": float(end_s)}
    source = media.VideoSource(
        asset.get("path"),
        start_s=float(start_s),
        end_s=float(end_s),
        orientation=int(video.get("orientation", 0) or 0),
        fps_override=video.get("fps_override"),
        sample_fps=video.get("sample_fps"),
        source_fps=metadata.get("fps"),
        decode_images=True,
    )
    job = {
        "assay": KEY,
        "run_id": new_id(),
        "job_id": "preview",
        "project_id": None,
        "window": window,
        "orientation": int(video.get("orientation", 0) or 0),
        "fps_override": video.get("fps_override"),
        "sample_fps": video.get("sample_fps"),
        "asset": dict(asset),
        "units": [dict(unit)],
        "style": {},
    }
    context = AnalysisContext(job=job, source=source)
    captured = capture(context, [unit])[unit["id"]]
    video_window = timeline.resolve_video_window(
        window, captured, metadata.get("duration_s"))
    first_seen = timeline.first_seen_in_window(
        KEY, captured, video_window, unit.get("geometry") or {})
    facts = process_unit(unit, captured, video_window, first_seen)
    summary = facts["summary"]
    return {
        "mode": KEY,
        "start_s": float(start_s),
        "end_s": float(end_s),
        "window_start_s": summary["analysis_window_start_s"],
        "window_end_s": summary["analysis_window_end_s"],
        "total_time_s": summary["total_time_s"],
        "observed_time_s": summary["observed_time_s"],
        "missing_time_s": summary["missing_time_s"],
        "open_time_s": summary["open_time_s"],
        "closed_time_s": summary["closed_time_s"],
        "boundary_time_s": summary["boundary_time_s"],
        "outside_time_s": summary["outside_time_s"],
        "unknown_time_s": summary["unknown_time_s"],
        "open_time_percent_total": summary["open_time_percent_total"],
        "open_time_percent_known": summary["open_time_percent_known"],
        "initial_zone": summary["initial_zone"],
        "open_entries": summary["open_entries"],
        "closed_entries": summary["closed_entries"],
        "first_open_observation_latency_s":
            summary["first_open_observation_latency_s"],
        "first_open_entry_latency_s": summary["first_open_entry_latency_s"],
        "first_open_entry_censored": summary["first_open_entry_censored"],
        "total_distance_cm": summary["total_distance_cm"],
        "mean_speed_cm_s": summary["mean_speed_cm_s"],
        "rows": len(facts["rows"]),
        "events": len(facts["events"]),
        "bins": len(facts["bins"]),
    }


def validate(job: Any) -> list[dict]:
    issues: list[dict] = []
    for unit in job.get("units") or []:
        prefix = f"units[{unit.get('id')}]"
        for issue in validate_unit(unit.get("options") or {},
                                   unit.get("geometry") or {},
                                   complete=True):
            entry = dict(issue)
            entry["field"] = f"{prefix}.{entry['field']}"
            issues.append(entry)
    return issues
