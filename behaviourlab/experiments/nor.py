"""NOR 新物体识别：四角透视、双物体接近与头向归属、探索与阶段指数。

场地四角映射到 [0,W]×[0,H] cm，身体、鼻尖、尾根、对象中心与自定义探索
范围共用同一变换。探索需要同时满足接近与速度门槛；开启头向时还要朝向
合格，头向不合格只退回距离判定，不当成"背向"证据。
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .. import detection, geometry as geo, media, observation, timeline
from ..schema import AnalysisResult, ValidationError, is_number
from .base import geometry_field, metric, parameter

KEY = "nor"

DEFAULTS = {
    "arena_width_cm": 50.0,
    "arena_height_cm": 50.0,
    "center_ratio_x": 0.5,
    "center_ratio_y": 0.5,
    "phase": "recognition",
    "interaction_radius_cm": 4.0,
    "exploration_max_speed_cm_s": 12.0,
    "movement_threshold_cm_s": 1.0,
    "use_head_direction": True,
    "head_angle_threshold_deg": 50.0,
    "head_confidence_threshold": 0.35,
    "min_valid_exploration_s": 20.0,
    "bin_seconds": 300.0,
}

OPTION_KEYS = tuple(DEFAULTS)
PHASES = ("recognition", "familiarization")
ROLES = ("novel", "familiar")
OBJECT_FIELDS = ("distance_cm", "nose_distance_cm", "zone_clearance_cm",
                 "nose_zone_clearance_cm", "head_angle_deg", "head_aligned",
                 "near", "exploring")
TIME_EPS = 1e-9
NEAR_EPS = 1e-8

DETECTION_DEFAULTS: dict = {}

FIRST_SEEN = "arena"

SPEC = {
    "key": KEY,
    "label": "新物体识别实验 / NOR",
    "short": "NOR",
    "parameters": [
        parameter("arena_width_cm", "场地宽度", "number", 50.0, unit="cm",
                  minimum=0, step="any"),
        parameter("arena_height_cm", "场地长度", "number", 50.0, unit="cm",
                  minimum=0, step="any"),
        parameter("phase", "实验阶段", "select", "recognition",
                  choices=[{"value": "recognition", "label": "识别期"},
                           {"value": "familiarization", "label": "熟悉期"}]),
        parameter("center_ratio_x", "中心区宽度比例", "number", 0.5,
                  minimum=0, maximum=1, step="any", section="实验参数"),
        parameter("center_ratio_y", "中心区长度比例", "number", 0.5,
                  minimum=0, maximum=1, step="any", section="实验参数"),
        parameter("interaction_radius_cm", "默认探索半径", "number", 4.0,
                  unit="cm", minimum=0, step="any", section="实验参数"),
        parameter("exploration_max_speed_cm_s", "探索最大速度", "number", 12.0,
                  unit="cm/s", minimum=0, step="any", section="实验参数"),
        parameter("use_head_direction", "使用头向辅助", "checkbox", True,
                  section="实验参数"),
        parameter("head_angle_threshold_deg", "头向夹角上限", "number", 50.0,
                  unit="°", minimum=0, maximum=180, step="any",
                  section="实验参数"),
        parameter("head_confidence_threshold", "头向质量下限", "number", 0.35,
                  minimum=0, maximum=1, step="any", section="实验参数"),
        parameter("movement_threshold_cm_s", "移动速度阈值", "number", 1.0,
                  unit="cm/s", minimum=0, step="any", section="实验参数"),
        parameter("min_valid_exploration_s", "最少探索时间", "number", 20.0,
                  unit="s", minimum=0, step="any", section="实验参数"),
        parameter("bin_seconds", "时间箱长度", "number", 300.0, unit="s",
                  minimum=0, step="any", section="实验参数"),
    ],
    "geometry": [
        geometry_field("arena", "场地四角（左上、右上、右下、左下）", "quad",
                       count=4),
        geometry_field("object1", "物体 1 中心", "point", count=1,
                       role="object"),
        geometry_field("object2", "物体 2 中心", "point", count=1,
                       role="object"),
        geometry_field("explore1", "物体 1 探索范围", "polygon", min_points=3,
                       optional=True, role="explore"),
        geometry_field("explore2", "物体 2 探索范围", "polygon", min_points=3,
                       optional=True, role="explore"),
        geometry_field("mask", "检测遮罩", "polygon", min_points=3,
                       optional=True),
    ],
    "metrics": [
        metric("total_time_s", "窗口时间", "s"),
        metric("observed_time_s", "有效检测时间", "s"),
        metric("missing_time_s", "缺失检测时间", "s"),
        metric("detected_rate_percent", "检测率", "%"),
        metric("measurable_motion_time_s", "可测量运动时间", "s"),
        metric("total_distance_cm", "总距离", "cm"),
        metric("mean_speed_cm_s", "平均速度", "cm/s"),
        metric("center_distance_cm", "中心区距离", "cm"),
        metric("outer_distance_cm", "外围区距离", "cm"),
        metric("outside_distance_cm", "场地外距离", "cm"),
        metric("observed_total_distance_cm", "全部检测路程", "cm"),
        metric("max_speed_cm_s", "最大速度", "cm/s"),
        metric("moving_time_s", "移动时间", "s"),
        metric("immobility_time_s", "静止时间", "s"),
        metric("center_time_s", "中心区时间", "s"),
        metric("center_percent", "中心区比例", "%"),
        metric("center_entries", "进入中心次数"),
        metric("outside_time_s", "场地外时间", "s"),
        metric("outside_percent", "场地外比例", "%"),
        metric("center_latency_s", "首次进入中心潜伏期", "s"),
        metric("object_1_exploration_s", "物体一探索时间", "s"),
        metric("object_2_exploration_s", "物体二探索时间", "s"),
        metric("object_1_bouts", "物体一探索次数"),
        metric("object_2_bouts", "物体二探索次数"),
        metric("object_1_latency_s", "首次探索物体一潜伏期", "s"),
        metric("object_2_latency_s", "首次探索物体二潜伏期", "s"),
        metric("object_1_zone_time_s", "物体一接近范围时间", "s"),
        metric("object_2_zone_time_s", "物体二接近范围时间", "s"),
        metric("object_1_zone_entries", "进入物体一范围次数"),
        metric("object_2_zone_entries", "进入物体二范围次数"),
        metric("total_object_exploration_s", "总探索时间", "s"),
        metric("first_explored_object", "首次探索物体"),
        metric("first_explored_role", "首次探索物体角色"),
        metric("novel_exploration_s", "新物体探索时间", "s"),
        metric("familiar_exploration_s", "熟悉物体探索时间", "s"),
        metric("discrimination_index", "辨别指数"),
        metric("preference_index_percent", "新物体偏好指数", "%"),
        metric("object_balance_index", "熟悉期物体平衡指数"),
        metric("head_supported_exploration_s", "头向支持探索时间", "s"),
        metric("fallback_exploration_s", "距离回退探索时间", "s"),
        metric("proximity_exploration_s", "距离判定探索时间", "s"),
        metric("head_estimated_rate_percent", "头向估计率", "%"),
        metric("head_confident_rate_percent", "头向合格率", "%"),
        metric("meets_min_exploration", "达到最少探索时间"),
    ],
    "template_keys": ["arena", "arena_width_cm", "arena_height_cm"],
}


# -- 参数与几何校验 -----------------------------------------------------

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


def _points(shape: Mapping[str, Any] | None) -> list[tuple[float, float]]:
    if not isinstance(shape, Mapping):
        return []
    result: list[tuple[float, float]] = []
    for item in shape.get("points") or []:
        point = _point(item)
        if point is None:
            return []
        result.append(point)
    return result


def _orientation(a: Sequence[float], b: Sequence[float],
                 c: Sequence[float]) -> int:
    value = ((float(b[0]) - float(a[0])) * (float(c[1]) - float(a[1]))
             - (float(b[1]) - float(a[1])) * (float(c[0]) - float(a[0])))
    if abs(value) <= 1e-12:
        return 0
    return 1 if value > 0 else -1


def _between(a: Sequence[float], b: Sequence[float],
             c: Sequence[float]) -> bool:
    return (min(a[0], b[0]) - 1e-12 <= c[0] <= max(a[0], b[0]) + 1e-12
            and min(a[1], b[1]) - 1e-12 <= c[1] <= max(a[1], b[1]) + 1e-12)


def _segments_meet(a: Sequence[float], b: Sequence[float],
                   c: Sequence[float], d: Sequence[float]) -> bool:
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    if o1 != o2 and o3 != o4:
        return True
    # 共线时只判"另一点落在本段内"，共线但分离的线段不算相交。
    if o1 == 0 and _between(a, b, c):
        return True
    if o2 == 0 and _between(a, b, d):
        return True
    if o3 == 0 and _between(c, d, a):
        return True
    if o4 == 0 and _between(c, d, b):
        return True
    return False


def _polygon_self_intersects(points: Sequence[Sequence[float]]) -> bool:
    count = len(points)
    for first in range(count):
        a, b = points[first], points[(first + 1) % count]
        for second in range(first + 1, count):
            if second in ((first + 1) % count, (first - 1) % count):
                continue
            c, d = points[second], points[(second + 1) % count]
            if _segments_meet(a, b, c, d):
                return True
    return False


def _zone_points(zone: Any) -> list[tuple[float, float]] | None:
    """自定义探索范围：≥3 点、正面积、不自交；不合格返回 None。"""

    if not isinstance(zone, Mapping):
        return None
    points = _points(zone)
    if len(points) < 3 or len(zone.get("points") or []) < 3:
        return None
    if abs(geo.polygon_area(points)) <= 1e-9:
        return None
    if _polygon_self_intersects(points):
        return None
    return points


def validate_unit(options: Mapping[str, Any] | None,
                  geometry: Mapping[str, Any] | None,
                  provided: Iterable[str] | None = None, *,
                  complete: bool = False) -> list[dict]:
    """只校验显式给出的值与已画完的几何；未完成草稿仍可保存。"""

    values = options if isinstance(options, Mapping) else {}
    provided = set(OPTION_KEYS if provided is None else provided)
    issues: list[dict] = []

    def number(key: str) -> float | None:
        value = _finite(values.get(key))
        if value is None:
            issues.append({"field": f"options.{key}", "message": "必须是有限数值"})
        return value

    width = number("arena_width_cm") if "arena_width_cm" in provided else None
    height = number("arena_height_cm") if "arena_height_cm" in provided else None
    if width is not None and width <= 0:
        issues.append({"field": "options.arena_width_cm",
                       "message": "场地宽度必须 > 0"})
    if height is not None and height <= 0:
        issues.append({"field": "options.arena_height_cm",
                       "message": "场地长度必须 > 0"})

    for key in ("center_ratio_x", "center_ratio_y"):
        if key not in provided:
            continue
        ratio = number(key)
        if ratio is not None and not 0 < ratio <= 1:
            issues.append({"field": f"options.{key}",
                           "message": "中心比例必须 > 0 且 ≤ 1"})

    phase = values.get("phase") if values.get("phase") in PHASES \
        else DEFAULTS["phase"]
    if "phase" in provided and values.get("phase") not in PHASES:
        issues.append({"field": "options.phase", "message": "未知取值"})

    if "interaction_radius_cm" in provided:
        radius = number("interaction_radius_cm")
        if radius is not None and radius <= 0:
            issues.append({"field": "options.interaction_radius_cm",
                           "message": "探索半径必须 > 0"})

    if "exploration_max_speed_cm_s" in provided:
        speed = number("exploration_max_speed_cm_s")
        if speed is not None and speed < 0:
            issues.append({"field": "options.exploration_max_speed_cm_s",
                           "message": "探索最大速度必须 ≥ 0"})

    if "movement_threshold_cm_s" in provided:
        threshold = number("movement_threshold_cm_s")
        if threshold is not None and threshold < 0:
            issues.append({"field": "options.movement_threshold_cm_s",
                           "message": "速度阈值必须 ≥ 0"})

    if "use_head_direction" in provided:
        switch = values.get("use_head_direction")
        if not isinstance(switch, bool):
            issues.append({"field": "options.use_head_direction",
                           "message": "必须是布尔值"})

    if "head_angle_threshold_deg" in provided:
        angle = number("head_angle_threshold_deg")
        if angle is not None and not 0 <= angle <= 180:
            issues.append({"field": "options.head_angle_threshold_deg",
                           "message": "头向夹角上限必须在 0 到 180 度之间"})

    if "head_confidence_threshold" in provided:
        quality = number("head_confidence_threshold")
        if quality is not None and not 0 <= quality <= 1:
            issues.append({"field": "options.head_confidence_threshold",
                           "message": "头向质量下限必须在 0 到 1 之间"})

    if "min_valid_exploration_s" in provided:
        minimum = number("min_valid_exploration_s")
        if minimum is not None and minimum < 0:
            issues.append({"field": "options.min_valid_exploration_s",
                           "message": "最少探索时间必须 ≥ 0"})

    if "bin_seconds" in provided:
        bin_width = number("bin_seconds")
        if bin_width is not None and bin_width <= 0:
            issues.append({"field": "options.bin_seconds",
                           "message": "时间箱长度必须 > 0"})

    issues.extend(validate_arena(geometry, require=complete))
    issues.extend(validate_objects(geometry, phase=phase, require=complete))
    return issues


def _is_convex(points: Sequence[Sequence[float]]) -> bool:
    signs = []
    for index in range(4):
        value = _orientation(points[index], points[(index + 1) % 4],
                             points[(index + 2) % 4])
        if value == 0:
            return False
        signs.append(value)
    return all(sign == signs[0] for sign in signs)


def validate_arena(geometry: Mapping[str, Any] | None,
                   *, require: bool = False) -> list[dict]:
    """场地必须是顺序四角 quad：四个有限点、凸、不交叉。"""

    shape = (geometry or {}).get("arena")
    if shape is None:
        if require:
            return [{"field": "geometry.arena", "message": "缺少场地四角"}]
        return []
    if not isinstance(shape, Mapping) or shape.get("type") != "quad":
        return [{"field": "geometry.arena", "message": "场地必须是四角四边形"}]
    if len(shape.get("points") or []) != 4:
        return [{"field": "geometry.arena", "message": "场地需要 4 个有限二维点"}]
    points = _points(shape)
    if len(points) != 4:
        return [{"field": "geometry.arena", "message": "场地需要 4 个有限二维点"}]
    if geo.quad_is_crossed(points) or not _is_convex(points):
        return [{"field": "geometry.arena",
                 "message": "场地四角必须凸且不交叉"}]
    return []


def object_entries(raw: Any) -> list[Any] | None:
    """物体数组；兼容编辑器直接产出的两点形状。"""

    if isinstance(raw, Mapping):
        points = _points(raw)
        if len(points) != 2:
            return None
        defaults = (("object_1", "物体 1", "familiar"),
                    ("object_2", "物体 2", "novel"))
        return [{"id": identity, "name": name, "role": role, "center": point,
                 "zone": None}
                for point, (identity, name, role) in zip(points, defaults)]
    if isinstance(raw, (list, tuple)):
        return list(raw)
    return None


def validate_objects(geometry: Mapping[str, Any] | None, *,
                     phase: str = DEFAULTS["phase"],
                     require: bool = False) -> list[dict]:
    """恰好两个物体：唯一 ID、显示名、角色、中心与可选探索范围。"""

    raw = (geometry or {}).get("objects")
    if raw is None:
        if require:
            return [{"field": "geometry.objects", "message": "缺少两个物体"}]
        return []
    entries = object_entries(raw)
    if entries is None:
        return [{"field": "geometry.objects", "message": "物体必须是两项数组"}]
    issues: list[dict] = []
    if len(entries) != 2:
        issues.append({"field": "geometry.objects",
                       "message": "必须正好两个物体"})
    identities: set[str] = set()
    roles: list[str] = []
    for index, item in enumerate(entries):
        prefix = f"geometry.objects[{index}]"
        if not isinstance(item, Mapping):
            issues.append({"field": prefix, "message": "物体必须是对象"})
            continue
        identity = item.get("id")
        if not isinstance(identity, str) or not identity.strip():
            issues.append({"field": f"{prefix}.id",
                           "message": "物体 ID 必须是非空字符串"})
        elif identity in identities:
            issues.append({"field": f"{prefix}.id",
                           "message": "物体 ID 必须唯一"})
        else:
            identities.add(identity)
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            issues.append({"field": f"{prefix}.name",
                           "message": "物体显示名不能为空"})
        role = item.get("role")
        if role not in ROLES:
            issues.append({"field": f"{prefix}.role",
                           "message": "物体角色必须是 familiar 或 novel"})
        else:
            roles.append(role)
        center = item.get("center")
        if center is None:
            # 未画完的草稿允许先存另一半；最终校验/开始任务才要求中心。
            if require:
                issues.append({"field": f"{prefix}.center",
                               "message": "缺少物体中心"})
        elif _point(center) is None:
            issues.append({"field": f"{prefix}.center",
                           "message": "物体中心必须是两个有限数值"})
        zone = item.get("zone")
        if zone is not None and _zone_points(zone) is None:
            issues.append({"field": f"{prefix}.zone",
                           "message": "探索范围至少 3 个有限点、正面积且不自交"})
    if phase == "recognition" and len(roles) == 2 and roles.count("novel") != 1:
        issues.append({"field": "geometry.objects",
                       "message": "识别期必须恰好一个新奇和一个熟悉物体"})
    return issues


def objects_of(geometry: Mapping[str, Any]) -> list[dict]:
    """规范化后的两个物体（像素坐标）；缺字段按默认补全。"""

    entries = object_entries((geometry or {}).get("objects")) or []
    objects: list[dict] = []
    for index, item in enumerate(entries):
        center = (_point(item.get("center"))
                  if isinstance(item, Mapping) else None)
        if center is None:
            raise ValidationError("缺少物体中心",
                                  f"geometry.objects[{index}].center")
        role = item.get("role")
        identity = item.get("id")
        name = item.get("name")
        objects.append({
            "index": index + 1,
            "id": identity if isinstance(identity, str) and identity
            else f"object_{index + 1}",
            "name": name if isinstance(name, str) and name
            else f"物体 {index + 1}",
            "role": role if role in ROLES else ("familiar", "novel")[index % 2],
            "center": center,
            "zone": _zone_points(item.get("zone")),
        })
    if len(objects) != 2:
        raise ValidationError("必须正好两个物体", "geometry.objects")
    return objects


# -- 坐标与几何 ---------------------------------------------------------

def arena_matrix(geometry: Mapping[str, Any],
                 options: Mapping[str, Any]) -> list[float]:
    issues = validate_arena(geometry)
    if issues:
        raise ValidationError(issues[0]["message"], issues[0]["field"])
    points = _points((geometry or {}).get("arena"))
    return geo.homography(points, float(options["arena_width_cm"]),
                          float(options["arena_height_cm"]))


def to_cm(matrix: Sequence[float],
          point: Sequence[float]) -> tuple[float, float]:
    return geo.apply_homography(matrix, point)


def center_rect(width: float, height: float, ratio_x: float,
                ratio_y: float) -> tuple[float, float, float, float]:
    return (width * (1.0 - ratio_x) / 2.0, height * (1.0 - ratio_y) / 2.0,
            width * (1.0 + ratio_x) / 2.0, height * (1.0 + ratio_y) / 2.0)


def zone_of(x: float, y: float, width: float, height: float,
            rect: tuple[float, float, float, float]) -> str:
    """Center / Outer / Outside；中心矩形边界计内部。"""

    if x < 0 or y < 0 or x > width or y > height:
        return "Outside"
    left, top, right, bottom = rect
    if left <= x <= right and top <= y <= bottom:
        return "Center"
    return "Outer"


def _clip_fraction(p0: Sequence[float], p1: Sequence[float],
                   rect: tuple[float, float, float, float]) -> float:
    """直线参数 u∈[0,1] 与矩形求交的长度/时间比例。"""

    x0, y0, x1, y1 = rect
    dx = float(p1[0]) - float(p0[0])
    dy = float(p1[1]) - float(p0[1])
    low, high = 0.0, 1.0
    for delta, offset in ((-dx, float(p0[0]) - x0), (dx, x1 - float(p0[0])),
                          (-dy, float(p0[1]) - y0), (dy, y1 - float(p0[1]))):
        if abs(delta) < 1e-12:
            if offset < 0:
                return 0.0
            continue
        ratio = offset / delta
        if delta < 0:
            if ratio > high:
                return 0.0
            low = max(low, ratio)
        else:
            if ratio < low:
                return 0.0
            high = min(high, ratio)
    return max(0.0, high - low)


def _point_polygon_distance(point: Sequence[float],
                            polygon: Sequence[Sequence[float]]) -> float:
    if geo.point_in_polygon(point, polygon):
        return 0.0
    count = len(polygon)
    best = float("inf")
    for index in range(count):
        best = min(best, geo.distance_to_segment(
            point, polygon[index], polygon[(index + 1) % count]))
    return best


def _polygons_intersect(first: Sequence[Sequence[float]],
                        second: Sequence[Sequence[float]]) -> bool:
    n, m = len(first), len(second)
    for i in range(n):
        a, b = first[i], first[(i + 1) % n]
        for j in range(m):
            if _segments_meet(a, b, second[j], second[(j + 1) % m]):
                return True
    return (geo.point_in_polygon(first[0], second)
            or geo.point_in_polygon(second[0], first))


def _polygons_distance(first: Sequence[Sequence[float]],
                       second: Sequence[Sequence[float]]) -> float:
    if _polygons_intersect(first, second):
        return 0.0
    n, m = len(first), len(second)
    best = float("inf")
    for i in range(n):
        a, b = first[i], first[(i + 1) % n]
        for j in range(m):
            c, d = second[j], second[(j + 1) % m]
            best = min(best, geo.distance_to_segment(a, c, d),
                       geo.distance_to_segment(c, a, b))
    return best


def _body_polygon(row: Mapping[str, Any],
                  matrix: Sequence[float]) -> list[tuple[float, float]] | None:
    """身体轮廓（cm）；没有可用轮廓时返回 None，由中心点代替。"""

    contour = row.get("contour") or []
    if len(contour) < 3:
        return None
    points: list[tuple[float, float]] = []
    for item in contour:
        point = _point(item)
        if point is None:
            return None
        try:
            points.append(to_cm(matrix, point))
        except ValidationError:
            return None
    if abs(geo.polygon_area(points)) <= 1e-9:
        return None
    return points


def _body_distance(body: Sequence[Sequence[float]] | None,
                   center: Sequence[float],
                   target: Sequence[float]) -> float:
    if body is None:
        return math.hypot(float(center[0]) - float(target[0]),
                          float(center[1]) - float(target[1]))
    return _point_polygon_distance(target, body)


# -- 观测行 -------------------------------------------------------------

def _unit_dict(unit: Mapping[str, Any]) -> dict:
    return {"id": unit.get("id"), "name": unit.get("name"),
            "animal_id": unit.get("animal_id")}


def observation_row(unit: Mapping[str, Any], frame: Any,
                    result: Any) -> dict:
    has_head = result.nose is not None and result.tail_base is not None
    return timeline.base_row(
        _unit_dict(unit), frame, result,
        extra={"detection_method": (unit.get("detection") or {}).get("bg_method"),
               "head_method": "classical_geometry" if has_head else None},
    )


def _region_for(unit: Mapping[str, Any], options: Mapping[str, Any],
                width: int, height: int) -> np.ndarray:
    geometry = unit.get("geometry") or {}
    region, _ = detection.geometry_region(geometry, KEY, options, width, height)
    return region


def _blank_objects(row: dict) -> None:
    for index in (1, 2):
        for field in OBJECT_FIELDS:
            row[f"object_{index}_{field}"] = None


def _annotate(row: dict, matrix: Sequence[float], width: float,
              height: float, rect: tuple[float, float, float, float],
              options: Mapping[str, Any],
              objects: Sequence[Mapping[str, Any]]) -> None:
    """物理坐标、区域、两物体距离/接近与头向夹角；缺失行全部 null。"""

    row["coordinate_unit"] = "cm"
    row["head_direction_estimated"] = None
    row["head_direction_confident"] = None
    row["head_direction_source"] = None
    row["exploration_mode"] = None
    row["current_object"] = None
    row["current_object_id"] = None
    row["zone_label"] = "Unknown"
    _blank_objects(row)
    if not row.get("detected") or row.get("x_px") is None:
        row.update({"x": None, "y": None, "x_cm": None, "y_cm": None,
                    "nose_x_cm": None, "nose_y_cm": None,
                    "tail_x_cm": None, "tail_y_cm": None,
                    "in_arena": None, "in_center": None, "in_outer": None,
                    "zone": "Unknown", "body_has_contour": False})
        return

    x_cm, y_cm = to_cm(matrix, (row["x_px"], row["y_px"]))
    row["x"] = x_cm
    row["y"] = y_cm
    row["x_cm"] = x_cm
    row["y_cm"] = y_cm
    zone = zone_of(x_cm, y_cm, width, height, rect)
    row["zone"] = zone
    row["in_arena"] = zone != "Outside"
    row["in_center"] = zone == "Center"
    row["in_outer"] = zone == "Outer"

    nose = None
    tail = None
    for prefix, x_key, y_key in (("nose", "nose_x_px", "nose_y_px"),
                                 ("tail", "tail_x_px", "tail_y_px")):
        if row.get(x_key) is None or row.get(y_key) is None:
            row[f"{prefix}_x_cm"] = None
            row[f"{prefix}_y_cm"] = None
            continue
        px, py = to_cm(matrix, (row[x_key], row[y_key]))
        row[f"{prefix}_x_cm"] = px
        row[f"{prefix}_y_cm"] = py
        if prefix == "nose":
            nose = (px, py)
        else:
            tail = (px, py)

    body = _body_polygon(row, matrix)
    row["body_has_contour"] = body is not None

    head = None
    estimated = False
    confident = False
    if nose is not None and tail is not None:
        hx, hy = nose[0] - tail[0], nose[1] - tail[1]
        norm = math.hypot(hx, hy)
        if norm > 0:
            estimated = True
            head = (hx / norm, hy / norm)
            quality = row.get("head_quality")
            confident = (is_number(quality)
                         and float(quality)
                         >= float(options["head_confidence_threshold"]))
    row["head_direction_estimated"] = estimated
    row["head_direction_confident"] = confident
    row["head_direction_source"] = "classical_geometry" if estimated else None

    radius = float(options["interaction_radius_cm"])
    angle_limit = float(options["head_angle_threshold_deg"])
    for position, obj in enumerate(objects, start=1):
        center = (float(obj["center"][0]), float(obj["center"][1]))
        distance = _body_distance(body, (x_cm, y_cm), center)
        row[f"object_{position}_distance_cm"] = distance
        nose_distance = (None if nose is None
                         else math.hypot(nose[0] - center[0],
                                         nose[1] - center[1]))
        row[f"object_{position}_nose_distance_cm"] = nose_distance
        zone_polygon = obj.get("zone")
        if zone_polygon:
            clearance = (_point_polygon_distance((x_cm, y_cm), zone_polygon)
                         if body is None
                         else _polygons_distance(body, zone_polygon))
            nose_clearance = (None if nose is None
                              else _point_polygon_distance(nose, zone_polygon))
        else:
            clearance = max(0.0, distance - radius)
            nose_clearance = (None if nose is None or nose_distance is None
                              else max(0.0, nose_distance - radius))
        row[f"object_{position}_zone_clearance_cm"] = clearance
        row[f"object_{position}_nose_zone_clearance_cm"] = nose_clearance
        near = clearance <= NEAR_EPS or (nose_clearance is not None
                                         and nose_clearance <= NEAR_EPS)
        row[f"object_{position}_near"] = bool(near)
        if confident and head is not None and nose is not None:
            vx, vy = center[0] - nose[0], center[1] - nose[1]
            norm = math.hypot(vx, vy)
            if norm <= 0:
                angle = 0.0
            else:
                dot = (head[0] * vx + head[1] * vy) / norm
                dot = min(1.0, max(-1.0, dot))
                angle = math.degrees(math.acos(dot))
            row[f"object_{position}_head_angle_deg"] = angle
            row[f"object_{position}_head_aligned"] = bool(angle <= angle_limit)
        else:
            row[f"object_{position}_head_angle_deg"] = None
            row[f"object_{position}_head_aligned"] = None


def annotate_exploration(rows: Sequence[dict], options: Mapping[str, Any],
                         objects: Sequence[Mapping[str, Any]]) -> None:
    """速度门槛、头向判定模式与唯一归属；没有完成速度就不探索。"""

    use_head = bool(options.get("use_head_direction"))
    max_speed = float(options["exploration_max_speed_cm_s"])
    ids = [str(item["id"]) for item in objects]

    for row in rows:
        if not row.get("in_window") or not row.get("detected"):
            continue
        speed = row.get("speed_cm_s")
        if speed is None:
            mode = "no_speed"
        elif not use_head:
            mode = "proximity"
        elif row.get("head_direction_confident"):
            mode = "head"
        else:
            mode = "fallback"
        row["exploration_mode"] = mode

        eligible: list[int] = []
        if speed is not None and float(speed) <= max_speed:
            for index in (1, 2):
                if not row.get(f"object_{index}_near"):
                    continue
                if mode == "head" and row.get(
                        f"object_{index}_head_aligned") is not True:
                    continue
                eligible.append(index)

        chosen = None
        best = None
        has_nose = row.get("nose_x_cm") is not None
        for index in eligible:
            key = row.get(f"object_{index}_nose_distance_cm") if has_nose \
                else None
            if key is None:
                key = row.get(f"object_{index}_distance_cm")
            if key is None:
                continue
            if best is None or float(key) < best - 1e-12:
                best = float(key)
                chosen = index
        for index in (1, 2):
            row[f"object_{index}_exploring"] = bool(index == chosen)
        if chosen is not None:
            row["current_object"] = chosen
            row["current_object_id"] = ids[chosen - 1]
            row["zone_label"] = "Exploring"
        elif row.get("object_1_near") or row.get("object_2_near"):
            row["zone_label"] = "Near"
        else:
            row["zone_label"] = "Arena"


# -- 事件与统计 ---------------------------------------------------------

def _track(rows: Sequence[Mapping[str, Any]], window: timeline.Window,
           objects: Sequence[Mapping[str, Any]], unit: Mapping[str, Any]
           ) -> dict:
    """对象接近状态机与探索 bout；缺失只标缺口，不补区域时间。"""

    ids = [str(item["id"]) for item in objects]
    events: list[dict] = []
    known: dict[int, bool | None] = {1: None, 2: None}
    zone_entries = {1: 0, 2: 0}
    bouts = {1: 0, 2: 0}
    latency: dict[int, float | None] = {1: None, 2: None}
    if not window.known:
        return {"events": events, "zone_entries": zone_entries,
                "bouts": bouts, "latency": latency}
    origin = float(window.start_s)
    gap = False
    previous: int | None = None
    for row in sorted(rows, key=lambda item: (item["time_s"], item["frame"])):
        if not row.get("in_window"):
            continue
        if not row.get("detected"):
            for index in (1, 2):
                if known[index] is not None:
                    gap = True
            previous = None
            continue
        span = float(row.get("dt_s") or 0.0)
        if span <= 0:
            continue
        moment = max(origin, float(row["time_s"]))
        for index in (1, 2):
            near = bool(row.get(f"object_{index}_near"))
            if near and known[index] is not True:
                zone_entries[index] += 1
                events.append({
                    "event_id": f"{unit.get('id')}:object_{index}_zone_entry:"
                                f"{zone_entries[index]}",
                    "unit_id": unit.get("id"),
                    "unit_name": unit.get("name"),
                    "kind": f"object_{index}_zone_entry",
                    "time_s": moment,
                    "confirmed_s": moment,
                    "target": ids[index - 1],
                    "source": "automatic",
                    "frame": row["frame"],
                    "evidence_complete": not gap,
                })
            known[index] = near
        gap = False
        current = row.get("current_object")
        if current is not None:
            if latency[current] is None:
                latency[current] = moment - origin
            if previous != current:
                bouts[current] += 1
                events.append({
                    "event_id": f"{unit.get('id')}:exploration:"
                                f"{bouts[1] + bouts[2]}",
                    "unit_id": unit.get("id"),
                    "unit_name": unit.get("name"),
                    "kind": "exploration",
                    "time_s": moment,
                    "confirmed_s": moment,
                    "target": ids[current - 1],
                    "source": "automatic",
                    "frame": row["frame"],
                    "mode": row.get("exploration_mode"),
                })
        previous = current
    events.sort(key=lambda item: (item["time_s"], item["frame"]))
    return {"events": events, "zone_entries": zone_entries,
            "bouts": bouts, "latency": latency}


def _center_tracking(rows: Sequence[Mapping[str, Any]],
                     window: timeline.Window) -> tuple[int, float | None]:
    """中心进入：缺失保留此前真假，恢复后从非中心再进才计一次。"""

    if not window.known:
        return 0, None
    known: bool | None = None
    entries = 0
    first: float | None = None
    for row in sorted(rows, key=lambda item: (item["time_s"], item["frame"])):
        if not row.get("in_window") or not row.get("detected"):
            continue
        if float(row.get("dt_s") or 0.0) <= 0:
            continue
        inside = bool(row.get("in_center"))
        if inside and known is not True:
            entries += 1
            if first is None:
                first = max(float(window.start_s), float(row["time_s"]))
        known = inside
    latency = None if first is None else first - float(window.start_s)
    return entries, latency


def _percent(numerator: float, denominator: float) -> float | None:
    return timeline.ratio(numerator * 100.0, denominator)


def _segment_geometry(segment: Mapping[str, Any], width: float,
                      height: float, rect: tuple[float, float, float, float]
                      ) -> dict:
    distance = float(segment["distance"])
    arena_fraction = _clip_fraction(segment["p0"], segment["p1"],
                                    (0.0, 0.0, width, height))
    center_fraction = _clip_fraction(segment["p0"], segment["p1"], rect)
    arena_distance = distance * arena_fraction
    center_distance = distance * center_fraction
    return {
        "distance": distance,
        "arena_fraction": arena_fraction,
        "arena_distance": arena_distance,
        "center_distance": center_distance,
        "outer_distance": max(0.0, arena_distance - center_distance),
        "outside_distance": max(0.0, distance - arena_distance),
    }


def _phase_indices(objects: Sequence[Mapping[str, Any]]) -> dict:
    novel = [item["index"] for item in objects if item["role"] == "novel"]
    familiar = [item["index"] for item in objects if item["role"] == "familiar"]
    return {"novel": novel, "familiar": familiar}


def _phase_metrics(phase: str, explore: Mapping[int, float],
                   objects: Sequence[Mapping[str, Any]]) -> dict:
    """识别期给辨别指数/新物体偏好；熟悉期只给物体平衡指数。"""

    roles = _phase_indices(objects)
    result = {"novel_exploration_s": None, "familiar_exploration_s": None,
              "discrimination_index": None, "preference_index_percent": None,
              "object_balance_index": None}
    if phase == "recognition":
        if not roles["novel"] or not roles["familiar"]:
            return result
        novel = sum(explore.get(index, 0.0) for index in roles["novel"])
        familiar = sum(explore.get(index, 0.0) for index in roles["familiar"])
        result["novel_exploration_s"] = novel
        result["familiar_exploration_s"] = familiar
        total = novel + familiar
        if total > 0:
            result["discrimination_index"] = (novel - familiar) / total
            result["preference_index_percent"] = 100.0 * novel / total
    elif phase == "familiarization":
        first = explore.get(1, 0.0)
        second = explore.get(2, 0.0)
        total = first + second
        if total > 0:
            result["object_balance_index"] = (first - second) / total
    return result


def _summary(unit: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
             segments: Sequence[Mapping[str, Any]], window: timeline.Window,
             options: Mapping[str, Any], width: float, height: float,
             rect: tuple[float, float, float, float],
             objects: Sequence[Mapping[str, Any]],
             tracking: Mapping[str, Any]) -> dict:
    threshold = float(options["movement_threshold_cm_s"])
    times = timeline.time_summary(rows)
    center_time = outer_time = outside_time = 0.0
    detected_frames = window_frames = 0
    explore = {1: 0.0, 2: 0.0}
    zone_time = {1: 0.0, 2: 0.0}
    head_supported = fallback = proximity = 0.0
    estimated = confident = 0.0
    for row in rows:
        span = float(row.get("dt_s") or 0.0)
        if row.get("in_window"):
            window_frames += 1
            if row.get("detected"):
                detected_frames += 1
        if not row.get("detected") or span <= 0:
            continue
        zone = row.get("zone")
        if zone == "Center":
            center_time += span
        elif zone == "Outer":
            outer_time += span
        else:
            outside_time += span
        if row.get("head_direction_estimated"):
            estimated += span
        if row.get("head_direction_confident"):
            confident += span
        for index in (1, 2):
            if row.get(f"object_{index}_near"):
                zone_time[index] += span
        current = row.get("current_object")
        if current in (1, 2):
            explore[current] += span
            mode = row.get("exploration_mode")
            if mode == "head":
                head_supported += span
            elif mode == "fallback":
                fallback += span
            elif mode == "proximity":
                proximity += span

    arena_distance = center_distance = outer_distance = outside_distance = 0.0
    observed_total = measurable = moving = immobility = 0.0
    max_speed: float | None = None
    for segment in segments:
        metrics = _segment_geometry(segment, width, height, rect)
        arena_distance += metrics["arena_distance"]
        center_distance += metrics["center_distance"]
        outer_distance += metrics["outer_distance"]
        outside_distance += metrics["outside_distance"]
        observed_total += metrics["distance"]
        span = float(segment["end_s"]) - float(segment["start_s"])
        in_arena = span * metrics["arena_fraction"]
        measurable += in_arena
        if float(segment["speed"]) > threshold:
            moving += in_arena
        else:
            immobility += in_arena
        if in_arena > 0:
            speed = float(segment["speed"])
            max_speed = speed if max_speed is None else max(max_speed, speed)

    observed = times["observed_time_s"]
    objects_summary = []
    for obj in objects:
        index = obj["index"]
        zone = obj.get("zone")
        objects_summary.append({
            "index": index,
            "id": obj["id"],
            "name": obj["name"],
            "role": obj["role"],
            "zone_type": "polygon" if zone else "circle",
            "point_count": len(zone) if zone else 0,
            "exploration_s": explore[index],
            "zone_time_s": zone_time[index],
            "zone_entries": tracking["zone_entries"][index],
            "bouts": tracking["bouts"][index],
            "latency_s": tracking["latency"][index],
        })

    total_object = explore[1] + explore[2]
    phase = str(options["phase"])
    phase_metrics = _phase_metrics(phase, explore, objects)
    first_object = None
    first_latency: float | None = None
    for item in objects_summary:
        latency = item["latency_s"]
        if latency is not None and (first_latency is None
                                    or latency < first_latency):
            first_latency = latency
            first_object = item
    entries, center_latency = _center_tracking(rows, window)

    summary = {
        "unit_id": unit.get("id"),
        "unit_name": unit.get("name"),
        "mode": KEY,
        "total_time_s": times["total_time_s"],
        "observed_time_s": observed,
        "missing_time_s": times["missing_time_s"],
        "detected_frames": detected_frames,
        "window_frames": window_frames,
        "total_frames": window_frames,
        "detected_rate_percent": _percent(detected_frames, window_frames),
        "center_time_s": center_time,
        "outer_time_s": outer_time,
        "outside_time_s": outside_time,
        "center_percent": _percent(center_time, observed),
        "outside_percent": _percent(outside_time, observed),
        "center_entries": entries,
        "center_latency_s": center_latency,
        "total_distance_cm": arena_distance,
        "observed_total_distance_cm": observed_total,
        "center_distance_cm": center_distance,
        "outer_distance_cm": outer_distance,
        "outside_distance_cm": outside_distance,
        "measurable_motion_time_s": measurable,
        "mean_speed_cm_s": (arena_distance / measurable
                            if measurable > 0 else None),
        "max_speed_cm_s": max_speed,
        "moving_time_s": moving,
        "immobility_time_s": immobility,
        "object_1_exploration_s": explore[1],
        "object_2_exploration_s": explore[2],
        "object_1_bouts": tracking["bouts"][1],
        "object_2_bouts": tracking["bouts"][2],
        "object_1_latency_s": tracking["latency"][1],
        "object_2_latency_s": tracking["latency"][2],
        "object_1_zone_time_s": zone_time[1],
        "object_2_zone_time_s": zone_time[2],
        "object_1_zone_entries": tracking["zone_entries"][1],
        "object_2_zone_entries": tracking["zone_entries"][2],
        "total_object_exploration_s": total_object,
        "first_explored_object": (None if first_object is None
                                  else first_object["name"]),
        "first_explored_object_id": (None if first_object is None
                                     else first_object["id"]),
        "first_explored_role": (None if first_object is None
                                else first_object["role"]),
        "head_supported_exploration_s": head_supported,
        "fallback_exploration_s": fallback,
        "proximity_exploration_s": proximity,
        "head_estimated_rate_percent": _percent(estimated, observed),
        "head_confident_rate_percent": _percent(confident, observed),
        "meets_min_exploration": bool(
            total_object + TIME_EPS >= float(options["min_valid_exploration_s"])),
        "phase": phase,
        "objects": objects_summary,
        "arena_width_cm": width,
        "arena_height_cm": height,
        "analysis_window_start_s": window.start_s if window.known else None,
        "analysis_window_end_s": window.end_s if window.known else None,
    }
    summary.update(phase_metrics)
    for key in OPTION_KEYS:
        summary.setdefault(key, options.get(key))
    return summary


def _bin_rows(rows: Sequence[Mapping[str, Any]],
              segments: Sequence[Mapping[str, Any]], window: timeline.Window,
              options: Mapping[str, Any], width: float, height: float,
              rect: tuple[float, float, float, float],
              objects: Sequence[Mapping[str, Any]],
              unit: Mapping[str, Any]) -> list[dict]:
    """按单位窗起点编号的 1 基分箱；跨箱曝光拆开，阶段公式同汇总。"""

    if not window.known or window.duration_s <= 0:
        return []
    bin_width = float(options["bin_seconds"])
    origin = float(window.start_s)
    duration = window.duration_s
    count = max(1, int(math.ceil((duration - 1e-12) / bin_width)))
    threshold = float(options["movement_threshold_cm_s"])
    phase = str(options["phase"])

    buckets = [{"observed": 0.0, "missing": 0.0, "center": 0.0,
                "outer": 0.0, "outside": 0.0, "explore": {1: 0.0, 2: 0.0},
                "zone": {1: 0.0, 2: 0.0}, "distance": 0.0,
                "observed_total": 0.0, "center_distance": 0.0,
                "outer_distance": 0.0, "outside_distance": 0.0,
                "moving": 0.0, "immobility": 0.0} for _ in range(count)]

    for row in rows:
        pieces = timeline.split_interval(row["interval_start_s"],
                                         row["interval_end_s"], origin,
                                         bin_width)
        for index, low, high in pieces:
            if not 0 <= index < count:
                continue
            bucket = buckets[index]
            span = high - low
            if not row.get("detected"):
                bucket["missing"] += span
                continue
            bucket["observed"] += span
            zone = row.get("zone")
            if zone == "Center":
                bucket["center"] += span
            elif zone == "Outer":
                bucket["outer"] += span
            else:
                bucket["outside"] += span
            for position in (1, 2):
                if row.get(f"object_{position}_near"):
                    bucket["zone"][position] += span
            current = row.get("current_object")
            if current in (1, 2):
                bucket["explore"][current] += span

    for segment in segments:
        span = float(segment["end_s"]) - float(segment["start_s"])
        if span <= 0:
            continue
        pieces = timeline.split_interval(segment["start_s"], segment["end_s"],
                                         origin, bin_width)
        for index, low, high in pieces:
            if not 0 <= index < count:
                continue
            ratio_low = (low - segment["start_s"]) / span
            ratio_high = (high - segment["start_s"]) / span
            p0 = [segment["p0"][0]
                  + (segment["p1"][0] - segment["p0"][0]) * ratio_low,
                  segment["p0"][1]
                  + (segment["p1"][1] - segment["p0"][1]) * ratio_low]
            p1 = [segment["p0"][0]
                  + (segment["p1"][0] - segment["p0"][0]) * ratio_high,
                  segment["p0"][1]
                  + (segment["p1"][1] - segment["p0"][1]) * ratio_high]
            piece = {"p0": p0, "p1": p1,
                     "distance": math.hypot(p1[0] - p0[0], p1[1] - p0[1]),
                     "speed": float(segment["speed"])}
            metrics = _segment_geometry(piece, width, height, rect)
            bucket = buckets[index]
            bucket["distance"] += metrics["arena_distance"]
            bucket["observed_total"] += metrics["distance"]
            bucket["center_distance"] += metrics["center_distance"]
            bucket["outer_distance"] += metrics["outer_distance"]
            bucket["outside_distance"] += metrics["outside_distance"]
            measurable = (high - low) * metrics["arena_fraction"]
            if float(segment["speed"]) > threshold:
                bucket["moving"] += measurable
            else:
                bucket["immobility"] += measurable

    bins: list[dict] = []
    for index, bucket in enumerate(buckets):
        start = index * bin_width
        end = min(start + bin_width, duration)
        measurable = bucket["moving"] + bucket["immobility"]
        phase_metrics = _phase_metrics(phase, bucket["explore"], objects)
        bins.append({
            "unit_id": unit.get("id"),
            "unit_name": unit.get("name"),
            "bin_index": index + 1,
            "start_s": start,
            "end_s": end,
            "duration_s": end - start,
            "observed_time_s": bucket["observed"],
            "missing_time_s": bucket["missing"],
            "center_time_s": bucket["center"],
            "outer_time_s": bucket["outer"],
            "outside_time_s": bucket["outside"],
            "object_1_exploration_s": bucket["explore"][1],
            "object_2_exploration_s": bucket["explore"][2],
            "object_1_zone_time_s": bucket["zone"][1],
            "object_2_zone_time_s": bucket["zone"][2],
            "total_object_exploration_s": (bucket["explore"][1]
                                           + bucket["explore"][2]),
            "distance_cm": bucket["distance"],
            "observed_total_distance_cm": bucket["observed_total"],
            "center_distance_cm": bucket["center_distance"],
            "outer_distance_cm": bucket["outer_distance"],
            "outside_distance_cm": bucket["outside_distance"],
            "moving_time_s": bucket["moving"],
            "immobility_time_s": bucket["immobility"],
            "measurable_motion_time_s": measurable,
            "mean_speed_cm_s": (bucket["distance"] / measurable
                                if measurable > 0 else None),
            "discrimination_index": phase_metrics["discrimination_index"],
            "preference_index_percent": phase_metrics[
                "preference_index_percent"],
            "object_balance_index": phase_metrics["object_balance_index"],
        })
    return bins


# -- 单只动物 -----------------------------------------------------------

def process_unit(unit: Mapping[str, Any],
                 captured: Sequence[Mapping[str, Any]],
                 video_window: timeline.Window,
                 first_seen: float | None) -> dict:
    """物理坐标、对象接近/头向、探索事件、汇总与分箱。"""

    unit_id = unit["id"]
    options = unit["options"]
    geometry = unit.get("geometry") or {}
    matrix = arena_matrix(geometry, options)
    width_cm = float(options["arena_width_cm"])
    height_cm = float(options["arena_height_cm"])
    rect = center_rect(width_cm, height_cm, float(options["center_ratio_x"]),
                       float(options["center_ratio_y"]))
    objects: list[dict] = []
    for item in objects_of(geometry):
        zone = item.get("zone")
        objects.append({
            "index": item["index"],
            "id": item["id"],
            "name": item["name"],
            "role": item["role"],
            "center": to_cm(matrix, item["center"]),
            "zone": (None if not zone
                     else [to_cm(matrix, point) for point in zone]),
        })

    window = timeline.resolve_unit_window(
        unit.get("window"), video_window, first_seen)
    rows = timeline.clip_rows(captured, window)
    method = (unit.get("detection") or {}).get("bg_method")
    for row in rows:
        row.setdefault("detection_method", method)
        row.setdefault(
            "head_method",
            "classical_geometry"
            if row.get("nose_x_px") is not None
            and row.get("tail_x_px") is not None else None)
        _annotate(row, matrix, width_cm, height_cm, rect, options, objects)
    segments = timeline.motion_segments(rows, window,
                                        x_key="x_cm", y_key="y_cm",
                                        require_continuity=True)
    timeline.attach_motion(rows, segments)
    for row in rows:
        row["speed_cm_s"] = row.get("speed")
    annotate_exploration(rows, options, objects)
    tracking = _track(rows, window, objects, unit)
    summary = _summary(unit, rows, segments, window, options, width_cm,
                       height_cm, rect, objects, tracking)
    bins = _bin_rows(rows, segments, window, options, width_cm, height_cm,
                     rect, objects, unit)
    metadata = {
        "coordinate_unit": "cm",
        "scale": "four-corner perspective transform",
        "occupancy_denominator": "observed duration",
        "distance_denominator": "completed motion duration inside the arena",
        "near_rule": "body clearance ≤ 0 or nose clearance ≤ 0",
        "body_distance": "shortest distance from body polygon (or center) to object",
        "head_source": "classical_geometry (heuristic, not a probability)",
        "exploration_rule": ("near, speed ≤ exploration_max_speed_cm_s, and "
                             "head angle ≤ threshold in head mode"),
        "phase_metrics": ("recognition: discrimination_index / "
                          "preference_index_percent; familiarization: "
                          "object_balance_index"),
        "objects": [{"id": item["id"], "name": item["name"],
                     "role": item["role"],
                     "zone_type": "polygon" if item["zone"] else "circle"}
                    for item in objects],
        "arena_width_cm": width_cm,
        "arena_height_cm": height_cm,
    }
    warning = None
    if not window.known:
        warning = ("未检测到动物，无法确定相对时间窗："
                   + str(unit.get("name") or unit_id))
    elif summary["fallback_exploration_s"] > 0:
        fallback = summary["fallback_exploration_s"]
        warning = f"{fallback:.2f} s 探索使用距离回退，这些时段没有合格头向支持。"
    return {"rows": rows, "events": tracking["events"], "bins": bins,
            "summary": summary, "metadata": metadata, "warning": warning}


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
            row = observation_row(unit, frame, result)
            captured[unit_id].append(row)
        context.progress(min(0.9, 0.1 + 0.8 * max(0.0, float(frame.time_s)) /
                             max(1e-9, progress_span or 1.0)))
    return captured


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
        unit_metadata[unit["id"]] = facts["metadata"]
        if facts["warning"]:
            warnings.append(facts["warning"])

    return AnalysisResult(
        observations=timeline.sort_rows(observations),
        summary=summary,
        events=events,
        bins=bins,
        tables={},
        warnings=warnings,
        metadata={"assay": KEY, "coordinate_unit": "cm",
                  "units": unit_metadata, "warnings": warnings},
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
        "total_object_exploration_s": summary["total_object_exploration_s"],
        "object_1_exploration_s": summary["object_1_exploration_s"],
        "object_2_exploration_s": summary["object_2_exploration_s"],
        "object_1_zone_time_s": summary["object_1_zone_time_s"],
        "object_2_zone_time_s": summary["object_2_zone_time_s"],
        "center_time_s": summary["center_time_s"],
        "total_distance_cm": summary["total_distance_cm"],
        "mean_speed_cm_s": summary["mean_speed_cm_s"],
        "phase": summary["phase"],
        "rows": len(facts["rows"]),
        "events": len(facts["events"]),
        "bins": len(facts["bins"]),
        "arena_width_cm": options.get("arena_width_cm"),
        "arena_height_cm": options.get("arena_height_cm"),
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
