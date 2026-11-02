"""LDB 明暗箱：明室/暗室/门区占用、稳定主室与转换评分。

可见模式在两室都是合法凸四角时建立相邻边物理映射（cm）；否则只用像素。
不可见模式不输出全场距离/速度，失检曝光按推断暗室计，并与缺失是同一
时间的不同事实。
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from .. import detection, geometry as geo, media, observation, timeline
from ..schema import AnalysisResult, ValidationError, is_number
from .base import geometry_field, metric, parameter

KEY = "ldb"

DEFAULTS = {
    "dark_box_visible": True,
    "door_open_delay_s": 3.0,
    "transition_min_dwell_s": 0.12,
    "doorway_reset_s": 0.2,
    "bin_seconds": 60.0,
    "light_box_length_cm": 40.0,
    "light_box_width_cm": 20.0,
    "dark_box_length_cm": 40.0,
    "dark_box_width_cm": 20.0,
    "shared_edge_length_cm": 40.0,
    "movement_threshold_cm_s": 1.0,
}

DETECTION_DEFAULTS: dict = {}

OPTION_KEYS = tuple(DEFAULTS)
SIZE_KEYS = ("light_box_length_cm", "light_box_width_cm",
             "dark_box_length_cm", "dark_box_width_cm",
             "shared_edge_length_cm")
ZONE_KEYS = ("light", "dark", "doorway")
ZONE_LABELS = {"light": "明室区域", "dark": "暗室区域", "doorway": "门区"}

SPEC = {
    "key": KEY,
    "label": "明暗箱",
    "short": "LDB",
    "parameters": [
        parameter("dark_box_visible", "暗室可见", "checkbox", True,
                  section="场地",
                  help="不可见时明室未检测即推断在暗室，不输出全场距离"),
        parameter("door_open_delay_s", "开门延迟", "number", 3.0, unit="s",
                  minimum=0, step="any", section="统计"),
        parameter("transition_min_dwell_s", "主室转换确认时间", "number",
                  0.12, unit="s", minimum=0, step="any", section="统计"),
        parameter("doorway_reset_s", "门区候选重置时间", "number", 0.2,
                  unit="s", minimum=0, step="any", section="统计"),
        parameter("bin_seconds", "时间箱长度", "number", 60.0, unit="s",
                  minimum=0, step="any", section="统计"),
        parameter("light_box_length_cm", "明室长度", "number", 40.0,
                  unit="cm", minimum=0, step="any", section="场地"),
        parameter("light_box_width_cm", "明室宽度", "number", 20.0,
                  unit="cm", minimum=0, step="any", section="场地"),
        parameter("dark_box_length_cm", "暗室长度", "number", 40.0,
                  unit="cm", minimum=0, step="any", section="场地"),
        parameter("dark_box_width_cm", "暗室宽度", "number", 20.0,
                  unit="cm", minimum=0, step="any", section="场地"),
        parameter("shared_edge_length_cm", "相邻公共边重合长度", "number",
                  40.0, unit="cm", minimum=0, step="any", section="场地"),
        parameter("movement_threshold_cm_s", "移动速度阈值", "number", 1.0,
                  unit="cm/s", minimum=0, step="any", section="统计"),
    ],
    "geometry": [
        geometry_field("light", "明室边界", "polygon", min_points=3),
        geometry_field("dark", "暗室边界", "polygon", min_points=3),
        geometry_field("doorway", "门区", "polygon", min_points=3,
                       optional=True),
        geometry_field("mask", "检测范围", "polygon", min_points=3,
                       optional=True,
                       help="仅在圈内检测；圈外不参与检测"),
    ],
    "metrics": [
        metric("light_time_s", "明室时间", "s"),
        metric("dark_time_s", "暗室时间", "s"),
        metric("doorway_time_s", "门区时间", "s"),
        metric("light_time_percent", "明室时间比例", "%"),
        metric("dark_time_percent", "暗室时间比例", "%"),
        metric("initial_chamber", "初始主室"),
        metric("total_transitions", "主室转换次数", "次"),
        metric("light_entries", "进入明室次数", "次"),
        metric("dark_entries", "进入暗室次数", "次"),
        metric("first_light_latency_s", "首次进入明室潜伏期", "s"),
        metric("total_distance_cm", "总距离", "cm"),
        metric("mean_speed_cm_s", "平均速度", "cm/s"),
    ],
    "template_keys": list(SIZE_KEYS),
}


# -- 数值与多边形校验 ---------------------------------------------------

def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not is_number(value) \
            or not math.isfinite(float(value)):
        return None
    return float(value)


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


def _is_convex(points: Sequence[Sequence[float]]) -> bool:
    signs = []
    for index in range(len(points)):
        ax, ay = points[index]
        bx, by = points[(index + 1) % len(points)]
        cx, cy = points[(index + 2) % len(points)]
        value = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
        if abs(value) <= 1e-12:
            return False
        signs.append(value > 0)
    return all(sign == signs[0] for sign in signs)


def _polygon(shape: Any) -> list[tuple[float, float]] | None:
    """普通 polygon：至少 3 点、正面积、不自交；type=quad 另须凸且四点。"""

    if not isinstance(shape, Mapping):
        return None
    points = _shape_points(shape)
    if points is None or len(points) < 3:
        return None
    if shape.get("type") == "quad":
        if len(points) != 4 or not _is_convex(points):
            return None
    if abs(geo.polygon_area(points)) <= 1e-9:
        return None
    if geo.polygon_self_intersects(points):
        return None
    return points


def _quad(shape: Any) -> list[tuple[float, float]] | None:
    """有效凸四角：恰好四点、凸、正面积、不交叉；不看 type 名称。"""

    if not isinstance(shape, Mapping):
        return None
    points = _shape_points(shape)
    if points is None or len(points) != 4:
        return None
    if geo.quad_is_crossed(points) or not _is_convex(points):
        return None
    if abs(geo.polygon_area(points)) <= 1e-9:
        return None
    return points


def zones_of(geometry: Mapping[str, Any]) -> dict[str, Any]:
    zones = (geometry or {}).get("zones")
    return zones if isinstance(zones, Mapping) else {}


def validate_geometry(geometry: Mapping[str, Any] | None, *,
                      visible: bool = True,
                      require: bool = False) -> list[dict]:
    """light 必填；visible 时 dark 必填；doorway/mask 可选但合法。"""

    geometry = geometry or {}
    zones = geometry.get("zones")
    issues: list[dict] = []
    if not isinstance(zones, Mapping):
        # 未画完的几何仍可暂存；但已填的遮罩照常校验，不因缺 zones 提前退出。
        if require:
            issues.append({"field": "geometry.zones.light",
                           "message": "缺少明室区域"})
            if visible:
                issues.append({"field": "geometry.zones.dark",
                               "message": "缺少暗室区域"})
    else:
        for key in ZONE_KEYS:
            field = f"geometry.zones.{key}"
            shape = zones.get(key)
            if shape is None:
                if key == "light" and require:
                    issues.append({"field": field, "message": "缺少明室区域"})
                elif key == "dark" and visible and require:
                    issues.append({"field": field, "message": "缺少暗室区域"})
                continue
            if _polygon(shape) is None:
                strict = "需要 4 个有限点、凸、正面积且不交叉" \
                    if isinstance(shape, Mapping) and shape.get("type") == "quad" \
                    else "需要至少 3 个有限点、正面积且不自交"
                issues.append({"field": field, "message": strict})
    mask = geometry.get("mask")
    if mask is not None and _polygon(mask) is None:
        issues.append({"field": "geometry.mask",
                       "message": "检测遮罩需要至少 3 个有限点、正面积且不自交"})
    return issues


def validate_unit(options: Mapping[str, Any] | None,
                  geometry: Mapping[str, Any] | None,
                  provided: Iterable[str] | None = None, *,
                  complete: bool = False) -> list[dict]:
    """只校验显式参数与已绘区域；未完成草稿仍可保存。"""

    values = options if isinstance(options, Mapping) else {}
    provided = set(OPTION_KEYS if provided is None else provided)
    issues: list[dict] = []

    def number(key: str) -> float | None:
        if key not in provided:
            return None
        value = _finite(values.get(key))
        if value is None:
            issues.append({"field": f"options.{key}", "message": "必须是有限数值"})
        return value

    if "dark_box_visible" in provided and not isinstance(
            values.get("dark_box_visible"), bool):
        issues.append({"field": "options.dark_box_visible",
                       "message": "必须是严格布尔值"})
    visible = values.get("dark_box_visible")
    visible = True if not isinstance(visible, bool) else visible

    for key in ("door_open_delay_s", "transition_min_dwell_s", "doorway_reset_s"):
        value = number(key)
        if value is not None and value < 0:
            issues.append({"field": f"options.{key}", "message": "必须 ≥ 0"})

    if "bin_seconds" in provided:
        width = number("bin_seconds")
        if width is not None and width <= 0:
            issues.append({"field": "options.bin_seconds",
                           "message": "时间箱长度必须 > 0"})

    sizes: dict[str, float | None] = {}
    for key in SIZE_KEYS:
        sizes[key] = number(key)
        if sizes[key] is not None and sizes[key] <= 0:
            issues.append({"field": f"options.{key}", "message": "必须大于 0"})

    if "movement_threshold_cm_s" in provided:
        threshold = number("movement_threshold_cm_s")
        if threshold is not None and threshold < 0:
            issues.append({"field": "options.movement_threshold_cm_s",
                           "message": "速度阈值必须 ≥ 0"})

    if visible:
        # 公共边约束用最终合并后的有效尺寸：某个值来自默认也不能跳过比较。
        effective = {key: _finite(values.get(key)) for key in SIZE_KEYS}
        if all(value is not None and value > 0
               for value in effective.values()):
            # 公共边重合长度不能超过两箱各自最长边中较小的一个。
            limit = min(max(effective["light_box_length_cm"],
                            effective["light_box_width_cm"]),
                        max(effective["dark_box_length_cm"],
                            effective["dark_box_width_cm"]))
            if effective["shared_edge_length_cm"] > limit + 1e-12:
                issues.append({
                    "field": "options.shared_edge_length_cm",
                    "message": f"公共边重合长度不能超过 {limit:g} cm",
                })

    issues.extend(validate_geometry(geometry, visible=visible,
                                    require=complete))
    return issues


def step_drawn(geometry: Mapping[str, Any] | None, key: str) -> bool:
    """画布、预览与最终校验共用的“该步骤是否已绘制”判定。"""

    geometry = geometry or {}
    if key in ZONE_KEYS:
        zones = geometry.get("zones")
        return _polygon((zones or {}).get(key)) is not None
    return geometry.get(key) is not None


def step_applies(step: Mapping[str, Any],
                 options: Mapping[str, Any] | None) -> bool:
    """当前可见性下的步骤集合：不可见暗室既不要求画，也不列为当前步骤。"""

    if step.get("key") == "dark":
        visible = (options or {}).get("dark_box_visible")
        return True if not isinstance(visible, bool) else visible
    return True


# -- 区域定位 -----------------------------------------------------------

def zone_of(point: Sequence[float],
            zones: Mapping[str, Sequence[Sequence[float]]], *,
            dark_visible: bool) -> str:
    """优先门区→明室→可见暗室；都不命中即 unknown。"""

    if geo.point_in_polygon(point, zones["doorway"]):
        return "doorway"
    if geo.point_in_polygon(point, zones["light"]):
        return "light"
    if dark_visible and geo.point_in_polygon(point, zones["dark"]):
        return "dark"
    return "unknown"


def classify_row(row: Mapping[str, Any], zones: Mapping[str, Any], *,
                 dark_visible: bool) -> tuple[str | None, str]:
    """返回 (zone, zone_evidence)；缺失按模式给出推断或缺失证据。"""

    if not row.get("detected"):
        if dark_visible:
            return None, "missing"
        return "dark", "inferred_absent"
    point = (row.get("x_px"), row.get("y_px"))
    zone = zone_of(point, zones, dark_visible=dark_visible)
    return zone, "detected"


# -- 可见双室的物理映射 -------------------------------------------------

def _centroid(points: Sequence[Sequence[float]]) -> tuple[float, float]:
    count = len(points)
    return (sum(float(point[0]) for point in points) / count,
            sum(float(point[1]) for point in points) / count)


def _along_across(length: float, width: float, shared: float) -> tuple[float, float]:
    """沿边选 ≥公共边且最接近它的尺寸；另一个为横向。"""

    candidates = [value for value in (length, width) if value >= shared - 1e-12]
    if not candidates:
        candidates = [max(length, width)]
    along = min(candidates, key=lambda value: value - shared)
    across = width if along == length else length
    return along, across


def chamber_mapping(light: Sequence[Sequence[float]],
                    dark: Sequence[Sequence[float]],
                    options: Mapping[str, Any]) -> dict:
    """最近相邻边 → 共同边正向 → 每室四角映到 [0,along]×[0,across]。"""

    shared = float(options["shared_edge_length_cm"])
    best: tuple[float, int, int] | None = None
    for i in range(4):
        light_mid = _centroid([light[i], light[(i + 1) % 4]])
        for j in range(4):
            dark_mid = _centroid([dark[j], dark[(j + 1) % 4]])
            distance = math.hypot(light_mid[0] - dark_mid[0],
                                  light_mid[1] - dark_mid[1])
            if best is None or distance < best[0] - 1e-12:
                best = (distance, i, j)
    _, light_index, dark_index = best
    light_mid = _centroid([light[light_index], light[(light_index + 1) % 4]])
    dark_mid = _centroid([dark[dark_index], dark[(dark_index + 1) % 4]])

    edge = (light[(light_index + 1) % 4][0] - light[light_index][0],
            light[(light_index + 1) % 4][1] - light[light_index][1])
    norm = math.hypot(edge[0], edge[1])
    unit = (edge[0] / norm, edge[1] / norm)

    dark_points = list(dark)
    dark_edge = (dark[(dark_index + 1) % 4][0] - dark[dark_index][0],
                 dark[(dark_index + 1) % 4][1] - dark[dark_index][1])
    if dark_edge[0] * unit[0] + dark_edge[1] * unit[1] < 0:
        # 暗室边向量与共同正向相反：反转顶点顺序后，共享边从原 j+1 号顶点
        # 开始（反转表中位置 (2-j)），方向才与明室一致。
        dark_points = list(reversed(dark))
        dark_start = (2 - dark_index) % 4
    else:
        dark_start = dark_index

    def ordered(points: Sequence[Sequence[float]], start: int) -> list:
        return [points[(start + step) % 4] for step in range(4)]

    light_ordered = ordered(light, light_index)
    dark_ordered = ordered(dark_points, dark_start)

    along_light, across_light = _along_across(
        float(options["light_box_length_cm"]),
        float(options["light_box_width_cm"]), shared)
    along_dark, across_dark = _along_across(
        float(options["dark_box_length_cm"]),
        float(options["dark_box_width_cm"]), shared)

    projection = ((dark_mid[0] - light_mid[0]) * unit[0]
                  + (dark_mid[1] - light_mid[1]) * unit[1])
    sign = -1.0 if projection < 0 else 1.0
    offset = sign * ((along_light + along_dark) / 2.0 - shared)

    def homography(quad: Sequence[Sequence[float]], along: float,
                   across: float) -> np.ndarray:
        source = np.array(quad, dtype=np.float32)
        target = np.array([[0.0, 0.0], [along, 0.0],
                           [along, across], [0.0, across]], dtype=np.float32)
        return cv2.getPerspectiveTransform(source, target)

    return {
        "light": homography(light_ordered, along_light, across_light),
        "dark": homography(dark_ordered, along_dark, across_dark),
        "light_center": _centroid(light_ordered),
        "dark_center": _centroid(dark_ordered),
        "along_light": along_light,
        "across_light": across_light,
        "along_dark": along_dark,
        "across_dark": across_dark,
        "offset": offset,
        "sign": sign,
        "shared": shared,
        "edge_midpoint_distance": best[0],
        "light_edge_index": light_index,
        "dark_edge_index": dark_index,
    }


def _apply_mapping(point: Sequence[float], zone: str,
                   mapping: Mapping[str, Any]) -> tuple[float, float] | None:
    """物理坐标：明室 +y、暗室 −y；unknown 区不出物理坐标。"""

    if zone == "unknown":
        return None

    def mapped(name: str) -> tuple[float, float] | None:
        vector = mapping[name] @ np.array([float(point[0]), float(point[1]), 1.0])
        weight = float(vector[2])
        if abs(weight) < 1e-12:
            return None
        return float(vector[0] / weight), float(vector[1] / weight)

    if zone == "doorway":
        # 门区点归入画面距离更近的室中心，再用那一室的映射。
        light_distance = math.hypot(point[0] - mapping["light_center"][0],
                                    point[1] - mapping["light_center"][1])
        dark_distance = math.hypot(point[0] - mapping["dark_center"][0],
                                   point[1] - mapping["dark_center"][1])
        zone = "light" if light_distance <= dark_distance else "dark"

    if zone == "light":
        local = mapped("light")
        if local is None:
            return None
        return local[0] - mapping["along_light"] / 2.0, +local[1]
    local = mapped("dark")
    if local is None:
        return None
    return (local[0] - mapping["along_dark"] / 2.0 + mapping["offset"],
            -local[1])


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


# -- 事件状态机 ---------------------------------------------------------

class _ChamberState:
    """开门后的稳定主室、同目标候选与门区暂停。"""

    def __init__(self) -> None:
        self.stable: str | None = None
        self.pending_target: str | None = None
        self.pending_start: float | None = None
        self.pending_last: float | None = None
        self.pending_elapsed = 0.0
        self.doorway_since: float | None = None
        self.confirmations = 0

    def observe(self, moment: float, zone: str | None, evidence: str, *,
                dwell: float, reset: float) -> dict | None:
        """返回确认事件；没有确认时只推进候选。"""

        if zone == "doorway":
            if self.pending_target is not None:
                if self.doorway_since is None:
                    self.doorway_since = moment
                    # 门区时段不计入停留；保留起点与已积累时间。
                    self.pending_last = None
                # 起始时刻本身也要判阈值：reset=0 时首个门区点即清空候选。
                if moment - self.doorway_since >= reset - 1e-9:
                    self._clear_pending()
            return None

        self.doorway_since = None
        if zone not in ("light", "dark") or zone == self.stable:
            # 稳定主室、unknown 与可见缺失只清候选，不动已确认主室。
            self._clear_pending()
            return None

        if self.pending_target != zone:
            self.pending_target = zone
            self.pending_start = moment
            self.pending_last = moment
            self.pending_elapsed = 0.0
        else:
            if self.pending_last is not None:
                self.pending_elapsed += moment - self.pending_last
            self.pending_last = moment

        if self.pending_elapsed >= dwell - 1e-9:
            self.confirmations += 1
            event = {
                "kind": "chamber_transition" if self.confirmations > 1
                else "initial_chamber",
                "time_s": self.pending_start,
                "confirmed_s": moment,
                "target": zone,
                "previous_chamber": self.stable,
                "source": "automatic",
                "evidence": evidence,
            }
            self.stable = zone
            self._clear_pending()
            return event
        return None

    def _clear_pending(self) -> None:
        self.pending_target = None
        self.pending_start = None
        self.pending_last = None
        self.pending_elapsed = 0.0


# -- 占用与运动 ---------------------------------------------------------

def _percent(numerator: float, denominator: float) -> float | None:
    return timeline.ratio(numerator * 100.0, denominator)


def _empty_zones() -> dict[str, float]:
    return {"light": 0.0, "dark": 0.0, "doorway": 0.0, "unknown": 0.0}


def _accumulate(rows: Sequence[Mapping[str, Any]], dark_visible: bool
                ) -> tuple[dict[str, float], float, float, float]:
    """窗内按真实曝光累计区域/检测/缺失/推断暗室。"""

    zones = _empty_zones()
    observed = missing = inferred_dark = 0.0
    for row in rows:
        if not row.get("in_window"):
            continue
        span = float(row.get("dt_s") or 0.0)
        if span <= 0:
            continue
        zone = row.get("zone")
        if zone in zones:
            zones[zone] += span
        if row.get("detected"):
            observed += span
        else:
            missing += span
            if not dark_visible:
                inferred_dark += span
    return zones, observed, missing, inferred_dark


def _bin_rows(rows: Sequence[Mapping[str, Any]], window: timeline.Window,
              options: Mapping[str, Any], unit: Mapping[str, Any],
              events: Sequence[Mapping[str, Any]],
              segments: Sequence[Mapping[str, Any]], *,
              mode: str) -> list[dict]:
    if not window.known or window.duration_s <= 0:
        return []
    width = float(options["bin_seconds"])
    origin = float(window.start_s)
    duration = window.duration_s
    count = max(1, int(math.ceil((duration - 1e-12) / width)))
    dark_visible = bool(options["dark_box_visible"])
    threshold = float(options["movement_threshold_cm_s"])

    buckets = []
    for _ in range(count):
        bucket = {"observed": 0.0, "missing": 0.0, "inferred": 0.0,
                  "initial": 0, "transitions": 0, "distance": 0.0,
                  "measurable": 0.0, "moving": 0.0, "immobility": 0.0}
        bucket.update(_empty_zones())
        buckets.append(bucket)

    for row in rows:
        if not row.get("in_window"):
            continue
        pieces = timeline.split_interval(row["interval_start_s"],
                                         row["interval_end_s"], origin, width)
        for index, low, high in pieces:
            if not 0 <= index < count:
                continue
            bucket = buckets[index]
            span = high - low
            zone = row.get("zone")
            # 只有真实 zone=unknown 才计入 unknown；None（可见缺失）只计缺失。
            if zone in bucket:
                bucket[zone] += span
            if row.get("detected"):
                bucket["observed"] += span
            else:
                bucket["missing"] += span
                if not dark_visible:
                    bucket["inferred"] += span

    for event in events:
        # 事件按起点入箱；零宽区间不能交给 split_interval（半开窗会丢掉）。
        moment = float(event["time_s"])
        index = int(math.floor((moment - origin) / width + timeline.SPLIT_EPS))
        if not 0 <= index < count:
            continue
        if event["kind"] == "initial_chamber":
            buckets[index]["initial"] += 1
        else:
            buckets[index]["transitions"] += 1

    for segment in segments:
        span = float(segment["end_s"]) - float(segment["start_s"])
        if span <= 0:
            continue
        pieces = timeline.split_interval(segment["start_s"], segment["end_s"],
                                         origin, width)
        for index, low, high in pieces:
            if not 0 <= index < count:
                continue
            bucket = buckets[index]
            ratio_low = (low - segment["start_s"]) / span
            ratio_high = (high - segment["start_s"]) / span
            p0 = [segment["p0"][0] + (segment["p1"][0] - segment["p0"][0]) * ratio_low,
                  segment["p0"][1] + (segment["p1"][1] - segment["p0"][1]) * ratio_low]
            p1 = [segment["p0"][0] + (segment["p1"][0] - segment["p0"][0]) * ratio_high,
                  segment["p0"][1] + (segment["p1"][1] - segment["p0"][1]) * ratio_high]
            bucket["distance"] += math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            measurable = high - low
            bucket["measurable"] += measurable
            if mode == "cm":
                if float(segment["speed"]) > threshold:
                    bucket["moving"] += measurable
                else:
                    bucket["immobility"] += measurable

    suffix = "cm" if mode == "cm" else ("px" if mode == "px" else None)
    bins: list[dict] = []
    for index, bucket in enumerate(buckets):
        start = index * width
        end = min(start + width, duration)
        entry = {
            "unit_id": unit.get("id"),
            "unit_name": unit.get("name"),
            "bin_index": index + 1,
            "start_s": start,
            "end_s": end,
            "duration_s": end - start,
            "observed_time_s": bucket["observed"],
            "missing_time_s": bucket["missing"],
            "inferred_dark_time_s": bucket["inferred"],
            "light_time_s": bucket["light"],
            "dark_time_s": bucket["dark"],
            "doorway_time_s": bucket["doorway"],
            "unknown_time_s": bucket["unknown"],
            "initial_entries": bucket["initial"],
            "transitions": bucket["transitions"],
        }
        if suffix:
            entry[f"total_distance_{suffix}"] = bucket["distance"]
            entry["measurable_motion_time_s"] = bucket["measurable"]
            entry[f"mean_speed_{suffix}_s"] = (
                bucket["distance"] / bucket["measurable"]
                if bucket["measurable"] > 0 else None)
        if mode == "cm":
            entry["moving_time_s"] = bucket["moving"]
            entry["immobility_time_s"] = bucket["immobility"]
        bins.append(entry)
    return bins


def process_unit(unit: Mapping[str, Any], captured: Sequence[Mapping[str, Any]],
                 video_window: timeline.Window,
                 first_seen: float | None) -> dict:
    """单只动物的区域事实、事件、运动与分箱；捕获后统一处理。"""

    options = unit["options"]
    zones_shape = zones_of(unit.get("geometry") or {})
    dark_visible = bool(options["dark_box_visible"])

    zone_points = {
        "light": _polygon(zones_shape.get("light")) or [],
        "dark": _polygon(zones_shape.get("dark")) or [],
        "doorway": _polygon(zones_shape.get("doorway")) or [],
    }

    window = timeline.resolve_unit_window(
        unit.get("window"), video_window, first_seen)
    rows = timeline.clip_rows(captured, window)

    for row in rows:
        zone, evidence = classify_row(row, zone_points, dark_visible=dark_visible)
        row["zone"] = zone
        row["zone_evidence"] = evidence
        row["stable_chamber"] = None
        row["transition_event"] = False
        row["entry_chamber"] = None

    # 物理映射只在可见且两室都是合法凸四角时建立。
    light_quad = _quad(zones_shape.get("light"))
    dark_quad = _quad(zones_shape.get("dark")) if dark_visible else None
    mapping = None
    if dark_visible and light_quad and dark_quad:
        mapping = chamber_mapping(light_quad, dark_quad, options)
        mode = "cm"
    elif dark_visible:
        mode = "px"
    else:
        mode = None

    if mode == "cm":
        for row in rows:
            row["coordinate_unit"] = "cm"
            physical = None
            if row.get("detected"):
                physical = _apply_mapping((row["x_px"], row["y_px"]),
                                          row.get("zone") or "unknown", mapping)
            if physical is None:
                row["x_cm"] = None
                row["y_cm"] = None
            else:
                row["x_cm"], row["y_cm"] = physical
    elif mode == "px":
        for row in rows:
            row["coordinate_unit"] = "px"

    cutoff = None
    if window.known:
        cutoff = float(window.start_s) + float(options["door_open_delay_s"])

    state = _ChamberState()
    events: list[dict] = []
    for row in sorted((item for item in rows if item.get("in_window")),
                      key=lambda item: (item["time_s"], item["frame"])):
        if cutoff is None or float(row["time_s"]) < cutoff - 1e-9:
            continue
        confirmed = state.observe(
            float(row["time_s"]), row.get("zone"), row.get("zone_evidence"),
            dwell=float(options["transition_min_dwell_s"]),
            reset=float(options["doorway_reset_s"]))
        row["stable_chamber"] = state.stable
        if confirmed is not None:
            confirmed.update({
                "event_id": f"{unit.get('id')}:chamber:{state.confirmations}",
                "unit_id": unit.get("id"),
                "unit_name": unit.get("name"),
                "frame": row["frame"],
            })
            events.append(confirmed)
            if confirmed["kind"] == "chamber_transition":
                row["transition_event"] = True
                row["entry_chamber"] = confirmed["target"]

    zones_time, observed, missing, inferred_dark = _accumulate(rows, dark_visible)
    total = timeline.time_summary(rows)["total_time_s"]

    segments: list[dict] = []
    if mode in ("cm", "px"):
        x_key = "x_cm" if mode == "cm" else "x_px"
        y_key = "y_cm" if mode == "cm" else "y_px"
        segments = timeline.motion_segments(rows, window, x_key=x_key,
                                            y_key=y_key,
                                            require_continuity=True)
        timeline.attach_motion(rows, segments)

    measurable = sum(float(item["end_s"]) - float(item["start_s"])
                     for item in segments)
    distance = sum(float(item["distance"]) for item in segments)
    threshold = float(options["movement_threshold_cm_s"])
    moving = immobility = 0.0
    max_speed: float | None = None
    for segment in segments:
        span = float(segment["end_s"]) - float(segment["start_s"])
        if span <= 0:
            continue
        if max_speed is None or float(segment["speed"]) > max_speed:
            max_speed = float(segment["speed"])
        if mode == "cm":
            if float(segment["speed"]) > threshold:
                moving += span
            else:
                immobility += span

    transitions = [item for item in events
                   if item["kind"] == "chamber_transition"]
    light_events = [item for item in events if item["target"] == "light"]
    initial = events[0] if events else None

    summary: dict[str, Any] = {
        "unit_id": unit.get("id"),
        "unit_name": unit.get("name"),
        "mode": KEY,
        "analysis_window_start_s": window.start_s if window.known else None,
        "analysis_window_end_s": window.end_s if window.known else None,
        "total_time_s": total,
        "observed_time_s": observed,
        "missing_time_s": missing,
        "light_time_s": zones_time["light"],
        "dark_time_s": zones_time["dark"],
        "doorway_time_s": zones_time["doorway"],
        "unknown_time_s": zones_time["unknown"],
        "light_time_percent": _percent(zones_time["light"], total),
        "dark_time_percent": _percent(zones_time["dark"], total),
        "doorway_time_percent": _percent(zones_time["doorway"], total),
        "dark_box_visible": dark_visible,
        "position_available": mode is not None,
        "tracking_reliable": mode == "cm",
        "coordinate_unit": {"cm": "cm", "px": "px", None: None}[mode],
        "inferred_dark_time_s": inferred_dark,
        "initial_chamber": initial["target"] if initial else None,
        "initial_chamber_time_s": initial["time_s"] if initial else None,
        "initial_chamber_evidence": initial["evidence"] if initial else None,
        "total_transitions": len(transitions),
        "light_entries": len([item for item in transitions
                              if item["target"] == "light"]),
        "dark_entries": len([item for item in transitions
                             if item["target"] == "dark"]),
        "first_light_latency_s": (
            light_events[0]["time_s"] - cutoff
            if light_events and cutoff is not None else None),
        "door_open_delay_s": float(options["door_open_delay_s"]),
        "transition_min_dwell_s": float(options["transition_min_dwell_s"]),
        "doorway_reset_s": float(options["doorway_reset_s"]),
        "bin_seconds": float(options["bin_seconds"]),
        "movement_threshold_cm_s": float(options["movement_threshold_cm_s"]),
    }
    for key in SIZE_KEYS:
        summary[key] = float(options[key])

    if mode == "cm":
        summary.update({
            "total_distance_cm": distance,
            "mean_speed_cm_s": (distance / measurable if measurable > 0 else None),
            "max_speed_cm_s": max_speed,
            "measurable_motion_time_s": measurable,
            "moving_time_s": moving,
            "immobility_time_s": immobility,
        })
    elif mode == "px":
        summary.update({
            "total_distance_px": distance,
            "mean_speed_px_s": (distance / measurable if measurable > 0 else None),
            "max_speed_px_s": max_speed,
            "measurable_motion_time_s": measurable,
        })

    metadata: dict[str, Any] = {
        "mode": KEY,
        "coordinate_unit": {"cm": "cm", "px": "px", None: None}[mode],
        "position_available": mode is not None,
        "tracking_reliable": mode == "cm",
        "zone_priority": ["doorway", "light", "dark"],
        "occupancy_denominator": "total window duration; door delay only "
                                 "gates chamber events, not occupancy",
        "inference_rule": ("missing exposures are counted as dark chamber "
                           "inferred absence; missing and dark are the same "
                           "time as different facts"
                           if not dark_visible else None),
        "latency_origin": "unit window start + door_open_delay_s",
        "mean_speed_denominator": "completed motion duration inside window",
    }
    if mode == "cm":
        metadata["scale"] = "adjacent-edge quad mapping with shared edge"
        metadata["mapping"] = {
            "method": "nearest adjacent edge pair; light edge defines the "
                      "common direction",
            "shared_edge_length_cm": mapping["shared"],
            "edge_midpoint_distance_px": mapping["edge_midpoint_distance"],
            "along_light_cm": mapping["along_light"],
            "across_light_cm": mapping["across_light"],
            "along_dark_cm": mapping["along_dark"],
            "across_dark_cm": mapping["across_dark"],
            "offset_cm": mapping["offset"],
            "sign": mapping["sign"],
        }
    elif mode == "px":
        metadata["scale"] = "uncalibrated pixels"
        metadata["pixel_motion_note"] = (
            "chambers are not both valid convex quads; positions stay in "
            "pixels and the cm/s movement threshold is not applied")
    else:
        metadata["scale"] = None
        metadata["pixel_note"] = (
            "detected pixels inside the light chamber are kept in the "
            "observation table but are not a reliable full-arena track")

    warning = None
    if not window.known:
        warning = ("未检测到动物，无法确定相对时间窗："
                   + str(unit.get("name") or unit.get("id")))

    bins = _bin_rows(rows, window, options, unit, events, segments, mode=mode)
    return {"rows": rows, "events": events, "bins": bins,
            "summary": summary, "metadata": metadata, "warning": warning}


# -- 捕获与分析 ---------------------------------------------------------

def _capture(context: Any, units: Sequence[Mapping[str, Any]],
             asset: Mapping[str, Any], metadata: Mapping[str, Any],
             source: Any) -> dict:
    """逐帧检测：正式分析与行为预览共用同一条捕获路径。"""

    detectors: dict[Any, Any] = {}
    regions: dict[Any, np.ndarray] = {}
    backgrounds: dict[tuple, Any] = {}
    captured: dict[Any, list[dict]] = {unit["id"]: [] for unit in units}
    unit_ids = [unit["id"] for unit in units]
    job = context.job
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
                        settings, asset, job, int(job.get("orientation", 0) or 0),
                        width, height, cancel=context.cancelled)
                detectors[unit_id] = detection.shared_detector(
                    run_key, settings, background=backgrounds[signature])
                regions[unit_id] = _region_for(unit, unit["options"] or {},
                                               width, height)
            options = unit["options"] or {}
            detector = detectors[unit_id]
            result = observation.track(
                run_key, unit_id,
                image=image, region=regions[unit_id],
                settings=unit["detection"],
                signature=observation.signature_of(
                    unit.get("geometry") or {}, unit["detection"],
                    bool(options.get("dark_box_visible"))),
                assay=KEY, frame_index=frame.index, time_s=frame.time_s,
                foreground=detector.foreground(image),
                retain_units=unit_ids,
            )
            captured[unit_id].append(observation_row(unit, frame, result))
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

    captured = _capture(context, units, asset, metadata, source)

    all_rows = [row for rows in captured.values() for row in rows]
    video_window = timeline.resolve_video_window(
        job.get("window"), all_rows, metadata.get("duration_s"))
    first_seen = {
        unit["id"]: timeline.first_seen_in_window(
            KEY, captured[unit["id"]], video_window, unit.get("geometry") or {})
        for unit in units
    }

    rows_all: list[dict] = []
    events_all: list[dict] = []
    bins_all: list[dict] = []
    summary_all: list[dict] = []
    warnings: list[str] = []
    metadata_units: dict[str, dict] = {}
    coordinate_units: set[str | None] = set()

    for unit in units:
        facts = process_unit(unit, captured[unit["id"]], video_window,
                             first_seen[unit["id"]])
        rows_all.extend(facts["rows"])
        events_all.extend(facts["events"])
        bins_all.extend(facts["bins"])
        summary_all.append(facts["summary"])
        metadata_units[unit["id"]] = facts["metadata"]
        coordinate_units.add(facts["metadata"]["coordinate_unit"])
        if facts["warning"]:
            warnings.append(facts["warning"])

    global_unit = coordinate_units.pop() if len(coordinate_units) == 1 else None
    return AnalysisResult(
        observations=timeline.sort_rows(rows_all),
        summary=summary_all,
        events=events_all,
        bins=bins_all,
        tables={},
        warnings=warnings,
        metadata={
            "assay": KEY,
            "coordinate_unit": global_unit,
            "units": metadata_units,
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
    captured = _capture(context, [unit], asset, metadata, source)[unit["id"]]
    video_window = timeline.resolve_video_window(
        window, captured, metadata.get("duration_s"))
    first_seen = timeline.first_seen_in_window(
        KEY, captured, video_window, unit.get("geometry") or {})
    facts = process_unit(unit, captured, video_window, first_seen)
    summary = facts["summary"]
    result = {
        "mode": KEY,
        "start_s": float(start_s),
        "end_s": float(end_s),
        "window_start_s": summary["analysis_window_start_s"],
        "window_end_s": summary["analysis_window_end_s"],
        "total_time_s": summary["total_time_s"],
        "observed_time_s": summary["observed_time_s"],
        "missing_time_s": summary["missing_time_s"],
        "light_time_s": summary["light_time_s"],
        "dark_time_s": summary["dark_time_s"],
        "doorway_time_s": summary["doorway_time_s"],
        "light_time_percent": summary["light_time_percent"],
        "dark_time_percent": summary["dark_time_percent"],
        "dark_box_visible": summary["dark_box_visible"],
        "initial_chamber": summary["initial_chamber"],
        "total_transitions": summary["total_transitions"],
        "light_entries": summary["light_entries"],
        "dark_entries": summary["dark_entries"],
        "first_light_latency_s": summary["first_light_latency_s"],
        "inferred_dark_time_s": summary["inferred_dark_time_s"],
        "rows": len(facts["rows"]),
        "events": len(facts["events"]),
        "bins": len(facts["bins"]),
    }
    if summary["coordinate_unit"] == "cm":
        result.update({
            "total_distance_cm": summary.get("total_distance_cm"),
            "mean_speed_cm_s": summary.get("mean_speed_cm_s"),
        })
    elif summary["coordinate_unit"] == "px":
        result.update({
            "total_distance_px": summary.get("total_distance_px"),
            "mean_speed_px_s": summary.get("mean_speed_px_s"),
        })
    return result


def validate(job: Any) -> list[dict]:
    issues: list[dict] = []
    for unit in job.get("units") or []:
        prefix = f"units[{unit.get('id')}]"
        for issue in validate_unit(unit.get("options") or {},
                                   unit.get("geometry") or {},
                                   complete=True):
            entry = dict(issue)
            entry["field"] = f"{prefix}.{issue['field']}"
            issues.append(entry)
    return issues
