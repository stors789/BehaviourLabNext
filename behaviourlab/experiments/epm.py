"""EPM 高架十字迷宫：五区占用、入臂深度与估计尺度。

五个区域都用手绘四角描述，没有把臂缩成中心线。尺度由"臂长/中心边长相对
臂在长轴上的投影跨度"估计，坐标因此是 ``cm_estimated``：它是手画尺寸的
比例换算，不是透视校正结果。入臂以沿长轴的投影深度判定，同一臂再次进入
需要先稳定回到中心区（或中间已进入另一臂）。
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .. import detection, geometry as geo, media, observation, timeline
from ..schema import AnalysisResult, ValidationError, is_number
from .base import geometry_field, metric, parameter

KEY = "epm"

DEFAULTS = {
    "center_size_cm": 5.0,
    "open_arm_length_cm": 30.0,
    "closed_arm_length_cm": 30.0,
    "entry_depth_ratio": 0.2,
    "entry_reset_center_s": 0.2,
    "movement_threshold_cm_s": 1.0,
    "bin_seconds": 60.0,
}

OPTION_KEYS = tuple(DEFAULTS)
ZONE_KEYS = ("center", "open_1", "open_2", "closed_1", "closed_2")
ARM_KEYS = ZONE_KEYS[1:]
ZONE_LABELS = {"center": "中心区", "open_1": "开放臂 1", "open_2": "开放臂 2",
               "closed_1": "封闭臂 1", "closed_2": "封闭臂 2"}
ZONE_GROUPS = {"center": "center", "open_1": "open", "open_2": "open",
               "closed_1": "closed", "closed_2": "closed"}
GROUPS = ("center", "open", "closed", "unknown")
TIME_EPS = 1e-9

DETECTION_DEFAULTS: dict = {}

SPEC = {
    "key": KEY,
    "label": "高架十字迷宫 / EPM",
    "short": "EPM",
    "parameters": [
        parameter("center_size_cm", "中心区边长", "number", 5.0, unit="cm",
                  minimum=0, step="any"),
        parameter("open_arm_length_cm", "开放臂长度", "number", 30.0, unit="cm",
                  minimum=0, step="any"),
        parameter("closed_arm_length_cm", "封闭臂长度", "number", 30.0,
                  unit="cm", minimum=0, step="any"),
        parameter("entry_depth_ratio", "入臂深度比例", "number", 0.2,
                  minimum=0, maximum=1, step="any", section="实验参数"),
        parameter("entry_reset_center_s", "中心区重置时间", "number", 0.2,
                  unit="s", minimum=0, step="any", section="实验参数"),
        parameter("movement_threshold_cm_s", "移动速度阈值", "number", 1.0,
                  unit="cm/s", minimum=0, step="any", section="实验参数"),
        parameter("bin_seconds", "时间箱长度", "number", 60.0, unit="s",
                  minimum=0, step="any", section="实验参数"),
    ],
    "geometry": [
        geometry_field("center", "中心区四角", "quad", count=4),
        geometry_field("open_1", "开放臂 1 四角", "quad", count=4),
        geometry_field("open_2", "开放臂 2 四角", "quad", count=4),
        geometry_field("closed_1", "封闭臂 1 四角", "quad", count=4),
        geometry_field("closed_2", "封闭臂 2 四角", "quad", count=4),
        geometry_field("mask", "检测遮罩", "polygon", min_points=3,
                       optional=True),
    ],
    "metrics": [
        metric("open_time_s", "开放臂时间", "s"),
        metric("closed_time_s", "封闭臂时间", "s"),
        metric("center_time_s", "中心区时间", "s"),
        metric("open_time_percent_total", "开放臂时间占总时长", "%"),
        metric("open_time_percent_arms_only", "开放臂时间占臂区时长", "%"),
        metric("open_entries", "开放臂进入次数"),
        metric("closed_entries", "封闭臂进入次数"),
        metric("total_arm_entries", "总入臂次数"),
        metric("open_entry_percent", "开放臂进入比例", "%"),
        metric("first_open_entry_latency_s", "首次进入开放臂潜伏期", "s"),
        metric("total_distance_cm", "估计总距离", "cm"),
        metric("mean_speed_cm_s", "估计平均速度", "cm/s"),
    ],
    "template_keys": ["zones", "center_size_cm", "open_arm_length_cm",
                      "closed_arm_length_cm"],
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


def _quad(shape: Any) -> list[tuple[float, float]] | None:
    """四角区：类型合法、恰好四个有限点、凸、正面积、不交叉。"""

    if not isinstance(shape, Mapping):
        return None
    if shape.get("type") not in ("quad", "polygon"):
        return None
    points = _shape_points(shape)
    if points is None or len(points) != 4:
        return None
    if geo.quad_is_crossed(points) or not _is_convex(points):
        return None
    if abs(geo.polygon_area(points)) <= 1e-9:
        return None
    return points


def _is_convex(points: Sequence[Sequence[float]]) -> bool:
    signs = []
    for index in range(4):
        ax, ay = points[index]
        bx, by = points[(index + 1) % 4]
        cx, cy = points[(index + 2) % 4]
        value = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
        if abs(value) <= 1e-12:
            return False
        signs.append(value > 0)
    return all(sign == signs[0] for sign in signs)


def _simple_polygon(shape: Any) -> list[tuple[float, float]] | None:
    points = _shape_points(shape)
    if points is None or len(points) < 3:
        return None
    if abs(geo.polygon_area(points)) <= 1e-9:
        return None
    if geo.polygon_self_intersects(points):
        return None
    return points


def _centroid(points: Sequence[Sequence[float]]) -> tuple[float, float]:
    count = len(points)
    return (sum(float(point[0]) for point in points) / count,
            sum(float(point[1]) for point in points) / count)


def arm_frames(zones: Mapping[str, Sequence[Sequence[float]]]) -> dict:
    """每臂的长轴单位向量、投影近远端与宽度（像素）。"""

    center = _centroid(zones["center"])
    frames: dict[str, dict] = {}
    for key in ARM_KEYS:
        points = zones[key]
        middle = _centroid(points)
        dx, dy = middle[0] - center[0], middle[1] - center[1]
        norm = math.hypot(dx, dy)
        if norm <= 1e-12:
            raise ValidationError("臂中心不能与中心区重合",
                                  f"geometry.zones.{key}")
        u = (dx / norm, dy / norm)
        v = (-u[1], u[0])
        along = [point[0] * u[0] + point[1] * u[1] for point in points]
        across = [point[0] * v[0] + point[1] * v[1] for point in points]
        near, far = min(along), max(along)
        width = max(across) - min(across)
        if far - near <= 1e-9 or width <= 1e-9:
            raise ValidationError("臂区长宽不能为 0", f"geometry.zones.{key}")
        frames[key] = {"u": u, "v": v, "center": middle, "near": near,
                       "far": far, "width": width}
    return frames


def estimated_scale(options: Mapping[str, Any],
                    frames: Mapping[str, Mapping[str, Any]]) -> float:
    """八个比例（四臂 × 臂长/宽度）的算术平均，单位 cm/px。"""

    ratios: list[float] = []
    for key, frame in frames.items():
        length = float(options["open_arm_length_cm"]
                       if ZONE_GROUPS[key] == "open"
                       else options["closed_arm_length_cm"])
        ratios.append(length / (float(frame["far"]) - float(frame["near"])))
        ratios.append(float(options["center_size_cm"]) / float(frame["width"]))
    return sum(ratios) / len(ratios)


def validate_unit(options: Mapping[str, Any] | None,
                  geometry: Mapping[str, Any] | None,
                  provided: Iterable[str] | None = None, *,
                  complete: bool = False) -> list[dict]:
    """只校验显式参数与已绘区域；未完成草稿仍可保存。"""

    values = options if isinstance(options, Mapping) else {}
    provided = set(OPTION_KEYS if provided is None else provided)
    issues: list[dict] = []

    def number(key: str) -> float | None:
        value = _finite(values.get(key))
        if value is None:
            issues.append({"field": f"options.{key}", "message": "必须是有限数值"})
        return value

    for key in ("center_size_cm", "open_arm_length_cm", "closed_arm_length_cm"):
        if key not in provided:
            continue
        value = number(key)
        if value is not None and value <= 0:
            issues.append({"field": f"options.{key}", "message": "必须大于 0"})

    if "entry_depth_ratio" in provided:
        ratio = number("entry_depth_ratio")
        if ratio is not None and not 0 <= ratio <= 1:
            issues.append({"field": "options.entry_depth_ratio",
                           "message": "入臂深度比例必须在 0 到 1 之间"})

    if "entry_reset_center_s" in provided:
        reset = number("entry_reset_center_s")
        if reset is not None and reset < 0:
            issues.append({"field": "options.entry_reset_center_s",
                           "message": "中心区重置时间必须 ≥ 0"})

    if "movement_threshold_cm_s" in provided:
        threshold = number("movement_threshold_cm_s")
        if threshold is not None and threshold < 0:
            issues.append({"field": "options.movement_threshold_cm_s",
                           "message": "速度阈值必须 ≥ 0"})

    if "bin_seconds" in provided:
        bin_width = number("bin_seconds")
        if bin_width is not None and bin_width <= 0:
            issues.append({"field": "options.bin_seconds",
                           "message": "时间箱长度必须 > 0"})

    issues.extend(validate_geometry(geometry, require=complete))
    return issues


def validate_geometry(geometry: Mapping[str, Any] | None,
                      *, require: bool = False) -> list[dict]:
    """五个四角区齐全、凸、正面积、不交叉；遮罩为合法简单多边形。"""

    geometry = geometry or {}
    zones = geometry.get("zones")
    if not isinstance(zones, Mapping):
        if require:
            return [{"field": "geometry.zones", "message": "缺少五个区域"}]
        return []
    issues: list[dict] = []
    valid: dict[str, list[tuple[float, float]]] = {}
    for key in ZONE_KEYS:
        field = f"geometry.zones.{key}"
        shape = zones.get(key)
        if shape is None:
            if require:
                issues.append({"field": field,
                               "message": "缺少" + ZONE_LABELS[key]})
            continue
        points = _quad(shape)
        if points is None:
            issues.append({"field": field,
                           "message": "需要 4 个有限点、凸、正面积且不交叉"})
            continue
        valid[key] = points
    if len(valid) == len(ZONE_KEYS):
        try:
            arm_frames(valid)
        except ValidationError as error:
            issues.append({"field": error.field, "message": error.message})
    mask = geometry.get("mask")
    if mask is not None and _simple_polygon(mask) is None:
        issues.append({"field": "geometry.mask",
                       "message": "检测遮罩至少 3 个有限点、正面积且不自交"})
    return issues


def step_drawn(geometry: Mapping[str, Any] | None, key: str) -> bool:
    """画布、预览与最终校验共用的“该步骤是否已绘制”判定。"""

    geometry = geometry or {}
    if key in ZONE_KEYS:
        zones = geometry.get("zones")
        if not isinstance(zones, Mapping):
            return False
        return _quad(zones.get(key)) is not None
    return geometry.get(key) is not None


# -- 区域与尺度 ---------------------------------------------------------

def zones_of(geometry: Mapping[str, Any]) -> dict[str, list[tuple[float, float]]]:
    zones = (geometry or {}).get("zones")
    parsed: dict[str, list[tuple[float, float]]] = {}
    for key in ZONE_KEYS:
        shape = zones.get(key) if isinstance(zones, Mapping) else None
        points = _quad(shape)
        if points is None:
            raise ValidationError("缺少" + ZONE_LABELS[key],
                                  f"geometry.zones.{key}")
        parsed[key] = points
    return parsed


def zone_of(point: Sequence[float],
            zones: Mapping[str, Sequence[Sequence[float]]]) -> str:
    """重叠按 center→open_1→open_2→closed_1→closed_2 顺序取首个命中。"""

    for key in ZONE_KEYS:
        if geo.point_in_polygon(point, zones[key]):
            return key
    return "unknown"


def arm_depth(point: Sequence[float], frame: Mapping[str, Any]) -> float:
    """沿长轴投影的入臂深度；不以离中心距离代替。"""

    u = frame["u"]
    projection = float(point[0]) * u[0] + float(point[1]) * u[1]
    span = float(frame["far"]) - float(frame["near"])
    if span <= 0:
        return 0.0
    return (projection - float(frame["near"])) / span


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
    geometry = unit.get("geometry") or {}
    region, _ = detection.geometry_region(geometry, KEY, options, width, height)
    return region


def _annotate(row: dict,
              zones: Mapping[str, Sequence[Sequence[float]]],
              frames: Mapping[str, Mapping[str, Any]],
              scale: float) -> None:
    row["coordinate_unit"] = "cm_estimated"
    row["arm_entry_event"] = False
    row["arm_entry_name"] = None
    row["arm_depth"] = None
    if not row.get("detected") or row.get("x_px") is None:
        row.update({"x_cm": None, "y_cm": None, "zone": None,
                    "zone_group": None})
        return
    point = (float(row["x_px"]), float(row["y_px"]))
    row["x_cm"] = point[0] * scale
    row["y_cm"] = point[1] * scale
    key = zone_of(point, zones)
    row["zone"] = key
    row["zone_group"] = ZONE_GROUPS.get(key, "unknown")
    if key in frames:
        row["arm_depth"] = arm_depth(point, frames[key])


def _track(rows: Sequence[Mapping[str, Any]], window: timeline.Window,
           frames: Mapping[str, Mapping[str, Any]],
           options: Mapping[str, Any], unit: Mapping[str, Any]) -> dict:
    """入臂状态机：深度门槛、稳定回中心重置、缺失保留 last_arm。"""

    events: list[dict] = []
    entries = {"open": 0, "closed": 0}
    first_open: float | None = None
    if not window.known:
        return {"events": events, "entries": entries,
                "first_open_latency_s": None}
    origin = float(window.start_s)
    reset = float(options["entry_reset_center_s"])
    ratio = float(options["entry_depth_ratio"])
    last_arm: str | None = None
    center_since: float | None = None
    for row in sorted(rows, key=lambda item: (item["time_s"], item["frame"])):
        if not row.get("in_window"):
            continue
        span = float(row.get("dt_s") or 0.0)
        if not row.get("detected") or span <= 0:
            # 失检中止中心计时，但不改写 last_arm。
            center_since = None
            continue
        moment = float(row["time_s"])
        if moment < origin - TIME_EPS:
            # 裁切前上下文只贡献占用，不推进状态也不产生事件。
            continue
        key = row.get("zone")
        if key == "center":
            if center_since is None:
                center_since = moment
            elif moment - center_since >= reset - TIME_EPS:
                last_arm = None
            continue
        center_since = None
        if key not in frames:
            # 区外不改变 last_arm，也不计进入。
            continue
        depth = row.get("arm_depth")
        if (key != last_arm and depth is not None
                and float(depth) + 1e-9 >= ratio):
            group = ZONE_GROUPS[key]
            entries[group] += 1
            row["arm_entry_event"] = True
            row["arm_entry_name"] = key
            if group == "open" and first_open is None:
                first_open = moment
            events.append({
                "event_id": f"{unit.get('id')}:arm_entry:{len(events) + 1}",
                "unit_id": unit.get("id"),
                "unit_name": unit.get("name"),
                "kind": "arm_entry",
                "time_s": moment,
                "confirmed_s": moment,
                "target": key,
                "arm_group": group,
                "source": "automatic",
                "frame": row["frame"],
            })
            last_arm = key
    latency = None if first_open is None else first_open - origin
    return {"events": events, "entries": entries,
            "first_open_latency_s": latency}


def _percent(numerator: float, denominator: float) -> float | None:
    return timeline.ratio(numerator * 100.0, denominator)


def _segment_geometry(segment: Mapping[str, Any]) -> dict:
    return {"distance": float(segment["distance"])}


def _summary(unit: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
             segments: Sequence[Mapping[str, Any]], window: timeline.Window,
             options: Mapping[str, Any],
             zones: Mapping[str, Sequence[Sequence[float]]],
             frames: Mapping[str, Mapping[str, Any]], scale: float,
             tracking: Mapping[str, Any]) -> dict:
    threshold = float(options["movement_threshold_cm_s"])
    times = timeline.time_summary(rows)
    seconds = {group: 0.0 for group in GROUPS}
    detected_frames = window_frames = 0
    for row in rows:
        span = float(row.get("dt_s") or 0.0)
        if row.get("in_window"):
            window_frames += 1
            if row.get("detected"):
                detected_frames += 1
        if not row.get("detected") or span <= 0:
            continue
        group = row.get("zone_group") or "unknown"
        if group in seconds:
            seconds[group] += span

    distance = measurable = moving = immobility = 0.0
    max_speed: float | None = None
    for segment in segments:
        length = float(segment["distance"])
        distance += length
        span = float(segment["end_s"]) - float(segment["start_s"])
        measurable += span
        if float(segment["speed"]) > threshold:
            moving += span
        else:
            immobility += span
        if span > 0:
            speed = float(segment["speed"])
            max_speed = speed if max_speed is None else max(max_speed, speed)

    open_entries = int(tracking["entries"]["open"])
    closed_entries = int(tracking["entries"]["closed"])
    total_entries = open_entries + closed_entries
    summary = {
        "unit_id": unit.get("id"),
        "unit_name": unit.get("name"),
        "mode": KEY,
        "coordinate_unit": "cm_estimated",
        "total_time_s": times["total_time_s"],
        "observed_time_s": times["observed_time_s"],
        "missing_time_s": times["missing_time_s"],
        "detected_frames": detected_frames,
        "window_frames": window_frames,
        "total_frames": window_frames,
        "open_time_s": seconds["open"],
        "closed_time_s": seconds["closed"],
        "center_time_s": seconds["center"],
        "unknown_time_s": seconds["unknown"],
        "zone_seconds": dict(seconds),
        "open_time_percent_total": _percent(seconds["open"],
                                            times["total_time_s"]),
        "open_time_percent_arms_only": _percent(
            seconds["open"], seconds["open"] + seconds["closed"]),
        "open_entries": open_entries,
        "closed_entries": closed_entries,
        "total_arm_entries": total_entries,
        "open_entry_percent": _percent(open_entries, total_entries),
        "first_open_entry_latency_s": tracking["first_open_latency_s"],
        "total_distance_cm": distance,
        "measurable_motion_time_s": measurable,
        "moving_time_s": moving,
        "immobility_time_s": immobility,
        "mean_speed_cm_s": (distance / measurable if measurable > 0 else None),
        "max_speed_cm_s": max_speed,
        "cm_per_px_est": scale,
        "analysis_window_start_s": window.start_s if window.known else None,
        "analysis_window_end_s": window.end_s if window.known else None,
    }
    for key in OPTION_KEYS:
        summary.setdefault(key, options.get(key))
    return summary


def _bin_rows(rows: Sequence[Mapping[str, Any]],
              segments: Sequence[Mapping[str, Any]],
              events: Sequence[Mapping[str, Any]],
              window: timeline.Window, options: Mapping[str, Any],
              unit: Mapping[str, Any]) -> list[dict]:
    """1 基分箱：曝光按秒拆，入臂事件计入起点所在箱（边界归后箱）。"""

    if not window.known or window.duration_s <= 0:
        return []
    bin_width = float(options["bin_seconds"])
    origin = float(window.start_s)
    duration = window.duration_s
    count = max(1, int(math.ceil((duration - 1e-12) / bin_width)))
    threshold = float(options["movement_threshold_cm_s"])

    buckets = [{"observed": 0.0, "missing": 0.0, "unknown": 0.0,
                "center": 0.0, "open": 0.0, "closed": 0.0,
                "open_entries": 0, "closed_entries": 0, "distance": 0.0,
                "measurable": 0.0, "moving": 0.0, "immobility": 0.0}
               for _ in range(count)]

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
            group = row.get("zone_group") or "unknown"
            if group in ("center", "open", "closed", "unknown"):
                bucket[group] += span

    for segment in segments:
        span = float(segment["end_s"]) - float(segment["start_s"])
        if span <= 0:
            continue
        pieces = timeline.split_interval(segment["start_s"], segment["end_s"],
                                         origin, bin_width)
        for index, low, high in pieces:
            if not 0 <= index < count:
                continue
            bucket = buckets[index]
            ratio = (high - low) / span
            bucket["distance"] += float(segment["distance"]) * ratio
            bucket["measurable"] += high - low
            if float(segment["speed"]) > threshold:
                bucket["moving"] += high - low
            else:
                bucket["immobility"] += high - low

    for event in events:
        index = int(math.floor((float(event["time_s"]) - origin) / bin_width
                               + 1e-9))
        if not 0 <= index < count:
            continue
        key = "open_entries" if event.get("arm_group") == "open" \
            else "closed_entries"
        buckets[index][key] += 1

    bins: list[dict] = []
    for index, bucket in enumerate(buckets):
        start = index * bin_width
        end = min(start + bin_width, duration)
        arms = bucket["open"] + bucket["closed"]
        bins.append({
            "unit_id": unit.get("id"),
            "unit_name": unit.get("name"),
            "bin_index": index + 1,
            "start_s": start,
            "end_s": end,
            "duration_s": end - start,
            "observed_time_s": bucket["observed"],
            "missing_time_s": bucket["missing"],
            "unknown_time_s": bucket["unknown"],
            "center_time_s": bucket["center"],
            "open_time_s": bucket["open"],
            "closed_time_s": bucket["closed"],
            "open_entries": bucket["open_entries"],
            "closed_entries": bucket["closed_entries"],
            "total_arm_entries": (bucket["open_entries"]
                                  + bucket["closed_entries"]),
            "open_time_percent_total": _percent(bucket["open"], end - start),
            "open_time_percent_arms_only": _percent(bucket["open"], arms),
            "distance_cm": bucket["distance"],
            "measurable_motion_time_s": bucket["measurable"],
            "moving_time_s": bucket["moving"],
            "immobility_time_s": bucket["immobility"],
            "mean_speed_cm_s": (bucket["distance"] / bucket["measurable"]
                                if bucket["measurable"] > 0 else None),
        })
    return bins


# -- 单只动物 -----------------------------------------------------------

def process_unit(unit: Mapping[str, Any],
                 captured: Sequence[Mapping[str, Any]],
                 video_window: timeline.Window,
                 first_seen: float | None) -> dict:
    """估计坐标、五区占用、入臂事件、汇总与分箱。"""

    unit_id = unit["id"]
    options = unit["options"]
    geometry = unit.get("geometry") or {}
    zones = zones_of(geometry)
    frames = arm_frames(zones)
    scale = estimated_scale(options, frames)

    window = timeline.resolve_unit_window(
        unit.get("window"), video_window, first_seen)
    rows = timeline.clip_rows(captured, window)
    method = (unit.get("detection") or {}).get("bg_method")
    for row in rows:
        row.setdefault("detection_method", method)
        _annotate(row, zones, frames, scale)
    tracking = _track(rows, window, frames, options, unit)
    segments = timeline.motion_segments(rows, window,
                                        x_key="x_cm", y_key="y_cm",
                                        require_continuity=True)
    timeline.attach_motion(rows, segments)
    cumulative = 0.0
    by_frame = {(segment["unit_id"], segment["frame"]): segment
                for segment in segments}
    for row in sorted(rows, key=lambda item: (item["time_s"], item["frame"])):
        segment = by_frame.get((row["unit_id"], row["frame"]))
        if segment is None:
            row["step_distance_cm"] = None
            row["speed_cm_s"] = None
        else:
            row["step_distance_cm"] = float(segment["distance"])
            row["speed_cm_s"] = float(segment["speed"])
            cumulative += float(segment["distance"])
        row["cumulative_distance_cm"] = cumulative

    summary = _summary(unit, rows, segments, window, options, zones, frames,
                       scale, tracking)
    bins = _bin_rows(rows, segments, tracking["events"], window, options, unit)
    metadata = {
        "coordinate_unit": "cm_estimated",
        "scale": "mean of 8 drawn-size ratios (arm length / projection span, "
                 "center size / arm width); not a perspective correction",
        "cm_per_px_est": scale,
        "zone_priority": list(ZONE_KEYS),
        "occupancy_denominator": "observed duration",
        "mean_speed_denominator": "measurable completed motion duration",
        "center_reset": "reset after entry_reset_center_s of real center "
                        "observations; missing aborts the center timer",
        "missing_rule": "missing exposure is counted as missing only",
    }
    warning = None
    if not window.known:
        warning = ("未检测到动物，无法确定相对时间窗："
                   + str(unit.get("name") or unit_id))
    return {"rows": rows, "segments": segments,
            "events": tracking["events"], "bins": bins,
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
        metadata={"assay": KEY, "coordinate_unit": "cm_estimated",
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
        "open_time_s": summary["open_time_s"],
        "closed_time_s": summary["closed_time_s"],
        "center_time_s": summary["center_time_s"],
        "open_entries": summary["open_entries"],
        "closed_entries": summary["closed_entries"],
        "total_distance_cm": summary["total_distance_cm"],
        "mean_speed_cm_s": summary["mean_speed_cm_s"],
        "cm_per_px_est": summary["cm_per_px_est"],
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
