"""OFT 旷场实验：四角透视、区域占用、中心进入与完成运动统计。

数值实现与表单共用同一套解析参数；观测、事件、分箱与汇总都从同一次捕获
派生，导出不再重新检测。
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from .. import detection, media, observation, timeline
from ..schema import AnalysisResult, ValidationError, is_number
from .base import geometry_field, metric, parameter

KEY = "oft"

DEFAULTS = {
    "arena_width_cm": 50.0,
    "arena_height_cm": 50.0,
    "center_ratio_x": 0.5,
    "center_ratio_y": 0.5,
    "wall_margin_cm": 8.0,
    "movement_threshold_cm_s": 1.0,
    "bin_seconds": 60.0,
}

DETECTION_DEFAULTS: dict = {"bg_method": "temporal"}

OPTION_KEYS = tuple(DEFAULTS)

SPEC = {
    "key": KEY,
    "label": "旷场实验",
    "short": "OFT",
    "parameters": [
        parameter("arena_width_cm", "场地宽度", "number", 50.0, unit="cm",
                  minimum=0, section="统计"),
        parameter("arena_height_cm", "场地长度", "number", 50.0, unit="cm",
                  minimum=0, section="统计"),
        parameter("center_ratio_x", "中心区宽度比例", "number", 0.5,
                  minimum=0, maximum=1, section="统计"),
        parameter("center_ratio_y", "中心区长度比例", "number", 0.5,
                  minimum=0, maximum=1, section="统计"),
        parameter("wall_margin_cm", "壁区宽度", "number", 8.0, unit="cm",
                  minimum=0, section="统计"),
        parameter("movement_threshold_cm_s", "移动速度阈值", "number", 1.0,
                  unit="cm/s", minimum=0, section="统计"),
        parameter("bin_seconds", "时间箱长度", "number", 60.0, unit="s",
                  minimum=0, section="统计"),
    ],
    "geometry": [
        geometry_field("arena", "场地四角", "quad", count=4),
        geometry_field("mask", "检测范围", "polygon", min_points=3,
                       optional=True,
                       help="仅在圈内检测；圈外不参与检测"),
    ],
    "metrics": [
        metric("center_percent", "中心区占比", "%"),
        metric("total_distance_cm", "场地内路程", "cm"),
        metric("mean_speed_cm_s", "平均速度", "cm/s"),
        metric("center_entries", "中心进入", "次"),
    ],
    "template_keys": ["arena_width_cm", "arena_height_cm"],
}


# -- 参数与四角校验 -----------------------------------------------------

def _finite(value: Any) -> float | None:
    if not is_number(value) or not math.isfinite(float(value)):
        return None
    return float(value)


def validate_unit(options: Mapping[str, Any] | None,
                  geometry: Mapping[str, Any] | None,
                  provided: Iterable[str] | None = None, *,
                  complete: bool = False) -> list[dict]:
    """只校验显式给出的值与已绘制的四角；未完成草稿仍可保存。

    complete=True 用于最终校验/开始任务：此时必须已有完整合法的场地四角。
    """

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
        issues.append({"field": "options.arena_width_cm", "message": "场地宽度必须 > 0"})
    if height is not None and height <= 0:
        issues.append({"field": "options.arena_height_cm", "message": "场地长度必须 > 0"})

    for key in ("center_ratio_x", "center_ratio_y"):
        if key not in provided:
            continue
        ratio = number(key)
        if ratio is not None and not 0 < ratio <= 1:
            issues.append({"field": f"options.{key}",
                           "message": "中心比例必须 > 0 且 ≤ 1"})

    if "wall_margin_cm" in provided:
        margin = number("wall_margin_cm")
        limit_width = width if width is not None else _finite(values.get("arena_width_cm"))
        limit_height = height if height is not None else _finite(values.get("arena_height_cm"))
        if margin is not None and margin < 0:
            issues.append({"field": "options.wall_margin_cm",
                           "message": "壁区宽度必须 ≥ 0"})
        elif (margin is not None and limit_width and limit_height
                and margin > min(limit_width, limit_height) / 2):
            issues.append({"field": "options.wall_margin_cm",
                           "message": "壁区宽度不能超过场地短边的一半"})

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

    issues.extend(validate_arena(geometry, require=complete))
    return issues


def _points(shape: Mapping[str, Any] | None) -> list[tuple[float, float]]:
    if not isinstance(shape, Mapping):
        return []
    result: list[tuple[float, float]] = []
    for point in shape.get("points") or []:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return []
        x, y = _finite(point[0]), _finite(point[1])
        if x is None or y is None:
            return []
        result.append((x, y))
    return result


def validate_arena(geometry: Mapping[str, Any] | None,
                   *, require: bool = False) -> list[dict]:
    """四角必须是四个有限点、严格凸、非退化且不交叉；不修成外接矩形。"""

    shape = (geometry or {}).get("arena")
    if shape is None:
        # 草稿保存允许还没画；最终校验/开始任务必须已有四角。
        if require:
            return [{"field": "geometry.arena", "message": "缺少场地四角"}]
        return []
    if not isinstance(shape, Mapping) or shape.get("type") not in ("quad", "polygon"):
        return [{"field": "geometry.arena", "message": "场地必须是四角四边形"}]
    points = _points(shape)
    if len(shape.get("points") or []) != 4 or len(points) != 4:
        return [{"field": "geometry.arena", "message": "场地需要 4 个有限二维点"}]
    signs = []
    for index in range(4):
        ax, ay = points[index]
        bx, by = points[(index + 1) % 4]
        cx, cy = points[(index + 2) % 4]
        cross = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
        if abs(cross) <= 1e-9:
            return [{"field": "geometry.arena", "message": "场地四角退化或共线"}]
        signs.append(cross > 0)
    if any(sign != signs[0] for sign in signs):
        return [{"field": "geometry.arena", "message": "场地四角必须凸且不交叉"}]
    return []


def arena_matrix(geometry: Mapping[str, Any],
                 options: Mapping[str, Any]) -> np.ndarray:
    """四角 → (0,0)(W,0)(W,H)(0,H) 厘米的单应矩阵。"""

    issues = validate_arena(geometry)
    if issues:
        raise ValidationError(issues[0]["message"], issues[0]["field"])
    points = _points((geometry or {}).get("arena"))
    width = float(options["arena_width_cm"])
    height = float(options["arena_height_cm"])
    source = np.array(points, dtype=np.float32)
    target = np.array([[0.0, 0.0], [width, 0.0], [width, height],
                       [0.0, height]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(source, target)
    if not np.all(np.isfinite(matrix)) or abs(float(np.linalg.det(matrix))) < 1e-12:
        raise ValidationError("场地四角无法建立透视变换", "geometry.arena")
    return matrix


def to_cm(matrix: np.ndarray, point: Sequence[float]) -> tuple[float, float]:
    vector = matrix @ np.array([float(point[0]), float(point[1]), 1.0])
    weight = float(vector[2])
    if abs(weight) < 1e-12:
        raise ValidationError("场地四角无法建立透视变换", "geometry.arena")
    return float(vector[0] / weight), float(vector[1] / weight)


def zone_of(x: float, y: float, width: float, height: float,
            ratio_x: float, ratio_y: float, margin: float) -> str:
    """Center / Wall / Outer / Outside；壁区是外围的子集。"""

    if x < 0 or y < 0 or x > width or y > height:
        return "Outside"
    left = width * (1.0 - ratio_x) / 2.0
    right = width * (1.0 + ratio_x) / 2.0
    top = height * (1.0 - ratio_y) / 2.0
    bottom = height * (1.0 + ratio_y) / 2.0
    if left <= x <= right and top <= y <= bottom:
        return "Center"
    if x <= margin or x >= width - margin or y <= margin or y >= height - margin:
        return "Wall"
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


def _center_rect(width: float, height: float, ratio_x: float,
                 ratio_y: float) -> tuple[float, float, float, float]:
    return (width * (1.0 - ratio_x) / 2.0, height * (1.0 - ratio_y) / 2.0,
            width * (1.0 + ratio_x) / 2.0, height * (1.0 + ratio_y) / 2.0)


# -- 捕获与统计 ---------------------------------------------------------

def _unit_dict(unit: Mapping[str, Any]) -> dict:
    return {"id": unit.get("id"), "name": unit.get("name"),
            "animal_id": unit.get("animal_id")}


def observation_row(unit: Mapping[str, Any], frame: Any,
                    detection: Any) -> dict:
    """源观测行：记录实际背景方法与头向来源，缺失时显式 null。"""

    has_head = detection.nose is not None and detection.tail_base is not None
    return timeline.base_row(
        _unit_dict(unit), frame, detection,
        extra={"detection_method": (unit.get("detection") or {}).get("bg_method"),
               "head_method": "classical_geometry" if has_head else None},
    )


def _region_for(unit: Mapping[str, Any], options: Mapping[str, Any],
                width: int, height: int) -> np.ndarray:
    geometry = unit.get("geometry") or {}
    region, _ = detection.geometry_region(geometry, KEY, options, width, height)
    return region


def _annotate(row: dict, matrix: np.ndarray, width: float, height: float,
              options: Mapping[str, Any]) -> None:
    """物理坐标与区域字段；缺失行全部 null。"""

    row["coordinate_unit"] = "cm"
    if not row.get("detected"):
        row.update({"x": None, "y": None, "x_cm": None, "y_cm": None,
                    "nose_x_cm": None, "nose_y_cm": None,
                    "tail_x_cm": None, "tail_y_cm": None,
                    "in_arena": None, "in_center": None, "in_outer": None,
                    "in_wall": None, "zone": "Unknown"})
        return
    x_cm, y_cm = to_cm(matrix, (row["x_px"], row["y_px"]))
    row["x"] = x_cm
    row["y"] = y_cm
    row["x_cm"] = x_cm
    row["y_cm"] = y_cm
    for prefix, x_key, y_key in (("nose", "nose_x_px", "nose_y_px"),
                                 ("tail", "tail_x_px", "tail_y_px")):
        if row.get(x_key) is not None and row.get(y_key) is not None:
            px, py = to_cm(matrix, (row[x_key], row[y_key]))
            row[f"{prefix}_x_cm"] = px
            row[f"{prefix}_y_cm"] = py
        else:
            row[f"{prefix}_x_cm"] = None
            row[f"{prefix}_y_cm"] = None
    ratio_x = float(options["center_ratio_x"])
    ratio_y = float(options["center_ratio_y"])
    margin = float(options["wall_margin_cm"])
    zone = zone_of(x_cm, y_cm, width, height, ratio_x, ratio_y, margin)
    row["zone"] = zone
    row["in_arena"] = zone != "Outside"
    row["in_center"] = zone == "Center"
    row["in_outer"] = zone in ("Outer", "Wall")
    row["in_wall"] = zone == "Wall"


def _events_for(rows: Sequence[Mapping[str, Any]], unit_id: str,
                unit_name: str, window: timeline.Window,
                latency_start: float) -> tuple[list[dict], int, float | None]:
    """中心进入状态机：缺失只标缺口，不覆盖上一个已知真假。"""

    events: list[dict] = []
    if not window.known:
        # 未知窗口没有统计区间，不产生任何事件，也不拿 None 当时间。
        return events, 0, None
    known: bool | None = None
    gap = False
    sequence = 0
    for row in sorted(rows, key=lambda item: (item["time_s"], item["frame"])):
        if not row.get("in_window"):
            # 窗外边界上下文只供运动段使用，不参与中心进入状态机。
            continue
        if not row.get("detected"):
            if known is not None:
                gap = True
            continue
        in_center = bool(row.get("in_center"))
        if in_center and known is not True:
            sequence += 1
            moment = max(float(window.start_s), float(row["time_s"]))
            events.append({
                "event_id": f"{unit_id}:center_entry:{sequence}",
                "unit_id": unit_id,
                "unit_name": unit_name,
                "kind": "center_entry",
                "time_s": moment,
                "confirmed_s": moment,
                "target": "center",
                "source": "automatic",
                "frame": row["frame"],
                "evidence_complete": not gap,
            })
        known = in_center
        gap = False
    latency = (events[0]["time_s"] - latency_start) if events else None
    return events, len(events), latency


def _segment_geometry(segment: Mapping[str, Any], width: float, height: float,
                      center_rect: tuple[float, float, float, float]) -> dict:
    distance = float(segment["distance"])
    arena_fraction = _clip_fraction(segment["p0"], segment["p1"],
                                    (0.0, 0.0, width, height))
    center_fraction = _clip_fraction(segment["p0"], segment["p1"], center_rect)
    arena_distance = distance * arena_fraction
    center_distance = distance * center_fraction
    return {
        "distance": distance,
        "arena_fraction": arena_fraction,
        "center_fraction": center_fraction,
        "arena_distance": arena_distance,
        "center_distance": center_distance,
        "outer_distance": max(0.0, arena_distance - center_distance),
        "outside_distance": max(0.0, distance - arena_distance),
    }


def _bin_rows(rows: Sequence[Mapping[str, Any]],
              segments: Sequence[Mapping[str, Any]], window: timeline.Window,
              width_cm: float, height_cm: float,
              center_rect: tuple[float, float, float, float],
              options: Mapping[str, Any], unit: Mapping[str, Any]) -> list[dict]:
    if not window.known or window.duration_s <= 0:
        return []
    bin_width = float(options["bin_seconds"])
    origin = float(window.start_s)
    duration = window.duration_s
    count = max(1, int(math.ceil((duration - 1e-12) / bin_width)))
    threshold = float(options["movement_threshold_cm_s"])

    buckets = [{"center": 0.0, "outer": 0.0, "outside": 0.0, "wall": 0.0,
                "observed": 0.0, "missing": 0.0, "distance": 0.0,
                "observed_total": 0.0, "center_distance": 0.0,
                "outer_distance": 0.0, "outside_distance": 0.0,
                "moving": 0.0, "immobility": 0.0} for _ in range(count)]

    for row in rows:
        pieces = timeline.split_interval(row["interval_start_s"],
                                         row["interval_end_s"], origin, bin_width)
        for index, low, high in pieces:
            if not 0 <= index < count:
                continue
            bucket = buckets[index]
            span = high - low
            if row.get("detected"):
                bucket["observed"] += span
                zone = row.get("zone")
                if zone == "Center":
                    bucket["center"] += span
                elif zone in ("Outer", "Wall"):
                    bucket["outer"] += span
                    if zone == "Wall":
                        bucket["wall"] += span
                else:
                    bucket["outside"] += span
            else:
                bucket["missing"] += span

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
            p0 = [segment["p0"][0] + (segment["p1"][0] - segment["p0"][0]) * ratio_low,
                  segment["p0"][1] + (segment["p1"][1] - segment["p0"][1]) * ratio_low]
            p1 = [segment["p0"][0] + (segment["p1"][0] - segment["p0"][0]) * ratio_high,
                  segment["p0"][1] + (segment["p1"][1] - segment["p0"][1]) * ratio_high]
            piece = {"p0": p0, "p1": p1,
                     "distance": math.hypot(p1[0] - p0[0], p1[1] - p0[1]),
                     "speed": float(segment["speed"])}
            metrics = _segment_geometry(piece, width_cm, height_cm, center_rect)
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
        bins.append({
            "unit_id": unit.get("id"),
            "unit_name": unit.get("name"),
            "bin_index": index + 1,
            "start_s": start,
            "end_s": end,
            "duration_s": end - start,
            "distance_cm": bucket["distance"],
            "observed_total_distance_cm": bucket["observed_total"],
            "center_distance_cm": bucket["center_distance"],
            "outer_distance_cm": bucket["outer_distance"],
            "outside_distance_cm": bucket["outside_distance"],
            "center_time_s": bucket["center"],
            "outer_time_s": bucket["outer"],
            "outside_time_s": bucket["outside"],
            "wall_time_s": bucket["wall"],
            "observed_time_s": bucket["observed"],
            "missing_time_s": bucket["missing"],
            "moving_time_s": bucket["moving"],
            "immobility_time_s": bucket["immobility"],
            "measurable_motion_time_s": measurable,
            "mean_speed_cm_s": (bucket["distance"] / measurable
                                if measurable > 0 else None),
        })
    return bins


def _percent(numerator: float, denominator: float) -> float | None:
    return timeline.ratio(numerator * 100.0, denominator)


def _unit_summary(unit: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
                  segments: Sequence[Mapping[str, Any]],
                  window: timeline.Window, options: Mapping[str, Any],
                  width_cm: float, height_cm: float,
                  center_rect: tuple[float, float, float, float]) -> dict:
    threshold = float(options["movement_threshold_cm_s"])
    times = timeline.time_summary(rows)
    center_time = outer_time = outside_time = wall_time = 0.0
    detected_frames = window_frames = 0
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
        elif zone in ("Outer", "Wall"):
            outer_time += span
            if zone == "Wall":
                wall_time += span
        else:
            outside_time += span

    arena_distance = center_distance = outer_distance = outside_distance = 0.0
    observed_total = measurable = moving = immobility = 0.0
    max_speed: float | None = None
    for segment in segments:
        metrics = _segment_geometry(segment, width_cm, height_cm, center_rect)
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
    summary = {
        "unit_id": unit.get("id"),
        "unit_name": unit.get("name"),
        "total_time_s": times["total_time_s"],
        "observed_time_s": observed,
        "missing_time_s": times["missing_time_s"],
        "center_time_s": center_time,
        "outer_time_s": outer_time,
        "outside_time_s": outside_time,
        "wall_time_s": wall_time,
        "center_percent": _percent(center_time, observed),
        "outer_percent": _percent(outer_time, observed),
        "outside_percent": _percent(outside_time, observed),
        "wall_percent": _percent(wall_time, observed),
        "moving_time_s": moving,
        "immobility_time_s": immobility,
        "measurable_motion_time_s": measurable,
        "total_distance_cm": arena_distance,
        "observed_total_distance_cm": observed_total,
        "center_distance_cm": center_distance,
        "outer_distance_cm": outer_distance,
        "outside_distance_cm": outside_distance,
        "analysis_window_start_s": window.start_s if window.known else None,
        "analysis_window_end_s": window.end_s if window.known else None,
        "detected_frames": detected_frames,
        "window_frames": window_frames,
        "detected_rate_percent": _percent(detected_frames, window_frames),
        "center_entries": 0,
        "center_latency_s": None,
        "mean_speed_cm_s": (arena_distance / measurable
                            if measurable > 0 else None),
        "max_speed_cm_s": max_speed,
    }
    for key in OPTION_KEYS:
        summary[key] = options.get(key)
    return summary


def _attach_distances(rows: list[dict], segments: Sequence[Mapping[str, Any]],
                      width_cm: float, height_cm: float,
                      center_rect: tuple[float, float, float, float]) -> None:
    """行级距离/速度与累计；无完成段保持 null 或既有累计。"""

    by_frame = {}
    for segment in segments:
        metrics = _segment_geometry(segment, width_cm, height_cm, center_rect)
        by_frame[(segment["unit_id"], segment["frame"])] = (segment, metrics)
    cumulative = 0.0
    cumulative_observed = 0.0
    for row in sorted(rows, key=lambda item: (item["time_s"], item["frame"])):
        entry = by_frame.get((row["unit_id"], row["frame"]))
        if entry is None:
            row["speed_cm_s"] = None
            row["step_distance_cm"] = None
            row["step_observed_total_distance_cm"] = None
            row["step_center_distance_cm"] = None
            row["step_outer_distance_cm"] = None
            row["step_outside_distance_cm"] = None
        else:
            segment, metrics = entry
            row["speed_cm_s"] = float(segment["speed"])
            row["step_distance_cm"] = metrics["arena_distance"]
            row["step_observed_total_distance_cm"] = metrics["distance"]
            row["step_center_distance_cm"] = metrics["center_distance"]
            row["step_outer_distance_cm"] = metrics["outer_distance"]
            row["step_outside_distance_cm"] = metrics["outside_distance"]
            cumulative += metrics["arena_distance"]
            cumulative_observed += metrics["distance"]
        row["cumulative_distance_cm"] = cumulative
        row["cumulative_observed_distance_cm"] = cumulative_observed


def process_unit(unit: Mapping[str, Any], captured: Sequence[Mapping[str, Any]],
                 video_window: timeline.Window,
                 first_seen: float | None) -> dict:
    """单只动物的物理坐标、区域、事件、运动与分箱；捕获后统一处理。"""

    unit_id = unit["id"]
    options = unit["options"]
    geometry = unit.get("geometry") or {}
    matrix = arena_matrix(geometry, options)
    width_cm = float(options["arena_width_cm"])
    height_cm = float(options["arena_height_cm"])
    center_rect = _center_rect(width_cm, height_cm,
                               float(options["center_ratio_x"]),
                               float(options["center_ratio_y"]))
    window = timeline.resolve_unit_window(
        unit.get("window"), video_window, first_seen)
    rows = timeline.clip_rows(captured, window)
    method = (unit.get("detection") or {}).get("bg_method")
    for row in rows:
        # 捕获阶段已写入实际方法；替身行在这里补默认，避免导出缺列。
        row.setdefault("detection_method", method)
        row.setdefault(
            "head_method",
            "classical_geometry"
            if row.get("nose_x_px") is not None and row.get("tail_x_px") is not None
            else None)
        _annotate(row, matrix, width_cm, height_cm, options)
    segments = timeline.motion_segments(rows, window,
                                        x_key="x_cm", y_key="y_cm")
    timeline.attach_motion(rows, segments)
    _attach_distances(rows, segments, width_cm, height_cm, center_rect)
    latency_start = float(window.start_s) if window.known else 0.0
    events, entries, latency = _events_for(
        rows, unit_id, unit.get("name") or "", window, latency_start)
    summary = _unit_summary(unit, rows, segments, window, options,
                            width_cm, height_cm, center_rect)
    summary["center_entries"] = entries
    summary["center_latency_s"] = latency
    bins = _bin_rows(rows, segments, window, width_cm, height_cm,
                     center_rect, options, unit)
    metadata = {
        "coordinate_unit": "cm",
        "scale": "four-corner perspective transform",
        "mean_speed_denominator": "completed motion duration inside the arena",
        "occupancy_denominator": "observed duration",
        "outer_includes_wall": True,
    }
    warning = None
    if not window.known:
        warning = ("未检测到动物，无法确定相对时间窗："
                   + str(unit.get("name") or unit_id))
    return {"rows": rows, "events": events, "bins": bins,
            "summary": summary, "metadata": metadata, "warning": warning}


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

    # 视频统计窗按捕获到的真实源区间确定；元数据时长只在没有采样行时回退。
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

    for unit in units:
        facts = process_unit(unit, captured[unit["id"]], video_window,
                             first_seen[unit["id"]])
        rows_all.extend(facts["rows"])
        events_all.extend(facts["events"])
        bins_all.extend(facts["bins"])
        summary_all.append(facts["summary"])
        metadata_units[unit["id"]] = facts["metadata"]
        if facts["warning"]:
            warnings.append(facts["warning"])

    return AnalysisResult(
        observations=timeline.sort_rows(rows_all),
        summary=summary_all,
        events=events_all,
        bins=bins_all,
        tables={},
        warnings=warnings,
        metadata={
            "assay": KEY,
            "coordinate_unit": "cm",
            "units": metadata_units,
            "warnings": warnings,
        },
    )


def preview_analysis(*, asset: Mapping[str, Any], video: Mapping[str, Any],
                     unit: Mapping[str, Any], start_s: float,
                     end_s: float, time_s: float | None = None) -> dict:
    """行为预览：对实际片段跑同一捕获→分箱→汇总核心，返回片段汇总。

    单帧检测仍走预览自己的路径，这里只补片段行为统计。
    """

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
    geometry = unit.get("geometry") or {}
    first_seen = timeline.first_seen_in_window(
        KEY, captured, video_window, geometry)
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
        "center_time_s": summary["center_time_s"],
        "center_percent": summary["center_percent"],
        "center_entries": summary["center_entries"],
        "center_latency_s": summary["center_latency_s"],
        "total_distance_cm": summary["total_distance_cm"],
        "mean_speed_cm_s": summary["mean_speed_cm_s"],
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
            entry["field"] = f"{prefix}.{issue['field']}"
            issues.append(entry)
    return issues
