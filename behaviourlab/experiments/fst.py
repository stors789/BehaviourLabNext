"""FST 强迫游泳实验：圆筒尺度、运动分数、滞回与片段修整。

运动分数 m = .60·F/D + .25·v/D + .15·min(1, a/3)：F 为身体掩膜内
Farneback 光流 P90（cm/s，按分析 Hz 名义换算），v 为质心速度
（cm/s，按真实相邻源时刻差），a 为身体外轮廓面积变化率（1/s），
D 是用户填写的实际内径。水面线只保存与显示，不参与评分与阈值。
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Mapping, Sequence

import cv2
import numpy as np

from .. import activity, detection, media, timeline
from ..schema import AnalysisResult, Detection, ValidationError, is_number
from .base import geometry_field, metric, parameter

KEY = "fst"

DEFAULTS = {
    "cylinder_diameter_cm": 10.0,
    "auto_threshold": True,
    "threshold_sensitivity": 0.0,
    "immobility_threshold": 0.08,
    "active_threshold": 0.108,
    "smoothing_window_s": 0.5,
    "min_immobility_duration_s": 1.0,
    "min_active_duration_s": 0.3,
    "gap_merge_s": 0.5,
    "bin_seconds": 60.0,
}

OPTION_KEYS = tuple(DEFAULTS)
NONNEGATIVE_KEYS = ("smoothing_window_s", "min_immobility_duration_s",
                    "min_active_duration_s", "gap_merge_s")
POSITIVE_KEYS = ("cylinder_diameter_cm", "immobility_threshold",
                 "active_threshold", "bin_seconds")

# 游泳：亮背景暗鼠；腐蚀 3×3 一次、膨胀 5×5 两次，面积 80–50000 px²。
DETECTION_DEFAULTS = {
    "bg_method": "threshold",
    "bg_type": "light",
    "gray_threshold": 127.0,
    "erosion_kernel": 3,
    "erosion_iter": 1,
    "dilation_kernel": 5,
    "dilation_iter": 2,
    "min_area": 80.0,
    "max_area": 50000.0,
}

# 首次检测必须落在圆筒区域内，才允许锚定相对窗口。
FIRST_SEEN = "arena"

SPEC = {
    "key": KEY,
    "label": "强迫游泳实验 / FST",
    "short": "FST",
    "parameters": [
        parameter("cylinder_diameter_cm", "圆筒实际内径", "number", 10.0,
                  unit="cm", minimum=0.01, step="any"),
        parameter("auto_threshold", "自动估计不动阈值", "checkbox", True,
                  section="实验参数"),
        parameter("threshold_sensitivity", "自动阈值灵敏度", "number", 0.0,
                  minimum=-3, maximum=3, step=0.1, section="实验参数"),
        parameter("immobility_threshold", "不动低阈值", "number", 0.08,
                  minimum=0.001, step="any", section="实验参数"),
        parameter("active_threshold", "活动高阈值", "number", 0.108,
                  minimum=0.001, step="any", section="实验参数"),
        parameter("smoothing_window_s", "运动评分平滑时间", "number", 0.5,
                  unit="s", minimum=0, step=0.1, section="实验参数"),
        parameter("min_immobility_duration_s", "最短不动时长", "number", 1.0,
                  unit="s", minimum=0, step=0.1, section="实验参数"),
        parameter("min_active_duration_s", "最短活动时长", "number", 0.3,
                  unit="s", minimum=0, step=0.1, section="实验参数"),
        parameter("gap_merge_s", "合并不动片段间的活动间隙", "number", 0.5,
                  unit="s", minimum=0, step=0.1, section="实验参数"),
        parameter("bin_seconds", "时间分箱长度", "number", 60.0, unit="s",
                  minimum=0.01, step="any", section="实验参数"),
    ],
    "geometry": [
        geometry_field("arena", "圆筒区域", "polygon", min_points=3),
        geometry_field("diameter_line", "圆筒内径线", "line", count=2),
        geometry_field("mask", "检测遮罩", "polygon", min_points=3,
                       optional=True),
        geometry_field("waterline", "水面线", "line", count=2, optional=True),
    ],
    "metrics": [
        metric("immobility_time_s", "不动时间", "s"),
        metric("active_time_s", "活动时间", "s"),
        metric("uncertain_time_s", "不确定时间", "s"),
        metric("immobility_percent", "不动比例", "%"),
        metric("immobility_bouts", "不动片段数", "次"),
        metric("latency_to_immobility_s", "首次不动潜伏期", "s"),
        metric("detected_rate_percent", "检测时间比例", "%"),
    ],
    "template_keys": ["cylinder_diameter_cm"],
}

CANCEL_STRIDE = 512
TIME_EPS = 1e-9


def _check_cancel(cancel: Callable[[], bool] | None) -> None:
    if cancel is not None and cancel():
        raise ValidationError("分析已取消", "job")


# -- 参数与几何校验 -----------------------------------------------------

def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not is_number(value):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def validate_unit(options: Mapping[str, Any] | None,
                  geometry: Mapping[str, Any] | None,
                  provided: Iterable[str] | None = None, *,
                  complete: bool = False) -> list[dict]:
    """参数范围与几何契约；未完成的草稿允许先缺几何。"""

    values = options if isinstance(options, Mapping) else {}
    provided = set(OPTION_KEYS if provided is None else provided)
    issues: list[dict] = []

    if "auto_threshold" in provided and not isinstance(
            values.get("auto_threshold"), bool):
        issues.append({"field": "options.auto_threshold",
                       "message": "必须是布尔值"})

    for key in POSITIVE_KEYS:
        if key not in provided:
            continue
        value = _finite(values.get(key))
        if value is None:
            issues.append({"field": f"options.{key}", "message": "必须是有限数值"})
        elif value <= 0:
            issues.append({"field": f"options.{key}", "message": "必须大于 0"})

    for key in NONNEGATIVE_KEYS:
        if key not in provided:
            continue
        value = _finite(values.get(key))
        if value is None or value < 0:
            issues.append({"field": f"options.{key}",
                           "message": "必须是 ≥ 0 的有限数值"})

    if "threshold_sensitivity" in provided:
        sensitivity = _finite(values.get("threshold_sensitivity"))
        if sensitivity is None:
            issues.append({"field": "options.threshold_sensitivity",
                           "message": "必须是有限数值"})
        elif not -3.0 <= sensitivity <= 3.0:
            issues.append({"field": "options.threshold_sensitivity",
                           "message": "灵敏度必须在 -3 到 3 之间"})

    if values.get("auto_threshold") is False:
        low = _finite(values.get("immobility_threshold"))
        high = _finite(values.get("active_threshold"))
        if low is not None and high is not None and not 0 < low < high:
            issues.append({"field": "options.active_threshold",
                           "message": "关闭自动阈值时必须满足 "
                                      "0 < 不动阈值 < 活动阈值"})

    issues.extend(validate_geometry(geometry, require=complete))
    return issues


def _line_points(shape: Mapping[str, Any] | None) -> list[tuple[float, float]]:
    points = activity.shape_points(shape)
    return points if len(points) == 2 else []


def validate_geometry(geometry: Mapping[str, Any] | None,
                      *, require: bool = False) -> list[dict]:
    geometry = geometry or {}
    issues: list[dict] = []

    arena = geometry.get("arena")
    if arena is None:
        if require:
            issues.append({"field": "geometry.arena", "message": "缺少圆筒区域"})
    elif not isinstance(arena, Mapping) or arena.get("type") != "polygon":
        # 线/椭圆等形状会被检测区域栅格成一条细线，不能当圆筒区域。
        issues.append({"field": "geometry.arena",
                       "message": "圆筒区域必须是多边形"})
    else:
        points = activity.shape_points(arena)
        if len(points) < 3:
            issues.append({"field": "geometry.arena",
                           "message": "圆筒区域至少需要 3 个有限点"})
        elif abs(_polygon_area(points)) <= 1e-9:
            issues.append({"field": "geometry.arena",
                           "message": "圆筒区域面积必须大于 0"})

    line = geometry.get("diameter_line")
    if line is None:
        if require:
            issues.append({"field": "geometry.diameter_line",
                           "message": "缺少圆筒内径线"})
    else:
        points = _line_points(line)
        if not points:
            issues.append({"field": "geometry.diameter_line",
                           "message": "圆筒内径线需要两个有限点"})
        elif math.hypot(points[1][0] - points[0][0],
                        points[1][1] - points[0][1]) <= 1e-9:
            issues.append({"field": "geometry.diameter_line",
                           "message": "圆筒内径线两个端点不能重合"})

    waterline = geometry.get("waterline")
    if waterline is not None:
        points = _line_points(waterline)
        if not points:
            issues.append({"field": "geometry.waterline",
                           "message": "水面线需要两个有限点"})
        elif math.hypot(points[1][0] - points[0][0],
                        points[1][1] - points[0][1]) <= 1e-9:
            issues.append({"field": "geometry.waterline",
                           "message": "水面线两个端点不能重合"})

    mask = geometry.get("mask")
    if mask is not None:
        points = activity.shape_points(mask)
        if len(points) < 3:
            issues.append({"field": "geometry.mask",
                           "message": "检测遮罩至少需要 3 个有限点"})
        elif abs(_polygon_area(points)) <= 1e-9:
            issues.append({"field": "geometry.mask",
                           "message": "检测遮罩面积必须大于 0"})
    return issues


def _polygon_area(points: Sequence[Sequence[float]]) -> float:
    total = 0.0
    for index, point in enumerate(points):
        nxt = points[(index + 1) % len(points)]
        total += float(point[0]) * float(nxt[1]) - float(nxt[0]) * float(point[1])
    return total / 2.0


def validate(job: Any) -> list[dict]:
    issues: list[dict] = []
    for unit in job.get("units") or []:
        for issue in validate_unit(unit.get("options") or {},
                                   unit.get("geometry") or {},
                                   complete=True):
            entry = dict(issue)
            entry["field"] = f"units[{unit.get('id')}].{issue['field']}"
            issues.append(entry)
    return issues


# -- 尺度与分数 ---------------------------------------------------------

def scale_of(geometry: Mapping[str, Any] | None,
             diameter_cm: float) -> tuple[float, float]:
    """实际内径 D cm ÷ 内径线像素长 → (cm/px, 线长 px)。"""

    points = _line_points((geometry or {}).get("diameter_line"))
    if len(points) != 2:
        raise ValidationError("圆筒内径线需要两个有限点",
                              "geometry.diameter_line")
    length = math.hypot(points[1][0] - points[0][0],
                        points[1][1] - points[0][1])
    if length <= 0:
        raise ValidationError("圆筒内径线两个端点不能重合",
                              "geometry.diameter_line")
    diameter = float(diameter_cm)
    if diameter <= 0:
        raise ValidationError("圆筒实际内径必须大于 0",
                              "options.cylinder_diameter_cm")
    return diameter / length, length


def centroid_speed(distance_px: float, cm_per_px: float,
                   elapsed_s: float) -> float:
    """质心速度 cm/s：像素位移 × cm/px ÷ 真实帧间秒数。"""

    if elapsed_s <= 0:
        return 0.0
    return float(distance_px) * float(cm_per_px) / float(elapsed_s)


def shape_component(rate: float) -> float:
    return min(1.0, max(0.0, float(rate)) / 3.0)


def motion_score(flow_cm_s: float, speed_cm_s: float, shape_rate: float,
                 diameter_cm: float) -> float:
    """m = .60·F/D + .25·v/D + .15·min(1, a/3)；D 为实际内径。"""

    diameter = max(float(diameter_cm), 1e-9)
    return (0.60 * float(flow_cm_s) / diameter
            + 0.25 * float(speed_cm_s) / diameter
            + 0.15 * shape_component(shape_rate))


def resolve_thresholds(scores: Sequence[float], fps: float,
                       options: Mapping[str, Any]) -> dict:
    """自动/手动阈值：样本不足时用配置低阈值与同一高阈值公式。"""

    manual_low = float(options.get("immobility_threshold") or 0.0)
    sensitivity = float(options.get("threshold_sensitivity") or 0.0)
    values = [float(value) for value in scores
              if is_number(value) and math.isfinite(float(value))]
    samples = len(values)
    if not bool(options.get("auto_threshold")):
        return {"low": manual_low,
                "high": float(options.get("active_threshold") or 0.0),
                "estimated": False, "samples": 0}

    need = max(10, int(math.ceil(2.0 * max(float(fps), 0.0))))
    if samples < need:
        low = manual_low
        high = max(1.18 * low, low + 0.002)
        return {"low": low, "high": high, "estimated": False,
                "samples": samples}

    array = np.asarray(values, dtype=np.float64)
    q20 = float(np.percentile(array, 20))
    q80 = float(np.percentile(array, 80))
    base = (max(0.005, 1.25 * q20) if q80 - q20 < 1e-6
            else q20 + 0.25 * (q80 - q20))
    low = max(0.001, base * (1.0 + 0.12 * sensitivity))
    high = max(1.18 * low, low + 0.002)
    return {"low": float(low), "high": float(high), "estimated": True,
            "samples": samples}


# -- 捕获 ----------------------------------------------------------------

def _extra(method: str | None) -> dict:
    return {
        "detection_method": method,
        "feature_ready": False,
        "flow_p90_cm_s": None,
        "flow_mean_cm_s": None,
        "centroid_speed_cm_s": None,
        "shape_change_rate": None,
        "shape_change_component": None,
        "motion_score": None,
        "smoothed_motion_score": None,
        "threshold_low": None,
        "threshold_high": None,
        "state": None,
        "realtime_state": None,
    }


def _detection_of(candidate: Mapping[str, Any] | None,
                  reason: str | None) -> Detection:
    if candidate is None:
        return Detection(missing_reason=reason or "no_candidate")
    contour = [[float(point[0]), float(point[1])]
               for point in np.asarray(candidate["contour"]).reshape(-1, 2)]
    return Detection(center=list(candidate["center"]),
                     quality=float(candidate["quality"]),
                     contour=contour, mask=candidate["mask"],
                     area=float(candidate["area"]),
                     bbox=[float(value) for value in candidate["bbox"]])


def capture(context: Any, units: Sequence[Mapping[str, Any]]):
    """源帧顺序：每只鼠一次检测，保存相邻成对特征（cm/s、cm、1/s）。"""

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

    run_key = (job.get("run_id"), job.get("job_id"), KEY)
    captured: dict[Any, list[dict]] = {unit["id"]: [] for unit in units}
    state: dict[Any, dict] = {}
    progress_span = float(metadata.get("duration_s") or 0.0)

    for frame in source.iter_frames():
        if context.cancelled():
            raise ValidationError("分析已取消", "job")
        image = frame.image
        if image is None:
            continue
        height, width = image.shape[:2]
        gray = image if image.ndim == 2 else cv2.cvtColor(
            image, cv2.COLOR_BGR2GRAY)
        for unit in units:
            unit_id = unit["id"]
            options = unit.get("options") or {}
            geometry = unit.get("geometry") or {}
            settings = unit.get("detection") or {}
            entry = state.get(unit_id)
            if entry is None:
                region, _ = detection.geometry_region(geometry, KEY, options,
                                                      width, height)
                background = None
                if settings.get("bg_method") in ("temporal", "static"):
                    background = detection.prepare_background(
                        settings, asset, job,
                        int(job.get("orientation", 0) or 0), width, height)
                entry = {
                    "detector": detection.shared_detector(
                        (run_key, unit_id), settings, background=background),
                    "region": region,
                    "arena": activity.shape_points(geometry.get("arena")),
                    "scale": scale_of(
                        geometry,
                        float(options.get("cylinder_diameter_cm") or 0.0))[0],
                    "diameter": float(
                        options.get("cylinder_diameter_cm") or 0.0),
                    "prev": None,
                }
                state[unit_id] = entry
            detector = entry["detector"]
            candidates, reason = detection.find_candidates(
                detector.foreground(gray), entry["region"], settings)
            candidate = candidates[0] if candidates else None
            prev = entry["prev"]
            body = None
            if candidate is not None:
                elapsed = None
                previous_area = None
                if prev is not None and prev.get("body") is not None:
                    elapsed = float(frame.time_s) - float(prev["time_s"])
                    previous_area = float(prev["body"]["body_area_px"])
                body = activity.body_features(
                    candidate, entry["arena"], elapsed, previous_area)
            source_start = float(frame.interval_start_s)
            source_end = float(frame.interval_end_s)
            pair = None
            if (candidate is not None and body is not None and prev is not None
                    and prev.get("body") is not None
                    and prev.get("interval_start") is not None
                    and float(prev["interval_end"]) > float(prev["interval_start"])
                    and abs(source_start - float(prev["interval_end"])) < 1e-7
                    and float(frame.dt_s) > 0 and source_end > source_start):
                current = activity.flow_features(
                    prev["gray"], gray, candidate["mask"], candidate["bbox"],
                    prev["center"], candidate["center"],
                    mm_per_px=entry["scale"],
                    fps=float(frame.analysis_fps),
                    elapsed_s=float(frame.time_s) - float(prev["time_s"]),
                )
                if current is not None and is_number(body["body_area_rate_s"]):
                    rate = float(body["body_area_rate_s"])
                    flow = float(current["flow_p90_mm_s"])
                    speed = float(current["speed_mm_s"])
                    pair = {
                        "flow_p90_cm_s": flow,
                        "flow_mean_cm_s": float(current["flow_mean_mm_s"]),
                        "centroid_speed_cm_s": speed,
                        "shape_change_rate": rate,
                        "shape_change_component": shape_component(rate),
                        "motion_score": motion_score(
                            flow, speed, rate, entry["diameter"]),
                        "feature_ready": True,
                    }
            result = _detection_of(candidate, reason)
            row = timeline.base_row(
                {"id": unit_id, "name": unit.get("name"),
                 "animal_id": unit.get("animal_id")},
                frame, result, extra=_extra(settings.get("bg_method")))
            if pair is not None:
                row.update(pair)
                row["pair_from_frame"] = prev["frame"]
                row["pair_previous_time_s"] = float(prev["time_s"])
            captured[unit_id].append(row)
            entry["prev"] = {
                "frame": frame.index,
                "time_s": float(frame.time_s),
                "interval_start": source_start,
                "interval_end": source_end,
                "center": list(candidate["center"]) if candidate else None,
                "bbox": candidate["bbox"] if candidate else None,
                "body": body,
                "gray": gray.copy(),
            }
        context.progress(min(0.85, 0.1 + 0.75 * max(0.0, float(frame.time_s)) /
                             max(1e-9, progress_span or 1.0)))

    return captured


# -- 平滑、阈值与状态 ---------------------------------------------------

def _groups(rows: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    groups: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []
    for row in rows:
        if not row.get("feature_ready"):
            if current:
                groups.append(current)
                current = []
            continue
        if current and abs(float(current[-1]["interval_end_s"])
                           - float(row["interval_start_s"])) > 1e-7:
            groups.append(current)
            current = []
        current.append(row)
    if current:
        groups.append(current)
    return groups


def smooth_scores(group: list[Mapping[str, Any]], window_s: float,
                  cancel: Callable[[], bool] | None = None) -> None:
    """以源时刻为中心的曝光交集加权均值；w=0 用原分。"""

    window = float(window_s)
    left = 0
    right = 0
    count = len(group)
    for index, row in enumerate(group):
        if index % CANCEL_STRIDE == 0:
            _check_cancel(cancel)
        score = row.get("motion_score")
        raw = float(score) if is_number(score) else None
        if window <= 0 or raw is None:
            row["smoothed_motion_score"] = raw
            continue
        center = float(row["time_s"])
        low = center - window / 2.0
        high = center + window / 2.0
        while (left < count
               and float(group[left]["interval_end_s"]) <= low + 1e-9):
            left += 1
        if right < left:
            right = left
        while (right < count
               and float(group[right]["interval_start_s"]) < high - 1e-9):
            right += 1
        total = 0.0
        weighted = 0.0
        for other in group[left:right]:
            value = other.get("motion_score")
            if not is_number(value):
                continue
            start = max(float(other["interval_start_s"]), low)
            end = min(float(other["interval_end_s"]), high)
            if end - start > 1e-9:
                total += end - start
                weighted += float(value) * (end - start)
        row["smoothed_motion_score"] = (weighted / total if total > 1e-9
                                        else raw)


def _runs(group: Sequence[Mapping[str, Any]]) -> list[dict]:
    runs: list[dict] = []
    for row in group:
        state = row.get("state")
        if runs and runs[-1]["state"] == state:
            runs[-1]["rows"].append(row)
            runs[-1]["end"] = float(row["interval_end_s"])
        else:
            runs.append({"state": state, "rows": [row],
                         "start": float(row["interval_start_s"]),
                         "end": float(row["interval_end_s"])})
    return runs


def _duration(run: Mapping[str, Any]) -> float:
    return float(run["end"]) - float(run["start"])


def _paint(run: Mapping[str, Any], state: str) -> None:
    for row in run["rows"]:
        row["state"] = state
        row["state_group"] = state


def offline_states(group: list[Mapping[str, Any]], low: float, high: float,
                   gap_merge_s: float, min_immobility_s: float,
                   min_active_s: float,
                   cancel: Callable[[], bool] | None = None) -> None:
    """初始 active；≤低不动、≥高活动、中间保持，再按固定顺序修整。"""

    state = "active"
    for index, row in enumerate(group):
        if index % CANCEL_STRIDE == 0:
            _check_cancel(cancel)
        score = row.get("smoothed_motion_score")
        value = float(score) if is_number(score) else 0.0
        if value <= float(low):
            state = "immobile"
        elif value >= float(high):
            state = "active"
        row["state"] = state
        row["state_group"] = state

    runs = _runs(group)
    for position in range(1, len(runs) - 1):
        run = runs[position]
        if run["state"] != "active":
            continue
        if (runs[position - 1]["state"] != "immobile"
                or runs[position + 1]["state"] != "immobile"):
            continue
        if _duration(run) <= float(gap_merge_s) + TIME_EPS:
            _paint(run, "immobile")

    runs = _runs(group)
    for run in runs:
        if run["state"] != "immobile":
            continue
        if _duration(run) < float(min_immobility_s) - TIME_EPS:
            _paint(run, "active")

    runs = _runs(group)
    for position in range(1, len(runs) - 1):
        run = runs[position]
        if run["state"] != "active":
            continue
        if (runs[position - 1]["state"] != "immobile"
                or runs[position + 1]["state"] != "immobile"):
            continue
        if _duration(run) < float(min_active_s) - TIME_EPS:
            _paint(run, "immobile")


def realtime_states(group: list[Mapping[str, Any]], low: float, high: float,
                    min_immobility_s: float, min_active_s: float,
                    cancel: Callable[[], bool] | None = None) -> None:
    """实时列：原始分数定候选，实际过去曝光满足最短时长才确认。"""

    candidate = "active"
    confirmed: str | None = None
    accumulated = 0.0
    previous: Mapping[str, Any] | None = None
    for index, row in enumerate(group):
        if index % CANCEL_STRIDE == 0:
            _check_cancel(cancel)
        score = row.get("motion_score")
        value = float(score) if is_number(score) else 0.0
        if value <= float(low):
            current = "immobile"
        elif value >= float(high):
            current = "active"
        else:
            current = confirmed if confirmed is not None else "active"
        if current != candidate:
            candidate = current
            accumulated = 0.0
        elif previous is not None and abs(
                float(previous["interval_end_s"])
                - float(row["interval_start_s"])) < 1e-7:
            accumulated += max(0.0, float(previous.get("dt_s") or 0.0))
        need = (float(min_immobility_s) if candidate == "immobile"
                else float(min_active_s))
        if confirmed != candidate and accumulated >= need - TIME_EPS:
            confirmed = candidate
        # 候选尚未确认前实时状态保持 unknown；中间值沿用已确认状态。
        row["realtime_state"] = (candidate if confirmed == candidate
                                 else "unknown")
        previous = row


def derive_features(rows: Sequence[dict], options: Mapping[str, Any],
                    cancel: Callable[[], bool] | None = None) -> dict:
    """阈值 → 平滑 → 离线状态 → 实时列；未知与窗外不算状态。"""

    ordered = sorted(rows, key=lambda item: (item["time_s"], item["frame"]))
    measured = [row for row in ordered
                if row.get("in_window") and row.get("feature_ready")]
    fps = 0.0
    if measured:
        fps = float(measured[0].get("analysis_fps") or 0.0)
    info = resolve_thresholds(
        [row["motion_score"] for row in measured], fps, options)
    low = float(info["low"])
    high = float(info["high"])

    gap = max(0.0, float(options.get("gap_merge_s") or 0.0))
    min_immobility = max(0.0, float(
        options.get("min_immobility_duration_s") or 0.0))
    min_active = max(0.0, float(options.get("min_active_duration_s") or 0.0))
    window = max(0.0, float(options.get("smoothing_window_s") or 0.0))
    groups = _groups(measured)
    for group in groups:
        _check_cancel(cancel)
        smooth_scores(group, window, cancel)
        offline_states(group, low, high, gap, min_immobility, min_active,
                       cancel)
        realtime_states(group, low, high, min_immobility, min_active, cancel)

    for row in ordered:
        row["threshold_low"] = low
        row["threshold_high"] = high
        if row.get("feature_ready") and row.get("in_window"):
            continue
        row["state"] = "unknown" if row.get("in_window") else "outside"
        row["state_group"] = row["state"]
        row["smoothed_motion_score"] = None
        row["realtime_state"] = "unknown"
    return info


# -- 汇总、事件与分箱 ---------------------------------------------------

def build_events(rows: Sequence[Mapping[str, Any]], unit_id: str,
                 unit_name: str) -> list[dict]:
    """最终不动片段；离线分段不声称实时确认。"""

    events: list[dict] = []
    sequence = 0
    previous: Mapping[str, Any] | None = None
    for row in sorted(rows, key=lambda item: (item["time_s"], item["frame"])):
        if row.get("state") != "immobile":
            previous = None
            continue
        adjacent = (previous is not None
                    and abs(float(previous["interval_end_s"])
                            - float(row["interval_start_s"])) < 1e-7)
        if not adjacent:
            sequence += 1
            events.append({
                "event_id": f"{unit_id}:immobile_bout:{sequence}",
                "unit_id": unit_id,
                "unit_name": unit_name,
                "kind": "immobile_bout",
                "time_s": float(row["interval_start_s"]),
                "end_s": None,
                "confirmed_s": None,
                "target": None,
                "source": "automatic",
                "frame": row["frame"],
            })
        events[-1]["end_s"] = float(row["interval_end_s"])
        previous = row
    return events


def _bout_spans(rows: Sequence[Mapping[str, Any]]) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    previous: Mapping[str, Any] | None = None
    for row in sorted(rows, key=lambda item: (item["time_s"], item["frame"])):
        if row.get("state") != "immobile" or not row.get("in_window"):
            previous = None
            continue
        adjacent = (previous is not None
                    and abs(float(previous["interval_end_s"])
                            - float(row["interval_start_s"])) < 1e-7)
        if not adjacent:
            spans.append((float(row["interval_start_s"]),
                          float(row["interval_end_s"])))
        else:
            spans[-1] = (spans[-1][0], float(row["interval_end_s"]))
        previous = row
    return spans


def summarize(rows: Sequence[Mapping[str, Any]], unit: Mapping[str, Any],
              window: timeline.Window, info: Mapping[str, Any],
              events: Sequence[Mapping[str, Any]]) -> dict:
    total = active = immobile = detected_time = 0.0
    total_frames = detected_frames = 0
    scores: list[float] = []
    for row in rows:
        if not row.get("in_window"):
            continue
        span = max(0.0, float(row.get("dt_s") or 0.0))
        total += span
        total_frames += 1
        if row.get("detected"):
            detected_time += span
            detected_frames += 1
        state = row.get("state")
        if state == "active":
            active += span
        elif state == "immobile":
            immobile += span
        score = row.get("motion_score")
        if is_number(score):
            scores.append(float(score))
    uncertain = max(0.0, total - active - immobile)

    def percent(value: float, denominator: float) -> float | None:
        return timeline.ratio(value * 100.0, denominator)

    spans = _bout_spans(rows)
    durations = [end - start for start, end in spans]
    latency = None
    if spans and window.known:
        latency = float(spans[0][0]) - float(window.start_s)

    summary = {
        "unit_id": unit.get("id"),
        "unit_name": unit.get("name"),
        "mode": "fst",
        "total_time_s": total,
        "active_time_s": active,
        "immobility_time_s": immobile,
        "uncertain_time_s": uncertain,
        "immobility_percent": percent(immobile, total),
        "active_percent": percent(active, total),
        "uncertain_percent": percent(uncertain, total),
        "detected_time_s": detected_time,
        "detected_rate_percent": percent(detected_time, total),
        "immobility_percent_detected": percent(immobile, detected_time),
        "detected_frames": detected_frames,
        "total_frames": total_frames,
        "immobility_bouts": len(spans),
        "longest_immobility_bout_s": max(durations) if durations else 0.0,
        "latency_to_immobility_s": latency,
        "mean_motion_score": (sum(scores) / len(scores) if scores else None),
        "max_motion_score": max(scores) if scores else None,
        "analysis_window_start_s": window.start_s if window.known else None,
        "analysis_window_end_s": window.end_s if window.known else None,
        "thresholds": {"low": float(info["low"]),
                       "high": float(info["high"])},
        "threshold_estimated": bool(info["estimated"]),
        "threshold_samples": int(info["samples"]),
    }
    for key in OPTION_KEYS:
        summary[key] = (unit.get("options") or {}).get(key)
    return summary


def build_bins(rows: Sequence[Mapping[str, Any]],
               window: timeline.Window, bin_seconds: float,
               unit: Mapping[str, Any]) -> list[dict]:
    """按单位窗起点分箱；曝光跨界拆分，分数按有效曝光加权。"""

    if not window.known or window.duration_s <= 0:
        return []
    width = float(bin_seconds)
    if width <= 0:
        return []
    origin = float(window.start_s)
    duration = window.duration_s
    count = max(1, int(math.ceil((duration - TIME_EPS) / width)))
    buckets = [{"active": 0.0, "immobile": 0.0, "unknown": 0.0,
                "score_sum": 0.0, "score_weight": 0.0}
               for _ in range(count)]
    for row in rows:
        if not row.get("in_window"):
            continue
        pieces = timeline.split_interval(row["interval_start_s"],
                                         row["interval_end_s"], origin, width)
        score = row.get("motion_score")
        valid = is_number(score)
        for index, low, high in pieces:
            if not 0 <= index < count:
                continue
            span = high - low
            bucket = buckets[index]
            state = row.get("state")
            if state == "active":
                bucket["active"] += span
            elif state == "immobile":
                bucket["immobile"] += span
            else:
                bucket["unknown"] += span
            if valid:
                bucket["score_sum"] += float(score) * span
                bucket["score_weight"] += span

    bins: list[dict] = []
    for index, bucket in enumerate(buckets):
        start = index * width
        end = min(start + width, duration)
        duration_s = end - start
        immobility = bucket["immobile"]
        bins.append({
            "unit_id": unit.get("id"),
            "unit_name": unit.get("name"),
            "bin_index": index + 1,
            "start_s": start,
            "end_s": end,
            "duration_s": duration_s,
            "active_time_s": bucket["active"],
            "immobility_time_s": immobility,
            "unknown_time_s": bucket["unknown"],
            "immobility_percent": timeline.ratio(immobility * 100.0,
                                                 duration_s),
            "mean_motion_score": (bucket["score_sum"] / bucket["score_weight"]
                                  if bucket["score_weight"] > 1e-9 else None),
        })
    return bins


# -- 分析入口 ------------------------------------------------------------

def _clear_pair(row: dict) -> None:
    row["feature_ready"] = False
    for key in ("flow_p90_cm_s", "flow_mean_cm_s", "centroid_speed_cm_s",
                "shape_change_rate", "shape_change_component", "motion_score"):
        row[key] = None


def _filter_pairs(rows: Sequence[dict], window: timeline.Window) -> None:
    """配对特征必须前后两条都真实、都在窗内且不早于单位窗起点。"""

    by_frame = {row["frame"]: row for row in rows}
    for row in rows:
        if not row.get("feature_ready"):
            continue
        previous = by_frame.get(row.get("pair_from_frame"))
        keep = (window.known and row.get("in_window")
                and previous is not None and previous.get("in_window")
                and float(previous["time_s"]) >= float(window.start_s) - 1e-9
                and float(row["time_s"]) >= float(window.start_s) - 1e-9)
        if not keep:
            _clear_pair(row)


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

    observations: list[dict] = []
    summary: list[dict] = []
    events: list[dict] = []
    bins: list[dict] = []
    unit_metadata: dict[str, dict] = {}
    warnings: list[str] = []

    for unit in units:
        if context.cancelled():
            raise ValidationError("分析已取消", "job")
        unit_id = unit["id"]
        geometry = unit.get("geometry") or {}
        options = unit.get("options") or {}
        first_seen = timeline.first_seen_in_window(
            KEY, captured[unit_id], video_window, geometry)
        window = timeline.resolve_unit_window(
            unit.get("window"), video_window, first_seen)
        rows = timeline.clip_rows(captured[unit_id], window)
        _filter_pairs(rows, window)
        info = derive_features(rows, options, context.cancelled)
        unit_events = build_events(rows, unit_id, unit.get("name") or "")
        unit_summary = summarize(rows, unit, window, info, unit_events)
        unit_bins = build_bins(rows, window, float(
            options.get("bin_seconds") or 0.0), unit)
        observations.extend(rows)
        summary.append(unit_summary)
        events.extend(unit_events)
        bins.extend(unit_bins)
        scale, line_px = scale_of(
            geometry, float(options.get("cylinder_diameter_cm") or 0.0))
        unit_metadata[unit_id] = {
            "cm_per_px": scale,
            "diameter_line_px": line_px,
            "cylinder_diameter_cm": float(options["cylinder_diameter_cm"]),
            "threshold_low": float(info["low"]),
            "threshold_high": float(info["high"]),
            "threshold_estimated": bool(info["estimated"]),
            "threshold_samples": int(info["samples"]),
            "motion_score_formula": ("m = .60·F/D + .25·v/D + "
                                     ".15·min(1, a/3)"),
            "flow_time_source": "分析采样帧率（光流位移×cm/px×Hz）",
            "centroid_speed_time_source": "相邻源帧真实时刻差",
            "waterline_role": "仅保存与显示，不参与评分、阈值或浮动判定",
        }

    return AnalysisResult(
        observations=timeline.sort_rows(observations),
        summary=summary,
        events=events,
        bins=bins,
        tables={},
        warnings=warnings,
        metadata={"assay": KEY, "units": unit_metadata, "warnings": warnings},
    )


def preview_analysis(*, asset: Mapping[str, Any], video: Mapping[str, Any],
                     unit: Mapping[str, Any], start_s: float,
                     end_s: float, time_s: float | None = None) -> dict:
    """行为预览：对实际片段跑同一捕获→阈值→状态→汇总核心。"""

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
    captured = capture(AnalysisContext(job=job, source=source), [unit])
    rows_all = captured[unit["id"]]
    video_window = timeline.resolve_video_window(
        window, rows_all, metadata.get("duration_s"))
    geometry = unit.get("geometry") or {}
    first_seen = timeline.first_seen_in_window(
        KEY, rows_all, video_window, geometry)
    unit_window = timeline.resolve_unit_window(
        unit.get("window"), video_window, first_seen)
    rows = timeline.clip_rows(rows_all, unit_window)
    _filter_pairs(rows, unit_window)
    info = derive_features(rows, options)
    events = build_events(rows, unit["id"], unit.get("name") or "")
    summary = summarize(rows, unit, unit_window, info, events)

    state_at_time = None
    if time_s is not None:
        for row in rows:
            if (float(row["interval_start_s"]) - 1e-9 <= float(time_s)
                    < float(row["interval_end_s"]) - 1e-9):
                state_at_time = row.get("state")
                break
    behavior = {
        "mode": "fst",
        "start_s": float(start_s),
        "end_s": float(end_s),
        "window_start_s": (unit_window.start_s if unit_window.known else None),
        "window_end_s": (unit_window.end_s if unit_window.known else None),
        "total_time_s": summary["total_time_s"],
        "active_time_s": summary["active_time_s"],
        "immobility_time_s": summary["immobility_time_s"],
        "uncertain_time_s": summary["uncertain_time_s"],
        "immobility_percent": summary["immobility_percent"],
        "rows": len(rows),
        "events": len(events),
        "thresholds": summary["thresholds"],
        "threshold_estimated": bool(summary["threshold_estimated"]),
        "threshold_samples": int(info["samples"]),
        "state_at_time": state_at_time,
    }
    behavior["threshold_note"] = (
        "行为预览为短段阈值估计，与全程分析可能不同"
        if bool(options.get("auto_threshold"))
        else "行为预览为短段结果，阈值按配置")
    return behavior
