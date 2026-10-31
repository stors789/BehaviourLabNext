"""TST 悬尾实验：classic 模式的成对运动、阈值滞回、摆动与最终时间线。

本块只实现 classic；lis 参数照常保存与校验，但最终校验/开始任务会拒绝，
不会静默按 classic 计算。体长线固定 60 mm，尺度不从身体框反推。
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from .. import activity, detection, media, timeline
from ..schema import AnalysisResult, Detection, ValidationError, is_number
from .base import geometry_field, metric, parameter

KEY = "tst"
MODES = ("classic", "lis")

DEFAULTS = {
    "mode": "classic",
    "auto_behavior_threshold": False,
    "behavior_threshold_sensitivity": 0.0,
    "flow_thr_hi": 50.0,
    "flow_thr_lo": 35.0,
    "speed_std_thr_hi": 40.0,
    "speed_std_thr_lo": 20.0,
    "lis_rotate_align": False,
    "lis_rescue_enabled": True,
    "lis_score_gap_s": 2.0,
    "lis_evidence_gap_s": 0.5,
    "lis_rescue_gap_s": 1.0,
    "lis_min_struggle_s": 0.3,
    "lis_rescue_before_s": 0.7,
    "lis_rescue_after_s": 0.3,
}

OPTION_KEYS = tuple(DEFAULTS)
BOOLEAN_KEYS = ("auto_behavior_threshold", "lis_rotate_align",
                "lis_rescue_enabled")
SECONDS_KEYS = ("lis_score_gap_s", "lis_evidence_gap_s", "lis_rescue_gap_s",
                "lis_min_struggle_s", "lis_rescue_before_s",
                "lis_rescue_after_s")

# 悬尾：亮背景暗鼠；形态学与面积按本块工厂默认。
DETECTION_DEFAULTS = {
    "bg_method": "threshold",
    "bg_type": "light",
    "gray_threshold": 127.0,
    "erosion_kernel": 3,
    "erosion_iter": 2,
    "dilation_kernel": 5,
    "dilation_iter": 3,
    "min_area": 100.0,
    "max_area": 50000.0,
}

FIRST_SEEN = "arena"

SPEC = {
    "key": KEY,
    "label": "尾悬实验 / TST",
    "short": "TST",
    "parameters": [
        parameter("mode", "行为算法", "select", "classic",
                  choices=[{"value": "classic", "label": "经典行为"},
                           {"value": "lis", "label": "微动残差"}]),
        parameter("auto_behavior_threshold", "自动估计经典阈值", "checkbox",
                  False, section="实验参数"),
        parameter("behavior_threshold_sensitivity", "经典自动阈值灵敏度",
                  "number", 0.0, minimum=-3, maximum=3, step=0.1,
                  section="实验参数"),
        parameter("flow_thr_hi", "经典光流高阈值", "number", 50.0,
                  unit="mm/s", minimum=0.001, step=1, section="实验参数"),
        parameter("flow_thr_lo", "经典光流低阈值", "number", 35.0,
                  unit="mm/s", minimum=0.001, step=1, section="实验参数"),
        parameter("speed_std_thr_hi", "经典速度波动高阈值", "number", 40.0,
                  unit="mm/s", minimum=0.001, step=1, section="实验参数"),
        parameter("speed_std_thr_lo", "经典速度波动低阈值", "number", 20.0,
                  unit="mm/s", minimum=0.001, step=1, section="实验参数"),
        parameter("lis_rotate_align", "残差旋转对齐", "checkbox", False,
                  section="实验参数"),
        parameter("lis_rescue_enabled", "启用局部微动补救", "checkbox", True,
                  section="实验参数"),
        parameter("lis_score_gap_s", "残差评分分支间隙填充", "number", 2.0,
                  unit="s", minimum=0, step=0.1, section="实验参数"),
        parameter("lis_evidence_gap_s", "残差证据分支间隙填充", "number", 0.5,
                  unit="s", minimum=0, step=0.1, section="实验参数"),
        parameter("lis_rescue_gap_s", "局部微动间隙填充", "number", 1.0,
                  unit="s", minimum=0, step=0.1, section="实验参数"),
        parameter("lis_min_struggle_s", "残差最短挣扎段", "number", 0.3,
                  unit="s", minimum=0, step=0.1, section="实验参数"),
        parameter("lis_rescue_before_s", "局部微动向前扩展", "number", 0.7,
                  unit="s", minimum=0, step=0.1, section="实验参数"),
        parameter("lis_rescue_after_s", "局部微动向后扩展", "number", 0.3,
                  unit="s", minimum=0, step=0.1, section="实验参数"),
    ],
    "geometry": [
        geometry_field("arena", "小鼠活动区域", "polygon", min_points=3),
        geometry_field("body_line", "小鼠体长线（固定 60 mm）", "line", count=2),
        geometry_field("mask", "检测遮罩", "polygon", min_points=3,
                       optional=True),
    ],
    "metrics": [
        metric("immobility_time_s", "非挣扎时间（含摆动）", "s"),
        metric("struggling_time_s", "挣扎时间", "s"),
        metric("uncertain_time_s", "不确定时间", "s"),
        metric("immobility_percent", "非挣扎比例", "%"),
        metric("struggling_percent", "挣扎比例", "%"),
        metric("latency_to_immobility_s", "首次挣扎转非挣扎潜伏期（经典）", "s"),
        metric("detected_rate_percent", "检测时间比例", "%"),
    ],
    "template_keys": ["arena", "body_line"],
}


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
    """两种模式的字段都校验；lis 参数照常保存，但本块不允许开始任务。"""

    values = options if isinstance(options, Mapping) else {}
    provided = set(OPTION_KEYS if provided is None else provided)
    issues: list[dict] = []

    mode = values.get("mode", "classic")
    if mode not in MODES:
        issues.append({"field": "options.mode", "message": "未知行为算法"})

    for key in BOOLEAN_KEYS:
        if key in provided and not isinstance(values.get(key), bool):
            issues.append({"field": f"options.{key}", "message": "必须是布尔值"})

    if "behavior_threshold_sensitivity" in provided:
        sensitivity = _finite(values.get("behavior_threshold_sensitivity"))
        if sensitivity is None:
            issues.append({"field": "options.behavior_threshold_sensitivity",
                           "message": "必须是有限数值"})
        elif not -3.0 <= sensitivity <= 3.0:
            issues.append({"field": "options.behavior_threshold_sensitivity",
                           "message": "灵敏度必须在 -3 到 3 之间"})

    for key in ("flow_thr_hi", "flow_thr_lo", "speed_std_thr_hi",
                "speed_std_thr_lo"):
        if key not in provided:
            continue
        value = _finite(values.get(key))
        if value is None:
            issues.append({"field": f"options.{key}", "message": "必须是有限数值"})
        elif value <= 0:
            issues.append({"field": f"options.{key}", "message": "阈值必须大于 0"})

    for high_key, low_key in (("flow_thr_hi", "flow_thr_lo"),
                              ("speed_std_thr_hi", "speed_std_thr_lo")):
        if high_key not in provided and low_key not in provided:
            continue
        high, low = _finite(values.get(high_key)), _finite(values.get(low_key))
        if high is not None and low is not None and not 0 < low < high:
            issues.append({"field": f"options.{low_key}",
                           "message": "必须满足 0 < 低阈值 < 高阈值"})

    for key in SECONDS_KEYS:
        if key not in provided:
            continue
        value = _finite(values.get(key))
        if value is None or value < 0:
            issues.append({"field": f"options.{key}",
                           "message": "必须是 ≥ 0 的有限秒数"})

    if complete and mode == "lis":
        issues.append({"field": "options.mode",
                       "message": "微动残差模式尚未接入，本块只能运行经典行为"})

    issues.extend(validate_geometry(geometry, require=complete))
    return issues


def validate_geometry(geometry: Mapping[str, Any] | None,
                      *, require: bool = False) -> list[dict]:
    geometry = geometry or {}
    issues: list[dict] = []

    arena = geometry.get("arena")
    if arena is None:
        if require:
            issues.append({"field": "geometry.arena", "message": "缺少小鼠活动区域"})
    else:
        points = activity.shape_points(arena)
        if len(points) < 3:
            issues.append({"field": "geometry.arena",
                           "message": "活动区域至少需要 3 个有限点"})
        elif abs(_polygon_area(points)) <= 1e-9:
            issues.append({"field": "geometry.arena",
                           "message": "活动区域面积必须大于 0"})

    line = geometry.get("body_line")
    if line is None:
        if require:
            issues.append({"field": "geometry.body_line",
                           "message": "缺少小鼠体长线（固定 60 mm）"})
    else:
        points = activity.shape_points(line)
        if len(points) != 2:
            issues.append({"field": "geometry.body_line",
                           "message": "体长线需要两个有限点"})
        elif math.hypot(points[1][0] - points[0][0],
                        points[1][1] - points[0][1]) <= 0:
            issues.append({"field": "geometry.body_line",
                           "message": "体长线两个端点不能重合"})
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


# -- 捕获 ----------------------------------------------------------------

def _extra(body: Mapping[str, Any] | None, pair: Mapping[str, Any] | None,
           method: str | None) -> dict:
    values: dict[str, Any] = {
        "detection_method": method,
        "feature_ready": False,
        "flow_mean_px": None, "flow_p50_px": None, "flow_p90_px": None,
        "flow_max_px": None, "flow_std_px": None,
        "flow_coherence": None, "flow_dispersion": None,
        "distance_px": None,
        "speed_nominal_mm_s": None, "speed_mm_s": None,
        "flow_p90_mm_s": None, "flow_mean_mm_s": None,
    }
    if body:
        values.update(body)
    else:
        values.update({
            "body_area_px": None, "body_perimeter_px": None,
            "body_bbox_w": None, "body_bbox_h": None, "body_major_px": None,
            "body_solidity": None, "body_fill_ratio": None,
            "body_aspect": None, "body_area_rate_s": None,
            "body_x_px": None, "body_y_px": None,
        })
    if pair:
        values.update(pair)
    return values


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
                     bbox=[float(v) for v in candidate["bbox"]])


def capture(context: Any, units: Sequence[Mapping[str, Any]]):
    """源帧顺序：每只鼠一次检测，保存身体形态与相邻成对特征。"""

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
                    "scale": activity.line_scale(geometry.get("body_line"))[0],
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
                # 面积变化率用前后源时刻差；首帧或前帧无身体时不给值。
                elapsed = None
                previous_area = None
                if prev is not None and prev.get("body") is not None:
                    elapsed = float(frame.time_s) - float(prev["time_s"])
                    previous_area = prev.get("area")
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
                pair = activity.flow_features(
                    prev["gray"], gray, candidate["mask"], candidate["bbox"],
                    prev["center"], candidate["center"],
                    mm_per_px=entry["scale"], fps=float(frame.analysis_fps),
                    elapsed_s=float(frame.time_s) - float(prev["time_s"]),
                )
            result = _detection_of(candidate, reason)
            row = timeline.base_row(
                {"id": unit_id, "name": unit.get("name"),
                 "animal_id": unit.get("animal_id")},
                frame, result,
                extra=_extra(body, pair, settings.get("bg_method")))
            if pair is not None:
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
                "area": float(candidate["area"]) if candidate else None,
                "body": body,
                "gray": gray.copy(),
            }
        context.progress(min(0.85, 0.1 + 0.75 * max(0.0, float(frame.time_s)) /
                             max(1e-9, progress_span or 1.0)))

    return captured


# -- 特征派生 ------------------------------------------------------------

def _flow_of(row: Mapping[str, Any]) -> float | None:
    value = row.get("flow_p90_mm_s")
    return float(value) if is_number(value) else None


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


def derive_features(rows: Sequence[dict], arena_height: float,
                    thresholds: Mapping[str, float],
                    *, estimate: bool = False,
                    sensitivity: float = 0.0) -> dict:
    """按连续组派生 F/S/C/A/B/Y、刚体、摆动候选与辅助分。"""

    ordered = sorted(rows, key=lambda item: (item["time_s"], item["frame"]))
    pairs = [row for row in ordered
             if row.get("in_window") and row.get("feature_ready")]
    for group in _groups(pairs):
        group_body = [row for row in group if row.get("body_x_px") is not None]
        for index, row in enumerate(group):
            _derive_row(row, group[:index + 1], group_body, arena_height,
                        thresholds)
    for row in ordered:
        row.setdefault("swing_motion_candidate", False)
        row.setdefault("swing_rigid", True)
        row.setdefault("swing_candidate", False)
        row.setdefault("swing_score", 0.0)
        row.setdefault("swing_autocorrelation", 0.0)
        row.setdefault("swing_frequency_hz", None)
        row.setdefault("swing_horizontal_ratio", 0.0)
        row.setdefault("swing_vertical_ratio", 0.0)
        row.setdefault("flow_ratio", 0.0)
        row.setdefault("decision_flow_p90_mm_s", 0.0)
        row.setdefault("speed_std_mm_s", 0.0)
        row.setdefault("coherence_median", 0.0)
        row.setdefault("acceleration_std_mm_s2", 0.0)
        row.setdefault("struggle_score", 0.0)
        row.setdefault("swing_elongated", False)

    samples = [row for row in pairs if not row.get("swing_candidate")]
    result = {
        "flow_thr_hi": float(thresholds["flow_hi"]),
        "flow_thr_lo": float(thresholds["flow_lo"]),
        "speed_thr_hi": float(thresholds["speed_hi"]),
        "speed_thr_lo": float(thresholds["speed_lo"]),
        "estimated": False,
        "samples": len(samples),
    }
    if estimate:
        result.update(_estimate(samples, thresholds, sensitivity))
    for row in ordered:
        row["flow_thr_hi_used"] = result["flow_thr_hi"]
        row["flow_thr_lo_used"] = result["flow_thr_lo"]
        row["speed_thr_hi_used"] = result["speed_thr_hi"]
        row["speed_thr_lo_used"] = result["speed_thr_lo"]
    return result


def _derive_row(row: dict, history: Sequence[Mapping[str, Any]],
                body_rows: Sequence[Mapping[str, Any]], arena_height: float,
                thresholds: Mapping[str, float]) -> None:
    moment = float(row["time_s"])
    fps = max(float(row["analysis_fps"]), 1e-9)
    recent = activity.recent_rows(history, moment, 0.3)
    flows = [value for value in (_flow_of(item) for item in recent)
             if value is not None]
    speeds = [float(item["speed_nominal_mm_s"]) for item in recent
              if is_number(item.get("speed_nominal_mm_s"))]
    current_flow = _flow_of(row) or 0.0
    decision = (float(np.percentile(flows, 90)) if len(flows) >= 3
                else current_flow)
    speed_std = float(np.std(speeds)) if len(speeds) >= 3 else 0.0
    nominal = float(row.get("speed_nominal_mm_s") or 0.0)
    ratio = decision / max(abs(nominal), 1e-6)

    coherence_rows = activity.recent_rows(history, moment, 0.5)
    coherence_values = [float(item["flow_coherence"]) for item in coherence_rows
                        if is_number(item.get("flow_coherence"))]
    coherence = float(np.median(coherence_values)) if coherence_values else 0.0

    speed_recent = [float(item["speed_nominal_mm_s"]) for item in coherence_rows
                    if is_number(item.get("speed_nominal_mm_s"))]
    acceleration = activity.acceleration_std(speed_recent, fps)

    span_rows = activity.recent_rows(body_rows, moment, 2.0, include_left=True)
    xs = [float(item["body_x_px"]) for item in span_rows]
    ys = [float(item["body_y_px"]) for item in span_rows]
    aspects = [float(item["body_aspect"]) for item in span_rows]
    span = (float(span_rows[-1]["time_s"]) - float(span_rows[0]["time_s"])
            if len(span_rows) >= 2 else 0.0)
    if span >= 2.0 - 1.5 / fps:
        correlation, frequency = activity.autocorrelation(xs, fps)
    else:
        correlation, frequency = 0.0, None
    horizontal = (float(np.std(xs)) / arena_height
                  if xs and arena_height > 1e-9 else 0.0)
    vertical = (float(np.std(ys)) / max(float(np.std(xs)), 1e-6)
                if len(ys) >= 2 else 0.0)
    rigid = activity.rigidity(body_rows, moment, fps)
    extended = activity.elongated(aspects, float(row.get("body_aspect") or 0.0),
                                  fps)
    # 0 是合法离散度，只有 None/缺失才用缺省 1。
    dispersion_value = row.get("flow_dispersion")
    dispersion = (float(dispersion_value) if is_number(dispersion_value)
                  else 1.0)
    motion = activity.motion_swing(
        frequency=frequency, autocorrelation=correlation,
        horizontal=horizontal, vertical=vertical, coherence=coherence,
        dispersion=dispersion, nominal=nominal, ratio=ratio)
    score = activity.swing_score(
        autocorrelation_value=correlation, frequency=frequency,
        coherence=coherence, dispersion=dispersion,
        is_elongated=extended, ratio=ratio)

    struggle = 0.0
    if coherence < 0.8:
        struggle += 0.6
    if dispersion >= 0.55:
        struggle += 1.5
    elif dispersion >= 0.4:
        struggle += 0.8
    if not extended:
        struggle += 0.4
    if ratio >= 3.0:
        struggle += 1.2
    elif ratio >= 2.0:
        struggle += 0.6
    if acceleration >= 150.0 and dispersion >= 0.35:
        struggle += 0.8

    row.update({
        "decision_flow_p90_mm_s": decision,
        "speed_std_mm_s": speed_std,
        "flow_ratio": ratio,
        "coherence_median": coherence,
        "acceleration_std_mm_s2": acceleration,
        "swing_autocorrelation": correlation,
        "swing_frequency_hz": frequency,
        "swing_horizontal_ratio": horizontal,
        "swing_vertical_ratio": vertical,
        "swing_rigid": rigid,
        "swing_elongated": extended,
        "swing_motion_candidate": bool(motion),
        "swing_candidate": bool(motion and rigid),
        "swing_score": score,
        "struggle_score": struggle,
    })


def _estimate(samples: Sequence[Mapping[str, Any]],
              thresholds: Mapping[str, float],
              sensitivity: float) -> dict:
    # 自动阈值用定义后的决策 F，而不是原始逐帧 P90。
    flows = [float(row["decision_flow_p90_mm_s"]) for row in samples
             if is_number(row.get("decision_flow_p90_mm_s"))]
    speeds = [float(row["speed_std_mm_s"]) for row in samples
              if is_number(row.get("speed_std_mm_s"))]
    fps = float(samples[0]["analysis_fps"]) if samples else 10.0
    minimum = max(30, int(math.ceil(3 * fps)))
    result = {
        "flow_thr_hi": float(thresholds["flow_hi"]),
        "flow_thr_lo": float(thresholds["flow_lo"]),
        "speed_thr_hi": float(thresholds["speed_hi"]),
        "speed_thr_lo": float(thresholds["speed_lo"]),
        "estimated": False,
        "samples": len(samples),
    }
    if len(samples) < minimum or not flows or not speeds:
        return result
    if (max(flows) < max(8.0, 0.2 * float(thresholds["flow_hi"]))
            and max(speeds) < max(6.0, 0.2 * float(thresholds["speed_hi"]))):
        return result
    flow_hi, flow_lo = activity.estimate_high_low(
        flows, thresholds["flow_hi"], thresholds["flow_lo"], sensitivity, 1.6)
    speed_hi, speed_lo = activity.estimate_high_low(
        speeds, thresholds["speed_hi"], thresholds["speed_lo"], sensitivity, 1.35)
    result.update({"flow_thr_hi": flow_hi, "flow_thr_lo": flow_lo,
                   "speed_thr_hi": speed_hi, "speed_thr_lo": speed_lo,
                   "estimated": True})
    return result


# -- 经典状态机 ----------------------------------------------------------

def classic_states(rows: Sequence[dict], *,
                   flow_hi: float, flow_lo: float,
                   speed_hi: float, speed_lo: float) -> None:
    """按连续组跑经典状态机；只读本组证据，不跨 unknown/outside。"""

    ordered = sorted(rows, key=lambda item: (item["time_s"], item["frame"]))
    for row in ordered:
        if not row.get("in_window"):
            row["state"] = "outside"
        elif not row.get("feature_ready"):
            row["state"] = "unknown"
        else:
            row["state"] = None
    thresholds = {"flow_hi": float(flow_hi), "flow_lo": float(flow_lo),
                  "speed_hi": float(speed_hi), "speed_lo": float(speed_lo)}
    pairs = [row for row in ordered if row.get("in_window")]
    for group in _groups(pairs):
        _run_group(group, thresholds)
    for row in ordered:
        state = row.get("state") or "unknown"
        row["state"] = state
        row["state_group"] = ("immobile" if state in ("immobile", "swing")
                              else state)
        row["is_struggling"] = (None if state in ("unknown", "outside")
                                else state == "struggling")
        row["is_swinging"] = (state == "swing")


def _run_group(group: list[Mapping[str, Any]],
               thresholds: Mapping[str, float]) -> None:
    """经典状态机：只累计已经过去的曝光，摆动按曝光秒数取比例。"""

    state = "immobile"
    first = group[0]
    anchor = first.get("pair_previous_time_s")
    anchor_time = float(anchor) if is_number(anchor) else float(first["time_s"])
    hold_until = anchor_time + 0.2
    swing = False
    confirmed_at = None
    swing_accum = 0.0
    enter_accum = 0.0
    exit_accum = 0.0
    previous: Mapping[str, Any] | None = None
    previous_medium = False
    previous_low = False
    window_start = 0

    for index, row in enumerate(group):
        moment = float(row["time_s"])
        continuous = (previous is not None
                      and abs(float(previous["interval_end_s"])
                              - float(row["interval_start_s"])) < 1e-7)
        # 当前行代表 [t, t+dt)，它的曝光尚未过去，不能提前计入计时器。
        if continuous:
            elapsed = max(0.0, float(previous.get("dt_s") or 0.0))
            if (swing and confirmed_at is not None
                    and float(previous["time_s"]) >= confirmed_at - 1e-9):
                swing_accum += elapsed
            if previous_medium and state == "immobile":
                enter_accum += elapsed
            if previous_low and state == "struggling":
                exit_accum += elapsed

        low_edge = moment - 1.5
        while (window_start < index
               and float(group[window_start]["interval_end_s"]) <= low_edge):
            window_start += 1
        coverage = 0.0
        candidate_seconds = 0.0
        for other in group[window_start:index + 1]:
            start = max(float(other["interval_start_s"]), low_edge)
            end = min(float(other["interval_end_s"]), moment)
            if end > start:
                coverage += end - start
                if other.get("swing_candidate"):
                    candidate_seconds += end - start
        fraction = candidate_seconds / coverage if coverage > 0 else 0.0

        if swing:
            if (coverage >= 1.5 - 1e-9 and fraction <= 0.35
                    and swing_accum >= 1.5 - 1e-9):
                swing = False
                confirmed_at = None
                hold_until = moment + 0.2
                swing_accum = 0.0

        if (not swing and coverage >= 1.5 - 1e-9 and fraction >= 0.65
                and row.get("swing_rigid")):
            swing = True
            confirmed_at = moment
            # 摆动结束后回到 immobile，而不是恢复进入摆动前的状态。
            state = "immobile"
            enter_accum = 0.0
            exit_accum = 0.0
            swing_accum = 0.0
            for other in group[:index + 1]:
                # 只回写与过去 2.5 秒有真实交集的曝光，边界零长度不算。
                start = max(float(other["interval_start_s"]), moment - 2.5)
                end = min(float(other["interval_end_s"]), moment)
                if end - start > 1e-9:
                    other["state"] = "swing"
                    other["swing_confirmed_at"] = moment

        previous_medium = False
        previous_low = False
        if not swing and moment + 1e-9 >= hold_until:
            decision = float(row.get("decision_flow_p90_mm_s") or 0.0)
            speed_std = float(row.get("speed_std_mm_s") or 0.0)
            ratio = float(row.get("flow_ratio") or 0.0)
            if state == "immobile":
                strong = (speed_std > thresholds["speed_hi"]
                          or (decision > thresholds["flow_hi"] and ratio >= 2.5))
                medium = (decision >= max(thresholds["flow_lo"],
                                          0.45 * thresholds["flow_hi"])
                          and ratio >= 2.5)
                dispersion_value = row.get("flow_dispersion")
                dispersion = (float(dispersion_value)
                              if is_number(dispersion_value) else 1.0)
                suppressed = bool(row.get("swing_motion_candidate")) or (
                    float(row.get("swing_score") or 0.0) >= 4.0
                    and dispersion <= 0.08
                    and float(row.get("swing_autocorrelation") or 0.0) >= 0.55
                    and bool(row.get("swing_rigid")))
                if strong:
                    state = "struggling"
                    enter_accum = 0.0
                    exit_accum = 0.0
                elif medium and not suppressed:
                    if enter_accum >= 0.45 - 1e-9:
                        state = "struggling"
                        enter_accum = 0.0
                        exit_accum = 0.0
                    else:
                        previous_medium = True
                else:
                    enter_accum = 0.0
            elif state == "struggling":
                low = (decision < thresholds["flow_lo"]
                       and speed_std < thresholds["speed_lo"])
                if low:
                    if exit_accum >= 0.4 - 1e-9:
                        state = "immobile"
                        exit_accum = 0.0
                        enter_accum = 0.0
                    else:
                        previous_low = True
                else:
                    exit_accum = 0.0
        row["state"] = "swing" if swing else state
        previous = row


# -- 事件与汇总 ----------------------------------------------------------

def build_events(rows: Sequence[Mapping[str, Any]], unit_id: str,
                 unit_name: str) -> list[dict]:
    events: list[dict] = []
    previous = None
    sequence = 0
    for row in sorted(rows, key=lambda item: (item["time_s"], item["frame"])):
        if row.get("state") in ("unknown", "outside"):
            previous = None
            continue
        if previous is not None:
            adjacent = abs(float(previous["interval_end_s"])
                           - float(row["interval_start_s"])) < 1e-7
            if adjacent and row["state_group"] != previous["state_group"]:
                sequence += 1
                moment = float(row["interval_start_s"])
                confirmed = moment
                if row.get("state") == "swing" and row.get("swing_confirmed_at"):
                    confirmed = float(row["swing_confirmed_at"])
                events.append({
                    "event_id": f"{unit_id}:{'struggling' if row['state_group'] == 'struggling' else 'immobile'}_transition:{sequence}",
                    "unit_id": unit_id,
                    "unit_name": unit_name,
                    "kind": ("struggling_transition"
                             if row["state_group"] == "struggling"
                             else "immobile_transition"),
                    "time_s": moment,
                    "confirmed_s": confirmed,
                    "target": None,
                    "source": "automatic",
                    "frame": row["frame"],
                })
        previous = row
    return events


def summarize(rows: Sequence[Mapping[str, Any]], unit: Mapping[str, Any],
              window: timeline.Window, info: Mapping[str, Any],
              events: Sequence[Mapping[str, Any]]) -> dict:
    total = immobility = struggling = swing = detected_time = 0.0
    total_frames = detected_frames = 0
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
        if state == "struggling":
            struggling += span
        elif state in ("immobile", "swing"):
            immobility += span
        if state == "swing":
            swing += span
    uncertain = max(0.0, total - immobility - struggling)

    immobility_events = sum(1 for event in events
                            if event["kind"] == "immobile_transition")
    struggling_events = sum(1 for event in events
                            if event["kind"] == "struggling_transition")
    latency = None
    for event in events:
        if event["kind"] == "immobile_transition":
            start = float(window.start_s) if window.known else 0.0
            latency = float(event["time_s"]) - start
            break
    immobility_bouts = _bouts(rows, "immobile")
    struggling_bouts = _bouts(rows, "struggling")

    def percent(value: float) -> float | None:
        return timeline.ratio(value * 100.0, total)

    summary = {
        "unit_id": unit.get("id"),
        "unit_name": unit.get("name"),
        "total_time_s": total,
        "immobility_time_s": immobility,
        "struggling_time_s": struggling,
        "uncertain_time_s": uncertain,
        "swing_time_s": swing,
        "immobility_percent": percent(immobility),
        "struggling_percent": percent(struggling),
        "uncertain_percent": percent(uncertain),
        "swing_percent": percent(swing),
        "detected_time_s": detected_time,
        "detected_rate_percent": percent(detected_time),
        "detected_frames": detected_frames,
        "total_frames": total_frames,
        "immobility_episodes": immobility_events,
        "struggling_episodes": struggling_events,
        "immobility_frequency_per_min": (timeline.ratio(
            immobility_events * 60.0, total)),
        "struggling_frequency_per_min": (timeline.ratio(
            struggling_events * 60.0, total)),
        "immobility_bouts": immobility_bouts,
        "struggling_bouts": struggling_bouts,
        "latency_to_immobility_s": latency,
        "analysis_window_start_s": window.start_s if window.known else None,
        "analysis_window_end_s": window.end_s if window.known else None,
        "thresholds": {
            "flow_thr_hi": info["flow_thr_hi"],
            "flow_thr_lo": info["flow_thr_lo"],
            "speed_std_thr_hi": info["speed_thr_hi"],
            "speed_std_thr_lo": info["speed_thr_lo"],
        },
        "threshold_estimated": bool(info["estimated"]),
        "threshold_samples": int(info["samples"]),
    }
    for key in OPTION_KEYS:
        summary[key] = unit.get("options", {}).get(key)
    return summary


def _bouts(rows: Sequence[Mapping[str, Any]], group: str) -> int:
    """最终连续曝光段数：标签相同但曝光断开也要算新的一段。"""

    count = 0
    previous: Mapping[str, Any] | None = None
    for row in rows:
        if row.get("state_group") != group:
            previous = None
            continue
        gap = (previous is None
               or abs(float(previous["interval_end_s"])
                      - float(row["interval_start_s"])) >= 1e-7)
        if gap:
            count += 1
        previous = row
    return count


# -- 分析入口 ------------------------------------------------------------

def _clear_pair(row: dict) -> None:
    row["feature_ready"] = False
    for key in ("flow_mean_px", "flow_p50_px", "flow_p90_px", "flow_max_px",
                "flow_std_px", "flow_coherence", "flow_dispersion",
                "distance_px", "speed_nominal_mm_s",
                "speed_mm_s", "flow_p90_mm_s", "flow_mean_mm_s"):
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
    asset = job.get("asset") or {}
    metadata = asset.get("metadata") or {}
    all_rows = [row for rows in captured.values() for row in rows]
    video_window = timeline.resolve_video_window(
        job.get("window"), all_rows, metadata.get("duration_s"))

    observations: list[dict] = []
    summary: list[dict] = []
    events: list[dict] = []
    unit_metadata: dict[str, dict] = {}
    warnings: list[str] = []

    for unit in units:
        unit_id = unit["id"]
        geometry = unit.get("geometry") or {}
        options = unit.get("options") or {}
        first_seen = timeline.first_seen_in_window(
            KEY, captured[unit_id], video_window, geometry)
        window = timeline.resolve_unit_window(
            unit.get("window"), video_window, first_seen)
        rows = timeline.clip_rows(captured[unit_id], window)
        _filter_pairs(rows, window)

        scale, line_px = activity.line_scale(geometry.get("body_line"))
        arena_points = activity.shape_points(geometry.get("arena"))
        arena_height = max(1.0, max(point[1] for point in arena_points)
                           - min(point[1] for point in arena_points))
        thresholds = {
            "flow_hi": float(options["flow_thr_hi"]),
            "flow_lo": float(options["flow_thr_lo"]),
            "speed_hi": float(options["speed_std_thr_hi"]),
            "speed_lo": float(options["speed_std_thr_lo"]),
        }
        info = derive_features(
            rows, arena_height, thresholds,
            estimate=bool(options.get("auto_behavior_threshold")),
            sensitivity=float(options.get("behavior_threshold_sensitivity") or 0.0))
        classic_states(rows, flow_hi=info["flow_thr_hi"],
                       flow_lo=info["flow_thr_lo"],
                       speed_hi=info["speed_thr_hi"],
                       speed_lo=info["speed_thr_lo"])
        unit_events = build_events(rows, unit_id, unit.get("name") or "")
        unit_summary = summarize(rows, unit, window, info, unit_events)

        observations.extend(rows)
        events.extend(unit_events)
        summary.append(unit_summary)
        unit_metadata[unit_id] = {
            "body_length_mm": activity.BODY_LENGTH_MM,
            "mm_per_px": scale,
            "body_line_px": line_px,
            "coordinate_unit": "mm",
            "thresholds": unit_summary["thresholds"],
            "auto_threshold": bool(options.get("auto_behavior_threshold")),
            "threshold_estimated": bool(info["estimated"]),
            "threshold_samples": int(info["samples"]),
            "speed_sources": {
                "nominal": "px × mm/px × analysis fps",
                "actual": "px × mm/px ÷ source dt",
            },
            "state_labels": ["struggling", "immobile", "swing", "unknown",
                             "outside"],
            "immobility_includes_swing": True,
        }
        if not window.known:
            warnings.append("未检测到动物，无法确定相对时间窗："
                            + str(unit.get("name") or unit_id))

    return AnalysisResult(
        observations=timeline.sort_rows(observations),
        summary=summary,
        events=events,
        bins=[],
        tables={},
        warnings=warnings,
        metadata={
            "assay": KEY,
            "body_length_mm": activity.BODY_LENGTH_MM,
            "coordinate_unit": "mm",
            "units": unit_metadata,
            "warnings": warnings,
        },
    )


def preview_analysis(*, asset: Mapping[str, Any], video: Mapping[str, Any],
                     unit: Mapping[str, Any], start_s: float,
                     end_s: float, time_s: float | None = None) -> dict:
    """行为预览共用同一捕获→特征→实际阈值→最终状态核心。"""

    from ..schema import AnalysisContext, new_id

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

    options = unit.get("options") or {}
    scale, _ = activity.line_scale(geometry.get("body_line"))
    arena_points = activity.shape_points(geometry.get("arena"))
    arena_height = max(1.0, max(point[1] for point in arena_points)
                       - min(point[1] for point in arena_points))
    thresholds = {
        "flow_hi": float(options["flow_thr_hi"]),
        "flow_lo": float(options["flow_thr_lo"]),
        "speed_hi": float(options["speed_std_thr_hi"]),
        "speed_lo": float(options["speed_std_thr_lo"]),
    }
    info = derive_features(
        rows, arena_height, thresholds,
        estimate=bool(options.get("auto_behavior_threshold")),
        sensitivity=float(options.get("behavior_threshold_sensitivity") or 0.0))
    classic_states(rows, flow_hi=info["flow_thr_hi"],
                   flow_lo=info["flow_thr_lo"],
                   speed_hi=info["speed_thr_hi"],
                   speed_lo=info["speed_thr_lo"])
    events = build_events(rows, unit["id"], unit.get("name") or "")
    summary = summarize(rows, unit, unit_window, info, events)
    state_at_time = None
    if time_s is not None:
        for row in rows:
            if (float(row["interval_start_s"]) - 1e-9 <= float(time_s)
                    < float(row["interval_end_s"]) - 1e-9):
                state_at_time = row.get("state")
                break
    return {
        "start_s": float(start_s),
        "end_s": float(end_s),
        "window_start_s": (unit_window.start_s if unit_window.known else None),
        "window_end_s": (unit_window.end_s if unit_window.known else None),
        "total_time_s": summary["total_time_s"],
        "immobility_time_s": summary["immobility_time_s"],
        "struggling_time_s": summary["struggling_time_s"],
        "uncertain_time_s": summary["uncertain_time_s"],
        "swing_time_s": summary["swing_time_s"],
        "rows": len(rows),
        "events": len(events),
        "mm_per_px": scale,
        "thresholds": summary["thresholds"],
        "threshold_estimated": bool(info["estimated"]),
        "threshold_samples": int(info["samples"]),
        "state_at_time": state_at_time,
    }
