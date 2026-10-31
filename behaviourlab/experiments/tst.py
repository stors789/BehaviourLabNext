"""TST 悬尾实验：classic 与 lis 两种行为算法。

classic 用成对光流与阈值滞回；lis 先保存配准残差等测量证据，再统一
按分位数阈值评分（评分/证据两分支）并做片段补救。体长线固定 60 mm，
尺度不从身体框反推。缺失与未知不参与判定，也不伪造检测。
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

    # lis 已接入执行分支；经典阈值字段保留并验证，但不参与 lis 判定。
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
        # lis 配对证据：非配对行保持 None，不伪造数值。
        "lis_r": None, "lis_mask_residual": None, "lis_gray_mean_d": None,
        "lis_peripheral_d": None, "lis_core_d": None, "lis_core_ratio": None,
        "lis_z": None, "lis_hu_distance": None, "lis_body_length_px": None,
        "lis_aspect_ratio": None, "lis_solidity": None, "lis_extent": None,
        "lis_displacement_ratio": None, "lis_segmentation_bad": None,
        "lis_climbing_suspected": None,
        "lis_h": None, "lis_f": None, "lis_s": None, "lis_b": None,
        "lis_g": None,
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
    lis_units = [
        unit for unit in units
        if str((unit.get("options") or {}).get("mode", "classic")) == "lis"
    ]
    if lis_units:
        # 提取阶段不判定状态：进度文字明确“状态待定”，不给临时挣扎结论。
        context.progress(0.1, "提取运动特征中，状态待定")

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
                roi_y, roi_x, roi_h, roi_w = cv2.boundingRect(region)
                entry = {
                    "detector": detection.shared_detector(
                        (run_key, unit_id), settings, background=background),
                    "region": region,
                    "arena": activity.shape_points(geometry.get("arena")),
                    "scale": activity.line_scale(geometry.get("body_line"))[0],
                    "roi": (float(roi_x), float(roi_y),
                            float(roi_w), float(roi_h)),
                    "lis": str(options.get("mode", "classic")) == "lis",
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
            lis_state = None
            if entry["lis"] and candidate is not None and body is not None:
                mask = candidate["mask"]
                metrics = {
                    "area": float(candidate["area"]),
                    "perimeter": float(body["body_perimeter_px"]),
                    "aspect": float(body["body_aspect"]),
                    "solidity": float(body["body_solidity"]),
                    "fill": float(body["body_fill_ratio"]),
                }
                lis_state = {
                    "mask": np.asarray(mask).copy(),
                    "angle": activity.lis_principal_angle(mask),
                    "metrics": metrics,
                    "hu": activity.lis_hu_vector(mask),
                }
            if (entry["lis"] and lis_state is not None and pair is not None
                    and prev is not None and prev.get("lis") is not None):
                pair = {**pair, **_lis_pair_evidence(
                    entry, gray, candidate, body, pair, prev, lis_state,
                    options)}
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
                "lis": lis_state,
            }
        context.progress(min(0.85, 0.1 + 0.75 * max(0.0, float(frame.time_s)) /
                             max(1e-9, progress_span or 1.0)))

    return captured


def _lis_pair_evidence(entry: Mapping[str, Any], gray, candidate, body,
                       pair: Mapping[str, Any], prev: Mapping[str, Any],
                       lis_state: Mapping[str, Any],
                       options: Mapping[str, Any]) -> dict:
    """lis 配对证据：配准、残差 R/P/核心量/r、形变 Z 与分割诊断。"""

    length = activity.lis_body_length(candidate["mask"], candidate["bbox"])
    aligned_gray, aligned_mask, _ = activity.lis_register(
        prev["gray"], gray, prev["lis"]["mask"], lis_state["mask"],
        prev["center"], candidate["center"],
        prev["lis"]["angle"], lis_state["angle"],
        rotate_align=bool(options.get("lis_rotate_align")))
    residual = activity.lis_residual(aligned_gray, aligned_mask, gray,
                                     lis_state["mask"])
    core = activity.lis_core_peripheral(lis_state["mask"], length,
                                        residual["diff"])
    hu_distance = activity.lis_hu_distance(lis_state["hu"],
                                           prev["lis"]["hu"])
    z = activity.lis_deformation(lis_state["metrics"], prev["lis"]["metrics"],
                                 hu_distance, residual["mask_residual"])
    roi_x, roi_y, roi_w, roi_h = entry["roi"]
    bbox_x, bbox_y, bbox_w, bbox_h = candidate["bbox"]
    roi_area = max(roi_w * roi_h, 1e-6)
    area_ratio = float(candidate["area"]) / roi_area
    bad = bool(area_ratio < 0.0005 or area_ratio > 0.70 or length < 14.0)
    climbing = bool(
        float(bbox_y) <= roi_y + 0.1 * roi_h
        and float(body["body_y_px"]) <= roi_y + 0.25 * roi_h
        and float(body["body_aspect"]) >= 1.8)
    return {
        "lis_r": float(residual["r"]),
        "lis_mask_residual": float(residual["mask_residual"]),
        "lis_gray_mean_d": float(residual["gray_mean_d"]),
        "lis_peripheral_d": float(core["peripheral_mean"]),
        "lis_core_d": float(core["core_mean"]),
        "lis_core_ratio": float(core["ratio"]),
        "lis_z": float(z),
        "lis_hu_distance": float(hu_distance),
        "lis_body_length_px": float(length),
        "lis_aspect_ratio": float(body["body_aspect"]),
        "lis_solidity": float(body["body_solidity"]),
        "lis_extent": float(body["body_fill_ratio"]),
        "lis_displacement_ratio": (float(pair["distance_px"])
                                   / max(length, 1e-6)),
        "lis_segmentation_bad": bad,
        "lis_climbing_suspected": climbing,
    }


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


# -- LIS 微动残差：滚动量、阈值、两分支状态机与片段后处理 -----------------

LIS_ROLLING_KEYS = ("lis_h", "lis_f", "lis_s", "lis_b", "lis_g")
LIS_SCORE_FEATURES = ("r", "p", "z", "h", "f", "s")
LIS_SCORE_WEIGHTS = {"r": 0.24, "p": 0.14, "z": 0.22, "h": 0.08,
                     "f": 0.18, "s": 0.14}
# 特征 key → 行字段；周边量 P 在行里叫 lis_peripheral_d。
LIS_ROW_FIELDS = {"r": "lis_r", "p": "lis_peripheral_d", "z": "lis_z",
                  "h": "lis_h", "f": "lis_f", "s": "lis_s",
                  "g": "lis_g", "b": "lis_b"}


def _lis_rolling(group: list[dict]) -> None:
    """组内滚动量：H(.5s)、F/S(.3s)、g、b(1.5s)，只看 (t−w, t] 的同组行。"""

    for index, row in enumerate(group):
        moment = float(row["time_s"])
        recent = [item for item in group[:index + 1]
                  if float(item["time_s"]) > moment - 0.5 + 1e-9]
        if len(recent) >= 3:
            row["lis_h"] = max(
                activity.lis_median_diff(
                    [float(item[key]) for item in recent
                     if is_number(item.get(key))])
                for key in ("lis_r", "lis_peripheral_d", "lis_z"))
        else:
            row["lis_h"] = 0.0
        short = [item for item in group[:index + 1]
                 if float(item["time_s"]) > moment - 0.3 + 1e-9]
        flows = [float(item["flow_p90_mm_s"]) for item in short
                 if is_number(item.get("flow_p90_mm_s"))]
        if len(flows) >= 3:
            row["lis_f"] = float(np.percentile(flows, 90))
        elif is_number(row.get("flow_p90_mm_s")):
            row["lis_f"] = float(row["flow_p90_mm_s"])
        else:
            row["lis_f"] = 0.0
        speeds = [float(item["speed_nominal_mm_s"]) for item in short
                  if is_number(item.get("speed_nominal_mm_s"))]
        row["lis_s"] = float(np.std(speeds)) if len(speeds) >= 3 else 0.0
        length = float(row.get("lis_body_length_px") or 0.0)
        distance = float(row.get("distance_px") or 0.0)
        row["lis_g"] = distance / max(length, 1e-6)
        span_rows = [item for item in group[:index + 1]
                     if float(item["time_s"]) > moment - 1.5 + 1e-9]
        row["lis_b"] = activity.lis_periodicity(
            [float(item["body_x_px"]) for item in span_rows],
            [float(item["body_y_px"]) for item in span_rows],
            [float(item["time_s"]) for item in span_rows])


def lis_score_rows(rows: Sequence[dict]) -> dict:
    """阈值估计、归一化与综合分：只用分割良好的配对行估计 R/P/Z/H/F/S。

    无良好分割样本时改用全部配对行并标全局 low_information；
    每行写 lis_score_*、lis_motion_score、lis_motion_prior、
    lis_score_mode 与 lis_thresholds_used。
    """

    pairs = [row for row in rows
             if row.get("in_window") and row.get("feature_ready")]
    good = [row for row in pairs if not row.get("lis_segmentation_bad")]
    low_information = not good
    sample = good if good else pairs
    thresholds: dict[str, dict] = {}
    for key in ("r", "p", "z", "h", "f", "s", "g", "b"):
        field = LIS_ROW_FIELDS[key]
        values = [float(row[field]) for row in sample
                  if is_number(row.get(field))]
        thresholds[key] = activity.lis_thresholds(values or [0.0])

    for row in rows:
        if not (row.get("in_window") and row.get("feature_ready")):
            for key in LIS_ROLLING_KEYS:
                row.setdefault(key, None)
            continue
        score = 0.0
        for key, weight in LIS_SCORE_WEIGHTS.items():
            field = LIS_ROW_FIELDS[key]
            norm = activity.lis_normalize(
                float(row[field]), thresholds[key]["q25"],
                thresholds[key]["q90"])
            row[f"lis_score_{key}"] = norm
            score += weight * norm
        row["lis_motion_score"] = score
        row["lis_motion_prior"] = activity.lis_motion_prior(
            thresholds["f"]["q50"], thresholds["f"]["q90"],
            thresholds["s"]["q50"], thresholds["s"]["q90"])
        row["lis_score_mode"] = ("score" if row["lis_motion_prior"] >= .35
                                 else "evidence")
        row["lis_thresholds_used"] = {
            key: {name: thresholds[key][name]
                  for name in ("low", "mid", "high", "very_high")}
            for key in ("r", "p", "z", "h", "f", "s", "g", "b")}
    return {"thresholds": thresholds, "samples": len(sample),
            "low_information": low_information}


def _lis_evidence(row: Mapping[str, Any], thresholds: Mapping[str, Any]
                  ) -> tuple[bool, bool, bool]:
    """(活动证据, 不动证据, 被动摆动)。比较带 2%/5% 相对容差。"""

    values = {
        "r": float(row["lis_r"]),
        "p": float(row["lis_peripheral_d"]),
        "z": float(row["lis_z"]),
        "h": float(row["lis_h"]),
    }
    count = sum(1 for key in ("p", "z", "h")
                if activity.lis_above(values[key], thresholds[key]["high"]))
    if activity.lis_above(float(row["lis_core_ratio"]),
                          max(1.15, thresholds["r"]["q80"])):
        count += 1
    active = ((activity.lis_above(values["r"], thresholds["r"]["high"])
               and count >= 1) or count >= 2)
    low_evidence = all(
        activity.lis_not_above(values[key], thresholds[key]["low"])
        for key in ("r", "p", "z", "h"))
    calm_local = all(
        activity.lis_not_above(values[key], thresholds[key]["mid"])
        for key in ("r", "p", "z"))
    passive = False
    if calm_local and not active:
        g = float(row["lis_g"])
        b = float(row["lis_b"])
        g_up_mid = activity.lis_above(g, thresholds["g"]["q50"])
        g_up_high = activity.lis_above(g, thresholds["g"]["q80"])
        b_ref = max(0.4, thresholds["b"]["q75"])
        passive = ((g_up_mid and activity.lis_above(b, b_ref))
                   or (g_up_high and activity.lis_not_above(
                       values["h"], thresholds["h"]["high"])))
    return active, bool(low_evidence or passive), passive


def _lis_window(group: Sequence[Mapping[str, Any]], index: int,
                seconds: float, evidence_key: str) -> tuple[float, float]:
    """(t−seconds, t) 内已曝光同组数据的覆盖秒数与证据占比。"""

    now = float(group[index]["time_s"])
    low = now - seconds
    coverage = 0.0
    marked = 0.0
    for row in group[:index]:
        start = float(row["interval_start_s"])
        end = float(row["interval_end_s"])
        overlap = min(end, now) - max(start, low)
        if overlap > 1e-9:
            coverage += overlap
            if row.get(evidence_key):
                marked += overlap
    return coverage, (marked / coverage if coverage > 1e-9 else 0.0)


def _run_lis_group(group: list[dict], thresholds: Mapping[str, Any],
                   low_information: bool) -> None:
    """lis 状态机：评分分支即时判定，证据分支按曝光窗滞回。"""

    state = "immobile"
    for index, row in enumerate(group):
        if row.get("lis_segmentation_bad"):
            # 坏分割仍有配对：保留上一状态、低可信并清空证据。
            row["state"] = state
            row["state_group"] = state
            row["lis_confidence"] = "low"
            row["lis_reason"] = "分割质量较差，保持上一状态"
            row["lis_active_evidence"] = False
            row["lis_immobile_evidence"] = False
            row["lis_passive_swing"] = False
            row["lis_evidence_coverage_s"] = None
            row["lis_evidence_ratio"] = None
            if row.get("lis_climbing_suspected"):
                row["lis_confidence"] = "low"
            continue
        active, immobile_ev, passive = _lis_evidence(row, thresholds)
        row["lis_active_evidence"] = active
        row["lis_immobile_evidence"] = immobile_ev
        row["lis_passive_swing"] = passive
        prior = float(row["lis_motion_prior"])
        coverage: float | None = None
        ratio: float | None = None
        if prior >= .35:
            state = ("struggling" if float(row["lis_motion_score"]) >= .5
                     else "immobile")
            row["lis_reason"] = ("综合评分高于阈值" if state == "struggling"
                                 else "综合评分低于阈值")
            margin = abs(float(row["lis_motion_score"]) - 0.5)
            row["lis_confidence"] = "high" if margin >= .15 else "medium"
        elif state != "struggling":
            coverage, ratio = _lis_window(group, index, .35,
                                          "lis_active_evidence")
            if coverage >= .28 - 1e-9 and ratio >= .45 - 1e-9:
                state = "struggling"
                row["lis_reason"] = "活动证据"
            elif passive:
                row["lis_reason"] = "被动摆动"
            elif immobile_ev:
                row["lis_reason"] = "不动证据"
            else:
                row["lis_reason"] = "状态保持"
            row["lis_confidence"] = "medium"
        else:
            coverage, ratio = _lis_window(group, index, .75,
                                          "lis_immobile_evidence")
            if coverage >= .60 - 1e-9 and ratio >= .55 - 1e-9:
                state = "immobile"
                row["lis_reason"] = "不动证据"
            else:
                row["lis_reason"] = "状态保持"
            row["lis_confidence"] = "medium"
        row["lis_evidence_coverage_s"] = coverage
        row["lis_evidence_ratio"] = ratio
        if row.get("lis_climbing_suspected"):
            row["lis_confidence"] = "low"
        row["state"] = state
        row["state_group"] = state
    if low_information:
        for row in group:
            row["lis_confidence"] = "low"


def _lis_state_segments(group: Sequence[Mapping[str, Any]]
                        ) -> list[dict[str, Any]]:
    """组内连续同状态段（曝光连续，组内天然相邻）。"""

    segments: list[dict[str, Any]] = []
    for row in group:
        state = row["state"]
        if segments and segments[-1]["state"] == state:
            segments[-1]["rows"].append(row)
            segments[-1]["end"] = float(row["interval_end_s"])
        else:
            segments.append({"state": state,
                             "rows": [row],
                             "start": float(row["interval_start_s"]),
                             "end": float(row["interval_end_s"])})
    return segments


def _segment_gap(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    """两侧挣扎段的判定间隙：同分支用自己的，混合取较小者。"""

    def mode_gap(segment: Mapping[str, Any]) -> float:
        mode = str(segment["rows"][0].get("lis_score_mode") or "score")
        return 2.0 if mode == "score" else 0.5

    return min(mode_gap(left), mode_gap(right))


def _lis_sandwich(group: list[dict], min_struggle_s: float) -> None:
    """夹心处理：短不动段填回挣扎，再删过短挣扎段；两端片段不动。"""

    segments = _lis_state_segments(group)
    for position in range(1, len(segments) - 1):
        segment = segments[position]
        if segment["state"] != "immobile":
            continue
        left, right = segments[position - 1], segments[position + 1]
        if left["state"] != "struggling" or right["state"] != "struggling":
            continue
        duration = segment["end"] - segment["start"]
        if duration < _segment_gap(left, right) - 1e-9:
            for row in segment["rows"]:
                row["state"] = "struggling"
                row["state_group"] = "struggling"
    segments = _lis_state_segments(group)
    for position in range(1, len(segments) - 1):
        segment = segments[position]
        if segment["state"] != "struggling":
            continue
        left, right = segments[position - 1], segments[position + 1]
        if left["state"] != "immobile" or right["state"] != "immobile":
            continue
        if segment["end"] - segment["start"] <= float(min_struggle_s) + 1e-9:
            for row in segment["rows"]:
                row["state"] = "immobile"
                row["state_group"] = "immobile"


def _lis_rescue_condition(row: Mapping[str, Any]) -> bool:
    """局部补救三组条件之一命中即可（仅 .28≤p<.35 的行参与）。"""

    aspect = float(row["lis_aspect_ratio"])
    score = float(row["lis_motion_score"])
    f = float(row["lis_f"])
    s = float(row["lis_s"])
    r = float(row["lis_r"])
    p = float(row["lis_peripheral_d"])
    h = float(row["lis_h"])
    b = float(row["lis_b"])
    g = float(row["lis_g"])
    if aspect >= 1.55 and score >= .45 and f >= 20 and s <= 6:
        return True
    if (aspect >= 1.10 and score >= .32 and r >= .026 and p >= .010
            and h >= .006 and f >= 15 and s <= 6.5 and b >= .30
            and g <= .012):
        return True
    if (aspect >= 1.70 and score >= .45 and r >= .025 and p >= .016
            and h >= .025 and f >= 14 and s <= 12 and b >= .48 and g <= .010):
        return True
    return False


def _lis_rescue(group: list[dict], rescue_gap_s: float, before_s: float,
                after_s: float) -> None:
    """局部微动补救：候选夹心、间隙填充、前后扩展；不跨组不造位置。"""

    for row in group:
        prior = float(row["lis_motion_prior"])
        raw = bool(.28 <= prior < .35 and _lis_rescue_condition(row))
        row["lis_rescue_raw"] = raw
        row["lis_rescue"] = False
    raw_positions = [index for index, row in enumerate(group)
                     if row["lis_rescue_raw"]]
    if not raw_positions:
        return
    # 夹心：两段候选之间严格短于补救间隙的连续非候选行补成候选。
    for left, right in zip(raw_positions, raw_positions[1:]):
        between = group[left + 1:right]
        if not between:
            continue
        duration = (float(between[-1]["interval_end_s"])
                    - float(between[0]["interval_start_s"]))
        if duration < float(rescue_gap_s) - 1e-9:
            for row in between:
                row["lis_rescue_raw"] = True
    # 候选段前后扩展：与 [起点−before, 终点+after) 有正交集的本组曝光。
    filled_positions = [index for index, row in enumerate(group)
                        if row["lis_rescue_raw"]]
    blocks = []
    run = [filled_positions[0]]
    for index in filled_positions[1:]:
        if index == run[-1] + 1:
            run.append(index)
        else:
            blocks.append(run)
            run = [index]
    blocks.append(run)
    for block in blocks:
        start = float(group[block[0]]["interval_start_s"]) - float(before_s)
        end = float(group[block[-1]]["interval_end_s"]) + float(after_s)
        for row in group:
            overlap = (min(float(row["interval_end_s"]), end)
                       - max(float(row["interval_start_s"]), start))
            if overlap > 1e-9:
                row["lis_rescue"] = True
    for row in group:
        if row["lis_rescue"]:
            row["state"] = "struggling"
            row["state_group"] = "struggling"


def derive_lis_features(rows: Sequence[dict], options: Mapping[str, Any]
                        ) -> dict:
    """lis 完整派生：滚动量 → 阈值/评分 → 状态机 → 夹心 → 补救。"""

    ordered = sorted(rows, key=lambda item: (item["time_s"], item["frame"]))
    pairs = [row for row in ordered
             if row.get("in_window") and row.get("feature_ready")]
    groups = _groups(pairs)
    for group in groups:
        _lis_rolling(group)
    info = lis_score_rows(ordered)
    for row in ordered:
        if not row.get("in_window"):
            row["state"] = "outside"
            row["state_group"] = "outside"
        elif not row.get("feature_ready"):
            row["state"] = "unknown"
            row["state_group"] = "unknown"
    for group in groups:
        _run_lis_group(group, info["thresholds"], info["low_information"])
    rescue_enabled = bool(options.get("lis_rescue_enabled"))
    for group in groups:
        _lis_sandwich(group, float(options.get("lis_min_struggle_s") or 0.3))
        if rescue_enabled:
            _lis_rescue(group, float(options.get("lis_rescue_gap_s") or 1.0),
                        float(options.get("lis_rescue_before_s") or 0.7),
                        float(options.get("lis_rescue_after_s") or 0.3))
        else:
            for row in group:
                row["lis_rescue_raw"] = False
                row["lis_rescue"] = False
    for row in ordered:
        state = row.get("state") or "unknown"
        if state not in ("unknown", "outside"):
            row["state_group"] = state
        row["is_struggling"] = (None if state in ("unknown", "outside")
                                else state == "struggling")
        row["is_swinging"] = False
    return info


# -- 事件与汇总 ----------------------------------------------------------

def build_events(rows: Sequence[Mapping[str, Any]], unit_id: str,
                 unit_name: str, *, confirm_swing: bool = True) -> list[dict]:
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
                if not confirm_swing:
                    # lis 是离线分段：不声称某帧实时确认。
                    confirmed = None
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
              events: Sequence[Mapping[str, Any]], *,
              mode: str = "classic") -> dict:
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
    if mode != "lis":
        # 首次经典转变潜伏期在 lis 恒为 null。
        for event in events:
            if event["kind"] == "immobile_transition":
                start = float(window.start_s) if window.known else 0.0
                latency = float(event["time_s"]) - start
                break
    immobility_bouts = _bouts(rows, "immobile")
    struggling_bouts = _bouts(rows, "struggling")

    def percent(value: float) -> float | None:
        return timeline.ratio(value * 100.0, total)

    if mode == "lis":
        threshold_block = {
            key: {name: info["thresholds"][key][name]
                  for name in ("low", "mid", "high", "very_high")}
            for key in ("r", "p", "z", "h", "f", "s", "g", "b")}
        estimated = True
    else:
        threshold_block = {
            "flow_thr_hi": info["flow_thr_hi"],
            "flow_thr_lo": info["flow_thr_lo"],
            "speed_std_thr_hi": info["speed_thr_hi"],
            "speed_std_thr_lo": info["speed_thr_lo"],
        }
        estimated = bool(info["estimated"])

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
        "thresholds": threshold_block,
        "threshold_estimated": estimated,
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
                "speed_mm_s", "flow_p90_mm_s", "flow_mean_mm_s",
                "lis_r", "lis_mask_residual", "lis_gray_mean_d",
                "lis_peripheral_d", "lis_core_d", "lis_core_ratio",
                "lis_z", "lis_hu_distance", "lis_body_length_px",
                "lis_aspect_ratio", "lis_solidity", "lis_extent",
                "lis_displacement_ratio", "lis_segmentation_bad",
                "lis_climbing_suspected"):
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
        mode = str(options.get("mode", "classic"))
        first_seen = timeline.first_seen_in_window(
            KEY, captured[unit_id], video_window, geometry)
        window = timeline.resolve_unit_window(
            unit.get("window"), video_window, first_seen)
        rows = timeline.clip_rows(captured[unit_id], window)
        _filter_pairs(rows, window)

        scale, line_px = activity.line_scale(geometry.get("body_line"))
        if mode == "lis":
            info = derive_lis_features(rows, options)
            unit_events = build_events(rows, unit_id,
                                       unit.get("name") or "",
                                       confirm_swing=False)
            unit_summary = summarize(rows, unit, window, info, unit_events,
                                     mode="lis")
        else:
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
                sensitivity=float(
                    options.get("behavior_threshold_sensitivity") or 0.0))
            classic_states(rows, flow_hi=info["flow_thr_hi"],
                           flow_lo=info["flow_thr_lo"],
                           speed_hi=info["speed_thr_hi"],
                           speed_lo=info["speed_thr_lo"])
            unit_events = build_events(rows, unit_id, unit.get("name") or "")
            unit_summary = summarize(rows, unit, window, info, unit_events)

        observations.extend(rows)
        events.extend(unit_events)
        summary.append(unit_summary)
        if mode == "lis":
            priors = [float(row["lis_motion_prior"]) for row in rows
                      if is_number(row.get("lis_motion_prior"))]
            mode_counts: dict[str, int] = {}
            for row in rows:
                key = str(row.get("lis_score_mode") or "")
                if key:
                    mode_counts[key] = mode_counts.get(key, 0) + 1
            unit_metadata[unit_id] = {
                "mode": "lis",
                "body_length_mm": activity.BODY_LENGTH_MM,
                "mm_per_px": scale,
                "body_line_px": line_px,
                "coordinate_unit": "mm",
                "thresholds": info["thresholds"],
                "motion_prior": {"min": min(priors) if priors else None,
                                 "max": max(priors) if priors else None},
                "score_mode": mode_counts,
                "samples": int(info["samples"]),
                "low_information": bool(info["low_information"]),
                "rotate_align": bool(options.get("lis_rotate_align")),
                "rescue_enabled": bool(options.get("lis_rescue_enabled")),
                "state_labels": ["struggling", "immobile", "unknown",
                                 "outside"],
            }
        else:
            unit_metadata[unit_id] = {
                "mode": "classic",
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

    options = unit.get("options") or {}
    mode = str(options.get("mode", "classic"))
    if mode not in MODES:
        raise ValidationError("未知行为算法", "options.mode")

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

    scale, _ = activity.line_scale(geometry.get("body_line"))
    if mode == "lis":
        info = derive_lis_features(rows, options)
        events = build_events(rows, unit["id"], unit.get("name") or "",
                              confirm_swing=False)
        summary = summarize(rows, unit, unit_window, info, events,
                            mode="lis")
    else:
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
            sensitivity=float(
                options.get("behavior_threshold_sensitivity") or 0.0))
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
    behavior = {
        "mode": mode,
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
        "threshold_estimated": bool(summary["threshold_estimated"]),
        "threshold_samples": int(info["samples"]),
        "state_at_time": state_at_time,
    }
    if mode == "lis":
        # 短段阈值样本来自预览范围自身，与全程分析的估计可能不同。
        behavior["threshold_note"] = "行为预览为短段阈值估计，与全程分析可能不同"
        behavior["low_information"] = bool(info["low_information"])
    return behavior
