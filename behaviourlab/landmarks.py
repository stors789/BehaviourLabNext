"""经典关键点人工标注与像素误差。

会话与实验结果分开存放（``data_root/landmarks/{id}.json``），只读源视频，不
修改任何实验指标。评估用与检测预览相同的经典检测器，按帧顺序跑到最大选中
帧以保留真实连续关联，只对选中帧记录预测。
"""

from __future__ import annotations

import math
import os
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import detection, experiments, media, observation
from .config import DataPaths
from .detection import common_detection_defaults
from .schema import (
    ValidationError,
    check_json_value,
    dumps_strict,
    is_number,
    is_valid_id,
    new_id,
)
from .store import Store, atomic_write_json, read_json_file, utc_now

POINTS = ("body", "nose", "tail", "left_ear", "right_ear")
# 预测只支持身体中心、鼻尖与尾根；耳点可标注但不给预测。
PREDICTED = ("body", "nose", "tail")
SUPPORTED_BY_ASSAY = {
    "nor": PREDICTED,
    "barnes": PREDICTED,
}
DEFAULT_SUPPORTED = ("body",)
ORIENTATIONS = (0, 90, 180, 270)
LOCK = threading.RLock()


def _assays() -> set[str]:
    return {entry["key"] for entry in experiments.catalog()}


def supported_points(assay: str) -> tuple[str, ...]:
    return SUPPORTED_BY_ASSAY.get(str(assay), DEFAULT_SUPPORTED)


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _non_negative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidationError("必须是非负整数", field)
    return int(value)


def _frames(value: Any) -> list[int]:
    """必须是非空、无重复的非负整数数组；不提供服务端默认帧。"""

    if not isinstance(value, (list, tuple)) or not value:
        raise ValidationError("frames 必须是非空数组", "frames")
    seen: set[int] = set()
    for index, item in enumerate(value):
        frame = _non_negative_int(item, f"frames[{index}]")
        if frame in seen:
            raise ValidationError("frames 不能重复", f"frames[{index}]")
        seen.add(frame)
    return sorted(seen)


def _point(value: Any, field: str) -> dict:
    if not isinstance(value, Mapping):
        raise ValidationError("点必须是对象", field)
    visible = value.get("visible")
    if not isinstance(visible, bool):
        raise ValidationError("visible 必须是布尔值", f"{field}.visible")
    if visible:
        if not is_number(value.get("x")) or not is_number(value.get("y")):
            raise ValidationError("visible=true 时坐标必须是有限数",
                                  f"{field}.x")
        return {"visible": True, "x": float(value["x"]), "y": float(value["y"])}
    return {"visible": False, "x": None, "y": None}


def _records(value: Any, frames: Sequence[int]) -> list[dict]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValidationError("records 必须是数组", "records")
    allowed = set(frames)
    seen: set[int] = set()
    prepared: list[dict] = []
    for index, item in enumerate(value):
        field = f"records[{index}]"
        if not isinstance(item, Mapping):
            raise ValidationError("记录必须是对象", field)
        frame = _non_negative_int(item.get("frame"), f"{field}.frame")
        if frame not in allowed:
            raise ValidationError("只记录已选中的帧", f"{field}.frame")
        if frame in seen:
            raise ValidationError("同一帧只能有一条记录", f"{field}.frame")
        seen.add(frame)
        points = item.get("points") or {}
        if not isinstance(points, Mapping):
            raise ValidationError("points 必须是对象", f"{field}.points")
        cleaned = {}
        for name, entry in points.items():
            if name not in POINTS:
                raise ValidationError(f"未知的点名：{name}", f"{field}.points")
            cleaned[name] = _point(entry, f"{field}.points.{name}")
        prepared.append({"frame": frame, "points": cleaned})
    prepared.sort(key=lambda entry: entry["frame"])
    return prepared


def _session_path(paths: DataPaths, session_id: str) -> Path:
    if not is_valid_id(session_id) or len(session_id) != 32:
        raise ValidationError("会话 ID 无效", "id")
    return Path(paths.landmarks) / f"{session_id}.json"


def list_sessions(paths: DataPaths) -> list[dict]:
    directory = Path(paths.landmarks)
    if not directory.is_dir():
        return []
    sessions = []
    for item in sorted(directory.glob("*.json")):
        try:
            record = read_json_file(item, default=None)
        except (OSError, ValueError):
            continue
        if isinstance(record, Mapping) and record.get("id"):
            sessions.append(dict(record))
    sessions.sort(key=lambda entry: str(entry.get("updated_at") or ""),
                  reverse=True)
    return sessions


def get_session(paths: DataPaths, session_id: str) -> dict | None:
    try:
        path = _session_path(paths, session_id)
    except ValidationError:
        return None
    try:
        record = read_json_file(path, default=None)
    except (OSError, ValueError):
        return None
    return dict(record) if isinstance(record, Mapping) else None


def create_session(store: Store, paths: DataPaths,
                   payload: Mapping[str, Any]) -> dict:
    if not isinstance(payload, Mapping):
        raise ValidationError("会话必须是对象", "")
    asset_id = payload.get("asset_id")
    if not is_valid_id(asset_id):
        raise ValidationError("素材 ID 无效", "asset_id")
    asset = store.get_asset(asset_id)
    if asset is None:
        raise ValidationError("素材未登记", "asset_id")
    assay = payload.get("assay") or "nor"
    if assay not in _assays():
        raise ValidationError("未知实验", "assay")
    orientation = payload.get("orientation", 0)
    if isinstance(orientation, bool) or orientation not in ORIENTATIONS:
        raise ValidationError("方向必须是 0/90/180/270", "orientation")
    unit = payload.get("unit")
    if not isinstance(unit, Mapping):
        raise ValidationError("缺少单位", "unit")
    geometry = unit.get("geometry")
    if not isinstance(geometry, Mapping):
        raise ValidationError("unit.geometry 必须是对象", "unit.geometry")
    frames = _frames(payload.get("frames"))
    tolerance = payload.get("tolerance_px", 10)
    if not is_number(tolerance) or not 0 < float(tolerance):
        raise ValidationError("容差必须是有限正数", "tolerance_px")
    fps_override = payload.get("fps_override")
    if fps_override is not None and (not is_number(fps_override)
                                     or float(fps_override) <= 0):
        raise ValidationError("解释帧率必须是有限正数", "fps_override")
    detection_settings = payload.get("detection") or {}
    if not isinstance(detection_settings, Mapping):
        raise ValidationError("detection 必须是对象", "detection")
    records = _records(payload.get("records"), frames)

    path = asset.get("path")
    if not isinstance(path, str) or not path:
        raise ValidationError("素材缺少路径", "asset_id")
    try:
        stat = os.stat(path)
    except OSError as error:
        raise ValidationError(f"源视频不可读取：{error}", "asset_id") from error

    session_id = new_id()
    now = utc_now()
    session = {
        "id": session_id,
        "asset_id": asset_id,
        "asset_name": asset.get("name"),
        "assay": assay,
        "orientation": int(orientation),
        "frames": frames,
        "unit": {
            "id": unit.get("id"),
            "name": unit.get("name"),
            "geometry": deepcopy(dict(geometry)),
            "options": deepcopy(dict(unit.get("options") or {})),
            "detection": deepcopy(dict(unit.get("detection") or {})),
        },
        "detection": deepcopy(dict(detection_settings)),
        "fps_override": (float(fps_override) if fps_override is not None
                         else None),
        "tolerance_px": float(tolerance),
        "records": records,
        "asset_identity": {
            "path": str(Path(path).resolve()),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        },
        "evaluation": None,
        "created_at": now,
        "updated_at": now,
    }
    check_json_value(session, "session")
    with LOCK:
        target = _session_path(paths, session_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(target, session)
    return session


def update_session(paths: DataPaths, session_id: str,
                   payload: Mapping[str, Any]) -> dict:
    existing = get_session(paths, session_id)
    if existing is None:
        raise ValidationError("会话不存在", "id")
    if not isinstance(payload, Mapping):
        raise ValidationError("更新必须是对象", "")
    for key in payload:
        if key not in ("records", "tolerance_px"):
            raise ValidationError(f"不允许修改该字段：{key}", key)
    updated = deepcopy(existing)
    if "frames" in payload:
        raise ValidationError("不允许修改该字段：frames", "frames")
    if "tolerance_px" in payload:
        tolerance = payload["tolerance_px"]
        if not is_number(tolerance) or not 0 < float(tolerance):
            raise ValidationError("容差必须是有限正数", "tolerance_px")
        updated["tolerance_px"] = float(tolerance)
    if "records" in payload:
        updated["records"] = _records(payload["records"], existing["frames"])
    updated["evaluation"] = None
    updated["updated_at"] = utc_now()
    check_json_value(updated, "session")
    with LOCK:
        atomic_write_json(_session_path(paths, session_id), updated)
    return updated


def _identity_issues(session: Mapping[str, Any]) -> list[dict]:
    identity = session.get("asset_identity") or {}
    path = identity.get("path")
    if not path:
        return [{"field": "asset_identity", "message": "缺少冻结身份"}]
    try:
        stat = os.stat(path)
    except OSError:
        return [{"field": "asset_identity", "message": "源视频已不可读取"}]
    issues = []
    if int(stat.st_size) != identity.get("size"):
        issues.append({"field": "asset_identity", "message": "源视频大小已变化"})
    if int(stat.st_mtime_ns) != identity.get("mtime_ns"):
        issues.append({"field": "asset_identity",
                       "message": "源视频修改时间已变化"})
    return issues


def _settings_of(session: Mapping[str, Any]) -> dict:
    assay = session.get("assay")
    module = experiments.get(assay) if experiments.exists(assay) else None
    base = common_detection_defaults()
    if module is not None:
        base.update(deepcopy(getattr(module, "DETECTION_DEFAULTS", {})))
    for layer in (session.get("detection"),
                  (session.get("unit") or {}).get("detection")):
        if isinstance(layer, Mapping):
            base.update(deepcopy(dict(layer)))
    return base


def _options_of(session: Mapping[str, Any]) -> dict:
    assay = session.get("assay")
    base = {}
    if experiments.exists(assay):
        base.update(deepcopy(experiments.get(assay).DEFAULTS))
    unit_options = (session.get("unit") or {}).get("options")
    if isinstance(unit_options, Mapping):
        base.update(deepcopy(dict(unit_options)))
    return base


def _predictions(result: Any) -> dict:
    body = result.center
    nose = result.nose
    tail = result.tail_base
    return {
        "body": [float(body[0]), float(body[1])] if body else None,
        "nose": [float(nose[0]), float(nose[1])] if nose else None,
        "tail": [float(tail[0]), float(tail[1])] if tail else None,
    }


def _distance(first: Sequence[float], second: Sequence[float]) -> float:
    return math.hypot(float(first[0]) - float(second[0]),
                      float(first[1]) - float(second[1]))


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * fraction
    low = int(math.floor(position))
    high = min(len(ordered) - 1, low + 1)
    weight = position - low
    return float(ordered[low] * (1 - weight) + ordered[high] * weight)


def evaluate(paths: DataPaths, session: Mapping[str, Any], *,
             cancel: Any = None) -> dict:
    """逐帧跑到最大选中帧，只对选中帧记录预测并算像素误差。"""

    issues = _identity_issues(session)
    if issues:
        raise ValidationError(issues[0]["message"], issues[0]["field"])
    frames = list(session.get("frames") or [])
    if not frames:
        raise ValidationError("会话没有选中帧", "frames")
    identity = session["asset_identity"]
    assay = str(session.get("assay"))
    settings = _settings_of(session)
    options = _options_of(session)
    unit = session.get("unit") or {}
    supported = supported_points(assay)
    orientation = int(session.get("orientation", 0) or 0)

    source = media.VideoSource(
        identity["path"], start_s=0.0, end_s=None, orientation=orientation,
        fps_override=session.get("fps_override"), sample_fps=None,
        decode_images=True,
    )
    wanted = set(frames)
    last = max(wanted)
    key = ("landmarks", session.get("id"))
    region = None
    detector = None
    predicted: dict[int, dict] = {}
    times: dict[int, float] = {}
    seen_frames: set[int] = set()
    frame_size: tuple[int, int] | None = None
    for frame in source.iter_frames():
        if cancel is not None and cancel():
            raise ValidationError("标注评估已取消", "job")
        if frame.image is None:
            continue
        if frame.index > last:
            break
        if region is None:
            height, width = frame.image.shape[:2]
            frame_size = (width, height)
            try:
                region, _ = detection.geometry_region(
                    unit.get("geometry") or {}, assay, options, width, height)
            except detection.DetectionError as error:
                raise ValidationError(str(error), "unit.geometry") from error
            background = detection.prepare_background(
                settings, {"path": identity["path"], "metadata": {}},
                # 背景支路与主源同一份冻结方向/解释FPS及既有无采样/
                # 原窗规则（从 0 起、无采样、窗口终点未知）；不恢复
                # 0°/null，避免旋转源上的背景尺寸错配。
                {"window": {"start_s": 0.0},
                 "fps_override": session.get("fps_override"),
                 "sample_fps": None},
                orientation, width, height,
                cancel=(lambda: bool(cancel and cancel())))
            detector = detection.shared_detector(key, settings,
                                                 background=background)
        # 连续处理到最大选中帧：检测与关联不因跳选断链；选中帧只控制
        # 哪些预测被记录，不改变 tracker 的真实连续状态。
        try:
            result = observation.track(
                key, unit.get("id") or session.get("id"),
                image=frame.image, region=region, settings=settings,
                signature=observation.signature_of(
                    unit.get("geometry") or {}, settings, False),
                assay=assay, frame_index=frame.index,
                time_s=float(frame.time_s),
                foreground=detector.foreground(frame.image),
                retain_units=[unit.get("id") or session.get("id")],
            )
        except detection.DetectionError as error:
            raise ValidationError(str(error), "detection") from error
        if frame.index in wanted:
            seen_frames.add(int(frame.index))
            times[int(frame.index)] = float(frame.time_s)
            predicted[int(frame.index)] = _predictions(result)

    missing = [index for index in frames if index not in seen_frames]
    if missing:
        raise ValidationError(
            f"选中帧无法解码：{missing[0]}", "frames")
    if frame_size is not None:
        _validate_record_bounds(session.get("records") or [], frame_size)

    evaluation = compute_evaluation(
        session.get("records") or [], predicted, times,
        tolerance=float(session.get("tolerance_px") or 10),
        supported=supported)
    return evaluation


def _validate_record_bounds(records: Sequence[Mapping[str, Any]],
                            frame_size: tuple[int, int]) -> None:
    """人工可见标记必须落在旋转后的源像素范围内，越界直接拒绝。"""

    width, height = frame_size
    for record in records:
        frame_index = int(record.get("frame"))
        for name, mark in (record.get("points") or {}).items():
            if not isinstance(mark, Mapping) or not mark.get("visible"):
                continue
            x, y = mark.get("x"), mark.get("y")
            inside = (isinstance(x, (int, float))
                      and isinstance(y, (int, float))
                      and not isinstance(x, bool) and not isinstance(y, bool)
                      and 0 <= float(x) < width and 0 <= float(y) < height)
            if not inside:
                raise ValidationError(
                    f"帧 {frame_index} 的 {name} 标注超出旋转后画面"
                    f"（0≤x<{width}，0≤y<{height}）",
                    f"records[{frame_index}].points.{name}")


def compute_evaluation(records: Sequence[Mapping[str, Any]],
                       predicted: Mapping[int, Mapping[str, Any]],
                       times: Mapping[int, float], *, tolerance: float,
                       supported: Sequence[str]) -> dict:
    """只做误差统计：人工可见点、覆盖、容差内命中与匹配误差分位。"""

    by_point: dict[str, dict] = {
        name: {"visible": 0, "covered": 0, "hit": 0, "errors": []}
        for name in POINTS
    }
    rows = []
    for record in records:
        frame_index = int(record["frame"])
        points_out = {}
        for name, mark in (record.get("points") or {}).items():
            stat = by_point[name]
            visible = bool(mark.get("visible"))
            if visible:
                stat["visible"] += 1
            prediction = ((predicted.get(frame_index) or {}).get(name)
                          if name in supported else None)
            error = None
            hit = None
            if visible and prediction is not None:
                error = _distance((mark["x"], mark["y"]), prediction)
                stat["covered"] += 1
                stat["errors"].append(error)
                hit = error <= tolerance
                if hit:
                    stat["hit"] += 1
            elif visible:
                # 人工可见但没有（或该点不支持）预测：明确未命中，不缩小分母。
                hit = False
            points_out[name] = {
                "visible": visible,
                "prediction": prediction,
                "error_px": error,
                "hit": hit,
            }
        rows.append({"frame": frame_index, "time_s": times.get(frame_index),
                     "points": points_out})

    total_visible = sum(stat["visible"] for stat in by_point.values())
    total_covered = sum(stat["covered"] for stat in by_point.values())
    total_hit = sum(stat["hit"] for stat in by_point.values())
    errors = [value for stat in by_point.values() for value in stat["errors"]]
    median = _percentile(errors, 0.5)
    p90 = _percentile(errors, 0.9)
    metrics = {
        "visible": total_visible,
        "covered": total_covered,
        "hit": total_hit,
        "coverage": (total_covered / total_visible) if total_visible else None,
        "hit_rate": (total_hit / total_visible) if total_visible else None,
        # 匹配误差包含超出容差的点，不筛选成功命中。
        "matched_error_median_px": median,
        "matched_error_p90_px": p90,
    }
    for stat in by_point.values():
        # 逐点误差分位：各点只汇总自己的匹配误差；无匹配为 null，
        # 不把总体分位数重复填给每个点。
        stat["matched_error_median_px"] = _percentile(stat["errors"], 0.5)
        stat["matched_error_p90_px"] = _percentile(stat["errors"], 0.9)
        stat.pop("errors", None)
    return {
        "tolerance_px": float(tolerance),
        "metrics": metrics,
        "by_point": by_point,
        "records": rows,
        "supported_points": list(supported),
        "matched_error_median_px": median,
        "matched_error_p90_px": p90,
        "note": "这是关键点像素误差，不是动物行为准确率。",
    }


def save_evaluation(paths: DataPaths, session: Mapping[str, Any],
                    evaluation: Mapping[str, Any]) -> dict:
    updated = deepcopy(dict(session))
    updated["evaluation"] = deepcopy(dict(evaluation))
    updated["updated_at"] = utc_now()
    check_json_value(updated, "session")
    with LOCK:
        atomic_write_json(_session_path(paths, updated["id"]), updated)
    return updated
