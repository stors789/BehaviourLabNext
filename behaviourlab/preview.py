"""检测/行为预览：真实源帧 + 当前几何叠画。

预览只做画面与配置核对；实验分析尚未实现，不返回任何统计量。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from . import detection, experiments, geometry as geo, media, observation
from .config import DataPaths
from .engine import is_manual_path, resolve_detection
from .schema import ValidationError, is_number, new_id
from .store import Store

GUIDE_COLOR = (150, 150, 150)


def _bgr(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    red, green, blue = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return (blue, green, red)


def _thickness(width: int, height: int) -> int:
    return max(2, int(round(min(width, height) / 240)))


def _draw_candidates(canvas, result, thickness: int) -> None:
    """画出真实候选轮廓；最佳候选与其他候选分色，不改动源帧。"""

    import cv2
    import numpy as np

    for index, candidate in enumerate(result.candidates):
        color = _bgr("#ff6e46") if index == 0 else _bgr("#8ed7c7")
        points = np.array(
            [[int(round(point[0])), int(round(point[1]))]
             for point in candidate.get("contour") or []],
            dtype=np.int32,
        )
        if len(points) >= 2:
            cv2.polylines(canvas, [points], True, color, thickness, cv2.LINE_AA)
    if result.center:
        cv2.drawMarker(
            canvas,
            (int(round(result.center[0])), int(round(result.center[1]))),
            _bgr("#ff6e46"), cv2.MARKER_CROSS, 12, 1, cv2.LINE_AA,
        )
    if result.nose and result.tail_base:
        # 只在有细尾证据时画鼻尾；未知时不补方向。
        nose = (int(round(result.nose[0])), int(round(result.nose[1])))
        tail = (int(round(result.tail_base[0])),
                int(round(result.tail_base[1])))
        cv2.line(canvas, tail, nose, _bgr("#1c5f9e"),
                 max(1, thickness - 1), cv2.LINE_AA)
        cv2.circle(canvas, nose, max(3, thickness + 1), _bgr("#1c5f9e"), -1,
                   cv2.LINE_AA)


def _interpreted_duration(metadata: Mapping[str, Any],
                          video: Mapping[str, Any]) -> float | None:
    return detection.interpreted_duration(metadata, video)


def _background_for(settings: Mapping[str, Any], asset: Mapping[str, Any],
                    video: Mapping[str, Any], orientation: int,
                    width: int, height: int):
    """按方法准备背景；threshold/local_otsu 不需要。"""

    return detection.prepare_background(settings, asset, video, orientation,
                                        width, height)


def _draw_shape(canvas, shape: Mapping[str, Any], color, thickness: int) -> None:
    import cv2

    kind = shape.get("type")
    if kind in ("polygon", "quad"):
        points = _integer_points(shape.get("points") or [])
        if len(points) >= 2:
            cv2.polylines(canvas, [points], True, color, thickness, cv2.LINE_AA)
    elif kind == "line":
        points = _integer_points(shape.get("points") or [])
        if len(points) >= 2:
            cv2.line(canvas, points[0], points[1], color, thickness, cv2.LINE_AA)
    elif kind == "point":
        points = _integer_points(shape.get("points") or [])
        if len(points):
            radius = max(3, thickness * 2)
            cv2.circle(canvas, points[0], radius, color, thickness, cv2.LINE_AA)
    elif kind == "ellipse":
        center = shape.get("center")
        radii = shape.get("radii")
        if isinstance(center, (list, tuple)) and radii:
            cv2.ellipse(
                canvas,
                (int(round(center[0])), int(round(center[1]))),
                (int(round(float(radii[0]))), int(round(float(radii[1])))),
                float(shape.get("angle", 0.0)),
                0,
                360,
                color,
                thickness,
                cv2.LINE_AA,
            )
        elif shape.get("points"):
            fitted = geo.fit_ellipse(shape["points"])
            _draw_shape(canvas, {**fitted, "type": "ellipse"}, color, thickness)
    elif kind == "holes":
        radius = max(4, thickness * 2)
        for hole in shape.get("holes") or []:
            center = hole.get("center") or [0, 0]
            point = (int(round(center[0])), int(round(center[1])))
            cv2.circle(canvas, point, radius, color, thickness, cv2.LINE_AA)
            label = str(hole.get("id", ""))
            if label:
                cv2.putText(canvas, label, (point[0] + radius + 2, point[1] - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    elif kind == "union":
        for part in shape.get("shapes") or []:
            _draw_shape(canvas, part, color, thickness)


def _integer_points(points) -> Any:
    import numpy as np

    result = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        result.append([int(round(float(point[0]))), int(round(float(point[1])))])
    return np.array(result, dtype=np.int32) if result else np.zeros((0, 2), np.int32)


def _draw_guides(canvas, assay: str, options: Mapping[str, Any],
                 unit_geometry: Mapping[str, Any], thickness: int) -> None:
    import cv2

    if assay not in ("mwm", "ezm"):
        return
    arena = unit_geometry.get("arena") or unit_geometry.get("outer")
    if not arena:
        return
    if arena.get("type") == "ellipse" and arena.get("center") and arena.get("radii"):
        center = arena["center"]
        rx, ry = float(arena["radii"][0]), float(arena["radii"][1])
        angle = float(arena.get("angle", 0.0))
    elif arena.get("points"):
        fitted = geo.fit_ellipse(arena["points"])
        center = fitted["center"]
        rx, ry = fitted["radii"]
        angle = fitted["angle"]
    else:
        return

    if assay == "mwm":
        offset = float(options.get("coordinate_rotation_deg", 0.0) or 0.0)
        length = max(rx, ry)
        for index in range(2):
            theta = math.radians(angle + offset + index * 90.0)
            dx, dy = math.cos(theta) * length, math.sin(theta) * length
            start = (int(center[0] - dx), int(center[1] - dy))
            end = (int(center[0] + dx), int(center[1] + dy))
            cv2.line(canvas, start, end, GUIDE_COLOR, 1, cv2.LINE_AA)
        return

    if assay == "ezm":
        ring = float(options.get("ring_width", 0.0) or 0.0)
        diameter = float(options.get("outer_diameter", 0.0) or 0.0)
        if ring <= 0 or diameter <= 0:
            return
        outer_px = max(rx, ry)
        inner_px = outer_px * max(0.0, (diameter / 2.0 - ring) / (diameter / 2.0))
        labels = ("O1", "C1", "O2", "C2")
        colors = (_bgr("#16684e"), _bgr("#1c5f9e"),
                  _bgr("#16684e"), _bgr("#1c5f9e"))
        for index, label in enumerate(labels):
            theta = math.radians(angle + 45.0 + index * 90.0)
            dx, dy = math.cos(theta), math.sin(theta)
            start = (int(center[0] + dx * inner_px), int(center[1] + dy * inner_px))
            end = (int(center[0] + dx * outer_px), int(center[1] + dy * outer_px))
            cv2.line(canvas, start, end, colors[index], 1, cv2.LINE_AA)
            cv2.putText(canvas, label, (end[0] + 3, end[1] - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors[index], 1, cv2.LINE_AA)


def _find_video(draft: Mapping[str, Any], asset_id: str | None) -> Mapping[str, Any]:
    videos = draft.get("videos") or []
    if asset_id:
        for video in videos:
            if video.get("asset_id") == asset_id:
                return video
        raise ValidationError("草稿中没有该视频", "asset_id")
    if not videos:
        raise ValidationError("草稿还没有视频", "asset_id")
    return videos[0]


def _find_unit(video: Mapping[str, Any], unit_id: str | None) -> Mapping[str, Any]:
    units = video.get("units") or []
    if unit_id:
        for unit in units:
            if unit.get("id") == unit_id:
                return unit
        raise ValidationError("该视频没有这只小鼠", "unit_id")
    if not units:
        raise ValidationError("该视频还没有小鼠", "unit_id")
    return units[0]


def _scope(assay: str, video: Mapping[str, Any], unit: Mapping[str, Any],
           options: Mapping[str, Any], time_s: float, kind: str,
           duration_s: float | None) -> dict:
    window = video.get("window") or {}
    start = float(window.get("start_s") or 0.0)
    end = window.get("end_s")
    end_value = float(end) if is_number(end) else duration_s
    if kind == "detection":
        return {"start_s": time_s, "end_s": time_s, "frame_count": 1,
                "mode": "frame"}
    if is_manual_path(assay, options.get("mode"), unit.get("manual")):
        scope_end = end_value if end_value is not None else max(start, time_s)
        scope_end = max(start, scope_end)
        return {
            "start_s": start,
            "end_s": scope_end,
            "duration_s": scope_end - start,
            "mode": "configured",
            "manual": True,
        }
    scope_start = max(start, time_s - 1.0)
    scope_end = time_s + 3.0
    if end_value is not None:
        scope_end = min(end_value, scope_end)
    scope_end = max(scope_start, scope_end)
    return {
        "start_s": scope_start,
        "end_s": scope_end,
        "duration_s": scope_end - scope_start,
        "mode": "range",
    }


def create_preview(
    store: Store,
    paths: DataPaths,
    draft_id: str,
    payload: Mapping[str, Any],
) -> dict:
    """生成一张真实源帧预览；失败抛 ValidationError / media.MediaError。"""

    draft = store.get_draft(draft_id)
    if draft is None:
        raise ValidationError("草稿不存在", "draft_id")

    asset_id = payload.get("asset_id")
    video = _find_video(draft, asset_id if isinstance(asset_id, str) else None)
    asset = store.get_asset(video.get("asset_id"))
    if asset is None:
        raise ValidationError("素材不存在", "asset_id")
    unit = _find_unit(video, payload.get("unit_id")
                      if isinstance(payload.get("unit_id"), str) else None)

    kind = payload.get("kind", "detection")
    if kind not in ("detection", "behavior"):
        raise ValidationError("预览类型必须是 detection 或 behavior", "kind")

    frame_index = payload.get("frame_index")
    if frame_index is not None:
        if isinstance(frame_index, bool) or not isinstance(frame_index, int):
            raise ValidationError("帧索引必须是非负整数", "frame_index")
        if frame_index < 0:
            raise ValidationError("帧索引必须是非负整数", "frame_index")
        orientation = int(video.get("orientation", 0) or 0)
        info = media.read_frame_info(
            asset["path"], frame_index, orientation,
            video.get("fps_override"),
        )
    else:
        time_s = payload.get("time_s")
        if not is_number(time_s):
            raise ValidationError("需要帧索引或时间", "frame_index")
        orientation = int(video.get("orientation", 0) or 0)
        info = media.read_frame_at_time(
            asset["path"], float(time_s), orientation, video.get("fps_override")
        )

    geometry = unit.get("geometry") or {}
    options: dict = {}
    for layer in (draft.get("options"), video.get("options"), unit.get("options")):
        if isinstance(layer, Mapping):
            options.update(layer)
    settings = resolve_detection(draft["assay"], draft=draft, video=video, unit=unit)

    height, width = info.image.shape[:2]
    orientation = int(video.get("orientation", 0) or 0)
    manual_scope = is_manual_path(
        draft.get("assay", ""), options.get("mode"), unit.get("manual"))
    # 人工记录路径不运行自动检测，也不为它伪造检测区域。
    detect_here = not (kind == "behavior" and manual_scope)
    region_source = None
    result = None
    if detect_here:
        try:
            region, region_source = detection.geometry_region(
                geometry, draft["assay"], options, width, height
            )
            background = _background_for(settings, asset, video, orientation,
                                         width, height)
            window_key = repr(video.get("window"))
            video_key = (asset.get("id"), orientation, window_key)
            detector = detection.shared_detector(
                video_key, settings, background=background,
            )
            session_key = (draft.get("id"),) + video_key
            # 证据随会话保留：视频被移出草稿、单位被删时才释放。
            observation.retain_videos(
                draft.get("id"),
                [(draft.get("id"), item.get("asset_id"),
                  int(item.get("orientation", 0) or 0),
                  repr(item.get("window")))
                 for item in draft.get("videos") or []
                 if isinstance(item, Mapping)],
            )
            result = observation.track(
                session_key, unit.get("id") or id(unit),
                image=info.image, region=region, settings=settings,
                signature=observation.signature_of(
                    geometry, settings, bool(options.get("dark_visible"))),
                assay=draft["assay"], frame_index=info.index,
                time_s=float(info.time_s),
                foreground=detector.foreground(info.image),
                retain_units=[item.get("id") for item in video.get("units") or []
                              if isinstance(item, Mapping) and item.get("id")],
            )
        except detection.DetectionError as error:
            raise ValidationError(str(error), "detection") from error

    canvas = info.image.copy()
    thickness = _thickness(width, height)
    if result is not None:
        _draw_candidates(canvas, result, thickness)
    drawn = []
    for key, shape in geometry.items():
        if not isinstance(shape, Mapping):
            continue
        _draw_shape(canvas, shape, _bgr("#ffd26f"), thickness)
        drawn.append(key)
    _draw_guides(canvas, draft.get("assay", ""), unit.get("options") or {},
                 geometry, thickness)

    for other in video.get("units") or []:
        if other.get("id") == unit.get("id"):
            continue
        for shape in (other.get("geometry") or {}).values():
            if isinstance(shape, Mapping):
                _draw_shape(canvas, shape, _bgr("#8ed7c7"), 1)

    import cv2

    filename = f"{new_id()}.png"
    target = Path(paths.previews) / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), canvas):
        raise ValidationError("无法保存预览图", "preview")

    metadata = asset.get("metadata") or {}
    duration_s = metadata.get("duration_s")
    scope = _scope(draft.get("assay", ""), video, unit, options,
                   float(info.time_s), kind,
                   float(duration_s) if is_number(duration_s) else None)

    steps = experiments.get(draft["assay"]).SPEC.get("geometry", [])
    missing = [
        step["label"] for step in steps
        if step.get("required") and step.get("key") not in geometry
    ]
    warnings: list[str] = []
    if result is None:
        warnings.append("人工记录路径不运行自动检测。")
    elif result.missing_reason == "foreground_saturated":
        warnings.append("前景覆盖过高，已拒绝整片掩膜。")
    elif result.missing_reason == "no_candidate":
        warnings.append("没有符合面积与质量要求的候选。")
    elif result.missing_reason == "inconsistent_candidate_size":
        warnings.append("候选尺寸与已接受体型不一致。")
    elif result.missing_reason == "no_nearby_candidate":
        warnings.append("附近没有符合位置门限的候选。")
    if (result is not None and result.detected
            and draft.get("assay") in observation.HEAD_ASSAYS
            and result.head_angle is None):
        warnings.append("头向未判定：缺少细尾证据。")
    if missing:
        warnings.append("尚未绘制：" + "、".join(missing))

    return {
        "image_url": f"/api/previews/{filename}",
        "frame_index": info.index,
        "time_s": float(info.time_s),
        "clock_source": info.clock_source,
        "preview_scope": scope,
        "summary": {
            "kind": kind,
            "drawn_steps": drawn,
            "missing_steps": missing,
            "detection_available": result is not None,
            "detection": None if result is None else {
                "candidates": len(result.candidates),
                "missing_reason": result.missing_reason,
                "region": region_source,
                "center": list(result.center) if result.center else None,
                "quality": float(result.quality),
                "nose": list(result.nose) if result.nose else None,
                "tail_base": (list(result.tail_base)
                              if result.tail_base else None),
                "head_angle": result.head_angle,
                "head_quality": float(result.head_quality),
            },
            "manual_preserved": is_manual_path(draft.get("assay", ""),
                                               options.get("mode"),
                                               unit.get("manual")),
        },
        "warnings": warnings,
    }
