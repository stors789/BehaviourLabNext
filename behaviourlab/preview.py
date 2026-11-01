"""检测/行为预览：真实源帧 + 当前几何叠画。

预览只做画面与配置核对；实验分析尚未实现，不返回任何统计量。
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from . import detection, experiments, geometry as geo, media, observation
from .config import DataPaths
from .engine import is_manual_path, resolve_detection, resolve_options
from .schema import ValidationError, is_number, new_id
from .store import Store

GUIDE_COLOR = (150, 150, 150)


def _bgr(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    red, green, blue = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return (blue, green, red)


def _thickness(width: int, height: int) -> int:
    return max(2, int(round(min(width, height) / 240)))


def _selected_index(result) -> int | None:
    """已关联候选在列表中的位置；缺失帧没有选中，不按面积最大冒认。"""

    center = result.center
    if not center:
        return None
    for index, candidate in enumerate(result.candidates):
        item = candidate.get("center")
        if not item:
            continue
        if (abs(float(item[0]) - float(center[0])) <= 1e-6
                and abs(float(item[1]) - float(center[1])) <= 1e-6):
            return index
    return None


def _draw_candidates(canvas, result, thickness: int) -> None:
    """画出真实候选轮廓；已关联候选与其他候选分色，不改动源帧。"""

    import cv2
    import numpy as np

    chosen = _selected_index(result)
    for index, candidate in enumerate(result.candidates):
        color = _bgr("#ff6e46") if index == chosen else _bgr("#8ed7c7")
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

    if assay == "nor":
        # 物体中心与自定义探索范围画在同一画布，角色随标签显示。
        entries = experiments.get("nor").object_entries(
            unit_geometry.get("objects")) or []
        for index, item in enumerate(entries, start=1):
            if not isinstance(item, Mapping):
                continue
            zone = item.get("zone")
            if isinstance(zone, Mapping) and zone.get("points"):
                _draw_shape(canvas, zone, _bgr("#b45f45"), thickness)
            center = item.get("center")
            if not (isinstance(center, (list, tuple)) and len(center) >= 2):
                continue
            point = (int(round(float(center[0]))),
                     int(round(float(center[1]))))
            cv2.circle(canvas, point, 10, _bgr("#1c5f9e"), 1, cv2.LINE_AA)
            role = ("新奇" if item.get("role") == "novel"
                    else "熟悉" if item.get("role") == "familiar" else "")
            label = f"O{index}" + (f" {role}" if role else "")
            width = int(canvas.shape[1])
            label_x = point[0] + 12
            if label_x + 8 * len(label) > width - 4:
                label_x = max(2, point[0] - 12 - 8 * len(label))
            cv2.putText(canvas, label, (label_x, max(12, point[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, _bgr("#1c5f9e"), 1,
                        cv2.LINE_AA)
        return
    if assay == "mwm":
        module = experiments.get("mwm")
        guides = module.quadrant_guides(unit_geometry, options)
        if not guides:
            return
        center = guides["center"]
        length = max(1.0, float(guides["length"]))
        for dx, dy in guides["directions"]:
            start = (int(round(center[0] - dx * length)),
                     int(round(center[1] - dy * length)))
            end = (int(round(center[0] + dx * length)),
                   int(round(center[1] + dy * length)))
            cv2.line(canvas, start, end, GUIDE_COLOR, 1, cv2.LINE_AA)
        return
    if assay != "ezm":
        return
    # EZM：内圈、四条分界线与四个区名都由保存几何与当前有效尺寸算出；
    # 开放绿、封闭蓝只做标识，统计仍在 Python 端按同一物理变换判定。
    guides = experiments.get("ezm").boundary_guides(unit_geometry, options)
    if not guides:
        return
    center = guides["center"]
    radii = guides["radii"]
    scale = float(guides["inner_scale"])
    if scale > 0:
        cv2.ellipse(
            canvas,
            (int(round(center[0])), int(round(center[1]))),
            (max(1, int(round(float(radii[0]) * scale))),
             max(1, int(round(float(radii[1]) * scale)))),
            float(guides["angle_deg"]), 0, 360, GUIDE_COLOR, 1, cv2.LINE_AA)
    for ray in guides["rays"]:
        cv2.line(canvas, _round_point(ray["from"]), _round_point(ray["to"]),
                 GUIDE_COLOR, max(1, thickness - 1), cv2.LINE_AA)
    for item in guides["labels"]:
        color = _bgr("#16684e") if item.get("open") else _bgr("#1c5f9e")
        point = _round_point(item["point"])
        cv2.putText(canvas, str(item["zone"]),
                    (point[0] + 3, point[1] + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)


def _round_point(point) -> tuple[int, int]:
    return (int(round(float(point[0]))), int(round(float(point[1]))))


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
    unit_window = unit.get("window")
    relative = (isinstance(unit_window, Mapping)
                and unit_window.get("mode") == "relative")
    manual = is_manual_path(assay, unit)
    if manual or relative:
        # 人工路径与相对单位窗都需要完整配置范围，不能只取 t±1/3 秒。
        scope_end = end_value if end_value is not None else max(start, time_s)
        scope_end = max(start, scope_end)
        return {
            "start_s": start,
            "end_s": scope_end,
            "duration_s": scope_end - start,
            "mode": "configured",
            "manual": manual,
            "relative": relative,
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
    *,
    settings: Mapping[str, Any] | None = None,
) -> dict:
    """生成一张真实源帧预览；失败抛 ValidationError / media.MediaError。

    ``settings`` 只作最底层：完整草稿字段（含快照默认）优先于它。
    """

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
    # 检测区域、签名与行为核心必须用同一份最终配置：只叠显式覆盖会在
    # 稀疏 API 配置下漏掉默认 dark_box_visible，让检测区域少掉暗室。
    options = resolve_options(draft["assay"], settings=settings, draft=draft,
                              video=video, unit=unit)
    detection_settings = resolve_detection(
        draft["assay"], settings=settings, draft=draft, video=video, unit=unit)
    metadata = asset.get("metadata") or {}
    duration_s = metadata.get("duration_s")
    scope = _scope(draft.get("assay", ""), video, unit, options,
                   float(info.time_s), kind,
                   float(duration_s) if is_number(duration_s) else None)

    height, width = info.image.shape[:2]
    orientation = int(video.get("orientation", 0) or 0)
    manual_scope = is_manual_path(draft.get("assay", ""), unit)
    # 纯人工 Y 记录不运行自动检测；Barnes 的人工只换洞访问序列，仍要真实捕获
    # 身体/鼻尖覆盖与路径，检测区域与候选照常显示。
    detect_here = not (kind == "behavior" and manual_scope
                       and draft.get("assay") == "ymaze")
    region_source = None
    result = None
    if detect_here:
        try:
            region, region_source = detection.geometry_region(
                geometry, draft["assay"], options, width, height
            )
            background = _background_for(detection_settings, asset, video,
                                         orientation, width, height)
            window_key = repr(video.get("window"))
            video_key = (asset.get("id"), orientation, window_key)
            detector = detection.shared_detector(
                video_key, detection_settings, background=background,
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
                image=info.image, region=region, settings=detection_settings,
                signature=observation.signature_of(
                    geometry, detection_settings,
                    bool(options.get("dark_box_visible"))),
                assay=draft["assay"], frame_index=info.index,
                time_s=float(info.time_s),
                foreground=detector.foreground(info.image),
                retain_units=[item.get("id") for item in video.get("units") or []
                              if isinstance(item, Mapping) and item.get("id")],
            )
        except detection.DetectionError as error:
            raise ValidationError(str(error), "detection") from error

    behavior = None
    module = experiments.get(draft["assay"])
    if kind == "behavior" and hasattr(module, "preview_analysis"):
        core_unit = {
            "id": unit.get("id"),
            "name": unit.get("name"),
            "animal_id": unit.get("animal_id"),
            "window": unit.get("window"),
            "geometry": geometry,
            "options": dict(options),
            "detection": detection_settings,
            # 人工记录按值传入：预览不得写回或改写草稿里的输入。
            "manual": deepcopy(unit.get("manual")),
        }
        try:
            behavior = module.preview_analysis(
                asset=asset, video=video, unit=core_unit,
                start_s=float(scope["start_s"]),
                end_s=float(scope["end_s"]),
                time_s=float(info.time_s),
            )
        except ValidationError as error:
            raise ValidationError(error.message, error.field) from error
    canvas = info.image.copy()
    thickness = _thickness(width, height)
    if result is not None:
        _draw_candidates(canvas, result, thickness)
    for key, shape in geometry.items():
        if key == "zones" and isinstance(shape, Mapping):
            # EPM 的五个区域嵌在 zones 下，逐个叠画而不是当成一个形状。
            for zone in shape.values():
                if isinstance(zone, Mapping):
                    _draw_shape(canvas, zone, _bgr("#ffd26f"), thickness)
            continue
        if key == "holes" and isinstance(shape, list):
            # Barnes 的洞位落盘为数组；画布上按序编号，与工作台一致。
            _draw_shape(canvas, {"type": "holes", "holes": shape},
                        _bgr("#ffd26f"), thickness)
            continue
        if not isinstance(shape, Mapping):
            continue
        _draw_shape(canvas, shape, _bgr("#ffd26f"), thickness)
    steps = module.SPEC.get("geometry", [])
    # 步骤集合随当前配置变化：暗室不可见时不把 dark 当成待绘制步骤。
    step_applies = getattr(module, "step_applies", None)

    def in_scope(step: Mapping[str, Any]) -> bool:
        return bool(step_applies(step, options)) if step_applies else True

    step_checker = getattr(module, "step_drawn", None)

    def is_drawn(step: Mapping[str, Any]) -> bool:
        return (bool(step_checker(geometry, step["key"])) if step_checker
                else step.get("key") in geometry)

    # 已绘制判定与画布、最终校验同源：NOR 的物体步骤落在 geometry.objects 里。
    drawn = [step["key"] for step in steps
             if in_scope(step) and is_drawn(step)]
    # 辅助线读合并后的有效参数：只给显式覆盖会在默认尺寸下漏画内外圈。
    _draw_guides(canvas, draft.get("assay", ""), options, geometry, thickness)

    for other in video.get("units") or []:
        if other.get("id") == unit.get("id"):
            continue
        for shape in (other.get("geometry") or {}).values():
            if isinstance(shape, Mapping):
                _draw_shape(canvas, shape, _bgr("#8ed7c7"), 1)

    import cv2

    if behavior and behavior.get("state_at_time"):
        label = module.SPEC.get("short") or draft["assay"].upper()
        cv2.putText(canvas, label + " " + str(behavior["state_at_time"]),
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (45, 45, 45), 2, cv2.LINE_AA)

    filename = f"{new_id()}.png"
    target = Path(paths.previews) / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), canvas):
        raise ValidationError("无法保存预览图", "preview")

    missing = [
        step["label"] for step in steps
        if step.get("required") and in_scope(step) and not is_drawn(step)
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

    if behavior is not None:
        # 人工序列只决定洞访问与入箱评分；身体/鼻尖覆盖仍来自本次真实捕获。
        start = behavior.get("window_start_s")
        end = behavior.get("window_end_s")
        span = (f"{float(start):.3f}–{float(end):.3f} s"
                if start is not None and end is not None else "未确定")
        source = behavior.get("visit_source")
        entry = behavior.get("manual_entry_time_s")
        if source == "manual":
            warnings.append("行为预览保留人工洞访问与入箱评分，统计范围为动物窗 "
                            + span + "；覆盖与路径仍来自本次真实捕获。")
        elif entry is not None:
            # 只有入箱是人工端点：洞访问仍自动，不能写成人工访问序列。
            warnings.append("洞访问仍为自动，入箱端点由人工确认；统计保留完整"
                            "动物窗 " + span + "，覆盖与路径来自本次真实捕获。")

    manual_behavior = bool(behavior) and behavior.get("manual") is True
    if manual_behavior and behavior.get("events") == 0:
        warnings.append("分析时间窗内没有人工入臂记录。")
    elif behavior is not None and not manual_behavior and behavior["rows"] == 0:
        warnings.append("预览范围内没有可用的行为数据。")
    elif (behavior is not None and not manual_behavior
            and behavior["window_start_s"] is None):
        warnings.append("未检测到动物，无法确定相对时间窗。")

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
            "behavior": behavior,
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
            "manual_preserved": is_manual_path(draft.get("assay", ""), unit),
        },
        "warnings": warnings,
    }
