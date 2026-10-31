"""配置冻结、校验与单视频执行边界。

合并规则是数据契约的一部分：对象递归合并，数组、标量、null 整体替换并深拷贝。
"""

from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from . import experiments, media
from .config import ALGORITHM_VERSION, SCHEMA_VERSION
from .detection import common_detection_defaults
from .schema import (
    ValidationError,
    check_json_value,
    is_number,
    is_valid_id,
)

COMMON_OUTPUT_DEFAULTS: dict = {"annotated_video": True}
COMMON_STYLE_DEFAULTS: dict = {
    "color": "#B86245",
    "colormap": "viridis",
    "line_width": 2,
    "dpi": 150,
}

HEX_COLOR = re.compile(r"#[0-9A-Fa-f]{6}")
ORIENTATIONS = (0, 90, 180, 270)
WINDOW_MODES = ("global", "exact", "relative")


# ---------------------------------------------------------------------------
# 合并与解析


def merge_two(base: Any, override: Any) -> Any:
    """对象递归合并；其他类型由 override 整体替换。"""

    if isinstance(base, Mapping) and isinstance(override, Mapping):
        merged = {key: deepcopy(value) for key, value in base.items()}
        for key, value in override.items():
            if key in base:
                merged[key] = merge_two(base[key], value)
            else:
                merged[key] = deepcopy(value)
        return merged
    return deepcopy(override)


def merge_layers(*layers: Any) -> dict:
    result: dict = {}
    for layer in layers:
        if layer is None:
            continue
        if not isinstance(layer, Mapping):
            raise ValidationError("配置层必须是对象", "")
        result = merge_two(result, layer)
    return result


def _module(assay: str):
    try:
        return experiments.get(assay)
    except KeyError as error:
        raise ValidationError("未知实验", "assay") from error


def resolve_option_layers(
    factory: Mapping[str, Any],
    *,
    saved: Mapping[str, Any] | None = None,
    draft: Mapping[str, Any] | None = None,
    video: Mapping[str, Any] | None = None,
    unit: Mapping[str, Any] | None = None,
) -> dict:
    """实验工厂 → 保存默认 → draft → video → unit。"""

    return merge_layers(
        factory,
        saved,
        (draft or {}).get("options"),
        (video or {}).get("options"),
        (unit or {}).get("options"),
    )


def resolve_options(
    assay: str,
    *,
    saved: Mapping[str, Any] | None = None,
    draft: Mapping[str, Any] | None = None,
    video: Mapping[str, Any] | None = None,
    unit: Mapping[str, Any] | None = None,
) -> dict:
    """实验工厂 → 保存默认 → draft → video → unit。"""

    module = _module(assay)
    return resolve_option_layers(
        module.DEFAULTS, saved=saved, draft=draft, video=video, unit=unit
    )


def resolve_detection(
    assay: str,
    *,
    saved: Mapping[str, Any] | None = None,
    draft: Mapping[str, Any] | None = None,
    video: Mapping[str, Any] | None = None,
    unit: Mapping[str, Any] | None = None,
) -> dict:
    """公共工厂 → 实验检测默认 → 保存检测默认 → draft → video → unit。"""

    module = _module(assay)
    return merge_layers(
        common_detection_defaults(),
        getattr(module, "DETECTION_DEFAULTS", {}),
        saved,
        (draft or {}).get("detection"),
        (video or {}).get("detection"),
        (unit or {}).get("detection"),
    )


def resolve_output(
    *, saved: Mapping[str, Any] | None = None, draft: Mapping[str, Any] | None = None
) -> dict:
    return merge_layers(COMMON_OUTPUT_DEFAULTS, saved, (draft or {}).get("output"))


def resolve_style(
    *, saved: Mapping[str, Any] | None = None, draft: Mapping[str, Any] | None = None
) -> dict:
    return merge_layers(COMMON_STYLE_DEFAULTS, saved, (draft or {}).get("style"))


def snapshot_draft(
    assay: str,
    *,
    defaults: Mapping[str, Any] | None = None,
    detection: Mapping[str, Any] | None = None,
    output: Mapping[str, Any] | None = None,
    style: Mapping[str, Any] | None = None,
) -> dict:
    """创建草稿时的独立快照：工厂值被保存默认按顶层字段覆盖。"""

    module = _module(assay)
    options = deepcopy(module.DEFAULTS)
    if defaults:
        if not isinstance(defaults, Mapping):
            raise ValidationError("默认参数必须是对象", "defaults")
        for key, value in defaults.items():
            options[key] = deepcopy(value)
    return {
        "options": options,
        "detection": merge_layers(
            common_detection_defaults(),
            getattr(module, "DETECTION_DEFAULTS", {}),
            detection,
        ),
        "output": merge_layers(COMMON_OUTPUT_DEFAULTS, output),
        "style": merge_layers(COMMON_STYLE_DEFAULTS, style),
    }


# ---------------------------------------------------------------------------
# 校验


def validate_style(style: Mapping[str, Any]) -> list[dict]:
    issues: list[dict] = []
    if not isinstance(style, Mapping):
        return [{"field": "style", "message": "必须是对象"}]

    color = style.get("color")
    if color is not None and (not isinstance(color, str) or not HEX_COLOR.fullmatch(color)):
        issues.append({"field": "style.color", "message": "必须是六位十六进制颜色"})

    line_width = style.get("line_width")
    if line_width is not None:
        if not is_number(line_width) or not 0.5 <= float(line_width) <= 10:
            issues.append({"field": "style.line_width", "message": "线宽必须在 0.5–10 之间"})

    dpi = style.get("dpi")
    if dpi is not None:
        if not is_number(dpi) or not 72 <= float(dpi) <= 600:
            issues.append({"field": "style.dpi", "message": "DPI 必须在 72–600 之间"})

    colormap = style.get("colormap")
    if colormap is not None:
        try:
            import matplotlib

            matplotlib.colormaps[str(colormap)]
        except (KeyError, ValueError):
            issues.append({"field": "style.colormap", "message": "绘图库中没有该色图"})

    try:
        check_json_value(style, "style")
    except ValidationError as error:
        issues.append(error.as_issue())
    return issues


def validate_output(output: Mapping[str, Any]) -> list[dict]:
    if not isinstance(output, Mapping):
        return [{"field": "output", "message": "必须是对象"}]
    issues: list[dict] = []
    if "annotated_video" in output and not isinstance(output["annotated_video"], bool):
        issues.append({"field": "output.annotated_video", "message": "必须是布尔值"})
    return issues


def _number(value: Any) -> bool:
    return is_number(value)


def _window_issues(window: Any, field: str, *, video_window: bool) -> list[dict]:
    if window is None:
        return []
    if not isinstance(window, Mapping):
        return [{"field": field, "message": "必须是对象或 null"}]
    issues: list[dict] = []
    if video_window:
        start = window.get("start_s")
        if not _number(start) or float(start) < 0:
            issues.append({"field": f"{field}.start_s", "message": "起点必须是非负数"})
        end = window.get("end_s")
        if end is not None:
            if not _number(end):
                issues.append({"field": f"{field}.end_s", "message": "终点必须是数值或 null"})
            elif _number(start) and float(end) <= float(start):
                issues.append({"field": f"{field}.end_s", "message": "终点必须大于起点"})
        return issues

    mode = window.get("mode")
    if mode not in WINDOW_MODES:
        return [{"field": f"{field}.mode", "message": "未知窗口模式"}]
    if mode == "global":
        return issues
    if mode == "exact":
        start = window.get("start_s")
        end = window.get("end_s")
        if not _number(start) or float(start) < 0:
            issues.append({"field": f"{field}.start_s", "message": "起点必须是非负数"})
        if not _number(end):
            issues.append({"field": f"{field}.end_s", "message": "终点必须是数值"})
        elif _number(start) and float(end) <= float(start):
            issues.append({"field": f"{field}.end_s", "message": "终点必须大于起点"})
        return issues
    if mode == "relative":
        offset = window.get("offset_s")
        duration = window.get("duration_s")
        if not _number(offset):
            issues.append({"field": f"{field}.offset_s", "message": "偏移必须是数值"})
        if not _number(duration) or float(duration) <= 0:
            issues.append({"field": f"{field}.duration_s", "message": "持续时间必须大于 0"})
    return issues


def validate_draft(draft: Mapping[str, Any]) -> list[dict]:
    """草稿结构校验；实验专属参数由各实验模块校验。"""

    if not isinstance(draft, Mapping):
        return [{"field": "", "message": "草稿必须是对象"}]

    issues: list[dict] = []
    assay = draft.get("assay")
    if not experiments.exists(assay):
        issues.append({"field": "assay", "message": "未知实验"})

    videos = draft.get("videos")
    if videos is None:
        videos = []
    if not isinstance(videos, list):
        issues.append({"field": "videos", "message": "必须是数组"})
        return issues

    video_ranges: list[tuple[int, int, str | None]] = []

    for index, video in enumerate(videos):
        prefix = f"videos[{index}]"
        video_start = len(issues)
        if not isinstance(video, Mapping):
            issues.append({"field": prefix, "message": "视频条目必须是对象"})
            video_ranges.append((video_start, len(issues), None))
            continue
        asset_id = video.get("asset_id")
        asset_key = asset_id if is_valid_id(asset_id) else None
        if not is_valid_id(asset_id):
            issues.append({"field": f"{prefix}.asset_id", "message": "缺少有效素材 ID"})
        issues.extend(_window_issues(video.get("window"), f"{prefix}.window",
                                     video_window=True))
        orientation = video.get("orientation", 0)
        if orientation not in ORIENTATIONS:
            issues.append({"field": f"{prefix}.orientation", "message": "旋转角度必须是 0/90/180/270"})
        fps_override = video.get("fps_override")
        if fps_override is not None and (not _number(fps_override) or float(fps_override) <= 0):
            issues.append({"field": f"{prefix}.fps_override", "message": "帧率必须是正数"})
        sample_fps = video.get("sample_fps")
        if sample_fps is not None and (not _number(sample_fps) or float(sample_fps) <= 0):
            issues.append({"field": f"{prefix}.sample_fps", "message": "采样率必须是正数"})
        for field in ("options", "detection"):
            if not isinstance(video.get(field, {}), Mapping):
                issues.append({"field": f"{prefix}.{field}", "message": "必须是对象"})

        units = video.get("units", [])
        if not isinstance(units, list):
            issues.append({"field": f"{prefix}.units", "message": "必须是数组"})
            video_ranges.append((video_start, len(issues), asset_key))
            continue
        seen_ids: set[str] = set()
        for unit_index, unit in enumerate(units):
            unit_prefix = f"{prefix}.units[{unit_index}]"
            if not isinstance(unit, Mapping):
                issues.append({"field": unit_prefix, "message": "单位条目必须是对象"})
                continue
            unit_start = len(issues)
            unit_id = unit.get("id")
            unit_key = unit_id if is_valid_id(unit_id) else None
            if not is_valid_id(unit_id):
                issues.append({"field": f"{unit_prefix}.id", "message": "缺少有效单位 ID"})
            elif unit_id in seen_ids:
                issues.append({"field": f"{unit_prefix}.id", "message": "单位 ID 重复"})
            else:
                seen_ids.add(unit_id)
            name = unit.get("name")
            if not isinstance(name, str) or not name.strip():
                issues.append({"field": f"{unit_prefix}.name", "message": "名称不能为空"})
            animal_id = unit.get("animal_id")
            if animal_id is not None and not is_valid_id(animal_id):
                issues.append({"field": f"{unit_prefix}.animal_id", "message": "动物 ID 无效"})
            issues.extend(_window_issues(unit.get("window"), f"{unit_prefix}.window",
                                         video_window=False))
            for field in ("geometry", "options", "detection"):
                if not isinstance(unit.get(field, {}), Mapping):
                    issues.append({"field": f"{unit_prefix}.{field}", "message": "必须是对象"})
            manual = unit.get("manual")
            if manual is not None and not isinstance(manual, Mapping):
                issues.append({"field": f"{unit_prefix}.manual", "message": "必须是对象或 null"})
            if assay == "ymaze" and isinstance(manual, Mapping) and manual:
                window = unit.get("window")
                if isinstance(window, Mapping) and window.get("mode") == "relative":
                    issues.append(
                        {"field": f"{unit_prefix}.window.mode",
                         "message": "人工 Y 只支持 global 或 exact 窗口"}
                    )
            for issue in issues[unit_start:]:
                if unit_key:
                    issue.setdefault("unit_id", unit_key)
                if asset_key:
                    issue.setdefault("asset_id", asset_key)
        video_ranges.append((video_start, len(issues), asset_key))

    for start, end, asset_key in video_ranges:
        if asset_key:
            for issue in issues[start:end]:
                issue.setdefault("asset_id", asset_key)

    return issues


def validate_settings_value(section: str, value: Any) -> list[dict]:
    if section in ("defaults", "detection"):
        if not isinstance(value, Mapping):
            return [{"field": section, "message": "必须是对象"}]
        issues: list[dict] = []
        for key, item in value.items():
            if not experiments.exists(key):
                issues.append({"field": f"{section}.{key}", "message": "未知实验"})
                continue
            if not isinstance(item, Mapping):
                issues.append({"field": f"{section}.{key}", "message": "必须是对象"})
        return issues
    if section == "output":
        return validate_output(value)
    if section == "style":
        return validate_style(value)
    return [{"field": "section", "message": "未知设置分区"}]


# ---------------------------------------------------------------------------
# 冻结与身份


def _stat_identity(path: str) -> dict:
    file_path = Path(path)
    try:
        stat = os.stat(file_path)
    except OSError as error:
        raise ValidationError("视频文件不可读取", "asset.path") from error
    return {
        "path": str(file_path.resolve()),
        "size": int(stat.st_size),
        "mtime": float(stat.st_mtime),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def build_resolved_job(
    *,
    draft: Mapping[str, Any],
    video: Mapping[str, Any],
    asset: Mapping[str, Any],
    job_id: str,
    run_id: str,
    project_id: str | None = None,
) -> dict:
    """为一个视频冻结一份完整解析结果；每个单位都带完整 options/detection。"""

    assay = draft.get("assay")
    module = _module(assay)

    path = asset.get("path")
    if not isinstance(path, str) or not path:
        raise ValidationError("素材缺少路径", "asset.path")
    identity = _stat_identity(path)

    stored_metadata = asset.get("metadata")
    if not isinstance(stored_metadata, Mapping):
        stored_metadata = {}
    try:
        probed = media.probe_video(path)
    except media.MediaError as error:
        raise ValidationError(str(error), "asset.path") from error

    units = []
    for unit in video.get("units", []) or []:
        unit_id = unit.get("id")
        if not is_valid_id(unit_id):
            raise ValidationError("缺少有效单位 ID", "units.id")
        units.append(
            {
                "id": unit_id,
                "name": unit.get("name"),
                "animal_id": unit.get("animal_id"),
                "window": deepcopy(unit.get("window")),
                "geometry": deepcopy(unit.get("geometry") or {}),
                "manual": deepcopy(unit.get("manual")),
                "options": resolve_options(assay, draft=draft, video=video, unit=unit),
                "detection": resolve_detection(assay, draft=draft, video=video,
                                               unit=unit),
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "job_id": job_id,
        "project_id": project_id if project_id is not None else draft.get("project_id"),
        "assay": assay,
        "asset": {
            "id": asset.get("id"),
            "name": asset.get("name"),
            "path": identity["path"],
            "metadata": {**dict(stored_metadata), **probed},
            "identity": identity,
        },
        "window": deepcopy(video.get("window")),
        "orientation": video.get("orientation", 0),
        "fps_override": video.get("fps_override"),
        "sample_fps": video.get("sample_fps"),
        "options": resolve_options(assay, draft=draft, video=video),
        "detection": resolve_detection(assay, draft=draft, video=video),
        "units": units,
        "output": resolve_output(draft=draft),
        "style": resolve_style(draft=draft),
        "algorithm_version": ALGORITHM_VERSION,
    }


def verify_identity(asset: Mapping[str, Any]) -> list[dict]:
    """运行前核对源文件身份；不匹配即拒绝该冻结任务。"""

    issues: list[dict] = []
    identity = asset.get("identity") or {}
    path = identity.get("path")
    if not path:
        return [{"field": "asset.identity", "message": "缺少冻结身份"}]
    try:
        stat = os.stat(path)
    except OSError:
        return [{"field": "asset.path", "message": "源视频已不可读取"}]
    if int(stat.st_size) != identity.get("size"):
        issues.append({"field": "asset.identity", "message": "源视频大小已变化"})
    if int(stat.st_mtime_ns) != identity.get("mtime_ns"):
        issues.append({"field": "asset.identity", "message": "源视频修改时间已变化"})
    if str(Path(path).resolve()) != str(Path(asset.get("path", path)).resolve()):
        issues.append({"field": "asset.path", "message": "源视频路径已变化"})
    return issues


def require_identity(asset: Mapping[str, Any]) -> None:
    """身份不匹配时抛错，供任务执行前拒绝该冻结任务。"""

    issues = verify_identity(asset)
    if issues:
        first = issues[0]
        raise ValidationError(first["message"], first["field"])


def execute(resolved: Mapping[str, Any]) -> Any:
    raise NotImplementedError("单视频分析引擎将在后续块接入")
