"""配置冻结、校验与单视频执行边界。

合并规则是数据契约的一部分：对象递归合并，数组、标量、null 整体替换并深拷贝。
"""

from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from . import experiments, media
from .config import ALGORITHM_VERSION, SCHEMA_VERSION
from .detection import common_detection_defaults, validate_detection
from .schema import (
    AnalysisContext,
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


SETTING_SECTIONS = ("defaults", "detection", "output", "style")


def setting_layer(settings: Mapping[str, Any] | None, assay: str,
                  section: str) -> dict | None:
    """取 section 层，两种形状分别处理。

    - ``defaults``：与 Store 一致，按实验名再取一层 ``[assay]``。
    - ``detection``：服务已经取出当前实验对象，不能再按实验名取，否则合法
      的检测覆盖会被静默丢弃。
    - ``output``/``style``：本来就是单对象。
    """

    if not isinstance(settings, Mapping):
        return None
    value = settings.get(section)
    if not isinstance(value, Mapping):
        return None
    if section == "defaults":
        sub = value.get(assay)
        return deepcopy(dict(sub)) if isinstance(sub, Mapping) else None
    return deepcopy(dict(value))


def resolve_option_layers(
    factory: Mapping[str, Any],
    *,
    settings: Mapping[str, Any] | None = None,
    saved: Mapping[str, Any] | None = None,
    draft: Mapping[str, Any] | None = None,
    video: Mapping[str, Any] | None = None,
    unit: Mapping[str, Any] | None = None,
    assay: str = "",
) -> dict:
    """实验工厂 → 设置默认 → 保存默认 → draft → video → unit。"""

    return merge_layers(
        factory,
        setting_layer(settings, assay, "defaults"),
        saved,
        (draft or {}).get("options"),
        (video or {}).get("options"),
        (unit or {}).get("options"),
    )


def resolve_options(
    assay: str,
    *,
    settings: Mapping[str, Any] | None = None,
    saved: Mapping[str, Any] | None = None,
    draft: Mapping[str, Any] | None = None,
    video: Mapping[str, Any] | None = None,
    unit: Mapping[str, Any] | None = None,
) -> dict:
    """实验工厂 → 设置默认 → 保存默认 → draft → video → unit。"""

    module = _module(assay)
    return resolve_option_layers(
        module.DEFAULTS, settings=settings, saved=saved, draft=draft,
        video=video, unit=unit, assay=assay,
    )


def resolve_detection(
    assay: str,
    *,
    settings: Mapping[str, Any] | None = None,
    saved: Mapping[str, Any] | None = None,
    draft: Mapping[str, Any] | None = None,
    video: Mapping[str, Any] | None = None,
    unit: Mapping[str, Any] | None = None,
) -> dict:
    """公共工厂 → 实验检测默认 → 设置检测 → draft → video → unit。"""

    module = _module(assay)
    return merge_layers(
        common_detection_defaults(),
        getattr(module, "DETECTION_DEFAULTS", {}),
        setting_layer(settings, assay, "detection"),
        saved,
        (draft or {}).get("detection"),
        (video or {}).get("detection"),
        (unit or {}).get("detection"),
    )


def resolve_output(
    *,
    settings: Mapping[str, Any] | None = None,
    saved: Mapping[str, Any] | None = None,
    draft: Mapping[str, Any] | None = None,
) -> dict:
    return merge_layers(COMMON_OUTPUT_DEFAULTS, setting_layer(settings, "", "output"),
                        saved, (draft or {}).get("output"))


def resolve_style(
    *,
    settings: Mapping[str, Any] | None = None,
    saved: Mapping[str, Any] | None = None,
    draft: Mapping[str, Any] | None = None,
) -> dict:
    return merge_layers(COMMON_STYLE_DEFAULTS, setting_layer(settings, "", "style"),
                        saved, (draft or {}).get("style"))


def resolve_job(
    assay: str,
    *,
    settings: Mapping[str, Any] | None = None,
    draft: Mapping[str, Any] | None = None,
    video: Mapping[str, Any] | None = None,
    unit: Mapping[str, Any] | None = None,
) -> dict:
    """通用解析入口：一次给出 options/detection/output/style 四层结果。

    后层优先；两侧都是对象时逐键递归合并，列表/标量/null 整体替换。
    """

    return {
        "options": resolve_options(assay, settings=settings, draft=draft,
                                   video=video, unit=unit),
        "detection": resolve_detection(assay, settings=settings, draft=draft,
                                       video=video, unit=unit),
        "output": resolve_output(settings=settings, draft=draft),
        "style": resolve_style(settings=settings, draft=draft),
    }


def snapshot_draft(
    assay: str,
    *,
    settings: Mapping[str, Any] | None = None,
    defaults: Mapping[str, Any] | None = None,
    detection: Mapping[str, Any] | None = None,
    output: Mapping[str, Any] | None = None,
    style: Mapping[str, Any] | None = None,
) -> dict:
    """创建草稿时的独立快照：工厂值叠加设置默认，再按顶层字段覆盖。

    快照是深拷贝，草稿不引用 settings 对象；之后的设置修改不影响它。
    """

    module = _module(assay)
    options = merge_layers(module.DEFAULTS,
                           setting_layer(settings, assay, "defaults"))
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
            setting_layer(settings, assay, "detection"),
            detection,
        ),
        "output": merge_layers(COMMON_OUTPUT_DEFAULTS,
                               setting_layer(settings, "", "output"), output),
        "style": merge_layers(COMMON_STYLE_DEFAULTS,
                              setting_layer(settings, "", "style"), style),
    }


# ---------------------------------------------------------------------------
# 校验


def validate_style(style: Mapping[str, Any]) -> list[dict]:
    """已提供的样式字段必须是合法值；显式 null 不是“用默认值”。"""

    issues: list[dict] = []
    if not isinstance(style, Mapping):
        return [{"field": "style", "message": "必须是对象"}]

    if "color" in style:
        color = style.get("color")
        if not isinstance(color, str) or not HEX_COLOR.fullmatch(color):
            issues.append({"field": "style.color",
                           "message": "必须是六位十六进制颜色"})

    if "line_width" in style:
        line_width = style.get("line_width")
        if not is_number(line_width) or not 0.5 <= float(line_width) <= 10:
            issues.append({"field": "style.line_width",
                           "message": "线宽必须在 0.5–10 之间"})

    if "dpi" in style:
        dpi = style.get("dpi")
        if not is_number(dpi) or not 72 <= float(dpi) <= 600:
            issues.append({"field": "style.dpi",
                           "message": "DPI 必须在 72–600 之间"})

    if "colormap" in style:
        colormap = style.get("colormap")
        try:
            import matplotlib

            matplotlib.colormaps[str(colormap)]
        except (KeyError, TypeError, ValueError):
            issues.append({"field": "style.colormap",
                           "message": "绘图库中没有该色图"})

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


def is_manual_unit(unit: Mapping[str, Any]) -> bool:
    """严格人工判定：只有 unit.manual.mode == "manual" 才算人工。"""

    manual = unit.get("manual") if isinstance(unit, Mapping) else None
    return isinstance(manual, Mapping) and manual.get("mode") == "manual"


def is_manual_path(assay: str, unit: Mapping[str, Any]) -> bool:
    """各层共用的人工路径判定。

    Y 迷宫只看 unit.manual.mode：options.mode 只是设置项，manual 为 null 或
    mode=auto 时都按自动单位处理。Barnes 的人工记录仍按记录字段判定。
    """

    if assay == "ymaze":
        return is_manual_unit(unit)
    if assay == "barnes":
        # 只有真正给出人工记录（含空数组）才算人工路径；缺省/null 是自动访问。
        manual = unit.get("manual")
        if not isinstance(manual, Mapping):
            return False
        return manual.get("hole_visits") is not None \
            or manual.get("escape_time_s") is not None
    return False


def _detection_gaps(values: Mapping[str, Any], prefix: str) -> list[dict]:
    """自动单位运行时真正缺少的检测配置；字段路径与运行时一致。"""

    issues: list[dict] = []
    if values.get("bg_method") == "static" and not values.get("static_background"):
        issues.append({"field": f"{prefix}.static_background",
                       "message": "静态背景未设置"})
    return issues


def _exact_misses_video(window: Mapping[str, Any], video_start: float | None,
                        video_end: float | None) -> bool:
    """显式窗口与已声明的视频范围不重叠；视频终点未知时留给运行期。"""

    if video_start is None or video_end is None:
        return False
    start, end = window.get("start_s"), window.get("end_s")
    if not _number(start) or not _number(end):
        return False
    return min(float(end), video_end) <= max(float(start), video_start)


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


def validate_draft(draft: Mapping[str, Any], *,
                   complete: bool = False,
                   animals: Iterable[str] | None = None,
                   settings: Mapping[str, Any] | None = None) -> list[dict]:
    """草稿结构校验；实验专属参数由各实验模块校验。

    complete=False 允许未画完的草稿保存；complete=True 用于最终校验与开始
    任务，实验要求的必需几何（如 OFT 四角）缺失即报错。animals 是当前项目
    可绑定的动物 ID；提供时校验跨项目绑定，None 表示没有项目上下文。
    """

    if not isinstance(draft, Mapping):
        return [{"field": "", "message": "草稿必须是对象"}]

    issues: list[dict] = []
    assay = draft.get("assay")
    if not experiments.exists(assay):
        issues.append({"field": "assay", "message": "未知实验"})
    # Y 迷宫的人工单位不消费检测配置：纯人工草稿各层都豁免，混合草稿按每个
    # 自动单位合并后的配置校验，不能整层跳过。
    ymaze = assay == "ymaze"
    if not ymaze:
        issues.extend(validate_detection(draft.get("detection"), "detection"))
    allowed_animals = None if animals is None else set(animals)

    videos = draft.get("videos")
    if videos is None:
        videos = []
    if not isinstance(videos, list):
        issues.append({"field": "videos", "message": "必须是数组"})
        return issues
    if complete and not videos:
        # 保存草稿可以还没有视频；最终校验必须明确报出来。
        issues.append({"field": "videos", "message": "草稿还没有视频"})

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
        video_window = video.get("window")
        video_bounds: tuple[float | None, float | None] = (0.0, None)
        if isinstance(video_window, Mapping):
            declared_start = video_window.get("start_s", 0.0)
            declared_end = video_window.get("end_s")
            video_bounds = (
                float(declared_start) if _number(declared_start) else None,
                float(declared_end) if _number(declared_end) else None,
            )
        orientation = video.get("orientation", 0)
        # 与 VideoSource 的类型约束一致：bool 与浮点都不是合法方向。
        if isinstance(orientation, bool) or not isinstance(orientation, int) \
                or orientation not in ORIENTATIONS:
            issues.append({"field": f"{prefix}.orientation",
                           "message": "旋转角度必须是 0/90/180/270 的整数"})
        fps_override = video.get("fps_override")
        if fps_override is not None and (not _number(fps_override) or float(fps_override) <= 0):
            issues.append({"field": f"{prefix}.fps_override", "message": "帧率必须是正数"})
        sample_fps = video.get("sample_fps")
        if sample_fps is not None and (not _number(sample_fps) or float(sample_fps) <= 0):
            issues.append({"field": f"{prefix}.sample_fps", "message": "采样率必须是正数"})
        for field in ("options", "detection"):
            if not isinstance(video.get(field, {}), Mapping):
                issues.append({"field": f"{prefix}.{field}", "message": "必须是对象"})
        if not ymaze and isinstance(video.get("detection"), Mapping):
            issues.extend(validate_detection(video["detection"],
                                             f"{prefix}.detection"))

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
                issues.append({"field": f"{unit_prefix}.animal_id",
                               "message": "动物 ID 无效", "animal_id": animal_id})
            elif animal_id is not None and allowed_animals is not None \
                    and animal_id not in allowed_animals:
                # 项目绑定：动物必须属于当前项目，跨项目绑定不能开始任务。
                issues.append({"field": f"{unit_prefix}.animal_id",
                               "message": "动物不属于当前项目",
                               "animal_id": animal_id})
            elif animal_id is None and allowed_animals is not None \
                    and complete:
                # 项目模式的最终校验：未绑定（含已删除）动物不能开始任务；
                # 保存阶段不受影响，可以先存草稿再回项目补动物。
                issues.append({"field": f"{unit_prefix}.animal_id",
                               "message": "项目分析必须为每只小鼠绑定动物",
                               "animal_id": None})
            issues.extend(_window_issues(unit.get("window"), f"{unit_prefix}.window",
                                         video_window=False))
            unit_window = unit.get("window")
            if isinstance(unit_window, Mapping):
                if unit_window.get("mode") == "exact" and _exact_misses_video(
                        unit_window, video_bounds[0], video_bounds[1]):
                    issues.append(
                        {"field": f"{unit_prefix}.window.end_s",
                         "message": "窗口与视频范围没有重叠"}
                    )
                # 只有人工 Y 受窗口模式限制；Barnes 的人工路径仍可用 relative。
                if (ymaze and is_manual_path(assay, unit)
                        and unit_window.get("mode") == "relative"):
                    issues.append(
                        {"field": f"{unit_prefix}.window.mode",
                         "message": "人工 Y 只支持 global 或 exact 窗口"}
                    )
            for field in ("geometry", "options", "detection"):
                if not isinstance(unit.get(field, {}), Mapping):
                    issues.append({"field": f"{unit_prefix}.{field}", "message": "必须是对象"})
            # 专用人工 Y 只消费合并后的 bin_seconds 与记录：几何、自动参数与
            # 检测参数都不参与运行，交给模块的人工校验，不在这里要求。
            manual_y = ymaze and is_manual_unit(unit)
            if manual_y:
                manual_checker = getattr(experiments.get(assay),
                                         "validate_manual", None)
                if manual_checker is not None:
                    merged_options = resolve_options(
                        assay, settings=settings, draft=draft,
                        video=video, unit=unit)
                    for issue in manual_checker(unit, options=merged_options,
                                                complete=complete):
                        entry = dict(issue)
                        field = issue.get("field")
                        entry["field"] = (f"{unit_prefix}.{field}" if field
                                          else unit_prefix)
                        issues.append(entry)
            else:
                checker = (getattr(experiments.get(assay), "validate_unit", None)
                           if experiments.exists(assay) else None)
                if checker is not None:
                    layers = (draft.get("options"), video.get("options"),
                              unit.get("options"))
                    provided = {key for layer in layers if isinstance(layer, Mapping)
                                for key in layer}
                    merged = resolve_options(assay, settings=settings,
                                             draft=draft, video=video,
                                             unit=unit)
                    for issue in checker(merged, unit.get("geometry") or {}, provided,
                                         complete=complete):
                        entry = dict(issue)
                        field = issue.get("field")
                        entry["field"] = (f"{unit_prefix}.{field}" if field
                                          else unit_prefix)
                        issues.append(entry)
                # 人工输入的专用校验（Barnes 的洞序列与入箱时刻）：与窗口
                # 无关的结构、洞号与顺序在这里报，运行期再检查真实动物窗。
                manual_checker = (getattr(experiments.get(assay),
                                          "validate_unit_manual", None)
                                  if experiments.exists(assay) else None)
                if manual_checker is not None:
                    merged = resolve_options(assay, settings=settings,
                                             draft=draft, video=video,
                                             unit=unit)
                    for issue in manual_checker(unit, options=merged,
                                                complete=complete):
                        entry = dict(issue)
                        field = issue.get("field")
                        entry["field"] = (f"{unit_prefix}.{field}" if field
                                          else unit_prefix)
                        issues.append(entry)
                if ymaze:
                    # 自动单位按合并后的检测配置校验：draft/video 层的无效值
                    # 若被单位层覆盖，不应再单独报错。
                    merged_detection = resolve_detection(
                        assay, settings=settings, draft=draft,
                        video=video, unit=unit)
                    prefix = f"{unit_prefix}.detection"
                    issues.extend(validate_detection(merged_detection, prefix))
                    issues.extend(_detection_gaps(merged_detection, prefix))
                elif isinstance(unit.get("detection"), Mapping):
                    issues.extend(validate_detection(unit["detection"],
                                                     f"{unit_prefix}.detection"))
            manual = unit.get("manual")
            if manual is not None and not isinstance(manual, Mapping):
                issues.append({"field": f"{unit_prefix}.manual", "message": "必须是对象或 null"})
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

    issues.extend(validate_style(draft.get("style") or {}))
    issues.extend(validate_output(draft.get("output") or {}))

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
    settings: Mapping[str, Any] | None = None,
) -> dict:
    """为一个视频冻结一份完整解析结果；每个单位都带完整 options/detection。

    ``settings`` 只作为最底层补充；Jobs.submit 传 ``{}``，因为完整默认值在
    草稿创建时已经快照，提交不得重新读取新的设置。
    """

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
                "options": resolve_options(assay, settings=settings,
                                           draft=draft, video=video,
                                           unit=unit),
                "detection": resolve_detection(assay, settings=settings,
                                                draft=draft, video=video,
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
        "options": resolve_options(assay, settings=settings, draft=draft,
                                   video=video),
        "detection": resolve_detection(assay, settings=settings, draft=draft,
                                        video=video),
        "units": units,
        "output": resolve_output(settings=settings, draft=draft),
        "style": resolve_style(settings=settings, draft=draft),
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
    # 冻结身份里保存了 mtime：只改它也必须是身份变化。
    stored_mtime = identity.get("mtime")
    if is_number(stored_mtime) and abs(float(stat.st_mtime)
                                       - float(stored_mtime)) > 1e-9:
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


def _job_draft(resolved: Mapping[str, Any]) -> dict:
    """把冻结任务还原成校验器认得的单视频草稿结构。

    各层都用同一份完整解析值，重复合并是幂等的；这样保存、提交与执行阶段
    共用一套字段规则，不需要第二条校验分支。
    """

    asset = resolved.get("asset") or {}
    return {
        "assay": resolved.get("assay"),
        "project_id": resolved.get("project_id"),
        "options": {},
        "detection": resolved.get("detection") or {},
        "output": deepcopy(resolved.get("output") or {}),
        "style": deepcopy(resolved.get("style") or {}),
        "videos": [{
            "asset_id": asset.get("id"),
            "window": deepcopy(resolved.get("window")),
            "orientation": resolved.get("orientation", 0),
            "fps_override": resolved.get("fps_override"),
            "sample_fps": resolved.get("sample_fps"),
            "options": deepcopy(resolved.get("options") or {}),
            "detection": deepcopy(resolved.get("detection") or {}),
            "units": deepcopy(resolved.get("units") or []),
        }],
    }


def _job_field(field: Any, units: Sequence[Mapping[str, Any]]) -> tuple[str, str | None]:
    """把 videos[0].units[2].x 变成 unit.x，并给出对应单位 ID。"""

    text = str(field or "")
    if text.startswith("videos[0]."):
        text = text[len("videos[0]."):]
    unit_key = None
    if text.startswith("units[") and units:
        head, _, tail = text.partition("]")
        try:
            index = int(head[len("units["):])
        except ValueError:
            index = -1
        if 0 <= index < len(units):
            unit_key = units[index].get("id")
        text = "unit" + tail
    return text, unit_key


def validate_job(resolved: Mapping[str, Any], *,
                 animals: Iterable[str] | None = None) -> list[dict]:
    """冻结任务在提交与执行前都要通过的完整校验。

    问题带 ``asset_id``/``unit_id``/``field``/``message``；身份、路径与解码
    问题归到该视频，不让异常掩盖其他字段。
    """

    if not isinstance(resolved, Mapping):
        return [{"field": "job", "message": "任务必须是对象"}]
    issues: list[dict] = []
    assay = resolved.get("assay")
    if not experiments.exists(assay):
        issues.append({"field": "assay", "message": "未知实验"})
        return issues
    asset = resolved.get("asset")
    if not isinstance(asset, Mapping):
        return [{"field": "asset", "message": "缺少素材信息"}]
    asset_key = asset.get("id") if is_valid_id(asset.get("id")) else None

    if resolved.get("schema_version") != SCHEMA_VERSION:
        issues.append({"field": "schema_version",
                       "message": f"任务版本必须是 {SCHEMA_VERSION}"})
    for issue in verify_identity(asset):
        issues.append(dict(issue))
    path = asset.get("path")
    if not isinstance(path, str) or not path:
        issues.append({"field": "asset.path", "message": "缺少源视频路径"})
    # 媒体事实在冻结/重跑解析时已经重新探测并存进 asset.metadata；执行前只
    # 核对身份与参数，解码问题由 VideoSource 在真正读帧时报出。

    units = resolved.get("units")
    if not isinstance(units, list) or not units:
        issues.append({"field": "units", "message": "至少需要一个单位"})
        units = units if isinstance(units, list) else []

    issues.extend(validate_draft(_job_draft(resolved), complete=True,
                                 animals=animals))

    normalized: list[dict] = []
    for issue in issues:
        entry = dict(issue)
        field, unit_key = _job_field(issue.get("field"), units)
        entry["field"] = field
        if asset_key:
            entry.setdefault("asset_id", asset_key)
        if unit_key:
            entry.setdefault("unit_id", unit_key)
        normalized.append(entry)
    return normalized


def _coverage_issues(source: Any, window: Mapping[str, Any],
                     metadata: Mapping[str, Any]) -> list[dict]:
    """实际覆盖必须盖住请求的半开窗；元数据时长不能掩盖提前结束。

    只有真实 ``VideoSource`` 才带 ``coverage``；替身源没有这一事实时不猜。
    """

    coverage = getattr(source, "coverage", None)
    if not isinstance(coverage, Mapping):
        return []
    if not coverage.get("frame_count"):
        return [{"field": "window", "message": "窗口内没有可解码的有效曝光"}]
    start = float(window.get("start_s") or 0.0)
    declared = window.get("end_s")
    target_end = float(declared) if is_number(declared) else None
    if target_end is None:
        duration = metadata.get("duration_s")
        if is_number(duration) and float(duration) > start:
            target_end = float(duration)
    issues: list[dict] = []
    if coverage.get("start_s") is None or abs(
            float(coverage["start_s"]) - start) > 1e-6:
        issues.append({"field": "window.start_s",
                       "message": "源覆盖没有从请求起点开始"})
    if target_end is not None:
        if coverage.get("end_s") is None or float(coverage["end_s"]) < target_end - 1e-6:
            issues.append({"field": "window.end_s",
                           "message": "源覆盖没有到达请求终点就结束了"})
    return issues


def run_job(resolved: Mapping[str, Any], *,
            progress: Callable[[float, str], None] | None = None,
            cancel: Callable[[], bool] | None = None) -> Any:
    """执行一个冻结任务：再验证、建源、捕获评分，并补入真实覆盖。

    捕获进度只占 0…0.75，导出由调用方接着报到 .99。
    """

    issues = validate_job(resolved)
    if issues:
        first = issues[0]
        raise ValidationError(first["message"], first["field"])

    module = _module(resolved.get("assay"))
    asset = resolved.get("asset") or {}
    require_identity(asset)
    metadata = asset.get("metadata") or {}
    window = resolved.get("window") or {}
    source = media.VideoSource(
        asset.get("path"),
        start_s=float(window.get("start_s") or 0.0),
        end_s=window.get("end_s"),
        orientation=resolved.get("orientation", 0),
        fps_override=resolved.get("fps_override"),
        sample_fps=resolved.get("sample_fps"),
        source_fps=metadata.get("fps"),
        metadata=dict(metadata) if isinstance(metadata, Mapping) else None,
        decode_images=True,
    )

    def report(fraction: float, message: str = "") -> None:
        if progress is not None:
            progress(min(0.75, max(0.0, 0.75 * float(fraction))), message)

    context = AnalysisContext(job=resolved, source=source,
                              cancel_check=cancel, progress_callback=report)
    result = module.analyze(context)

    gaps = _coverage_issues(source, window,
                            metadata if isinstance(metadata, Mapping) else {})
    if gaps:
        raise ValidationError(gaps[0]["message"], gaps[0]["field"])

    merged = dict(result.metadata or {})
    coverage = getattr(source, "coverage", None)
    if isinstance(coverage, Mapping):
        merged["coverage"] = dict(coverage)
    merged["video"] = {
        "name": asset.get("name"),
        "fps": (float(source.source_fps)
                if is_number(getattr(source, "source_fps", None)) else None),
        # 来源说明：override/media 为真实解释时钟；default 表示源帧率不可用、
        # 名义值取默认 20。旧记录没有这两个键，等价于 media 的实测帧率。
        "fps_source": getattr(source, "source_fps_source", None),
        "analysis_fps": getattr(source, "analysis_fps", None),
        "orientation": getattr(source, "orientation", None),
    }
    source_note = getattr(source, "source_fps_note", None)
    if source_note:
        merged["video"]["fps_note"] = source_note
        if source_note not in result.warnings:
            result.warnings.append(source_note)
    merged["algorithm_version"] = ALGORITHM_VERSION
    merged.setdefault("assay", resolved.get("assay"))
    result.metadata = merged
    if progress is not None:
        progress(0.75, "正在导出结果")
    return result


def execute(resolved: Mapping[str, Any]) -> Any:
    """执行一个冻结任务；实验模块自己完成一次捕获与统计。"""

    return run_job(resolved)
