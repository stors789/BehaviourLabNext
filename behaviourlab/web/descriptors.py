"""设置页使用的字段描述：与引擎/检测允许范围同源，供服务端渲染真实表单。"""

from __future__ import annotations

from typing import Any, Mapping

from ..detection import BG_METHODS, BG_TYPES, common_detection_defaults

BG_METHOD_LABELS = {
    "threshold": "固定灰度阈值",
    "temporal": "时间背景差",
    "static": "静态背景",
    "mog2": "MOG2",
    "knn": "KNN",
    "local_otsu": "局部 Otsu",
}

BG_TYPE_LABELS = {"light": "亮背景（深色动物）", "dark": "暗背景（亮色动物）"}

DETECTION_FIELDS: tuple[dict[str, Any], ...] = (
    {"key": "bg_method", "label": "背景方法", "type": "select",
     "choices": [{"value": value, "label": BG_METHOD_LABELS.get(value, value)}
                 for value in BG_METHODS]},
    {"key": "bg_type", "label": "明暗", "type": "select",
     "choices": [{"value": value, "label": BG_TYPE_LABELS.get(value, value)}
                 for value in BG_TYPES]},
    {"key": "gray_threshold", "label": "灰度阈值", "type": "number",
     "min": 0, "max": 255},
    {"key": "body_gray_threshold", "label": "身体灰度阈值", "type": "number",
     "min": 0, "max": 255},
    {"key": "temporal_diff_threshold", "label": "时间差阈值", "type": "number",
     "min": 0, "max": 255},
    {"key": "erosion_kernel", "label": "腐蚀核", "type": "number", "min": 1,
     "integer": True},
    {"key": "erosion_iter", "label": "腐蚀次数", "type": "number", "min": 0,
     "integer": True},
    {"key": "dilation_kernel", "label": "膨胀核", "type": "number", "min": 1,
     "integer": True},
    {"key": "dilation_iter", "label": "膨胀次数", "type": "number", "min": 0,
     "integer": True},
    {"key": "min_area", "label": "最小面积", "type": "number", "min": 0,
     "unit": "px²", "integer": True},
    {"key": "max_area", "label": "最大面积", "type": "number", "min": 0,
     "unit": "px²", "integer": True},
    {"key": "temporal_bg_samples", "label": "背景样本数", "type": "number",
     "min": 1, "integer": True},
    {"key": "quality_threshold", "label": "检测质量阈值", "type": "number",
     "min": 0, "max": 1, "step": 0.05},
    {"key": "static_background", "label": "静态背景路径", "type": "text"},
)

COLORMAPS = ("viridis", "magma", "inferno", "plasma", "cividis")

STYLE_FIELDS: tuple[dict[str, Any], ...] = (
    {"key": "color", "label": "轨迹颜色", "type": "color"},
    {"key": "colormap", "label": "热图配色", "type": "select",
     "choices": [{"value": name, "label": name} for name in COLORMAPS]},
    {"key": "line_width", "label": "线条宽度", "type": "number",
     "min": 0.5, "max": 10, "step": 0.1},
    {"key": "dpi", "label": "DPI", "type": "number", "min": 72, "max": 600,
     "step": 1},
)

OUTPUT_FIELDS: tuple[dict[str, Any], ...] = (
    {"key": "annotated_video", "label": "生成标注视频", "type": "checkbox"},
)


def detection_defaults(assay: str | None = None) -> dict:
    """共享默认；给定实验时叠加该实验的检测覆盖。"""

    defaults = dict(common_detection_defaults())
    if assay:
        from .. import experiments

        if experiments.exists(assay):
            overrides = getattr(experiments.get(assay), "DETECTION_DEFAULTS", {})
            if isinstance(overrides, Mapping):
                defaults.update(dict(overrides))
    return defaults


def field_value(values: Mapping[str, Any], key: str,
                default: Any = None) -> Any:
    return values[key] if key in values else default
