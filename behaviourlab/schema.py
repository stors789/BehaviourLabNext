"""跨层数据契约：ID、严格 JSON、Frame/Detection、分析上下文与结果、观测行。

这里的对象被媒体、检测、实验、引擎、任务与导出共享；界面层不得另造一套
与之冲突的数据模型。
"""

from __future__ import annotations

import json
import math
import re
import secrets
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,100}")
HEX32_PATTERN = re.compile(r"[0-9a-f]{32}")


class ValidationError(ValueError):
    """带字段命名空间的输入错误。"""

    def __init__(self, message: str, field: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.field = field

    def as_issue(
        self, *, asset_id: str | None = None, unit_id: str | None = None
    ) -> dict:
        issue: dict[str, Any] = {"field": self.field, "message": self.message}
        if asset_id is not None:
            issue["asset_id"] = asset_id
        if unit_id is not None:
            issue["unit_id"] = unit_id
        return issue


def new_id() -> str:
    """生成随机 32 位十六进制 ID。"""

    return secrets.token_hex(16)


def is_valid_id(value: Any) -> bool:
    return isinstance(value, str) and bool(ID_PATTERN.fullmatch(value))


def require_id(value: Any, field: str = "id") -> str:
    if not is_valid_id(value):
        raise ValidationError("ID 必须是 1–100 个字母、数字、下划线或短横线", field)
    return value


def is_number(value: Any) -> bool:
    """布尔值不是数值。"""

    return isinstance(value, (int, float)) and not isinstance(value, bool)


def require_finite(value: Any, field: str, *, minimum: float | None = None,
                   maximum: float | None = None) -> float:
    if not is_number(value):
        raise ValidationError("必须是数值", field)
    number = float(value)
    if not math.isfinite(number):
        raise ValidationError("必须是有限数值", field)
    if minimum is not None and number < minimum:
        raise ValidationError(f"不能小于 {minimum:g}", field)
    if maximum is not None and number > maximum:
        raise ValidationError(f"不能大于 {maximum:g}", field)
    return number


def ensure_finite_tree(value: Any, prefix: str = "") -> None:
    """拒绝配置/结果中出现的 NaN 与 Infinity。"""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValidationError("对象的键必须是字符串", prefix or "value")
            ensure_finite_tree(item, f"{prefix}.{key}" if prefix else key)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            ensure_finite_tree(item, f"{prefix}[{index}]")
        return
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValidationError("必须是有限数值", prefix or "value")
        return


def check_json_value(value: Any, field: str = "") -> None:
    """校验对象可被严格 JSON 编码（不允许 NaN/Infinity 与非 JSON 类型）。"""

    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValidationError("JSON 中不允许 NaN 或 Infinity", field)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValidationError("JSON 对象的键必须是字符串", field)
            check_json_value(item, f"{field}.{key}" if field else key)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            check_json_value(item, f"{field}[{index}]")
        return
    raise ValidationError(f"不支持的值类型 {type(value).__name__}", field)


def sanitize_json(value: Any) -> Any:
    """把计算结果转换为可严格编码的 JSON；非有限浮点归一为 null。"""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return sanitize_json(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        return sanitize_json(value.tolist())
    raise TypeError(f"无法序列化的值类型 {type(value).__name__}")


def dumps_strict(value: Any, *, indent: int | None = None) -> str:
    sanitized = sanitize_json(value)
    ensure_finite_tree(sanitized)
    return json.dumps(
        sanitized,
        ensure_ascii=False,
        allow_nan=False,
        indent=indent,
        separators=(",", ":") if indent is None else None,
    )


# ---------------------------------------------------------------------------
# 帧与检测


@dataclass(slots=True)
class Frame:
    """一个分析采样帧；image 为 BGR 数组，纯人工路径可为 None。"""

    image: Any | None
    index: int
    time_s: float
    dt_s: float
    elapsed_s: float
    analysis_fps: float
    interval_start_s: float
    interval_end_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.index, int) or isinstance(self.index, bool) or self.index < 0:
            raise ValidationError("源帧号必须是非负整数", "frame")
        for name in ("time_s", "dt_s", "elapsed_s", "analysis_fps",
                     "interval_start_s", "interval_end_s"):
            require_finite(getattr(self, name), name)
        if self.interval_end_s < self.interval_start_s:
            raise ValidationError("有效区间终点不能早于起点", "interval_end_s")


Point = list[float]


def _normalize_point(value: Any, field: str) -> Point | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValidationError("点必须是 [x, y]", field)
    x = require_finite(value[0], f"{field}[0]")
    y = require_finite(value[1], f"{field}[1]")
    return [x, y]


def _point_xy(value: Any) -> tuple[float | None, float | None]:
    if value is None:
        return None, None
    return float(value[0]), float(value[1])


@dataclass(slots=True)
class Detection:
    """单帧检测结果；detected 由 center 是否为空决定。"""

    center: Point | None = None
    quality: float = 0.0
    contour: list[Point] = field(default_factory=list)
    mask: Any | None = None
    area: float = 0.0
    bbox: list[float] | None = None
    nose: Point | None = None
    tail_base: Point | None = None
    left_ear: Point | None = None
    right_ear: Point | None = None
    head_angle: float | None = None
    head_quality: float = 0.0
    missing_reason: str | None = None
    candidates: list[Any] = field(default_factory=list)

    @property
    def detected(self) -> bool:
        return self.center is not None

    def __post_init__(self) -> None:
        self.center = _normalize_point(self.center, "detection.center")
        if self.bbox is not None:
            if not isinstance(self.bbox, (list, tuple)) or len(self.bbox) != 4:
                raise ValidationError("bbox 必须是 [x, y, width, height]", "detection.bbox")
            self.bbox = [
                require_finite(value, f"detection.bbox[{index}]")
                for index, value in enumerate(self.bbox)
            ]
        self.nose = _normalize_point(self.nose, "detection.nose")
        self.tail_base = _normalize_point(self.tail_base, "detection.tail_base")
        self.left_ear = _normalize_point(self.left_ear, "detection.left_ear")
        self.right_ear = _normalize_point(self.right_ear, "detection.right_ear")
        self.quality = require_finite(self.quality, "detection.quality", minimum=0.0)
        self.head_quality = require_finite(self.head_quality, "detection.head_quality", minimum=0.0)
        self.area = require_finite(self.area, "detection.area", minimum=0.0)
        if self.head_angle is not None:
            self.head_angle = require_finite(self.head_angle, "detection.head_angle")


OBSERVATION_FIELDS = (
    "unit_id",
    "unit_name",
    "animal_id",
    "frame",
    "time_s",
    "source_dt_s",
    "elapsed_s",
    "analysis_fps",
    "source_interval_start_s",
    "source_interval_end_s",
    "interval_start_s",
    "interval_end_s",
    "dt_s",
    "in_window",
    "detected",
    "x_px",
    "y_px",
    "quality",
    "contour",
    "missing_reason",
    "head_angle",
    "head_quality",
    "nose_x_px",
    "nose_y_px",
    "tail_x_px",
    "tail_y_px",
    "left_ear_x_px",
    "left_ear_y_px",
    "right_ear_x_px",
    "right_ear_y_px",
)


def make_observation(
    *,
    unit_id: str,
    unit_name: str,
    animal_id: str | None,
    frame: int,
    time_s: float,
    source_dt_s: float,
    elapsed_s: float,
    analysis_fps: float,
    source_interval_start_s: float,
    source_interval_end_s: float,
    interval_start_s: float,
    interval_end_s: float,
    detection: Detection | None = None,
    in_window: bool = True,
    extra: Mapping[str, Any] | None = None,
) -> dict:
    """构造一条通用观测行；缺失位置与头向用 null，不填 0。"""

    require_id(unit_id, "unit_id")
    if not isinstance(unit_name, str) or not unit_name.strip():
        raise ValidationError("单位名称不能为空", "unit_name")
    if not isinstance(frame, int) or isinstance(frame, bool) or frame < 0:
        raise ValidationError("源帧号必须是非负整数", "frame")

    source_dt = require_finite(source_dt_s, "source_dt_s", minimum=0.0)
    start = require_finite(interval_start_s, "interval_start_s")
    end = require_finite(interval_end_s, "interval_end_s")
    if end < start:
        raise ValidationError("有效区间终点不能早于起点", "interval_end_s")

    detection = detection or Detection()
    detected = detection.detected
    x_px, y_px = _point_xy(detection.center)
    nose_x, nose_y = _point_xy(detection.nose)
    tail_x, tail_y = _point_xy(detection.tail_base)
    left_x, left_y = _point_xy(detection.left_ear)
    right_x, right_y = _point_xy(detection.right_ear)

    row: dict[str, Any] = {
        "unit_id": unit_id,
        "unit_name": unit_name,
        "animal_id": animal_id,
        "frame": frame,
        "time_s": require_finite(time_s, "time_s"),
        "source_dt_s": source_dt,
        "elapsed_s": require_finite(elapsed_s, "elapsed_s"),
        "analysis_fps": require_finite(analysis_fps, "analysis_fps"),
        "source_interval_start_s": require_finite(
            source_interval_start_s, "source_interval_start_s"
        ),
        "source_interval_end_s": require_finite(
            source_interval_end_s, "source_interval_end_s"
        ),
        "interval_start_s": start,
        "interval_end_s": end,
        "dt_s": end - start,
        "in_window": bool(in_window),
        "detected": detected,
        "x_px": x_px,
        "y_px": y_px,
        "quality": float(detection.quality) if detected else 0.0,
        "contour": [list(point) for point in detection.contour],
        "missing_reason": None if detected else detection.missing_reason,
        "head_angle": detection.head_angle if detected else None,
        "head_quality": float(detection.head_quality) if detected else 0.0,
        "nose_x_px": nose_x if detected else None,
        "nose_y_px": nose_y if detected else None,
        "tail_x_px": tail_x if detected else None,
        "tail_y_px": tail_y if detected else None,
        "left_ear_x_px": left_x if detected else None,
        "left_ear_y_px": left_y if detected else None,
        "right_ear_x_px": right_x if detected else None,
        "right_ear_y_px": right_y if detected else None,
    }

    if extra:
        for key, value in extra.items():
            if key in row:
                raise ValidationError(f"扩展字段与通用观测冲突：{key}", key)
            row[key] = value

    ensure_finite_tree(row, "observation")
    return row


# ---------------------------------------------------------------------------
# 分析上下文与结果


@dataclass
class AnalysisContext:
    """传给实验模块的运行上下文；纯人工 Y 的 detector 可为 None。"""

    job: Any
    source: Any = None
    detector: Any | None = None
    cancel_check: Callable[[], bool] | None = None
    progress_callback: Callable[[float, str], None] | None = None

    def cancelled(self) -> bool:
        return bool(self.cancel_check and self.cancel_check())

    def progress(self, fraction: float, message: str = "") -> None:
        value = require_finite(fraction, "progress", minimum=0.0, maximum=1.0)
        if self.progress_callback is not None:
            self.progress_callback(value, message)


@dataclass
class AnalysisResult:
    """实验输出；空列表是有效空结果，不用虚构行占位。"""

    observations: list = field(default_factory=list)
    summary: list = field(default_factory=list)
    events: list = field(default_factory=list)
    bins: list = field(default_factory=list)
    tables: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @classmethod
    def empty(cls) -> "AnalysisResult":
        return cls()

    def as_dict(self) -> dict:
        return {
            "observations": self.observations,
            "summary": self.summary,
            "events": self.events,
            "bins": self.bins,
            "tables": self.tables,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }
