"""Y 迷宫：中心三角形成员、三臂入臂与自发交替的自动评分。

中心区是手绘三点三角形，A/B/C 是手绘四角臂区；坐标由臂长/臂宽相对像素跨度
的六个比例平均换算，因此是 ``cm_estimated``。"入臂"以沿臂轴的投影深度判定，
候选关联用两点预测与双重面积参考，缺失只保留关联证据，不输出保持点。
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .. import detection, geometry as geo, media, timeline
from ..schema import AnalysisResult, Detection, ValidationError, is_number
from .base import geometry_field, metric, parameter

KEY = "ymaze"

DEFAULTS = {
    "mode": "auto",
    "arm_length_cm": 30.0,
    "arm_width_cm": 10.0,
    "arm_entry_rule": "depth",
    "entry_depth_ratio": 0.2,
    "max_jump_cm": 12.0,
    "tracking_hold_s": 0.4,
    "entry_confirm_s": 0.12,
    "entry_reset_center_s": 0.2,
    "require_center": True,
    "movement_threshold_cm_s": 1.0,
    "bin_seconds": 60.0,
}

# 入臂规则：depth=进入指定深度并持续确认（原规则，默认）；
# center=中心点处于新区域的首个有效观测即计一次（无防抖）。
ARM_ENTRY_RULES = ("depth", "center")
# 仅深度规则消费的参数；中心点规则下不生效（界面需明确提示）。
DEPTH_RULE_ONLY_KEYS = ("entry_depth_ratio", "entry_confirm_s",
                        "entry_reset_center_s", "require_center")

OPTION_KEYS = tuple(DEFAULTS)
PATH_MODES = ("auto", "manual")
ZONE_KEYS = ("center", "A", "B", "C")
ARM_KEYS = ("A", "B", "C")
ZONE_LABELS = {"center": "中心区", "A": "A 臂", "B": "B 臂", "C": "C 臂"}
TIME_EPS = 1e-9
AREA_LOW = 0.25
AREA_HIGH = 4.0
EVIDENCE_SAMPLES = 9
MISSING_GATE = "no_candidate_within_gate"

DETECTION_DEFAULTS: dict = {}

SPEC = {
    "key": KEY,
    "label": "Y 迷宫",
    "short": "Y 迷宫",
    "parameters": [
        parameter("arm_length_cm", "臂长", "number", 30.0, unit="cm",
                  minimum=0, step="any"),
        parameter("arm_width_cm", "臂宽", "number", 10.0, unit="cm",
                  minimum=0, step="any"),
        parameter("arm_entry_rule", "入臂规则", "select", "depth",
                  choices=[
                      {"value": "depth",
                       "label": "进入深度+持续确认（原规则，默认）"},
                      {"value": "center",
                       "label": "中心点入臂（新区域首个有效观测即计）"},
                  ],
                  section="实验参数",
                  help="参考 IMPC“自动追踪以小鼠中心进入新区域判定”的约定；"
                       "本软件具体化为中心点处于新区域的首个有效观测即记一次，"
                       "不要求额外深度或确认时间。臂/中心边界反复切换会按观测"
                       "转换计数（本规则无时间防抖）。中心点规则下入臂深度比例、"
                       "入臂确认时间、中心区重置时间、再次入臂须经过中心区均"
                       "不生效。"),
        parameter("entry_depth_ratio", "入臂深度比例", "number", 0.2,
                  minimum=0, maximum=1, step="any", section="实验参数"),
        parameter("max_jump_cm", "预测位置最大偏差", "number", 12.0,
                  unit="cm", minimum=0, step="any", section="实验参数"),
        parameter("tracking_hold_s", "短时丢失跟踪时间", "number", 0.4,
                  unit="s", minimum=0, step="any", section="实验参数"),
        parameter("entry_confirm_s", "入臂确认时间", "number", 0.12,
                  unit="s", minimum=0, step="any", section="实验参数"),
        parameter("entry_reset_center_s", "中心区重置时间", "number", 0.2,
                  unit="s", minimum=0, step="any", section="实验参数"),
        parameter("require_center", "再次入臂须经过中心区", "checkbox", True,
                  section="实验参数"),
        parameter("movement_threshold_cm_s", "移动速度阈值", "number", 1.0,
                  unit="cm/s", minimum=0, step="any", section="实验参数"),
        parameter("bin_seconds", "时间箱长度", "number", 60.0, unit="s",
                  minimum=0, step="any", section="实验参数"),
    ],
    "geometry": [
        geometry_field("center", "中心区三点", "polygon", count=3, fixed=True,
                       auto_commit=True),
        geometry_field("A", "A 臂四角", "quad", count=4),
        geometry_field("B", "B 臂四角", "quad", count=4),
        geometry_field("C", "C 臂四角", "quad", count=4),
        geometry_field("mask", "检测遮罩", "polygon", min_points=3,
                       optional=True),
    ],
    "paths": {
        "auto": ["select", "range", "arena", "detection"],
        "manual": ["select", "range", "arena", "review"],
    },
    "metrics": [
        metric("total_arm_entries", "总入臂次数"),
        metric("alternation_count", "交替次数"),
        metric("alternation_opportunities", "交替机会数"),
        metric("alternation_percent", "自发交替率", "%"),
        metric("center_time_s", "中心区时间", "s"),
        metric("arm_A_time_s", "A 臂时间", "s"),
        metric("arm_B_time_s", "B 臂时间", "s"),
        metric("arm_C_time_s", "C 臂时间", "s"),
        metric("total_distance_cm", "估计总距离", "cm"),
        metric("mean_speed_cm_s", "估计平均速度", "cm/s"),
    ],
    "template_keys": ["arm_length_cm", "arm_width_cm"],
}


# -- 参数与几何校验 -----------------------------------------------------

def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not is_number(value):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _point(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    x, y = _finite(value[0]), _finite(value[1])
    if x is None or y is None:
        return None
    return x, y


def _shape_points(shape: Any) -> list[tuple[float, float]] | None:
    if not isinstance(shape, Mapping):
        return None
    raw = shape.get("points")
    if not isinstance(raw, (list, tuple)):
        return None
    points: list[tuple[float, float]] = []
    for item in raw:
        point = _point(item)
        if point is None:
            return None
        points.append(point)
    return points


def _is_convex(points: Sequence[Sequence[float]]) -> bool:
    signs = []
    for index in range(len(points)):
        ax, ay = points[index]
        bx, by = points[(index + 1) % len(points)]
        cx, cy = points[(index + 2) % len(points)]
        value = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
        if abs(value) <= 1e-12:
            return False
        signs.append(value > 0)
    return all(sign == signs[0] for sign in signs)


def _adjacent_conflict(points: Sequence[Sequence[float]]) -> bool:
    """相邻顶点重合（含首尾）不是有效区域。"""

    count = len(points)
    for index in range(count):
        ax, ay = points[index]
        bx, by = points[(index + 1) % count]
        if abs(ax - bx) <= 1e-9 and abs(ay - by) <= 1e-9:
            return True
    return False


def _quad(shape: Any) -> list[tuple[float, float]] | None:
    """四角臂区：恰好四个有限点、凸、正面积、不交叉、相邻点不重合。"""

    if not isinstance(shape, Mapping):
        return None
    if shape.get("type") not in ("quad", "polygon"):
        return None
    points = _shape_points(shape)
    if points is None or len(points) != 4:
        return None
    if _adjacent_conflict(points) or geo.quad_is_crossed(points):
        return None
    if not _is_convex(points):
        return None
    if abs(geo.polygon_area(points)) <= 1e-9:
        return None
    return points


def _triangle(shape: Any) -> list[tuple[float, float]] | None:
    """中心区：恰好三个有限点、正面积、相邻点不重合。"""

    if not isinstance(shape, Mapping):
        return None
    if shape.get("type") not in ("polygon", "quad"):
        return None
    points = _shape_points(shape)
    if points is None or len(points) != 3:
        return None
    if _adjacent_conflict(points):
        return None
    if abs(geo.polygon_area(points)) <= 1e-9:
        return None
    return points


def _simple_polygon(shape: Any) -> list[tuple[float, float]] | None:
    points = _shape_points(shape)
    if points is None or len(points) < 3:
        return None
    if _adjacent_conflict(points):
        return None
    if abs(geo.polygon_area(points)) <= 1e-9:
        return None
    if geo.polygon_self_intersects(points):
        return None
    return points


def _centroid(points: Sequence[Sequence[float]]) -> tuple[float, float]:
    count = len(points)
    return (sum(float(point[0]) for point in points) / count,
            sum(float(point[1]) for point in points) / count)


def arm_frames(zones: Mapping[str, Sequence[Sequence[float]]]) -> dict:
    """每臂的长轴单位向量、投影近远端与宽度（像素）。"""

    center = _centroid(zones["center"])
    frames: dict[str, dict] = {}
    for key in ARM_KEYS:
        points = zones[key]
        middle = _centroid(points)
        dx, dy = middle[0] - center[0], middle[1] - center[1]
        norm = math.hypot(dx, dy)
        if norm <= 1e-12:
            raise ValidationError("臂中心不能与中心区重合",
                                  f"geometry.zones.{key}")
        u = (dx / norm, dy / norm)
        v = (-u[1], u[0])
        along = [point[0] * u[0] + point[1] * u[1] for point in points]
        across = [point[0] * v[0] + point[1] * v[1] for point in points]
        near, far = min(along), max(along)
        width = max(across) - min(across)
        if far - near <= 1e-9 or width <= 1e-9:
            raise ValidationError("臂区长宽不能为 0", f"geometry.zones.{key}")
        frames[key] = {"u": u, "v": v, "center": middle, "near": near,
                       "far": far, "width": width}
    return frames


def estimated_scale(options: Mapping[str, Any],
                    frames: Mapping[str, Mapping[str, Any]]) -> float:
    """六个比例（三臂 × 臂长/跨度、臂宽/像素宽）的算术平均，单位 cm/px。"""

    ratios: list[float] = []
    length = float(options["arm_length_cm"])
    width_cm = float(options["arm_width_cm"])
    for frame in frames.values():
        ratios.append(length / (float(frame["far"]) - float(frame["near"])))
        ratios.append(width_cm / float(frame["width"]))
    return sum(ratios) / len(ratios)


def validate_unit(options: Mapping[str, Any] | None,
                  geometry: Mapping[str, Any] | None,
                  provided: Iterable[str] | None = None, *,
                  complete: bool = False) -> list[dict]:
    """只校验显式参数与已绘区域；未完成草稿仍可保存。"""

    values = options if isinstance(options, Mapping) else {}
    provided = set(OPTION_KEYS if provided is None else provided)
    issues: list[dict] = []

    def number(key: str) -> float | None:
        if key not in provided:
            return None
        value = _finite(values.get(key))
        if value is None:
            issues.append({"field": f"options.{key}", "message": "必须是有限数值"})
        return value

    for key in ("arm_length_cm", "arm_width_cm", "max_jump_cm"):
        value = number(key)
        if value is not None and value <= 0:
            issues.append({"field": f"options.{key}", "message": "必须大于 0"})

    ratio = number("entry_depth_ratio")
    if ratio is not None and not 0 <= ratio <= 1:
        issues.append({"field": "options.entry_depth_ratio",
                       "message": "必须在 0 到 1 之间"})

    for key in ("tracking_hold_s", "entry_confirm_s", "entry_reset_center_s"):
        value = number(key)
        if value is not None and value < 0:
            issues.append({"field": f"options.{key}", "message": "必须 ≥ 0"})

    threshold = number("movement_threshold_cm_s")
    if threshold is not None and threshold < 0:
        issues.append({"field": "options.movement_threshold_cm_s",
                       "message": "必须 ≥ 0"})

    bin_width = number("bin_seconds")
    if bin_width is not None and bin_width <= 0:
        issues.append({"field": "options.bin_seconds",
                       "message": "必须大于 0"})

    if "require_center" in provided:
        if not isinstance(values.get("require_center"), bool):
            issues.append({"field": "options.require_center",
                           "message": "必须是严格布尔值"})

    if "mode" in provided:
        mode = values.get("mode")
        if mode not in PATH_MODES:
            issues.append({"field": "options.mode",
                           "message": "路径必须是自动或人工"})

    # 缺字段（旧草稿/旧冻结任务）按原规则运行；只有显式非法值才报错。
    rule = values.get("arm_entry_rule")
    if rule is not None and rule not in ARM_ENTRY_RULES:
        issues.append({"field": "options.arm_entry_rule",
                       "message": "入臂规则必须是 depth 或 center"})

    # 人工 Y 由 engine 按 unit.manual.mode 走 validate_manual，不会进入这里；
    # options.mode 只是设置项，不能用来豁免自动几何。
    issues.extend(validate_geometry(geometry, require=complete))
    return issues


def validate_geometry(geometry: Mapping[str, Any] | None,
                      *, require: bool = False) -> list[dict]:
    """中心三角形 + 三个凸四角臂区齐全；遮罩为合法简单多边形。"""

    geometry = geometry or {}
    zones = geometry.get("zones")
    if not isinstance(zones, Mapping):
        if require:
            return [{"field": "geometry.zones", "message": "缺少中心区与三个臂区"}]
        return []
    issues: list[dict] = []
    valid: dict[str, list[tuple[float, float]]] = {}
    for key in ZONE_KEYS:
        field = f"geometry.zones.{key}"
        shape = zones.get(key)
        if shape is None:
            if require:
                issues.append({"field": field,
                               "message": "缺少" + ZONE_LABELS[key]})
            continue
        points = _triangle(shape) if key == "center" else _quad(shape)
        if points is None:
            issues.append({
                "field": field,
                "message": ("需要 3 个有限点、正面积且相邻点不重合"
                            if key == "center"
                            else "需要 4 个有限点、凸、正面积、不交叉"
                                 "且相邻点不重合"),
            })
            continue
        valid[key] = points
    if len(valid) == len(ZONE_KEYS):
        try:
            arm_frames(valid)
        except ValidationError as error:
            issues.append({"field": error.field, "message": error.message})
    mask = geometry.get("mask")
    if mask is not None and _simple_polygon(mask) is None:
        issues.append({"field": "geometry.mask",
                       "message": "检测遮罩至少 3 个有限点、正面积且不自交"})
    return issues


def step_drawn(geometry: Mapping[str, Any] | None, key: str) -> bool:
    """画布、预览与最终校验共用的“该步骤是否已绘制”判定。"""

    geometry = geometry or {}
    if key in ZONE_KEYS:
        zones = geometry.get("zones")
        if not isinstance(zones, Mapping):
            return False
        shape = zones.get(key)
        return (_triangle(shape) if key == "center" else _quad(shape)) is not None
    return geometry.get(key) is not None


# -- 区域、深度与尺度 ---------------------------------------------------

def zones_of(geometry: Mapping[str, Any]) -> dict[str, list[tuple[float, float]]]:
    zones = (geometry or {}).get("zones")
    parsed: dict[str, list[tuple[float, float]]] = {}
    for key in ZONE_KEYS:
        shape = zones.get(key) if isinstance(zones, Mapping) else None
        points = _triangle(shape) if key == "center" else _quad(shape)
        if points is None:
            raise ValidationError("缺少" + ZONE_LABELS[key],
                                  f"geometry.zones.{key}")
        parsed[key] = points
    return parsed


def zone_of(point: Sequence[float],
            zones: Mapping[str, Sequence[Sequence[float]]]) -> str:
    """重叠按 center→A→B→C 顺序取首个命中；都不在即 unknown。"""

    for key in ZONE_KEYS:
        if geo.point_in_polygon(point, zones[key]):
            return key
    return "unknown"


def arm_depth(point: Sequence[float], frame: Mapping[str, Any]) -> float:
    """沿臂轴投影的入臂深度；不以离中心距离代替。"""

    u = frame["u"]
    projection = float(point[0]) * u[0] + float(point[1]) * u[1]
    span = float(frame["far"]) - float(frame["near"])
    if span <= 0:
        return 0.0
    return (projection - float(frame["near"])) / span


# -- 候选关联 -----------------------------------------------------------

class CandidateLink:
    """单位级候选关联：两点预测 + 双重面积参考，缺失只保留证据。

    位置历史只保留最近两个被接受的时刻；面积参考分别是最近九个正有限
    面积的中位数 r 与最初九个（集满即固定）的中位数 a。短缺口保留关联
    证据用于接续，但不输出保持点；缺口超过 ``tracking_hold_s`` 才按面积
    重新获取，并保留已建立的面积参考。
    """

    def __init__(self, options: Mapping[str, Any], scale: float) -> None:
        self.max_jump_cm = float(options["max_jump_cm"])
        self.hold_s = float(options["tracking_hold_s"])
        self.scale = float(scale)
        self.history: list[tuple[float, tuple[float, float]]] = []
        self.areas: list[float] = []
        self.reference: list[float] = []
        self.last_accepted_s: float | None = None
        self.missed = False

    # -- 证据 --
    def area_ok(self, area: Any) -> bool:
        """有面积参考时须同时落在 r 与 a 的 [1/4, 4] 内；无证据即兼容。"""

        value = _finite(area)
        if value is None or value <= 0:
            return True
        local = statistics.median(self.areas) if self.areas else None
        reference = (statistics.median(self.reference)
                     if self.reference else None)
        if local is None or reference is None or local <= 0 or reference <= 0:
            return True
        return (AREA_LOW <= value / local <= AREA_HIGH
                and AREA_LOW <= value / reference <= AREA_HIGH)

    def accept(self, candidate: Mapping[str, Any], time_s: float) -> None:
        now = float(time_s)
        center = (float(candidate["center"][0]), float(candidate["center"][1]))
        self.history.append((now, center))
        del self.history[:-2]
        area = _finite(candidate.get("area"))
        if area is not None and area > 0:
            self.areas.append(area)
            del self.areas[:-EVIDENCE_SAMPLES]
            if len(self.reference) < EVIDENCE_SAMPLES:
                self.reference.append(area)
        self.last_accepted_s = now
        self.missed = False

    def miss(self) -> None:
        """失败只标记缺失；历史与面积参考都保留。"""

        self.missed = True

    # -- 选择 --
    def pick(self, candidates: Sequence[Mapping[str, Any]],
             time_s: float) -> Mapping[str, Any] | None:
        compatible = [item for item in candidates
                      if self.area_ok(item.get("area"))]
        if not compatible:
            return None
        if self._reacquire(float(time_s)):
            selected = self._by_area(compatible)
            if selected is not None:
                # 重新获取即重置位置历史，面积参考不清空。
                self.history = []
            return selected
        predicted = self.predict(float(time_s))
        if predicted is None:
            return None
        best: Mapping[str, Any] | None = None
        best_cm: float | None = None
        for item in compatible:
            center = item["center"]
            distance_cm = math.hypot(float(center[0]) - predicted[0],
                                     float(center[1]) - predicted[1]) * self.scale
            if distance_cm > self.max_jump_cm + 1e-9:
                continue
            if best is None or distance_cm < best_cm - 1e-12:
                best, best_cm = item, distance_cm
        return best

    def predict(self, time_s: float) -> tuple[float, float] | None:
        """一点历史取原位置；两点按真实时间差估计速度外推。"""

        if not self.history:
            return None
        if len(self.history) == 1:
            return self.history[0][1]
        (t0, p0), (t1, p1) = self.history[-2], self.history[-1]
        span = t1 - t0
        if span <= 0:
            return p1
        ratio = max(0.0, (float(time_s) - t1) / span)
        return (p1[0] + (p1[0] - p0[0]) * ratio,
                p1[1] + (p1[1] - p0[1]) * ratio)

    def _reacquire(self, time_s: float) -> bool:
        if not self.history:
            return True
        if not self.missed or self.last_accepted_s is None:
            return False
        return time_s - self.last_accepted_s > self.hold_s + TIME_EPS

    def _by_area(self, compatible: Sequence[Mapping[str, Any]]
                 ) -> Mapping[str, Any] | None:
        """有面积参考选 |ln(面积/r)| 最小，未知面积排最后；无参考取首个。"""

        local = statistics.median(self.areas) if self.areas else None
        if local is None or local <= 0:
            return compatible[0] if compatible else None
        scored: list[tuple[int, float, Mapping[str, Any]]] = []
        for item in compatible:
            area = _finite(item.get("area"))
            if area is not None and area > 0:
                scored.append((0, abs(math.log(area / local)), item))
            else:
                scored.append((1, 0.0, item))
        scored.sort(key=lambda entry: (entry[0], entry[1]))
        return scored[0][2]


# -- 观测行 -------------------------------------------------------------

def _unit_dict(unit: Mapping[str, Any]) -> dict:
    return {"id": unit.get("id"), "name": unit.get("name"),
            "animal_id": unit.get("animal_id")}


def observation_row(unit: Mapping[str, Any], frame: Any,
                    result: Any) -> dict:
    return timeline.base_row(
        _unit_dict(unit), frame, result,
        extra={"detection_method": (unit.get("detection") or {}).get("bg_method")},
    )


def _region_for(unit: Mapping[str, Any], options: Mapping[str, Any],
                width: int, height: int) -> np.ndarray:
    geometry = unit.get("geometry") or {}
    region, _ = detection.geometry_region(geometry, KEY, options, width, height)
    return region


def _prepare(unit: Mapping[str, Any]) -> tuple[dict, dict, float]:
    geometry = unit.get("geometry") or {}
    zones = zones_of(geometry)
    frames = arm_frames(zones)
    return zones, frames, estimated_scale(unit["options"], frames)


def _annotate(row: dict,
              zones: Mapping[str, Sequence[Sequence[float]]],
              scale: float) -> None:
    row["coordinate_unit"] = "cm_estimated"
    row["zone"] = None
    row["current_arm"] = None
    row["arm_entry_event"] = False
    row["arm_entry_name"] = None
    row["alternation_event"] = False
    row["recent_entry_sequence"] = None
    if not row.get("detected") or row.get("x_px") is None:
        row.update({"x_cm": None, "y_cm": None})
        return
    point = (float(row["x_px"]), float(row["y_px"]))
    row["x_cm"] = point[0] * scale
    row["y_cm"] = point[1] * scale
    row["zone"] = zone_of(point, zones)


def _track(rows: Sequence[Mapping[str, Any]], window: timeline.Window,
           frames: Mapping[str, Mapping[str, Any]],
           options: Mapping[str, Any], unit: Mapping[str, Any]) -> dict:
    """入臂状态机：深度门槛、稳定回中心重置、真实时间确认与交替序列。"""

    events: list[dict] = []
    if not window.known:
        return {"events": events}
    origin = float(window.start_s)
    depth_ratio = float(options["entry_depth_ratio"])
    confirm = float(options["entry_confirm_s"])
    reset = float(options["entry_reset_center_s"])
    require_center = bool(options["require_center"])
    armed = True
    last_arm: str | None = None
    pending: dict | None = None
    center_since: float | None = None
    sequence: list[str] = []

    for row in sorted(rows, key=lambda item: (item["time_s"], item["frame"])):
        if not row.get("in_window"):
            continue
        moment = float(row["time_s"])
        if moment < origin - TIME_EPS:
            # 裁切前上下文只贡献占用，不初始化/推进或重置状态。
            continue
        zone = row.get("zone")
        if not row.get("detected") or zone is None:
            center_since = None
            pending = None
            row["current_arm"] = last_arm
            continue
        if zone == "center":
            pending = None
            if center_since is None:
                center_since = moment
            elif moment - center_since >= reset - TIME_EPS:
                armed = True
            row["current_arm"] = last_arm
            continue
        center_since = None
        if zone not in frames:
            pending = None
            row["current_arm"] = last_arm
            continue

        depth = arm_depth((row["x_px"], row["y_px"]), frames[zone])
        allowed = armed or (not require_center and zone != last_arm)
        if depth + 1e-9 < depth_ratio or not allowed:
            pending = None
            row["current_arm"] = last_arm
            continue
        if pending is None or pending["arm"] != zone:
            pending = {"arm": zone, "start": moment}
        if moment - float(pending["start"]) + 1e-9 >= confirm:
            start = float(pending["start"])
            sequence.append(zone)
            index = len(sequence)
            recent = sequence[-3:]
            opportunity = index >= 3
            alternation = opportunity and len(set(recent)) == 3
            events.append({
                "event_id": f"{unit.get('id')}:arm_entry:{index}",
                "unit_id": unit.get("id"),
                "unit_name": unit.get("name"),
                "kind": "arm_entry",
                "time_s": start,
                "confirmed_s": moment,
                "target": zone,
                "source": "automatic",
                "frame": row["frame"],
                "entry_index": index,
                "recent_sequence": "-".join(recent),
                "alternation_opportunity": opportunity,
                "alternation": alternation,
            })
            row["arm_entry_event"] = True
            row["arm_entry_name"] = zone
            row["alternation_event"] = alternation
            row["recent_entry_sequence"] = "-".join(recent)
            armed = False
            last_arm = zone
            pending = None
        row["current_arm"] = last_arm
    return {"events": events}


def _track_center(rows: Sequence[Mapping[str, Any]], window: timeline.Window,
                  frames: Mapping[str, Mapping[str, Any]],
                  options: Mapping[str, Any], unit: Mapping[str, Any]) -> dict:
    """中心点入臂规则：中心点处于新区域的首个有效观测即记一次。

    与深度规则共用窗口/行序约定：窗外与裁切前行只贡献占用、不推进状态；
    不读入臂深度比例、入臂确认时间、中心区重置时间与再次入臂须经过
    中心区——这些参数在本规则下不生效。缺失、无效或未知区域不构成
    入臂，也不单独制造离臂/重入：它们不更新“上一有效区域”，恢复后
    仍在原臂不重复计数，恢复后观察到其他有效区域按转换计数。臂↔中心
    边界反复切换按观测转换计数（本规则无时间防抖）。事件时间与确认
    时间均取该次有效观测的源时间。
    """

    events: list[dict] = []
    if not window.known:
        return {"events": events}
    origin = float(window.start_s)
    last_zone: str | None = None   # 上一有效观测的区域；缺失/未知不更新。
    last_arm: str | None = None    # 最近一次计入的臂（供观测行展示）。
    sequence: list[str] = []

    for row in sorted(rows, key=lambda item: (item["time_s"], item["frame"])):
        if not row.get("in_window"):
            continue
        moment = float(row["time_s"])
        if moment < origin - TIME_EPS:
            # 裁切前上下文只贡献占用，不初始化/推进状态，不产生窗外事件。
            continue
        zone = row.get("zone")
        if not row.get("detected") or zone is None:
            row["current_arm"] = last_arm
            continue
        if zone in ARM_KEYS and zone != last_zone:
            sequence.append(zone)
            index = len(sequence)
            recent = sequence[-3:]
            opportunity = index >= 3
            alternation = opportunity and len(set(recent)) == 3
            events.append({
                "event_id": f"{unit.get('id')}:arm_entry:{index}",
                "unit_id": unit.get("id"),
                "unit_name": unit.get("name"),
                "kind": "arm_entry",
                "time_s": moment,
                "confirmed_s": moment,
                "target": zone,
                "source": "automatic",
                "frame": row["frame"],
                "entry_index": index,
                "recent_sequence": "-".join(recent),
                "alternation_opportunity": opportunity,
                "alternation": alternation,
            })
            row["arm_entry_event"] = True
            row["arm_entry_name"] = zone
            row["alternation_event"] = alternation
            row["recent_entry_sequence"] = "-".join(recent)
            last_arm = zone
        # 只有有效区域（中心/三臂）推进“上一有效区域”；unknown 保持中性。
        if zone in ZONE_KEYS:
            last_zone = zone
        row["current_arm"] = last_arm
    return {"events": events}


def _percent(numerator: float, denominator: float) -> float | None:
    return timeline.ratio(numerator * 100.0, denominator)


def _summary(unit: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
             segments: Sequence[Mapping[str, Any]], window: timeline.Window,
             options: Mapping[str, Any], scale: float,
             events: Sequence[Mapping[str, Any]]) -> dict:
    threshold = float(options["movement_threshold_cm_s"])
    times = timeline.time_summary(rows)
    seconds = {key: 0.0 for key in (*ZONE_KEYS, "unknown")}
    detected_frames = window_frames = 0
    for row in rows:
        span = float(row.get("dt_s") or 0.0)
        if row.get("in_window"):
            window_frames += 1
            if row.get("detected"):
                detected_frames += 1
        if not row.get("detected") or span <= 0:
            continue
        zone = row.get("zone")
        if zone in seconds:
            seconds[zone] += span
        elif zone is not None:
            seconds["unknown"] += span

    distance = measurable = moving = immobility = 0.0
    max_speed: float | None = None
    for segment in segments:
        length = float(segment["distance"])
        distance += length
        span = float(segment["end_s"]) - float(segment["start_s"])
        measurable += span
        if float(segment["speed"]) > threshold:
            moving += span
        else:
            immobility += span
        if span > 0:
            speed = float(segment["speed"])
            max_speed = speed if max_speed is None else max(max_speed, speed)

    entries = {key: 0 for key in ARM_KEYS}
    alternation = opportunities = 0
    for event in events:
        target = event.get("target")
        if target in entries:
            entries[target] += 1
        if event.get("alternation_opportunity"):
            opportunities += 1
        if event.get("alternation"):
            alternation += 1
    total_entries = sum(entries.values())

    summary = {
        "unit_id": unit.get("id"),
        "unit_name": unit.get("name"),
        "mode": KEY,
        "coordinate_unit": "cm_estimated",
        "total_time_s": times["total_time_s"],
        "observed_time_s": times["observed_time_s"],
        "missing_time_s": times["missing_time_s"],
        "detected_frames": detected_frames,
        "window_frames": window_frames,
        "total_frames": window_frames,
        "center_time_s": seconds["center"],
        "arm_A_time_s": seconds["A"],
        "arm_B_time_s": seconds["B"],
        "arm_C_time_s": seconds["C"],
        "unknown_time_s": seconds["unknown"],
        "zone_seconds": dict(seconds),
        "total_arm_entries": total_entries,
        "arm_A_entries": entries["A"],
        "arm_B_entries": entries["B"],
        "arm_C_entries": entries["C"],
        "alternation_count": alternation,
        "alternation_opportunities": opportunities,
        "alternation_percent": _percent(alternation, opportunities),
        "total_distance_cm": distance,
        "measurable_motion_time_s": measurable,
        "moving_time_s": moving,
        "immobility_time_s": immobility,
        "mean_speed_cm_s": (distance / measurable if measurable > 0 else None),
        "max_speed_cm_s": max_speed,
        "cm_per_px_est": scale,
        "analysis_window_start_s": window.start_s if window.known else None,
        "analysis_window_end_s": window.end_s if window.known else None,
    }
    for key in OPTION_KEYS:
        if key == "mode":
            continue
        summary.setdefault(key, options.get(key))
    return summary


def _bin_rows(rows: Sequence[Mapping[str, Any]],
              segments: Sequence[Mapping[str, Any]],
              events: Sequence[Mapping[str, Any]],
              window: timeline.Window, options: Mapping[str, Any],
              unit: Mapping[str, Any]) -> list[dict]:
    """1 基分箱：曝光按秒拆，事件按起点入箱，交替用全序列判定后入箱。"""

    if not window.known or window.duration_s <= 0:
        return []
    bin_width = float(options["bin_seconds"])
    origin = float(window.start_s)
    duration = window.duration_s
    count = max(1, int(math.ceil((duration - 1e-12) / bin_width)))
    threshold = float(options["movement_threshold_cm_s"])

    buckets = [{"observed": 0.0, "missing": 0.0, "unknown": 0.0,
                "center": 0.0, "A": 0.0, "B": 0.0, "C": 0.0,
                "entries": {key: 0 for key in ARM_KEYS},
                "alternation": 0, "opportunities": 0,
                "distance": 0.0, "measurable": 0.0, "moving": 0.0,
                "immobility": 0.0}
               for _ in range(count)]

    for row in rows:
        pieces = timeline.split_interval(row["interval_start_s"],
                                         row["interval_end_s"], origin,
                                         bin_width)
        for index, low, high in pieces:
            if not 0 <= index < count:
                continue
            bucket = buckets[index]
            span = high - low
            if not row.get("detected"):
                bucket["missing"] += span
                continue
            bucket["observed"] += span
            zone = row.get("zone")
            if zone in ("center", "A", "B", "C"):
                bucket[zone] += span
            else:
                bucket["unknown"] += span

    for segment in segments:
        span = float(segment["end_s"]) - float(segment["start_s"])
        if span <= 0:
            continue
        pieces = timeline.split_interval(segment["start_s"], segment["end_s"],
                                         origin, bin_width)
        for index, low, high in pieces:
            if not 0 <= index < count:
                continue
            bucket = buckets[index]
            ratio = (high - low) / span
            bucket["distance"] += float(segment["distance"]) * ratio
            bucket["measurable"] += high - low
            if float(segment["speed"]) > threshold:
                bucket["moving"] += high - low
            else:
                bucket["immobility"] += high - low

    for event in events:
        index = int(math.floor((float(event["time_s"]) - origin) / bin_width
                               + 1e-9))
        if not 0 <= index < count:
            continue
        bucket = buckets[index]
        target = event.get("target")
        if target in bucket["entries"]:
            bucket["entries"][target] += 1
        if event.get("alternation_opportunity"):
            bucket["opportunities"] += 1
        if event.get("alternation"):
            bucket["alternation"] += 1

    bins: list[dict] = []
    for index, bucket in enumerate(buckets):
        start = index * bin_width
        end = min(start + bin_width, duration)
        total = sum(bucket["entries"].values())
        bins.append({
            "unit_id": unit.get("id"),
            "unit_name": unit.get("name"),
            "bin_index": index + 1,
            "start_s": start,
            "end_s": end,
            "duration_s": end - start,
            "observed_time_s": bucket["observed"],
            "missing_time_s": bucket["missing"],
            "unknown_time_s": bucket["unknown"],
            "center_time_s": bucket["center"],
            "arm_A_time_s": bucket["A"],
            "arm_B_time_s": bucket["B"],
            "arm_C_time_s": bucket["C"],
            "total_arm_entries": total,
            "arm_A_entries": bucket["entries"]["A"],
            "arm_B_entries": bucket["entries"]["B"],
            "arm_C_entries": bucket["entries"]["C"],
            "alternation_count": bucket["alternation"],
            "alternation_opportunities": bucket["opportunities"],
            "alternation_percent": _percent(bucket["alternation"],
                                            bucket["opportunities"]),
            "distance_cm": bucket["distance"],
            "measurable_motion_time_s": bucket["measurable"],
            "moving_time_s": bucket["moving"],
            "immobility_time_s": bucket["immobility"],
            "mean_speed_cm_s": (bucket["distance"] / bucket["measurable"]
                                if bucket["measurable"] > 0 else None),
        })
    return bins


# -- 单只动物 -----------------------------------------------------------

def process_unit(unit: Mapping[str, Any],
                 captured: Mapping[str, Any],
                 video_window: timeline.Window,
                 first_seen: float | None) -> dict:
    """估计坐标、分区占用、入臂事件、交替序列、汇总与分箱。"""

    unit_id = unit["id"]
    options = unit["options"]
    zones, frames, scale = _prepare(unit)

    window = timeline.resolve_unit_window(
        unit.get("window"), video_window, first_seen)
    rows = timeline.clip_rows(captured.get("rows") or [], window)
    method = (unit.get("detection") or {}).get("bg_method")
    for row in rows:
        row.setdefault("detection_method", method)
        _annotate(row, zones, scale)
    # 缺字段按原规则运行；只有显式 "center" 走中心点规则，其余非法值
    # 已在冻结校验拒绝，这里防御性地按原规则处理。
    entry_rule = options.get("arm_entry_rule") or "depth"
    tracking = (_track_center(rows, window, frames, options, unit)
                if entry_rule == "center"
                else _track(rows, window, frames, options, unit))
    rule_warning = None
    if entry_rule == "center":
        rule_warning = (f"「{unit.get('name') or unit_id}」使用中心点入臂规则："
                        "臂/中心边界反复切换会按观测转换计数（无时间防抖）；"
                        "入臂深度比例、入臂确认时间、中心区重置时间、"
                        "再次入臂须经过中心区在本规则下不生效。")
    segments = timeline.motion_segments(rows, window,
                                        x_key="x_cm", y_key="y_cm",
                                        require_continuity=True)
    timeline.attach_motion(rows, segments)
    cumulative = 0.0
    by_frame = {(segment["unit_id"], segment["frame"]): segment
                for segment in segments}
    for row in sorted(rows, key=lambda item: (item["time_s"], item["frame"])):
        segment = by_frame.get((row["unit_id"], row["frame"]))
        if segment is None:
            row["step_distance_cm"] = None
            row["speed_cm_s"] = None
        else:
            row["step_distance_cm"] = float(segment["distance"])
            row["speed_cm_s"] = float(segment["speed"])
            cumulative += float(segment["distance"])
        row["cumulative_distance_cm"] = cumulative

    summary = _summary(unit, rows, segments, window, options, scale,
                       tracking["events"])
    # 旧冻结任务缺字段时 summary 里是 None：覆盖为实际生效的规则。
    summary["arm_entry_rule"] = entry_rule
    bins = _bin_rows(rows, segments, tracking["events"], window, options, unit)
    candidates = _window_candidates(captured.get("candidates") or [], rows)
    metadata = {
        "mode": "automatic",
        "coordinate_unit": "cm_estimated",
        "scale": "mean of 6 drawn-size ratios (arm length / projection span, "
                 "arm width / drawn width); not a perspective correction",
        "cm_per_px_est": scale,
        "zone_priority": list(ZONE_KEYS),
        "candidate_link": "two accepted positions predict the next point; "
                          "candidates beyond max_jump_cm are rejected and the "
                          "nearest accepted",
        "area_reference": "median of the last 9 accepted areas and of the "
                          "first 9 (fixed once full); both within [1/4, 4]",
        "tracking_hold": "short gaps keep association evidence only; after "
                         "tracking_hold_s the link re-acquires by area",
        "entry_confirm": "same arm for entry_confirm_s of real source time",
        "center_reset": "armed again after entry_reset_center_s of center "
                        "observations; missing aborts the center timer",
        "alternation": "entry index >= 3 with three distinct arms in the "
                       "recent sequence",
        "missing_rule": "missing exposure is counted as missing only and no "
                        "held position is written; every association failure "
                        "reports no_candidate_within_gate, the raw detector "
                        "reason stays in the row field detection_reason",
    }
    # 规则事实进报告“计量 metadata”段：不同入臂规则的结果可区分。
    # 旧冻结任务缺字段时记录实际生效的规则（depth），不凭空改含义。
    metadata["arm_entry_rule"] = entry_rule
    if entry_rule == "center":
        metadata["arm_entry_rule_note"] = ("center-point rule: no entry depth, "
                                           "confirm time, center reset or "
                                           "require-center; no debounce on "
                                           "arm/center boundary flips")
    warning = rule_warning
    if not window.known:
        unknown_window = ("未检测到动物，无法确定相对时间窗："
                          + str(unit.get("name") or unit_id))
        warning = (f"{unknown_window}；{rule_warning}" if rule_warning
                   else unknown_window)
    return {"rows": rows, "segments": segments, "events": tracking["events"],
            "bins": bins, "summary": summary, "metadata": metadata,
            "candidates": candidates, "warning": warning}


def _window_candidates(candidates: Sequence[Mapping[str, Any]],
                       rows: Sequence[Mapping[str, Any]]) -> list[dict]:
    """候选表只保留窗口内的帧，与观测行一一对应。"""

    frames = {row["frame"] for row in rows if row.get("in_window")}
    return [dict(item) for item in candidates
            if item.get("frame") in frames]


# -- 捕获与分析入口 -----------------------------------------------------

def capture(context: Any, units: Sequence[Mapping[str, Any]]) -> dict:
    """逐帧候选 + 关联：正式分析与行为预览共用同一条捕获路径。"""

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

    prepared = {unit["id"]: _prepare(unit) for unit in units}
    links = {unit["id"]: CandidateLink(unit["options"], prepared[unit["id"]][2])
             for unit in units}
    detectors: dict[Any, Any] = {}
    regions: dict[Any, np.ndarray] = {}
    backgrounds: dict[tuple, Any] = {}
    captured: dict[Any, dict] = {
        unit["id"]: {"rows": [], "candidates": []} for unit in units}
    unit_ids = [unit["id"] for unit in units]
    run_key = (job.get("run_id"), job.get("job_id"), KEY)
    progress_span = float(metadata.get("duration_s") or 0.0)

    for frame in source.iter_frames():
        if context.cancelled():
            raise ValidationError("分析已取消", "job")
        image = frame.image
        if image is None:
            continue
        height, width = image.shape[:2]
        for unit in units:
            unit_id = unit["id"]
            settings = unit.get("detection") or {}
            if unit_id not in detectors:
                signature = tuple(settings.get(name) for name in
                                  detection.FOREGROUND_KEYS)
                if signature not in backgrounds:
                    backgrounds[signature] = detection.prepare_background(
                        settings, asset, job,
                        int(job.get("orientation", 0) or 0), width, height,
                        cancel=context.cancelled)
                detectors[unit_id] = detection.shared_detector(
                    run_key, settings, background=backgrounds[signature])
                regions[unit_id] = _region_for(unit, unit.get("options") or {},
                                               width, height)
            detector = detectors[unit_id]
            candidates, reason = detection.find_candidates(
                detector.foreground(image), regions[unit_id], settings)
            link = links[unit_id]
            selected = (link.pick(candidates, frame.time_s)
                        if candidates else None)
            entry = captured[unit_id]
            for index, item in enumerate(candidates, start=1):
                center = item["center"]
                entry["candidates"].append({
                    "unit_id": unit_id,
                    "unit_name": unit.get("name"),
                    "frame": frame.index,
                    "time_s": float(frame.time_s),
                    "candidate_index": index,
                    "x_px": float(center[0]),
                    "y_px": float(center[1]),
                    "quality": float(item["quality"]),
                    "area_px": float(item["area"]),
                    "contour": item["contour"],
                    "selected": item is selected,
                })
            if selected is not None:
                link.accept(selected, frame.time_s)
                center = selected["center"]
                result = Detection(
                    center=[float(center[0]), float(center[1])],
                    quality=float(selected["quality"]),
                    contour=selected["contour"], mask=selected["mask"],
                    area=float(selected["area"]), bbox=selected["bbox"],
                    candidates=candidates,
                )
            else:
                link.miss()
                # 自动 Y 没有接受候选就统一报关联失败；底层检测原因只作诊断，
                # 不覆盖这一对外枚举，也不保持、不外推位置。
                result = Detection(center=None, quality=0.0, contour=[],
                                   mask=None, missing_reason=MISSING_GATE,
                                   candidates=candidates)
            row = observation_row(unit, frame, result)
            row["detection_reason"] = reason
            entry["rows"].append(row)
        context.progress(min(0.9, 0.1 + 0.8 * max(0.0, float(frame.time_s)) /
                             max(1e-9, progress_span or 1.0)))
    return captured


def is_manual_unit(unit: Mapping[str, Any]) -> bool:
    """专用人工 Y 的判定：只有 unit.manual.mode == "manual" 才算。"""

    manual = unit.get("manual")
    return isinstance(manual, Mapping) and manual.get("mode") == "manual"


def validate_manual(unit: Mapping[str, Any], *,
                    options: Mapping[str, Any] | None = None,
                    complete: bool = False) -> list[dict]:
    """人工 Y：只校验时间箱、单位窗与有序非空入臂记录。

    几何、自动臂尺寸/追踪参数与检测参数在人工模式不参与运行，因此不校验。
    options 传合并后的有效参数（draft → video → unit），保证校验的就是运行
    时真正消费的 bin_seconds；不传时退回单位层。
    """

    issues: list[dict] = []
    if options is None:
        options = unit.get("options") or {}
    if isinstance(options, Mapping) and "bin_seconds" in options:
        width = _finite(options.get("bin_seconds"))
        if width is None:
            issues.append({"field": "options.bin_seconds",
                           "message": "必须是有限数值"})
        elif width <= 0:
            issues.append({"field": "options.bin_seconds",
                           "message": "必须大于 0"})

    window = unit.get("window")
    if isinstance(window, Mapping) and (window.get("mode") or "global") == "relative":
        issues.append({"field": "window.mode",
                       "message": "人工 Y 只支持跟随视频或绝对窗口"})

    manual = unit.get("manual")
    entries = manual.get("entries") if isinstance(manual, Mapping) else None
    if not isinstance(entries, list):
        issues.append({"field": "manual.entries", "message": "必须是数组"})
        return issues
    if not entries:
        # 清空后仍可保存草稿，开始分析时再要求至少一条。
        if complete:
            issues.append({"field": "manual.entries",
                           "message": "至少需要一条入臂记录"})
        return issues

    previous: float | None = None
    for index, entry in enumerate(entries):
        prefix = f"manual.entries[{index}]"
        if not isinstance(entry, Mapping):
            issues.append({"field": prefix, "message": "必须是对象"})
            continue
        raw = entry.get("time_s")
        if isinstance(raw, bool) or not is_number(raw) \
                or not math.isfinite(float(raw)):
            issues.append({"field": f"{prefix}.time_s",
                           "message": "必须是有限数值"})
            continue
        moment = float(raw)
        if moment < 0:
            issues.append({"field": f"{prefix}.time_s", "message": "必须 ≥ 0"})
        if previous is not None and moment <= previous:
            issues.append({"field": f"{prefix}.time_s",
                           "message": "入臂时间必须晚于上一条记录"})
        previous = moment
        if entry.get("arm") not in ARM_KEYS:
            issues.append({"field": f"{prefix}.arm",
                           "message": "必须是 A、B 或 C"})
    return issues


def _manual_entries(unit: Mapping[str, Any]) -> list[dict]:
    manual = unit.get("manual")
    raw = manual.get("entries") if isinstance(manual, Mapping) else None
    entries: list[dict] = []
    for item in raw or []:
        if not isinstance(item, Mapping):
            continue
        moment = _finite(item.get("time_s"))
        arm = item.get("arm")
        if moment is None or arm not in ARM_KEYS:
            continue
        entries.append({"time_s": moment, "arm": arm})
    return entries


def _manual_bins(unit: Mapping[str, Any], window: timeline.Window,
                 events: Sequence[Mapping[str, Any]],
                 width: float) -> list[dict]:
    """人工分箱：相对单位窗从 0 起，机会按全序列判定后入箱。"""

    if not window.known or window.duration_s <= 0:
        return []
    origin = float(window.start_s)
    duration = window.duration_s
    count = max(1, int(math.ceil((duration - 1e-12) / width)))
    buckets = [{"entries": {key: 0 for key in ARM_KEYS},
                "alternation": 0, "opportunities": 0}
               for _ in range(count)]
    for event in events:
        index = int(math.floor((float(event["time_s"]) - origin) / width
                               + 1e-9))
        if not 0 <= index < count:
            continue
        bucket = buckets[index]
        target = event.get("target")
        if target in bucket["entries"]:
            bucket["entries"][target] += 1
        if event.get("alternation_opportunity"):
            bucket["opportunities"] += 1
        if event.get("alternation"):
            bucket["alternation"] += 1

    bins: list[dict] = []
    for index, bucket in enumerate(buckets):
        start = index * width
        end = min(start + width, duration)
        total = sum(bucket["entries"].values())
        bins.append({
            "unit_id": unit.get("id"),
            "unit_name": unit.get("name"),
            "bin_index": index + 1,
            "start_s": start,
            "end_s": end,
            "duration_s": end - start,
            # 原文键：本箱入臂总数与交替数；下面两个是同一数值的兼容字段。
            "entries": total,
            "alternations": bucket["alternation"],
            "total_arm_entries": total,
            "arm_A_entries": bucket["entries"]["A"],
            "arm_B_entries": bucket["entries"]["B"],
            "arm_C_entries": bucket["entries"]["C"],
            "alternation_count": bucket["alternation"],
            "alternation_opportunities": bucket["opportunities"],
            "alternation_percent": _percent(bucket["alternation"],
                                            bucket["opportunities"]),
        })
    return bins


def _manual_facts(unit: Mapping[str, Any],
                  window: timeline.Window) -> dict:
    """人工入臂：只用已记录事件，不产生位置、占用、距离或速度。"""

    options = unit.get("options") or {}
    width = _finite(options.get("bin_seconds"))
    if width is None:
        width = float(DEFAULTS["bin_seconds"])
    entries = _manual_entries(unit)
    kept: list[dict] = []
    if window.known:
        low, high = float(window.start_s), float(window.end_s)
        kept = [entry for entry in entries if low <= entry["time_s"] < high]

    events: list[dict] = []
    sequence: list[str] = []
    counts = {key: 0 for key in ARM_KEYS}
    alternation = opportunities = 0
    for index, entry in enumerate(kept, start=1):
        arm = entry["arm"]
        sequence.append(arm)
        recent = sequence[-3:]
        opportunity = index >= 3
        is_alternation = opportunity and len(set(recent)) == 3
        counts[arm] += 1
        if opportunity:
            opportunities += 1
        if is_alternation:
            alternation += 1
        events.append({
            "event_id": f"{unit.get('id')}:arm_entry:{index}",
            "unit_id": unit.get("id"),
            "unit_name": unit.get("name"),
            "kind": "arm_entry",
            "time_s": entry["time_s"],
            "confirmed_s": entry["time_s"],
            "target": arm,
            "source": "manual",
            "entry_index": index,
            "recent_sequence": "-".join(recent),
            "alternation_opportunity": opportunity,
            "alternation": is_alternation,
        })

    total = len(kept)
    duration = window.duration_s if window.known else 0.0
    summary = {
        "unit_id": unit.get("id"),
        "unit_name": unit.get("name"),
        "mode": KEY,
        "analysis_window_start_s": window.start_s if window.known else None,
        "analysis_window_end_s": window.end_s if window.known else None,
        "total_time_s": duration,
        "total_arm_entries": total,
        "alternation_count": alternation,
        "alternation_opportunities": opportunities,
        "alternation_percent": _percent(alternation, opportunities),
        "arm_A_entries": counts["A"],
        "arm_B_entries": counts["B"],
        "arm_C_entries": counts["C"],
        "bin_seconds": width,
    }
    warning = None
    if window.known and total == 0:
        warning = "分析时间窗内没有人工入臂记录"
    elif not window.known:
        warning = ("未检测到动物，无法确定相对时间窗："
                   + str(unit.get("name") or unit.get("id")))
    metadata = {
        "mode": "manual",
        "event_time": "video absolute seconds",
        "position_available": False,
        "tracking_reliable": False,
    }
    return {"events": events,
            "bins": _manual_bins(unit, window, events, width),
            "summary": summary, "metadata": metadata, "warning": warning}


def _source_for(context: Any):
    """分析/预览共用的源；预览没有解码源时按任务配置构造。"""

    source = context.source
    if source is not None:
        return source
    job = context.job
    asset = job.get("asset") or {}
    metadata = asset.get("metadata") or {}
    window = job.get("window") or {}
    return media.VideoSource(
        asset.get("path"),
        start_s=float(window.get("start_s") or 0.0),
        end_s=window.get("end_s"),
        orientation=int(job.get("orientation", 0) or 0),
        fps_override=job.get("fps_override"),
        sample_fps=job.get("sample_fps"),
        source_fps=metadata.get("fps"),
        decode_images=True,
    )


def _decode_clock(context: Any) -> list[dict]:
    """纯人工：完整解码源帧只为取真实时钟与覆盖，不创建检测器。"""

    rows: list[dict] = []
    for frame in _source_for(context).iter_frames():
        if context.cancelled():
            raise ValidationError("分析已取消", "job")
        rows.append({
            "unit_id": "",
            "frame": frame.index,
            "time_s": frame.time_s,
            "source_interval_start_s": frame.interval_start_s,
            "source_interval_end_s": frame.interval_end_s,
            "interval_start_s": frame.interval_start_s,
            "interval_end_s": frame.interval_end_s,
            "dt_s": max(0.0, float(frame.interval_end_s)
                        - float(frame.interval_start_s)),
        })
    return rows


def analyze(context: Any) -> AnalysisResult:
    job = context.job
    units = list(job.get("units") or [])
    if not units:
        raise ValidationError("草稿还没有小鼠", "units")
    manual_units = [unit for unit in units if is_manual_unit(unit)]
    auto_units = [unit for unit in units if not is_manual_unit(unit)]
    for unit in auto_units:
        issues = validate_unit(unit.get("options") or {},
                               unit.get("geometry") or {}, complete=True)
        if issues:
            raise ValidationError(issues[0]["message"], issues[0]["field"])
    for unit in manual_units:
        issues = validate_manual(unit, complete=True)
        if issues:
            raise ValidationError(issues[0]["message"], issues[0]["field"])

    # 纯人工不创建检测器、不读背景；混合时只把自动单位交给检测。
    captured = capture(context, auto_units) if auto_units else {}
    if context.cancelled():
        raise ValidationError("分析已取消", "job")
    asset = job.get("asset") or {}
    metadata = asset.get("metadata") or {}
    if captured:
        clock_rows = [row for facts in captured.values()
                      for row in facts["rows"]]
    elif manual_units:
        clock_rows = _decode_clock(context)
    else:
        clock_rows = []
    video_window = timeline.resolve_video_window(
        job.get("window"), clock_rows, metadata.get("duration_s"))
    first_seen = {
        unit["id"]: timeline.first_seen_in_window(
            KEY, captured[unit["id"]]["rows"], video_window,
            unit.get("geometry") or {})
        for unit in auto_units
    }

    observations: list[dict] = []
    summary: list[dict] = []
    events: list[dict] = []
    bins: list[dict] = []
    candidates: list[dict] = []
    warnings: list[str] = []
    unit_metadata: dict[str, dict] = {}
    for unit in units:
        if context.cancelled():
            raise ValidationError("分析已取消", "job")
        if is_manual_unit(unit):
            window = timeline.resolve_unit_window(unit.get("window"),
                                                  video_window, None)
            facts = _manual_facts(unit, window)
            summary.append(facts["summary"])
            events.extend(facts["events"])
            bins.extend(facts["bins"])
            unit_metadata[unit["id"]] = facts["metadata"]
            if facts["warning"]:
                warnings.append(facts["warning"])
            continue
        facts = process_unit(unit, captured[unit["id"]], video_window,
                             first_seen[unit["id"]])
        observations.extend(facts["rows"])
        summary.append(facts["summary"])
        events.extend(facts["events"])
        bins.extend(facts["bins"])
        candidates.extend(facts["candidates"])
        unit_metadata[unit["id"]] = facts["metadata"]
        if facts["warning"]:
            warnings.append(facts["warning"])

    return AnalysisResult(
        observations=timeline.sort_rows(observations),
        summary=summary,
        events=events,
        bins=bins,
        tables={"candidates": candidates, "arm_entries": events},
        warnings=warnings,
        metadata={"assay": KEY, "coordinate_unit": "cm_estimated",
                  "units": unit_metadata, "warnings": warnings},
    )


def _manual_preview(*, asset: Mapping[str, Any], video: Mapping[str, Any],
                    unit: Mapping[str, Any], start_s: float,
                    end_s: float) -> dict:
    """纯人工预览：只迭代源时钟，不创建检测器也不解码画面。"""

    metadata = asset.get("metadata") or {}
    source = media.VideoSource(
        asset.get("path"),
        start_s=float(start_s),
        end_s=float(end_s),
        orientation=int(video.get("orientation", 0) or 0),
        fps_override=video.get("fps_override"),
        sample_fps=video.get("sample_fps"),
        source_fps=metadata.get("fps"),
        decode_images=False,
    )
    rows = [{"source_interval_end_s": frame.interval_end_s}
            for frame in source.iter_frames()]
    video_window = timeline.resolve_video_window(
        {"start_s": float(start_s), "end_s": float(end_s)}, rows,
        metadata.get("duration_s"))
    window = timeline.resolve_unit_window(unit.get("window"), video_window,
                                          None)
    facts = _manual_facts(unit, window)
    summary = facts["summary"]
    return {
        "mode": KEY,
        "manual": True,
        "start_s": float(start_s),
        "end_s": float(end_s),
        "window_start_s": summary["analysis_window_start_s"],
        "window_end_s": summary["analysis_window_end_s"],
        "total_time_s": summary["total_time_s"],
        "total_arm_entries": summary["total_arm_entries"],
        "arm_A_entries": summary["arm_A_entries"],
        "arm_B_entries": summary["arm_B_entries"],
        "arm_C_entries": summary["arm_C_entries"],
        "alternation_count": summary["alternation_count"],
        "alternation_opportunities": summary["alternation_opportunities"],
        "alternation_percent": summary["alternation_percent"],
        "bin_seconds": summary["bin_seconds"],
        "rows": 0,
        "events": len(facts["events"]),
        "bins": len(facts["bins"]),
    }


def preview_analysis(*, asset: Mapping[str, Any], video: Mapping[str, Any],
                     unit: Mapping[str, Any], start_s: float,
                     end_s: float, time_s: float | None = None) -> dict:
    """行为预览：人工只走时钟；自动仍跑同一捕获→处理核心。"""

    from ..schema import AnalysisContext, new_id

    options = unit.get("options") or {}
    if is_manual_unit(unit):
        issues = validate_manual(unit, complete=True)
        if issues:
            raise ValidationError(issues[0]["message"], issues[0]["field"])
        return _manual_preview(asset=asset, video=video, unit=unit,
                               start_s=start_s, end_s=end_s)

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
    context = AnalysisContext(job=job, source=source)
    captured = capture(context, [unit])[unit["id"]]
    video_window = timeline.resolve_video_window(
        window, captured["rows"], metadata.get("duration_s"))
    first_seen = timeline.first_seen_in_window(
        KEY, captured["rows"], video_window, unit.get("geometry") or {})
    facts = process_unit(unit, captured, video_window, first_seen)
    summary = facts["summary"]
    return {
        "mode": KEY,
        "start_s": float(start_s),
        "end_s": float(end_s),
        "window_start_s": summary["analysis_window_start_s"],
        "window_end_s": summary["analysis_window_end_s"],
        "total_time_s": summary["total_time_s"],
        "observed_time_s": summary["observed_time_s"],
        "missing_time_s": summary["missing_time_s"],
        "center_time_s": summary["center_time_s"],
        "arm_A_time_s": summary["arm_A_time_s"],
        "arm_B_time_s": summary["arm_B_time_s"],
        "arm_C_time_s": summary["arm_C_time_s"],
        "total_arm_entries": summary["total_arm_entries"],
        "arm_A_entries": summary["arm_A_entries"],
        "arm_B_entries": summary["arm_B_entries"],
        "arm_C_entries": summary["arm_C_entries"],
        "arm_entry_rule": summary["arm_entry_rule"],
        "alternation_count": summary["alternation_count"],
        "alternation_opportunities": summary["alternation_opportunities"],
        "alternation_percent": summary["alternation_percent"],
        "total_distance_cm": summary["total_distance_cm"],
        "mean_speed_cm_s": summary["mean_speed_cm_s"],
        "cm_per_px_est": summary["cm_per_px_est"],
        "rows": len(facts["rows"]),
        "events": len(facts["events"]),
        "bins": len(facts["bins"]),
    }


def validate(job: Any) -> list[dict]:
    issues: list[dict] = []
    for unit in job.get("units") or []:
        prefix = f"units[{unit.get('id')}]"
        found = (validate_manual(unit, complete=True) if is_manual_unit(unit)
                 else validate_unit(unit.get("options") or {},
                                    unit.get("geometry") or {}, complete=True))
        for issue in found:
            entry = dict(issue)
            entry["field"] = f"{prefix}.{entry['field']}"
            issues.append(entry)
    return issues
