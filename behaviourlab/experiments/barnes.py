"""Barnes 迷宫：椭圆圆盘归一化、洞访问、目标扇区与覆盖。

同一批源观测先落成事实行（物理坐标、身体/鼻尖资格、洞归属、目标扇区），
访问、逃避候选、路径、分箱与导出都从这份事实派生；结果阶段不再重新探洞。
洞访问默认来自自动鼻尖证据，人工覆盖留给后续块，二者不混合计数。
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .. import detection, geometry as geo, media, observation, timeline
from ..schema import AnalysisResult, ValidationError, is_number
from .base import geometry_field, metric, parameter

KEY = "barnes"

DEFAULTS = {
    "disc_diameter_cm": 90.0,
    "hole_count": 20,
    "hole_radius_cm": 2.0,
    "target_hole": "1",
    "trial_type": "training",
    "visit_confirm_s": 0.1,
    "visit_reset_s": 0.1,
    "bin_seconds": 60.0,
    "quality_threshold": 0.4,
}

# 实验级检测覆盖：背景方法与腐蚀按巴恩斯场景固定，其余继承共享默认。
DETECTION_DEFAULTS = {
    "bg_method": "local_otsu",
    "erosion_kernel": 3,
    "erosion_iter": 1,
    "quality_threshold": 0.4,
}

OPTION_KEYS = tuple(DEFAULTS)
TRIAL_TYPES = ("training", "reversal", "probe")
POSITIVE_KEYS = ("disc_diameter_cm", "hole_radius_cm", "visit_confirm_s",
                 "visit_reset_s", "bin_seconds")
# 相邻曝光端点相接的容差；确认阈值另用更紧的比较容差。
CONTINUITY_EPS = 1e-7
CONFIRM_EPS = 1e-9

SPEC = {
    "key": KEY,
    "label": "Barnes 迷宫",
    "short": "Barnes",
    "parameters": [
        parameter("disc_diameter_cm", "圆盘直径", "number", 90.0,
                  unit="cm", minimum=0, step="any"),
        parameter("hole_count", "洞数", "number", 20, unit="个",
                  minimum=2, maximum=100, step=1),
        parameter("hole_radius_cm", "洞半径", "number", 2.0,
                  unit="cm", minimum=0, step="any"),
        parameter("target_hole", "目标洞", "select", "1", choices=[]),
        parameter("trial_type", "试次", "select", "training",
                  choices=[{"value": "training", "label": "训练"},
                           {"value": "reversal", "label": "反转"},
                           {"value": "probe", "label": "探测"}]),
        parameter("visit_confirm_s", "访问确认时间", "number", 0.1,
                  unit="s", minimum=0, step="any", section="统计"),
        parameter("visit_reset_s", "洞外重置时间", "number", 0.1,
                  unit="s", minimum=0, step="any", section="统计"),
        parameter("quality_threshold", "统计质量阈值", "number", 0.4,
                  minimum=0, maximum=1, step="any", section="统计"),
        parameter("bin_seconds", "时间箱长度", "number", 60.0,
                  unit="s", minimum=0, step="any", section="统计"),
    ],
    "geometry": [
        geometry_field("arena", "圆盘外缘", "ellipse", min_points=5),
        geometry_field("holes", "洞位", "holes", count_key="hole_count"),
        geometry_field("mask", "检测遮罩", "polygon", min_points=3,
                       optional=True),
    ],
    "metrics": [
        metric("visits", "洞访问", "次"),
        metric("primary_latency_s", "首次目标潜伏期", "s"),
        metric("primary_errors", "首次目标前错误", "次"),
        metric("primary_path_cm", "首次目标路径", "cm"),
        metric("total_distance_cm", "总距离", "cm"),
    ],
    "template_keys": ["disc_diameter_cm", "hole_count", "hole_radius_cm"],
}


# -- 解析与校验 ---------------------------------------------------------

def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not is_number(value):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _pair(value: Any) -> tuple[float, float] | None:
    """源像素点/半轴必须恰好两个有限数；多余分量与布尔值都算非法。"""

    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    x, y = _finite(value[0]), _finite(value[1])
    if x is None or y is None:
        return None
    return x, y


def _shape_points(shape: Any) -> list[tuple[float, float]]:
    if not isinstance(shape, Mapping):
        return []
    points: list[tuple[float, float]] = []
    for item in shape.get("points") or []:
        pair = _pair(item)
        if pair is None:
            return []
        points.append(pair)
    return points


def ellipse_params(geometry: Mapping[str, Any] | None
                   ) -> tuple[list[float], list[float], float] | None:
    """圆盘椭圆 (center, radii, angle)；支持中心+半轴或 ≥5 个边缘点。"""

    arena = (geometry or {}).get("arena")
    if not isinstance(arena, Mapping) or arena.get("type") != "ellipse":
        return None
    center = arena.get("center")
    radii = arena.get("radii")
    if isinstance(center, (list, tuple)) and isinstance(radii, (list, tuple)):
        # 中心与半轴都必须是恰好两个有限数，多余分量不做截断。
        if len(center) != 2 or len(radii) != 2:
            return None
        cx, cy = _finite(center[0]), _finite(center[1])
        rx, ry = _finite(radii[0]), _finite(radii[1])
        angle = _finite(arena.get("angle", 0.0))
        if None in (cx, cy, rx, ry, angle) or rx <= 0 or ry <= 0:
            return None
        return [cx, cy], [rx, ry], float(angle)
    points = _shape_points(arena)
    if len(points) < 5:
        return None
    try:
        fitted = geo.fit_ellipse(points)
    except ValidationError:
        return None
    if fitted["radii"][0] <= 0 or fitted["radii"][1] <= 0:
        return None
    return ([float(fitted["center"][0]), float(fitted["center"][1])],
            [float(fitted["radii"][0]), float(fitted["radii"][1])],
            float(fitted["angle"]))


def holes_of(geometry: Mapping[str, Any] | None,
             count: Any = None) -> tuple[list[dict], list[dict]]:
    """已标洞位与问题；保留标记顺序，洞号一律按字符串比较。"""

    raw = (geometry or {}).get("holes")
    if raw is None:
        return [], []
    if not isinstance(raw, list):
        return [], [{"field": "geometry.holes", "message": "洞位必须是数组"}]

    issues: list[dict] = []
    entries: list[dict] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        field = f"geometry.holes[{index}]"
        if not isinstance(item, Mapping):
            issues.append({"field": field, "message": "洞必须是对象"})
            continue
        raw_id = item.get("id")
        if isinstance(raw_id, bool) or raw_id is None \
                or not str(raw_id).strip():
            issues.append({"field": f"{field}.id", "message": "洞号不能为空"})
            continue
        hole_id = str(raw_id)
        if hole_id in seen:
            issues.append({"field": f"{field}.id",
                           "message": f"洞号 {hole_id} 重复"})
            continue
        seen.add(hole_id)
        center = _pair(item.get("center"))
        if center is None:
            issues.append({"field": f"{field}.center",
                           "message": "洞心需要两个有限数值"})
            continue
        entries.append({"id": hole_id, "center": center})

    if isinstance(count, int) and not isinstance(count, bool) \
            and len(raw) != count:
        issues.append({
            "field": "geometry.holes",
            "message": f"洞位数必须与洞数一致（当前 {len(raw)} / 设置 {count}）",
        })
    return entries, issues


def validate_geometry(geometry: Mapping[str, Any] | None,
                      options: Mapping[str, Any] | None = None, *,
                      require: bool = False,
                      target: str | None = None) -> list[dict]:
    geometry = geometry or {}
    options = options if isinstance(options, Mapping) else {}
    issues: list[dict] = []

    arena = geometry.get("arena")
    if arena is None:
        if require:
            issues.append({"field": "geometry.arena", "message": "缺少圆盘外缘"})
    elif not isinstance(arena, Mapping) or arena.get("type") != "ellipse":
        issues.append({"field": "geometry.arena",
                       "message": "圆盘外缘必须是椭圆"})
    elif ellipse_params(geometry) is None:
        issues.append({"field": "geometry.arena",
                       "message": "圆盘中心与半轴必须是两个有限数，"
                                  "或给至少 5 个二维边缘点"})

    holes, hole_issues = holes_of(geometry, options.get("hole_count"))
    issues.extend(hole_issues)
    raw_holes = geometry.get("holes")
    if raw_holes is None:
        if require:
            issues.append({"field": "geometry.holes", "message": "缺少洞位"})
    elif isinstance(raw_holes, list) and holes and len(holes) == len(raw_holes) \
            and target is not None \
            and target not in {item["id"] for item in holes}:
        # 洞位本身有错时只报洞位，不追加目标洞的连带错误。
        issues.append({"field": "options.target_hole",
                       "message": f"目标洞 {target} 不在已标洞号中"})

    mask = geometry.get("mask")
    if mask is not None:
        issues.extend(mask_issues(mask))
    return issues


def mask_issues(mask: Any) -> list[dict]:
    """可选检测遮罩：本块只接受合法 polygon（二维、正面积、不自交）。"""

    if not isinstance(mask, Mapping) or mask.get("type") != "polygon":
        return [{"field": "geometry.mask", "message": "检测遮罩必须是 polygon"}]
    points = _shape_points(mask)
    if len(points) < 3:
        return [{"field": "geometry.mask",
                 "message": "检测遮罩需要至少 3 个有限二维点"}]
    if abs(_polygon_area(points)) <= 1e-9:
        return [{"field": "geometry.mask",
                 "message": "检测遮罩面积必须大于 0"}]
    if geo.polygon_self_intersects(points):
        return [{"field": "geometry.mask", "message": "检测遮罩不能自交"}]
    return []


def _polygon_area(points: Sequence[Sequence[float]]) -> float:
    total = 0.0
    for index, point in enumerate(points):
        nxt = points[(index + 1) % len(points)]
        total += float(point[0]) * float(nxt[1]) - float(nxt[0]) * float(point[1])
    return total / 2.0


def validate_unit(options: Mapping[str, Any] | None,
                  geometry: Mapping[str, Any] | None,
                  provided: Iterable[str] | None = None, *,
                  complete: bool = False) -> list[dict]:
    """只校验显式参数与已画区域；未完成草稿仍可保存。"""

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

    for key in POSITIVE_KEYS:
        value = number(key)
        if value is not None and value <= 0:
            issues.append({"field": f"options.{key}", "message": "必须大于 0"})

    if "hole_count" in provided:
        count = values.get("hole_count")
        if isinstance(count, bool) or not isinstance(count, int):
            issues.append({"field": "options.hole_count",
                           "message": "洞数必须是 2–100 的整数"})
        elif not 2 <= count <= 100:
            issues.append({"field": "options.hole_count",
                           "message": "洞数必须在 2–100 之间"})

    if "quality_threshold" in provided:
        value = number("quality_threshold")
        if value is not None and not 0.0 <= value <= 1.0:
            issues.append({"field": "options.quality_threshold",
                           "message": "统计质量阈值必须在 0–1 之间"})

    if "trial_type" in provided and values.get("trial_type") not in TRIAL_TYPES:
        issues.append({"field": "options.trial_type",
                       "message": "试次必须是 training、reversal 或 probe"})

    raw_target = values.get("target_hole")
    target = None
    if "target_hole" in provided:
        if isinstance(raw_target, bool) or raw_target is None \
                or not isinstance(raw_target, (str, int, float)):
            issues.append({"field": "options.target_hole",
                           "message": "目标洞必须是洞号"})
        elif not str(raw_target).strip():
            issues.append({"field": "options.target_hole",
                           "message": "目标洞不能为空"})
    if raw_target is not None and not isinstance(raw_target, bool) \
            and isinstance(raw_target, (str, int, float)):
        target = str(raw_target).strip() or None

    issues.extend(validate_geometry(geometry, values, require=complete,
                                    target=target))
    return issues


def target_hole_id(options: Mapping[str, Any]) -> str:
    raw = options.get("target_hole")
    if isinstance(raw, bool) or raw is None:
        raise ValidationError("目标洞必须是洞号", "options.target_hole")
    value = str(raw).strip()
    if not value:
        raise ValidationError("目标洞不能为空", "options.target_hole")
    return value


def validate_unit_manual(unit: Mapping[str, Any], *,
                         options: Mapping[str, Any] | None = None,
                         complete: bool = False) -> list[dict]:
    """人工洞序列与确认入箱的结构校验；与动物窗相关的检查留到运行期。"""

    manual = unit.get("manual")
    if manual is None:
        return []
    if not isinstance(manual, Mapping):
        return [{"field": "manual", "message": "必须是对象或 null"}]
    values = options if isinstance(options, Mapping) else {}
    geometry = unit.get("geometry") or {}
    holes, _ = holes_of(geometry, values.get("hole_count"))
    raw_holes = geometry.get("holes")
    # 洞位本身有错时只报洞位，不对洞号做连带判断。
    ids = ({item["id"] for item in holes}
           if isinstance(raw_holes, list) and holes
           and len(holes) == len(raw_holes) else None)

    issues: list[dict] = []
    visits = manual.get("hole_visits")
    if visits is not None:
        if not isinstance(visits, list):
            issues.append({"field": "manual.hole_visits",
                           "message": "必须是数组或 null"})
        else:
            previous: float | None = None
            for index, item in enumerate(visits):
                field = f"manual.hole_visits[{index}]"
                if not isinstance(item, Mapping):
                    issues.append({"field": field,
                                   "message": "每条访问必须是 {time_s, hole_id}"})
                    continue
                seconds = _finite(item.get("time_s"))
                if seconds is None or seconds < 0:
                    issues.append({"field": f"{field}.time_s",
                                   "message": "时间必须是非负的有限秒数"})
                elif previous is not None and seconds < previous - 1e-12:
                    issues.append({"field": f"{field}.time_s",
                                   "message": "访问时间必须非递减"})
                else:
                    previous = seconds
                raw_id = item.get("hole_id")
                if isinstance(raw_id, bool) or raw_id is None \
                        or not str(raw_id).strip():
                    issues.append({"field": f"{field}.hole_id",
                                   "message": "洞号不能为空"})
                elif ids is not None and str(raw_id) not in ids:
                    issues.append({"field": f"{field}.hole_id",
                                   "message": f"洞号 {raw_id} 不在已标洞号中"})

    escape = manual.get("escape_time_s")
    if escape is not None:
        if _finite(escape) is None or float(escape) < 0:
            issues.append({"field": "manual.escape_time_s",
                           "message": "必须是非负的有限秒数或 null"})
        elif values.get("trial_type") == "probe":
            issues.append({"field": "manual.escape_time_s",
                           "message": "探测试次不接受确认入箱时间"})
    return issues


def require_geometry(geometry: Mapping[str, Any] | None,
                     options: Mapping[str, Any]) -> tuple[
        tuple[list[float], list[float], float], list[dict]]:
    """分析与预览共用的几何解析；不合法即抛字段错误。"""

    ellipse = ellipse_params(geometry)
    if ellipse is None:
        raise ValidationError("圆盘需要两个正半轴，或至少 5 个有限边缘点",
                              "geometry.arena")
    holes, issues = holes_of(geometry, options.get("hole_count"))
    if issues:
        raise ValidationError(issues[0]["message"], issues[0]["field"])
    if not holes:
        raise ValidationError("缺少洞位", "geometry.holes")
    return ellipse, holes


# -- 像素 → 物理坐标 ----------------------------------------------------

def affine_matrix(ellipse: tuple[list[float], list[float], float],
                  diameter_cm: float) -> np.ndarray:
    """A = R·diag(D/2rx, D/2ry)·Rᵀ；保持画面轴向，图像 y 向下。"""

    _center, radii, angle = ellipse
    rx, ry = float(radii[0]), float(radii[1])
    a = float(diameter_cm) / (2.0 * rx)
    b = float(diameter_cm) / (2.0 * ry)
    theta = math.radians(float(angle))
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    rotate = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)
    return rotate @ np.diag([a, b]) @ rotate.T


def to_cm(matrix: np.ndarray, center: Sequence[float],
          point: Sequence[float]) -> tuple[float, float]:
    vector = np.array([float(point[0]) - float(center[0]),
                       float(point[1]) - float(center[1])], dtype=np.float64)
    result = matrix @ vector
    return float(result[0]), float(result[1])


def radius_cm(point: Sequence[float]) -> float:
    return math.hypot(float(point[0]), float(point[1]))


def angular_gap(first: float, second: float) -> float:
    """两个角度之间的最短环形差，范围 [0, π]。"""

    delta = (float(first) - float(second) + math.pi) % (2.0 * math.pi) - math.pi
    return abs(delta)


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
    region, _ = detection.geometry_region(unit.get("geometry") or {}, KEY,
                                          options, width, height)
    return region


def _annotate(row: dict, matrix: np.ndarray, center: Sequence[float],
              holes: Sequence[Mapping[str, Any]],
              hole_points: Sequence[Sequence[float]],
              options: Mapping[str, Any], state: dict) -> None:
    """一行事实：物理坐标、身体/鼻尖资格、洞归属、目标扇区与运动坐标。"""

    threshold = float(options["quality_threshold"])
    row["coordinate_unit"] = "cm"
    if row.get("detected"):
        state["held_x_px"] = row.get("x_px")
        state["held_y_px"] = row.get("y_px")
    row["held_x_px"] = state.get("held_x_px")
    row["held_y_px"] = state.get("held_y_px")

    body_point = None
    if row.get("detected") and _finite(row.get("x_px")) is not None \
            and _finite(row.get("y_px")) is not None:
        body_point = to_cm(matrix, center, (row["x_px"], row["y_px"]))
    if body_point is None:
        row["x_cm"] = row["y_cm"] = None
    else:
        row["x_cm"], row["y_cm"] = body_point
    row["x"] = row["x_cm"]
    row["y"] = row["y_cm"]

    for prefix, x_key, y_key in (("nose", "nose_x_px", "nose_y_px"),
                                 ("tail", "tail_x_px", "tail_y_px")):
        px, py = row.get(x_key), row.get(y_key)
        if body_point is not None and _finite(px) is not None \
                and _finite(py) is not None:
            row[f"{prefix}_x_cm"], row[f"{prefix}_y_cm"] = to_cm(
                matrix, center, (px, py))
        else:
            row[f"{prefix}_x_cm"] = None
            row[f"{prefix}_y_cm"] = None

    body_qualified = bool(
        body_point is not None
        and float(row.get("quality") or 0.0) >= threshold)
    row["body_qualified"] = body_qualified
    row["body_radius_cm"] = radius_cm(body_point) if body_point else None
    limit = float(options["disc_diameter_cm"]) / 2.0
    inside = bool(body_qualified and row["body_radius_cm"] <= limit + 1e-8)
    row["inside_arena"] = inside

    nose_qualified = bool(
        row["nose_x_cm"] is not None
        and float(row.get("head_quality") or 0.0) >= threshold)
    row["nose_qualified"] = nose_qualified
    row["nose_hole"] = None
    row["nose_hole_distance_cm"] = None
    if nose_qualified:
        nose_point = (row["nose_x_cm"], row["nose_y_cm"])
        best_index = None
        best_distance = None
        for index, point in enumerate(hole_points):
            distance = math.hypot(nose_point[0] - point[0],
                                  nose_point[1] - point[1])
            if best_distance is None or distance < best_distance - 1e-12:
                best_index, best_distance = index, distance
        if best_index is not None and best_distance is not None \
                and best_distance <= float(options["hole_radius_cm"]):
            row["nose_hole"] = holes[best_index]["id"]
            row["nose_hole_distance_cm"] = best_distance

    sector = False
    if inside:
        half = math.pi / float(options["hole_count"])
        angle = math.atan2(row["y_cm"], row["x_cm"])
        sector = angular_gap(angle, state["target_angle"]) <= half + 1e-12
    row["target_sector"] = sector

    if inside:
        row["motion_x"], row["motion_y"] = row["x_cm"], row["y_cm"]
    else:
        row["motion_x"], row["motion_y"] = None, None


# -- 洞访问状态机 -------------------------------------------------------

class _HoleState:
    """每个洞独立的进入/退出候选；确认只由更晚的真实观测触发。"""

    def __init__(self, hole_id: str) -> None:
        self.id = hole_id
        self.active = False
        self.inside_start: float | None = None
        self.outside_start: float | None = None

    def reset_pending(self) -> None:
        self.inside_start = None
        self.outside_start = None


def _visits_for(rows: Sequence[Mapping[str, Any]],
                holes: Sequence[Mapping[str, Any]],
                options: Mapping[str, Any]) -> list[dict]:
    """自动洞访问：进入需更晚观测确认，持续同洞不重复，缺失不算退出。"""

    confirm = float(options["visit_confirm_s"])
    reset = float(options["visit_reset_s"])
    states = [_HoleState(hole["id"]) for hole in holes]
    visits: list[dict] = []
    previous: Mapping[str, Any] | None = None

    for row in sorted((item for item in rows if item.get("in_window")),
                      key=lambda item: (item["time_s"], item["frame"])):
        continuous = (previous is not None
                      and abs(float(previous["interval_end_s"])
                              - float(row["interval_start_s"]))
                      <= CONTINUITY_EPS)
        if not continuous:
            for state in states:
                state.reset_pending()
        previous = row

        if not row.get("nose_qualified"):
            # 鼻尖缺失只清候选计时，不把缺失当成退出。
            for state in states:
                state.reset_pending()
            continue

        instant = max(float(row["time_s"]), float(row["interval_start_s"]))
        current = row.get("nose_hole")
        for state in states:
            if state.id == current:
                state.outside_start = None
                if state.active:
                    continue
                if state.inside_start is None:
                    # 第一条只建立候选：一条长曝光不能自己确认。
                    state.inside_start = instant
                    continue
                if instant - state.inside_start >= confirm - CONFIRM_EPS:
                    visits.append({
                        "kind": "hole_visit",
                        "hole_id": state.id,
                        "target": state.id,
                        "time_s": state.inside_start,
                        "confirmed_s": instant,
                        "source": "automatic",
                        "frame": row["frame"],
                    })
                    state.active = True
                    state.inside_start = None
                continue

            if not state.active:
                # 合格鼻尖真实离开该洞：未确认的进入候选立即作废，不能跨洞或
                # 跨一段洞外时间继续累计。
                state.reset_pending()
                continue
            if state.outside_start is None:
                state.outside_start = instant
            elif instant - state.outside_start >= reset - CONFIRM_EPS:
                state.active = False
                state.outside_start = None
    return visits


# -- 逃避候选 -----------------------------------------------------------

def escape_candidates(rows: Sequence[Mapping[str, Any]],
                      target_point: Sequence[float],
                      hole_radius_cm: float,
                      target: str) -> list[dict]:
    """目标附近身体消失：只记未确认候选，不产生总逃避潜伏期。"""

    events: list[dict] = []
    previous: tuple[float, float] | None = None
    for row in sorted((item for item in rows if item.get("in_window")),
                      key=lambda item: (item["time_s"], item["frame"])):
        qualified = bool(row.get("body_qualified"))
        if previous is not None and not qualified:
            near = math.hypot(previous[0] - float(target_point[0]),
                              previous[1] - float(target_point[1]))
            if near <= 2.0 * float(hole_radius_cm) + 1e-12:
                events.append({
                    "kind": "escape_candidate",
                    "target": target,
                    "time_s": float(row["interval_start_s"]),
                    "confirmed_s": None,
                    "source": "automatic",
                    "frame": row["frame"],
                    "basis": "目标附近身体消失，未确认",
                })
            previous = None
        if qualified:
            previous = (float(row["x_cm"]), float(row["y_cm"]))
    return events


# -- 评分 ---------------------------------------------------------------

def _scored(visits: list[dict], target: str,
            escape: float | None) -> dict:
    """按人工逃避时刻给访问打 scored，再从已评分序列取首次目标。"""

    for visit in visits:
        start, confirmed = visit.get("time_s"), visit.get("confirmed_s")
        if escape is None:
            visit["scored"] = start is not None and confirmed is not None
        else:
            visit["scored"] = bool(
                start is not None and confirmed is not None
                and float(start) < float(escape)
                and float(confirmed) <= float(escape))

    scored = [item for item in visits if item["scored"]]
    primary_index = next((index for index, item in enumerate(scored)
                          if item["target"] == target), None)
    primary = scored[primary_index] if primary_index is not None else None
    errors = (len([item for item in scored[:primary_index]
                   if item["target"] != target])
              if primary_index is not None
              else len([item for item in scored if item["target"] != target]))
    return {"scored": scored, "primary": primary, "errors": errors,
            "censored": primary is None}


def _path_until(segments: Sequence[Mapping[str, Any]], cutoff: float,
                origin: float) -> float:
    """只累计与截止时刻交叠的那部分路径，不计整条跨截止段。"""

    total = 0.0
    for segment in segments:
        start = max(float(segment["start_s"]), float(origin))
        end = min(float(segment["end_s"]), float(cutoff))
        span = float(segment["end_s"]) - float(segment["start_s"])
        if end <= start or span <= 0:
            continue
        total += float(segment["distance"]) * (end - start) / span
    return total


# -- 汇总与分箱 ---------------------------------------------------------

def _percent(numerator: float, denominator: float) -> float | None:
    return timeline.ratio(numerator * 100.0, denominator)


def _bin_rows(rows: Sequence[Mapping[str, Any]],
              segments: Sequence[Mapping[str, Any]], window: timeline.Window,
              options: Mapping[str, Any], unit: Mapping[str, Any]) -> list[dict]:
    if not window.known or window.duration_s <= 0:
        return []
    width = float(options["bin_seconds"])
    origin = float(window.start_s)
    duration = window.duration_s
    count = max(1, int(math.ceil((duration - 1e-12) / width)))

    buckets = [{"observed": 0.0, "missing": 0.0, "body": 0.0,
                "nose": 0.0, "sector": 0.0, "distance": 0.0,
                "measurable": 0.0} for _ in range(count)]

    for row in rows:
        if not row.get("in_window"):
            continue
        for index, low, high in timeline.split_interval(
                row["interval_start_s"], row["interval_end_s"], origin, width):
            if not 0 <= index < count:
                continue
            bucket = buckets[index]
            span = high - low
            if row.get("detected"):
                bucket["observed"] += span
            else:
                bucket["missing"] += span
            if row.get("body_qualified"):
                bucket["body"] += span
            if row.get("nose_qualified"):
                bucket["nose"] += span
            if row.get("target_sector"):
                bucket["sector"] += span

    for segment in segments:
        span = float(segment["end_s"]) - float(segment["start_s"])
        if span <= 0:
            continue
        for index, low, high in timeline.split_interval(
                segment["start_s"], segment["end_s"], origin, width):
            if not 0 <= index < count:
                continue
            share = (high - low) / span
            buckets[index]["distance"] += float(segment["distance"]) * share
            buckets[index]["measurable"] += high - low

    bins: list[dict] = []
    for index, bucket in enumerate(buckets):
        start = index * width
        end = min(start + width, duration)
        bins.append({
            "unit_id": unit.get("id"),
            "unit_name": unit.get("name"),
            "bin_index": index + 1,
            "start_s": start,
            "end_s": end,
            "duration_s": end - start,
            "observed_time_s": bucket["observed"],
            "missing_time_s": bucket["missing"],
            "body_time_s": bucket["body"],
            "nose_time_s": bucket["nose"],
            "target_sector_time_s": bucket["sector"],
            "distance_cm": bucket["distance"],
            "measurable_motion_time_s": bucket["measurable"],
            "mean_speed_cm_s": (bucket["distance"] / bucket["measurable"]
                                if bucket["measurable"] > 0 else None),
        })
    return bins


# -- 单动物处理 ---------------------------------------------------------

def process_unit(unit: Mapping[str, Any], captured: Sequence[Mapping[str, Any]],
                 video_window: timeline.Window,
                 first_seen: float | None) -> dict:
    """单只动物：物理坐标、资格、访问、候选、路径、分箱与逐洞表。"""

    options = unit["options"]
    geometry = unit.get("geometry") or {}
    ellipse, holes = require_geometry(geometry, options)
    matrix = affine_matrix(ellipse, float(options["disc_diameter_cm"]))
    center = ellipse[0]
    holes = list(holes)
    hole_points = [to_cm(matrix, center, hole["center"]) for hole in holes]
    target = target_hole_id(options)
    if target not in {hole["id"] for hole in holes}:
        raise ValidationError(f"目标洞 {target} 不在已标洞号中",
                              "options.target_hole")
    target_index = next(index for index, hole in enumerate(holes)
                        if hole["id"] == target)
    target_point = hole_points[target_index]

    window = timeline.resolve_unit_window(
        unit.get("window"), video_window, first_seen)
    rows = timeline.clip_rows(captured, window)
    method = (unit.get("detection") or {}).get("bg_method")
    state = {"target_angle": math.atan2(target_point[1], target_point[0])}
    for row in rows:
        row.setdefault("detection_method", method)
        _annotate(row, matrix, center, holes, hole_points, options, state)

    segments: list[dict] = []
    if window.known:
        # 只有相邻且真实曝光相接的合格身体点才能构成完成段；断口不补点。
        segments = timeline.motion_segments(rows, window,
                                            x_key="motion_x", y_key="motion_y",
                                            require_continuity=True)
    timeline.attach_motion(rows, segments)

    manual = unit.get("manual") if isinstance(unit.get("manual"), Mapping) else {}
    manual_visits = manual.get("hole_visits")
    manual_source = manual_visits is not None
    escape = _finite(manual.get("escape_time_s"))
    probe = options.get("trial_type") == "probe"

    # 自动洞访问始终计算：人工模式只换主序列，不删除自动审计记录。
    automatic_visits = _visits_for(rows, holes, options)
    visits = (_manual_visits(manual_visits) if manual_source
              else automatic_visits)
    if manual_source or escape is not None:
        # 人工时刻必须落在真实动物窗内；不裁剪、不延长、不改 null 迁就输入。
        check_manual_window(visits if manual_source else [], escape, window)
    if probe:
        # 探测试次没有入箱概念：既不产生候选，也不保留逃避字段。
        candidates: list[dict] = []
        escape = None
    else:
        candidates = escape_candidates(rows, target_point,
                                       float(options["hole_radius_cm"]), target)

    scored_escape = None if probe else escape
    scoring = _scored(visits, target, scored_escape)
    events = _event_rows(unit, visits, candidates, scored_escape, target)

    times = timeline.time_summary(rows)
    total = times["total_time_s"]
    body_time = nose_time = sector_time = outside_time = 0.0
    hole_nose: dict[str, float] = {hole["id"]: 0.0 for hole in holes}
    for row in rows:
        span = float(row.get("dt_s") or 0.0)
        if not row.get("in_window") or span <= 0:
            continue
        if row.get("body_qualified"):
            body_time += span
            if not row.get("inside_arena"):
                outside_time += span
        if row.get("nose_qualified"):
            nose_time += span
            hole_id = row.get("nose_hole")
            if hole_id in hole_nose:
                hole_nose[hole_id] += span
        if row.get("target_sector"):
            sector_time += span

    distance = sum(float(item["distance"]) for item in segments)
    measurable = sum(float(item["end_s"]) - float(item["start_s"])
                     for item in segments)
    max_speed: float | None = None
    for segment in segments:
        speed = float(segment["speed"])
        max_speed = speed if max_speed is None else max(max_speed, speed)

    origin = float(window.start_s) if window.known else 0.0
    primary = scoring["primary"]
    primary_latency = (float(primary["time_s"]) - origin) if primary else None
    primary_path = (_path_until(segments, float(primary["time_s"]), origin)
                    if primary else None)

    scored = scoring["scored"]
    unscored = [item for item in visits if not item["scored"]]
    post_escape = ([item for item in visits
                    if escape is not None and item.get("time_s") is not None
                    and float(item["time_s"]) >= float(escape)]
                   if escape is not None else [])

    if probe:
        total_latency = total_errors = total_path = total_censored = None
    elif escape is None:
        total_latency = total_errors = total_path = None
        total_censored = True
    else:
        total_latency = float(escape) - origin
        total_errors = len([item for item in scored if item["target"] != target])
        total_path = _path_until(segments, float(escape), origin)
        total_censored = False

    summary: dict[str, Any] = {
        "unit_id": unit.get("id"),
        "unit_name": unit.get("name"),
        "mode": KEY,
        "coordinate_unit": "cm",
        "analysis_window_start_s": window.start_s if window.known else None,
        "analysis_window_end_s": window.end_s if window.known else None,
        "total_time_s": total,
        "observed_time_s": times["observed_time_s"],
        "missing_time_s": times["missing_time_s"],
        "trial_type": options.get("trial_type"),
        "visit_source": "manual" if manual_source else "automatic",
        "target_hole": target,
        "visits": len(scored),
        "unscored_visits": len(unscored),
        "post_escape_visits": len(post_escape),
        "primary_latency_s": primary_latency,
        "primary_censored": scoring["censored"],
        "primary_errors": scoring["errors"],
        "primary_path_cm": primary_path,
        "total_latency_s": total_latency,
        "total_censored": total_censored,
        "total_errors": total_errors,
        "total_path_cm": total_path,
        "escape_candidates": len(candidates),
        "body_time_s": body_time,
        "nose_time_s": nose_time,
        "target_sector_time_s": sector_time,
        "outside_time_s": outside_time,
        "body_coverage_percent": _percent(body_time, total),
        "nose_coverage_percent": _percent(nose_time, total),
        "total_distance_cm": distance,
        "measurable_motion_time_s": measurable,
        "mean_speed_cm_s": (distance / measurable if measurable > 0 else None),
        "max_speed_cm_s": max_speed,
    }
    for key in OPTION_KEYS:
        summary[key] = options.get(key)

    holes_table = []
    for index, hole in enumerate(holes):
        point = hole_points[index]
        holes_table.append({
            "unit_id": unit.get("id"),
            "unit_name": unit.get("name"),
            "hole_id": hole["id"],
            "target": hole["id"] == target,
            "x_cm": point[0],
            "y_cm": point[1],
            "nose_time_s": hole_nose.get(hole["id"], 0.0),
            "visits": len([item for item in scored
                           if item["hole_id"] == hole["id"]]),
            "automatic_visits": len([item for item in automatic_visits
                                     if item["hole_id"] == hole["id"]]),
        })

    if manual_source:
        # 人工模式：自动访问只作审计记录，用独立事件号避免与主序列混淆。
        automatic_table = []
        for index, item in enumerate(automatic_visits, start=1):
            entry = dict(item)
            entry.update({
                "event_id": f"{unit.get('id')}:automatic_hole_visit:{index}",
                "unit_id": unit.get("id"),
                "unit_name": unit.get("name"),
            })
            automatic_table.append(entry)
    else:
        automatic_table = [dict(item) for item in events
                           if item["kind"] == "hole_visit"
                           and item["source"] == "automatic"]

    metadata = {
        "mode": KEY,
        "coordinate_unit": "cm",
        "scale": "ellipse affine A = R·diag(D/2rx, D/2ry)·Rᵀ; image y down",
        "disc_diameter_cm": float(options["disc_diameter_cm"]),
        "hole_count": int(options["hole_count"]),
        "hole_radius_cm": float(options["hole_radius_cm"]),
        "target_hole": target,
        "visit_source": summary["visit_source"],
        "coverage_denominator": "full animal window exposure",
        "primary_origin": "animal window start",
        "manual_entry_endpoint_s": escape,
        "manual_empty_means": ("zero visits" if manual_source else None),
        "mean_speed_denominator": "completed motion duration inside the disc",
        "sector_width_rad": 2.0 * math.pi / float(options["hole_count"]),
    }

    warning = None
    if not window.known:
        warning = ("未检测到动物，无法确定相对时间窗："
                   + str(unit.get("name") or unit.get("id")))
    elif summary["visit_source"] == "automatic" and nose_time < total - 1e-9:
        percent = _percent(nose_time, total) or 0.0
        warning = ("自动洞访问仅来自可用鼻尖观测"
                   f"（覆盖率 {percent:.1f}%）；缺失时段不能判定为无访问")

    bins = _bin_rows(rows, segments, window, options, unit)
    return {"rows": rows, "events": events, "bins": bins,
            "summary": summary, "metadata": metadata, "warning": warning,
            "tables": {"holes": holes_table,
                       "automatic_hole_visits": automatic_table}}


def _manual_visits(entries: Any) -> list[dict]:
    """人工覆盖记录：确认时刻一律由输入时刻派生，额外字段不改变评分。"""

    visits: list[dict] = []
    for item in entries if isinstance(entries, list) else []:
        if isinstance(item, str):
            hole_id, moment = item, None
        elif isinstance(item, Mapping):
            raw_id = item.get("hole_id", item.get("id", item.get("target")))
            hole_id = (None if isinstance(raw_id, bool) or raw_id is None
                       else str(raw_id).strip() or None)
            moment = _finite(item.get("time_s"))
        else:
            continue
        if hole_id is None:
            continue
        visits.append({
            "kind": "hole_visit",
            "hole_id": hole_id,
            "target": hole_id,
            "time_s": moment,
            # 人工事件的确认时刻就是输入时刻；调用方传入的 confirmed_s 不采信。
            "confirmed_s": moment,
            "source": "manual",
            "frame": None,
        })
    return visits


def check_manual_window(visits: Sequence[Mapping[str, Any]],
                        escape: float | None,
                        window: timeline.Window) -> None:
    """人工时刻必须落在真实动物窗内：访问是半开窗，入箱时间右端可等于窗尾。"""

    if not visits and escape is None:
        return
    if not window.known or window.duration_s <= 0:
        # 零时长窗不能确认任何人工时刻：不延窗、不裁记录、不清 null。
        field = ("manual.escape_time_s" if escape is not None
                 else "manual.hole_visits")
        raise ValidationError("没有有效动物窗，无法校验人工记录的时刻", field)
    start = float(window.start_s)
    end = float(window.end_s)
    for index, item in enumerate(visits):
        moment = item.get("time_s")
        if moment is None or not (start - 1e-9 <= float(moment) < end):
            raise ValidationError(
                f"人工访问必须落在动物窗 {start:g}–{end:g} s 内",
                f"manual.hole_visits[{index}].time_s")
    if escape is not None and not (start - 1e-9 <= float(escape)
                                   <= end + 1e-9):
        raise ValidationError(
            f"确认入箱时间必须落在动物窗 {start:g}–{end:g} s 内",
            "manual.escape_time_s")


def _event_rows(unit: Mapping[str, Any],
                visits: Sequence[Mapping[str, Any]],
                candidates: Sequence[Mapping[str, Any]],
                escape: float | None,
                target: str) -> list[dict]:
    """事件 id 按 kind 与顺序生成，再按 time_s、kind 稳定排序。"""

    unit_id = unit.get("id")
    counters: dict[str, int] = {}
    events: list[dict] = []
    rows = list(visits) + list(candidates)
    if escape is not None:
        # 人工确认入箱是独立端点，不能由未确认的消失候选替代。
        rows.append({"kind": "escape_confirmed", "target": target,
                     "time_s": float(escape), "confirmed_s": float(escape),
                     "source": "manual"})
    for item in rows:
        kind = item["kind"]
        counters[kind] = counters.get(kind, 0) + 1
        entry = dict(item)
        entry.update({
            "event_id": f"{unit_id}:{kind}:{counters[kind]}",
            "unit_id": unit_id,
            "unit_name": unit.get("name"),
        })
        events.append(entry)
    events.sort(key=lambda item: (item["time_s"] is None,
                                  item["time_s"] if item["time_s"] is not None
                                  else 0.0, item["kind"]))
    return events


# -- 捕获与分析 ---------------------------------------------------------

def _capture(context: Any, units: Sequence[Mapping[str, Any]],
             asset: Mapping[str, Any], metadata: Mapping[str, Any],
             source: Any) -> dict:
    """逐帧检测：正式分析与行为预览共用同一条捕获路径。"""

    detectors: dict[Any, Any] = {}
    regions: dict[Any, np.ndarray] = {}
    backgrounds: dict[tuple, Any] = {}
    captured: dict[Any, list[dict]] = {unit["id"]: [] for unit in units}
    unit_ids = [unit["id"] for unit in units]
    job = context.job
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
            if unit_id not in detectors:
                settings = unit.get("detection") or {}
                signature = tuple(settings.get(name) for name in
                                  detection.FOREGROUND_KEYS)
                if signature not in backgrounds:
                    backgrounds[signature] = detection.prepare_background(
                        settings, asset, job, int(job.get("orientation", 0) or 0),
                        width, height)
                detectors[unit_id] = detection.shared_detector(
                    run_key, settings, background=backgrounds[signature])
                regions[unit_id] = _region_for(unit, unit.get("options") or {},
                                               width, height)
            detector = detectors[unit_id]
            result = observation.track(
                run_key, unit_id,
                image=image, region=regions[unit_id],
                settings=unit["detection"],
                signature=observation.signature_of(
                    unit.get("geometry") or {}, unit["detection"], False),
                assay=KEY, frame_index=frame.index, time_s=frame.time_s,
                foreground=detector.foreground(image),
                retain_units=unit_ids,
            )
            captured[unit_id].append(observation_row(unit, frame, result))
        context.progress(min(0.9, 0.1 + 0.8 * max(0.0, float(frame.time_s)) /
                             max(1e-9, progress_span or 1.0)))
    return captured


def first_seen(rows: Iterable[Mapping[str, Any]],
               window: timeline.Window) -> float | None:
    """相对窗锚点：视频范围内首个有正曝光的实际检测。"""

    if not window.known:
        return None
    start, end = float(window.start_s), float(window.end_s)
    for row in timeline.sort_rows(rows):
        moment = float(row["time_s"])
        if moment < start or moment >= end:
            continue
        if not row.get("detected"):
            continue
        low = float(row.get("source_interval_start_s") or 0.0)
        high = float(row.get("source_interval_end_s") or 0.0)
        if high <= low:
            continue
        return moment
    return None


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
        # 冻结配置也可能带着未经过本次保存的人工输入：运行前再校验一次结构。
        manual_issues = validate_unit_manual(
            unit, options=unit.get("options") or {}, complete=True)
        if manual_issues:
            raise ValidationError(manual_issues[0]["message"],
                                  manual_issues[0]["field"])

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

    captured = _capture(context, units, asset, metadata, source)

    all_rows = [row for rows in captured.values() for row in rows]
    video_window = timeline.resolve_video_window(
        job.get("window"), all_rows, metadata.get("duration_s"))
    anchors = {unit["id"]: first_seen(captured[unit["id"]], video_window)
               for unit in units}

    rows_all: list[dict] = []
    events_all: list[dict] = []
    bins_all: list[dict] = []
    summary_all: list[dict] = []
    warnings: list[str] = []
    metadata_units: dict[str, dict] = {}
    tables: dict[str, list] = {"holes": [], "automatic_hole_visits": []}

    for unit in units:
        facts = process_unit(unit, captured[unit["id"]], video_window,
                             anchors[unit["id"]])
        rows_all.extend(facts["rows"])
        events_all.extend(facts["events"])
        bins_all.extend(facts["bins"])
        summary_all.append(facts["summary"])
        metadata_units[unit["id"]] = facts["metadata"]
        for name, items in facts["tables"].items():
            tables.setdefault(name, []).extend(items)
        if facts["warning"]:
            warnings.append(facts["warning"])

    return AnalysisResult(
        observations=timeline.sort_rows(rows_all),
        summary=summary_all,
        events=events_all,
        bins=bins_all,
        tables=tables,
        warnings=warnings,
        metadata={
            "assay": KEY,
            "coordinate_unit": "cm",
            "units": metadata_units,
            "warnings": warnings,
        },
    )


def preview_analysis(*, asset: Mapping[str, Any], video: Mapping[str, Any],
                     unit: Mapping[str, Any], start_s: float,
                     end_s: float, time_s: float | None = None) -> dict:
    """行为预览：对实际片段跑同一捕获→事实→访问核心，返回片段摘要。"""

    from ..schema import AnalysisContext, new_id

    if float(end_s) <= float(start_s):
        raise ValidationError("预览片段没有正区间", "time_s")

    options = unit.get("options") or {}
    issues = validate_unit(options, unit.get("geometry") or {}, complete=True)
    if issues:
        raise ValidationError(issues[0]["message"], issues[0]["field"])
    # 预览不假设持久草稿里的输入都经过本次保存：先校验结构与顺序再捕获。
    manual_issues = validate_unit_manual(unit, options=options, complete=True)
    if manual_issues:
        raise ValidationError(manual_issues[0]["message"],
                              manual_issues[0]["field"])

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
    captured = _capture(AnalysisContext(job=job, source=source),
                        [unit], asset, metadata, source)[unit["id"]]
    video_window = timeline.resolve_video_window(
        window, captured, metadata.get("duration_s"))
    anchor = first_seen(captured, video_window)
    facts = process_unit(unit, captured, video_window, anchor)
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
        "trial_type": summary["trial_type"],
        "visit_source": summary["visit_source"],
        "manual_entry_time_s": facts["metadata"].get("manual_entry_endpoint_s"),
        "target_hole": summary["target_hole"],
        "visits": summary["visits"],
        "unscored_visits": summary["unscored_visits"],
        "post_escape_visits": summary["post_escape_visits"],
        "primary_latency_s": summary["primary_latency_s"],
        "primary_errors": summary["primary_errors"],
        "primary_censored": summary["primary_censored"],
        "primary_path_cm": summary["primary_path_cm"],
        "total_latency_s": summary["total_latency_s"],
        "total_errors": summary["total_errors"],
        "total_path_cm": summary["total_path_cm"],
        "total_censored": summary["total_censored"],
        "nose_coverage_percent": summary["nose_coverage_percent"],
        "body_coverage_percent": summary["body_coverage_percent"],
        "target_sector_time_s": summary["target_sector_time_s"],
        "total_distance_cm": summary["total_distance_cm"],
        "mean_speed_cm_s": summary["mean_speed_cm_s"],
        "escape_candidates": summary["escape_candidates"],
        "rows": len(facts["rows"]),
        "events": len(facts["events"]),
        "bins": len(facts["bins"]),
    }


def validate(job: Any) -> list[dict]:
    issues: list[dict] = []
    for unit in job.get("units") or []:
        prefix = f"units[{unit.get('id')}]"
        for issue in validate_unit(unit.get("options") or {},
                                   unit.get("geometry") or {},
                                   complete=True):
            entry = dict(issue)
            entry["field"] = f"{prefix}.{issue['field']}"
            issues.append(entry)
    return issues
