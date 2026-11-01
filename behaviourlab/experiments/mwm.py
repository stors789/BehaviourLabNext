"""MWM Morris 水迷宫：椭圆泳池归一化、平台事件、路径与分箱。

像素点先按 S = Rθ·diag(D/2rx, D/2ry)·Rθᵀ 归一化到圆，再翻转 y 并按
坐标轴旋转得到北为正的物理厘米坐标。平台、释放点与轨迹共用这套坐标；
轨迹只由连续的真实观测形成，缺失不补点。
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .. import activity, detection, geometry as geo, media, observation, timeline
from ..schema import AnalysisResult, ValidationError, is_number
from .base import geometry_field, metric, parameter

KEY = "mwm"

DEFAULTS = {
    "pool_diameter_cm": 120.0,
    "platform_radius_cm": 5.0,
    "trial_type": "hidden",
    "coordinate_rotation_deg": 0.0,
    "target_quadrant": "auto",
    "start_position": "N",
    "platform_dwell_s": 1.0,
    "thigmotaxis_band_cm": 10.0,
    "corridor_width_cm": 10.0,
    "movement_threshold_cm_s": 1.0,
    "bin_seconds": 60.0,
}

OPTION_KEYS = tuple(DEFAULTS)
POSITIVE_KEYS = ("pool_diameter_cm", "platform_radius_cm", "platform_dwell_s",
                 "corridor_width_cm", "bin_seconds")
NONNEGATIVE_KEYS = ("thigmotaxis_band_cm", "movement_threshold_cm_s")
ROTATION_RANGE = (-360.0, 360.0)
TRIAL_TYPES = ("hidden", "visible", "probe")
TARGET_QUADRANTS = ("auto", "NE", "NW", "SW", "SE")
START_POSITIONS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
START_ANGLES = {"E": 0.0, "NE": 45.0, "N": 90.0, "NW": 135.0,
                "W": 180.0, "SW": 225.0, "S": 270.0, "SE": 315.0}
QUADRANTS = ("NE", "NW", "SW", "SE")
QUADRANT_ROMAN = {"NE": "I", "NW": "II", "SW": "III", "SE": "IV"}
TIME_EPS = 1e-9
# 轴上点的旋转残差（如 90° 旋转后的 −1.8e−15）相对量级可忽略：按此比例归零，
# 真实负坐标不受影响。
ZERO_EPS = 1e-12

DETECTION_DEFAULTS: dict = {}

# 首次检测必须落在泳池内，才允许锚定相对窗口。
FIRST_SEEN = "arena"

SPEC = {
    "key": KEY,
    "label": "Morris 水迷宫 / MWM",
    "short": "MWM",
    "parameters": [
        parameter("pool_diameter_cm", "泳池直径", "number", 120.0,
                  unit="cm", minimum=0, step="any"),
        parameter("platform_radius_cm", "平台半径", "number", 5.0,
                  unit="cm", minimum=0, step="any"),
        parameter("trial_type", "试次", "select", "hidden",
                  choices=[{"value": "hidden", "label": "隐藏平台"},
                           {"value": "visible", "label": "可见平台"},
                           {"value": "probe", "label": "探测试次"}]),
        parameter("coordinate_rotation_deg", "坐标轴旋转", "number", 0.0,
                  unit="°", minimum=-360, maximum=360, step="any"),
        parameter("target_quadrant", "目标象限", "select", "auto",
                  choices=[{"value": "auto", "label": "按平台位置"},
                           {"value": "NE", "label": "I（东北）"},
                           {"value": "NW", "label": "II（西北）"},
                           {"value": "SW", "label": "III（西南）"},
                           {"value": "SE", "label": "IV（东南）"}]),
        parameter("start_position", "释放位置", "select", "N",
                  choices=[{"value": item, "label": item}
                           for item in START_POSITIONS]),
        parameter("platform_dwell_s", "平台确认停留", "number", 1.0,
                  unit="s", minimum=0, step="any", section="实验参数"),
        parameter("thigmotaxis_band_cm", "趋边区宽度", "number", 10.0,
                  unit="cm", minimum=0, step="any", section="实验参数"),
        parameter("corridor_width_cm", "释放点至平台通道宽度", "number", 10.0,
                  unit="cm", minimum=0, step="any", section="实验参数"),
        parameter("movement_threshold_cm_s", "移动速度阈值", "number", 1.0,
                  unit="cm/s", minimum=0, step="any", section="实验参数"),
        parameter("bin_seconds", "时间箱长度", "number", 60.0, unit="s",
                  minimum=0, step="any", section="实验参数"),
    ],
    "geometry": [
        geometry_field("arena", "泳池边界（椭圆或至少五点）", "ellipse",
                       min_points=5),
        geometry_field("platform", "平台中心或平台椭圆",
                       "ellipse_or_point", min_points=1,
                       modes=["center", "ellipse"]),
        geometry_field("mask", "检测遮罩", "polygon", min_points=3,
                       optional=True),
    ],
    "metrics": [
        metric("escape_latency_s", "平台确认潜伏期", "s"),
        metric("first_platform_entry_latency_s", "首次进入平台区潜伏期", "s"),
        metric("path_length_to_platform_cm", "确认平台前路径", "cm"),
        metric("platform_entries", "平台区进入次数", "次"),
        metric("platform_time_s", "平台区时间", "s"),
        metric("total_distance_cm", "总路径", "cm"),
        metric("mean_speed_cm_s", "平均速度", "cm/s"),
        metric("target_quadrant_time_s", "目标象限时间", "s"),
        metric("target_quadrant_percent", "目标象限比例", "%"),
        metric("target_quadrant_distance_cm", "目标象限路径", "cm"),
        metric("thigmotaxis_time_s", "趋边时间", "s"),
        metric("corridor_time_s", "通道时间", "s"),
        metric("mean_distance_to_platform_cm", "按时间加权的平台距离", "cm"),
        metric("path_weighted_distance_to_platform_cm",
               "按路程加权的平台距离", "cm"),
        metric("moving_time_s", "移动时间", "s"),
        metric("floating_time_s", "低速或漂浮时间", "s"),
        metric("quadrant_ne_time_s", "东北象限时间", "s"),
        metric("quadrant_nw_time_s", "西北象限时间", "s"),
        metric("quadrant_sw_time_s", "西南象限时间", "s"),
        metric("quadrant_se_time_s", "东南象限时间", "s"),
        metric("search_strategy", "探索性搜索策略"),
    ],
    "template_keys": ["pool_diameter_cm", "platform_radius_cm"],
}


# -- 参数与几何校验 -----------------------------------------------------

def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not is_number(value):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _pool_params(geometry: Mapping[str, Any] | None
                 ) -> tuple[list[float], list[float], float] | None:
    """泳池椭圆参数 (center, radii, angle)；无法解析时 None。"""

    arena = (geometry or {}).get("arena")
    if not isinstance(arena, Mapping) or arena.get("type") != "ellipse":
        return None
    center = arena.get("center")
    radii = arena.get("radii")
    if isinstance(center, Sequence) and radii:
        cx, cy = _finite(center[0]), _finite(center[1])
        rx, ry = _finite(radii[0]), _finite(radii[1])
        angle = _finite(arena.get("angle", 0.0))
        if None in (cx, cy, rx, ry, angle) or rx <= 0 or ry <= 0:
            return None
        return [cx, cy], [rx, ry], angle
    points = activity.shape_points(arena)
    if len(points) < 5:
        return None
    try:
        fitted = geo.fit_ellipse(points)
    except ValidationError:
        return None
    rx, ry = float(fitted["radii"][0]), float(fitted["radii"][1])
    if rx <= 0 or ry <= 0:
        return None
    return [float(fitted["center"][0]), float(fitted["center"][1])], \
        [rx, ry], float(fitted["angle"])


def _platform_center(geometry: Mapping[str, Any] | None
                     ) -> list[float] | None:
    shape = (geometry or {}).get("platform")
    if not isinstance(shape, Mapping):
        return None
    kind = shape.get("type")
    if kind == "point":
        points = activity.shape_points(shape)
        return list(points[0]) if len(points) == 1 else None
    if kind != "ellipse":
        return None
    center = shape.get("center")
    radii = shape.get("radii")
    if isinstance(center, Sequence) and radii:
        cx, cy = _finite(center[0]), _finite(center[1])
        rx, ry = _finite(radii[0]), _finite(radii[1])
        if None in (cx, cy, rx, ry) or rx <= 0 or ry <= 0:
            return None
        return [cx, cy]
    points = activity.shape_points(shape)
    if len(points) < 5:
        return None
    try:
        fitted = geo.fit_ellipse(points)
    except ValidationError:
        return None
    return [float(fitted["center"][0]), float(fitted["center"][1])]


def validate_unit(options: Mapping[str, Any] | None,
                  geometry: Mapping[str, Any] | None,
                  provided: Iterable[str] | None = None, *,
                  complete: bool = False) -> list[dict]:
    values = options if isinstance(options, Mapping) else {}
    provided = set(OPTION_KEYS if provided is None else provided)
    issues: list[dict] = []

    for key in POSITIVE_KEYS:
        if key not in provided:
            continue
        value = _finite(values.get(key))
        if value is None:
            issues.append({"field": f"options.{key}", "message": "必须是有限数值"})
        elif value <= 0:
            issues.append({"field": f"options.{key}", "message": "必须大于 0"})

    for key in NONNEGATIVE_KEYS:
        if key not in provided:
            continue
        value = _finite(values.get(key))
        if value is None:
            issues.append({"field": f"options.{key}", "message": "必须是有限数值"})
        elif value < 0:
            issues.append({"field": f"options.{key}", "message": "必须是 ≥ 0 的数值"})

    if "coordinate_rotation_deg" in provided:
        # 旋转角是有符号量：只要求有限非布尔并在 [-360, 360] 内。
        rotation = _finite(values.get("coordinate_rotation_deg"))
        if rotation is None:
            issues.append({"field": "options.coordinate_rotation_deg",
                           "message": "必须是有限数值"})
        elif not ROTATION_RANGE[0] <= rotation <= ROTATION_RANGE[1]:
            issues.append({"field": "options.coordinate_rotation_deg",
                           "message": "坐标轴旋转必须在 -360 到 360 之间"})

    for key, allowed in (("trial_type", TRIAL_TYPES),
                         ("target_quadrant", TARGET_QUADRANTS),
                         ("start_position", START_POSITIONS)):
        if key in provided and values.get(key) not in allowed:
            issues.append({"field": f"options.{key}", "message": "未知取值"})

    diameter = _finite(values.get("pool_diameter_cm"))
    radius = _finite(values.get("platform_radius_cm"))
    if diameter is not None and radius is not None and radius > diameter / 2.0:
        issues.append({"field": "options.platform_radius_cm",
                       "message": "平台半径不能大于泳池半径"})
    band = _finite(values.get("thigmotaxis_band_cm"))
    if diameter is not None and band is not None and band > diameter / 2.0:
        issues.append({"field": "options.thigmotaxis_band_cm",
                       "message": "趋边区宽度不能大于泳池半径"})

    issues.extend(validate_geometry(geometry, require=complete))
    return issues


def validate_geometry(geometry: Mapping[str, Any] | None,
                      *, require: bool = False) -> list[dict]:
    geometry = geometry or {}
    issues: list[dict] = []
    pool = _pool_params(geometry)
    arena = geometry.get("arena")

    if arena is None:
        if require:
            issues.append({"field": "geometry.arena", "message": "缺少泳池边界"})
    elif not isinstance(arena, Mapping) or arena.get("type") != "ellipse":
        issues.append({"field": "geometry.arena",
                       "message": "泳池边界必须是椭圆"})
    elif pool is None:
        issues.append({"field": "geometry.arena",
                       "message": "泳池椭圆需要两个正半轴，或至少 5 个有限点"})

    platform = geometry.get("platform")
    if platform is None:
        if require:
            issues.append({"field": "geometry.platform",
                           "message": "缺少平台中心或平台椭圆"})
    elif _platform_center(geometry) is None:
        issues.append({"field": "geometry.platform",
                       "message": "平台需要一个中心点或合法椭圆"})
    elif pool is not None:
        center = _platform_center(geometry)
        if not geo.ellipse_contains(center, pool[0], pool[1], pool[2]):
            issues.append({"field": "geometry.platform",
                           "message": "平台中心必须在泳池内"})

    mask = geometry.get("mask")
    if mask is not None:
        points = activity.shape_points(mask)
        if len(points) < 3:
            issues.append({"field": "geometry.mask",
                           "message": "检测遮罩至少需要 3 个有限点"})
        elif abs(_polygon_area(points)) <= 1e-9:
            issues.append({"field": "geometry.mask",
                           "message": "检测遮罩面积必须大于 0"})
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


# -- 像素 → 物理坐标 ----------------------------------------------------

def _linear_matrix(pool: tuple[list[float], list[float], float],
                   diameter_cm: float, rotation_deg: float) -> np.ndarray:
    """S = Rα·F·Rθ·diag(D/2rx, D/2ry)·Rθᵀ。"""

    center, radii, angle = pool
    rx, ry = float(radii[0]), float(radii[1])
    a = float(diameter_cm) / (2.0 * rx)
    b = float(diameter_cm) / (2.0 * ry)
    theta = math.radians(float(angle))
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    local = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)
    scale = np.diag([a, b])
    flip = np.diag([1.0, -1.0])
    alpha = math.radians(float(rotation_deg))
    cos_a, sin_a = math.cos(alpha), math.sin(alpha)
    rotate = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)
    return rotate @ flip @ local @ scale @ local.T


def _snap_axis(value: float, scale: float) -> float:
    """把相对量级可忽略的数值零残差归零；真实正负坐标保持原符号。"""

    limit = ZERO_EPS * max(abs(float(scale)), 1.0)
    return 0.0 if abs(float(value)) <= limit else float(value)


def _transform(matrix: np.ndarray, center: Sequence[float],
               point: Sequence[float]) -> tuple[float, float]:
    vector = np.array([float(point[0]) - float(center[0]),
                       float(point[1]) - float(center[1])], dtype=np.float64)
    result = matrix @ vector
    x, y = float(result[0]), float(result[1])
    scale = max(abs(x), abs(y))
    return _snap_axis(x, scale), _snap_axis(y, scale)


def quadrant_of(x: float, y: float) -> str:
    """x≥0 东、y≥0 北；原点归东北。轴上残差与物理坐标同一归零规则。"""

    scale = max(abs(float(x)), abs(float(y)))
    x = _snap_axis(x, scale)
    y = _snap_axis(y, scale)
    if x >= 0.0:
        return "NE" if y >= 0.0 else "SE"
    return "NW" if y >= 0.0 else "SW"


def start_point(position: str, radius: float) -> tuple[float, float]:
    """释放点在半径 R 圆周，按最终物理角度（不重复轴旋转）。"""

    angle = math.radians(START_ANGLES.get(position, 90.0))
    return radius * math.cos(angle), radius * math.sin(angle)


def quadrant_guides(geometry: Mapping[str, Any] | None,
                    options: Mapping[str, Any] | None) -> dict | None:
    """预览画布用的象限辅助线：像素中心、两条单位方向与像素长度。"""

    pool = _pool_params(geometry)
    if pool is None:
        return None
    diameter = _finite((options or {}).get("pool_diameter_cm"))
    if diameter is None or diameter <= 0:
        return None
    rotation = _finite((options or {}).get("coordinate_rotation_deg")) or 0.0
    matrix = _linear_matrix(pool, diameter, rotation)
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return None
    x_direction = inverse @ np.array([1.0, 0.0])
    y_direction = inverse @ np.array([0.0, 1.0])
    length = max(float(pool[1][0]), float(pool[1][1]))
    return {
        "center": [float(pool[0][0]), float(pool[0][1])],
        "directions": [_unit(x_direction), _unit(y_direction)],
        "length": length,
    }


def _unit(vector: np.ndarray) -> list[float]:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return [1.0, 0.0]
    return [float(vector[0]) / norm, float(vector[1]) / norm]


# -- 捕获 ----------------------------------------------------------------

def _unit_row(unit: Mapping[str, Any], frame: Any, result: Any) -> dict:
    return timeline.base_row(
        {"id": unit.get("id"), "name": unit.get("name"),
         "animal_id": unit.get("animal_id")},
        frame, result,
        extra={"detection_method": (unit.get("detection") or {}).get(
            "bg_method")},
    )


def capture(context: Any, units: Sequence[Mapping[str, Any]]):
    """逐帧检测：正式分析与行为预览共用同一条捕获路径。"""

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

    detectors: dict[Any, Any] = {}
    regions: dict[Any, np.ndarray] = {}
    backgrounds: dict[tuple, Any] = {}
    captured: dict[Any, list[dict]] = {unit["id"]: [] for unit in units}
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
            if unit_id not in detectors:
                settings = unit.get("detection") or {}
                signature = tuple(settings.get(name) for name in
                                  detection.FOREGROUND_KEYS)
                if signature not in backgrounds:
                    backgrounds[signature] = detection.prepare_background(
                        settings, asset, job,
                        int(job.get("orientation", 0) or 0), width, height)
                detectors[unit_id] = detection.shared_detector(
                    run_key, settings, background=backgrounds[signature])
                regions[unit_id] = detection.geometry_region(
                    unit.get("geometry") or {}, KEY,
                    unit.get("options") or {}, width, height)[0]
            result = observation.track(
                run_key, unit_id,
                image=image, region=regions[unit_id],
                settings=unit["detection"],
                signature=observation.signature_of(
                    unit.get("geometry") or {}, unit["detection"], False),
                assay=KEY, frame_index=frame.index, time_s=frame.time_s,
                foreground=detectors[unit_id].foreground(image),
                retain_units=unit_ids,
            )
            captured[unit_id].append(_unit_row(unit, frame, result))
        context.progress(min(0.85, 0.1 + 0.75 * max(0.0, float(frame.time_s)) /
                             max(1e-9, progress_span or 1.0)))
    return captured


# -- 池内事实、平台事件与路径 -------------------------------------------

def _annotate(row: dict, matrix: np.ndarray, center: Sequence[float],
              radius: float, platform: Sequence[float],
              platform_radius: float, band: float,
              corridor_half: float, start: Sequence[float],
              platform_center_distance: float) -> None:
    row["in_pool"] = None
    row["x_cm"] = None
    row["y_cm"] = None
    row["quadrant"] = None
    row["distance_to_platform_cm"] = None
    row["in_platform"] = None
    row["thigmotaxis"] = None
    row["in_corridor"] = None
    row["near_platform"] = None
    row["platform_annulus"] = None
    row["distance_to_center_cm"] = None
    if not row.get("detected") or row.get("x_px") is None:
        return
    x_cm, y_cm = _transform(matrix, center, (row["x_px"], row["y_px"]))
    distance_center = math.hypot(x_cm, y_cm)
    if distance_center > radius + 1e-8:
        row["in_pool"] = False
        return
    row["in_pool"] = True
    row["x_cm"] = x_cm
    row["y_cm"] = y_cm
    row["distance_to_center_cm"] = distance_center
    row["quadrant"] = quadrant_of(x_cm, y_cm)
    distance = math.hypot(x_cm - float(platform[0]),
                          y_cm - float(platform[1]))
    row["distance_to_platform_cm"] = distance
    row["in_platform"] = distance <= platform_radius + 1e-9
    row["thigmotaxis"] = distance_center >= radius - band - 1e-9
    row["in_corridor"] = geo.distance_to_segment(
        (x_cm, y_cm), start, platform) <= corridor_half + 1e-9
    row["near_platform"] = distance <= 2.0 * platform_radius + 1e-9
    row["platform_annulus"] = (
        abs(distance_center - platform_center_distance) <= platform_radius + 1e-9)


def _continuous(previous: Mapping[str, Any] | None,
                row: Mapping[str, Any]) -> bool:
    return timeline.exposures_continuous(previous, row)


def _platform_events(rows: Sequence[Mapping[str, Any]],
                     window: timeline.Window, dwell: float,
                     trial: str) -> dict:
    entries: list[dict] = []
    first_observation: float | None = None
    first_confirmed: dict | None = None
    platform_time = 0.0
    candidate: float | None = None
    confirmed_run = False
    previous: Mapping[str, Any] | None = None

    for row in rows:
        if not row.get("in_window"):
            continue
        span = float(row.get("dt_s") or 0.0)
        if span <= 0:
            continue
        moment = max(float(row["time_s"]), float(row["interval_start_s"]))
        if row.get("in_platform"):
            platform_time += span
            continuous = _continuous(previous, row)
            if candidate is not None and not continuous:
                candidate = None
                confirmed_run = False
            if first_observation is None:
                first_observation = moment
            if candidate is None:
                candidate = moment
                confirmed_run = False
                if (previous is not None and previous.get("in_pool")
                        and previous.get("in_platform") is False
                        and continuous):
                    entries.append({"time_s": moment, "frame": row["frame"]})
            elif (not confirmed_run and moment > candidate + TIME_EPS
                  and moment - candidate >= dwell - TIME_EPS):
                confirmed_run = True
                if trial != "probe" and first_confirmed is None:
                    first_confirmed = {
                        "entry_s": candidate, "confirmed_s": moment,
                        "frame": row["frame"]}
        else:
            candidate = None
            confirmed_run = False
        previous = row

    start = float(window.start_s) if window.known else None
    first_entry = entries[0]["time_s"] if entries else None
    latencies = {
        "observation": (None if first_observation is None or start is None
                        else first_observation - start),
        "entry": (None if first_entry is None or start is None
                  else first_entry - start),
        "escape": (None if first_confirmed is None or start is None
                   else first_confirmed["confirmed_s"] - start),
    }
    return {
        "entries": entries,
        "first_observation_s": first_observation,
        "first_confirmed": first_confirmed,
        "platform_time_s": platform_time,
        "latencies": latencies,
        "escape_censored": (None if trial == "probe"
                            else first_confirmed is None),
    }


def _axis_pieces(segment: Mapping[str, Any]) -> list[dict]:
    """按 x=0 / y=0 交点把段切成子段，象限取子段中点。"""

    x0, y0 = float(segment["p0"][0]), float(segment["p0"][1])
    x1, y1 = float(segment["p1"][0]), float(segment["p1"][1])
    cuts = [0.0, 1.0]
    if abs(x1 - x0) > 1e-12:
        value = -x0 / (x1 - x0)
        if 0.0 < value < 1.0:
            cuts.append(value)
    if abs(y1 - y0) > 1e-12:
        value = -y0 / (y1 - y0)
        if 0.0 < value < 1.0:
            cuts.append(value)
    cuts = sorted(set(cuts))
    pieces = []
    for low, high in zip(cuts, cuts[1:]):
        px0 = x0 + (x1 - x0) * low
        py0 = y0 + (y1 - y0) * low
        px1 = x0 + (x1 - x0) * high
        py1 = y0 + (y1 - y0) * high
        mid_x, mid_y = (px0 + px1) / 2.0, (py0 + py1) / 2.0
        pieces.append({
            "t0": low, "t1": high,
            "p0": (px0, py0), "p1": (px1, py1),
            "length": float(segment["distance"]) * (high - low),
            "quadrant": quadrant_of(mid_x, mid_y),
        })
    return pieces


def _g_function(value: float, height: float) -> float:
    if abs(height) <= 1e-12:
        return value * abs(value) / 2.0
    return (value * math.hypot(value, height)
            + height * height * math.asinh(value / height)) / 2.0


def _segment_integral(segment: Mapping[str, Any],
                      target: Sequence[float]) -> float:
    length = float(segment["distance"])
    if length <= 0:
        return 0.0
    x0, y0 = float(segment["p0"][0]), float(segment["p0"][1])
    x1, y1 = float(segment["p1"][0]), float(segment["p1"][1])
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    wx, wy = x0 - float(target[0]), y0 - float(target[1])
    a = wx * ux + wy * uy
    height = abs(wx * uy - wy * ux)
    return _g_function(a + length, height) - _g_function(a, height)


def _path_until(segments: Sequence[Mapping[str, Any]],
                confirmed_s: float) -> float:
    total = 0.0
    for segment in segments:
        start = float(segment["start_s"])
        end = float(segment["end_s"])
        if start >= confirmed_s - TIME_EPS or end <= start:
            continue
        stop = min(end, confirmed_s)
        ratio = (stop - start) / (end - start)
        x0, y0 = float(segment["p0"][0]), float(segment["p0"][1])
        x1, y1 = float(segment["p1"][0]), float(segment["p1"][1])
        px = x0 + (x1 - x0) * ratio
        py = y0 + (y1 - y0) * ratio
        total += math.hypot(px - x0, py - y0)
    return total


def _ratio(numerator: float, denominator: float,
           scale: float = 1.0) -> float | None:
    return timeline.ratio(numerator * scale, denominator)


# -- 汇总与分箱 ---------------------------------------------------------

def process_unit(unit: Mapping[str, Any],
                 captured: Sequence[Mapping[str, Any]],
                 video_window: timeline.Window,
                 first_seen: float | None) -> dict:
    options = unit.get("options") or {}
    geometry = unit.get("geometry") or {}
    pool = _pool_params(geometry)
    if pool is None:
        raise ValidationError("泳池椭圆无效", "geometry.arena")
    platform_px = _platform_center(geometry)
    if platform_px is None:
        raise ValidationError("平台中心无效", "geometry.platform")
    diameter = float(options["pool_diameter_cm"])
    radius = diameter / 2.0
    rotation = float(options.get("coordinate_rotation_deg") or 0.0)
    matrix = _linear_matrix(pool, diameter, rotation)
    platform = _transform(matrix, pool[0], platform_px)
    platform_center_distance = math.hypot(platform[0], platform[1])
    platform_radius = float(options["platform_radius_cm"])
    band = float(options["thigmotaxis_band_cm"])
    corridor_half = float(options["corridor_width_cm"]) / 2.0
    start = start_point(str(options["start_position"]), radius)
    auto_quadrant = quadrant_of(platform[0], platform[1])
    target = (auto_quadrant if options.get("target_quadrant") == "auto"
              else str(options.get("target_quadrant")))

    window = timeline.resolve_unit_window(
        unit.get("window"), video_window, first_seen)
    rows = timeline.clip_rows(captured, window)
    for row in rows:
        _annotate(row, matrix, pool[0], radius, platform, platform_radius,
                  band, corridor_half, start, platform_center_distance)
    segments = timeline.motion_segments(rows, window,
                                        x_key="x_cm", y_key="y_cm",
                                        require_continuity=True)
    timeline.attach_motion(rows, segments)
    facts = _platform_events(
        rows, window, float(options["platform_dwell_s"]),
        str(options["trial_type"]))
    summary = summarize(rows, segments, unit, window, options, facts,
                        radius, platform, start, target)
    bins = build_bins(rows, segments, window, options, target)
    return {"rows": rows, "segments": segments, "summary": summary,
            "events": _events(unit, facts), "bins": bins,
            "platform": platform, "start": start,
            "metadata": {
                "coordinate_unit": "cm",
                "coordinate_origin": "pool center",
                "coordinate_transform": (
                    "S=Rθ·diag(D/2rx,D/2ry)·Rθᵀ，随后 y 变号并按"
                    "coordinate_rotation_deg 旋转"),
                "coordinate_rotation_deg": rotation,
                "pool_radius_cm": radius,
                "platform_x_cm": platform[0],
                "platform_y_cm": platform[1],
                "start_x_cm": start[0],
                "start_y_cm": start[1],
                "target_quadrant": target,
                "occupancy_denominator": "有效池内曝光",
                "confirmation_rule": (
                    "后续真实观测时刻 > 入段起点且经过 ≥ platform_dwell_s"),
                "path_integral": (
                    "∫|p−target|ds = G(a+l)−G(a)，h 为到段所在直线的垂距"),
                "search_strategy_note": "探索性规则标签，未经行为学验证",
            }}


def _events(unit: Mapping[str, Any], facts: Mapping[str, Any]) -> list[dict]:
    unit_id = unit.get("id")
    unit_name = unit.get("name") or ""
    events: list[dict] = []
    for index, entry in enumerate(facts["entries"], start=1):
        events.append({
            "event_id": f"{unit_id}:platform_entry:{index}",
            "unit_id": unit_id,
            "unit_name": unit_name,
            "kind": "platform_entry",
            "time_s": float(entry["time_s"]),
            "confirmed_s": float(entry["time_s"]),
            "target": "platform",
            "source": "automatic",
            "frame": entry["frame"],
        })
    confirmed = facts["first_confirmed"]
    if confirmed is not None:
        events.append({
            "event_id": f"{unit_id}:platform_confirmed:1",
            "unit_id": unit_id,
            "unit_name": unit_name,
            "kind": "platform_confirmed",
            "time_s": float(confirmed["entry_s"]),
            "confirmed_s": float(confirmed["confirmed_s"]),
            "target": "platform",
            "source": "automatic",
            "frame": confirmed["frame"],
        })
    return events


def summarize(rows: Sequence[Mapping[str, Any]],
              segments: Sequence[Mapping[str, Any]],
              unit: Mapping[str, Any], window: timeline.Window,
              options: Mapping[str, Any], facts: Mapping[str, Any],
              radius: float, platform: Sequence[float],
              start: Sequence[float], target: str) -> dict:
    total = observed = missing = valid = 0.0
    platform_time = thigmotaxis = corridor = 0.0
    near_time = annulus_time = 0.0
    distance_sum = radius_sum = 0.0
    quadrant_time = {key: 0.0 for key in QUADRANTS}
    target_time = 0.0
    for row in rows:
        if not row.get("in_window"):
            continue
        span = max(0.0, float(row.get("dt_s") or 0.0))
        total += span
        if row.get("detected"):
            observed += span
        else:
            missing += span
        if not row.get("in_pool"):
            continue
        valid += span
        distance_sum += float(row["distance_to_platform_cm"]) * span
        radius_sum += float(row["distance_to_center_cm"]) * span
        if row.get("in_platform"):
            platform_time += span
        if row.get("thigmotaxis"):
            thigmotaxis += span
        if row.get("in_corridor"):
            corridor += span
        if row.get("near_platform"):
            near_time += span
        if row.get("platform_annulus"):
            annulus_time += span
        quadrant = row.get("quadrant")
        if quadrant in quadrant_time:
            quadrant_time[quadrant] += span
        if quadrant == target:
            target_time += span

    total_distance = sum(float(segment["distance"]) for segment in segments)
    measurable = sum(float(segment["end_s"]) - float(segment["start_s"])
                     for segment in segments)
    threshold = float(options["movement_threshold_cm_s"])
    moving = floating = 0.0
    quadrant_distance = {key: 0.0 for key in QUADRANTS}
    for segment in segments:
        span = float(segment["end_s"]) - float(segment["start_s"])
        if float(segment["speed"]) > threshold:
            moving += span
        else:
            floating += span
        for piece in _axis_pieces(segment):
            quadrant_distance[piece["quadrant"]] += piece["length"]

    confirmed = facts["first_confirmed"]
    trial = str(options["trial_type"])
    pre_path = None
    efficiency = None
    if confirmed is not None and trial != "probe":
        pre_path = _path_until(segments, float(confirmed["confirmed_s"]))
        straight = math.hypot(platform[0] - start[0], platform[1] - start[1])
        if pre_path > TIME_EPS:
            efficiency = min(1.0, straight / pre_path)

    integral = sum(_segment_integral(segment, platform)
                   for segment in segments)
    path_weighted = (integral / total_distance
                     if total_distance > TIME_EPS else None)
    mean_distance = _ratio(distance_sum, valid)
    mean_radius = _ratio(radius_sum, valid)
    max_speed = (max((float(segment["speed"]) for segment in segments),
                     default=None) if segments else None)

    summary = {
        "unit_id": unit.get("id"),
        "unit_name": unit.get("name"),
        "mode": "mwm",
        "total_time_s": total,
        "observed_time_s": observed,
        "missing_time_s": missing,
        "valid_pool_time_s": valid,
        "detected_rate_percent": _ratio(observed, total, 100.0),
        "escape_latency_s": facts["latencies"]["escape"],
        "first_platform_observation_latency_s": facts["latencies"]["observation"],
        "first_platform_entry_latency_s": facts["latencies"]["entry"],
        "path_length_to_platform_cm": pre_path,
        "escape_censored": facts["escape_censored"],
        "path_efficiency": efficiency,
        "platform_entries": len(facts["entries"]),
        "platform_time_s": platform_time,
        "platform_percent": _ratio(platform_time, valid, 100.0),
        "total_distance_cm": total_distance,
        "max_speed_cm_s": max_speed,
        "measurable_motion_time_s": measurable,
        "moving_time_s": moving,
        "floating_time_s": floating,
        "mean_speed_cm_s": (total_distance / measurable
                            if measurable > TIME_EPS else None),
        "target_quadrant_resolved": target,
        "target_quadrant_roman": QUADRANT_ROMAN.get(target),
        "target_quadrant_time_s": target_time,
        "target_quadrant_percent": _ratio(target_time, valid, 100.0),
        "target_quadrant_distance_cm": quadrant_distance.get(target, 0.0),
        "thigmotaxis_time_s": thigmotaxis,
        "thigmotaxis_percent": _ratio(thigmotaxis, valid, 100.0),
        "corridor_time_s": corridor,
        "corridor_percent": _ratio(corridor, valid, 100.0),
        "near_platform_time_s": near_time,
        "near_platform_percent": _ratio(near_time, valid, 100.0),
        "platform_annulus_time_s": annulus_time,
        "platform_annulus_percent": _ratio(annulus_time, valid, 100.0),
        "mean_distance_to_platform_cm": mean_distance,
        "path_weighted_distance_to_platform_cm": path_weighted,
        "mean_radius_cm": mean_radius,
        "analysis_window_start_s": window.start_s if window.known else None,
        "analysis_window_end_s": window.end_s if window.known else None,
        "pool_diameter_cm": float(options["pool_diameter_cm"]),
        "platform_radius_cm": float(options["platform_radius_cm"]),
        "trial_type": trial,
        "start_position": str(options["start_position"]),
        "pool_radius_cm": radius,
        "platform_x_cm": platform[0],
        "platform_y_cm": platform[1],
        "start_x_cm": start[0],
        "start_y_cm": start[1],
    }
    for key in QUADRANTS:
        summary[f"quadrant_{key.lower()}_time_s"] = quadrant_time[key]
        summary[f"quadrant_{key.lower()}_percent"] = _ratio(
            quadrant_time[key], valid, 100.0)
        summary[f"quadrant_{key.lower()}_distance_cm"] = quadrant_distance[key]
    for key in OPTION_KEYS:
        summary[key] = options.get(key)
    summary["search_strategy"] = search_strategy(summary)
    return summary


def search_strategy(summary: Mapping[str, Any]) -> str:
    """探索性规则，严格按序取首个满足项。"""

    valid = float(summary.get("valid_pool_time_s") or 0.0)
    if valid <= TIME_EPS:
        return "unknown"

    def fraction(key: str) -> float:
        value = summary.get(key)
        return float(value) / 100.0 if is_number(value) else 0.0

    if fraction("thigmotaxis_percent") >= 0.5:
        return "thigmotaxis"
    efficiency = summary.get("path_efficiency")
    if is_number(efficiency):
        if float(efficiency) >= 0.8:
            return "direct"
        if float(efficiency) >= 0.5:
            return "directed"
    if fraction("near_platform_percent") >= 0.5:
        return "focal"
    if fraction("platform_annulus_percent") >= 0.5:
        return "chaining"
    if fraction("target_quadrant_percent") >= 0.35:
        return "indirect"
    mean_radius = summary.get("mean_radius_cm")
    diameter = float(summary.get("pool_diameter_cm") or 0.0)
    if is_number(mean_radius) and diameter > 0 \
            and float(mean_radius) < 0.3 * diameter:
        return "scanning"
    return "random"


def build_bins(rows: Sequence[Mapping[str, Any]],
               segments: Sequence[Mapping[str, Any]],
               window: timeline.Window, options: Mapping[str, Any],
               target: str) -> list[dict]:
    if not window.known or window.duration_s <= 0:
        return []
    width = float(options["bin_seconds"])
    if width <= 0:
        return []
    origin = float(window.start_s)
    duration = window.duration_s
    count = max(1, int(math.ceil((duration - TIME_EPS) / width)))
    buckets = [{"observed": 0.0, "missing": 0.0, "valid": 0.0,
                "platform": 0.0, "target": 0.0, "thigmotaxis": 0.0,
                "corridor": 0.0,
                "quadrant": {key: 0.0 for key in QUADRANTS},
                "distance": 0.0,
                "quadrant_distance": {key: 0.0 for key in QUADRANTS},
                "measurable": 0.0, "moving": 0.0, "floating": 0.0}
               for _ in range(count)]
    threshold = float(options["movement_threshold_cm_s"])

    for row in rows:
        if not row.get("in_window"):
            continue
        for index, low, high in timeline.split_interval(
                row["interval_start_s"], row["interval_end_s"], origin, width):
            if not 0 <= index < count:
                continue
            span = high - low
            bucket = buckets[index]
            if row.get("detected"):
                bucket["observed"] += span
            else:
                bucket["missing"] += span
            if not row.get("in_pool"):
                continue
            bucket["valid"] += span
            if row.get("in_platform"):
                bucket["platform"] += span
            if row.get("thigmotaxis"):
                bucket["thigmotaxis"] += span
            if row.get("in_corridor"):
                bucket["corridor"] += span
            quadrant = row.get("quadrant")
            if quadrant in QUADRANTS:
                bucket["quadrant"][quadrant] += span
                if quadrant == target:
                    bucket["target"] += span

    for segment in segments:
        start = float(segment["start_s"])
        end = float(segment["end_s"])
        span = end - start
        if span <= 0:
            continue
        for index, low, high in timeline.split_interval(start, end, origin,
                                                        width):
            if not 0 <= index < count:
                continue
            ratio_low = (low - start) / span
            ratio_high = (high - start) / span
            x0, y0 = float(segment["p0"][0]), float(segment["p0"][1])
            x1, y1 = float(segment["p1"][0]), float(segment["p1"][1])
            px0 = x0 + (x1 - x0) * ratio_low
            py0 = y0 + (y1 - y0) * ratio_low
            px1 = x0 + (x1 - x0) * ratio_high
            py1 = y0 + (y1 - y0) * ratio_high
            length = math.hypot(px1 - px0, py1 - py0)
            piece_span = high - low
            bucket = buckets[index]
            bucket["distance"] += length
            bucket["measurable"] += piece_span
            if float(segment["speed"]) > threshold:
                bucket["moving"] += piece_span
            else:
                bucket["floating"] += piece_span
            piece = {"distance": length,
                     "p0": (px0, py0), "p1": (px1, py1)}
            for axis_piece in _axis_pieces(piece):
                bucket["quadrant_distance"][axis_piece["quadrant"]] += \
                    axis_piece["length"]

    bins: list[dict] = []
    for index, bucket in enumerate(buckets):
        start = index * width
        end = min(start + width, duration)
        span = end - start
        bins.append({
            "unit_id": rows[0].get("unit_id") if rows else None,
            "unit_name": rows[0].get("unit_name") if rows else None,
            "bin_index": index + 1,
            "start_s": start,
            "end_s": end,
            "duration_s": span,
            "observed_time_s": bucket["observed"],
            "missing_time_s": bucket["missing"],
            "valid_pool_time_s": bucket["valid"],
            "platform_time_s": bucket["platform"],
            "target_quadrant_time_s": bucket["target"],
            "thigmotaxis_time_s": bucket["thigmotaxis"],
            "corridor_time_s": bucket["corridor"],
            "quadrant_ne_time_s": bucket["quadrant"]["NE"],
            "quadrant_nw_time_s": bucket["quadrant"]["NW"],
            "quadrant_sw_time_s": bucket["quadrant"]["SW"],
            "quadrant_se_time_s": bucket["quadrant"]["SE"],
            "quadrant_ne_distance_cm": bucket["quadrant_distance"]["NE"],
            "quadrant_nw_distance_cm": bucket["quadrant_distance"]["NW"],
            "quadrant_sw_distance_cm": bucket["quadrant_distance"]["SW"],
            "quadrant_se_distance_cm": bucket["quadrant_distance"]["SE"],
            "total_distance_cm": bucket["distance"],
            "measurable_motion_time_s": bucket["measurable"],
            "moving_time_s": bucket["moving"],
            "floating_time_s": bucket["floating"],
            "mean_speed_cm_s": (bucket["distance"] / bucket["measurable"]
                                if bucket["measurable"] > TIME_EPS else None),
        })
    return bins


# -- 分析入口 ------------------------------------------------------------

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
    if context.cancelled():
        raise ValidationError("分析已取消", "job")
    asset = job.get("asset") or {}
    metadata = asset.get("metadata") or {}
    all_rows = [row for rows in captured.values() for row in rows]
    video_window = timeline.resolve_video_window(
        job.get("window"), all_rows, metadata.get("duration_s"))

    observations: list[dict] = []
    summary: list[dict] = []
    events: list[dict] = []
    bins: list[dict] = []
    unit_metadata: dict[str, dict] = {}
    for unit in units:
        if context.cancelled():
            raise ValidationError("分析已取消", "job")
        geometry = unit.get("geometry") or {}
        first_seen = timeline.first_seen_in_window(
            KEY, captured[unit["id"]], video_window, geometry)
        facts = process_unit(unit, captured[unit["id"]], video_window,
                             first_seen)
        observations.extend(facts["rows"])
        summary.append(facts["summary"])
        events.extend(facts["events"])
        bins.extend(facts["bins"])
        unit_metadata[unit["id"]] = facts["metadata"]

    return AnalysisResult(
        observations=timeline.sort_rows(observations),
        summary=summary,
        events=events,
        bins=bins,
        tables={},
        warnings=[],
        metadata={"assay": KEY, "units": unit_metadata, "warnings": []},
    )


def preview_analysis(*, asset: Mapping[str, Any], video: Mapping[str, Any],
                     unit: Mapping[str, Any], start_s: float,
                     end_s: float, time_s: float | None = None) -> dict:
    """行为预览：实际片段跑同一捕获→池内事实→事件→汇总核心。"""

    from ..schema import AnalysisContext, new_id

    options = unit.get("options") or {}
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
    captured = capture(AnalysisContext(job=job, source=source),
                       [unit])[unit["id"]]
    video_window = timeline.resolve_video_window(
        window, captured, metadata.get("duration_s"))
    geometry = unit.get("geometry") or {}
    first_seen = timeline.first_seen_in_window(
        KEY, captured, video_window, geometry)
    facts = process_unit(unit, captured, video_window, first_seen)
    summary = facts["summary"]
    return {
        "mode": "mwm",
        "start_s": float(start_s),
        "end_s": float(end_s),
        "window_start_s": (summary["analysis_window_start_s"]),
        "window_end_s": summary["analysis_window_end_s"],
        "total_time_s": summary["total_time_s"],
        "platform_entries": summary["platform_entries"],
        "platform_time_s": summary["platform_time_s"],
        "total_distance_cm": summary["total_distance_cm"],
        "escape_latency_s": summary["escape_latency_s"],
        "first_platform_entry_latency_s":
            summary["first_platform_entry_latency_s"],
        "target_quadrant": summary["target_quadrant_resolved"],
        "target_quadrant_roman": summary["target_quadrant_roman"],
        "search_strategy": summary["search_strategy"],
        "rows": len(facts["rows"]),
        "preview_note": "行为预览为短段结果，与全程分析可能不同",
    }
