"""共享活动特征：身体形态、成对运动、光流与摆动证据。

只消费保存下来的灰度图、鼠体掩膜与质心；判定阈值与状态机在实验模块里，
这里的函数都是纯数值，便于用数组夹具直接核对。
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from . import geometry as geo

BODY_LENGTH_MM = 60.0
FLOW_PAD_PX = 10
FLOW_DOWNSCALE = 0.35
FLOW_DOWNSCALE_ABOVE = 40
ACCEL_WINDOW_S = 0.5


def shape_points(shape: Mapping[str, Any] | None) -> list[tuple[float, float]]:
    if not isinstance(shape, Mapping):
        return []
    raw = shape.get("points")
    if not isinstance(raw, (list, tuple)):
        return []
    points: list[tuple[float, float]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return []
        try:
            x, y = float(item[0]), float(item[1])
        except (TypeError, ValueError):
            return []
        if not (math.isfinite(x) and math.isfinite(y)):
            return []
        points.append((x, y))
    return points


def line_scale(body_line: Mapping[str, Any] | None) -> tuple[float, float]:
    """体长线固定 60 mm：返回 (mm/px, 线长 px)，不从身体框反推尺度。"""

    points = shape_points(body_line)
    if len(points) != 2:
        raise ValueError("体长线需要两个有限点")
    length = math.hypot(points[1][0] - points[0][0], points[1][1] - points[0][1])
    if length <= 0:
        raise ValueError("体长线两个端点不能重合")
    return BODY_LENGTH_MM / length, length


def speeds(distance_px: float, mm_per_px: float, fps: float,
           elapsed_s: float) -> tuple[float, float]:
    """名义速度按分析频率，实际速度按真实源时间差；两者分开保存。"""

    nominal = float(distance_px) * float(mm_per_px) * float(fps)
    if elapsed_s <= 0:
        return nominal, 0.0
    return nominal, float(distance_px) * float(mm_per_px) / float(elapsed_s)


def body_features(candidate: Mapping[str, Any],
                  arena_points: Sequence[Sequence[float]],
                  elapsed_s: float | None,
                  prev_area: float | None) -> dict | None:
    """最大外轮廓的形态特征；质心不在场地内就不形成身体形态。

    身体面积是外轮廓面积，不是检测候选减去孔洞后的净面积：孔洞属于
    身体内部，形态量按实体外轮廓计算；候选检测仍保留孔洞规则。
    """

    contour = np.asarray(candidate.get("contour"), dtype=np.float32)
    if contour.size < 6:
        return None
    contour = contour.reshape(-1, 1, 2)
    center = candidate.get("center")
    if center is None or not geo.point_in_polygon(center, arena_points):
        return None
    x, y, width, height = candidate["bbox"]
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 1e-9 else 0.0
    fill = area / (float(width) * float(height)) if width > 0 and height > 0 else 0.0
    major = float(max(cv2.minAreaRect(contour)[1]))
    rate = None
    if (prev_area is not None and prev_area > 0
            and elapsed_s is not None and elapsed_s > 0):
        rate = (abs(area - float(prev_area)) / max(float(prev_area), 1e-6)
                / float(elapsed_s))
    return {
        "body_area_px": area,
        "body_perimeter_px": perimeter,
        "body_bbox_w": float(width),
        "body_bbox_h": float(height),
        "body_major_px": major,
        "body_solidity": float(solidity),
        "body_fill_ratio": float(fill),
        "body_aspect": float(height) / float(width) if width > 0 else 0.0,
        "body_area_rate_s": rate,
        "body_x_px": float(center[0]),
        "body_y_px": float(center[1]),
    }


def _flow_crop(gray: np.ndarray, mask: np.ndarray, bbox: Sequence[float]):
    x, y, width, height = (float(v) for v in bbox)
    image_h, image_w = gray.shape[:2]
    x0 = max(0, int(math.floor(x)) - FLOW_PAD_PX)
    y0 = max(0, int(math.floor(y)) - FLOW_PAD_PX)
    x1 = min(image_w, int(math.ceil(x + width)) + FLOW_PAD_PX)
    y1 = min(image_h, int(math.ceil(y + height)) + FLOW_PAD_PX)
    if x1 - x0 < 5 or y1 - y0 < 5:
        return None
    crop = gray[y0:y1, x0:x1]
    body = mask[y0:y1, x0:x1]
    scale = 1.0
    if max(x1 - x0, y1 - y0) > FLOW_DOWNSCALE_ABOVE:
        scale = FLOW_DOWNSCALE
        size = (max(8, int(round((x1 - x0) * scale))),
                max(8, int(round((y1 - y0) * scale))))
        crop = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
        body = cv2.resize(body, size, interpolation=cv2.INTER_NEAREST)
    return crop, body, scale


def direction_stats(vectors: np.ndarray) -> tuple[float, float]:
    """(C, E)：C=平均向量模长/平均模长；E=1−||mean(单位方向向量)||。"""

    magnitudes = np.linalg.norm(vectors, axis=1)
    directional = magnitudes > 1e-6
    selected = vectors[directional]
    sizes = magnitudes[directional]
    if sizes.size >= 50:
        p60 = float(np.percentile(sizes, 60))
        strong = sizes >= p60
        if int(strong.sum()) >= 20:
            selected = selected[strong]
            sizes = sizes[strong]
    if not sizes.size:
        return 0.0, 1.0
    mean_vector = selected.mean(axis=0)
    coherence = float(np.linalg.norm(mean_vector)
                      / max(float(sizes.mean()), 1e-9))
    unit = selected / sizes[:, None]
    dispersion = float(1.0 - np.linalg.norm(unit.mean(axis=0)))
    return (max(0.0, min(1.0, coherence)),
            max(0.0, min(1.0, dispersion)))


def acceleration_std(speeds: Sequence[float], fps: float) -> float:
    """最近窗口的有符号相邻名义速度差 × 分析 Hz 的总体标准差。"""

    diffs = [(float(b) - float(a)) * float(fps)
             for a, b in zip(speeds, speeds[1:])]
    return float(np.std(diffs)) if len(diffs) >= 3 else 0.0


def flow_features(prev_gray: np.ndarray, cur_gray: np.ndarray,
                  cur_mask: np.ndarray, bbox: Sequence[float],
                  prev_center: Sequence[float], cur_center: Sequence[float],
                  *, mm_per_px: float, fps: float,
                  elapsed_s: float) -> dict | None:
    """在当前身体框外扩 10px 的小图上做 Farneback；只统计当前鼠体掩膜。"""

    crops = _flow_crop(cur_gray, cur_mask, bbox)
    prev_crops = _flow_crop(prev_gray, cur_mask, bbox)
    if crops is None or prev_crops is None:
        return None
    cur_crop, body, scale = crops
    prev_crop, _, _ = prev_crops
    if prev_crop.shape != cur_crop.shape:
        height = min(prev_crop.shape[0], cur_crop.shape[0])
        width = min(prev_crop.shape[1], cur_crop.shape[1])
        prev_crop = prev_crop[:height, :width]
        cur_crop = cur_crop[:height, :width]
        body = body[:height, :width]
    flow = cv2.calcOpticalFlowFarneback(
        prev_crop, cur_crop, None, 0.5, 2, 15, 2, 5, 1.2, 0)
    if flow is None:
        return None
    if scale != 1.0:
        flow = flow / scale
    selected = body > 0
    vectors = flow[selected]
    if vectors.size == 0:
        return None
    magnitudes = np.linalg.norm(vectors, axis=1)
    if magnitudes.size == 0:
        return None

    coherence, dispersion = direction_stats(vectors)

    distance = math.hypot(float(cur_center[0]) - float(prev_center[0]),
                          float(cur_center[1]) - float(prev_center[1]))
    nominal, actual = speeds(distance, mm_per_px, fps, elapsed_s)
    return {
        "feature_ready": True,
        "flow_mean_px": float(magnitudes.mean()),
        "flow_p50_px": float(np.percentile(magnitudes, 50)),
        "flow_p90_px": float(np.percentile(magnitudes, 90)),
        "flow_max_px": float(magnitudes.max()),
        "flow_std_px": float(magnitudes.std()),
        "flow_coherence": max(0.0, min(1.0, coherence)),
        "flow_dispersion": max(0.0, min(1.0, dispersion)),
        "distance_px": float(distance),
        "speed_nominal_mm_s": float(nominal),
        "speed_mm_s": float(actual),
        "flow_p90_mm_s": float(np.percentile(magnitudes, 90)) * mm_per_px * fps,
        "flow_mean_mm_s": float(magnitudes.mean()) * mm_per_px * fps,
    }


def recent_rows(rows: Sequence[Mapping[str, Any]], time_s: float,
                seconds: float, *, include_left: bool = False
                ) -> list[Mapping[str, Any]]:
    """时间有序行里取 (t−w, t]；2 秒历史含左端，1 秒刚体窗不含左端。"""

    low = float(time_s) - float(seconds)
    selected: list[Mapping[str, Any]] = []
    for row in reversed(rows):
        moment = float(row["time_s"])
        if moment > float(time_s) + 1e-9:
            continue
        if include_left:
            if moment < low - 1e-9:
                break
            selected.append(row)
        else:
            if moment <= low + 1e-9:
                break
            selected.append(row)
    selected.reverse()
    return selected


def window_rows(rows: Iterable[Mapping[str, Any]], time_s: float,
                seconds: float, *, include_left: bool = False
                ) -> list[Mapping[str, Any]]:
    """无序行的特征窗；有序列表请用 recent_rows 以避免整段扫描。"""

    low = float(time_s) - float(seconds)
    selected = []
    for row in rows:
        moment = float(row["time_s"])
        if moment > float(time_s) + 1e-9:
            continue
        if include_left:
            if moment >= low - 1e-9:
                selected.append(row)
        elif moment > low + 1e-9:
            selected.append(row)
    return selected


def _population_std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    array = np.asarray(values, dtype=np.float64)
    return float(array.std())


def _spread(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, 90) - np.percentile(values, 10))


def autocorrelation(values: Sequence[float], fps: float) -> tuple[float, float | None]:
    """2 秒 x 序列的自相关峰；A 截到 [0,1]，频率按峰延迟换算。"""

    count = len(values)
    if count < 5 or fps <= 0:
        return 0.0, None
    series = np.asarray(values, dtype=np.float64)
    series = series - series.mean()
    denominator = float(np.dot(series, series))
    if denominator <= 1e-12:
        return 0.0, None
    full = np.correlate(series, series, mode="full")[count - 1:]
    normalized = full / denominator
    min_lag = max(1, int(math.ceil(fps / 3.0)))
    max_lag = min(count - 2, int(math.floor(1.25 * fps)))
    if max_lag < min_lag:
        return 0.0, None
    best_lag = None
    best_value = -2.0
    for lag in range(min_lag, max_lag + 1):
        value = float(normalized[lag])
        left = float(normalized[lag - 1])
        right = float(normalized[lag + 1]) if lag + 1 < count else -2.0
        if value + 1e-12 < left or value + 1e-12 < right:
            continue
        if value > best_value:
            best_value = value
            best_lag = lag
    if best_lag is None:
        return 0.0, None
    return max(0.0, min(1.0, best_value)), float(fps) / float(best_lag)


def rigidity(body_rows: Sequence[Mapping[str, Any]], time_s: float,
            fps: float) -> bool:
    """最近 1 秒的 P90−P10 波动；历史不足时暂记刚体。"""

    window = recent_rows(list(body_rows), time_s, 1.0, include_left=False)
    if len(window) < 2:
        return True
    span = float(window[-1]["time_s"]) - float(window[0]["time_s"])
    if span < 1.0 - 1.5 / max(fps, 1e-9):
        return True
    aspects = [float(row["body_aspect"]) for row in window]
    ys = [float(row["body_y_px"]) for row in window]
    areas = [float(row["body_area_px"]) for row in window]
    heights = [float(row["body_bbox_h"]) for row in window]
    aspect_median = float(np.median(aspects))
    if aspect_median <= 1e-9:
        return True
    if _spread(aspects) / aspect_median > 0.24:
        return False
    height_median = float(np.median(heights))
    if height_median <= 1e-9:
        return False
    if _spread(ys) / height_median > 0.025:
        return False
    area_median = float(np.median(areas))
    if area_median <= 1e-9:
        return False
    if _spread(areas) / area_median > 0.10:
        return False
    return True


def elongated(aspects: Sequence[float], current: float, fps: float) -> bool:
    """延长判据：当前比值 > max(1.2, .75×最近 2 秒中位数)。"""

    if len(aspects) < max(1, int(math.ceil(fps))):
        return float(current) > 1.8
    return float(current) > max(1.2, 0.75 * float(np.median(aspects)))


def motion_swing(*, frequency: float | None, autocorrelation: float,
                horizontal: float, vertical: float, coherence: float,
                dispersion: float, nominal: float, ratio: float) -> bool:
    """运动摆动候选：频率/自相关/位移比/一致性/离散度与速度门。"""

    return bool(
        frequency is not None and 0.8 <= float(frequency) <= 3.0
        and float(autocorrelation) >= 0.45
        and float(horizontal) >= 0.012
        and float(vertical) <= 0.55
        and float(coherence) >= 0.88
        and float(dispersion) <= 0.30
        and (float(nominal) < 8.0 or float(ratio) <= 2.2)
    )


def swing_score(*, autocorrelation_value: float | None,
                frequency: float | None, coherence: float, dispersion: float,
                is_elongated: bool, ratio: float) -> float:
    """摆动辅助分 W：每项只取命中的最高档。"""

    score = 0.0
    a = float(autocorrelation_value or 0.0)
    if a >= 0.8:
        score += 2.5
    elif a >= 0.7:
        score += 2.0
    elif a >= 0.6:
        score += 1.0
    if frequency is not None and 0.8 <= float(frequency) <= 3.0:
        score += 0.8
    if coherence >= 0.9:
        score += 1.2
    elif coherence >= 0.8:
        score += 0.8
    if dispersion <= 0.15:
        score += 1.2
    elif dispersion <= 0.30:
        score += 0.8
    if is_elongated:
        score += 0.6
    if ratio <= 1.2:
        score += 0.4
    return float(score)


def estimate_high_low(values: Sequence[float], original_high: float,
                      original_low: float, sensitivity: float,
                      cap: float) -> tuple[float, float]:
    """按线性分位数估计一对阈值；调用方负责样本量与幅度门。"""

    array = np.asarray(values, dtype=np.float64)
    p20, p35, p50 = (float(np.percentile(array, q)) for q in (20, 35, 50))
    p80, p90 = float(np.percentile(array, 80)), float(np.percentile(array, 90))
    base = p80
    if p90 - p20 < 1e-6:
        base = max(p50, 1.25 * p20)
    k = min(1.35, max(0.65, 1.0 - 0.1 * float(sensitivity)))
    high = min(max(base * k, 0.6 * float(original_high)),
               cap * float(original_high))
    low = max(0.001, min(0.65 * high, max(0.4 * high, p35)))
    return float(high), float(low)


# -- LIS 微动残差：配准、残差量与分位数（纯数值，夹具可直接核对） ----------

LIS_ROTATE_MAX_DEG = 30.0
LIS_HU_SCALE = 25.0


def _fold_angle(degrees: float) -> float:
    """角差折到 [-90°, 90°)。"""

    return (float(degrees) + 90.0) % 180.0 - 90.0


def lis_principal_angle(mask: np.ndarray) -> float | None:
    """身体像素主轴方向角（度，atan2(dy, dx)）；像素不足返回 None。"""

    points = np.column_stack(np.nonzero(mask))
    if points.shape[0] < 5:
        return None
    centered = points.astype(np.float64) - points.mean(axis=0)
    cov = centered.T @ centered / max(points.shape[0] - 1, 1)
    values, vectors = np.linalg.eigh(cov)
    principal = vectors[:, int(np.argmax(values))]
    return math.degrees(math.atan2(float(principal[0]), float(principal[1])))


def lis_register(prev_gray: np.ndarray, cur_gray: np.ndarray,
                 prev_mask: np.ndarray, cur_mask: np.ndarray,
                 prev_center: Sequence[float], cur_center: Sequence[float],
                 prev_angle: float | None, cur_angle: float | None, *,
                 rotate_align: bool) -> tuple[np.ndarray, np.ndarray, float]:
    """平移上一图至当前质心；可选主轴旋转对齐（角差 >30° 不旋转）。

    灰度双线性、掩膜最近邻插值，图像外填 0。返回 (对齐灰度, 对齐掩膜,
    实际使用的旋转角差)。
    """

    height, width = cur_gray.shape[:2]
    delta = 0.0
    if rotate_align and prev_angle is not None and cur_angle is not None:
        delta = _fold_angle(float(cur_angle) - float(prev_angle))
        if abs(delta) > LIS_ROTATE_MAX_DEG:
            delta = 0.0
    cos_t = math.cos(math.radians(delta))
    sin_t = math.sin(math.radians(delta))
    px, py = float(prev_center[0]), float(prev_center[1])
    tx, ty = float(cur_center[0]), float(cur_center[1])
    # p → R·(p − prev_center) + cur_center
    matrix = np.array([
        [cos_t, -sin_t, tx - (cos_t * px - sin_t * py)],
        [sin_t, cos_t, ty - (sin_t * px + cos_t * py)],
    ], dtype=np.float64)
    aligned_gray = cv2.warpAffine(prev_gray, matrix, (width, height),
                                  flags=cv2.INTER_LINEAR, borderValue=0)
    aligned_mask = cv2.warpAffine(prev_mask, matrix, (width, height),
                                  flags=cv2.INTER_NEAREST, borderValue=0)
    return aligned_gray, aligned_mask, delta


def lis_residual(aligned_gray: np.ndarray, aligned_mask: np.ndarray,
                 cur_gray: np.ndarray, cur_mask: np.ndarray) -> dict:
    """掩膜残差、灰差图与综合残差 R = max(掩膜残差, mean_U(D))。"""

    union = (aligned_mask > 0) | (cur_mask > 0)
    inter = (aligned_mask > 0) & (cur_mask > 0)
    union_count = int(union.sum())
    mask_residual = 1.0 - (int(inter.sum()) / max(union_count, 1))
    diff = np.abs(cur_gray.astype(np.float32)
                  - aligned_gray.astype(np.float32)) / 255.0
    gray_mean = float(diff[union].mean()) if union_count else 0.0
    return {"mask_residual": mask_residual, "gray_mean_d": gray_mean,
            "r": max(mask_residual, gray_mean), "diff": diff}


def lis_body_length(mask: np.ndarray, bbox: Sequence[float]) -> float:
    """L = max(主轴投影 P95−P5, 框最长边)。"""

    projected = 0.0
    points = np.column_stack(np.nonzero(mask)).astype(np.float64)
    if points.shape[0] >= 5:
        centered = points - points.mean(axis=0)
        cov = centered.T @ centered / max(points.shape[0] - 1, 1)
        values, vectors = np.linalg.eigh(cov)
        axis = vectors[:, int(np.argmax(values))]
        projection = centered @ axis
        projected = float(np.percentile(projection, 95)
                          - np.percentile(projection, 5))
    _, _, width, height = (float(v) for v in bbox)
    return max(projected, width, height)


def lis_core_peripheral(mask: np.ndarray, length_px: float,
                        diff: np.ndarray) -> dict:
    """以 .08L（最近整数、≥3、偶数加 1）椭圆核腐蚀分核心/周边。

    任一区为空就用整个身体。返回核心/周边灰差均值与比值 r。
    """

    body = mask > 0
    if not body.any():
        return {"core_mean": 0.0, "peripheral_mean": 0.0, "ratio": 0.0,
                "kernel": 0}
    size = int(round(0.08 * float(length_px)))
    size = max(3, size)
    if size % 2 == 0:
        size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    core = cv2.erode(mask, kernel) > 0
    peripheral = body & ~core
    if not core.any() or not peripheral.any():
        core = body
        peripheral = body
    core_mean = float(diff[core].mean())
    peripheral_mean = float(diff[peripheral].mean())
    return {"core_mean": core_mean, "peripheral_mean": peripheral_mean,
            "ratio": peripheral_mean / max(core_mean, 1e-6), "kernel": size}


def lis_hu_log(value: float) -> float:
    """Hu 分量的有符号对数 −sign(h)·log10(max(|h|,1e−30))。

    注意先取对数再乘符号；h = 0 时 sign(h) = 0，分量必须正好是 0。
    """

    number = float(value)
    if number == 0.0:
        return 0.0
    magnitude = math.log10(max(abs(number), 1e-30))
    return -magnitude if number > 0 else magnitude


def lis_hu_vector(mask: np.ndarray) -> list[float] | None:
    """Hu 矩的有符号对数向量，逐分量按 lis_hu_log 计算。"""

    if not isinstance(mask, np.ndarray) or not (mask > 0).any():
        return None
    moments = cv2.moments(mask, binaryImage=True)
    raw = cv2.HuMoments(moments).flatten()
    return [lis_hu_log(float(value)) for value in raw]


def lis_hu_distance(cur: Sequence[float] | None,
                    prev: Sequence[float] | None) -> float:
    """Hu 对数向量的欧氏距离；任一侧缺失按 0。"""

    if cur is None or prev is None:
        return 0.0
    return float(np.linalg.norm(np.asarray(cur, dtype=np.float64)
                                - np.asarray(prev, dtype=np.float64)))


def lis_deformation(cur: Mapping[str, float], prev: Mapping[str, float],
                    hu_distance: float, mask_residual: float) -> float:
    """形变量 Z：七量截到 [0,1] 后的中位数。"""

    parts = []
    for key in ("area", "perimeter", "aspect"):
        base = max(abs(float(prev[key])), 1e-6)
        parts.append(min(1.0, abs(float(cur[key]) - float(prev[key])) / base))
    for key in ("solidity", "fill"):
        parts.append(min(1.0, abs(float(cur[key]) - float(prev[key]))))
    parts.append(min(1.0, max(0.0, hu_distance / LIS_HU_SCALE)))
    parts.append(min(1.0, max(0.0, mask_residual)))
    return float(np.median(parts))


def lis_median_diff(values: Sequence[float]) -> float:
    """相邻差绝对值的中位数；不足两条为 0。"""

    if len(values) < 2:
        return 0.0
    diffs = [abs(float(b) - float(a)) for a, b in zip(values, values[1:])]
    return float(np.median(diffs))


def lis_periodicity(xs: Sequence[float], ys: Sequence[float],
                    times: Sequence[float]) -> float:
    """最近窗的周期量 b：零延迟归一自相关在 .2–1 秒延迟的最大值。

    两轴各减中位数后计算取较大者；少于 8 条或跨度 ≤1 秒为 0。
    采样率按 (条数−1)/真实跨度 估计，延迟范围换算为帧 lag。
    """

    count = len(times)
    if count < 8:
        return 0.0
    span = float(times[-1]) - float(times[0])
    if span <= 1.0:
        return 0.0
    fps_est = (count - 1) / span
    min_lag = max(1, int(math.ceil(0.2 * fps_est)))
    max_lag = min(count - 2, int(math.floor(1.0 * fps_est)))
    if max_lag < min_lag:
        return 0.0
    best = 0.0
    for series in (xs, ys):
        array = np.asarray(series, dtype=np.float64)
        array = array - float(np.median(array))
        denominator = float(np.dot(array, array))
        if denominator <= 1e-12:
            continue
        full = np.correlate(array, array, mode="full")[count - 1:]
        for lag in range(min_lag, max_lag + 1):
            best = max(best, float(full[lag]) / denominator)
    return max(0.0, min(1.0, best))


def lis_thresholds(values: Sequence[float]) -> dict:
    """线性分位数阈值：low=min(q50, max(0, q20+.5·MAD))、mid=q50、
    high=q80、very_high=q90；q80−q20<1e-4 记 low_information。"""

    array = np.asarray(values, dtype=np.float64)
    q20, q25, q50, q75, q80, q90 = (
        float(np.percentile(array, q)) for q in (20, 25, 50, 75, 80, 90))
    mad = float(np.median(np.abs(array - q50)))
    low = min(q50, max(0.0, q20 + 0.5 * mad))
    return {"q20": q20, "q25": q25, "q50": q50, "q75": q75, "q80": q80,
            "q90": q90, "mad": mad, "low": low, "mid": q50, "high": q80,
            "very_high": q90, "low_information": bool((q80 - q20) < 1e-4)}


def lis_normalize(value: float, q25: float, q90: float) -> float:
    """nX = clip((X−q25)/max(q90−q25, 1e-6), 0, 1)。"""

    span = max(float(q90) - float(q25), 1e-6)
    return max(0.0, min(1.0, (float(value) - float(q25)) / span))


def lis_motion_prior(f_q50: float, f_q90: float, s_q50: float,
                     s_q90: float) -> float:
    """运动先验 p = max(.5×[c(Fq50/45)+c(Sq50/25)], .375×[c(Fq90/90)+c(Sq90/40)])。"""

    def clamp(value: float, reference: float) -> float:
        return max(0.0, min(1.0, float(value) / max(float(reference), 1e-9)))

    slow = 0.5 * (clamp(f_q50, 45.0) + clamp(s_q50, 25.0))
    fast = 0.375 * (clamp(f_q90, 90.0) + clamp(s_q90, 40.0))
    return max(slow, fast)


def lis_above(value: float, threshold: float) -> bool:
    """"高于 a"：值 > a + max(1e-6, .02|a|)。"""

    threshold = float(threshold)
    return float(value) > threshold + max(1e-6, 0.02 * abs(threshold))


def lis_not_above(value: float, threshold: float) -> bool:
    """"不高于 a"：值 ≤ a + max(1e-6, .05|a|)。"""

    threshold = float(threshold)
    return float(value) <= threshold + max(1e-6, 0.05 * abs(threshold))
