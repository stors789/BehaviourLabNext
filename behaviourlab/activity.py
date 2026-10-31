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
                  dt_s: float, prev_area: float | None) -> dict | None:
    """最大外轮廓的形态特征；质心不在场地内就不形成身体形态。"""

    contour = np.asarray(candidate.get("contour"), dtype=np.float32)
    if contour.size < 6:
        return None
    contour = contour.reshape(-1, 1, 2)
    center = candidate.get("center")
    if center is None or not geo.point_in_polygon(center, arena_points):
        return None
    x, y, width, height = candidate["bbox"]
    area = float(candidate["area"])
    perimeter = float(cv2.arcLength(contour, True))
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 1e-9 else 0.0
    fill = area / (float(width) * float(height)) if width > 0 and height > 0 else 0.0
    major = float(max(cv2.minAreaRect(contour)[1]))
    rate = None
    if prev_area is not None and prev_area > 0 and dt_s > 0:
        rate = abs(area - float(prev_area)) / max(float(prev_area), 1e-6) / dt_s
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

    directional = magnitudes > 1e-6
    direction_vectors = vectors[directional]
    direction_magnitudes = magnitudes[directional]
    if direction_magnitudes.size >= 50:
        p60 = float(np.percentile(direction_magnitudes, 60))
        strong = direction_magnitudes >= p60
        if int(strong.sum()) >= 20:
            direction_vectors = direction_vectors[strong]
            direction_magnitudes = direction_magnitudes[strong]
    if direction_magnitudes.size:
        mean_vector = direction_vectors.mean(axis=0)
        coherence = float(np.linalg.norm(mean_vector)
                          / max(float(direction_magnitudes.mean()), 1e-9))
        unit = direction_vectors / direction_magnitudes[:, None]
        dispersion = float(1.0 - np.linalg.norm(unit, axis=1).mean())
    else:
        coherence = 0.0
        dispersion = 1.0

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
