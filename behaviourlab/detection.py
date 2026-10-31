"""经典前景检测：参数、背景、区域、候选与质量。

本模块只做单帧候选；跨帧关联留给后续块。前景像素不等于目标动物，
候选按面积降序返回，缺失时给出明确原因。
"""

from __future__ import annotations

import math
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from .schema import Detection, is_number

BG_METHODS = ("threshold", "temporal", "static", "mog2", "knn")
BG_TYPES = ("light", "dark")

DETECTION_DEFAULTS: dict = {
    "bg_method": "threshold",
    "bg_type": "light",
    "gray_threshold": 127.0,
    "body_gray_threshold": 95.0,
    "erosion_kernel": 3,
    "erosion_iter": 1,
    "dilation_kernel": 5,
    "dilation_iter": 2,
    "min_area": 120.0,
    "max_area": 50000.0,
    "temporal_bg_samples": 90,
    "temporal_diff_threshold": 25.0,
    "quality_threshold": 0.4,
    "static_background": None,
}

INTEGER_FIELDS = ("erosion_kernel", "erosion_iter", "dilation_kernel",
                  "dilation_iter", "temporal_bg_samples")

SATURATION_RATIO = 0.85
TILE = 64
STRIP_ROWS = 64
SHADOW_FRACTION = 0.98


class DetectionError(RuntimeError):
    """检测无法进行：区域、背景或参数不满足要求。"""


def common_detection_defaults() -> dict:
    return deepcopy(DETECTION_DEFAULTS)


def _issue(field_name: str, message: str) -> dict:
    return {"field": field_name, "message": message}


def validate_detection(values: Any, prefix: str = "detection") -> list[dict]:
    """按服务器有效范围校验检测命名空间；未知字段不在此处扩展。"""

    if values is None:
        return []
    if not isinstance(values, Mapping):
        return [_issue(prefix, "必须是对象")]

    issues: list[dict] = []
    if "bg_method" in values and values["bg_method"] not in BG_METHODS:
        issues.append(_issue(f"{prefix}.bg_method", "未知背景方法"))
    if "bg_type" in values and values["bg_type"] not in BG_TYPES:
        issues.append(_issue(f"{prefix}.bg_type", "未知明暗类型"))

    for key in ("gray_threshold", "body_gray_threshold",
                "temporal_diff_threshold"):
        if key not in values:
            continue
        value = values[key]
        if not is_number(value) or not math.isfinite(float(value)):
            issues.append(_issue(f"{prefix}.{key}", "必须是有限数值"))
        elif not 0 <= float(value) <= 255:
            issues.append(_issue(f"{prefix}.{key}", "必须在 0–255 之间"))

    for key in INTEGER_FIELDS:
        if key not in values:
            continue
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, int):
            issues.append(_issue(f"{prefix}.{key}", "必须是整数"))
            continue
        minimum = 0 if key.endswith("_iter") else 1
        if value < minimum:
            issues.append(_issue(f"{prefix}.{key}", f"必须 ≥ {minimum}"))

    for key in ("min_area", "max_area"):
        if key not in values:
            continue
        value = values[key]
        if not is_number(value) or not math.isfinite(float(value)):
            issues.append(_issue(f"{prefix}.{key}", "必须是有限数值"))
        elif float(value) < 0:
            issues.append(_issue(f"{prefix}.{key}", "必须 ≥ 0"))

    if "min_area" in values and "max_area" in values:
        minimum, maximum = values["min_area"], values["max_area"]
        if is_number(minimum) and is_number(maximum) and float(maximum) < float(minimum):
            issues.append(_issue(f"{prefix}.max_area", "必须 ≥ min_area"))

    if "quality_threshold" in values:
        value = values["quality_threshold"]
        if not is_number(value) or not math.isfinite(float(value)):
            issues.append(_issue(f"{prefix}.quality_threshold", "必须是有限数值"))
        elif not 0 <= float(value) <= 1:
            issues.append(_issue(f"{prefix}.quality_threshold", "必须在 0–1 之间"))

    if "static_background" in values:
        value = values["static_background"]
        if value is not None and (not isinstance(value, str) or not value.strip()):
            issues.append(_issue(f"{prefix}.static_background", "必须是路径或 null"))

    return issues


def merge_settings(values: Mapping[str, Any] | None) -> dict:
    settings = deepcopy(DETECTION_DEFAULTS)
    if isinstance(values, Mapping):
        for key in DETECTION_DEFAULTS:
            if key in values:
                settings[key] = values[key]
    return settings


def normalize_settings(values: Mapping[str, Any] | None) -> dict:
    settings = merge_settings(values)
    issues = validate_detection(settings)
    if issues:
        raise DetectionError(issues[0]["message"])
    return settings


# -- 灰度与前景 ---------------------------------------------------------

def to_gray(image: Any) -> np.ndarray:
    """BGR → GRAY；灰度数组直接使用，不做写操作。"""

    array = np.asarray(image)
    if array.ndim == 2:
        return array.astype(np.uint8, copy=False)
    if array.ndim != 3 or array.shape[2] < 3:
        raise DetectionError("图像必须是灰度或三通道 BGR")
    return cv2.cvtColor(array[:, :, :3], cv2.COLOR_BGR2GRAY)


def threshold_foreground(gray: np.ndarray, settings: Mapping[str, Any]) -> np.ndarray:
    threshold = float(settings.get("gray_threshold", 127.0))
    if settings.get("bg_type", "light") == "light":
        mask = (gray <= threshold).astype(np.uint8) * 255
    else:
        mask = (gray > threshold).astype(np.uint8) * 255
    return mask


def _otsu_threshold(tile: np.ndarray) -> float:
    values = tile.reshape(-1)
    if values.size == 0 or int(values.max()) == int(values.min()):
        return -1.0
    threshold, _ = cv2.threshold(tile, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(threshold)


def local_otsu_foreground(gray: np.ndarray, bg_type: str = "light") -> np.ndarray:
    """64×64 非重叠图块独立 Otsu；常量图块没有分割证据，输出 0。"""

    height, width = gray.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    for top in range(0, height, TILE):
        bottom = min(height, top + TILE)
        for left in range(0, width, TILE):
            right = min(width, left + TILE)
            tile = gray[top:bottom, left:right]
            threshold = _otsu_threshold(tile)
            if threshold < 0:
                continue
            if bg_type == "light":
                block = (tile <= threshold).astype(np.uint8) * 255
            else:
                block = (tile > threshold).astype(np.uint8) * 255
            mask[top:bottom, left:right] = block
    return mask


def temporal_foreground(gray: np.ndarray, background: np.ndarray,
                        bg_type: str = "light",
                        difference: float = 25.0) -> np.ndarray:
    """有符号差分：light 取背景−当前，dark 取当前−背景，严格大于阈值。"""

    gray_i = gray.astype(np.int16)
    background_i = background.astype(np.int16)
    if bg_type == "light":
        delta = background_i - gray_i
    else:
        delta = gray_i - background_i
    return ((delta > float(difference)) * 255).astype(np.uint8)


def apply_morphology(mask: np.ndarray, settings: Mapping[str, Any]) -> np.ndarray:
    result = mask
    erosion_kernel = int(settings.get("erosion_kernel", 3))
    erosion_iter = int(settings.get("erosion_iter", 1))
    dilation_kernel = int(settings.get("dilation_kernel", 5))
    dilation_iter = int(settings.get("dilation_iter", 2))
    if erosion_iter > 0 and erosion_kernel >= 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (erosion_kernel, erosion_kernel))
        result = cv2.erode(result, kernel, iterations=erosion_iter)
    if dilation_iter > 0 and dilation_kernel >= 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (dilation_kernel, dilation_kernel))
        result = cv2.dilate(result, kernel, iterations=dilation_iter)
    return result


# -- 区域 ---------------------------------------------------------------

def _shape_points(shape: Mapping[str, Any]) -> list[tuple[float, float]]:
    return [(float(point[0]), float(point[1]))
            for point in shape.get("points") or []
            if isinstance(point, (list, tuple)) and len(point) >= 2]


def rasterize_shape(shape: Mapping[str, Any], width: int, height: int,
                    canvas: np.ndarray | None = None) -> np.ndarray:
    mask = canvas if canvas is not None else np.zeros((height, width), dtype=np.uint8)
    if not isinstance(shape, Mapping):
        return mask
    kind = shape.get("type")
    if kind in ("polygon", "quad"):
        points = np.array(_shape_points(shape), dtype=np.int32)
        if len(points) >= 3:
            cv2.fillPoly(mask, [points], 255)
    elif kind == "ellipse":
        center = shape.get("center")
        radii = shape.get("radii")
        if isinstance(center, (list, tuple)) and radii:
            cv2.ellipse(mask,
                        (int(round(center[0])), int(round(center[1]))),
                        (max(1, int(round(float(radii[0])))),
                         max(1, int(round(float(radii[1]))))),
                        float(shape.get("angle", 0.0)), 0, 360, 255, -1)
        elif shape.get("points"):
            from . import geometry as geo

            fitted = geo.fit_ellipse(shape["points"])
            cv2.ellipse(mask,
                        (int(round(fitted["center"][0])), int(round(fitted["center"][1]))),
                        (max(1, int(round(fitted["radii"][0]))),
                         max(1, int(round(fitted["radii"][1])))),
                        float(fitted["angle"]), 0, 360, 255, -1)
    elif kind == "union":
        for part in shape.get("shapes") or []:
            rasterize_shape(part, width, height, mask)
    elif kind == "point":
        for x, y in _shape_points(shape)[:1]:
            cv2.circle(mask, (int(round(x)), int(round(y))), 1, 255, -1)
    elif kind == "line":
        points = _shape_points(shape)
        if len(points) >= 2:
            cv2.line(mask, tuple(int(round(v)) for v in points[0]),
                     tuple(int(round(v)) for v in points[1]), 255, 1)
    return mask


def _polygon_union(geometry: Mapping[str, Any], keys: Sequence[str],
                   width: int, height: int) -> np.ndarray | None:
    canvas = np.zeros((height, width), dtype=np.uint8)
    used = False
    for key in keys:
        shape = geometry.get(key)
        if isinstance(shape, Mapping):
            rasterize_shape(shape, width, height, canvas)
            used = True
    return canvas if used else None


def geometry_region(geometry: Mapping[str, Any] | None, assay: str,
                    options: Mapping[str, Any] | None,
                    width: int, height: int) -> tuple[np.ndarray, str]:
    """按合同选择检测区域；缺区域报错，不退回全图。"""

    geometry = geometry or {}
    options = options or {}

    explicit = geometry.get("mask")
    if isinstance(explicit, Mapping) and explicit.get("type"):
        return rasterize_shape(explicit, width, height), "mask"

    if assay == "ldb":
        light = geometry.get("light")
        if not isinstance(light, Mapping):
            raise DetectionError("缺少明室区域")
        region = rasterize_shape(light, width, height)
        dark = geometry.get("dark")
        if options.get("dark_visible") and isinstance(dark, Mapping):
            rasterize_shape(dark, width, height, region)
            source = "light+dark"
        else:
            source = "light"
        door = geometry.get("door")
        if isinstance(door, Mapping) and _shape_points(door):
            cv2.fillPoly(region, [np.array(_shape_points(door), dtype=np.int32)], 0)
        return region, source

    if assay in ("epm", "ymaze"):
        zones = geometry.get("zones")
        if isinstance(zones, Mapping) and zones:
            canvas = np.zeros((height, width), dtype=np.uint8)
            for shape in zones.values():
                rasterize_shape(shape, width, height, canvas)
            return canvas, "zones"
        keys = ("center", "open_arm_1", "open_arm_2", "closed_arm_1",
                "closed_arm_2", "arm_a", "arm_b", "arm_c")
        union = _polygon_union(geometry, keys, width, height)
        if union is not None:
            return union, "zones"

    for key in ("arena", "outer"):
        shape = geometry.get(key)
        if isinstance(shape, Mapping) and (shape.get("points") or shape.get("center")):
            return rasterize_shape(shape, width, height), key

    raise DetectionError("缺少检测区域")


# -- 候选 ---------------------------------------------------------------

def _saturated(raw: np.ndarray, region: np.ndarray) -> bool:
    region_count = int(cv2.countNonZero(region))
    if region_count == 0:
        return False
    return cv2.countNonZero(raw) >= SATURATION_RATIO * region_count


def _candidate(outer: np.ndarray, holes: list[np.ndarray],
               width: int, height: int) -> dict | None:
    area = float(cv2.contourArea(outer)) - sum(
        float(cv2.contourArea(hole)) for hole in holes
    )
    if area <= 0:
        return None

    moments = cv2.moments(outer)
    m00 = moments["m00"]
    m10 = moments["m10"]
    m01 = moments["m01"]
    for hole in holes:
        hole_moments = cv2.moments(hole)
        m00 -= hole_moments["m00"]
        m10 -= hole_moments["m10"]
        m01 -= hole_moments["m01"]
    if m00 <= 0:
        return None

    hull_area = float(cv2.contourArea(cv2.convexHull(outer)))
    _, (w, h), _ = cv2.minAreaRect(outer)
    w, h = float(w), float(h)
    extent = min(1.0, area / (w * h)) if w * h > 0 else 0.0
    aspect = max(w, h) / max(min(w, h), 1.0)
    if hull_area <= 0:
        quality = 0.0
    else:
        quality = (min(1.0, area / hull_area) * math.sqrt(extent)
                   * min(1.0, 4.0 / aspect))

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(mask, [outer], -1, 255, -1)
    for hole in holes:
        cv2.drawContours(mask, [hole], -1, 0, -1)

    x, y, bw, bh = cv2.boundingRect(outer)
    contour = [[float(point[0][0]), float(point[0][1])] for point in outer]
    return {
        "center": (m10 / m00, m01 / m00),
        "contour": contour,
        "mask": mask,
        "area": area,
        "bbox": [float(x), float(y), float(bw), float(bh)],
        "quality": max(0.0, min(1.0, quality)),
    }


def find_candidates(foreground: np.ndarray, region: np.ndarray,
                    settings: Mapping[str, Any]) -> tuple[list[dict], str | None]:
    """前景∩区域 → 饱和拒绝 → 形态学 → 再次饱和拒绝 → 轮廓候选。"""

    raw = cv2.bitwise_and(foreground, region)
    if _saturated(raw, region):
        return [], "foreground_saturated"
    processed = apply_morphology(raw, settings)
    processed = cv2.bitwise_and(processed, region)
    if _saturated(processed, region):
        return [], "foreground_saturated"

    contours, hierarchy = cv2.findContours(processed, cv2.RETR_CCOMP,
                                           cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return [], "no_candidate"

    height, width = processed.shape[:2]
    min_area = float(settings.get("min_area", 0.0))
    max_area = float(settings.get("max_area", 1e9))
    quality_threshold = float(settings.get("quality_threshold", 0.0))

    candidates: list[dict] = []
    for index, contour in enumerate(contours):
        if hierarchy[0][index][3] != -1:
            continue
        holes = [contours[child] for child in range(len(contours))
                 if hierarchy[0][child][3] == index]
        item = _candidate(contour, holes, width, height)
        if item is None:
            continue
        if item["area"] < min_area or item["area"] > max_area:
            continue
        if item["quality"] < quality_threshold:
            continue
        candidates.append(item)

    candidates.sort(key=lambda item: item["area"], reverse=True)
    if not candidates:
        return [], "no_candidate"
    return candidates, None


def _detection_from(candidates: list[dict], reason: str | None) -> Detection:
    if not candidates:
        return Detection(center=None, missing_reason=reason, candidates=[])
    best = candidates[0]
    return Detection(
        center=[float(best["center"][0]), float(best["center"][1])],
        quality=best["quality"],
        contour=best["contour"], mask=best["mask"], area=best["area"],
        bbox=best["bbox"], candidates=candidates,
    )


def detect_frame(image: Any, settings: Mapping[str, Any],
                 region: np.ndarray | None = None) -> Detection:
    """对单帧生成 Detection；center 取面积最大的候选，不作身份关联。"""

    gray = to_gray(image)
    height, width = gray.shape[:2]
    if region is None:
        region = np.full((height, width), 255, dtype=np.uint8)
    resolved = merge_settings(settings)
    method = resolved["bg_method"]
    if method == "threshold":
        foreground = threshold_foreground(gray, resolved)
    elif method == "local_otsu":
        foreground = local_otsu_foreground(gray, resolved["bg_type"])
    else:
        raise DetectionError("该背景方法需要背景模型")

    candidates, reason = find_candidates(foreground, region, resolved)
    return _detection_from(candidates, reason)


def detect_with_background(image: Any, background: np.ndarray,
                           settings: Mapping[str, Any],
                           region: np.ndarray | None = None) -> Detection:
    gray = to_gray(image)
    resolved = merge_settings(settings)
    height, width = gray.shape[:2]
    if background.shape[:2] != (height, width):
        raise DetectionError("背景尺寸与源帧不一致")
    if region is None:
        region = np.full((height, width), 255, dtype=np.uint8)
    foreground = temporal_foreground(gray, background,
                                     resolved["bg_type"],
                                     resolved["temporal_diff_threshold"])
    candidates, reason = find_candidates(foreground, region, resolved)
    return _detection_from(candidates, reason)


# -- 背景模型 -----------------------------------------------------------

def load_static_background(path: str | Path, orientation: int,
                           width: int, height: int) -> np.ndarray:
    """按字节读取并解码静态背景，旋转后尺寸必须与源帧一致。"""

    file_path = Path(path)
    if not file_path.is_file():
        raise DetectionError("静态背景不可读取")
    data = np.fromfile(str(file_path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise DetectionError("静态背景无法解码")
    if orientation == 90:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 180:
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif orientation == 270:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if image.shape[1] != width or image.shape[0] != height:
        raise DetectionError("静态背景尺寸与源帧不一致")
    return to_gray(image)


def _median_strips(stack: np.ndarray) -> np.ndarray:
    height, width = stack.shape[1:]
    result = np.empty((height, width), dtype=np.uint8)
    for top in range(0, height, STRIP_ROWS):
        bottom = min(height, top + STRIP_ROWS)
        block = np.median(stack[:, top:bottom, :], axis=0)
        result[top:bottom, :] = np.clip(block, 0, 255).astype(np.uint8)
    return result


def temporal_background(frames: Iterable[tuple[float, Any]],
                        samples: int, start: float, end: float | None,
                        *, cancel=None) -> np.ndarray:
    """均匀目标时间采集真实帧；临时磁盘映射按条带求中位数。"""

    if samples < 1:
        raise DetectionError("背景样本数必须 ≥ 1")
    targets = None
    if end is not None and end > start:
        targets = [start + (end - start) * index / samples
                   for index in range(samples)]

    handle, temp_path = tempfile.mkstemp(suffix=".bg", prefix="blnext-")
    os.close(handle)
    stack = None
    used = 0
    target_index = 0
    try:
        for time_s, image in frames:
            if cancel is not None and cancel():
                raise DetectionError("已取消")
            gray = image if np.asarray(image).ndim == 2 else to_gray(image)
            if stack is None:
                height, width = gray.shape[:2]
                stack = np.memmap(temp_path, dtype=np.uint8, mode="w+",
                                  shape=(samples, height, width))
            if targets is None:
                stack[used] = gray
                used += 1
            else:
                sample_time = max(float(time_s), float(start))
                if (target_index < len(targets)
                        and sample_time + 1e-9 >= targets[target_index]):
                    stack[used] = gray
                    used += 1
                    target_index += 1
            if targets is None and used >= samples:
                break
            if targets is not None and target_index >= len(targets):
                break
        if stack is None or used == 0:
            raise DetectionError("没有可用的背景样本")
        return _median_strips(np.asarray(stack[:used]))
    finally:
        stack = None
        try:
            os.unlink(temp_path)
        except OSError:
            pass


class ClassicalDetector:
    """同一配置复用背景模型与原始前景缓存；按帧只缓存原始前景。"""

    def __init__(self, settings: Mapping[str, Any] | None = None, *,
                 background: np.ndarray | None = None) -> None:
        self.settings = normalize_settings(settings)
        self.background = background
        self._subtractor = None
        self._foreground_cache: dict[tuple, np.ndarray] = {}
        method = self.settings["bg_method"]
        if method in ("mog2", "knn"):
            factory = (cv2.createBackgroundSubtractorMOG2 if method == "mog2"
                       else cv2.createBackgroundSubtractorKNN)
            self._subtractor = factory(history=500, detectShadows=False)

    def _signature(self) -> tuple:
        keys = ("bg_method", "bg_type", "gray_threshold", "temporal_bg_samples",
                "temporal_diff_threshold", "static_background")
        return tuple(self.settings[key] for key in keys)

    def foreground(self, image: Any) -> np.ndarray:
        gray = to_gray(image)
        cache_key = (id(image), self._signature())
        cached = self._foreground_cache.get(cache_key)
        if cached is not None:
            return cached

        method = self.settings["bg_method"]
        if method == "threshold":
            mask = threshold_foreground(gray, self.settings)
        elif method == "local_otsu":
            mask = local_otsu_foreground(gray, self.settings["bg_type"])
        elif method in ("temporal", "static"):
            if self.background is None:
                raise DetectionError("缺少背景模型")
            if self.background.shape[:2] != gray.shape[:2]:
                raise DetectionError("背景尺寸与源帧不一致")
            mask = temporal_foreground(gray, self.background,
                                       self.settings["bg_type"],
                                       self.settings["temporal_diff_threshold"])
        else:
            mask = self._subtractor.apply(gray)
            if cv2.countNonZero(mask) >= SHADOW_FRACTION * mask.size:
                mask = np.zeros_like(mask)
        self._foreground_cache[cache_key] = mask
        if len(self._foreground_cache) > 8:
            self._foreground_cache.pop(next(iter(self._foreground_cache)))
        return mask

    def detect(self, image: Any, region: np.ndarray) -> Detection:
        foreground = self.foreground(image)
        candidates, reason = find_candidates(foreground, region, self.settings)
        return _detection_from(candidates, reason)


def select_candidate(candidates: Iterable[Mapping[str, Any]] | None) -> dict | None:
    """按 quality 选择最佳候选；空列表返回 None。"""

    best: dict | None = None
    best_score = float("-inf")
    for candidate in candidates or ():
        if not isinstance(candidate, Mapping):
            continue
        score = candidate.get("quality", 0)
        if not is_number(score):
            continue
        if best is None or float(score) > best_score:
            best = dict(candidate)
            best_score = float(score)
    return best


_SHARED_DETECTORS: dict[tuple, "ClassicalDetector"] = {}


def shared_detector(key: tuple, settings: Mapping[str, Any],
                    background: np.ndarray | None = None) -> "ClassicalDetector":
    """MOG2/KNN 同配置复用学习状态；其他方法每次新建，避免背景占内存。"""

    resolved = normalize_settings(settings)
    if resolved["bg_method"] not in ("mog2", "knn"):
        return ClassicalDetector(resolved, background=background)
    signature = tuple(resolved[name] for name in (
        "bg_method", "bg_type", "gray_threshold", "temporal_bg_samples",
        "temporal_diff_threshold", "static_background"))
    cache_key = (key, signature)
    detector = _SHARED_DETECTORS.get(cache_key)
    if detector is None:
        detector = ClassicalDetector(resolved, background=background)
        _SHARED_DETECTORS[cache_key] = detector
        if len(_SHARED_DETECTORS) > 4:
            _SHARED_DETECTORS.pop(next(iter(_SHARED_DETECTORS)))
    return detector
