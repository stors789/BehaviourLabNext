"""经典前景检测：参数、背景、区域、候选与质量。

本模块只做单帧候选；跨帧关联留给后续块。前景像素不等于目标动物，
候选按面积降序返回，缺失时给出明确原因。
"""

from __future__ import annotations

import math
import os
import tempfile
import threading
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import cv2
import numpy as np

from .schema import Detection, is_number

BG_METHODS = ("threshold", "temporal", "static", "mog2", "knn", "local_otsu")
FOREGROUND_KEYS = ("bg_method", "bg_type", "gray_threshold", "temporal_bg_samples",
                   "temporal_diff_threshold", "static_background")
BG_TYPES = ("light", "dark")

DETECTION_DEFAULTS: dict = {
    "bg_method": "threshold",
    "bg_type": "light",
    "gray_threshold": 127.0,
    "body_gray_threshold": 95.0,
    "erosion_kernel": 3,
    "erosion_iter": 1,
    "dilation_kernel": 5,
    "dilation_iter": 1,
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

# 时序背景的算法版本：取样/中位数规则一旦变化就必须改，缓存键据此失效。
TEMPORAL_BG_VERSION = 1
# 背景缓存按条目与字节双上限，避免大图、多视频累积。
BACKGROUND_CACHE_ENTRIES = 8
BACKGROUND_CACHE_BYTES = 96 * 1024 * 1024
# 同键并发构建时等待对方完成的上限；超时明确报错，不无限阻塞。
BACKGROUND_WAIT_SECONDS = 900.0
BACKGROUND_POLL_SECONDS = 0.05


class DetectionError(RuntimeError):
    """检测无法进行：区域、背景或参数不满足要求。"""


class BackgroundUnavailableError(DetectionError):
    """背景内容不可用，预览可以降级的一类失败。

    只覆盖"换一份可解码的背景/视频就能继续"的内容性问题：时序背景
    视频无法解码、静态背景未设置/文件缺失/无法解码/尺寸不符。
    源身份变化、共享构建等待超时、取消等流程性失败仍用 DetectionError
    本类抛出，调用方不得据此降级。
    """


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
    dilation_iter = int(settings.get("dilation_iter", 1))
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

    if assay == "barnes":
        # 显式检测遮罩就是检测区域；只有没有 mask 时才用圆盘。统计圆盘与
        # 物理尺度仍由 ellipse + 直径决定，不随检测区域改变。
        explicit = geometry.get("mask")
        if isinstance(explicit, Mapping) and explicit.get("type") == "polygon" \
                and len(_shape_points(explicit)) >= 3:
            return rasterize_shape(explicit, width, height), "mask"
        arena = geometry.get("arena")
        if not isinstance(arena, Mapping):
            raise DetectionError("缺少圆盘边界")
        return rasterize_shape(arena, width, height), "arena"

    explicit = geometry.get("mask")
    if isinstance(explicit, Mapping) and explicit.get("type"):
        return rasterize_shape(explicit, width, height), "mask"

    if assay == "ldb":
        zones = geometry.get("zones")
        if not isinstance(zones, Mapping):
            zones = {key: geometry.get(key) for key in ("light", "dark", "door")}
        light = zones.get("light")
        if not isinstance(light, Mapping):
            raise DetectionError("缺少明室区域")
        region = rasterize_shape(light, width, height)
        source = "light"
        # 可见模式才把暗室纳入检测范围；不可见时只看明室，失检由评分层推断。
        if options.get("dark_box_visible") and isinstance(zones.get("dark"),
                                                          Mapping):
            rasterize_shape(zones["dark"], width, height, region)
            source = "light+dark"
        door = zones.get("doorway") or zones.get("door")
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
        # 场地范围不足以代表 EPM/Y 的分区；不退回 arena，也不退回全图。
        raise DetectionError("缺少检测区域")

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


def _component_label(outer: np.ndarray, labels: np.ndarray | None) -> int:
    """轮廓点落在连通体边界上；取出现最多的非零标签作为该候选的实体。"""

    if labels is None:
        return 0
    points = outer.reshape(-1, 2)
    values = labels[points[:, 1], points[:, 0]]
    values = values[values > 0]
    if values.size == 0:
        return 0
    counts = np.bincount(values.astype(np.int64))
    return int(np.argmax(counts))


def _candidate(outer: np.ndarray, holes: list[np.ndarray],
               labels: np.ndarray | None,
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
    label = _component_label(outer, labels)
    x, y, bw, bh = cv2.boundingRect(outer)
    if label:
        # 直接取真实连通体像素：洞内背景与内部独立候选都不属于该实体，
        # 也不会像重绘孔洞轮廓那样削掉内边缘的一圈墙。
        window = mask[y:y + bh, x:x + bw]
        window[labels[y:y + bh, x:x + bw] == label] = 255
    else:
        cv2.drawContours(mask, [outer], -1, 255, -1)
        for hole in holes:
            cv2.drawContours(mask, [hole], -1, 0, -1)

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
    labels = cv2.connectedComponents(processed, connectivity=8)[1]
    min_area = float(settings.get("min_area", 0.0))
    max_area = float(settings.get("max_area", 1e9))
    quality_threshold = float(settings.get("quality_threshold", 0.0))

    candidates: list[dict] = []
    for index, contour in enumerate(contours):
        if hierarchy[0][index][3] != -1:
            continue
        holes = [contours[child] for child in range(len(contours))
                 if hierarchy[0][child][3] == index]
        item = _candidate(contour, holes, labels, width, height)
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
        raise BackgroundUnavailableError("静态背景不可读取")
    data = np.fromfile(str(file_path), dtype=np.uint8)
    if data.size == 0:
        # 0 字节文件：cv2.imdecode 对空数组会断言失败（500），先明确拒绝。
        raise BackgroundUnavailableError("静态背景为空文件")
    try:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except cv2.error as error:
        raise BackgroundUnavailableError(f"静态背景无法解码：{error}") from error
    if image is None:
        raise BackgroundUnavailableError("静态背景无法解码")
    if orientation == 90:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 180:
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif orientation == 270:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if image.shape[1] != width or image.shape[0] != height:
        raise BackgroundUnavailableError("静态背景尺寸与源帧不一致")
    return to_gray(image)


def interpreted_duration(metadata: Mapping[str, Any],
                         video: Mapping[str, Any]) -> float | None:
    """把已知时长换到与 Frame.time_s 相同的解释时钟；无法换算时返回 None。

    有合法解释帧率且帧数可信时，帧数/解释帧率本身就能定出终点，不要求
    同时有原容器时长；缺帧数时才退回“原时长×原帧率/解释帧率”。
    """

    override = video.get("fps_override")
    valid_override = is_number(override) and float(override) > 0
    frames = metadata.get("frame_count")
    if valid_override and is_number(frames) and float(frames) > 0:
        # 覆盖帧率下帧时间按 index/fps_override 解释，原始容器时长不能直接使用。
        return float(frames) / float(override)

    duration = metadata.get("duration_s")
    if not is_number(duration) or float(duration) <= 0:
        return None
    seconds = float(duration)
    if not valid_override:
        return seconds
    source_fps = metadata.get("fps")
    if is_number(source_fps) and float(source_fps) > 0:
        return seconds * float(source_fps) / float(override)
    return None


def prepare_background(settings: Mapping[str, Any], asset: Mapping[str, Any],
                       video: Mapping[str, Any], orientation: int,
                       width: int, height: int, *,
                       cancel: Callable[[], bool] | None = None):
    """按方法准备背景；threshold/local_otsu 不需要。预览与分析共用。

    temporal 走“先按采样规则定目标源帧、再只对这些帧取像素”的两段式，
    并对同一背景事实做有界复用；其余方法与取消语义保持原样。
    """

    from . import media

    method = settings.get("bg_method")
    if method == "static":
        path = settings.get("static_background")
        if not path:
            raise BackgroundUnavailableError("静态背景未设置")
        return load_static_background(path, orientation, width, height)
    if method == "temporal":
        metadata = asset.get("metadata") or {}
        window = video.get("window") or {}
        start = float(window.get("start_s") or 0.0)
        end = window.get("end_s")
        end_value = float(end) if is_number(end) else None
        # 窗口终点未写时用同一解释时钟的已知时长安排采样；时长也未知才取前 N 帧。
        sample_end = end_value
        known_end = interpreted_duration(metadata, video)
        if sample_end is None and known_end is not None and known_end > start:
            sample_end = known_end
        samples = int(settings.get("temporal_bg_samples", 90))
        prepared = _temporal_background_key(
            asset, video, metadata, orientation, width, height,
            start, end_value, sample_end, samples)
        if prepared is None:
            # 源身份不可归属：现建现用，不写缓存，也不采纳任何索引证据。
            return _build_temporal_background(
                asset, video, orientation, start, end_value, sample_end,
                samples, cancel, expected_identity=None)

        key, expected_identity = prepared

        def build():
            return _build_temporal_background(
                asset, video, orientation, start, end_value, sample_end,
                samples, cancel, expected_identity=expected_identity)

        return _cached_background(key, build, cancel,
                                  path=asset.get("path"),
                                  expected_identity=expected_identity)
    return None


# -- 时序背景：取样规则（单一定义，规划与建栈共用） --------------------
def temporal_sample_targets(samples: int, start: float,
                            end: float | None) -> list[float] | None:
    """按有效区间均分的目标时刻；终点未知返回 None（取前 N 个有效帧）。"""

    if end is not None and end > start:
        return [start + (end - start) * index / samples
                for index in range(samples)]
    return None


def _sample_reached(sample_time: float, targets: list[float] | None,
                    taken: int) -> bool:
    """已取 taken 个样本时，这一帧是否到达下一个目标时刻。"""

    return (targets is not None and taken < len(targets)
            and sample_time + 1e-9 >= targets[taken])


def temporal_frame_plan(
    source: Any, *, start: float, sample_end: float | None, samples: int,
    cancel: Callable[[], bool] | None = None,
) -> tuple[list[tuple[int, float, float]], Any]:
    """在已构造的采样源上确定要用的真实源帧：只读时刻与曝光，不取像素。

    返回 ([(index, time_s, 有效曝光)], 索引证据)。证据可在同文件、同时钟
    模式下交给取像素的读取器，避免二次整段扫描。
    """

    if samples < 1:
        raise DetectionError("背景样本数必须 ≥ 1")
    targets = temporal_sample_targets(samples, start, sample_end)
    plan: list[tuple[int, float, float]] = []
    for frame in source.iter_frames():
        if cancel is not None and cancel():
            raise DetectionError("已取消")
        exposure = frame.interval_end_s - frame.interval_start_s
        if exposure <= 0:
            continue
        if targets is None:
            plan.append((frame.index, float(frame.time_s), exposure))
            if len(plan) >= samples:
                break
            continue
        sample_time = max(float(frame.time_s), float(start))
        if _sample_reached(sample_time, targets, len(plan)):
            plan.append((frame.index, float(frame.time_s), exposure))
            if len(plan) >= len(targets):
                break
    evidence = source.index_evidence()
    if not plan:
        raise DetectionError("没有可用的背景样本")
    return plan, evidence


def _build_temporal_background(asset: Mapping[str, Any],
                               video: Mapping[str, Any], orientation: int,
                               start: float, end_value: float | None,
                               sample_end: float | None, samples: int,
                               cancel: Callable[[], bool] | None = None,
                               expected_identity: tuple | None = None):
    """规划目标帧 → 只对这些帧取方向像素与灰度 → 同一中位数规则。

    三个量互相独立，不互相代替：源帧解码次数（规划扫描 + 定位补解）、
    目标像素转换次数（应等于实际样本数）、背景样本数（median 的帧数）。
    """

    from . import media

    path = asset.get("path")
    metadata = asset.get("metadata") or {}
    pool = media.ReaderPool(max_readers=2)
    try:
        # 规划、取像素与发布必须属于同一份源。``expected_identity`` 来自缓存
        # key 的计算时刻：入口先按它核对一次，避免把另一份文件的背景写回旧 key。
        if expected_identity is None:
            try:
                source_identity = media.identity_key(media.file_identity(path))
            except (OSError, TypeError):
                # 身份不可归属（路径不存在或测试替身）：不做绑定核对，也不采纳
                # 任何旧索引证据；真实读取仍由 ReaderPool 拒绝不可读的源。
                source_identity = ()
        else:
            source_identity = tuple(expected_identity)
            if source_identity:
                _require_same_source(media, path, source_identity, "背景构建入口")

        def _decode_io(action, *, when: str):
            """把源解码失败转为背景内容不可用（预览可降级）。

            只包媒体 IO；身份核对（``_require_same_source``，含文件不可
            再访问）与取消走各自原有异常，保持失败/取消语义。
            时钟不可解释（ClockError）与覆盖不足（VideoSourceError）是
            正确性错误，不是"换份背景就能继续"，原样上抛。
            访问被拒绝（如 Windows 独占打开）按异常类型/errno 结构化判定，
            同样原样上抛；源文件不可 stat 也属于访问失败。
            """

            try:
                return action()
            except (media.ClockError, media.VideoSourceError):
                raise
            except media.MediaError as error:
                if media.is_access_denied(error):
                    # 换一份文件才可能继续：不是内容问题，明确失败。
                    raise error
                try:
                    media.file_identity(path)
                except OSError:
                    # 源文件不可访问：重抛原始 MediaError（访问类失败），
                    # 不是嵌套 try 里的新 OSError。
                    raise error
                raise BackgroundUnavailableError(
                    f"背景视频无法解码（{when}）：{error}") from error

        # 规划阶段只读时刻/曝光：解码但不转 BGR、不复制方向像素。
        source = _decode_io(
            lambda: media.VideoSource(
                path, start_s=start, end_s=end_value, orientation=orientation,
                fps_override=video.get("fps_override"),
                sample_fps=video.get("sample_fps"),
                source_fps=metadata.get("fps"),
                decode_images=False, pool=pool,
            ),
            when="规划扫描")
        plan, evidence = _decode_io(
            lambda: temporal_frame_plan(
                source, start=start, sample_end=sample_end, samples=samples,
                cancel=cancel),
            when="规划扫描")
        if source_identity:
            _require_same_source(media, path, source_identity,
                                 "背景规划与取像素之间")
        if cancel is not None and cancel():
            raise DetectionError("已取消")
        # 只对确定下来的源帧取像素：中间帧不再转 BGR、复制或转灰。
        pixels = _decode_io(
            lambda: source.frames_at([item[0] for item in plan], orientation,
                                     evidence),
            when="取像素")

        def frames():
            for entry, image in zip(plan, pixels):
                if cancel is not None and cancel():
                    raise DetectionError("已取消")
                yield entry[1], image, entry[2]

        # 取像素是生成器：解码失败往往在消费时才抛（FrameReadError），
        # 所以包装必须覆盖实际迭代，而不只是生成器创建。
        background = _decode_io(
            lambda: temporal_background(frames(), samples, start, sample_end,
                                        cancel=cancel),
            when="取像素")
        if source_identity:
            _require_same_source(media, path, source_identity, "背景构建期间")
        return background
    finally:
        pool.close_all()


def _require_same_source(media_module: Any, path: Any, identity: tuple,
                         when: str) -> None:
    """源身份变了就明确失败：不把旧时间/索引与新像素混成一份背景。"""

    try:
        current = media_module.identity_key(media_module.file_identity(path))
    except (OSError, TypeError) as error:
        raise media_module.MediaError(
            f"源文件在{when}不可再访问：{path}") from error
    if current != identity:
        raise DetectionError(
            f"源文件在{when}发生变化（身份不同），已放弃本次背景构建")


# -- 背景复用：同语义键有界缓存 + 同键并发受控 ------------------------
class _PendingBackground:
    __slots__ = ("event", "error")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.error: BaseException | None = None


_BACKGROUND_CACHE: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
_BACKGROUND_BYTES = 0
_BACKGROUND_INFLIGHT: dict[tuple, _PendingBackground] = {}
_BACKGROUND_LOCK = threading.Lock()


def _temporal_background_key(asset: Mapping[str, Any], video: Mapping[str, Any],
                             metadata: Mapping[str, Any], orientation: int,
                             width: int, height: int, start: float,
                             end_value: float | None,
                             sample_end: float | None,
                             samples: int) -> tuple[tuple, tuple] | None:
    """只含影响背景像素的事实：同一化后相同才可命中。

    返回 ``(key, expected_identity)``：身份在**取 key 的这一刻**唯一确定，
    之后从缓存命中、构建入口、规划取像素到发布都要拿它核对，不能换一份
    文件继续用同一个 key。

    源文件身份取不到（文件缺失/不可 stat）时返回 None：这种路径无法安全
    归属，一律现建不缓存，也不与别的输入共用条目。
    """

    from . import media

    path = asset.get("path")
    try:
        identity_key = media.identity_key(media.file_identity(path))
    except (OSError, TypeError):
        return None
    if not identity_key:
        return None
    override = video.get("fps_override")
    if is_number(override) and float(override) > 0:
        fps_mode = ("override", float(override))
    else:
        fps_mode = ("native", None)
    number = lambda value: float(value) if is_number(value) else None
    key = (
        "temporal-background", TEMPORAL_BG_VERSION, identity_key,
        int(orientation), fps_mode,
        ("sample_fps", number(video.get("sample_fps"))),
        ("window", float(start), number(end_value)),
        ("samples", int(samples)),
        ("targets_end", number(sample_end)),
        # 用于末端推导与采样判定的元数据事实
        ("metadata", number(metadata.get("fps")),
         number(metadata.get("frame_count")), number(metadata.get("duration_s"))),
        ("frame", int(width), int(height)),
    )
    return key, identity_key


def _cache_store(key: tuple, value: np.ndarray) -> None:
    global _BACKGROUND_BYTES
    _BACKGROUND_BYTES += int(value.nbytes)
    _BACKGROUND_CACHE[key] = value
    _BACKGROUND_CACHE.move_to_end(key)
    while _BACKGROUND_CACHE and (
        len(_BACKGROUND_CACHE) > BACKGROUND_CACHE_ENTRIES
        or _BACKGROUND_BYTES > BACKGROUND_CACHE_BYTES
    ):
        _, removed = _BACKGROUND_CACHE.popitem(last=False)
        _BACKGROUND_BYTES -= int(removed.nbytes)


def _verify_expected_identity(path: Any, expected_identity: tuple,
                              when: str) -> None:
    """key 里记下的身份必须仍是当前文件的身份；否则拒绝本次结果。

    身份为空（不可 stat 的路径或测试替身）时不做绑定核对：这类请求本来
    就不进缓存。承诺范围只覆盖 `file_identity` 里可观测的字段。
    """

    if not expected_identity or not isinstance(path, (str, Path)):
        return
    from . import media

    _require_same_source(media, path, expected_identity, when)


def _cached_background(key: tuple, build: Callable[[], np.ndarray],
                       cancel: Callable[[], bool] | None = None, *,
                       path: Any = None,
                       expected_identity: tuple = ()) -> np.ndarray:
    """同键复用；并发同键只建一次，失败/取消不发布半成品。

    身份从取 key 那一刻起贯穿：命中返回前、构建入口、发布前、等待者取回
    结果前都按同一个 expected_identity 核对；任何一处不符即拒绝本次请求，
    既不返回、也不写入、也不覆盖旧条目。
    """

    with _BACKGROUND_LOCK:
        hit = _BACKGROUND_CACHE.get(key)
        if hit is not None:
            _BACKGROUND_CACHE.move_to_end(key)
            _verify_expected_identity(path, expected_identity, "缓存命中返回前")
            return hit
        pending = _BACKGROUND_INFLIGHT.get(key)
        owner = pending is None
        if owner:
            pending = _PendingBackground()
            _BACKGROUND_INFLIGHT[key] = pending
    if not owner:
        waited = 0.0
        while not pending.event.wait(BACKGROUND_POLL_SECONDS):
            waited += BACKGROUND_POLL_SECONDS
            if cancel is not None and cancel():
                raise DetectionError("已取消")
            if waited >= BACKGROUND_WAIT_SECONDS:
                raise DetectionError("等待背景构建超时")
        with _BACKGROUND_LOCK:
            hit = _BACKGROUND_CACHE.get(key)
        if hit is not None:
            _verify_expected_identity(path, expected_identity, "等待背景结果时")
            return hit
        if pending.error is not None:
            raise pending.error
        raise DetectionError("背景构建未完成")

    value = None
    error: BaseException | None = None
    try:
        _verify_expected_identity(path, expected_identity, "背景构建开始前")
        value = np.asarray(build())
        # 缓存数组只读：后续检测不会改到共享背景。
        value.setflags(write=False)
    except BaseException as build_error:  # noqa: BLE001 - 原样上抛
        error = build_error
        raise
    finally:
        with _BACKGROUND_LOCK:
            _BACKGROUND_INFLIGHT.pop(key, None)
            if error is None and value is not None:
                try:
                    _verify_expected_identity(path, expected_identity,
                                              "背景发布前")
                except BaseException as publish_error:  # noqa: BLE001
                    # 发布前换源：丢掉这次的 B 结果，不写 A 的 key。
                    error = publish_error
                else:
                    _cache_store(key, value)
            pending.error = error
        pending.event.set()
    if error is not None and value is not None and isinstance(error, BaseException):
        # build 已成功但发布被拒：把拒绝原因抛给调用者，不返回混合结果。
        raise error
    return value


def background_cache_state() -> tuple[int, int]:
    """诊断用：(条目数, 字节数)。"""

    with _BACKGROUND_LOCK:
        return len(_BACKGROUND_CACHE), int(_BACKGROUND_BYTES)


def clear_background_cache() -> None:
    """清空背景缓存（测试与显式失效用）。"""

    global _BACKGROUND_BYTES
    with _BACKGROUND_LOCK:
        _BACKGROUND_CACHE.clear()
        _BACKGROUND_INFLIGHT.clear()
        _BACKGROUND_BYTES = 0


def _median_strips(stack: np.ndarray, cancel=None) -> np.ndarray | None:
    """逐条带求中位数；取消时返回 None，让调用者先释放映射再报错。"""

    height, width = stack.shape[1:]
    result = np.empty((height, width), dtype=np.uint8)
    for top in range(0, height, STRIP_ROWS):
        if cancel is not None and cancel():
            return None
        bottom = min(height, top + STRIP_ROWS)
        block = np.median(stack[:, top:bottom, :], axis=0)
        result[top:bottom, :] = np.clip(block, 0, 255).astype(np.uint8)
    return result


def _sample_parts(item: Any) -> tuple[float, Any, float]:
    """样本项：(time_s, image) 或 (time_s, image, 有效曝光秒数)。"""

    if len(item) >= 3:
        return float(item[0]), item[1], float(item[2])
    return float(item[0]), item[1], 1.0


def temporal_background(frames: Iterable[tuple[float, Any]],
                        samples: int, start: float, end: float | None,
                        *, cancel=None) -> np.ndarray:
    """按同一源时钟的已知有效终点均匀取真实帧；未知终点才取前 N 帧。

    零曝光的窗口外上下文帧不参与背景；跨起点但有曝光的帧保留。
    """

    if samples < 1:
        raise DetectionError("背景样本数必须 ≥ 1")
    targets = temporal_sample_targets(samples, start, end)

    handle, temp_path = tempfile.mkstemp(suffix=".bg", prefix="blnext-")
    os.close(handle)
    stack = None
    used = 0
    target_index = 0
    try:
        for item in frames:
            if cancel is not None and cancel():
                raise DetectionError("已取消")
            time_s, image, exposure = _sample_parts(item)
            if exposure <= 0:
                continue
            gray = image if np.asarray(image).ndim == 2 else to_gray(image)
            if stack is None:
                height, width = gray.shape[:2]
                stack = np.memmap(temp_path, dtype=np.uint8, mode="w+",
                                  shape=(samples, height, width))
            if targets is None:
                stack[used] = gray
                used += 1
                if used >= samples:
                    break
                continue
            sample_time = max(time_s, float(start))
            if _sample_reached(sample_time, targets, target_index):
                stack[used] = gray
                used += 1
                target_index += 1
            if target_index >= len(targets):
                break
        if stack is None or used == 0:
            raise DetectionError("没有可用的背景样本")
        background = _median_strips(np.asarray(stack[:used]), cancel)
        if background is None:
            stack = None
            raise DetectionError("已取消")
        return background
    finally:
        stack = None
        try:
            os.unlink(temp_path)
        except OSError:
            pass


class ForegroundModel:
    """共享的减除器与原始前景缓存；只含背景参数，不含单位过滤参数。

    缓存的前景数组只读，调用者清零会立即报错，而不是污染下一只小鼠。
    """

    def __init__(self, settings: Mapping[str, Any] | None = None, *,
                 background: np.ndarray | None = None) -> None:
        self.settings = normalize_settings(settings)
        self.background = background
        self.method = self.settings["bg_method"]
        self._subtractor = None
        if self.method in ("mog2", "knn"):
            factory = (cv2.createBackgroundSubtractorMOG2 if self.method == "mog2"
                       else cv2.createBackgroundSubtractorKNN)
            self._subtractor = factory(history=500, detectShadows=False)
        # 值里保留图像引用：id 只在对象存活期间唯一，引用防止地址被复用后误命中。
        self._cache: dict[int, tuple[Any, np.ndarray]] = {}

    def foreground(self, image: Any) -> np.ndarray:
        gray = to_gray(image)
        cache_key = id(image)
        cached = self._cache.get(cache_key)
        if (cached is not None and cached[0] is image
                and cached[1].shape[:2] == gray.shape[:2]):
            return cached[1]

        if self.method == "threshold":
            mask = threshold_foreground(gray, self.settings)
        elif self.method == "local_otsu":
            mask = local_otsu_foreground(gray, self.settings["bg_type"])
        elif self.method in ("temporal", "static"):
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
        mask.setflags(write=False)
        self._cache[cache_key] = (image, mask)
        if len(self._cache) > 8:
            self._cache.pop(next(iter(self._cache)))
        return mask


class ClassicalDetector:
    """单位级管线：区域、形态学与过滤各自独立，只共享原始前景。"""

    def __init__(self, settings: Mapping[str, Any] | None = None, *,
                 background: np.ndarray | None = None,
                 model: "ForegroundModel | None" = None) -> None:
        self.settings = normalize_settings(settings)
        self.model = model or ForegroundModel(self.settings,
                                              background=background)
        self.background = self.model.background

    def foreground(self, image: Any) -> np.ndarray:
        return self.model.foreground(image)

    def detect(self, image: Any, region: np.ndarray) -> Detection:
        candidates, reason = find_candidates(self.foreground(image), region,
                                             self.settings)
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


_SHARED_MODELS: dict[tuple, ForegroundModel] = {}


def shared_detector(key: tuple, settings: Mapping[str, Any],
                    background: np.ndarray | None = None) -> "ClassicalDetector":
    """MOG2/KNN 同配置复用减除器与原始前景；单位参数各自独立。"""

    resolved = normalize_settings(settings)
    if resolved["bg_method"] not in ("mog2", "knn"):
        return ClassicalDetector(resolved, background=background)
    signature = tuple(resolved[name] for name in FOREGROUND_KEYS)
    cache_key = (key, signature)
    model = _SHARED_MODELS.get(cache_key)
    if model is None:
        model = ForegroundModel(resolved, background=background)
        _SHARED_MODELS[cache_key] = model
        if len(_SHARED_MODELS) > 4:
            _SHARED_MODELS.pop(next(iter(_SHARED_MODELS)))
    return ClassicalDetector(resolved, model=model)
