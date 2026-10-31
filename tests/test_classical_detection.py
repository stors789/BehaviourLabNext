"""经典检测三项固定预期：候选、饱和拒绝、极性与孔洞。"""

from __future__ import annotations

import cv2
import numpy as np

from behaviourlab import detection

TOL = 1e-6


def settings(**overrides) -> dict:
    values = dict(detection.DETECTION_DEFAULTS)
    values.update({
        "bg_type": "light",
        "gray_threshold": 127.0,
        "erosion_iter": 0,
        "dilation_iter": 0,
        "min_area": 20.0,
        "max_area": 10000.0,
        "quality_threshold": 0.4,
    })
    values.update(overrides)
    return values


def two_rectangles() -> np.ndarray:
    image = np.full((120, 160), 255, dtype=np.uint8)
    cv2.rectangle(image, (15, 40), (55, 60), 0, -1)
    cv2.rectangle(image, (110, 45), (130, 60), 0, -1)
    return image


def test_two_candidates_and_explicit_mask_priority():
    image = two_rectangles()
    original = image.copy()
    region = np.full((120, 160), 255, dtype=np.uint8)
    candidates, reason = detection.find_candidates(
        detection.threshold_foreground(image, settings()), region, settings()
    )
    assert reason is None
    assert len(candidates) == 2
    centers = [candidate["center"] for candidate in candidates]
    assert abs(centers[0][0] - 35.0) <= TOL
    assert abs(centers[0][1] - 50.0) <= TOL
    assert abs(centers[1][0] - 120.0) <= TOL
    assert abs(centers[1][1] - 52.5) <= TOL

    # 显式 mask 优先于 arena，即使 arena 在另一侧。
    geometry = {
        "arena": {"type": "quad",
                  "points": [[0, 0], [65, 0], [65, 119], [0, 119]]},
        "mask": {"type": "polygon",
                 "points": [[90, 0], [159, 0], [159, 119], [90, 119]]},
    }
    mask, source = detection.geometry_region(geometry, "tst", {}, 160, 120)
    assert source == "mask"
    masked, _ = detection.find_candidates(
        detection.threshold_foreground(image, settings()), mask, settings()
    )
    assert len(masked) == 1
    assert abs(masked[0]["center"][0] - 120.0) <= TOL
    assert abs(masked[0]["center"][1] - 52.5) <= TOL

    # 源数组逐值不变。
    assert np.array_equal(image, original)


def noisy_frame() -> np.ndarray:
    image = np.full((120, 160), 110, dtype=np.uint8)
    image[:10, :] = 255
    cv2.rectangle(image, (52, 1), (98, 8), 0, -1)
    cv2.ellipse(image, (118, 70), (14, 20), 0, 0, 360, 0, -1)
    return image


def test_saturation_and_threshold_recovery():
    image = noisy_frame()
    region = np.full((120, 160), 255, dtype=np.uint8)

    saturated = detection.detect_frame(image, settings(gray_threshold=127.0), region)
    assert saturated.center is None
    assert saturated.mask is None
    assert saturated.missing_reason == "foreground_saturated"

    recovered = detection.detect_frame(image, settings(gray_threshold=50.0), region)
    assert recovered.center is not None
    assert abs(recovered.center[0] - 118.0) <= 1.0
    assert abs(recovered.center[1] - 70.0) <= 1.0


def test_dilation_saturation_is_rejected():
    dots = np.full((120, 160), 255, dtype=np.uint8)
    for y in range(4, 120, 8):
        for x in range(4, 160, 8):
            dots[y, x] = 0
    region = np.full((120, 160), 255, dtype=np.uint8)
    values = settings(dilation_kernel=17, dilation_iter=1)
    candidates, reason = detection.find_candidates(
        detection.threshold_foreground(dots, values), region, values
    )
    assert candidates == []
    assert reason == "foreground_saturated"


def test_signed_polarity_and_constant_tiles():
    gray = np.array([[80]], dtype=np.uint8)
    background = np.array([[180]], dtype=np.uint8)
    foreground = detection.temporal_foreground(gray, background, "light", 25.0)
    assert int(foreground[0, 0]) == 255
    # 旧位置恢复到亮地板，不得产生反极性鬼影。
    ghost = detection.temporal_foreground(background, gray, "light", 25.0)
    assert int(ghost[0, 0]) == 0
    dark = detection.temporal_foreground(background, gray, "dark", 25.0)
    assert int(dark[0, 0]) == 255

    constant = np.full((64, 64), 110, dtype=np.uint8)
    assert int(detection.local_otsu_foreground(constant, "light").max()) == 0


def test_wall_frame_keeps_its_hole():
    image = np.full((120, 160), 255, dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (140, 100), 0, 10)
    cv2.ellipse(image, (80, 60), (14, 20), 0, 0, 360, 0, -1)
    region = np.full((120, 160), 255, dtype=np.uint8)

    # 默认质量阈值下，空心墙框因真实支持少被丢弃，只留独立椭圆。
    strict, reason = detection.find_candidates(
        detection.threshold_foreground(image, settings()), region, settings()
    )
    assert reason is None
    assert len(strict) == 1
    assert abs(strict[0]["center"][0] - 80.0) <= 1.0
    assert abs(strict[0]["center"][1] - 60.0) <= 1.0

    values = settings(quality_threshold=0.0)
    candidates, _ = detection.find_candidates(
        detection.threshold_foreground(image, values), region, values
    )
    wall = max(candidates, key=lambda item: item["area"])
    # 墙框实体像素在内；洞内背景与独立椭圆都不属于墙候选。
    assert int(wall["mask"][60, 20]) == 255
    assert int(wall["mask"][60, 45]) == 0
    assert int(wall["mask"][60, 80]) == 0
    assert wall["quality"] < 0.4
    # 有效面积扣掉孔洞，小于外接矩形面积。
    x, y, width, height = wall["bbox"]
    assert wall["area"] < width * height
