"""经典检测固定预期：候选、饱和拒绝、极性、孔洞、区域与共享前景。"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

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


def wall_and_hole() -> np.ndarray:
    """160×120 白底：黑外框 (20,20)–(140,100)、白洞 (26,26)–(134,94)、洞内黑椭圆。"""

    image = np.full((120, 160), 255, dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (140, 100), 0, -1)
    cv2.rectangle(image, (26, 26), (134, 94), 255, -1)
    cv2.ellipse(image, (80, 60), (14, 20), 0, 0, 360, 0, -1)
    return image


def test_wall_mask_keeps_the_real_component():
    image = wall_and_hole()
    region = np.full((120, 160), 255, dtype=np.uint8)
    values = settings(quality_threshold=0.0)
    candidates, reason = detection.find_candidates(
        detection.threshold_foreground(image, values), region, values
    )
    assert reason is None
    wall = max(candidates, key=lambda item: item["area"])

    # 墙连通体本身是 2280 像素；内边缘那一圈墙不能被孔洞轮廓削掉。
    assert int(cv2.countNonZero(wall["mask"])) == 2280
    assert int(wall["mask"][40, 25]) == 255
    assert int(wall["mask"][40, 20]) == 255
    # 白洞与洞内独立椭圆都不属于墙。
    assert int(wall["mask"][40, 40]) == 0
    assert int(wall["mask"][40, 26]) == 0
    assert int(wall["mask"][60, 80]) == 0


def test_epm_and_automatic_y_require_partition():
    arena_only = {"arena": {"type": "quad",
                            "points": [[0, 0], [159, 0], [159, 119], [0, 119]]}}
    for assay in ("epm", "ymaze"):
        with pytest.raises(detection.DetectionError):
            detection.geometry_region(arena_only, assay, {}, 160, 120)
        with pytest.raises(detection.DetectionError):
            detection.geometry_region({}, assay, {}, 160, 120)


def test_epm_and_automatic_y_keep_zones_and_explicit_mask():
    def quad(x):
        return {"type": "quad",
                "points": [[x, 10], [x + 40, 10], [x + 40, 60], [x, 60]]}

    zones = {"center": quad(10), "open_arm_1": quad(60)}
    region, source = detection.geometry_region({"zones": zones}, "epm", {}, 160, 120)
    assert source == "zones"
    assert int(cv2.countNonZero(region)) == 2 * 41 * 51

    arms = {"center": quad(10), "arm_a": quad(60), "arm_b": quad(110)}
    region, source = detection.geometry_region(arms, "ymaze", {}, 160, 120)
    assert source == "zones"
    assert int(cv2.countNonZero(region)) == 3 * 41 * 51

    explicit = {"arena": quad(0), "mask": quad(100)}
    region, source = detection.geometry_region(explicit, "ymaze", {}, 160, 120)
    assert source == "mask"
    assert int(cv2.countNonZero(region)) == 41 * 51


class _CountingSubtractor:
    def __init__(self, mask: np.ndarray) -> None:
        self.mask = mask
        self.calls = 0

    def apply(self, gray: np.ndarray) -> np.ndarray:
        self.calls += 1
        return self.mask.copy()


def shared_settings(**overrides) -> dict:
    values = settings(bg_method="mog2", bg_type="dark", erosion_iter=0,
                      dilation_iter=0, max_area=10000.0, quality_threshold=0.0)
    values.update(overrides)
    return values


def blob_frame() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = np.full((120, 160, 3), 200, dtype=np.uint8)
    mask = np.zeros((120, 160), dtype=np.uint8)
    mask[40:61, 60:81] = 255          # 21×21 像素，contourArea = 400
    region = np.full((120, 160), 255, dtype=np.uint8)
    return image, mask, region


def test_shared_detector_keeps_unit_settings_apart():
    image, mask, region = blob_frame()
    key = ("asset-b05-isolation", 0, "window")
    a = detection.shared_detector(key, shared_settings(min_area=20.0))
    b = detection.shared_detector(key, shared_settings(min_area=1000.0))
    assert a.model is b.model
    a.model._subtractor = _CountingSubtractor(mask)

    first = a.detect(image, region)
    second = b.detect(image, region)

    assert a.model._subtractor.calls == 1
    assert len(first.candidates) == 1
    assert abs(first.candidates[0]["area"] - 400.0) <= 1e-6
    assert second.candidates == []
    assert second.missing_reason == "no_candidate"


def test_shared_foreground_cache_is_read_only():
    image, mask, region = blob_frame()
    key = ("asset-b05-readonly", 0, "window")
    detector = detection.shared_detector(key, shared_settings(min_area=20.0))
    detector.model._subtractor = _CountingSubtractor(mask)

    first = detector.foreground(image)
    second = detector.foreground(image)
    assert first is second
    with pytest.raises(ValueError):
        first[0, 0] = 0
    assert int(second[40, 60]) == 255
