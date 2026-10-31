"""第 06 块三项固定预期：关联/缺失、门限与体型证据、细尾头向。"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from behaviourlab import detection, observation

TOL = 1e-6


def settings(**overrides) -> dict:
    values = dict(detection.DETECTION_DEFAULTS)
    values.update({
        "bg_type": "light",
        "gray_threshold": 127.0,
        "body_gray_threshold": 95.0,
        "min_area": 20.0,
        "max_area": 10000.0,
        "quality_threshold": 0.4,
        "erosion_iter": 0,
        "dilation_iter": 0,
    })
    values.update(overrides)
    return values


def full_region(width: int, height: int) -> np.ndarray:
    return np.full((height, width), 255, dtype=np.uint8)


def observe(tracker, image, region, values, *, index, time_s,
            unit="unit-1", signature="sig-a", assay="oft", previous=None):
    detector = detection.ClassicalDetector(values)
    candidates, reason = detection.find_candidates(
        detector.foreground(image), region, values)
    return tracker.observe(
        unit, candidates=candidates, reason=reason, frame_index=index,
        time_s=time_s, signature=signature, image=image, region=region,
        settings=values, assay=assay, previous=previous)


def test_association_keeps_size_evidence_across_missing():
    width, height = 240, 160
    values = settings(min_area=20.0, max_area=20000.0, quality_threshold=0.0)
    region = full_region(width, height)
    tracker = observation.ObservationTracker(("fixture-1", 0))
    centers = [(38, 80), (42, 80), None, None, None, (90, 80), (90, 80)]

    results = []
    for index, center in enumerate(centers):
        image = np.full((height, width), 255, dtype=np.uint8)
        if center is not None:
            cv2.ellipse(image, center, (12, 8), 0, 0, 360, 0, -1)
        if index >= 2:
            cv2.rectangle(image, (150, 20), (230, 135), 0, -1)
        results.append(observe(tracker, image, region, values,
                               index=index, time_s=index * 0.1))

    assert [item.detected for item in results] == [
        True, True, False, False, False, True, True]
    for item in results[2:5]:
        assert item.center is None
        assert item.mask is None
        assert item.contour == []
        assert item.nose is None and item.tail_base is None
        assert item.head_angle is None and item.head_quality == 0
        assert item.missing_reason == "inconsistent_candidate_size"
        # 候选列表保留，但不附保持位置/掩膜。
        assert len(item.candidates) == 1
    assert abs(results[5].center[0] - 90.0) <= 0.2
    assert abs(results[5].center[1] - 80.0) <= 0.2
    assert abs(results[6].center[0] - 90.0) <= 0.2


def candidate(area, center, span_box):
    return {"area": float(area), "center": list(center),
            "bbox": [center[0] - span_box[0] / 2, center[1] - span_box[1] / 2,
                     span_box[0], span_box[1]],
            "contour": [], "quality": 0.9}


def accept(tracker, area, span_box, *, index, time_s, center=(100.0, 100.0),
           signature="sig-a"):
    return tracker.observe(
        "unit-1", candidates=[candidate(area, center, span_box)],
        reason=None, frame_index=index, time_s=time_s, signature=signature)


def test_spatial_gate_and_fixed_reference_median():
    assert abs(observation.spatial_gate(10.0, 0.1) - 42.0) <= TOL

    tracker = observation.ObservationTracker(("fixture-2", 0))
    accept(tracker, 48.0, (8.0, 6.0), index=0, time_s=0.0)   # span = 10
    far = tracker.observe("unit-1", candidates=[candidate(48.0, (143.0, 100.0),
                                                          (8.0, 6.0))],
                          reason=None, frame_index=1, time_s=0.1,
                          signature="sig-a")
    assert far.detected is False
    assert far.missing_reason == "no_nearby_candidate"
    assert len(far.candidates) == 1

    near = tracker.observe("unit-1", candidates=[candidate(48.0, (141.0, 100.0),
                                                           (8.0, 6.0))],
                           reason=None, frame_index=1, time_s=0.1,
                           signature="sig-a")
    assert near.detected is True
    assert abs(near.center[0] - 141.0) <= TOL

    # 只有调用方 previous、尚无尺寸历史：S 取当前候选最小 span，大候选不能放大门限。
    fresh = observation.ObservationTracker(("fixture-2c", 0))
    picked = fresh.observe(
        "unit-1",
        candidates=[candidate(9000.0, (100.0, 135.0), (90.0, 30.0)),
                    candidate(100.0, (100.0, 100.0), (10.0, 10.0))],
        reason=None, frame_index=0, time_s=0.0, signature="sig-a",
        previous=(100.0, 100.0),
    )
    assert picked.detected is True
    assert abs(picked.area - 100.0) <= TOL
    assert abs(picked.center[1] - 100.0) <= TOL

    # 9 个 100 → 9 个 50：滚动中位数变 50，固定参考仍是 100。
    grown = observation.ObservationTracker(("fixture-2b", 0))
    for step in range(9):
        assert accept(grown, 100.0, (10.0, 10.0), index=step,
                      time_s=step * 0.1).detected is True
    for step in range(9, 18):
        assert accept(grown, 50.0, (7.0, 7.0), index=step,
                      time_s=step * 0.1).detected is True

    state = grown.state("unit-1")
    assert state.reference_areas[:9] == [100.0] * 9
    assert state.areas == [50.0] * 9

    # 重复同一源时刻不增加体型样本（与上次接受同为 1.7）。
    repeated = accept(grown, 50.0, (7.0, 7.0), index=18, time_s=1.7)
    assert repeated.detected is True
    assert state.areas == [50.0] * 9
    assert len(state.reference_areas) == 9

    rejected = accept(grown, 20.0, (4.5, 4.5), index=19, time_s=1.8)
    assert rejected.detected is False
    assert rejected.missing_reason == "inconsistent_candidate_size"

    # 配置签名改变后允许重新建立不同面积体型。
    rebuilt = accept(grown, 20.0, (4.5, 4.5), index=20, time_s=1.9,
                     signature="sig-b")
    assert rebuilt.detected is True
    assert grown.state("unit-1").areas == [20.0]

    # 逆向跳帧同样重建证据。
    accept(grown, 60.0, (7.7, 7.7), index=21, time_s=2.0, signature="sig-b")
    jumped = accept(grown, 60.0, (7.7, 7.7), index=3, time_s=0.3,
                    signature="sig-b")
    assert jumped.detected is True
    assert grown.state("unit-1").areas == [60.0]


def test_head_only_with_thin_tail_evidence():
    width, height = 160, 120
    region = full_region(width, height)
    values = settings(gray_threshold=127.0, body_gray_threshold=95.0,
                      max_area=10000.0, quality_threshold=0.4)

    symmetric = np.full((height, width), 255, dtype=np.uint8)
    cv2.ellipse(symmetric, (75, 60), (24, 12), 0, 0, 360, 0, -1)
    plain = observe(observation.ObservationTracker(("head-a", 0)), symmetric,
                    region, values, index=0, time_s=0.0, assay="nor")
    assert plain.detected is True
    assert plain.nose is None and plain.tail_base is None
    assert plain.head_angle is None
    assert plain.head_quality == 0

    tailed = symmetric.copy()
    cv2.line(tailed, (50, 60), (20, 60), 0, 2)
    with_tail = observe(observation.ObservationTracker(("head-b", 0)), tailed,
                        region, values, index=0, time_s=0.0, assay="nor")
    assert with_tail.detected is True
    assert with_tail.head_quality > 0.5
    assert with_tail.nose[0] > with_tail.tail_base[0]
    assert abs(with_tail.head_angle) < 5.0

    floor_values = settings(gray_threshold=50.0, body_gray_threshold=127.0,
                            max_area=10000.0, quality_threshold=0.4)
    floor = np.full((height, width), 110, dtype=np.uint8)
    cv2.ellipse(floor, (75, 60), (24, 12), 0, 0, 360, 0, -1)
    merged = observe(observation.ObservationTracker(("head-c", 0)), floor,
                     region, floor_values, index=0, time_s=0.0, assay="nor")
    assert merged.detected is True
    assert abs(merged.center[0] - plain.center[0]) <= TOL
    assert abs(merged.center[1] - plain.center[1]) <= TOL
    assert merged.nose is None and merged.tail_base is None
    assert merged.head_angle is None
    assert merged.head_quality == 0
