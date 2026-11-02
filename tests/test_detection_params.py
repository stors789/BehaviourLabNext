"""检测参数表的服务器校验与默认合并。"""

from __future__ import annotations

import pytest

from behaviourlab import detection


def fields(issues):
    return [issue["field"] for issue in issues]


def test_defaults_cover_full_table():
    defaults = detection.DETECTION_DEFAULTS
    assert defaults["bg_method"] == "threshold"
    assert defaults["bg_type"] == "light"
    assert defaults["gray_threshold"] == 127.0
    assert defaults["body_gray_threshold"] == 95.0
    assert defaults["erosion_kernel"] == 3
    assert defaults["erosion_iter"] == 1
    assert defaults["dilation_kernel"] == 5
    assert defaults["dilation_iter"] == 1
    assert defaults["min_area"] == 120.0
    assert defaults["max_area"] == 50000.0
    assert defaults["quality_threshold"] == 0.4
    assert defaults["static_background"] is None


def test_validate_rejects_invalid_values():
    assert fields(detection.validate_detection({"erosion_kernel": 2.5})) == \
        ["detection.erosion_kernel"]
    assert fields(detection.validate_detection({"erosion_iter": -1})) == \
        ["detection.erosion_iter"]
    assert fields(detection.validate_detection({"dilation_kernel": 0})) == \
        ["detection.dilation_kernel"]
    assert fields(detection.validate_detection({"gray_threshold": 256})) == \
        ["detection.gray_threshold"]
    assert fields(detection.validate_detection({"temporal_diff_threshold": -1})) == \
        ["detection.temporal_diff_threshold"]
    assert fields(detection.validate_detection({"quality_threshold": 1.2})) == \
        ["detection.quality_threshold"]
    assert fields(detection.validate_detection({"bg_method": "flow"})) == \
        ["detection.bg_method"]
    assert fields(detection.validate_detection({"bg_type": "gray"})) == \
        ["detection.bg_type"]
    assert fields(detection.validate_detection({"min_area": 100, "max_area": 10})) == \
        ["detection.max_area"]
    assert fields(detection.validate_detection({"static_background": "  "})) == \
        ["detection.static_background"]


def test_validate_accepts_integer_even_kernels():
    assert detection.validate_detection({"erosion_kernel": 4, "dilation_kernel": 2,
                                         "erosion_iter": 0}) == []


def test_normalize_merges_sparse_override():
    resolved = detection.normalize_settings({"gray_threshold": 50,
                                             "bg_type": "dark"})
    assert resolved["gray_threshold"] == 50
    assert resolved["bg_type"] == "dark"
    assert resolved["min_area"] == 120.0
    with pytest.raises(detection.DetectionError):
        detection.normalize_settings({"erosion_kernel": 1.5})
