"""时间中位数背景：有效曝光、已知端点与条带取消。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from behaviourlab import detection


def test_zero_exposure_context_never_enters_median():
    # 窗口 [.15,.25)：80 帧覆盖有效区间，200 是零曝光的窗口外上下文。
    frames = [(0.15, np.full((4, 4), 80, np.uint8), 0.10),
              (0.25, np.full((4, 4), 200, np.uint8), 0.0)]
    background = detection.temporal_background(frames, 2, 0.15, None)
    assert int(background[0, 0]) == 80


def test_frame_crossing_start_with_exposure_is_kept():
    # 起点前开始但覆盖窗口的帧有曝光，应保留。
    frames = [(0.0, np.full((4, 4), 40, np.uint8), 0.20),
              (0.20, np.full((4, 4), 90, np.uint8), 0.10)]
    background = detection.temporal_background(frames, 1, 0.15, 0.30)
    assert int(background[0, 0]) == 40


def test_known_duration_schedules_targets_on_source_clock():
    # duration=10、end_s=null：目标 0/3.333/6.667 取真实帧 0/4/7，中位数 50。
    frames = [(float(index), np.full((4, 4), (index + 1) * 10, np.uint8), 1.0)
              for index in range(10)]
    known = detection.temporal_background(frames, 3, 0.0, 10.0)
    assert int(known[0, 0]) == 50

    # 真正未知终点时才取前 N 帧，不伪造帧补样本。
    unknown = detection.temporal_background(frames, 3, 0.0, None)
    assert int(unknown[0, 0]) == 20


def test_cancel_between_median_strips(monkeypatch, tmp_path):
    created = {}
    real_mkstemp = detection.tempfile.mkstemp

    def fake_mkstemp(*args, **kwargs):
        handle, path = real_mkstemp(*args, dir=str(tmp_path), **kwargs)
        created["path"] = path
        return handle, path

    monkeypatch.setattr(detection.tempfile, "mkstemp", fake_mkstemp)

    # 130 行 = 64 + 64 + 2，三条带。
    frames = [(0.0, np.zeros((130, 4), np.uint8), 1.0),
              (1.0, np.zeros((130, 4), np.uint8), 1.0)]
    calls = []

    def cancel():
        calls.append(1)
        return len(calls) >= 4

    with pytest.raises(detection.DetectionError):
        detection.temporal_background(frames, 2, 0.0, None, cancel=cancel)

    # 两次帧检查 + 第一条带 + 第二条带前：第二条带没有开始计算。
    assert len(calls) == 4
    assert created.get("path")
    assert not Path(created["path"]).exists()


def test_background_without_exposed_frames_is_rejected():
    frames = [(0.0, np.zeros((4, 4), np.uint8), 0.0)]
    with pytest.raises(detection.DetectionError):
        detection.temporal_background(frames, 2, 0.0, None)
