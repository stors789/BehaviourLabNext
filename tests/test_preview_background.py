"""时间中位数背景的接线：有效曝光与已知终点不在这里丢失。"""

from __future__ import annotations

import numpy as np

from behaviourlab import preview


class FakeFrame:
    def __init__(self, time_s: float, value: int, exposure: float) -> None:
        self.time_s = time_s
        self.image = np.full((4, 4), value, dtype=np.uint8)
        self.interval_start_s = time_s
        self.interval_end_s = time_s + exposure


class FakeSource:
    frames: list = []

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    def iter_frames(self):
        return iter(self.frames)


def run_background(monkeypatch, metadata, frames):
    seen = {}

    def fake_background(items, samples, start, end, *, cancel=None):
        seen["frames"] = [(time_s, int(image[0, 0]), exposure)
                          for time_s, image, exposure in items]
        seen["samples"] = samples
        seen["start"] = start
        seen["end"] = end
        return np.zeros((4, 4), dtype=np.uint8)

    FakeSource.frames = frames
    monkeypatch.setattr(preview.media, "VideoSource", FakeSource)
    monkeypatch.setattr(preview.detection, "temporal_background", fake_background)
    preview._background_for(
        {"bg_method": "temporal", "temporal_bg_samples": 3},
        {"path": "clip.mp4", "metadata": metadata},
        {"window": {"start_s": 0.0, "end_s": None}, "fps_override": None,
         "sample_fps": None},
        0, 4, 4,
    )
    return seen


def test_known_duration_supplies_sampling_end(monkeypatch):
    seen = run_background(
        monkeypatch, {"fps": 10.0, "duration_s": 10.0},
        [FakeFrame(0.0, 10, 0.1), FakeFrame(10.0, 200, 0.0)],
    )
    assert seen["samples"] == 3
    assert seen["start"] == 0.0
    # 窗口终点未写时用同一源时钟的已知时长；零曝光的片尾仍是 0 秒曝光。
    assert seen["end"] == 10.0
    assert seen["frames"][1][2] == 0.0


def test_unknown_duration_leaves_end_unknown(monkeypatch):
    seen = run_background(
        monkeypatch, {"fps": 10.0, "duration_s": None},
        [FakeFrame(0.0, 10, 0.1)],
    )
    assert seen["end"] is None
