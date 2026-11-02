"""导出工作量：段只栅格化一次、端点只换算一次、切分与线性版逐值一致。

计数器只统计真实调用次数（cv2.polylines / 坐标换算 / _split），不测时间。
"""

from __future__ import annotations

import random

import cv2
import pytest

from behaviourlab import annotated
from behaviourlab.schema import Frame

UNIT_A = "a" * 32
UNIT_B = "b" * 32


class _Source:
    frames_spec: list[dict] = []

    def __init__(self, path=None, *, start_s=0.0, end_s=None,
                 decode_images=True, **kwargs):
        self.frames = []
        for item in self.frames_spec:
            low = max(item["start"], start_s)
            high = item["end"] if end_s is None else min(item["end"], end_s)
            if high - low <= 1e-12:
                continue
            image = (cv2.cvtColor(
                __import__("numpy").full((36, 48), 60 + item["index"] % 100,
                                         dtype="uint8"),
                cv2.COLOR_GRAY2BGR) if decode_images else None)
            self.frames.append(Frame(
                image=image, index=item["index"], time_s=item["time_s"],
                dt_s=item["end"] - item["start"], elapsed_s=item["time_s"],
                analysis_fps=5.0, interval_start_s=low, interval_end_s=high))
        self.coverage = {"start_s": None, "end_s": None, "duration_s": 0.0,
                         "frame_count": 0}

    def iter_frames(self):
        return iter(self.frames)


def _spec(count: int) -> list[dict]:
    return [{"index": index, "time_s": index * 0.2,
             "start": index * 0.2, "end": index * 0.2 + 0.2}
            for index in range(count)]


def _units(count: int) -> list[dict]:
    units = []
    for slot, unit_id in enumerate((UNIT_A, UNIT_B)):
        rows = []
        for index in range(count):
            x, y = 5 + (index * 3 + slot) % 30, 6 + (index * 2) % 20
            row = {"unit_id": unit_id, "frame": index, "time_s": index * 0.2,
                   "x_px": x, "y_px": y, "x_cm": x, "y_cm": y,
                   "coordinate_unit": "px", "detected": True,
                   "position_valid": True}
            if index + 1 < count:
                row.update({"motion_x0": x, "motion_y0": y,
                            "motion_x1": 6 + (index * 3 + slot) % 30,
                            "motion_y1": 8 + (index * 2) % 20})
            rows.append(row)
        units.append({"unit_id": unit_id, "unit_name": f"u{slot}",
                      "window_start_s": 0.0, "window_end_s": count * 0.2,
                      "geometry": {}, "observations": rows})
    return units


def _run(tmp_path, count: int, monkeypatch):
    """每次在独立的补丁上下文里跑，计数互不叠加。"""

    counted = {"segments": 0, "conversions": 0, "splits": 0}
    with monkeypatch.context() as patch:
        _run_inner(tmp_path, count, patch, counted)
    return counted


def _run_inner(tmp_path, count, monkeypatch, counted):
    _Source.frames_spec = _spec(count)
    monkeypatch.setattr(annotated.media, "VideoSource", _Source)

    real_add = annotated._TraceLayer.add

    def add(self, start, end):
        counted["segments"] += 1
        return real_add(self, start, end)

    monkeypatch.setattr(annotated._TraceLayer, "add", add)

    real_mapper = annotated._cm_to_px_mapper

    def mapper(rows):
        inner = real_mapper(rows)
        if inner is None:
            return None

        def wrapped(point):
            counted["conversions"] += 1
            return inner(point)

        return wrapped

    monkeypatch.setattr(annotated, "_cm_to_px_mapper", mapper)
    real_split = annotated._split

    def split(start, end, points):
        counted["splits"] += 1
        return real_split(start, end, points)

    monkeypatch.setattr(annotated, "_split", split)

    written, _ = annotated.write_run_video(
        tmp_path / f"n{count}.mp4",
        job={"asset": {"path": "memory", "metadata": {"fps": 5.0}},
             "window": {"start_s": 0.0, "end_s": count * 0.2},
             "orientation": 0, "fps_override": None},
        units=_units(count))
    assert written == count
    return counted


def test_each_segment_rasterised_exactly_once(tmp_path, monkeypatch):
    small = _run(tmp_path, 5, monkeypatch)
    assert small["segments"] == 2 * (5 - 1), "每段只栅格化一次"
    assert small["conversions"] == 2 * small["segments"], "端点每段换算一次"
    assert small["splits"] == 5, "每个输出片段只切分一次"

    large = _run(tmp_path, 10, monkeypatch)
    assert large["segments"] == 2 * (10 - 1)
    assert large["conversions"] == 2 * large["segments"]
    assert large["splits"] == 10
    # 新增 5 帧只带来 10 次新栅格化：与历史段数无关的线性增长。
    assert large["segments"] - small["segments"] == 2 * (10 - 5)


def test_split_matches_linear_reference():
    from test_annotated_render_equivalence import _split_linear

    rng = random.Random(11)
    for _ in range(40):
        points = sorted({round(rng.uniform(0, 10), 6) for _ in range(12)})
        start, end = 0.0, 10.0
        assert annotated._split(start, end, points) == \
            _split_linear(start, end, points)
    # 边界：切点与端点重合、切点相邻 1e-12 以内、空表。
    assert annotated._split(1.0, 2.0, [1.0, 2.0]) == [(1.0, 2.0)]
    assert annotated._split(1.0, 2.0, []) == [(1.0, 2.0)]
    assert annotated._split(1.0, 1.0 + 1e-13, [1.0]) == []
    assert annotated._split(0.0, 1.0, [0.5, 0.5]) == [(0.0, 0.5), (0.5, 1.0)]


def test_duplicate_frame_rows_keep_first_and_prefix_semantics():
    """重复 frame 取首条；超出 upto 的第一段之后一律不取（原顺序语义）。"""

    unit = {"unit_id": UNIT_A, "unit_name": "u", "window_start_s": 0.0,
            "window_end_s": 10.0,
            "geometry": {},
            "observations": [
                {"unit_id": UNIT_A, "frame": 1, "time_s": 0.2,
                 "x_px": 3, "y_px": 4, "x_cm": 3, "y_cm": 4,
                 "coordinate_unit": "px", "detected": True,
                 "position_valid": True, "state": "first",
                 "motion_x0": 3, "motion_y0": 4, "motion_x1": 5,
                 "motion_y1": 6},
                {"unit_id": UNIT_A, "frame": 1, "time_s": 0.2,
                 "x_px": 30, "y_px": 40, "x_cm": 30, "y_cm": 40,
                 "coordinate_unit": "px", "detected": True,
                 "position_valid": True, "state": "second"},
                # 时间倒退的行不会被走到：超出 upto 的那一行之后即停止。
                {"unit_id": UNIT_A, "frame": 2, "time_s": 5.0,
                 "x_px": 7, "y_px": 8, "x_cm": 7, "y_cm": 8,
                 "coordinate_unit": "px", "detected": True,
                 "position_valid": True,
                 "motion_x0": 7, "motion_y0": 8, "motion_x1": 9,
                 "motion_y1": 10},
                {"unit_id": UNIT_A, "frame": 3, "time_s": 0.4,
                 "x_px": 11, "y_px": 12, "x_cm": 11, "y_cm": 12,
                 "coordinate_unit": "px", "detected": True,
                 "position_valid": True,
                 "motion_x0": 11, "motion_y0": 12, "motion_x1": 13,
                 "motion_y1": 14},
            ]}
    plan = annotated._segment_plan(unit)
    assert [moment for moment, _ in plan] == [0.2, 5.0, 0.4]
    # 到 1.0 秒为止：只取第一段（遇到 5.0 秒的段即停，与逐帧旧实现一致）。
    assert annotated._segments_upto(plan, 1.0) == [[(3, 4), (5, 6)]]
    assert len(annotated._segments_upto(plan, 6.0)) == 3

    prepared = annotated._unit_windows([unit])[0]
    track = annotated._UnitTrack(prepared, (20, 20), 1)
    assert track.state(1) == "first", "重复 frame 取首条"
    assert track.body(1) == (3, 4)
    assert track.state(2) != "first"
    assert track.body(99) is None
