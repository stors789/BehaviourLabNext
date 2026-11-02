"""叠画渲染：改前（8f39454 逐段重放）与改后（增量轨迹层）的编码前像素对照。

改前实现逐帧重建全部历史段并逐段 ``cv2.polylines``；改后把同色段的覆盖度
累积在增量层里、每帧合成一次。这里用同一批冻结事实跑两条路径，比较**编码
前**画布，避免有损编码掩盖绘制差异。

参照实现 ``legacy_canvases`` 逐行复刻 8f39454 的渲染循环，不改写任何断言。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from behaviourlab import annotated
from behaviourlab.schema import Frame

UNIT_A = "a" * 32
UNIT_B = "b" * 32


class _Source:
    """确定性合成源：每帧像素随帧号变化，用于暴露抗锯齿混合差异。"""

    frames_spec: list[dict] = []

    def __init__(self, path=None, *, start_s=0.0, end_s=None,
                 decode_images=True, **kwargs):
        self.frames = []
        for item in self.frames_spec:
            low = max(item["start"], start_s)
            high = item["end"] if end_s is None else min(item["end"], end_s)
            if high - low <= 1e-12:
                continue
            image = _frame_image(item["index"]) if decode_images else None
            self.frames.append(Frame(
                image=image, index=item["index"], time_s=item["time_s"],
                dt_s=item["end"] - item["start"],
                elapsed_s=item["time_s"], analysis_fps=5.0,
                interval_start_s=low, interval_end_s=high))
        self.coverage = {"start_s": None, "end_s": None, "duration_s": 0.0,
                         "frame_count": 0}

    def iter_frames(self):
        return iter(self.frames)


def _frame_image(index: int) -> np.ndarray:
    rows = np.arange(72, dtype=np.int32)[:, None]
    cols = np.arange(96, dtype=np.int32)[None, :]
    base = (rows * 3 + cols * 2 + index * 11) % 256
    base = base.astype(np.uint8)
    return np.stack([base, np.roll(base, 5, axis=1),
                     np.roll(base, 9, axis=0)], axis=-1)


def _job(frames: list[dict], *, orientation: int = 0) -> dict:
    return {
        "asset": {"path": "memory", "metadata": {"fps": 5.0}},
        "window": {"start_s": 0.0, "end_s": frames[-1]["end"]},
        "orientation": orientation, "fps_override": None,
    }


def _active_units_linear(units, start: float, end: float):
    middle = (start + end) / 2.0
    return [unit for unit in units
            if unit["start"] - 1e-9 <= middle < unit["end"] - 1e-9]


def _prepare(job, units, events):
    prepared_units = annotated._unit_windows(units)
    prepared_events = []
    for event in events or ():
        moment, confirmed = annotated._display_time(event)
        if moment is None:
            continue
        prepared_events.append({"display_s": float(moment),
                                "confirmed": confirmed,
                                "target": event.get("target"),
                                "kind": event.get("kind"),
                                "unit_id": event.get("unit_id")})
    prepared_events.sort(key=lambda item: item["display_s"])
    return prepared_units, prepared_events


def _split_linear(start: float, end: float, points) -> list[tuple]:
    """8f39454 的切分：每帧逐点扫全表，二分只用于对照结果。"""

    cuts = [start]
    for point in points:
        if start + 1e-9 < point < end - 1e-9:
            cuts.append(point)
    cuts.append(end)
    return [(cuts[index], cuts[index + 1])
            for index in range(len(cuts) - 1)
            if cuts[index + 1] - cuts[index] > 1e-12]


def legacy_canvases(job, units, events, collect=True):
    """8f39454 的渲染循环：逐帧重建全部完成段并逐段绘制。

    ``collect=False`` 只统计画布数量（测时间/内存时不保留像素）。
    """

    import cv2

    window = job.get("window") or {}
    window_start = float(window.get("start_s") or 0.0)
    end = window.get("end_s")
    window_end = float(end) if end is not None else math.inf
    prepared_units, prepared_events = _prepare(job, units, events)

    def clipped(frame):
        low = annotated._finite(frame.interval_start_s)
        high = annotated._finite(frame.interval_end_s)
        if low is None or high is None:
            return None
        low = max(float(low), window_start)
        high = min(float(high), window_end)
        if high - low <= 1e-12:
            return None
        return low, high

    timing = []
    for frame in _Source(decode_images=False).iter_frames():
        pair = clipped(frame)
        if pair is None:
            continue
        timing.append({"index": frame.index, "start": pair[0],
                       "end": pair[1], "time_s": frame.time_s})
    bounds = annotated._segment_bounds(timing, prepared_units, prepared_events)

    canvases: list[np.ndarray] = []
    for frame in _Source(decode_images=True).iter_frames():
        base = annotated._to_bgr(frame.image)
        pair = clipped(frame)
        if base is None or pair is None:
            continue
        low, high = pair
        for clip_start, clip_end in _split_linear(low, high, bounds):
            canvas = base.copy()
            thickness = max(1, int(round(min(canvas.shape[:2]) / 320)))
            lines = [f"t={annotated.clock_text(clip_start)}  frame={frame.index}"]
            for unit in _active_units_linear(prepared_units, clip_start,
                                             clip_end):
                annotated._draw_geometry(canvas, unit["geometry"], thickness)
                for segment in annotated._trajectory_segments(unit, clip_end):
                    if len(segment) >= 2:
                        cv2.polylines(canvas, [np.array(segment, "int32")],
                                      False, (95, 98, 184), thickness,
                                      cv2.LINE_AA)
                row = annotated._row_for_frame(unit, int(frame.index))
                point = annotated._body_point(row)
                if point is not None:
                    cv2.circle(canvas, point, 4 + thickness, (78, 104, 22),
                               -1, cv2.LINE_AA)
                lines.append(f"{annotated._ascii_label(unit.get('unit_name'))} "
                             f"{annotated._row_state(row)}")
            for event in prepared_events:
                if event["display_s"] > clip_start + 1e-9:
                    continue
                label = "Confirmed" if event["confirmed"] else "Unconfirmed"
                lines.append(f"{label} {event.get('kind') or 'event'} "
                             f"{event.get('target') or ''}".strip())
            canvas = annotated._draw_label(canvas, lines)
            canvas = annotated._even_canvas(canvas)
            canvases.append(canvas.copy() if collect else None)
    return canvases


def new_canvases(monkeypatch, job, units, events):
    """生产 write_run_video：只截获编码前画布，不写真实文件。"""

    captured: list[np.ndarray] = []

    class _Stream:
        width = 0
        height = 0
        pix_fmt = ""
        time_base = None

        class codec_context:
            time_base = None
            options = {}

        def encode(self, frame):
            return []

    class _Writer:
        def add_stream(self, *args, **kwargs):
            return _Stream()

        def mux(self, packet):
            return None

        def close(self):
            return None

    class _VFrame:
        def __init__(self, canvas):
            self.canvas = canvas
            self.pts = 0
            self.time_base = None

    def make(canvas, format=None):
        captured.append(np.array(canvas, copy=True))
        return _VFrame(canvas)

    monkeypatch.setattr(annotated, "_open_writer",
                        lambda *a, **k: (_Writer(), _Stream()))
    monkeypatch.setattr(annotated.av, "VideoFrame",
                        type("_VF", (), {"from_ndarray": staticmethod(make)}))
    monkeypatch.setattr(annotated, "verify_video", lambda *a, **k: [])
    annotated.write_run_video("memory.mp4", job=job, units=units, events=events)
    return captured


def _compare(old, new) -> tuple[int, int, int, int]:
    assert len(old) == len(new), f"画布数量不同：{len(old)} vs {len(new)}"
    worst = 0
    diff_pixels = 0
    painted = 0
    for before, after in zip(old, new):
        assert before.shape == after.shape
        delta = np.abs(before.astype(np.int16) - after.astype(np.int16))
        worst = max(worst, int(delta.max()))
        changed = delta.max(axis=2) > 0
        diff_pixels += int(changed.sum())
        painted += int((before != after).any(axis=2).sum())
    return diff_pixels, worst, len(old), painted


def _frames(count: int, step: float = 0.2) -> list[dict]:
    return [{"index": index, "time_s": round(index * step, 6),
             "start": round(index * step, 6),
             "end": round(index * step + step, 6)} for index in range(count)]


def _rows(unit_id, points):
    rows = []
    for item in points:
        frame, moment, x, y = item[0], item[1], item[2], item[3]
        extra = item[4] if len(item) > 4 else None
        row = {"unit_id": unit_id, "frame": frame, "time_s": moment,
               "x_px": x, "y_px": y, "x_cm": x, "y_cm": y,
               "coordinate_unit": "px", "detected": True,
               "position_valid": True}
        row.update(extra or {})
        rows.append(row)
    return rows


def _with_segments(rows):
    for index, row in enumerate(rows[:-1]):
        nxt = rows[index + 1]
        if row.get("position_valid") is False or row.get("jump"):
            continue
        if nxt.get("position_valid") is False or nxt.get("jump"):
            continue
        if nxt["time_s"] - row["time_s"] > 0.2 + 1e-9:
            continue
        row["motion_x0"] = row["x_px"]
        row["motion_y0"] = row["y_px"]
        row["motion_x1"] = nxt["x_px"]
        row["motion_y1"] = nxt["y_px"]
    return rows


@pytest.fixture(autouse=True)
def _frames_spec():
    _Source.frames_spec = _frames(8)
    yield
    _Source.frames_spec = []


def _report(name, old, new):
    diff, worst, canvases, _painted = _compare(old, new)
    print(f"[{name}] 画布 {canvases} 张；差异像素 {diff}；最大差 {worst}")
    # 差异只来自舍入：既有断言不放宽，这里如实记录实际幅度。
    assert worst <= 2, f"{name} 出现超过 2 的像素差：{worst}"
    return diff, worst


def test_single_segment_diff_confined_to_partial_coverage():
    """单段：完全覆盖与未画像素逐值相同，差异只在 AA 部分覆盖像素。"""

    import cv2

    base = _frame_image(0)[:40, :60].copy()
    start, end, thickness = (5, 5), (52, 33), 2
    direct = base.copy()
    cv2.polylines(direct, [np.array([start, end], "int32")], False,
                  (95, 98, 184), thickness, cv2.LINE_AA)
    mask = np.zeros(base.shape[:2], np.uint8)
    cv2.polylines(mask, [np.array([start, end], "int32")], False, 255,
                  thickness, cv2.LINE_AA)

    layer = annotated._TraceLayer(base.shape[:2], (95, 98, 184), thickness)
    layer.add(start, end)
    ours = base.copy()
    layer.compose(ours)

    delta = np.abs(direct.astype(np.int16) - ours.astype(np.int16))
    changed = delta.max(axis=2) > 0
    # 实测上界（见报告“未解决差异”）不是等价结论：逐段整数舍入与一次性
    # 合成舍入不同，差异只允许出现在部分覆盖像素上。
    assert int(delta.max()) <= 2
    assert not changed[mask == 255].any(), "完全覆盖的像素必须逐值相同"
    assert not changed[mask == 0].any(), "未画到的像素必须逐值相同"
    partial = (mask > 0) & (mask < 255)
    assert changed[partial].any(), "部分覆盖像素才有舍入差"
    print(f"[单段] 部分覆盖像素 {int(partial.sum())}，差异 "
          f"{int(changed[partial].sum())}，最大差 {int(delta.max())}")


def test_crossing_and_overlapping_trajectories(monkeypatch):
    monkeypatch.setattr(annotated.media, "VideoSource", _Source)
    rows = _with_segments(_rows(UNIT_A, [
        (0, 0.0, 10, 10), (1, 0.2, 40, 30), (2, 0.4, 70, 12),
        (3, 0.6, 30, 55), (4, 0.8, 80, 60), (5, 1.0, 12, 60),
        (6, 1.2, 60, 20), (7, 1.4, 20, 40),
    ]))
    units = [{"unit_id": UNIT_A, "unit_name": "u1",
              "window_start_s": 0.0, "window_end_s": 1.6,
              "geometry": {}, "observations": rows}]
    events = [{"time_s": 0.4, "confirmed_s": 0.6, "kind": "zone_entry",
               "target": "open", "unit_id": UNIT_A}]
    job = _job(_Source.frames_spec)
    _report("交叉/重叠", legacy_canvases(job, units, events),
            new_canvases(monkeypatch, job, units, events))


def test_two_units_window_switch_and_geometry_overlap(monkeypatch):
    monkeypatch.setattr(annotated.media, "VideoSource", _Source)
    rows_a = _with_segments(_rows(UNIT_A, [
        (0, 0.0, 10, 10), (1, 0.2, 30, 20), (2, 0.4, 55, 40),
        (3, 0.6, 20, 55), (4, 0.8, 70, 30),
    ]))
    rows_b = _with_segments(_rows(UNIT_B, [
        (2, 0.4, 80, 60), (3, 0.6, 60, 40), (4, 0.8, 30, 20),
        (5, 1.0, 12, 12), (6, 1.2, 66, 62),
    ]))
    geometry = {"arena": {"type": "quad",
                          "points": [[4, 4], [92, 4], [92, 66], [4, 66]]},
                "body_line": {"type": "line", "points": [[30, 10], [60, 50]]}}
    units = [
        {"unit_id": UNIT_A, "unit_name": "first",
         "window_start_s": 0.0, "window_end_s": 0.9,
         "geometry": geometry, "observations": rows_a},
        {"unit_id": UNIT_B, "unit_name": "second",
         "window_start_s": 0.4, "window_end_s": 1.6,
         "geometry": geometry, "observations": rows_b},
    ]
    events = [{"time_s": 0.2, "confirmed_s": 0.4, "kind": "zone_entry",
               "target": "center", "unit_id": UNIT_A},
              {"time_s": 0.6, "kind": "rearing", "target": "wall",
               "unit_id": UNIT_B}]
    job = _job(_Source.frames_spec)
    _report("双动物/窗切换/几何", legacy_canvases(job, units, events),
            new_canvases(monkeypatch, job, units, events))


def test_gaps_jumps_suppressed_and_manual(monkeypatch):
    monkeypatch.setattr(annotated.media, "VideoSource", _Source)
    rows = _with_segments(_rows(UNIT_A, [
        (0, 0.0, 12, 12), (1, 0.2, 30, 18),
        (2, 0.4, 50, 26),                               # 缺口起点（下一行越界）
        (4, 0.8, 70, 40, {"jump": True}),
        (5, 1.0, 20, 60), (6, 1.2, 30, 60, {"state": "freeze"}),
    ]))
    manual = _with_segments(_rows(UNIT_B, [
        (0, 0.0, 40, 40), (1, 0.2, 50, 44), (2, 0.4, 60, 48),
    ]))
    units = [
        {"unit_id": UNIT_A, "unit_name": "tracked",
         "window_start_s": 0.0, "window_end_s": 1.6,
         "geometry": {}, "observations": rows},
        {"unit_id": UNIT_B, "unit_name": "manual", "manual": True,
         "window_start_s": 0.0, "window_end_s": 1.6,
         "geometry": {}, "observations": manual},
        {"unit_id": "c" * 32, "unit_name": "suppressed",
         "suppress_positions": True,
         "window_start_s": 0.0, "window_end_s": 1.6,
         "geometry": {}, "observations": rows},
    ]
    job = _job(_Source.frames_spec)
    _report("缺口/jump/人工/抑制", legacy_canvases(job, units, []),
            new_canvases(monkeypatch, job, units, []))


def test_multiple_events_in_one_exposure(monkeypatch):
    monkeypatch.setattr(annotated.media, "VideoSource", _Source)
    rows = _with_segments(_rows(UNIT_A, [
        (0, 0.0, 20, 20), (1, 0.2, 40, 24),
    ]))
    units = [{"unit_id": UNIT_A, "unit_name": "u",
              "window_start_s": 0.0, "window_end_s": 1.6,
              "geometry": {}, "observations": rows}]
    events = [
        {"time_s": 0.1, "confirmed_s": 0.1, "kind": "zone_entry",
         "target": "A", "unit_id": UNIT_A},
        {"time_s": 0.1, "confirmed_s": 0.15, "kind": "zone_exit",
         "target": "A", "unit_id": UNIT_A},
        {"time_s": 0.3, "confirmed_s": 0.35, "kind": "rearing",
         "target": "B", "unit_id": UNIT_A},
    ]
    job = _job(_Source.frames_spec)
    _report("同曝光多事件", legacy_canvases(job, units, events),
            new_canvases(monkeypatch, job, units, events))
