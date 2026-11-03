"""尾迹节点预算与逐帧工作量：只从保存完成段计节点，超限淘汰最旧节点。

节点共享的保存事实证明：``timeline.attach_motion`` 按 (unit_id, frame) 把
“上一有效观测 → 本行观测”的段写进行内，因此源行相邻且都存有完成段的两段
共享同一节点；其余情况（不同观测、断点两侧、同坐标不同观测）独立计数。
计数以节点记录为准，不按秒/帧，也不因两点 contour 重复计算共享节点。
"""

from __future__ import annotations

import random

import cv2
import numpy as np
import pytest

from behaviourlab import annotated, figures
from behaviourlab.schema import Frame

UNIT_A = "a" * 32
UNIT_B = "b" * 32
LIMIT = annotated.TRAJECTORY_NODE_LIMIT


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
                np.full((36, 48), 60 + item["index"] % 100, dtype="uint8"),
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


def _row(index: int, x: int, y: int, *, x1: int | None = None,
         y1: int | None = None, **extra):
    row = {"unit_id": UNIT_A, "frame": index, "time_s": index * 0.2,
           "x_px": x, "y_px": y, "x_cm": x, "y_cm": y,
           "coordinate_unit": "px", "detected": True, "position_valid": True}
    if x1 is not None:
        row.update({"motion_x0": x, "motion_y0": y,
                    "motion_x1": x1, "motion_y1": y1})
    row.update(extra)
    return row


def _chain(count: int, *, step: int = 3) -> list[dict]:
    """count 条完成段（count+1 行）：与管线同构——段存在“到达行”内，
    连接上一有效观测与本行观测；首行没有段。共享节点按相邻行证明。"""

    def point(index: int) -> tuple[int, int]:
        return (5 + (index * step) % 40, 6 + (index * 2) % 30)

    rows = [_row(0, *point(0))]
    for index in range(1, count + 1):
        previous = point(index - 1)
        current = point(index)
        rows.append(_row(index, current[0], current[1],
                         x1=previous[0], y1=previous[1]))
    return rows


def _track(rows, unit_id=UNIT_A, **unit_extra):
    unit = annotated._unit_windows([{
        "unit_id": unit_id, "unit_name": "u",
        "window_start_s": 0.0, "window_end_s": 1e9,
        "geometry": {}, "observations": rows, **unit_extra}])[0]
    return annotated._UnitTrack(unit)


def _state(track):
    return {"nodes": len(track.nodes), "edges": len(track.edges),
            "dropped": track.dropped}


def test_budget_999_1000_1001_nodes(tmp_path):
    for segments in (998, 999, 1000, 1001, 1200):
        track = _track(_chain(segments))
        track.advance(1e9)
        state = _state(track)
        # 连续链：segments 段 = segments+1 个节点、segments 条边；
        # 超过 1000 节点后淘汰最旧节点及引用它的边，剩 1000 节点 / 999 边。
        assert state["nodes"] == min(segments + 1, LIMIT), (segments, state)
        assert state["edges"] == min(segments, LIMIT - 1), (segments, state)
        assert state["dropped"] == max(0, segments + 1 - LIMIT), (segments, state)
        if state["nodes"] == LIMIT:
            assert state["edges"] == LIMIT - 1, "连续链保留节点-边=1"


def test_shared_endpoint_counts_once_but_coordinates_do_not_deduplicate():
    # 相邻源行且坐标完全相同：共享节点按保存事实证明，只计一次。
    chained = _track([
        _row(0, 10, 10, x1=20, y1=20),
        _row(1, 20, 20, x1=30, y1=30),        # 与上一段共享 obs0→obs1 的节点
        _row(2, 30, 30, x1=40, y1=40),
    ])
    chained.advance(1e9)
    assert _state(chained) == {"nodes": 4, "edges": 3, "dropped": 0}

    # 不相邻的两段（中间隔着无段观测）即使坐标完全相同，也是 4 个独立
    # 节点——不按坐标去重。
    same_coords = _track([
        _row(0, 10, 10, x1=20, y1=20),
        _row(1, 20, 20),                      # 无完成段：断链
        _row(2, 10, 10, x1=20, y1=20),
    ])
    same_coords.advance(1e9)
    assert _state(same_coords) == {"nodes": 4, "edges": 2, "dropped": 0}

    # 断点（中间行无效被跳过）：两侧节点分别计数，同坐标也不合并。
    broken = _track([
        _row(0, 10, 10, x1=20, y1=20),
        _row(1, 20, 20, position_valid=False),
        _row(2, 20, 20, x1=30, y1=30),
    ])
    broken.advance(1e9)
    assert _state(broken) == {"nodes": 4, "edges": 2, "dropped": 0}


def test_future_segments_never_introduced(tmp_path):
    track = _track(_chain(10))
    track.advance(1.0)          # 只到第 6 个完成段（0.2*5=1.0）
    assert _state(track) == {"nodes": 6, "edges": 5, "dropped": 0}
    track.advance(0.5)          # 时间倒退：游标单调，不引入也不回退
    assert _state(track) == {"nodes": 6, "edges": 5, "dropped": 0}
    track.advance(1e9)
    assert _state(track) == {"nodes": 11, "edges": 10, "dropped": 0}


def test_same_time_segments_all_land_in_order():
    rows = [_row(0, 5, 5, x1=10, y1=10, ),
            _row(1, 10, 10, x1=15, y1=15),
            _row(2, 15, 15, x1=20, y1=20)]
    rows[1]["time_s"] = rows[2]["time_s"] = 0.4
    track = _track(rows)
    track.advance(0.4)
    assert _state(track) == {"nodes": 4, "edges": 3, "dropped": 0}
    # 同刻多段按保存顺序推进：第一段（t=0.0）先成为边。
    assert list(track.edges)[0].time_s == 0.0
    assert [edge.time_s for edge in track.edges] == [0.0, 0.4, 0.4]


def test_eviction_drops_only_referencing_edges_and_never_bridges():
    # 连续链淘汰最旧节点：只少一条边，剩余边顺序不变，不补连。
    rows = _chain(1003)
    track = _track(rows)
    track.advance(1003 * 0.2 - 0.1)      # 前 1002 段已完成，最后一段未完成
    assert len(track.nodes) == LIMIT and len(track.edges) == LIMIT - 1
    before = [np.array(edge.array, copy=True) for edge in track.edges]
    order_before = [edge.start for edge in track.edges]
    track.advance(1e9)                   # 最后一段完成，再淘汰一个最旧节点
    after = [edge.array for edge in track.edges]
    assert len(track.nodes) == LIMIT
    assert len(track.edges) == LIMIT - 1
    # 淘汰最旧节点后整体前移一格：剩余保存边顺序与归属不变，不补连。
    assert [int(e[0, 0]) for e in after[:-1]] == [
        int(e[0, 0]) for e in before[1:]], "剩余保存边顺序不变"
    assert [edge.start for edge in track.edges][:-1] == order_before[1:], \
        "节点序号与边的归属保持原身份"

    # 独立两点段淘汰一端：只删引用它的边，另一端成为孤立节点，不补连。
    rows = [_row(0, 5, 5, x1=8, y1=8)]
    rows += _chain(LIMIT - 2)[1:]
    rows[1] = _row(1, 8, 8, x1=11, y1=11)   # 与首段断开（源行仍相邻？否——
    # 行 0 与行 1 相邻会被判共享；这里用断开行的构造改为：
    rows = [_row(0, 5, 5, x1=8, y1=8),
            _row(1, 8, 8),
            _row(2, 8, 8, x1=11, y1=11)] + _chain(LIMIT - 3)[1:]
    track = _track(rows)
    track.advance(1e9)
    assert len(track.nodes) == LIMIT
    # 首段（节点0-1）与后续链断开：淘汰节点 0 后，引用它的边消失，
    # 节点 1 保留为孤立节点，不与后续链补连。
    assert track.dropped >= 1
    assert all(edge.start != 0 for edge in track.edges)


def test_per_animal_budget_is_isolated():
    rows_a = _chain(20)
    rows_b = _chain(3)
    track_a = _track(rows_a, UNIT_A)
    track_b = _track([{**row, "unit_id": UNIT_B} for row in rows_b], UNIT_B)
    track_a.advance(1e9)
    assert _state(track_a)["nodes"] == 21
    assert _state(track_b) == {"nodes": 0, "edges": 0, "dropped": 0}
    track_b.advance(1e9)
    assert _state(track_b) == {"nodes": 4, "edges": 3, "dropped": 0}
    assert _state(track_a)["nodes"] == 21


def test_manual_and_suppressed_units_have_no_trail():
    rows = _chain(5)
    for flag in ({"manual": True}, {"suppress_positions": True}):
        track = _track(rows, **flag)
        track.advance(1e9)
        assert _state(track) == {"nodes": 0, "edges": 0, "dropped": 0}
        buffer = np.zeros((48, 64, 3), np.uint8)
        track.draw(buffer, 1)
        assert int(buffer.sum()) == 0


def _run(tmp_path, count, monkeypatch, rows_builder=None):
    counted = {"arrays": 0, "calls": 0, "conversions": 0, "splits": 0,
               "max_arrays": 0}
    with monkeypatch.context() as patch:
        _run_inner(tmp_path, count, patch, counted, rows_builder)
    return counted


def _run_inner(tmp_path, count, monkeypatch, counted, rows_builder=None):
    _Source.frames_spec = _spec(count)
    monkeypatch.setattr(annotated.media, "VideoSource", _Source)
    real_poly = cv2.polylines

    def poly(canvas, polys, *args, **kwargs):
        if isinstance(polys, list):
            counted["calls"] += 1
            counted["arrays"] += len(polys)
            counted["max_arrays"] = max(counted["max_arrays"], len(polys))
        return real_poly(canvas, polys, *args, **kwargs)

    real_mapper = annotated._cm_to_px_mapper

    def mapper(rows):
        inner = real_mapper(rows)
        if inner is None:
            return None

        def wrapped(point):
            counted["conversions"] += 1
            return inner(point)

        return wrapped

    monkeypatch.setattr(cv2, "polylines", poly)
    monkeypatch.setattr(annotated, "_cm_to_px_mapper", mapper)

    rows = (rows_builder or _chain)(count)
    units = [{"unit_id": UNIT_A, "unit_name": "u",
              "window_start_s": 0.0, "window_end_s": (count + 1) * 0.2,
              "geometry": {}, "observations": rows}]
    written, _ = annotated.write_run_video(
        tmp_path / f"n{count}.mp4",
        job={"asset": {"path": "memory", "metadata": {"fps": 5.0}},
             "window": {"start_s": 0.0, "end_s": (count + 1) * 0.2},
             "orientation": 0, "fps_override": None},
        units=units)
    assert written == count
    return counted


def test_per_frame_work_is_bounded_by_trail(tmp_path, monkeypatch):
    small = _run(tmp_path, 300, monkeypatch)
    large = _run(tmp_path, 1200, monkeypatch)
    # 端点每段只换算一次（预处理），不随帧数重复。
    assert small["conversions"] == 2 * 300
    assert large["conversions"] == 2 * 1200
    # 每帧画的是当前保留段：预算内随段数增长，超过预算后被 1000 封顶。
    assert small["max_arrays"] == 300
    assert large["max_arrays"] == LIMIT - 1
    assert large["arrays"] < large["calls"] * LIMIT
    # 每帧一次合并调用：调用次数 == 输出片段数，不随历史段数增长。
    assert small["calls"] == 300
    assert large["calls"] == 1200
    # 增长规律：1200 帧（其中 200 帧处于满预算）总栅格化数组数远低于
    # “每帧重放全部历史”的平方级成本。
    assert large["arrays"] < 1200 * LIMIT


def _split_linear(start: float, end: float, points) -> list[tuple]:
    """8f39454 的切分写法：每帧逐点扫全表，用于对照二分版结果。"""

    cuts = [start]
    for point in points:
        if start + 1e-9 < point < end - 1e-9:
            cuts.append(point)
    cuts.append(end)
    return [(cuts[index], cuts[index + 1])
            for index in range(len(cuts) - 1)
            if cuts[index + 1] - cuts[index] > 1e-12]


def test_split_matches_linear_reference():
    rng = random.Random(11)
    for _ in range(40):
        points = sorted({round(rng.uniform(0, 10), 6) for _ in range(12)})
        assert annotated._split(0.0, 10.0, points) == \
            _split_linear(0.0, 10.0, points)
    assert annotated._split(1.0, 2.0, [1.0, 2.0]) == [(1.0, 2.0)]
    assert annotated._split(1.0, 2.0, []) == [(1.0, 2.0)]
    assert annotated._split(1.0, 1.0 + 1e-13, [1.0]) == []
    assert annotated._split(0.0, 1.0, [0.5, 0.5]) == [(0.0, 0.5), (0.5, 1.0)]


def test_video_trail_bounded_while_static_figure_keeps_full_data(
        tmp_path, monkeypatch):
    count = LIMIT + 200
    _Source.frames_spec = _spec(count)
    monkeypatch.setattr(annotated.media, "VideoSource", _Source)
    counted = {"max_arrays": 0}
    real_poly = cv2.polylines

    def poly(canvas, polys, *args, **kwargs):
        if isinstance(polys, list):
            counted["max_arrays"] = max(counted["max_arrays"], len(polys))
        return real_poly(canvas, polys, *args, **kwargs)

    monkeypatch.setattr(cv2, "polylines", poly)
    rows = _chain(count)
    units = [{"unit_id": UNIT_A, "unit_name": "u",
              "window_start_s": 0.0, "window_end_s": (count + 1) * 0.2,
              "geometry": {}, "observations": rows}]
    written, _ = annotated.write_run_video(
        tmp_path / "bounded.mp4",
        job={"asset": {"path": "memory", "metadata": {"fps": 5.0}},
             "window": {"start_s": 0.0, "end_s": (count + 1) * 0.2},
             "orientation": 0, "fps_override": None},
        units=units)
    assert written == count
    # 视频尾迹有界：任何一帧画出的段都不超过 1000 节点对应的 999 条边。
    assert counted["max_arrays"] == LIMIT - 1
    # 静态完整轨迹图仍使用全部完成段，不随尾迹淘汰。
    assert len(figures.motion_segments(rows)) == count
