"""第 07 块三项固定预期：曝光/缺失、窗口裁切与分箱、窗口解析。"""

from __future__ import annotations

import pytest

from behaviourlab import timeline
from behaviourlab.schema import Detection, Frame, ValidationError

TOL = 1e-10
UNIT = {"id": "unit-1", "name": "小鼠 1", "animal_id": None}


def frame(index: int, time_s: float, start_s: float, end_s: float) -> Frame:
    return Frame(
        image=None, index=index, time_s=time_s, dt_s=end_s - start_s,
        elapsed_s=time_s, analysis_fps=10.0,
        interval_start_s=start_s, interval_end_s=end_s,
    )


def rows_for(samples, unit=UNIT) -> list[dict]:
    rows = []
    for index, (time_s, start_s, end_s, point) in enumerate(samples):
        detection = Detection(center=list(point)) if point else Detection()
        rows.append(timeline.base_row(
            unit, frame(index, time_s, start_s, end_s), detection))
    return rows


def test_exposure_segment_and_heatmap_weights():
    rows = rows_for([
        (0.0, 0.0, 1.0, (1.0, 1.0)),
        (1.0, 1.0, 3.0, None),
        (3.0, 3.0, 6.0, (10.0, 10.0)),
        (6.0, 6.0, 7.0, (11.0, 10.0)),
    ])
    window = timeline.Window(0.0, 7.0)
    clipped = timeline.clip_rows(rows, window)
    summary = timeline.time_summary(clipped)
    assert summary["total_time_s"] == pytest.approx(7.0, abs=TOL)
    assert summary["observed_time_s"] == pytest.approx(5.0, abs=TOL)
    assert summary["missing_time_s"] == pytest.approx(2.0, abs=TOL)

    segments = timeline.motion_segments(clipped, window)
    assert len(segments) == 1
    segment = segments[0]
    assert (segment["start_s"], segment["end_s"]) == (3.0, 6.0)
    assert segment["distance"] == pytest.approx(1.0, abs=TOL)
    assert segment["speed"] == pytest.approx(1.0 / 3.0, abs=TOL)

    timeline.attach_motion(clipped, segments)
    assert clipped[1]["step"] is None and clipped[1]["speed"] is None
    assert clipped[2]["step"] is None and clipped[2]["speed"] is None
    assert clipped[3]["step"] == pytest.approx(1.0, abs=TOL)
    assert clipped[3]["speed"] == pytest.approx(1.0 / 3.0, abs=TOL)

    weights = timeline.heatmap_weights(clipped)
    assert [round(weight, 10) for _, _, weight in weights] == [1.0, 3.0, 1.0]


def test_window_clipping_segments_and_bins():
    rows = rows_for([
        (0.0, 0.0, 0.1, (0.0, 0.0)),
        (0.1, 0.1, 0.2, (10.0, 0.0)),
        (0.2, 0.2, 0.3, (20.0, 0.0)),
        (0.3, 0.3, 0.4, (30.0, 0.0)),
    ])
    window = timeline.Window(0.15, 0.25)
    clipped = timeline.clip_rows(rows, window)
    assert [row["dt_s"] for row in clipped] == pytest.approx(
        [0.0, 0.05, 0.05, 0.0], abs=TOL)
    assert [row["in_window"] for row in clipped] == [False, True, True, False]
    # 源区间身份保持原样，窗裁切不改成 0.15 起点。
    assert clipped[1]["source_interval_start_s"] == pytest.approx(0.1, abs=TOL)

    summary = timeline.time_summary(clipped)
    assert summary["total_time_s"] == pytest.approx(0.10, abs=TOL)
    assert summary["observed_time_s"] == pytest.approx(0.10, abs=TOL)

    segments = timeline.motion_segments(clipped, window)
    assert [(round(item["start_s"], 10), round(item["end_s"], 10))
            for item in segments] == [(0.15, 0.2), (0.2, 0.25)]
    assert [round(item["distance"], 10) for item in segments] == [5.0, 5.0]
    assert [round(item["speed"], 10) for item in segments] == [100.0, 100.0]

    bins = timeline.build_bins(clipped, segments, window, 0.05)
    assert [item["bin_index"] for item in bins] == [1, 2]
    assert [(round(item["start_s"], 10), round(item["end_s"], 10))
            for item in bins] == [(0.0, 0.05), (0.05, 0.10)]
    assert [round(item["duration_s"], 10) for item in bins] == [0.05, 0.05]
    assert [round(item["exposure_s"], 10) for item in bins] == [0.05, 0.05]
    assert [round(item["distance"], 10) for item in bins] == [5.0, 5.0]

    # 边界半开：恰好落在箱界的时间不重复分配。
    pieces = timeline.split_interval(0.15, 0.25, 0.15, 0.05)
    assert [(index, round(stop - start, 10)) for index, start, stop in pieces] \
        == [(0, 0.05), (1, 0.05)]


def test_window_resolution_and_unresolved_cases():
    video = timeline.Window(0.0, 10.0)
    relative = {"mode": "relative", "offset_s": -1.0, "duration_s": 3.0}
    resolved = timeline.resolve_unit_window(relative, video, first_seen=0)
    assert (resolved.start_s, resolved.end_s) == (0.0, 2.0)
    assert resolved.duration_s == pytest.approx(2.0, abs=TOL)

    unknown = timeline.resolve_unit_window(relative, video, first_seen=None)
    assert unknown.known is False
    assert unknown.duration_s == 0.0
    rows = timeline.clip_rows(
        rows_for([(0.0, 0.0, 1.0, (1.0, 1.0))]), unknown)
    assert rows[0]["window_start_s"] is None
    assert rows[0]["dt_s"] == 0.0
    assert rows[0]["in_window"] is False
    assert timeline.time_summary(rows)["total_time_s"] == 0.0
    assert timeline.build_bins(rows, [], unknown, 5.0) == []
    assert timeline.window_warning("oft", unknown)
    assert timeline.window_warning("nor", unknown)
    assert timeline.window_warning("epm", unknown) is None

    past = {"mode": "relative", "offset_s": 5.0, "duration_s": 2.0}
    empty = timeline.resolve_unit_window(past, video, first_seen=8.0)
    assert (empty.start_s, empty.end_s) == (13.0, 13.0)
    assert empty.duration_s == 0.0

    with pytest.raises(ValidationError):
        timeline.resolve_unit_window(
            {"mode": "exact", "start_s": 11.0, "end_s": 12.0}, video)

    # TST/FST：场外中心不算起点，进入场地后才锚定相对窗。
    assert timeline.first_seen_time(
        "tst", Detection(center=[200.0, 200.0]), 1.0,
        {"arena": {"type": "quad",
                   "points": [[0, 0], [100, 0], [100, 100], [0, 100]]}}) is None
    assert timeline.first_seen_time(
        "tst", Detection(center=[50.0, 50.0]), 3.0,
        {"arena": {"type": "quad",
                   "points": [[0, 0], [100, 0], [100, 100], [0, 100]]}}) == 3.0
    shifted = timeline.resolve_unit_window(
        {"mode": "relative", "offset_s": 0.0, "duration_s": 2.0},
        video, first_seen=3.0)
    assert (shifted.start_s, shifted.end_s) == (3.0, 5.0)

    assert timeline.ratio(1.0, 0.0) is None
    assert timeline.ratio(1.0, 4.0) == 0.25

    # 视频窗口：显式终点不超过实际末端；无观测才用元数据时长。
    observed = rows_for([(0.0, 0.0, 6.5, (1.0, 1.0))])
    capped = timeline.resolve_video_window(
        {"start_s": 0.0, "end_s": 10.0}, observed)
    assert (capped.start_s, capped.end_s) == (0.0, 6.5)
    fallback = timeline.resolve_video_window(
        {"start_s": 0.0, "end_s": None}, [], metadata_duration=8.0)
    assert (fallback.start_s, fallback.end_s) == (0.0, 8.0)
    nothing = timeline.resolve_video_window(
        {"start_s": 2.0, "end_s": None}, [], metadata_duration=None)
    assert (nothing.start_s, nothing.end_s) == (2.0, 2.0)
    unit_global = timeline.resolve_unit_window(None, capped)
    assert (unit_global.start_s, unit_global.end_s) == (0.0, 6.5)
