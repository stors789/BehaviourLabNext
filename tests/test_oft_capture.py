"""第 08 块收口：捕获时钟、首检、行来源与中心事件的固定失败例。"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from behaviourlab import detection, timeline
from behaviourlab.experiments import oft
from behaviourlab.schema import AnalysisContext, Detection, Frame

TOL = 1e-8
ARENA = {"type": "quad", "points": [[0, 0], [199, 0], [199, 199], [0, 199]]}
SQUARE = {"type": "quad", "points": [[0, 0], [200, 0], [200, 200], [0, 200]]}


class Source:
    def __init__(self, frames):
        self._frames = frames

    def iter_frames(self):
        return iter(self._frames)


def frame(index: int, time_s: float, span: float = 0.1, image=None) -> Frame:
    return Frame(image=image, index=index, time_s=time_s, dt_s=span,
                 elapsed_s=time_s, analysis_fps=1.0 / span,
                 interval_start_s=time_s, interval_end_s=time_s + span)


def blank(dark: bool = True):
    image = np.full((200, 200, 3), 255, dtype=np.uint8)
    if dark:
        cv2.rectangle(image, (95, 95), (105, 105), (0, 0, 0), -1)
    return image


def unit(unit_id: str, name: str, *, method: str | None = None,
         window=None, geometry=None) -> dict:
    settings = dict(detection.common_detection_defaults())
    if method:
        settings["bg_method"] = method
    return {"id": unit_id, "name": name, "animal_id": None, "window": window,
            "geometry": {"arena": geometry or ARENA},
            "options": dict(oft.DEFAULTS), "detection": settings}


def job(units, *, metadata=None, window=None) -> dict:
    return {
        "assay": "oft", "run_id": "run-capture", "job_id": "job-capture",
        "project_id": None,
        "window": window or {"start_s": 0.0, "end_s": None},
        "orientation": 0, "fps_override": None, "sample_fps": None,
        "asset": {"id": "a" * 32, "path": "memory",
                  "metadata": metadata or {}},
        "units": units,
        "style": {"color": "#B86245", "colormap": "viridis",
                  "line_width": 2, "dpi": 100},
    }


@pytest.mark.parametrize("metadata", [
    {},
    {"duration_s": 0.5, "fps": 20.0, "frame_count": 10},
    {"duration_s": 2.0, "fps": 20.0, "frame_count": 10},
])
def test_video_window_follows_captured_clock(metadata):
    # 10 帧、每帧曝光 .1 秒，实际末端 1 秒；元数据时长不能把窗口定死。
    frames = [frame(index, index * 0.1, 0.1, blank())
              for index in range(10)]
    result = oft.analyze(AnalysisContext(
        job=job([unit("u" * 32, "小鼠 1")], metadata=metadata),
        source=Source(frames)))
    summary = result.summary[0]
    assert summary["analysis_window_end_s"] == pytest.approx(1.0, abs=TOL)
    assert summary["total_time_s"] == pytest.approx(1.0, abs=TOL)
    assert summary["observed_time_s"] == pytest.approx(1.0, abs=TOL)


def test_first_seen_ignores_tail_context():
    # 视频窗 [0,.1)；只有 time_s=.1 的尾上下文检测成功，不能拿来锚定相对窗。
    frames = [frame(0, 0.0, 0.1, blank(False)),
              frame(1, 0.1, 0.1, blank(True))]
    result = oft.analyze(AnalysisContext(
        job=job([unit("u" * 32, "小鼠 1",
                      window={"mode": "relative", "offset_s": 0.0,
                              "duration_s": 0.1})],
                metadata={"duration_s": 0.2},
                window={"start_s": 0.0, "end_s": 0.1}),
        source=Source(frames)))
    summary = result.summary[0]
    assert summary["analysis_window_start_s"] is None
    assert summary["analysis_window_end_s"] is None
    assert summary["total_time_s"] == pytest.approx(0.0, abs=TOL)
    assert result.events == []
    assert any("未检测到动物，无法确定相对时间窗" in text
               for text in result.warnings)


def test_rows_record_detection_and_head_methods():
    frames = [frame(index, index * 0.1, 0.1, blank()) for index in range(3)]
    result = oft.analyze(AnalysisContext(
        job=job([unit("a" * 32, "小鼠 1", method="threshold"),
                 unit("b" * 32, "小鼠 2", method="local_otsu")]),
        source=Source(frames)))
    seen = {(row["unit_id"], row["detection_method"]) for row in result.observations}
    assert seen == {("a" * 32, "threshold"), ("b" * 32, "local_otsu")}
    assert {row["head_method"] for row in result.observations} == {None}

    # 有双端证据才标 classical_geometry；缺一端仍是显式 null。
    two_ends = Detection(center=[100.0, 100.0], nose=[120.0, 100.0],
                         tail_base=[80.0, 100.0])
    one_end = Detection(center=[100.0, 100.0], nose=[120.0, 100.0])
    spec = unit("c" * 32, "小鼠 3", method="threshold")
    head_row = oft.observation_row(spec, frame(0, 0.0), two_ends)
    tail_row = oft.observation_row(spec, frame(0, 0.0), one_end)
    assert head_row["head_method"] == "classical_geometry"
    assert tail_row["head_method"] is None
    assert head_row["detection_method"] == "threshold"


def _rows(points, span=1.0):
    rows = []
    for index, point in enumerate(points):
        moment = index * span
        rows.append(timeline.base_row(
            {"id": "unit-1", "name": "小鼠 1", "animal_id": None},
            frame(index, moment, span),
            Detection(center=list(point)) if point else Detection()))
    return rows


def _spec(window):
    return {"id": "unit-1", "name": "小鼠 1", "animal_id": None,
            "options": dict(oft.DEFAULTS), "geometry": {"arena": SQUARE},
            "window": window}


def test_center_events_only_use_window_rows():
    # 动物窗 [1,2)：窗外的前后中心上下文不算进入，也不生成窗外事件。
    facts = oft.process_unit(
        _spec({"mode": "exact", "start_s": 1.0, "end_s": 2.0}),
        _rows([(100, 100), (40, 100), (100, 100)]),
        timeline.Window(0.0, 3.0), None)
    assert facts["summary"]["center_entries"] == 0
    assert facts["summary"]["center_latency_s"] is None
    assert facts["events"] == []
    assert facts["summary"]["total_time_s"] == pytest.approx(1.0, abs=TOL)


def test_unknown_window_has_no_events_and_no_crash():
    # first_seen 为空：窗口未确定，只给警告，不把 None 当时间。
    facts = oft.process_unit(
        _spec({"mode": "relative", "offset_s": 0.0, "duration_s": 1.0}),
        _rows([(100, 100), None]),
        timeline.Window(1.0, 2.0), None)
    assert facts["summary"]["analysis_window_start_s"] is None
    assert facts["summary"]["analysis_window_end_s"] is None
    assert facts["events"] == []
    assert facts["summary"]["center_entries"] == 0
    assert facts["summary"]["total_time_s"] == pytest.approx(0.0, abs=TOL)
    assert "未检测到动物" in facts["warning"]
