"""第 08 块三项固定预期：OFT 区域、事件、运动与分箱。"""

from __future__ import annotations

import pytest

from behaviourlab import timeline
from behaviourlab.experiments import oft
from behaviourlab.schema import Detection, Frame, ValidationError

TOL = 1e-8
ARENA = {"type": "quad",
         "points": [[0, 0], [200, 0], [200, 200], [0, 200]]}
UNIT = {"id": "unit-1", "name": "小鼠 1", "animal_id": None}


def options(**overrides) -> dict:
    values = dict(oft.DEFAULTS)
    values.update(overrides)
    return values


def frame(index: int, time_s: float, span: float) -> Frame:
    return Frame(image=None, index=index, time_s=time_s, dt_s=span,
                 elapsed_s=time_s, analysis_fps=5.0,
                 interval_start_s=time_s, interval_end_s=time_s + span)


def rows_for(points, span: float = 0.2) -> list[dict]:
    rows = []
    for index, point in enumerate(points):
        time_s = index * span
        detection = Detection(center=list(point)) if point else Detection()
        rows.append(timeline.base_row(UNIT, frame(index, time_s, span),
                                      detection))
    return rows


def unit(options_, window=None) -> dict:
    return {"id": UNIT["id"], "name": UNIT["name"], "animal_id": None,
            "options": options_, "geometry": {"arena": ARENA},
            "window": window}


def process(points, options_=None, window=None, span=0.2, duration=4.0):
    return oft.process_unit(
        unit(options_ or options(), window),
        rows_for(points, span),
        timeline.Window(0.0, duration),
        0.0,
    )


def test_first_fixture_matches_fixed_expectations():
    points = ([(40, 100)] * 5 + [(80, 100)] * 5 + [(120, 100)] * 5
              + [(160, 100)] * 5)
    facts = process(points)
    summary = facts["summary"]

    assert summary["total_time_s"] == pytest.approx(4.0, abs=TOL)
    assert summary["observed_time_s"] == pytest.approx(4.0, abs=TOL)
    assert summary["missing_time_s"] == pytest.approx(0.0, abs=TOL)
    assert summary["center_time_s"] == pytest.approx(2.0, abs=TOL)
    assert summary["outer_time_s"] == pytest.approx(2.0, abs=TOL)
    assert summary["outside_time_s"] == pytest.approx(0.0, abs=TOL)
    assert summary["wall_time_s"] == pytest.approx(0.0, abs=TOL)
    assert summary["center_percent"] == pytest.approx(50.0, abs=TOL)
    assert summary["outer_percent"] == pytest.approx(50.0, abs=TOL)
    assert summary["detected_frames"] == 20
    assert summary["window_frames"] == 20
    assert summary["detected_rate_percent"] == pytest.approx(100.0, abs=TOL)
    assert summary["center_entries"] == 1
    assert summary["center_latency_s"] == pytest.approx(1.0, abs=TOL)

    assert summary["total_distance_cm"] == pytest.approx(30.0, abs=TOL)
    assert summary["observed_total_distance_cm"] == pytest.approx(30.0, abs=TOL)
    assert summary["center_distance_cm"] == pytest.approx(25.0, abs=TOL)
    assert summary["outer_distance_cm"] == pytest.approx(5.0, abs=TOL)
    assert summary["outside_distance_cm"] == pytest.approx(0.0, abs=TOL)
    assert summary["measurable_motion_time_s"] == pytest.approx(3.8, abs=TOL)
    assert summary["moving_time_s"] == pytest.approx(0.6, abs=TOL)
    assert summary["immobility_time_s"] == pytest.approx(3.2, abs=TOL)
    assert summary["mean_speed_cm_s"] == pytest.approx(150.0 / 19.0, abs=TOL)
    assert summary["max_speed_cm_s"] == pytest.approx(50.0, abs=TOL)

    assert len(facts["bins"]) == 1
    single = facts["bins"][0]
    assert single["bin_index"] == 1
    assert single["start_s"] == pytest.approx(0.0, abs=TOL)
    assert single["end_s"] == pytest.approx(4.0, abs=TOL)
    assert single["distance_cm"] == pytest.approx(30.0, abs=TOL)
    assert single["center_distance_cm"] == pytest.approx(25.0, abs=TOL)
    assert single["observed_time_s"] == pytest.approx(4.0, abs=TOL)
    assert single["mean_speed_cm_s"] == pytest.approx(150.0 / 19.0, abs=TOL)

    assert [round(row["x_cm"], 8) for row in facts["rows"][:1]] == [10.0]
    assert [round(row["x_cm"], 8) for row in facts["rows"][5:6]] == [20.0]
    assert facts["events"][0]["time_s"] == pytest.approx(1.0, abs=TOL)
    assert facts["events"][0]["evidence_complete"] is True


def test_second_fixture_matches_fixed_expectations():
    points = ([(40, 100)] * 4 + [(60, 100)] * 4 + [None] * 4
              + [(100, 100)] * 4 + [(120, 100)] * 4)
    facts = process(points)
    summary = facts["summary"]

    assert summary["observed_time_s"] == pytest.approx(3.2, abs=TOL)
    assert summary["missing_time_s"] == pytest.approx(0.8, abs=TOL)
    assert summary["detected_frames"] == 16
    assert summary["window_frames"] == 20
    assert summary["detected_rate_percent"] == pytest.approx(80.0, abs=TOL)
    assert summary["center_time_s"] == pytest.approx(2.4, abs=TOL)
    assert summary["outer_time_s"] == pytest.approx(0.8, abs=TOL)
    assert summary["center_percent"] == pytest.approx(75.0, abs=TOL)
    assert summary["outer_percent"] == pytest.approx(25.0, abs=TOL)
    assert summary["center_entries"] == 1
    assert summary["center_latency_s"] == pytest.approx(0.8, abs=TOL)

    assert summary["total_distance_cm"] == pytest.approx(10.0, abs=TOL)
    assert summary["center_distance_cm"] == pytest.approx(7.5, abs=TOL)
    assert summary["outer_distance_cm"] == pytest.approx(2.5, abs=TOL)
    assert summary["measurable_motion_time_s"] == pytest.approx(2.8, abs=TOL)
    assert summary["moving_time_s"] == pytest.approx(0.4, abs=TOL)
    assert summary["immobility_time_s"] == pytest.approx(2.4, abs=TOL)
    assert summary["mean_speed_cm_s"] == pytest.approx(25.0 / 7.0, abs=TOL)
    assert summary["max_speed_cm_s"] == pytest.approx(25.0, abs=TOL)

    recovery = facts["rows"][12]
    assert recovery["time_s"] == pytest.approx(2.4, abs=TOL)
    assert recovery["step"] is None
    assert recovery["speed"] is None
    assert recovery["step_distance_cm"] is None


def test_third_fixture_zones_and_unknown_window_validation():
    facts = process([(20, 100), None, (100, 100)], span=1.0, duration=3.0)
    summary = facts["summary"]

    assert summary["total_time_s"] == pytest.approx(3.0, abs=TOL)
    assert summary["observed_time_s"] == pytest.approx(2.0, abs=TOL)
    assert summary["missing_time_s"] == pytest.approx(1.0, abs=TOL)
    assert summary["outer_time_s"] == pytest.approx(1.0, abs=TOL)
    assert summary["center_time_s"] == pytest.approx(1.0, abs=TOL)
    assert summary["wall_time_s"] == pytest.approx(1.0, abs=TOL)
    assert summary["outside_time_s"] == pytest.approx(0.0, abs=TOL)
    assert summary["center_percent"] == pytest.approx(50.0, abs=TOL)
    assert summary["outer_percent"] == pytest.approx(50.0, abs=TOL)
    assert summary["wall_percent"] == pytest.approx(50.0, abs=TOL)

    assert summary["total_distance_cm"] == pytest.approx(0.0, abs=TOL)
    assert summary["measurable_motion_time_s"] == pytest.approx(0.0, abs=TOL)
    assert summary["moving_time_s"] == pytest.approx(0.0, abs=TOL)
    assert summary["immobility_time_s"] == pytest.approx(0.0, abs=TOL)
    assert summary["mean_speed_cm_s"] is None
    assert summary["max_speed_cm_s"] is None
    assert facts["rows"][0]["zone"] == "Wall"
    assert facts["rows"][1]["zone"] == "Unknown"
    assert facts["rows"][1]["x_cm"] is None

    assert summary["center_entries"] == 1
    assert summary["center_latency_s"] == pytest.approx(2.0, abs=TOL)
    event = facts["events"][0]
    assert event["time_s"] == pytest.approx(2.0, abs=TOL)
    assert event["evidence_complete"] is False
    assert event["kind"] == "center_entry"


def test_parameter_and_quad_validation():
    assert oft.validate_unit(options(), {"arena": ARENA}) == []
    assert oft.validate_unit(options(arena_width_cm=0.0), {"arena": ARENA}) == [
        {"field": "options.arena_width_cm", "message": "场地宽度必须 > 0"}]
    assert oft.validate_unit(options(bin_seconds=0.0), {"arena": ARENA}) == [
        {"field": "options.bin_seconds", "message": "时间箱长度必须 > 0"}]
    assert oft.validate_unit(options(wall_margin_cm=26.0),
                             {"arena": ARENA}) == [
        {"field": "options.wall_margin_cm",
         "message": "壁区宽度不能超过场地短边的一半"}]
    assert oft.validate_unit(options(wall_margin_cm=0.0, center_ratio_x=1.0,
                                     movement_threshold_cm_s=0.0),
                             {"arena": ARENA}) == []

    crossed = {"type": "quad", "points": [[0, 0], [200, 200], [200, 0], [0, 200]]}
    with pytest.raises(ValidationError):
        oft.arena_matrix({"arena": crossed}, options())
    with pytest.raises(ValidationError):
        oft.arena_matrix({"arena": {"type": "quad",
                                    "points": [[0, 0], [0, 0], [0, 0], [0, 0]]}},
                         options())

    assert oft.zone_of(12.5, 25.0, 50.0, 50.0, 0.5, 0.5, 8.0) == "Center"
    assert oft.zone_of(5.0, 25.0, 50.0, 50.0, 0.5, 0.5, 8.0) == "Wall"


class FakeSource:
    def __init__(self, frames):
        self._frames = frames

    def iter_frames(self):
        return iter(self._frames)


def fake_frame(index: int, image) -> Frame:
    start = index * 0.2
    return Frame(image=image, index=index, time_s=start, dt_s=0.2,
                 elapsed_s=start, analysis_fps=5.0,
                 interval_start_s=start, interval_end_s=start + 0.2)


def test_analyze_captures_once_and_builds_result():
    import cv2
    import numpy as np

    from behaviourlab import detection
    from behaviourlab.schema import AnalysisContext

    image = np.full((200, 200, 3), 255, dtype=np.uint8)
    cv2.circle(image, (40, 100), 8, (0, 0, 0), -1)
    frames = [fake_frame(index, image.copy()) for index in range(5)]
    job = {
        "assay": "oft", "run_id": "run-fixed", "job_id": "job-fixed",
        "project_id": None,
        "window": {"start_s": 0, "end_s": None},
        "orientation": 0, "fps_override": None, "sample_fps": None,
        "asset": {"id": "asset-fixed", "path": "memory",
                  "metadata": {"duration_s": 1.0, "fps": 5.0}},
        "units": [{
            "id": "unit-fixed", "name": "小鼠 1", "animal_id": None,
            "window": None, "geometry": {"arena": ARENA},
            "options": options(), "detection": dict(detection.DETECTION_DEFAULTS),
        }],
    }
    context = AnalysisContext(job=job, source=FakeSource(frames))
    result = oft.analyze(context)

    assert len(result.observations) == 5
    assert all(row["detected"] for row in result.observations)
    assert result.summary[0]["window_frames"] == 5
    assert result.summary[0]["detected_frames"] == 5
    assert result.metadata["units"]["unit-fixed"]["coordinate_unit"] == "cm"
    assert result.observations[0]["coordinate_unit"] == "cm"
    assert abs(result.observations[0]["x_cm"] - 10.0) <= 1e-6
