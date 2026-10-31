"""第 09 块 TST classic：单位换算、固定状态序列与参数校验。"""

from __future__ import annotations

import pytest

from behaviourlab import activity, timeline
from behaviourlab.experiments import tst

TOL = 1e-9
UNIT = {"id": "unit-1", "name": "小鼠 1", "animal_id": None}
ARENA = {"type": "polygon",
         "points": [[5, 5], [120, 5], [120, 120], [5, 120]]}
WIDE_ARENA = {"type": "polygon",
              "points": [[0, 0], [199, 0], [199, 199], [0, 199]]}
BODY_LINE = {"type": "line", "points": [[10, 10], [70, 10]]}


def options(**overrides) -> dict:
    values = dict(tst.DEFAULTS)
    values.update(overrides)
    return values


def test_body_line_scale_and_vfr_speeds():
    scale, length = activity.line_scale(BODY_LINE)
    assert length == pytest.approx(60.0, abs=TOL)
    assert scale == pytest.approx(1.0, abs=TOL)

    nominal, actual = activity.speeds(5.0, scale, 10.0, 0.1)
    assert nominal == pytest.approx(50.0, abs=TOL)
    assert actual == pytest.approx(50.0, abs=TOL)

    # VFR：帧 2 的真实时间差 .22 秒，名义速度仍按分析频率。
    nominal, actual = activity.speeds(5.0, scale, 10.0, 0.22)
    assert nominal == pytest.approx(50.0, abs=TOL)
    assert actual == pytest.approx(5.0 / 0.22, abs=TOL)
    assert actual != pytest.approx(nominal, abs=1.0)


def evidence_row(index: int, *, feature: bool, swing: bool) -> dict:
    start = 0.1 * index
    row = {
        "unit_id": UNIT["id"],
        "unit_name": UNIT["name"],
        "frame": index,
        "time_s": start,
        "elapsed_s": start,
        "analysis_fps": 10.0,
        "interval_start_s": start,
        "interval_end_s": start + 0.1,
        "source_interval_start_s": start,
        "source_interval_end_s": start + 0.1,
        "dt_s": 0.1,
        "in_window": True,
        "feature_ready": feature,
        "detected": feature,
        "swing_motion_candidate": swing,
        "swing_candidate": swing,
        "swing_rigid": True,
        "swing_score": 4.0 if swing else 0.0,
        "swing_autocorrelation": 0.8 if swing else 0.0,
        "flow_dispersion": 0.05 if swing else 0.0,
        "decision_flow_p90_mm_s": 60.0 if swing else 0.0,
        "speed_std_mm_s": 0.0,
        "flow_ratio": 1.5 if swing else 0.0,
    }
    return row


def test_classic_state_fixture_matches_fixed_sequence():
    rows = [evidence_row(index, feature=index != 20, swing=index < 20)
            for index in range(35)]
    tst.classic_states(rows, flow_hi=50.0, flow_lo=35.0,
                       speed_hi=40.0, speed_lo=20.0)

    assert [row["state"] for row in rows[:20]] == ["swing"] * 20
    assert rows[20]["state"] == "unknown"
    assert [row["state"] for row in rows[21:]] == ["immobile"] * 14

    for row in rows[21:]:
        assert row["flow_ratio"] == 0.0
        assert row["swing_score"] == 0.0
        assert row["swing_autocorrelation"] == 0.0
        assert row["swing_motion_candidate"] is False
        assert row["swing_candidate"] is False
        assert row["swing_rigid"] is True
    for row in rows[:20]:
        assert row["swing_motion_candidate"] is True
        assert row["is_struggling"] is False
        assert row["is_swinging"] is True
    assert rows[20]["is_struggling"] is None
    assert rows[20]["is_swinging"] is False

    events = tst.build_events(rows, UNIT["id"], UNIT["name"])
    assert events == []
    info = {"flow_thr_hi": 50.0, "flow_thr_lo": 35.0,
            "speed_thr_hi": 40.0, "speed_thr_lo": 20.0,
            "estimated": False, "samples": 0}
    summary = tst.summarize(rows, {"id": UNIT["id"], "name": UNIT["name"],
                                   "options": options()},
                            timeline.Window(0.0, 3.5), info, events)
    assert summary["swing_time_s"] == pytest.approx(2.0, abs=TOL)
    assert summary["struggling_time_s"] == pytest.approx(0.0, abs=TOL)
    assert summary["uncertain_time_s"] == pytest.approx(0.1, abs=TOL)
    assert summary["total_time_s"] == pytest.approx(3.5, abs=TOL)
    # 原始 immobile 标签 1.4 秒，汇总把 2 秒摆动并入非挣扎时间。
    assert summary["immobility_time_s"] == pytest.approx(3.4, abs=TOL)
    assert summary["immobility_percent"] == pytest.approx(3.4 / 3.5 * 100.0,
                                                          abs=1e-9)
    assert summary["latency_to_immobility_s"] is None
    assert summary["immobility_episodes"] == 0
    assert summary["struggling_episodes"] == 0


def test_parameters_and_geometry_validation():
    geometry = {"arena": ARENA, "body_line": BODY_LINE}
    assert tst.validate_unit(options(), geometry) == []

    crossed = {"arena": {"type": "polygon",
                         "points": [[5, 5], [5, 5], [5, 5]]},
               "body_line": BODY_LINE}
    assert tst.validate_unit(options(), crossed, complete=True) != []

    bad_pair = tst.validate_unit(options(flow_thr_lo=60.0), geometry,
                                 provided={"flow_thr_lo"})
    assert bad_pair[0]["field"] == "options.flow_thr_lo"

    bad_sensitivity = tst.validate_unit(
        options(behavior_threshold_sensitivity=4.0), geometry,
        provided={"behavior_threshold_sensitivity"})
    assert bad_sensitivity[0]["field"] == "options.behavior_threshold_sensitivity"

    not_boolean = tst.validate_unit(options(auto_behavior_threshold="yes"),
                                    geometry,
                                    provided={"auto_behavior_threshold"})
    assert not_boolean[0]["message"] == "必须是布尔值"

    # lis 参数可以保存，但不能在本块开始任务。
    assert tst.validate_unit(options(mode="lis"), geometry) == []
    rejected = tst.validate_unit(options(mode="lis"), geometry, complete=True)
    assert any(issue["field"] == "options.mode" and "尚未接入" in issue["message"]
               for issue in rejected)

    missing = tst.validate_unit(options(), {}, complete=True)
    fields = {issue["field"] for issue in missing}
    assert {"geometry.arena", "geometry.body_line"}.issubset(fields)


def test_short_line_gives_finer_scale():
    scale, length = activity.line_scale(
        {"type": "line", "points": [[0, 0], [30, 0]]})
    assert length == pytest.approx(30.0, abs=TOL)
    assert scale == pytest.approx(2.0, abs=TOL)


class FakeSource:
    def __init__(self, frames):
        self._frames = frames

    def iter_frames(self):
        return iter(self._frames)


def frame(index, center=None, size=200, span=0.1):
    from behaviourlab.schema import Frame
    import cv2
    import numpy as np

    image = np.full((size, size, 3), 255, dtype=np.uint8)
    if center is not None:
        cv2.ellipse(image, center, (24, 12), 0, 0, 360, (0, 0, 0), -1)
    start = index * span
    return Frame(image=image, index=index, time_s=start, dt_s=span,
                 elapsed_s=start, analysis_fps=1.0 / span,
                 interval_start_s=start, interval_end_s=start + span)


def analyze_frames(centers, options_override=None):
    from behaviourlab import detection
    from behaviourlab.schema import AnalysisContext

    frames = [frame(index, center) for index, center in enumerate(centers)]
    settings = dict(detection.common_detection_defaults())
    settings.update(tst.DETECTION_DEFAULTS)
    job = {
        "assay": "tst", "run_id": "run-tst", "job_id": "job-tst",
        "project_id": None,
        "window": {"start_s": 0.0, "end_s": None},
        "orientation": 0, "fps_override": None, "sample_fps": None,
        "asset": {"id": "a" * 32, "path": "memory", "metadata": {}},
        "units": [{
            "id": "u" * 32, "name": "小鼠 1", "animal_id": None,
            "window": None,
            "geometry": {"arena": WIDE_ARENA, "body_line": BODY_LINE},
            "options": options(**(options_override or {})),
            "detection": settings,
        }],
        "style": {},
    }
    return tst.analyze(AnalysisContext(job=job, source=FakeSource(frames)))


def test_capture_features_and_recovery_rows():
    centers = [(60 + 5 * index, 60) for index in range(10)]
    centers[5] = None
    centers[6] = None
    result = analyze_frames(centers)
    rows = result.observations

    assert len(rows) == 10
    assert rows[0]["feature_ready"] is False
    assert rows[0]["distance_px"] is None
    assert rows[1]["feature_ready"] is True
    assert rows[1]["distance_px"] == pytest.approx(5.0, abs=1e-6)
    assert rows[1]["speed_nominal_mm_s"] == pytest.approx(50.0, abs=1e-6)
    assert rows[1]["speed_mm_s"] == pytest.approx(50.0, abs=1e-6)

    # 缺失后恢复首帧没有凭空配对；下一帧才重新有段。
    assert rows[7]["feature_ready"] is False
    assert rows[7]["state"] == "unknown"
    assert rows[8]["feature_ready"] is True
    assert rows[8]["distance_px"] == pytest.approx(5.0, abs=1e-6)
    assert result.bins == []
    assert result.summary[0]["total_time_s"] == pytest.approx(1.0, abs=1e-9)
    assert result.summary[0]["detected_frames"] == 8
    assert result.metadata["units"]["u" * 32]["body_length_mm"] == 60.0
    assert result.metadata["units"]["u" * 32]["mm_per_px"] == pytest.approx(1.0)
    assert {row["state"] for row in rows} <= {
        "struggling", "immobile", "swing", "unknown", "outside"}


def test_lis_mode_is_rejected_before_capture():
    from behaviourlab.schema import AnalysisContext, ValidationError

    job = {"assay": "tst", "units": [{
        "id": "u" * 32, "name": "小鼠 1", "animal_id": None, "window": None,
        "geometry": {"arena": ARENA, "body_line": BODY_LINE},
        "options": options(mode="lis"),
        "detection": dict(tst.DETECTION_DEFAULTS),
    }]}
    with pytest.raises(ValidationError) as caught:
        tst.analyze(AnalysisContext(job=job, source=FakeSource([])))
    assert "尚未接入" in str(caught.value)


def test_medium_activity_needs_point_45s_and_exit_needs_point_4s():
    rows = []
    for index in range(12):
        row = evidence_row(index, feature=True, swing=False)
        row.update({"decision_flow_p90_mm_s": 0.0, "speed_std_mm_s": 0.0,
                    "flow_ratio": 0.0, "flow_dispersion": 0.05,
                    "swing_autocorrelation": 0.0, "swing_rigid": True,
                    "swing_motion_candidate": False, "swing_candidate": False,
                    "swing_score": 0.0})
        if 2 <= index <= 6:
            # 中等活动：F ≥ max(35, .45×50)=35 且 Q ≥ 2.5。
            row.update({"decision_flow_p90_mm_s": 40.0, "flow_ratio": 3.0})
        rows.append(row)
    tst.classic_states(rows, flow_hi=50.0, flow_lo=35.0,
                       speed_hi=40.0, speed_lo=20.0)

    assert [row["state"] for row in rows[:6]] == ["immobile"] * 6
    assert rows[6]["state"] == "struggling"  # 第 5 个中等活动行才满 .45 秒
    assert [row["state"] for row in rows[7:10]] == ["struggling"] * 3
    assert rows[10]["state"] == "immobile"   # 连续 .4 秒低于低阈值才退出
    assert rows[11]["state"] == "immobile"

    events = tst.build_events(rows, UNIT["id"], UNIT["name"])
    assert [event["kind"] for event in events] == [
        "struggling_transition", "immobile_transition"]
    assert events[0]["time_s"] == pytest.approx(0.6, abs=TOL)
    assert events[1]["time_s"] == pytest.approx(1.0, abs=TOL)
