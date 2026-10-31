"""第 09 块收口：公式/零值/自动阈值与状态计时的固定反例。"""

from __future__ import annotations

import numpy as np
import pytest

from behaviourlab import activity, timeline
from behaviourlab.experiments import tst

TOL = 1e-9
UNIT = {"id": "unit-1", "name": "小鼠 1", "animal_id": None}
ARENA = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]


def row(index, *, dt=0.1, start=None, candidate=False, **fields):
    begin = 0.1 * index if start is None else float(start)
    item = {
        "unit_id": UNIT["id"], "unit_name": UNIT["name"],
        "frame": index, "time_s": begin, "elapsed_s": begin,
        "analysis_fps": 10.0,
        "interval_start_s": begin, "interval_end_s": begin + dt,
        "source_interval_start_s": begin, "source_interval_end_s": begin + dt,
        "dt_s": dt, "in_window": True, "feature_ready": True,
        "detected": True, "swing_candidate": candidate,
        "swing_motion_candidate": False, "swing_rigid": True,
        "swing_score": 0.0, "swing_autocorrelation": 0.0,
        "flow_dispersion": 0.05, "flow_ratio": 0.0,
        "decision_flow_p90_mm_s": 0.0, "speed_std_mm_s": 0.0,
    }
    item.update(fields)
    return item


def test_dispersion_is_one_minus_mean_unit_vector():
    vectors = np.array([[2.0, 0.0]] * 50 + [[-2.0, 0.0]] * 50)
    coherence, dispersion = activity.direction_stats(vectors)
    assert coherence == pytest.approx(0.0, abs=TOL)
    assert dispersion == pytest.approx(1.0, abs=TOL)

    aligned = np.array([[3.0, 0.0]] * 20)
    coherence, dispersion = activity.direction_stats(aligned)
    assert coherence == pytest.approx(1.0, abs=TOL)
    assert dispersion == pytest.approx(0.0, abs=TOL)


def test_zero_dispersion_is_valid_for_candidate_and_score():
    assert activity.motion_swing(
        frequency=2.5, autocorrelation=0.8, horizontal=0.02, vertical=0.3,
        coherence=1.0, dispersion=0.0, nominal=5.0, ratio=2.0) is True
    score = activity.swing_score(
        autocorrelation_value=0.8, frequency=2.5, coherence=1.0,
        dispersion=0.0, is_elongated=True, ratio=2.0)
    assert score == pytest.approx(6.3, abs=TOL)


def test_zero_dispersion_suppresses_medium_activity():
    rows = [row(index, decision_flow_p90_mm_s=40.0, flow_ratio=3.0,
                flow_dispersion=0.0, swing_score=4.0,
                swing_autocorrelation=0.6, swing_rigid=True)
            for index in range(12)]
    tst.classic_states(rows, flow_hi=50.0, flow_lo=35.0,
                       speed_hi=40.0, speed_lo=20.0)
    assert [item["state"] for item in rows] == ["immobile"] * 12


def test_acceleration_uses_signed_speed_differences():
    assert activity.acceleration_std([0.0, 10.0, 0.0, 10.0, 0.0], 10.0) == \
        pytest.approx(100.0, abs=TOL)
    assert activity.acceleration_std([0.0, 10.0], 10.0) == 0.0


def test_area_rate_uses_source_time_difference():
    candidate = {
        "contour": [[10, 10], [20, 10], [20, 20], [10, 20]],
        "bbox": [10.0, 10.0, 10.0, 10.0], "center": [15.0, 15.0],
        "area": 120.0,
    }
    body = activity.body_features(candidate, ARENA, 0.22, 100.0)
    assert body["body_area_rate_s"] == pytest.approx(0.2 / 0.22, abs=TOL)
    first = activity.body_features(candidate, ARENA, None, None)
    assert first["body_area_rate_s"] is None


def test_auto_threshold_samples_decision_flow():
    raw = [0.0, 40.0, 0.0] * 10
    decision = [0.0, 40.0] + [32.0] * 28
    samples = [{
        "analysis_fps": 10.0,
        "flow_p90_mm_s": raw[index],
        "decision_flow_p90_mm_s": decision[index],
        "speed_std_mm_s": 0.0,
        "swing_candidate": False,
    } for index in range(30)]
    info = tst._estimate(
        samples, {"flow_hi": 50.0, "flow_lo": 35.0,
                  "speed_hi": 40.0, "speed_lo": 20.0}, 0.0)
    assert info["estimated"] is True
    assert info["flow_thr_hi"] == pytest.approx(40.0, abs=TOL)
    assert info["flow_thr_lo"] == pytest.approx(26.0, abs=TOL)


def test_initial_protection_starts_at_previous_frame_of_first_pair():
    rows = [row(0, feature_ready=False, pair_previous_time_s=None)]
    for index in range(1, 6):
        rows.append(row(index, pair_previous_time_s=0.0,
                        decision_flow_p90_mm_s=60.0, flow_ratio=3.0))
    tst.classic_states(rows, flow_hi=50.0, flow_lo=35.0,
                       speed_hi=40.0, speed_lo=20.0)
    assert rows[1]["state"] == "immobile"   # .1 秒仍在 .2 秒保护内
    assert rows[2]["state"] == "struggling"  # .2 秒即可进入


def test_swing_confirmation_uses_exposure_seconds():
    rows = [row(0, dt=0.5, candidate=True),
            row(1, dt=0.5, start=0.5, candidate=True)]
    for index in range(2, 8):
        rows.append(row(index, dt=0.1, start=1.0 + 0.1 * (index - 2)))
    tst.classic_states(rows, flow_hi=50.0, flow_lo=35.0,
                       speed_hi=40.0, speed_lo=20.0)
    # 候选 1.0 秒 + 非候选 .5 秒 = 2/3，按秒数在 1.5 秒确认；
    # 确认后回写过去 2.5 秒，所以更早的行也会标成 swing。
    assert {item.get("swing_confirmed_at") for item in rows[:-1]} == {1.5}
    assert rows[-1]["state"] == "swing"  # 确认行本身不回写未来曝光


def test_swing_confirmation_at_four_seconds():
    rows = [row(index, candidate=index >= 30) for index in range(41)]
    tst.classic_states(rows, flow_hi=50.0, flow_lo=35.0,
                       speed_hi=40.0, speed_lo=20.0)
    # 3.0 秒起才有候选，1.5 秒窗在 4.0 秒才满 2/3；不能提前到 3.9。
    assert rows[28].get("swing_confirmed_at") == pytest.approx(4.0, abs=TOL)
    assert rows[40]["state"] == "swing"
    assert rows[14]["state"] != "swing"  # 回写只覆盖过去 2.5 秒


def test_long_exposure_cannot_exit_swing_early():
    rows = [row(index, candidate=True) for index in range(26)]  # 0.0–2.5
    rows.append(row(26, dt=1.0, start=2.6, candidate=False,
                    decision_flow_p90_mm_s=0.0))
    rows.append(row(27, start=3.6, candidate=False,
                    decision_flow_p90_mm_s=60.0, flow_ratio=3.0))
    rows.append(row(28, start=3.7, candidate=False,
                    decision_flow_p90_mm_s=60.0, flow_ratio=3.0))
    rows.append(row(29, start=3.8, candidate=False,
                    decision_flow_p90_mm_s=60.0, flow_ratio=3.0))
    tst.classic_states(rows, flow_hi=50.0, flow_lo=35.0,
                       speed_hi=40.0, speed_lo=20.0)
    assert rows[26]["state"] == "swing"       # [2.6,3.6) 尚未过去
    assert rows[27]["state"] == "immobile"    # 3.6 退出
    assert rows[28]["state"] == "immobile"    # .2 秒保护到 3.8
    assert rows[29]["state"] == "struggling"  # 3.8 可重新挣扎


def test_bouts_count_exposure_gaps():
    rows = [row(0, dt=0.1), row(1, dt=0.1, start=0.3)]
    for item in rows:
        item["state"] = "immobile"
        item["state_group"] = "immobile"
    assert tst._bouts(rows, "immobile") == 2
    events = tst.build_events(rows, UNIT["id"], UNIT["name"])
    assert events == []


def test_timeline_window_still_counts_unknown_gap():
    # 主门仍成立：unknown 之后的新组从 immobile 重新开始。
    rows = [row(index) for index in range(3)]
    rows[1]["feature_ready"] = False
    tst.classic_states(rows, flow_hi=50.0, flow_lo=35.0,
                       speed_hi=40.0, speed_lo=20.0)
    assert rows[1]["state"] == "unknown"
    assert rows[2]["state"] == "immobile"
