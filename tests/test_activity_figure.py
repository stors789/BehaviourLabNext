"""第 09 块活动图：真实数值曲线、缺口断开与缺省值排除。"""

from __future__ import annotations

import math

import cv2
import numpy as np

from behaviourlab import figures

F = "decision_flow_p90_mm_s"
S = "speed_std_mm_s"


def rows_with_features(count: int = 20, *, decision: float = 40.0,
                       speed: float = 10.0) -> list[dict]:
    rows = []
    for index in range(count):
        state = "struggling" if index % 4 == 0 else (
            "immobile" if index % 4 == 1 else "swing")
        start = 0.1 * index
        rows.append({
            "unit_id": "e" * 32, "frame": index, "time_s": start,
            "interval_start_s": start, "interval_end_s": start + 0.1,
            "dt_s": 0.1, "in_window": True, "state": state,
            "feature_ready": True,
            F: decision if index % 3 else 0.0,
            S: speed if index % 3 else 0.0,
            "flow_thr_hi_used": 50.0, "flow_thr_lo_used": 35.0,
        })
    return rows


def test_activity_series_uses_exposure_and_breaks_at_gaps():
    rows = [
        {"in_window": True, "feature_ready": True,
         "interval_start_s": 0.0, "interval_end_s": 0.1, F: 10.0},
        {"in_window": True, "feature_ready": True,
         "interval_start_s": 0.1, "interval_end_s": 0.2, F: 30.0},
        # 未配对行的 0 是派生缺省，不是真实证据，不进入曲线。
        {"in_window": True, "feature_ready": False,
         "interval_start_s": 0.2, "interval_end_s": 0.3, F: 0.0},
        {"in_window": True, "feature_ready": True,
         "interval_start_s": 0.4, "interval_end_s": 0.5, F: 20.0},
    ]
    xs, ys = figures.activity_series(rows, F)
    # 每行值覆盖自己的曝光区间：[0,0.1)@10 → [0.1,0.2)@30，0.1 处垂直跳变。
    assert len(xs) == len(ys) == 7
    assert xs[0] == 0.0 and ys[0] == 10.0
    assert xs[1] == 0.1 and ys[1] == 10.0
    assert xs[2] == 0.1 and ys[2] == 30.0
    assert xs[3] == 0.2 and ys[3] == 30.0
    assert math.isnan(xs[4]) and math.isnan(ys[4])
    assert xs[5] == 0.4 and ys[5] == 20.0
    assert xs[6] == 0.5 and ys[6] == 20.0


def test_activity_series_value_at_time_matches_own_interval():
    rows = [
        {"in_window": True, "feature_ready": True,
         "interval_start_s": 0.0, "interval_end_s": 0.1, F: 10.0},
        {"in_window": True, "feature_ready": True,
         "interval_start_s": 0.1, "interval_end_s": 0.2, F: 30.0},
    ]
    xs, ys = figures.activity_series(rows, F)
    # 0.15 秒落在 [0.1,0.2) 内，必须读到本行的 30 而不是上一行的 10。
    step = [(x, y) for x, y in zip(xs, ys) if x <= 0.15][-1]
    assert step == (0.1, 30.0)


def test_activity_series_skips_rows_without_evidence():
    rows = [
        {"in_window": True, "feature_ready": True,
         "interval_start_s": 0.0, "interval_end_s": 0.1, F: None},
        {"in_window": False, "feature_ready": True,
         "interval_start_s": 0.1, "interval_end_s": 0.2, F: 99.0},
    ]
    assert figures.activity_series(rows, F) == ([], [])


def test_threshold_values_deduplicates():
    rows = [{"flow_thr_hi_used": 50.0}, {"flow_thr_hi_used": 50.0},
            {"flow_thr_hi_used": 35.0}, {"flow_thr_hi_used": None}]
    assert figures.threshold_values(rows, "flow_thr_hi_used") == [50.0, 35.0]


def test_activity_png_changes_when_features_change(tmp_path):
    base = tmp_path / "base.png"
    assert figures.activity_png(base, rows_with_features())
    scaled = []
    for row in rows_with_features():
        scaled.append({**row, F: row[F] * 5.0, S: row[S] * 5.0,
                       "flow_thr_hi_used": row["flow_thr_hi_used"] * 5.0,
                       "flow_thr_lo_used": row["flow_thr_lo_used"] * 5.0})
    other = tmp_path / "scaled.png"
    assert figures.activity_png(other, scaled)

    first = cv2.imread(str(base))
    second = cv2.imread(str(other))
    assert first is not None and second is not None
    # 特征放大 5 倍后曲线必须可见地改变，不能只有状态色带。
    assert not np.array_equal(first, second)
    changed = cv2.absdiff(first, second)
    assert int(np.count_nonzero(changed)) > 500


def test_activity_png_keeps_state_band_and_gap_whitespace(tmp_path):
    rows = rows_with_features()
    # 中间 1 秒没有行：状态带与曲线都留白，不跨缺口连接。
    for row in rows:
        if 1.0 <= row["interval_start_s"] < 2.0:
            row["in_window"] = False
    gapped = tmp_path / "gapped.png"
    assert figures.activity_png(gapped, rows)

    dense = tmp_path / "dense.png"
    assert figures.activity_png(dense, rows_with_features())
    first = cv2.imread(str(gapped))
    second = cv2.imread(str(dense))
    assert first is not None and second is not None
    assert not np.array_equal(first, second)


def test_activity_png_falls_back_to_state_band_without_evidence(tmp_path):
    rows = []
    for index in range(10):
        start = 0.1 * index
        rows.append({
            "interval_start_s": start, "interval_end_s": start + 0.1,
            "dt_s": 0.1, "in_window": True, "state": "immobile",
            "feature_ready": False, F: 0.0, S: 0.0,
        })
    target = tmp_path / "band.png"
    assert figures.activity_png(target, rows)
    image = cv2.imread(str(target))
    assert image is not None
