"""第 10 块只读验收补齐：五组固定反例与既有接线差距。

夹具来自验收书：ROI 顺序、Hu 零分量、综合分权重、外轮廓面积、
周边/核心比门、坏分割清史、后处理参数 0 值、补救副状态、lis 曲线、
取消合同、OFT 行为预览与相对 elapsed。
"""

from __future__ import annotations

from fractions import Fraction

import cv2
import numpy as np
import pytest

from behaviourlab import activity, detection, figures, media, preview
from behaviourlab.experiments import tst
from behaviourlab.schema import AnalysisContext, Frame, ValidationError

TOL = 1e-9
ARENA = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
BODY_LINE = {"type": "line", "points": [[10, 10], [70, 10]]}
# 场地包围盒正好是 [x=245, y=75, w=196, h=166]。
CLIMB_ARENA = {"type": "polygon",
               "points": [[245, 75], [440, 75], [440, 240], [245, 240]]}


def options(**overrides) -> dict:
    values = dict(tst.DEFAULTS)
    values.update(overrides)
    return values


def detection_settings() -> dict:
    settings = dict(detection.common_detection_defaults())
    settings.update(tst.DETECTION_DEFAULTS)
    return settings


class FixedSource:
    def __init__(self, frames):
        self._frames = list(frames)

    def iter_frames(self):
        return iter(self._frames)


def tst_job(arena=CLIMB_ARENA) -> dict:
    return {
        "assay": "tst", "run_id": "run", "job_id": "job", "project_id": None,
        "window": {"start_s": 0.0, "end_s": None},
        "orientation": 0, "fps_override": None, "sample_fps": None,
        "asset": {"id": "a" * 32, "path": "memory", "metadata": {}},
        "units": [{
            "id": "u" * 32, "name": "小鼠 1", "animal_id": None,
            "window": None,
            "geometry": {"arena": arena, "body_line": BODY_LINE},
            "options": options(mode="lis"),
            "detection": detection_settings(),
        }],
        "style": {},
    }


def climbing_frame(index: int) -> Frame:
    """身体框顶约 140、质心约 164 的竖直鼠体，位于 ROI 上半部。"""

    image = np.full((480, 480, 3), 255, dtype=np.uint8)
    cv2.ellipse(image, (300, 164), (8, 24), 0, 0, 360, (0, 0, 0), -1)
    start = 0.1 * index
    return Frame(image=image, index=index, time_s=start, dt_s=0.1,
                 elapsed_s=start, analysis_fps=10.0,
                 interval_start_s=start, interval_end_s=start + 0.1)


def lis_row(index: int, *, start: float | None = None, dt: float = 0.1,
            **fields) -> dict:
    begin = 0.1 * index if start is None else float(start)
    row = {
        "unit_id": "u" * 32, "unit_name": "小鼠 1", "frame": index,
        "time_s": begin, "interval_start_s": begin,
        "interval_end_s": begin + dt, "dt_s": dt, "in_window": True,
        "feature_ready": True, "analysis_fps": 10.0,
        "body_x_px": 50.0, "body_y_px": 50.0, "lis_body_length_px": 40.0,
        "distance_px": 1.0, "flow_p90_mm_s": 1.0,
        "speed_nominal_mm_s": 1.0,
        "lis_r": 0.01, "lis_peripheral_d": 0.01, "lis_z": 0.01,
        "lis_h": 0.01, "lis_f": 1.0, "lis_s": 1.0, "lis_g": 0.025,
        "lis_b": 0.0, "lis_core_ratio": 1.0,
        "lis_segmentation_bad": False, "lis_aspect_ratio": 1.2,
        "lis_motion_score": 0.0, "lis_motion_prior": 0.01,
        "lis_climbing_suspected": False, "state": None,
    }
    row.update(fields)
    return row


def rescue_row(index: int, hit: bool, **fields) -> dict:
    row = lis_row(
        index,
        lis_motion_prior=0.30 if hit else 0.50,
        lis_aspect_ratio=2.0 if hit else 1.0,
        lis_motion_score=0.6 if hit else 0.1,
        lis_f=30.0 if hit else 1.0, lis_s=2.0, lis_r=0.03,
        lis_peripheral_d=0.02, lis_h=0.03, lis_b=0.5, lis_g=0.0,
        state="immobile", state_group="immobile")
    row.update(fields)
    return row


def segment_row(index: int, state: str, mode: str = "score") -> dict:
    start = 0.1 * index
    return {"frame": index, "time_s": start, "interval_start_s": start,
            "interval_end_s": start + 0.1, "state": state,
            "state_group": state, "lis_score_mode": mode}


EVIDENCE_THRESHOLDS = {
    key: {"high": 0.1, "low": 0.01, "mid": 0.05, "q50": 0.05,
          "q75": 0.2, "q80": 0.2}
    for key in ("r", "p", "z", "h", "f", "s", "g", "b")}


# -- 一、身体与残差证据、评分公式 -----------------------------------------

def test_climbing_gate_uses_bounding_rect_order():
    # 正确 ROI (x, y, w, h)：框顶 140 不满足 75+16.6=91.6。
    assert tst._lis_climbing(140.0, 164.0, 2.0,
                             (245.0, 75.0, 196.0, 166.0)) is False
    # 交换成 (y, x, h, w) 会把同一身体误判成攀爬。
    assert tst._lis_climbing(140.0, 164.0, 2.0,
                             (75.0, 245.0, 166.0, 196.0)) is True


def test_capture_keeps_roi_in_xywh_order():
    job = tst_job()
    context = AnalysisContext(
        job=job, source=FixedSource([climbing_frame(0), climbing_frame(1)]))
    captured = tst.capture(context, job["units"])
    row = captured["u" * 32][1]
    assert row["feature_ready"] is True
    assert row["lis_segmentation_bad"] is False
    assert row["lis_aspect_ratio"] >= 1.8
    assert row["lis_climbing_suspected"] is False
    # 同一行在错误的 ROI 解释下会命中攀爬门，说明夹具能分辨两种顺序。
    bbox_top = row["body_y_px"] - row["body_bbox_h"] / 2.0
    assert tst._lis_climbing(bbox_top, row["body_y_px"],
                             row["lis_aspect_ratio"],
                             (75.0, 245.0, 166.0, 196.0)) is True


def test_hu_component_zero_is_zero_and_distance_is_four():
    assert activity.lis_hu_log(0.0) == 0.0
    assert activity.lis_hu_log(1e-4) == pytest.approx(4.0, abs=TOL)
    # 分量 0 与 1e−4 的向量距离是 4；修复前 0 变成 30，距离 26。
    assert activity.lis_hu_distance([0.0], [4.0]) == pytest.approx(4.0,
                                                                  abs=TOL)
    # 圆形掩膜的 Hu 分量精确为 0，不能落成 ±30。
    circle = np.zeros((64, 64), dtype=np.uint8)
    cv2.circle(circle, (32, 32), 12, 255, -1)
    vector = activity.lis_hu_vector(circle)
    assert vector is not None
    assert vector[1:] == [0.0] * 6


def test_deformation_seven_values_median_uses_zero_hu():
    cur = {"area": 110.0, "perimeter": 120.0, "aspect": 1.3,
           "solidity": 0.9, "fill": 0.8}
    prev = {"area": 100.0, "perimeter": 100.0, "aspect": 1.0,
            "solidity": 0.5, "fill": 0.3}
    # 七量：.1 .2 .3 .4 .5 / 0 / .6 → 中位数 .3（旧 Hu 会变成 1.0→.4）。
    assert activity.lis_deformation(cur, prev, 0.0, 0.6) == pytest.approx(
        0.3, abs=TOL)


def test_motion_score_weights_hu_point_22_and_z_point_08():
    rows = []
    for index in range(10):
        value = 0.0 if index < 4 else (0.5 if index == 4 else 1.0)
        rows.append(lis_row(
            index, lis_r=value, lis_peripheral_d=value,
            lis_z=(0.0 if index < 5 else 1.0),
            lis_h=(0.0 if index < 4 else 1.0),
            lis_f=value, lis_s=value))
    tst.lis_score_rows(rows)
    target = rows[4]
    assert target["lis_score_r"] == pytest.approx(0.5, abs=TOL)
    assert target["lis_score_p"] == pytest.approx(0.5, abs=TOL)
    assert target["lis_score_f"] == pytest.approx(0.5, abs=TOL)
    assert target["lis_score_s"] == pytest.approx(0.5, abs=TOL)
    assert target["lis_score_h"] == pytest.approx(1.0, abs=TOL)
    assert target["lis_score_z"] == pytest.approx(0.0, abs=TOL)
    # .24×.5 + .14×.5 + .08×0 + .22×1 + .18×.5 + .14×.5 = .57
    assert target["lis_motion_score"] == pytest.approx(0.57, abs=1e-9)


def test_body_area_uses_outer_contour_and_same_definition_for_rate():
    candidate = {
        "contour": [[0, 0], [10, 0], [10, 10], [0, 10]],
        "bbox": [0.0, 0.0, 11.0, 11.0], "center": [5.0, 5.0],
        "area": 84.0,  # 外轮廓 100 − 孔洞 16
    }
    body = activity.body_features(candidate, ARENA, None, None)
    assert body["body_area_px"] == pytest.approx(100.0, abs=TOL)
    assert body["body_solidity"] == pytest.approx(1.0, abs=TOL)
    assert body["body_fill_ratio"] == pytest.approx(100.0 / 121.0, abs=TOL)

    wider = {
        "contour": [[0, 0], [12, 0], [12, 10], [0, 10]],
        "bbox": [0.0, 0.0, 13.0, 11.0], "center": [6.0, 5.0],
        "area": 100.0,
    }
    grown = activity.body_features(wider, ARENA, 0.2, body["body_area_px"])
    assert grown["body_area_px"] == pytest.approx(120.0, abs=TOL)
    # 变化率同样用外轮廓面积：(120−100)/100/0.2 = 1.0
    assert grown["body_area_rate_s"] == pytest.approx(1.0, abs=TOL)


# -- 二、证据分支的阈值来源与历史 -----------------------------------------

def test_core_ratio_gate_uses_ratio_sample_q80():
    rows = []
    for index in range(10):
        rows.append(lis_row(index, lis_core_ratio=10.0))
    rows[5]["lis_r"] = 0.9
    info = tst.lis_score_rows(rows)
    assert info["core_ratio_gate"] == pytest.approx(10.0, abs=TOL)
    # 全体比值 10、只有 R 高：不能借 R 的 q80 把比值算成活动证据。
    active, immobile, passive = tst._lis_evidence(
        rows[5], info["thresholds"], info["core_ratio_gate"])
    assert active is False


def test_bad_segmentation_keeps_state_and_clears_history():
    rows = [
        lis_row(0, start=0.0, lis_r=0.5, lis_peripheral_d=0.5, lis_z=0.5,
                lis_h=0.5, lis_motion_prior=0.1),
        lis_row(1, start=0.1, lis_r=0.5, lis_peripheral_d=0.5, lis_z=0.5,
                lis_h=0.5, lis_motion_prior=0.1),
        lis_row(2, start=0.2, dt=0.08, lis_r=0.5, lis_peripheral_d=0.5,
                lis_z=0.5, lis_h=0.5, lis_motion_prior=0.1),
        lis_row(3, start=0.28, lis_segmentation_bad=True,
                lis_motion_prior=0.1),
        lis_row(4, start=0.38, lis_r=0.5, lis_peripheral_d=0.5, lis_z=0.5,
                lis_h=0.5, lis_motion_prior=0.1),
        lis_row(5, start=0.48, lis_r=0.5, lis_peripheral_d=0.5, lis_z=0.5,
                lis_h=0.5, lis_motion_prior=0.1),
        lis_row(6, start=0.58, lis_r=0.5, lis_peripheral_d=0.5, lis_z=0.5,
                lis_h=0.5, lis_motion_prior=0.1),
        lis_row(7, start=0.68, lis_r=0.5, lis_peripheral_d=0.5, lis_z=0.5,
                lis_h=0.5, lis_motion_prior=0.1),
    ]
    tst._run_lis_group(rows, EVIDENCE_THRESHOLDS, False, 1.15)

    assert rows[3]["state"] == "immobile"
    assert rows[3]["lis_confidence"] == "low"
    # 恢复首行不能借坏分割之前的 .25 秒活动证据（旧值覆盖 .35、比 .714286）。
    assert rows[4]["state"] == "immobile"
    assert rows[4]["lis_evidence_coverage_s"] == pytest.approx(0.0, abs=TOL)
    assert rows[4]["lis_evidence_ratio"] == pytest.approx(0.0, abs=TOL)
    # 恢复后重新积累满 .28 秒才进入挣扎。
    assert rows[6]["state"] == "immobile"
    assert rows[7]["state"] == "struggling"


# -- 三、后处理参数生效及诊断来源不被覆盖 ---------------------------------

def test_sandwich_uses_actual_gaps_and_zero_is_legal():
    group = [segment_row(0, "struggling"), segment_row(1, "immobile"),
             segment_row(2, "struggling")]
    tst._lis_sandwich(group, 0.3, 0.05, 0.5)
    # .1 秒不严格短于 .05：不填。
    assert [row["state"] for row in group] == [
        "struggling", "immobile", "struggling"]

    group = [segment_row(0, "struggling"), segment_row(1, "immobile"),
             segment_row(2, "struggling")]
    tst._lis_sandwich(group, 0.3, 0.2, 0.5)
    assert [row["state"] for row in group] == ["struggling"] * 3

    group = [segment_row(0, "struggling", "score"), segment_row(1, "immobile"),
             segment_row(2, "struggling", "evidence")]
    # 混合分支取较小值 .05：.1 秒不填，不能沿用评分分支的 2.0。
    tst._lis_sandwich(group, 1.0, 2.0, 0.05)
    assert group[1]["state"] == "immobile"

    group = [segment_row(0, "immobile"), segment_row(1, "struggling"),
             segment_row(2, "struggling"), segment_row(3, "immobile")]
    tst._lis_sandwich(group, 0.0, 2.0, 0.5)
    # min_struggle=0 是合法值：.2 秒挣扎段保留。
    assert [row["state"] for row in group] == [
        "immobile", "struggling", "struggling", "immobile"]
    tst._lis_sandwich(group, 0.3, 2.0, 0.5)
    assert [row["state"] for row in group] == ["immobile"] * 4


def test_rescue_zero_gap_does_not_fill_or_extend():
    group = [rescue_row(index, index < 4 or index > 5) for index in range(10)]
    tst._lis_rescue(group, 0.0, 0.0, 0.0, 0.3)
    assert [row["lis_rescue"] for row in group] == [
        True, True, True, True, False, False, True, True, True, True]
    assert [row["lis_rescue_raw"] for row in group] == [
        True, True, True, True, False, False, True, True, True, True]


def test_rescue_raw_keeps_original_hits_only():
    group = [rescue_row(0, True), rescue_row(1, False), rescue_row(2, False),
             rescue_row(3, False), rescue_row(4, True)]
    tst._lis_rescue(group, 1.0, 0.0, 0.0, 0.3)
    # 原始 [T,F,F,F,T] 保持原序列；夹心只写进 separate 的副状态。
    assert [row["lis_rescue_raw"] for row in group] == [
        True, False, False, False, True]
    assert [row["lis_rescue"] for row in group] == [True] * 5


def test_rescue_substate_sandwich_and_short_revert():
    group = [rescue_row(0, False), rescue_row(1, True), rescue_row(2, False)]
    tst._lis_rescue(group, 1.0, 0.0, 0.0, 0.3)
    # 两端非补救之间的 .1 秒补救段在默认 .3 秒门下删除。
    assert [row["lis_rescue"] for row in group] == [False, False, False]

    group = [rescue_row(0, False), rescue_row(1, True), rescue_row(2, False)]
    tst._lis_rescue(group, 1.0, 0.0, 0.0, 0.0)
    assert [row["lis_rescue"] for row in group] == [False, True, False]


def test_rescue_extension_stays_inside_group_contacts():
    group = [rescue_row(index, 3 <= index <= 6) for index in range(10)]
    tst._lis_rescue(group, 1.0, 0.2, 0.1, 0.3)
    assert [row["lis_rescue"] for row in group] == [
        False, True, True, True, True, True, True, True, False, False]
    assert [row["lis_rescue_raw"] for row in group] == [
        False, False, False, True, True, True, True, False, False, False]


def test_model_state_saved_before_rescue_and_rescue_is_independent():
    rows = []
    for index in range(10):
        value = 0.0 if index < 4 else (0.5 if index == 4 else 1.0)
        rows.append(lis_row(
            index, lis_r=value, lis_peripheral_d=value, lis_z=value,
            lis_h=(0.0 if index < 4 else 1.0), flow_p90_mm_s=30.0,
            lis_aspect_ratio=2.0))
    tst.derive_lis_features(rows, options())
    # 证据分支本身没有活动证据：模型状态全是 immobile。
    assert {row["lis_model_state"] for row in rows} == {"immobile"}
    # 补救只改最终状态，不改模型状态，也不改写原始条件命中。
    assert [row["lis_rescue_raw"] for row in rows] == [
        False, False, False, False, False, True, True, True, True, True]
    assert {row["state"] for row in rows} == {"struggling"}
    assert all(row["lis_rescue"] for row in rows)


def test_lis_diagnostics_kept_for_unpaired_and_outside_rows(tmp_path):
    from behaviourlab import outputs
    from behaviourlab.schema import dumps_strict

    rows = [lis_row(index) for index in range(12)]
    rows.append(lis_row(12, feature_ready=False))
    outside = lis_row(13, in_window=False)
    outside.update({"interval_start_s": 1.3, "interval_end_s": 1.3,
                    "dt_s": 0.0})
    rows.append(outside)
    tst.derive_lis_features(rows, options())

    assert rows[12]["state"] == "unknown"
    assert rows[12]["lis_model_state"] == "unknown"
    assert rows[12]["lis_reason"] == "无连续合格配对"
    assert rows[13]["state"] == "outside"
    assert rows[13]["lis_model_state"] == "outside"
    assert rows[13]["lis_reason"] == "统计窗外"
    assert rows[12]["lis_rescue_raw"] is False
    assert rows[13]["lis_rescue"] is False
    for row in rows[:12]:
        assert row["lis_model_state"] in ("immobile", "struggling")
    # 诊断列必须是可解析的状态值，中文说明只留在 lis_reason。
    assert {row["lis_model_state"] for row in rows} <= {
        "struggling", "immobile", "unknown", "outside"}

    # JSON 与 CSV 都保留诊断，不只显示最终状态。
    dumps_strict(rows)
    target = tmp_path / "observations.csv"
    outputs.write_observations(target, rows)
    text = target.read_text(encoding="utf-8-sig")
    header = text.splitlines()[0]
    for column in ("lis_model_state", "lis_rescue_raw", "lis_rescue",
                   "lis_reason"):
        assert column in header
    assert "无连续合格配对" in text
    assert "统计窗外" in text


# -- 四、活动曲线与取消接线 -----------------------------------------------

def lis_figure_rows(scale: float = 1.0) -> list[dict]:
    rows = []
    for index in range(20):
        start = 0.1 * index
        rows.append({
            "unit_id": "u" * 32, "frame": index, "time_s": start,
            "interval_start_s": start, "interval_end_s": start + 0.1,
            "dt_s": 0.1, "in_window": True, "feature_ready": True,
            "state": "struggling" if index % 3 == 0 else "immobile",
            "lis_f": (10.0 + 2.0 * index) * scale,
            "lis_s": (3.0 + index) * scale,
        })
    return rows


def test_activity_png_draws_lis_f_s_curve(tmp_path):
    base = tmp_path / "lis.png"
    assert figures.activity_png(base, lis_figure_rows(), mode="lis")
    other = tmp_path / "lis-scaled.png"
    assert figures.activity_png(other, lis_figure_rows(5.0), mode="lis")
    first = cv2.imread(str(base))
    second = cv2.imread(str(other))
    assert first is not None and second is not None
    assert not np.array_equal(first, second)
    assert int(np.count_nonzero(cv2.absdiff(first, second))) > 500


def test_exported_figure_uses_lis_curves(tmp_path):
    from behaviourlab import outputs

    summary = {"unit_id": "u" * 32, "unit_name": "小鼠 1", "mode": "lis",
               "immobility_time_s": 1.0}
    info = outputs._unit_figures(tmp_path, summary, lis_figure_rows(), "u",
                                 None)
    assert info["activity"]
    produced = cv2.imread(str(tmp_path / info["activity"]))
    assert produced is not None
    reference = tmp_path / "ref-lis.png"
    classic = tmp_path / "ref-classic.png"
    figures.activity_png(reference, lis_figure_rows(), mode="lis",
                         title="小鼠 1")
    figures.activity_png(classic, lis_figure_rows(), mode="classic",
                         title="小鼠 1")
    assert np.array_equal(produced, cv2.imread(str(reference)))
    # classic 模式下这些行没有经典 F/S，只能退化到状态色带。
    assert not np.array_equal(produced, cv2.imread(str(classic)))


def test_analyze_checks_cancel_after_capture():
    state = {"done": False, "after": 0}
    frames = [climbing_frame(index) for index in range(10)]

    class Source:
        def iter_frames(self):
            for frame in frames:
                yield frame
            state["done"] = True

    def cancel() -> bool:
        if state["done"]:
            state["after"] += 1
        return state["done"]

    job = tst_job()
    job["window"] = {"start_s": 0.0, "end_s": 1.0}
    context = AnalysisContext(job=job, source=Source(), cancel_check=cancel)
    with pytest.raises(ValidationError) as error:
        tst.analyze(context)
    assert error.value.message == "分析已取消"
    assert error.value.field == "job"
    # 捕获结束后仍有取消检查，并且真的停下。
    assert state["after"] >= 1


def test_lis_rolling_checks_cancel_inside_long_group():
    rows = [lis_row(index) for index in range(1025)]
    for row in rows:
        row["lis_f"] = None
    calls = {"count": 0}

    def cancel() -> bool:
        calls["count"] += 1
        return calls["count"] > 1

    with pytest.raises(ValidationError) as error:
        tst._lis_rolling(rows, cancel)
    assert error.value.message == "分析已取消"
    # 首行之后取消：在固定步长处停下，不把 1025 行整组算完。
    assert calls["count"] == 2
    assert rows[511]["lis_f"] is not None
    assert rows[512]["lis_f"] is None

    # 不取消时结果与旧路径一致。
    fresh = [lis_row(index) for index in range(1025)]
    quiet = [lis_row(index) for index in range(1025)]
    tst._lis_rolling(fresh)
    tst._lis_rolling(quiet, lambda: False)
    for key in ("lis_h", "lis_f", "lis_s", "lis_g", "lis_b"):
        assert [row[key] for row in fresh] == [row[key] for row in quiet]
    assert all(row["lis_f"] is not None for row in fresh)


def test_derive_lis_features_honours_cancel():
    rows = [lis_row(index) for index in range(20)]
    calls = {"count": 0}

    def cancel() -> bool:
        calls["count"] += 1
        return calls["count"] > 2

    with pytest.raises(ValidationError) as error:
        tst.derive_lis_features(rows, options(), cancel)
    assert error.value.message == "分析已取消"
    assert calls["count"] >= 3


def moving_frames(count: int = 10, span: float = 0.1) -> list[Frame]:
    frames = []
    for index in range(count):
        image = np.full((200, 200, 3), 255, dtype=np.uint8)
        cv2.ellipse(image, (40 + 5 * index, 60), (24, 12), 0, 0, 360,
                    (0, 0, 0), -1)
        start = index * span
        frames.append(Frame(
            image=image, index=index, time_s=start, dt_s=span,
            elapsed_s=start, analysis_fps=1.0 / span,
            interval_start_s=start, interval_end_s=start + span))
    return frames


def test_lis_preview_matches_formal_analysis_on_same_window(monkeypatch):
    arena = {"type": "polygon",
             "points": [[5, 5], [120, 5], [120, 120], [5, 120]]}
    frames = moving_frames()
    job = tst_job(arena)
    job["window"] = {"start_s": 0.0, "end_s": 1.0}
    result = tst.analyze(AnalysisContext(job=job, source=FixedSource(frames)))
    monkeypatch.setattr(media, "VideoSource",
                        lambda *args, **kwargs: FixedSource(frames))
    behavior = tst.preview_analysis(
        asset=job["asset"],
        video={"orientation": 0, "fps_override": None, "sample_fps": None},
        unit=job["units"][0], start_s=0.0, end_s=1.0)

    summary = result.summary[0]
    assert behavior["rows"] == len(result.observations)
    for key in ("total_time_s", "immobility_time_s", "struggling_time_s",
                "uncertain_time_s", "swing_time_s"):
        assert behavior[key] == pytest.approx(summary[key], abs=1e-9)
    assert behavior["thresholds"] == summary["thresholds"]
    assert behavior["threshold_samples"] == summary["threshold_samples"]


# -- 五、两个既有公共接线差距 ---------------------------------------------

def test_frame_elapsed_is_relative_to_window_start(tmp_path):
    # 源时钟以首帧为 0；窗口取 [30,35) 的前上下文与窗口内三帧。
    pts = [200 * index for index in range(176)]
    spec = [(value, 0.2) for value in pts]

    class TimedDecoder:
        def __init__(self) -> None:
            self.start = 0

        def frame_at(self, index):
            value, duration = spec[index]
            return media.RawFrame(
                index=-1, image=np.full((4, 6, 3), index, dtype=np.uint16),
                pts=value, time_base=Fraction(1, 1000), keyframe=True,
                duration_s=duration)

        def frames(self):
            for index in range(self.start, len(spec)):
                yield self.frame_at(index)

        def seek(self, pts_value, keyframe=True):
            self.start = 0

        def close(self):
            pass

    path = tmp_path / "clip.mp4"
    path.write_bytes(b"placeholder")
    source = media.VideoSource(
        path, start_s=30.0, end_s=35.0, source_fps=5.0,
        decoder_factory=lambda *args, **kwargs: TimedDecoder())
    frames = list(source.iter_frames())
    by_time = {round(frame.time_s, 6): frame for frame in frames}

    assert by_time[29.8].elapsed_s == pytest.approx(-0.2, abs=1e-9)
    assert by_time[30.0].elapsed_s == pytest.approx(0.0, abs=1e-9)
    assert by_time[30.2].elapsed_s == pytest.approx(0.2, abs=1e-9)
    # 源时刻与有效曝光不变。
    assert by_time[29.8].time_s == pytest.approx(29.8, abs=1e-9)
    assert by_time[29.8].interval_start_s == pytest.approx(30.0, abs=1e-9)
    assert by_time[30.2].interval_end_s - by_time[30.2].interval_start_s == \
        pytest.approx(0.2, abs=1e-9)


def test_oft_behavior_preview_runs_core_and_returns_summary(
        client, store, tmp_path, monkeypatch):
    from behaviourlab.media import FrameInfo

    source = tmp_path / "oft.mp4"
    source.write_bytes(b"placeholder")
    asset = store.create_asset(
        name="oft.mp4", path=str(source), project_id=None,
        metadata={"width": 640, "height": 480, "duration_s": 40.0,
                  "fps": 25.0, "frame_count": 1000})
    draft = store.create_draft("oft", videos=[{
        "asset_id": asset["id"],
        "window": {"start_s": 30.0, "end_s": 35.0},
        "orientation": 0, "fps_override": None, "sample_fps": 5,
        "units": [{
            "id": "a" * 32, "name": "小鼠 1", "window": None,
            "geometry": {"arena": {"type": "quad",
                                   "points": [[0, 0], [639, 0],
                                              [639, 479], [0, 479]]}},
            "options": {}, "detection": {"bg_method": "threshold"},
            "manual": None,
        }],
    }])

    calls = {"count": 0}

    class Source:
        def __init__(self, path, *, start_s=0.0, end_s=None, **kwargs):
            calls["count"] += 1
            self._frames = []
            begin = float(start_s)
            stop = float(end_s if end_s is not None else begin + 1.0)
            step = 0.2
            count = max(1, int(round((stop - begin) / step)))
            for index in range(count):
                moment = begin + index * step
                image = np.full((480, 640, 3), 255, dtype=np.uint8)
                cv2.rectangle(image, (300, 200), (360, 260), (0, 0, 0), -1)
                self._frames.append(Frame(
                    image=image, index=800 + index, time_s=moment, dt_s=step,
                    elapsed_s=moment - begin, analysis_fps=5.0,
                    interval_start_s=moment, interval_end_s=moment + step))

        def iter_frames(self):
            return iter(self._frames)

    monkeypatch.setattr(media, "VideoSource", Source)
    monkeypatch.setattr(
        preview.media, "read_frame_at_time",
        lambda *args, **kwargs: FrameInfo(
            image=np.full((480, 640, 3), 255, dtype=np.uint8), index=800,
            time_s=32.0, pts=None, time_base=None, clock_source="pts"))

    response = client.post(
        f"/api/drafts/{draft['id']}/preview",
        json={"asset_id": asset["id"], "unit_id": "a" * 32,
              "time_s": 32.0, "kind": "behavior"})
    assert response.status_code == 201, response.get_json()
    data = response.get_json()["data"]
    assert data["frame_index"] == 800
    scope = data["preview_scope"]
    assert scope["start_s"] == pytest.approx(31.0, abs=1e-9)
    assert scope["end_s"] == pytest.approx(35.0, abs=1e-9)
    behavior = data["summary"]["behavior"]
    assert behavior is not None
    assert behavior["rows"] == 20
    assert behavior["total_time_s"] == pytest.approx(4.0, abs=1e-9)
    assert behavior["observed_time_s"] == pytest.approx(4.0, abs=1e-9)
    assert behavior["center_percent"] is not None
    # 同一核心只调用一次；单帧检测路径保持。
    assert calls["count"] == 1
    assert data["summary"]["detection"] is not None
