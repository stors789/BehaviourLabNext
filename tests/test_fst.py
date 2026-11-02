"""FST 强迫游泳：分数/尺度、45 条固定状态向量与预览/页面接线。"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from behaviourlab import experiments, timeline
from behaviourlab.experiments import fst

TOL = 1e-9
ARENA = {"type": "polygon",
         "points": [[20, 20], [220, 20], [220, 220], [20, 220]]}
DIAMETER = {"type": "line", "points": [[70, 40], [170, 40]]}
WATERLINE = {"type": "line", "points": [[30, 60], [210, 60]]}
OPTIONS = {
    "cylinder_diameter_cm": 10.0,
    "auto_threshold": False,
    "threshold_sensitivity": 0.0,
    "immobility_threshold": 0.08,
    "active_threshold": 0.108,
    "smoothing_window_s": 0.0,
    "min_immobility_duration_s": 1.0,
    "min_active_duration_s": 0.3,
    "gap_merge_s": 0.5,
    "bin_seconds": 60.0,
}


def test_score_scale_and_threshold_fixture():
    geometry = {"diameter_line": {"type": "line",
                                  "points": [[10, 10], [110, 10]]}}
    scale, length = fst.scale_of(geometry, 10.0)
    assert length == pytest.approx(100.0, abs=TOL)
    assert scale == pytest.approx(0.1, abs=TOL)
    # .1 秒移动 5 px → 5 cm/s
    assert fst.centroid_speed(5.0, scale, 0.1) == pytest.approx(5.0, abs=TOL)
    # .60×2/10 + .25×4/10 + .15×min(1, 1.5/3) = .295
    assert fst.motion_score(2.0, 4.0, 1.5, 10.0) == pytest.approx(
        0.295, abs=TOL)

    values = [0.01] * 10 + [0.2] * 10
    info = fst.resolve_thresholds(values, 10.0, {
        "auto_threshold": True, "threshold_sensitivity": 0.0,
        "immobility_threshold": 0.08, "active_threshold": 0.108})
    # base = .01 + .25×(.2−.01) = .0575；高 = max(1.18×低, 低+.002)
    assert info["samples"] == 20
    assert info["estimated"] is True
    assert info["low"] == pytest.approx(0.0575, abs=TOL)
    assert info["high"] == pytest.approx(0.06785, abs=TOL)

    sensitive = fst.resolve_thresholds(values, 10.0, {
        "auto_threshold": True, "threshold_sensitivity": 1.0,
        "immobility_threshold": 0.08, "active_threshold": 0.108})
    assert sensitive["low"] == pytest.approx(0.0644, abs=TOL)
    assert sensitive["high"] == pytest.approx(0.075992, abs=TOL)

    # 样本不足：用配置低阈值与同一高阈值公式，标未估计。
    short = fst.resolve_thresholds([0.1] * 5, 10.0, {
        "auto_threshold": True, "threshold_sensitivity": 0.0,
        "immobility_threshold": 0.08, "active_threshold": 0.108})
    assert short["estimated"] is False
    assert short["samples"] == 5
    assert short["low"] == pytest.approx(0.08, abs=TOL)
    assert short["high"] == pytest.approx(0.0944, abs=TOL)

    # 平滑：以 t 为中心、按曝光交集加权；w=0 用原分。
    group = [
        {"time_s": 0.0, "interval_start_s": 0.0, "interval_end_s": 0.1,
         "motion_score": 0.1},
        {"time_s": 0.1, "interval_start_s": 0.1, "interval_end_s": 0.2,
         "motion_score": 0.9},
        {"time_s": 0.2, "interval_start_s": 0.2, "interval_end_s": 0.3,
         "motion_score": 0.1},
    ]
    fst.smooth_scores(group, 0.2)
    assert group[1]["smoothed_motion_score"] == pytest.approx(0.5, abs=TOL)
    fst.smooth_scores(group, 0.0)
    assert group[1]["smoothed_motion_score"] == pytest.approx(0.9, abs=TOL)

    # 关闭自动阈值时才要求高 > 低；自动模式只要求两阈值为正。
    geometry = {"arena": ARENA, "diameter_line": DIAMETER}
    bad = dict(OPTIONS, active_threshold=0.05)
    issues = fst.validate_unit(bad, geometry, complete=True)
    assert [item["field"] for item in issues] == ["options.active_threshold"]
    assert fst.validate_unit(dict(OPTIONS, auto_threshold=True), geometry,
                             complete=True) == []


def fixed_rows() -> list[dict]:
    scores = [0.2] * 3 + [0.01] * 15 + [0.2] * 4 + [0.01] * 12
    scores += [None] * 3 + [0.01] * 8
    rows = []
    for index, score in enumerate(scores):
        start = 0.1 * index
        rows.append({
            "unit_id": "u" * 32, "unit_name": "小鼠 1", "frame": index,
            "time_s": start, "interval_start_s": start,
            "interval_end_s": start + 0.1, "dt_s": 0.1,
            "in_window": True, "detected": True,
            "feature_ready": score is not None,
            "motion_score": score,
        })
    return rows


def test_fixed_45_row_states_and_summary():
    rows = fixed_rows()
    info = fst.derive_features(rows, OPTIONS)
    states = [row["state"] for row in rows]
    assert states[:3] == ["active"] * 3
    assert states[3:34] == ["immobile"] * 31
    assert states[34:37] == ["unknown"] * 3
    assert states[37:] == ["active"] * 8

    window = timeline.Window(0.0, 4.5)
    events = fst.build_events(rows, "u" * 32, "小鼠 1")
    summary = fst.summarize(
        rows, {"id": "u" * 32, "name": "小鼠 1", "options": OPTIONS},
        window, info, events)
    assert summary["total_time_s"] == pytest.approx(4.5, abs=1e-9)
    assert summary["active_time_s"] == pytest.approx(1.1, abs=1e-9)
    assert summary["immobility_time_s"] == pytest.approx(3.1, abs=1e-9)
    assert summary["uncertain_time_s"] == pytest.approx(0.3, abs=1e-9)
    assert summary["immobility_percent"] == pytest.approx(
        3.1 / 4.5 * 100.0, abs=1e-9)
    assert summary["immobility_bouts"] == 1
    assert summary["longest_immobility_bout_s"] == pytest.approx(
        3.1, abs=1e-9)
    assert summary["latency_to_immobility_s"] == pytest.approx(
        0.3, abs=1e-9)
    assert summary["mean_motion_score"] == pytest.approx(
        1.75 / 42.0, abs=1e-9)
    assert summary["max_motion_score"] == pytest.approx(0.2, abs=1e-9)
    assert len(events) == 1
    assert events[0]["kind"] == "immobile_bout"
    assert events[0]["time_s"] == pytest.approx(0.3, abs=1e-9)
    assert events[0]["end_s"] == pytest.approx(3.4, abs=1e-9)
    assert events[0]["confirmed_s"] is None

    bins = fst.build_bins(
        rows, window, 60.0, {"id": "u" * 32, "name": "小鼠 1"})
    assert len(bins) == 1
    assert bins[0]["bin_index"] == 1
    assert bins[0]["duration_s"] == pytest.approx(4.5, abs=1e-9)
    assert bins[0]["active_time_s"] == pytest.approx(1.1, abs=1e-9)
    assert bins[0]["immobility_time_s"] == pytest.approx(3.1, abs=1e-9)
    assert bins[0]["unknown_time_s"] == pytest.approx(0.3, abs=1e-9)
    assert bins[0]["mean_motion_score"] == pytest.approx(
        1.75 / 42.0, abs=1e-9)

    realtime = {row["realtime_state"] for row in rows}
    assert realtime <= {"active", "immobile", "unknown"}
    assert {row["realtime_state"] for row in rows[34:37]} == {"unknown"}
    # 候选改变重新计时：索引 3 起候选不动，过去曝光满 1 秒才在 13 确认；
    # 最后 8 行只有 .8 秒，不能用未来曝光提前确认。
    assert rows[12]["realtime_state"] == "unknown"
    assert rows[13]["realtime_state"] == "immobile"
    assert {row["realtime_state"] for row in rows[37:]} == {"unknown"}


class PreviewSource:
    """内存解码头：前 2 秒静止、后 1 秒平移。"""

    def __init__(self, path, *, start_s=0.0, end_s=None, **kwargs):
        from behaviourlab.schema import Frame

        self._frames = []
        begin = float(start_s or 0.0)
        stop = float(end_s if end_s is not None else begin + 3.0)
        step = 0.1
        count = max(1, int(round((stop - begin) / step)))
        for index in range(count):
            moment = begin + index * step
            offset = 0 if moment < 2.0 else int(round((moment - 1.9) * 10))
            image = np.full((240, 240, 3), 255, dtype=np.uint8)
            cv2.ellipse(image, (90 + offset, 120), (10, 22), 0, 0, 360,
                        (0, 0, 0), -1)
            self._frames.append(Frame(
                image=image, index=index, time_s=moment, dt_s=step,
                elapsed_s=moment - begin, analysis_fps=1.0 / step,
                interval_start_s=moment, interval_end_s=moment + step))

    def iter_frames(self):
        return iter(self._frames)


def test_arena_requires_polygon_type(client, store, tmp_path, monkeypatch):
    geometry = {"arena": {"type": "line",
                          "points": [[20, 20], [220, 20],
                                     [220, 220], [20, 220]]},
                "diameter_line": DIAMETER}
    issues = fst.validate_unit(dict(fst.DEFAULTS), geometry, complete=True)
    assert [item["field"] for item in issues] == ["geometry.arena"]
    assert fst.validate_unit(dict(fst.DEFAULTS),
                             {"arena": ARENA, "diameter_line": DIAMETER},
                             complete=True) == []

    source = tmp_path / "line-arena.mp4"
    source.write_bytes(b"placeholder")
    asset = store.create_asset(
        name="line-arena.mp4", path=str(source), project_id=None,
        metadata={"width": 240, "height": 240, "duration_s": 3.0,
                  "fps": 10.0, "frame_count": 30})
    draft = store.create_draft("fst", videos=[{
        "asset_id": asset["id"],
        "window": {"start_s": 0.0, "end_s": 3.0},
        "orientation": 0, "fps_override": None, "sample_fps": 10,
        "units": [{
            "id": "f" * 32, "name": "FST 鼠", "window": None,
            "geometry": geometry,
            "options": dict(fst.DEFAULTS), "detection": {}, "manual": None,
        }],
    }])
    monkeypatch.setattr(
        "behaviourlab.engine.media.probe_video",
        lambda path: {"width": 240, "height": 240, "duration_s": 3.0,
                      "fps": 10.0, "frame_count": 30})
    response = client.post(f"/api/drafts/{draft['id']}/analyze")
    assert response.status_code == 400, response.get_json()
    fields = [item["field"] for item in response.get_json()["error"]["fields"]]
    assert "videos[0].units[0].geometry.arena" in fields
    assert store.list_jobs() == []


def state_rows(state: str, count: int = 50) -> list[dict]:
    rows = []
    for index in range(count):
        start = 0.1 * index
        rows.append({
            "in_window": True, "feature_ready": True,
            "interval_start_s": start, "interval_end_s": start + 0.1,
            "dt_s": 0.1, "state": state,
            "motion_score": 0.0, "smoothed_motion_score": 0.0,
            "threshold_low": 0.08, "threshold_high": 0.108,
        })
    return rows


def test_state_band_draws_active_and_keeps_gaps(tmp_path):
    from behaviourlab import figures

    active = tmp_path / "active.png"
    # 全 active 的 5 秒数据也必须出图（此前 active 不在色表时返回 False）。
    assert figures.activity_png(active, state_rows("active"), mode="fst")
    image = cv2.imread(str(active))
    assert image is not None
    active_bgr = np.array([177, 107, 140])  # #8C6BB1
    assert int(np.all(image == active_bgr, axis=2).sum()) > 5000

    unknown = tmp_path / "unknown.png"
    assert figures.activity_png(unknown, state_rows("unknown"), mode="fst")
    assert not np.array_equal(image, cv2.imread(str(unknown)))

    # 3.8 秒 active + 1.2 秒 immobile：两种色带都要出现。
    mixed = state_rows("active")
    for row in mixed[38:]:
        row["state"] = "immobile"
    target = tmp_path / "mixed.png"
    assert figures.activity_png(target, mixed, mode="fst")
    mixed_image = cv2.imread(str(target))
    immobile_bgr = np.array([124, 143, 111])  # #6F8F7C
    assert int(np.all(mixed_image == active_bgr, axis=2).sum()) > 1000
    assert int(np.all(mixed_image == immobile_bgr, axis=2).sum()) > 1000

    # 真正缺失不填平：缺口区间不出图，仍能生成其余部分。
    gapped = state_rows("active")
    for row in gapped[20:30]:
        row["in_window"] = False
    gap_target = tmp_path / "gap.png"
    assert figures.activity_png(gap_target, gapped, mode="fst")
    assert not np.array_equal(cv2.imread(str(gap_target)), image)


def test_fst_registration_preview_and_result_page(client, store, tmp_path,
                                                  monkeypatch):
    from behaviourlab import media, preview
    from behaviourlab.media import FrameInfo

    entry = [item for item in experiments.catalog()
             if item["key"] == "fst"][0]
    assert entry["label"] == "强迫游泳实验 / FST"
    assert entry["defaults"]["cylinder_diameter_cm"] == 10.0
    assert entry["defaults"]["auto_threshold"] is True
    labels = [item["label"] for item in entry["geometry"]]
    assert labels == ["圆筒区域", "圆筒内径线", "检测遮罩", "水面线"]
    assert {item["key"] for item in entry["parameters"]} == set(
        fst.DEFAULTS)
    assert entry["detection_defaults"]["erosion_iter"] == 1
    assert entry["detection_defaults"]["dilation_iter"] == 1
    assert entry["detection_defaults"]["min_area"] == 80.0

    source = tmp_path / "fst.mp4"
    source.write_bytes(b"placeholder")
    asset = store.create_asset(
        name="fst.mp4", path=str(source), project_id=None,
        metadata={"width": 240, "height": 240, "duration_s": 3.0,
                  "fps": 10.0, "frame_count": 30})
    draft = store.create_draft("fst", videos=[{
        "asset_id": asset["id"],
        "window": {"start_s": 0.0, "end_s": 3.0},
        "orientation": 0, "fps_override": None, "sample_fps": 10,
        "units": [{
            "id": "f" * 32, "name": "FST 鼠", "window": None,
            "geometry": {"arena": ARENA, "diameter_line": DIAMETER,
                         "waterline": WATERLINE},
            "options": dict(fst.DEFAULTS), "detection": {}, "manual": None,
        }],
    }])
    monkeypatch.setattr(media, "VideoSource", PreviewSource)
    monkeypatch.setattr(
        "behaviourlab.engine.media.probe_video",
        lambda path: {"width": 240, "height": 240, "duration_s": 3.0,
                      "fps": 10.0, "frame_count": 30})
    monkeypatch.setattr(
        preview.media, "read_frame_at_time",
        lambda *args, **kwargs: FrameInfo(
            image=np.full((240, 240, 3), 255, dtype=np.uint8), index=15,
            time_s=1.5, pts=None, time_base=None, clock_source="pts"))

    response = client.post(
        f"/api/drafts/{draft['id']}/preview",
        json={"asset_id": asset["id"], "unit_id": "f" * 32,
              "time_s": 1.5, "kind": "behavior"})
    assert response.status_code == 201, response.get_json()
    behavior = response.get_json()["data"]["summary"]["behavior"]
    assert behavior["mode"] == "fst"
    # 行为预览只取 [t−1, t+3) 与视频窗的交集：[.5, 3) 共 2.5 秒。
    assert behavior["start_s"] == pytest.approx(0.5, abs=1e-9)
    assert behavior["end_s"] == pytest.approx(3.0, abs=1e-9)
    assert behavior["total_time_s"] == pytest.approx(2.5, abs=1e-9)
    assert behavior["rows"] == 25
    assert behavior["immobility_time_s"] > 0
    assert "短段阈值估计" in behavior["threshold_note"]

    started = client.post(f"/api/drafts/{draft['id']}/analyze")
    assert started.status_code == 202, started.get_json()
    payload = started.get_json()["data"]
    job = payload["jobs"][0]
    assert job["status"] == "completed"
    manifest = job["manifest"]
    summary = manifest["units"][0]["summary"]
    assert summary["mode"] == "fst"
    assert summary["total_time_s"] == pytest.approx(3.0, abs=1e-9)
    assert summary["active_time_s"] > 0
    assert summary["immobility_time_s"] > 0
    unit_meta = manifest["metadata"]["units"]["f" * 32]
    assert unit_meta["cm_per_px"] == pytest.approx(0.1, abs=1e-9)
    assert unit_meta["waterline_role"].startswith("仅保存与显示")
    assert manifest["units"][0]["figures"]["activity"]

    page = client.get(f"/results/{payload['run_id']}").get_data(as_text=True)
    for label in ("不动时间", "活动时间", "不确定时间", "不动比例",
                  "不动片段数", "首次不动潜伏期", "检测时间比例"):
        assert label in page
    assert "视觉运动规则使用可调经验阈值，请结合视频与人工复核。" in page
