"""Y 迷宫人工评分：固定事件夹具、纯人工边界、混合检测与完整窗预览。"""

from __future__ import annotations

import numpy as np
import pytest

from behaviourlab import detection, engine, media, preview
from behaviourlab.experiments import ymaze
from behaviourlab.media import FrameInfo
from behaviourlab.schema import AnalysisContext, Frame, ValidationError

UNIT = "u" * 32
AUTO = "a" * 32
RUN = "r" * 32
JOB = "j" * 32

ENTRIES = [{"time_s": 0.15, "arm": "A"}, {"time_s": 0.35, "arm": "A"},
           {"time_s": 0.55, "arm": "B"}, {"time_s": 0.75, "arm": "C"}]

AUTO_GEOMETRY = {"zones": {
    "center": {"type": "polygon",
               "points": [[58, 42], [70, 42], [64, 54]]},
    "A": {"type": "quad", "points": [[58, 4], [70, 4], [70, 38], [58, 38]]},
    "B": {"type": "quad", "points": [[74, 48], [120, 48], [120, 60], [74, 60]]},
    "C": {"type": "quad", "points": [[8, 48], [54, 48], [54, 60], [8, 60]]},
}}

AUTO_OPTIONS = dict(ymaze.DEFAULTS, arm_length_cm=30.0, arm_width_cm=10.0,
                    entry_depth_ratio=0.2, max_jump_cm=12.0,
                    tracking_hold_s=0.4, entry_confirm_s=0.12,
                    entry_reset_center_s=0.2, require_center=True,
                    movement_threshold_cm_s=1.0, bin_seconds=60.0)


class Frames:
    """内存源：10 张 128×96 灰度 240 图，t=.1i、曝光 .1 秒。"""

    def __init__(self, path=None, *, start_s=0.0, end_s=None,
                 decode_images=True, **kwargs):
        self._frames = []
        begin = float(start_s or 0.0)
        stop = float(end_s if end_s is not None else begin + 1.0)
        index = 0
        moment = begin
        while moment < stop - 1e-9:
            image = (np.full((96, 128), 240, dtype=np.uint8)
                     if decode_images else None)
            end = round(moment + 0.1, 6)
            self._frames.append(Frame(
                image=image, index=index, time_s=round(moment, 6), dt_s=0.1,
                elapsed_s=round(moment - begin, 6), analysis_fps=10.0,
                interval_start_s=round(moment, 6), interval_end_s=end))
            index += 1
            moment = end

    def iter_frames(self):
        return iter(self._frames)


def manual_unit(entries=None, *, window=None, options=None, detection_settings=None):
    return {
        "id": UNIT, "name": "人工鼠", "animal_id": None,
        "window": window if window is not None else {
            "mode": "exact", "start_s": 0.1, "end_s": 0.9},
        "geometry": {},
        "options": options if options is not None else {"bin_seconds": 0.4},
        "detection": detection_settings if detection_settings is not None else {
            "bg_method": "static", "static_background": "缺失背景.png",
            "bg_type": "xx", "gray_threshold": -2,
        },
        "manual": {"mode": "manual",
                   "entries": [dict(item) for item in
                               (ENTRIES if entries is None else entries)]},
    }


def auto_unit():
    return {
        "id": AUTO, "name": "自动鼠", "animal_id": None, "window": None,
        "geometry": AUTO_GEOMETRY, "options": dict(AUTO_OPTIONS),
        "detection": {"bg_method": "threshold", "bg_type": "light"},
        "manual": None,
    }


def job_for(units, *, window=None):
    return {"assay": "ymaze", "run_id": RUN, "job_id": JOB,
            "asset": {"id": "asset", "name": "y.mp4", "path": "memory",
                      "metadata": {"duration_s": 1.0}},
            "window": window or {"start_s": 0.0, "end_s": 1.0},
            "orientation": 0, "fps_override": None, "sample_fps": None,
            "units": units, "style": {}}


def forbid_detection(monkeypatch):
    """任何检测器/背景创建都应让纯人工用例失败。"""

    def refuse(*args, **kwargs):
        raise AssertionError("人工 Y 不应创建检测器或读取背景")

    monkeypatch.setattr(detection, "prepare_background", refuse)
    monkeypatch.setattr(detection, "shared_detector", refuse)
    monkeypatch.setattr(detection, "find_candidates", refuse)


# -- 1. 固定事件夹具：AABC / 4 / 2 / 1 / 50% ----------------------------

def test_manual_fixture_scores_without_detection(monkeypatch):
    monkeypatch.setattr(media, "VideoSource", Frames)
    forbid_detection(monkeypatch)

    result = ymaze.analyze(AnalysisContext(job=job_for([manual_unit()])))
    summary = result.summary[0]

    assert summary["total_time_s"] == pytest.approx(0.8, abs=1e-9)
    assert summary["analysis_window_start_s"] == pytest.approx(0.1, abs=1e-9)
    assert summary["analysis_window_end_s"] == pytest.approx(0.9, abs=1e-9)
    assert summary["total_arm_entries"] == 4
    assert summary["alternation_count"] == 1
    assert summary["alternation_opportunities"] == 2
    assert summary["alternation_percent"] == pytest.approx(50.0, abs=1e-9)
    assert (summary["arm_A_entries"], summary["arm_B_entries"],
            summary["arm_C_entries"]) == (2, 1, 1)
    assert summary["bin_seconds"] == pytest.approx(0.4, abs=1e-9)
    # 人工结果不含占用、位置时长、距离或速度。
    for key in ("observed_time_s", "missing_time_s", "center_time_s",
                "arm_A_time_s", "total_distance_cm", "mean_speed_cm_s",
                "cm_per_px_est"):
        assert key not in summary, key

    assert [event["target"] for event in result.events] == ["A", "A", "B", "C"]
    assert [event["entry_index"] for event in result.events] == [1, 2, 3, 4]
    assert [event["time_s"] for event in result.events] == [
        0.15, 0.35, 0.55, 0.75]
    assert [event["confirmed_s"] for event in result.events] == [
        0.15, 0.35, 0.55, 0.75]
    assert [event["source"] for event in result.events] == ["manual"] * 4
    assert [event["recent_sequence"] for event in result.events] == [
        "A", "A-A", "A-A-B", "A-B-C"]

    assert len(result.bins) == 2
    assert [(item["bin_index"], item["total_arm_entries"])
            for item in result.bins] == [(1, 2), (2, 2)]
    assert result.bins[0]["alternation_percent"] is None
    assert result.bins[1]["alternation_count"] == 1
    assert result.bins[1]["alternation_opportunities"] == 2
    assert result.bins[1]["alternation_percent"] == pytest.approx(50.0, abs=1e-9)

    # 人工单位没有逐帧观测与候选，入臂表与事件同源。
    assert result.observations == []
    assert result.tables["candidates"] == []
    assert result.tables["arm_entries"] == result.events
    metadata = result.metadata["units"][UNIT]
    assert metadata == {"mode": "manual",
                        "event_time": "video absolute seconds",
                        "position_available": False,
                        "tracking_reliable": False}
    assert result.warnings == []


# -- 2. 混合单位：检测器只收自动单位 -----------------------------------

def test_mixed_units_only_auto_reaches_detector(monkeypatch):
    monkeypatch.setattr(media, "VideoSource", Frames)
    prepared: list[str] = []
    real_prepare = ymaze._prepare

    def spy_prepare(unit):
        prepared.append(unit["id"])
        return real_prepare(unit)

    monkeypatch.setattr(ymaze, "_prepare", spy_prepare)
    monkeypatch.setattr(detection, "find_candidates",
                        lambda *args, **kwargs: ([], "no_candidate"))

    result = ymaze.analyze(
        AnalysisContext(job=job_for([manual_unit(), auto_unit()])))
    assert AUTO in prepared and UNIT not in prepared, "人工单位不得进入捕获/检测"
    manual_summary = result.summary[0]
    assert manual_summary["total_arm_entries"] == 4
    auto_summary = result.summary[1]
    assert auto_summary["total_arm_entries"] == 0
    assert result.metadata["units"][UNIT]["mode"] == "manual"
    assert result.metadata["units"][AUTO]["mode"] == "automatic"
    # 自动单位仍产生逐帧观测；人工单位不产生。
    assert all(row["unit_id"] == AUTO for row in result.observations)

    # 自动单位的静态背景缺失会让整个作业在分析前失败。
    broken = auto_unit()
    broken["detection"] = {"bg_method": "static",
                           "static_background": "缺失背景.png"}
    with pytest.raises(detection.DetectionError) as captured:
        ymaze.analyze(AnalysisContext(job=job_for([manual_unit(), broken])))
    assert "静态背景" in str(captured.value)

    # 该豁免只属于 Y 人工：其他实验的人工单位仍走常规检测校验。
    barnes = {"id": "b" * 32, "name": "Barnes", "animal_id": None,
              "window": None, "geometry": {}, "options": {},
              "detection": {"bg_type": "xx"},
              "manual": {"hole_sequence": [3, 7, 11], "entry": 7}}
    barnes_draft = {"assay": "barnes", "videos": [{
        "asset_id": "asset", "window": {"start_s": 0.0, "end_s": 10.0},
        "orientation": 0, "fps_override": None, "sample_fps": None,
        "units": [barnes]}]}
    issues = engine.validate_draft(barnes_draft, complete=True)
    assert any(issue["field"].endswith("detection.bg_type")
               for issue in issues), "Barnes 人工仍校验检测参数"

    manual_draft = {"assay": "ymaze", "videos": [{
        "asset_id": "asset", "window": {"start_s": 0.0, "end_s": 10.0},
        "orientation": 0, "fps_override": None, "sample_fps": None,
        "units": [manual_unit(detection_settings={"bg_type": "xx"})]}]}
    assert engine.validate_draft(manual_draft, complete=True) == [], \
        "Y 人工不校验未使用的检测参数"


# -- 3. 完整窗预览与人工校验 -------------------------------------------

def make_frame(index=0, time_s=0.0):
    return FrameInfo(image=np.full((96, 128, 3), 240, dtype=np.uint8),
                     index=index, time_s=time_s, pts=0, time_base=None,
                     clock_source="pts")


def test_manual_preview_full_window_and_validation(client, store, tmp_path,
                                                   monkeypatch):
    source = tmp_path / "y.mp4"
    source.write_bytes(b"placeholder")
    asset = store.create_asset(
        name="y.mp4", path=str(source), project_id=None,
        metadata={"width": 128, "height": 96, "duration_s": 4.0,
                  "fps": 10.0, "frame_count": 40})
    entries = [{"time_s": 3.2, "arm": "A"}, {"time_s": 3.4, "arm": "A"},
               {"time_s": 3.6, "arm": "B"}, {"time_s": 3.8, "arm": "C"}]
    draft = store.create_draft("ymaze", videos=[{
        "asset_id": asset["id"],
        "window": {"start_s": 0.0, "end_s": 4.0},
        "orientation": 0, "fps_override": None, "sample_fps": None,
        "units": [{"id": UNIT, "name": "人工鼠", "window": None,
                   "geometry": {}, "options": {"bin_seconds": 1.0},
                   "detection": {}, "manual": {"mode": "manual",
                                               "entries": entries}}],
    }])
    monkeypatch.setattr(media, "VideoSource", Frames)
    monkeypatch.setattr(preview.media, "read_frame_info",
                        lambda *args, **kwargs: make_frame(0, 0.0))
    monkeypatch.setattr(preview.media, "read_frame_at_time",
                        lambda *args, **kwargs: make_frame(0, 0.0))

    def preview_here():
        return client.post(
            f"/api/drafts/{draft['id']}/preview",
            json={"asset_id": asset["id"], "unit_id": UNIT,
                  "frame_index": 0, "time_s": 0.0, "kind": "behavior"})

    response = preview_here()
    assert response.status_code == 201, response.get_json()
    data = response.get_json()["data"]
    scope = data["preview_scope"]
    assert scope["mode"] == "configured"
    assert scope["start_s"] == pytest.approx(0.0, abs=1e-9)
    assert scope["end_s"] == pytest.approx(4.0, abs=1e-9)
    behavior = data["summary"]["behavior"]
    assert behavior["manual"] is True
    assert behavior["total_time_s"] == pytest.approx(4.0, abs=1e-9)
    assert behavior["total_arm_entries"] == 4
    assert behavior["alternation_percent"] == pytest.approx(50.0, abs=1e-9)
    assert data["summary"]["detection_available"] is False

    # 同一草稿走 /api/jobs：4 次/50%，页面只显示人工实际有的字段。
    monkeypatch.setattr(
        "behaviourlab.engine.media.probe_video",
        lambda path: {"width": 128, "height": 96, "duration_s": 4.0,
                      "fps": 10.0, "frame_count": 40})
    started = client.post("/api/jobs", json={"draft_id": draft["id"]})
    assert started.status_code == 202, started.get_json()
    run = started.get_json()["data"]
    job = run["jobs"][0]
    assert job["status"] == "completed", job
    summary = job["manifest"]["units"][0]["summary"]
    assert summary["total_arm_entries"] == 4
    assert summary["alternation_percent"] == pytest.approx(50.0, abs=1e-9)
    sequence = job["manifest"]["units"][0]["figures"]["sequence"]
    assert sequence

    page = client.get(f"/results/{run['run_id']}").get_data(as_text=True)
    for label in ("总入臂次数", "交替次数", "交替机会数", "自发交替率",
                  "来源 人工记录", "入臂序列（"):
        assert label in page, label
    assert "中心区时间" not in page, "人工结果不伪造中心占用"
    assert "估计总距离" not in page
    assert "候选（" not in page
    assert f'src="/runs/{run["run_id"]}/{job["job_id"]}/{sequence}"' in page

    # 配置窗缩到 [0,1)：窗内 0 条 → 空比例，但草稿事件不删。
    updated = client.put(f"/api/drafts/{draft['id']}", json={"videos": [{
        "asset_id": asset["id"], "window": {"start_s": 0.0, "end_s": 1.0},
        "orientation": 0, "fps_override": None, "sample_fps": None,
        "units": [{"id": UNIT, "name": "人工鼠", "window": None,
                   "geometry": {}, "options": {"bin_seconds": 1.0},
                   "detection": {}, "manual": {"mode": "manual",
                                               "entries": entries}}],
    }]})
    assert updated.status_code == 200, updated.get_json()
    narrowed = preview_here().get_json()["data"]
    narrowed_behavior = narrowed["summary"]["behavior"]
    assert narrowed_behavior["total_arm_entries"] == 0
    assert narrowed_behavior["alternation_percent"] is None
    assert any("没有人工入臂记录" in text for text in narrowed["warnings"])
    stored = client.get(f"/api/drafts/{draft['id']}").get_json()["data"]
    assert stored["draft"]["videos"][0]["units"][0]["manual"]["entries"] == entries

    # 人工校验：重复时刻/倒序/未知臂/相对窗口/非正箱宽都报错，A,A 合法。
    def issues_for(entries_value, **overrides):
        unit = manual_unit(entries_value, **overrides)
        return [issue["field"] for issue in
                ymaze.validate_manual(unit, complete=True)]

    assert issues_for([{"time_s": 0.2, "arm": "A"},
                       {"time_s": 0.2, "arm": "B"}]) == [
        "manual.entries[1].time_s"]
    assert issues_for([{"time_s": 0.3, "arm": "A"},
                       {"time_s": 0.2, "arm": "B"}]) == [
        "manual.entries[1].time_s"]
    assert issues_for([{"time_s": 0.2, "arm": "D"}]) == [
        "manual.entries[0].arm"]
    assert issues_for([{"time_s": 0.2, "arm": "A"},
                       {"time_s": 0.4, "arm": "A"}]) == []
    relative = manual_unit(window={"mode": "relative", "offset_s": 0,
                                   "duration_s": 60})
    assert [issue["field"] for issue in
            ymaze.validate_manual(relative, complete=True)] == ["window.mode"]
    assert [issue["field"] for issue in ymaze.validate_manual(
        manual_unit(options={"bin_seconds": 0}), complete=True)] == [
        "options.bin_seconds"]
    # 清空后仍可保存，complete 校验才要求非空。
    assert ymaze.validate_manual(manual_unit([]), complete=False) == []
    assert [issue["field"] for issue in ymaze.validate_manual(
        manual_unit([]), complete=True)] == ["manual.entries"]
