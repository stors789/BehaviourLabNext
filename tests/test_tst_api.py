"""第 09 块 TST 接口：开始任务校验、结果页概览与指标顺序。"""

from __future__ import annotations

import pytest

from behaviourlab import worker
from behaviourlab.experiments import tst
from behaviourlab.schema import AnalysisResult

ARENA = {"type": "polygon",
         "points": [[5, 5], [120, 5], [120, 120], [5, 120]]}
BODY_LINE = {"type": "line", "points": [[10, 10], [70, 10]]}


def seed(store, tmp_path, *, geometry=None, options=None):
    source = tmp_path / "tst.mp4"
    source.write_bytes(b"placeholder")
    asset = store.create_asset(
        name="悬尾.mp4", path=str(source), project_id=None,
        metadata={"width": 200, "height": 200, "duration_s": 1.0,
                  "fps": 10.0, "frame_count": 10},
    )
    draft = store.create_draft("tst", videos=[{
        "asset_id": asset["id"],
        "window": {"start_s": 0.0, "end_s": 1.0},
        "orientation": 0, "fps_override": None, "sample_fps": None,
        "units": [{
            "id": "e" * 32, "name": "小鼠 1", "window": None,
            "geometry": geometry if geometry is not None else {
                "arena": ARENA, "body_line": BODY_LINE},
            "options": dict(tst.DEFAULTS if options is None else options),
            "detection": {}, "manual": None,
        }],
    }])
    return asset, draft


def fake_result() -> AnalysisResult:
    summary = {
        "unit_id": "e" * 32, "unit_name": "小鼠 1",
        "total_time_s": 3.5, "immobility_time_s": 3.4,
        "struggling_time_s": 0.0, "uncertain_time_s": 0.1,
        "swing_time_s": 2.0,
        "immobility_percent": 3.4 / 3.5 * 100.0,
        "struggling_percent": 0.0, "uncertain_percent": 0.1 / 3.5 * 100.0,
        "swing_percent": 2.0 / 3.5 * 100.0,
        "detected_time_s": 3.4, "detected_rate_percent": 3.4 / 3.5 * 100.0,
        "detected_frames": 34, "total_frames": 35,
        "immobility_episodes": 0, "struggling_episodes": 0,
        "immobility_frequency_per_min": 0.0,
        "struggling_frequency_per_min": 0.0,
        "immobility_bouts": 2, "struggling_bouts": 0,
        "latency_to_immobility_s": None,
        "analysis_window_start_s": 0.0, "analysis_window_end_s": 3.5,
        "thresholds": {"flow_thr_hi": 50.0, "flow_thr_lo": 35.0,
                       "speed_std_thr_hi": 40.0, "speed_std_thr_lo": 20.0},
        "threshold_estimated": False, "threshold_samples": 34,
        "mode": "classic", "auto_behavior_threshold": False,
        "behavior_threshold_sensitivity": 0.0,
    }
    observations = []
    for index in range(35):
        if index < 20:
            state = "swing"
        elif index == 20:
            state = "unknown"
        else:
            state = "immobile"
        start = 0.1 * index
        observations.append({
            "unit_id": "e" * 32, "unit_name": "小鼠 1", "frame": index,
            "time_s": start, "interval_start_s": start,
            "interval_end_s": start + 0.1, "dt_s": 0.1, "in_window": True,
            "detected": state != "unknown", "state": state,
            "state_group": "immobile" if state in ("swing", "immobile") else state,
        })
    return AnalysisResult(observations=observations, summary=[summary],
                          events=[], bins=[], warnings=[],
                          metadata={"assay": "tst", "body_length_mm": 60.0})


def test_tst_result_page_order_and_labels(client, store, tmp_path, monkeypatch):
    asset, draft = seed(store, tmp_path)
    monkeypatch.setattr(
        "behaviourlab.engine.media.probe_video",
        lambda path: {"width": 200, "height": 200, "duration_s": 1.0,
                      "fps": 10.0, "frame_count": 10})
    monkeypatch.setattr(worker, "run_frozen_job", lambda job: fake_result())

    response = client.post(f"/api/drafts/{draft['id']}/analyze")
    assert response.status_code == 201, response.get_json()
    run_id = response.get_json()["data"]["run_id"]
    page = client.get(f"/results/{run_id}").get_data(as_text=True)

    assert "尾悬实验 / TST" in page
    labels = ["非挣扎时间（含摆动）", "挣扎时间", "不确定时间",
              "非挣扎比例", "挣扎比例", "首次挣扎转非挣扎潜伏期（经典）"]
    positions = [page.index(label) for label in labels]
    assert positions == sorted(positions)
    assert "检测时间比例" in page
    assert "视觉运动规则使用可调经验阈值" in page
    assert "—" in page  # 潜伏期为空时显示破折号

    manifest = response.get_json()["data"]["jobs"][0]["manifest"]
    activity = manifest["units"][0]["figures"]["activity"]
    assert activity and activity.endswith("_activity.png")
    assert activity in manifest["files"]
    job_id = response.get_json()["data"]["jobs"][0]["job_id"]
    assert f'alt="活动曲线"' in page
    assert "没有可用的场地尺寸" not in page
    image = client.get(f"/runs/{run_id}/{job_id}/{activity}")
    assert image.status_code == 200
    assert image.mimetype == "image/png"


def test_tst_analyze_requires_geometry_and_classic_mode(client, store, tmp_path,
                                                      monkeypatch):
    asset, draft = seed(store, tmp_path, geometry={})
    store.update_draft(draft["id"], {"videos": draft["videos"]})
    monkeypatch.setattr(
        "behaviourlab.engine.media.probe_video",
        lambda path: {"width": 200, "height": 200, "duration_s": 1.0,
                      "fps": 10.0, "frame_count": 10})
    response = client.post(f"/api/drafts/{draft['id']}/analyze")
    assert response.status_code == 400
    fields = [issue["field"] for issue in response.get_json()["error"]["fields"]]
    assert "videos[0].units[0].geometry.arena" in fields
    assert "videos[0].units[0].geometry.body_line" in fields
    assert store.list_jobs() == []

    # lis 参数可保存，最终校验拒绝，不静默按 classic 跑。
    values = dict(tst.DEFAULTS)
    values["mode"] = "lis"
    asset2, draft2 = seed(store, tmp_path, options=values)
    saved = client.put(f"/api/drafts/{draft2['id']}",
                       json={"videos": draft2["videos"]})
    assert saved.status_code == 200
    blocked = client.post(f"/api/drafts/{draft2['id']}/analyze")
    assert blocked.status_code == 400
    messages = [issue["message"]
                for issue in blocked.get_json()["error"]["fields"]]
    assert any("微动残差模式尚未接入" in message for message in messages)


def test_relative_unit_window_previews_configured_range():
    from behaviourlab import preview

    scope = preview._scope(
        "tst", {"window": {"start_s": 0.0, "end_s": 10.0}},
        {"window": {"mode": "relative", "offset_s": 0.0, "duration_s": 2.0}},
        {"mode": "classic"}, 5.0, "behavior", 10.0)
    assert scope["mode"] == "configured"
    assert scope["start_s"] == 0.0
    assert scope["end_s"] == 10.0
    assert scope["relative"] is True


class PreviewSource:
    """行为预览用的小段解码头：与 VideoSource 相同的 iter_frames 接口。"""

    def __init__(self, path, *, start_s=0.0, end_s=None, **kwargs):
        import cv2
        import numpy as np
        from behaviourlab.schema import Frame

        self._frames = []
        begin = float(start_s or 0.0)
        stop = float(end_s if end_s is not None else begin + 1.0)
        count = max(1, int(round((stop - begin) / 0.1)))
        for index in range(count):
            moment = begin + index * 0.1
            image = np.full((200, 200, 3), 255, dtype=np.uint8)
            cv2.ellipse(image, (40 + 4 * index, 60), (24, 12), 0, 0, 360,
                        (0, 0, 0), -1)
            self._frames.append(Frame(
                image=image, index=index, time_s=moment, dt_s=0.1,
                elapsed_s=moment, analysis_fps=10.0,
                interval_start_s=moment, interval_end_s=moment + 0.1))

    def iter_frames(self):
        return iter(self._frames)


def test_tst_behavior_preview_runs_core(client, store, tmp_path, monkeypatch):
    from behaviourlab import media, preview
    from behaviourlab.media import FrameInfo

    asset, draft = seed(store, tmp_path)
    videos = draft["videos"]
    videos[0]["window"] = {"start_s": 1.0, "end_s": 4.0}
    store.update_draft(draft["id"], {"videos": videos})
    monkeypatch.setattr(media, "VideoSource", PreviewSource)
    monkeypatch.setattr(
        preview.media, "read_frame_at_time",
        lambda *args, **kwargs: FrameInfo(
            image=__import__("numpy").full((200, 200, 3), 255, dtype="uint8"),
            index=20, time_s=2.0, pts=None, time_base=None,
            clock_source="fps_override"))

    response = client.post(
        f"/api/drafts/{draft['id']}/preview",
        json={"asset_id": asset["id"], "unit_id": "e" * 32,
              "time_s": 2.0, "kind": "behavior"})
    assert response.status_code == 201, response.get_json()
    data = response.get_json()["data"]
    assert data["preview_scope"]["start_s"] == 1.0
    assert data["preview_scope"]["end_s"] == 4.0
    behavior = data["summary"]["behavior"]
    assert behavior is not None
    assert behavior["start_s"] == 1.0
    assert behavior["end_s"] == 4.0
    assert behavior["total_time_s"] == pytest.approx(3.0, abs=1e-9)
    assert behavior["rows"] >= 25
    assert set(behavior["thresholds"]) == {"flow_thr_hi", "flow_thr_lo",
                                           "speed_std_thr_hi",
                                           "speed_std_thr_lo"}


def test_tst_relative_behavior_preview_uses_configured_range(
        client, store, tmp_path, monkeypatch):
    from behaviourlab import media, preview
    from behaviourlab.media import FrameInfo

    asset, draft = seed(store, tmp_path)
    videos = draft["videos"]
    videos[0]["window"] = {"start_s": 1.0, "end_s": 4.0}
    videos[0]["units"][0]["window"] = {"mode": "relative", "offset_s": 0.0,
                                       "duration_s": 2.0}
    store.update_draft(draft["id"], {"videos": videos})
    monkeypatch.setattr(media, "VideoSource", PreviewSource)
    monkeypatch.setattr(
        preview.media, "read_frame_at_time",
        lambda *args, **kwargs: FrameInfo(
            image=__import__("numpy").full((200, 200, 3), 255, dtype="uint8"),
            index=20, time_s=2.0, pts=None, time_base=None,
            clock_source="fps_override"))

    response = client.post(
        f"/api/drafts/{draft['id']}/preview",
        json={"asset_id": asset["id"], "unit_id": "e" * 32,
              "time_s": 2.0, "kind": "behavior"})
    assert response.status_code == 201, response.get_json()
    data = response.get_json()["data"]
    assert data["preview_scope"]["mode"] == "configured"
    assert data["preview_scope"]["start_s"] == 1.0
    assert data["preview_scope"]["end_s"] == 4.0
    behavior = data["summary"]["behavior"]
    assert behavior["window_start_s"] == pytest.approx(1.0, abs=1e-9)
    assert behavior["window_end_s"] == pytest.approx(3.0, abs=1e-9)
    assert behavior["total_time_s"] == pytest.approx(2.0, abs=1e-9)
