"""预览接口：真实源帧叠画、范围与错误分支。"""

from __future__ import annotations

import cv2
import numpy as np

from behaviourlab import media, preview
from behaviourlab.media import FrameInfo


def make_frame(index: int = 7, time_s: float = 0.28) -> FrameInfo:
    return FrameInfo(
        image=np.zeros((48, 64, 3), dtype=np.uint8),
        index=index,
        time_s=time_s,
        pts=700,
        time_base=None,
        clock_source="pts",
    )


def seed(store, tmp_path, geometry=None, duration=10.0, *, assay="oft",
         window=None, unit_options=None, manual=None):
    if geometry is None:
        geometry = {"arena": {"type": "quad",
                              "points": [[10, 10], [300, 10], [300, 260], [10, 260]]}}
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"placeholder")
    asset = store.create_asset(
        name="示例.mp4",
        path=str(source),
        project_id=None,
        metadata={"width": 640, "height": 480, "duration_s": duration,
                  "fps": 25.0, "frame_count": 250},
    )
    draft = store.create_draft(
        assay,
        videos=[{
            "asset_id": asset["id"],
            "window": window or {"start_s": 0, "end_s": None},
            "orientation": 0,
            "fps_override": None,
            "sample_fps": None,
            "units": [{
                "id": "a" * 32,
                "name": "小鼠 1",
                "window": None,
                "geometry": geometry or {},
                "options": unit_options or {},
                "detection": {},
                "manual": manual,
            }],
        }],
    )
    return asset, draft


def test_preview_draws_frame_and_reports_scope(client, store, tmp_path, monkeypatch):
    asset, draft = seed(store, tmp_path, geometry={
        "arena": {"type": "quad",
                  "points": [[20, 20], [300, 20], [300, 260], [20, 260]]},
    })
    monkeypatch.setattr(preview.media, "read_frame_info",
                        lambda *args, **kwargs: make_frame())

    response = client.post(
        f"/api/drafts/{draft['id']}/preview",
        json={"asset_id": asset["id"], "unit_id": "a" * 32,
              "frame_index": 7, "kind": "detection"},
    )
    assert response.status_code == 201, response.get_json()
    data = response.get_json()["data"]
    assert data["frame_index"] == 7
    assert abs(data["time_s"] - 0.28) <= 1e-12
    assert data["clock_source"] == "pts"
    assert data["preview_scope"]["mode"] == "frame"
    assert data["preview_scope"]["start_s"] == data["preview_scope"]["end_s"]
    assert data["summary"]["detection_available"] is True
    assert data["summary"]["drawn_steps"] == ["arena"]
    detection = data["summary"]["detection"]
    assert detection["candidates"] == 0
    assert detection["missing_reason"] == "foreground_saturated"
    assert detection["region"] == "arena"
    assert data["warnings"]

    image = client.get(data["image_url"])
    assert image.status_code == 200
    assert image.data[:8] == b"\x89PNG\r\n\x1a\n"


def test_behavior_scope_is_bounded_by_window(client, store, tmp_path, monkeypatch):
    asset, draft = seed(store, tmp_path)
    monkeypatch.setattr(preview.media, "read_frame_at_time",
                        lambda *args, **kwargs: make_frame(index=50, time_s=2.0))

    response = client.post(
        f"/api/drafts/{draft['id']}/preview",
        json={"asset_id": asset["id"], "unit_id": "a" * 32,
              "time_s": 2.0, "kind": "behavior"},
    )
    assert response.status_code == 201, response.get_json()
    scope = response.get_json()["data"]["preview_scope"]
    assert scope["mode"] == "range"
    assert abs(scope["start_s"] - 1.0) <= 1e-12
    assert abs(scope["end_s"] - 5.0) <= 1e-12
    assert abs(scope["duration_s"] - 4.0) <= 1e-12


def test_preview_rejects_bad_frame_index(client, store, tmp_path):
    asset, draft = seed(store, tmp_path)
    for value in (True, -1, 1.5, "3"):
        response = client.post(
            f"/api/drafts/{draft['id']}/preview",
            json={"asset_id": asset["id"], "unit_id": "a" * 32,
                  "frame_index": value},
        )
        assert response.status_code == 400, value
        fields = response.get_json()["error"]["fields"]
        assert fields and fields[0]["field"] == "frame_index"


def test_nor_object_point_preview_returns_png(client, store, tmp_path, monkeypatch):
    asset, draft = seed(store, tmp_path, assay="nor", geometry={
        "arena": {"type": "quad",
                  "points": [[20, 20], [300, 20], [300, 260], [20, 260]]},
        "object1": {"type": "point", "points": [[100, 120]]},
        "object2": {"type": "point", "points": [[300, 220]]},
    })
    monkeypatch.setattr(preview.media, "read_frame_info",
                        lambda *args, **kwargs: make_frame(index=12, time_s=0.48))

    response = client.post(
        f"/api/drafts/{draft['id']}/preview",
        json={"asset_id": asset["id"], "unit_id": "a" * 32,
              "frame_index": 12, "kind": "detection"},
    )
    assert response.status_code == 201, response.get_json()
    data = response.get_json()["data"]
    assert data["frame_index"] == 12
    assert data["summary"]["drawn_steps"] == ["arena", "object1", "object2"]
    image = client.get(data["image_url"])
    assert image.status_code == 200
    assert image.data[:8] == b"\x89PNG\r\n\x1a\n"


def test_manual_y_keeps_configured_window(client, store, tmp_path, monkeypatch):
    asset, draft = seed(
        store, tmp_path, assay="ymaze", duration=60.0,
        window={"start_s": 1.0, "end_s": 20.0},
        unit_options={"mode": "manual"},
        manual={"events": [{"time_s": 2.5, "frame": 75, "kind": "event"}]},
    )
    monkeypatch.setattr(preview.media, "read_frame_at_time",
                        lambda *args, **kwargs: make_frame(index=90, time_s=2.99997))

    response = client.post(
        f"/api/drafts/{draft['id']}/preview",
        json={"asset_id": asset["id"], "unit_id": "a" * 32,
              "time_s": 2.99997, "kind": "behavior"},
    )
    assert response.status_code == 201, response.get_json()
    data = response.get_json()["data"]
    scope = data["preview_scope"]
    assert scope["mode"] == "configured"
    assert scope.get("manual") is True
    assert abs(scope["start_s"] - 1.0) <= 1e-12
    assert abs(scope["end_s"] - 20.0) <= 1e-12
    assert data["summary"]["manual_preserved"] is True
    assert data["frame_index"] == 90


def test_manual_barnes_keeps_configured_window(client, store, tmp_path, monkeypatch):
    asset, draft = seed(
        store, tmp_path, assay="barnes", duration=60.0,
        window={"start_s": 1.0, "end_s": 20.0},
        unit_options={"hole_count": 20},
        manual={"hole_sequence": [3, 7, 11], "entry": 7},
    )
    monkeypatch.setattr(preview.media, "read_frame_at_time",
                        lambda *args, **kwargs: make_frame(index=90, time_s=2.99997))

    response = client.post(
        f"/api/drafts/{draft['id']}/preview",
        json={"asset_id": asset["id"], "unit_id": "a" * 32,
              "time_s": 2.99997, "kind": "behavior"},
    )
    assert response.status_code == 201, response.get_json()
    scope = response.get_json()["data"]["preview_scope"]
    assert scope["mode"] == "configured"
    assert abs(scope["start_s"] - 1.0) <= 1e-12
    assert abs(scope["end_s"] - 20.0) <= 1e-12


def test_automatic_behavior_scope_stays_local(client, store, tmp_path, monkeypatch):
    asset, draft = seed(store, tmp_path,
                        window={"start_s": 1.0, "end_s": 20.0})
    monkeypatch.setattr(preview.media, "read_frame_at_time",
                        lambda *args, **kwargs: make_frame(index=90, time_s=2.99997))

    response = client.post(
        f"/api/drafts/{draft['id']}/preview",
        json={"asset_id": asset["id"], "unit_id": "a" * 32,
              "time_s": 2.99997, "kind": "behavior"},
    )
    assert response.status_code == 201, response.get_json()
    scope = response.get_json()["data"]["preview_scope"]
    assert scope["mode"] == "range"
    assert abs(scope["start_s"] - 1.99997) <= 1e-9
    assert abs(scope["end_s"] - 5.99997) <= 1e-9


def test_nor_object_point_preview_returns_png(client, store, tmp_path, monkeypatch):
    asset, draft = seed(store, tmp_path, assay="nor", geometry={
        "arena": {"type": "quad",
                  "points": [[20, 20], [300, 20], [300, 260], [20, 260]]},
        "object1": {"type": "point", "points": [[100, 120]]},
        "object2": {"type": "point", "points": [[300, 220]]},
    })
    monkeypatch.setattr(preview.media, "read_frame_info",
                        lambda *args, **kwargs: make_frame(index=12, time_s=0.48))

    response = client.post(
        f"/api/drafts/{draft['id']}/preview",
        json={"asset_id": asset["id"], "unit_id": "a" * 32,
              "frame_index": 12, "kind": "detection"},
    )
    assert response.status_code == 201, response.get_json()
    data = response.get_json()["data"]
    assert data["frame_index"] == 12
    assert data["summary"]["drawn_steps"] == ["arena", "object1", "object2"]
    image = client.get(data["image_url"])
    assert image.status_code == 200
    assert image.data[:8] == b"\x89PNG\r\n\x1a\n"


def test_manual_y_keeps_configured_window(client, store, tmp_path, monkeypatch):
    asset, draft = seed(
        store, tmp_path, assay="ymaze", duration=60.0,
        window={"start_s": 1.0, "end_s": 20.0},
        unit_options={"mode": "manual"},
        manual={"events": [{"time_s": 2.5, "frame": 75, "kind": "event"}]},
    )
    monkeypatch.setattr(preview.media, "read_frame_at_time",
                        lambda *args, **kwargs: make_frame(index=90, time_s=2.99997))

    response = client.post(
        f"/api/drafts/{draft['id']}/preview",
        json={"asset_id": asset["id"], "unit_id": "a" * 32,
              "time_s": 2.99997, "kind": "behavior"},
    )
    assert response.status_code == 201, response.get_json()
    data = response.get_json()["data"]
    scope = data["preview_scope"]
    assert scope["mode"] == "configured"
    assert scope.get("manual") is True
    assert abs(scope["start_s"] - 1.0) <= 1e-12
    assert abs(scope["end_s"] - 20.0) <= 1e-12
    assert data["summary"]["manual_preserved"] is True
    assert data["frame_index"] == 90


def test_manual_barnes_keeps_configured_window(client, store, tmp_path, monkeypatch):
    asset, draft = seed(
        store, tmp_path, assay="barnes", duration=60.0,
        window={"start_s": 1.0, "end_s": 20.0},
        unit_options={"hole_count": 20},
        manual={"hole_sequence": [3, 7, 11], "entry": 7},
    )
    monkeypatch.setattr(preview.media, "read_frame_at_time",
                        lambda *args, **kwargs: make_frame(index=90, time_s=2.99997))

    response = client.post(
        f"/api/drafts/{draft['id']}/preview",
        json={"asset_id": asset["id"], "unit_id": "a" * 32,
              "time_s": 2.99997, "kind": "behavior"},
    )
    assert response.status_code == 201, response.get_json()
    scope = response.get_json()["data"]["preview_scope"]
    assert scope["mode"] == "configured"
    assert abs(scope["start_s"] - 1.0) <= 1e-12
    assert abs(scope["end_s"] - 20.0) <= 1e-12


def test_automatic_behavior_scope_stays_local(client, store, tmp_path, monkeypatch):
    asset, draft = seed(store, tmp_path,
                        window={"start_s": 1.0, "end_s": 20.0})
    monkeypatch.setattr(preview.media, "read_frame_at_time",
                        lambda *args, **kwargs: make_frame(index=90, time_s=2.99997))

    response = client.post(
        f"/api/drafts/{draft['id']}/preview",
        json={"asset_id": asset["id"], "unit_id": "a" * 32,
              "time_s": 2.99997, "kind": "behavior"},
    )
    assert response.status_code == 201, response.get_json()
    scope = response.get_json()["data"]["preview_scope"]
    assert scope["mode"] == "range"
    assert abs(scope["start_s"] - 1.99997) <= 1e-9
    assert abs(scope["end_s"] - 5.99997) <= 1e-9


def test_preview_reports_real_candidate(client, store, tmp_path, monkeypatch):
    asset, draft = seed(store, tmp_path, geometry={
        "arena": {"type": "quad",
                  "points": [[0, 0], [159, 0], [159, 119], [0, 119]]},
    })
    image = np.full((120, 160, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (60, 50), (90, 80), (0, 0, 0), -1)
    frame = FrameInfo(image=image, index=3, time_s=0.12, pts=300,
                      time_base=None, clock_source="pts")
    monkeypatch.setattr(preview.media, "read_frame_info",
                        lambda *args, **kwargs: frame)

    response = client.post(
        f"/api/drafts/{draft['id']}/preview",
        json={"asset_id": asset["id"], "unit_id": "a" * 32,
              "frame_index": 3, "kind": "detection"},
    )
    assert response.status_code == 201, response.get_json()
    data = response.get_json()["data"]
    found = data["summary"]["detection"]
    assert found["candidates"] == 1
    assert found["missing_reason"] is None
    assert found["region"] == "arena"
    assert abs(found["center"][0] - 75.0) <= 1.0
    assert abs(found["center"][1] - 65.0) <= 1.0
    image_response = client.get(data["image_url"])
    assert image_response.status_code == 200
    assert image_response.data[:8] == b"\x89PNG\r\n\x1a\n"


def test_local_otsu_saves_and_previews(client, store, tmp_path, monkeypatch):
    asset, draft = seed(store, tmp_path, geometry={
        "arena": {"type": "quad",
                  "points": [[0, 0], [159, 0], [159, 119], [0, 119]]},
    })
    videos = draft["videos"]
    videos[0]["units"][0]["detection"] = {"bg_method": "local_otsu"}
    saved = client.put(f"/api/drafts/{draft['id']}", json={"videos": videos})
    assert saved.status_code == 200, saved.get_json()
    stored = client.get(f"/api/drafts/{draft['id']}").get_json()["data"]["draft"]
    assert stored["videos"][0]["units"][0]["detection"]["bg_method"] == "local_otsu"

    image = np.full((120, 160, 3), 235, dtype=np.uint8)
    cv2.rectangle(image, (40, 40), (80, 80), (20, 20, 20), -1)
    frame = FrameInfo(image=image, index=5, time_s=0.2, pts=500,
                      time_base=None, clock_source="pts")
    monkeypatch.setattr(preview.media, "read_frame_info",
                        lambda *args, **kwargs: frame)

    response = client.post(
        f"/api/drafts/{draft['id']}/preview",
        json={"asset_id": asset["id"], "unit_id": "a" * 32,
              "frame_index": 5, "kind": "detection"},
    )
    assert response.status_code == 201, response.get_json()
    found = response.get_json()["data"]["summary"]
    assert found["detection_available"] is True
    assert found["detection"]["candidates"] >= 1
    assert found["detection"]["missing_reason"] is None


def test_unknown_background_method_is_still_rejected(client, store, tmp_path):
    asset, draft = seed(store, tmp_path)
    videos = draft["videos"]
    videos[0]["units"][0]["detection"] = {"bg_method": "flow"}
    response = client.put(f"/api/drafts/{draft['id']}", json={"videos": videos})
    assert response.status_code == 400
    issue = response.get_json()["error"]["fields"][0]
    assert issue["field"] == "videos[0].units[0].detection.bg_method"
    assert issue["message"] == "未知背景方法"


def test_manual_y_previews_without_detection_region(client, store, tmp_path,
                                                    monkeypatch):
    asset, draft = seed(
        store, tmp_path, assay="ymaze", duration=60.0,
        geometry={},
        window={"start_s": 1.0, "end_s": 20.0},
        unit_options={"mode": "manual"},
        manual={"events": [{"time_s": 2.5, "frame": 75, "kind": "event"}]},
    )
    monkeypatch.setattr(preview.media, "read_frame_at_time",
                        lambda *args, **kwargs: make_frame(index=90, time_s=2.99997))

    response = client.post(
        f"/api/drafts/{draft['id']}/preview",
        json={"asset_id": asset["id"], "unit_id": "a" * 32,
              "time_s": 2.99997, "kind": "behavior"},
    )
    assert response.status_code == 201, response.get_json()
    data = response.get_json()["data"]
    scope = data["preview_scope"]
    assert scope["mode"] == "configured"
    assert abs(scope["start_s"] - 1.0) <= 1e-12
    assert abs(scope["end_s"] - 20.0) <= 1e-12
    assert data["frame_index"] == 90
    assert data["summary"]["manual_preserved"] is True
    assert data["summary"]["detection_available"] is False
    assert data["summary"]["detection"] is None
    assert any("人工记录" in text for text in data["warnings"])


def test_automatic_y_with_only_arena_is_rejected(client, store, tmp_path,
                                                 monkeypatch):
    asset, draft = seed(store, tmp_path, assay="ymaze",
                        window={"start_s": 1.0, "end_s": 20.0})
    monkeypatch.setattr(preview.media, "read_frame_info",
                        lambda *args, **kwargs: make_frame(index=7, time_s=0.28))

    response = client.post(
        f"/api/drafts/{draft['id']}/preview",
        json={"asset_id": asset["id"], "unit_id": "a" * 32,
              "frame_index": 7, "kind": "detection"},
    )
    assert response.status_code == 400, response.get_json()
    assert response.get_json()["error"]["message"] == "缺少检测区域"


def test_local_otsu_draft_is_created_from_saved_settings(client):
    saved = client.put("/api/settings/detection",
                       json={"value": {"oft": {"bg_method": "local_otsu"}}})
    assert saved.status_code == 200, saved.get_json()

    created = client.post("/api/drafts", json={"assay": "oft"})
    assert created.status_code == 201, created.get_json()
    draft = created.get_json()["data"]["draft"]
    assert draft["detection"]["bg_method"] == "local_otsu"


def test_nor_preview_reports_head_evidence(client, store, tmp_path, monkeypatch):
    asset, draft = seed(store, tmp_path, assay="nor", geometry={
        "arena": {"type": "quad",
                  "points": [[0, 0], [159, 0], [159, 119], [0, 119]]},
    })
    image = np.full((120, 160, 3), 255, dtype=np.uint8)
    cv2.ellipse(image, (75, 60), (24, 12), 0, 0, 360, (0, 0, 0), -1)
    cv2.line(image, (50, 60), (20, 60), (0, 0, 0), 2)
    frame = FrameInfo(image=image, index=0, time_s=0.0, pts=0,
                      time_base=None, clock_source="pts")
    monkeypatch.setattr(preview.media, "read_frame_info",
                        lambda *args, **kwargs: frame)

    response = client.post(
        f"/api/drafts/{draft['id']}/preview",
        json={"asset_id": asset["id"], "unit_id": "a" * 32,
              "frame_index": 0, "kind": "detection"},
    )
    assert response.status_code == 201, response.get_json()
    found = response.get_json()["data"]["summary"]["detection"]
    assert found["candidates"] == 1
    assert found["head_quality"] > 0.5
    assert found["nose"][0] > found["tail_base"][0]
    assert found["head_angle"] is not None


def test_nor_preview_reports_head_evidence(client, store, tmp_path, monkeypatch):
    asset, draft = seed(store, tmp_path, assay="nor", geometry={
        "arena": {"type": "quad",
                  "points": [[0, 0], [159, 0], [159, 119], [0, 119]]},
    })
    image = np.full((120, 160, 3), 255, dtype=np.uint8)
    cv2.ellipse(image, (75, 60), (24, 12), 0, 0, 360, (0, 0, 0), -1)
    cv2.line(image, (50, 60), (20, 60), (0, 0, 0), 2)
    frame = FrameInfo(image=image, index=0, time_s=0.0, pts=0,
                      time_base=None, clock_source="pts")
    monkeypatch.setattr(preview.media, "read_frame_info",
                        lambda *args, **kwargs: frame)

    response = client.post(
        f"/api/drafts/{draft['id']}/preview",
        json={"asset_id": asset["id"], "unit_id": "a" * 32,
              "frame_index": 0, "kind": "detection"},
    )
    assert response.status_code == 201, response.get_json()
    found = response.get_json()["data"]["summary"]["detection"]
    assert found["candidates"] == 1
    assert found["head_quality"] > 0.5
    assert found["nose"][0] > found["tail_base"][0]
    assert found["head_angle"] is not None


def test_nor_preview_states_unknown_head(client, store, tmp_path, monkeypatch):
    asset, draft = seed(store, tmp_path, assay="nor", geometry={
        "arena": {"type": "quad",
                  "points": [[0, 0], [159, 0], [159, 119], [0, 119]]},
    })
    image = np.full((120, 160, 3), 255, dtype=np.uint8)
    cv2.ellipse(image, (75, 60), (24, 12), 0, 0, 360, (0, 0, 0), -1)
    frame = FrameInfo(image=image, index=0, time_s=0.0, pts=0,
                      time_base=None, clock_source="pts")
    monkeypatch.setattr(preview.media, "read_frame_info",
                        lambda *args, **kwargs: frame)

    response = client.post(
        f"/api/drafts/{draft['id']}/preview",
        json={"asset_id": asset["id"], "unit_id": "a" * 32,
              "frame_index": 0, "kind": "detection"},
    )
    assert response.status_code == 201, response.get_json()
    data = response.get_json()["data"]
    found = data["summary"]["detection"]
    assert found["candidates"] == 1
    assert found["nose"] is None
    assert found["tail_base"] is None
    assert found["head_angle"] is None
    assert found["head_quality"] == 0
    assert any("头向未判定" in text for text in data["warnings"])


def test_nor_preview_states_unknown_head(client, store, tmp_path, monkeypatch):
    asset, draft = seed(store, tmp_path, assay="nor", geometry={
        "arena": {"type": "quad",
                  "points": [[0, 0], [159, 0], [159, 119], [0, 119]]},
    })
    image = np.full((120, 160, 3), 255, dtype=np.uint8)
    cv2.ellipse(image, (75, 60), (24, 12), 0, 0, 360, (0, 0, 0), -1)
    frame = FrameInfo(image=image, index=0, time_s=0.0, pts=0,
                      time_base=None, clock_source="pts")
    monkeypatch.setattr(preview.media, "read_frame_info",
                        lambda *args, **kwargs: frame)

    response = client.post(
        f"/api/drafts/{draft['id']}/preview",
        json={"asset_id": asset["id"], "unit_id": "a" * 32,
              "frame_index": 0, "kind": "detection"},
    )
    assert response.status_code == 201, response.get_json()
    data = response.get_json()["data"]
    found = data["summary"]["detection"]
    assert found["candidates"] == 1
    assert found["nose"] is None
    assert found["tail_base"] is None
    assert found["head_angle"] is None
    assert found["head_quality"] == 0
    assert any("头向未判定" in text for text in data["warnings"])
