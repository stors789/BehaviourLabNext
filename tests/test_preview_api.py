"""预览接口：真实源帧叠画、范围与错误分支。"""

from __future__ import annotations

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


def seed(store, tmp_path, geometry=None, duration=10.0):
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
        "oft",
        videos=[{
            "asset_id": asset["id"],
            "window": {"start_s": 0, "end_s": None},
            "orientation": 0,
            "fps_override": None,
            "sample_fps": None,
            "units": [{
                "id": "a" * 32,
                "name": "小鼠 1",
                "window": None,
                "geometry": geometry or {},
                "options": {},
                "detection": {},
                "manual": None,
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
    assert data["summary"]["detection_available"] is False
    assert data["summary"]["drawn_steps"] == ["arena"]
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
