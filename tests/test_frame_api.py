"""源帧接口：JPEG、缓存头、源时间与错误状态。"""

from __future__ import annotations

import numpy as np
import pytest

from behaviourlab import media as media_service
from behaviourlab.media import ClockError, FrameInfo
from behaviourlab.web import app as app_module


def make_asset(store, tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"fake-video")
    return store.create_asset(name="clip.mp4", path=str(path), metadata={})


def test_frame_endpoint_returns_jpeg_and_headers(client, store, tmp_path, monkeypatch):
    asset = make_asset(store, tmp_path)

    def fake_info(path, index, orientation=0, fps_override=None):
        assert index == 4
        assert orientation == 90
        assert fps_override == 20.0
        return FrameInfo(
            image=np.zeros((6, 4, 3), dtype=np.uint8),
            index=4,
            time_s=0.4,
            pts=400,
            time_base=0.001,
            clock_source="fps_override",
        )

    monkeypatch.setattr(media_service, "read_frame_info", fake_info)
    response = client.get(
        f"/api/assets/{asset['id']}/frame",
        query_string={"index": 4, "orientation": 90, "fps_override": 20},
    )

    assert response.status_code == 200
    assert response.mimetype == "image/jpeg"
    assert response.data.startswith(b"\xff\xd8")
    assert response.headers["Cache-Control"] == "no-store"
    assert float(response.headers["X-Source-Time"]) == 0.4
    assert response.headers["X-Source-Frame"] == "4"
    assert response.headers["X-Clock-Source"] == "fps_override"


def test_frame_endpoint_rejects_invalid_settings(client, store, tmp_path):
    asset = make_asset(store, tmp_path)
    base = f"/api/assets/{asset['id']}/frame"

    assert client.get(base).status_code == 400
    assert client.get(base, query_string={"index": "abc"}).status_code == 400
    assert client.get(base, query_string={"index": -1}).status_code == 400
    assert client.get(
        base, query_string={"index": 0, "orientation": 45}
    ).status_code == 400
    assert client.get(
        base, query_string={"index": 0, "fps_override": "nan"}
    ).status_code == 400
    assert client.get(
        base, query_string={"index": 0, "fps_override": 0}
    ).status_code == 400


def test_frame_endpoint_reports_unreadable_or_clock_error(client, store, tmp_path, monkeypatch):
    asset = make_asset(store, tmp_path)

    def broken(path, index, orientation=0, fps_override=None):
        raise ClockError("源时钟不可解释")

    monkeypatch.setattr(media_service, "read_frame_info", broken)
    response = client.get(
        f"/api/assets/{asset['id']}/frame", query_string={"index": 2}
    )
    assert response.status_code == 422
    payload = response.get_json()
    assert payload["ok"] is False
    assert "源时钟不可解释" in payload["error"]["message"]
    assert payload["error"]["fields"][0]["field"] == "index"


def test_frame_endpoint_asset_not_found(client):
    response = client.get("/api/assets/missing/frame", query_string={"index": 0})
    assert response.status_code == 404
    assert response.get_json()["ok"] is False


def test_frame_endpoint_unreadable_media_returns_422(client, store, tmp_path):
    asset = make_asset(store, tmp_path)
    response = client.get(
        f"/api/assets/{asset['id']}/frame", query_string={"index": 0}
    )
    assert response.status_code == 422
    assert response.get_json()["ok"] is False
