"""新草稿的窗口合同：首只小鼠使用 null 窗口，不伪造模式。"""

from __future__ import annotations


def payload(unit_window):
    return {
        "assay": "oft",
        "project_id": None,
        "videos": [{
            "asset_id": "a" * 32,
            "window": {"start_s": 0, "end_s": None},
            "orientation": 0,
            "fps_override": None,
            "sample_fps": None,
            "units": [{
                "id": "b" * 32,
                "name": "小鼠 1",
                "window": unit_window,
                "geometry": {},
                "options": {},
                "detection": {},
                "manual": None,
            }],
        }],
    }


def test_unit_window_null_creates_draft(client):
    response = client.post("/api/drafts", json=payload(None))
    assert response.status_code == 201, response.get_json()
    draft = response.get_json()["data"]["draft"]
    assert draft["videos"][0]["units"][0]["window"] is None


def test_unit_window_without_mode_is_rejected(client):
    response = client.post("/api/drafts", json=payload({"start_s": 0, "end_s": None}))
    assert response.status_code == 400
    fields = response.get_json()["error"]["fields"]
    assert fields[0]["field"] == "videos[0].units[0].window.mode"
    assert fields[0]["message"] == "未知窗口模式"


def test_repeated_payloads_create_separate_drafts(client):
    first = client.post("/api/drafts", json=payload(None))
    second = client.post("/api/drafts", json=payload(None))
    assert first.status_code == 201
    assert second.status_code == 201
    assert (first.get_json()["data"]["draft"]["id"]
            != second.get_json()["data"]["draft"]["id"])
