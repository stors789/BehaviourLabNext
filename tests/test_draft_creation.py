"""新草稿的窗口合同：首只小鼠使用 null 窗口，不伪造模式。"""

from __future__ import annotations


def payload(unit_window, assay="oft", options=None):
    return {
        "assay": assay,
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
                "options": options or {},
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


def test_barnes_hole_count_persists(client):
    created = client.post("/api/drafts", json=payload(None, assay="barnes"))
    assert created.status_code == 201, created.get_json()
    draft = created.get_json()["data"]["draft"]
    assert draft["videos"][0]["units"][0]["options"] == {}

    videos = draft["videos"]
    videos[0]["units"][0]["options"] = {"hole_count": 3}
    videos[0]["units"][0]["geometry"] = {
        "disk": {"type": "ellipse", "center": [100, 100],
                 "radii": [50, 50], "angle": 0},
    }
    saved = client.put(f"/api/drafts/{draft['id']}", json={"videos": videos})
    assert saved.status_code == 200, saved.get_json()

    reloaded = client.get(f"/api/drafts/{draft['id']}").get_json()["data"]["draft"]
    unit = reloaded["videos"][0]["units"][0]
    assert unit["options"]["hole_count"] == 3
    assert unit["geometry"]["disk"]["radii"] == [50, 50]


def test_barnes_hole_count_persists(client):
    created = client.post("/api/drafts", json=payload(None, assay="barnes"))
    assert created.status_code == 201, created.get_json()
    draft = created.get_json()["data"]["draft"]
    assert draft["videos"][0]["units"][0]["options"] == {}

    videos = draft["videos"]
    videos[0]["units"][0]["options"] = {"hole_count": 3}
    videos[0]["units"][0]["geometry"] = {
        "disk": {"type": "ellipse", "center": [100, 100],
                 "radii": [50, 50], "angle": 0},
    }
    saved = client.put(f"/api/drafts/{draft['id']}", json={"videos": videos})
    assert saved.status_code == 200, saved.get_json()

    reloaded = client.get(f"/api/drafts/{draft['id']}").get_json()["data"]["draft"]
    unit = reloaded["videos"][0]["units"][0]
    assert unit["options"]["hole_count"] == 3
    assert unit["geometry"]["disk"]["radii"] == [50, 50]
