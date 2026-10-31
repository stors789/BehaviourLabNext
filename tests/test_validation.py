"""人工 Y 窗口约束与源身份校验。"""

from __future__ import annotations

import os

from behaviourlab import engine
from behaviourlab.schema import new_id


def test_ymaze_manual_only_allows_global_or_exact_window(client):
    response = client.post(
        "/api/drafts",
        json={
            "assay": "ymaze",
            "videos": [
                {
                    "asset_id": new_id(),
                    "units": [
                        {
                            "name": "小鼠 1",
                            "window": {"mode": "relative", "offset_s": 0,
                                       "duration_s": 60},
                            "manual": {"entries": [{"time_s": 1.0, "arm": "A"}]},
                        }
                    ],
                }
            ],
        },
    )
    assert response.status_code == 400
    issues = response.get_json()["error"]["fields"]
    assert any(issue["field"].endswith("window.mode") for issue in issues)
    assert all(issue.get("unit_id") for issue in issues if "unit_id" in issue)


def test_identity_mismatch_is_reported(tmp_path):
    video = tmp_path / "clip.bin"
    video.write_bytes(b"abcd")
    stat = os.stat(video)
    asset = {
        "path": str(video),
        "identity": {
            "path": str(video.resolve()),
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "mtime_ns": stat.st_mtime_ns,
        },
    }
    assert engine.verify_identity(asset) == []

    asset["identity"]["size"] = stat.st_size + 1
    issues = engine.verify_identity(asset)
    assert issues and issues[0]["field"] == "asset.identity"


def test_exact_window_outside_video_range_is_rejected(client, app):
    payload = {
        "assay": "oft",
        "videos": [{
            "asset_id": new_id(),
            "window": {"start_s": 0.0, "end_s": 10.0},
            "units": [{
                "name": "小鼠 1",
                "window": {"mode": "exact", "start_s": 11.0, "end_s": 12.0},
            }],
        }],
    }
    created = client.post("/api/drafts", json=payload)
    assert created.status_code == 400
    fields = [issue["field"] for issue in created.get_json()["error"]["fields"]]
    assert "videos[0].units[0].window.end_s" in fields
    assert any(issue["message"] == "窗口与视频范围没有重叠"
               for issue in created.get_json()["error"]["fields"])

    draft = client.post("/api/drafts", json={
        "assay": "oft",
        "videos": [{"asset_id": new_id(),
                    "window": {"start_s": 0.0, "end_s": 10.0},
                    "units": [{"name": "小鼠 1"}]}],
    }).get_json()["data"]["draft"]
    updated = client.put(f"/api/drafts/{draft['id']}", json={
        "videos": [{
            "asset_id": draft["videos"][0]["asset_id"],
            "window": {"start_s": 0.0, "end_s": 10.0},
            "units": [{
                "id": draft["videos"][0]["units"][0]["id"],
                "name": "小鼠 1",
                "window": {"mode": "exact", "start_s": 11.0, "end_s": 12.0},
            }],
        }],
    })
    assert updated.status_code == 400

    # 已保存的坏窗口在最终校验入口仍要报告。
    store = app.config["STORE"]
    store.update_draft(draft["id"], {"videos": [{
        "asset_id": draft["videos"][0]["asset_id"],
        "window": {"start_s": 0.0, "end_s": 10.0},
        "units": [{
            "id": draft["videos"][0]["units"][0]["id"],
            "name": "小鼠 1",
            "window": {"mode": "exact", "start_s": 11.0, "end_s": 12.0},
        }],
    }]})
    checked = client.get(f"/api/drafts/{draft['id']}/validate")
    issues = checked.get_json()["data"]["issues"]
    assert any(issue["message"] == "窗口与视频范围没有重叠"
               for issue in issues)
    posted = client.post(f"/api/drafts/{draft['id']}/validate")
    assert any(issue["message"] == "窗口与视频范围没有重叠"
               for issue in posted.get_json()["data"]["issues"])


def test_manual_y_restriction_follows_mode_not_records(client):
    response = client.post("/api/drafts", json={
        "assay": "ymaze",
        "videos": [{
            "asset_id": new_id(),
            "units": [{
                "name": "小鼠 1",
                "options": {"mode": "manual"},
                "manual": None,
                "window": {"mode": "relative", "offset_s": 0,
                           "duration_s": 60},
            }],
        }],
    })
    assert response.status_code == 400
    fields = [issue["field"] for issue in response.get_json()["error"]["fields"]]
    assert "videos[0].units[0].window.mode" in fields


def test_exact_window_outside_video_range_is_rejected(client, app):
    payload = {
        "assay": "oft",
        "videos": [{
            "asset_id": new_id(),
            "window": {"start_s": 0.0, "end_s": 10.0},
            "units": [{
                "name": "小鼠 1",
                "window": {"mode": "exact", "start_s": 11.0, "end_s": 12.0},
            }],
        }],
    }
    created = client.post("/api/drafts", json=payload)
    assert created.status_code == 400
    fields = [issue["field"] for issue in created.get_json()["error"]["fields"]]
    assert "videos[0].units[0].window.end_s" in fields
    assert any(issue["message"] == "窗口与视频范围没有重叠"
               for issue in created.get_json()["error"]["fields"])

    draft = client.post("/api/drafts", json={
        "assay": "oft",
        "videos": [{"asset_id": new_id(),
                    "window": {"start_s": 0.0, "end_s": 10.0},
                    "units": [{"name": "小鼠 1"}]}],
    }).get_json()["data"]["draft"]
    updated = client.put(f"/api/drafts/{draft['id']}", json={
        "videos": [{
            "asset_id": draft["videos"][0]["asset_id"],
            "window": {"start_s": 0.0, "end_s": 10.0},
            "units": [{
                "id": draft["videos"][0]["units"][0]["id"],
                "name": "小鼠 1",
                "window": {"mode": "exact", "start_s": 11.0, "end_s": 12.0},
            }],
        }],
    })
    assert updated.status_code == 400

    # 已保存的坏窗口在最终校验入口仍要报告。
    store = app.config["STORE"]
    store.update_draft(draft["id"], {"videos": [{
        "asset_id": draft["videos"][0]["asset_id"],
        "window": {"start_s": 0.0, "end_s": 10.0},
        "units": [{
            "id": draft["videos"][0]["units"][0]["id"],
            "name": "小鼠 1",
            "window": {"mode": "exact", "start_s": 11.0, "end_s": 12.0},
        }],
    }]})
    checked = client.get(f"/api/drafts/{draft['id']}/validate")
    issues = checked.get_json()["data"]["issues"]
    assert any(issue["message"] == "窗口与视频范围没有重叠"
               for issue in issues)
    posted = client.post(f"/api/drafts/{draft['id']}/validate")
    assert any(issue["message"] == "窗口与视频范围没有重叠"
               for issue in posted.get_json()["data"]["issues"])


def test_manual_y_restriction_follows_mode_not_records(client):
    response = client.post("/api/drafts", json={
        "assay": "ymaze",
        "videos": [{
            "asset_id": new_id(),
            "units": [{
                "name": "小鼠 1",
                "options": {"mode": "manual"},
                "manual": None,
                "window": {"mode": "relative", "offset_s": 0,
                           "duration_s": 60},
            }],
        }],
    })
    assert response.status_code == 400
    fields = [issue["field"] for issue in response.get_json()["error"]["fields"]]
    assert "videos[0].units[0].window.mode" in fields


def test_automatic_y_keeps_relative_window(client):
    response = client.post("/api/drafts", json={
        "assay": "ymaze",
        "videos": [{
            "asset_id": new_id(),
            "units": [{
                "name": "小鼠 1",
                "window": {"mode": "relative", "offset_s": 0,
                           "duration_s": 60},
            }],
        }],
    })
    assert response.status_code == 201, response.get_json()
    draft = response.get_json()["data"]["draft"]
    assert draft["videos"][0]["units"][0]["window"]["mode"] == "relative"


def test_automatic_y_keeps_relative_window(client):
    response = client.post("/api/drafts", json={
        "assay": "ymaze",
        "videos": [{
            "asset_id": new_id(),
            "units": [{
                "name": "小鼠 1",
                "window": {"mode": "relative", "offset_s": 0,
                           "duration_s": 60},
            }],
        }],
    })
    assert response.status_code == 201, response.get_json()
    draft = response.get_json()["data"]["draft"]
    assert draft["videos"][0]["units"][0]["window"]["mode"] == "relative"
