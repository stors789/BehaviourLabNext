"""第 01 块接口负向检查：原子分组、null 数组、样式/输出、assay 类型。"""

from __future__ import annotations

from behaviourlab.schema import new_id


def test_bulk_group_creation_is_atomic(client, project):
    kept = client.post(
        f"/api/projects/{project['id']}/groups", json={"name": "保留组"}
    )
    assert kept.status_code == 201

    response = client.post(
        f"/api/projects/{project['id']}/groups",
        json={"groups": [{"name": "A"}, {}]},
    )
    assert response.status_code == 400
    fields = response.get_json()["error"]["fields"]
    assert fields[0]["field"] == "groups[1].name"

    groups = client.get(f"/api/projects/{project['id']}/groups").get_json()["data"]["groups"]
    assert [group["name"] for group in groups] == ["保留组"]


def test_bulk_group_creation_rolls_back_every_row(client, project):
    response = client.post(
        f"/api/projects/{project['id']}/groups",
        json={"groups": [{"name": "A"}, {"name": "B"}, {"name": ""}]},
    )
    assert response.status_code == 400
    groups = client.get(f"/api/projects/{project['id']}/groups").get_json()["data"]["groups"]
    assert groups == []


def test_draft_videos_null_is_rejected_without_change(client):
    asset_id = new_id()
    created = client.post(
        "/api/drafts",
        json={"assay": "oft", "videos": [{"asset_id": asset_id, "units": [{}]}]},
    )
    assert created.status_code == 201
    draft = created.get_json()["data"]["draft"]
    original_videos = draft["videos"]

    response = client.put(f"/api/drafts/{draft['id']}", json={"videos": None})
    assert response.status_code == 400
    assert response.get_json()["error"]["fields"][0]["field"] == "videos"

    unchanged = client.get(f"/api/drafts/{draft['id']}").get_json()["data"]["draft"]
    assert unchanged["videos"] == original_videos

    created_null = client.post("/api/drafts", json={"assay": "oft", "videos": None})
    assert created_null.status_code == 400


def test_draft_units_null_is_rejected_without_change(client):
    asset_id = new_id()
    created = client.post(
        "/api/drafts",
        json={"assay": "oft", "videos": [{"asset_id": asset_id, "units": [{}]}]},
    )
    draft = created.get_json()["data"]["draft"]
    original_videos = draft["videos"]

    response = client.put(
        f"/api/drafts/{draft['id']}",
        json={"videos": [{"asset_id": asset_id, "units": None}]},
    )
    assert response.status_code == 400
    assert response.get_json()["error"]["fields"][0]["field"] == "videos[0].units"

    unchanged = client.get(f"/api/drafts/{draft['id']}").get_json()["data"]["draft"]
    assert unchanged["videos"] == original_videos


def test_draft_empty_video_array_is_still_allowed(client):
    created = client.post(
        "/api/drafts",
        json={"assay": "oft", "videos": [{"asset_id": new_id()}]},
    )
    draft = created.get_json()["data"]["draft"]
    assert len(draft["videos"]) == 1

    response = client.put(f"/api/drafts/{draft['id']}", json={"videos": []})
    assert response.status_code == 200
    assert response.get_json()["data"]["draft"]["videos"] == []


def test_draft_rejects_invalid_style_without_change(client):
    draft = client.post("/api/drafts", json={"assay": "oft"}).get_json()["data"]["draft"]
    original = draft["style"]

    response = client.put(
        f"/api/drafts/{draft['id']}",
        json={"style": {"color": "red", "line_width": 0, "dpi": 1,
                        "colormap": "no-such-map"}},
    )
    assert response.status_code == 400
    fields = {issue["field"] for issue in response.get_json()["error"]["fields"]}
    assert {"style.color", "style.line_width", "style.dpi",
            "style.colormap"}.issubset(fields)

    unchanged = client.get(f"/api/drafts/{draft['id']}").get_json()["data"]["draft"]
    assert unchanged["style"] == original


def test_draft_rejects_invalid_output_without_change(client):
    draft = client.post("/api/drafts", json={"assay": "oft"}).get_json()["data"]["draft"]
    original = draft["output"]

    response = client.put(
        f"/api/drafts/{draft['id']}",
        json={"output": {"annotated_video": "yes"}},
    )
    assert response.status_code == 400
    issues = response.get_json()["error"]["fields"]
    assert issues[0]["field"] == "output.annotated_video"

    unchanged = client.get(f"/api/drafts/{draft['id']}").get_json()["data"]["draft"]
    assert unchanged["output"] == original


def test_validate_reports_saved_style_issues(client, app):
    draft = client.post("/api/drafts", json={"assay": "oft"}).get_json()["data"]["draft"]
    store = app.config["STORE"]
    store.update_draft(
        draft["id"],
        {"style": {"color": "red", "line_width": 0, "dpi": 1, "colormap": "nope"}},
    )

    response = client.get(f"/api/drafts/{draft['id']}/validate")
    assert response.status_code == 200
    fields = {issue["field"] for issue in response.get_json()["data"]["issues"]}
    assert {"style.color", "style.line_width", "style.dpi",
            "style.colormap"}.issubset(fields)


def test_incomplete_draft_still_saves(client):
    response = client.post("/api/drafts", json={"assay": "oft"})
    assert response.status_code == 201
    draft = response.get_json()["data"]["draft"]
    assert draft["videos"] == []
    assert draft["style"]["color"] == "#B86245"


def test_draft_assay_type_errors_return_400(client):
    before = client.get("/api/drafts").get_json()["data"]["drafts"]

    for value in ([], {}):
        response = client.post("/api/drafts", json={"assay": value})
        assert response.status_code == 400
        assert response.get_json()["error"]["fields"][0]["field"] == "assay"

    after = client.get("/api/drafts").get_json()["data"]["drafts"]
    assert len(after) == len(before)


def test_draft_update_assay_type_error_keeps_draft(client):
    draft = client.post("/api/drafts", json={"assay": "oft"}).get_json()["data"]["draft"]

    response = client.put(f"/api/drafts/{draft['id']}", json={"assay": {}})
    assert response.status_code == 400
    assert response.get_json()["error"]["fields"][0]["field"] == "assay"

    unchanged = client.get(f"/api/drafts/{draft['id']}").get_json()["data"]["draft"]
    assert unchanged["assay"] == "oft"
