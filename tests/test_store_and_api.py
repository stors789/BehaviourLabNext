"""存储级联、API 信封、草稿接口与页面路由的轻量检查。"""

from __future__ import annotations

import json

from behaviourlab.schema import new_id


def test_project_group_animal_flow(client, store):
    created = client.post("/api/projects", json={"name": "学习与记忆"})
    assert created.status_code == 201
    payload = created.get_json()
    assert payload["ok"] is True
    project = payload["data"]["project"]

    group = client.post(
        f"/api/projects/{project['id']}/groups", json={"name": "对照组"}
    )
    assert group.status_code == 201
    group_id = group.get_json()["data"]["groups"][0]["id"]

    animal = client.post(
        f"/api/projects/{project['id']}/animals",
        json={"name": "小鼠 1", "group_id": group_id, "sex": "female"},
    )
    assert animal.status_code == 201
    animals = animal.get_json()["data"]["animals"]
    assert animals[0]["group_name"] == "对照组"

    detail = client.get(f"/api/projects/{project['id']}").get_json()["data"]["project"]
    assert detail["group_count"] == 1
    assert detail["animal_count"] == 1


def test_animal_group_must_belong_to_same_project(client):
    first = client.post("/api/projects", json={"name": "A"}).get_json()["data"]["project"]
    second = client.post("/api/projects", json={"name": "B"}).get_json()["data"]["project"]
    foreign_group = client.post(
        f"/api/projects/{second['id']}/groups", json={"name": "外来组"}
    ).get_json()["data"]["groups"][0]

    response = client.post(
        f"/api/projects/{first['id']}/animals",
        json={"name": "小鼠 1", "group_id": foreign_group["id"]},
    )
    assert response.status_code == 400
    assert response.get_json()["error"]["fields"][0]["field"] == "group_id"


def test_delete_project_cascades_groups_and_animals(store):
    project = store.create_project("临时")
    group = store.create_group(project["id"], "组")
    store.create_animals(project["id"], [{"name": "小鼠 1", "group_id": group["id"]}])

    store.delete_project(project["id"])

    assert store.get_project(project["id"]) is None
    assert store.list_groups(project["id"]) == []
    assert store.list_animals(project["id"]) == []


def test_delete_group_sets_animal_group_null(store):
    project = store.create_project("临时")
    group = store.create_group(project["id"], "组")
    store.create_animals(project["id"], [{"name": "小鼠 1", "group_id": group["id"]}])

    with store.transaction() as connection:
        connection.execute("DELETE FROM groups WHERE id = ?", (group["id"],))

    animals = store.list_animals(project["id"])
    assert animals[0]["group_id"] is None


def test_delete_project_clears_asset_and_draft_bindings(store):
    project = store.create_project("临时")
    asset = store.create_asset(
        name="片段", path="C:/clips/sample.mp4", project_id=project["id"]
    )
    draft = store.create_draft("oft", project_id=project["id"])

    store.delete_project(project["id"])

    assert store.get_asset(asset["id"])["project_id"] is None
    assert store.get_draft(draft["id"])["project_id"] is None


def test_bulk_group_creation(client, project):
    response = client.post(
        f"/api/projects/{project['id']}/groups",
        json={"groups": [{"name": "对照组"}, {"name": "模型组"}]},
    )
    assert response.status_code == 201
    names = [item["name"] for item in response.get_json()["data"]["groups"]]
    assert names == ["对照组", "模型组"]


def test_create_animal_returns_the_new_record(store):
    project = store.create_project("临时")
    store.create_animal(project["id"], "小鼠 1")
    second = store.create_animal(project["id"], "小鼠 2")
    assert second["name"] == "小鼠 2"


def test_error_envelope_and_status_codes(client):
    missing = client.get("/api/projects/does-not-exist")
    assert missing.status_code == 404
    body = missing.get_json()
    assert body["ok"] is False
    assert set(body["error"].keys()) == {"message", "fields"}

    invalid = client.post("/api/projects", json={"name": ""})
    assert invalid.status_code == 400
    fields = invalid.get_json()["error"]["fields"]
    assert fields and fields[0]["field"] == "name"

    not_object = client.post("/api/projects", json=[1, 2, 3])
    assert not_object.status_code == 400

    no_json = client.post("/api/projects", data="name=x",
                          content_type="application/x-www-form-urlencoded")
    assert no_json.status_code == 400


def test_settings_roundtrip_and_validation(client):
    style = client.get("/api/settings/style").get_json()["data"]["value"]
    assert style["color"] == "#B86245"

    invalid = client.put("/api/settings/style",
                         json={"value": {"color": "red", "dpi": 900}})
    assert invalid.status_code == 400
    fields = {item["field"] for item in invalid.get_json()["error"]["fields"]}
    assert {"style.color", "style.dpi"}.issubset(fields)

    accepted = client.put(
        "/api/settings/style",
        json={"value": {"color": "#112233", "line_width": 3, "dpi": 200,
                        "colormap": "magma"}},
    )
    assert accepted.status_code == 200
    stored = client.get("/api/settings/style").get_json()["data"]["value"]
    assert stored["color"] == "#112233"

    unknown_assay = client.put("/api/settings/defaults",
                               json={"value": {"nope": {"a": 1}}})
    assert unknown_assay.status_code == 400


def test_draft_creation_snapshots_saved_defaults(client):
    client.put("/api/settings/defaults", json={"value": {"tst": {"mode": "lis"}}})
    created = client.post("/api/drafts", json={"assay": "tst"})
    assert created.status_code == 201
    draft = created.get_json()["data"]["draft"]
    assert draft["options"]["mode"] == "lis"
    assert draft["output"] == {"annotated_video": True}

    client.put("/api/settings/defaults", json={"value": {"tst": {"mode": "classic"}}})
    reread = client.get(f"/api/drafts/{draft['id']}").get_json()["data"]["draft"]
    assert reread["options"]["mode"] == "lis"

    validate = client.get(f"/api/drafts/{draft['id']}/validate")
    assert validate.status_code == 200
    assert validate.get_json()["data"]["issues"] == []


def test_draft_with_video_and_unit_defaults(client):
    asset_id = new_id()
    created = client.post(
        "/api/drafts",
        json={
            "assay": "oft",
            "videos": [{"asset_id": asset_id, "units": [{}]}],
        },
    )
    assert created.status_code == 201
    draft = created.get_json()["data"]["draft"]
    video = draft["videos"][0]
    assert video["window"] == {"start_s": 0, "end_s": None}
    assert video["orientation"] == 0
    assert video["units"][0]["name"] == "小鼠 1"
    assert video["units"][0]["id"]

    replaced = client.put(
        f"/api/drafts/{draft['id']}", json={"options": {"a": 7}}
    )
    assert replaced.status_code == 200
    assert replaced.get_json()["data"]["draft"]["options"] == {"a": 7}


def test_draft_rejects_unknown_assay(client):
    response = client.post("/api/drafts", json={"assay": "unknown"})
    assert response.status_code == 400
    assert response.get_json()["error"]["fields"][0]["field"] == "assay"


def test_catalog_endpoint(client):
    payload = client.get("/api/catalog").get_json()
    assert payload["ok"] is True
    assays = payload["data"]["assays"]
    assert [item["key"] for item in assays] == [
        "tst", "fst", "mwm", "oft", "nor", "epm", "ymaze", "ldb", "barnes", "ezm"
    ]
    assert payload["data"]["style_defaults"]["colormap"] == "viridis"


def test_page_routes_render(client, store):
    project = store.create_project("页面项目")
    for path in ("/", "/projects", f"/projects/{project['id']}", "/analyze",
                 "/tasks", "/results/abc", "/settings", "/landmarks"):
        response = client.get(path)
        assert response.status_code == 200, path
        assert "text/html" in response.content_type

    assert client.get("/projects/missing").status_code == 404
    assert client.get("/api/nope").status_code == 404


def test_store_writes_strict_json(client, store):
    client.put("/api/settings/output", json={"value": {"annotated_video": False}})
    with store.read() as connection:
        raw = connection.execute(
            "SELECT value FROM settings WHERE key = 'settings.output'"
        ).fetchone()["value"]
    assert json.loads(raw) == {"annotated_video": False}
    assert "NaN" not in raw
