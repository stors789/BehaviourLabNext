"""第 21 块：数据目录语义、默认值快照、模板白名单与多动物隔离。"""

from __future__ import annotations

import json
import sqlite3
from copy import deepcopy
from pathlib import Path
from unittest import mock

import pytest

from behaviourlab import engine, field_templates
from behaviourlab.config import DataPaths
from behaviourlab.schema import ValidationError, new_id
from behaviourlab.store import Store, atomic_write_json
from behaviourlab.web.app import create_app

TOL = 1e-9
ARENA = {"type": "quad", "points": [[0, 0], [5, 0], [5, 5], [0, 5]]}


@pytest.fixture
def sandbox(tmp_path):
    """专用空 SQLite 与它自己的应用实例。"""

    paths = DataPaths.from_dir(tmp_path / "block21")
    store = Store(paths)
    app = create_app(data_dir=str(paths.root), store=store, testing=True)
    with app.test_client() as client:
        yield store, client
    store.close()


def project_of(client, name):
    response = client.post("/api/projects", json={"name": name})
    assert response.status_code == 201, response.get_json()
    return response.get_json()["data"]["project"]


# -- 自检 1：空库命名、跨项目绑定、删组与原子写 -------------------------

def test_batch_naming_cross_project_and_group_delete(sandbox):
    store, client = sandbox
    project = project_of(client, "实验甲")
    group = client.post(f"/api/projects/{project['id']}/groups",
                        json={"name": "对照"}).get_json()["data"]["groups"][0]

    created = client.post(
        f"/api/projects/{project['id']}/animals",
        json={"group_id": group["id"], "prefix": "小鼠", "start": 3, "count": 2},
    )
    assert created.status_code == 201, created.get_json()
    animals = created.get_json()["data"]["animals"]
    assert [item["name"] for item in animals] == ["小鼠3", "小鼠4"]
    assert {item["group_id"] for item in animals} == {group["id"]}

    # 跨项目组绑定失败，且整批不留半行。
    other = project_of(client, "实验乙")
    foreign = client.post(f"/api/projects/{other['id']}/groups",
                          json={"name": "外来组"}).get_json()["data"]["groups"][0]
    before = len(store.list_animals(project["id"]))
    rejected = client.post(
        f"/api/projects/{project['id']}/animals",
        json={"animals": [{"name": "小鼠5"},
                          {"name": "小鼠6", "group_id": foreign["id"]}]},
    )
    assert rejected.status_code == 400
    assert rejected.get_json()["error"]["fields"][0]["field"] == "group_id"
    assert len(store.list_animals(project["id"])) == before

    # 删除分组保留动物，只把 group_id 置空。
    store.delete_group(group["id"])
    remaining = store.list_animals(project["id"])
    assert [item["name"] for item in remaining] == ["小鼠3", "小鼠4"]
    assert all(item["group_id"] is None for item in remaining)
    assert store.list_groups(project["id"]) == []


def test_atomic_json_failure_keeps_previous_file(tmp_path):
    target = tmp_path / "state.json"
    atomic_write_json(target, {"ok": 1})
    before = target.read_text(encoding="utf-8")

    # 不可序列化的值在写盘前失败：旧文件原样保留。
    with pytest.raises(TypeError):
        atomic_write_json(target, {"bad": object()})
    assert target.read_text(encoding="utf-8") == before

    # Windows 共享冲突：有限重试后仍失败，抛出原异常且不留临时文件。
    with mock.patch("behaviourlab.store.os.replace",
                    side_effect=PermissionError("locked")):
        with pytest.raises(PermissionError):
            atomic_write_json(target, {"ok": 2})
    assert target.read_text(encoding="utf-8") == before
    assert not list(tmp_path.glob(".state.json.*.tmp"))


def test_old_database_is_migrated_without_rebuild(tmp_path):
    """旧库只补列：动物与模板记录都保留，payload 里的模板内容搬进新列。"""

    paths = DataPaths.from_dir(tmp_path / "legacy")
    paths.ensure()
    connection = sqlite3.connect(paths.database)
    try:
        connection.executescript(
            """
            CREATE TABLE projects (id TEXT PRIMARY KEY, name TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL);
            CREATE TABLE groups (id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                name TEXT NOT NULL, created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL);
            CREATE TABLE animals (id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                group_id TEXT REFERENCES groups(id) ON DELETE SET NULL,
                name TEXT NOT NULL, sex TEXT, birth_date TEXT, notes TEXT,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
            CREATE TABLE templates (id TEXT PRIMARY KEY, key TEXT NOT NULL,
                assay TEXT NOT NULL, name TEXT NOT NULL,
                payload TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL);
            """
        )
        now = "2026-01-01T00:00:00Z"
        connection.execute(
            "INSERT INTO projects VALUES('p1', '旧项目', '', ?, ?)", (now, now))
        connection.execute(
            "INSERT INTO animals VALUES('a1', 'p1', NULL, '旧鼠', NULL, NULL, NULL, ?, ?)",
            (now, now))
        connection.execute(
            "INSERT INTO templates VALUES('t1', 'oft', 'oft', '旧模板', ?, ?, ?)",
            (json.dumps({"geometry": {"arena": ARENA},
                         "options": {"arena_width_cm": 30}}), now, now))
        connection.commit()
    finally:
        connection.close()

    store = Store(paths)
    try:
        animal = store.get_animal("a1")
        assert animal["name"] == "旧鼠"
        assert animal["code"] is None

        template = store.get_template("t1")
        assert template["geometry"]["arena"]["points"] == ARENA["points"]
        assert template["options"] == {"arena_width_cm": 30}

        # 迁移是幂等的：再次打开不重复加列、不改记录。
        store.close()
        again = Store(paths)
        assert again.get_animal("a1")["name"] == "旧鼠"
        again.close()
    finally:
        store.close()


# -- 自检 2：默认值快照、逐视频/逐动物覆盖与隔离 -----------------------

def _asset(tmp_path, monkeypatch):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"placeholder")
    monkeypatch.setattr(
        "behaviourlab.engine.media.probe_video",
        lambda path: {"width": 100, "height": 100, "duration_s": 4.0,
                      "fps": 10.0, "frame_count": 40})
    return {"id": "a" * 32, "name": "clip.mp4", "path": str(source),
            "metadata": {"duration_s": 4.0, "fps": 10.0}}


def test_drafts_snapshot_settings_and_units_do_not_leak(sandbox, tmp_path,
                                                        monkeypatch):
    store, client = sandbox
    client.put("/api/settings/defaults",
               json={"value": {"oft": {"arena_width_cm": 33}}})
    client.put("/api/settings/style", json={"value": {"dpi": 125}})
    first = client.post("/api/drafts", json={"assay": "oft"}).get_json()["data"]["draft"]

    client.put("/api/settings/defaults",
               json={"value": {"oft": {"arena_width_cm": 99}}})
    client.put("/api/settings/style", json={"value": {"dpi": 150}})
    second = client.post("/api/drafts", json={"assay": "oft"}).get_json()["data"]["draft"]

    assert first["options"]["arena_width_cm"] == 33
    assert first["style"]["dpi"] == 125
    assert second["options"]["arena_width_cm"] == 99
    assert second["style"]["dpi"] == 150

    # 提交时使用草稿快照：再次改设置也不影响冻结值（submit 传 settings={}）。
    asset = _asset(tmp_path, monkeypatch)
    video = {
        "asset_id": asset["id"],
        "window": {"start_s": 0, "end_s": 4},
        "options": {"arena_width_cm": 40},
        "detection": {"gray_threshold": 110},
        "units": [
            {"id": "u1" * 16, "name": "小鼠 1",
             "options": {"arena_width_cm": 42}, "geometry": {"arena": ARENA}},
            {"id": "u2" * 16, "name": "小鼠 2", "options": {},
             "geometry": {"arena": deepcopy(ARENA)}},
        ],
    }
    draft = {"id": new_id(), "assay": "oft", "project_id": None,
             "videos": [video], "options": first["options"],
             "detection": first["detection"], "output": first["output"],
             "style": first["style"]}
    job = engine.build_resolved_job(
        draft=draft, video=video, asset=asset, job_id=new_id(),
        run_id=new_id(), settings={})

    assert job["style"]["dpi"] == 125
    # 视频层覆盖草稿快照（40 覆盖 33），动物层再覆盖视频层。
    assert job["options"]["arena_width_cm"] == 40
    assert [unit["options"]["arena_width_cm"] for unit in job["units"]] == [42, 40]
    assert [unit["detection"]["gray_threshold"] for unit in job["units"]] == [110, 110]

    # 冻结结果与草稿、视频、动物互不共享引用。
    job["units"][0]["geometry"]["arena"]["points"][0][0] = 99
    assert video["units"][0]["geometry"]["arena"]["points"][0][0] == 0
    assert video["units"][1]["geometry"]["arena"]["points"][0][0] == 0

    # 提交后再改设置：草稿仍按自己的快照解析。
    client.put("/api/settings/defaults",
               json={"value": {"oft": {"arena_width_cm": 120}}})
    reread = client.get(f"/api/drafts/{first['id']}").get_json()["data"]["draft"]
    assert reread["options"]["arena_width_cm"] == 33


# -- 自检 3：模板白名单、应用保留与样式默认 -----------------------------

def test_template_allow_list_and_apply(sandbox):
    store, client = sandbox
    unit = {
        "geometry": {
            "arena": deepcopy(ARENA),
            "mask": {"type": "polygon",
                     "points": [[1, 1], [2, 1], [2, 2]]},
            "objects": [{"id": "o1", "center": [1, 1],
                         "role": "familiar", "target": "x"}],
            "platform": {"type": "point", "points": [[2, 2]]},
        },
        "options": {"arena_width_cm": 33, "arena_height_cm": 44,
                    "bin_seconds": 7, "animal_id": "a" * 32,
                    "window": {"start_s": 4}, "manual": {},
                    "trial_type": "probe", "entry_depth_ratio": 0.3},
    }
    response = client.post("/api/templates",
                           json={"name": "OFT 方场", "assay": "oft",
                                 "unit": unit})
    assert response.status_code == 201, response.get_json()
    template = response.get_json()["data"]["template"]

    assert template["options"] == {"arena_width_cm": 33,
                                  "arena_height_cm": 44}
    assert "bin_seconds" not in template["options"]
    assert "animal_id" not in template["options"]
    assert "window" not in template["options"]
    assert template["geometry"]["arena"]["points"] == ARENA["points"]
    assert "mask" in template["geometry"]
    # 递归去掉角色与平台，洞心编号这类几何内容仍然保留。
    assert "role" not in template["geometry"]["objects"][0]
    assert "target" not in template["geometry"]["objects"][0]
    assert "platform" not in template["geometry"]

    current = {"geometry": {"arena": {"type": "quad",
                                      "points": [[9, 9], [9, 10], [10, 10], [10, 9]]},
                            "platform": {"type": "point", "points": [[5, 5]]}},
               "options": {"bin_seconds": 9, "animal_id": "b" * 32}}
    applied = field_templates.apply_template("oft", template, current)
    assert applied["options"]["bin_seconds"] == 9
    assert applied["options"]["animal_id"] == "b" * 32
    assert applied["options"]["arena_width_cm"] == 33
    assert applied["geometry"]["arena"]["points"] == ARENA["points"]
    # 平台属于当次试验：模板里没有，应用时保留当前值。
    assert applied["geometry"]["platform"]["points"] == [[5, 5]]

    # 深拷贝：改应用结果不会反向污染模板或原单位。
    applied["geometry"]["arena"]["points"][0][0] = 99
    assert template["geometry"]["arena"]["points"][0][0] == 0
    assert current["geometry"]["arena"]["points"][0][0] == 9

    listed = client.get("/api/templates?assay=oft").get_json()["data"]["templates"]
    assert [item["name"] for item in listed] == ["OFT 方场"]
    assert client.get("/api/templates?assay=ezm").get_json()["data"]["templates"] == []
    assert client.get(f"/api/templates/{template['id']}").status_code == 200
    assert client.delete(f"/api/templates/{template['id']}").status_code == 200
    assert client.get(f"/api/templates/{template['id']}").status_code == 404


def test_style_defaults_and_preview_do_not_touch_formal_style(sandbox):
    store, client = sandbox
    style = client.get("/api/settings/style").get_json()["data"]
    assert style["key"] == "style"
    assert style["value"]["color"] == "#B86245"
    assert style["value"]["colormap"] == "viridis"
    assert style["value"]["dpi"] == 150
    assert style["value"]["color"] != "#087b80"

    response = client.post("/api/settings/style/preview",
                           json={"value": {"color": "#B86245",
                                           "colormap": "magma",
                                           "line_width": 3, "dpi": 600}})
    assert response.status_code == 200, response.get_json()
    data = response.get_json()["data"]
    assert client.get(data["image_url"]).status_code == 200
    # 预览把 DPI 上限用在图上，但不把上限写回正式样式。
    assert store.get_setting("settings.style", default=None) is None

    saved = client.put("/api/settings/style",
                       json={"value": {"color": "#112233", "dpi": 200}})
    assert saved.status_code == 200
    value = saved.get_json()["data"]["value"]
    assert value["color"] == "#112233"
    assert value["dpi"] == 200
    assert value["colormap"] == "viridis", "保存时补全 factory 字段"

    invalid = client.post("/api/settings/style/preview",
                          json={"value": {"colormap": "not-a-map"}})
    assert invalid.status_code == 400


def test_settings_value_rules(sandbox):
    from behaviourlab import experiments

    store, client = sandbox
    empty = client.get("/api/settings/defaults").get_json()["data"]
    assert empty["key"] == "defaults"
    # 没有保存值时给的是工厂值，不是空对象。
    assert set(empty["value"]) == {entry["key"] for entry in experiments.catalog()}
    assert empty["value"]["oft"] == experiments.get("oft").DEFAULTS
    detection_empty = client.get("/api/settings/detection").get_json()["data"]
    assert detection_empty["value"]["oft"]["gray_threshold"] == 127

    unknown = client.get("/api/settings/nope")
    assert unknown.status_code == 404
    not_object = client.put("/api/settings/defaults", json={"value": [1, 2]})
    assert not_object.status_code == 400
    assert not_object.get_json()["error"]["fields"][0]["field"] == "value"
    # detection 也按实验键保存，完整命名空间与 options 分开。
    client.put("/api/settings/detection",
               json={"value": {"oft": {"quality_threshold": 0.7}}})
    client.put("/api/settings/defaults",
               json={"value": {"oft": {"quality_threshold": 0.2}}})
    saved_detection = client.get("/api/settings/detection").get_json()["data"][
        "value"]["oft"]
    assert saved_detection["quality_threshold"] == 0.7
    assert saved_detection["gray_threshold"] == 127, "保存一个字段不清掉其他工厂值"
    saved_defaults = client.get("/api/settings/defaults").get_json()["data"][
        "value"]["oft"]
    assert saved_defaults["quality_threshold"] == 0.2
    assert saved_defaults["arena_width_cm"] == experiments.get(
        "oft").DEFAULTS["arena_width_cm"]


def test_draft_defaults_to_oft_and_accepts_partial_fields(sandbox):
    store, client = sandbox
    created = client.post("/api/drafts", json={})
    assert created.status_code == 201
    draft = created.get_json()["data"]["draft"]
    assert draft["assay"] == "oft"
    assert draft["output"] == {"annotated_video": True}
    assert draft["style"]["color"] == "#B86245"
    assert draft["options"]["arena_width_cm"] == 50.0

    titled = client.post("/api/drafts",
                         json={"assay": "tst", "options": {"mode": "lis"}})
    assert titled.get_json()["data"]["draft"]["options"]["mode"] == "lis"


def test_mutate_draft_reads_and_writes_in_one_transaction(sandbox):
    store, client = sandbox
    draft = store.create_draft("oft", options={"a": 1})

    def add_unit(record):
        videos = record["videos"] or []
        videos.append({"asset_id": "a" * 32, "units": []})
        return {"videos": videos}

    updated = store.mutate_draft(draft["id"], add_unit)
    assert len(updated["videos"]) == 1
    assert updated["options"] == {"a": 1}, "patch 之外的顶层字段保留"
    with pytest.raises(ValidationError):
        store.mutate_draft(draft["id"], lambda record: {"id": "x"})
    store.close()
    assert store.closed is True
    store.close()
    with pytest.raises(Exception):
        store.connect()


def test_storage_failure_log_falls_back_when_logs_unwritable(tmp_path):
    """主日志目录不可写时回退到临时目录，并返回可见的实际路径。"""

    from behaviourlab.store import log_storage_failure, storage_log_path

    root = tmp_path / "root-with-file-logs"
    root.mkdir()
    (root / "logs").write_text("not a directory", encoding="utf-8")
    paths = DataPaths.from_dir(root)

    target = storage_log_path(paths)
    assert target.name == "storage.log"
    assert paths.logs not in target.parents
    target.unlink(missing_ok=True)

    database = root / "behaviourlab.sqlite3"
    written = log_storage_failure(paths, "transaction",
                                  sqlite3.OperationalError("disk I/O error"),
                                  operation="insert draft",
                                  database=str(database))
    assert written == target
    text = written.read_text(encoding="utf-8")
    assert "phase=transaction" in text
    assert "operation=insert draft" in text
    assert str(database) in text
    assert "disk I/O error" in text


def test_health_reports_storage_log_path(sandbox):
    store, client = sandbox
    data = client.get("/api/health").get_json()["data"]
    assert data["storage_log"].endswith("storage.log")
    assert Path(data["data_root"]) == store.paths.root


def test_storage_failure_log_falls_back_when_logs_unwritable(tmp_path):
    """主日志目录不可写时回退到临时目录，并返回可见的实际路径。"""

    from behaviourlab.store import log_storage_failure, storage_log_path

    root = tmp_path / "root-with-file-logs"
    root.mkdir()
    (root / "logs").write_text("not a directory", encoding="utf-8")
    paths = DataPaths.from_dir(root)

    target = storage_log_path(paths)
    assert target.name == "storage.log"
    assert paths.logs not in target.parents
    target.unlink(missing_ok=True)

    database = root / "behaviourlab.sqlite3"
    written = log_storage_failure(paths, "transaction",
                                  sqlite3.OperationalError("disk I/O error"),
                                  operation="insert draft",
                                  database=str(database))
    assert written == target
    text = written.read_text(encoding="utf-8")
    assert "phase=transaction" in text
    assert "operation=insert draft" in text
    assert str(database) in text
    assert "disk I/O error" in text


def test_health_reports_storage_log_path(sandbox):
    store, client = sandbox
    data = client.get("/api/health").get_json()["data"]
    assert data["storage_log"].endswith("storage.log")
    assert Path(data["data_root"]) == store.paths.root


def test_nor_template_restores_roles_by_object_order():
    template = {
        "geometry": {"objects": [{"id": "o1", "center": [1, 1]},
                                 {"id": "o2", "center": [9, 9]}]},
        "options": {},
    }
    current = {"geometry": {"objects": [{"role": "novel"}, {"role": "familiar"}]},
               "options": {}}
    applied = field_templates.apply_template("nor", template, current)
    assert [item["role"] for item in applied["geometry"]["objects"]] == [
        "novel", "familiar"], "沿用当前单位同序号的角色"

    fresh = field_templates.apply_template("nor", template,
                                           {"geometry": {}, "options": {}})
    assert [item["role"] for item in fresh["geometry"]["objects"]] == [
        "familiar", "novel"], "没有当前角色时第一熟悉、第二新奇"
