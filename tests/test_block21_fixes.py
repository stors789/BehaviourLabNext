"""第 21 块五组合并修正的定点自检。

只覆盖确认过的反例：默认值解析与设置页接线、Store 失败分支、项目空绑定、
模板白名单、项目页上传的 Content-Type。
"""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from behaviourlab import experiments
from behaviourlab.config import DataPaths
from behaviourlab.engine import resolve_job
from behaviourlab.schema import ValidationError
from behaviourlab.store import Store, atomic_write_json, log_storage_failure

ROOT = Path(__file__).resolve().parents[1]
OPTIONS = {"quality_threshold": 0.4}


@pytest.fixture
def sandbox(tmp_path):
    """专用空 SQLite 与它自己的应用实例（与第 21 块自检同一形状）。"""

    from behaviourlab.web.app import create_app

    paths = DataPaths.from_dir(tmp_path / "block21-fixes")
    store = Store(paths)
    app = create_app(data_dir=str(paths.root), store=store, testing=True)
    with app.test_client() as client:
        yield store, client
    store.close()


# -- 一、默认值解析与设置页接线 ------------------------------------------


def test_resolve_job_takes_current_assay_detection_object():
    """合同形状：settings.detection 已是当前实验的检测对象。"""

    resolved = resolve_job("oft", settings={
        "detection": {"gray_threshold": 110, "quality_threshold": 0.7}})
    assert resolved["detection"]["gray_threshold"] == 110
    assert resolved["detection"]["quality_threshold"] == 0.7
    # options.quality_threshold 与 detection.quality_threshold 各管一段。
    # OFT 的 options 没有 quality_threshold：两者是不同命名空间。
    assert "quality_threshold" not in resolved["options"]

    # 嵌套实验键不再被当作参数：空对象等价于没有保存值。
    empty = resolve_job("oft", settings={"detection": {}})
    assert empty["detection"]["gray_threshold"] == 127


def test_draft_snapshot_uses_same_detection_object(sandbox):
    """服务取值与解析入口一致：保存的检测覆盖进入新草稿。"""

    store, client = sandbox
    saved = client.put("/api/settings/detection",
                       json={"value": {"oft": {"gray_threshold": 110}}})
    assert saved.status_code == 200, saved.get_json()
    draft = client.post("/api/drafts", json={"assay": "oft"}).get_json()["data"][
        "draft"]
    assert draft["detection"]["gray_threshold"] == 110
    # 提交不重读新默认：快照在创建时已经固定。
    client.put("/api/settings/detection",
               json={"value": {"oft": {"gray_threshold": 90}}})
    again = client.get(f"/api/drafts/{draft['id']}").get_json()["data"]["draft"]
    assert again["detection"]["gray_threshold"] == 110


def test_settings_api_returns_factory_when_unset(sandbox):
    store, client = sandbox
    data = client.get("/api/settings/defaults").get_json()["data"]
    assert data["key"] == "defaults"
    assert data["value"]["tst"] == experiments.get("tst").DEFAULTS
    detection = client.get("/api/settings/detection").get_json()["data"]
    assert detection["key"] == "detection"
    assert detection["value"]["oft"]["gray_threshold"] == 127


def _number_inputs(html: str) -> list[dict]:
    items = []
    for match in re.finditer(r"<input[^>]*type=\"number\"[^>]*>", html):
        tag = match.group(0)
        attrs = dict(re.findall(r'([a-z-]+)="([^"]*)"', tag))
        items.append(attrs)
    return items


def test_settings_page_number_fields_are_natively_valid(sandbox):
    """默认值必须落在 min/step 网格上，否则整份表单被原生校验拦住。"""

    store, client = sandbox
    html = client.get("/settings").get_data(as_text=True)
    checked = 0
    for attrs in _number_inputs(html):
        value = attrs.get("value")
        if value in (None, ""):
            continue
        number = float(value)
        minimum = float(attrs["min"]) if attrs.get("min") else None
        maximum = float(attrs["max"]) if attrs.get("max") else None
        step = attrs.get("step", "any")
        if minimum is not None:
            assert number >= minimum, (attrs, "低于 min")
        if maximum is not None:
            assert number <= maximum, (attrs, "高于 max")
        if step != "any":
            base = minimum if minimum is not None else 0.0
            steps = (number - base) / float(step)
            assert abs(steps - round(steps)) < 1e-9, (
                attrs, "默认值不在 min/step 网格上")
        checked += 1
    assert checked > 10, "设置页应有可校验的数值控件"


def test_settings_page_detection_text_field_carries_assay(sandbox):
    store, client = sandbox
    html = client.get("/settings").get_data(as_text=True)
    forms = html.split('data-settings-section="detection"', 1)[1]
    block = forms.split("</form>", 1)[0]
    text_inputs = re.findall(r"<input[^>]*type=\"text\"[^>]*>", block)
    assert text_inputs, "检测分区应有文本字段"
    for tag in text_inputs:
        assert "data-assay=" in tag, tag


@pytest.mark.parametrize("field", ["dpi", "line_width", "color", "colormap"])
def test_style_rejects_explicit_null(sandbox, field):
    store, client = sandbox
    response = client.put("/api/settings/style", json={"value": {field: None}})
    assert response.status_code == 400, response.get_json()
    assert response.get_json()["error"]["fields"][0]["field"] == f"style.{field}"
    assert client.get("/api/settings/style").get_json()["data"]["value"][field] \
        is not None


def test_style_accepts_600_and_preview_clamps(sandbox):
    store, client = sandbox
    saved = client.put("/api/settings/style", json={"value": {"dpi": 600}})
    assert saved.status_code == 200, saved.get_json()
    value = client.get("/api/settings/style").get_json()["data"]["value"]
    assert value["dpi"] == 600
    assert value["color"] == "#B86245"
    preview = client.post("/api/settings/style/preview",
                          json={"value": {"dpi": 600}})
    assert preview.status_code == 200, preview.get_json()
    assert preview.get_json()["data"]["style"]["dpi"] == 240
    # 预览上限不回写正式设置。
    assert client.get("/api/settings/style").get_json()["data"]["value"][
        "dpi"] == 600


# -- 二、Store 失败分支 --------------------------------------------------


def _store_with_fast_busy(paths: DataPaths, monkeypatch) -> Store:
    """先建好两个 Store（建库需要写锁），再把第二条的 busy_timeout 缩到 1ms。"""

    Store(paths)
    store = Store(paths)

    def connect(*, shared=False):
        connection = sqlite3.connect(store.db_path, timeout=0,
                                     isolation_level=None,
                                     check_same_thread=not shared)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 1")
        return connection

    monkeypatch.setattr(store, "connect", connect)
    return store


def test_transaction_keeps_sqlite_busy_error(tmp_path, monkeypatch):
    paths = DataPaths.from_dir(tmp_path / "busy")
    other = _store_with_fast_busy(paths, monkeypatch)
    blocker = sqlite3.connect(paths.database, timeout=0, isolation_level=None)
    blocker.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(sqlite3.OperationalError) as captured:
            other.create_project("写不进去")
        message = str(captured.value).lower()
        assert "locked" in message or "busy" in message
        assert "no transaction" not in message
        log = paths.logs / "storage.log"
        assert log.is_file() and log.stat().st_size > 0
        text = log.read_text(encoding="utf-8")
        assert "phase=transaction" in text
        assert "SQLITE_BUSY" in text or "locked" in text.lower()
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()


def test_connect_closes_connection_when_pragma_fails(tmp_path, monkeypatch):
    paths = DataPaths.from_dir(tmp_path / "pragma")
    Store(paths)
    state = {"closed": False}

    class Proxy:
        def __init__(self, real):
            self._real = real
            self.row_factory = None
            self.fail_on = 0
            self.calls = 0

        def execute(self, sql, *args):
            self.calls += 1
            if self.calls == 2:
                raise sqlite3.OperationalError("busy_timeout 注入失败")
            return self._real.execute(sql, *args)

        def close(self):
            state["closed"] = True
            self._real.close()

    original_connect = sqlite3.connect

    def fake_connect(*args, **kwargs):
        return Proxy(original_connect(*args, **kwargs))

    monkeypatch.setattr(sqlite3, "connect", fake_connect)
    with pytest.raises(sqlite3.OperationalError):
        Store(paths)
    assert state["closed"] is True, "配置失败后连接必须关闭"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_atomic_write_rejects_non_finite_and_keeps_old_file(tmp_path, bad):
    target = tmp_path / "state.json"
    atomic_write_json(target, {"ok": 1})
    before = target.read_text(encoding="utf-8")
    with pytest.raises(ValidationError):
        atomic_write_json(target, {"bad": bad})
    assert target.read_text(encoding="utf-8") == before
    assert not list(tmp_path.glob(".state.json.*.tmp"))
    # 合法 null 仍然是 null。
    atomic_write_json(target, {"bad": None})
    assert json.loads(target.read_text(encoding="utf-8")) == {"bad": None}


def test_log_write_failure_uses_fallback(tmp_path, monkeypatch):
    from behaviourlab import store as store_module

    paths = DataPaths.from_dir(tmp_path / "logs")
    paths.ensure()
    primary = paths.logs / "storage.log"
    fallback = tmp_path / "fallback" / "storage.log"
    monkeypatch.setattr(store_module, "_storage_log_candidates",
                        lambda paths: (primary, fallback))

    real_open = Path.open

    class BadHandle:
        def __init__(self, handle):
            self._handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self._handle.close()
            return False

        def write(self, text):
            raise PermissionError("写入被拒绝")

        def flush(self):
            pass

        def fileno(self):
            return self._handle.fileno()

    def fake_open(self, *args, **kwargs):
        handle = real_open(self, *args, **kwargs)
        if self == primary:
            return BadHandle(handle)
        return handle

    monkeypatch.setattr(Path, "open", fake_open)
    written = log_storage_failure(paths, "transaction",
                                  sqlite3.OperationalError("disk I/O error"),
                                  operation="insert", database=str(paths.database))
    assert written == fallback
    assert fallback.is_file()
    text = fallback.read_text(encoding="utf-8")
    assert "phase=transaction" in text
    assert str(paths.database) in text
    assert "disk I/O error" in text


# -- 三、项目模式必须绑定动物 --------------------------------------------


def _project_draft(client, store, tmp_path, *, project_mode: bool,
                   animal_id=None):
    project = store.create_project("绑定检查", "")
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"placeholder")
    asset = store.create_asset(name="clip.mp4", path=str(source),
                               project_id=project["id"] if project_mode else None,
                               metadata={"width": 200, "height": 200,
                                         "duration_s": 1.0, "fps": 5.0,
                                         "frame_count": 5})
    draft = store.create_draft("oft", project_id=project["id"] if project_mode else None,
                               videos=[{
        "asset_id": asset["id"],
        "window": {"start_s": 0.0, "end_s": 1.0},
        "orientation": 0, "fps_override": None, "sample_fps": None,
        "units": [{
            "id": "u" * 32, "name": "小鼠 1", "animal_id": animal_id,
            "window": None,
            "geometry": {"arena": {"type": "quad", "points": [
                [10, 10], [110, 10], [110, 110], [10, 110]]}},
            "options": dict(experiments.get("oft").DEFAULTS),
            "detection": {}, "manual": None,
        }],
    }])
    return project, draft


def test_project_submit_requires_bound_animal(sandbox, tmp_path, monkeypatch):
    store, client = sandbox
    from behaviourlab import engine

    monkeypatch.setattr(
        engine.media, "probe_video",
        lambda path: {"width": 200, "height": 200, "duration_s": 1.0,
                      "fps": 5.0, "frame_count": 5})
    project, draft = _project_draft(client, store, tmp_path,
                                    project_mode=True, animal_id=None)

    issues = client.get(f"/api/drafts/{draft['id']}/validate").get_json()[
        "data"]["issues"]
    binding = [item for item in issues if item["field"].endswith("animal_id")]
    assert binding, issues
    assert binding[0]["unit_id"] == "u" * 32
    assert binding[0]["asset_id"] == draft["videos"][0]["asset_id"]
    assert "animal_id" in binding[0]

    submitted = client.post("/api/jobs", json={"draft_id": draft["id"]})
    assert submitted.status_code == 400, submitted.get_json()
    fields = submitted.get_json()["error"]["fields"]
    assert any(item["field"].endswith("animal_id") for item in fields)
    # 没有半个批次。
    assert client.get("/api/jobs").get_json()["data"]["batches"] == []
    # 保存草稿不受影响：可以先存，再回项目补动物。
    saved = client.put(f"/api/drafts/{draft['id']}", json={"options": {"bin_seconds": 60}})
    assert saved.status_code == 200, saved.get_json()

    # 绑定本项目动物后可以提交（执行结果不影响接口是否放行）。
    animal = store.create_animal(project["id"], "小鼠 1")
    videos = deepcopy(draft["videos"])
    videos[0]["units"][0]["animal_id"] = animal["id"]
    store.update_draft(draft["id"], {"videos": videos})
    accepted = client.post("/api/jobs", json={"draft_id": draft["id"]})
    assert accepted.status_code == 202, accepted.get_json()


def test_independent_submit_still_allows_no_animal(sandbox, tmp_path, monkeypatch):
    store, client = sandbox
    from behaviourlab import engine

    monkeypatch.setattr(
        engine.media, "probe_video",
        lambda path: {"width": 200, "height": 200, "duration_s": 1.0,
                      "fps": 5.0, "frame_count": 5})
    project, draft = _project_draft(client, store, tmp_path,
                                    project_mode=False, animal_id=None)
    assert draft["project_id"] is None
    issues = client.get(f"/api/drafts/{draft['id']}/validate").get_json()[
        "data"]["issues"]
    assert not [item for item in issues if item["field"].endswith("animal_id")]
    accepted = client.post("/api/jobs", json={"draft_id": draft["id"]})
    assert accepted.status_code == 202, accepted.get_json()
    assert project is not None


def test_deleted_animal_binding_is_still_rejected(sandbox, tmp_path):
    store, client = sandbox
    project, draft = _project_draft(client, store, tmp_path, project_mode=True)
    animal = store.create_animal(project["id"], "会被删的小鼠")
    videos = deepcopy(draft["videos"])
    videos[0]["units"][0]["animal_id"] = animal["id"]
    store.update_draft(draft["id"], {"videos": videos})
    store.delete_animal(animal["id"])
    issues = client.get(f"/api/drafts/{draft['id']}/validate").get_json()[
        "data"]["issues"]
    assert any(item["field"].endswith("animal_id") for item in issues)


# -- 四、模板 options 精确白名单 -----------------------------------------


def test_template_options_are_exact_whitelist(sandbox):
    store, client = sandbox
    response = client.post("/api/templates", json={
        "name": "绕道检查", "assay": "tst",
        "unit": {
            "geometry": {"arena": {"type": "quad",
                                   "points": [[0, 0], [5, 0], [5, 5], [0, 5]]}},
            "options": {
                "arena": {"animal_id": "A", "trial": "probe"},
                "body_line": {"manual": [1, 2]},
                "bin_seconds": 7,
                "behavior_threshold_sensitivity": 0.5,
            },
        },
    })
    assert response.status_code == 201, response.get_json()
    template = response.get_json()["data"]["template"]
    assert template["options"] == {}, template

    oft = client.post("/api/templates", json={
        "name": "OFT 尺寸", "assay": "oft",
        "unit": {
            "geometry": {"arena": {"type": "quad",
                                   "points": [[0, 0], [33, 0], [33, 44], [0, 44]]}},
            "options": {"arena": {"animal_id": "A"}, "arena_width_cm": 33,
                        "arena_height_cm": 44, "bin_seconds": 7},
        },
    }).get_json()["data"]["template"]
    assert oft["options"] == {"arena_width_cm": 33, "arena_height_cm": 44}

    epm = client.post("/api/templates", json={
        "name": "EPM 尺寸", "assay": "epm",
        "unit": {
            "geometry": {"arena": {"type": "quad",
                                   "points": [[0, 0], [60, 0], [60, 60], [0, 60]]}},
            "options": {"zones": {"manual": [1]}, "center_size_cm": 5,
                        "open_arm_length_cm": 30, "closed_arm_length_cm": 30},
        },
    }).get_json()["data"]["template"]
    assert epm["options"] == {"center_size_cm": 5, "open_arm_length_cm": 30,
                              "closed_arm_length_cm": 30}


def test_template_apply_does_not_merge_legacy_options():
    from behaviourlab import field_templates

    template = {"geometry": {"arena": {"type": "quad",
                                       "points": [[0, 0], [5, 0], [5, 5]]}},
                "options": {"arena_width_cm": 33, "arena_height_cm": 44}}
    unit = {"geometry": {}, "options": {"arena": {"animal_id": "A"},
                                        "bin_seconds": 9,
                                        "animal_id": "a" * 32}}
    applied = field_templates.apply_template("oft", template, unit)
    assert applied["options"] == {"bin_seconds": 9, "animal_id": "a" * 32,
                                  "arena_width_cm": 33, "arena_height_cm": 44}


# -- 五、上传 Content-Type 与共享脚本 ------------------------------------


def test_upload_requires_multipart_boundary(sandbox, tmp_path):
    store, client = sandbox
    project = store.create_project("上传检查", "")
    boundary = "----bln-boundary"
    body = "\r\n".join([
        f"--{boundary}",
        'Content-Disposition: form-data; name="project_id"', "", project["id"],
        f"--{boundary}",
        'Content-Disposition: form-data; name="files"; filename="clip.mp4"',
        "Content-Type: video/mp4", "", "not-a-real-video",
        f"--{boundary}--", "",
    ]).encode("utf-8")

    wrong = client.post("/api/assets/upload", data=body,
                        headers={"Content-Type": "application/json"})
    assert wrong.status_code == 400
    assert "缺少上传文件" in wrong.get_json()["error"]["message"]

    right = client.post(
        "/api/assets/upload", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    assert right.status_code in (200, 201), right.get_json()
    payload = right.get_json()["data"]
    # 文件被解析到了：失败原因是媒体不可解码，而不是“缺少上传文件”。
    assert payload["assets"] == []
    assert payload["errors"], payload
    assert "clip.mp4" in str(payload["errors"])
    assert "缺少上传文件" not in str(payload["errors"])


def _run_node(script: str, expect: str) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node 不可用")
    result = subprocess.run([node, str(Path(__file__).parent / "js" / script)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert expect in result.stdout


def test_settings_async_callbacks_belong_to_their_snapshot():
    _run_node("test_settings_async.cjs", "settings-async OK")


def test_public_request_wrapper_distinguishes_form_data():
    _run_node("test_app_request.cjs", "app-request OK")
