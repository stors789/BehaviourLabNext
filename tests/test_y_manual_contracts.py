"""第 16 块第 1 组：统一人工判定与合并后的最终校验。

固定输入按本轮说明：严格人工只看 unit.manual.mode；纯人工豁免未消费的
检测配置；混合按每个自动单位的合并配置校验；绑定跨项目的动物必须拒绝。
"""

from __future__ import annotations

from behaviourlab import engine
from behaviourlab.schema import new_id

UNIT = "u" * 32
AUTO = "a" * 32

GEOMETRY = {"zones": {
    "center": {"type": "polygon", "points": [[90, 90], [110, 90], [100, 110]]},
    "A": {"type": "quad", "points": [[90, 10], [110, 10], [110, 85], [90, 85]]},
    "B": {"type": "quad", "points": [[115, 95], [185, 95], [185, 115], [115, 115]]},
    "C": {"type": "quad", "points": [[15, 95], [85, 95], [85, 115], [15, 115]]},
}}


def manual_unit(entries=None, *, options=None, detection=None, window=None):
    return {
        "id": UNIT, "name": "人工鼠", "animal_id": None,
        "window": window,
        "geometry": {},
        "options": dict(options or {}),
        "detection": dict(detection or {}),
        "manual": {"mode": "manual",
                   "entries": [{"time_s": 0.2, "arm": "A"},
                               {"time_s": 0.4, "arm": "A"}]
                   if entries is None else entries},
    }


def auto_unit(*, geometry=None, detection=None):
    return {
        "id": AUTO, "name": "自动鼠", "animal_id": None, "window": None,
        "geometry": dict(GEOMETRY if geometry is None else geometry),
        "options": {}, "detection": dict(detection or {}), "manual": None,
    }


def video_of(*units, window=None, detection=None):
    return {
        "asset_id": new_id(),
        "window": window or {"start_s": 0.0, "end_s": 10.0},
        "orientation": 0, "fps_override": None, "sample_fps": None,
        "options": {}, "detection": dict(detection or {}),
        "units": list(units),
    }


def fields_of(issues):
    return [issue["field"] for issue in issues]


# -- 1. options.mode 不是人工判定 --------------------------------------

def test_options_mode_manual_without_records_is_automatic(client, store):
    unit = manual_unit()
    unit["manual"] = None
    unit["options"] = {"mode": "manual"}
    draft = store.create_draft("ymaze", videos=[video_of(unit)])

    assert engine.is_manual_path("ymaze", unit) is False
    checked = client.post(f"/api/drafts/{draft['id']}/validate")
    fields = fields_of(checked.get_json()["data"]["issues"])
    assert "videos[0].units[0].geometry.zones" in fields

    before = len(store.list_jobs())
    rejected = client.post("/api/jobs", json={"draft_id": draft["id"]})
    assert rejected.status_code == 400, rejected.get_json()
    assert "videos[0].units[0].geometry.zones" in fields_of(
        rejected.get_json()["error"]["fields"])
    assert len(store.list_jobs()) == before, "校验失败不能留下作业"


def test_manual_mode_auto_keeps_auto_window_rules(client, store):
    unit = auto_unit()
    unit["manual"] = {"mode": "auto"}
    unit["window"] = {"mode": "relative", "offset_s": 0, "duration_s": 60}
    draft = store.create_draft("ymaze", videos=[video_of(unit)])

    assert engine.is_manual_path("ymaze", unit) is False
    checked = client.post(f"/api/drafts/{draft['id']}/validate")
    issues = checked.get_json()["data"]["issues"]
    assert fields_of(issues) == []
    assert all("人工 Y 只支持" not in issue["message"] for issue in issues)


# -- 2. 合并后的 bin_seconds -------------------------------------------

def test_manual_bin_seconds_uses_merged_value(client, store):
    unit = manual_unit(options={})
    draft = store.create_draft("ymaze", options={"bin_seconds": 0},
                               videos=[video_of(unit)])

    checked = client.post(f"/api/drafts/{draft['id']}/validate")
    fields = fields_of(checked.get_json()["data"]["issues"])
    assert "videos[0].units[0].options.bin_seconds" in fields

    before = len(store.list_jobs())
    rejected = client.post("/api/jobs", json={"draft_id": draft["id"]})
    assert rejected.status_code == 400
    assert "videos[0].units[0].options.bin_seconds" in fields_of(
        rejected.get_json()["error"]["fields"])
    assert len(store.list_jobs()) == before

    # 合法的继承值继续有效，不需要在单位层重复写一遍。
    store.update_draft(draft["id"], {"options": {"bin_seconds": 1.0}})
    fixed = client.post(f"/api/drafts/{draft['id']}/validate")
    assert fixed.get_json()["data"]["issues"] == []


# -- 3. 纯人工各层一致豁免 / 混合按合并配置 ------------------------------

def test_pure_manual_ignores_detection_at_every_layer(client, store):
    bad = {"bg_type": "invalid", "gray_threshold": -2}
    draft = store.create_draft(
        "ymaze", detection=bad,
        videos=[video_of(manual_unit(detection=bad), detection=bad)])

    checked = client.post(f"/api/drafts/{draft['id']}/validate")
    assert checked.get_json()["data"]["issues"] == []


def test_mixed_draft_validates_each_automatic_unit(client, store):
    # 草稿层的无效值被自动单位逐项覆盖 → 合并后合法，不报错。
    covered = store.create_draft(
        "ymaze", detection={"bg_type": "invalid", "gray_threshold": -2},
        videos=[video_of(manual_unit(),
                         auto_unit(detection={"bg_type": "light",
                                              "gray_threshold": 40}))])
    assert client.post(
        f"/api/drafts/{covered['id']}/validate").get_json()["data"]["issues"] == []

    # 自动单位的静态背景缺失 → 创建作业前就拒绝，并定位到该单位字段。
    broken = store.create_draft(
        "ymaze",
        videos=[video_of(manual_unit(),
                         auto_unit(detection={
                             "bg_method": "static",
                             "static_background": None}))])
    checked = client.post(f"/api/drafts/{broken['id']}/validate")
    fields = fields_of(checked.get_json()["data"]["issues"])
    assert "videos[0].units[1].detection.static_background" in fields

    before = len(store.list_jobs())
    rejected = client.post("/api/jobs", json={"draft_id": broken["id"]})
    assert rejected.status_code == 400
    assert len(store.list_jobs()) == before


# -- 4. 项目动物绑定 ---------------------------------------------------

def test_cross_project_animal_binding_is_rejected(client, store):
    home = store.create_project("项目一")
    other = store.create_project("项目二")
    outsider = store.create_animal(other["id"], "别组的鼠")
    insider = store.create_animal(home["id"], "本组的鼠")

    bad = manual_unit()
    bad["animal_id"] = outsider["id"]
    draft = store.create_draft("ymaze", project_id=home["id"],
                               videos=[video_of(bad)])

    checked = client.post(f"/api/drafts/{draft['id']}/validate")
    issues = checked.get_json()["data"]["issues"]
    target = [issue for issue in issues
              if issue["field"] == "videos[0].units[0].animal_id"]
    assert target and target[0]["message"] == "动物不属于当前项目"

    before = len(store.list_jobs())
    rejected = client.post("/api/jobs", json={"draft_id": draft["id"]})
    assert rejected.status_code == 400
    assert len(store.list_jobs()) == before, "跨项目绑定不能创建作业"

    # 同项目绑定与独立分析都正常。
    bad["animal_id"] = insider["id"]
    store.update_draft(draft["id"], {"videos": [video_of(bad)]})
    assert client.post(
        f"/api/drafts/{draft['id']}/validate").get_json()["data"]["issues"] == []

    standalone = manual_unit()
    standalone["animal_id"] = outsider["id"]
    loose = store.create_draft("ymaze", videos=[video_of(standalone)])
    assert client.post(
        f"/api/drafts/{loose['id']}/validate").get_json()["data"]["issues"] == []
