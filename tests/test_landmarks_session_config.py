"""关键点会话有效配置继承：真实页面选项 → 原生产 JS 创建处理器 →
POST /api/landmarks → 读取保存会话。不只直调辅助函数或手造 payload。

固定反例：OFT 草稿，视频 orientation=90、fps_override=12.5；
draft.detection={bg_method:"temporal",temporal_diff_threshold:33}；
video.detection={gray_threshold:111}；unit.detection={min_area:200}。
固定预期：会话 orientation=90、fps_override=12.5；有效检测仍为
temporal/33/111/200，不退为 0°/null/threshold/25/127。
"""

from __future__ import annotations

import json
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest

from behaviourlab import landmarks
from behaviourlab.engine import resolve_detection

NODE_SCRIPT = Path(__file__).parent / "js" / "test_landmarks_config.cjs"

DRAFT_DETECTION = {"bg_method": "temporal", "temporal_diff_threshold": 33}
VIDEO_DETECTION = {"gray_threshold": 111}
UNIT_DETECTION = {"min_area": 200}
EFFECTIVE = {"bg_method": "temporal", "temporal_diff_threshold": 33,
             "gray_threshold": 111, "min_area": 200}
GEOMETRY_90 = {"zones": {"arena": {"type": "quad",
                                   "points": [[20, 10], [80, 10],
                                              [80, 70], [20, 70]]}}}


class _OptionCollector(HTMLParser):
    """从真实渲染页面收集 <option> 的 data-* 属性。"""

    def __init__(self):
        super().__init__()
        self.options: list[dict] = []
        self._current: dict | None = None
        self._in_media = False
        self.media_raw: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "option":
            self._current = dict(attrs)
        elif tag == "script":
            values = dict(attrs)
            if values.get("id") == "landmark-media":
                self._in_media = True

    def handle_endtag(self, tag):
        if tag == "option" and self._current is not None:
            self.options.append(self._current)
            self._current = None
        elif tag == "script":
            self._in_media = False

    def handle_data(self, data):
        if self._in_media:
            self.media_raw.append(data)


def _camel(name: str) -> str:
    parts = name.split("-")
    return parts[0] + "".join(item.title() for item in parts[1:])


def _page_options(client) -> tuple[list[dict], dict]:
    """渲染 /landmarks 并解析真实 option dataset 与 media JSON。"""

    page = client.get("/landmarks")
    assert page.status_code == 200
    parser = _OptionCollector()
    parser.feed(page.get_data(as_text=True))
    assert parser.options, "页面应枚举可选的视频与动物"
    media = json.loads("".join(parser.media_raw) or "null")
    rows = []
    for attributes in parser.options:
        dataset = {}
        for key, value in attributes.items():
            if key.startswith("data-"):
                dataset[_camel(key[5:])] = value
        dataset["value"] = attributes.get("value", "")
        rows.append(dataset)
    return rows, media


def _payload_via_real_js(dataset: dict) -> dict:
    """把真实页面选项喂给真实 landmarks.js 的 createSession 处理器。"""

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node 不可用")
    result = subprocess.run(
        [node, str(NODE_SCRIPT), "extract"],
        input=json.dumps(dataset), capture_output=True, text=True,
        timeout=30, encoding="utf-8",
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _make_draft(store, asset_id: str, *, orientation: int,
                fps_override, unit_detection: dict,
                draft_detection: dict | None = None,
                video_detection: dict | None = None,
                unit_id: str = "u" * 32, unit_name: str = "小鼠 1",
                geometry: dict | None = None) -> dict:
    video = {
        "asset_id": asset_id,
        "orientation": orientation,
        "fps_override": fps_override,
        "sample_fps": None,
        "detection": dict(video_detection or {}),
        "units": [{"id": unit_id, "name": unit_name, "animal_id": None,
                   "window": None,
                   "geometry": GEOMETRY_90 if geometry is None else geometry,
                   "options": {"arena_width_cm": 50.0,
                               "arena_height_cm": 50.0},
                   "detection": unit_detection}],
    }
    return store.create_draft(
        "oft", videos=[video],
        detection=dict(DRAFT_DETECTION if draft_detection is None
                       else draft_detection))


def test_fixed_counterexample_full_chain(client, store, tmp_path):
    """固定反例：方向/FPS/有效检测全部进入会话并被 _settings_of 读出。"""

    source = tmp_path / "source.mp4"
    source.write_bytes(b"\x00")
    asset = store.create_asset(name="oft.mp4", path=str(source))
    draft = _make_draft(store, asset["id"], orientation=90,
                        fps_override=12.5, unit_detection=UNIT_DETECTION,
                        video_detection=VIDEO_DETECTION)

    rows, media = _page_options(client)
    unit_id = draft["videos"][0]["units"][0]["id"]
    dataset = next(item for item in rows if item["value"] == unit_id)
    # option dataset 与同一渲染的 media JSON 同源一致；完整有效检测是
    # 默认全集上的三层覆盖合并，覆盖键等于固定预期。
    entry = next(unit for item in media for unit in item["units"]
                 if unit["id"] == unit_id)
    assert entry["orientation"] == 90
    assert entry["fps_override"] == 12.5
    effective = entry["effective_detection"]
    for key, value in EFFECTIVE.items():
        assert effective[key] == value, key

    payload = _payload_via_real_js(dataset)
    assert payload["orientation"] == 90
    assert payload["fps_override"] == 12.5
    for key, value in EFFECTIVE.items():
        assert payload["detection"][key] == value, key
    assert payload["unit"]["detection"] == UNIT_DETECTION
    assert payload["unit"]["geometry"] == GEOMETRY_90, \
        "几何保持该 90° 方向下保存的源像素坐标，不被再次旋转"

    response = client.post("/api/landmarks", json=payload)
    assert response.status_code in (200, 201), response.get_json()
    session = landmarks.get_session(store.paths,
                                    response.get_json()["data"]["session"]["id"])
    assert session["orientation"] == 90
    assert session["fps_override"] == 12.5
    for key, value in EFFECTIVE.items():
        assert session["detection"][key] == value, key
    assert session["unit"]["detection"] == UNIT_DETECTION
    assert session["unit"]["geometry"] == GEOMETRY_90
    # _settings_of 实际读出值 = 完整有效检测语义；不退回工厂默认。
    settings = landmarks._settings_of(session)
    assert settings["bg_method"] == "temporal"
    assert settings["temporal_diff_threshold"] == 33
    assert settings["gray_threshold"] == 111
    assert settings["min_area"] == 200
    # 生产解析与引擎规则一致（不含今天的全局设置层）。
    assert settings == resolve_detection(
        "oft", draft={"detection": DRAFT_DETECTION},
        video={"detection": VIDEO_DETECTION},
        unit={"detection": UNIT_DETECTION})


def test_override_priority_and_units_do_not_swap(client, store, tmp_path):
    """同一键三层同时出现：单位覆盖视频、视频覆盖草稿；只草稿有的键
    保留；两只单位不同配置各自会话不串。"""

    source = tmp_path / "source.mp4"
    source.write_bytes(b"\x00")
    asset = store.create_asset(name="oft.mp4", path=str(source))
    draft = store.create_draft("oft", videos=[{
        "asset_id": asset["id"], "orientation": 90, "fps_override": 12.5,
        "sample_fps": None, "detection": {"gray_threshold": 111},
        "units": [
            {"id": "u" * 32, "name": "甲", "animal_id": None,
             "window": None, "geometry": GEOMETRY_90, "options": {},
             "detection": {"min_area": 200, "gray_threshold": 200}},
            {"id": "f" * 32, "name": "乙", "animal_id": None,
             "window": None, "geometry": GEOMETRY_90, "options": {},
             "detection": {"min_area": 300}},
        ],
    }], detection={"bg_method": "temporal", "temporal_diff_threshold": 33,
                   "gray_threshold": 100})

    rows, _ = _page_options(client)
    payloads = {}
    for unit_id in ("u" * 32, "f" * 32):
        dataset = next(item for item in rows if item["value"] == unit_id)
        payload = _payload_via_real_js(dataset)
        payloads[unit_id] = payload
    # 单位层 gray_threshold=200 覆盖视频层 111；乙没有单位层该键，
    # 得到视频层 111（视频覆盖草稿 100）；min_area 各自保留。
    assert payloads["u" * 32]["detection"]["gray_threshold"] == 200
    assert payloads["u" * 32]["detection"]["min_area"] == 200
    assert payloads["f" * 32]["detection"]["gray_threshold"] == 111
    assert payloads["f" * 32]["detection"]["min_area"] == 300
    for unit_id in ("u" * 32, "f" * 32):
        assert payloads[unit_id]["detection"]["bg_method"] == "temporal"
        assert payloads[unit_id]["detection"][
            "temporal_diff_threshold"] == 33

    sessions = {}
    for unit_id, payload in payloads.items():
        response = client.post("/api/landmarks", json=payload)
        assert response.status_code in (200, 201)
        session_id = response.get_json()["data"]["session"]["id"]
        sessions[unit_id] = landmarks.get_session(store.paths, session_id)
    first = landmarks._settings_of(sessions["u" * 32])
    second = landmarks._settings_of(sessions["f" * 32])
    assert first["gray_threshold"] == 200 and first["min_area"] == 200
    assert second["gray_threshold"] == 111 and second["min_area"] == 300
    assert first["bg_method"] == "temporal", "只草稿有的键保留"


def test_frozen_isolation_after_draft_and_settings_change(
        client, store, tmp_path):
    """创建后修改草稿与全局默认：原会话完整配置/几何/records/evaluation
    不变；新建会话使用新选择；旧会话仍可继续标注。"""

    source = tmp_path / "source.mp4"
    source.write_bytes(b"\x00")
    asset = store.create_asset(name="oft.mp4", path=str(source))
    draft = _make_draft(store, asset["id"], orientation=90,
                        fps_override=12.5, unit_detection=UNIT_DETECTION,
                        video_detection=VIDEO_DETECTION)

    rows, _ = _page_options(client)
    unit_id = draft["videos"][0]["units"][0]["id"]
    payload = _payload_via_real_js(
        next(item for item in rows if item["value"] == unit_id))
    response = client.post("/api/landmarks", json=payload)
    assert response.status_code in (200, 201)
    session_id = response.get_json()["data"]["session"]["id"]
    before = landmarks.get_session(store.paths, session_id)
    assert landmarks._settings_of(before) == resolve_detection(
        "oft", draft={"detection": DRAFT_DETECTION},
        video={"detection": VIDEO_DETECTION},
        unit={"detection": UNIT_DETECTION})

    # 之后修改草稿与全局检测默认。
    store.update_draft(draft["id"], {
        "detection": {"bg_method": "threshold", "gray_threshold": 77}})
    client.put("/api/settings/detection", json={
        "assay": "oft", "values": {"min_area": 999}})

    after = landmarks.get_session(store.paths, session_id)
    assert after == before, "已创建会话不被任何后续修改回写"
    assert landmarks._settings_of(after)["bg_method"] == "temporal"
    assert landmarks._settings_of(after)["min_area"] == 200

    # 再标注一次仍成功：旧会话不被自动重写。
    saved = client.put(f"/api/landmarks/{session_id}", json={
        "records": [{"frame": 0, "points": {
            "body": {"visible": True, "x": 40.0, "y": 30.0}}}]})
    assert saved.status_code in (200, 201), saved.get_json()
    reopened = landmarks.get_session(store.paths, session_id)
    for key, value in EFFECTIVE.items():
        assert reopened["detection"][key] == value, key
    assert reopened["records"][0]["points"]["body"]["visible"] is True

    # 新建会话使用新的草稿快照：bg_method 来自更新后的草稿层；
    # 视频层 gray_threshold=111 仍在覆盖链中（覆盖语义一致），
    # 单位层 min_area=200 保留。
    rows2, _ = _page_options(client)
    payload2 = _payload_via_real_js(
        next(item for item in rows2 if item["value"] == unit_id))
    response2 = client.post("/api/landmarks", json=payload2)
    assert response2.status_code in (200, 201)
    fresh = landmarks.get_session(
        store.paths, response2.get_json()["data"]["session"]["id"])
    assert fresh["detection"]["bg_method"] == "threshold"
    assert fresh["detection"]["gray_threshold"] == 111
    assert landmarks._settings_of(fresh)["min_area"] == 200
    # 草稿层被整体替换后不再含 temporal_diff_threshold=33：新会话的
    # 该键回落到工厂默认 25——新建确实使用新快照，而非旧草稿漂移。
    assert fresh["detection"]["temporal_diff_threshold"] == 25.0


def test_default_zero_null_still_works(client, store, tmp_path):
    """0° 与 fps_override=null 的默认例正常；缺省时不写死。"""

    source = tmp_path / "source.mp4"
    source.write_bytes(b"\x00")
    asset = store.create_asset(name="oft.mp4", path=str(source))
    draft = _make_draft(store, asset["id"], orientation=0,
                        fps_override=None, unit_detection={},
                        draft_detection={}, video_detection={})

    rows, _ = _page_options(client)
    unit_id = draft["videos"][0]["units"][0]["id"]
    payload = _payload_via_real_js(
        next(item for item in rows if item["value"] == unit_id))
    assert payload["orientation"] == 0
    assert payload["fps_override"] is None

    response = client.post("/api/landmarks", json=payload)
    assert response.status_code in (200, 201)
    session = landmarks.get_session(store.paths,
                                    response.get_json()["data"]["session"]["id"])
    assert session["orientation"] == 0
    assert session["fps_override"] is None
    settings = landmarks._settings_of(session)
    assert settings["bg_method"] == "threshold"
    assert settings["gray_threshold"] == 127.0
    assert settings["temporal_diff_threshold"] == 25.0
    assert settings["min_area"] == 120.0


@pytest.mark.parametrize("payload_patch,field", [
    ({"orientation": 45}, "orientation"),
    ({"fps_override": 0}, "fps_override"),
    ({"fps_override": -3}, "fps_override"),
    ({"tolerance_px": 0}, "tolerance_px"),
    ({"frames": [0, 0]}, None),
    ({"frames": [-1]}, None),
    ({"detection": "not-an-object"}, "detection"),
])
def test_invalid_inputs_are_rejected_not_silently_defaulted(
        client, store, tmp_path, payload_patch, field):
    """非法方向/FPS/容差/frames/检测类型明确报错，不静默回退默认。"""

    source = tmp_path / "source.mp4"
    source.write_bytes(b"\x00")
    asset = store.create_asset(name="oft.mp4", path=str(source))
    _make_draft(store, asset["id"], orientation=90, fps_override=12.5,
                unit_detection=UNIT_DETECTION)

    rows, _ = _page_options(client)
    dataset = next(item for item in rows
                   if item["value"] == "u" * 32)
    payload = _payload_via_real_js(dataset)
    payload.update(payload_patch)
    response = client.post("/api/landmarks", json=payload)
    assert response.status_code >= 400, response.get_json()
    if field is not None:
        body = response.get_json()
        fields = [item.get("field")
                  for item in (body.get("error", {}).get("fields") or [])]
        assert field in fields or body["error"].get("message"), body


def test_source_identity_change_still_rejected(client, store, tmp_path):
    """源身份校验保持：文件变化后验证仍拒绝，不为配置继承放宽。"""

    source = tmp_path / "source.mp4"
    source.write_bytes(b"\x00")
    asset = store.create_asset(name="oft.mp4", path=str(source))
    _make_draft(store, asset["id"], orientation=90, fps_override=12.5,
                unit_detection=UNIT_DETECTION)
    rows, _ = _page_options(client)
    payload = _payload_via_real_js(
        next(item for item in rows if item["value"] == "u" * 32))
    response = client.post("/api/landmarks", json=payload)
    assert response.status_code in (200, 201)
    session_id = response.get_json()["data"]["session"]["id"]

    source.write_bytes(b"\x00changed")
    records = [{"frame": 0, "points": {
        "body": {"visible": True, "x": 40.0, "y": 30.0}}}]
    saved = client.put(f"/api/landmarks/{session_id}", json={
        "records": records})
    assert saved.status_code in (200, 201)
    evaluated = client.post(f"/api/landmarks/{session_id}/evaluate", json={})
    assert evaluated.status_code >= 400, "源身份变化后验证仍拒绝"


def test_session_orientation_and_fps_reach_engine(
        client, store, tmp_path, monkeypatch):
    """最小方向例：90°/12.5 会话的原图请求与评估使用旋转后源坐标——
    VideoSource 收到会话冻结的方向与解释FPS；标注点按同一旋转后
    画布校验（超出即拒绝），不重复转换。"""

    import numpy as np

    source = tmp_path / "source.mp4"
    source.write_bytes(b"\x00")
    asset = store.create_asset(name="oft.mp4", path=str(source))
    _make_draft(store, asset["id"], orientation=90, fps_override=12.5,
                unit_detection=UNIT_DETECTION,
                video_detection=VIDEO_DETECTION)
    rows, _ = _page_options(client)
    payload = _payload_via_real_js(
        next(item for item in rows if item["value"] == "u" * 32))
    response = client.post("/api/landmarks", json=payload)
    assert response.status_code in (200, 201)
    session_id = response.get_json()["data"]["session"]["id"]

    captured: dict = {}

    class FakeSource:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

        def iter_frames(self):
            # 真实 VideoSource 按 orientation 旋转后产出：60x40 源
            # 旋转 90° 后为 60 高 × 40 宽。
            for index in range(4):
                yield type("F", (), {
                    "image": np.zeros((60, 40, 3), dtype=np.uint8),
                    "index": index, "time_s": index * 0.1,
                })()

    class FakeDetector:
        def foreground(self, image):
            return object()

    fake_result = type("R", (), {"center": (30.0, 20.0),
                                 "nose": (30.0, 20.0),
                                 "tail_base": None})()
    monkeypatch.setattr(landmarks.media, "VideoSource", FakeSource)
    monkeypatch.setattr(landmarks.detection, "geometry_region",
                        lambda *args, **kwargs: (None, None))
    monkeypatch.setattr(landmarks.detection, "prepare_background",
                        lambda *args, **kwargs: object())
    monkeypatch.setattr(landmarks.detection, "shared_detector",
                        lambda *args, **kwargs: FakeDetector())
    monkeypatch.setattr(landmarks.observation, "signature_of",
                        lambda *args, **kwargs: object())
    monkeypatch.setattr(landmarks.observation, "track",
                        lambda *args, **kwargs: fake_result)

    session = landmarks.get_session(store.paths, session_id)
    evaluation = landmarks.evaluate(store.paths, session)
    assert evaluation is not None
    # 会话冻结的方向与解释FPS原样进入引擎：原图按 90° 旋转后显示，
    # 标注点与几何使用同一旋转后源坐标，不在评估时二次转换。
    assert captured["kwargs"]["orientation"] == 90
    assert captured["kwargs"]["fps_override"] == 12.5
    # 越界标注按旋转后画面拒绝（旋转后画布 40 宽 × 60 高，
    # 点 x=55 超出宽度）。
    bad = dict(session, records=[{"frame": 0, "points": {
        "body": {"visible": True, "x": 55.0, "y": 30.0}}}], frames=[0])
    try:
        landmarks.evaluate(store.paths, bad)
        raised = None
    except Exception as error:  # noqa: BLE001 - 具体类型由实现决定
        raised = error
    assert raised is not None and "超出旋转后画面" in str(raised)
