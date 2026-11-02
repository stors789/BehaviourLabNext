"""U03 新指定窗默认终点 + R06 有帧数时的有效时长。

前端规则由测试驱动真实的 editor-core 函数（`effectiveVideoDuration`、
`windowFieldValue`、`assignWindow`），随后把结果写进真实草稿再读回；
预览侧走生产 create_preview → 实验 preview_analysis → VideoSource 链路。
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from behaviourlab import detection
from behaviourlab.experiments import oft

ROOT = Path(__file__).resolve().parents[1]


# -- U03：默认终点 -----------------------------------------------------
@pytest.fixture(scope="module")
def window_rules():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node 不可用")
    script = Path(__file__).parent / "js" / "test_window_default.cjs"
    result = subprocess.run([node, str(script)], capture_output=True,
                            text=True, timeout=60, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_default_endpoint_rules(window_rules):
    assert window_rules["fresh"] == {"mode": "exact", "start_s": 1, "end_s": 6}
    assert window_rules["kept"]["end_s"] == 5.5, "已保存的终点不被改写"
    assert window_rules["typed"]["end_s"] == 7.25, "用户当前输入优先"
    assert "终点" in window_rules["blankError"], "无有效时长且未填终点必须报错"


def test_frontend_matches_backend_duration_rule(window_rules):
    """同一表：前端有效时长与 detection.interpreted_duration 逐项相同。"""

    cases = window_rules["durationCases"]
    assert len(cases) >= 8
    for case in cases:
        expected = detection.interpreted_duration(case["meta"], case["video"])
        assert case["value"] == expected, case
    # 有帧数但没有原 duration 的固定例。
    assert detection.interpreted_duration(
        {"frame_count": 120, "fps": None, "duration_s": None},
        {"fps_override": 20}) == 6.0


def test_workspace_uses_shared_effective_duration():
    source = (ROOT / "behaviourlab/web/static/workspace.js").read_text(
        encoding="utf-8")
    helper = (source.split("function knownVideoDuration()", 1)[1]
              .split("\n  }", 1)[0])
    assert "BLCore.effectiveVideoDuration" in helper
    # 新窗默认终点取可证实的有效时长，而不是旧的 videoDuration 估计。
    field = source.split("function animalWindowField(unit, options)", 1)[1]
    field = field.split("\n  }", 1)[0]
    assert "const duration = knownVideoDuration();" in field
    assert "videoDuration()" not in field
    assert 'numberInput("end_s", duration, 0)' in field
    # 模式 change 仍走同一提交入口。
    assert 'select.addEventListener("change", () => {' in field
    assert "commit();" in field
    assert "window.BLCore.assignWindow(null, select.value, values)" in field

    display = (source.split("function videoDuration()", 1)[1]
               .split("\n  }", 1)[0])
    assert "knownVideoDuration()" in display, "总时长显示与默认终点同一规则"


def test_new_window_persists_to_draft(client, store, tmp_path, window_rules):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"placeholder")
    asset = store.create_asset(
        name="clip.mp4", path=str(source), project_id=None,
        metadata={"width": 640, "height": 480, "fps": 30.0,
                  "frame_count": 120, "duration_s": 4.0})
    draft = store.create_draft("oft", videos=[{
        "asset_id": asset["id"],
        "window": {"start_s": 0, "end_s": None},
        "orientation": 0, "fps_override": 20, "sample_fps": None,
        "units": [
            {"id": "a" * 32, "name": "新窗", "window": None},
            {"id": "b" * 32, "name": "已有窗",
             "window": {"mode": "exact", "start_s": 1, "end_s": 5.5}},
            {"id": "c" * 32, "name": "手填窗", "window": None},
        ],
    }])
    videos = draft["videos"]
    videos[0]["units"][0]["window"] = window_rules["fresh"]
    videos[0]["units"][2]["window"] = window_rules["typed"]

    response = client.put(f"/api/drafts/{draft['id']}", json={"videos": videos})
    assert response.status_code == 200, response.get_json()

    stored = client.get(f"/api/drafts/{draft['id']}").get_json()["data"]["draft"]
    units = {unit["id"]: unit for unit in stored["videos"][0]["units"]}
    assert units["a" * 32]["window"] == {"mode": "exact", "start_s": 1, "end_s": 6}
    assert units["b" * 32]["window"]["end_s"] == 5.5
    assert units["c" * 32]["window"]["end_s"] == 7.25
    # 无覆盖时同一素材的有效时长仍是原 4 秒（默认值只影响新窗）。
    assert detection.interpreted_duration(
        asset["metadata"], {"fps_override": None}) == 4.0


# -- R06：有帧数就不依赖原 duration -----------------------------------
@pytest.mark.parametrize("metadata,video,expected", [
    ({"frame_count": 250, "fps": 25.0}, {"fps_override": 12.5}, 20.0),
    ({"frame_count": 250, "fps": 25.0, "duration_s": None},
     {"fps_override": 12.5}, 20.0),
    ({"frame_count": 250, "fps": 25.0, "duration_s": 10.0},
     {"fps_override": 12.5}, 20.0),
    ({"fps": 25.0, "duration_s": 10.0}, {"fps_override": 12.5}, 20.0),
    ({"duration_s": 10.0}, {"fps_override": 12.5}, None),
    ({"frame_count": 250, "fps": 25.0}, {"fps_override": None}, None),
    ({"frame_count": 250, "fps": 25.0, "duration_s": 10.0},
     {"fps_override": None}, 10.0),
    ({"frame_count": 0, "fps": None, "duration_s": None},
     {"fps_override": 12.5}, None),
])
def test_interpreted_duration_uses_frame_count_first(metadata, video, expected):
    assert detection.interpreted_duration(metadata, video) == expected


def _preview_by_behavior(client, asset, draft, monkeypatch, **kwargs):
    from test_media_preview_defaults import behavior_preview

    return behavior_preview(client, asset, draft, monkeypatch, **kwargs)


def test_preview_endpoint_without_container_duration(client, store, tmp_path,
                                                     monkeypatch):
    from test_media_preview_defaults import seed_draft

    asset, draft = seed_draft(
        store, tmp_path,
        metadata={"width": 200, "height": 200, "fps": 25.0,
                  "frame_count": 250})
    response, record = _preview_by_behavior(client, asset, draft, monkeypatch)
    assert response.status_code == 201, response.get_json()
    data = response.get_json()["data"]
    # 250 帧 ÷ 12.5 = 20 秒；没有原 duration 也能给出完整终点。
    assert data["preview_scope"]["end_s"] == pytest.approx(20.0)
    assert data["preview_scope"]["duration_s"] == pytest.approx(20.0)
    behavior = data["summary"]["behavior"]
    assert behavior["end_s"] == pytest.approx(20.0)
    assert behavior["window_end_s"] == pytest.approx(20.0)
    assert record[0]["end_s"] == pytest.approx(20.0)
    assert record[0]["orientation"] == 90
    assert record[0]["sample_fps"] == 2


def test_preview_explicit_end_still_wins(client, store, tmp_path, monkeypatch):
    from test_media_preview_defaults import seed_draft

    asset, draft = seed_draft(
        store, tmp_path, window={"start_s": 0, "end_s": 18},
        metadata={"width": 200, "height": 200, "fps": 25.0,
                  "frame_count": 250})
    response, record = _preview_by_behavior(client, asset, draft, monkeypatch)
    assert response.status_code == 201, response.get_json()
    data = response.get_json()["data"]
    assert data["preview_scope"]["end_s"] == pytest.approx(18.0)
    assert record[0]["end_s"] == pytest.approx(18.0)


def test_preview_does_not_invent_endpoint_without_facts(client, store,
                                                        tmp_path, monkeypatch):
    from test_media_preview_defaults import seed_draft

    asset, draft = seed_draft(
        store, tmp_path,
        metadata={"width": 200, "height": 200, "fps": None,
                  "frame_count": None})
    response, record = _preview_by_behavior(client, asset, draft, monkeypatch)
    assert response.status_code == 201, response.get_json()
    data = response.get_json()["data"]
    # 事实不足：仍用既有“到当前帧为止”的扫描上界，不得扩到 20 秒。
    assert data["preview_scope"]["end_s"] == pytest.approx(14.96)
    assert record[0]["end_s"] == pytest.approx(14.96)
    assert detection.interpreted_duration(
        {"width": 200, "height": 200, "fps": None, "frame_count": None},
        {"fps_override": 12.5}) is None
