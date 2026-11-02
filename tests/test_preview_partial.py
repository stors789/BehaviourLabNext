"""未知时长时的预览范围与总时长：只能如实标为部分范围。

受控解码器说明：帧来自内存合成器与受控 PTS，不是真实录像；浏览器渲染
与真实素材不在本文件覆盖范围。
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from behaviourlab import detection

ROOT = Path(__file__).resolve().parents[1]
UNKNOWN = {"width": 200, "height": 200, "fps": None, "frame_count": None}


def preview_payload(client, asset, draft, monkeypatch, **kwargs):
    from test_media_preview_defaults import behavior_preview

    return behavior_preview(client, asset, draft, monkeypatch, **kwargs)


def test_unknown_duration_is_marked_partial(client, store, tmp_path, monkeypatch):
    from test_media_preview_defaults import seed_draft

    asset, draft = seed_draft(store, tmp_path, metadata=dict(UNKNOWN))
    before = client.get(f"/api/drafts/{draft['id']}").get_json()["data"]["draft"]
    response, record = preview_payload(client, asset, draft, monkeypatch)
    assert response.status_code == 201, response.get_json()
    data = response.get_json()["data"]

    scope = data["preview_scope"]
    # 只扫到当前帧：明确是部分范围，不是完整配置范围。
    assert scope["mode"] == "partial"
    assert scope["partial"] is True
    assert scope["unknown_end"] is True
    assert scope["start_s"] == pytest.approx(0.0)
    assert scope["end_s"] == pytest.approx(14.96)
    assert scope["duration_s"] == pytest.approx(14.96)

    behavior = data["summary"]["behavior"]
    assert behavior["end_s"] == pytest.approx(14.96), "不暗示 relative 20 秒已覆盖"
    assert behavior["window_end_s"] == pytest.approx(14.96)
    assert behavior["total_time_s"] == pytest.approx(14.96)
    assert any("部分范围" in text and "完整配置范围" in text
               for text in data["warnings"]), data["warnings"]
    assert record[0]["end_s"] == pytest.approx(14.96)

    # 预览不写回草稿：视频窗、单位窗与几何逐值不变，也不产生任务/冻结件。
    after = client.get(f"/api/drafts/{draft['id']}").get_json()["data"]["draft"]
    assert after["videos"] == before["videos"]
    assert after["detection"] == before["detection"]
    assert store.list_batches() == []


def test_explicit_video_end_still_configured(client, store, tmp_path, monkeypatch):
    from test_media_preview_defaults import seed_draft

    asset, draft = seed_draft(store, tmp_path, metadata=dict(UNKNOWN),
                              window={"start_s": 0, "end_s": 18})
    response, record = preview_payload(client, asset, draft, monkeypatch)
    assert response.status_code == 201, response.get_json()
    data = response.get_json()["data"]
    assert data["preview_scope"]["mode"] == "configured"
    assert "partial" not in data["preview_scope"]
    assert data["preview_scope"]["end_s"] == pytest.approx(18.0)
    assert not any("部分范围" in text for text in data["warnings"])
    assert record[0]["end_s"] == pytest.approx(18.0)


def test_known_frame_count_stays_configured(client, store, tmp_path, monkeypatch):
    from test_media_preview_defaults import seed_draft

    asset, draft = seed_draft(
        store, tmp_path,
        metadata={"width": 200, "height": 200, "fps": 25.0,
                  "frame_count": 250})
    response, _ = preview_payload(client, asset, draft, monkeypatch)
    assert response.status_code == 201, response.get_json()
    data = response.get_json()["data"]
    assert data["preview_scope"]["mode"] == "configured"
    assert data["preview_scope"]["end_s"] == pytest.approx(20.0)
    assert not any("部分范围" in text for text in data["warnings"])


def test_detection_preview_keeps_single_frame_scope(client, store, tmp_path,
                                                    monkeypatch):
    from behaviourlab import preview
    from behaviourlab.media import FrameInfo
    from test_media_preview_defaults import seed_draft

    asset, draft = seed_draft(store, tmp_path, metadata=dict(UNKNOWN))
    monkeypatch.setattr(
        preview.media, "read_frame_info",
        lambda *args, **kwargs: FrameInfo(
            image=__import__("numpy").zeros((200, 200, 3), dtype="uint8"),
            index=187, time_s=14.96, pts=None, time_base=None,
            clock_source="fps_override"))
    response = client.post(
        f"/api/drafts/{draft['id']}/preview",
        json={"asset_id": asset["id"], "unit_id": "a" * 32,
              "frame_index": 187, "kind": "detection"})
    assert response.status_code == 201, response.get_json()
    scope = response.get_json()["data"]["preview_scope"]
    assert scope["mode"] == "frame"
    assert scope["start_s"] == scope["end_s"] == pytest.approx(14.96)


def test_duration_facts_preserved():
    # 已知帧数推导不回退；缺帧数才用原时长×原帧率；事实不足仍是 None。
    assert detection.interpreted_duration(
        {"frame_count": 250, "fps": 25.0, "duration_s": None},
        {"fps_override": 12.5}) == 20.0
    assert detection.interpreted_duration(
        {"frame_count": 250, "fps": 25.0, "duration_s": 10.0},
        {"fps_override": 12.5}) == 20.0
    assert detection.interpreted_duration(
        {"fps": 25.0, "duration_s": 10.0}, {"fps_override": 12.5}) == 20.0
    assert detection.interpreted_duration(
        {"fps": 25.0, "duration_s": 10.0}, {"fps_override": None}) == 10.0
    assert detection.interpreted_duration(
        {"fps": None, "duration_s": None}, {"fps_override": 12.5}) is None


def test_workspace_duration_display_is_honest():
    source = (ROOT / "behaviourlab/web/static/workspace.js").read_text(
        encoding="utf-8")
    duration = (source.split("function videoDuration()", 1)[1]
                .split("\n  }", 1)[0])
    # 总时长只取可证实值：不再用(lastFrame+1)/fps 或名义帧率补片尾。
    assert "return knownVideoDuration();" in duration
    assert "lastFrame" not in duration
    label = (source.split("function durationText(digits)", 1)[1]
             .split("\n  }", 1)[0])
    assert '"总时长未知"' in label
    assert "timeText(total, digits)" in label
    frame = (source.split("function frameLabel(info)", 1)[1]
             .split("\n  }", 1)[0])
    assert "durationText(1)" in frame
    # 范围面板两处同样用诚实文案。
    assert source.count("durationText(1)") >= 3
    # 新 exact 终点仍只取可证实时长；未知时留空。
    field = source.split("function animalWindowField(unit, options)", 1)[1]
    field = field.split("\n  }", 1)[0]
    assert "const duration = knownVideoDuration();" in field
    # 预览状态行区分部分范围与完整配置。
    assert "部分范围预览（完整终点未知）" in source


def test_partial_scope_note_text():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node 不可用")
    script = Path(__file__).parent / "js" / "test_preview_partial.cjs"
    result = subprocess.run([node, str(script)], capture_output=True, text=True,
                            timeout=60, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    assert "preview-partial OK" in result.stdout
