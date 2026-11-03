"""中心点入臂规则：设置保存→冻结→分析→结果/报告 的真实链路一致性。

接口级走 client（生产 PUT/GET/preview/analyze），渲染层用源码接线断言
（parameterField 的规则选择、说明与不生效标注）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from test_ymaze import PreviewSource, geometry

ROOT = Path(__file__).resolve().parents[1]
UNIT_A = "e" * 32
UNIT_B = "f" * 32


def videos_payload(asset_id: str) -> list[dict]:
    return [{
        "asset_id": asset_id,
        "window": {"start_s": 0.0, "end_s": 4.0},
        "orientation": 0, "fps_override": None, "sample_fps": 10,
        "units": [
            {"id": UNIT_A, "name": "中心点鼠", "window": None,
             "geometry": geometry(),
             "options": {"arm_entry_rule": "center"}, "detection": {},
             "manual": None},
            {"id": UNIT_B, "name": "原规则鼠", "window": None,
             "geometry": geometry(), "options": {}, "detection": {},
             "manual": None},
        ],
    }]


@pytest.fixture
def ymaze_env(client, store, tmp_path, monkeypatch):
    source = tmp_path / "y.mp4"
    source.write_bytes(b"placeholder")
    asset = store.create_asset(
        name="y.mp4", path=str(source), project_id=None,
        metadata={"width": 240, "height": 240, "duration_s": 4.0,
                  "fps": 10.0, "frame_count": 40})
    draft = store.create_draft("ymaze", videos=videos_payload(asset["id"]))
    monkeypatch.setattr("behaviourlab.media.VideoSource", PreviewSource)
    monkeypatch.setattr(
        "behaviourlab.engine.media.probe_video",
        lambda path: {"width": 240, "height": 240, "duration_s": 4.0,
                      "fps": 10.0, "frame_count": 40})
    from behaviourlab.media import FrameInfo

    monkeypatch.setattr(
        "behaviourlab.preview.media.read_frame_at_time",
        lambda *args, **kwargs: FrameInfo(
            image=np.full((240, 240, 3), 255, dtype=np.uint8), index=15,
            time_s=1.5, pts=None, time_base=None, clock_source="pts"))
    return draft, asset


def test_rule_persists_through_save_reload(ymaze_env, client):
    draft, _asset = ymaze_env
    # 工作台保存路径：PUT 全量 videos（含单鼠 options）→ 重载逐值一致。
    stored = client.get(f"/api/drafts/{draft['id']}").get_json()["data"]["draft"]
    response = client.put(f"/api/drafts/{draft['id']}",
                          json={"videos": stored["videos"]})
    assert response.status_code == 200, response.get_json()
    reloaded = client.get(f"/api/drafts/{draft['id']}").get_json()["data"]["draft"]
    units = reloaded["videos"][0]["units"]
    assert units[0]["options"]["arm_entry_rule"] == "center"
    assert "arm_entry_rule" not in units[1]["options"], "原规则鼠未被写入新字段"
    # 同一视频不同小鼠各自携带规则，冻结解析不串用。
    from behaviourlab.engine import resolve_options
    resolved = [resolve_options("ymaze", draft=reloaded,
                                video=reloaded["videos"][0], unit=u)
                for u in units]
    assert resolved[0]["arm_entry_rule"] == "center"
    assert resolved[1]["arm_entry_rule"] == "depth"


def test_preview_and_analysis_record_rule(ymaze_env, client, store):
    draft, _asset = ymaze_env
    behavior = client.post(
        f"/api/drafts/{draft['id']}/preview",
        json={"asset_id": draft["videos"][0]["asset_id"],
              "unit_id": UNIT_A, "time_s": 1.5, "kind": "behavior"})
    assert behavior.status_code == 201, behavior.get_json()
    core = behavior.get_json()["data"]["summary"]["behavior"]
    assert core["arm_entry_rule"] == "center", "行为预览与正式分析同一判定"
    # 停在 B 臂：中心点规则首个有效观测即计（深度规则同为 1 次，时点更晚）。
    assert core["total_arm_entries"] == 1
    assert core["arm_B_entries"] == 1

    started = client.post(f"/api/drafts/{draft['id']}/analyze")
    assert started.status_code == 202, started.get_json()
    payload = started.get_json()["data"]
    assert payload["jobs"][0]["status"] == "completed"
    first, second = payload["jobs"][0]["manifest"]["units"]
    assert first["summary"]["arm_entry_rule"] == "center"
    assert second["summary"]["arm_entry_rule"] == "depth"
    assert first["summary"]["arm_B_entries"] == 1
    assert second["summary"]["arm_B_entries"] == 1

    run_dir = Path(store.paths.runs) / payload["run_id"]
    report = (run_dir / "report.md").read_text("utf-8")
    assert "arm_entry_rule=center" in report, "报告计量段记录实际规则"
    assert "arm_entry_rule=depth" in report


def test_catalog_exposes_rule_choices():
    from behaviourlab.experiments import catalog
    spec = [item for item in catalog() if item["key"] == "ymaze"][0]
    rule = [p for p in spec["parameters"] if p["key"] == "arm_entry_rule"][0]
    assert rule["type"] == "select"
    assert [c["value"] for c in rule["choices"]] == ["depth", "center"]
    assert "IMPC" in rule["help"] and "无时间防抖" in rule["help"]


def test_workspace_wires_rule_hint_and_depth_only_marks():
    source = (ROOT / "behaviourlab/web/static/workspace.js").read_text(
        encoding="utf-8")
    # 规则选择说明随当前值切换：区别、失效参数与无防抖边界都写明。
    assert 'param.key === "arm_entry_rule"' in source
    assert "中心点处于新区域的首个有效观测即记一次" in source
    assert "无防抖" in source
    assert "均不生效" in source
    # 深度规则专属参数在中心点模式下的标题标注。
    assert "YMAZE_DEPTH_ONLY_KEYS" in source
    assert '"entry_depth_ratio"' in source and '"require_center"' in source
    assert "（中心点规则下不生效）" in source
    # 规则切换走既有失效+保存路径（所有参数共用 change 处理）。
    change = source.split('input.addEventListener("change", () => {', 1)[1]
    assert "invalidatePreview()" in change
    assert "markDirty()" in change
