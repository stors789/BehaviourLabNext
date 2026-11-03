"""中心点入臂规则：设置保存→冻结→分析→结果/报告 的真实链路一致性。

接口级走 client（生产 PUT/GET/preview/analyze），渲染层用源码接线断言
（parameterField 的规则选择、说明与不生效标注）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from behaviourlab.experiments import ymaze
from test_ymaze import PreviewSource, geometry

ROOT = Path(__file__).resolve().parents[1]
UNIT_A = "e" * 32
UNIT_B = "f" * 32


def videos_payload(asset_id: str) -> list[dict]:
    return [{
        "asset_id": asset_id,
        "window": {"start_s": 0.0, "end_s": 4.0},
        "orientation": 0, "fps_override": None, "sample_fps": 10,
        # 这组测试验证规则接线与展示，不验证背景方法：显式钉回阈值，
        # 避开合成场景在时序法下的前景不稳（时序默认由
        # test_factory_defaults_center_rule_and_temporal_background 断言）。
        "units": [
            {"id": UNIT_A, "name": "中心点鼠", "window": None,
             "geometry": geometry(),
             "options": {"arm_entry_rule": "center"},
             "detection": {"bg_method": "threshold"}, "manual": None},
            {"id": UNIT_B, "name": "原规则鼠", "window": None,
             "geometry": geometry(),
             "options": {"arm_entry_rule": "depth"},
             "detection": {"bg_method": "threshold"}, "manual": None},
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
    assert units[1]["options"]["arm_entry_rule"] == "depth", \
        "原规则鼠显式携带深度规则"
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
    # 方法说明与实际规则一致：中心点单位不再输出深度/确认/重置的旧条件。
    assert report.count("entry_confirm=") == 1, \
        "只有深度规则单位携带 entry_confirm 说明"
    assert report.count("center_reset=") == 1
    assert report.count("arm_entry_rule_note=") == 1, \
        "只有中心点单位携带规则附注"
    assert "boundary flips count as observed" in report


def test_result_page_shows_rule_per_unit(ymaze_env, client, store):
    draft, _asset = ymaze_env
    started = client.post(f"/api/drafts/{draft['id']}/analyze")
    assert started.status_code == 202, started.get_json()
    payload = started.get_json()["data"]
    page = client.get(f"/results/{payload['run_id']}").get_data(as_text=True)
    # 两只动物各自按实际生效规则展示。
    assert "入臂规则 中心点（首个有效观测即计，无防抖）" in page
    assert "入臂规则 深度+持续确认" in page
    # 深度单位仍展示实际参数值。
    assert "入臂确认 0.12" in page
    assert "中心重置 0.2" in page
    assert "须经中心 是" in page


def _unit_sections(page: str) -> dict[str, str]:
    """按 data-unit-id 把结果页切成各鼠自己的 article 区块。"""

    sections: dict[str, str] = {}
    for block in page.split('<article class="unit-block"')[1:]:
        header = block[:block.find(">")]
        if f'data-unit-id="{UNIT_A}"' in header:
            sections["中心点鼠"] = block
        elif f'data-unit-id="{UNIT_B}"' in header:
            sections["原规则鼠"] = block
    assert set(sections) == {"中心点鼠", "原规则鼠"}, "两只鼠区块都应存在"
    return sections


def test_result_page_center_unit_never_shows_confirm_as_effective(
        ymaze_env, client, store):
    draft, _asset = ymaze_env
    started = client.post(f"/api/drafts/{draft['id']}/analyze")
    assert started.status_code == 202, started.get_json()
    payload = started.get_json()["data"]
    page = client.get(f"/results/{payload['run_id']}").get_data(as_text=True)
    sections = _unit_sections(page)

    center = sections["中心点鼠"]
    assert "入臂规则 中心点" in center
    assert "本规则不适用" in center
    assert "中心点规则：首个有效观测即计，无防抖；入臂确认时间不适用" in center
    assert "入臂确认 0.12" not in center, \
        "中心点鼠不得把确认时间展示为生效条件"
    assert "须经中心 是" not in center

    depth = sections["原规则鼠"]
    assert "入臂规则 深度+持续确认" in depth
    assert "入臂确认 0.12" in depth, "深度鼠保留实际参数值"
    assert "中心重置 0.2" in depth
    assert "须经中心 是" in depth
    assert "本规则不适用" not in depth


def test_invalid_rule_values_rejected_by_api(client, store, tmp_path):
    source = tmp_path / "y.mp4"
    source.write_bytes(b"placeholder")
    asset = store.create_asset(
        name="y.mp4", path=str(source), project_id=None,
        metadata={"width": 240, "height": 240, "duration_s": 4.0,
                  "fps": 10.0, "frame_count": 40})
    draft = store.create_draft("ymaze", videos=[{
        "asset_id": asset["id"], "window": {"start_s": 0.0, "end_s": 4.0},
        "orientation": 0, "fps_override": None, "sample_fps": 10,
        "units": [{"id": UNIT_A, "name": "鼠", "window": None,
                   "geometry": geometry(), "options": {}, "detection": {},
                   "manual": None}],
    }])
    # 显式 null、空串、未知值、错误类型：保存必须带字段错误，不得静默回退。
    for bad in (None, "", "impc", 1, True):
        stored = client.get(
            f"/api/drafts/{draft['id']}").get_json()["data"]["draft"]
        stored["videos"][0]["units"][0]["options"]["arm_entry_rule"] = bad
        response = client.put(f"/api/drafts/{draft['id']}",
                              json={"videos": stored["videos"]})
        assert response.status_code == 400, (bad, response.get_json())
        fields = response.get_json()["error"]["fields"]
        assert any(item["field"].endswith("options.arm_entry_rule")
                   for item in fields), (bad, fields)
        # 保存被拒后草稿未被污染。
        reloaded = client.get(
            f"/api/drafts/{draft['id']}").get_json()["data"]["draft"]
        assert "arm_entry_rule" not in \
            reloaded["videos"][0]["units"][0]["options"]


def test_analysis_rejects_explicit_invalid_rule(client, store, tmp_path,
                                                monkeypatch):
    """绕过保存直接喂给冻结/分析链（旧式直读路径）也必须拒绝。"""

    import test_ymaze as harness
    from behaviourlab.schema import AnalysisContext

    values = harness.options()
    values["arm_entry_rule"] = None          # 显式 null：非法
    source = harness.rows([harness.ARM_A])
    job = {"assay": "ymaze", "run_id": "r" * 32, "job_id": "j" * 32,
           "asset": {"id": "a", "name": "y.mp4", "path": "memory",
                     "metadata": {"duration_s": 0.1}},
           "window": {"start_s": 0.0, "end_s": 0.1},
           "units": [harness.unit(values)], "style": {}, "orientation": 0}
    monkeypatch.setattr(
        ymaze, "capture",
        lambda context, units: {harness.UNIT_ID: {
            "rows": source, "candidates": []}})
    with pytest.raises(Exception) as exc_info:
        ymaze.analyze(AnalysisContext(job=job))
    assert "入臂规则" in str(exc_info.value)

    # 对照：缺字段的旧冻结任务直接分析仍是原深度规则。
    legacy_options = {key: value for key, value in harness.options().items()
                      if key != "arm_entry_rule"}
    legacy = harness.unit(legacy_options)
    job["units"] = [legacy]
    monkeypatch.setattr(
        ymaze, "capture",
        lambda context, units: {legacy["id"]: {
            "rows": source, "candidates": []}})
    result = ymaze.analyze(AnalysisContext(job=job))
    assert result.summary[0]["arm_entry_rule"] == "depth"


def test_catalog_exposes_rule_choices():
    from behaviourlab.experiments import catalog
    spec = [item for item in catalog() if item["key"] == "ymaze"][0]
    rule = [p for p in spec["parameters"] if p["key"] == "arm_entry_rule"][0]
    assert rule["type"] == "select"
    assert [c["value"] for c in rule["choices"]] == ["center", "depth"]
    assert "IMPC" in rule["help"] and "无时间防抖" in rule["help"]


def test_workspace_wires_rule_hint_and_depth_only_marks():
    """源码接线断言：可执行前端逻辑在 tests/js/test_ymaze_rule_note.cjs。"""

    source = (ROOT / "behaviourlab/web/static/workspace.js").read_text(
        encoding="utf-8")
    # 说明与“不生效”标注来自 editor-core 的可执行实现（切换即更新）。
    assert "window.BLCore.ymazeEntryRuleNote(" in source
    assert "window.BLCore.ymazeParamNotApplicable(" in source
    assert "（中心点规则下不生效）" in source
    # 说明挂在 arm_entry_rule 字段下，随当前有效值取文案。
    block = source.split('param.key === "arm_entry_rule"', 1)[1]
    assert "ymazeEntryRuleNote(" in block
    assert "effectiveOption(unit" in block
    # 规则切换走既有失效+保存路径（所有参数共用 change 处理）。
    change = source.split('input.addEventListener("change", () => {', 1)[1]
    assert "invalidatePreview()" in change
    assert "markDirty()" in change


def test_rule_note_frontend_logic_executable():
    """可执行前端测试：真实 editor-core 函数 + 假 DOM（Node）。"""

    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node 不可用")
    script = ROOT / "tests/js/test_ymaze_rule_note.cjs"
    result = subprocess.run([node, str(script)], capture_output=True,
                            text=True, timeout=60, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    assert "ymaze-rule-note OK" in result.stdout
