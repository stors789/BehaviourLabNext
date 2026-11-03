"""检测设置批量应用与校验前配置摘要。

前端合并规则由真实 editor-core 的 ``effectiveDetection`` 驱动（分层用例
与后端 ``engine.resolve_detection`` 逐值比对）；同步结果走真实草稿
PUT/GET 保存并重载；界面接线做源码级断言。
"""

from __future__ import annotations

import json
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from behaviourlab import engine
from behaviourlab.experiments import catalog

ROOT = Path(__file__).resolve().parents[1]
UNIT_A = "a" * 32
UNIT_B = "b" * 32
UNIT_C = "c" * 32


# -- 前端合并规则（真实 editor-core） -----------------------------------
@pytest.fixture(scope="module")
def node_available():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node 不可用")
    return node


def frontend_effective(node, payload: dict) -> dict:
    job = Path(__file__).parent / "_sync_job.json"
    job.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    result = subprocess.run(
        [node, str(ROOT / "tests/js/test_detection_sync.cjs"), str(job)],
        capture_output=True, text=True, timeout=60, encoding="utf-8")
    job.unlink(missing_ok=True)
    assert result.returncode == 0, result.stderr
    line = [line for line in result.stdout.splitlines() if line.startswith("{")]
    assert line, result.stdout
    return json.loads(line[-1])["effective"]


def merge_like_frontend(base, override):
    """engine.merge_two 的 Python 镜像：喂给前端的是同样的层 JSON。"""

    if isinstance(base, dict) and isinstance(override, dict):
        merged = deepcopy(base)
        for key, value in override.items():
            merged[key] = (merge_like_frontend(merged[key], value)
                           if key in merged else deepcopy(value))
        return merged
    return deepcopy(override)


def frontend_layers(defaults, draft, video, unit):
    result: dict = {}
    for layer in (defaults, draft, video, unit):
        if layer is None:
            continue
        result = merge_like_frontend(result, layer)
    return result


@pytest.fixture
def layered_draft(client, store, tmp_path):
    """四层各不相同的真实草稿：设置 → 草稿 → 视频 → 鼠，两只视频各两鼠。"""

    store.set_setting("settings.detection", {
        "oft": {"dilation_iter": 4, "gray_threshold": 111.0},
    })
    first = tmp_path / "one.mp4"
    second = tmp_path / "two.mp4"
    for path in (first, second):
        path.write_bytes(b"controlled-not-real-video")
    assets = [
        store.create_asset(name="视频一.mp4", path=str(first),
                           project_id=None,
                           metadata={"fps": 25.0, "frame_count": 250,
                                     "duration_s": 10.0}),
        store.create_asset(name="视频二.mp4", path=str(second),
                           project_id=None,
                           metadata={"fps": 25.0, "frame_count": 250,
                                     "duration_s": 10.0}),
    ]

    def unit(unit_id: str, name: str, detection: dict) -> dict:
        return {
            "id": unit_id, "name": name, "animal_id": None,
            "window": None, "geometry": {}, "manual": None,
            "detection": detection,
        }

    videos = [
        {
            "asset_id": assets[0]["id"],
            "window": {"start_s": 0, "end_s": None},
            "orientation": 0, "fps_override": None, "sample_fps": None,
            "options": {}, "detection": {"erosion_iter": 2},
            "units": [
                unit(UNIT_A, "源鼠", {"bg_method": "temporal",
                                      "temporal_bg_samples": 45}),
                unit(UNIT_B, "同伴", {"bg_method": "mog2"}),
            ],
        },
        {
            "asset_id": assets[1]["id"],
            "window": {"start_s": 0, "end_s": None},
            "orientation": 90, "fps_override": 12.5, "sample_fps": 2,
            "detection": {"gray_threshold": 200.0},
            "units": [unit(UNIT_C, "另一视频的鼠", {"min_area": 500.0})],
        },
    ]
    response = client.post("/api/drafts", json={
        "assay": "oft", "videos": videos,
        "detection": {"min_area": 300.0},
    })
    assert response.status_code == 201, response.get_json()
    draft = response.get_json()["data"]["draft"]
    catalog_defaults = [entry for entry in catalog()
                        if entry["key"] == "oft"][0]["detection_defaults"]
    return draft, catalog_defaults


def test_frontend_merge_matches_backend_resolution(
        layered_draft, node_available):
    """前端 effectiveDetection 的层序与正式任务冻结逐值一致（含未显示字段）。"""

    draft, catalog_defaults = layered_draft
    settings = {"detection": {"dilation_iter": 4, "gray_threshold": 111.0}}
    for video in draft["videos"]:
        for unit in video["units"]:
            frontend = frontend_effective(node_available, {
                "defaults": catalog_defaults,
                "draft": draft["detection"],
                "video": video["detection"],
                "unit": unit["detection"],
            })
            backend = engine.resolve_detection(
                "oft", settings=settings, draft=draft, video=video, unit=unit)
            assert frontend == backend, unit["name"]


def test_sync_copies_full_effective_config(layered_draft, client):
    """同步（按钮结果 = 各鼠写入完整有效配置）保存重载后与源鼠逐值一致。"""

    draft, _catalog_defaults = layered_draft
    draft_id = draft["id"]
    stored = client.get(f"/api/drafts/{draft_id}").get_json()["data"]["draft"]
    first_video = stored["videos"][0]
    source, target = first_video["units"]

    # 按钮的写入动作：目标鼠获得源鼠完整有效配置的独立副本。
    effective = frontend_layers(
        [entry for entry in catalog() if entry["key"] == "oft"][0]
        ["detection_defaults"],
        stored["detection"], first_video["detection"], source["detection"])
    payload = deepcopy(stored)
    payload["videos"][0]["units"][1]["detection"] = deepcopy(effective)
    response = client.put(f"/api/drafts/{draft_id}", json={
        "videos": payload["videos"]})
    assert response.status_code == 200, response.get_json()

    reloaded = client.get(f"/api/drafts/{draft_id}").get_json()["data"]["draft"]
    video = reloaded["videos"][0]
    assert video["units"][0]["detection"] == source["detection"], "源鼠不动"
    assert video["units"][1]["detection"] == effective, "同伴获得完整副本"

    # 两只鼠的最终有效配置逐值一致，含设置层的未显示字段（dilation_iter=4）。
    settings = {"detection": {"dilation_iter": 4, "gray_threshold": 111.0}}
    resolved = [
        engine.resolve_detection("oft", settings=settings, draft=reloaded,
                                 video=video, unit=unit)
        for unit in video["units"]
    ]
    assert resolved[0] == resolved[1]
    assert resolved[1]["dilation_iter"] == 4
    assert resolved[1]["temporal_bg_samples"] == 45
    assert resolved[1]["erosion_iter"] == 2
    assert resolved[1]["bg_method"] == "temporal"

    # 同步只动 detection：目标鼠的身份/几何/窗/人工/实验参数逐值不变。
    for before, after in zip(first_video["units"], video["units"]):
        for field in ("id", "name", "animal_id", "window", "geometry",
                      "manual", "options"):
            assert after[field] == before[field], (before["name"], field)
    assert video["detection"] == first_video["detection"], "视频层不动"
    assert video["window"] == first_video["window"]

    # 第二段视频完全不受影响；几何/窗/身份/人工/实验参数逐值不变。
    other = reloaded["videos"][1]
    assert other["units"][0]["detection"] == {"min_area": 500.0}
    assert other["detection"] == {"gray_threshold": 200.0}
    assert other["orientation"] == 90 and other["fps_override"] == 12.5
    for before, after in zip(draft["videos"][1]["units"], other["units"]):
        assert after == before


def test_individual_adjust_after_sync_is_independent(layered_draft, client):
    """同步后单独调整一只：其他鼠不变；保存重载后结果一致。"""

    draft, _catalog_defaults = layered_draft
    draft_id = draft["id"]
    stored = client.get(f"/api/drafts/{draft_id}").get_json()["data"]["draft"]
    first_video = stored["videos"][0]
    effective = frontend_layers(
        [entry for entry in catalog() if entry["key"] == "oft"][0]
        ["detection_defaults"],
        stored["detection"], first_video["detection"],
        first_video["units"][0]["detection"])

    payload = deepcopy(stored)
    payload["videos"][0]["units"][1]["detection"] = deepcopy(effective)
    assert client.put(f"/api/drafts/{draft_id}",
                      json={"videos": payload["videos"]}).status_code == 200

    # 同步后单独调整同伴：提高膨胀次数为 0。
    reloaded = client.get(f"/api/drafts/{draft_id}").get_json()["data"]["draft"]
    payload = deepcopy(reloaded)
    payload["videos"][0]["units"][1]["detection"]["dilation_iter"] = 0
    assert client.put(f"/api/drafts/{draft_id}",
                      json={"videos": payload["videos"]}).status_code == 200

    final = client.get(f"/api/drafts/{draft_id}").get_json()["data"]["draft"]
    units = final["videos"][0]["units"]
    assert units[0]["detection"] == first_video["units"][0]["detection"]
    assert units[1]["detection"]["dilation_iter"] == 0
    assert units[1]["detection"]["bg_method"] == "temporal"
    settings = {"detection": {"dilation_iter": 4, "gray_threshold": 111.0}}
    resolved_source = engine.resolve_detection(
        "oft", settings=settings, draft=final, video=final["videos"][0],
        unit=units[0])
    resolved_target = engine.resolve_detection(
        "oft", settings=settings, draft=final, video=final["videos"][0],
        unit=units[1])
    assert resolved_source["dilation_iter"] == 4
    assert resolved_target["dilation_iter"] == 0, "只有被调整的鼠变化"


# -- 界面接线（源码级） --------------------------------------------------

def test_workspace_wires_sync_button_and_summary():
    source = (ROOT / "behaviourlab/web/static/workspace.js").read_text(
        encoding="utf-8")
    # 同步按钮：显式确认覆盖，明确说明影响范围，走既有失效+保存+重渲染。
    apply = source.split("function applyDetectionToVideoUnits", 1)[1]
    apply = apply.split("\n  }\n", 1)[0]
    assert 'window.confirm(message)' in apply
    assert "覆盖应用" in apply and "其他视频不受影响" in apply
    assert "effectiveUnitDetection(source)" in apply
    assert "clone(effective)" in apply
    assert "invalidatePreview()" in apply
    assert "markDirty()" in apply
    assert "renderInspector()" in apply
    # 按钮只在检测检查器出现，且不自动触发。
    assert 'methods.append(applyDetectionRow(unit))' in source
    assert 'button.textContent = "应用到本视频所有小鼠"' in source

    # 摘要常驻三个入口：检测视图与两个复核检查器，且默认展开。
    assert "body.append(detectionSummarySection())" in source
    assert source.count("body.append(detectionSummarySection())") == 3
    assert "人工记录，不执行检测" in source
    section = source.split("function detectionSummarySection", 1)[1]
    assert "wrap.open = true" in section, "提交前必须直接可见"

    # 人工判定与 engine.is_manual_path 同一规则。
    skips = source.split("function unitSkipsDetection", 1)[1]
    skips = skips.split("\n  }\n", 1)[0]
    assert 'state.assay === "ymaze"' in skips
    assert 'manual.mode === "manual"' in skips
    assert 'state.assay === "barnes"' in skips
    assert "hole_visits != null" in skips and "escape_time_s != null" in skips

    # 有效配置助手来自 editor-core 的共享实现。
    assert "window.BLCore.effectiveDetection" in source


def test_manual_path_matrix_matches_backend():
    """同一条人工判定规则：前端源码谓词与 engine.is_manual_path 逐例一致。"""

    cases = [
        ("ymaze", {"manual": {"mode": "manual"}}, True),
        ("ymaze", {"manual": {"mode": "auto"}}, False),
        ("ymaze", {"manual": None}, False),
        ("ymaze", {}, False),
        ("barnes", {"manual": {"hole_visits": []}}, True),
        ("barnes", {"manual": {"hole_visits": None,
                               "escape_time_s": 3.0}}, True),
        ("barnes", {"manual": {"hole_visits": None,
                               "escape_time_s": None}}, False),
        ("barnes", {"manual": None}, False),
        ("oft", {"manual": {"mode": "manual"}}, False),
        ("oft", {}, False),
    ]
    for assay, unit, expected in cases:
        assert engine.is_manual_path(assay, unit) is expected, (assay, unit)


def test_summary_states_for_edge_cases():
    """纯人工与无鼠场景有明确状态，不假称执行了检测。"""

    source = (ROOT / "behaviourlab/web/static/workspace.js").read_text(
        encoding="utf-8")
    summary = source.split("function detectionSummarySection", 1)[1]
    summary = summary.split("\n  }\n", 1)[0]
    assert "当前视频还没有小鼠" in summary, "无鼠场景有明确状态"
    assert "unitSkipsDetection(unit)" in summary
    assert "detectionMethodLabel(" in summary
