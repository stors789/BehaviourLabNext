"""画完自动换步骤：真实实验步骤定义 + 工作台提交接线。

步骤序列在 Node 里用真实 EditorCore 状态机跑；接线（提交入口必须都经过
commitGeometry，拖动/撤销不得换步骤）在源码层核对。
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from behaviourlab import experiments

ROOT = Path(__file__).resolve().parents[1]
JS_DIR = Path(__file__).parent / "js"


def workspace_source() -> str:
    return (ROOT / "behaviourlab/web/static/workspace.js").read_text(
        encoding="utf-8")


def block(source: str, start: str, end: str) -> str:
    return source.split(start, 1)[1].split(end, 1)[0]


@pytest.fixture(scope="module")
def steps_file(tmp_path_factory):
    """真实的 catalog 几何步骤：顺序、required 与 count 都来自实验 SPEC。"""

    payload = {entry["key"]: entry["geometry"]
               for entry in experiments.catalog()}
    path = tmp_path_factory.mktemp("steps") / "geometry_steps.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_commit_advances_to_next_required_step(steps_file):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node 不可用")
    script = JS_DIR / "test_step_advance.cjs"
    result = subprocess.run(
        [node, str(script), str(steps_file)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "step-advance OK" in result.stdout


def test_every_commit_entry_goes_through_commit_geometry():
    source = workspace_source()
    commit = block(source, "function commitGeometry(geometry)", "\n  }")
    # 提交后按有效步骤顺序取下一个未完成必需步骤，必要时换步骤。
    assert "nextRequiredGeometryStep()" in commit
    assert "selectStep(next.key)" in commit
    # 全部必需步骤完成时留在原地，不自动进入检测确认或分析。
    assert "primaryAction" not in commit
    assert "startAnalysis" not in commit
    assert "runPreview" not in commit

    # 点/线/四角/洞心/固定三点/中心点：提交结果统一走 commitGeometry。
    add_point = block(source, "function addPointAt(event)", "\n  }")
    assert "commitGeometry(result.geometry)" in add_point
    # 拖出椭圆：只有真实提交才换步骤，过小椭圆只提示。
    pointer_up = block(source, "function onPointerUp(event)", "\n  }")
    assert "if (geometry) commitGeometry(geometry);" in pointer_up
    # 多边形/边缘点拟合椭圆：完成按钮。
    finish = block(source, 'el.toolFinish.addEventListener("click"', "});")
    assert "state.editor.finish()" in finish
    assert "commitGeometry(geometry)" in finish

    # 拖动已有点位/顶点是编辑，不换步骤。
    move = block(source, "function onPointerMove(event)", "\n  }")
    assert "selectStep(" not in move
    assert "commitGeometry(" not in move
    for handler in ("el.toolUndo.addEventListener", "el.toolRedraw.addEventListener"):
        body = block(source, handler, "});")
        assert "selectStep(" not in body
        assert "commitGeometry(" not in body


def test_next_step_uses_current_unit_and_effective_steps():
    source = workspace_source()
    helper = block(source, "function nextRequiredGeometryStep()", "\n  }")
    # 有效步骤顺序（含 LDB 可见性过滤）与当前动物的已绘制状态。
    assert "geometrySteps()" in helper
    assert "currentUnit()" in helper
    assert "stepShape(unit, key)" in helper

    core = (ROOT / "behaviourlab/web/static/editor-core.js").read_text(
        encoding="utf-8")
    assert "function nextRequiredStep(steps, isDrawn)" in core
    assert "step.required !== true" in core
    assert "    nextRequiredStep," in core


def test_optional_steps_never_block_progress():
    core = (ROOT / "behaviourlab/web/static/editor-core.js").read_text(
        encoding="utf-8")
    body = block(core, "function nextRequiredStep(steps, isDrawn)", "\n  }")
    assert "isDrawn" in body
    assert "return null;" in body
    # 只按 required 判推进：可选步骤既不返回也不挡住后面的必需步骤。
    assert "optional" not in body
