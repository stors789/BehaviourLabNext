"""界面补齐（BLN-UI-A004）：样式生效值、页面结构与工作台接线的静态契约。

这里只核对改到的显示层与确有必要的界面接线；真实浏览器渲染、同视口对照
和 Windows 原生操作不在开发自检内。
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSS = (ROOT / "behaviourlab/web/static/app.css").read_text(encoding="utf-8")
BASE = (ROOT / "behaviourlab/web/templates/base.html").read_text(encoding="utf-8")
HOME = (ROOT / "behaviourlab/web/templates/home.html").read_text(encoding="utf-8")
ANALYZE = (ROOT / "behaviourlab/web/templates/analyze.html").read_text(encoding="utf-8")
WORKSPACE = (ROOT / "behaviourlab/web/static/workspace.js").read_text(encoding="utf-8")


def rules(selector: str) -> list[str]:
    """取某个选择器的全部规则块（按源码顺序）。

    先去掉注释再按块解析：`html,\nbody {` 这种选择器列表也要能取到，
    媒体查询里的同名规则排在后面，可以单独按顺序判断。
    """

    clean = re.sub(r"/\*.*?\*/", "", CSS, flags=re.S)
    found = []
    for match in re.finditer(r"([^{}]+)\{([^{}]*)\}", clean):
        selectors = [part.strip() for part in match.group(1).split(",")]
        if selector in selectors:
            found.append(match.group(2))
    assert found, f"缺少规则：{selector}"
    return found


def rule(selector: str) -> str:
    """某个选择器的第一条（默认状态）规则块。"""

    return rules(selector)[0]


def test_design_tokens_and_typography():
    for token in ("--paper: #f2f5f8", "--ink: #1c3040", "--navy: #142c40",
                  "--accent: #087b80", "--accent-dark: #06646a",
                  "--line: #d8e1e8", "--muted: #5c6e7e",
                  "--accent-soft: #eaf5f5", "--disabled-bg: #e9eef3"):
        assert token in CSS, token
    body = " ".join(rules("body"))
    assert "font-size: 14px" in body
    assert "line-height: 1.5" in body
    assert '"Segoe UI"' in body and '"Microsoft YaHei"' in body

    page = rule(".page")
    assert "max-width: 1480px" in page
    assert "padding: 24px 24px 96px" in page

    assert "font-size: 24px" in rule("h1")
    assert "font-size: 17px" in rule("h2")
    assert "font-size: 14px" in rule("h3")


def test_control_heights_radii_and_disabled_state():
    button = rule(".button")
    assert "min-height: 38px" in button
    assert "padding: 8px 14px" in button
    assert "border-radius: var(--radius-control)" in button

    small = rule(".button.small")
    assert "min-height: 32px" in small
    assert "padding: 5px 10px" in small

    inputs = CSS.split('input[type="text"],', 1)[1].split("}", 1)[0]
    assert "min-height: 38px" in inputs
    assert "border-radius: var(--radius-control)" in inputs

    root = rule(":root")
    assert "--radius: 6px" in root
    assert "--radius-control: 5px" in root

    disabled = CSS.split(".button.primary:disabled,", 1)[1].split("}", 1)[0]
    assert "background: var(--disabled-bg)" in disabled
    assert "color: var(--disabled-ink)" in disabled
    assert "border-color: var(--disabled-line)" in disabled
    assert "opacity" not in disabled


def test_shell_and_workspace_geometry():
    topbar = rule(".topbar")
    assert "height: 60px" in topbar
    assert "position: sticky" in topbar
    assert "z-index: 30" in topbar
    assert "padding: 0 20px" in topbar
    assert "gap: 18px" in topbar

    assert "width: 32px" in rule(".brand-mark")
    brand = rule(".brand")
    assert "font-size: 19px" in brand and "font-weight: 700" in brand

    # 工作台占满视口用自己的内边距；选择与视频设置走普通页 padding。
    arena = rule('.work[data-step="arena"]')
    assert "padding: 14px 20px 18px" in arena
    assert "height: calc(100vh - 60px)" in arena
    assert "min-height: 620px" in arena
    assert "gap: 12px" in rule(".work")
    assert "padding" not in rule(".work")

    stage = rule(".stage")
    assert "minmax(0, 1fr) 288px" in stage
    assert "gap: 18px" in stage

    toolhead = rule(".toolhead")
    assert "height: 40px" in toolhead
    assert "min-height: 40px" in toolhead

    canvaswrap = rule(".canvaswrap")
    assert "position: relative" in canvaswrap
    assert "height: 0" in canvaswrap
    assert "min-height: 250px" in canvaswrap

    assert "position: absolute" in rule("#stage-canvas")
    assert "padding: 14px" in rule(".inspect-body")

    media = CSS.split("@media (max-width: 1100px)", 1)[1].split("}", 1)[0]
    assert "272px" in media


def test_video_setup_three_columns_and_action_bar():
    grid = rule(".range-grid")
    assert "192px minmax(0, 1fr) 304px" in grid
    assert "gap: 16px" in grid

    narrow = CSS.split("@media (max-width: 1100px)", 1)[1].split("\n}", 1)[0]
    assert "192px minmax(0, 1fr)" in narrow

    bar = rule(".action-bar")
    assert "position: fixed" in bar
    assert "min-height: 64px" in bar
    assert "border-top: 1px solid var(--line)" in bar

    inner = rule(".action-bar-inner")
    assert "padding: 12px 24px" in inner


def test_shell_markup_context_and_tasknav():
    # 本地内联标记，不使用外站资源；品牌始终是产品名。
    assert "<svg class=\"brand-mark\"" in BASE
    assert "http://" not in BASE and "https://" not in BASE
    assert "{{ app_name }}" in BASE
    assert "page-context" in BASE
    assert "跳到主要内容" in BASE
    assert "aria-current=page" in BASE

    # 视频设置属于“视频”阶段：任务导航不能把它显示成已进入场地。
    nav = WORKSPACE.split("function renderTasknav()", 1)[1].split("\n  }", 1)[0]
    assert 'if (state.step === "range") activeIndex = 1;' not in nav
    assert 'if (state.step === "arena") activeIndex' in nav

    # 三段各自进真实阶段：场地→几何视图、检测→检测视图，视频不再映射到 range。
    entries = WORKSPACE.split("function tasknavEntries()", 1)[1].split("\n  }", 1)[0]
    assert '{ step: "arena", view: "geometry" }' in entries
    assert '{ step: "arena", view: "detection" }' in entries
    assert '"range"' not in entries
    click = WORKSPACE.split('el.tasknav.addEventListener("click"', 1)[1].split(
        "\n      });", 1)[0]
    assert 'declared === "detection" ? "detection" : "geometry"' in click
    # 没有草稿时留在选视频，有草稿但没视频时回视频设置：都不落到空画布。
    assert 'if (step === "arena" && !state.draftId)' in click
    assert 'status("先选择视频并设置范围", "warn")' in click
    assert 'goStep("range")' in click


def test_home_intro_and_compact_entries():
    assert 'class="intro"' in HOME
    assert "行为视频分析" in HOME
    assert HOME.count('class="entry-card"') == 2
    assert 'href="/projects"' in HOME
    assert 'href="/analyze?mode=independent"' in HOME
    assert 'class="entry-icon"' in HOME
    assert "最近项目" in HOME
    assert "新建项目" in HOME

    intro = rule(".intro")
    assert "min-height: 154px" in intro
    assert "padding: 28px 32px" in intro
    assert "border-left: 5px solid var(--accent)" in intro
    assert "font-size: 32px" in rule(".intro h1")

    card = rule(".entry-card")
    assert "min-height: 152px" in card
    assert "padding: 24px" in card
    assert "font-size: 18px" in rule(".entry-body h2")

    grid = rule(".entry-grid")
    assert "repeat(2, minmax(0, 1fr))" in grid
    assert "gap: 18px" in grid


def test_select_page_import_and_chosen_list():
    assert 'class="view-head"' in ANALYZE
    assert "拖入视频文件" in ANALYZE
    assert "可同时选择多个视频" in ANALYZE
    assert 'id="selected-list"' in ANALYZE
    assert 'id="asset-list"' in ANALYZE
    # 已有资产复用降级为次要折叠，不再与已选列表并列成两个大栏。
    assert 'class="fold asset-fold"' in ANALYZE
    assert 'class="selected-list"' not in ANALYZE

    thumb = rule(".thumb")
    assert "width: 82px" in thumb
    assert "height: 48px" in thumb
    assert "object-fit: cover" in rule(".thumb img")

    # 底部动作条：返回在左，唯一主操作在右。
    bar = ANALYZE.split('<div class="action-bar">', 1)[1].split("</div>\n  </section>", 1)[0]
    assert "返回" in bar
    assert 'id="setup-videos"' in bar
    assert 'id="confirm-range"' in ANALYZE.split("data-panel=\"range\"", 1)[1]


def test_video_setup_columns_and_frame_bar():
    assert 'id="range-list"' in ANALYZE
    assert 'class="range-frame-card"' in ANALYZE
    assert 'id="range-frame-range"' in ANALYZE
    assert 'id="range-frame-number"' in ANALYZE
    assert 'class="settings-card"' in ANALYZE
    assert 'id="range-video-name"' in ANALYZE
    assert 'id="fps-fold"' in ANALYZE
    assert "应用到其他视频" in ANALYZE

    assert "loadRangeFrame(video, state.rangeFrame)" in WORKSPACE
    assert "rangeToken" in WORKSPACE
    # 帧号来自真实源帧响应头，不按名义 FPS 推算。
    assert 'response.headers.get("X-Source-Frame")' in WORKSPACE


def test_animal_window_entry_wiring():
    # 普通实验在“实验设置”里给出三种模式；人工 Y 只允许 global/exact。
    assert "animalWindowField(unit, { allowRelative: true })" in WORKSPACE
    assert "animalWindowField(unit, { allowRelative: false })" in WORKSPACE
    field = WORKSPACE.split("function animalWindowField", 1)[1].split("\n  }", 1)[0]
    assert '["exact", "指定时间"]' in field
    # relative 只在允许的选择集合里出现：人工 Y 的字段不给这一项。
    assert ('if (allowRelative) choices.push(["relative", "首次检测后"]);'
            in field)
    # 偏移可为负：只有起止与持续时间带下限。
    assert "if (minimum != null) input.min = String(minimum);" in field
    assert "offsetInput = numberInput(current && current.offset_s" in field

    # 只写当前动物，改值使旧预览确认失效。
    assert "assignWindow(target, select.value, values)" in field
    assert "invalidatePreview();" in field
    assert "target.window = " not in field


def test_unit_delete_entry_scope():
    assert 'button.textContent = "删除当前小鼠"' in WORKSPACE
    assert "(video.units || []).length < 2" in WORKSPACE
    delete = WORKSPACE.split("async function deleteCurrentUnit()", 1)[1].split("\n  }", 1)[0]
    assert "if (!window.confirm(" in delete
    # 取消发生在任何写入之前：确认框之后的代码才是删除。
    assert delete.index("window.confirm(") < delete.index("window.BLCore.removeUnit")
    assert "video.units = plan.units" in delete
    assert "manualCache().delete(unit.id)" in delete
    assert "invalidatePreview();" in delete
    # 只动当前视频：不触碰其他视频的单位或已完成结果。
    assert "state.draft.videos.forEach" not in delete
    assert "state.draft.videos" not in delete


def test_read_only_layers_do_not_write_geometry():
    draw = WORKSPACE.split("\n  function draw() {", 1)[1].split("\n  }", 1)[0]
    for writer in ("stepApply", "commitGeometry", "markDirty", "setStepShape"):
        assert writer not in draw, writer
    assert 'drawShape(item.shape, "#ffd26f", 1.6, false)' in draw
    assert 'drawShape(shape, "#ffd26f", 2.8, true)' in draw
    assert 'drawShape(item.shape, "#8ed7c7", 1.2, false)' in draw
    # LDB 不可见暗室不进绘制集合。
    shapes = WORKSPACE.split("function unitGeometryShapes(unit)", 1)[1].split("\n  }", 1)[0]
    assert 'step.key !== "dark"' in shapes


def test_analyze_page_renders_shell_with_draft(client, store, tmp_path):
    """有草稿时顶栏同时给出实验名与任务导航，工作台元素齐全。"""

    source = tmp_path / "vid.mp4"
    source.write_bytes(b"placeholder")
    asset = store.create_asset(
        name="小鼠A.mp4", path=str(source), project_id=None,
        metadata={"width": 640, "height": 480, "duration_s": 4.0,
                  "fps": 25.0, "frame_count": 100})
    draft = store.create_draft("tst", videos=[{
        "asset_id": asset["id"],
        "window": {"start_s": 0.0, "end_s": 4.0},
        "orientation": 0, "fps_override": None, "sample_fps": None,
        "units": [{"id": "a" * 32, "name": "小鼠 1", "animal_id": None,
                   "window": None, "geometry": {}, "options": {},
                   "detection": {}, "manual": None}],
    }])

    page = client.get(
        f"/analyze?mode=independent&draft_id={draft['id']}&step=range"
    ).get_data(as_text=True)
    assert 'class="top-assay"' in page
    assert "TST" in page or "强迫" in page
    assert 'id="tasknav"' in page
    assert 'data-step="range"' in page
    assert 'id="range-list"' in page
    assert 'id="range-frame-range"' in page
    assert 'id="range-status"' in page
    assert 'id="window-start"' in page and 'id="window-end"' in page
    assert 'id="setup-videos"' in page
    assert 'id="preview-note"' in page
    assert 'id="toolhead-title"' in page
    assert "{{" not in page


def test_analyze_page_renders_shell_with_draft(client, store, tmp_path):
    """有草稿时顶栏同时给出实验名与任务导航，工作台元素齐全。"""

    source = tmp_path / "vid.mp4"
    source.write_bytes(b"placeholder")
    asset = store.create_asset(
        name="小鼠A.mp4", path=str(source), project_id=None,
        metadata={"width": 640, "height": 480, "duration_s": 4.0,
                  "fps": 25.0, "frame_count": 100})
    draft = store.create_draft("tst", videos=[{
        "asset_id": asset["id"],
        "window": {"start_s": 0.0, "end_s": 4.0},
        "orientation": 0, "fps_override": None, "sample_fps": None,
        "units": [{"id": "a" * 32, "name": "小鼠 1", "animal_id": None,
                   "window": None, "geometry": {}, "options": {},
                   "detection": {}, "manual": None}],
    }])

    page = client.get(
        f"/analyze?mode=independent&draft_id={draft['id']}&step=range"
    ).get_data(as_text=True)
    assert 'class="top-assay"' in page
    assert "TST" in page or "强迫" in page
    assert 'id="tasknav"' in page
    assert 'data-step="range"' in page
    assert 'id="range-list"' in page
    assert 'id="range-frame-range"' in page
    assert 'id="range-status"' in page
    assert 'id="window-start"' in page and 'id="window-end"' in page
    assert 'id="setup-videos"' in page
    assert 'id="preview-note"' in page
    assert 'id="toolhead-title"' in page
    assert "{{" not in page


def test_geometry_tools_only_in_geometry_stage():
    """撤销/重画只在可编辑的场地阶段出现，检测阶段不摆几何工具。"""

    tools = WORKSPACE.split("function syncTools()", 1)[1].split("\n  }", 1)[0]
    assert 'const editable = !state.recordMode && state.view !== "detection";' in tools
    assert "el.toolUndo.hidden = !editable;" in tools
    assert "el.toolRedraw.hidden = !editable;" in tools
    assert "el.toolFinish.hidden = !editable || !state.editor.canFinish();" in tools

    detection = WORKSPACE.split("function renderDetectionInspector", 1)[1].split(
        "\n  }", 1)[0]
    assert "syncTools();" in detection
    # 工具头固定高度：标题与按钮同处一条 40px 行，不因按钮出现改变布局。
    head = (ROOT / "behaviourlab/web/templates/analyze.html").read_text(
        encoding="utf-8")
    toolhead = head.split('class="toolhead" id="toolhead"', 1)[1].split("</div>\n        <div class=\"canvaswrap\"", 1)[0]
    assert 'id="toolhead-title"' in toolhead
    assert 'id="tool-undo"' in toolhead and 'id="tool-redraw"' in toolhead
    assert 'id="tool-finish"' in toolhead
