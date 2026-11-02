"""四类实验空间图补齐的有限自检：EPM/Y 自动（估计厘米）、LDB 可见
（源像素）、Barnes（圆盘物理厘米）的轨迹与热图。

四个正例都经过真实 process_unit → 生产方式组装 AnalysisResult →
真实 write_outputs；不制作测试录像，不安装新依赖。
"""

from __future__ import annotations

import copy
import math

import pytest
from matplotlib.image import imread

from behaviourlab import figures, outputs, timeline
from behaviourlab.experiments import barnes, epm, ldb, ymaze
from behaviourlab.schema import AnalysisResult

import test_ldb

PNG_MAGIC = bytes.fromhex("89504e470d0a1a0a")
STYLE = {"dpi": 90, "colormap": "viridis", "line_width": 2,
         "color": "#B86245"}


def _window(start: float, end: float, rows):
    return timeline.resolve_video_window(
        {"start_s": start, "end_s": end}, rows, None)


def _job(assay: str, units: list[dict]) -> dict:
    return {"run_id": "r" * 32, "job_id": "j" * 32, "assay": assay,
            "asset": {"id": "a" * 32, "name": "s.mp4", "path": "x"},
            "window": {"start_s": 0.0, "end_s": 1.0},
            "orientation": 0, "fps_override": None, "sample_fps": None,
            "options": {}, "detection": {}, "units": units,
            "output": {"annotated_video": False}, "style": dict(STYLE),
            "algorithm_version": "0.1.0"}


def _assemble(module, facts_list: list[dict]) -> AnalysisResult:
    """按各实验 analyze 的生产方式合并单单位事实。"""

    observations = timeline.sort_rows(
        [row for facts in facts_list for row in facts["rows"]])
    summary = [facts["summary"] for facts in facts_list]
    events = [event for facts in facts_list for event in facts["events"]]
    bins = [item for facts in facts_list for item in facts["bins"]]
    tables: dict[str, list] = {}
    for facts in facts_list:
        for name, items in (facts.get("tables") or {}).items():
            tables.setdefault(name, []).extend(items)
    unit_metadata = {facts["summary"]["unit_id"]: facts["metadata"]
                     for facts in facts_list}
    warnings = [facts["warning"] for facts in facts_list
                if facts.get("warning")]
    metadata: dict = {"assay": module.KEY, "units": unit_metadata,
                      "warnings": warnings}
    if module.KEY in ("epm", "ymaze"):
        metadata["coordinate_unit"] = "cm_estimated"
    return AnalysisResult(observations=observations, summary=summary,
                          events=events, bins=bins, tables=tables,
                          warnings=warnings, metadata=metadata)


def _contract_points(rows, x_key: str, y_key: str):
    """独立实现资格合同，用于核对 figures.qualified_points 的筛选。"""

    points: list[tuple[float, float, float]] = []
    for row in rows:
        if not row.get("in_window") or not row.get("detected"):
            continue
        if row.get("position_valid") is False or row.get("jump") is True:
            continue
        if "motion_x" in row and row.get("motion_x") is None:
            continue
        span = row.get("dt_s")
        x, y = row.get(x_key), row.get(y_key)
        if not isinstance(span, (int, float)) or span <= 0 \
                or not isinstance(x, (int, float)) \
                or not isinstance(y, (int, float)):
            continue
        points.append((float(x), float(y), float(span)))
    return points


def _assert_png(path) -> None:
    assert path.is_file() and path.read_bytes()[:8] == PNG_MAGIC
    image = imread(path)
    assert image.shape[0] > 0 and image.shape[1] > 0


def _assert_space_pair(manifest: dict, unit: dict, tmp_path) -> None:
    figures_map = manifest["figures"][unit["id"]]
    assert "unit_01_trajectory.png" in figures_map, figures_map
    assert "unit_01_heatmap.png" in figures_map, figures_map
    entry = next(item for item in manifest["units"]
                 if item["unit_id"] == unit["id"])
    assert entry["figures"]["trajectory"] == "unit_01_trajectory.png"
    assert entry["figures"]["heatmap"] == "unit_01_heatmap.png"
    assert entry["figure_files"] == figures_map
    assert "unit_01_trajectory.png" in manifest["files"]
    assert "unit_01_heatmap.png" in manifest["files"]
    for name in (entry["figures"]["trajectory"],
                 entry["figures"]["heatmap"]):
        _assert_png(tmp_path / "out" / name)


def test_epm_space_figures_via_production_export(tmp_path):
    """A. EPM：估计厘米轨迹+热图，区域柱图保留，科学事实不被改写。"""

    import test_epm

    unit = test_epm.unit()
    rows = test_epm.rows([(150, 100), (150, 100), (100, 100), (100, 100),
                          (100, 100), (150, 100), (100, 40)])
    window = _window(0.0, 0.7, rows)
    anchor = timeline.first_seen_in_window(
        epm.KEY, rows, window, unit["geometry"])
    facts = epm.process_unit(unit, rows, window, anchor)
    snapshot = copy.deepcopy((facts["rows"], facts["summary"],
                              facts["events"], facts["bins"],
                              facts["metadata"]))
    assert facts["metadata"]["coordinate_unit"] == "cm_estimated"
    assert facts["metadata"]["cm_per_px_est"] > 0

    result = _assemble(epm, [facts])
    job = _job("epm", [unit])
    manifest = outputs.write_outputs(tmp_path / "out", job, result)
    _assert_space_pair(manifest, unit, tmp_path)
    figures_map = manifest["figures"][unit["id"]]
    assert "unit_01_zones.png" in figures_map, "区域柱图必须保留"
    _assert_png(tmp_path / "out" / "unit_01_zones.png")

    # 资格筛选与独立合同实现一致；热图总曝光来自这些行的 dt_s。
    expected = _contract_points(facts["rows"], "x_cm", "y_cm")
    selected = figures.qualified_points(facts["rows"], x_key="x_cm",
                                        y_key="y_cm")
    assert selected == expected
    assert sum(span for _, _, span in expected) > 0

    # 导出不改写科学事实。
    assert (facts["rows"], facts["summary"], facts["events"],
            facts["bins"], facts["metadata"]) == snapshot
    assert result.summary[0] == facts["summary"]


def test_ymaze_auto_space_figures_with_sequence(tmp_path):
    """B. Y 自动：估计厘米轨迹+热图，完整序列图保留。"""

    import test_ymaze

    unit = test_ymaze.unit()
    rows = test_ymaze.rows(test_ymaze.FIXTURE)
    assert len(rows) == 23
    assert sum(1 for row in rows if row["detected"]) == 22
    candidates = test_ymaze.candidates(rows)
    window = _window(0.0, 2.3, rows)
    anchor = timeline.first_seen_in_window(
        ymaze.KEY, rows, window, unit["geometry"])
    facts = ymaze.process_unit(unit, {"rows": rows, "candidates": candidates},
                               window, anchor)
    snapshot = copy.deepcopy((facts["rows"], facts["summary"],
                              facts["events"], facts["bins"]))
    result = _assemble(ymaze, [facts])
    job = _job("ymaze", [unit])
    manifest = outputs.write_outputs(tmp_path / "out", job, result)
    _assert_space_pair(manifest, unit, tmp_path)
    figures_map = manifest["figures"][unit["id"]]
    assert "unit_01_sequence.png" in figures_map, "完整序列图必须保留"
    _assert_png(tmp_path / "out" / "unit_01_sequence.png")
    assert (facts["rows"], facts["summary"], facts["events"],
            facts["bins"]) == snapshot


def test_ldb_visible_space_figures_in_source_pixels(tmp_path):
    """C. LDB 可见：源像素轴轨迹+热图；厘米完成段经可校验逆映射。"""

    import test_ldb

    unit = test_ldb.unit()
    rows = test_ldb.source_rows(test_ldb.light_rows())
    assert len(rows) == 20 and all(row["detected"] for row in rows)
    window = _window(0.0, 4.0, rows)
    anchor = timeline.first_seen_in_window(
        ldb.KEY, rows, window, unit["geometry"])
    facts = ldb.process_unit(unit, rows, window, anchor)
    assert facts["metadata"]["position_available"] is True
    assert facts["metadata"]["tracking_reliable"] is True
    assert facts["metadata"]["coordinate_unit"] == "cm"
    snapshot = copy.deepcopy((facts["rows"], facts["summary"],
                              facts["events"], facts["bins"]))

    result = _assemble(ldb, [facts])
    job = _job("ldb", [unit])
    manifest = outputs.write_outputs(tmp_path / "out", job, result)
    _assert_space_pair(manifest, unit, tmp_path)
    assert (facts["rows"], facts["summary"], facts["events"],
            facts["bins"]) == snapshot

    # 完成段是厘米：源像素端点由真实源行与同一裁窗比例重建，
    # 落在明/暗室四角范围内且不按 y 符号猜室。
    segments_cm = figures.motion_segments(facts["rows"])
    assert segments_cm, "可见双室应有完成运动段"
    converted, insufficient = outputs._ldb_cm_segments_to_px(facts["rows"])
    assert converted and insufficient == 0
    for x0, y0, x1, y1 in converted:
        for x, y in ((x0, y0), (x1, y1)):
            assert 5.0 <= x <= 195.0 and 5.0 <= y <= 195.0, (x, y)


def test_barnes_disc_space_figures_physical_axes(tmp_path):
    """D. Barnes：圆盘物理厘米轨迹+热图，访问事实与逐洞表保留。"""

    import test_barnes

    options = {"hole_count": 3, "bin_seconds": 0.2}
    unit = test_barnes.unit(options=options)
    rows = test_barnes.fixture_rows()
    assert len(rows) == 12
    assert sum(1 for row in rows if row["detected"]) == 11
    window = _window(0.0, 1.2, rows)
    anchor = barnes.first_seen(rows, window)
    facts = barnes.process_unit(unit, rows, window, anchor)
    snapshot = copy.deepcopy((facts["rows"], facts["summary"],
                              facts["events"], facts["bins"],
                              facts["tables"]))
    assert facts["metadata"]["coordinate_unit"] == "cm"
    assert facts["metadata"]["disc_diameter_cm"] == 90.0

    result = _assemble(barnes, [facts])
    job = _job("barnes", [unit])
    manifest = outputs.write_outputs(tmp_path / "out", job, result)
    _assert_space_pair(manifest, unit, tmp_path)
    assert result.tables["holes"] == snapshot[4]["holes"]
    assert result.tables["automatic_hole_visits"] == \
        snapshot[4]["automatic_hole_visits"]
    visits = [item for item in result.events if item["kind"] == "hole_visit"]
    assert len(visits) == 4, "访问事实不被补图改写"
    assert (facts["rows"], facts["summary"], facts["events"],
            facts["bins"]) == snapshot[:4]


def test_ldb_two_units_isolated(tmp_path):
    """多动物隔离：不同名字/几何，第一只有效、第二只被抑制。"""

    import test_ldb

    unit1 = test_ldb.unit()
    unit2 = test_ldb.unit(dark_visible=False, dark=None)
    unit2 = dict(unit2)
    unit2["id"] = "f" * 32
    unit2["name"] = "小鼠 2"
    unit2["geometry"] = {"zones": {"light": {
        "type": "quad",
        "points": [[20, 20], [380, 20], [380, 180], [20, 180]]}}}
    rows1 = test_ldb.source_rows(test_ldb.light_rows())
    rows2 = test_ldb.source_rows(test_ldb.light_rows())
    # source_rows 夹具硬编码 unit1 的 id：第二只动物的行必须改成
    # 自己的 unit_id，否则两只动物的观测在结果里交错、段无法归属。
    for row in rows2:
        row["unit_id"] = unit2["id"]
        row["unit_name"] = "小鼠 2"
    window = _window(0.0, 4.0, rows1 + rows2)
    facts1 = ldb.process_unit(
        unit1, rows1, window,
        timeline.first_seen_in_window(ldb.KEY, rows1, window,
                                      unit1["geometry"]))
    facts2 = ldb.process_unit(
        unit2, rows2, window,
        timeline.first_seen_in_window(ldb.KEY, rows2, window,
                                      unit2["geometry"]))
    assert facts1["metadata"]["position_available"] is True
    assert facts2["metadata"]["position_available"] is False

    result = _assemble(ldb, [facts1, facts2])
    job = _job("ldb", [unit1, unit2])
    manifest = outputs.write_outputs(tmp_path / "out", job, result)
    figures_map = manifest["figures"]
    assert set(figures_map) == {unit1["id"], unit2["id"]}, "单位不串用"
    assert "unit_01_trajectory.png" in figures_map[unit1["id"]]
    assert "unit_01_heatmap.png" in figures_map[unit1["id"]]
    assert figures_map[unit2["id"]] == [], "被抑制单位没有空间图"
    assert "unit_02_trajectory.png" not in manifest["files"]
    entry1 = next(item for item in manifest["units"]
                  if item["unit_id"] == unit1["id"])
    entry2 = next(item for item in manifest["units"]
                  if item["unit_id"] == unit2["id"])
    assert entry1["unit_name"] == "小鼠 1" and entry1["figure_files"]
    assert entry2["unit_name"] == "小鼠 2" and entry2["figure_files"] == []
    assert "位置不可靠或不可用" in (entry2["figures"].get("note") or "")
    assert entry1["figures"]["trajectory"] == "unit_01_trajectory.png"


@pytest.mark.parametrize("assay,mode,geometry,options", [
    ("epm", "epm",
     {"zones": {"center": {"type": "quad", "points": [[90, 90], [110, 110],
                                                      [90, 110], [110, 90]]}}},
      {}),
    ("ymaze", "ymaze",
     {"zones": {"center": {"type": "quad", "points": [[90, 90], [110, 110],
                                                      [90, 110], [110, 90]]},
                "A": {"type": "quad", "points": [[90, 10], [110, 10],
                                                 [110, 90], [90, 90]]},
                "B": {"type": "quad", "points": [[110, 90], [190, 90],
                                                 [190, 110], [110, 110]]},
                "C": {"type": "quad", "points": [[10, 90], [90, 90],
                                                 [90, 110], [10, 110]]}}},
      {}),
    ("ldb", "ldb",
     {"zones": {"light": {"type": "quad", "points": [[10, 10], [190, 10],
                                                     [190, 95], [10, 95]]},
                "dark": {"type": "quad", "points": [[10, 105], [190, 105],
                                                    [190, 190], [10, 190]]}}},
      dict(ldb.DEFAULTS, dark_box_visible=True)),
    ("barnes", "barnes",
     {"arena": {"type": "ellipse", "center": [100.0, 100.0],
                "radii": [45.0, 45.0], "angle": 0.0},
      "holes": [{"id": "1", "center": [130.0, 100.0]}]},
     {"hole_count": 1, "disc_diameter_cm": 90.0, "hole_radius_cm": 5.0}),
])
def test_suppressed_units_produce_no_space_figures(
        tmp_path, assay, mode, geometry, options):
    """tracking_reliable=false / position_available=false：不出空间图。"""

    rows = [{"unit_id": "u1", "unit_name": "鼠", "frame": 0,
             "time_s": 0.0, "dt_s": 0.1, "in_window": True,
             "detected": True, "x_cm": 1.0, "y_cm": 1.0,
             "x_px": 100.0, "y_px": 100.0, "motion_x": 1.0,
             "motion_y": 1.0, "motion_x0": 1.0, "motion_y0": 1.0,
             "motion_x1": 2.0, "motion_y1": 1.0}]
    summary = {"unit_id": "u1", "unit_name": "鼠", "mode": mode}
    info = outputs._unit_figures(
        tmp_path, summary, rows, "unit_01", dict(STYLE),
        suppress_positions=True,
        unit_metadata={"tracking_reliable": False, "position_available": False},
        geometry=geometry, options=options)
    assert info["trajectory"] is None and info["heatmap"] is None
    assert "位置不可靠或不可用" in (info.get("note") or "")
    assert not (tmp_path / "unit_01_trajectory.png").exists()
    assert not (tmp_path / "unit_01_heatmap.png").exists()


def test_manual_ymaze_produces_no_space_figures(tmp_path):
    """纯人工 Y：不强行生成位置/运动图；序列图不受影响。"""

    rows = []
    info = outputs._unit_figures(
        tmp_path, {"unit_id": "u1", "unit_name": "Y 鼠", "mode": "ymaze"},
        rows, "unit_01", dict(STYLE), events=[], bins=[],
        unit_metadata={"mode": "manual"},
        geometry={"zones": {"center": {"type": "quad", "points": [
            [90, 90], [110, 110], [90, 110], [110, 90]]}}}, options={})
    assert info["trajectory"] is None and info["heatmap"] is None
    assert "纯人工记录" in (info.get("note") or "")
    assert not (tmp_path / "unit_01_trajectory.png").exists()


def test_unqualified_rows_produce_no_space_figures(tmp_path):
    """全部缺失或被判无效的行：不出空间图，原因如实记录。"""

    missing = [{"unit_id": "u1", "unit_name": "鼠", "frame": index,
                "time_s": 0.1 * index, "dt_s": 0.1, "in_window": True,
                "detected": False, "x_cm": None, "y_cm": None,
                "motion_x": None, "motion_y": None}
               for index in range(4)]
    invalid = [dict(row, detected=True, x_cm=1.0, y_cm=1.0,
                    motion_x=1.0, motion_y=1.0, position_valid=False)
               for row in missing]
    geometry = {"zones": {"center": {"type": "quad", "points": [
        [0, 0], [10, 0], [10, 10], [0, 10]]}}}
    metadata = {"cm_per_px_est": 0.5, "coordinate_unit": "cm_estimated"}
    for rows in (missing, invalid):
        info = outputs._unit_figures(
            tmp_path, {"unit_id": "u1", "unit_name": "鼠", "mode": "epm"},
            rows, "unit_01", dict(STYLE),
            unit_metadata=metadata, geometry=geometry, options={})
        assert info["trajectory"] is None and info["heatmap"] is None
        assert not (tmp_path / "unit_01_trajectory.png").exists()
        assert "没有完成的连续运动段" in (info.get("note") or "")
        assert "没有合格位置" in (info.get("note") or "")


def test_motion_x_null_rows_never_reach_heatmap(tmp_path):
    """motion_x=null 的行被剔除：热图曝光等于合格行的 dt_s 之和。"""

    rows = [
        {"unit_id": "u1", "frame": 0, "time_s": 0.0, "dt_s": 1.0,
         "in_window": True, "detected": True, "x_cm": 5.0, "y_cm": 5.0,
         "motion_x": None, "motion_y": None},
        {"unit_id": "u1", "frame": 1, "time_s": 1.0, "dt_s": 1.0,
         "in_window": True, "detected": True, "x_cm": 10.0, "y_cm": 10.0,
         "motion_x": 10.0, "motion_y": 10.0, "motion_x0": 10.0,
         "motion_y0": 10.0, "motion_x1": 11.0, "motion_y1": 10.0},
    ]
    expected = _contract_points(rows, "x_cm", "y_cm")
    assert [point[:2] for point in expected] == [(10.0, 10.0)]
    selected = figures.qualified_points(rows, x_key="x_cm", y_key="y_cm")
    assert selected == expected
    grid, used, total = figures.space_heat_grid(
        selected, (0.0, 20.0, 0.0, 20.0))
    assert used == 1
    assert math.isclose(total, 1.0, abs_tol=1e-9)
    assert math.isclose(sum(sum(line) for line in grid), 1.0, abs_tol=1e-9)


def test_gap_segments_are_not_connected(tmp_path):
    """缺口两侧各有完成段：只画保存段，不跨缺失补连线。"""

    rows = [
        {"unit_id": "u1", "frame": 0, "time_s": 0.0, "dt_s": 0.1,
         "in_window": True, "detected": True, "x_cm": 1.0, "y_cm": 1.0,
         "motion_x": 1.0, "motion_y": 1.0, "motion_x0": 1.0,
         "motion_y0": 1.0, "motion_x1": 2.0, "motion_y1": 1.0},
        {"unit_id": "u1", "frame": 1, "time_s": 0.1, "dt_s": 0.1,
         "in_window": True, "detected": False, "x_cm": None, "y_cm": None,
         "motion_x": None, "motion_y": None},
        {"unit_id": "u1", "frame": 2, "time_s": 0.2, "dt_s": 0.1,
         "in_window": True, "detected": True, "x_cm": 5.0, "y_cm": 1.0,
         "motion_x": 5.0, "motion_y": 1.0, "motion_x0": 5.0,
         "motion_y0": 1.0, "motion_x1": 6.0, "motion_y1": 1.0},
    ]
    segments = figures.motion_segments(rows)
    assert segments == [(1.0, 1.0, 2.0, 1.0), (5.0, 1.0, 6.0, 1.0)]
    outlines = [[(0.0, 0.0), (8.0, 0.0), (8.0, 8.0), (0.0, 8.0)]]
    extent = figures.space_extent(
        outlines, figures.qualified_points(rows, x_key="x_cm",
                                           y_key="y_cm"))
    path = tmp_path / "gap_trajectory.png"
    assert figures.space_trajectory_png(
        path, segments=segments, outlines=outlines, extent=extent,
        xlabel="x (估计 cm)", ylabel="y (估计 cm)", style=dict(STYLE))
    assert path.read_bytes()[:8] == PNG_MAGIC


def test_space_extent_margin_and_y_direction():
    """范围：约 6%、至少 1 单位边距；画布 y 轴向下。"""

    outline = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
    extent = figures.space_extent([outline], [])
    assert extent == (-1.0, 11.0, -1.0, 11.0), "10 单位的 6% 不足 1，取 1"
    wide = figures.space_extent([[[-50.0, 0.0], [50.0, 0.0]]], [])
    assert wide == (-56.0, 56.0, -1.0, 1.0), \
        "零跨度轴按至少 1 单位边距；6% 只对有跨度的轴生效"
    figure, axes = figures._space_base(extent, [outline], "x", "y")
    assert axes.get_ylim() == (11.0, -1.0), "图像 y 轴向下"
    assert axes.get_aspect() == 1.0


# -- LDB 源像素端点返修：两个固定自然反例 + 裁窗 + 缺失断段 --------------


def _ldb_endpoint_rows(unit, positions, start, end):
    rows = test_ldb.source_rows(positions)
    window = _window(start, end, rows)
    anchor = timeline.first_seen_in_window(
        ldb.KEY, rows, window, unit["geometry"])
    return ldb.process_unit(unit, rows, window, anchor), rows


def _assert_endpoint(actual, expected) -> None:
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        for a, b in zip(got, want):
            assert abs(a - b) <= 1e-6, (got, want)


def _assert_manifest_pair(manifest: dict, unit: dict, out_dir) -> None:
    """单次核对四处清单一致 + 两图真实可解码。"""

    figures_map = manifest["figures"][unit["id"]]
    assert "unit_01_trajectory.png" in figures_map
    assert "unit_01_heatmap.png" in figures_map
    entry = next(item for item in manifest["units"]
                 if item["unit_id"] == unit["id"])
    assert entry["figure_files"] == figures_map
    assert entry["figures"]["trajectory"] == "unit_01_trajectory.png"
    assert entry["figures"]["heatmap"] == "unit_01_heatmap.png"
    for name in ("unit_01_trajectory.png", "unit_01_heatmap.png"):
        assert name in manifest["files"], name
        _assert_png(out_dir / name)


def test_ldb_doorway_crossing_segments_exact(tmp_path):
    """反例 1：门区穿越。轨迹端点必须逐段回到源像素 90→98→102→110，
    不再出现 90→108→92 的猜室折返；热图仍用真实 98/102。"""

    unit = test_ldb.unit(doorway={"type": "quad",
                                  "points": [[80, 90], [120, 90],
                                             [120, 110], [80, 110]]})
    facts, rows = _ldb_endpoint_rows(
        unit, [(100, 90), (100, 98), (100, 102), (100, 110)], 0.0, 0.8)
    assert facts["metadata"]["coordinate_unit"] == "cm"
    snapshot = copy.deepcopy((facts["rows"], facts["summary"],
                              facts["events"], facts["bins"],
                              facts["metadata"]))

    converted, insufficient = outputs._ldb_cm_segments_to_px(facts["rows"])
    assert insufficient == 0
    _assert_endpoint(converted, [(100.0, 90.0, 100.0, 98.0),
                                 (100.0, 98.0, 100.0, 102.0),
                                 (100.0, 102.0, 100.0, 110.0)])

    result = _assemble(ldb, [facts])
    job = _job("ldb", [unit])
    manifest = outputs.write_outputs(tmp_path / "out", job, result)
    _assert_manifest_pair(manifest, unit, tmp_path / "out")

    # 热图资格与独立合同实现一致；曝光来自真实 98/102 行的 dt_s。
    selected = figures.qualified_points(rows, x_key="x_px", y_key="y_px")
    assert selected == _contract_points(rows, "x_px", "y_px")
    assert sum(span for _, _, span in selected) == pytest.approx(0.8, abs=1e-9)

    # 科学事实不变：导出前后观测/汇总/事件/分箱/元数据逐项相等。
    assert (facts["rows"], facts["summary"], facts["events"],
            facts["bins"], facts["metadata"]) == snapshot
    assert result.summary[0] == facts["summary"]


def test_ldb_dark_boundary_segments_exact(tmp_path):
    """反例 2：无门区，暗室边界 y=105。边界点必须画成 105，
    不再被猜成明室边界 95。"""

    unit = test_ldb.unit()
    facts, rows = _ldb_endpoint_rows(
        unit, [(100, 150), (100, 105), (100, 150)], 0.0, 0.6)
    snapshot = copy.deepcopy((facts["rows"], facts["summary"],
                              facts["events"], facts["bins"],
                              facts["metadata"]))

    converted, insufficient = outputs._ldb_cm_segments_to_px(facts["rows"])
    assert insufficient == 0
    _assert_endpoint(converted, [(100.0, 150.0, 100.0, 105.0),
                                 (100.0, 105.0, 100.0, 150.0)])

    result = _assemble(ldb, [facts])
    job = _job("ldb", [unit])
    manifest = outputs.write_outputs(tmp_path / "out", job, result)
    _assert_manifest_pair(manifest, unit, tmp_path / "out")
    selected = figures.qualified_points(rows, x_key="x_px", y_key="y_px")
    assert selected == _contract_points(rows, "x_px", "y_px")
    assert (facts["rows"], facts["summary"], facts["events"],
            facts["bins"], facts["metadata"]) == snapshot


def test_ldb_cropped_window_keeps_ratio_and_segment_identity(tmp_path):
    """有限裁窗：源像素端点用同一裁窗比例插值；段的时间边界、
    frame 归属与存储的科学完成段不被改写，段数量不变。"""

    unit = test_ldb.unit()
    facts, _ = _ldb_endpoint_rows(
        unit, [(100, 150), (100, 105), (100, 150)], 0.1, 0.3)
    # 段 1：[0.1,0.2]（起点按窗起点裁切，比例 0.5）；
    # 段 2：[0.2,0.3]（终点按窗尾裁切，比例 0.5）。
    segments_cm = figures.motion_segments(facts["rows"])
    assert len(segments_cm) == 2
    converted, insufficient = outputs._ldb_cm_segments_to_px(facts["rows"])
    assert insufficient == 0
    _assert_endpoint(converted, [(100.0, 127.5, 100.0, 105.0),
                                 (100.0, 105.0, 100.0, 127.5)])

    identity = [(row.get("unit_id"), row.get("frame"),
                 row.get("motion_start_s"), row.get("motion_end_s"),
                 row.get("motion_x0"), row.get("motion_y0"),
                 row.get("motion_x1"), row.get("motion_y1"))
                for row in facts["rows"]]
    result = _assemble(ldb, [facts])
    job = _job("ldb", [unit])
    manifest = outputs.write_outputs(tmp_path / "out", job, result)
    _assert_manifest_pair(manifest, unit, tmp_path / "out")
    # 导出只换绘图表示：存储段的 unit_id/frame/时间/厘米端点不变。
    assert [(row.get("unit_id"), row.get("frame"),
             row.get("motion_start_s"), row.get("motion_end_s"),
             row.get("motion_x0"), row.get("motion_y0"),
             row.get("motion_x1"), row.get("motion_y1"))
            for row in facts["rows"]] == identity


def test_ldb_missing_row_breaks_segment_no_bridge(tmp_path):
    """缺失断段：中间行缺失即断开，不跨缺失补线；转换段数与
    完成段一一对应，端点仍精确回到源像素。"""

    unit = test_ldb.unit()
    facts, _ = _ldb_endpoint_rows(
        unit, [(100, 150), (100, 105), None, (100, 150)], 0.0, 0.8)
    segments_cm = figures.motion_segments(facts["rows"])
    assert len(segments_cm) == 1, "缺失行之后的点不与前段相连"
    converted, insufficient = outputs._ldb_cm_segments_to_px(facts["rows"])
    assert insufficient == 0
    _assert_endpoint(converted, [(100.0, 150.0, 100.0, 105.0)])

    result = _assemble(ldb, [facts])
    job = _job("ldb", [unit])
    manifest = outputs.write_outputs(tmp_path / "out", job, result)
    figures_map = manifest["figures"][unit["id"]]
    assert "unit_01_trajectory.png" in figures_map
    assert "unit_01_heatmap.png" in figures_map
    entry = next(item for item in manifest["units"]
                 if item["unit_id"] == unit["id"])
    assert entry["figures"]["trajectory"] == "unit_01_trajectory.png"
    _assert_png(tmp_path / "out" / "unit_01_trajectory.png")
    _assert_png(tmp_path / "out" / "unit_01_heatmap.png")
