"""EZM 高架零迷宫：三项固定夹具、七类占用、相邻进入与几何校验。"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from behaviourlab import experiments, timeline
from behaviourlab.experiments import ezm

UNIT = "u" * 32
TOL = 1e-8

ARENA = {"type": "ellipse", "center": [100.0, 100.0],
         "radii": [60.0, 30.0], "angle": 0.0}
START = {"type": "point", "points": [[160.0, 100.0]]}

OPTIONS = {
    "outer_diameter_cm": 60.0,
    "ring_width_cm": 5.0,
    "boundary_margin_cm": 0.5,
    "confirm_s": 0.12,
    "quality_threshold": 0.9,
    "bin_seconds": 3.0,
}


def geometry(arena=None, start=None, **extra):
    shapes = {"arena": arena or ARENA, "open_boundary": start or START}
    shapes.update(extra)
    return shapes


def unit(*, options=None, geometric=None):
    merged = dict(OPTIONS)
    if options:
        merged.update(options)
    return {"id": UNIT, "name": "小鼠 1", "animal_id": None, "window": None,
            "geometry": geometric or geometry(), "options": merged,
            "detection": {}}


def ring_point(theta_deg, *, center=(100.0, 100.0), radii=(55.0, 27.5)):
    """夹具点：像素椭圆上物理半径恒为 27.5 cm（环带中线）。"""

    radians = math.radians(theta_deg)
    return (center[0] + radii[0] * math.cos(radians),
            center[1] + radii[1] * math.sin(radians))


def source_rows(positions, *, exposure=1.0, quality=1.0):
    rows = []
    for index, seen in enumerate(positions):
        start = round(exposure * index, 10)
        end = round(exposure * (index + 1), 10)
        detected = seen is not None
        rows.append({
            "unit_id": UNIT, "unit_name": "小鼠 1", "animal_id": None,
            "frame": index, "time_s": float(start), "dt_s": exposure,
            "source_dt_s": exposure, "elapsed_s": float(start),
            "analysis_fps": 1.0 / exposure,
            "source_interval_start_s": float(start),
            "source_interval_end_s": float(end),
            "interval_start_s": float(start), "interval_end_s": float(end),
            "in_window": True, "detected": detected,
            "x_px": float(seen[0]) if detected else None,
            "y_px": float(seen[1]) if detected else None,
            "quality": quality if detected else 0.0,
        })
    return rows


def window(seconds):
    return timeline.Window(0.0, float(seconds))


# -- 1. 椭圆场地固定夹具：七类占用、相邻进入与两个潜伏期 ----------------

def test_ezm_ring_fixture_occupancy_and_entries():
    positions = [ring_point(45), ring_point(45), ring_point(135),
                 ring_point(135), ring_point(180), ring_point(225),
                 ring_point(225), None, ring_point(315), (100.0, 100.0),
                 ring_point(135)]
    facts = ezm.process_unit(unit(), source_rows(positions), window(11), 0.0)
    summary = facts["summary"]

    # 七类曝光：O1 2 / C1 3 / O2 2 / C2 1 / 边界 1 / 区外 1 / 未知 1。
    assert [row["zone"] for row in facts["rows"]] == [
        "O1", "O1", "C1", "C1", "boundary", "O2", "O2", "unknown",
        "C2", "outside", "C1"]
    assert summary["total_time_s"] == pytest.approx(11.0, abs=TOL)
    assert summary["observed_time_s"] == pytest.approx(10.0, abs=TOL)
    assert summary["missing_time_s"] == pytest.approx(1.0, abs=TOL)
    assert summary["open_time_s"] == pytest.approx(4.0, abs=TOL)
    assert summary["closed_time_s"] == pytest.approx(4.0, abs=TOL)
    assert summary["boundary_time_s"] == pytest.approx(1.0, abs=TOL)
    assert summary["outside_time_s"] == pytest.approx(1.0, abs=TOL)
    assert summary["unknown_time_s"] == pytest.approx(1.0, abs=TOL)
    assert summary["known_zone_time_s"] == pytest.approx(10.0, abs=TOL)

    # 两个开放分母：整窗曝光含缺失；已知时间只算开放+封闭。
    assert summary["open_time_percent_total"] == pytest.approx(
        400.0 / 11.0, abs=TOL)
    assert summary["open_time_percent_known"] == pytest.approx(50.0, abs=TOL)

    assert summary["initial_zone"] == "O1"
    assert summary["first_open_observation_latency_s"] == pytest.approx(
        0.0, abs=TOL)
    assert summary["first_open_entry_latency_s"] == pytest.approx(5.0, abs=TOL)
    assert summary["first_open_entry_censored"] is False
    assert summary["open_entries"] == 1
    assert summary["closed_entries"] == 1

    events = facts["events"]
    assert [(item["target"], item["from_zone"], item["time_s"],
             item["confirmed_s"]) for item in events] == [
        ("C1", "O1", 2.0, 3.0), ("O2", "C1", 5.0, 6.0)]
    assert all(item["kind"] == "zone_entry" for item in events)
    assert all(item["source"] == "automatic" for item in events)

    # 七类明细表固定七行，entries 只数真正以该区为 target 的事件。
    zones = {row["zone"]: row for row in facts["zones"]}
    assert list(zones) == list(ezm.ALL_ZONES)
    assert zones["O1"]["entries"] == 0
    assert zones["O2"]["entries"] == 1
    assert zones["C1"]["entries"] == 1
    assert zones["C2"]["entries"] == 0
    assert zones["boundary"]["entries"] == 0
    assert zones["unknown"]["entries"] == 0
    assert zones["outside"]["time_s"] == pytest.approx(1.0, abs=TOL)

    # 分箱：箱间曝光与路径都必须守恒。
    bins = facts["bins"]
    assert [item["duration_s"] for item in bins] == pytest.approx(
        [3.0, 3.0, 3.0, 2.0], abs=TOL)
    assert sum(item["open_time_s"] + item["closed_time_s"]
               + item["boundary_time_s"] + item["outside_time_s"]
               + item["unknown_time_s"] for item in bins) == pytest.approx(
        summary["total_time_s"], abs=TOL)
    assert sum(item["distance_cm"] for item in bins) == pytest.approx(
        summary["total_distance_cm"], abs=TOL)

    # 区外与未知不产生运动坐标；边界位置仍算有效跑道位置。
    assert facts["rows"][9]["motion_x"] is None
    assert facts["rows"][7]["motion_x"] is None
    assert facts["rows"][4]["motion_x"] is not None
    assert summary["total_distance_cm"] > 0
    assert facts["metadata"]["coordinate_unit"] == "cm"


# -- 2. 缺少确认帧：不给开放进入补事件 ----------------------------------

def test_ezm_unconfirmed_open_entry_is_censored():
    positions = [ring_point(45), None, ring_point(225)]
    facts = ezm.process_unit(unit(), source_rows(positions), window(3), 0.0)
    summary = facts["summary"]

    assert facts["events"] == []
    assert summary["open_entries"] == 0
    assert summary["closed_entries"] == 0
    assert summary["first_open_entry_latency_s"] is None
    assert summary["first_open_entry_censored"] is True
    assert summary["initial_zone"] == "O1"
    assert summary["first_open_observation_latency_s"] == pytest.approx(
        0.0, abs=TOL)
    # 缺失是未知而不是封闭区；纯中心点落在环外。
    assert summary["unknown_time_s"] == pytest.approx(1.0, abs=TOL)
    assert summary["closed_time_s"] == pytest.approx(0.0, abs=TOL)
    assert summary["outside_time_s"] == pytest.approx(0.0, abs=TOL)
    facts_center = ezm.process_unit(unit(), source_rows([(100.0, 100.0)]),
                                    window(1), 0.0)
    assert facts_center["rows"][0]["zone"] == "outside"
    assert facts_center["summary"]["outside_time_s"] == pytest.approx(
        1.0, abs=TOL)
    assert facts_center["summary"]["unknown_time_s"] == pytest.approx(
        0.0, abs=TOL)


# -- 3. 圆形场地：确认必须来自真实更晚观测与半开窗 ----------------------

def circular():
    return dict(ARENA, radii=[60.0, 60.0])


def circle_point(theta_deg):
    radians = math.radians(theta_deg)
    return (100.0 + 55.0 * math.cos(radians),
            100.0 + 55.0 * math.sin(radians))


def test_ezm_confirmation_needs_later_observation():
    options = {"confirm_s": 0.4, "bin_seconds": 60.0}
    geometric = geometry(arena=circular())
    two = [circle_point(135), circle_point(45)]
    rows = source_rows(two, exposure=0.6)
    facts = ezm.process_unit(unit(options=options, geometric=geometric),
                             rows, window(1.2), 0.0)
    assert facts["events"] == []
    assert facts["summary"]["initial_zone"] == "C1"
    assert facts["summary"]["open_entries"] == 0

    three = [circle_point(135), circle_point(45), circle_point(45)]
    rows = source_rows(three, exposure=0.6)
    facts = ezm.process_unit(unit(options=options, geometric=geometric),
                             rows, window(1.8), 0.0)
    events = facts["events"]
    assert [(item["target"], item["time_s"], item["confirmed_s"])
            for item in events] == [("O1", 0.6, 1.2)]
    assert facts["summary"]["open_entries"] == 1
    assert facts["summary"]["first_open_entry_latency_s"] == pytest.approx(
        0.6, abs=TOL)

    # 动物窗右端 1.2 是半开终点：确认帧不在窗内，就没有事件。
    rows = source_rows(three, exposure=0.6)
    facts = ezm.process_unit(unit(options=options, geometric=geometric),
                             rows, window(1.2), 0.0)
    assert facts["events"] == []
    assert facts["summary"]["first_open_entry_censored"] is True


# -- 注册、校验与预览接线 ------------------------------------------------

def test_registration_validation_and_preview():
    from behaviourlab import preview

    entry = [item for item in experiments.catalog()
             if item["key"] == "ezm"][0]
    assert entry["label"] == "高架零迷宫 / O迷宫"
    assert entry["short"] == "EZM"
    assert [param["key"] for param in entry["parameters"]] == [
        "outer_diameter_cm", "ring_width_cm", "boundary_margin_cm",
        "confirm_s", "quality_threshold", "bin_seconds"]
    assert [item["label"] for item in entry["geometry"]] == [
        "外缘", "O1 开放区起始边界", "检测遮罩"]
    assert [bool(item["required"]) for item in entry["geometry"]] == [
        True, True, False]
    assert entry["template_keys"] == ["outer_diameter_cm", "ring_width_cm"]
    assert set(entry["defaults"]) == set(ezm.DEFAULTS)
    # 检测默认值属于检测层，与统计身体质量阈值分开。
    assert entry["detection_defaults"]["bg_method"] == "local_otsu"
    assert entry["detection_defaults"]["erosion_kernel"] == 3
    assert entry["detection_defaults"]["erosion_iter"] == 1
    assert entry["detection_defaults"]["quality_threshold"] == 0.4

    assert ezm.validate_unit(dict(OPTIONS), geometry(), complete=True) == []
    # 未画完仍可保存，最终校验才报缺。
    assert ezm.validate_unit(dict(OPTIONS), {}) == []
    fields = [item["field"] for item in ezm.validate_unit(
        dict(OPTIONS), {}, complete=True)]
    assert fields == ["geometry.arena", "geometry.open_boundary"]

    # 半径约束用最终有效值比较，默认宽度也不能绕过。
    bad = dict(OPTIONS, outer_diameter_cm=8.0)
    issues = ezm.validate_unit(bad, geometry())
    assert [item["field"] for item in issues] == ["options.ring_width_cm"]
    # 布尔值不是数字。
    issues = ezm.validate_unit(dict(OPTIONS, confirm_s=True), geometry())
    assert [item["field"] for item in issues] == ["options.confirm_s"]
    for key in ("outer_diameter_cm", "ring_width_cm", "confirm_s",
                "bin_seconds"):
        issues = ezm.validate_unit(dict(OPTIONS, **{key: 0}), geometry())
        assert [item["field"] for item in issues] == [f"options.{key}"]
    issues = ezm.validate_unit(dict(OPTIONS, boundary_margin_cm=-1.0),
                               geometry())
    assert [item["field"] for item in issues] == ["options.boundary_margin_cm"]
    issues = ezm.validate_unit(dict(OPTIONS, quality_threshold=1.5),
                               geometry())
    assert [item["field"] for item in issues] == ["options.quality_threshold"]

    # 起始边界：必须是一个点，且不能落在圆心。
    centered = geometry(start={"type": "point",
                               "points": [[100.0, 100.0]]})
    assert [item["field"] for item in ezm.validate_unit(
        dict(OPTIONS), centered, complete=True)] == ["geometry.open_boundary"]
    two_points = geometry(start={"type": "point",
                                 "points": [[160.0, 100.0], [161.0, 100.0]]})
    assert [item["field"] for item in ezm.validate_unit(
        dict(OPTIONS), two_points)] == ["geometry.open_boundary"]
    line = geometry(start={"type": "line",
                           "points": [[160.0, 100.0]]})
    assert [item["field"] for item in ezm.validate_unit(
        dict(OPTIONS), line)] == ["geometry.open_boundary"]
    # ≥5 点拟合与参数式椭圆等价进入分析。
    five = geometry(arena={"type": "ellipse",
                           "points": [[160.0, 100.0], [100.0, 130.0],
                                      [40.0, 100.0], [100.0, 70.0],
                                      [130.0, 85.0]]})
    assert ezm.validate_unit(dict(OPTIONS), five, complete=True) == []

    # 环上判定：边界余量只由 margin 决定，不额外限制跑道宽与余量关系。
    guides = ezm.boundary_guides(geometry(), OPTIONS)
    assert guides is not None
    assert guides["boundary_angle_deg"] == pytest.approx(0.0, abs=TOL)
    assert guides["rings"]["inner"] == pytest.approx(25.0, abs=TOL)
    assert [item["zone"] for item in guides["labels"]] == [
        "O1", "C1", "O2", "C2"]
    assert guides["labels"][0]["open"] is True
    assert guides["labels"][1]["open"] is False
    # 起始射线沿 +x：外端落在椭圆右端点，内端落在 25/30 的等比位置。
    ray = guides["rays"][0]
    assert ray["to"][0] == pytest.approx(160.0, abs=1e-6)
    assert ray["to"][1] == pytest.approx(100.0, abs=1e-6)
    assert ray["from"][0] == pytest.approx(150.0, abs=1e-6)

    # 预览画布：内外圈、四条分界线与四个标签都由保存几何与有效尺寸算出。
    canvas = np.zeros((200, 200, 3), dtype=np.uint8)
    preview._draw_guides(canvas, "ezm", dict(OPTIONS), geometry(), 2)
    assert int(canvas.sum()) > 0
    # 起始边界点旋转后分界线随之改变，但场地椭圆不变。
    rotated = ezm.boundary_guides(
        geometry(start={"type": "point", "points": [[100.0, 40.0]]}),
        dict(OPTIONS))
    assert rotated["boundary_angle_deg"] == pytest.approx(270.0, abs=1e-6)
    assert rotated["rays"][0]["to"][1] == pytest.approx(70.0, abs=1e-6)


def test_analysis_writes_zones_table_and_unit_metadata(tmp_path, monkeypatch):
    """分析入口：七类表、事件、分箱与每只动物的 metadata.units 都落盘。"""

    from behaviourlab import engine, media, outputs
    from behaviourlab.schema import Frame

    class Source:
        def __init__(self, path, *, start_s=0.0, end_s=None, **kwargs):
            self._frames = []
            positions = [ring_point(45), ring_point(45), ring_point(135),
                         ring_point(135)]
            for index, seen in enumerate(positions):
                moment = float(index)
                image = np.zeros((200, 200, 3), dtype=np.uint8)
                self._frames.append(Frame(
                    image=image, index=index, time_s=moment, dt_s=1.0,
                    elapsed_s=moment, analysis_fps=1.0,
                    interval_start_s=moment, interval_end_s=moment + 1.0))

        def iter_frames(self):
            return iter(self._frames)

    monkeypatch.setattr(media, "VideoSource", Source)
    monkeypatch.setattr(engine, "require_identity", lambda asset: None)
    source_file = tmp_path / "ezm.mp4"
    source_file.write_bytes(b"placeholder")
    job = {
        "schema_version": "1",
        "run_id": "r" * 32,
        "job_id": "j" * 32,
        "assay": "ezm",
        "asset": {"id": "a" * 32, "name": "ezm.mp4",
                  "path": str(source_file),
                  "metadata": {"duration_s": 4.0, "fps": 1.0}},
        "window": {"start_s": 0.0, "end_s": None},
        "orientation": 0,
        "fps_override": None,
        "sample_fps": None,
        "options": dict(OPTIONS),
        "detection": dict(ezm.DETECTION_DEFAULTS),
        "units": [unit()],
        "output": {"annotated_video": False},
        "style": {},
        "algorithm_version": "test",
    }
    result = engine.execute(job)
    assert [row["unit_id"] for row in result.summary] == [UNIT]
    assert len(result.tables["zones"]) == len(ezm.ALL_ZONES)
    assert result.metadata["coordinate_unit"] == "cm"
    assert result.metadata["algorithm_version"]
    assert result.metadata["coverage"]["units"] == 1
    unit_meta = result.metadata["units"][UNIT]
    assert unit_meta["zone_order_keys"] == list(ezm.ZONE_KEYS)
    assert unit_meta["occupancy_denominator"]["open_time_percent_known"]
    # 导出：七类明细进 zones.csv，轨迹与热图按环带绘制。
    manifest = outputs.export_result(tmp_path / "run", job, result)
    files = manifest["files"]
    assert "summary.csv" in files
    info = manifest["units"][0]["figures"]
    assert info["trajectory"] or info["note"]
    assert (tmp_path / "run" / "results.json").is_file()


class _RingSource:
    """内存解码头：一只小鼠沿环带从 O1 走向 C1。"""

    def __init__(self, path, *, start_s=0.0, end_s=None, **kwargs):
        from behaviourlab.schema import Frame

        self._frames = []
        begin = float(start_s or 0.0)
        stop = float(end_s if end_s is not None else begin + 2.0)
        step = 0.1
        count = max(1, int(round((stop - begin) / step)))
        for index in range(count):
            moment = round(begin + index * step, 6)
            image = np.full((200, 200, 3), 255, dtype=np.uint8)
            theta = math.radians(45.0 + 45.0 * index / max(1, count - 1))
            center = (int(round(100.0 + 55.0 * math.cos(theta))),
                      int(round(100.0 + 27.5 * math.sin(theta))))
            cv2.ellipse(image, center, (7, 5), 0, 0, 360, (0, 0, 0), -1)
            self._frames.append(Frame(
                image=image, index=index, time_s=moment, dt_s=step,
                elapsed_s=moment - begin, analysis_fps=1.0 / step,
                interval_start_s=moment, interval_end_s=moment + step))

    def iter_frames(self):
        return iter(self._frames)


def test_workspace_flow_writes_ezm_results(client, store, tmp_path, monkeypatch):
    """从草稿到结果页：预览、行为预览、分析与 zones.csv 全部可达。"""

    import csv

    from behaviourlab import media, preview
    from behaviourlab.media import FrameInfo

    source = tmp_path / "ezm-ui.mp4"
    source.write_bytes(b"placeholder")
    asset = store.create_asset(
        name="ezm-ui.mp4", path=str(source), project_id=None,
        metadata={"width": 200, "height": 200, "duration_s": 2.0,
                  "fps": 10.0, "frame_count": 20})
    draft = store.create_draft("ezm", videos=[{
        "asset_id": asset["id"],
        "window": {"start_s": 0.0, "end_s": 2.0},
        "orientation": 0, "fps_override": None, "sample_fps": 10,
        "units": [{"id": "z" * 32, "name": "EZM 鼠", "window": None,
                   "geometry": geometry(),
                   "options": {"quality_threshold": 0.4}, "detection": {},
                   "manual": None}],
    }])
    monkeypatch.setattr(media, "VideoSource", _RingSource)
    monkeypatch.setattr(
        "behaviourlab.engine.media.probe_video",
        lambda path: {"width": 200, "height": 200, "duration_s": 2.0,
                      "fps": 10.0, "frame_count": 20})
    monkeypatch.setattr(
        preview.media, "read_frame_at_time",
        lambda *args, **kwargs: FrameInfo(
            image=np.full((200, 200, 3), 255, dtype=np.uint8), index=10,
            time_s=1.0, pts=None, time_base=None, clock_source="pts"))
    monkeypatch.setattr(
        preview.media, "read_frame_info",
        lambda *args, **kwargs: FrameInfo(
            image=np.full((200, 200, 3), 255, dtype=np.uint8), index=10,
            time_s=1.0, pts=None, time_base=None, clock_source="pts"))

    detection = client.post(
        f"/api/drafts/{draft['id']}/preview",
        json={"asset_id": asset["id"], "unit_id": "z" * 32,
              "frame_index": 10, "kind": "detection"})
    assert detection.status_code == 201, detection.get_json()
    data = detection.get_json()["data"]
    assert data["summary"]["drawn_steps"] == ["arena", "open_boundary"]
    assert data["summary"]["missing_steps"] == []
    assert client.get(data["image_url"]).status_code == 200

    behavior = client.post(
        f"/api/drafts/{draft['id']}/preview",
        json={"asset_id": asset["id"], "unit_id": "z" * 32,
              "frame_index": 10, "kind": "behavior"})
    assert behavior.status_code == 201, behavior.get_json()
    core = behavior.get_json()["data"]["summary"]["behavior"]
    assert core["mode"] == "ezm"
    assert core["rows"] > 0
    for key in ("open_time_s", "closed_time_s", "boundary_time_s",
                "outside_time_s", "unknown_time_s", "open_time_percent_total",
                "open_time_percent_known", "first_open_observation_latency_s",
                "first_open_entry_latency_s", "first_open_entry_censored"):
        assert key in core

    started = client.post(f"/api/drafts/{draft['id']}/analyze")
    assert started.status_code == 201, started.get_json()
    payload = started.get_json()["data"]
    job = payload["jobs"][0]
    assert job["status"] == "done"
    summary = job["manifest"]["units"][0]["summary"]
    assert summary["total_time_s"] == pytest.approx(2.0, abs=1e-6)
    assert summary["observed_time_s"] > 0
    assert summary["open_time_s"] + summary["closed_time_s"] > 0
    assert summary["coordinate_unit"] == "cm"
    figures = job["manifest"]["units"][0]["figures"]
    assert figures["trajectory"]
    assert figures["heatmap"]

    run_dir = Path(store.paths.runs) / payload["run_id"] / job["job_id"]
    zones_file = run_dir / "zones.csv"
    assert zones_file.is_file()
    with zones_file.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["zone"] for row in rows] == list(ezm.ALL_ZONES)
    assert client.get(
        f"/runs/{payload['run_id']}/{job['job_id']}/zones.csv").status_code == 200
    for name in (figures["trajectory"], figures["heatmap"]):
        assert client.get(
            f"/runs/{payload['run_id']}/{job['job_id']}/{name}").status_code == 200

    page = client.get(f"/results/{payload['run_id']}").get_data(as_text=True)
    for label in ("开放区时间", "封闭区时间", "边界时间", "区外时间", "未知时间",
                  "开放区时间占总时长", "开放区时间占开放+封闭",
                  "开放区进入", "封闭区进入", "首次观察开放区潜伏期",
                  "首次确认进入开放区潜伏期", "zones.csv"):
        assert label in page, label


def test_editor_bridge_saves_ring_geometry():
    """Node 编辑产物：外缘与起始边界落在同一 arena/point 结构并通过校验。"""

    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node 不可用")
    script = Path(__file__).parent / "js" / "test_ezm_editor.cjs"
    result = subprocess.run([node, str(script)], capture_output=True,
                            text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "ezm-editor OK" in result.stdout
    assert ezm.validate_unit(dict(OPTIONS), geometry(), complete=True) == []


def test_workspace_and_preview_wire_ezm_guides():
    """真实页面资源：画布与预览都调用同一套环上方向计算。"""

    root = Path(__file__).resolve().parents[1] / "behaviourlab" / "web"
    workspace = (root / "static" / "workspace.js").read_text(encoding="utf-8")
    assert 'if (state.assay === "ezm") drawEzmGuides(unit);' in workspace
    assert "BLCore.ezmGuides(" in workspace
    assert '"outer_diameter_cm"' in workspace and '"ring_width_cm"' in workspace

    core = (root / "static" / "editor-core.js").read_text(encoding="utf-8")
    assert "function ezmGuides(arena, boundary, diameterCm, ringWidthCm)" in core
    assert "    ezmGuides," in core
    assert 'behavior.mode === "ezm"' in core

    preview_source = (Path(__file__).resolve().parents[1]
                      / "behaviourlab" / "preview.py").read_text(encoding="utf-8")
    assert 'experiments.get("ezm").boundary_guides(unit_geometry, options)' \
        in preview_source
    assert "_draw_guides(canvas, draft.get(\"assay\", \"\"), options" \
        in preview_source


# -- 有限修正一：已提供但类型错误的几何在四个入口都被拒绝 ----------------

WRONG_ARENA = {"type": "polygon", "center": [100.0, 100.0],
               "radii": [60.0, 30.0], "angle": 0.0}
WRONG_MASK = {"type": "line", "points": [[10, 10], [90, 10], [90, 90], [10, 90]]}
GOOD_MASK = {"type": "polygon", "points": [[10, 10], [90, 10], [90, 90], [10, 90]]}


def test_wrong_geometry_types_are_rejected_everywhere(client, store):
    from behaviourlab import engine
    from behaviourlab.schema import AnalysisContext, ValidationError

    bad_arena = geometry(arena=WRONG_ARENA)
    issues = ezm.validate_unit(dict(OPTIONS), bad_arena, complete=True)
    assert [item["field"] for item in issues] == ["geometry.arena"]
    assert "ellipse" in issues[0]["message"]
    # 未完成的草稿（缺几何）仍可保存；类型错误与“尚未绘制”不同。
    assert ezm.validate_unit(dict(OPTIONS), {}) == []
    assert ezm.validate_unit(dict(OPTIONS), {"arena": WRONG_ARENA}) == [
        {"field": "geometry.arena", "message": "外缘必须是 ellipse 类型"}]

    bad_mask = geometry(mask=WRONG_MASK)
    assert ezm.validate_unit(dict(OPTIONS), bad_mask, complete=True) == [
        {"field": "geometry.mask", "message": "检测遮罩必须是 polygon 类型"}]
    # 正确的可选遮罩保持原行为。
    assert ezm.validate_unit(dict(OPTIONS), geometry(mask=GOOD_MASK),
                             complete=True) == []

    # 保存：POST/PUT 都拒绝，字段指向具体步骤。
    created = client.post("/api/drafts", json={
        "assay": "ezm",
        "videos": [{"asset_id": "a" * 32,
                    "units": [{"name": "小鼠 1", "geometry": bad_arena}]}],
    })
    assert created.status_code == 400
    assert created.get_json()["error"]["fields"][0]["field"].endswith(
        "geometry.arena")

    draft = store.create_draft("ezm", videos=[{
        "asset_id": "a" * 32,
        "units": [{"id": "e" * 32, "name": "小鼠 1", "geometry": geometry()}],
    }])
    updated = client.put(f"/api/drafts/{draft['id']}", json={"videos": [{
        "asset_id": "a" * 32,
        "units": [{"id": "e" * 32, "name": "小鼠 1",
                   "geometry": geometry(arena=WRONG_ARENA)}],
    }]})
    assert updated.status_code == 400
    assert updated.get_json()["error"]["fields"][0]["field"].endswith(
        "geometry.arena")

    # 未完成草稿仍可保存并进入预览。
    incomplete = store.create_draft("ezm", videos=[{
        "asset_id": "a" * 32,
        "units": [{"id": "e" * 32, "name": "小鼠 1", "geometry": {}}],
    }])
    assert client.put(f"/api/drafts/{incomplete['id']}",
                      json={"videos": incomplete["videos"]}).status_code == 200

    # 最终校验与分析入口同样拒绝。
    validate = client.get(f"/api/drafts/{draft['id']}/validate")
    assert validate.status_code == 200
    bad_draft = {**draft, "videos": [{
        "asset_id": "a" * 32,
        "units": [{"id": "e" * 32, "name": "小鼠 1",
                   "geometry": geometry(arena=WRONG_ARENA)}],
    }]}
    assert [item["field"] for item in engine.validate_draft(
        bad_draft, complete=True)] == ["videos[0].units[0].geometry.arena"]

    context = AnalysisContext(job={"assay": "ezm", "units": [unit(
        geometric=bad_arena)]})
    with pytest.raises(ValidationError) as captured:
        ezm.analyze(context)
    assert captured.value.field == "geometry.arena"

    context_mask = AnalysisContext(job={"assay": "ezm", "units": [unit(
        geometric=bad_mask)]})
    with pytest.raises(ValidationError) as captured_mask:
        ezm.analyze(context_mask)
    assert captured_mask.value.field == "geometry.mask"


# -- 有限修正一：已提供但类型错误的几何在四个入口都被拒绝 ----------------

WRONG_ARENA = {"type": "polygon", "center": [100.0, 100.0],
               "radii": [60.0, 30.0], "angle": 0.0}
WRONG_MASK = {"type": "line", "points": [[10, 10], [90, 10], [90, 90], [10, 90]]}
GOOD_MASK = {"type": "polygon", "points": [[10, 10], [90, 10], [90, 90], [10, 90]]}


def test_wrong_geometry_types_are_rejected_everywhere(client, store):
    from behaviourlab import engine
    from behaviourlab.schema import AnalysisContext, ValidationError

    bad_arena = geometry(arena=WRONG_ARENA)
    issues = ezm.validate_unit(dict(OPTIONS), bad_arena, complete=True)
    assert [item["field"] for item in issues] == ["geometry.arena"]
    assert "ellipse" in issues[0]["message"]
    # 未完成的草稿（缺几何）仍可保存；类型错误与“尚未绘制”不同。
    assert ezm.validate_unit(dict(OPTIONS), {}) == []
    assert ezm.validate_unit(dict(OPTIONS), {"arena": WRONG_ARENA}) == [
        {"field": "geometry.arena", "message": "外缘必须是 ellipse 类型"}]

    bad_mask = geometry(mask=WRONG_MASK)
    assert ezm.validate_unit(dict(OPTIONS), bad_mask, complete=True) == [
        {"field": "geometry.mask", "message": "检测遮罩必须是 polygon 类型"}]
    # 正确的可选遮罩保持原行为。
    assert ezm.validate_unit(dict(OPTIONS), geometry(mask=GOOD_MASK),
                             complete=True) == []

    # 保存：POST/PUT 都拒绝，字段指向具体步骤。
    created = client.post("/api/drafts", json={
        "assay": "ezm",
        "videos": [{"asset_id": "a" * 32,
                    "units": [{"name": "小鼠 1", "geometry": bad_arena}]}],
    })
    assert created.status_code == 400
    assert created.get_json()["error"]["fields"][0]["field"].endswith(
        "geometry.arena")

    draft = store.create_draft("ezm", videos=[{
        "asset_id": "a" * 32,
        "units": [{"id": "e" * 32, "name": "小鼠 1", "geometry": geometry()}],
    }])
    updated = client.put(f"/api/drafts/{draft['id']}", json={"videos": [{
        "asset_id": "a" * 32,
        "units": [{"id": "e" * 32, "name": "小鼠 1",
                   "geometry": geometry(arena=WRONG_ARENA)}],
    }]})
    assert updated.status_code == 400
    assert updated.get_json()["error"]["fields"][0]["field"].endswith(
        "geometry.arena")

    # 未完成草稿仍可保存并进入预览。
    incomplete = store.create_draft("ezm", videos=[{
        "asset_id": "a" * 32,
        "units": [{"id": "e" * 32, "name": "小鼠 1", "geometry": {}}],
    }])
    assert client.put(f"/api/drafts/{incomplete['id']}",
                      json={"videos": incomplete["videos"]}).status_code == 200

    # 最终校验与分析入口同样拒绝。
    validate = client.get(f"/api/drafts/{draft['id']}/validate")
    assert validate.status_code == 200
    bad_draft = {**draft, "videos": [{
        "asset_id": "a" * 32,
        "units": [{"id": "e" * 32, "name": "小鼠 1",
                   "geometry": geometry(arena=WRONG_ARENA)}],
    }]}
    assert [item["field"] for item in engine.validate_draft(
        bad_draft, complete=True)] == ["videos[0].units[0].geometry.arena"]

    context = AnalysisContext(job={"assay": "ezm", "units": [unit(
        geometric=bad_arena)]})
    with pytest.raises(ValidationError) as captured:
        ezm.analyze(context)
    assert captured.value.field == "geometry.arena"

    context_mask = AnalysisContext(job={"assay": "ezm", "units": [unit(
        geometric=bad_mask)]})
    with pytest.raises(ValidationError) as captured_mask:
        ezm.analyze(context_mask)
    assert captured_mask.value.field == "geometry.mask"


# -- 有限修正二：近圆心判据统一按 D=1 ------------------------------------

LEGAL_START = {"type": "point", "points": [[100.0000024, 100.0]]}
ILLEGAL_START = {"type": "point", "points": [[100.0000006, 100.0]]}
SMALL = {"outer_diameter_cm": 0.1, "ring_width_cm": 1.0 / 120.0}


@pytest.mark.parametrize("sizes,zone", [
    # 0.5 cm 的边界余量在 0.05 cm 半径的环上覆盖全程：占用结果不同是物理事实，
    # 本轮只要求“有效性判据与方向”不随外径改变。
    (SMALL, "boundary"),
    ({"outer_diameter_cm": 60.0, "ring_width_cm": 5.0}, "O1"),
])
def test_near_center_rule_is_independent_of_outer_diameter(sizes, zone):
    from behaviourlab.schema import ValidationError

    options = dict(OPTIONS, **sizes)
    legal = geometry(start=LEGAL_START)
    illegal = geometry(start=ILLEGAL_START)

    # D=1 映射下范数约 2e-8：任何实际外径下都合法，方向恒为 +x。
    assert ezm.validate_unit(options, legal, complete=True) == []
    assert ezm.boundary_angle(legal) == pytest.approx(0.0, abs=1e-15)
    guides = ezm.boundary_guides(legal, options)
    assert guides is not None
    assert guides["boundary_angle_deg"] == pytest.approx(0.0, abs=1e-12)
    facts = ezm.process_unit(unit(options=options, geometric=legal),
                             source_rows([ring_point(45)]), window(1), 0.0)
    assert facts["rows"][0]["zone"] == zone
    assert facts["summary"]["boundary_angle_deg"] == pytest.approx(
        0.0, abs=1e-12)

    # 范数约 5e-9：两组尺寸下都拒绝，字段指向起始边界。
    fields = [item["field"] for item in ezm.validate_unit(
        options, illegal, complete=True)]
    assert fields == ["geometry.open_boundary"]
    assert ezm.boundary_angle(illegal) is None
    assert ezm.boundary_guides(illegal, options) is None
    with pytest.raises(ValidationError) as captured:
        ezm.process_unit(unit(options=options, geometric=illegal),
                         source_rows([ring_point(45)]), window(1), 0.0)
    assert captured.value.field == "geometry.open_boundary"


def test_preview_rejects_wrong_geometry_types(client, store, tmp_path,
                                              monkeypatch):
    from behaviourlab import preview
    from behaviourlab.media import FrameInfo

    monkeypatch.setattr(
        preview.media, "read_frame_info",
        lambda *args, **kwargs: FrameInfo(
            image=np.full((200, 200, 3), 255, dtype=np.uint8), index=0,
            time_s=0.0, pts=None, time_base=None, clock_source="pts"))

    source = tmp_path / "ezm-types.mp4"
    source.write_bytes(b"placeholder")
    asset = store.create_asset(name="ezm-types.mp4", path=str(source),
                               project_id=None,
                               metadata={"width": 200, "height": 200,
                                         "duration_s": 1.0, "fps": 10.0,
                                         "frame_count": 10})

    def make(bad_geometry):
        return store.create_draft("ezm", videos=[{
            "asset_id": asset["id"],
            "units": [{"id": "e" * 32, "name": "小鼠 1",
                       "geometry": bad_geometry}],
        }])

    for bad_geometry, field in (
            (geometry(arena=WRONG_ARENA), "geometry.arena"),
            (geometry(mask=WRONG_MASK), "geometry.mask")):
        draft = make(bad_geometry)
        for kind in ("detection", "behavior"):
            response = client.post(
                f"/api/drafts/{draft['id']}/preview",
                json={"asset_id": asset["id"], "unit_id": "e" * 32,
                      "frame_index": 0, "kind": kind})
            assert response.status_code == 400, (kind, response.get_json())
            assert response.get_json()["error"]["fields"][0]["field"] == field

    # 正确类型与“尚未绘制”都能正常预览。
    for good in (geometry(), geometry(mask=GOOD_MASK)):
        draft = make(good)
        response = client.post(
            f"/api/drafts/{draft['id']}/preview",
            json={"asset_id": asset["id"], "unit_id": "e" * 32,
                  "frame_index": 0, "kind": "detection"})
        assert response.status_code == 201, response.get_json()
