"""Barnes 迷宫：圆盘归一化、洞访问、首次目标与覆盖的固定夹具。"""

from __future__ import annotations

import math

import pytest

from behaviourlab import timeline
from behaviourlab.experiments import barnes

UNIT = "u" * 32

DISK = {"type": "ellipse", "center": [100.0, 100.0],
        "radii": [45.0, 45.0], "angle": 0.0}


def geometry(holes, disk=DISK, **extra):
    payload = {"arena": disk, "holes": holes}
    payload.update(extra)
    return payload


def unit(*, holes=None, options=None, manual=None):
    merged = dict(barnes.DEFAULTS)
    if options:
        merged.update(options)
    return {"id": UNIT, "name": "小鼠 1", "animal_id": None, "window": None,
            "geometry": geometry(holes if holes is not None else [
                {"id": "1", "center": [130.0, 100.0]},
                {"id": "2", "center": [100.0, 130.0]},
                {"id": "3", "center": [70.0, 100.0]},
            ]),
            "options": merged, "detection": {}, "manual": manual}


def rows_at(entries, *, quality=0.9, head_quality=0.9):
    """显式曝光行：(time_s, 起点, 终点, 身体位置, 鼻尖位置)。"""

    rows = []
    for index, (time_s, start, end, body, nose) in enumerate(entries):
        detected = body is not None
        has_nose = detected and nose is not None
        rows.append({
            "unit_id": UNIT, "unit_name": "小鼠 1", "animal_id": None,
            "frame": index, "time_s": float(time_s), "dt_s": float(end - start),
            "source_dt_s": float(end - start), "elapsed_s": float(time_s),
            "analysis_fps": 1.0,
            "source_interval_start_s": float(start),
            "source_interval_end_s": float(end),
            "interval_start_s": float(start), "interval_end_s": float(end),
            "in_window": True, "detected": detected,
            "x_px": float(body[0]) if detected else None,
            "y_px": float(body[1]) if detected else None,
            "quality": quality if detected else 0.0,
            "contour": [], "missing_reason": None,
            "head_angle": None,
            "head_quality": head_quality if has_nose else 0.0,
            "nose_x_px": float(nose[0]) if has_nose else None,
            "nose_y_px": float(nose[1]) if has_nose else None,
            "tail_x_px": None, "tail_y_px": None,
            "left_ear_x_px": None, "left_ear_y_px": None,
            "right_ear_x_px": None, "right_ear_y_px": None,
        })
    return rows


def fixture_rows():
    """12 行 0.1 秒曝光：洞2、洞2、中心、中心、洞2、洞2、洞3、洞3、洞1、洞1、缺失、中心。"""

    positions = [
        (100, 130), (100, 130), (100, 100), (100, 100),
        (100, 130), (100, 130), (70, 100), (70, 100),
        (130, 100), (130, 100), None, (100, 100),
    ]
    entries = []
    for index, seen in enumerate(positions):
        start = round(0.1 * index, 10)
        entries.append((start, start, round(start + 0.1, 10),
                        seen, seen))
    return rows_at(entries)


def process(rows, *, unit_payload, window=(0.0, 1.2)):
    return barnes.process_unit(unit_payload, rows,
                               timeline.Window(*window), None)


# -- 夹具 1：固定序列、首次目标与覆盖 -----------------------------------

def test_barnes_fixture_sequence_and_primary():
    options = {"hole_count": 3, "target_hole": "1",
               "visit_confirm_s": 0.1, "visit_reset_s": 0.1,
               "quality_threshold": 0.9, "bin_seconds": 0.2}
    result = process(fixture_rows(), unit_payload=unit(options=options))
    summary = result["summary"]

    visits = [item for item in result["events"]
              if item["kind"] == "hole_visit"]
    assert [(item["hole_id"], item["time_s"], item["confirmed_s"])
            for item in visits] == [("2", 0.0, 0.1), ("2", 0.4, 0.5),
                                    ("3", 0.6, 0.7), ("1", 0.8, 0.9)]
    assert summary["visits"] == 4
    assert summary["unscored_visits"] == 0
    assert summary["post_escape_visits"] == 0
    assert summary["primary_censored"] is False
    assert summary["primary_latency_s"] == pytest.approx(0.8, abs=1e-8)
    assert summary["primary_errors"] == 3
    expected_path = 120.0 + 30.0 * math.sqrt(2.0)
    assert summary["primary_path_cm"] == pytest.approx(expected_path, abs=1e-8)

    assert summary["escape_candidates"] == 1
    assert summary["total_latency_s"] is None
    assert summary["total_errors"] is None
    assert summary["total_path_cm"] is None
    assert summary["total_censored"] is True

    assert summary["nose_coverage_percent"] == pytest.approx(
        1100.0 / 12.0, abs=1e-8)
    assert f"{summary['nose_coverage_percent']:.1f}" == "91.7"
    assert summary["body_coverage_percent"] == pytest.approx(
        1100.0 / 12.0, abs=1e-8)

    # 逐洞表：鼻尖曝光直接累计，不要求先形成访问事件。
    holes = {item["hole_id"]: item for item in result["tables"]["holes"]}
    assert holes["1"]["target"] is True
    assert holes["1"]["nose_time_s"] == pytest.approx(0.2, abs=1e-8)
    assert holes["2"]["nose_time_s"] == pytest.approx(0.4, abs=1e-8)
    assert holes["3"]["nose_time_s"] == pytest.approx(0.2, abs=1e-8)
    assert holes["1"]["visits"] == 1
    assert holes["2"]["visits"] == 2
    assert holes["2"]["x_cm"] == pytest.approx(0.0, abs=1e-8)
    assert holes["2"]["y_cm"] == pytest.approx(30.0, abs=1e-8)

    # 目标扇区：中心与目标方向的合格身体曝光。
    assert summary["target_sector_time_s"] == pytest.approx(0.5, abs=1e-8)
    assert summary["outside_time_s"] == pytest.approx(0.0, abs=1e-8)
    assert summary["total_time_s"] == pytest.approx(1.2, abs=1e-8)
    assert summary["missing_time_s"] == pytest.approx(0.1, abs=1e-8)

    # 分箱：观测/缺失/身体/鼻尖/扇区按曝光切分，距离按段比例切分。
    bins = result["bins"]
    assert len(bins) == 6
    assert [item["start_s"] for item in bins] == pytest.approx(
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], abs=1e-8)
    assert sum(item["observed_time_s"] for item in bins) == pytest.approx(
        1.1, abs=1e-8)
    assert sum(item["missing_time_s"] for item in bins) == pytest.approx(
        0.1, abs=1e-8)
    assert sum(item["nose_time_s"] for item in bins) == pytest.approx(
        1.1, abs=1e-8)
    assert sum(item["distance_cm"] for item in bins) == pytest.approx(
        summary["total_distance_cm"], abs=1e-8)
    assert summary["total_distance_cm"] == pytest.approx(expected_path, abs=1e-8)
    assert summary["measurable_motion_time_s"] == pytest.approx(0.9, abs=1e-8)

    # 自动访问表的原记录与事件一致。
    automatic = result["tables"]["automatic_hole_visits"]
    assert [item["hole_id"] for item in automatic] == ["2", "2", "3", "1"]
    assert all(item["source"] == "automatic" for item in automatic)

    warning = result["warning"]
    assert warning and "自动洞访问仅来自可用鼻尖观测" in warning
    assert "91.7%" in warning


# -- 夹具 2：确认需要更晚观测，退出需要连续洞外 -------------------------

def two_holes():
    return [{"id": "1", "center": [130.0, 100.0]},
            {"id": "2", "center": [70.0, 100.0]}]


def test_barnes_visit_needs_later_observation():
    options = {"hole_count": 2, "target_hole": "1",
               "visit_confirm_s": 0.4, "visit_reset_s": 0.4,
               "quality_threshold": 0.4}
    payload = unit(holes=two_holes(), options=options)

    # 一条 0.6 秒的长曝光本身不能确认。
    single = rows_at([(0.0, 0.0, 0.6, (130, 100), (130, 100))])
    facts = process(single, unit_payload=payload, window=(0.0, 0.6))
    assert [item for item in facts["events"]
            if item["kind"] == "hole_visit"] == []

    # 同一洞的两条真实相邻观测：time=0、confirmed=0.6。
    pair = rows_at([(0.0, 0.0, 0.6, (130, 100), (130, 100)),
                    (0.6, 0.6, 0.7, (130, 100), (130, 100))])
    facts = process(pair, unit_payload=payload, window=(0.0, 0.7))
    visits = [item for item in facts["events"] if item["kind"] == "hole_visit"]
    assert [(item["time_s"], item["confirmed_s"]) for item in visits] == [
        (0.0, 0.6)]
    assert facts["summary"]["visits"] == 1


def test_barnes_exit_needs_continuous_outside_time():
    options = {"hole_count": 2, "target_hole": "1",
               "visit_confirm_s": 0.4, "visit_reset_s": 0.4,
               "quality_threshold": 0.4}
    payload = unit(holes=two_holes(), options=options)
    in_hole = (130, 100)
    outside = (100, 100)

    # 一次洞外 + 回到同洞：不重置，不产生第二次访问。
    rows = rows_at([
        (0.0, 0.0, 0.6, in_hole, in_hole),
        (0.6, 0.6, 0.7, in_hole, in_hole),
        (0.7, 0.7, 0.8, outside, outside),
        (0.8, 0.8, 1.0, in_hole, in_hole),
    ])
    facts = process(rows, unit_payload=payload, window=(0.0, 1.0))
    visits = [item for item in facts["events"] if item["kind"] == "hole_visit"]
    assert len(visits) == 1

    # 两条连续洞外间隔 0.6 秒：允许确认后的重复访问。
    rows = rows_at([
        (0.0, 0.0, 0.6, in_hole, in_hole),
        (0.6, 0.6, 0.7, in_hole, in_hole),
        (0.7, 0.7, 1.3, outside, outside),
        (1.3, 1.3, 1.4, outside, outside),
        (1.4, 1.4, 1.8, in_hole, in_hole),
        (1.8, 1.8, 1.9, in_hole, in_hole),
    ])
    facts = process(rows, unit_payload=payload, window=(0.0, 1.9))
    visits = [item for item in facts["events"] if item["kind"] == "hole_visit"]
    assert [(item["time_s"], item["confirmed_s"]) for item in visits] == [
        (0.0, 0.6), (1.4, 1.8)]
    assert facts["summary"]["visits"] == 2

    # 缺鼻尖保持 active：不制造退出，也不产生新访问。
    rows = rows_at([
        (0.0, 0.0, 0.6, in_hole, in_hole),
        (0.6, 0.6, 0.7, in_hole, in_hole),
        (0.7, 0.7, 0.8, in_hole, None),
        (0.8, 0.8, 1.0, in_hole, in_hole),
    ])
    facts = process(rows, unit_payload=payload, window=(0.0, 1.0))
    visits = [item for item in facts["events"] if item["kind"] == "hole_visit"]
    assert len(visits) == 1
    assert facts["summary"]["nose_coverage_percent"] == pytest.approx(
        90.0, abs=1e-8)


# -- 夹具 3：零鼻尖覆盖与人工覆盖 ---------------------------------------

def test_barnes_zero_nose_coverage_and_manual_override():
    options = {"hole_count": 2, "target_hole": "1",
               "visit_confirm_s": 0.1, "visit_reset_s": 0.1,
               "quality_threshold": 0.4}
    rows = rows_at([(0.0, 0.0, 1.0, (100, 100), None),
                    (1.0, 1.0, 2.0, (100, 100), None)])

    automatic = process(rows, unit_payload=unit(holes=two_holes(),
                                                options=options),
                        window=(0.0, 2.0))
    summary = automatic["summary"]
    assert summary["visit_source"] == "automatic"
    assert summary["nose_coverage_percent"] == pytest.approx(0.0, abs=1e-8)
    assert summary["visits"] == 0
    assert automatic["warning"] and "自动洞访问仅来自可用鼻尖观测" in automatic["warning"]
    assert "0.0%" in automatic["warning"]

    manual = process(rows, unit_payload=unit(holes=two_holes(),
                                             options=options,
                                             manual={"hole_visits": []}),
                     window=(0.0, 2.0))
    summary = manual["summary"]
    assert summary["visit_source"] == "manual"
    assert summary["visits"] == 0
    assert summary["nose_coverage_percent"] == pytest.approx(0.0, abs=1e-8)
    assert manual["warning"] is None


# -- 几何与参数校验 -----------------------------------------------------

def test_barnes_validation_rules():
    holes = two_holes()
    options = dict(barnes.DEFAULTS, hole_count=2, target_hole="1")
    assert barnes.validate_unit(options, geometry(holes), complete=True) == []

    mismatch = dict(options, hole_count=3)
    issues = barnes.validate_unit(mismatch, geometry(holes), complete=True)
    assert any(item["field"] == "geometry.holes" for item in issues)

    missing_target = dict(options, target_hole="9")
    issues = barnes.validate_unit(missing_target, geometry(holes),
                                  complete=True)
    assert any(item["field"] == "options.target_hole" for item in issues)

    duplicate = [{"id": "1", "center": [130.0, 100.0]},
                 {"id": "1", "center": [70.0, 100.0]}]
    issues = barnes.validate_unit(options, geometry(duplicate), complete=True)
    assert any(item["field"].endswith(".id") for item in issues)

    bad_count = dict(options, hole_count=1)
    issues = barnes.validate_unit(bad_count, geometry(holes))
    assert any(item["field"] == "options.hole_count" for item in issues)

    fractional = dict(options, hole_count=2.5)
    issues = barnes.validate_unit(fractional, geometry(holes))
    assert any(item["field"] == "options.hole_count" for item in issues)

    # 未画完的草稿仍可保存；最终校验才要求圆盘与洞位。
    assert barnes.validate_unit(options, {}) == []
    issues = barnes.validate_unit(options, {}, complete=True)
    fields = [item["field"] for item in issues]
    assert "geometry.arena" in fields and "geometry.holes" in fields


def test_barnes_affine_mapping_keeps_image_axes():
    ellipse = ([100.0, 100.0], [45.0, 45.0], 0.0)
    matrix = barnes.affine_matrix(ellipse, 90.0)
    # 1 cm/px，图像 y 向下：圆周点映到半径 45 的圆盘边界。
    assert barnes.to_cm(matrix, ellipse[0], (130.0, 100.0)) == pytest.approx(
        (30.0, 0.0), abs=1e-8)
    assert barnes.to_cm(matrix, ellipse[0], (100.0, 130.0)) == pytest.approx(
        (0.0, 30.0), abs=1e-8)
    rotated = ([100.0, 100.0], [45.0, 20.0], 90.0)
    rotated_matrix = barnes.affine_matrix(rotated, 90.0)
    # 90° 旋转后长半轴沿画面 y 方向：椭圆边界点仍映到 45 cm。
    x, y = barnes.to_cm(rotated_matrix, rotated[0], (100.0, 145.0))
    assert math.hypot(x, y) == pytest.approx(45.0, abs=1e-8)


# -- 端到端：作业、导出与结果页 -----------------------------------------

DISK = {"type": "ellipse", "center": [100.0, 100.0],
        "radii": [45.0, 45.0], "angle": 0.0}
THREE_HOLES = [{"id": "1", "center": [130.0, 100.0]},
               {"id": "2", "center": [70.0, 100.0]},
               {"id": "3", "center": [100.0, 130.0]}]


def _asset_and_draft(store, tmp_path, *, geometry_payload=None, options=None):
    source = tmp_path / "barnes.mp4"
    source.write_bytes(b"placeholder")
    asset = store.create_asset(
        name="barnes.mp4", path=str(source), project_id=None,
        metadata={"width": 200, "height": 200, "duration_s": 4.0,
                  "fps": 5.0, "frame_count": 20})
    merged = dict(barnes.DEFAULTS, hole_count=3, target_hole="1")
    if options:
        merged.update(options)
    payload = geometry_payload if geometry_payload is not None else {
        "arena": DISK, "holes": THREE_HOLES}
    draft = store.create_draft("barnes", videos=[{
        "asset_id": asset["id"],
        "window": {"start_s": 0.0, "end_s": 4.0},
        "orientation": 0, "fps_override": None, "sample_fps": None,
        "units": [{"id": UNIT, "name": "小鼠 1", "animal_id": None,
                   "window": None, "geometry": payload,
                   "options": merged, "detection": {}, "manual": None}],
    }])
    return asset, draft


def _patch_frames(monkeypatch):
    import numpy as np

    from behaviourlab import media
    from behaviourlab.schema import Detection, Frame

    class Frames:
        def __init__(self, path=None, *, start_s=0.0, end_s=None, **kwargs):
            self._frames = []
            for index in range(20):
                moment = round(0.2 * index, 10)
                self._frames.append(Frame(
                    image=np.full((200, 200), 240, dtype=np.uint8),
                    index=index, time_s=moment, dt_s=0.2, elapsed_s=moment,
                    analysis_fps=5.0, interval_start_s=moment,
                    interval_end_s=moment + 0.2))

        def iter_frames(self):
            return iter(self._frames)

    def track(run_key, unit_id, *, frame_index, **kwargs):
        nose = (130.0, 100.0) if frame_index < 10 else (70.0, 100.0)
        return Detection(center=(100.0, 100.0), quality=0.9, contour=[],
                         mask=None, nose=nose, tail_base=(100.0, 120.0),
                         head_quality=0.9)

    monkeypatch.setattr(media, "VideoSource", Frames)
    monkeypatch.setattr("behaviourlab.engine.media.probe_video",
                        lambda path: {"width": 200, "height": 200,
                                      "duration_s": 4.0, "fps": 5.0,
                                      "frame_count": 20})
    monkeypatch.setattr(barnes.observation, "track", track)


def test_barnes_job_exports_and_result_page(client, store, tmp_path,
                                            monkeypatch):
    _asset, draft = _asset_and_draft(store, tmp_path)
    _patch_frames(monkeypatch)

    payload = client.post("/api/jobs",
                          json={"draft_id": draft["id"]}).get_json()["data"]
    job = payload["jobs"][0]
    assert job["status"] == "done", job
    summary = job["manifest"]["units"][0]["summary"]
    assert summary["visits"] == 2
    assert summary["primary_latency_s"] == pytest.approx(0.0, abs=1e-8)
    assert summary["primary_errors"] == 0
    assert summary["target_hole"] == "1"
    assert summary["nose_coverage_percent"] == pytest.approx(100.0, abs=1e-8)
    assert summary["total_latency_s"] is None
    assert summary["total_censored"] is True
    assert summary["escape_candidates"] == 0

    run_id, job_id = payload["run_id"], job["job_id"]
    results = client.get(f"/runs/{run_id}/{job_id}/results.json").get_json()
    holes = results["tables"]["holes"]
    assert [item["hole_id"] for item in holes] == ["1", "2", "3"]
    assert holes[0]["target"] is True and holes[0]["visits"] == 1
    assert holes[1]["automatic_visits"] == 1
    automatic = results["tables"]["automatic_hole_visits"]
    assert [(item["hole_id"], item["time_s"]) for item in automatic] == [
        ("1", 0.0), ("2", 2.0)]

    page = client.get(f"/results/{run_id}").get_data(as_text=True)
    for label in ("洞访问", "首次目标潜伏期", "目标扇区", "自动访问",
                  "暂无图表", "洞访问来自鼻尖进入洞半径的证据"):
        assert label in page, label
    assert "没有可用的场地尺寸" not in page
    assert 'src=""' not in page


def test_barnes_preview_steps_scope_and_errors(client, store, tmp_path,
                                               monkeypatch):
    from behaviourlab import preview as preview_service
    from behaviourlab.media import FrameInfo
    import numpy as np

    _asset, draft = _asset_and_draft(store, tmp_path)
    _patch_frames(monkeypatch)

    def fake_frame(*args, **kwargs):
        return FrameInfo(image=np.full((200, 200), 240, dtype=np.uint8),
                         index=0, time_s=0.0, pts=0, time_base=None,
                         clock_source="pts")

    monkeypatch.setattr(preview_service.media, "read_frame_at_time", fake_frame)
    monkeypatch.setattr(preview_service.media, "read_frame_info", fake_frame)

    response = client.post(
        f"/api/drafts/{draft['id']}/preview",
        json={"asset_id": _asset["id"], "unit_id": UNIT,
              "frame_index": 0, "kind": "detection"})
    assert response.status_code == 201, response.get_json()
    summary = response.get_json()["data"]["summary"]
    assert summary["drawn_steps"] == ["arena", "holes"]
    assert summary["missing_steps"] == []
    assert summary["detection"]["region"] == "arena"

    # 自动行为预览只估计当前片段，且摘要标明是片段。
    response = client.post(
        f"/api/drafts/{draft['id']}/preview",
        json={"asset_id": _asset["id"], "unit_id": UNIT,
              "time_s": 0.0, "kind": "behavior"})
    assert response.status_code == 201, response.get_json()
    data = response.get_json()["data"]
    assert data["preview_scope"]["mode"] == "range"
    assert abs(data["preview_scope"]["end_s"] - 3.0) <= 1e-9
    behavior = data["summary"]["behavior"]
    assert behavior["mode"] == "barnes"
    assert behavior["visit_source"] == "automatic"
    assert behavior["visits"] >= 1

    # 缺洞位：最终校验留在场地阶段并定位 geometry.holes。
    _asset2, draft2 = _asset_and_draft(
        store, tmp_path, geometry_payload={"arena": DISK})
    response = client.post(
        f"/api/drafts/{draft2['id']}/preview",
        json={"asset_id": _asset2["id"], "unit_id": UNIT,
              "time_s": 0.0, "kind": "behavior"})
    assert response.status_code == 400, response.get_json()
    fields = [item["field"] for item in response.get_json()["error"]["fields"]]
    assert any(field.endswith("geometry.holes") for field in fields), fields
    issues = client.get(f"/api/drafts/{draft2['id']}/validate") \
        .get_json()["data"]["issues"]
    assert any(item["field"].endswith("geometry.holes") for item in issues)


def test_barnes_editor_bridge_round_trip():
    """Node 复用的编辑器把洞位写成数组，后端直接校验同一份几何。"""

    import json
    import shutil
    import subprocess
    from pathlib import Path

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node 不可用")
    script = Path(__file__).parent / "js" / "test_barnes_editor.cjs"
    result = subprocess.run([node, str(script)], capture_output=True,
                            text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "barnes-editor OK" in result.stdout
    line = [item for item in result.stdout.splitlines()
            if item.startswith("barnes-geometry ")][0]
    geometry = json.loads(line.split(" ", 1)[1])
    assert sorted(geometry) == ["arena", "holes"]
    assert [hole["id"] for hole in geometry["holes"]] == ["1", "2", "3"]
    options = dict(barnes.DEFAULTS, hole_count=3, target_hole="1")
    assert barnes.validate_unit(options, geometry, complete=True) == []


# -- 探测试次与无锚点相对窗 ---------------------------------------------

def test_barnes_probe_drops_escape_fields():
    options = {"hole_count": 3, "target_hole": "1", "trial_type": "probe",
               "visit_confirm_s": 0.1, "visit_reset_s": 0.1,
               "quality_threshold": 0.4}
    rows = fixture_rows()
    facts = process(rows, unit_payload=unit(options=options))
    summary = facts["summary"]
    assert summary["trial_type"] == "probe"
    assert summary["escape_candidates"] == 0
    assert not [item for item in facts["events"]
                if item["kind"] == "escape_candidate"]
    assert summary["total_latency_s"] is None
    assert summary["total_errors"] is None
    assert summary["total_path_cm"] is None
    assert summary["total_censored"] is None


def test_barnes_relative_window_without_anchor_is_empty():
    options = {"hole_count": 3, "target_hole": "1"}
    payload = unit(options=options)
    payload["window"] = {"mode": "relative", "offset_s": 0.0,
                         "duration_s": 2.0}
    rows = rows_at([(0.0, 0.0, 0.2, None, None),
                    (0.2, 0.2, 0.4, None, None)])
    facts = barnes.process_unit(payload, rows, timeline.Window(0.0, 2.0), None)
    summary = facts["summary"]
    assert summary["analysis_window_start_s"] is None
    assert summary["total_time_s"] == pytest.approx(0.0, abs=1e-8)
    assert summary["body_coverage_percent"] is None
    assert summary["nose_coverage_percent"] is None
    assert summary["mean_speed_cm_s"] is None
    assert summary["visits"] == 0
    assert facts["warning"] and "无法确定相对时间窗" in facts["warning"]


# -- 探测试次与无锚点相对窗 ---------------------------------------------

def test_barnes_probe_drops_escape_fields():
    options = {"hole_count": 3, "target_hole": "1", "trial_type": "probe",
               "visit_confirm_s": 0.1, "visit_reset_s": 0.1,
               "quality_threshold": 0.4}
    rows = fixture_rows()
    facts = process(rows, unit_payload=unit(options=options))
    summary = facts["summary"]
    assert summary["trial_type"] == "probe"
    assert summary["escape_candidates"] == 0
    assert not [item for item in facts["events"]
                if item["kind"] == "escape_candidate"]
    assert summary["total_latency_s"] is None
    assert summary["total_errors"] is None
    assert summary["total_path_cm"] is None
    assert summary["total_censored"] is None


def test_barnes_relative_window_without_anchor_is_empty():
    options = {"hole_count": 3, "target_hole": "1"}
    payload = unit(options=options)
    payload["window"] = {"mode": "relative", "offset_s": 0.0,
                         "duration_s": 2.0}
    rows = rows_at([(0.0, 0.0, 0.2, None, None),
                    (0.2, 0.2, 0.4, None, None)])
    facts = barnes.process_unit(payload, rows, timeline.Window(0.0, 2.0), None)
    summary = facts["summary"]
    assert summary["analysis_window_start_s"] is None
    assert summary["total_time_s"] == pytest.approx(0.0, abs=1e-8)
    assert summary["body_coverage_percent"] is None
    assert summary["nose_coverage_percent"] is None
    assert summary["mean_speed_cm_s"] is None
    assert summary["visits"] == 0
    assert facts["warning"] and "无法确定相对时间窗" in facts["warning"]
