"""NOR 新物体识别：探索手算、头向回退与注册/预览/页面接线。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from behaviourlab import experiments, timeline
from behaviourlab.experiments import nor

TOL = 1e-9
ARENA = {"type": "quad",
         "points": [[0.0, 0.0], [50.0, 0.0], [50.0, 50.0], [0.0, 50.0]]}
OBJECTS = [
    {"id": "object_1", "name": "物体 1", "role": "familiar",
     "center": [20.0, 25.0], "zone": None},
    {"id": "object_2", "name": "物体 2", "role": "novel",
     "center": [10.0, 25.0], "zone": None},
]


def geometry(novel=(10.0, 25.0), familiar=(20.0, 25.0)) -> dict:
    return {
        "arena": ARENA,
        "objects": [
            dict(OBJECTS[0], center=list(familiar)),
            dict(OBJECTS[1], center=list(novel)),
        ],
    }


def unit(options=None, geom=None) -> dict:
    return {
        "id": "u" * 32, "name": "小鼠 1", "animal_id": None, "window": None,
        "geometry": geom or geometry(),
        "options": dict(nor.DEFAULTS, **(options or {})),
        "detection": {},
    }


def rows(points, heads=None) -> list[dict]:
    """源观测：1 Hz、每条曝光 1 秒；head 为 (鼻尖, 尾根, 质量) 或 None。"""

    result = []
    for index, point in enumerate(points):
        detected = point is not None
        nose, tail, quality = (None, None, 0.0)
        if heads is not None and heads[index] is not None:
            nose, tail, quality = heads[index]
        result.append({
            "unit_id": "u" * 32, "unit_name": "小鼠 1", "frame": index,
            "time_s": float(index), "source_dt_s": 1.0,
            "elapsed_s": float(index), "analysis_fps": 1.0,
            "source_interval_start_s": float(index),
            "source_interval_end_s": float(index + 1),
            "interval_start_s": float(index),
            "interval_end_s": float(index + 1), "dt_s": 1.0,
            "in_window": True, "detected": detected,
            "x_px": None if not detected else float(point[0]),
            "y_px": None if not detected else float(point[1]),
            "quality": 1.0, "contour": [],
            "missing_reason": None if detected else "no_candidate",
            "head_angle": None, "head_quality": float(quality),
            "nose_x_px": None if nose is None else float(nose[0]),
            "nose_y_px": None if nose is None else float(nose[1]),
            "tail_x_px": None if tail is None else float(tail[0]),
            "tail_y_px": None if tail is None else float(tail[1]),
            "left_ear_x_px": None, "left_ear_y_px": None,
            "right_ear_x_px": None, "right_ear_y_px": None,
            "detection_method": "threshold",
        })
    return result


def test_training_fixture_exploration_and_fallback_warning():
    points = [(10.0, 25.0)] * 6 + [(20.0, 25.0)] * 4
    facts = nor.process_unit(unit(), rows(points), timeline.Window(0.0, 10.0),
                             None)
    summary = facts["summary"]
    # 首条无完成速度不计探索：新 5 秒、熟悉 4 秒、总 9 秒。
    assert summary["object_2_exploration_s"] == pytest.approx(5.0, abs=TOL)
    assert summary["object_1_exploration_s"] == pytest.approx(4.0, abs=TOL)
    assert summary["total_object_exploration_s"] == pytest.approx(9.0, abs=TOL)
    assert summary["object_1_bouts"] == 1
    assert summary["object_2_bouts"] == 1
    assert summary["object_1_latency_s"] == pytest.approx(6.0, abs=TOL)
    assert summary["object_2_latency_s"] == pytest.approx(1.0, abs=TOL)
    assert summary["discrimination_index"] == pytest.approx(1.0 / 9.0, abs=TOL)
    assert summary["preference_index_percent"] == pytest.approx(
        100.0 * 5.0 / 9.0, abs=TOL)
    assert summary["novel_exploration_s"] == pytest.approx(5.0, abs=TOL)
    assert summary["familiar_exploration_s"] == pytest.approx(4.0, abs=TOL)
    assert summary["object_balance_index"] is None
    assert summary["fallback_exploration_s"] == pytest.approx(9.0, abs=TOL)
    assert summary["head_supported_exploration_s"] == 0.0
    assert summary["proximity_exploration_s"] == 0.0
    assert summary["head_estimated_rate_percent"] == 0.0
    assert summary["head_confident_rate_percent"] == 0.0
    assert summary["meets_min_exploration"] is False
    assert summary["first_explored_role"] == "novel"
    assert facts["warning"] == (
        "9.00 s 探索使用距离回退，这些时段没有合格头向支持。")

    first = facts["rows"][0]
    assert first["speed_cm_s"] is None
    assert first["exploration_mode"] == "no_speed"
    assert first["current_object"] is None
    assert first["zone_label"] == "Near"
    # 进入物体 2 一次、物体 1 一次；探索各一段。
    kinds = [(item["kind"], item["time_s"], item["target"])
             for item in facts["events"]]
    assert kinds == [
        ("object_2_zone_entry", 0.0, "object_2"),
        ("exploration", 1.0, "object_2"),
        ("object_1_zone_entry", 6.0, "object_1"),
        ("exploration", 6.0, "object_1"),
    ]

    bins = facts["bins"]
    assert len(bins) == 1
    assert bins[0]["start_s"] == 0.0
    assert bins[0]["end_s"] == pytest.approx(10.0, abs=TOL)
    assert bins[0]["object_1_exploration_s"] == pytest.approx(4.0, abs=TOL)
    assert bins[0]["object_2_exploration_s"] == pytest.approx(5.0, abs=TOL)
    assert bins[0]["discrimination_index"] == pytest.approx(1.0 / 9.0, abs=TOL)


def test_head_direction_fixture_attribution_and_fallback():
    geom = geometry(novel=(14.0, 25.0), familiar=(20.0, 25.0))
    heads = [((11.0, 25.0), (9.0, 25.0), 0.8),
             ((11.0, 25.0), (9.0, 25.0), 0.8),
             ((9.0, 25.0), (11.0, 25.0), 0.8),
             ((9.0, 25.0), (11.0, 25.0), 0.1)]
    facts = nor.process_unit(unit(geom=geom), rows([(10.0, 25.0)] * 4, heads),
                             timeline.Window(0.0, 4.0), None)
    rows_out = facts["rows"]
    assert [row["current_object"] for row in rows_out] == [None, 2, None, 2]
    assert [row["exploration_mode"] for row in rows_out] == [
        "no_speed", "head", "head", "fallback"]
    # 前两条朝向物体 2：夹角 0°、aligned；第三条 180° 不对齐。
    assert rows_out[0]["object_2_head_angle_deg"] == pytest.approx(0.0, abs=TOL)
    assert rows_out[0]["object_2_head_aligned"] is True
    assert rows_out[1]["object_2_head_angle_deg"] == pytest.approx(0.0, abs=TOL)
    assert rows_out[1]["object_2_head_aligned"] is True
    assert rows_out[2]["object_2_head_angle_deg"] == pytest.approx(
        180.0, abs=TOL)
    assert rows_out[2]["object_2_head_aligned"] is False
    # 低质量：夹角与 aligned 都是 null，但仍有 estimated 与来源。
    assert rows_out[3]["head_direction_estimated"] is True
    assert rows_out[3]["head_direction_confident"] is False
    assert rows_out[3]["head_direction_source"] == "classical_geometry"
    assert rows_out[3]["object_2_head_angle_deg"] is None
    assert rows_out[3]["object_2_head_aligned"] is None

    summary = facts["summary"]
    assert summary["head_supported_exploration_s"] == pytest.approx(1.0, abs=TOL)
    assert summary["fallback_exploration_s"] == pytest.approx(1.0, abs=TOL)
    assert summary["total_object_exploration_s"] == pytest.approx(2.0, abs=TOL)
    assert summary["head_estimated_rate_percent"] == pytest.approx(
        100.0, abs=TOL)
    assert summary["head_confident_rate_percent"] == pytest.approx(
        75.0, abs=TOL)
    assert facts["warning"].startswith("1.00 s 探索使用距离回退")


def test_familiarization_uses_object_balance_only():
    facts = nor.process_unit(
        unit(options={"phase": "familiarization"}),
        rows([(10.0, 25.0)] * 3 + [(20.0, 25.0)]),
        timeline.Window(0.0, 4.0), None)
    summary = facts["summary"]
    # 首条无速度；物体 2（第 2、3 条）2 秒、物体 1（第 4 条）1 秒。
    assert summary["object_2_exploration_s"] == pytest.approx(2.0, abs=TOL)
    assert summary["object_1_exploration_s"] == pytest.approx(1.0, abs=TOL)
    assert summary["object_balance_index"] == pytest.approx(-1.0 / 3.0,
                                                           abs=TOL)
    assert summary["novel_exploration_s"] is None
    assert summary["familiar_exploration_s"] is None
    assert summary["discrimination_index"] is None
    assert summary["preference_index_percent"] is None
    assert summary["phase"] == "familiarization"


class PreviewSource:
    """内存解码头：动物停在物体 2 的中心。"""

    def __init__(self, path, *, start_s=0.0, end_s=None, **kwargs):
        from behaviourlab.schema import Frame

        self._frames = []
        begin = float(start_s or 0.0)
        stop = float(end_s if end_s is not None else begin + 4.0)
        step = 0.1
        count = max(1, int(round((stop - begin) / step)))
        for index in range(count):
            moment = round(begin + index * step, 6)
            image = np.full((240, 240, 3), 255, dtype=np.uint8)
            cv2.ellipse(image, (120, 120), (10, 22), 0, 0, 360, (0, 0, 0), -1)
            self._frames.append(Frame(
                image=image, index=index, time_s=moment, dt_s=step,
                elapsed_s=moment - begin, analysis_fps=1.0 / step,
                interval_start_s=moment, interval_end_s=moment + step))

    def iter_frames(self):
        return iter(self._frames)


def test_registration_validation_preview_and_page(client, store, tmp_path,
                                                  monkeypatch):
    from behaviourlab import media, preview
    from behaviourlab.media import FrameInfo

    entry = [item for item in experiments.catalog()
             if item["key"] == "nor"][0]
    assert entry["label"] == "新物体识别实验 / NOR"
    assert [item["label"] for item in entry["geometry"]] == [
        "场地四角（左上、右上、右下、左下）", "物体 1 中心", "物体 2 中心",
        "物体 1 探索范围", "物体 2 探索范围", "检测遮罩"]
    assert [bool(item["required"]) for item in entry["geometry"]] == [
        True, True, True, False, False, False]
    assert entry["template_keys"] == ["arena_width_cm",
                                      "arena_height_cm"]
    assert set(entry["defaults"]) == set(nor.DEFAULTS)

    geom = geometry()
    assert nor.validate_unit(dict(nor.DEFAULTS), geom, complete=True) == []
    # 四角必须是 quad；类型不对直接拒绝。
    line_arena = {"arena": {"type": "line", "points": [[0, 0], [50, 50]]},
                  "objects": geom["objects"]}
    assert [item["field"] for item in nor.validate_unit(
        dict(nor.DEFAULTS), line_arena, complete=True)] == ["geometry.arena"]
    # 识别期角色必须一新奇一熟悉。
    same_roles = {"arena": ARENA, "objects": [
        dict(geom["objects"][0], role="novel"),
        dict(geom["objects"][1], role="novel")]}
    assert [item["field"] for item in nor.validate_unit(
        dict(nor.DEFAULTS), same_roles, complete=True)] == ["geometry.objects"]
    # 探索范围至少 3 点且正面积。
    bad_zone = {"arena": ARENA, "objects": [
        dict(geom["objects"][0],
             zone={"type": "polygon", "points": [[1, 1], [2, 1]]}),
        geom["objects"][1]]}
    assert [item["field"] for item in nor.validate_unit(
        dict(nor.DEFAULTS), bad_zone, complete=True)] == [
        "geometry.objects[0].zone"]
    # 参数范围。
    for options, field in (
            ({"interaction_radius_cm": 0}, "options.interaction_radius_cm"),
            ({"head_angle_threshold_deg": 181},
             "options.head_angle_threshold_deg"),
            ({"head_confidence_threshold": 1.5},
             "options.head_confidence_threshold"),
            ({"use_head_direction": 1}, "options.use_head_direction"),
            ({"bin_seconds": 0}, "options.bin_seconds"),
            ({"phase": "other"}, "options.phase")):
        assert [item["field"] for item in nor.validate_unit(
            dict(nor.DEFAULTS, **options), geom, complete=True)] == [field]

    arena = {"type": "quad",
             "points": [[10.0, 10.0], [230.0, 10.0], [230.0, 230.0],
                        [10.0, 230.0]]}
    source = tmp_path / "nor.mp4"
    source.write_bytes(b"placeholder")
    asset = store.create_asset(
        name="nor.mp4", path=str(source), project_id=None,
        metadata={"width": 240, "height": 240, "duration_s": 4.0,
                  "fps": 10.0, "frame_count": 40})
    draft = store.create_draft("nor", videos=[{
        "asset_id": asset["id"],
        "window": {"start_s": 0.0, "end_s": 4.0},
        "orientation": 0, "fps_override": None, "sample_fps": 10,
        "units": [{
            "id": "n" * 32, "name": "NOR 鼠", "window": None,
            "geometry": {"arena": arena, "objects": [
                {"id": "object_1", "name": "物体 1", "role": "familiar",
                 "center": [220.0, 120.0], "zone": None},
                {"id": "object_2", "name": "物体 2", "role": "novel",
                 "center": [120.0, 120.0], "zone": None}]},
            "options": dict(nor.DEFAULTS), "detection": {}, "manual": None,
        }],
    }])
    monkeypatch.setattr(media, "VideoSource", PreviewSource)
    monkeypatch.setattr(
        "behaviourlab.engine.media.probe_video",
        lambda path: {"width": 240, "height": 240, "duration_s": 4.0,
                      "fps": 10.0, "frame_count": 40})
    monkeypatch.setattr(
        preview.media, "read_frame_at_time",
        lambda *args, **kwargs: FrameInfo(
            image=np.full((240, 240, 3), 255, dtype=np.uint8), index=15,
            time_s=1.5, pts=None, time_base=None, clock_source="pts"))

    bad = store.create_draft("nor", videos=[{
        "asset_id": asset["id"],
        "window": {"start_s": 0.0, "end_s": 4.0},
        "orientation": 0, "fps_override": None, "sample_fps": 10,
        "units": [{
            "id": "b" * 32, "name": "缺物体", "window": None,
            "geometry": {"arena": arena}, "options": dict(nor.DEFAULTS),
            "detection": {}, "manual": None,
        }],
    }])
    rejected = client.post(f"/api/drafts/{bad['id']}/analyze")
    assert rejected.status_code == 400, rejected.get_json()
    fields = [item["field"] for item in
              rejected.get_json()["error"]["fields"]]
    assert "videos[0].units[0].geometry.objects" in fields
    assert store.list_jobs() == []

    response = client.post(
        f"/api/drafts/{draft['id']}/preview",
        json={"asset_id": asset["id"], "unit_id": "n" * 32,
              "time_s": 1.5, "kind": "behavior"})
    assert response.status_code == 201, response.get_json()
    behavior = response.get_json()["data"]["summary"]["behavior"]
    assert behavior["mode"] == "nor"
    # [t−1, t+3) 与视频窗 [0,4) 的交集为 [0.5, 4)，共 3.5 秒。
    assert behavior["total_time_s"] == pytest.approx(3.5, abs=1e-9)
    assert behavior["total_object_exploration_s"] > 0
    assert behavior["rows"] > 0

    started = client.post(f"/api/drafts/{draft['id']}/analyze")
    assert started.status_code == 202, started.get_json()
    payload = started.get_json()["data"]
    job = payload["jobs"][0]
    assert job["status"] == "completed"
    summary = job["manifest"]["units"][0]["summary"]
    assert summary["total_time_s"] == pytest.approx(4.0, abs=1e-9)
    assert summary["object_2_exploration_s"] > 0
    assert len(summary["objects"]) == 2
    figures = job["manifest"]["units"][0]["figures"]
    assert figures["objects"]
    assert figures["trajectory"]

    # objects 图必须能内联显示与附件下载，且与本次生成文件逐字节相同。
    from pathlib import Path as _Path

    run_dir = _Path(store.paths.runs) / payload["run_id"]
    stored = (run_dir / figures["objects"]).read_bytes()
    inline = client.get(
        f"/runs/{payload['run_id']}/{job['job_id']}/{figures['objects']}")
    assert inline.status_code == 200
    assert inline.mimetype == "image/png"
    assert inline.data[:8] == b"\x89PNG\r\n\x1a\n"
    attachment = client.get(
        f"/runs/{payload['run_id']}/{job['job_id']}/{figures['objects']}"
        "?download=1")
    assert attachment.status_code == 200
    assert attachment.headers["Content-Disposition"].startswith("attachment")
    assert attachment.data == inline.data == stored
    # 轨迹/热图与 CSV 的既有通路保持可用。
    for name in (figures["trajectory"], figures["heatmap"],
                 "observations.csv", "bins.csv"):
        assert client.get(
            f"/runs/{payload['run_id']}/{job['job_id']}/{name}").status_code == 200
    assert client.get(
        f"/runs/{payload['run_id']}/{job['job_id']}/notes.txt").status_code == 404

    page = client.get(f"/results/{payload['run_id']}").get_data(as_text=True)
    for label in ("窗口时间", "有效检测时间", "缺失检测时间", "检测率",
                  "可测量运动时间", "总距离", "辨别指数", "新物体偏好指数",
                  "首次探索物体", "距离回退", "物体探索与接近范围（秒）"):
        assert label in page


# -- 复验反例：编辑落盘结构、两种多边形距离、曝光断口、物体图下载 --------

ZONE = {"type": "polygon",
        "points": [[20.0, 24.0], [24.0, 24.0], [24.0, 26.0], [20.0, 26.0]]}


def zone_geometry() -> dict:
    return {
        "arena": ARENA,
        "objects": [
            {"id": "object_1", "name": "物体 1", "role": "familiar",
             "center": [22.0, 25.0], "zone": ZONE},
            {"id": "object_2", "name": "物体 2", "role": "novel",
             "center": [45.0, 45.0], "zone": None},
        ],
    }


def body_rows(points, contours, spans=None) -> list[dict]:
    result = []
    for index, point in enumerate(points):
        start, end = (spans[index] if spans else (float(index),
                                                  float(index) + 1.0))
        result.append({
            "unit_id": "u" * 32, "unit_name": "小鼠 1", "frame": index,
            "time_s": start, "source_dt_s": end - start,
            "elapsed_s": start, "analysis_fps": 1.0,
            "source_interval_start_s": start,
            "source_interval_end_s": end,
            "interval_start_s": start, "interval_end_s": end,
            "dt_s": end - start, "in_window": True, "detected": True,
            "x_px": float(point[0]), "y_px": float(point[1]),
            "quality": 1.0,
            "contour": [[float(x), float(y)] for x, y in contours[index]],
            "missing_reason": None, "head_angle": None, "head_quality": 0.0,
            "nose_x_px": None, "nose_y_px": None,
            "tail_x_px": None, "tail_y_px": None,
            "left_ear_x_px": None, "left_ear_y_px": None,
            "right_ear_x_px": None, "right_ear_y_px": None,
            "detection_method": "threshold",
        })
    return result


def test_point_body_distance_uses_zone_not_object_center():
    # 无轮廓：身体中心 (21,25) 在多边形内，clearance 必须是 0。
    facts = nor.process_unit(
        unit(geom=zone_geometry()),
        body_rows([(21.0, 25.0)] * 2, [[], []]),
        timeline.Window(0.0, 2.0), None)
    second = facts["rows"][1]
    assert second["object_1_zone_clearance_cm"] == pytest.approx(0.0, abs=TOL)
    assert second["object_1_near"] is True
    assert second["current_object"] == 1
    assert facts["summary"]["object_1_exploration_s"] == pytest.approx(1.0,
                                                                      abs=TOL)


def test_body_polygon_distance_keeps_collinear_gap():
    body = [(8.0, 24.0), (12.0, 24.0), (12.0, 26.0), (8.0, 26.0)]
    facts = nor.process_unit(
        unit(geom=zone_geometry()),
        body_rows([(10.0, 25.0)] * 2, [body, body]),
        timeline.Window(0.0, 2.0), None)
    second = facts["rows"][1]
    # 身体矩形 x∈[8,12] 与 zone x∈[20,24] 分离 8 cm，共线边不算相交。
    assert second["object_1_zone_clearance_cm"] == pytest.approx(8.0, abs=TOL)
    assert second["object_1_near"] is False
    assert second["current_object"] is None
    assert facts["summary"]["object_1_exploration_s"] == 0.0

    # 真重叠仍为 0，且不因修共线分支而失效。
    overlap = [(19.0, 24.0), (23.0, 24.0), (23.0, 26.0), (19.0, 26.0)]
    facts = nor.process_unit(
        unit(geom=zone_geometry()),
        body_rows([(10.0, 25.0)] * 2, [overlap, overlap]),
        timeline.Window(0.0, 2.0), None)
    second = facts["rows"][1]
    assert second["object_1_zone_clearance_cm"] == pytest.approx(0.0, abs=TOL)
    assert second["object_1_near"] is True
    assert second["current_object"] == 1


def test_motion_and_exploration_do_not_cross_exposure_gap():
    geom = geometry()
    rows = body_rows([(10.0, 25.0), (10.0, 25.0)], [[], []],
                     spans=[(0.0, 0.5), (1.0, 1.5)])
    facts = nor.process_unit(unit(geom=geom), rows, timeline.Window(0.0, 2.0),
                             None)
    second = facts["rows"][1]
    assert second["speed_cm_s"] is None
    assert second["exploration_mode"] == "no_speed"
    assert second["current_object"] is None
    assert facts["summary"]["total_object_exploration_s"] == 0.0
    assert facts["summary"]["measurable_motion_time_s"] == 0.0
    assert facts["summary"]["object_2_bouts"] == 0
    assert facts["summary"]["object_2_latency_s"] is None
    assert [item["kind"] for item in facts["events"]] == [
        "object_2_zone_entry"]

    # 真正相接的曝光仍成段并正常探索。
    joined = body_rows([(10.0, 25.0), (10.0, 25.0)], [[], []],
                       spans=[(0.0, 0.5), (0.5, 1.0)])
    solid = nor.process_unit(unit(geom=geom), joined,
                             timeline.Window(0.0, 1.0), None)
    assert solid["summary"]["measurable_motion_time_s"] == pytest.approx(
        0.5, abs=TOL)
    assert solid["summary"]["object_2_exploration_s"] == pytest.approx(
        0.5, abs=TOL)


def test_geometry_steps_allow_half_drawn_draft():
    half = {"arena": ARENA, "objects": [
        {"id": "object_1", "name": "物体 1", "role": "familiar",
         "center": [10.0, 25.0], "zone": None},
        {"id": "object_2", "name": "物体 2", "role": "novel",
         "center": None, "zone": None}]}
    # 只画好一个中心时仍可保存；最终校验要求两个中心都在。
    assert nor.validate_unit(dict(nor.DEFAULTS), half) == []
    assert [item["field"] for item in nor.validate_unit(
        dict(nor.DEFAULTS), half, complete=True)] == [
        "geometry.objects[1].center"]


def test_editor_output_saves_previews_and_analyzes(client, store, tmp_path,
                                                   monkeypatch):
    """JS 编辑产物直接走保存→刷新→预览→分析，不经手写对象结构。"""

    import json
    import shutil
    import subprocess
    from pathlib import Path

    from behaviourlab import media, preview
    from behaviourlab.media import FrameInfo

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node 不可用")
    script = Path(__file__).parent / "js" / "test_nor_editor.cjs"
    result = subprocess.run([node, str(script)], capture_output=True,
                            text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "nor-editor OK" in result.stdout
    line = [item for item in result.stdout.splitlines()
            if item.startswith("nor-geometry ")][0]
    geometry = json.loads(line.split(" ", 1)[1])
    # 画布只写 arena + 恰好两个对象，没有平行步骤键。
    assert sorted(geometry) == ["arena", "objects"]
    assert nor.validate_unit(dict(nor.DEFAULTS), geometry, complete=True) == []

    source = tmp_path / "nor-ui.mp4"
    source.write_bytes(b"placeholder")
    asset = store.create_asset(
        name="nor-ui.mp4", path=str(source), project_id=None,
        metadata={"width": 240, "height": 240, "duration_s": 4.0,
                  "fps": 10.0, "frame_count": 40})
    draft = store.create_draft("nor", videos=[{
        "asset_id": asset["id"],
        "window": {"start_s": 0.0, "end_s": 4.0},
        "orientation": 0, "fps_override": None, "sample_fps": 10,
        "units": [{"id": "e" * 32, "name": "NOR 鼠", "window": None,
                   "geometry": {}, "options": {}, "detection": {},
                   "manual": None}],
    }])

    def save(unit_geometry):
        return client.put(f"/api/drafts/{draft['id']}", json={"videos": [{
            "asset_id": asset["id"],
            "window": {"start_s": 0.0, "end_s": 4.0},
            "orientation": 0, "fps_override": None, "sample_fps": 10,
            "units": [{"id": "e" * 32, "name": "NOR 鼠", "window": None,
                       "geometry": unit_geometry, "options": {},
                       "detection": {}, "manual": None}],
        }]})

    # 只画好第一个中心也能保存（编辑未完成不挡住落盘）。
    half = json.loads(json.dumps(geometry))
    half["objects"][1]["center"] = None
    assert save(half).status_code == 200
    assert nor.validate_unit(dict(nor.DEFAULTS), half, complete=True) != []

    saved = save(geometry)
    assert saved.status_code == 200, saved.get_json()
    reloaded = client.get(f"/api/drafts/{draft['id']}").get_json()["data"]
    stored = reloaded["draft"]["videos"][0]["units"][0]["geometry"]
    assert stored == geometry
    assert client.get(
        f"/api/drafts/{draft['id']}/validate").get_json()["data"]["issues"] == []

    monkeypatch.setattr(media, "VideoSource", PreviewSource)
    monkeypatch.setattr(
        "behaviourlab.engine.media.probe_video",
        lambda path: {"width": 240, "height": 240, "duration_s": 4.0,
                      "fps": 10.0, "frame_count": 40})
    monkeypatch.setattr(
        preview.media, "read_frame_at_time",
        lambda *args, **kwargs: FrameInfo(
            image=np.full((240, 240, 3), 255, dtype=np.uint8), index=15,
            time_s=1.5, pts=None, time_base=None, clock_source="pts"))

    behavior = client.post(
        f"/api/drafts/{draft['id']}/preview",
        json={"asset_id": asset["id"], "unit_id": "e" * 32,
              "time_s": 1.5, "kind": "behavior"})
    assert behavior.status_code == 201, behavior.get_json()
    behavior_data = behavior.get_json()["data"]
    assert behavior_data["summary"]["behavior"]["mode"] == "nor"
    assert behavior_data["summary"]["missing_steps"] == []
    assert not any("尚未绘制" in text for text in behavior_data["warnings"])

    started = client.post(f"/api/drafts/{draft['id']}/analyze")
    assert started.status_code == 202, started.get_json()
    payload = started.get_json()["data"]
    job = payload["jobs"][0]
    summary = job["manifest"]["units"][0]["summary"]
    assert summary["total_time_s"] == pytest.approx(4.0, abs=1e-9)
    assert [item["role"] for item in summary["objects"]] == [
        "familiar", "novel"]
    assert summary["objects"][0]["zone_type"] == "polygon"
    assert summary["objects"][1]["zone_type"] == "circle"
    assert job["manifest"]["units"][0]["figures"]["objects"]


def test_editor_invalidates_preview_on_object_edits():
    source = (Path(__file__).parent.parent
              / "behaviourlab/web/static/workspace.js").read_text(
                  encoding="utf-8")
    # 提交几何与改角色都必须让旧预览失效，且预览前先保存。
    commit = source.split("function commitGeometry", 1)[1].split("\n  }", 1)[0]
    assert "invalidatePreview()" in commit
    role = source.split("function norRoleRow", 1)[1].split("\n  }", 1)[0]
    assert "invalidatePreview()" in role
    assert "markDirty()" in role
    assert "norSetRole" in role


def test_workspace_reads_and_writes_object_steps_through_bridge():
    source = (Path(__file__).parent.parent
              / "behaviourlab/web/static/workspace.js").read_text(
                  encoding="utf-8")
    # 物体中心/探索范围不再直接读写 unit.geometry[stepKey]。
    assert "unit.geometry[state.stepKey]" not in source
    for marker in ("function stepShape(unit, key)",
                   "stepShape(unit, state.stepKey)",
                   "window.BLCore.stepApply(state.assay, unit.geometry, key, shape)",
                   "window.BLCore.stepClear(state.assay, unit.geometry, state.stepKey)",
                   "function norRoleRow(index)",
                   'state.assay === "nor" && step.role === "object"'):
        assert marker in source, marker


def test_two_objects_near_still_explore_only_one():
    geom = geometry(novel=(10.0, 25.0), familiar=(14.0, 25.0))
    facts = nor.process_unit(unit(geom=geom),
                             body_rows([(12.0, 25.0)] * 2, [[], []]),
                             timeline.Window(0.0, 2.0), None)
    second = facts["rows"][1]
    assert second["object_1_near"] is True
    assert second["object_2_near"] is True
    # 完全相等时取列表先出现者，且只探索一个。
    assert second["current_object"] == 1
    assert second["object_2_exploring"] is False
    assert facts["summary"]["total_object_exploration_s"] == pytest.approx(
        1.0, abs=TOL)


def test_preview_missing_steps_follow_objects_array(client, store, tmp_path,
                                                    monkeypatch):
    """预览的缺失步骤必须按 objects 数组判断，不能查顶层 object1/object2。"""

    from behaviourlab import preview
    from behaviourlab.media import FrameInfo

    arena = {"type": "quad",
             "points": [[10.0, 10.0], [230.0, 10.0], [230.0, 230.0],
                        [10.0, 230.0]]}
    full = {"arena": arena, "objects": [
        {"id": "object_1", "name": "物体 1", "role": "familiar",
         "center": [120.0, 120.0], "zone": None},
        {"id": "object_2", "name": "物体 2", "role": "novel",
         "center": [200.0, 120.0], "zone": None}]}
    source = tmp_path / "nor-preview.mp4"
    source.write_bytes(b"placeholder")
    asset = store.create_asset(
        name="nor-preview.mp4", path=str(source), project_id=None,
        metadata={"width": 240, "height": 240, "duration_s": 4.0,
                  "fps": 10.0, "frame_count": 40})

    def make(unit_geometry):
        return store.create_draft("nor", videos=[{
            "asset_id": asset["id"],
            "window": {"start_s": 0.0, "end_s": 4.0},
            "orientation": 0, "fps_override": None, "sample_fps": 10,
            "units": [{"id": "p" * 32, "name": "NOR 鼠", "window": None,
                       "geometry": dict(unit_geometry), "options": {},
                       "detection": {}, "manual": None}],
        }])

    monkeypatch.setattr(
        preview.media, "read_frame_at_time",
        lambda *args, **kwargs: FrameInfo(
            image=np.full((240, 240, 3), 255, dtype=np.uint8), index=15,
            time_s=1.5, pts=None, time_base=None, clock_source="pts"))

    draft = make(full)
    response = client.post(
        f"/api/drafts/{draft['id']}/preview",
        json={"asset_id": asset["id"], "unit_id": "p" * 32,
              "time_s": 1.5, "kind": "detection"})
    assert response.status_code == 201, response.get_json()
    summary = response.get_json()["data"]["summary"]
    assert summary["drawn_steps"] == ["arena", "object1", "object2"]
    assert summary["missing_steps"] == []
    assert not any("尚未绘制" in text
                   for text in response.get_json()["data"]["warnings"])

    # 只缺物体 2 中心时只报这一项，不牵连物体 1。
    partial = {"arena": arena, "objects": [
        dict(full["objects"][0]),
        dict(full["objects"][1], center=None)]}
    half = make(partial)
    response = client.post(
        f"/api/drafts/{half['id']}/preview",
        json={"asset_id": asset["id"], "unit_id": "p" * 32,
              "time_s": 1.5, "kind": "detection"})
    assert response.status_code == 201, response.get_json()
    data = response.get_json()["data"]
    assert data["summary"]["drawn_steps"] == ["arena", "object1"]
    assert data["summary"]["missing_steps"] == ["物体 2 中心"]
    assert any("尚未绘制：物体 2 中心" == text for text in data["warnings"])

    # 真正缺场地仍然要报，不因这次修复放宽。
    assert [item["field"] for item in nor.validate_unit(
        dict(nor.DEFAULTS), {"objects": full["objects"]},
        complete=True)] == ["geometry.arena"]


def test_preview_missing_steps_follow_objects_array(client, store, tmp_path,
                                                    monkeypatch):
    """预览的缺失步骤必须按 objects 数组判断，不能查顶层 object1/object2。"""

    from behaviourlab import preview
    from behaviourlab.media import FrameInfo

    arena = {"type": "quad",
             "points": [[10.0, 10.0], [230.0, 10.0], [230.0, 230.0],
                        [10.0, 230.0]]}
    full = {"arena": arena, "objects": [
        {"id": "object_1", "name": "物体 1", "role": "familiar",
         "center": [120.0, 120.0], "zone": None},
        {"id": "object_2", "name": "物体 2", "role": "novel",
         "center": [200.0, 120.0], "zone": None}]}
    source = tmp_path / "nor-preview.mp4"
    source.write_bytes(b"placeholder")
    asset = store.create_asset(
        name="nor-preview.mp4", path=str(source), project_id=None,
        metadata={"width": 240, "height": 240, "duration_s": 4.0,
                  "fps": 10.0, "frame_count": 40})

    def make(unit_geometry):
        return store.create_draft("nor", videos=[{
            "asset_id": asset["id"],
            "window": {"start_s": 0.0, "end_s": 4.0},
            "orientation": 0, "fps_override": None, "sample_fps": 10,
            "units": [{"id": "p" * 32, "name": "NOR 鼠", "window": None,
                       "geometry": dict(unit_geometry), "options": {},
                       "detection": {}, "manual": None}],
        }])

    monkeypatch.setattr(
        preview.media, "read_frame_at_time",
        lambda *args, **kwargs: FrameInfo(
            image=np.full((240, 240, 3), 255, dtype=np.uint8), index=15,
            time_s=1.5, pts=None, time_base=None, clock_source="pts"))

    draft = make(full)
    response = client.post(
        f"/api/drafts/{draft['id']}/preview",
        json={"asset_id": asset["id"], "unit_id": "p" * 32,
              "time_s": 1.5, "kind": "detection"})
    assert response.status_code == 201, response.get_json()
    summary = response.get_json()["data"]["summary"]
    assert summary["drawn_steps"] == ["arena", "object1", "object2"]
    assert summary["missing_steps"] == []
    assert not any("尚未绘制" in text
                   for text in response.get_json()["data"]["warnings"])

    # 只缺物体 2 中心时只报这一项，不牵连物体 1。
    partial = {"arena": arena, "objects": [
        dict(full["objects"][0]),
        dict(full["objects"][1], center=None)]}
    half = make(partial)
    response = client.post(
        f"/api/drafts/{half['id']}/preview",
        json={"asset_id": asset["id"], "unit_id": "p" * 32,
              "time_s": 1.5, "kind": "detection"})
    assert response.status_code == 201, response.get_json()
    data = response.get_json()["data"]
    assert data["summary"]["drawn_steps"] == ["arena", "object1"]
    assert data["summary"]["missing_steps"] == ["物体 2 中心"]
    assert any("尚未绘制：物体 2 中心" == text for text in data["warnings"])

    # 真正缺场地仍然要报，不因这次修复放宽。
    assert [item["field"] for item in nor.validate_unit(
        dict(nor.DEFAULTS), {"objects": full["objects"]},
        complete=True)] == ["geometry.arena"]


def test_undo_refreshes_inspector_without_reload():
    source = (Path(__file__).parent.parent
              / "behaviourlab/web/static/workspace.js").read_text(
                  encoding="utf-8")
    block = source.split(
        'el.toolUndo.addEventListener("click", () => {', 1)[1].split(
            "\n    });", 1)[0]
    # 撤销几何仍沿用"清除该步骤"的语义，并即时刷新侧栏与预览确认。
    assert "window.BLCore.stepClear(state.assay, unit.geometry, state.stepKey)" in block
    assert "state.previewGate.invalidate()" in block
    assert "renderInspector()" in block
    assert '} else if (result === "points") {' not in block


def test_undo_refreshes_inspector_without_reload():
    source = (Path(__file__).parent.parent
              / "behaviourlab/web/static/workspace.js").read_text(
                  encoding="utf-8")
    block = source.split(
        'el.toolUndo.addEventListener("click", () => {', 1)[1].split(
            "\n    });", 1)[0]
    # 撤销几何仍沿用"清除该步骤"的语义，并即时刷新侧栏与预览确认。
    assert "window.BLCore.stepClear(state.assay, unit.geometry, state.stepKey)" in block
    assert "state.previewGate.invalidate()" in block
    assert "renderInspector()" in block
    assert '} else if (result === "points") {' not in block
