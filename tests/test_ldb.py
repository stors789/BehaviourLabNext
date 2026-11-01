"""LDB 明暗箱：可见/不可见固定夹具、校验与界面接线。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from behaviourlab import detection, experiments, media
from behaviourlab.experiments import ldb
from behaviourlab.schema import AnalysisContext, Detection, Frame, ValidationError
from behaviourlab import timeline

UNIT = "u" * 32

LIGHT = {"type": "quad",
         "points": [[10, 10], [190, 10], [190, 95], [10, 95]]}
DARK = {"type": "quad",
        "points": [[10, 105], [190, 105], [190, 190], [10, 190]]}

OPTIONS = dict(ldb.DEFAULTS, door_open_delay_s=0.0,
               transition_min_dwell_s=0.12, doorway_reset_s=0.2,
               bin_seconds=60.0)


def geometry(dark=DARK, **extra):
    zones = {"light": LIGHT}
    if dark is not None:
        zones["dark"] = dark
    zones.update(extra)
    return {"zones": zones}


def unit(*, dark_visible=True, dark=DARK, options=None, **extra):
    merged = dict(OPTIONS)
    merged["dark_box_visible"] = dark_visible
    if options:
        merged.update(options)
    return {"id": UNIT, "name": "小鼠 1", "animal_id": None, "window": None,
            "geometry": geometry(dark, **extra), "options": merged,
            "detection": {}}


def source_rows(positions):
    """20 条源观测：t=.2i、曝光 .2 秒、窗 [0,4)。"""

    rows = []
    for index, seen in enumerate(positions):
        start = round(0.2 * index, 10)
        end = round(0.2 * (index + 1), 10)
        detected = seen is not None
        rows.append({
            "unit_id": UNIT, "unit_name": "小鼠 1", "animal_id": None,
            "frame": index, "time_s": float(start), "dt_s": 0.2,
            "source_dt_s": 0.2, "elapsed_s": float(start), "analysis_fps": 5.0,
            "source_interval_start_s": float(start),
            "source_interval_end_s": float(end),
            "interval_start_s": float(start), "interval_end_s": float(end),
            "in_window": True, "detected": detected,
            "x_px": float(seen[0]) if detected else None,
            "y_px": float(seen[1]) if detected else None,
            "quality": 1.0 if detected else 0.0, "nose_x_px": None,
            "nose_y_px": None, "tail_x_px": None, "tail_y_px": None,
        })
    return rows


def light_rows():
    return [(100, 50)] * 10 + [(100, 150)] * 10


def inferred_rows():
    return [(100, 50)] * 10 + [None] * 10


# -- 1. 可见双室：固定数值与物理映射 ------------------------------------

def test_ldb_visible_fixture_scores():
    result = ldb.process_unit(unit(), source_rows(light_rows()),
                              timeline.Window(0.0, 4.0), 0.0)
    summary = result["summary"]

    assert summary["total_time_s"] == pytest.approx(4.0, abs=1e-9)
    assert summary["observed_time_s"] == pytest.approx(4.0, abs=1e-9)
    assert summary["missing_time_s"] == pytest.approx(0.0, abs=1e-9)
    assert summary["light_time_s"] == pytest.approx(2.0, abs=1e-9)
    assert summary["dark_time_s"] == pytest.approx(2.0, abs=1e-9)
    assert summary["light_time_percent"] == pytest.approx(50.0, abs=1e-9)
    assert summary["dark_time_percent"] == pytest.approx(50.0, abs=1e-9)
    assert summary["total_transitions"] == 1
    assert summary["light_entries"] == 0
    assert summary["dark_entries"] == 1
    assert summary["initial_chamber"] == "light"
    assert summary["initial_chamber_time_s"] == pytest.approx(0.0, abs=1e-9)
    assert summary["first_light_latency_s"] == pytest.approx(0.0, abs=1e-9)
    assert summary["coordinate_unit"] == "cm"
    assert summary["tracking_reliable"] is True

    kinds = [(item["kind"], item["time_s"], item["confirmed_s"], item["target"])
             for item in result["events"]]
    assert kinds == [("initial_chamber", 0.0, 0.2, "light"),
                     ("chamber_transition", 2.0, 2.2, "dark")]
    assert result["events"][0]["previous_chamber"] is None
    assert result["events"][1]["previous_chamber"] == "light"
    assert all(item["source"] == "automatic" for item in result["events"])

    zones = [row["zone"] for row in result["rows"]]
    assert zones == ["light"] * 10 + ["dark"] * 10
    assert all(row["zone_evidence"] == "detected" for row in result["rows"])
    transitions = [row["transition_event"] for row in result["rows"]]
    assert transitions.count(True) == 1
    assert result["rows"][11]["entry_chamber"] == "dark"
    assert result["rows"][0]["entry_chamber"] is None

    # 物理映射：相邻边为明室下边/暗室上边，公共边 40cm → along 40、across 20。
    mapping = result["metadata"]["mapping"]
    assert mapping["along_light_cm"] == pytest.approx(40.0)
    assert mapping["across_light_cm"] == pytest.approx(20.0)
    assert mapping["offset_cm"] == pytest.approx(0.0)
    # (100,50)：明室高 85px → across=(95-50)/85*20；暗室对称。
    light_point = result["rows"][0]
    assert light_point["x_cm"] == pytest.approx(0.0, abs=1e-6)
    assert light_point["y_cm"] == pytest.approx(45.0 / 85.0 * 20.0, abs=1e-6)
    dark_point = result["rows"][10]
    assert dark_point["x_cm"] == pytest.approx(0.0, abs=1e-6)
    assert dark_point["y_cm"] == pytest.approx(-45.0 / 85.0 * 20.0, abs=1e-6)
    # 共享边对齐：跨室一步的物理距离 = 两侧 across 之和（该夹具像素高宽比
    # 与声明尺寸不成比例，距离由映射坐标计算）。
    cross = 2.0 * 45.0 / 85.0 * 20.0
    assert (abs(light_point["y_cm"]) + abs(dark_point["y_cm"])
            == pytest.approx(cross, abs=1e-6))

    # 运动：18 段同点 + 1 段跨室；第二阶段首条不被丢掉。
    assert summary["total_distance_cm"] == pytest.approx(cross, abs=1e-9)
    assert summary["measurable_motion_time_s"] == pytest.approx(3.8, abs=1e-9)
    # 同点段速度 0 → 静止；只有跨室一段是移动。
    assert summary["moving_time_s"] == pytest.approx(0.2, abs=1e-9)
    assert summary["immobility_time_s"] == pytest.approx(3.6, abs=1e-9)
    assert summary["mean_speed_cm_s"] == pytest.approx(cross / 3.8, abs=1e-9)

    bins = result["bins"]
    assert len(bins) == 1
    assert bins[0]["light_time_s"] == pytest.approx(2.0, abs=1e-9)
    assert bins[0]["dark_time_s"] == pytest.approx(2.0, abs=1e-9)
    assert bins[0]["initial_entries"] == 1
    assert bins[0]["transitions"] == 1
    assert bins[0]["total_distance_cm"] == pytest.approx(cross, abs=1e-9)


# -- 2. 不可见暗室：推断与缺失是同一时间的不同事实 ----------------------

def test_ldb_invisible_fixture_infers_dark():
    result = ldb.process_unit(unit(dark_visible=False),
                              source_rows(inferred_rows()),
                              timeline.Window(0.0, 4.0), 0.0)
    summary = result["summary"]

    assert summary["light_time_s"] == pytest.approx(2.0, abs=1e-9)
    assert summary["dark_time_s"] == pytest.approx(2.0, abs=1e-9)
    assert summary["missing_time_s"] == pytest.approx(2.0, abs=1e-9)
    assert summary["inferred_dark_time_s"] == pytest.approx(2.0, abs=1e-9)
    assert summary["observed_time_s"] == pytest.approx(2.0, abs=1e-9)
    assert summary["total_transitions"] == 1
    assert summary["coordinate_unit"] is None
    assert summary["position_available"] is False
    assert summary["tracking_reliable"] is False
    assert "total_distance_cm" not in summary
    assert "total_distance_px" not in summary
    assert "moving_time_s" not in summary

    zones = [(row["zone"], row["zone_evidence"]) for row in result["rows"]]
    assert zones[:10] == [("light", "detected")] * 10
    assert zones[10:] == [("dark", "inferred_absent")] * 10
    assert [(item["kind"], item["evidence"]) for item in result["events"]] == [
        ("initial_chamber", "detected"),
        ("chamber_transition", "inferred_absent"),
    ]

    # 失检与推断暗室是同一时间：不叠加成 6 秒。
    assert (summary["light_time_s"] + summary["dark_time_s"]
            == pytest.approx(summary["total_time_s"], abs=1e-9))

    # 明室局部像素留在观察表，但没有可靠全场轨迹字段。
    assert result["rows"][0]["x_px"] == pytest.approx(100.0)
    assert "step" not in result["rows"][0]
    assert result["metadata"]["inference_rule"]


# -- 3. 校验、px 退化与检测预览 ----------------------------------------

def test_ldb_validation_rules():
    issues = ldb.validate_unit(dict(OPTIONS), geometry(), complete=True)
    assert issues == []

    issues = ldb.validate_unit(dict(OPTIONS), geometry(dark=None),
                               complete=True)
    fields = [item["field"] for item in issues]
    assert "geometry.zones.dark" in fields, "可见模式必须有暗室"

    # 不可见模式可以没有暗室几何；但保留的旧几何仍须合法。
    assert ldb.validate_unit(dict(OPTIONS, dark_box_visible=False),
                             geometry(dark=None), complete=True) == []
    crossed = {"type": "quad",
               "points": [[10, 105], [190, 190], [190, 105], [10, 190]]}
    issues = ldb.validate_unit(dict(OPTIONS, dark_box_visible=False),
                               geometry(dark=crossed))
    assert any(item["field"] == "geometry.zones.dark" for item in issues)

    # 普通 polygon 允许简单凹形；quad 必须凸。
    concave = {"type": "polygon",
               "points": [[10, 10], [190, 10], [190, 95], [100, 60], [10, 95]]}
    assert ldb._polygon(concave) is not None
    issues = ldb.validate_unit(dict(OPTIONS), {"zones": {"light": concave,
                                                         "dark": DARK}},
                               complete=True)
    assert issues == []

    crossed_light = {"type": "quad",
                     "points": [[10, 10], [190, 95], [190, 10], [10, 95]]}
    issues = ldb.validate_unit(dict(OPTIONS), {"zones": {"light": crossed_light,
                                                         "dark": DARK}},
                               complete=True)
    assert any(item["field"] == "geometry.zones.light" for item in issues)

    # 严格布尔与公共边约束。
    issues = ldb.validate_unit(dict(OPTIONS, dark_box_visible="yes"),
                               geometry())
    assert any(item["field"] == "options.dark_box_visible" for item in issues)
    issues = ldb.validate_unit(dict(OPTIONS, shared_edge_length_cm=50.0),
                               geometry())
    assert any(item["field"] == "options.shared_edge_length_cm"
               for item in issues)
    issues = ldb.validate_unit(dict(OPTIONS, dark_box_visible=False,
                                    shared_edge_length_cm=50.0), geometry())
    assert not any(item["field"] == "options.shared_edge_length_cm"
                   for item in issues), "不可见模式不做公共边约束"
    issues = ldb.validate_unit(dict(OPTIONS, bin_seconds=0), geometry())
    assert any(item["field"] == "options.bin_seconds" for item in issues)
    issues = ldb.validate_unit(dict(OPTIONS, door_open_delay_s=-1), geometry())
    assert any(item["field"] == "options.door_open_delay_s" for item in issues)


def test_ldb_pixel_fallback_when_dark_not_quad():
    pentagon = {"type": "polygon",
                "points": [[10, 105], [80, 105], [190, 130], [190, 190],
                           [10, 190]]}
    result = ldb.process_unit(unit(dark=pentagon), source_rows(light_rows()),
                              timeline.Window(0.0, 4.0), 0.0)
    summary = result["summary"]
    assert summary["coordinate_unit"] == "px"
    assert summary["position_available"] is True
    assert summary["tracking_reliable"] is False
    assert "total_distance_cm" not in summary
    assert "moving_time_s" not in summary
    # 19 段中只有跨室一段有位移（100 px），cm/s 阈值不生效。
    assert summary["total_distance_px"] == pytest.approx(100.0, abs=1e-9)
    assert summary["mean_speed_px_s"] == pytest.approx(100.0 / 3.8, abs=1e-9)


def test_ldb_preview_and_result_page(client, store, tmp_path, monkeypatch):
    import numpy as np

    source = tmp_path / "ldb.mp4"
    source.write_bytes(b"placeholder")
    asset = store.create_asset(
        name="ldb.mp4", path=str(source), project_id=None,
        metadata={"width": 200, "height": 200, "duration_s": 4.0,
                  "fps": 5.0, "frame_count": 20})
    draft = store.create_draft("ldb", videos=[{
        "asset_id": asset["id"],
        "window": {"start_s": 0.0, "end_s": 4.0},
        "orientation": 0, "fps_override": None, "sample_fps": None,
        "units": [dict(unit(), id=UNIT)],
    }])

    class Frames:
        def __init__(self, path=None, *, start_s=0.0, end_s=None, **kwargs):
            self._frames = []
            for index in range(20):
                moment = round(0.2 * index, 10)
                image = np.full((200, 200), 240, dtype=np.uint8)
                self._frames.append(Frame(
                    image=image, index=index, time_s=moment, dt_s=0.2,
                    elapsed_s=moment, analysis_fps=5.0,
                    interval_start_s=moment, interval_end_s=moment + 0.2))

        def iter_frames(self):
            return iter(self._frames)

    def track(run_key, unit_id, *, frame_index, **kwargs):
        point = (100.0, 50.0) if frame_index < 10 else (100.0, 150.0)
        return Detection(center=point, quality=1.0, contour=[], mask=None)

    monkeypatch.setattr(media, "VideoSource", Frames)
    monkeypatch.setattr("behaviourlab.engine.media.probe_video",
                        lambda path: {"width": 200, "height": 200,
                                      "duration_s": 4.0, "fps": 5.0,
                                      "frame_count": 20})
    monkeypatch.setattr(ldb.observation, "track", track)

    payload = client.post("/api/jobs",
                          json={"draft_id": draft["id"]}).get_json()["data"]
    job = payload["jobs"][0]
    assert job["status"] == "completed", job
    summary = job["manifest"]["units"][0]["summary"]
    assert summary["light_time_s"] == pytest.approx(2.0, abs=1e-9)
    assert summary["dark_time_s"] == pytest.approx(2.0, abs=1e-9)
    assert summary["total_transitions"] == 1

    page = client.get(f"/results/{payload['run_id']}").get_data(as_text=True)
    for label in ("明室时间", "暗室时间", "门区时间", "明室时间比例",
                  "暗室时间比例", "初始主室", "进入明室", "进入暗室",
                  "首次进入明室潜伏期", "初始"):
        assert label in page, label
    assert "推断暗室" not in page, "可见模式不显示推断行"
    assert "没有可用的场地尺寸" not in page
    # 清单没有图形时给诚实空态：不放空 src，也不谎称生成了轨迹/热图。
    assert "暂无图表" in page
    assert 'src=""' not in page
    assert "_trajectory.png" not in page and "_heatmap.png" not in page


def test_ldb_dark_initial_has_no_light_latency():
    result = ldb.process_unit(unit(), source_rows([(100, 150)] * 20),
                              timeline.Window(0.0, 4.0), 0.0)
    summary = result["summary"]
    assert summary["initial_chamber"] == "dark"
    assert summary["first_light_latency_s"] is None
    assert summary["total_transitions"] == 0
    assert summary["light_entries"] == 0
    assert summary["dark_entries"] == 0


def test_ldb_behavior_preview_uses_short_window(client, store, tmp_path,
                                                monkeypatch):
    import numpy as np

    from behaviourlab import preview as preview_service

    source = tmp_path / "ldb.mp4"
    source.write_bytes(b"placeholder")
    asset = store.create_asset(
        name="ldb.mp4", path=str(source), project_id=None,
        metadata={"width": 200, "height": 200, "duration_s": 4.0,
                  "fps": 5.0, "frame_count": 20})
    draft = store.create_draft("ldb", videos=[{
        "asset_id": asset["id"],
        "window": {"start_s": 0.0, "end_s": 4.0},
        "orientation": 0, "fps_override": None, "sample_fps": None,
        "units": [dict(unit(), id=UNIT)],
    }])

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
        point = (100.0, 50.0) if frame_index < 10 else (100.0, 150.0)
        return Detection(center=point, quality=1.0, contour=[], mask=None)

    from behaviourlab.media import FrameInfo

    monkeypatch.setattr(media, "VideoSource", Frames)

    def fake_frame(*args, **kwargs):
        return FrameInfo(image=np.full((200, 200), 240, dtype=np.uint8),
                         index=0, time_s=0.0, pts=0, time_base=None,
                         clock_source="pts")

    monkeypatch.setattr(preview_service.media, "read_frame_at_time", fake_frame)
    monkeypatch.setattr(preview_service.media, "read_frame_info", fake_frame)
    monkeypatch.setattr(ldb.observation, "track", track)

    response = client.post(
        f"/api/drafts/{draft['id']}/preview",
        json={"asset_id": asset["id"], "unit_id": UNIT,
              "frame_index": 0, "time_s": 0.0, "kind": "behavior"},
    )
    assert response.status_code == 201, response.get_json()
    data = response.get_json()["data"]
    assert data["preview_scope"]["mode"] == "range"
    behavior = data["summary"]["behavior"]
    assert behavior["mode"] == "ldb"
    assert behavior["light_time_s"] == pytest.approx(2.0, abs=1e-9)
    assert behavior["dark_time_s"] == pytest.approx(1.0, abs=1e-9)
    assert behavior["initial_chamber"] == "light"
    assert behavior["total_transitions"] == 1


# -- 4. 第 17 块有限修正：稀疏配置、非法已填区域与数值输入 ---------------

def asset_and_video(store, tmp_path, *, units=None, options=None):
    source = tmp_path / "ldb.mp4"
    source.write_bytes(b"placeholder")
    asset = store.create_asset(
        name="ldb.mp4", path=str(source), project_id=None,
        metadata={"width": 200, "height": 200, "duration_s": 4.0,
                  "fps": 5.0, "frame_count": 20})
    video = {
        "asset_id": asset["id"],
        "window": {"start_s": 0.0, "end_s": 4.0},
        "orientation": 0, "fps_override": None, "sample_fps": None,
        "units": units if units is not None else [dict(unit(), id=UNIT)],
    }
    if options is not None:
        video["options"] = options
    return asset, video


def rows_at(entries):
    """显式曝光行：(time_s, 起点, 终点, 位置或 None)。"""

    rows = []
    for index, (time_s, start, end, seen) in enumerate(entries):
        detected = seen is not None
        rows.append({
            "unit_id": UNIT, "unit_name": "小鼠 1", "animal_id": None,
            "frame": index, "time_s": float(time_s), "dt_s": float(end - start),
            "source_dt_s": float(end - start), "elapsed_s": float(time_s),
            "analysis_fps": 1.0,
            "source_interval_start_s": float(start),
            "source_interval_end_s": float(end),
            "interval_start_s": float(start), "interval_end_s": float(end),
            "in_window": True, "detected": detected,
            "x_px": float(seen[0]) if detected else None,
            "y_px": float(seen[1]) if detected else None,
            "quality": 1.0 if detected else 0.0,
            "nose_x_px": None, "nose_y_px": None,
            "tail_x_px": None, "tail_y_px": None,
        })
    return rows


def test_ldb_sparse_option_overlay_rejects_shared_edge(client, store, tmp_path):
    _asset, video = asset_and_video(store, tmp_path)
    draft = store.create_draft("ldb", videos=[video])
    issues = client.get(f"/api/drafts/{draft['id']}/validate") \
        .get_json()["data"]["issues"]
    assert issues == [], issues

    def sparse(unit_options):
        return {"options": {}, "videos": [dict(
            video, options={},
            units=[{"id": UNIT, "name": "小鼠 1", "animal_id": None,
                    "window": None, "geometry": geometry(),
                    "options": unit_options, "detection": {}}])]}

    # 稀疏覆盖：单位只设共享边，其余尺寸继承默认；上限是 40。
    response = client.put(f"/api/drafts/{draft['id']}",
                          json=sparse({"shared_edge_length_cm": 50.0}))
    assert response.status_code == 400, response.get_json()
    fields = [item["field"] for item in response.get_json()["error"]["fields"]]
    assert "videos[0].units[0].options.shared_edge_length_cm" in fields, fields

    # 明室长度 5、其余默认：上限变成 20，公共边 40 仍须拒绝。
    response = client.put(f"/api/drafts/{draft['id']}",
                          json=sparse({"light_box_length_cm": 5.0}))
    assert response.status_code == 400, response.get_json()
    fields = response.get_json()["error"]["fields"]
    assert "videos[0].units[0].options.shared_edge_length_cm" in [
        item["field"] for item in fields], fields
    message = next(item["message"] for item in fields
                   if item["field"].endswith("options.shared_edge_length_cm"))
    assert "20" in message, message

    # 已持久化的稀疏配置：最终校验与开始入口都要在作业之前拒绝。
    jobs_before = len(store.list_jobs())
    store.update_draft(draft["id"], {
        "options": {},
        "videos": [dict(video, options={}, units=[{
            "id": UNIT, "name": "小鼠 1", "animal_id": None, "window": None,
            "geometry": geometry(),
            "options": {"shared_edge_length_cm": 50.0}, "detection": {}}])],
    })
    issues = client.get(f"/api/drafts/{draft['id']}/validate") \
        .get_json()["data"]["issues"]
    assert any(item["field"].endswith("options.shared_edge_length_cm")
               for item in issues), issues
    start = client.post("/api/jobs", json={"draft_id": draft["id"]})
    assert start.status_code == 400, start.get_json()
    assert len(store.list_jobs()) == jobs_before

    # 正常五尺寸仍通过。
    normal = {"light_box_length_cm": 40.0, "light_box_width_cm": 20.0,
              "dark_box_length_cm": 40.0, "dark_box_width_cm": 20.0,
              "shared_edge_length_cm": 40.0}
    store.update_draft(draft["id"], {
        "videos": [dict(video, options={}, units=[{
            "id": UNIT, "name": "小鼠 1", "animal_id": None, "window": None,
            "geometry": geometry(), "options": normal, "detection": {}}])],
    })
    assert client.get(f"/api/drafts/{draft['id']}/validate") \
        .get_json()["data"]["issues"] == []


def test_ldb_mask_validated_even_without_zones(client, store, tmp_path):
    _asset, video = asset_and_video(store, tmp_path)
    crossed = {"type": "polygon",
               "points": [[1, 1], [10, 10], [1, 10], [10, 1]]}
    payload = dict(video, units=[{
        "id": UNIT, "name": "小鼠 1", "animal_id": None, "window": None,
        "geometry": {"mask": crossed}, "options": {}, "detection": {}}])
    response = client.post("/api/drafts",
                           json={"assay": "ldb", "videos": [payload]})
    assert response.status_code == 400, response.get_json()
    fields = [item["field"]
              for item in response.get_json()["error"]["fields"]]
    assert "videos[0].units[0].geometry.mask" in fields, fields

    # 未画完的草稿仍可暂存：只有合法遮罩、没有 zones。
    legal = {"type": "polygon",
             "points": [[0, 0], [10, 0], [10, 10], [0, 10]]}
    payload = dict(video, units=[{
        "id": UNIT, "name": "小鼠 1", "animal_id": None, "window": None,
        "geometry": {"mask": legal}, "options": {}, "detection": {}}])
    response = client.post("/api/drafts",
                           json={"assay": "ldb", "videos": [payload]})
    assert response.status_code == 201, response.get_json()


def test_ldb_numeric_controls_accept_fractional_input():
    numbers = [item for item in ldb.SPEC["parameters"]
               if item["type"] == "number"]
    assert len(numbers) == 10
    for item in numbers:
        assert item["min"] == 0, item["key"]
        assert item["step"] == "any", item["key"]


# -- 5. 第 17 块有限修正：缺失/分箱、运动断口与门区重置 ------------------

def test_ldb_visible_missing_is_not_unknown_in_bins():
    result = ldb.process_unit(unit(), source_rows([None]),
                              timeline.Window(0.0, 0.2), None)
    summary = result["summary"]
    assert result["rows"][0]["zone"] is None
    assert result["rows"][0]["zone_evidence"] == "missing"
    assert summary["missing_time_s"] == pytest.approx(0.2, abs=1e-9)
    assert summary["unknown_time_s"] == pytest.approx(0.0, abs=1e-9)
    assert summary["light_time_s"] == pytest.approx(0.0, abs=1e-9)
    assert len(result["bins"]) == 1
    assert result["bins"][0]["unknown_time_s"] == pytest.approx(0.0, abs=1e-9)
    assert result["bins"][0]["missing_time_s"] == pytest.approx(0.2, abs=1e-9)

    # 跨箱：缺失不得在分箱里补成 unknown，合计必须与汇总一致。
    rows = rows_at([
        (0.0, 0.0, 0.2, None),
        (0.2, 0.2, 0.4, (2, 2)),
        (0.4, 0.4, 0.6, (2, 2)),
        (0.6, 0.6, 0.8, (100, 50)),
        (0.8, 0.8, 1.0, (100, 50)),
        (1.0, 1.0, 1.2, (100, 50)),
    ])
    result = ldb.process_unit(unit(options={"bin_seconds": 0.6}), rows,
                              timeline.Window(0.0, 1.2), None)
    summary = result["summary"]
    assert summary["unknown_time_s"] == pytest.approx(0.4, abs=1e-9)
    assert summary["missing_time_s"] == pytest.approx(0.2, abs=1e-9)
    total_unknown = sum(item["unknown_time_s"] for item in result["bins"])
    total_missing = sum(item["missing_time_s"] for item in result["bins"])
    assert total_unknown == pytest.approx(summary["unknown_time_s"], abs=1e-9)
    assert total_missing == pytest.approx(summary["missing_time_s"], abs=1e-9)


def test_ldb_invisible_missing_keeps_dark_and_missing_together():
    rows = rows_at([(0.0, 0.0, 0.2, None), (0.2, 0.2, 0.4, (100, 50))])
    result = ldb.process_unit(unit(dark_visible=False), rows,
                              timeline.Window(0.0, 0.4), None)
    summary = result["summary"]
    assert summary["missing_time_s"] == pytest.approx(0.2, abs=1e-9)
    assert summary["inferred_dark_time_s"] == pytest.approx(0.2, abs=1e-9)
    assert summary["dark_time_s"] == pytest.approx(0.2, abs=1e-9)
    assert result["bins"][0]["missing_time_s"] == pytest.approx(0.2, abs=1e-9)
    assert result["bins"][0]["inferred_dark_time_s"] == pytest.approx(0.2, abs=1e-9)
    assert result["bins"][0]["dark_time_s"] == pytest.approx(0.2, abs=1e-9)
    assert result["bins"][0]["unknown_time_s"] == pytest.approx(0.0, abs=1e-9)


FLAT_LIGHT = {"type": "quad",
              "points": [[0, 0], [100, 0], [100, 40], [0, 40]]}
FLAT_DARK = {"type": "quad",
             "points": [[0, 50], [100, 50], [100, 90], [0, 90]]}


def test_ldb_motion_segment_does_not_cross_exposure_gap():
    rows = rows_at([(0.0, 0.0, 0.5, (10, 20)),
                    (1.0, 1.0, 2.0, (30, 20))])
    result = ldb.process_unit(unit(dark=FLAT_DARK), rows,
                              timeline.Window(0.0, 2.0), None)
    summary = result["summary"]
    assert summary["coordinate_unit"] == "cm"
    assert summary["total_distance_cm"] == pytest.approx(0.0, abs=1e-9)
    assert summary["measurable_motion_time_s"] == pytest.approx(0.0, abs=1e-9)
    assert summary["mean_speed_cm_s"] is None
    assert summary["max_speed_cm_s"] is None

    # 像素降级同样不能在真实曝光断口上凭空造段。
    pentagon = {"type": "polygon",
                "points": [[0, 50], [60, 50], [100, 70], [100, 90], [0, 90]]}
    result = ldb.process_unit(unit(dark=pentagon), rows,
                              timeline.Window(0.0, 2.0), None)
    summary = result["summary"]
    assert summary["coordinate_unit"] == "px"
    assert summary["total_distance_px"] == pytest.approx(0.0, abs=1e-9)
    assert summary["measurable_motion_time_s"] == pytest.approx(0.0, abs=1e-9)
    assert summary["mean_speed_px_s"] is None

    # 相邻连续点仍要连段。
    continuous = rows_at([(0.0, 0.0, 0.5, (10, 20)),
                          (1.0, 0.5, 1.5, (30, 20))])
    result = ldb.process_unit(unit(dark=FLAT_DARK), continuous,
                              timeline.Window(0.0, 2.0), None)
    summary = result["summary"]
    assert summary["measurable_motion_time_s"] == pytest.approx(1.0, abs=1e-9)
    assert summary["total_distance_cm"] > 0.0


def test_ldb_doorway_reset_zero_clears_pending_at_first_doorway_frame():
    doorway = {"type": "polygon",
               "points": [[10, 95], [190, 95], [190, 105], [10, 105]]}
    rows = rows_at([
        (0.0, 0.0, 0.1, (100, 50)),
        (0.1, 0.1, 0.2, (100, 50)),
        (0.2, 0.2, 0.3, (100, 100)),
        (0.3, 0.3, 0.4, (100, 50)),
        (0.4, 0.4, 0.5, (100, 50)),
        (0.5, 0.5, 0.6, (100, 50)),
    ])
    options = {"door_open_delay_s": 0.0, "transition_min_dwell_s": 0.2,
               "doorway_reset_s": 0.0}
    result = ldb.process_unit(unit(options=options, doorway=doorway), rows,
                              timeline.Window(0.0, 0.6), None)
    events = result["events"]
    assert [(item["kind"], item["time_s"], item["confirmed_s"])
            for item in events] == [("initial_chamber", 0.3, 0.5)]


# -- 6. 第 17 块有限修正：预览步骤与检测区域跟随可见性 -------------------

def _stub_frames(monkeypatch, preview_service):
    import numpy as np

    from behaviourlab.media import FrameInfo

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

    def fake_frame(*args, **kwargs):
        return FrameInfo(image=np.full((200, 200), 240, dtype=np.uint8),
                         index=0, time_s=0.0, pts=0, time_base=None,
                         clock_source="pts")

    monkeypatch.setattr(media, "VideoSource", Frames)
    monkeypatch.setattr(preview_service.media, "read_frame_at_time", fake_frame)
    monkeypatch.setattr(preview_service.media, "read_frame_info", fake_frame)
    monkeypatch.setattr(ldb.observation, "track",
                        lambda run_key, unit_id, **kwargs: Detection(
                            center=(100.0, 50.0), quality=1.0, contour=[],
                            mask=None))


def test_ldb_preview_does_not_require_hidden_dark(client, store, tmp_path,
                                                  monkeypatch):
    from behaviourlab import preview as preview_service

    hidden = {"id": UNIT, "name": "小鼠 1", "animal_id": None, "window": None,
              "geometry": geometry(dark=None),
              "options": {"dark_box_visible": False}, "detection": {}}
    _asset, video = asset_and_video(store, tmp_path, units=[hidden])
    draft = store.create_draft("ldb", videos=[video])
    assert client.get(f"/api/drafts/{draft['id']}/validate") \
        .get_json()["data"]["issues"] == []

    _stub_frames(monkeypatch, preview_service)
    asset_id = video["asset_id"]
    response = client.post(
        f"/api/drafts/{draft['id']}/preview",
        json={"asset_id": asset_id, "unit_id": UNIT,
              "frame_index": 0, "time_s": 0.0, "kind": "detection"},
    )
    assert response.status_code == 201, response.get_json()
    summary = response.get_json()["data"]["summary"]
    assert summary["missing_steps"] == []
    assert summary["drawn_steps"] == ["light"]
    # 显式 false 时检测区域只看明室。
    assert summary["detection"]["region"] == "light"
    warnings = response.get_json()["data"]["warnings"]
    assert not any("暗室边界" in text for text in warnings), warnings

    # 切回可见、又没有暗室几何：必须重新要求暗室边界。
    response = client.put(
        f"/api/drafts/{draft['id']}",
        json={"videos": [dict(video, units=[dict(hidden, options={})])]})
    assert response.status_code == 200, response.get_json()
    response = client.post(
        f"/api/drafts/{draft['id']}/preview",
        json={"asset_id": asset_id, "unit_id": UNIT,
              "frame_index": 0, "time_s": 0.0, "kind": "detection"},
    )
    assert response.status_code == 201, response.get_json()
    summary = response.get_json()["data"]["summary"]
    assert summary["missing_steps"] == ["暗室边界"]


def test_ldb_preview_detection_region_uses_merged_defaults(client, store,
                                                           tmp_path,
                                                           monkeypatch):
    from behaviourlab import preview as preview_service

    _asset, video = asset_and_video(store, tmp_path)
    draft = store.create_draft("ldb", videos=[video])

    sparse_unit = {"id": UNIT, "name": "小鼠 1", "animal_id": None,
                   "window": None, "geometry": geometry(), "options": {},
                   "detection": {}}
    sparse_video = dict(video, options={}, units=[sparse_unit])
    response = client.put(f"/api/drafts/{draft['id']}",
                          json={"options": {}, "videos": [sparse_video]})
    assert response.status_code == 200, response.get_json()
    assert client.get(f"/api/drafts/{draft['id']}/validate") \
        .get_json()["data"]["issues"] == []

    _stub_frames(monkeypatch, preview_service)
    response = client.post(
        f"/api/drafts/{draft['id']}/preview",
        json={"asset_id": video["asset_id"], "unit_id": UNIT,
              "frame_index": 0, "time_s": 0.0, "kind": "behavior"},
    )
    assert response.status_code == 201, response.get_json()
    summary = response.get_json()["data"]["summary"]
    assert summary["detection"]["region"] == "light+dark"
    assert summary["behavior"]["dark_box_visible"] is True


def test_ldb_result_page_lists_manifest_figures(client, store):
    facts = ldb.process_unit(unit(), source_rows(light_rows()),
                             timeline.Window(0.0, 4.0), 0.0)
    run_id = "a" * 32
    job_dir = Path(store.paths.runs) / run_id / ("b" * 32)
    job_dir.mkdir(parents=True, exist_ok=True)
    result = {"assay": "ldb", "summary": [facts["summary"]],
              "events": facts["events"], "bins": facts["bins"],
              "observations": [], "tables": {}, "warnings": [],
              "metadata": {"assay": "ldb", "coordinate_unit": "cm",
                           "units": {}, "warnings": []}}
    (job_dir / "results.json").write_text(json.dumps(result), encoding="utf-8")
    manifest = {
        "status": "completed", "run_id": run_id, "job_id": "b" * 32,
        "units": [{"unit_id": UNIT, "unit_name": "小鼠 1",
                   "summary": facts["summary"],
                   "figures": {"stem": "u", "trajectory": "u_trajectory.png"}}],
        "files": ["results.json", "u_trajectory.png"],
    }
    (job_dir / "result_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8")
    (job_dir / "u_trajectory.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    page = client.get(f"/results/{run_id}").get_data(as_text=True)
    # 有图按清单显示，不因为当前单位没有图形算法就一律隐藏。
    assert "u_trajectory.png" in page
    assert "暂无图表" not in page
    assert 'src=""' not in page
