"""LDB 明暗箱：可见/不可见固定夹具、校验与界面接线。"""

from __future__ import annotations

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


def unit(*, dark_visible=True, dark=DARK, options=None):
    merged = dict(OPTIONS)
    merged["dark_box_visible"] = dark_visible
    if options:
        merged.update(options)
    return {"id": UNIT, "name": "小鼠 1", "animal_id": None, "window": None,
            "geometry": geometry(dark), "options": merged, "detection": {}}


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
    assert job["status"] == "done", job
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
