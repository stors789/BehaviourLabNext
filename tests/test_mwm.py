"""MWM Morris 水迷宫：固定手算夹具、probe 空值与预览/页面接线。"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from behaviourlab import experiments, timeline
from behaviourlab.experiments import mwm

TOL = 1e-9
ARENA = {"type": "ellipse", "center": [100.0, 100.0],
         "radii": [60.0, 30.0], "angle": 0.0}
PLATFORM = {"type": "point", "points": [[130.0, 100.0]]}
OPTIONS = dict(mwm.DEFAULTS, platform_dwell_s=1.0, bin_seconds=2.0)


def fixture_rows(positions=None, trial: str = "hidden") -> list[dict]:
    if positions is None:
        positions = [(100.0, 100.0), (120.0, 100.0), (130.0, 100.0),
                     (130.0, 100.0), None, (140.0, 100.0)]
    rows = []
    for index, point in enumerate(positions):
        detected = point is not None
        rows.append({
            "unit_id": "u" * 32, "unit_name": "小鼠 1", "frame": index,
            "time_s": float(index), "source_dt_s": 1.0,
            "elapsed_s": float(index), "analysis_fps": 1.0,
            "source_interval_start_s": float(index),
            "source_interval_end_s": float(index + 1),
            "interval_start_s": float(index),
            "interval_end_s": float(index + 1), "dt_s": 1.0,
            "in_window": True, "detected": detected,
            "x_px": None if point is None else point[0],
            "y_px": None if point is None else point[1],
            "quality": 1.0, "contour": [],
            "missing_reason": None if detected else "no_candidate",
            "head_angle": None, "head_quality": 0.0,
            "nose_x_px": None, "nose_y_px": None,
            "tail_x_px": None, "tail_y_px": None,
            "left_ear_x_px": None, "left_ear_y_px": None,
            "right_ear_x_px": None, "right_ear_y_px": None,
            "detection_method": "threshold",
            "trial_type": trial,
        })
    return rows


def fixture_unit(options=None) -> dict:
    return {
        "id": "u" * 32, "name": "小鼠 1", "animal_id": None,
        "window": None,
        "geometry": {"arena": ARENA, "platform": PLATFORM},
        "options": dict(OPTIONS, **(options or {})),
        "detection": {},
    }


def test_training_fixture_numbers_and_no_bridging():
    unit = fixture_unit()
    facts = mwm.process_unit(unit, fixture_rows(),
                             timeline.Window(0.0, 6.0), None)
    summary = facts["summary"]
    assert summary["escape_latency_s"] == pytest.approx(3.0, abs=TOL)
    assert summary["first_platform_entry_latency_s"] == pytest.approx(
        2.0, abs=TOL)
    assert summary["path_length_to_platform_cm"] == pytest.approx(
        30.0, abs=TOL)
    assert summary["platform_entries"] == 1
    assert summary["platform_time_s"] == pytest.approx(2.0, abs=TOL)
    assert summary["total_distance_cm"] == pytest.approx(30.0, abs=TOL)
    assert summary["mean_distance_to_platform_cm"] == pytest.approx(
        10.0, abs=TOL)
    assert (summary["path_weighted_distance_to_platform_cm"]
            == pytest.approx(15.0, abs=TOL))
    assert summary["mean_speed_cm_s"] == pytest.approx(10.0, abs=TOL)
    assert summary["moving_time_s"] == pytest.approx(2.0, abs=TOL)
    assert summary["floating_time_s"] == pytest.approx(1.0, abs=TOL)
    assert summary["valid_pool_time_s"] == pytest.approx(5.0, abs=TOL)
    assert summary["observed_time_s"] == pytest.approx(5.0, abs=TOL)
    assert summary["missing_time_s"] == pytest.approx(1.0, abs=TOL)
    assert summary["target_quadrant_resolved"] == "NE"
    assert summary["escape_censored"] is False
    assert summary["path_efficiency"] == pytest.approx(1.0, abs=TOL)
    assert summary["search_strategy"] == "direct"

    kinds = [(event["kind"], event["time_s"], event["confirmed_s"])
             for event in facts["events"]]
    assert kinds == [("platform_entry", 2.0, 2.0),
                     ("platform_confirmed", 2.0, 3.0)]

    # 缺失不桥接：第 4 行没有坐标，第 5 行不产生 t3→t5 的段。
    assert facts["rows"][4]["x_cm"] is None
    assert facts["rows"][5]["x_cm"] == pytest.approx(40.0, abs=TOL)
    assert facts["segments"][-1]["end_s"] == pytest.approx(3.0, abs=TOL)

    bins = facts["bins"]
    assert [bin["bin_index"] for bin in bins] == [1, 2, 3]
    assert [bin["start_s"] for bin in bins] == [0.0, 2.0, 4.0]
    assert bins[0]["total_distance_cm"] == pytest.approx(30.0, abs=TOL)
    assert bins[1]["platform_time_s"] == pytest.approx(2.0, abs=TOL)
    assert bins[2]["valid_pool_time_s"] == pytest.approx(1.0, abs=TOL)


def test_probe_reports_entries_without_confirmation():
    unit = fixture_unit({"trial_type": "probe"})
    facts = mwm.process_unit(unit, fixture_rows(),
                             timeline.Window(0.0, 6.0), None)
    summary = facts["summary"]
    assert summary["platform_entries"] == 1
    assert summary["platform_time_s"] == pytest.approx(2.0, abs=TOL)
    assert summary["escape_latency_s"] is None
    assert summary["path_length_to_platform_cm"] is None
    assert summary["path_efficiency"] is None
    assert summary["escape_censored"] is None
    assert [event["kind"] for event in facts["events"]] == ["platform_entry"]

    # 单个长曝光不能提前确认，也不能伪造 platform_entry。
    rows = fixture_rows([(130.0, 100.0)])
    rows[0].update({"interval_end_s": 10.0, "source_interval_end_s": 10.0,
                    "dt_s": 10.0})
    unit = fixture_unit()
    single = mwm.process_unit(unit, rows, timeline.Window(0.0, 10.0), None)
    assert single["summary"]["platform_time_s"] == pytest.approx(10.0, abs=TOL)
    assert single["summary"]["platform_entries"] == 0
    assert single["summary"]["escape_latency_s"] is None
    assert single["summary"]["first_platform_observation_latency_s"] == \
        pytest.approx(0.0, abs=TOL)
    assert single["events"] == []


class PreviewSource:
    """内存解码头：从池心走到平台后停留。"""

    def __init__(self, path, *, start_s=0.0, end_s=None, **kwargs):
        from behaviourlab.schema import Frame

        self._frames = []
        begin = float(start_s or 0.0)
        stop = float(end_s if end_s is not None else begin + 5.0)
        step = 0.1
        count = max(1, int(round((stop - begin) / step)))
        for index in range(count):
            moment = round(begin + index * step, 6)
            # 池心 (120,120)，平台 (150,120)；前 10 帧走到平台，再停留。
            offset = min(30.0, index * 3.0) if begin == 0.0 else 30.0
            image = np.full((240, 240, 3), 255, dtype=np.uint8)
            cv2.ellipse(image, (int(120 + offset), 120), (10, 22), 0, 0, 360,
                        (0, 0, 0), -1)
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
             if item["key"] == "mwm"][0]
    assert entry["label"] == "Morris 水迷宫 / MWM"
    assert [item["label"] for item in entry["geometry"]] == [
        "泳池边界（椭圆或至少五点）", "平台中心或平台椭圆", "检测遮罩"]
    assert entry["template_keys"] == ["pool_diameter_cm",
                                      "platform_radius_cm"]
    assert set(entry["defaults"]) == set(mwm.DEFAULTS)

    # 象限辅助线与物理变换一致（与前端 mwmLines 同值）。
    rotated = {"arena": {"type": "ellipse", "center": [100.0, 100.0],
                         "radii": [60.0, 30.0], "angle": 30.0}}
    guides = mwm.quadrant_guides(rotated, {"pool_diameter_cm": 120.0,
                                           "coordinate_rotation_deg": 25.0})
    assert guides["directions"][0] == pytest.approx(
        [0.8870492613930958, 0.461674785820022], abs=1e-9)
    assert guides["directions"][1] == pytest.approx(
        [0.343250149967973, -0.9392440228965868], abs=1e-9)

    geometry = {"arena": ARENA, "platform": PLATFORM}
    assert mwm.validate_unit(dict(mwm.DEFAULTS), geometry,
                             complete=True) == []
    outside = {"arena": ARENA,
               "platform": {"type": "point", "points": [[300.0, 100.0]]}}
    assert [item["field"] for item in mwm.validate_unit(
        dict(mwm.DEFAULTS), outside, complete=True)] == ["geometry.platform"]
    assert [item["field"] for item in mwm.validate_unit(
        dict(mwm.DEFAULTS, platform_radius_cm=80.0), geometry,
        complete=True)] == ["options.platform_radius_cm"]
    assert [item["field"] for item in mwm.validate_unit(
        dict(mwm.DEFAULTS, start_position="X"), geometry,
        complete=True)] == ["options.start_position"]

    # 平台椭圆只取中心；泳池也接受五点以上的椭圆拟合。
    ellipse_platform = {"type": "ellipse", "center": [130.0, 100.0],
                        "radii": [10.0, 6.0], "angle": 0.0}
    assert mwm.validate_unit(dict(mwm.DEFAULTS),
                             {"arena": ARENA, "platform": ellipse_platform},
                             complete=True) == []
    fitted_arena = {"type": "ellipse", "points": [
        [160.0, 100.0], [142.4, 121.2], [100.0, 130.0], [57.6, 121.2],
        [40.0, 100.0], [57.6, 78.8], [100.0, 70.0], [142.4, 78.8]]}
    assert mwm.validate_unit(dict(mwm.DEFAULTS),
                             {"arena": fitted_arena, "platform": PLATFORM},
                             complete=True) == []

    pool = dict(ARENA, center=[120.0, 120.0], radii=[80.0, 50.0])
    platform = {"type": "point", "points": [[150.0, 120.0]]}
    source = tmp_path / "mwm.mp4"
    source.write_bytes(b"placeholder")
    asset = store.create_asset(
        name="mwm.mp4", path=str(source), project_id=None,
        metadata={"width": 240, "height": 240, "duration_s": 5.0,
                  "fps": 10.0, "frame_count": 50})
    draft = store.create_draft("mwm", videos=[{
        "asset_id": asset["id"],
        "window": {"start_s": 0.0, "end_s": 5.0},
        "orientation": 0, "fps_override": None, "sample_fps": 10,
        "units": [{
            "id": "m" * 32, "name": "MWM 鼠", "window": None,
            "geometry": {"arena": pool, "platform": platform},
            "options": dict(mwm.DEFAULTS), "detection": {}, "manual": None,
        }],
    }])
    monkeypatch.setattr(media, "VideoSource", PreviewSource)
    monkeypatch.setattr(
        "behaviourlab.engine.media.probe_video",
        lambda path: {"width": 240, "height": 240, "duration_s": 5.0,
                      "fps": 10.0, "frame_count": 50})
    monkeypatch.setattr(
        preview.media, "read_frame_at_time",
        lambda *args, **kwargs: FrameInfo(
            image=np.full((240, 240, 3), 255, dtype=np.uint8), index=20,
            time_s=2.0, pts=None, time_base=None, clock_source="pts"))

    bad = store.create_draft("mwm", videos=[{
        "asset_id": asset["id"],
        "window": {"start_s": 0.0, "end_s": 5.0},
        "orientation": 0, "fps_override": None, "sample_fps": 10,
        "units": [{
            "id": "b" * 32, "name": "越界平台", "window": None,
            "geometry": {"arena": pool,
                         "platform": {"type": "point",
                                      "points": [[300.0, 120.0]]}},
            "options": dict(mwm.DEFAULTS), "detection": {}, "manual": None,
        }],
    }])
    rejected = client.post(f"/api/drafts/{bad['id']}/analyze")
    assert rejected.status_code == 400, rejected.get_json()
    fields = [item["field"] for item in
              rejected.get_json()["error"]["fields"]]
    assert "videos[0].units[0].geometry.platform" in fields
    assert store.list_jobs() == []

    response = client.post(
        f"/api/drafts/{draft['id']}/preview",
        json={"asset_id": asset["id"], "unit_id": "m" * 32,
              "time_s": 2.0, "kind": "behavior"})
    assert response.status_code == 201, response.get_json()
    behavior = response.get_json()["data"]["summary"]["behavior"]
    assert behavior["mode"] == "mwm"
    # [t−1, t+3) 与视频窗 [0,5) 的交集为 [1,5)，共 4 秒。
    assert behavior["total_time_s"] == pytest.approx(4.0, abs=1e-9)
    assert behavior["platform_entries"] >= 0
    assert behavior["preview_note"].startswith("行为预览")

    started = client.post(f"/api/drafts/{draft['id']}/analyze")
    assert started.status_code == 202, started.get_json()
    payload = started.get_json()["data"]
    job = payload["jobs"][0]
    assert job["status"] == "completed"
    summary = job["manifest"]["units"][0]["summary"]
    assert summary["total_time_s"] == pytest.approx(5.0, abs=1e-9)
    assert summary["platform_entries"] >= 1
    assert summary["escape_latency_s"] is not None
    figures = job["manifest"]["units"][0]["figures"]
    assert figures["trajectory"] and figures["heatmap"]
    unit_meta = job["manifest"]["metadata"]["units"]["m" * 32]
    assert unit_meta["coordinate_origin"] == "pool center"
    assert unit_meta["occupancy_denominator"] == "有效池内曝光"

    page = client.get(f"/results/{payload['run_id']}").get_data(as_text=True)
    for label in ("平台确认潜伏期", "首次进入平台区潜伏期", "确认平台前路径",
                  "平台区进入次数", "平台区时间", "总路径",
                  "探索性规则标签，不是经验证的行为分类。"):
        assert label in page


# -- 复验反例：旋转角范围、轴上象限、曝光断口、平台圈半径 ----------------

def _draft(unit: dict) -> dict:
    return {
        "assay": "mwm", "options": {},
        "videos": [{"asset_id": "a" * 32, "options": {},
                    "window": {"start_s": 0.0, "end_s": 6.0},
                    "units": [unit]}],
    }


def _unit(options: dict, geometry: dict | None = None) -> dict:
    return {
        "id": "u" * 32, "name": "小鼠 1", "animal_id": None, "window": None,
        "geometry": geometry or {"arena": ARENA, "platform": PLATFORM},
        "options": options, "detection": {},
    }


def _span_rows(points, spans) -> list[dict]:
    """按给定 (start, end) 曝光构造源观察；点缺失即 detected=False。"""

    rows = []
    for index, (point, (start, end)) in enumerate(zip(points, spans)):
        detected = point is not None
        rows.append({
            "unit_id": "u" * 32, "unit_name": "小鼠 1", "frame": index,
            "time_s": float(start), "source_dt_s": float(end - start),
            "elapsed_s": float(start), "analysis_fps": 1.0,
            "source_interval_start_s": float(start),
            "source_interval_end_s": float(end),
            "interval_start_s": float(start), "interval_end_s": float(end),
            "dt_s": float(end - start), "in_window": True,
            "detected": detected,
            "x_px": None if not detected else float(point[0]),
            "y_px": None if not detected else float(point[1]),
            "quality": 1.0, "contour": [],
            "missing_reason": None if detected else "no_candidate",
            "head_angle": None, "head_quality": 0.0,
            "nose_x_px": None, "nose_y_px": None,
            "tail_x_px": None, "tail_y_px": None,
            "left_ear_x_px": None, "left_ear_y_px": None,
            "right_ear_x_px": None, "right_ear_y_px": None,
            "detection_method": "threshold",
        })
    return rows


def test_signed_rotation_angle_range_shared_by_save_and_analyze():
    from behaviourlab import engine

    geometry = {"arena": ARENA, "platform": PLATFORM}
    for value in (-360.0, -30.0, 0.0, 360.0):
        options = dict(mwm.DEFAULTS, coordinate_rotation_deg=value)
        assert mwm.validate_unit(options, geometry, complete=True) == []
        unit = _unit(options)
        draft = _draft(unit)
        # 保存（complete=False）、最终开始（complete=True）与任务校验同源。
        assert engine.validate_draft(draft) == []
        assert engine.validate_draft(draft, complete=True) == []
        assert mwm.validate({"units": [unit]}) == []
        facts = mwm.process_unit(unit, fixture_rows(),
                                 timeline.Window(0.0, 6.0), None)
        assert facts["metadata"]["coordinate_rotation_deg"] == pytest.approx(
            value, abs=TOL)

    for value in (-360.001, 360.001, -720.0):
        issues = mwm.validate_unit(
            dict(mwm.DEFAULTS, coordinate_rotation_deg=value), geometry,
            complete=True)
        assert [item["field"] for item in issues] == [
            "options.coordinate_rotation_deg"]
        assert issues[0]["message"] == "坐标轴旋转必须在 -360 到 360 之间"
        saved = engine.validate_draft(
            _draft(_unit(dict(mwm.DEFAULTS,
                              coordinate_rotation_deg=value))))
        assert [(item["field"], item["message"]) for item in saved] == [
            ("videos[0].units[0].options.coordinate_rotation_deg",
             "坐标轴旋转必须在 -360 到 360 之间")]

    for value in (True, "30", None):
        issues = mwm.validate_unit(
            dict(mwm.DEFAULTS, coordinate_rotation_deg=value), geometry,
            complete=True)
        assert [item["field"] for item in issues] == [
            "options.coordinate_rotation_deg"]
        assert issues[0]["message"] == "必须是有限数值"


def test_axis_point_after_quarter_turn_stays_northeast():
    arena = {"type": "ellipse", "center": [100.0, 100.0],
             "radii": [60.0, 30.0], "angle": 0.0}
    platform = {"type": "point", "points": [[100.0, 115.0]]}
    unit = _unit(dict(mwm.DEFAULTS, coordinate_rotation_deg=90.0,
                      bin_seconds=2.0),
                 {"arena": arena, "platform": platform})
    facts = mwm.process_unit(unit, fixture_rows([(100.0, 115.0)] * 2),
                             timeline.Window(0.0, 2.0), None)
    first = facts["rows"][0]
    # 椭圆归一化(0,30) → 翻 y(0,−30) → 旋转 90°(30,0)。
    assert first["x_cm"] == pytest.approx(30.0, abs=TOL)
    assert first["y_cm"] == pytest.approx(0.0, abs=TOL)
    assert first["quadrant"] == "NE"
    assert facts["summary"]["target_quadrant_resolved"] == "NE"
    assert facts["summary"]["quadrant_ne_time_s"] == pytest.approx(2.0, abs=TOL)
    assert facts["summary"]["quadrant_se_time_s"] == pytest.approx(0.0, abs=TOL)
    assert facts["summary"]["target_quadrant_time_s"] == pytest.approx(
        2.0, abs=TOL)

    # 非轴上点与真实负坐标不受归零影响。
    plain = mwm.process_unit(fixture_unit(), fixture_rows([(95.0, 85.0)] * 2),
                             timeline.Window(0.0, 2.0), None)
    assert plain["rows"][0]["x_cm"] == pytest.approx(-5.0, abs=TOL)
    assert plain["rows"][0]["y_cm"] == pytest.approx(30.0, abs=TOL)
    assert plain["rows"][0]["quadrant"] == "NW"

    slight = mwm.process_unit(fixture_unit(),
                              fixture_rows([(100.0, 100.001)] * 2),
                              timeline.Window(0.0, 2.0), None)
    assert slight["rows"][0]["y_cm"] == pytest.approx(-0.002, abs=1e-12)
    assert slight["rows"][0]["quadrant"] == "SE"


def test_exposure_gap_between_two_points_is_not_a_path():
    unit = fixture_unit()
    # t=0/1，源曝光 [0,.5)/[1,1.5)，物理点 (0,0)→(20,0)，中间 .5 秒无曝光。
    rows = _span_rows([(100.0, 100.0), (120.0, 100.0)],
                      [(0.0, 0.5), (1.0, 1.5)])
    facts = mwm.process_unit(unit, rows, timeline.Window(0.0, 2.0), None)
    summary = facts["summary"]
    assert facts["segments"] == []
    assert summary["total_distance_cm"] == 0
    assert summary["measurable_motion_time_s"] == 0
    assert summary["mean_speed_cm_s"] is None
    assert summary["max_speed_cm_s"] is None
    for item in facts["bins"]:
        assert item["total_distance_cm"] == 0
        assert item["measurable_motion_time_s"] == 0
    assert facts["rows"][0]["step"] is None
    assert facts["rows"][1]["step"] is None

    # 真正相接的曝光仍成段。
    joined = _span_rows([(100.0, 100.0), (120.0, 100.0)],
                        [(0.0, 0.5), (0.5, 1.0)])
    solid = mwm.process_unit(unit, joined, timeline.Window(0.0, 2.0), None)
    assert len(solid["segments"]) == 1
    assert solid["summary"]["total_distance_cm"] == pytest.approx(20.0, abs=TOL)
    assert solid["summary"]["mean_speed_cm_s"] == pytest.approx(40.0, abs=TOL)

    # 窗口端点仍用真实上下文插值裁剪，不删除边界外的相邻点。
    three = _span_rows([(100.0, 100.0), (120.0, 100.0), (140.0, 100.0)],
                       [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5)])
    clipped = mwm.process_unit(unit, three, timeline.Window(0.1, 0.7), None)
    assert len(clipped["segments"]) == 2
    first, second = clipped["segments"]
    assert first["start_s"] == pytest.approx(0.1, abs=TOL)
    assert first["end_s"] == pytest.approx(0.5, abs=TOL)
    assert first["p0"] == pytest.approx([4.0, 0.0], abs=TOL)
    assert first["p1"] == pytest.approx([20.0, 0.0], abs=TOL)
    assert second["start_s"] == pytest.approx(0.5, abs=TOL)
    assert second["end_s"] == pytest.approx(0.7, abs=TOL)
    assert second["p0"] == pytest.approx([20.0, 0.0], abs=TOL)
    assert second["p1"] == pytest.approx([28.0, 0.0], abs=TOL)


def test_platform_circle_uses_configured_radius(monkeypatch, tmp_path):
    from matplotlib.figure import Figure

    from behaviourlab import figures

    facts = mwm.process_unit(fixture_unit(), fixture_rows(),
                             timeline.Window(0.0, 6.0), None)
    rows = facts["rows"]
    seen: list[list[float]] = []

    def fake_savefig(self, *args, **kwargs):
        seen.append([float(patch.radius) for axis in self.axes
                     for patch in axis.patches if hasattr(patch, "radius")])

    monkeypatch.setattr(Figure, "savefig", fake_savefig)

    assert figures.mwm_trajectory_png(
        tmp_path / "traj.png", rows, 60.0, platform=(30.0, 0.0),
        platform_radius=0.25, start=(0.0, 60.0), title="轨迹")
    assert figures.mwm_heatmap_png(
        tmp_path / "heat.png", rows, 60.0, platform=(30.0, 0.0),
        platform_radius=0.25, title="热图")
    assert len(seen) == 2
    trajectory, heatmap = seen
    for radii in (trajectory, heatmap):
        assert 0.25 in radii
        assert 0.5 not in radii
        assert 60.0 in radii
