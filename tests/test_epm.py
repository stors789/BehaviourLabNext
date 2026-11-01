"""EPM 高架十字迷宫：五区占用、入臂状态、估计尺度与页面接线。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from behaviourlab import experiments, timeline
from behaviourlab.experiments import epm

TOL = 1e-9


def rect(x1: float, y1: float, x2: float, y2: float) -> dict:
    return {"type": "quad",
            "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]}


def geometry() -> dict:
    return {"zones": {
        "center": rect(90, 90, 110, 110),
        "open_1": rect(111, 90, 190, 110),
        "open_2": rect(10, 90, 89, 110),
        "closed_1": rect(90, 10, 110, 89),
        "closed_2": rect(90, 111, 110, 190),
    }}


def unit(options=None, geom=None) -> dict:
    return {
        "id": "u" * 32, "name": "小鼠 1", "animal_id": None, "window": None,
        "geometry": geom or geometry(),
        "options": dict(epm.DEFAULTS, **(options or {})),
        "detection": {},
    }


def rows(points, spans=None, detected=None) -> list[dict]:
    """源观测：索引 i 默认 time_s=.1i、曝光 .1 秒。"""

    result = []
    for index, point in enumerate(points):
        start, end = ((spans[index]) if spans
                      else (round(0.1 * index, 10), round(0.1 * (index + 1), 10)))
        seen = (detected[index] if detected else point is not None)
        result.append({
            "unit_id": "u" * 32, "unit_name": "小鼠 1", "frame": index,
            "time_s": float(start), "source_dt_s": float(end - start),
            "elapsed_s": float(start), "analysis_fps": 10.0,
            "source_interval_start_s": float(start),
            "source_interval_end_s": float(end),
            "interval_start_s": float(start), "interval_end_s": float(end),
            "dt_s": float(end - start), "in_window": True,
            "detected": bool(seen),
            "x_px": None if not seen else float(point[0]),
            "y_px": None if not seen else float(point[1]),
            "quality": 1.0, "contour": [],
            "missing_reason": None if seen else "no_candidate",
            "head_angle": None, "head_quality": 0.0,
            "nose_x_px": None, "nose_y_px": None,
            "tail_x_px": None, "tail_y_px": None,
            "left_ear_x_px": None, "left_ear_y_px": None,
            "right_ear_x_px": None, "right_ear_y_px": None,
            "detection_method": "threshold",
        })
    return result


def test_zone_fixture_numbers_and_bins():
    points = [(150, 100), (150, 100), (100, 100), (100, 100), (100, 100),
              (150, 100), (100, 40)]
    facts = epm.process_unit(unit(options={"bin_seconds": 0.3}),
                             rows(points), timeline.Window(0.0, 0.7), None)
    summary = facts["summary"]
    # 手画尺度：8 个比例的平均 = (30/79 + 5/20)/2。
    assert summary["cm_per_px_est"] == pytest.approx(
        (30.0 / 79.0 + 0.25) / 2.0, abs=TOL)
    assert summary["coordinate_unit"] == "cm_estimated"
    assert summary["open_time_s"] == pytest.approx(0.3, abs=TOL)
    assert summary["closed_time_s"] == pytest.approx(0.1, abs=TOL)
    assert summary["center_time_s"] == pytest.approx(0.3, abs=TOL)
    assert summary["unknown_time_s"] == pytest.approx(0.0, abs=TOL)
    assert summary["open_entries"] == 2
    assert summary["closed_entries"] == 1
    assert summary["total_arm_entries"] == 3
    assert summary["open_time_percent_total"] == pytest.approx(
        100.0 * 0.3 / 0.7, abs=TOL)
    assert summary["open_time_percent_arms_only"] == pytest.approx(
        75.0, abs=TOL)
    assert summary["open_entry_percent"] == pytest.approx(
        100.0 * 2.0 / 3.0, abs=TOL)
    assert summary["first_open_entry_latency_s"] == pytest.approx(0.0, abs=TOL)
    assert summary["total_time_s"] == pytest.approx(0.7, abs=TOL)
    assert summary["observed_time_s"] == pytest.approx(0.7, abs=TOL)
    assert summary["missing_time_s"] == pytest.approx(0.0, abs=TOL)
    assert summary["total_distance_cm"] > 0

    events = [(item["target"], item["arm_group"], round(item["time_s"], 6))
              for item in facts["events"]]
    assert events == [("open_1", "open", 0.0), ("open_1", "open", 0.5),
                      ("closed_1", "closed", 0.6)]
    kinds = [(row["frame"], row["arm_entry_name"]) for row in facts["rows"]
             if row["arm_entry_event"]]
    assert kinds == [(0, "open_1"), (5, "open_1"), (6, "closed_1")]
    # 入臂深度按长轴投影，不按离中心距离。
    assert facts["rows"][0]["arm_depth"] == pytest.approx(
        (150.0 - 111.0) / 79.0, abs=TOL)

    bins = facts["bins"]
    assert [(item["bin_index"], item["open_entries"], item["closed_entries"])
            for item in bins] == [(1, 1, 0), (2, 1, 0), (3, 0, 1)]
    assert bins[0]["start_s"] == 0.0
    assert bins[2]["end_s"] == pytest.approx(0.7, abs=TOL)
    assert bins[2]["closed_time_s"] == pytest.approx(0.1, abs=TOL)
    assert bins[0]["open_time_percent_arms_only"] == pytest.approx(
        100.0, abs=TOL)


def test_center_reset_missing_and_unknown_rules():
    # 单条长曝光只算一次真实中心观测：未来曝光不能提前完成重置。
    long_stay = epm.process_unit(
        unit(),
        rows([(150, 100), (100, 100), (150, 100)],
             spans=[(0.0, 0.1), (1.0, 11.0), (11.0, 11.1)]),
        timeline.Window(0.0, 11.1), None)
    assert long_stay["summary"]["open_entries"] == 1
    assert [item["target"] for item in long_stay["events"]] == ["open_1"]

    # 两条相隔达到重置时长的真实中心观测才清除 last_arm。
    reset = epm.process_unit(
        unit(),
        rows([(150, 100), (100, 100), (100, 100), (150, 100)],
             spans=[(0.0, 0.1), (1.0, 1.1), (1.2, 1.3), (1.4, 1.5)]),
        timeline.Window(0.0, 1.5), None)
    assert reset["summary"]["open_entries"] == 2

    # 缺失回到同臂不计第二次；换臂立即计。
    gap = epm.process_unit(
        unit(),
        rows([(150, 100), None, (150, 100), (100, 40)],
             detected=[True, False, True, True]),
        timeline.Window(0.0, 0.4), None)
    assert [item["target"] for item in gap["events"]] == ["open_1", "closed_1"]
    assert gap["summary"]["open_entries"] == 1
    assert gap["summary"]["closed_entries"] == 1
    assert gap["summary"]["missing_time_s"] == pytest.approx(0.1, abs=TOL)

    # 区外是 unknown 而不是封闭臂；运动链包含标区外的真实位置。
    outside = epm.process_unit(
        unit(),
        rows([(150, 100), (10, 10), (100, 40)]),
        timeline.Window(0.0, 0.3), None)
    assert outside["rows"][1]["zone"] == "unknown"
    assert outside["rows"][1]["zone_group"] == "unknown"
    assert outside["summary"]["unknown_time_s"] == pytest.approx(0.1, abs=TOL)
    assert outside["summary"]["closed_time_s"] == pytest.approx(0.1, abs=TOL)
    assert [item["target"] for item in outside["events"]] == [
        "open_1", "closed_1"]
    # 未知位置仍产生完成段：三行两段。
    assert len(outside["segments"]) == 2
    assert outside["summary"]["total_distance_cm"] > 0


def test_registration_validation_preview_and_page(client, store, tmp_path,
                                                  monkeypatch):
    from behaviourlab import media, preview
    from behaviourlab.media import FrameInfo

    entry = [item for item in experiments.catalog()
             if item["key"] == "epm"][0]
    assert entry["label"] == "高架十字迷宫 / EPM"
    assert [item["label"] for item in entry["geometry"]] == [
        "中心区四角", "开放臂 1 四角", "开放臂 2 四角", "封闭臂 1 四角",
        "封闭臂 2 四角", "检测遮罩"]
    assert [bool(item["required"]) for item in entry["geometry"]] == [
        True, True, True, True, True, False]
    assert entry["template_keys"] == ["zones", "center_size_cm",
                                      "open_arm_length_cm",
                                      "closed_arm_length_cm"]
    assert set(entry["defaults"]) == set(epm.DEFAULTS)

    geom = geometry()
    assert epm.validate_unit(dict(epm.DEFAULTS), geom, complete=True) == []
    # 缺一个区域就按字段报缺；未画完仍可保存。
    partial = {"zones": {key: value for key, value in geom["zones"].items()
                         if key != "closed_2"}}
    assert epm.validate_unit(dict(epm.DEFAULTS), partial) == []
    assert [item["field"] for item in epm.validate_unit(
        dict(epm.DEFAULTS), partial, complete=True)] == [
        "geometry.zones.closed_2"]
    # 非凸/自交叉四角被拒绝。
    crossed = {"zones": dict(geom["zones"]) | {
        "center": {"type": "quad", "points": [[90, 90], [110, 110],
                                              [90, 110], [110, 90]]}}}
    assert [item["field"] for item in epm.validate_unit(
        dict(epm.DEFAULTS), crossed, complete=True)] == ["geometry.zones.center"]
    # 参数范围。
    for options, field in (
            ({"center_size_cm": 0}, "options.center_size_cm"),
            ({"open_arm_length_cm": 0}, "options.open_arm_length_cm"),
            ({"entry_depth_ratio": 1.5}, "options.entry_depth_ratio"),
            ({"entry_reset_center_s": -0.1}, "options.entry_reset_center_s"),
            ({"bin_seconds": 0}, "options.bin_seconds")):
        assert [item["field"] for item in epm.validate_unit(
            dict(epm.DEFAULTS, **options), geom, complete=True)] == [field]

    source = tmp_path / "epm.mp4"
    source.write_bytes(b"placeholder")
    asset = store.create_asset(
        name="epm.mp4", path=str(source), project_id=None,
        metadata={"width": 240, "height": 240, "duration_s": 4.0,
                  "fps": 10.0, "frame_count": 40})
    draft = store.create_draft("epm", videos=[{
        "asset_id": asset["id"],
        "window": {"start_s": 0.0, "end_s": 4.0},
        "orientation": 0, "fps_override": None, "sample_fps": 10,
        "units": [{"id": "e" * 32, "name": "EPM 鼠", "window": None,
                   "geometry": geom, "options": {}, "detection": {},
                   "manual": None}],
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

    response = client.post(
        f"/api/drafts/{draft['id']}/preview",
        json={"asset_id": asset["id"], "unit_id": "e" * 32,
              "time_s": 1.5, "kind": "detection"})
    assert response.status_code == 201, response.get_json()
    data = response.get_json()["data"]
    assert data["summary"]["missing_steps"] == []
    assert data["summary"]["drawn_steps"] == [
        "center", "open_1", "open_2", "closed_1", "closed_2"]
    assert client.get(data["image_url"]).status_code == 200

    behavior = client.post(
        f"/api/drafts/{draft['id']}/preview",
        json={"asset_id": asset["id"], "unit_id": "e" * 32,
              "time_s": 1.5, "kind": "behavior"})
    assert behavior.status_code == 201, behavior.get_json()
    behavior_data = behavior.get_json()["data"]
    assert behavior_data["summary"]["behavior"]["mode"] == "epm"
    assert behavior_data["summary"]["missing_steps"] == []

    started = client.post(f"/api/drafts/{draft['id']}/analyze")
    assert started.status_code == 201, started.get_json()
    payload = started.get_json()["data"]
    job = payload["jobs"][0]
    assert job["status"] == "done"
    summary = job["manifest"]["units"][0]["summary"]
    assert summary["total_time_s"] == pytest.approx(4.0, abs=1e-9)
    assert summary["open_entries"] >= 1
    assert summary["cm_per_px_est"] > 0
    assert summary["coordinate_unit"] == "cm_estimated"
    figures = job["manifest"]["units"][0]["figures"]
    assert figures["zones"]

    from behaviourlab import outputs  # noqa: F401  保证导出层已加载

    run_dir = Path(store.paths.runs) / payload["run_id"] / job["job_id"]
    stored = (run_dir / figures["zones"]).read_bytes()
    inline = client.get(
        f"/runs/{payload['run_id']}/{job['job_id']}/{figures['zones']}")
    assert inline.status_code == 200
    assert inline.mimetype == "image/png"
    download = client.get(
        f"/runs/{payload['run_id']}/{job['job_id']}/{figures['zones']}"
        "?download=1")
    assert download.status_code == 200
    assert download.data == stored

    import csv

    with open(run_dir / "observations.csv", encoding="utf-8-sig") as handle:
        header = next(csv.reader(handle))
    for column in ("zone", "zone_group", "arm_entry_event", "arm_entry_name",
                   "step_distance_cm", "speed_cm_s",
                   "cumulative_distance_cm"):
        assert column in header

    page = client.get(f"/results/{payload['run_id']}").get_data(as_text=True)
    for label in ("开放臂时间", "封闭臂时间", "中心区时间",
                  "开放臂时间占总时长", "开放臂时间占臂区时长",
                  "开放臂进入次数", "首次进入开放臂潜伏期", "估计总距离",
                  "估计平均速度", "区域占用时长（秒，估计）"):
        assert label in page

    metadata = job["manifest"]["metadata"]["units"]["e" * 32]
    assert metadata["coordinate_unit"] == "cm_estimated"
    assert metadata["cm_per_px_est"] == pytest.approx(
        summary["cm_per_px_est"], abs=TOL)
    assert "missing exposure is counted as missing only" in metadata[
        "missing_rule"]


class PreviewSource:
    """内存解码头：小鼠停在开放臂 1 中部。"""

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
            cv2.ellipse(image, (150, 100), (10, 22), 0, 0, 360, (0, 0, 0), -1)
            self._frames.append(Frame(
                image=image, index=index, time_s=moment, dt_s=step,
                elapsed_s=moment - begin, analysis_fps=1.0 / step,
                interval_start_s=moment, interval_end_s=moment + step))

    def iter_frames(self):
        return iter(self._frames)


def test_editor_bridge_writes_nested_zones(client, store, tmp_path,
                                            monkeypatch):
    """Node 编辑产物走保存→校验→分析，五区落在 geometry.zones。"""

    import json
    import shutil
    import subprocess

    from behaviourlab import media, preview
    from behaviourlab.media import FrameInfo

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node 不可用")
    script = Path(__file__).parent / "js" / "test_epm_editor.cjs"
    result = subprocess.run([node, str(script)], capture_output=True,
                            text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "epm-editor OK" in result.stdout
    # 与 nor 一致：桥接落盘的结构必须能通过最终校验。
    assert epm.validate_unit(dict(epm.DEFAULTS), geometry(),
                             complete=True) == []

    source = tmp_path / "epm-ui.mp4"
    source.write_bytes(b"placeholder")
    asset = store.create_asset(
        name="epm-ui.mp4", path=str(source), project_id=None,
        metadata={"width": 240, "height": 240, "duration_s": 4.0,
                  "fps": 10.0, "frame_count": 40})
    draft = store.create_draft("epm", videos=[{
        "asset_id": asset["id"],
        "window": {"start_s": 0.0, "end_s": 4.0},
        "orientation": 0, "fps_override": None, "sample_fps": 10,
        "units": [{"id": "z" * 32, "name": "EPM 鼠", "window": None,
                   "geometry": {}, "options": {}, "detection": {},
                   "manual": None}],
    }])

    def save(unit_geometry):
        return client.put(f"/api/drafts/{draft['id']}", json={"videos": [{
            "asset_id": asset["id"],
            "window": {"start_s": 0.0, "end_s": 4.0},
            "orientation": 0, "fps_override": None, "sample_fps": 10,
            "units": [{"id": "z" * 32, "name": "EPM 鼠", "window": None,
                       "geometry": unit_geometry, "options": {},
                       "detection": {}, "manual": None}],
        }]})

    # 只画好部分区域也能保存，最终校验才要求五区齐全。
    half = {"zones": {"center": geometry()["zones"]["center"]}}
    assert save(half).status_code == 200
    assert epm.validate_unit(dict(epm.DEFAULTS), half, complete=True) != []

    full = geometry()
    assert save(full).status_code == 200, "save failed"
    stored = client.get(f"/api/drafts/{draft['id']}").get_json()["data"]
    assert stored["draft"]["videos"][0]["units"][0]["geometry"] == full
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

    started = client.post(f"/api/drafts/{draft['id']}/analyze")
    assert started.status_code == 201, started.get_json()
    job = started.get_json()["data"]["jobs"][0]
    summary = job["manifest"]["units"][0]["summary"]
    assert summary["total_time_s"] == pytest.approx(4.0, abs=1e-9)
    assert summary["open_entries"] >= 1
    assert summary["zone_seconds"]["open"] > 0
    assert job["manifest"]["units"][0]["figures"]["zones"]
