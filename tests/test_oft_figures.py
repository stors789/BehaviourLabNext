"""第 08 块收口：每动物轨迹/热图、清单映射与结果页交付。"""

from __future__ import annotations

import json
import re
import re

import pytest

from behaviourlab import figures, outputs, worker
from behaviourlab.experiments import oft
from behaviourlab.schema import AnalysisResult

UNIT_A = "a" * 32
UNIT_B = "b" * 32
ARENA = {"type": "quad", "points": [[0, 0], [199, 0], [199, 199], [0, 199]]}


def summary(unit_id: str, name: str, width: float, height: float) -> dict:
    return {
        "unit_id": unit_id, "unit_name": name,
        "arena_width_cm": width, "arena_height_cm": height,
        "center_ratio_x": 0.5, "center_ratio_y": 0.5,
        "wall_margin_cm": 8.0, "movement_threshold_cm_s": 1.0,
        "bin_seconds": 60.0,
        "total_time_s": 4.0, "observed_time_s": 4.0, "missing_time_s": 0.0,
        "center_time_s": 2.0, "outer_time_s": 2.0, "outside_time_s": 0.0,
        "wall_time_s": 0.0, "center_percent": 50.0, "outer_percent": 50.0,
        "outside_percent": 0.0, "wall_percent": 0.0,
        "moving_time_s": 0.6, "immobility_time_s": 3.2,
        "measurable_motion_time_s": 3.8,
        "total_distance_cm": 30.0, "observed_total_distance_cm": 30.0,
        "center_distance_cm": 25.0, "outer_distance_cm": 5.0,
        "outside_distance_cm": 0.0,
        "analysis_window_start_s": 0.0, "analysis_window_end_s": 4.0,
        "detected_frames": 20, "window_frames": 20,
        "detected_rate_percent": 100.0, "center_entries": 1,
        "center_latency_s": 1.0, "mean_speed_cm_s": 150.0 / 19.0,
        "max_speed_cm_s": 50.0,
    }


def row(unit_id: str, index: int, moment: float, dt: float,
        x: float | None, y: float | None, *, segment=None) -> dict:
    item = {"unit_id": unit_id, "frame": index, "time_s": moment,
            "dt_s": dt, "in_window": True, "detected": x is not None,
            "x_cm": x, "y_cm": y, "motion_x0": None, "motion_y0": None,
            "motion_x1": None, "motion_y1": None}
    if segment:
        item["motion_x0"], item["motion_y0"] = segment[0]
        item["motion_x1"], item["motion_y1"] = segment[1]
    return item


def result_two_units() -> AnalysisResult:
    rows = []
    # 两只动物各两段，中间断开：轨迹只连各自保存的段。
    rows.append(row(UNIT_A, 0, 0.0, 1.0, 5.0, 5.0,
                    segment=((0.0, 5.0), (10.0, 5.0))))
    rows.append(row(UNIT_A, 1, 1.0, 1.0, 15.0, 5.0))
    rows.append(row(UNIT_A, 2, 2.0, 1.0, None, None))
    rows.append(row(UNIT_A, 3, 3.0, 1.0, 30.0, 15.0,
                    segment=((30.0, 15.0), (40.0, 15.0))))
    rows.append(row(UNIT_B, 0, 0.5, 0.5, 60.0, 30.0,
                    segment=((60.0, 30.0), (70.0, 30.0))))
    rows.append(row(UNIT_B, 1, 1.0, 0.5, 75.0, 30.0))
    return AnalysisResult(
        observations=rows,
        summary=[summary(UNIT_A, "小鼠 A", 50.0, 50.0),
                 summary(UNIT_B, "小鼠 B", 80.0, 60.0)],
        events=[], bins=[], warnings=["示例警告"],
        metadata={"assay": "oft", "coordinate_unit": "cm"},
    )


def job_payload() -> dict:
    return {"run_id": "r" * 32, "job_id": "j" * 32, "assay": "oft",
            "style": {"color": "#B86245", "colormap": "viridis",
                      "line_width": 2, "dpi": 100},
            "asset": {"id": "asset", "name": "旷场.mp4", "path": "memory",
                      "metadata": {}},
            "window": {"start_s": 0.0, "end_s": 4.0}}


def test_export_writes_two_figures_per_unit(tmp_path):
    manifest = outputs.export_result(tmp_path, job_payload(), result_two_units())
    names = manifest["files"]
    units = {unit["unit_id"]: unit for unit in manifest["units"]}
    png = bytes.fromhex("89504e470d0a1a0a")

    for unit_id in (UNIT_A, UNIT_B):
        info = units[unit_id]["figures"]
        for key in ("trajectory", "heatmap"):
            name = info[key]
            assert name in names
            assert re.fullmatch(r"[A-Za-z0-9_-]+_(?:trajectory|heatmap)\.png",
                                name), name
            assert (tmp_path / name).read_bytes()[:8] == png
    assert units[UNIT_A]["figures"]["trajectory"] !=         units[UNIT_B]["figures"]["trajectory"]
    # 8 个基础文件 + 4 张图，清单自身与报告仍在。
    assert len([name for name in names if name.endswith(".png")]) == 4
    for base in ("observations.csv", "summary.csv", "events.csv", "bins.csv",
                 "frozen_job.json", "results.json", "result_manifest.json",
                 "report.md"):
        assert base in names
        assert (tmp_path / base).is_file()
    stored = json.loads((tmp_path / "result_manifest.json").read_text("utf-8"))
    assert stored["units"][0]["figures"] == units[UNIT_A]["figures"]


def test_heatmap_weights_are_seconds_per_bin():
    rows = [row(UNIT_A, 0, 0.0, 1.0, 5.0, 5.0),
            row(UNIT_A, 1, 1.0, 3.0, 15.0, 15.0),
            row(UNIT_A, 2, 4.0, 1.0, 25.0, 25.0)]
    grid, count = figures.heat_grid(rows, 50.0, 50.0)
    # 40 箱覆盖 50cm：每箱 1.25cm；5/15/25cm 分别落在第 4/12/20 箱。
    assert count == 3
    assert grid[4][4] == pytest.approx(1.0, abs=1e-12)
    assert grid[12][12] == pytest.approx(3.0, abs=1e-12)
    assert grid[20][20] == pytest.approx(1.0, abs=1e-12)
    assert sum(sum(line) for line in grid) == pytest.approx(5.0, abs=1e-12)


def test_trajectory_uses_only_saved_segments():
    rows = result_two_units().observations
    segments = figures.motion_segments([item for item in rows
                                        if item["unit_id"] == UNIT_A])
    assert len(segments) == 2
    assert segments[0] == (0.0, 5.0, 10.0, 5.0)
    assert segments[1] == (30.0, 15.0, 40.0, 15.0)


def test_missing_positions_skip_figures_with_note(tmp_path):
    empty = AnalysisResult(
        observations=[row(UNIT_A, 0, 0.0, 1.0, None, None)],
        summary=[summary(UNIT_A, "小鼠 A", 50.0, 50.0)],
        warnings=[], metadata={"assay": "oft"},
    )
    manifest = outputs.export_result(tmp_path, job_payload(), empty)
    info = manifest["units"][0]["figures"]
    assert info["trajectory"] is None and info["heatmap"] is None
    assert "轨迹图" in info["note"] and "热图" in info["note"]
    assert not [name for name in manifest["files"] if name.endswith(".png")]


def test_result_page_shows_figures_and_downloads(client, store, tmp_path,
                                                 monkeypatch):
    source = tmp_path / "oft.mp4"
    source.write_bytes(b"placeholder")
    asset = store.create_asset(
        name="旷场.mp4", path=str(source), project_id=None,
        metadata={"width": 200, "height": 200, "duration_s": 1.0,
                  "fps": 5.0, "frame_count": 5})
    draft = store.create_draft("oft", videos=[{
        "asset_id": asset["id"],
        "window": {"start_s": 0.0, "end_s": 1.0},
        "orientation": 0, "fps_override": None, "sample_fps": None,
        "units": [{"id": "d" * 32, "name": "小鼠 1", "window": None,
                   "geometry": {"arena": ARENA},
                   "options": dict(oft.DEFAULTS), "detection": {},
                   "manual": None}],
    }])
    monkeypatch.setattr(
        "behaviourlab.engine.media.probe_video",
        lambda path: {"width": 200, "height": 200, "duration_s": 1.0,
                      "fps": 5.0, "frame_count": 5})
    monkeypatch.setattr(worker, "run_frozen_job",
                        lambda job, **kwargs: result_two_units())

    response = client.post(f"/api/drafts/{draft['id']}/analyze")
    assert response.status_code == 202, response.get_json()
    data = response.get_json()["data"]
    run_id = data["run_id"]
    job_id = data["jobs"][0]["job_id"]
    figures_written = [name for name in data["jobs"][0]["manifest"]["files"]
                       if name.endswith(".png")]
    assert len(figures_written) == 4

    page = client.get(f"/results/{run_id}")
    body = page.get_data(as_text=True)
    assert "小鼠 A" in body and "小鼠 B" in body
    assert "壁区包含在外围内" in body
    assert "示例警告" in body
    for name in figures_written:
        assert f'src="/runs/{run_id}/{job_id}/{name}"' in body
        inline = client.get(f"/runs/{run_id}/{job_id}/{name}")
        assert inline.status_code == 200
        assert inline.mimetype == "image/png"
        download = client.get(f"/runs/{run_id}/{job_id}/{name}?download=1")
        assert download.status_code == 200
        assert "attachment" in download.headers.get("Content-Disposition", "")
    assert client.get(f"/runs/{run_id}/{job_id}/secret.txt").status_code == 404
    assert client.get(f"/runs/{run_id}/{job_id}/x_trajectory.txt"
                      ).status_code == 404
