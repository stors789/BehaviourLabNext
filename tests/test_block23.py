"""第 23 块三项固定自检：输出文件/图形尺寸、叠画时间轴、关键点误差。"""

from __future__ import annotations

import csv
import io
import math
from pathlib import Path

import numpy as np
import pytest

from behaviourlab import annotated, figures, landmarks, outputs
from behaviourlab.schema import AnalysisResult, Frame


# -- 夹具 1：文件与图形 ---------------------------------------------------


def _rows():
    """t=0/1/3/6，曝光 1/2/3/1 秒；第 2 帧缺失，后两点有效。

    位置与物理坐标一致；只有最后两个相邻有效观测构成完成段（保存在起点行），
    第一帧没有有效后继，不产生任何段。
    """

    return [
        {"unit_id": "u" * 32, "frame": 0, "time_s": 0.0, "dt_s": 1.0,
         "in_window": True, "detected": True, "x_px": 1.0, "y_px": 1.0,
         "x_cm": 1.0, "y_cm": 1.0, "motion_x": 1.0, "motion_y": 1.0},
        {"unit_id": "u" * 32, "frame": 1, "time_s": 1.0, "dt_s": 2.0,
         "in_window": True, "detected": False, "x_px": None, "y_px": None},
        {"unit_id": "u" * 32, "frame": 2, "time_s": 3.0, "dt_s": 3.0,
         "in_window": True, "detected": True, "x_px": 10.0, "y_px": 10.0,
         "x_cm": 10.0, "y_cm": 10.0, "motion_x": 10.0, "motion_y": 10.0,
         "motion_x0": 10.0, "motion_y0": 10.0, "motion_x1": 11.0,
         "motion_y1": 10.0},
        {"unit_id": "u" * 32, "frame": 3, "time_s": 6.0, "dt_s": 1.0,
         "in_window": True, "detected": True, "x_px": 11.0, "y_px": 10.0,
         "x_cm": 11.0, "y_cm": 10.0, "motion_x": 11.0, "motion_y": 10.0},
    ]


def test_output_module_files_csv_and_sizes(tmp_path):
    rows = _rows()
    summary = [{"unit_id": "u" * 32, "unit_name": "小鼠, 一",
                "analysis_window_start_s": 0.0, "analysis_window_end_s": 7.0,
                "arena_width_cm": 20.0, "arena_height_cm": 20.0,
                "ratio": None}]
    result = AnalysisResult(
        observations=rows, summary=summary, events=[],
        bins=[], tables={"zones": []}, warnings=[],
        metadata={"assay": "oft", "units": {"u" * 32: {}}},
    )
    job = {
        "schema_version": 1, "run_id": "r" * 32, "job_id": "j" * 32,
        "assay": "oft",
        "asset": {"id": "a" * 32, "name": "open field, 1.mp4", "path": "x"},
        "window": {"start_s": 0.0, "end_s": 7.0},
        "orientation": 0, "fps_override": None, "sample_fps": None,
        "options": {"arena_width_cm": 20.0, "arena_height_cm": 20.0},
        "detection": {}, "units": [{"id": "u" * 32, "name": "小鼠, 一",
                                    "geometry": {}}],
        "output": {"annotated_video": False},
        "style": {"color": "#B86245", "colormap": "viridis",
                  "line_width": 2, "dpi": 90},
        "algorithm_version": "0.1.0",
    }
    manifest = outputs.write_outputs(tmp_path, job, result)

    assert manifest["status"] == "completed"
    assert manifest["files"] == sorted(manifest["files"], key=manifest["files"].index)
    # 空事件表仍有七公共表头。
    header = (tmp_path / "events.csv").read_text(encoding="utf-8-sig").splitlines()[0]
    assert header == ("event_id,unit_id,kind,time_s,confirmed_s,target,source")
    # 空附加表也保留 unit_id/unit_name 头，且不因空被跳过。
    assert "zones.csv" in manifest["files"]
    zones_header = (tmp_path / "zones.csv").read_text(
        encoding="utf-8-sig").splitlines()[0]
    assert zones_header.startswith("unit_id,unit_name")
    # null 在 JSON 是 null、在 CSV 是空单元格；中文/逗号/数组可回读。
    summary_csv = list(csv.DictReader(
        io.StringIO((tmp_path / "summary.csv").read_text(encoding="utf-8-sig"))))
    assert summary_csv[0]["unit_name"] == "小鼠, 一"
    assert summary_csv[0]["ratio"] == ""
    import json

    payload = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert payload["summary"][0]["ratio"] is None
    # 轨迹只连后两点：完成段由保存的 motion_x0..y1 决定，缺口不跨。
    assert figures.motion_segments(rows) == [(10.0, 10.0, 11.0, 10.0)]


def test_heat_weights_and_figure_pixel_size(tmp_path):
    rows = _rows()
    # 有效点曝光 1 + 3 + 1 秒；缺失点不进入热图。
    grid, count = figures.heat_grid(rows, 20.0, 20.0)
    assert count == 3
    assert math.isclose(sum(sum(line) for line in grid), 5.0, abs_tol=1e-9)

    style = {"dpi": 90, "colormap": "viridis", "line_width": 2,
             "color": "#B86245"}
    target = tmp_path / "traj.png"
    import cv2

    ok = figures.trajectory_png(target, rows, 20.0, 20.0, style=style)
    assert ok is True, "完成段画出轨迹"
    image = cv2.imread(str(target))
    assert image is not None
    assert image.shape[1] == 540 and image.shape[0] == 450, "6×5 英寸 @90dpi"
    heat = tmp_path / "heat.png"
    assert figures.heatmap_png(heat, rows, 20.0, 20.0, style=style) is True
    image = cv2.imread(str(heat))
    assert image is not None
    assert image.shape[1] == 540 and image.shape[0] == 450, "6×5 英寸 @90dpi"


def test_trajectory_connects_only_completed_segments(tmp_path):
    rows = [
        {"motion_x0": 1.0, "motion_y0": 1.0, "motion_x1": 10.0,
         "motion_y1": 10.0},
        {"motion_x0": 10.0, "motion_y0": 10.0, "motion_x1": 11.0,
         "motion_y1": 10.0},
        {"motion_x0": None, "motion_y0": None, "motion_x1": None,
         "motion_y1": None},
    ]
    segments = figures.motion_segments(rows)
    assert segments == [(1.0, 1.0, 10.0, 10.0), (10.0, 10.0, 11.0, 10.0)]
    assert len(segments) == 2, "缺失之后不产生跨缺口的段"


def test_unlisted_paths_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        outputs.table_filename("../bad")
    with pytest.raises(ValueError):
        outputs.table_filename("events")
    assert outputs.table_filename("zones") == "zones.csv"


# -- 夹具 2：叠画时间轴 ---------------------------------------------------


class _Source128:
    """128×96 源，帧在 0/.2/.../1.0，每帧曝光 .2 秒。"""

    def __init__(self, path=None, *, start_s=0.0, end_s=None,
                 decode_images=True, **kwargs):
        self._frames = []
        for index in range(6):
            moment = round(0.2 * index, 6)
            low = max(moment, start_s)
            high = moment + 0.2
            if end_s is not None:
                high = min(high, end_s)
            if high - low <= 1e-12:
                continue
            self._frames.append(Frame(
                image=(np.full((96, 128), 200, dtype=np.uint8)
                       if decode_images else None),
                index=index, time_s=moment, dt_s=0.2, elapsed_s=moment,
                analysis_fps=5.0, interval_start_s=low, interval_end_s=high))
        self.coverage = {"start_s": None, "end_s": None, "duration_s": 0.0,
                         "frame_count": 0}

    def iter_frames(self):
        return iter(self._frames)


def test_annotated_timeline_and_event_display_time(tmp_path, monkeypatch):
    monkeypatch.setattr(annotated.media, "VideoSource", _Source128)
    job = {"asset": {"path": "memory", "metadata": {"fps": 5.0}},
           "window": {"start_s": 0.1, "end_s": 0.9},
           "orientation": 90, "fps_override": None}
    units = [{"unit_id": "u" * 32, "unit_name": "unit",
              "window_start_s": 0.3, "window_end_s": 0.7,
              "geometry": {}, "observations": []}]
    events = [{"time_s": 0.3, "confirmed_s": 0.6, "target": "A",
               "kind": "zone_entry", "unit_id": "u" * 32}]

    # 事件显示时刻取 confirmed_s，不从 time_s 提前出现。
    moment, confirmed = annotated._display_time(events[0])
    assert moment == 0.6 and confirmed is True

    target = tmp_path / "annotated.mp4"
    written, seconds = annotated.write_run_video(
        target, job=job, units=units, events=events)

    assert written == 7
    assert seconds == pytest.approx(0.8, abs=1e-6)

    import av

    with av.open(str(target)) as container:
        stream = container.streams.video[0]
        frames = list(container.decode(stream))
        time_base = stream.time_base
    assert len(frames) == 7
    assert frames[0].to_ndarray(format="bgr24").shape[:2] == (96, 128)
    starts = [round(float(item.pts * time_base), 6) for item in frames]
    durations = [round(float(item.duration * time_base), 6) for item in frames]
    assert starts == [0.0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7]
    assert durations == [0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1]
    assert sum(durations) == pytest.approx(0.8, abs=1e-6)
    # 绝对端点量化：每帧起点与真实切分点相差不超过 1µs。
    bounds = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8]
    for index, item in enumerate(frames):
        got = float(item.pts * time_base)
        assert abs(got - (bounds[index] - 0.1)) <= 1e-6


def test_annotated_cancel_removes_partial(tmp_path, monkeypatch):
    monkeypatch.setattr(annotated.media, "VideoSource", _Source128)
    job = {"asset": {"path": "memory", "metadata": {"fps": 5.0}},
           "window": {"start_s": 0.0, "end_s": 1.0},
           "orientation": 0, "fps_override": None}
    units = [{"unit_id": "u" * 32, "unit_name": "unit",
              "window_start_s": 0.0, "window_end_s": 1.0,
              "geometry": {}, "observations": []}]
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 1

    target = tmp_path / "cancel.mp4"
    with pytest.raises(OSError):
        annotated.write_run_video(target, job=job, units=units, cancel=cancel)
    assert not target.exists(), "取消后不留未完成视频"


# -- 夹具 3：关键点误差公式 ----------------------------------------------


def test_landmark_error_metrics_formula():
    records = [{"frame": index,
                "points": {"body": {"visible": True,
                                    "x": manual[0], "y": manual[1]}}}
               for index, manual in enumerate(
                   [(53.0, 40.0), (54.0, 40.0), (50.0, 40.0)])]
    predicted = {0: {"body": [50.0, 40.0]}, 1: {"body": [50.0, 40.0]},
                 2: {"body": None}}
    evaluation = landmarks.compute_evaluation(
        records, predicted, {0: 0.0, 1: 0.1, 2: 0.2}, tolerance=3.5,
        supported=("body",))

    metrics = evaluation["metrics"]
    assert metrics["visible"] == 3
    assert metrics["covered"] == 2
    assert metrics["hit"] == 1
    assert metrics["coverage"] == pytest.approx(2 / 3)
    assert metrics["hit_rate"] == pytest.approx(1 / 3)
    assert metrics["matched_error_median_px"] == pytest.approx(3.5)
    assert metrics["matched_error_p90_px"] == pytest.approx(3.9)
    # 缺失预测的第三帧不从分母删除：可见 3，命中 1。
    assert evaluation["records"][2]["points"]["body"]["error_px"] is None
    assert evaluation["records"][2]["points"]["body"]["hit"] is False


def test_landmark_unsupported_point_is_visible_but_not_covered():
    records = [{"frame": 0, "points": {
        "left_ear": {"visible": True, "x": 10.0, "y": 10.0}}}]
    evaluation = landmarks.compute_evaluation(
        records, {0: {"left_ear": None}}, {0: 0.0}, tolerance=3.0,
        supported=("body",))
    entry = evaluation["records"][0]["points"]["left_ear"]
    assert entry["prediction"] is None
    assert entry["error_px"] is None
    assert entry["hit"] is False
    assert evaluation["by_point"]["left_ear"]["visible"] == 1
    assert evaluation["metrics"]["covered"] == 0
    assert evaluation["metrics"]["coverage"] == 0


def test_landmark_session_fields(tmp_path):
    """会话字段：frames 排序去重拒绝、点名与坐标校验、PUT 字段白名单。"""

    from behaviourlab.config import DataPaths
    from behaviourlab.schema import ValidationError
    from behaviourlab.store import Store

    source = tmp_path / "source.mp4"
    source.write_bytes(b"\x00")
    paths = DataPaths.from_dir(tmp_path / "data")
    store = Store(paths)
    asset = store.create_asset(name="nor.mp4", path=str(source))
    unit = {"id": "u" * 32, "name": "unit", "geometry": {"arena": {}},
            "options": {}, "detection": {}}

    # 默认 assay=nor；frames 排序保存；身份冻结；评估初始为空。
    session = landmarks.create_session(store, paths, {
        "asset_id": asset["id"], "frames": [4, 0, 2], "unit": unit,
        "records": [], "tolerance_px": 12.5,
    })
    assert session["assay"] == "nor"
    assert session["frames"] == [0, 2, 4]
    assert session["tolerance_px"] == 12.5
    assert session["evaluation"] is None
    assert session["asset_identity"]["size"] == source.stat().st_size
    assert landmarks.get_session(paths, session["id"])["id"] == session["id"]

    # 重复帧、负数帧、小数帧都拒绝，且不静默去重。
    for frames in ([1, 1], [-1], [1.5], []):
        with pytest.raises(ValidationError):
            landmarks.create_session(store, paths, {
                "asset_id": asset["id"], "frames": frames, "unit": unit})

    # 未知点名拒绝；visible=true 要求有限坐标。
    with pytest.raises(ValidationError):
        landmarks.create_session(store, paths, {
            "asset_id": asset["id"], "frames": [0], "unit": unit,
            "records": [{"frame": 0, "points": {
                "whisker": {"visible": True, "x": 1.0, "y": 1.0}}}]})
    with pytest.raises(ValidationError):
        landmarks.create_session(store, paths, {
            "asset_id": asset["id"], "frames": [0], "unit": unit,
            "records": [{"frame": 0, "points": {
                "body": {"visible": True, "x": None, "y": 1.0}}}]})

    # PUT 只允许 records 与 tolerance_px；替换 records 会清空评估。
    landmarks.save_evaluation(paths, session, {"metrics": {}})
    updated = landmarks.update_session(paths, session["id"], {
        "tolerance_px": 4.0,
        "records": [{"frame": 0, "points": {
            "body": {"visible": False, "x": None, "y": None}}}],
    })
    assert updated["tolerance_px"] == 4.0
    assert updated["evaluation"] is None
    assert updated["records"][0]["points"]["body"] == {
        "visible": False, "x": None, "y": None}
    with pytest.raises(ValidationError):
        landmarks.update_session(paths, session["id"], {"frames": [7]})
    # 非法会话 ID 不能逃逸目录，读取直接返回不存在。
    assert landmarks.get_session(paths, "../escape") is None
    store.close()


def test_result_page_unit_switch_and_file_rail(client, store):
    """结果页：动物选择器、manifest 图、视频播放器、文件栏与空态。"""

    import json
    from pathlib import Path

    run_id = "b" * 32
    run_dir = Path(store.paths.runs) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summaries = [
        {"unit_id": "u1", "unit_name": "甲鼠", "mode": "oft",
         "arena_width_cm": 20.0, "arena_height_cm": 20.0},
        {"unit_id": "u2", "unit_name": "乙鼠", "mode": "oft",
         "arena_width_cm": 20.0, "arena_height_cm": 20.0},
    ]
    (run_dir / "results.json").write_text(json.dumps(
        {"assay": "oft", "summary": summaries, "events": [], "bins": [],
         "observations": [], "tables": {}, "warnings": ["样本警告"],
         "metadata": {"assay": "oft", "units": {}}}), encoding="utf-8")
    for name in ("unit_01_trajectory.png", "unit_02_trajectory.png"):
        (run_dir / name).write_bytes(b"png")
    (run_dir / "annotated.mp4").write_bytes(b"mp4")
    manifest = {
        "status": "completed", "run_id": run_id, "job_id": "j" * 32,
        "assay": "oft", "asset": {"id": "a" * 32, "name": "双鼠.mp4"},
        "coverage": None, "algorithm_version": "0.1.0",
        "units": [
            {"unit_id": "u1", "unit_name": "甲鼠", "summary": summaries[0],
             "figures": {"trajectory": "unit_01_trajectory.png"},
             "figure_files": ["unit_01_trajectory.png"],
             "annotated_video": "annotated.mp4"},
            {"unit_id": "u2", "unit_name": "乙鼠", "summary": summaries[1],
             "figures": {}, "figure_files": [], "annotated_video": None},
        ],
        "figures": {"u1": ["unit_01_trajectory.png"], "u2": []},
        "annotated_video": "annotated.mp4",
        "files": ["results.json", "unit_01_trajectory.png",
                  "unit_02_trajectory.png", "annotated.mp4"],
        "warnings": ["样本警告"],
        "metadata": {},
    }
    (run_dir / "result_manifest.json").write_text(json.dumps(manifest),
                                                  encoding="utf-8")

    page = client.get(f"/results/{run_id}").get_data(as_text=True)
    # 顶部：源视频名、下载全部主操作与打开目录次要动作。
    assert "双鼠.mp4" in page
    assert f'href="/api/runs/{run_id}/bundle"' in page
    assert 'id="open-run-dir"' in page
    assert "样本警告" in page
    # 动物选择器列出两只动物；第二只默认隐藏，图与视频各自渲染。
    assert 'id="unit-switch"' in page
    assert 'data-unit-id="u2"' in page and "hidden" in page
    job = "j" * 32
    assert f'src="/runs/{run_id}/{job}/unit_01_trajectory.png"' in page
    assert '<video controls preload="metadata"' in page
    assert f'src="/runs/{run_id}/{job}/annotated.mp4"' in page
    assert "data-empty=\"figures\"" in page
    assert "data-empty=\"video\"" in page
    # 文件栏只列声明文件并全部带下载链接；脚本接线存在。
    assert 'data-file-rail' in page
    assert f"/runs/{run_id}/{job}/annotated.mp4?download=1" in page
    assert "/static/result.js" in page
    # 声明之外的文件不出现在页面链接里。
    assert "frozen_job.json" not in page
