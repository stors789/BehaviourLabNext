"""第 16 块第 4 组：人工分箱原文键与标注视频输出合同。"""

from __future__ import annotations

import numpy as np
import pytest

from behaviourlab import annotated, media, outputs
from behaviourlab.schema import AnalysisResult, Frame

UNIT = "u" * 32


class Frames:
    """内存源：10 张 128×96 灰度 240 图，t=.1i、曝光 .1 秒。"""

    def __init__(self, path=None, *, start_s=0.0, end_s=None,
                 decode_images=True, **kwargs):
        self._frames = []
        begin = float(start_s or 0.0)
        stop = float(end_s if end_s is not None else begin + 1.0)
        index = 0
        moment = begin
        while moment < stop - 1e-9:
            image = (np.full((96, 128), 240, dtype=np.uint8)
                     if decode_images else None)
            end = round(moment + 0.1, 6)
            self._frames.append(Frame(
                image=image, index=index, time_s=round(moment, 6), dt_s=0.1,
                elapsed_s=round(moment - begin, 6), analysis_fps=10.0,
                interval_start_s=round(moment, 6), interval_end_s=end))
            index += 1
            moment = end

    def iter_frames(self):
        return iter(self._frames)


class SilentFrames(Frames):
    """能解码但没有任何可写画面：用于编码失败的明确报错。"""

    def iter_frames(self):
        return iter([])


ENTRIES = [{"time_s": 0.15, "arm": "A"}, {"time_s": 0.35, "arm": "A"},
           {"time_s": 0.55, "arm": "B"}, {"time_s": 0.75, "arm": "C"}]


def manual_unit(entries=None, *, bin_seconds=0.4):
    return {
        "id": UNIT, "name": "人工鼠", "animal_id": None,
        "window": {"mode": "exact", "start_s": 0.1, "end_s": 0.9},
        "geometry": {},
        "options": {"bin_seconds": bin_seconds},
        "detection": {},
        "manual": {"mode": "manual",
                   "entries": [dict(item) for item in
                               (ENTRIES if entries is None else entries)]},
    }


def seed(client, store, tmp_path, *, annotated=True):
    source = tmp_path / "y.mp4"
    source.write_bytes(b"placeholder")
    asset = store.create_asset(
        name="y.mp4", path=str(source), project_id=None,
        metadata={"width": 128, "height": 96, "duration_s": 1.0,
                  "fps": 10.0, "frame_count": 10})
    draft = store.create_draft(
        "ymaze", output={"annotated_video": annotated},
        videos=[{
            "asset_id": asset["id"],
            "window": {"start_s": 0.0, "end_s": 1.0},
            "orientation": 0, "fps_override": None, "sample_fps": None,
            "units": [manual_unit()],
        }])
    return draft


# -- 1. 分箱原文键 -----------------------------------------------------

def test_manual_bins_expose_original_keys(client, store, tmp_path,
                                          monkeypatch):
    monkeypatch.setattr(media, "VideoSource", Frames)
    monkeypatch.setattr(
        "behaviourlab.engine.media.probe_video",
        lambda path: {"width": 128, "height": 96, "duration_s": 1.0,
                      "fps": 10.0, "frame_count": 10})
    draft = seed(client, store, tmp_path)
    started = client.post("/api/jobs", json={"draft_id": draft["id"]})
    assert started.status_code == 201, started.get_json()
    job = started.get_json()["data"]["jobs"][0]
    assert job["status"] == "done", job
    run_id = started.get_json()["data"]["run_id"]

    stored = client.get(
        f"/runs/{run_id}/{job['job_id']}/results.json").get_json()
    header = client.get(
        f"/runs/{run_id}/{job['job_id']}/bins.csv").get_data(as_text=True)
    first_line = header.splitlines()[0].lstrip("﻿")
    assert "entries" in first_line.split(",")
    assert "alternations" in first_line.split(",")
    header = client.get(
        f"/runs/{run_id}/{job['job_id']}/bins.csv").get_data(as_text=True)
    first_line = header.splitlines()[0].lstrip("\ufeff")
    assert "entries" in first_line.split(",")
    assert "alternations" in first_line.split(",")
    bins = stored["bins"]
    assert [(b["entries"], b["alternations"]) for b in bins] == [(2, 0), (2, 1)]
    assert [b["alternation_opportunities"] for b in bins] == [0, 2]
    assert [b["alternation_percent"] for b in bins] == [None, 50.0]
    assert [b["duration_s"] for b in bins] == [0.4, 0.4]
    # 兼容字段与原文键同源。
    assert [b["total_arm_entries"] for b in bins] == [2, 2]
    assert [b["alternation_count"] for b in bins] == [0, 1]
    assert stored["summary"][0]["total_arm_entries"] == 4

    page = client.get(f"/results/{run_id}").get_data(as_text=True)
    assert "标注视频" in page


# -- 2. 标注视频：开启时真实存在、可解码、可下载 ------------------------

def test_annotated_video_is_generated_and_downloadable(client, store, tmp_path,
                                                       monkeypatch):
    import av

    monkeypatch.setattr(media, "VideoSource", Frames)
    monkeypatch.setattr(
        "behaviourlab.engine.media.probe_video",
        lambda path: {"width": 128, "height": 96, "duration_s": 1.0,
                      "fps": 10.0, "frame_count": 10})
    draft = seed(client, store, tmp_path)
    started = client.post("/api/jobs", json={"draft_id": draft["id"]})
    payload = started.get_json()["data"]
    job = payload["jobs"][0]
    assert job["status"] == "done", job
    unit = job["manifest"]["units"][0]
    name = unit["annotated_video"]
    assert name and name.endswith("_annotated.mp4")
    assert name in job["manifest"]["files"]

    run_dir = store.paths.runs / payload["run_id"] / job["job_id"]
    target = run_dir / name
    assert target.is_file() and target.stat().st_size > 0

    with av.open(str(target)) as container:
        stream = container.streams.video[0]
        assert stream.codec_context.name == "h264", "浏览器可播的编码"
        assert float(container.duration) / av.time_base == pytest.approx(
            0.8, abs=1e-4)
        frames = list(container.decode(video=0))
    assert len(frames) == 8, "标注视频只覆盖单位分析窗 [0.1, 0.9)"
    assert frames[0].to_ndarray(format="bgr24").shape[:2] == (96, 128)
    assert [round(float(item.duration * item.time_base), 6) for item in frames] \
        == [0.1] * 8
    assert unit["annotated_video_duration_s"] == pytest.approx(0.8, abs=1e-4)
    assert unit["annotated_video_time_base"] == "1/1000000"

    inline = client.get(f"/runs/{payload['run_id']}/{job['job_id']}/{name}")
    assert inline.status_code == 200
    assert inline.mimetype == "video/mp4"
    assert inline.data == target.read_bytes()
    download = client.get(
        f"/runs/{payload['run_id']}/{job['job_id']}/{name}?download=1")
    assert download.status_code == 200
    assert download.data == target.read_bytes()
    assert "attachment" in download.headers["Content-Disposition"]

    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "标注视频" in report and name in report
    assert f"]({name})" not in report, "MP4 不能拼成图片引用"


# -- 3. 关闭开关：不生成视频，事件与汇总不变 ---------------------------

def test_annotated_video_off_changes_nothing_else(client, store, tmp_path,
                                                  monkeypatch):
    monkeypatch.setattr(media, "VideoSource", Frames)
    monkeypatch.setattr(
        "behaviourlab.engine.media.probe_video",
        lambda path: {"width": 128, "height": 96, "duration_s": 1.0,
                      "fps": 10.0, "frame_count": 10})
    on = seed(client, store, tmp_path, annotated=True)
    off = seed(client, store, tmp_path, annotated=False)

    first = client.post("/api/jobs", json={"draft_id": on["id"]}).get_json()["data"]
    second = client.post("/api/jobs", json={"draft_id": off["id"]}).get_json()["data"]
    assert first["jobs"][0]["status"] == "done"
    assert second["jobs"][0]["status"] == "done"

    unit = second["jobs"][0]["manifest"]["units"][0]
    assert unit["annotated_video"] is None
    names = second["jobs"][0]["manifest"]["files"]
    assert not [item for item in names if item.endswith(".mp4")]

    results = []
    for payload in (first, second):
        data = client.get(
            f"/runs/{payload['run_id']}/{payload['jobs'][0]['job_id']}/results.json"
        ).get_json()
        results.append((data["summary"], data["events"], data["bins"]))
    assert results[0] == results[1], "开关不能改变事件、汇总或分箱"

    page = client.get(f"/results/{second['run_id']}").get_data(as_text=True)
    assert "<video" not in page


# -- 4. 编码失败明确报错，不静默完成 -----------------------------------

def test_annotated_video_failure_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(media, "VideoSource", SilentFrames)
    summary = {
        "unit_id": UNIT, "unit_name": "人工鼠", "mode": "ymaze",
        "analysis_window_start_s": 0.1, "analysis_window_end_s": 0.9,
        "total_time_s": 0.8, "total_arm_entries": 4,
        "alternation_count": 1, "alternation_opportunities": 2,
        "alternation_percent": 50.0, "arm_A_entries": 2,
        "arm_B_entries": 1, "arm_C_entries": 1, "bin_seconds": 0.4,
    }
    result = AnalysisResult(
        summary=[summary],
        events=[],
        metadata={"assay": "ymaze",
                  "units": {UNIT: {"mode": "manual"}}},
    )
    job = {
        "assay": "ymaze", "run_id": "r" * 32, "job_id": "j" * 32,
        "asset": {"path": str(tmp_path / "y.mp4"),
                  "metadata": {"fps": 10.0}},
        "window": {"start_s": 0.0, "end_s": 1.0},
        "orientation": 0, "fps_override": None,
        "output": {"annotated_video": True},
    }
    with pytest.raises(OSError) as captured:
        outputs.export_result(tmp_path / "run", job, result)
    assert "标注视频" in str(captured.value)
    assert not (tmp_path / "run" / "result_manifest.json").exists()


def test_clock_text_matches_frame_label():
    assert annotated.clock_text(3.2) == "00:03.200"
    assert annotated.clock_text(0.0) == "00:00.000"
    assert annotated.clock_text(None) == "--:--.---"


def test_encoding_failure_marks_job_error(client, store, tmp_path, monkeypatch):
    monkeypatch.setattr(media, "VideoSource", Frames)
    monkeypatch.setattr(
        "behaviourlab.engine.media.probe_video",
        lambda path: {"width": 128, "height": 96, "duration_s": 1.0,
                      "fps": 10.0, "frame_count": 10})

    def refuse(*args, **kwargs):
        raise OSError("标注视频编码器不可用（mp4v）")

    monkeypatch.setattr(annotated, "write_manual_y_video", refuse)
    draft = seed(client, store, tmp_path)
    payload = client.post("/api/jobs", json={"draft_id": draft["id"]}).get_json()["data"]
    job = payload["jobs"][0]
    assert job["status"] == "error"
    assert "标注视频" in str(job.get("error")), job
    assert "manifest" not in job
    run_dir = store.paths.runs / payload["run_id"] / job["job_id"]
    assert not (run_dir / "result_manifest.json").exists()


def test_encoding_failure_marks_job_error(client, store, tmp_path, monkeypatch):
    monkeypatch.setattr(media, "VideoSource", Frames)
    monkeypatch.setattr(
        "behaviourlab.engine.media.probe_video",
        lambda path: {"width": 128, "height": 96, "duration_s": 1.0,
                      "fps": 10.0, "frame_count": 10})

    def refuse(*args, **kwargs):
        raise OSError("标注视频编码器不可用（mp4v）")

    monkeypatch.setattr(annotated, "write_manual_y_video", refuse)
    draft = seed(client, store, tmp_path)
    payload = client.post("/api/jobs", json={"draft_id": draft["id"]}).get_json()["data"]
    job = payload["jobs"][0]
    assert job["status"] == "error"
    assert "标注视频" in str(job.get("error")), job
    assert "manifest" not in job
    run_dir = store.paths.runs / payload["run_id"] / job["job_id"]
    assert not (run_dir / "result_manifest.json").exists()


def test_annotated_video_skips_zero_exposure_boundary_frames(tmp_path,
                                                             monkeypatch):
    """真实源在窗口两端会给零曝光上下文帧，视频只收窗口内的真实曝光。"""

    class BoundaryFrames(Frames):
        """像真实解码：窗口外两端各多给一帧零曝光的上下文。"""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._frames = []
            for index in range(10):
                moment = round(0.1 * index, 6)
                exposure = (0.1, 0.1) if index == 0 else (
                    (0.9, 0.9) if index == 9 else (moment, round(moment + 0.1, 6)))
                self._frames.append(Frame(
                    image=np.full((96, 128), 240, dtype=np.uint8), index=index,
                    time_s=moment, dt_s=0.1, elapsed_s=moment, analysis_fps=10.0,
                    interval_start_s=exposure[0], interval_end_s=exposure[1]))

    monkeypatch.setattr(media, "VideoSource", BoundaryFrames)
    job = {"asset": {"path": "memory", "metadata": {"fps": 10.0}},
           "window": {"start_s": 0.0, "end_s": 1.0},
           "orientation": 0, "fps_override": None}
    summary = {"unit_name": "unit", "analysis_window_start_s": 0.1,
               "analysis_window_end_s": 0.9}
    events = [{"time_s": 0.15, "target": "A", "entry_index": 1},
              {"time_s": 0.35, "target": "A", "entry_index": 2},
              {"time_s": 0.55, "target": "B", "entry_index": 3},
              {"time_s": 0.75, "target": "C", "entry_index": 4}]
    target = tmp_path / "boundary.mp4"
    written, seconds = annotated.write_manual_y_video(
        target, job=job, summary=summary, events=events)
    assert written == 8
    assert seconds == pytest.approx(0.8, abs=1e-4)

    import av

    with av.open(str(target)) as container:
        assert len(list(container.decode(video=0))) == 8


class Frames29:
    """29FPS 源：帧 i 在 i/29 秒，曝光到 (i+1)/29 秒。"""

    def __init__(self, path=None, *, start_s=0.0, end_s=None,
                 decode_images=True, **kwargs):
        self._frames = []
        for index in range(28):
            moment = index / 29.0
            self._frames.append(Frame(
                image=(np.full((96, 128), 200, dtype=np.uint8)
                       if decode_images else None),
                index=index, time_s=moment, dt_s=1.0 / 29.0,
                elapsed_s=moment, analysis_fps=29.0,
                interval_start_s=moment, interval_end_s=(index + 1) / 29.0))

    def iter_frames(self):
        return iter(self._frames)


def test_vfr_keeps_cropped_exposures(tmp_path, monkeypatch):
    """窗口首尾的半帧曝光不能被补成完整帧，总时长仍等于 .8s。"""

    monkeypatch.setattr(media, "VideoSource", Frames29)
    job = {"asset": {"path": "memory", "metadata": {"fps": 29.0}},
           "window": {"start_s": 0.0, "end_s": 1.0},
           "orientation": 0, "fps_override": None}
    summary = {"unit_name": "unit", "analysis_window_start_s": 0.1,
               "analysis_window_end_s": 0.9}
    events = [{"time_s": 0.2, "target": "A", "entry_index": 1}]
    target = tmp_path / "vfr.mp4"
    frames, seconds = annotated.write_manual_y_video(
        target, job=job, summary=summary, events=events)
    assert frames == 25, "源帧 2–26 是窗内正曝光"
    assert seconds == pytest.approx(0.8, abs=1e-4)

    import av

    with av.open(str(target)) as container:
        assert container.streams.video[0].codec_context.name == "h264"
        assert float(container.duration) / av.time_base == pytest.approx(
            0.8, abs=1e-4)
        decoded = list(container.decode(video=0))
    assert len(decoded) == 25
    durations = [float(item.duration * item.time_base) for item in decoded]
    assert durations[0] == pytest.approx(3 / 29.0 - 0.1, abs=2e-6)
    assert durations[-1] == pytest.approx(0.9 - 26 / 29.0, abs=2e-6)
    assert all(abs(item - 1 / 29.0) < 2e-6 for item in durations[1:-1])
    assert sum(durations) == pytest.approx(0.8, abs=1e-4)
    starts = [float(item.pts * item.time_base) for item in decoded]
    assert starts[0] == pytest.approx(0.0, abs=1e-9)
    # 时间基 1µs：帧时间按微秒取整，容忍累计 5µs 的量化误差。
    assert starts[-1] == pytest.approx(0.8 - durations[-1], abs=1e-5)


def test_encode_write_failure_cleans_up(tmp_path, monkeypatch):
    """写帧阶段抛错也要变成明确的 OSError，并删除半成品。"""

    monkeypatch.setattr(media, "VideoSource", Frames)
    calls = {"count": 0}
    original = annotated._mux

    def flaky(container, packet, queued):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("模拟写入失败")
        return original(container, packet, queued)

    monkeypatch.setattr(annotated, "_mux", flaky)
    target = tmp_path / "broken.mp4"
    job = {"asset": {"path": "memory", "metadata": {"fps": 10.0}},
           "window": {"start_s": 0.0, "end_s": 1.0},
           "orientation": 0, "fps_override": None}
    summary = {"unit_name": "unit", "analysis_window_start_s": 0.1,
               "analysis_window_end_s": 0.9}
    with pytest.raises(OSError) as captured:
        annotated.write_manual_y_video(target, job=job, summary=summary,
                                       events=ENTRIES)
    assert "标注视频编码失败" in str(captured.value)
    assert not target.exists()


def test_write_failure_marks_job_error(client, store, tmp_path, monkeypatch):
    monkeypatch.setattr(media, "VideoSource", Frames)
    monkeypatch.setattr(
        "behaviourlab.engine.media.probe_video",
        lambda path: {"width": 128, "height": 96, "duration_s": 1.0,
                      "fps": 10.0, "frame_count": 10})
    original = annotated._mux
    calls = {"count": 0}

    def flaky(container, packet, queued):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("模拟写入失败")
        return original(container, packet, queued)

    monkeypatch.setattr(annotated, "_mux", flaky)
    draft = seed(client, store, tmp_path)
    payload = client.post("/api/jobs",
                          json={"draft_id": draft["id"]}).get_json()["data"]
    job = payload["jobs"][0]
    assert job["status"] == "error", job
    assert "标注视频" in str(job.get("error")), job
    assert "manifest" not in job
    run_dir = store.paths.runs / payload["run_id"] / job["job_id"]
    assert not (run_dir / "result_manifest.json").exists()
    assert not list(run_dir.glob("*_annotated.mp4"))
