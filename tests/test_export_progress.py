"""导出进度与取消：叠画视频阶段有真实进度，取消不留半成品。

只覆盖接线与失败隔离，不改变编码参数或输出格式。
"""

from __future__ import annotations

import numpy as np
import pytest

from behaviourlab import annotated, outputs
from behaviourlab.schema import Frame

UNIT = "a" * 32


class _Source:
    frames_spec: list[dict] = []

    def __init__(self, path=None, *, start_s=0.0, end_s=None,
                 decode_images=True, **kwargs):
        self.frames = []
        for item in self.frames_spec:
            low = max(item["start"], start_s)
            high = item["end"] if end_s is None else min(item["end"], end_s)
            if high - low <= 1e-12:
                continue
            image = (np.full((32, 40, 3), 90 + item["index"], np.uint8)
                     if decode_images else None)
            self.frames.append(Frame(
                image=image, index=item["index"], time_s=item["time_s"],
                dt_s=item["end"] - item["start"], elapsed_s=item["time_s"],
                analysis_fps=5.0, interval_start_s=low, interval_end_s=high))
        self.coverage = {"start_s": None, "end_s": None, "duration_s": 0.0,
                         "frame_count": 0}

    def iter_frames(self):
        return iter(self.frames)


def _spec(count: int = 6) -> list[dict]:
    return [{"index": index, "time_s": index * 0.2,
             "start": index * 0.2, "end": index * 0.2 + 0.2}
            for index in range(count)]


def _job(count: int = 6) -> dict:
    return {"asset": {"path": "memory", "metadata": {"fps": 5.0}},
            "window": {"start_s": 0.0, "end_s": count * 0.2},
            "orientation": 0, "fps_override": None}


def _units(count: int = 6) -> list[dict]:
    rows = []
    for index in range(count):
        row = {"unit_id": UNIT, "frame": index, "time_s": index * 0.2,
               "x_px": 6 + index, "y_px": 8 + index, "x_cm": 6 + index,
               "y_cm": 8 + index, "coordinate_unit": "px", "detected": True,
               "position_valid": True}
        if index + 1 < count:
            row.update({"motion_x0": 6 + index, "motion_y0": 8 + index,
                        "motion_x1": 7 + index, "motion_y1": 9 + index})
        rows.append(row)
    return [{"unit_id": UNIT, "unit_name": "u", "window_start_s": 0.0,
             "window_end_s": count * 0.2, "geometry": {}, "observations": rows}]


@pytest.fixture(autouse=True)
def _source(monkeypatch):
    _Source.frames_spec = _spec()
    monkeypatch.setattr(annotated.media, "VideoSource", _Source)
    yield
    _Source.frames_spec = []


def test_video_progress_is_monotone_and_reports_counts(tmp_path):
    seen: list[tuple[float, str]] = []
    written, _seconds = annotated.write_run_video(
        tmp_path / "annotated.mp4", job=_job(), units=_units(),
        progress=lambda f, msg: seen.append((float(f), msg)))

    assert written == 6
    fractions = [item[0] for item in seen]
    assert fractions == sorted(fractions), "进度必须单调"
    assert all(message for _, message in seen), "每步都要说明阶段"
    assert any("定位源帧" in msg for _, msg in seen)
    assert any("写叠画视频" in msg and "/6" in msg for _, msg in seen)
    assert any("校验叠画视频" in msg for _, msg in seen)
    # 未完成前不报 100%。
    assert max(fractions) <= 1.0
    assert fractions[-1] == pytest.approx(1.0, abs=1e-9)


def test_cancel_during_encode_removes_partial(tmp_path):
    target = tmp_path / "cancel.mp4"
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 2

    with pytest.raises(OSError):
        annotated.write_run_video(target, job=_job(), units=_units(),
                                  cancel=cancel)
    assert not target.exists(), "取消后不留未完成视频"


def test_cancel_during_verify_removes_partial(tmp_path):
    target = tmp_path / "verify-cancel.mp4"
    flags = {"armed": False}

    def cancel():
        return flags["armed"]

    real_verify = annotated.verify_video

    def verify(path, expected, *, cancel=None, progress=None):
        flags["armed"] = True
        return real_verify(path, expected, cancel=cancel, progress=progress)

    import behaviourlab.annotated as module

    original = module.verify_video
    module.verify_video = verify
    try:
        with pytest.raises(OSError) as error:
            annotated.write_run_video(target, job=_job(), units=_units(),
                                      cancel=cancel)
    finally:
        module.verify_video = original
    assert "取消" in str(error.value)
    assert not target.exists()


def test_verify_rejects_wrong_frame_count(tmp_path):
    target = tmp_path / "ok.mp4"
    written, _ = annotated.write_run_video(target, job=_job(), units=_units())
    expected = [(annotated._ticks(i * 0.2), annotated._ticks(0.2))
                for i in range(written)]
    assert annotated.verify_video(target, expected) == []
    issues = annotated.verify_video(target, expected[:-1])
    assert issues and "帧数不一致" in issues[0]
    issues = annotated.verify_video(target, expected + [(0, 1)])
    assert issues and "帧数不一致" in issues[0]


def test_verify_rejects_wrong_pts_and_missing_file(tmp_path):
    target = tmp_path / "ok.mp4"
    written, _ = annotated.write_run_video(target, job=_job(), units=_units())
    expected = [(annotated._ticks(i * 0.2), annotated._ticks(0.2))
                for i in range(written)]
    broken = list(expected)
    pts, duration = broken[2]
    broken[2] = (pts + annotated._ticks(0.05), duration)
    issues = annotated.verify_video(target, broken)
    assert any("PTS 偏差" in item for item in issues)
    assert annotated.verify_video(tmp_path / "missing.mp4", expected)


def test_write_outputs_video_progress_slot_is_monotone(tmp_path, monkeypatch):
    _Source.frames_spec = _spec(4)
    job = _job(4)
    job["output"] = {"annotated_video": True}
    job.update({"run_id": "r" * 32, "job_id": "j" * 32, "assay": "oft",
                "units": [{"id": UNIT, "name": "u", "geometry": {}}],
                "style": {}})

    class _Result:
        summary = [{"unit_id": UNIT, "unit_name": "u",
                    "analysis_window_start_s": 0.0,
                    "analysis_window_end_s": 0.8}]
        observations: list = []
        events: list = []
        bins: list = []
        tables: dict = {}
        warnings: list = []
        metadata = {"assay": "oft", "units": {UNIT: {}}}

        def as_dict(self):
            return {"summary": self.summary}

    seen: list[float] = []
    manifest = outputs.write_outputs(
        tmp_path / "out", job, _Result(),
        progress=lambda f, msg="": seen.append(float(f)))
    assert manifest["annotated_video"] == "annotated.mp4"
    assert seen == sorted(seen), f"进度必须单调：{seen}"
    assert seen[-1] == pytest.approx(0.99)
    # 视频阶段有自己的区间，不停在同一档上。
    video = [value for value in seen if 0.84 <= value <= 0.985]
    assert len(set(video)) >= 2
    assert max(seen) < 1.0 or seen[-1] == pytest.approx(1.0)


def test_write_outputs_cancel_skips_video(tmp_path):
    _Source.frames_spec = _spec(4)
    job = _job(4)
    job["output"] = {"annotated_video": True}
    job.update({"run_id": "r" * 32, "job_id": "j" * 32, "assay": "oft",
                "units": [{"id": UNIT, "name": "u", "geometry": {}}],
                "style": {}})

    class _Result:
        summary = [{"unit_id": UNIT, "unit_name": "u",
                    "analysis_window_start_s": 0.0,
                    "analysis_window_end_s": 0.8}]
        observations: list = []
        events: list = []
        bins: list = []
        tables: dict = {}
        warnings: list = []
        metadata = {"assay": "oft", "units": {UNIT: {}}}

        def as_dict(self):
            return {"summary": self.summary}

    with pytest.raises(RuntimeError):
        outputs.write_outputs(tmp_path / "out", job, _Result(),
                              cancel=lambda: True)
    assert not (tmp_path / "out" / "annotated.mp4").exists()
