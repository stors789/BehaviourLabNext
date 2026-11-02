"""时序背景在实际调用点的复用：预览与冻结任务，多单位与多轮。

受控解码器说明：帧来自内存合成（8×6 BGR），不是真实录像；Windows 与
真实解码表现不在本文件覆盖范围。
"""

from __future__ import annotations

import os
from fractions import Fraction

import numpy as np
import pytest

from behaviourlab import detection, media, preview
from behaviourlab.experiments import oft
from behaviourlab.media import FrameInfo, RawFrame
from test_media_source import FakeDecoder, make_spec, write_file

ARENA = {"type": "quad", "points": [[1, 1], [6, 1], [6, 5], [1, 5]]}
SPEC = make_spec([80 * index for index in range(300)], last_duration=0.08)
DETECTION = {"bg_method": "temporal", "temporal_bg_samples": 3,
             "gray_threshold": 127.0, "min_area": 2.0, "max_area": 10000.0}


class Decoder(FakeDecoder):
    def __init__(self, spec, *, decode_images=True, **kwargs):
        super().__init__(spec, **kwargs)
        self.decode_images = decode_images

    def frame_at(self, index):
        item = self.spec[index]
        image = None
        if self.decode_images:
            image = np.full((6, 8, 3), 245, dtype=np.uint8)
            image[2:4, 2:4] = (index % 200, 0, 0)
        return RawFrame(index=-1, image=image, pts=item.get("pts"),
                        time_base=item.get("time_base", Fraction(1, 1000)),
                        keyframe=item.get("keyframe", False),
                        duration_s=item.get("duration_s"))


def factory(spec=SPEC):
    def build(path, decode_images=True):
        return Decoder(spec, decode_images=decode_images, version=1)
    return build


def inject(monkeypatch):
    real_pool = media.ReaderPool
    monkeypatch.setattr(
        media, "ReaderPool",
        lambda **kwargs: real_pool(
            **{**kwargs, "decoder_factory": factory()}))
    detection.clear_background_cache()


def track_builds(monkeypatch):
    calls = []
    real = detection._build_temporal_background

    def wrapper(asset, video, orientation, start, end_value, sample_end,
                samples, cancel=None):
        calls.append((float(start), end_value, samples))
        return real(asset, video, orientation, start, end_value, sample_end,
                    samples, cancel)

    monkeypatch.setattr(detection, "_build_temporal_background", wrapper)
    return calls


def seed(store, tmp_path, assay="oft"):
    source = write_file(tmp_path / "clip.mp4")
    asset = store.create_asset(
        name="clip.mp4", path=str(source), project_id=None,
        metadata={"width": 8, "height": 6, "fps": 12.5, "frame_count": 300,
                  "duration_s": 24.0})
    draft = store.create_draft(assay, detection=dict(DETECTION), videos=[{
        "asset_id": asset["id"],
        "window": {"start_s": 0.0, "end_s": 6.0},
        "orientation": 0, "fps_override": 12.5, "sample_fps": None,
        "units": [{
            "id": "a" * 32, "name": "小鼠 1", "animal_id": None,
            "window": {"mode": "relative", "offset_s": 0.0, "duration_s": 3.0},
            "geometry": {"arena": ARENA}, "options": {}, "detection": {},
            "manual": None,
        }],
    }])
    return asset, draft


def test_behavior_preview_prepares_background_once(client, store, tmp_path,
                                                   monkeypatch):
    inject(monkeypatch)
    calls = track_builds(monkeypatch)
    asset, draft = seed(store, tmp_path)
    monkeypatch.setattr(
        preview.media, "read_frame_info",
        lambda *a, **k: FrameInfo(
            image=np.full((6, 8, 3), 245, dtype=np.uint8), index=10,
            time_s=0.8, pts=800, time_base=Fraction(1, 1000),
            clock_source="fps_override"))
    try:
        body = {"asset_id": asset["id"], "unit_id": "a" * 32,
                "frame_index": 10, "kind": "behavior"}
        first = client.post(f"/api/drafts/{draft['id']}/preview", json=body)
        assert first.status_code == 201, first.get_json()
        assert len(calls) == 1, "同请求的两处背景范围事实相同，只构建一次"

        second = client.post(f"/api/drafts/{draft['id']}/preview", json=body)
        assert second.status_code == 201, second.get_json()
        assert len(calls) == 1, "重复请求命中同一背景"

        # 改窗口后再预览：不同背景事实各自构建。
        draft = store.get_draft(draft["id"])
        videos = draft["videos"]
        videos[0]["window"] = {"start_s": 1.0, "end_s": 6.0}
        assert client.put(f"/api/drafts/{draft['id']}",
                          json={"videos": videos}).status_code == 200
        third = client.post(f"/api/drafts/{draft['id']}/preview", json=body)
        assert third.status_code == 201, third.get_json()
        assert len(calls) == 2, "窗口不同时保留各自背景"
    finally:
        detection.clear_background_cache()


def test_frozen_job_reuses_background_across_runs(tmp_path, monkeypatch):
    from behaviourlab import engine
    from behaviourlab.config import ALGORITHM_VERSION, SCHEMA_VERSION

    inject(monkeypatch)
    calls = track_builds(monkeypatch)
    path = write_file(tmp_path / "job.mp4")
    stat = os.stat(path)
    identity = {"path": str(path.resolve()), "size": int(stat.st_size),
                "mtime": float(stat.st_mtime),
                "mtime_ns": int(stat.st_mtime_ns)}
    settings = dict(detection.common_detection_defaults())
    settings.update(DETECTION)
    units = [{
        "id": unit_id * 32, "name": f"小鼠 {index + 1}", "animal_id": None,
        "window": {"mode": "relative", "offset_s": 0.0, "duration_s": 3.0},
        "geometry": {"arena": ARENA}, "options": dict(oft.DEFAULTS),
        "detection": dict(settings), "manual": None,
    } for index, unit_id in enumerate(("a", "b"))]
    resolved = {
        "schema_version": SCHEMA_VERSION,
        "run_id": "r" * 32, "job_id": "j" * 32, "project_id": None,
        "assay": "oft",
        "asset": {"id": "x" * 32, "name": "job.mp4",
                  "path": str(path.resolve()),
                  "metadata": {"width": 8, "height": 6, "fps": 12.5,
                               "frame_count": 300, "duration_s": 24.0},
                  "identity": identity},
        "window": {"start_s": 0.0, "end_s": 6.0},
        "orientation": 0, "fps_override": 12.5, "sample_fps": None,
        "options": dict(oft.DEFAULTS), "detection": dict(settings),
        "units": units,
        "output": {"annotated_video": False},
        "style": {"color": "#B86245", "colormap": "viridis",
                  "line_width": 2, "dpi": 100},
        "algorithm_version": ALGORITHM_VERSION,
    }
    try:
        engine.run_job(resolved)
        assert len(calls) == 1, "两个单位同语义背景只构建一次"
        engine.run_job(resolved)
        assert len(calls) == 1, "同冻结事实重跑命中缓存"
    finally:
        detection.clear_background_cache()


def test_fst_multi_unit_shares_one_background(tmp_path, monkeypatch):
    """FST 按单位初始化检测器：同语义背景只构建一次。"""

    from behaviourlab.experiments import fst
    from behaviourlab.schema import AnalysisContext

    inject(monkeypatch)
    calls = track_builds(monkeypatch)
    path = write_file(tmp_path / "fst.mp4")
    settings = dict(detection.common_detection_defaults())
    settings.update(DETECTION)
    geometry = {
        "arena": {"type": "polygon",
                  "points": [[1, 1], [6, 1], [6, 5], [1, 5]]},
        "diameter_line": {"type": "line", "points": [[1, 3], [6, 3]]},
    }
    units = [{
        "id": unit_id * 32, "name": f"小鼠 {index + 1}", "animal_id": None,
        "window": None, "geometry": dict(geometry),
        "options": dict(fst.DEFAULTS), "detection": dict(settings),
        "manual": None,
    } for index, unit_id in enumerate(("a", "b"))]
    job = {
        "assay": "fst", "run_id": "r" * 32, "job_id": "j" * 32,
        "project_id": None,
        "window": {"start_s": 0.0, "end_s": 6.0},
        "orientation": 0, "fps_override": 12.5, "sample_fps": None,
        "asset": {"id": "x" * 32, "path": str(path),
                  "metadata": {"width": 8, "height": 6, "fps": 12.5,
                               "frame_count": 300, "duration_s": 24.0}},
        "units": units,
        "style": {},
    }
    try:
        fst.analyze(AnalysisContext(job=job, source=None))
        assert len(calls) == 1, "两只鼠同语义背景只构建一次"
    finally:
        detection.clear_background_cache()


def test_fst_multi_unit_shares_one_background(tmp_path, monkeypatch):
    """FST 按单位初始化检测器：同语义背景只构建一次。"""

    from behaviourlab.experiments import fst
    from behaviourlab.schema import AnalysisContext

    inject(monkeypatch)
    calls = track_builds(monkeypatch)
    path = write_file(tmp_path / "fst.mp4")
    settings = dict(detection.common_detection_defaults())
    settings.update(DETECTION)
    geometry = {
        "arena": {"type": "polygon",
                  "points": [[1, 1], [6, 1], [6, 5], [1, 5]]},
        "diameter_line": {"type": "line", "points": [[1, 3], [6, 3]]},
    }
    units = [{
        "id": unit_id * 32, "name": f"小鼠 {index + 1}", "animal_id": None,
        "window": None, "geometry": dict(geometry),
        "options": dict(fst.DEFAULTS), "detection": dict(settings),
        "manual": None,
    } for index, unit_id in enumerate(("a", "b"))]
    job = {
        "assay": "fst", "run_id": "r" * 32, "job_id": "j" * 32,
        "project_id": None,
        "window": {"start_s": 0.0, "end_s": 6.0},
        "orientation": 0, "fps_override": 12.5, "sample_fps": None,
        "asset": {"id": "x" * 32, "path": str(path),
                  "metadata": {"width": 8, "height": 6, "fps": 12.5,
                               "frame_count": 300, "duration_s": 24.0}},
        "units": units,
        "style": {},
    }
    try:
        fst.analyze(AnalysisContext(job=job, source=None))
        assert len(calls) == 1, "两只鼠同语义背景只构建一次"
    finally:
        detection.clear_background_cache()
