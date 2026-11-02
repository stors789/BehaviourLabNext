"""默认源帧率 20 与解释帧率下的行为预览范围。

受控帧说明：这里的“视频”是内存解码器给出的合成帧 + 受控 PTS/时长，
不是真实录像；真实素材上的行为未在此验证。
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from behaviourlab import media, preview
from behaviourlab.experiments import oft
from behaviourlab.media import FrameInfo, RawFrame, VideoSource
from behaviourlab.schema import AnalysisContext, ValidationError
from test_media_source import FakeDecoder, FakeFactory, make_spec, write_file

TOL = 1e-9


# -- 受控解码器：合成 200×200 帧 + 受控 PTS ----------------------------
class GrayDecoder(FakeDecoder):
    def frame_at(self, index):
        item = self.spec[index]
        image = np.full((200, 200, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (85, 85), (115, 115), (0, 0, 0), -1)
        return RawFrame(
            index=-1,
            image=image,
            pts=item.get("pts"),
            time_base=item.get("time_base"),
            keyframe=item.get("keyframe", False),
            duration_s=item.get("duration_s"),
        )


class GrayFactory(FakeFactory):
    def __call__(self, path, decode_images=True):
        self.opens += 1
        decoder = GrayDecoder(self.spec, version=self.version)
        self.decoders.append(decoder)
        return decoder


def controlled(monkeypatch, factory, record=None, probe=None, sources=None):
    """保留真实 VideoSource 的时钟与窗口规则，只替换解码器与探测。"""

    real = media.VideoSource

    def build(path, **kwargs):
        if record is not None:
            record.append(dict(kwargs))
        source = real(path, decoder_factory=factory, **kwargs)
        if sources is not None:
            sources.append(source)
        return source

    monkeypatch.setattr(media, "VideoSource", build)
    if probe is not None:
        monkeypatch.setattr(media, "probe_video", lambda path: dict(probe))
    return build


# -- 一、名义源帧率优先级 -----------------------------------------------
def test_nominal_source_fps_priority():
    assert media.nominal_source_fps(25.0) == (25.0, "media")
    assert media.nominal_source_fps(29.97) == (29.97, "media")
    assert media.nominal_source_fps(None) == (20.0, "default")
    assert media.nominal_source_fps(0) == (20.0, "default")
    assert media.nominal_source_fps(-3.0) == (20.0, "default")
    assert media.nominal_source_fps(float("nan")) == (20.0, "default")
    assert media.nominal_source_fps(float("inf")) == (20.0, "default")
    assert media.nominal_source_fps(True) == (20.0, "default")
    # 合法显式覆盖优先于原生值，也优先于默认。
    assert media.nominal_source_fps(25.0, fps_override=12.5) == (12.5, "override")
    assert media.nominal_source_fps(None, fps_override=12.5) == (12.5, "override")
    assert media.source_fps_note("default") == media.DEFAULT_SOURCE_FPS_NOTE
    assert media.source_fps_note("media") is None
    assert media.source_fps_note("override") is None


def test_missing_source_fps_uses_20_on_real_pts(tmp_path):
    # 0.05 秒间隔：源帧率缺失时按默认 20 采样，但时刻仍来自真实 PTS。
    pts = [0, 50, 100, 150, 200]
    path = write_file(tmp_path / "no-fps.mp4")
    factory = FakeFactory(make_spec(pts, last_duration=0.05))
    source = VideoSource(path, source_fps=None, metadata={"fps": None},
                         decoder_factory=factory)
    frames = list(source.iter_frames())

    assert source.source_fps == 20.0
    assert source.source_fps_source == "default"
    assert source.source_fps_note == "源FPS不可用，使用默认 20"
    assert source.analysis_fps == 20.0
    assert source.sample_fps is None
    assert [round(frame.time_s, 6) for frame in frames] == \
        [0.0, 0.05, 0.1, 0.15, 0.2]
    assert abs(source.coverage["end_s"] - 0.25) <= TOL


def test_default_fps_never_rewrites_reliable_pts(tmp_path):
    # 非 1/20 间隔：若被偷偷换成 index/20，时刻会变成 0/0.05/0.10/0.15。
    pts = [0, 40, 100, 160]
    spec = make_spec(pts, last_duration=0.06)
    path = write_file(tmp_path / "odd-pts.mp4")

    native = VideoSource(path, source_fps=None, metadata={},
                         decoder_factory=FakeFactory(spec))
    native_frames = list(native.iter_frames())
    assert [round(frame.time_s, 6) for frame in native_frames] == \
        [0.0, 0.04, 0.10, 0.16]
    assert native.source_fps_source == "default"
    assert native.fps_override is None

    # 对照：真正的显式 20 覆盖才按 index/20 解释。
    override = VideoSource(path, fps_override=20, metadata={},
                           decoder_factory=FakeFactory(spec))
    assert [round(frame.time_s, 6) for frame in override.iter_frames()] == \
        [0.0, 0.05, 0.10, 0.15]
    assert override.source_fps_source == "override"


def test_real_source_fps_and_override_are_unchanged(tmp_path):
    pts = [0, 40, 80, 120]
    path = write_file(tmp_path / "fps25.mp4")
    source = VideoSource(path, source_fps=25, decoder_factory=FakeFactory(
        make_spec(pts, last_duration=0.04)))
    frames = list(source.iter_frames())
    assert source.source_fps == 25.0
    assert source.source_fps_source == "media"
    assert source.source_fps_note is None
    assert [round(frame.time_s, 6) for frame in frames] == [0.0, 0.04, 0.08, 0.12]

    # 显式 12.5 覆盖：12.5 保持 12.5，不因默认值改为 20。
    over = VideoSource(path, fps_override=12.5, source_fps=25,
                       decoder_factory=FakeFactory(make_spec(pts)))
    assert over.source_fps == 12.5
    assert over.analysis_fps == 12.5
    assert [round(frame.time_s, 6) for frame in over.iter_frames()] == \
        [0.0, 0.08, 0.16, 0.24]


@pytest.mark.parametrize("bad", [0, -1, True, "12.5", float("nan"), float("inf")])
def test_invalid_explicit_override_still_fails(tmp_path, bad):
    path = write_file(tmp_path / "bad-override.mp4")
    with pytest.raises(ValidationError):
        VideoSource(path, fps_override=bad, source_fps=25)


def test_missing_pts_still_errors_under_default_fps(tmp_path):
    # 默认 20 只是名义源帧率：缺 PTS 仍按既有明确错误处理，不改成 index/20。
    spec = make_spec([0, None, 100])
    path = write_file(tmp_path / "no-pts.mp4")
    source = VideoSource(path, source_fps=None, metadata={"fps": None},
                         decoder_factory=FakeFactory(spec))
    assert source.source_fps == 20.0
    assert source.fps_override is None
    with pytest.raises(media.ClockError):
        list(source.iter_frames())

    # 合法的显式解释帧率才可以接管不可解释的源时钟。
    rescued = VideoSource(path, fps_override=25, metadata={"fps": None},
                          decoder_factory=FakeFactory(spec))
    assert [round(frame.time_s, 6) for frame in rescued.iter_frames()] == \
        [0.0, 0.04, 0.08]


def test_non_monotonic_pts_still_errors_under_default_fps(tmp_path):
    spec = make_spec([0, 100, 50, 200])
    path = write_file(tmp_path / "backwards.mp4")
    source = VideoSource(path, source_fps=None, metadata={},
                         decoder_factory=FakeFactory(spec))
    assert source.source_fps_source == "default"
    with pytest.raises(media.ClockError):
        list(source.iter_frames())


def test_default_fps_reaches_analysis_frames(tmp_path, monkeypatch):
    # 分析入口（oft.analyze）自己建源；元数据没有 fps 时用默认 20。
    spec = make_spec([40 * index for index in range(25)], last_duration=0.04)
    path = write_file(tmp_path / "analysis-no-fps.mp4")
    record = []
    sources = []
    controlled(monkeypatch, GrayFactory(spec), record,
               probe={"width": 200, "height": 200, "fps": None,
                      "frame_count": None, "duration_s": None},
               sources=sources)

    unit = {
        "id": "u" * 32, "name": "小鼠 1", "animal_id": None, "window": None,
        "geometry": {"arena": {"type": "quad",
                               "points": [[0, 0], [199, 0], [199, 199], [0, 199]]}},
        "options": dict(oft.DEFAULTS),
        "detection": dict(oft_module_defaults()),
    }
    job = {
        "assay": "oft", "run_id": "run-default-fps", "job_id": "job",
        "project_id": None,
        "window": {"start_s": 0.0, "end_s": None},
        "orientation": 0, "fps_override": None, "sample_fps": None,
        "asset": {"id": "a" * 32, "path": str(path),
                  "metadata": {"width": 200, "height": 200}},
        "units": [unit],
        "style": {},
    }
    result = oft.analyze(AnalysisContext(job=job, source=None))

    assert record and record[0]["source_fps"] is None
    assert record[0]["fps_override"] is None
    # 真实构造出来的源按默认 20 解析名义帧率，PTS 时钟未被覆盖。
    assert len(sources) == 1
    assert sources[0].source_fps == 20.0
    assert sources[0].source_fps_source == "default"
    assert sources[0].fps_override is None
    assert sources[0].analysis_fps == 20.0
    assert result.summary[0]["total_time_s"] == pytest.approx(1.0, abs=1e-6)
    assert result.observations
    assert all(row["time_s"] <= 1.0 for row in result.observations)


def oft_module_defaults():
    from behaviourlab import detection

    return detection.common_detection_defaults()


def test_run_job_records_default_source_fps(tmp_path, monkeypatch):
    """生产分析入口 run_job：默认 20 要记来源，不能冒充实测帧率。"""

    import os

    from behaviourlab import detection, engine
    from behaviourlab.config import ALGORITHM_VERSION, SCHEMA_VERSION

    spec = make_spec([40 * index for index in range(25)], last_duration=0.04)
    path = write_file(tmp_path / "run-no-fps.mp4")
    stat = os.stat(path)
    identity = {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime": float(stat.st_mtime),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    sources = []
    controlled(monkeypatch, GrayFactory(spec), sources=sources,
               probe={"width": 200, "height": 200, "fps": None,
                      "frame_count": None, "duration_s": None})
    settings = dict(detection.common_detection_defaults())
    unit = {
        "id": "u" * 32, "name": "小鼠 1", "animal_id": None, "window": None,
        "geometry": {"arena": {"type": "quad",
                               "points": [[0, 0], [199, 0], [199, 199], [0, 199]]}},
        "options": dict(oft.DEFAULTS), "detection": settings, "manual": None,
    }
    resolved = {
        "schema_version": SCHEMA_VERSION,
        "run_id": "r" * 32, "job_id": "j" * 32, "project_id": None,
        "assay": "oft",
        "asset": {"id": "a" * 32, "name": "run-no-fps.mp4",
                  "path": str(path.resolve()),
                  "metadata": {"width": 200, "height": 200},
                  "identity": identity},
        "window": {"start_s": 0.0, "end_s": None},
        "orientation": 0, "fps_override": None, "sample_fps": None,
        "options": dict(oft.DEFAULTS), "detection": settings,
        "units": [unit],
        "output": {"annotated_video": False},
        "style": {"color": "#B86245", "colormap": "viridis",
                  "line_width": 2, "dpi": 100},
        "algorithm_version": ALGORITHM_VERSION,
    }
    result = engine.run_job(resolved)

    video = result.metadata["video"]
    assert video["fps"] == 20.0
    assert video["fps_source"] == "default"
    assert video["fps_note"] == media.DEFAULT_SOURCE_FPS_NOTE
    assert media.DEFAULT_SOURCE_FPS_NOTE in result.warnings
    # 帧来自真实 PTS（0.04 秒），不是 index/20。
    times = [row["time_s"] for row in result.observations][:4]
    assert [round(value, 6) for value in times] == [0.0, 0.04, 0.08, 0.12]


# -- 二、解释帧率下行为预览的有效时长 -----------------------------------
def seed_draft(store, tmp_path, *, fps_override=12.5, window=None,
               metadata=None):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"controlled-not-real-video")
    asset = store.create_asset(
        name="示例.mp4", path=str(source), project_id=None,
        metadata=metadata or {"width": 200, "height": 200, "fps": 25.0,
                              "frame_count": 250, "duration_s": 10.0},
    )
    draft = store.create_draft("oft", videos=[{
        "asset_id": asset["id"],
        "window": window or {"start_s": 0, "end_s": None},
        "orientation": 90,
        "fps_override": fps_override,
        "sample_fps": 2,
        "units": [{
            "id": "a" * 32, "name": "小鼠 1", "animal_id": None,
            "window": {"mode": "relative", "offset_s": 0.0, "duration_s": 20.0},
            "geometry": {"arena": {"type": "quad",
                                   "points": [[0, 0], [199, 0],
                                              [199, 199], [0, 199]]}},
            "options": dict(oft.DEFAULTS),
            "detection": {},
            "manual": None,
        }],
    }])
    return asset, draft


def behavior_preview(client, asset, draft, monkeypatch, *, frame_index=187,
                     frame_time=14.96, spec=None, clock="fps_override"):
    monkeypatch.setattr(
        preview.media, "read_frame_info",
        lambda *args, **kwargs: FrameInfo(
            image=np.zeros((200, 200, 3), dtype=np.uint8), index=frame_index,
            time_s=frame_time, pts=None, time_base=None, clock_source=clock))
    record = []
    controlled(monkeypatch, GrayFactory(
        spec or make_spec([80 * index for index in range(250)])), record,
        probe={"width": 200, "height": 200, "fps": None,
               "frame_count": None, "duration_s": None})
    response = client.post(
        f"/api/drafts/{draft['id']}/preview",
        json={"asset_id": asset["id"], "unit_id": "a" * 32,
              "frame_index": frame_index, "kind": "behavior"},
    )
    return response, record


def test_override_fps_preview_uses_full_interpreted_duration(
        client, store, tmp_path, monkeypatch):
    asset, draft = seed_draft(store, tmp_path)
    response, record = behavior_preview(client, asset, draft, monkeypatch)
    assert response.status_code == 201, response.get_json()
    data = response.get_json()["data"]

    # 250 帧 ÷ 12.5 = 20 秒；原始 10 秒容器时长不是覆盖后的终点。
    assert data["preview_scope"]["end_s"] == pytest.approx(20.0)
    assert data["preview_scope"]["mode"] == "configured"
    assert data["preview_scope"]["duration_s"] == pytest.approx(20.0)
    behavior = data["summary"]["behavior"]
    assert behavior["end_s"] == pytest.approx(20.0)
    assert behavior["window_end_s"] == pytest.approx(20.0)
    assert behavior["total_time_s"] == pytest.approx(20.0, abs=0.1)
    # 真实生产入口确实用 20 秒窗捕帧。
    assert record[0]["end_s"] == pytest.approx(20.0)
    assert record[0]["start_s"] == pytest.approx(0.0)
    assert record[0]["fps_override"] == 12.5


def test_explicit_video_end_wins(client, store, tmp_path, monkeypatch):
    asset, draft = seed_draft(store, tmp_path,
                              window={"start_s": 0, "end_s": 20})
    response, record = behavior_preview(client, asset, draft, monkeypatch)
    assert response.status_code == 201, response.get_json()
    data = response.get_json()["data"]
    assert data["preview_scope"]["end_s"] == pytest.approx(20.0)
    assert record[0]["end_s"] == pytest.approx(20.0)


def test_uncovered_fps_keeps_container_duration(client, store, tmp_path,
                                                monkeypatch):
    asset, draft = seed_draft(store, tmp_path, fps_override=None)
    response, record = behavior_preview(
        client, asset, draft, monkeypatch, frame_index=120, frame_time=4.8,
        spec=make_spec([40 * index for index in range(250)]), clock="pts")
    assert response.status_code == 201, response.get_json()
    data = response.get_json()["data"]
    # 25 FPS、无覆盖：沿用原有 10 秒范围，不因本次改动放大。
    assert data["preview_scope"]["end_s"] == pytest.approx(10.0)
    assert record[0]["end_s"] == pytest.approx(10.0)


def test_preview_notes_default_source_fps(client, store, tmp_path, monkeypatch):
    asset, draft = seed_draft(
        store, tmp_path, fps_override=None,
        metadata={"width": 200, "height": 200, "frame_count": 250})
    response, _ = behavior_preview(
        client, asset, draft, monkeypatch, frame_index=120, frame_time=4.8,
        spec=make_spec([40 * index for index in range(250)]), clock="pts")
    assert response.status_code == 201, response.get_json()
    data = response.get_json()["data"]
    # 源帧率不可用且无覆盖：如实说明用的默认 20，不冒充实测帧率。
    assert any(media.DEFAULT_SOURCE_FPS_NOTE in text for text in data["warnings"])
