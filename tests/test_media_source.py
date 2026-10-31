"""内存解码器替身：源时钟、采样窗口、有界读取与缓存失效。"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from behaviourlab.media import (
    ClockError,
    RawFrame,
    ReaderPool,
    VideoSource,
    VideoSourceError,
    read_frame,
    read_frame_at_time,
    read_frame_info,
)
from behaviourlab.schema import ValidationError


class FakeDecoder:
    def __init__(self, spec, *, version=1, corrupt_seek=False, fail_seek=False):
        self.spec = spec
        self.version = version
        self.corrupt_seek = corrupt_seek
        self.fail_seek = fail_seek
        self.start = 0
        self.closed = False
        self.seek_calls = 0

    def frame_at(self, index):
        item = self.spec[index]
        image = np.full((4, 6, 3), index + self.version * 100, dtype=np.uint16)
        return RawFrame(
            index=-1,
            image=image,
            pts=item.get("pts"),
            time_base=item.get("time_base", Fraction(1, 1000)),
            keyframe=item.get("keyframe", False),
            duration_s=item.get("duration_s"),
        )

    def frames(self):
        for index in range(self.start, len(self.spec)):
            yield self.frame_at(index)

    def seek(self, pts, keyframe=True):
        self.seek_calls += 1
        if self.fail_seek:
            self.start = 0
            return
        target = next(
            (index for index, item in enumerate(self.spec)
             if item.get("pts") == pts and item.get("keyframe")),
            0,
        )
        self.start = target + 1 if self.corrupt_seek else target

    def close(self):
        self.closed = True


class FakeFactory:
    def __init__(self, spec, *, corrupt_seek=False):
        self.spec = spec
        self.corrupt_seek = corrupt_seek
        self.version = 0
        self.opens = 0
        self.decoders = []

    def __call__(self, path, decode_images=True):
        self.opens += 1
        decoder = FakeDecoder(
            self.spec,
            version=self.version,
            corrupt_seek=self.corrupt_seek,
        )
        self.decoders.append(decoder)
        return decoder


def write_file(path: Path, content: bytes = b"fake-video") -> Path:
    path.write_bytes(content)
    return path


def make_spec(pts_values, *, tb=Fraction(1, 1000), keyframe_step=12,
              last_duration=None):
    spec = []
    for index, pts in enumerate(pts_values):
        duration = None
        if last_duration is not None and index == len(pts_values) - 1:
            duration = last_duration
        spec.append(
            {
                "pts": pts,
                "time_base": tb,
                "keyframe": index % keyframe_step == 0,
                "duration_s": duration,
            }
        )
    return spec


def test_native_clock_and_window_sampling(tmp_path):
    pts = [5000, 5050, 5270, 5430]
    path = write_file(tmp_path / "clock.mp4")
    factory = FakeFactory(make_spec(pts, last_duration=0.04))

    source = VideoSource(path, source_fps=20, decoder_factory=factory)
    frames = list(source.iter_frames())

    assert [round(frame.time_s, 6) for frame in frames] == [0.0, 0.05, 0.27, 0.43]
    assert [round(frame.dt_s, 6) for frame in frames] == [0.05, 0.22, 0.16, 0.04]
    assert abs(source.coverage["duration_s"] - 0.47) <= 1e-12
    assert source.coverage["frame_count"] == 4
    assert source.coverage["start_s"] == 0.0
    assert abs(source.coverage["end_s"] - 0.47) <= 1e-12

    window_factory = FakeFactory(make_spec(pts, last_duration=0.04))
    window = VideoSource(
        path, start_s=0.15, end_s=0.25, source_fps=20,
        decoder_factory=window_factory,
    )
    window_frames = list(window.iter_frames())
    context = next(frame for frame in window_frames if frame.index == 1)
    assert abs(context.time_s - 0.05) <= 1e-12
    assert abs(context.dt_s - 0.22) <= 1e-12
    assert abs(context.interval_start_s - 0.15) <= 1e-12
    assert abs(context.interval_end_s - 0.25) <= 1e-12
    assert abs(window.coverage["duration_s"] - 0.10) <= 1e-12
    assert window.coverage["frame_count"] == 1
    assert abs(window.coverage["end_s"] - 0.25) <= 1e-12

    pool = ReaderPool(decoder_factory=FakeFactory(make_spec(pts, last_duration=0.04)))
    assert read_frame_at_time(path, 0.269, pool=pool).index == 1
    assert read_frame_at_time(path, 0.30, pool=pool).index == 2
    assert read_frame_at_time(path, 0.43, pool=pool).index == 3

    override_factory = FakeFactory(make_spec(pts, last_duration=0.04))
    override = VideoSource(
        path, fps_override=20, source_fps=20,
        decoder_factory=override_factory,
    )
    override_frames = list(override.iter_frames())
    assert [round(frame.time_s, 6) for frame in override_frames] == [0.0, 0.05, 0.10, 0.15]
    assert abs(override.coverage["duration_s"] - 0.20) <= 1e-12
    info = read_frame_info(
        path, 2, fps_override=20,
        pool=ReaderPool(decoder_factory=FakeFactory(make_spec(pts, last_duration=0.04))),
    )
    assert info.time_s == 0.10
    assert info.pts == 5270
    assert info.clock_source == "fps_override"


def test_sequential_reader_reuses_decoder_and_returns_copies(tmp_path):
    count = 360
    pts = [1000 + 40 * index for index in range(count)]
    spec = make_spec(pts, keyframe_step=12, last_duration=0.04)
    path = write_file(tmp_path / "long.mp4")
    factory = FakeFactory(spec)
    pool = ReaderPool(decoder_factory=factory)

    info = read_frame_info(path, 300, pool=pool)
    assert abs(info.time_s - 12.0) <= 1e-12
    assert info.index == 300

    for index in range(301, 330):
        read_frame(path, index, pool=pool)

    reader = pool.get(path)
    assert factory.opens == 1
    assert reader.decoded_count() == 330

    image = read_frame(path, 300, pool=pool)
    image[:, :, :] = 0
    again = read_frame(path, 300, pool=pool)
    assert int(again[0, 0, 0]) == 300

    fallback_pool = ReaderPool(
        decoder_factory=FakeFactory(spec, corrupt_seek=True)
    )
    read_frame_info(path, 200, pool=fallback_pool)
    frame = read_frame(path, 100, pool=fallback_pool)
    assert int(frame[0, 0, 0]) == 100
    fallback_reader = fallback_pool.get(path)
    assert fallback_reader.decoded_count() == 101


def test_clock_errors_pixels_and_identity(tmp_path):
    pts = [1, 4, 3, 5, 2]
    spec = make_spec(pts, tb=Fraction(1, 29), keyframe_step=1)
    path = write_file(tmp_path / "broken.mp4")
    factory = FakeFactory(spec)
    pool = ReaderPool(decoder_factory=factory)

    assert read_frame_info(path, 0, pool=pool).time_s == 0.0
    assert abs(read_frame_info(path, 1, pool=pool).time_s - 3 / 29) <= 1e-12

    with pytest.raises(ClockError):
        read_frame_info(path, 2, pool=pool)
    with pytest.raises(ClockError):
        read_frame_info(path, 4, pool=pool)

    pixel = read_frame(path, 3, pool=pool)
    assert int(pixel[0, 0, 0]) == 3

    override_pool = ReaderPool(decoder_factory=FakeFactory(spec))
    info = read_frame_info(path, 4, fps_override=29, pool=override_pool)
    assert abs(info.time_s - 4 / 29) <= 1e-12
    assert info.clock_source == "fps_override"

    for bad_index in (-1, 1.5, True):
        with pytest.raises(ValidationError):
            read_frame(path, bad_index, pool=pool)
    for bad_fps in (0, -1, float("nan"), float("inf"), True):
        with pytest.raises(ValidationError):
            read_frame_info(path, 0, fps_override=bad_fps, pool=pool)
    with pytest.raises(ValidationError):
        read_frame_at_time(path, -0.1, pool=pool)

    identity_pool = ReaderPool(decoder_factory=factory)
    first = read_frame(path, 0, pool=identity_pool)
    assert int(first[0, 0, 0]) == 0
    path.write_bytes(b"changed-file")
    factory.version = 2
    second = read_frame(path, 0, pool=identity_pool)
    assert int(second[0, 0, 0]) == 200
    assert factory.opens >= 2

    identity_pool.close_all()
    pool.close_all()
    assert all(decoder.closed for decoder in factory.decoders)


def test_sampling_uses_real_frames_and_reports_short_coverage(tmp_path):
    count = 30
    pts = [40 * index for index in range(count)]
    spec = make_spec(pts, keyframe_step=12, last_duration=0.04)
    path = write_file(tmp_path / "sampled.mp4")
    factory = FakeFactory(spec)

    source = VideoSource(
        path, source_fps=25, sample_fps=10, decoder_factory=factory
    )
    frames = list(source.iter_frames())
    indices = [frame.index for frame in frames]
    assert len(indices) > 2
    assert indices != list(range(indices[0], indices[0] + len(indices)))
    assert all(abs(frame.time_s - round(frame.time_s, 6)) <= 1e-9 for frame in frames)
    assert abs(source.coverage["end_s"] - 1.20) <= 1e-12
    assert abs(source.coverage["duration_s"] - 1.20) <= 1e-12

    short_factory = FakeFactory(spec)
    short = VideoSource(
        path, start_s=0.5, end_s=1.5, source_fps=25,
        decoder_factory=short_factory,
    )
    with pytest.raises(VideoSourceError):
        list(short.iter_frames())


def test_orientation_swaps_dimensions(tmp_path):
    path = write_file(tmp_path / "rotate.mp4")
    spec = make_spec([0, 40], last_duration=0.04)
    factory = FakeFactory(spec)

    upright = read_frame(path, 0, 90, decoder_factory=factory)
    assert upright.shape == (6, 4, 3)
    plain = read_frame(path, 0, 0, decoder_factory=factory)
    assert plain.shape == (4, 6, 3)
