"""时序背景：两段式（先定目标源帧、再取像素）与 255322e 旧实现逐值等价。

受控解码器：帧由内存合成（BGR 值随索引变化）+ 受控 PTS；不是真实录像。
真实解码与 Windows 表现不在本文件覆盖范围。
"""

from __future__ import annotations

import hashlib
from fractions import Fraction

import numpy as np
import pytest

from behaviourlab import detection, media
from behaviourlab.media import RawFrame
from test_media_source import FakeDecoder, make_spec, write_file

TOL = 1e-9


class Counter:
    def __init__(self):
        self.decoded = 0
        self.materialized = 0
        self.opens = 0
        self.gray = 0


class PendingPixel:
    """延迟物化的受控帧：只有 to_ndarray 被调用时才计数并产生像素。"""

    def __init__(self, index, counter):
        self.index = index
        self.counter = counter

    def to_ndarray(self, format=None):
        self.counter.materialized += 1
        return np.full((4, 6, 3), (self.index * 7 + 3) % 251, dtype=np.uint8)


class CountingDecoder(FakeDecoder):
    """按 decode_images/defer_images 决定是否真的产生 BGR 像素，并分别计数。"""

    def __init__(self, spec, counter, *, decode_images=True,
                 defer_images=False, **kwargs):
        super().__init__(spec, **kwargs)
        self.counter = counter
        self.decode_images = decode_images
        self.defer_images = defer_images

    def frame_at(self, index):
        self.counter.decoded += 1
        item = self.spec[index]
        image = None
        pending = None
        if self.decode_images:
            if self.defer_images:
                pending = PendingPixel(index, self.counter)
            else:
                image = PendingPixel(index, self.counter).to_ndarray()
        return RawFrame(
            index=-1, image=image, pts=item.get("pts"),
            time_base=item.get("time_base", Fraction(1, 1000)),
            keyframe=item.get("keyframe", False),
            duration_s=item.get("duration_s"), pending=pending,
        )


class CountingFactory:
    def __init__(self, spec, counter):
        self.spec = spec
        self.counter = counter
        self.decoders = []

    def __call__(self, path, decode_images=True, defer_images=False):
        self.counter.opens += 1
        decoder = CountingDecoder(self.spec, self.counter,
                                  decode_images=decode_images,
                                  defer_images=defer_images, version=1)
        self.decoders.append(decoder)
        return decoder


def sha(array) -> str:
    data = np.ascontiguousarray(array)
    return hashlib.sha256(data.tobytes() + repr(data.shape).encode()).hexdigest()


def legacy_background(path, settings, video, metadata, orientation, factory):
    """255322e 的单遍实现：VideoSource 取全像素 → 逐帧转灰 → 按目标取中位数。"""

    samples = int(settings.get("temporal_bg_samples", 90))
    window = video.get("window") or {}
    start = float(window.get("start_s") or 0.0)
    end = window.get("end_s")
    end_value = float(end) if detection.is_number(end) else None
    sample_end = end_value
    known_end = detection.interpreted_duration(metadata, video)
    if sample_end is None and known_end is not None and known_end > start:
        sample_end = known_end

    pool = media.ReaderPool(decoder_factory=factory, max_readers=1)
    source = media.VideoSource(
        path, start_s=start, end_s=end_value, orientation=orientation,
        fps_override=video.get("fps_override"),
        sample_fps=video.get("sample_fps"),
        source_fps=metadata.get("fps"),
        decode_images=True, pool=pool,
    )
    targets = None
    if sample_end is not None and sample_end > start:
        targets = [start + (sample_end - start) * index / samples
                   for index in range(samples)]
    stack = []
    chosen = []
    target_index = 0
    for frame in source.iter_frames():
        if frame.image is None:
            continue
        exposure = frame.interval_end_s - frame.interval_start_s
        if exposure <= 0:
            continue
        gray = frame.image if np.asarray(frame.image).ndim == 2 \
            else detection.to_gray(frame.image)
        if targets is None:
            stack.append(gray)
            chosen.append((frame.index, frame.time_s, exposure,
                           sha(frame.image)))
            if len(stack) >= samples:
                break
            continue
        sample_time = max(frame.time_s, float(start))
        if (target_index < len(targets)
                and sample_time + 1e-9 >= targets[target_index]):
            stack.append(gray)
            chosen.append((frame.index, frame.time_s, exposure,
                           sha(frame.image)))
            target_index += 1
            if target_index >= len(targets):
                break
    pool.close_all()
    if not stack:
        raise detection.DetectionError("没有可用的背景样本")
    return detection._median_strips(np.asarray(stack)), chosen


def new_plan(path, video, metadata, orientation, factory, settings):
    """生产两段式：规划目标源帧 → 只对这些帧取方向像素。"""

    samples = int(settings.get("temporal_bg_samples", 90))
    window = video.get("window") or {}
    start = float(window.get("start_s") or 0.0)
    end = window.get("end_s")
    end_value = float(end) if detection.is_number(end) else None
    sample_end = end_value
    known_end = detection.interpreted_duration(metadata, video)
    if sample_end is None and known_end is not None and known_end > start:
        sample_end = known_end
    pool = media.ReaderPool(decoder_factory=factory, max_readers=2)
    source = media.VideoSource(
        path, start_s=start, end_s=end_value, orientation=orientation,
        fps_override=video.get("fps_override"),
        sample_fps=video.get("sample_fps"),
        source_fps=metadata.get("fps"),
        decode_images=False, pool=pool,
    )
    plan, evidence = detection.temporal_frame_plan(
        source, start=start, sample_end=sample_end, samples=samples)
    images = list(source.frames_at([item[0] for item in plan], orientation,
                                  evidence))
    pool.close_all()
    return plan, images


def prepare(monkeypatch, path, settings, video, metadata, orientation,
            factory, width=6, height=4):
    real_pool = media.ReaderPool
    monkeypatch.setattr(
        media, "ReaderPool",
        lambda **kwargs: real_pool(decoder_factory=factory, **kwargs))
    detection.clear_background_cache()
    try:
        return detection.prepare_background(
            settings, {"path": str(path), "metadata": dict(metadata)},
            dict(video), orientation, width, height)
    finally:
        detection.clear_background_cache()


CASES = {
    # 名称: (spec, metadata, video, orientation)
    "fps_override": (None, {"fps": 25.0, "frame_count": 300,
                            "duration_s": 20.0},
                     {"fps_override": 12.5, "sample_fps": None,
                      "window": {"start_s": 0.0, "end_s": None}}, 0),
    "fps_override_sample": (None, {"fps": 25.0, "frame_count": 300,
                                   "duration_s": 20.0},
                            {"fps_override": 12.5, "sample_fps": 5,
                             "window": {"start_s": 0.0, "end_s": None}}, 0),
    "native_cfr": (None, {"fps": 12.5},
                   {"fps_override": None, "sample_fps": None,
                    "window": {"start_s": 0.0, "end_s": None}}, 0),
    "vfr": ("vfr", {"fps": 12.5},
            {"fps_override": None, "sample_fps": None,
             "window": {"start_s": 0.0, "end_s": None}}, 0),
    "start_crossing": (None, {"fps": 12.5, "frame_count": 300,
                              "duration_s": 24.0},
                       {"fps_override": None, "sample_fps": None,
                        "window": {"start_s": 0.05, "end_s": None}}, 0),
    "explicit_end": (None, {"fps": 12.5, "frame_count": 300,
                            "duration_s": 24.0},
                     {"fps_override": None, "sample_fps": None,
                      "window": {"start_s": 0.0, "end_s": 5.0}}, 0),
    "derived_end": (None, {"fps": 25.0, "frame_count": 300},
                    {"fps_override": 12.5, "sample_fps": None,
                     "window": {"start_s": 0.0, "end_s": None}}, 0),
    "unknown_end": (None, {}, {"fps_override": None, "sample_fps": None,
                               "window": {"start_s": 0.0, "end_s": None}}, 0),
    "rotated": (None, {"fps": 12.5, "frame_count": 300, "duration_s": 24.0},
                {"fps_override": None, "sample_fps": None,
                 "window": {"start_s": 0.0, "end_s": None}}, 90),
}


def spec_for(kind, frames=300, step_ms=80):
    if kind == "vfr":
        pts, moment = [], 0
        for index in range(frames):
            pts.append(moment)
            moment += 40 if index % 3 else 110
        return make_spec(pts, last_duration=0.04)
    return make_spec([step_ms * index for index in range(frames)],
                     last_duration=step_ms / 1000.0)


@pytest.mark.parametrize("name", sorted(CASES))
def test_two_phase_matches_legacy(name, tmp_path, monkeypatch):
    monkeypatch.setattr(media, "probe_video", lambda *a, **k: {"fps": None})
    kind, metadata, video, orientation = CASES[name]
    spec = spec_for(kind)
    path = write_file(tmp_path / f"{name}.mp4")
    settings = {"bg_method": "temporal", "temporal_bg_samples": 30}

    old_bg, chosen = legacy_background(
        path, settings, video, metadata, orientation,
        CountingFactory(spec, Counter()))
    plan, images = new_plan(path, video, metadata, orientation,
                            CountingFactory(spec, Counter()), settings)

    # 1) 选中的有序源 index、时间、曝光逐值相同。
    assert [item[0] for item in plan] == [item[0] for item in chosen]
    assert [item[1] for item in plan] == pytest.approx(
        [item[1] for item in chosen])
    assert [item[2] for item in plan] == pytest.approx(
        [item[2] for item in chosen])
    # 2) 每个样本的方向像素相同。
    assert [sha(image) for image in images] == [item[3] for item in chosen]


@pytest.mark.parametrize("name", sorted(CASES))
def test_prepare_background_matches_legacy(name, tmp_path, monkeypatch):
    monkeypatch.setattr(media, "probe_video", lambda *a, **k: {"fps": None})
    kind, metadata, video, orientation = CASES[name]
    spec = spec_for(kind)
    path = write_file(tmp_path / f"{name}-bg.mp4")
    settings = {"bg_method": "temporal", "temporal_bg_samples": 30}

    old_bg, _chosen = legacy_background(
        path, settings, video, metadata, orientation,
        CountingFactory(spec, Counter()))
    new_bg = prepare(monkeypatch, path, settings, video, metadata, orientation,
                     CountingFactory(spec, Counter()))
    assert np.array_equal(np.asarray(old_bg), np.asarray(new_bg))
    assert sha(old_bg) == sha(new_bg)


def test_sample_shortage_and_eof_keep_previous_behaviour(tmp_path, monkeypatch):
    spec = spec_for(None, frames=12)          # 只有 12 帧
    path = write_file(tmp_path / "short.mp4")
    settings = {"bg_method": "temporal", "temporal_bg_samples": 40}
    video = {"fps_override": None, "sample_fps": None,
             "window": {"start_s": 0.0, "end_s": None}}
    metadata = {"fps": 12.5, "frame_count": 12, "duration_s": 0.96}

    old_bg, chosen = legacy_background(path, settings, video, metadata, 0,
                                       CountingFactory(spec, Counter()))
    plan, images = new_plan(path, video, metadata, 0,
                            CountingFactory(spec, Counter()), settings)
    assert len(plan) == len(chosen) == 12, "帧数不足时按实际可用样本"
    assert [item[0] for item in plan] == [item[0] for item in chosen]
    new_bg = prepare(monkeypatch, path, settings, video, metadata, 0,
                     CountingFactory(spec, Counter()))
    assert np.array_equal(np.asarray(old_bg), np.asarray(new_bg))


def test_zero_exposure_context_frame_is_not_a_sample(tmp_path):
    """显式 end 时尾帧只作上下文（零曝光），不进入背景样本。"""

    spec = spec_for(None, frames=60)
    path = write_file(tmp_path / "tail.mp4")
    video = {"fps_override": 12.5, "sample_fps": None,
             "window": {"start_s": 0.0, "end_s": 2.0}}
    pool = media.ReaderPool(decoder_factory=CountingFactory(spec, Counter()),
                            max_readers=1)
    source = media.VideoSource(path, start_s=0.0, end_s=2.0,
                               fps_override=12.5, decode_images=False,
                               pool=pool)
    plan, _evidence = detection.temporal_frame_plan(
        source, start=0.0, sample_end=2.0, samples=5)
    pool.close_all()
    assert plan, "应有可用样本"
    assert all(item[2] > 0 for item in plan), "零曝光上下文帧不得计入"
    assert all(item[0] < 25 for item in plan), "样本只来自窗口内帧"

def test_bad_pts_still_rejected_in_planning(tmp_path, monkeypatch):
    """坏 PTS 在规划阶段就按既有规则失败，不因性能改动被跳过。"""

    spec = make_spec([0, 80, None, 240, 320])
    path = write_file(tmp_path / "bad-pts.mp4")
    settings = {"bg_method": "temporal", "temporal_bg_samples": 3}
    video = {"fps_override": None, "sample_fps": None,
             "window": {"start_s": 0.0, "end_s": None}}
    with pytest.raises(media.ClockError):
        legacy_background(path, settings, video, {"fps": 12.5}, 0,
                          CountingFactory(spec, Counter()))
    real_pool = media.ReaderPool
    monkeypatch.setattr(
        media, "ReaderPool",
        lambda **kwargs: real_pool(
            **{**kwargs, "decoder_factory": CountingFactory(spec, Counter())}))
    detection.clear_background_cache()
    try:
        with pytest.raises(media.ClockError):
            detection.prepare_background(
                settings, {"path": str(path), "metadata": {"fps": 12.5}},
                dict(video), 0, 6, 4)
    finally:
        detection.clear_background_cache()


def test_window_end_beyond_eof_keeps_existing_error(tmp_path, monkeypatch):
    """显式 end 超过真实覆盖：与旧实现同样拒绝，不悄悄少取样本。"""

    spec = spec_for(None, frames=20)
    path = write_file(tmp_path / "eof-end.mp4")
    settings = {"bg_method": "temporal", "temporal_bg_samples": 5}
    video = {"fps_override": 12.5, "sample_fps": None,
             "window": {"start_s": 0.0, "end_s": 30.0}}
    with pytest.raises(media.VideoSourceError):
        legacy_background(path, settings, video, {"fps": 12.5}, 0,
                          CountingFactory(spec, Counter()))
    real_pool = media.ReaderPool
    monkeypatch.setattr(
        media, "ReaderPool",
        lambda **kwargs: real_pool(
            **{**kwargs, "decoder_factory": CountingFactory(spec, Counter())}))
    detection.clear_background_cache()
    try:
        with pytest.raises(media.VideoSourceError):
            detection.prepare_background(
                settings, {"path": str(path), "metadata": {"fps": 12.5}},
                dict(video), 0, 6, 4)
    finally:
        detection.clear_background_cache()
