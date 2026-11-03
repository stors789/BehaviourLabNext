"""时序背景复用：有界缓存、同键并发、失效与失败不污染。

受控解码器说明：帧来自内存合成；真实录像与 Windows 表现不在本文件覆盖。
"""

from __future__ import annotations

import threading
import time
from fractions import Fraction

import numpy as np
import pytest

from behaviourlab import detection, media
from behaviourlab.media import RawFrame
from test_media_source import FakeDecoder, make_spec, write_file


class Counter:
    def __init__(self):
        self.decoded = 0
        self.materialized = 0
        self.opens = 0


class SlowDecoder(FakeDecoder):
    """按索引出像素；可指定在第几次解码抛错、是否延迟。"""

    def __init__(self, spec, counter, *, decode_images=True, fail_at=None,
                 delay=0.0, **kwargs):
        super().__init__(spec, **kwargs)
        self.counter = counter
        self.decode_images = decode_images
        self.fail_at = fail_at
        self.delay = delay

    def frame_at(self, index):
        if self.delay:
            time.sleep(self.delay)
        self.counter.decoded += 1
        if self.fail_at is not None and self.counter.decoded >= self.fail_at:
            raise RuntimeError("受控解码失败")
        item = self.spec[index]
        image = None
        if self.decode_images:
            self.counter.materialized += 1
            image = np.full((4, 6, 3), (index * 5 + 7) % 251, dtype=np.uint8)
        return RawFrame(index=-1, image=image, pts=item.get("pts"),
                        time_base=item.get("time_base", Fraction(1, 1000)),
                        keyframe=item.get("keyframe", False),
                        duration_s=item.get("duration_s"))


class Factory:
    def __init__(self, spec, counter, *, fail_at=None, delay=0.0):
        self.spec = spec
        self.counter = counter
        self.fail_at = fail_at
        self.delay = delay
        self.decoders = []

    def __call__(self, path, decode_images=True):
        self.counter.opens += 1
        decoder = SlowDecoder(self.spec, self.counter,
                              decode_images=decode_images, version=1,
                              fail_at=self.fail_at, delay=self.delay)
        self.decoders.append(decoder)
        return decoder


SPEC = make_spec([80 * index for index in range(300)], last_duration=0.08)
SETTINGS = {"bg_method": "temporal", "temporal_bg_samples": 10}
VIDEO = {"fps_override": 12.5, "sample_fps": None,
         "window": {"start_s": 0.0, "end_s": None}}
METADATA = {"fps": 25.0, "frame_count": 300, "duration_s": 24.0}


REAL_POOL = media.ReaderPool


def inject(monkeypatch, factory):
    monkeypatch.setattr(
        media, "ReaderPool",
        lambda **kwargs: REAL_POOL(
            **{**kwargs, "decoder_factory": factory}))
    detection.clear_background_cache()


def track_builds(monkeypatch):
    calls = []
    real = detection._build_temporal_background

    def wrapper(asset, video, orientation, start, end_value, sample_end,
                samples, cancel=None, expected_identity=None):
        calls.append((str(asset.get("path")), float(start), end_value, samples))
        return real(asset, video, orientation, start, end_value, sample_end,
                    samples, cancel, expected_identity=expected_identity)

    monkeypatch.setattr(detection, "_build_temporal_background", wrapper)
    return calls


def prepare(path, asset_metadata=None, video=None, settings=None,
            orientation=0, width=6, height=4, cancel=None):
    return detection.prepare_background(
        dict(settings or SETTINGS),
        {"path": str(path), "metadata": dict(asset_metadata or METADATA)},
        dict(video or VIDEO), orientation, width, height, cancel=cancel)


def test_same_facts_build_once(tmp_path, monkeypatch):
    path = write_file(tmp_path / "same.mp4")
    counter = Counter()
    inject(monkeypatch, Factory(SPEC, counter))
    calls = track_builds(monkeypatch)
    try:
        first = prepare(path)
        second = prepare(path)
    finally:
        detection.clear_background_cache()
    assert len(calls) == 1, "同语义事实只构建一次"
    assert first is second, "命中缓存返回同一数组"
    assert np.array_equal(first, second)


def test_different_window_or_facts_build_separately(tmp_path, monkeypatch):
    path = write_file(tmp_path / "diff.mp4")
    inject(monkeypatch, Factory(SPEC, Counter()))
    calls = track_builds(monkeypatch)
    try:
        base = prepare(path)
        other_window = prepare(path, video=dict(
            VIDEO, window={"start_s": 1.0, "end_s": None}))
        other_orientation = prepare(path, orientation=90)
        other_samples = prepare(path, settings=dict(
            SETTINGS, temporal_bg_samples=6))
        other_fps = prepare(path, video=dict(VIDEO, fps_override=25.0))
        assert len(calls) == 5, "窗口/方向/样本数/解释FPS 各自成键"
        assert not np.array_equal(base, other_window)
        assert base.shape == other_orientation.T.shape
        assert detection.background_cache_state()[0] == 5
    finally:
        detection.clear_background_cache()


def test_source_replacement_invalidates(tmp_path, monkeypatch):
    path = write_file(tmp_path / "swap.mp4")
    inject(monkeypatch, Factory(SPEC, Counter()))
    calls = track_builds(monkeypatch)
    try:
        before = prepare(path)
        # 换文件：字节与 mtime 变化，identity 不同。
        time.sleep(0.01)
        path.write_bytes(b"replaced-source-bytes")
        after = prepare(path)
        assert len(calls) == 2, "源替换必须重建"
        assert detection.background_cache_state()[0] == 2
        # 同一份受控解码器给同一组像素；重建证据是构建次数与独立条目。
        assert np.array_equal(before, after)
    finally:
        detection.clear_background_cache()


def test_cached_array_is_read_only(tmp_path, monkeypatch):
    path = write_file(tmp_path / "ro.mp4")
    inject(monkeypatch, Factory(SPEC, Counter()))
    try:
        background = prepare(path)
        assert background.flags.writeable is False
        with pytest.raises(ValueError):
            background[0, 0] = 1
    finally:
        detection.clear_background_cache()


def test_cache_bounds_evict_oldest(tmp_path, monkeypatch):
    path = write_file(tmp_path / "bounds.mp4")
    inject(monkeypatch, Factory(SPEC, Counter()))
    monkeypatch.setattr(detection, "BACKGROUND_CACHE_ENTRIES", 2)
    calls = track_builds(monkeypatch)
    try:
        prepare(path)
        prepare(path, video=dict(VIDEO, window={"start_s": 1.0, "end_s": None}))
        prepare(path, video=dict(VIDEO, window={"start_s": 2.0, "end_s": None}))
        entries, _bytes = detection.background_cache_state()
        assert entries == 2, "按条目上限淘汰"
        # 已淘汰的最早条目需要重建。
        prepare(path)
        assert len(calls) == 4
    finally:
        detection.clear_background_cache()


def test_concurrent_same_key_builds_once(tmp_path, monkeypatch):
    path = write_file(tmp_path / "threads.mp4")
    counter = Counter()
    inject(monkeypatch, Factory(SPEC, counter, delay=0.002))
    calls = track_builds(monkeypatch)
    results = []
    errors = []

    def worker():
        try:
            results.append(prepare(path))
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    try:
        threads = [threading.Thread(target=worker) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)
        assert not errors, errors
        assert len(calls) == 1, "同键并发只建一次"
        assert len(results) == 3
        assert all(item is results[0] for item in results)
    finally:
        detection.clear_background_cache()


def test_cancel_and_failure_are_not_cached(tmp_path, monkeypatch):
    path = write_file(tmp_path / "cancel.mp4")
    inject(monkeypatch, Factory(SPEC, Counter()))
    calls = track_builds(monkeypatch)
    try:
        with pytest.raises(detection.DetectionError):
            prepare(path, cancel=lambda: True)
        assert detection.background_cache_state()[0] == 0, "取消不留半成品"
        assert len(calls) == 1, "取消发生在构建过程中，但结果不发布"

        # 失败：解码中途抛错，也不得写入缓存。
        detection.clear_background_cache()
        inject(monkeypatch, Factory(SPEC, Counter(), fail_at=3))
        with pytest.raises(RuntimeError):
            prepare(path)
        assert detection.background_cache_state()[0] == 0
    finally:
        detection.clear_background_cache()

    # 失败后下一次正常请求不受污染。
    inject(monkeypatch, Factory(SPEC, Counter()))
    calls = track_builds(monkeypatch)
    try:
        background = prepare(path)
        assert background.shape == (4, 6)
        assert len(calls) == 1
    finally:
        detection.clear_background_cache()


def test_unreadable_path_is_not_cached(tmp_path, monkeypatch):
    inject(monkeypatch, Factory(SPEC, Counter()))
    calls = track_builds(monkeypatch)
    try:
        missing = tmp_path / "missing.mp4"
        with pytest.raises(media.MediaError):
            prepare(missing)
        assert detection.background_cache_state()[0] == 0
        # 身份不可归属：每次现建（不进缓存，也不与别的输入共用条目）。
        with pytest.raises(media.MediaError):
            prepare(tmp_path / "missing-2.mp4")
        assert len(calls) == 2
    finally:
        detection.clear_background_cache()
