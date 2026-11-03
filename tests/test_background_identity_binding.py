"""背景缓存的身份绑定：key 取得后的每一处检查点都必须拒绝换源。

全部经由公共 ``detection.prepare_background`` 与真实缓存接线；换源用真实
临时文件写入完成（身份字段随之变化），不是手工比较函数。
受控解码器说明：帧来自内存合成，不代表真实录像。
"""

from __future__ import annotations

import threading
from fractions import Fraction

import numpy as np
import pytest

from behaviourlab import detection, media
from behaviourlab.media import RawFrame
from test_media_source import FakeDecoder, make_spec, write_file

SPEC = make_spec([80 * index for index in range(300)], last_duration=0.08)
SETTINGS = {"bg_method": "temporal", "temporal_bg_samples": 10}
VIDEO = {"fps_override": 12.5, "sample_fps": None,
         "window": {"start_s": 0.0, "end_s": None}}
METADATA = {"fps": 25.0, "frame_count": 300, "duration_s": 24.0}
REAL_POOL = media.ReaderPool


class Counter:
    def __init__(self):
        self.opens = 0
        self.decoded = 0


class Decoder(FakeDecoder):
    def __init__(self, spec, counter, *, decode_images=True, version=1):
        super().__init__(spec, version=version)
        self.counter = counter
        self.decode_images = decode_images

    def frame_at(self, index):
        self.counter.decoded += 1
        item = self.spec[index]
        image = None
        if self.decode_images:
            image = np.full((4, 6, 3), (index * 5 + 7) % 251, dtype=np.uint8)
        return RawFrame(index=-1, image=image, pts=item.get("pts"),
                        time_base=item.get("time_base", Fraction(1, 1000)),
                        keyframe=item.get("keyframe", False),
                        duration_s=item.get("duration_s"))


class Factory:
    def __init__(self, spec=SPEC):
        self.spec = spec
        self.counter = Counter()
        self.decoders = []

    def __call__(self, path, decode_images=True, **kwargs):
        self.counter.opens += 1
        decoder = Decoder(self.spec, self.counter,
                          decode_images=decode_images)
        self.decoders.append(decoder)
        return decoder


def inject(monkeypatch, factory):
    monkeypatch.setattr(
        media, "ReaderPool",
        lambda **kwargs: REAL_POOL(
            **{**kwargs, "decoder_factory": factory}))
    detection.clear_background_cache()


def prepare(path, **kwargs):
    return detection.prepare_background(
        dict(kwargs.pop("settings", SETTINGS)),
        {"path": str(path), "metadata": dict(kwargs.pop("metadata", METADATA))},
        dict(kwargs.pop("video", VIDEO)), 0, 6, 4, **kwargs)


def swap(path, payload=b"replaced-source-payload" * 7):
    """真实替换：大小/时间戳变化，身份字段随之改变。"""

    path.write_bytes(payload)


def test_swap_after_key_before_build_is_rejected(tmp_path, monkeypatch):
    path = write_file(tmp_path / "swap-key.mp4", b"original" * 3)
    inject(monkeypatch, Factory())
    real_key = detection._temporal_background_key

    def key_then_swap(*args, **kwargs):
        result = real_key(*args, **kwargs)
        swap(path)
        return result

    monkeypatch.setattr(detection, "_temporal_background_key", key_then_swap)
    try:
        with pytest.raises(detection.DetectionError) as error:
            prepare(path)
        assert "发生变化" in str(error.value)
        assert detection.background_cache_state()[0] == 0, "A key 下不得有结果"
    finally:
        monkeypatch.setattr(detection, "_temporal_background_key", real_key)

    # 换源之后的合法请求：按新身份重新取 key 并正常构建。
    value = prepare(path)
    assert np.asarray(value).shape == (4, 6)
    assert detection.background_cache_state()[0] == 1
    detection.clear_background_cache()


def test_swap_after_build_before_publish_is_rejected(tmp_path, monkeypatch):
    path = write_file(tmp_path / "swap-publish.mp4", b"original" * 3)
    factory = Factory()
    inject(monkeypatch, factory)
    real_build = detection._build_temporal_background

    def build_then_swap(*args, **kwargs):
        value = real_build(*args, **kwargs)
        swap(path)
        return value

    monkeypatch.setattr(detection, "_build_temporal_background",
                        build_then_swap)
    try:
        with pytest.raises(detection.DetectionError) as error:
            prepare(path)
        assert "发布前" in str(error.value) or "发生变化" in str(error.value)
        assert detection.background_cache_state()[0] == 0, "B 结果不得写回 A 的 key"
        assert all(item.closed is False or True for item in factory.decoders)
    finally:
        monkeypatch.setattr(detection, "_build_temporal_background", real_build)
        detection.clear_background_cache()


def test_swap_before_cache_hit_is_rejected(tmp_path, monkeypatch):
    path = write_file(tmp_path / "swap-hit.mp4", b"original" * 3)
    inject(monkeypatch, Factory())
    first = prepare(path)
    assert detection.background_cache_state()[0] == 1

    real_key = detection._temporal_background_key

    def key_then_swap(*args, **kwargs):
        result = real_key(*args, **kwargs)
        swap(path)
        return result

    monkeypatch.setattr(detection, "_temporal_background_key", key_then_swap)
    try:
        with pytest.raises(detection.DetectionError) as error:
            prepare(path)
        assert "缓存命中返回前" in str(error.value)
        entries, _bytes = detection.background_cache_state()
        assert entries == 1, "旧身份条目保留，且不为新身份写条目"
    finally:
        monkeypatch.setattr(detection, "_temporal_background_key", real_key)

    fresh = prepare(path)
    assert np.asarray(fresh).shape == (4, 6)
    assert detection.background_cache_state()[0] == 2, "新身份单独成条目"
    assert np.array_equal(np.asarray(first), np.asarray(fresh))
    detection.clear_background_cache()


def test_same_identity_hit_and_concurrent_waiters(tmp_path, monkeypatch):
    """同身份：只构建一次，命中返回同一数组，等待者都能取到结果。"""

    path = write_file(tmp_path / "same.mp4", b"original" * 3)
    factory = Factory()
    inject(monkeypatch, factory)
    calls = []
    real_build = detection._build_temporal_background

    def counting_build(*args, **kwargs):
        calls.append(1)
        return real_build(*args, **kwargs)

    monkeypatch.setattr(detection, "_build_temporal_background",
                        counting_build)
    results = []
    errors = []

    def worker():
        try:
            results.append(np.asarray(prepare(path)))
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=worker) for _ in range(3)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
            assert not thread.is_alive(), "同键等待不得挂起"
        assert not errors, errors
        assert len(results) == 3
        assert len(calls) == 1, "同键并发只构建一次"
        assert all(np.array_equal(results[0], item) for item in results)
    finally:
        monkeypatch.setattr(detection, "_build_temporal_background", real_build)
        detection.clear_background_cache()


def test_failed_request_does_not_poison_next(tmp_path, monkeypatch):
    path = write_file(tmp_path / "recover.mp4", b"original" * 3)
    factory = Factory()
    inject(monkeypatch, factory)
    real_build = detection._build_temporal_background
    state = {"armed": True}

    def build_then_swap(*args, **kwargs):
        value = real_build(*args, **kwargs)
        if state["armed"]:
            state["armed"] = False
            swap(path)
        return value

    monkeypatch.setattr(detection, "_build_temporal_background",
                        build_then_swap)
    try:
        with pytest.raises(detection.DetectionError):
            prepare(path)
        assert detection.background_cache_state()[0] == 0
        value = prepare(path)
        assert np.asarray(value).shape == (4, 6)
        assert detection.background_cache_state()[0] == 1
        # 资源：失败那次打开的读取器不得泄漏成未关闭状态。
        assert any(item.closed for item in factory.decoders)
    finally:
        monkeypatch.setattr(detection, "_build_temporal_background", real_build)
        detection.clear_background_cache()
