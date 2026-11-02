"""时序背景取帧正确性：seek 失败不得返回错像素；换源不得嫁接旧索引证据。

两项都走生产链 ``prepare_background → temporal_frame_plan → frames_at``，
像素真值由受控解码器按源帧号编码，不 mock 返回值。
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from behaviourlab import detection, media
from behaviourlab.media import RawFrame
from test_media_source import FakeDecoder, make_spec, write_file

SIZE = (4, 6)


class _Pending:
    """延迟物化的受控帧：只有 to_ndarray 才真正产生像素。"""

    def __init__(self, image, counter):
        self.image = image
        self.counter = counter

    def to_ndarray(self, format=None):
        self.counter["materialized"] += 1
        return self.image


class _Decoder(FakeDecoder):
    """像素值 = 真实源帧号，seek 行为可按用例配置。"""

    def __init__(self, spec, counter, *, decode_images=True,
                 defer_images=False, **kwargs):
        super().__init__(spec, **kwargs)
        self.counter = counter
        self.decode_images = decode_images
        self.defer_images = defer_images

    def frame_at(self, index):
        self.counter["decoded"] += 1
        item = self.spec[index]
        image = None
        pending = None
        if self.decode_images:
            image = np.full((*SIZE, 3), index, dtype=np.uint16)
            if self.defer_images:
                pending = _Pending(image, self.counter)
                image = None
        return RawFrame(index=-1, image=image, pts=item.get("pts"),
                        time_base=item.get("time_base", Fraction(1, 1000)),
                        keyframe=item.get("keyframe", False),
                        duration_s=item.get("duration_s"), pending=pending)


class _Factory:
    def __init__(self, spec, counter, **kwargs):
        self.spec = spec
        self.counter = counter
        self.options = kwargs
        self.decoders = []

    def __call__(self, path, decode_images=True, defer_images=False, **kwargs):
        self.counter["opens"] += 1
        decoder = _Decoder(self.spec, self.counter,
                           decode_images=decode_images,
                           defer_images=defer_images, version=0,
                           **self.options)
        self.decoders.append(decoder)
        return decoder


def _spec25(count: int = 400, step_ms: int = 40):
    return make_spec([step_ms * index for index in range(count)],
                     keyframe_step=12)


def _value(image) -> int:
    return int(np.asarray(image).flat[0])


def _run(tmp_path, name, spec, *, factory_kwargs=None, samples=2,
         start=4.0, end=8.0, override=25.0, prepare=False, counter=None):
    counter = counter if counter is not None else {
        "decoded": 0, "materialized": 0, "opens": 0}
    path = write_file(tmp_path / f"{name}.mp4")
    factory = _Factory(spec, counter, **(factory_kwargs or {}))
    real_pool = media.ReaderPool
    media.ReaderPool = lambda **kw: real_pool(
        **{**kw, "decoder_factory": factory})
    detection.clear_background_cache()
    try:
        asset = {"path": str(path),
                 "metadata": {"width": SIZE[1], "height": SIZE[0],
                              "fps": 25.0, "frame_count": len(spec),
                              "duration_s": len(spec) / 25.0}}
        video = {"fps_override": override, "sample_fps": None,
                 "window": {"start_s": start, "end_s": end}}
        if prepare:
            background = detection.prepare_background(
                {"bg_method": "temporal", "temporal_bg_samples": samples},
                asset, dict(video), 0, *SIZE)
            plan = None
        else:
            pool = media.ReaderPool(decoder_factory=factory, max_readers=2)
            source = media.VideoSource(
                asset["path"], start_s=start, end_s=end, orientation=0,
                fps_override=override, sample_fps=None, source_fps=25.0,
                decode_images=False, pool=pool)
            plan, evidence = detection.temporal_frame_plan(
                source, start=start, sample_end=end, samples=samples)
            values = [_value(image) for image in
                      source.frames_at([item[0] for item in plan], 0, evidence)]
            pool.close_all()
            background = None
            return {"plan": plan, "values": values, "counter": counter,
                    "factory": factory, "path": path, "video": video,
                    "asset": asset}
    finally:
        media.ReaderPool = real_pool
        detection.clear_background_cache()
    return {"background": np.asarray(background), "counter": counter,
            "factory": factory, "path": path, "video": video, "asset": asset}


def test_normal_seek_returns_true_pixels(tmp_path):
    result = _run(tmp_path, "normal", _spec25())
    assert [item[0] for item in result["plan"]] == [100, 150]
    assert result["values"] == [100, 150]


def test_overshoot_seek_falls_back_to_sequential_with_true_pixels(tmp_path):
    """seek 越过锚点：必须回到顺序读取，返回 100/150 而不是 301/351。"""

    result = _run(tmp_path, "overshoot", _spec25(),
                  factory_kwargs={"overshoot_to": 201})
    assert [item[0] for item in result["plan"]] == [100, 150]
    assert result["values"] == [100, 150], "不得返回越过锚点后的错像素"
    # 有界回退：失败后停用 seek，剩余目标顺序读取，不反复重开。
    assert result["counter"]["opens"] <= 6


def test_failing_seek_still_returns_true_pixels(tmp_path):
    result = _run(tmp_path, "failseek", _spec25(),
                  factory_kwargs={"fail_seek": True})
    assert result["values"] == [100, 150]


@pytest.mark.parametrize("options", [{}, {"overshoot_to": 201},
                                     {"fail_seek": True}])
def test_background_median_and_samples_unchanged(tmp_path, options):
    """生产入口：三种 seek 行为下背景中位数与样本时刻/曝光完全一致。"""

    base = _run(tmp_path, f"bg-ref-{len(options)}", _spec25(), prepare=True,
                samples=2)
    other = _run(tmp_path, f"bg-{len(options)}-{sorted(options)}", _spec25(),
                 prepare=True, samples=2, factory_kwargs=options)
    assert np.array_equal(base["background"], other["background"])
    # 两个目标帧 100/150 的中位数。
    assert int(base["background"].flat[0]) == 125


def test_same_identity_reuses_seeded_evidence(tmp_path):
    """同一文件身份：证据被采纳，向后跳转用 seek 而不是从头重扫。"""

    result = _run(tmp_path, "same", _spec25(), samples=3)
    assert result["values"] == [item[0] for item in result["plan"]]
    seeks = sum(decoder.seek_calls for decoder in result["factory"].decoders)
    assert seeks >= 1, "同身份证据应带来 seek 快路"


def test_source_replaced_between_plan_and_pixels_is_rejected(tmp_path):
    """规划后换片：明确报源文件变化，旧索引证据不嫁接。"""

    counter = {"decoded": 0, "materialized": 0, "opens": 0}
    old_spec = _spec25()
    new_spec = make_spec([40 * (index + 2) for index in range(400)],
                         keyframe_step=1)
    factory = _Factory(old_spec, counter)
    path = write_file(tmp_path / "swap.mp4", b"A" * 100)
    real_pool = media.ReaderPool
    media.ReaderPool = lambda **kw: real_pool(
        **{**kw, "decoder_factory": factory})
    detection.clear_background_cache()
    try:
        pool = media.ReaderPool(decoder_factory=factory, max_readers=2)
        source = media.VideoSource(
            path, start_s=4.0, end_s=8.0, orientation=0, fps_override=25.0,
            sample_fps=None, source_fps=25.0, decode_images=False, pool=pool)
        plan, evidence = detection.temporal_frame_plan(
            source, start=4.0, sample_end=8.0, samples=2)
        assert [item[0] for item in plan] == [100, 150]

        # 同一路径换成另一份文件（身份与关键帧布局都不同）。
        factory.spec = new_spec
        write_file(path, b"B" * 137)
        with pytest.raises(media.VideoSourceError) as error:
            list(source.frames_at([item[0] for item in plan], 0, evidence))
        assert "发生变化" in str(error.value)
        pool.close_all()
    finally:
        media.ReaderPool = real_pool
        detection.clear_background_cache()


def test_evidence_bound_to_other_identity_or_clock_is_rejected(tmp_path):
    """证据投给另一份源或另一时钟参数：拒绝采纳，退回顺序读取。"""

    spec = _spec25()
    path = write_file(tmp_path / "clip.mp4", b"x" * 64)
    factory = _Factory(spec, {"decoded": 0, "materialized": 0, "opens": 0})
    pool = media.ReaderPool(decoder_factory=factory, max_readers=2)
    try:
        source = media.VideoSource(
            path, start_s=4.0, end_s=8.0, orientation=0, fps_override=25.0,
            sample_fps=None, source_fps=25.0, decode_images=False, pool=pool)
        plan, evidence = detection.temporal_frame_plan(
            source, start=4.0, sample_end=8.0, samples=2)
        assert evidence.identity == media.identity_key(
            media.file_identity(path))

        other = media.FrameReader(path, decoder_factory=factory,
                                  fps_override=12.5, decode_images=True,
                                  defer_images=True)
        assert other.seed_index_evidence(evidence) is False, "解释帧率不同"
        other.close()

        wrong = media.IndexEvidence(evidence.clock_source,
                                    evidence.first_abs, evidence.anchors,
                                    (0, 0, 1, 1, 1), evidence.fps_override)
        same = media.FrameReader(path, decoder_factory=factory,
                                 fps_override=25.0, decode_images=True,
                                 defer_images=True)
        assert same.seed_index_evidence(wrong) is False, "身份不同"
        assert same.seed_index_evidence(evidence) is True, "同身份应采纳"
        same.close()

        # 即使证据被拒，取帧结果仍正确（只是退回顺序读取）。
        values = [_value(image) for image in
                  source.frames_at([item[0] for item in plan], 0, evidence)]
        assert values == [100, 150]
    finally:
        pool.close_all()


def test_replacement_fails_background_and_is_not_cached(tmp_path, monkeypatch):
    """换片使背景构建失败：不发布背景，也不写缓存；之后正常请求可恢复。"""

    counter = {"decoded": 0, "materialized": 0, "opens": 0}
    spec = _spec25()
    path = write_file(tmp_path / "bg.mp4", b"x" * 32)
    factory = _Factory(spec, counter)
    real_pool = media.ReaderPool
    media.ReaderPool = lambda **kw: real_pool(
        **{**kw, "decoder_factory": factory})
    detection.clear_background_cache()

    real_plan = detection.temporal_frame_plan

    def plan_then_swap(source, **kwargs):
        result = real_plan(source, **kwargs)
        # 规划与取像素之间换片。
        factory.spec = make_spec([40 * (index + 2) for index in range(400)],
                                 keyframe_step=1)
        write_file(path, b"Y" * 211)
        return result

    monkeypatch.setattr(detection, "temporal_frame_plan", plan_then_swap)
    try:
        with pytest.raises((media.VideoSourceError, detection.DetectionError)):
            detection.prepare_background(
                {"bg_method": "temporal", "temporal_bg_samples": 2},
                {"path": str(path), "metadata": {"fps": 25.0}},
                {"fps_override": 25.0, "sample_fps": None,
                 "window": {"start_s": 4.0, "end_s": 8.0}},
                0, *SIZE)
        assert detection.background_cache_state()[0] == 0, "失败不入缓存"
    finally:
        monkeypatch.setattr(detection, "temporal_frame_plan", real_plan)

    # 恢复成新文件的完整新规划：正常请求仍然可用且不沿用旧时间。
    counter2 = {"decoded": 0, "materialized": 0, "opens": 0}
    media.ReaderPool = lambda **kw: real_pool(
        **{**kw, "decoder_factory": _Factory(
            make_spec([40 * (index + 2) for index in range(400)],
                      keyframe_step=12), counter2)})
    try:
        background = detection.prepare_background(
            {"bg_method": "temporal", "temporal_bg_samples": 2},
            {"path": str(path), "metadata": {"fps": 25.0}},
            {"fps_override": 25.0, "sample_fps": None,
             "window": {"start_s": 4.0, "end_s": 8.0}},
            0, *SIZE)
        assert np.asarray(background).shape == SIZE
    finally:
        media.ReaderPool = real_pool
        detection.clear_background_cache()
