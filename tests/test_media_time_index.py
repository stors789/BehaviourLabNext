"""按时间读取：先定位真实源索引，再按索引读像素与时钟。

受控解码器说明：这里用内存合成帧与受控 PTS/计数器，不是真实录像；
真实视频解码与浏览器显示不在本文件覆盖范围。
"""

from __future__ import annotations

from contextlib import contextmanager
from fractions import Fraction

import numpy as np
import pytest

from behaviourlab import media, preview
from behaviourlab.media import ClockError, ReaderPool, read_frame_at_time
from behaviourlab.schema import ValidationError
from test_media_source import FakeDecoder, FakeFactory, make_spec

TOL = 1e-9


class Counter:
    def __init__(self):
        self.decoded = 0
        self.opens = 0


class CountedDecoder(FakeDecoder):
    def __init__(self, spec, counter, **kwargs):
        super().__init__(spec, **kwargs)
        self.counter = counter

    def frame_at(self, index):
        self.counter.decoded += 1
        return super().frame_at(index)


class CountedFactory(FakeFactory):
    def __init__(self, spec, counter, **kwargs):
        super().__init__(spec, **kwargs)
        self.counter = counter

    def __call__(self, path, decode_images=True):
        self.counter.opens += 1
        decoder = CountedDecoder(
            self.spec, self.counter, version=self.version,
            corrupt_seek=self.corrupt_seek, overshoot_to=self.overshoot_to,
        )
        self.decoders.append(decoder)
        return decoder


def fps_spec(count: int = 250, step_ms: int = 80):
    """80 ms 间隔：与 12.5 FPS 的解释时钟一致。"""

    return make_spec([step_ms * index for index in range(count)],
                     last_duration=step_ms / 1000.0)


class Harness:
    def __init__(self, spec, tmp_path, name="clip.mp4"):
        self.path = tmp_path / name
        self.path.write_bytes(b"controlled-source")
        self.counter = Counter()
        self.factory = CountedFactory(spec, self.counter)
        self.pool = ReaderPool(decoder_factory=self.factory, max_readers=1)

    def read(self, target, orientation=0, override=None):
        return read_frame_at_time(self.path, target, orientation, override,
                                  pool=self.pool)

    def close(self):
        self.pool.close_all()


@contextmanager
def harness(spec, tmp_path, name="clip.mp4"):
    item = Harness(spec, tmp_path, name)
    try:
        yield item
    finally:
        item.close()


def legacy_scan(path, target, orientation=0, fps_override=None, factory=None):
    """优化前的实现：无条件 rewind 后从头顺序遍历，作为对照基准。"""

    pool = ReaderPool(decoder_factory=factory, max_readers=1)
    reader = pool.get(path, fps_override=fps_override, decode_images=True)
    try:
        with reader._lock:
            reader.rewind()
            best = None
            while True:
                try:
                    raw = reader._next_raw()
                except StopIteration:
                    break
                meta = reader._info_cache.get(raw.index)
                if meta is None or meta.time_s is None:
                    raise ClockError("源时钟不可解释")
                if meta.time_s <= target + media.TIME_EPS:
                    best = (raw, meta)
                elif best is not None:
                    break
            if best is None:
                raise media.FrameReadError("时间点之前没有可解码帧")
            raw, meta = best
            image = reader._image_at(raw)
            return media.FrameInfo(
                image=media._orient(image, orientation),
                index=raw.index,
                time_s=float(meta.time_s),
                pts=meta.pts,
                time_base=meta.time_base,
                clock_source=meta.clock_source,
            )
    finally:
        pool.close_all()


# -- 一、显式解释帧率：时间直接换算真实索引 -----------------------------
def test_fps_override_maps_time_to_real_index(tmp_path):
    with harness(fps_spec(), tmp_path, "fps.mp4") as h:
        info = h.read(14.96, override=12.5)
        assert info.index == 187
        assert info.time_s == pytest.approx(14.96, abs=TOL)
        assert info.pts == 80 * 187, "原 PTS 字段保持源值"
        assert info.time_base == Fraction(1, 1000)
        assert info.clock_source == "fps_override"
        assert info.image.shape == (4, 6, 3)

        earlier = h.read(14.959, override=12.5)
        assert earlier.index == 186
        assert earlier.time_s == pytest.approx(14.88, abs=TOL)

        # 90° 像素方向按请求给出，索引与时间不受旋转影响。
        rotated = h.read(14.96, orientation=90, override=12.5)
        assert rotated.index == 187
        assert rotated.image.shape == (6, 4, 3)
        assert int(rotated.image[0, 0, 0]) == 187

        # 方向切换不串：切回 0° 仍是未旋转原帧。
        plain = h.read(14.96, override=12.5)
        assert plain.image.shape == (4, 6, 3)
        assert int(plain.image[0, 0, 0]) == 187
        assert plain.time_s == pytest.approx(14.96, abs=TOL)


@pytest.mark.parametrize("bad", [0, -1, True, "12.5", float("nan"), float("inf")])
def test_invalid_override_still_rejected(tmp_path, bad):
    path = tmp_path / "bad-override.mp4"
    path.write_bytes(b"x")
    with pytest.raises(ValidationError):
        read_frame_at_time(path, 1.0, 0, bad,
                           pool=ReaderPool(decoder_factory=CountedFactory(
                               fps_spec(10), Counter())))


def test_fps_lookup_reuses_index_and_cache(tmp_path):
    with harness(fps_spec(), tmp_path, "heat.mp4") as h:
        first = h.read(14.96, override=12.5)
        assert first.index == 187
        assert h.counter.decoded == 188, "冷读只解码到目标帧"
        assert h.counter.opens == 1

        again = h.read(14.96, override=12.5)
        assert (again.index, again.time_s) == (first.index, first.time_s)
        assert h.counter.decoded == 188, "重复目标不应再解码"
        assert h.counter.opens == 1

        forward = h.read(15.04, override=12.5)
        assert forward.index == 188
        assert h.counter.decoded == 189, "向前相邻只增解码下一帧"

        backward = h.read(14.88, override=12.5)
        assert backward.index == 186
        assert h.counter.decoded == 189, "向后相邻命中缓存"
        assert int(backward.image[0, 0, 0]) == 186


def test_fps_backward_after_eviction_is_bounded(tmp_path):
    with harness(fps_spec(400), tmp_path, "evict.mp4") as h:
        h.read(1.0, override=12.5)
        cold = h.counter.decoded
        far = h.read(30.0, override=12.5)
        assert far.index == 375
        back = h.read(1.0, override=12.5)
        assert back.index == 12
        assert back.time_s == pytest.approx(0.96, abs=TOL)
        assert h.counter.opens == 2, "只重开一次，不循环重试"
        assert h.counter.decoded <= cold + 376 + 13


def test_seek_failure_falls_back_to_sequential(tmp_path):
    with harness(fps_spec(120), tmp_path, "no-seek.mp4") as h:
        h.factory.fail_seek = True
        first = h.read(8.0, override=12.5)
        assert first.index == 100
        back = h.read(0.16, override=12.5)
        assert back.index == 2
        assert back.time_s == pytest.approx(0.16, abs=TOL)
        assert h.counter.opens == 2


# -- 二、可靠 PTS / VFR：按真实时间顺序定位 -----------------------------
def test_pts_lookup_uses_real_times(tmp_path):
    with harness(make_spec([0, 40, 110, 150]), tmp_path, "vfr.mp4") as h:
        info = h.read(0.10)
        assert info.index == 1, "0.10 秒落在 0.04 那帧，不能被平均帧率推到 0.11"
        assert info.time_s == pytest.approx(0.04, abs=TOL)
        assert info.clock_source == "pts"
        assert h.counter.decoded == 3

        later = h.read(0.11)
        assert later.index == 2
        assert later.time_s == pytest.approx(0.11, abs=TOL)
        assert h.counter.decoded == 4, "向前相邻只补一帧"

        repeat = h.read(0.10)
        assert (repeat.index, repeat.time_s) == (1, info.time_s)
        assert h.counter.decoded == 4, "重复目标由已解码证据回答"


def test_pts_backward_reuses_evidence_then_rewinds_once(tmp_path):
    with harness(make_spec([40 * index for index in range(200)]),
                 tmp_path, "pts-evict.mp4") as h:
        h.read(0.4)
        forward = h.read(7.0)
        assert forward.index == 175
        opens = h.counter.opens
        back = h.read(0.4)
        assert back.index == 10
        assert back.time_s == pytest.approx(0.4, abs=TOL)
        assert h.counter.opens == opens + 1, "目标在已解码窗口之前只重开一次"


def test_bad_pts_still_rejected(tmp_path):
    missing = tmp_path / "missing-pts.mp4"
    missing.write_bytes(b"x")
    with pytest.raises(ClockError):
        read_frame_at_time(missing, 0.2, 0, None,
                           pool=ReaderPool(decoder_factory=FakeFactory(
                               make_spec([0, 40, None, 120]))))
    backwards = tmp_path / "backwards-pts.mp4"
    backwards.write_bytes(b"x")
    with pytest.raises(ClockError):
        read_frame_at_time(backwards, 0.2, 0, None,
                           pool=ReaderPool(decoder_factory=FakeFactory(
                               make_spec([0, 100, 50, 200]))))


def test_time_beyond_eof_returns_last_frame(tmp_path):
    with harness(fps_spec(10), tmp_path, "eof.mp4") as h:
        info = h.read(100.0, override=12.5)
        assert info.index == 9, "超过真实 EOF 仍返回最后可解码帧"
        assert info.time_s == pytest.approx(0.72, abs=TOL)
    with harness(fps_spec(10), tmp_path, "eof-pts.mp4") as h:
        pts = h.read(100.0)
        assert pts.index == 9
        assert pts.time_s == pytest.approx(0.72, abs=TOL)


def test_identity_change_and_mode_switch_do_not_leak(tmp_path):
    with harness(fps_spec(40), tmp_path, "switch.mp4") as h:
        fps = h.read(1.0, override=12.5)
        assert (fps.index, fps.clock_source) == (12, "fps_override")
        pts = h.read(1.0)
        assert pts.index == 12
        assert pts.time_s == pytest.approx(0.96, abs=TOL), "无覆盖时按真实 PTS"
        assert pts.clock_source == "pts"

        h.factory.version = 2
        h.path.write_bytes(b"replaced-source")
        replaced = h.read(1.0, override=12.5)
        assert int(replaced.image[0, 0, 0]) == 12 + 200, "换文件后不得返回旧画面"
        assert replaced.time_s == pytest.approx(0.96, abs=TOL)


# -- 三、与优化前逐值一致 ---------------------------------------------
@pytest.mark.parametrize("override,target", [
    (12.5, 14.96), (12.5, 3.2), (12.5, 0.0),
    (None, 0.10), (None, 0.15), (None, 7.32),
])
def test_frame_info_matches_legacy_scan(tmp_path, override, target):
    spec = fps_spec(250)
    with harness(spec, tmp_path, "parity.mp4") as h:
        produced = h.read(target, override=override)
        reference = legacy_scan(h.path, target, 0, override,
                                CountedFactory(spec, Counter()))

    assert produced.index == reference.index
    assert produced.time_s == reference.time_s
    assert produced.pts == reference.pts
    assert produced.time_base == reference.time_base
    assert produced.clock_source == reference.clock_source
    assert np.array_equal(produced.image, reference.image)


def test_preview_by_time_matches_legacy_entry(client, store, tmp_path,
                                              monkeypatch):
    """同一选定帧：优化前后经预览入口产生的检测事实一致。"""

    from test_media_preview_defaults import GrayDecoder, GrayFactory
    from behaviourlab.media import FrameInfo

    spec = make_spec([80 * index for index in range(250)])
    path = tmp_path / "preview-parity.mp4"
    path.write_bytes(b"controlled-not-real-video")

    class PreviewFactory(GrayFactory):
        def __call__(self, decoder_path, decode_images=True):
            self.opens += 1
            decoder = GrayDecoder(self.spec, version=self.version)
            self.decoders.append(decoder)
            return decoder

    asset = store.create_asset(
        name="parity.mp4", path=str(path), project_id=None,
        metadata={"width": 200, "height": 200, "fps": 25.0,
                  "frame_count": 250, "duration_s": 10.0})
    draft = store.create_draft("oft", videos=[{
        "asset_id": asset["id"],
        "window": {"start_s": 0, "end_s": None},
        "orientation": 90, "fps_override": 12.5, "sample_fps": 2,
        "units": [{
            "id": "a" * 32, "name": "小鼠 1", "animal_id": None,
            "window": None,
            "geometry": {"arena": {"type": "quad",
                                   "points": [[10, 10], [180, 10],
                                              [180, 180], [10, 180]]}},
            "options": {}, "detection": {}, "manual": None,
        }],
    }])
    monkeypatch.setattr(
        preview.media, "read_frame_info",
        lambda *args, **kwargs: FrameInfo(
            image=np.zeros((200, 200, 3), dtype=np.uint8), index=187,
            time_s=14.96, pts=None, time_base=None,
            clock_source="fps_override"))

    import behaviourlab.media as media_module

    real_entry = media_module.read_frame_at_time

    def legacy_entry(entry_path, time_s, orientation=0, fps_override=None, **kw):
        return legacy_scan(entry_path, float(time_s), orientation,
                           fps_override, PreviewFactory(spec))

    def production_entry(entry_path, time_s, orientation=0, fps_override=None,
                         **kw):
        # 生产入口本身不变，只把受控解码器换进来。
        return real_entry(entry_path, time_s, orientation, fps_override,
                          decoder_factory=PreviewFactory(spec))

    body = {"asset_id": asset["id"], "unit_id": "a" * 32,
            "time_s": 14.96, "kind": "detection"}

    monkeypatch.setattr(preview.media, "read_frame_at_time", legacy_entry)
    before = client.post(f"/api/drafts/{draft['id']}/preview", json=body)
    assert before.status_code == 201, before.get_json()

    monkeypatch.setattr(preview.media, "read_frame_at_time", production_entry)
    after = client.post(f"/api/drafts/{draft['id']}/preview", json=body)
    assert after.status_code == 201, after.get_json()

    first = before.get_json()["data"]
    second = after.get_json()["data"]
    assert first["frame_index"] == second["frame_index"] == 187
    assert first["time_s"] == second["time_s"] == pytest.approx(14.96)
    assert first["clock_source"] == second["clock_source"] == "fps_override"
    assert first["preview_scope"] == second["preview_scope"]
    assert first["summary"]["detection"] == second["summary"]["detection"]
    assert first["summary"]["drawn_steps"] == second["summary"]["drawn_steps"]
    assert first["warnings"] == second["warnings"]
