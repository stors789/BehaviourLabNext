"""源视频探测、真实帧访问、解释时钟与有界读取缓存。

像素访问与定量时钟是两条边界：read_frame 只要求帧可解码；带时钟的
read_frame_info / read_frame_at_time / VideoSource 要求源时钟可解释。
"""

from __future__ import annotations

import atexit
import math
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from .schema import Frame, ValidationError, is_number

VIDEO_EXTENSIONS = (
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".m4v",
    ".wmv",
    ".mpg",
    ".mpeg",
    ".mts",
    ".m2ts",
    ".webm",
    ".flv",
)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")

ORIENTATIONS = (0, 90, 180, 270)
IMAGE_CACHE_COUNT = 8
IMAGE_CACHE_BYTES = 16 * 1024 * 1024
MAX_READERS = 4
INFO_CACHE_COUNT = 64
ANCHOR_LIMIT = 2048
SEEK_DECODE_LIMIT = 4096
TIME_EPS = 1e-9


class MediaError(RuntimeError):
    """无法读取指定媒体。"""


class FrameReadError(MediaError):
    """指定源帧不可读。"""


class ClockError(MediaError):
    """源时钟不可解释。"""


class VideoSourceError(MediaError):
    """视频窗口或覆盖不满足要求。"""


def has_video_extension(name: str) -> bool:
    return Path(name).suffix.lower() in VIDEO_EXTENSIONS


def has_image_extension(name: str) -> bool:
    return Path(name).suffix.lower() in IMAGE_EXTENSIONS


def video_dialog_filter() -> str:
    return "视频文件 (" + ";".join(f"*{suffix}" for suffix in VIDEO_EXTENSIONS) + ")"


def image_dialog_filter() -> str:
    return "图片 (" + ";".join(f"*{suffix}" for suffix in IMAGE_EXTENSIONS) + ")"


def validate_index(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidationError("帧索引必须是非负整数", "index")
    return value


def validate_orientation(value: Any = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in ORIENTATIONS:
        raise ValidationError("方向必须是 0/90/180/270", "orientation")
    return value


def validate_fps_override(value: Any) -> float | None:
    if value is None:
        return None
    if not is_number(value):
        raise ValidationError("解释帧率必须是有限正数", "fps_override")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValidationError("解释帧率必须是有限正数", "fps_override")
    return number


def validate_time(value: Any) -> float:
    if not is_number(value):
        raise ValidationError("时间必须是有限非负数", "time_s")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValidationError("时间必须是有限非负数", "time_s")
    return number


def probe_video(path: str | Path) -> dict:
    """探测第一视频流元数据；未知字段保持 None。"""

    import av

    file_path = Path(path)
    if not file_path.exists():
        raise MediaError(f"视频文件不存在：{file_path}")

    try:
        with av.open(str(file_path)) as container:
            stream = next(
                (item for item in container.streams if item.type == "video"), None
            )
            if stream is None:
                raise MediaError("文件中没有视频流")
            context = stream.codec_context
            width = int(context.width) if context.width else None
            height = int(context.height) if context.height else None
            fps: float | None = None
            if stream.average_rate:
                fps = float(stream.average_rate)
            frame_count = int(stream.frames) if stream.frames else None
            duration_s: float | None = None
            if stream.duration is not None and stream.time_base is not None:
                duration_s = float(stream.duration * stream.time_base)
            elif container.duration:
                duration_s = float(container.duration) / 1_000_000.0
            codec = context.name or None
    except MediaError:
        raise
    except Exception as error:
        raise MediaError(f"无法读取视频：{error}") from error

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_s": duration_s,
        "codec": codec,
    }


@dataclass(slots=True)
class RawFrame:
    index: int
    image: Any | None
    pts: int | None
    time_base: Any | None
    keyframe: bool = False
    duration_s: float | None = None


@dataclass(slots=True)
class ClockMeta:
    time_s: float | None
    pts: int | None
    time_base: Any | None
    duration_s: float | None
    clock_source: str


@dataclass(slots=True)
class FrameInfo:
    image: Any | None
    index: int
    time_s: float
    pts: int | None
    time_base: Any | None
    clock_source: str

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def as_dict(self) -> dict:
        return {
            "image": self.image,
            "index": self.index,
            "time_s": self.time_s,
            "pts": self.pts,
            "time_base": self.time_base,
            "clock_source": self.clock_source,
        }


def file_identity(path: str | Path) -> dict:
    stat = os.stat(path)
    ctime_ns = getattr(stat, "st_ctime_ns", None)
    if ctime_ns is None:
        ctime_ns = int(stat.st_ctime * 1_000_000_000)
    return {
        "device": int(getattr(stat, "st_dev", 0)),
        "inode": int(getattr(stat, "st_ino", 0)),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(ctime_ns),
    }


def _orient(image: Any, orientation: int) -> Any:
    import cv2

    if orientation == 0:
        return image.copy()
    if orientation == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if orientation == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)


class AVDecoder:
    """PyAV 顺序解码器；seek 只在关键帧锚点恢复时使用。"""

    def __init__(self, path: str, decode_images: bool = True) -> None:
        import av

        self.path = str(path)
        self.decode_images = decode_images
        self.container = av.open(self.path)
        self.stream = next(
            (item for item in self.container.streams if item.type == "video"), None
        )
        if self.stream is None:
            self.container.close()
            raise MediaError("文件中没有视频流")
        self._iterator = self.container.decode(self.stream)

    def frames(self) -> Iterator[RawFrame]:
        return self._convert(self._iterator)

    def _convert(self, iterator: Iterator[Any]) -> Iterator[RawFrame]:
        for frame in iterator:
            time_base = frame.time_base or self.stream.time_base
            duration_s: float | None = None
            if frame.duration is not None and time_base is not None:
                duration_s = float(frame.duration * time_base)
            image = None
            if self.decode_images:
                image = frame.to_ndarray(format="bgr24")
            yield RawFrame(
                index=-1,
                image=image,
                pts=frame.pts,
                time_base=time_base,
                keyframe=bool(frame.key_frame),
                duration_s=duration_s,
            )

    def seek(self, pts: int, keyframe: bool = True) -> None:
        self.container.seek(int(pts), stream=self.stream, backward=True, any_frame=False)
        self._iterator = self.container.decode(self.stream)

    def close(self) -> None:
        try:
            self.container.close()
        finally:
            self._iterator = iter(())


class FrameReader:
    """顺序计数读取器；稀疏锚点只用于向后恢复，缓存有界。"""

    def __init__(
        self,
        path: str | Path,
        *,
        decoder_factory: Callable[..., Any] | None = None,
        fps_override: float | None = None,
        decode_images: bool = True,
    ) -> None:
        self.path = str(Path(path).resolve())
        try:
            self.identity = file_identity(self.path)
        except OSError as error:
            raise MediaError(f"视频文件不存在：{self.path}") from error
        self.fps_override = validate_fps_override(fps_override)
        self.clock_source = "fps_override" if self.fps_override is not None else "pts"
        self.decode_images = decode_images
        self._factory = decoder_factory or AVDecoder
        self._lock = threading.RLock()
        self._decoder: Any = None
        self._iterator: Iterator[RawFrame] = iter(())
        self._decoded = 0
        self._last_raw: RawFrame | None = None
        self._first_abs: float | None = None
        self._previous_abs: float | None = None
        self._clock_error_index: int | None = None
        self._clock_dirty = False
        self._anchors: list[tuple[int, int]] = []
        self._info_cache: OrderedDict[int, ClockMeta] = OrderedDict()
        self._image_cache: OrderedDict[int, Any] = OrderedDict()
        self._image_bytes = 0
        self._open()

    def _open(self) -> None:
        try:
            try:
                self._decoder = self._factory(
                    self.path, decode_images=self.decode_images
                )
            except TypeError:
                self._decoder = self._factory(self.path)
            self._iterator = iter(self._decoder.frames())
        except MediaError:
            raise
        except Exception as error:
            raise MediaError(f"无法打开视频：{error}") from error
        self._decoded = 0
        self._last_raw = None
        self._reset_clock()

    def _reset_clock(self) -> None:
        self._first_abs = None
        self._previous_abs = None
        self._clock_error_index = None
        self._clock_dirty = False
        self._info_cache.clear()

    def rewind(self) -> None:
        with self._lock:
            self.close()
            self._open()

    def close(self) -> None:
        if self._decoder is not None:
            try:
                self._decoder.close()
            finally:
                self._decoder = None
                self._iterator = iter(())
        self._image_cache.clear()
        self._image_bytes = 0

    def _record_clock(self, raw: RawFrame) -> None:
        index = raw.index
        if self._clock_dirty:
            meta = ClockMeta(None, raw.pts, raw.time_base, raw.duration_s,
                             self.clock_source)
            self._info_cache[index] = meta
            self._trim_info()
            return

        if self.clock_source == "fps_override":
            meta = ClockMeta(
                index / float(self.fps_override),
                raw.pts,
                raw.time_base,
                1.0 / float(self.fps_override),
                self.clock_source,
            )
        elif raw.pts is None or raw.time_base is None:
            if self._clock_error_index is None:
                self._clock_error_index = index
            meta = ClockMeta(None, raw.pts, raw.time_base, raw.duration_s,
                             self.clock_source)
        else:
            absolute = float(raw.pts * raw.time_base)
            if self._clock_error_index is not None:
                time_s = None
            elif index == 0:
                self._first_abs = absolute
                time_s = 0.0
            else:
                time_s = absolute - (self._first_abs or 0.0)
                if self._previous_abs is not None and absolute <= self._previous_abs:
                    if self._clock_error_index is None:
                        self._clock_error_index = index
                    time_s = None
            if time_s is not None and self._clock_error_index is None:
                self._previous_abs = absolute
            meta = ClockMeta(time_s, raw.pts, raw.time_base, raw.duration_s,
                             self.clock_source)
        self._info_cache[index] = meta
        self._trim_info()

    def _trim_info(self) -> None:
        while len(self._info_cache) > INFO_CACHE_COUNT:
            self._info_cache.popitem(last=False)

    def _add_anchor(self, raw: RawFrame) -> None:
        if not raw.keyframe or raw.pts is None or self._clock_error_index is not None:
            return
        if self._anchors and self._anchors[-1][0] >= raw.index:
            return
        self._anchors.append((raw.index, int(raw.pts)))
        if len(self._anchors) > ANCHOR_LIMIT:
            old = self._anchors
            thinned = old[::2]
            if thinned[-1] != old[-1]:
                thinned.append(old[-1])
            self._anchors = thinned

    def _next_raw(self) -> RawFrame:
        try:
            raw = next(self._iterator)
        except StopIteration:
            raise
        raw.index = self._decoded
        self._decoded += 1
        self._last_raw = raw
        self._record_clock(raw)
        self._add_anchor(raw)
        return raw

    def _store_image(self, index: int, image: Any) -> None:
        if image is None:
            return
        size = int(getattr(image, "nbytes", 0))
        if size > IMAGE_CACHE_BYTES:
            return
        existing = self._image_cache.pop(index, None)
        if existing is not None:
            self._image_bytes -= int(getattr(existing, "nbytes", 0))
        self._image_cache[index] = image
        self._image_bytes += size
        while self._image_cache and (
            len(self._image_cache) > IMAGE_CACHE_COUNT
            or self._image_bytes > IMAGE_CACHE_BYTES
        ):
            _, removed = self._image_cache.popitem(last=False)
            self._image_bytes -= int(getattr(removed, "nbytes", 0))

    def _image_at(self, raw: RawFrame) -> Any:
        cached = self._image_cache.get(raw.index)
        if cached is not None:
            return cached
        if raw.image is None:
            raise FrameReadError("该帧没有可用像素")
        self._store_image(raw.index, raw.image)
        return raw.image

    def _raw_at(self, index: int) -> RawFrame:
        if index == self._decoded - 1 and self._last_raw is not None:
            return self._last_raw
        cached = self._image_cache.get(index)
        if cached is not None:
            meta = self._info_cache.get(index)
            return RawFrame(
                index,
                cached,
                meta.pts if meta else None,
                meta.time_base if meta else None,
                False,
                meta.duration_s if meta else None,
            )
        if index < self._decoded:
            self._backward_to(index)
        while self._decoded <= index:
            self._next_raw()
        if self._last_raw is None or self._last_raw.index != index:
            raise FrameReadError("源帧不可读")
        return self._last_raw

    def _select_anchor(self, index: int) -> tuple[int, int] | None:
        if self._clock_error_index is not None:
            return None
        candidate = None
        for anchor in self._anchors:
            if anchor[0] <= index:
                candidate = anchor
            else:
                break
        return candidate

    def _try_seek(self, anchor: tuple[int, int]) -> bool:
        anchor_index, anchor_pts = anchor
        try:
            self._decoder.seek(anchor_pts, keyframe=True)
        except Exception:
            return False
        iterator = iter(self._decoder.frames())
        for _ in range(SEEK_DECODE_LIMIT):
            try:
                raw = next(iterator)
            except StopIteration:
                break
            except Exception:
                return False
            if raw.pts is None:
                return False
            if raw.pts > anchor_pts:
                # seek 落在锚点之后，无法恢复原索引计数。
                return False
            if raw.pts == anchor_pts:
                if not raw.keyframe:
                    return False
                raw.index = anchor_index
                self._iterator = iterator
                self._decoded = anchor_index + 1
                self._last_raw = raw
                self._clock_dirty = True
                return True
        return False

    def _backward_to(self, index: int) -> None:
        anchor = self._select_anchor(index)
        if anchor is not None and self._try_seek(anchor):
            return
        self.rewind()

    def frame(self, index: int, orientation: int = 0) -> Any:
        validate_index(index)
        validate_orientation(orientation)
        with self._lock:
            raw = self._raw_at(index)
            image = self._image_at(raw)
            return _orient(image, orientation)

    def _ensure_sequential(self, index: int) -> RawFrame:
        needs_rewind = (
            self._clock_dirty
            or index < self._decoded
            and (index not in self._info_cache or index not in self._image_cache)
        )
        if needs_rewind:
            self.rewind()
        while self._decoded <= index:
            try:
                self._next_raw()
            except StopIteration:
                raise FrameReadError("指定帧超出可解码范围")
        if self._clock_error_index is not None and index >= self._clock_error_index:
            raise ClockError(
                f"源时钟在第 {self._clock_error_index} 帧起不可解释，"
                "请设置显式解释帧率"
            )
        if self._last_raw is None or self._last_raw.index != index:
            raise FrameReadError("源帧不可读")
        return self._last_raw

    def info(self, index: int, orientation: int = 0) -> FrameInfo:
        validate_index(index)
        validate_orientation(orientation)
        with self._lock:
            if (
                index in self._info_cache
                and index in self._image_cache
                and not self._clock_dirty
                and index < self._decoded
            ):
                meta = self._info_cache[index]
                raw = RawFrame(index, self._image_cache[index], meta.pts,
                               meta.time_base, False, meta.duration_s)
            else:
                raw = self._ensure_sequential(index)
                meta = self._info_cache.get(index)
            if meta is None or meta.time_s is None:
                raise ClockError(
                    f"源时钟在第 {self._clock_error_index} 帧起不可解释，"
                    "请设置显式解释帧率"
                )
            image = self._image_at(raw)
            return FrameInfo(
                image=_orient(image, orientation),
                index=index,
                time_s=float(meta.time_s),
                pts=meta.pts,
                time_base=meta.time_base,
                clock_source=meta.clock_source,
            )

    def iter_timed(self) -> Iterator[tuple[RawFrame, ClockMeta]]:
        with self._lock:
            self.rewind()
            while True:
                try:
                    raw = self._next_raw()
                except StopIteration:
                    return
                meta = self._info_cache.get(raw.index)
                if meta is None or meta.time_s is None:
                    raise ClockError(
                        f"源时钟在第 {raw.index} 帧起不可解释，"
                        "请设置显式解释帧率"
                    )
                yield raw, meta

    def decoded_count(self) -> int:
        return self._decoded

    def image_cache_size(self) -> int:
        return len(self._image_cache)


class ReaderPool:
    """最多四个读取器的 LRU；路径身份变化时强制重建。"""

    def __init__(
        self,
        *,
        decoder_factory: Callable[..., Any] | None = None,
        max_readers: int = MAX_READERS,
    ) -> None:
        self._factory = decoder_factory
        self._max_readers = max_readers
        self._readers: OrderedDict[tuple, FrameReader] = OrderedDict()
        self._lock = threading.RLock()

    def get(
        self,
        path: str | Path,
        *,
        fps_override: float | None = None,
        decode_images: bool = True,
    ) -> FrameReader:
        resolved = str(Path(path).resolve())
        try:
            identity = file_identity(resolved)
        except OSError as error:
            raise MediaError(f"视频文件不存在：{resolved}") from error
        mode = "fps" if fps_override is not None else "pts"
        key = (resolved, mode, float(fps_override) if fps_override else None,
               bool(decode_images))
        with self._lock:
            existing = self._readers.get(key)
            if existing is not None and existing.identity == identity:
                self._readers.move_to_end(key)
                return existing
            if existing is not None:
                existing.close()
                self._readers.pop(key, None)
            reader = FrameReader(
                resolved,
                decoder_factory=self._factory,
                fps_override=fps_override,
                decode_images=decode_images,
            )
            self._readers[key] = reader
            while len(self._readers) > self._max_readers:
                _, old = self._readers.popitem(last=False)
                old.close()
            return reader

    def close_all(self) -> None:
        with self._lock:
            for reader in self._readers.values():
                reader.close()
            self._readers.clear()


_DEFAULT_POOL = ReaderPool()
_FACTORY_POOLS: dict[Any, ReaderPool] = {}
_FACTORY_POOLS_LOCK = threading.Lock()
atexit.register(_DEFAULT_POOL.close_all)


def _reader_for(
    path: str | Path,
    *,
    pool: ReaderPool | None,
    decoder_factory: Callable[..., Any] | None,
    fps_override: float | None = None,
    decode_images: bool = True,
) -> FrameReader:
    if pool is not None:
        return pool.get(path, fps_override=fps_override,
                        decode_images=decode_images)
    if decoder_factory is None:
        return _DEFAULT_POOL.get(path, fps_override=fps_override,
                                 decode_images=decode_images)
    with _FACTORY_POOLS_LOCK:
        factory_pool = _FACTORY_POOLS.get(decoder_factory)
        if factory_pool is None:
            factory_pool = ReaderPool(decoder_factory=decoder_factory,
                                      max_readers=1)
            _FACTORY_POOLS[decoder_factory] = factory_pool
    return factory_pool.get(path, fps_override=fps_override,
                            decode_images=decode_images)


def read_frame(
    path: str | Path,
    index: int,
    orientation: int = 0,
    *,
    pool: ReaderPool | None = None,
    decoder_factory: Callable[..., Any] | None = None,
) -> Any:
    """只读指定真实源帧像素；不要求源时钟合法。"""

    validate_index(index)
    validate_orientation(orientation)
    reader = _reader_for(path, pool=pool, decoder_factory=decoder_factory)
    return reader.frame(index, orientation)


def read_frame_info(
    path: str | Path,
    index: int,
    orientation: int = 0,
    fps_override: float | None = None,
    *,
    pool: ReaderPool | None = None,
    decoder_factory: Callable[..., Any] | None = None,
) -> FrameInfo:
    validate_index(index)
    validate_orientation(orientation)
    validate_fps_override(fps_override)
    reader = _reader_for(
        path, pool=pool, decoder_factory=decoder_factory,
        fps_override=fps_override,
    )
    return reader.info(index, orientation)


def read_frame_at_time(
    path: str | Path,
    time_s: float,
    orientation: int = 0,
    fps_override: float | None = None,
    *,
    pool: ReaderPool | None = None,
    decoder_factory: Callable[..., Any] | None = None,
) -> FrameInfo:
    """返回 time_s≤t 的最后一帧；不把 t 回填为源时间。"""

    target = validate_time(time_s)
    validate_orientation(orientation)
    validate_fps_override(fps_override)
    reader = _reader_for(
        path, pool=pool, decoder_factory=decoder_factory,
        fps_override=fps_override,
    )
    with reader._lock:
        reader.rewind()
        best: tuple[RawFrame, ClockMeta] | None = None
        while True:
            try:
                raw = reader._next_raw()
            except StopIteration:
                break
            meta = reader._info_cache.get(raw.index)
            if meta is None or meta.time_s is None:
                raise ClockError(
                    f"源时钟在第 {raw.index} 帧起不可解释，请设置显式解释帧率"
                )
            if meta.time_s <= target + TIME_EPS:
                best = (raw, meta)
            elif best is not None:
                break
        if best is None:
            raise FrameReadError("时间点之前没有可解码帧")
        raw, meta = best
        image = reader._image_at(raw)
        return FrameInfo(
            image=_orient(image, orientation),
            index=raw.index,
            time_s=float(meta.time_s),
            pts=meta.pts,
            time_base=meta.time_base,
            clock_source=meta.clock_source,
        )


class VideoSource:
    """按窗口与采样率遍历真实源帧，并计算有界覆盖。"""

    def __init__(
        self,
        path: str | Path,
        *,
        start_s: float = 0.0,
        end_s: float | None = None,
        orientation: int = 0,
        fps_override: float | None = None,
        sample_fps: float | None = None,
        source_fps: float | None = None,
        metadata: dict | None = None,
        decode_images: bool = True,
        pool: ReaderPool | None = None,
        decoder_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.path = str(Path(path).resolve())
        self.start_s = self._validate_start(start_s)
        self.end_s = self._validate_end(end_s, self.start_s)
        self.orientation = validate_orientation(orientation)
        self.fps_override = validate_fps_override(fps_override)
        self.sample_fps = self._validate_sample_fps(sample_fps)
        self.decode_images = bool(decode_images)

        resolved_source = self.fps_override
        if resolved_source is None:
            if source_fps is not None:
                resolved_source = source_fps
            elif metadata is not None:
                resolved_source = metadata.get("fps")
            else:
                metadata = probe_video(self.path)
                resolved_source = metadata.get("fps")
        if not is_number(resolved_source) or float(resolved_source) <= 0:
            raise VideoSourceError("源帧率不可用或不为正数")
        self.source_fps = float(resolved_source)
        self.analysis_fps = (
            self.source_fps
            if self.sample_fps is None or self.sample_fps >= self.source_fps
            else self.sample_fps
        )

        self._pool = pool or ReaderPool(decoder_factory=decoder_factory,
                                        max_readers=1)
        self.coverage: dict = {
            "start_s": None,
            "end_s": None,
            "duration_s": 0.0,
            "frame_count": 0,
        }

    @staticmethod
    def _validate_start(value: Any) -> float:
        if not is_number(value):
            raise VideoSourceError("窗口起点必须是有限非负数")
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise VideoSourceError("窗口起点必须是有限非负数")
        return number

    @staticmethod
    def _validate_end(value: Any, start: float) -> float | None:
        if value is None:
            return None
        if not is_number(value):
            raise VideoSourceError("窗口终点必须是有限数值或 null")
        number = float(value)
        if not math.isfinite(number) or number <= start:
            raise VideoSourceError("窗口终点必须大于起点")
        return number

    @staticmethod
    def _validate_sample_fps(value: Any) -> float | None:
        if value is None:
            return None
        if not is_number(value):
            raise VideoSourceError("采样率必须是有限正数")
        number = float(value)
        if not math.isfinite(number) or number <= 0:
            raise VideoSourceError("采样率必须是有限正数")
        return number

    def _eof_boundary(
        self, last: ClockMeta, previous_time: float | None
    ) -> float:
        last_time = float(last.time_s or 0.0)
        if self.fps_override is not None:
            return last_time + 1.0 / self.source_fps
        if last.duration_s is not None and last.duration_s > 0:
            return last_time + last.duration_s
        if previous_time is not None and last_time > previous_time:
            return last_time + (last_time - previous_time)
        return last_time + 1.0 / self.source_fps

    def _frame(
        self,
        raw: RawFrame,
        meta: ClockMeta,
        next_time: float,
        *,
        context_after: bool = False,
    ) -> Frame:
        time_s = float(meta.time_s or 0.0)
        dt_s = 0.0 if context_after else max(0.0, next_time - time_s)
        window_end = self.end_s if self.end_s is not None else next_time
        interval_start = min(max(time_s, self.start_s), window_end)
        interval_end = max(interval_start, min(next_time, window_end))
        image = None
        if self.decode_images and raw.image is not None:
            image = _orient(raw.image, self.orientation)
        frame = Frame(
            image=image,
            index=raw.index,
            time_s=time_s,
            dt_s=dt_s,
            elapsed_s=time_s,
            analysis_fps=self.analysis_fps,
            interval_start_s=interval_start,
            interval_end_s=interval_end,
        )
        if interval_end > interval_start:
            self.coverage["duration_s"] += interval_end - interval_start
            self.coverage["frame_count"] += 1
            if self.coverage["start_s"] is None:
                self.coverage["start_s"] = interval_start
            self.coverage["end_s"] = interval_end
        return frame

    def _check_coverage(self) -> None:
        if self.coverage["frame_count"] == 0:
            raise VideoSourceError("窗口内没有可解码的有效曝光")
        start = float(self.coverage["start_s"])
        end = float(self.coverage["end_s"])
        if start > self.start_s + TIME_EPS:
            raise VideoSourceError("实际覆盖起点晚于请求")
        if self.end_s is not None:
            if end < self.end_s - TIME_EPS:
                raise VideoSourceError("实际覆盖终点早于请求")
            expected = self.end_s - self.start_s
            if self.coverage["duration_s"] < expected - 1e-6:
                raise VideoSourceError("窗口有效覆盖不足")

    def iter_frames(self) -> Iterator[Frame]:
        self.coverage = {
            "start_s": None,
            "end_s": None,
            "duration_s": 0.0,
            "frame_count": 0,
        }
        keep_all = (
            self.sample_fps is None or self.sample_fps >= self.source_fps
        )
        next_target = 0.0
        pending: tuple[RawFrame, ClockMeta] | None = None
        phase = "before"
        last_meta: ClockMeta | None = None
        previous_time: float | None = None
        eof_boundary: float | None = None

        reader = self._pool.get(
            self.path,
            fps_override=self.fps_override,
            decode_images=self.decode_images,
        )
        for raw, meta in reader.iter_timed():
            time_s = float(meta.time_s or 0.0)
            previous_time = last_meta.time_s if last_meta else previous_time
            last_meta = meta

            selected = keep_all or time_s + TIME_EPS >= next_target
            if selected and not keep_all:
                next_target = (
                    math.floor((time_s + TIME_EPS) * self.analysis_fps) + 1
                ) / self.analysis_fps

            if not selected:
                continue

            if phase == "before":
                if time_s < self.start_s:
                    pending = (raw, meta)
                    continue
                if pending is not None:
                    yield self._frame(pending[0], pending[1], time_s)
                    pending = None
                pending = (raw, meta)
                phase = "inside" if self.end_s is None or time_s < self.end_s else "after"
                if phase == "after":
                    yield self._frame(raw, meta, time_s, context_after=True)
                    break
                continue

            if phase == "inside":
                if self.end_s is not None and time_s >= self.end_s:
                    if pending is not None:
                        yield self._frame(pending[0], pending[1], time_s)
                        pending = None
                    yield self._frame(raw, meta, time_s, context_after=True)
                    break
                if pending is not None:
                    yield self._frame(pending[0], pending[1], time_s)
                pending = (raw, meta)
                continue

            # phase == after 已在上面立即结束
            break

        if last_meta is None:
            raise VideoSourceError("没有可解码的源帧")

        eof_boundary = self._eof_boundary(last_meta, previous_time)
        if pending is not None:
            yield self._frame(pending[0], pending[1], eof_boundary)

        self._check_coverage()
