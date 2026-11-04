"""源视频探测、真实帧访问、解释时钟与有界读取缓存。

像素访问与定量时钟是两条边界：read_frame 只要求帧可解码；带时钟的
read_frame_info / read_frame_at_time / VideoSource 要求源时钟可解释。
"""

from __future__ import annotations

import atexit
import errno
import math
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

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
# 连续这么多次 seek 未获证实就停用 seek：后续只顺序读取，避免每个目标重扫。
SEEK_FAILURE_LIMIT = 2
TIME_EPS = 1e-9

# 取不到有效源帧率、也没有合法覆盖时的名义源帧率。它只参与采样与
# analysis_fps 的名义换算，不接管源时钟：可靠 PTS 仍按真实 PTS 解释。
DEFAULT_SOURCE_FPS = 20.0
DEFAULT_SOURCE_FPS_NOTE = "源FPS不可用，使用默认 20"


class MediaError(RuntimeError):
    """无法读取指定媒体。"""


class FrameReadError(MediaError):
    """指定源帧不可读。"""


class ClockError(MediaError):
    """源时钟不可解释。"""


class VideoSourceError(MediaError):
    """视频窗口或覆盖不满足要求。"""


ACCESS_DENIED_ERRNOS = (errno.EACCES, errno.EPERM)


def is_access_denied(error: BaseException) -> bool:
    """按异常类型/errno 判定"访问被拒绝"（含 PyAV 的 PermissionError）。

    访问拒绝不是内容/解码问题：换一份文件不会有帮助，调用方必须保留
    失败语义。沿 ``__cause__``/``__context__`` 链检查，只看类型与 errno，
    不匹配提示文字。
    """

    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, PermissionError):
            return True
        errno_value = getattr(current, "errno", None)
        if isinstance(errno_value, int) and errno_value in ACCESS_DENIED_ERRNOS:
            return True
        current = current.__cause__ or current.__context__
    return False


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


def nominal_source_fps(value: Any, *, fps_override: float | None = None
                       ) -> tuple[float, str]:
    """名义源帧率与来源：合法覆盖 > 有效原生帧率 > 默认 20。

    返回 (帧率, 来源)，来源为 ``override`` / ``media`` / ``default``。
    默认值只补名义帧率，不改变源时钟来源（仍为真实 PTS）。
    """

    if fps_override is not None:
        return float(fps_override), "override"
    if is_number(value):
        number = float(value)
        if math.isfinite(number) and number > 0:
            return number, "media"
    return DEFAULT_SOURCE_FPS, "default"


def source_fps_note(source: str | None) -> str | None:
    """默认源帧率的可读说明；真实帧率与显式覆盖不给说明。"""

    return DEFAULT_SOURCE_FPS_NOTE if source == "default" else None


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
    # 延迟物化：只有真正要像素时才 to_ndarray（见 defer_images）。
    pending: Any | None = None


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


@dataclass(slots=True)
class IndexEvidence:
    """一遍扫描留下的有界索引证据：可信锚点 + 首帧绝对 PTS。

    只记录已验证的（索引, PTS, 时间）。证据绑定**产生它的源文件身份**与
    解释时钟参数：换文件、换解释帧率或换时钟模式都不得复用——索引与时间的
    对应关系只对那一份源成立。
    """

    clock_source: str
    first_abs: float | None
    anchors: tuple[tuple[int, int, float | None], ...]
    identity: tuple = ()
    fps_override: float | None = None

    def __bool__(self) -> bool:
        return bool(self.anchors)


def identity_key(identity: Mapping[str, Any] | None) -> tuple:
    """源文件身份的稳定比较键；取不到身份时返回空元组。"""

    if not isinstance(identity, Mapping):
        return ()
    return (int(identity.get("device") or 0), int(identity.get("inode") or 0),
            int(identity.get("size") or 0), int(identity.get("mtime_ns") or 0),
            int(identity.get("ctime_ns") or 0))


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


def _materialize(pending: Any) -> Any:
    """把延迟物化的解码帧转成 BGR；只在真正需要像素时调用。"""

    return pending.to_ndarray(format="bgr24")


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

    def __init__(self, path: str, decode_images: bool = True,
                 defer_images: bool = False) -> None:
        import av

        self.path = str(path)
        self.decode_images = decode_images
        self.defer_images = bool(defer_images)
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
            pending = None
            if self.decode_images:
                if self.defer_images:
                    # 先保留解码帧，等确定要这一帧的像素时再转 BGR。
                    pending = frame
                else:
                    image = frame.to_ndarray(format="bgr24")
            yield RawFrame(
                index=-1,
                image=image,
                pts=frame.pts,
                time_base=time_base,
                keyframe=bool(frame.key_frame),
                duration_s=duration_s,
                pending=pending,
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
        defer_images: bool = False,
    ) -> None:
        self.path = str(Path(path).resolve())
        try:
            self.identity = file_identity(self.path)
        except OSError as error:
            raise MediaError(f"视频文件不存在：{self.path}") from error
        self.fps_override = validate_fps_override(fps_override)
        self.clock_source = "fps_override" if self.fps_override is not None else "pts"
        self.decode_images = decode_images
        self.defer_images = bool(defer_images) and decode_images
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
        self._seek_failures = 0
        self._seek_disabled = False
        self._anchors: list[tuple[int, int]] = []
        self._info_cache: OrderedDict[int, ClockMeta] = OrderedDict()
        self._image_cache: OrderedDict[int, Any] = OrderedDict()
        self._image_bytes = 0
        self._open()

    def _open(self) -> None:
        try:
            try:
                self._decoder = self._factory(
                    self.path, decode_images=self.decode_images,
                    defer_images=self.defer_images,
                )
            except TypeError:
                try:
                    self._decoder = self._factory(
                        self.path, decode_images=self.decode_images)
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
        # 锚点连同这一遍已验证的索引时间一起保存：缓存淘汰后可直接用它
        # 恢复，不必从第 0 帧重扫；时间无法证实（坏时钟）时存 None。
        meta = self._info_cache.get(raw.index)
        self._anchors.append(
            (raw.index, int(raw.pts), meta.time_s if meta is not None else None))
        if len(self._anchors) > ANCHOR_LIMIT:
            old = self._anchors
            thinned = old[::2]
            if thinned[-1] != old[-1]:
                thinned.append(old[-1])
            self._anchors = thinned

    def _next_raw(self) -> RawFrame:
        self._ensure_open()
        try:
            raw = next(self._iterator)
        except StopIteration:
            raise
        raw.index = self._decoded
        self._decoded += 1
        self._last_raw = raw
        self._record_clock(raw)
        self._add_anchor(raw)
        # 顺序解码顺带进有界图像缓存：按索引回读相邻帧不必重解同一段。
        self._store_image(raw.index, raw.image)
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
        if raw.image is None and raw.pending is not None:
            # 延迟物化：到这里才把这一帧转成 BGR。
            raw.image = _materialize(raw.pending)
            raw.pending = None
        if raw.image is None:
            raise FrameReadError("该帧没有可用像素")
        self._store_image(raw.index, raw.image)
        return raw.image

    def _raw_at(self, index: int) -> RawFrame:
        self._ensure_open()
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

    def _select_anchor(self, index: int) -> tuple[int, int, float | None] | None:
        if self._clock_error_index is not None:
            return None
        candidate = None
        for anchor in self._anchors:
            if anchor[0] <= index:
                candidate = anchor
            else:
                break
        return candidate

    def _seek_anchor_time(self, raw: RawFrame, stored: float | None) -> float | None:
        """seek 落到锚点后，能不能证明这一帧的时间；不能就返回 None。"""

        if self.clock_source == "fps_override":
            # 解释帧率下时间就是 index/fps；索引已由 seek 的 PTS 核对证明。
            return raw.index / float(self.fps_override)
        if stored is None or raw.pts is None or raw.time_base is None \
                or self._first_abs is None:
            return None
        current = float(raw.pts * raw.time_base) - float(self._first_abs)
        # 锚点记录的时间必须与本次基准换算吻合，否则不当已验证。
        if abs(current - stored) > 1e-6 * max(1.0, abs(stored)):
            return None
        return float(stored)

    def _adopt_anchor_clock(self, raw: RawFrame, time_s: float) -> None:
        """把已验证锚点作为新的时钟起点，后续顺序帧按同一规则续算。"""

        meta = ClockMeta(float(time_s), raw.pts, raw.time_base, raw.duration_s,
                         self.clock_source)
        self._info_cache[raw.index] = meta
        self._trim_info()
        if self.clock_source == "pts" and raw.pts is not None \
                and raw.time_base is not None:
            self._previous_abs = float(raw.pts * raw.time_base)
        self._clock_dirty = False
        self._store_image(raw.index, raw.image)

    def _invalidate_position(self, *, seek_failed: bool = False) -> None:
        """丢弃“当前位置”：底层解码器可能已被 seek 移动，索引计数不再可信。

        只重置位置与时钟状态，保留按索引缓存的原帧（同一文件身份下仍有效）。
        下一次读取从头重建解码器；不重复重试、不猜测位置。连续失败的 seek
        会被停用，后续只用顺序读取走完，避免每个目标都重扫一遍。
        """

        if seek_failed:
            self._seek_failures += 1
            if self._seek_failures >= SEEK_FAILURE_LIMIT:
                self._seek_disabled = True
        if self._decoder is not None:
            try:
                self._decoder.close()
            except Exception:
                pass
            self._decoder = None
            self._iterator = iter(())
        self._decoded = 0
        self._last_raw = None
        self._first_abs = None
        self._previous_abs = None
        self._clock_dirty = False
        self._info_cache.clear()

    def _ensure_open(self) -> None:
        if self._decoder is None:
            self._open()

    def _try_seek(self, anchor: tuple[int, int, float | None]) -> bool:
        """按锚点 seek 并在 PTS 上核对归属；任何不能证实的结果都丢弃位置。"""

        anchor_index, anchor_pts = anchor[0], anchor[1]
        try:
            self._decoder.seek(anchor_pts, keyframe=True)
        except Exception:
            self._invalidate_position(seek_failed=True)
            return False
        iterator = iter(self._decoder.frames())
        for _ in range(SEEK_DECODE_LIMIT):
            try:
                raw = next(iterator)
            except StopIteration:
                break
            except Exception:
                self._invalidate_position(seek_failed=True)
                return False
            if raw.pts is None:
                self._invalidate_position(seek_failed=True)
                return False
            if raw.pts > anchor_pts:
                # seek 落在锚点之后，无法恢复原索引计数。
                self._invalidate_position(seek_failed=True)
                return False
            if raw.pts == anchor_pts:
                if not raw.keyframe:
                    self._invalidate_position(seek_failed=True)
                    return False
                self._seek_failures = 0
                raw.index = anchor_index
                self._iterator = iterator
                self._decoded = anchor_index + 1
                self._last_raw = raw
                time_s = self._seek_anchor_time(raw, anchor[2])
                if time_s is None:
                    self._clock_dirty = True
                else:
                    self._adopt_anchor_clock(raw, time_s)
                return True
        # 探测耗尽仍未落到锚点：同样不能假设位置没变。
        self._invalidate_position(seek_failed=True)
        return False

    def _backward_to(self, index: int) -> None:
        anchor = self._select_anchor(index) if not self._seek_disabled else None
        if anchor is not None and self._try_seek(anchor):
            return
        # 锚点不可用或 seek 未获证实：从头重建（一次，不循环重试）。
        if self._decoder is None:
            self._open()
        else:
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
            # 先试已验证锚点（索引/PTS 可核对）恢复；锚点证据不足或
            # seek 不可证明时回到从头读取，保证这里能给出真实时间。
            self._backward_to(index)
            if self._clock_dirty:
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

    # -- 时间定位 -------------------------------------------------------
    # 时间→真实索引→指定帧读取：只复用本读取器的索引计数、时钟元信息、
    # 缓存与关键帧锚点，不另建解码器、坐标或第二套时钟。

    def _forward_time(
        self, limit: float, best: tuple[int, ClockMeta] | None
    ) -> tuple[int, ClockMeta]:
        """顺序向前解码，返回 time_s≤limit 的最后一帧；坏时钟照旧报错。"""

        while True:
            try:
                raw = self._next_raw()
            except StopIteration:
                break
            meta = self._info_cache.get(raw.index)
            if meta is None or meta.time_s is None:
                raise ClockError(
                    f"源时钟在第 {raw.index} 帧起不可解释，"
                    "请设置显式解释帧率"
                )
            if meta.time_s <= limit:
                # 候选帧原图进有界图像缓存：定位与取像素不必再解一遍。
                self._store_image(raw.index, raw.image)
                best = (raw.index, meta)
            elif best is not None:
                break
        if best is None:
            raise FrameReadError("时间点之前没有可解码帧")
        return best

    def _cached_time_frame(self, limit: float) -> tuple[int, ClockMeta] | None:
        """已解码窗口内的证据：同一遍顺序解码的相邻两帧即可定出答案。"""

        times = {index: meta.time_s for index, meta in self._info_cache.items()
                 if meta.time_s is not None}
        for index in sorted(times):
            if times[index] <= limit:
                continue
            previous = index - 1
            if previous in times and times[previous] <= limit:
                return previous, self._info_cache[previous]
            # 窗口不连续或前一帧不可解释：这条证据不成立，退回顺序解码。
            return None
        return None

    def _anchor_time_frame(self, limit: float) -> tuple[int, ClockMeta] | None:
        """已验证锚点（索引+时间）里最后一个不越过目标的起点。

        锚点时间来自顺序解码那一遍的 ClockMeta，seek 后再按 PTS 基准复核；
        任一环节证明不了就返回 None，由调用方退回顺序读取。
        """

        best = None
        for anchor in self._anchors:
            if anchor[2] is None:
                continue
            if anchor[2] <= limit:
                best = anchor
            else:
                break
        if best is None:
            return None
        if not self._try_seek(best):
            return None
        meta = self._info_cache.get(best[0])
        if meta is None or meta.time_s is None:
            return None
        return best[0], meta

    def _pts_time_frame(self, target: float) -> int:
        """可靠 PTS 下按已验证的源时间顺序定位；不用名义帧率猜帧号。"""

        limit = target + TIME_EPS
        cached = self._cached_time_frame(limit)
        if cached is not None:
            return cached[0]
        # 当前位置已看过且还没越过目标：继续向前，不回到第 0 帧重建。
        if not self._clock_dirty and self._last_raw is not None \
                and self._last_raw.index == self._decoded - 1:
            meta = self._info_cache.get(self._last_raw.index)
            if meta is not None and meta.time_s is not None \
                    and meta.time_s <= limit:
                self._store_image(self._last_raw.index, self._last_raw.image)
                return self._forward_time(
                    limit, (self._last_raw.index, meta))[0]
        # 目标在已解码窗口之前：先用可信锚点做有界恢复（关键帧→目标）。
        anchored = self._anchor_time_frame(limit)
        if anchored is not None:
            return self._forward_time(limit, anchored)[0]
        # 没有可用锚点（无关键帧/seek 失败/时钟不可信）才从头顺序定位；
        # 刚打开的读取器本来就在开头，不必再重开一次。
        if self._decoded == 0 and self._last_raw is None \
                and not self._clock_dirty:
            return self._forward_time(limit, None)[0]
        self.rewind()
        return self._forward_time(limit, None)[0]

    def _fps_time_index(self, target: float) -> int:
        """显式解释帧率下源时间就是 index/fps_override，直接算真实索引。"""

        fps = float(self.fps_override)
        return max(0, int(math.floor((target + TIME_EPS) * fps)))

    def frame_at_time(self, target: float, orientation: int = 0) -> FrameInfo:
        """返回 time_s≤target（沿用 TIME_EPS）的最后一帧；不回填请求时间。"""

        target = validate_time(target)
        validate_orientation(orientation)
        with self._lock:
            index = (self._fps_time_index(target) if self.fps_override is not None
                     else self._pts_time_frame(target))
            try:
                # 按真实索引取像素与时钟：命中缓存或从最近锚点顺序恢复。
                return self.info(index, orientation)
            except FrameReadError:
                last = self._decoded - 1
                if self.fps_override is None or last < 0 or index < self._decoded:
                    raise
                # 超过真实 EOF：沿用“返回最后可解码帧”，时间仍是该帧时间。
                return self.info(last, orientation)

    def decoded_count(self) -> int:
        return self._decoded

    def image_cache_size(self) -> int:
        return len(self._image_cache)

    # -- 索引证据与按索引取像素 -----------------------------------------
    def index_evidence(self) -> IndexEvidence:
        """导出这一遍已验证的有界索引证据；不导出像素。

        证据带上产生它的源文件身份与解释时钟参数：换文件、换解释帧率或换
        时钟模式时，目标读取器必须拒绝这份证据。
        """

        return IndexEvidence(self.clock_source, self._first_abs,
                             tuple(self._anchors), identity_key(self.identity),
                             self.fps_override)

    def seed_index_evidence(self, evidence: IndexEvidence | None) -> bool:
        """复用同一文件、同一解释时钟的索引证据；不相符一律拒绝。

        只比 ``clock_source`` 不足以证明索引归属：同一路径可能已被替换成
        另一份关键帧布局不同的文件。身份、解释帧率、时钟模式三者都对上才
        采纳，否则退回顺序读取（结果仍正确，只是慢）。
        """

        if evidence is None or not evidence:
            return False
        if self._anchors or self._first_abs is not None:
            return False
        if evidence.clock_source != self.clock_source:
            return False
        if not evidence.identity or evidence.identity != identity_key(self.identity):
            return False
        if evidence.fps_override != self.fps_override:
            return False
        self._anchors = list(evidence.anchors)
        self._first_abs = evidence.first_abs
        return True

    def frame_at_index(self, index: int, orientation: int = 0) -> Any:
        """按真实索引取方向像素；向前跳跃时先用可信锚点缩短解码。

        只服务像素物化：不读时钟，也不因此接受未验证的时间。
        """

        validate_index(index)
        validate_orientation(orientation)
        with self._lock:
            cached = self._image_cache.get(index)
            if cached is not None and index < self._decoded:
                return _orient(cached, orientation)
            at_target = (self._last_raw is not None
                         and self._last_raw.index == index
                         and self._decoded == index + 1)
            if not at_target and not self._seek_disabled:
                anchor = self._select_anchor(index)
                # 只在能省解码时 seek：目标在当前位之前，或锚点在当前位之后。
                if anchor is not None and (
                        index < self._decoded - 1
                        or anchor[0] > self._decoded - 1):
                    # 返回 False 时位置已被作废，_raw_at 会从头顺序读。
                    self._try_seek(anchor)
            raw = self._raw_at(index)
            return _orient(self._image_at(raw), orientation)


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
        defer_images: bool = False,
    ) -> FrameReader:
        resolved = str(Path(path).resolve())
        try:
            identity = file_identity(resolved)
        except OSError as error:
            raise MediaError(f"视频文件不存在：{resolved}") from error
        mode = "fps" if fps_override is not None else "pts"
        deferred = bool(defer_images) and bool(decode_images)
        key = (resolved, mode, float(fps_override) if fps_override else None,
               bool(decode_images), deferred)
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
                defer_images=deferred,
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
    defer_images: bool = False,
) -> FrameReader:
    if pool is not None:
        return pool.get(path, fps_override=fps_override,
                        decode_images=decode_images,
                        defer_images=defer_images)
    if decoder_factory is None:
        return _DEFAULT_POOL.get(path, fps_override=fps_override,
                                 decode_images=decode_images,
                                 defer_images=defer_images)
    with _FACTORY_POOLS_LOCK:
        factory_pool = _FACTORY_POOLS.get(decoder_factory)
        if factory_pool is None:
            factory_pool = ReaderPool(decoder_factory=decoder_factory,
                                      max_readers=1)
            _FACTORY_POOLS[decoder_factory] = factory_pool
    return factory_pool.get(path, fps_override=fps_override,
                            decode_images=decode_images,
                            defer_images=defer_images)


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

    validate_time(time_s)
    validate_orientation(orientation)
    validate_fps_override(fps_override)
    reader = _reader_for(
        path, pool=pool, decoder_factory=decoder_factory,
        fps_override=fps_override,
    )
    # 定位在读取器内部完成：先定真实源索引，再按索引读像素与时钟。
    return reader.frame_at_time(float(time_s), orientation)


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
        # 无覆盖且取不到有效源帧率时补 20；显式覆盖仍压在原生值之上。
        self.source_fps, self.source_fps_source = nominal_source_fps(
            resolved_source, fps_override=self.fps_override)
        self.source_fps_note = source_fps_note(self.source_fps_source)
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
            # 相对视频窗起点的时间；源时刻与有效曝光保持原样。
            elapsed_s=time_s - self.start_s,
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

    def index_evidence(self) -> IndexEvidence:
        """本次窗口扫描留下的索引证据（可信锚点与首帧绝对 PTS）。"""

        reader = self._pool.get(
            self.path,
            fps_override=self.fps_override,
            decode_images=self.decode_images,
        )
        return reader.index_evidence()

    def frames_at(self, indices: Iterable[int], orientation: int | None = None,
                  evidence: IndexEvidence | None = None) -> Iterator[Any]:
        """按真实索引产出方向像素；复用同一池内的像素读取器与索引证据。

        只有确定要用的帧才会解码成像素，供两段式背景取样使用。规划与取像素
        必须属于同一份源：目标读取器的身份与证据身份不符时明确报源文件变化，
        不把旧索引/时间嫁接到新 READER 上。
        """

        angle = self.orientation if orientation is None else orientation
        planned = evidence if evidence is not None else self.index_evidence()
        reader = self._pool.get(
            self.path,
            fps_override=self.fps_override,
            decode_images=True,
            defer_images=True,
        )
        if isinstance(planned, IndexEvidence) and planned.identity \
                and planned.identity != identity_key(reader.identity):
            raise VideoSourceError(
                "源文件在背景规划与取像素之间发生变化（身份不同），"
                "已放弃本次取帧")
        if planned is not None:
            reader.seed_index_evidence(planned)
        for index in indices:
            yield reader.frame_at_index(index, angle)

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
        ended_at_window = False
        last_meta: ClockMeta | None = None
        previous_time: float | None = None

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

            # 窗口之前只保留最近一帧作前向上下文，不输出。
            if time_s < self.start_s:
                pending = (raw, meta)
                continue

            # 进入窗口：先结算上一帧的前向曝光。
            if pending is not None:
                yield self._frame(pending[0], pending[1], time_s)
                pending = None

            # 到达窗口终点：本帧只作后向上下文，随即结束，不是源视频结束。
            if self.end_s is not None and time_s >= self.end_s:
                yield self._frame(raw, meta, time_s, context_after=True)
                ended_at_window = True
                break

            pending = (raw, meta)

        if last_meta is None:
            raise VideoSourceError("没有可解码的源帧")

        if pending is not None and not ended_at_window:
            eof_boundary = self._eof_boundary(last_meta, previous_time)
            yield self._frame(pending[0], pending[1], eof_boundary)

        self._check_coverage()
