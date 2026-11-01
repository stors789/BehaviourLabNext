"""人工 Y 的源图标注视频：原始画面 + 实际源时间与人工入臂事件。

只叠加可解释的文本（源秒、帧号、已记录事件），不做位置估计，不画固定鼠点、
轨迹或热图。输出为浏览器可播的 H.264/MP4，按裁切后的真实曝光逐帧给时长
（VFR），不把首尾短曝光补成完整帧。编码失败抛 OSError，由导出层如实报错。
"""

from __future__ import annotations

import math
import os
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

import av

from . import media

ARM_KEYS = ("A", "B", "C")

# 帧时间基：微秒。标注视频的时长与每帧持续时间都按这个单位取整。
TIMEBASE = Fraction(1, 1_000_000)
TIMEBASE_LABEL = "1/1000000"


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def clock_text(seconds: Any) -> str:
    """源绝对秒 → mm:ss.sss，与界面帧标签同一写法。"""

    moment = _finite(seconds)
    if moment is None:
        return "--:--.---"
    total = max(0.0, moment)
    minutes = int(total // 60)
    return f"{minutes:02d}:{total - minutes * 60:06.3f}"


def _ascii_label(text: Any) -> str:
    """cv2 只能可靠画 ASCII；非 ASCII 名称退回通用标签。"""

    value = str(text or "")
    return value if value and value.isascii() else "unit"


def _ordered_events(events: Sequence[Mapping[str, Any]]) -> list[dict]:
    kept = []
    for item in events or ():
        if not isinstance(item, Mapping):
            continue
        if item.get("target") not in ARM_KEYS:
            continue
        moment = _finite(item.get("time_s"))
        if moment is None:
            continue
        kept.append({"time_s": moment, "target": item["target"],
                     "entry_index": item.get("entry_index")})
    kept.sort(key=lambda entry: entry["time_s"])
    return kept


def _clipped_exposure(frame: Any, start: float, end: float) -> float | None:
    """帧在分析窗内的真实正曝光时长；没有曝光信息时退回名义步长。"""

    exposure_start = _finite(getattr(frame, "interval_start_s", None))
    exposure_end = _finite(getattr(frame, "interval_end_s", None))
    if exposure_start is None or exposure_end is None:
        moment = _finite(getattr(frame, "time_s", None))
        step = _finite(getattr(frame, "dt_s", None))
        if moment is None or step is None or step <= 0:
            return None
        return step if start <= moment < end else None
    low = max(exposure_start, start)
    high = min(exposure_end, end)
    if high - low <= 1e-12:
        return None
    return high - low


def _even_size(canvas: Any) -> Any:
    """H.264 的 yuv420p 需要偶数宽高；奇数边缘补 1 像素。"""

    import cv2

    height, width = canvas.shape[:2]
    pad_h = height % 2
    pad_w = width % 2
    if not pad_h and not pad_w:
        return canvas
    return cv2.copyMakeBorder(canvas, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)


def _open_writer(target: Path, width: int, height: int):
    """打开 H.264/MP4 输出：浏览器与桌面 WebView 都能直接播放。"""

    container = av.open(
        str(target), "w", format="mp4",
        options={"movflags": "+faststart"},
    )
    stream = container.add_stream("libx264", rate=None)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.time_base = TIMEBASE
    stream.codec_context.time_base = TIMEBASE
    stream.codec_context.options = {
        "crf": "20", "preset": "veryfast", "bf": "0",
        "profile": "main", "level": "4.0",
    }
    return container, stream


def _mux(container: Any, packet: Any, queued: list[int]) -> None:
    """按帧顺序补上每帧持续时间，保证容器总时长等于曝光之和。"""

    if queued:
        packet.duration = queued.pop(0)
    container.mux(packet)


def _to_bgr(image: Any):
    import cv2

    if image is None:
        return None
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def _draw_label(canvas: Any, lines: Sequence[str]) -> Any:
    """在画面左上叠一条半透明底栏，保持原画面内容不变。

    字号按画面宽度收缩，底栏高度按实际行高计算，保证文字完整可读不溢出。
    """

    import cv2

    height, width = canvas.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    scale = max(0.3, min(0.6, width / 700.0))
    sizes = [cv2.getTextSize(text, font, scale, thickness)[0] for text in lines]
    widest = max(size[0] for size in sizes) if sizes else 0
    limit = max(20, width - 20)
    if widest > limit:
        scale *= limit / widest
        sizes = [cv2.getTextSize(text, font, scale, thickness)[0]
                 for text in lines]
    line_height = max((size[1] for size in sizes), default=10) + 6
    band = min(height, 8 + line_height * len(lines))
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (width, band), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0.0, canvas)
    y = 5
    for text in lines:
        y += line_height
        cv2.putText(canvas, text, (10, y - 4), font, scale,
                    (246, 246, 246), thickness, cv2.LINE_AA)
    return canvas


def write_manual_y_video(path: str | Path, *, job: Mapping[str, Any],
                         summary: Mapping[str, Any],
                         events: Sequence[Mapping[str, Any]]
                         ) -> tuple[int, float]:
    """按单位分析窗写标注视频；返回 (帧数, 总时长秒)。"""

    asset = job.get("asset") or {}
    source_path = asset.get("path")
    if not isinstance(source_path, str) or not source_path:
        raise OSError("标注视频缺少源文件路径")
    metadata = asset.get("metadata") or {}
    video_window = job.get("window") or {}
    start = _finite(summary.get("analysis_window_start_s"))
    if start is None:
        start = _finite(video_window.get("start_s")) or 0.0
    end = _finite(summary.get("analysis_window_end_s"))
    if end is None:
        end = _finite(video_window.get("end_s"))

    source = media.VideoSource(
        source_path,
        start_s=float(start),
        end_s=float(end) if end is not None else None,
        orientation=int(job.get("orientation", 0) or 0),
        fps_override=job.get("fps_override"),
        sample_fps=None,
        source_fps=metadata.get("fps"),
        decode_images=True,
    )

    label = _ascii_label(summary.get("unit_name"))
    ordered = _ordered_events(events)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    window_start = float(start)
    window_end = float(end) if end is not None else math.inf

    container = None
    stream = None
    queued: list[int] = []
    cursor = 0
    sequence: list[str] = []
    current: dict | None = None
    written = 0
    # 累计真实曝光（秒）与已量化端点：每帧 duration 用相邻端点之差，
    # 不把逐帧取整后的 ticks 相加，避免误差累积。
    elapsed = 0.0
    start_ticks = 0
    end_ticks = 0
    last_ticks = 0
    try:
        for frame in source.iter_frames():
            canvas = _to_bgr(frame.image)
            if canvas is None:
                continue
            moment = _finite(frame.time_s)
            if moment is None:
                continue
            # 只收分析窗内的真实正曝光；窗口两端的零曝光上下文帧不进入视频。
            duration = _clipped_exposure(frame, window_start, window_end)
            if duration is None:
                continue
            while cursor < len(ordered) and ordered[cursor]["time_s"] <= moment + 1e-9:
                current = ordered[cursor]
                sequence.append(str(current["target"]))
                cursor += 1
            lines = [f"{label}  manual  t={clock_text(moment)}  frame={frame.index}"]
            lines.append(f"entries={len(sequence)}  sequence="
                         + ("-".join(sequence[-8:]) if sequence else "-"))
            lines.append(
                f"entry #{current['entry_index']} {current['target']} at "
                f"{clock_text(current['time_s'])}"
                if current else "entry -")
            canvas = _even_size(_draw_label(canvas, lines))
            if container is None:
                container, stream = _open_writer(target, canvas.shape[1],
                                                 canvas.shape[0])
            video_frame = av.VideoFrame.from_ndarray(canvas, format="bgr24")
            elapsed += duration
            end_ticks = int(round(elapsed * TIMEBASE.denominator))
            ticks = max(1, end_ticks - start_ticks)
            video_frame.pts = start_ticks
            video_frame.time_base = TIMEBASE
            queued.append(ticks)
            for packet in stream.encode(video_frame):
                _mux(container, packet, queued)
            start_ticks = end_ticks
            last_ticks = ticks
            written += 1
        if container is None:
            raise OSError("分析范围内没有可写入的源帧，未生成标注视频")
        for packet in stream.encode(None):
            if not queued:
                queued.append(last_ticks)
            _mux(container, packet, queued)
        container.close()
        container = None
        stream = None
    except BaseException as error:
        if container is not None:
            try:
                container.close()
            except Exception:
                pass
        try:
            os.unlink(target)
        except OSError:
            pass
        if isinstance(error, OSError):
            raise
        # 编码/写入失败统一变成明确的输出失败，交给导出层标 error。
        raise OSError(f"标注视频编码失败：{error}") from error

    if not written:
        try:
            os.unlink(target)
        except OSError:
            pass
        raise OSError("分析范围内没有可写入的源帧，未生成标注视频")
    return written, end_ticks / TIMEBASE.denominator
