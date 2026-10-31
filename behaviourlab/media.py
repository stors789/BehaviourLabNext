"""源视频探测与帧读取边界。

本块只实现只读元数据探测；真实源帧、时钟与预览读取在媒体块接入。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class MediaError(RuntimeError):
    """无法读取指定媒体。"""


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


def has_video_extension(name: str) -> bool:
    return Path(name).suffix.lower() in VIDEO_EXTENSIONS


def has_image_extension(name: str) -> bool:
    return Path(name).suffix.lower() in IMAGE_EXTENSIONS


def video_dialog_filter() -> str:
    return "视频文件 (" + ";".join(f"*{suffix}" for suffix in VIDEO_EXTENSIONS) + ")"


def image_dialog_filter() -> str:
    return "图片 (" + ";".join(f"*{suffix}" for suffix in IMAGE_EXTENSIONS) + ")"


def probe_video(path: str | Path) -> dict:
    """探测视频元数据；未知字段保持 None，不猜测。"""

    import av  # 延迟导入，避免仅导入包时加载解码器

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
            rate = stream.average_rate or stream.base_rate
            if rate:
                fps = float(rate)
            frame_count = int(stream.frames) if stream.frames else None
            duration_s: float | None = None
            if stream.duration is not None and stream.time_base is not None:
                duration_s = float(stream.duration * stream.time_base)
            elif container.duration:
                duration_s = float(container.duration) / 1_000_000.0
            codec = context.name or None
    except MediaError:
        raise
    except Exception as error:  # av 的异常类型随容器格式变化
        raise MediaError(f"无法读取视频：{error}") from error

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_s": duration_s,
        "codec": codec,
    }
