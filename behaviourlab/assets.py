"""真实媒体导入：源路径链接、上传副本与背景图。

导入只读原片；上传只在数据根 videos 下生成应用管理的副本。探测失败按项记录，
不拖垮同一批次的其他文件。
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from . import media
from .config import DataPaths
from .schema import ValidationError, new_id
from .store import Store

CHUNK_SIZE = 1024 * 1024
UPLOAD_SUFFIX = re.compile(r"\.[a-z0-9]{1,8}")


class BackgroundError(RuntimeError):
    """背景图无法解码或编码。"""


def _error(name: str, message: str) -> dict:
    return {"name": name or "未命名文件", "message": message}


def _display_name(path: str) -> str:
    name = Path(path).name
    return name or path


def _canonical_file(path: str) -> str:
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise OSError("不是普通文件")
    return str(resolved)


def _walk_directory(directory: Path) -> Iterable[str]:
    for root, dirnames, filenames in os.walk(directory):
        dirnames.sort(key=str.casefold)
        for name in sorted(filenames, key=str.casefold):
            candidate = Path(root) / name
            if candidate.is_symlink() or not candidate.is_file():
                continue
            if media.has_video_extension(name):
                yield str(candidate)


def _expand(paths: Sequence[str]) -> Iterable[str]:
    for raw in paths:
        candidate = Path(raw).expanduser()
        if candidate.is_dir():
            yield from _walk_directory(candidate)
        else:
            # 显式文件不经后缀筛选，交给探测决定是否可用。
            yield str(candidate)


def _metadata_or_error(probe: Callable[[str], Any], path: str) -> tuple[dict | None, str | None]:
    try:
        metadata = probe(path)
    except Exception as error:  # 每个文件独立失败
        return None, str(error) or error.__class__.__name__
    if not isinstance(metadata, Mapping):
        return None, "探测器未返回元数据"
    return dict(metadata), None


def import_paths(
    store: Store,
    paths: Sequence[str],
    project_id: str | None,
    *,
    probe: Callable[[str], Any] | None = None,
) -> dict:
    """导入源路径；同项目同路径复用旧资产，同请求内按规范路径去重。"""

    probe = probe or media.probe_video
    assets: list[dict] = []
    errors: list[dict] = []
    seen: set[str] = set()

    for raw in paths:
        if not isinstance(raw, str) or not raw.strip():
            errors.append(_error("未命名路径", "路径必须是非空字符串"))
            continue
        try:
            expanded = list(_expand([raw]))
        except OSError as error:
            errors.append(_error(_display_name(raw), str(error)))
            continue
        if not expanded:
            errors.append(_error(_display_name(raw), "没有找到受支持的视频文件"))
            continue

        for candidate in expanded:
            name = _display_name(candidate)
            try:
                canonical = _canonical_file(candidate)
            except (OSError, RuntimeError) as error:
                errors.append(_error(name, f"无法读取文件：{error}"))
                continue

            folded = os.path.normcase(canonical).casefold()
            if folded in seen:
                continue
            seen.add(folded)

            existing = store.find_asset_by_path(project_id, canonical)
            if existing is not None:
                assets.append(existing)
                continue

            metadata, message = _metadata_or_error(probe, canonical)
            if metadata is None:
                errors.append(_error(name, message or "无法探测视频"))
                continue

            try:
                asset = store.create_asset(
                    name=name,
                    path=canonical,
                    ownership="linked",
                    project_id=project_id,
                    metadata=metadata,
                )
            except (ValidationError, OSError) as error:
                errors.append(_error(name, str(error)))
                continue
            assets.append(asset)

    return {"assets": assets, "errors": errors}


def _safe_suffix(display_name: str) -> str:
    suffix = Path(display_name).suffix.lower()
    return suffix if UPLOAD_SUFFIX.fullmatch(suffix) else ""


def _upload_display_name(filename: str | None) -> str:
    raw = (filename or "").replace("\\", "/")
    name = raw.rsplit("/", 1)[-1].strip()
    return name or "video"


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def import_uploads(
    store: Store,
    files: Sequence[Any],
    project_id: str | None,
    *,
    probe: Callable[[str], Any] | None = None,
) -> dict:
    """保存上传副本；同项目同内容哈希且旧副本存在时复用旧资产。"""

    probe = probe or media.probe_video
    assets: list[dict] = []
    errors: list[dict] = []
    videos_dir = Path(store.paths.videos)

    for upload in files:
        display = _upload_display_name(getattr(upload, "filename", None))
        asset_id = new_id()
        suffix = _safe_suffix(display)
        target = videos_dir / f"{asset_id}{suffix}"
        partial = target.with_name(target.name + ".partial")
        digest = hashlib.sha256()

        try:
            stream = getattr(upload, "stream", None)
            if stream is None:
                raise OSError("无法读取上传内容")
            with partial.open("wb") as handle:
                while True:
                    chunk = stream.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    digest.update(chunk)
                    handle.write(chunk)
            os.replace(partial, target)
        except OSError as error:
            _unlink(partial)
            _unlink(target)
            errors.append(_error(display, f"无法保存上传副本：{error}"))
            continue

        checksum = digest.hexdigest()
        existing = store.find_asset_by_sha256(project_id, checksum)
        if existing is not None and Path(existing["path"]).is_file():
            _unlink(target)
            assets.append(existing)
            continue

        metadata, message = _metadata_or_error(probe, str(target))
        if metadata is None:
            _unlink(target)
            errors.append(_error(display, message or "无法探测视频"))
            continue

        try:
            asset = store.create_asset(
                asset_id=asset_id,
                name=display,
                path=str(target),
                ownership="owned",
                project_id=project_id,
                metadata=metadata,
                upload_sha256=checksum,
            )
        except (ValidationError, OSError) as error:
            _unlink(target)
            errors.append(_error(display, str(error)))
            continue
        assets.append(asset)

    return {"assets": assets, "errors": errors}


def save_background(paths: DataPaths, data: bytes) -> dict:
    """把上传内容实际解码后重编码为 PNG，存到 backgrounds/随机ID.png。"""

    import cv2
    import numpy as np

    if not data:
        raise BackgroundError("文件为空")
    try:
        buffer = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except Exception as error:
        raise BackgroundError(f"无法解码图片：{error}") from error
    if image is None or getattr(image, "ndim", 0) != 3 or image.shape[2] != 3:
        raise BackgroundError("无法解码为彩色图像")

    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise BackgroundError("无法编码背景图")

    filename = f"{new_id()}.png"
    target = Path(paths.backgrounds) / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(encoded.tobytes())
    return {"path": str(target), "url": f"/api/backgrounds/{filename}"}


def validate_background_path(path: str) -> None:
    """确认 linked 原生背景当前可按背景语义解码；失败抛 BackgroundError。

    与 load_static_background 使用同一解码语义（IMREAD_COLOR），只读
    不拷贝；前端必须在本校验通过后才替换旧有效背景。
    """

    import cv2
    import numpy as np

    file_path = Path(path)
    if not file_path.is_file():
        raise BackgroundError("背景文件不存在")
    try:
        data = file_path.read_bytes()
    except OSError as error:
        raise BackgroundError(f"背景文件不可读取：{error}") from error
    if not data:
        raise BackgroundError("背景文件为空")
    try:
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8),
                             cv2.IMREAD_COLOR)
    except Exception as error:
        # cv2 对空内容/畸形数据可能直接抛错：统一转成可理解的背景
        # 业务错误，而不是让接口冒泡成服务器内部错误。
        raise BackgroundError(f"背景无法解码：{error}") from error
    if image is None or getattr(image, "ndim", 0) != 3:
        raise BackgroundError("背景无法解码为彩色图像")
