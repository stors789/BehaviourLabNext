"""桌面窗口桥与原生文件选择。

桥在构造时没有窗口；create_window 返回后先绑定，再启动事件循环。
公开接口只有四个选择/打开方法，窗口与辅助方法均为私有，避免 GUI 枚举
进入 pywebview 窗口内部。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .media import image_dialog_filter, video_dialog_filter

WINDOW_WIDTH = 1360
WINDOW_HEIGHT = 900
WINDOW_MIN_SIZE = (1000, 680)


def _open_directory_default(path: str) -> bool:
    if os.name == "nt":
        os.startfile(path)  # type: ignore[attr-defined]
        return True
    command = ["open", path] if sys.platform == "darwin" else ["xdg-open", path]
    subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return True


class DesktopBridge:
    """暴露给前端的四个方法；窗口未就绪时明确失败。"""

    def __init__(self, *, webview_module: Any = None,
                 path_opener: Any = None) -> None:
        self._window: Any = None
        self._webview = webview_module
        self._path_opener = path_opener or _open_directory_default

    def _bind(self, window: Any) -> None:
        self._window = window

    def _require_window(self) -> Any:
        if self._window is None:
            raise RuntimeError("窗口尚未准备好")
        return self._window

    def _module(self) -> Any:
        if self._webview is None:
            import webview

            self._webview = webview
        return self._webview

    @staticmethod
    def _first(result: Any) -> str | None:
        if not result:
            return None
        if isinstance(result, (str, bytes)):
            return result.decode() if isinstance(result, bytes) else result
        first = result[0]
        return first.decode() if isinstance(first, bytes) else str(first)

    def choose_videos(self) -> list[str]:
        window = self._require_window()
        module = self._module()
        result = window.create_file_dialog(
            module.OPEN_DIALOG,
            allow_multiple=True,
            file_types=(video_dialog_filter(),),
        )
        if isinstance(result, (str, bytes)):
            return [self._first([result]) or ""]
        return [str(item) for item in (result or [])]

    def choose_folder(self) -> str | None:
        window = self._require_window()
        module = self._module()
        result = window.create_file_dialog(module.FOLDER_DIALOG, allow_multiple=False)
        return self._first(result)

    def choose_image(self) -> str | None:
        window = self._require_window()
        module = self._module()
        result = window.create_file_dialog(
            module.OPEN_DIALOG,
            allow_multiple=False,
            file_types=(image_dialog_filter(),),
        )
        return self._first(result)

    def open_directory(self, path: str) -> bool:
        try:
            target = Path(path).expanduser().resolve(strict=True)
        except (OSError, RuntimeError, TypeError):
            return False
        if not target.is_dir():
            return False
        try:
            return bool(self._path_opener(str(target)))
        except Exception:
            return False


def open_window(url: str, *, title: str = "BehaviourLabNext") -> None:
    """创建唯一窗口，先绑定桥，再启动事件循环。"""

    import webview

    bridge = DesktopBridge(webview_module=webview)
    window = webview.create_window(
        title,
        url,
        js_api=bridge,
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
        min_size=WINDOW_MIN_SIZE,
    )
    if window is None:
        raise RuntimeError("窗口创建失败")
    bridge._bind(window)
    webview.start()
