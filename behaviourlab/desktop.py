"""桌面窗口桥、WebView2 运行时门槛与原生文件选择。

桥在构造时没有窗口；create_window 返回后先绑定，再启动事件循环。
公开接口只有四个选择/打开方法，窗口与辅助方法均为私有，避免 GUI 枚举
进入 pywebview 窗口内部。正式启动请求 gui="edgechromium"，但该参数不
足以证明实际引擎：订阅 initialized(renderer) 同步事件记录真实渲染器，
只有严格等于 edgechromium 才算通过选择门槛。
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .media import image_dialog_filter, video_dialog_filter

LOGGER = logging.getLogger("behaviourlab.desktop")

WINDOW_WIDTH = 1360
WINDOW_HEIGHT = 900
WINDOW_MIN_SIZE = (1000, 680)
WINDOW_TITLE = "BehaviourLabNext"

REQUIRED_RENDERER = "edgechromium"
WEBVIEW2_FIX_URL = "https://developer.microsoft.com/microsoft-edge/webview2/"


class DesktopRuntimeError(RuntimeError):
    """WebView2 运行时不满足要求；消息包含可行动的修复指引。"""


def runtime_accepted(renderer: Any) -> bool:
    """只有严格等于 edgechromium 才通过；mshtml/None/未知一律拒绝。"""

    return renderer == REQUIRED_RENDERER


def runtime_error_message(renderer: Any) -> str:
    """可行动错误：实际引擎、运行时要求、官方修复入口与日志位置。"""

    observed = str(renderer) if renderer else "未报告"
    return (
        f"无法启动桌面窗口：实际渲染引擎是“{observed}”，"
        f"不是必需的 {REQUIRED_RENDERER}。BehaviourLabNext 需要 "
        "Microsoft Edge WebView2 Evergreen Runtime（该依赖要求 "
        ".NET Framework 4.6.2 或更新版本）。请打开官方页面安装或修复 "
        "WebView2：https://developer.microsoft.com/microsoft-edge/webview2/ ，"
        "完成后重新启动本程序；如仍失败建议重启计算机并检查 .NET 环境。"
        "详细信息见数据目录 logs/application.log 与 logs/startup.log。"
    )


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


def _subscribe_initialized(window: Any, observed: dict) -> None:
    """记录 initialized(renderer) 的真实渲染器；回调内不 raise。"""

    events = getattr(window, "events", None)
    event = getattr(events, "initialized", None)
    if event is None:
        # 没有可订阅的事件就无法证明引擎：保持未报告，门槛会拒绝。
        LOGGER.warning("窗口没有 initialized 事件，无法记录实际渲染引擎")
        return

    def on_initialized(renderer: Any = None) -> None:
        observed["renderer"] = renderer

    try:
        event += on_initialized
    except Exception:
        LOGGER.exception("订阅 initialized 事件失败，实际渲染引擎将未报告")


def open_window(url: str, *, title: str = WINDOW_TITLE) -> None:
    """创建唯一窗口并等待生命周期；引擎门槛不满足时抛可行动错误。

    收到正确 renderer 只证明选择门槛通过，不证明后续异步初始化、
    ES modules 或绘图已成功。
    """

    try:
        import webview
        # 允许用户点击附件后的底层下载（下载对话框由 WebView2 后端处理）；
        # 不忽略证书错误，不用降低安全设置规避启动失败。
        webview.settings["ALLOW_DOWNLOADS"] = True
        webview.settings["IGNORE_SSL_ERRORS"] = False
    except Exception as error:
        # import/依赖初始化失败与引擎不符走同一可行动错误，保留根因。
        raise DesktopRuntimeError(runtime_error_message(None)) from error

    bridge = DesktopBridge(webview_module=webview)
    try:
        window = webview.create_window(
            title,
            url,
            js_api=bridge,
            width=WINDOW_WIDTH,
            height=WINDOW_HEIGHT,
            min_size=WINDOW_MIN_SIZE,
        )
    except Exception as error:
        raise DesktopRuntimeError(runtime_error_message(None)) from error
    if window is None:
        raise DesktopRuntimeError(runtime_error_message(None))

    observed: dict = {"renderer": None}
    _subscribe_initialized(window, observed)
    # 先由 Python 启动逻辑绑定私有引用，再启动事件循环；窗口不是
    # js_api 的公开成员。
    bridge._bind(window)
    try:
        webview.start(gui="edgechromium", debug=False)
    except Exception as error:
        raise DesktopRuntimeError(
            runtime_error_message(observed["renderer"])) from error
    if not runtime_accepted(observed["renderer"]):
        # 参数不足以证明引擎：以事件记录的实际渲染器为准，未报告/其他
        # 值都阻止窗口作为正常结果返回。
        raise DesktopRuntimeError(runtime_error_message(observed["renderer"]))
