"""桌面窗口桥：只在用户主动调用时创建窗口。"""

from __future__ import annotations


def open_window(url: str, *, title: str = "BehaviourLabNext",
                width: int = 1440, height: int = 900) -> None:
    import webview  # 延迟导入，导入包本身不创建窗口

    webview.create_window(
        title, url, width=width, height=height, min_size=(1024, 700)
    )
    webview.start()
