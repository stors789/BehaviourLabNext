"""桌面桥与原生窗口生命周期的替身检查。"""

from __future__ import annotations

import sys
import types

import pytest

from behaviourlab.desktop import DesktopBridge, open_window


class FakeWebview:
    OPEN_DIALOG = "open"
    FOLDER_DIALOG = "folder"


class FakeWindow:
    def __init__(self, result=None):
        self.result = result
        self.calls = []

    def create_file_dialog(self, dialog_type, **kwargs):
        self.calls.append((dialog_type, kwargs))
        return self.result


def test_bridge_has_no_window_until_bound():
    bridge = DesktopBridge(webview_module=FakeWebview())
    assert bridge.window is None
    with pytest.raises(RuntimeError) as captured:
        bridge.choose_videos()
    assert "窗口尚未准备好" in str(captured.value)


def test_bridge_returns_selected_paths_unchanged():
    bridge = DesktopBridge(webview_module=FakeWebview())
    window = FakeWindow(
        ("C:/实验数据/动物 甲.mp4", "C:/实验数据/动物 乙.mp4")
    )
    bridge.bind(window)

    assert bridge.choose_videos() == [
        "C:/实验数据/动物 甲.mp4",
        "C:/实验数据/动物 乙.mp4",
    ]
    assert len(window.calls) == 1
    dialog_type, kwargs = window.calls[0]
    assert dialog_type == "open"
    assert kwargs["allow_multiple"] is True
    filters = " ".join(kwargs["file_types"])
    for suffix in (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".mpg",
                   ".mpeg", ".mts", ".m2ts", ".webm", ".flv"):
        assert f"*{suffix}" in filters

    selection = ["C:/实验数据/动物 甲.mp4", "C:/实验数据/动物 乙.mp4"]
    window.result = []
    assert bridge.choose_videos() == []
    assert selection == ["C:/实验数据/动物 甲.mp4", "C:/实验数据/动物 乙.mp4"]


def test_bridge_cancel_keeps_image_and_folder():
    bridge = DesktopBridge(webview_module=FakeWebview())
    window = FakeWindow(("C:/背景/场地.png",))
    bridge.bind(window)

    background = "C:/背景/旧图.png"
    chosen = bridge.choose_image()
    assert chosen == "C:/背景/场地.png"
    background = chosen
    assert background == "C:/背景/场地.png"
    filters = " ".join(window.calls[-1][1]["file_types"])
    for suffix in (".png", ".jpg", ".jpeg", ".bmp"):
        assert f"*{suffix}" in filters

    window.result = None
    assert bridge.choose_image() is None
    assert background == "C:/背景/场地.png"

    window.result = ("C:/实验数据/动物",)
    assert bridge.choose_folder() == "C:/实验数据/动物"
    window.result = None
    assert bridge.choose_folder() is None


def test_window_binds_bridge_before_event_loop(monkeypatch):
    created = {}
    window = FakeWindow()

    def create_window(title, url, **kwargs):
        created.update(kwargs)
        created["title"] = title
        created["url"] = url
        return window

    def start():
        bridge = created["js_api"]
        assert bridge.window is window
        created["started"] = True

    fake_module = types.SimpleNamespace(
        create_window=create_window,
        start=start,
    )
    monkeypatch.setitem(sys.modules, "webview", fake_module)

    open_window("http://127.0.0.1:8765")

    assert created["started"] is True
    assert created["width"] == 1360
    assert created["height"] == 900
    assert created["min_size"] == (1000, 680)
    assert created["js_api"].window is window
