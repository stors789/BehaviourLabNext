"""桌面桥与原生窗口生命周期的替身检查。"""

from __future__ import annotations

import sys
import types

import pytest

from behaviourlab.desktop import DesktopBridge, DesktopRuntimeError, open_window


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


class TrackingWindow(FakeWindow):
    def __init__(self, result=None):
        self.reads = []
        super().__init__(result)

    def __getattr__(self, name):
        self.reads.append(name)
        raise AttributeError(name)


PUBLIC_API = {"choose_videos", "choose_folder", "choose_image", "open_directory"}


def public_names(bridge):
    return {name for name in dir(bridge) if not name.startswith("_")}


def test_public_api_is_stable_and_does_not_touch_window():
    bridge = DesktopBridge(webview_module=FakeWebview())
    assert public_names(bridge) == PUBLIC_API

    window = TrackingWindow(())
    bridge._bind(window)
    assert public_names(bridge) == PUBLIC_API
    for name in PUBLIC_API:
        assert callable(getattr(bridge, name))
    assert window.reads == []


def test_bridge_has_no_window_until_bound():
    bridge = DesktopBridge(webview_module=FakeWebview())
    assert bridge._window is None
    with pytest.raises(RuntimeError) as captured:
        bridge.choose_videos()
    assert "窗口尚未准备好" in str(captured.value)


def test_bridge_returns_selected_paths_unchanged():
    bridge = DesktopBridge(webview_module=FakeWebview())
    window = FakeWindow(
        ("C:/实验数据/动物 甲.mp4", "C:/实验数据/动物 乙.mp4")
    )
    bridge._bind(window)

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
    bridge._bind(window)

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


def test_open_directory_validates_and_reports(tmp_path):
    opened = []
    bridge = DesktopBridge(
        webview_module=FakeWebview(),
        path_opener=lambda path: opened.append(path) or True,
    )
    folder = tmp_path / "实验 数据"
    folder.mkdir()

    assert bridge.open_directory(str(folder)) is True
    assert opened == [str(folder.resolve())]

    assert bridge.open_directory(str(tmp_path / "不存在")) is False

    file_path = tmp_path / "note.txt"
    file_path.write_text("x", encoding="utf-8")
    assert bridge.open_directory(str(file_path)) is False

    failing = DesktopBridge(
        webview_module=FakeWebview(), path_opener=lambda path: False
    )
    assert failing.open_directory(str(folder)) is False


def test_window_binds_bridge_before_event_loop(monkeypatch):
    """自检 1：假原生后端执行真实 initialized(renderer) 选择契约。"""

    created = {}
    window = FakeWindow()

    def create_window(title, url, **kwargs):
        created.update(kwargs)
        created["title"] = title
        created["url"] = url
        return window

    def start(**kwargs):
        bridge = created["js_api"]
        assert bridge._window is window, "开始 GUI 前私有窗口引用已绑定"
        created["started"] = True
        created["start_kwargs"] = kwargs
        window.events.initialized.fire("edgechromium")

    fake_module = types.SimpleNamespace(
        create_window=create_window,
        start=start,
        settings={"ALLOW_DOWNLOADS": False, "IGNORE_SSL_ERRORS": True},
    )
    window.events = types.SimpleNamespace(initialized=_FakeEvent())

    monkeypatch.setitem(sys.modules, "webview", fake_module)

    open_window("http://127.0.0.1:8765")

    assert created["started"] is True
    assert created["width"] == 1360
    assert created["height"] == 900
    assert created["min_size"] == (1000, 680)
    assert created["js_api"]._window is window
    assert created["start_kwargs"] == {"gui": "edgechromium", "debug": False}
    # 安全设置：允许用户触发的下载，不忽略证书错误。
    assert fake_module.settings["ALLOW_DOWNLOADS"] is True
    assert fake_module.settings["IGNORE_SSL_ERRORS"] is False


class _FakeEvent:
    """initialized 事件替身：真实事件用 += 订阅，同步触发。"""

    def __init__(self):
        self._items = []

    def __iadd__(self, item):
        self._items.append(item)
        return self

    def fire(self, renderer):
        for item in list(self._items):
            item(renderer)


class _Tracked:
    """公开属性替身：记录任何读取，验证桥枚举不会递归访问。"""

    def __init__(self):
        self.reads = []

    def __getattr__(self, name):
        self.reads.append(name)
        raise AttributeError(name)


class _NativeWindow:
    def __init__(self):
        self.events = types.SimpleNamespace(initialized=_FakeEvent())
        self.payload = _Tracked()


class _FakeBackend:
    """假原生后端：只有正确引擎才真正创建原生窗口。"""

    OPEN_DIALOG = "open"
    FOLDER_DIALOG = "folder"

    def __init__(self, renderer):
        self.renderer = renderer
        self.settings = {"ALLOW_DOWNLOADS": False, "IGNORE_SSL_ERRORS": True}
        self.created = []
        self.start_calls = []
        self.native_created = 0

    def create_window(self, title, url, **kwargs):
        native = _NativeWindow()
        self.created.append({"title": title, "url": url, "kwargs": kwargs,
                             "window": native})
        return native

    def start(self, *args, **kwargs):
        self.start_calls.append(kwargs)
        entry = self.created[-1]
        js_api = entry["kwargs"]["js_api"]
        assert js_api._window is entry["window"], "开始 GUI 前私有窗口引用已绑定"
        entry["window"].events.initialized.fire(self.renderer)
        if self.renderer == "edgechromium":
            self.native_created += 1


def _run_open_window(monkeypatch, renderer):
    backend = _FakeBackend(renderer)
    monkeypatch.setitem(sys.modules, "webview", backend)
    error = None
    try:
        open_window("http://127.0.0.1:8765")
    except DesktopRuntimeError as caught:
        error = caught
    return backend, error


def test_runtime_gate_accepts_only_edgechromium(monkeypatch):
    backend, error = _run_open_window(monkeypatch, "edgechromium")
    assert error is None
    assert len(backend.created) == 1
    assert backend.native_created == 1
    assert backend.start_calls == [{"gui": "edgechromium", "debug": False}]
    assert backend.settings["ALLOW_DOWNLOADS"] is True
    assert backend.settings["IGNORE_SSL_ERRORS"] is False
    # 公开 API 仍只有四个选择/目录方法；窗口替身的公开属性没有被
    # 桥枚举递归访问。
    bridge = backend.created[0]["kwargs"]["js_api"]
    assert {name for name in dir(bridge) if not name.startswith("_")} == \
        {"choose_videos", "choose_folder", "choose_image", "open_directory"}
    assert backend.created[0]["window"].payload.reads == []


@pytest.mark.parametrize("renderer", ["mshtml", None])
def test_runtime_gate_rejects_other_engines(monkeypatch, renderer):
    backend, error = _run_open_window(monkeypatch, renderer)
    assert error is not None, "非 edgechromium 必须产生可行动错误"
    assert backend.native_created == 0, "错误引擎不创建原生窗口"
    message = str(error)
    assert "WebView2" in message
    assert "https://developer.microsoft.com/microsoft-edge/webview2/" in message
    assert "重新启动" in message
    assert ".NET" in message


def test_runtime_import_failure_is_actionable(monkeypatch):
    class BrokenImporter:
        def __getattr__(self, name):
            raise ImportError("webview 依赖不可用")

    monkeypatch.setitem(sys.modules, "webview", BrokenImporter())
    with pytest.raises(DesktopRuntimeError) as captured:
        open_window("http://127.0.0.1:8765")
    assert "WebView2" in str(captured.value)
    assert isinstance(captured.value.__cause__, Exception)


def test_runtime_start_failure_keeps_cause(monkeypatch):
    backend = _FakeBackend("edgechromium")

    def failing_start(*args, **kwargs):
        raise OSError("WebView2 异步初始化失败")

    backend.start = failing_start
    monkeypatch.setitem(sys.modules, "webview", backend)
    with pytest.raises(DesktopRuntimeError) as captured:
        open_window("http://127.0.0.1:8765")
    assert isinstance(captured.value.__cause__, OSError)


def test_runtime_keyboard_interrupt_not_disguised(monkeypatch):
    backend = _FakeBackend("edgechromium")

    def interrupting_start(*args, **kwargs):
        raise KeyboardInterrupt

    backend.start = interrupting_start
    monkeypatch.setitem(sys.modules, "webview", backend)
    with pytest.raises(KeyboardInterrupt):
        open_window("http://127.0.0.1:8765")
