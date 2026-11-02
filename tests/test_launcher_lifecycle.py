"""第 24 块自检：下载/另存为统一契约与启动生命周期的替身检查。

下载契约仿 pywebview EdgeChromium 后端的 DownloadStarting 处理：
设置开启后，只有用户点击附件才弹一次保存对话框，确认写入所选路径、
取消不产生文件；生命周期用假服务器/Jobs/GUI 驱动 launcher.main。
真实 Windows 原生表现不在此验证，留待独立验收。
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from behaviourlab import desktop as desktop_module
from behaviourlab import launcher as launcher_module
from behaviourlab.config import DataPaths
from behaviourlab.desktop import DesktopRuntimeError
from behaviourlab.jobs import JobScheduler
from behaviourlab.launcher import main
from behaviourlab.service import LocalService
from behaviourlab.store import Store


# -- 自检 2：四类附件统一的下载/另存为契约 --------------------------------

ALLOW_DOWNLOADS = True  # desktop.open_window 打开的同一设置


def download_starting(args: dict, user_picks, dialogs: list) -> None:
    """仿 edgechromium.on_download_starting：默认名来自响应，取消不保存。"""

    if not ALLOW_DOWNLOADS:
        args["Cancel"] = True
        return
    # 每次真实下载只进入一次 SaveFileDialog；页面加载/预览/切换设置
    # 不会走到这里。
    dialogs.append(args["ResultFilePath"])
    chosen = user_picks(os.path.basename(args["ResultFilePath"]))
    if chosen is None:
        args["Cancel"] = True
    else:
        args["ResultFilePath"] = chosen
        args["Cancel"] = False


def test_download_save_dialog_contract():
    dialogs: list[str] = []
    written: list[str] = []

    def user_confirm(suggested: str):
        target = "D:/实验结果/" + suggested
        written.append(target)  # 确认后底层写入所选路径
        return target

    def user_cancel(_suggested: str):
        return None

    kinds = ["结果 鼠一.csv", "结果 鼠一.zip",
             "unit_01_trajectory.png", "annotated.mp4"]
    for name in kinds:
        args = {"ResultFilePath": f"C:/Downloads/{name}", "Cancel": True}
        before = len(dialogs)
        download_starting(args, user_confirm, dialogs)
        assert len(dialogs) == before + 1, "只有点击附件才弹一次对话框"
        assert args["Cancel"] is False
        assert args["ResultFilePath"] == f"D:/实验结果/{name}"
        assert written[-1] == f"D:/实验结果/{name}"

        args = {"ResultFilePath": f"C:/Downloads/{name}", "Cancel": True}
        before_written = len(written)
        download_starting(args, user_cancel, dialogs)
        assert args["Cancel"] is True, "取消必须 Cancel=true"
        assert args["ResultFilePath"] == f"C:/Downloads/{name}"
        assert len(written) == before_written, "取消不产生文件"

    assert len(dialogs) == 2 * len(kinds), "四类附件复用同一合同，无额外弹出"


# -- 自检 3：假服务器/Jobs/GUI 的启动生命周期 -----------------------------


class FakeServer:
    def __init__(self, events: list, workers: set):
        self._events = events
        self.task_dispatcher = SimpleNamespace(threads=workers)
        self.active_channels = {}

    def run(self):
        self._events.append("serve")

    def close(self):
        self._events.append("http")


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """提供真实 Store/Jobs + 假 HTTP/GUI 的 main 驱动环境。"""

    paths = DataPaths.from_dir(tmp_path / "data")
    paths.ensure()
    store = Store(paths)
    real_store_close = store.close
    jobs = JobScheduler(store, paths, inline=False)
    real_jobs_shutdown = jobs.shutdown
    state: dict = {"events": [], "boxes": [], "ready": (True, ""),
                   "workers": set(), "service": None}

    def wrapped_store_close():
        state["events"].append("store")
        real_store_close()

    def wrapped_jobs_shutdown(*args, **kwargs):
        state["events"].append("jobs")
        return real_jobs_shutdown(*args, **kwargs)

    monkeypatch.setattr(store, "close", wrapped_store_close)
    monkeypatch.setattr(jobs, "shutdown", wrapped_jobs_shutdown)

    def fake_create_app(*, data_dir, testing=False):
        assert data_dir == str(paths.root)
        return SimpleNamespace(config={
            "JOBS": jobs, "STORE": store, "JOB_OWNER": jobs.owner})

    def wrapped_local_service(app, service_paths, **kwargs):
        kwargs["server_factory"] = (
            lambda *a, **k: FakeServer(state["events"], state["workers"]))
        kwargs["ready_probe"] = lambda url, root: state["ready"]
        created = LocalService(app, service_paths, **kwargs)
        created._http_drain_timeout = 0.2
        state["service"] = created
        return created

    monkeypatch.setattr(launcher_module, "LocalService", wrapped_local_service)
    monkeypatch.setattr("behaviourlab.web.create_app", fake_create_app)
    monkeypatch.setattr(launcher_module, "show_native_error",
                        lambda message: state["boxes"].append(message))
    state["paths"] = paths
    state["store"] = store
    state["jobs"] = jobs
    return state


def test_gui_error_returns_1_and_cleans(harness, monkeypatch, capsys):
    """GUI 抛可行动错误：返回 1、startup.log 保留中文原异常、链路清理。"""

    def failing_window(url):
        raise DesktopRuntimeError(
            "无法启动桌面窗口：实际渲染引擎是“mshtml”。请安装或修复 "
            "WebView2 后重新启动程序。")

    monkeypatch.setattr(desktop_module, "open_window", failing_window)
    code = main(["--data-dir", str(harness["paths"].root)])
    assert code == 1
    assert harness["jobs"].running is False
    assert harness["jobs"].owner.held is False, "退出时释放本实例所有权"
    assert harness["events"].count("http") == 1, "server.close 已执行"
    assert harness["store"].closed is True
    assert harness["boxes"] and "无法启动" in harness["boxes"][0]
    startup = (harness["paths"].logs / "startup.log").read_text(
        encoding="utf-8")
    assert "mshtml" in startup and "DesktopRuntimeError" in startup
    assert "Traceback" in startup
    payload = json.loads((harness["paths"].root / "server.json").read_text(
        encoding="utf-8"))
    assert payload["pid"] == os.getpid()
    assert payload["data_root"] == str(harness["paths"].root)
    assert payload["url"] in capsys.readouterr().out


def test_normal_window_return_returns_0_and_cleans(harness, monkeypatch, capsys):
    """正常窗口返回：返回 0，jobs.shutdown 与 server.close 各执行一次。"""

    monkeypatch.setattr(desktop_module, "open_window",
                        lambda url: harness["events"].append("window"))
    code = main(["--data-dir", str(harness["paths"].root)])
    assert code == 0
    events = harness["events"]
    assert "window" in events
    assert harness["jobs"].running is False
    assert harness["jobs"].owner.held is False
    assert events.count("jobs") == 1
    assert events.count("http") == 1
    assert events.count("store") == 1
    # 清理链顺序：先停调度，再排空 HTTP，最后关闭 Store。
    assert events.index("jobs") < events.index("http") < events.index("store")
    payload = json.loads((harness["paths"].root / "server.json").read_text(
        encoding="utf-8"))
    assert payload["url"].startswith("http://127.0.0.1:")
    out = capsys.readouterr().out
    assert json.loads(out.strip().splitlines()[-1]) == payload


def test_browser_error_shows_no_message_box(harness, monkeypatch, capsys):
    """--browser 模式错误不调用 MessageBox，返回 1 并写 startup.log。"""

    harness["ready"] = (False, "服务在 60 次检查后仍未就绪")
    code = main(["--data-dir", str(harness["paths"].root), "--browser"])
    assert code == 1
    assert harness["boxes"] == [], "--browser 不创建错误框"
    assert harness["jobs"].running is False
    assert harness["store"].closed is True
    startup = (harness["paths"].logs / "startup.log").read_text(
        encoding="utf-8")
    assert "未就绪" in startup
    assert capsys.readouterr().out == "", "服务未就绪时不输出 server JSON"


def test_lock_taken_reports_and_leaves_other_instance(harness, monkeypatch):
    """未取到锁：返回 2，不关闭其他实例的 Jobs，只清自己的对象。"""

    assert harness["jobs"].start() is True  # 第一实例先持有数据根
    second_store = Store(harness["paths"])
    second_jobs = JobScheduler(second_store, harness["paths"], inline=False)

    def fake_create_app(*, data_dir, testing=False):
        return SimpleNamespace(config={
            "JOBS": second_jobs, "STORE": second_store,
            "JOB_OWNER": second_jobs.owner})

    monkeypatch.setattr("behaviourlab.web.create_app", fake_create_app)

    code = main(["--data-dir", str(harness["paths"].root), "--browser"])
    assert code == 2
    assert second_jobs.running is False
    assert second_jobs.owner.held is False
    assert second_store.closed is True
    assert harness["jobs"].running is True, "不关闭其他实例的调度"
    assert harness["jobs"].owner.held is True, "不释放其他实例的所有权"
    assert harness["store"].closed is False
    assert harness["jobs"].shutdown() is True
    assert harness["jobs"].owner.held is False


def test_blocked_http_keeps_lock_until_drained(harness, monkeypatch):
    """HTTP 未排空：stop=False 返回 1、不放锁；第二所有者拿不到锁；
    解除阻塞后可继续收尾，全部退出才关 Store/放锁。"""

    import threading

    from behaviourlab.processes import JobOwnerRegistry, OwnershipError

    monkeypatch.setattr(desktop_module, "open_window",
                        lambda url: harness["events"].append("window"))
    workers = harness["workers"]
    workers.add(threading.current_thread())  # 模拟未退出的 HTTP 工作线程

    code = main(["--data-dir", str(harness["paths"].root)])
    assert code == 1, "stop=False 不得以 0 冒称成功"
    assert harness["jobs"].running is False, "Jobs 已先收尾"
    assert harness["jobs"].owner.held is True, "同根锁仍由整体生命周期持有"
    assert harness["store"].closed is False, "使用者未退出不关 Store"
    assert harness["events"].count("jobs") == 1
    assert harness["events"].count("http") == 1, "监听与连接已关闭"

    # 真实阻塞未排空时，第二所有者不能取得同根锁。
    with pytest.raises(OwnershipError):
        JobOwnerRegistry(harness["paths"]).acquire()

    # 解除阻塞后同一实例可继续收尾：排空 → Store → 放锁。
    workers.clear()
    assert harness["service"].stop() is True
    assert harness["events"].index("http") < harness["events"].index("store")
    assert harness["store"].closed is True
    assert harness["jobs"].owner.held is False, "全部使用者退出后才放锁"


def test_waitress_factory_recovers_dispatcher_on_bind_failure(monkeypatch):
    """工厂部分构造/bind 失败：收掉自己创建的 dispatcher 线程。"""

    import waitress as waitress_module

    from behaviourlab.service import _default_server_factory

    captured = {}

    def failing_create_server(*args, **kwargs):
        # 工厂把自己的 dispatcher 通过 _dispatcher 传入，这里捕获后失败。
        captured["dispatcher"] = kwargs["_dispatcher"]
        raise OSError("端口被占用")

    monkeypatch.setattr(waitress_module, "create_server", failing_create_server)
    with pytest.raises(OSError):
        _default_server_factory(object(), "127.0.0.1", 39011, 2)
    dispatcher = captured["dispatcher"]
    assert dispatcher.threads == set(), "bind 失败后 dispatcher 线程已回收"


def test_waitress_factory_closes_owned_map_on_failure(monkeypatch):
    """工厂失败分支必须逐项显式 close 本实例 map 登记的资源。"""

    import os as os_module

    import waitress as waitress_module

    from behaviourlab.service import _default_server_factory

    captured = {}

    def failing_create_server(*args, **kwargs):
        owned_map = kwargs["map"]
        captured["map"] = owned_map
        captured["dispatcher"] = kwargs["_dispatcher"]
        # 模拟半构造已登记的资源：trigger 管道两端各自登记。
        read_fd, write_fd = os_module.pipe()
        captured["pipe"] = (read_fd, write_fd)

        class _Registered:
            def __init__(self, fd):
                self._fd = fd
                self.closed = False

            def fileno(self):
                return self._fd

            def close(self):
                if not self.closed:
                    self.closed = True
                    os_module.close(self._fd)

        owned_map[read_fd] = _Registered(read_fd)
        owned_map[write_fd] = _Registered(write_fd)
        raise OSError("端口被占用")

    monkeypatch.setattr(waitress_module, "create_server", failing_create_server)
    with pytest.raises(OSError):
        _default_server_factory(object(), "127.0.0.1", 39012, 1)
    assert all(entry.closed for entry in captured["map"].values())
    # 显式收尾（非垃圾回收）：pipe 两端文件描述符已不存在。
    with pytest.raises(OSError):
        os_module.fstat(captured["pipe"][0])
    with pytest.raises(OSError):
        os_module.fstat(captured["pipe"][1])
    assert captured["dispatcher"].threads == set()


def test_waitress_factory_real_bind_failure_closes_resources(monkeypatch):
    """真实构造链：trigger/监听 socket 先登记再 bind 失败，必须被显式关闭。"""

    import waitress as waitress_module
    from waitress.server import TcpWSGIServer

    from behaviourlab.service import _default_server_factory

    captured = {}

    def failing_bind(self):
        captured["map"] = self._map
        captured["entries"] = list(self._map.values())
        raise OSError("端口被占用")

    monkeypatch.setattr(TcpWSGIServer, "bind_server_socket", failing_bind)
    with pytest.raises(OSError):
        _default_server_factory(object(), "127.0.0.1", 39013, 2)
    assert captured["entries"], "半构造对象已登记 trigger/监听资源"
    for entry in captured["entries"]:
        trigger_closed = getattr(entry, "_closed", None)
        socket_gone = getattr(entry, "socket", None) is None
        assert trigger_closed or socket_gone, f"所属资源未显式关闭：{entry!r}"
    assert len(captured["map"]) == 0, "asyncore map 中无残留登记"


def test_port_bind_failure_leaves_nothing_behind(harness, monkeypatch):
    """占端口构造失败：无剩余线程/Store/持锁，原异常日志不被替换。"""

    def failing_factory(*args, **kwargs):
        raise OSError("端口被占用")

    def wrapped_local_service(app, service_paths, **kwargs):
        kwargs["server_factory"] = failing_factory
        kwargs["ready_probe"] = lambda url, root: state["ready"]
        created = LocalService(app, service_paths, **kwargs)
        created._http_drain_timeout = 0.2
        state["service"] = created
        return created

    state = harness
    monkeypatch.setattr(launcher_module, "LocalService", wrapped_local_service)
    code = main(["--data-dir", str(harness["paths"].root), "--browser"])
    assert code == 1
    assert harness["jobs"].owner.held is False, "本实例未持有任何锁"
    assert harness["jobs"].running is False
    assert harness["store"].closed is True
    startup = (harness["paths"].logs / "startup.log").read_text(
        encoding="utf-8")
    assert "端口被占用" in startup, "原始异常保留在 startup.log"
    assert "Traceback" in startup


def test_service_construction_failure_recovers_store(harness, monkeypatch):
    """app/Store 创建成功而 LocalService 构造失败：回收 Store，无锁残留。"""

    def broken_service(*args, **kwargs):
        raise OSError("无法申请空闲端口")

    monkeypatch.setattr(launcher_module, "LocalService", broken_service)
    code = main(["--data-dir", str(harness["paths"].root), "--browser"])
    assert code == 1
    assert harness["jobs"].running is False
    assert harness["jobs"].owner.held is False, "构造失败时未取得锁"
    assert harness["store"].closed is True, "已创建的 Store 被回收"
    startup = (harness["paths"].logs / "startup.log").read_text(
        encoding="utf-8")
    assert "无法申请空闲端口" in startup


def test_owner_release_failure_does_not_mask_original_error(
        harness, monkeypatch, caplog):
    """精确注入所属记录删除权限失败：主错误仍是原始 GUI 错误。

    清理异常只作为补充日志；退出码 1 可判定清理状态；解除注入后
    同一实例可继续收尾，不产生新的未处理异常覆盖主因。
    """

    import logging

    def failing_window(url):
        raise DesktopRuntimeError(
            "无法启动桌面窗口：实际渲染引擎是“mshtml”。请安装或修复 "
            "WebView2 后重新启动程序。")

    registry = harness["jobs"].owner
    real_release = registry.release
    blocked = {"on": True}

    def failing_release():
        if blocked["on"]:
            raise PermissionError("拒绝删除所属记录")
        return real_release()

    monkeypatch.setattr(desktop_module, "open_window", failing_window)
    monkeypatch.setattr(registry, "release", failing_release)
    with caplog.at_level(logging.ERROR, logger="behaviourlab"):
        code = main(["--data-dir", str(harness["paths"].root)])
    assert code == 1, "清理未完成不得返回 0"
    # 原始 GUI 错误仍是启动日志主因。
    startup = (harness["paths"].logs / "startup.log").read_text(
        encoding="utf-8")
    assert "mshtml" in startup and "DesktopRuntimeError" in startup
    messages = [record.getMessage() for record in caplog.records]
    assert any("释放数据根所有权失败" in message for message in messages), \
        "清理失败作为补充日志出现"
    assert any("退出时清理未完成" in message for message in messages), \
        "入口侧同样只补日志，不向外冒泡"
    # 释放失败保留所有权与可收尾状态。
    assert registry.held is True
    assert harness["store"].closed is True, "Store 已先于释放关闭"
    blocked["on"] = False
    assert harness["service"].stop() is True, "解除阻塞后可继续收尾"
    assert registry.held is False, "最终完成释放"
