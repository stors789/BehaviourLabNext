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
    def __init__(self, events: list):
        self._events = events

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
    state: dict = {"events": [], "boxes": [], "ready": (True, "")}

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
            lambda *a, **k: FakeServer(state["events"]))
        kwargs["ready_probe"] = lambda url, root: state["ready"]
        return LocalService(app, service_paths, **kwargs)

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
