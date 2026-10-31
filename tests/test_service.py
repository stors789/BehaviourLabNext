"""本地服务生命周期、所有权与 Windows 子进程参数检查。"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from behaviourlab.config import DataPaths
from behaviourlab.launcher import record_startup_failure
from behaviourlab.processes import (
    ManagedProcess,
    child_creation_flags,
)
from behaviourlab.service import (
    AlreadyRunningError,
    LocalService,
    ReadinessError,
)


class FakeServer:
    def __init__(self, events: list | None = None) -> None:
        self.closed = False
        self.run_called = False
        self.events = events if events is not None else []

    def run(self) -> None:
        self.run_called = True
        self.events.append("server-run")

    def close(self) -> None:
        self.closed = True
        self.events.append("server-close")


def make_service(tmp_path, *, ready=(True, ""), server=None, events=None,
                 port=39001, callbacks=()):
    paths = DataPaths.from_dir(tmp_path / "data").ensure()
    server = server or FakeServer(events)
    service = LocalService(
        object(),
        paths,
        port=port,
        server_factory=lambda app, host, port, threads: server,
        ready_probe=lambda url, root: ready,
        stop_callbacks=list(callbacks),
    )
    return service, server, paths


def test_service_starts_and_cleans_up(tmp_path):
    service, server, paths = make_service(tmp_path)
    assert service.host == "127.0.0.1"
    assert service.threads == 6

    url = service.start()
    assert url == "http://127.0.0.1:39001"
    assert server.run_called is True
    assert service._owner.is_active() is True
    assert (paths.logs / "job_owner.json").exists()

    service.stop()
    assert server.closed is True
    assert not (paths.logs / "job_owner.json").exists()
    service.stop()  # 清理幂等


def test_service_readiness_failure_releases_ownership(tmp_path):
    service, server, paths = make_service(tmp_path, ready=(False, "健康检查失败"))

    with pytest.raises(ReadinessError) as captured:
        service.start()
    assert "健康检查失败" in str(captured.value)
    assert server.closed is True
    assert not (paths.logs / "job_owner.json").exists()


def test_service_refuses_second_instance_for_same_data_root(tmp_path):
    first, first_server, paths = make_service(tmp_path)
    first.start()
    try:
        second = LocalService(
            object(),
            paths,
            port=39002,
            server_factory=lambda *args: FakeServer(),
            ready_probe=lambda url, root: (True, ""),
        )
        with pytest.raises(AlreadyRunningError):
            second.start()
        assert first_server.closed is False
        assert (paths.logs / "job_owner.json").exists()
    finally:
        first.stop()


def test_stop_order_stops_workers_before_server_and_owner(tmp_path):
    events: list[str] = []
    service, server, paths = make_service(
        tmp_path,
        events=events,
        callbacks=[lambda: events.append("worker-stop")],
    )
    service.start()
    events.clear()

    service.stop()

    assert events == ["worker-stop", "server-close"]
    assert not (paths.logs / "job_owner.json").exists()


def test_service_chooses_free_port(tmp_path):
    paths = DataPaths.from_dir(tmp_path / "data").ensure()
    service = LocalService(object(), paths, port=0)
    assert service.port > 0


def test_stop_callback_failure_still_closes_and_releases(tmp_path):
    events: list[str] = []

    def broken_callback() -> None:
        raise RuntimeError("工作进程停止失败")

    service, server, paths = make_service(
        tmp_path,
        callbacks=[broken_callback, lambda: events.append("after")],
    )
    service.start()
    service.stop()

    assert events == ["after"]
    assert server.closed is True
    assert not (paths.logs / "job_owner.json").exists()


def test_startup_log_preserves_original_error(tmp_path):
    paths = DataPaths.from_dir(tmp_path / "data").ensure()
    record_startup_failure(paths, "Traceback\n启动失败：无法创建窗口")

    content = (paths.logs / "startup.log").read_text(encoding="utf-8")
    assert "启动失败：无法创建窗口" in content


def test_unwritable_startup_log_keeps_original_error(tmp_path, monkeypatch, caplog):
    paths = DataPaths.from_dir(tmp_path / "data").ensure()

    def fail_write(self, *args, **kwargs):
        raise OSError("日志目录只读")

    monkeypatch.setattr(Path, "write_text", fail_write)
    record_startup_failure(paths, "原始启动错误：端口占用")

    assert "原始启动错误：端口占用" in caplog.text


def test_launcher_records_startup_failure(tmp_path, monkeypatch):
    import logging

    from behaviourlab import launcher, web

    def broken_app(**kwargs):
        raise RuntimeError("启动失败测试")

    monkeypatch.setattr(web, "create_app", broken_app)
    try:
        code = launcher.main(
            ["--browser", "--data-dir", str(tmp_path / "data"), "--port", "39003"]
        )
    finally:
        root = logging.getLogger()
        for handler in list(root.handlers):
            if getattr(handler, "_behaviourlab", False):
                root.removeHandler(handler)
                handler.close()

    assert code == 1
    content = (tmp_path / "data" / "logs" / "startup.log").read_text(encoding="utf-8")
    assert "启动失败测试" in content


def test_windows_creation_flags_preserve_existing_value():
    flags = 0x00000200
    assert child_creation_flags(flags) == 0x08000200
    assert flags == 0x00000200
    assert child_creation_flags() == 0x08000000


class FakeJob:
    def __init__(self) -> None:
        self.terminated = False
        self.closed = False

    def terminate(self) -> None:
        self.terminated = True

    def close(self) -> None:
        self.closed = True


class FakeProcess:
    def __init__(self) -> None:
        self.waits = []
        self.killed = False

    def wait(self, timeout=None):
        self.waits.append(timeout)
        return 0

    def kill(self) -> None:
        self.killed = True


def test_managed_process_stops_tree_then_releases_handle():
    process = FakeProcess()
    job = FakeJob()
    ManagedProcess(process, job).stop()

    assert job.terminated is True
    assert process.waits == [3.0]
    assert job.closed is True


def test_managed_process_forces_after_timeout():
    class SlowProcess(FakeProcess):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def wait(self, timeout=None):
            self.calls += 1
            self.waits.append(timeout)
            if self.calls == 1:
                raise subprocess.TimeoutExpired(cmd="worker", timeout=timeout)
            return 0

    process = SlowProcess()
    job = FakeJob()
    ManagedProcess(process, job).stop()

    assert process.killed is True
    assert job.closed is True
    assert process.waits == [3.0, 3.0]
