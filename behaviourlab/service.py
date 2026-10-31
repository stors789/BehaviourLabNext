"""本地服务生命周期：所有权、就绪检查与统一清理。"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Sequence

from .config import DataPaths
from .processes import JobOwnerRegistry

LOGGER = logging.getLogger("behaviourlab.service")

READY_ATTEMPTS = 60
READY_TIMEOUT_S = 1.0
READY_INTERVAL_S = 0.1


class ServiceError(RuntimeError):
    """本地服务无法按预期启动或关闭。"""


class AlreadyRunningError(ServiceError):
    """同一数据目录已有运行实例。"""


class ReadinessError(ServiceError):
    """服务进程已启动，但健康检查未确认就绪。"""


def find_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def check_health(url: str, data_root: str | Path) -> tuple[bool, str]:
    """最多 60 次请求 /api/health，核对 data_root 完全一致。"""

    expected = str(Path(data_root).resolve())
    last_error = ""
    endpoint = url.rstrip("/") + "/api/health"
    for _ in range(READY_ATTEMPTS):
        try:
            with urllib.request.urlopen(endpoint, timeout=READY_TIMEOUT_S) as response:
                payload = json.loads(response.read().decode("utf-8"))
            data = payload.get("data") or {}
            if payload.get("ok") is True and data.get("status") == "ok":
                returned = data.get("data_root")
                if returned and str(Path(returned).resolve()) == expected:
                    return True, ""
                last_error = f"健康检查返回的数据目录不一致：{returned}"
            else:
                last_error = "健康检查未返回 ok"
        except Exception as error:  # 连接拒绝、超时、JSON 异常
            last_error = f"{type(error).__name__}: {error}"
        time.sleep(READY_INTERVAL_S)
    return False, f"服务在 {READY_ATTEMPTS} 次检查后仍未就绪（{last_error}）"


def _default_server_factory(app: Any, host: str, port: int, threads: int) -> Any:
    from waitress import create_server

    return create_server(app, host=host, port=port, threads=threads)


class LocalService:
    def __init__(
        self,
        app: Any,
        paths: DataPaths,
        *,
        port: int | None = None,
        host: str = "127.0.0.1",
        threads: int = 6,
        server_factory: Callable[[Any, str, int, int], Any] | None = None,
        ready_probe: Callable[[str, str | Path], tuple[bool, str]] | None = None,
        owner_registry: JobOwnerRegistry | None = None,
        stop_callbacks: Sequence[Callable[[], None]] | None = None,
    ) -> None:
        self.app = app
        self.paths = paths
        self.host = host
        self.port = int(port) if port else find_free_port(host)
        self.threads = threads
        self.url = f"http://{host}:{self.port}"
        self._server_factory = server_factory or _default_server_factory
        self._ready_probe = ready_probe or check_health
        self._owner = owner_registry or JobOwnerRegistry(paths)
        self._stop_callbacks = list(stop_callbacks or [])
        self._server: Any = None
        self._thread: threading.Thread | None = None
        self._owner_acquired = False
        self._stopped = False
        self._lock = threading.RLock()

    def start(self) -> str:
        with self._lock:
            if self._server is not None:
                return self.url
            self._stopped = False
            record = self._owner.read()
            if record and self._owner.is_active():
                pid = record.get("pid")
                raise AlreadyRunningError(
                    f"数据目录已有运行实例（PID {pid}），请先关闭后再启动。"
                )

            self._owner.acquire(pid=os.getpid())
            self._owner_acquired = True
            try:
                server = self._server_factory(
                    self.app, self.host, self.port, self.threads
                )
                self._server = server
                self._thread = threading.Thread(
                    target=server.run,
                    name="behaviourlab-http",
                    daemon=True,
                )
                self._thread.start()
                ready, detail = self._ready_probe(self.url, self.paths.root)
                if not ready:
                    raise ReadinessError(detail or "本地服务未就绪")
            except BaseException:
                self.stop()
                raise
        LOGGER.info("本地服务已就绪：%s", self.url)
        return self.url

    def wait(self) -> None:
        """浏览器模式：服务保持运行，直到用户中断。"""

        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            return

    def stop(self) -> None:
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            for callback in self._stop_callbacks:
                try:
                    callback()
                except Exception:
                    LOGGER.exception("停止任务/工作进程失败")
            server = self._server
            self._server = None
            if server is not None:
                try:
                    server.close()
                except Exception:
                    LOGGER.exception("关闭本地服务失败")
            thread = self._thread
            self._thread = None
            if thread is not None and thread.is_alive():
                thread.join(timeout=3.0)
            if self._owner_acquired:
                self._owner.release()
                self._owner_acquired = False
            LOGGER.info("本地服务已停止")
