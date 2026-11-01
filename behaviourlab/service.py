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
from .processes import JobOwnerRegistry, OwnershipError

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
        stop_callbacks: Sequence[Callable[[], object]] | None = None,
        final_callbacks: Sequence[Callable[[], object]] | None = None,
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
        # 先停任务、再排空 HTTP、最后收尾（例如关闭 Store），顺序不可交换。
        self._stop_callbacks = list(stop_callbacks or [])
        self._final_callbacks = list(final_callbacks or [])
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

            try:
                self._owner.acquire(pid=os.getpid())
            except OwnershipError as error:
                record = self._owner.read() or {}
                pid = record.get("pid")
                detail = f"（PID {pid}）" if isinstance(pid, int) else ""
                raise AlreadyRunningError(
                    f"数据目录已有运行实例{detail}，请先关闭后再启动。"
                ) from error
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

    def _shutdown_server(self, server: Any) -> bool:
        """关闭监听并尽量排空所属连接；返回是否确认排空。"""

        complete = True
        try:
            server.close()
        except Exception:
            LOGGER.exception("关闭本地服务监听失败")
            complete = False

        channels = getattr(server, "active_channels", None)
        if isinstance(channels, dict):
            for channel in list(channels.values()):
                try:
                    channel.close()
                except Exception:
                    LOGGER.exception("关闭所属连接失败")

        dispatcher = getattr(server, "task_dispatcher", None)
        if dispatcher is not None and hasattr(dispatcher, "shutdown"):
            try:
                dispatcher.shutdown(cancel_pending=True, timeout=3.0)
            except TypeError:
                dispatcher.shutdown()
            except Exception:
                LOGGER.exception("关闭服务调度线程失败")
                complete = False

        remaining = getattr(server, "active_channels", None)
        if isinstance(remaining, dict) and remaining:
            # 仍有连接在用数据库：这是“尚在关闭”，不能当作完成。
            complete = False
        return complete

    def stop(self) -> None:
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            complete = True
            for callback in self._stop_callbacks:
                try:
                    result = callback()
                    if result is False:
                        complete = False
                except Exception:
                    LOGGER.exception("停止任务/工作进程失败")
                    complete = False
            server = self._server
            self._server = None
            if server is not None:
                if not self._shutdown_server(server):
                    complete = False
            thread = self._thread
            self._thread = None
            if thread is not None and thread.is_alive():
                thread.join(timeout=3.0)
                if thread.is_alive():
                    LOGGER.warning("服务线程在 3 秒内未退出")
                    complete = False
            for callback in self._final_callbacks:
                try:
                    result = callback()
                    if result is False:
                        complete = False
                except Exception:
                    LOGGER.exception("关闭收尾失败")
                    complete = False
            if self._owner_acquired:
                if complete:
                    self._owner.release()
                    self._owner_acquired = False
                else:
                    # 未完成关闭：保留数据根所有权，等旧线程真正退出。
                    LOGGER.warning("关闭未完成，保留数据根所有权")
            LOGGER.info("本地服务已停止（complete=%s）", complete)
