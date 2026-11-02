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
HTTP_DRAIN_TIMEOUT_S = 5.0


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
    """创建 Waitress 服务；部分构造/bind 失败时收掉自己创建的 dispatcher。

    Waitress 在 bind 之前就创建 ``ThreadedTaskDispatcher`` 并启动工作
    线程：端口被占等 bind 失败会把这些线程留在半构造对象里。这里由
    工厂自己持有 dispatcher 并在失败时 shutdown 回收。
    """

    from waitress import create_server
    from waitress.task import ThreadedTaskDispatcher

    dispatcher = ThreadedTaskDispatcher()
    dispatcher.set_thread_count(threads)
    try:
        return create_server(
            app,
            host=host,
            port=port,
            threads=threads,
            _dispatcher=dispatcher,
            clear_untrusted_proxy_headers=True,
        )
    except BaseException:
        try:
            dispatcher.shutdown(cancel_pending=True, timeout=1.0)
        except Exception:
            LOGGER.exception("回收 Waitress dispatcher 失败")
        raise


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
        self._http_drain_timeout = HTTP_DRAIN_TIMEOUT_S
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

    def _drain_http(self, server: Any) -> bool:
        """关监听、关连接，并等待真实 HTTP 工作线程退出。

        ``active_channels`` 清空不等于工作线程已退出：阻塞中的任务不在
        channels 里。只有 dispatcher 的 ``threads`` 集合清空，才能确认
        没有线程还在使用数据库。返回是否确认排空。
        """

        complete = self._shutdown_server(server)
        dispatcher = getattr(server, "task_dispatcher", None)
        threads = getattr(dispatcher, "threads", None)
        if isinstance(threads, (set, frozenset)) and threads:
            deadline = time.monotonic() + float(self._http_drain_timeout)
            while threads:
                if time.monotonic() >= deadline:
                    LOGGER.warning("HTTP 工作线程 %.1f 秒内未退出（%d 个）",
                                   float(self._http_drain_timeout),
                                   len(threads))
                    return False
                time.sleep(0.05)
        return complete

    def stop(self) -> bool:
        """统一清理：任务收尾 → HTTP 排空 → Store.close → 释放所有权。

        返回是否完成全部收尾。任务未停完或仍有工作线程在使用数据时，
        不关闭 Store、不释放所有权、不丢弃引用，也不进入最终停止状态；
        解除阻塞后可再次调用 ``stop()`` 继续收尾。
        """

        with self._lock:
            if self._stopped:
                return True
            complete = True
            # 1. 停任务与工作进程；False/异常 = 未在时限内收尾。
            for callback in self._stop_callbacks:
                try:
                    result = callback()
                    if result is False:
                        complete = False
                except Exception:
                    LOGGER.exception("停止任务/工作进程失败")
                    complete = False
            # 2. 排空真实 HTTP 使用者：监听、连接与 dispatcher 线程。
            server = self._server
            if server is not None and not self._drain_http(server):
                complete = False
            thread = self._thread
            if thread is not None and thread.is_alive():
                thread.join(timeout=3.0)
                if thread.is_alive():
                    LOGGER.warning("服务线程在 3 秒内未退出")
                    complete = False
            if not complete:
                # 仍被使用的 Store 不能关：保留引用与所有权，允许再次关闭。
                LOGGER.warning("关闭未完成，保留引用与数据根所有权")
                return False
            # 3. 所有使用者已退出，才执行收尾回调（例如 Store.close）。
            for callback in self._final_callbacks:
                try:
                    result = callback()
                    if result is False:
                        complete = False
                except Exception:
                    LOGGER.exception("关闭收尾失败")
                    complete = False
            if not complete:
                return False
            # 4. 全部使用者退出后才释放数据根所有权。锁由整体生命周期
            # 持有：无论锁是本服务还是同注册表的调度器取得的，最终
            # 释放责任都在这里（Store 关闭之后）；注册表未被本实例
            # 持有时 held 为 False，不会碰其他实例的锁。
            if self._owner.held:
                self._owner.release()
            self._owner_acquired = False
            self._server = None
            self._thread = None
            self._stopped = True
            LOGGER.info("本地服务已停止")
            return True
