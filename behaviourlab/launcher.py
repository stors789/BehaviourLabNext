"""启动器：数据根所有权、单一窗口、本地服务与统一清理。

顺序固定：create_app（不启动第二调度器）→ 构造 LocalService → 显式
jobs.start 取得数据根所有权 → 锁的最终释放责任移交整体生命周期 →
本地服务监听与就绪检查 → server.json → 桌面窗口/浏览器等待。
窗口正常返回或启动任一步失败，都走同一异常安全清理链：先停调度、
再排空 HTTP、最后关闭 Store，全部使用者退出后才释放同根锁；清理未
完成时保留可收尾状态（atexit 不注销），不以退出码 0 冒称成功。
"""

from __future__ import annotations

import argparse
import atexit
import ctypes
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from .config import DataPaths
from .service import AlreadyRunningError, LocalService

LOGGER = logging.getLogger("behaviourlab")

LOCK_TAKEN_MESSAGE = ("此数据目录已有运行实例，请关闭原实例或选择其他数据目录")


def configure_logging(paths: DataPaths) -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    if any(getattr(handler, "_behaviourlab", False) for handler in root.handlers):
        return

    try:
        file_handler = logging.FileHandler(
            Path(paths.logs) / "application.log", encoding="utf-8"
        )
        file_handler._behaviourlab = True  # type: ignore[attr-defined]
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError:
        pass

    console = logging.StreamHandler()
    console._behaviourlab = True  # type: ignore[attr-defined]
    console.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(console)


def record_startup_failure(paths: DataPaths, detail: str) -> Path | None:
    """把完整 cause/traceback 追加到 logs/startup.log；不可写时说明权限。"""

    target = Path(paths.logs) / "startup.log"
    try:
        paths.logs.mkdir(parents=True, exist_ok=True)
        existing = target.read_text(encoding="utf-8") if target.exists() else ""
        target.write_text(existing + detail.rstrip("\n") + "\n",
                          encoding="utf-8")
        return target
    except OSError as error:
        LOGGER.error("启动日志写入失败（请检查 logs 目录写入权限）：%s", error)
        LOGGER.error("原始错误：\n%s", detail)
        return None


def show_native_error(message: str) -> None:
    """不依赖 WebView2 的原生错误框；自身失败只补日志，不抛出。"""

    if os.name != "nt":
        return
    try:
        ctypes.windll.user32.MessageBoxW(
            None, message, "BehaviourLabNext", 0x00000010
        )
    except Exception:
        LOGGER.exception("原生错误框显示失败")


def _write_stream(stream: Any, text: str) -> None:
    """stdout/stderr 不存在或不可写时不再因 print 崩溃。"""

    try:
        if stream is None:
            return
        stream.write(text + "\n")
        stream.flush()
    except Exception:
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="behaviourlab", description="BehaviourLabNext")
    parser.add_argument("--data-dir", default=None,
                        help="显式数据根；默认使用本地应用数据目录")
    parser.add_argument("--port", type=int, default=0,
                        help="监听端口；0 表示自动申请空闲端口")
    parser.add_argument("--browser", action="store_true",
                        help="只运行本地服务，不创建桌面窗口")
    return parser


def _record_server(paths: DataPaths, url: str) -> None:
    """就绪后把 {url, data_root, pid} 记录到数据根 server.json。"""

    payload = json.dumps(
        {"url": url, "data_root": str(paths.root), "pid": os.getpid()},
        ensure_ascii=False,
    )
    try:
        (Path(paths.root) / "server.json").write_text(
            payload, encoding="utf-8")
    except OSError:
        LOGGER.exception("记录 server.json 失败")
    _write_stream(sys.stdout, payload)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = DataPaths.from_dir(args.data_dir)
    service: LocalService | None = None
    jobs: Any = None
    store: Any = None

    def cleanup() -> bool:
        """入口统一的异常安全清理链；返回是否完整收尾。

        服务已构造：停调度 → 排空 HTTP → Store.close → 最后释放同根
        锁（最终释放责任在 LocalService）。服务构造失败：只回收已
        创建的 Store/Jobs，绝不触碰其他实例。清理异常只记日志，不
        替换最初的 cause。
        """

        nonlocal service
        if service is not None:
            try:
                stopped = service.stop()
            except Exception:
                # 清理异常（如所有权记录删除失败）作为补充日志处理：
                # 不替换最初的启动异常/cause，也不把未完成当成功。
                LOGGER.exception("服务清理未完成")
                stopped = False
            if stopped:
                atexit.unregister(service.stop)
            return stopped
        stopped = True
        if jobs is not None:
            try:
                if not jobs.shutdown():
                    stopped = False
            except Exception:
                LOGGER.exception("回收调度器失败")
                stopped = False
        if store is not None:
            try:
                store.close()
            except Exception:
                LOGGER.exception("关闭 Store 失败")
                stopped = False
        return stopped

    try:
        paths.ensure()
        configure_logging(paths)
        LOGGER.info("数据目录：%s", paths.root)

        from .web import create_app

        app = create_app(data_dir=str(paths.root))
        jobs = app.config["JOBS"]
        store = app.config["STORE"]
        # 关闭顺序固定：停调度 → 排空 HTTP → 关闭 Store → 释放唯一所有权。
        service = LocalService(
            app, paths, port=args.port,
            owner_registry=app.config["JOB_OWNER"],
            stop_callbacks=[jobs.shutdown],
            final_callbacks=[store.close],
        )
        atexit.register(service.stop)
        # 显式取得数据根所有权：取不到锁就不是第二个分析者，直接报错。
        if not jobs.start():
            raise AlreadyRunningError(LOCK_TAKEN_MESSAGE)
        # 同根锁由整体生命周期持有到最后：shutdown 只收尾，不提前放锁。
        jobs.defer_lock_release()
        service.start()
        LOGGER.info("本地服务：%s", service.url)
        _record_server(paths, service.url)
        stopped = True
        try:
            if args.browser:
                # 只保持服务等待用户中断，不创建桌面窗口/错误框。
                service.wait()
            else:
                from .desktop import open_window

                open_window(service.url)
        finally:
            # stop=False：不注销 atexit 清理，保留可收尾状态。
            stopped = cleanup()
        if not stopped:
            LOGGER.error("关闭未完成：清理已保留，不以 0 冒称成功")
            return 1
        return 0
    except AlreadyRunningError as error:
        LOGGER.error("%s", error)
        _write_stream(sys.stderr, str(error))
        if not args.browser:
            show_native_error(str(error))
        cleanup()
        return 2
    except KeyboardInterrupt:
        if cleanup():
            return 0
        LOGGER.error("中断后清理未完成：清理已保留，不以 0 冒称成功")
        return 1
    except Exception as error:
        detail = traceback.format_exc()
        # 先落盘并展示最初异常；清理异常只补日志，不替换根因。
        LOGGER.error("启动失败：\n%s", detail)
        log_path = record_startup_failure(paths, detail)
        message = f"无法启动：{error}"
        if log_path is not None:
            message += f"\n详细信息已写入：{log_path}"
        else:
            message += "\n启动日志写入失败，请检查数据目录 logs 的写入权限。"
        _write_stream(sys.stderr, message)
        if not args.browser:
            show_native_error(message)
        if not cleanup():
            LOGGER.error("退出时清理未完成，已保留可收尾状态")
        return 1


if __name__ == "__main__":
    sys.exit(main())
