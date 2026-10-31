"""启动器：单一窗口、本地服务与统一清理。"""

from __future__ import annotations

import argparse
import atexit
import ctypes
import logging
import os
import sys
import traceback
from pathlib import Path

from .config import DataPaths
from .service import AlreadyRunningError, LocalService

LOGGER = logging.getLogger("behaviourlab")


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


def record_startup_failure(paths: DataPaths, detail: str) -> None:
    try:
        paths.logs.mkdir(parents=True, exist_ok=True)
        (Path(paths.logs) / "startup.log").write_text(detail, encoding="utf-8")
    except OSError:
        LOGGER.error("无法写入 startup.log；原始错误：\n%s", detail)


def show_native_error(message: str) -> None:
    if os.name != "nt":
        return
    try:
        ctypes.windll.user32.MessageBoxW(
            None, message, "BehaviourLabNext", 0x00000010
        )
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = DataPaths.from_dir(args.data_dir)

    try:
        paths.ensure()
        configure_logging(paths)
        LOGGER.info("数据目录：%s", paths.root)

        from .web import create_app

        app = create_app(data_dir=str(paths.root))
        service = LocalService(app, paths, port=args.port)
        atexit.register(service.stop)
        service.start()
        LOGGER.info("本地服务：%s", service.url)
        try:
            if args.browser:
                service.wait()
            else:
                from .desktop import open_window

                open_window(service.url)
        finally:
            service.stop()
        return 0
    except AlreadyRunningError as error:
        LOGGER.error("%s", error)
        return 2
    except KeyboardInterrupt:
        return 0
    except Exception as error:
        detail = traceback.format_exc()
        LOGGER.error("启动失败：\n%s", detail)
        record_startup_failure(paths, detail)
        if not args.browser:
            show_native_error(f"启动失败：{error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
