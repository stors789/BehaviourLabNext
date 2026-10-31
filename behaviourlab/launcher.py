"""启动器：本地服务、端口选择与关闭。

默认只启动本地 HTTP 服务；桌面窗口仅在显式传入 --window 时创建。
"""

from __future__ import annotations

import argparse
import logging
import socket
import sys
import threading

from .config import DEFAULT_HOST, DEFAULT_PORT, DataPaths
from .web import create_app

LOGGER = logging.getLogger("behaviourlab")


def find_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="behaviourlab", description="BehaviourLabNext")
    parser.add_argument("--data-dir", default=None, help="显式数据根；默认使用本地应用数据目录")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help="监听端口；0 表示自动选择")
    parser.add_argument("--window", action="store_true", help="启动桌面窗口")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    paths = DataPaths.from_dir(args.data_dir)
    app = create_app(data_dir=str(paths.root))

    host = args.host
    port = args.port if args.port else find_free_port(host)
    url = f"http://{host}:{port}"
    LOGGER.info("数据目录：%s", paths.root)
    LOGGER.info("本地服务：%s", url)

    if args.window:
        from waitress import serve

        from .desktop import open_window

        server = threading.Thread(
            target=serve,
            kwargs={"app": app, "host": host, "port": port, "threads": 8, "_quiet": True},
            daemon=True,
        )
        server.start()
        try:
            open_window(url)
        except Exception as error:  # 窗口不可用时保留服务提示
            LOGGER.error("无法创建桌面窗口：%s", error)
            return 1
        return 0

    from waitress import serve

    try:
        serve(app, host=host, port=port, threads=8, _quiet=True)
    except KeyboardInterrupt:
        LOGGER.info("已停止")
    return 0


if __name__ == "__main__":
    sys.exit(main())
