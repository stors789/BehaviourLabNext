"""运行环境分派：区分当前 Python 解释器与冻结 EXE 的工作进程命令。

调度器用 ``worker_command`` 启动后台分析进程：源码模式走
``python -m behaviourlab.worker``；冻结 EXE 模式走同一可执行文件加
私有参数 ``--behaviorlab-worker``，确保 EXE 后台 worker 不会递归
启动 GUI、数据库或 HTTP。每个路径都是单独的 ``str`` 参数，保留
Unicode 与空格，不经过 shell 拼接。
"""

from __future__ import annotations

import os
import sys

WORKER_FLAG = "--behaviorlab-worker"


def worker_command(job_file: str | os.PathLike[str],
                   output_dir: str | os.PathLike[str],
                   progress_file: str | os.PathLike[str]) -> list[str]:
    """后台工作进程命令；冻结模式复用同一 EXE 加私有参数分派。

    源码模式返回 6 项（解释器、``-m``、模块名、三个路径）；``sys.frozen``
    为真时返回 5 项（EXE、私有 flag、三个路径），绝不把 EXE 当 Python
    使用 ``-m``。
    """

    job = str(job_file)
    output = str(output_dir)
    progress = str(progress_file)
    if getattr(sys, "frozen", False):
        return [sys.executable, WORKER_FLAG, job, output, progress]
    return [sys.executable, "-m", "behaviourlab.worker",
            job, output, progress]
