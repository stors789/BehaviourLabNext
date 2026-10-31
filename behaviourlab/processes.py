"""所属进程与任务所有者记录。

健康检查据此报告 ``job_owner``；不匹配或进程不存在时视为没有任务所有者。
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

from .config import DataPaths
from .store import atomic_write_json, read_json_file, utc_now

OWNER_FILENAME = "job_owner.json"


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid)
        )
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return bool(ok) and code.value == 259  # STILL_ACTIVE
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class JobOwnerRegistry:
    def __init__(self, paths: DataPaths) -> None:
        self.paths = paths
        self.path = Path(paths.logs) / OWNER_FILENAME

    def read(self) -> dict | None:
        try:
            record = read_json_file(self.path, default=None)
        except (OSError, ValueError):
            return None
        if isinstance(record, dict):
            return record
        return None

    def is_active(self) -> bool:
        record = self.read()
        if not record:
            return False
        pid = record.get("pid")
        if not isinstance(pid, int):
            return False
        return pid_alive(pid)

    def acquire(self, *, pid: int | None = None, run_id: str | None = None) -> dict:
        record = {
            "pid": int(pid if pid is not None else os.getpid()),
            "run_id": run_id,
            "started_at": utc_now(),
        }
        atomic_write_json(self.path, record)
        return record

    def release(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
