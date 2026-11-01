"""所属进程与任务所有者记录。

健康检查据此报告 ``job_owner``；不匹配或进程不存在时视为没有任务所有者。
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import threading
from pathlib import Path

from .config import DataPaths
from .schema import new_id
from .store import atomic_write_json, read_json_file, utc_now

OWNER_FILENAME = "job_owner.json"
OWNER_LOCK_FILENAME = "job_owner.lock"
CREATE_NO_WINDOW = 0x08000000
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9

_LOCAL_LOCKS: set[str] = set()
_LOCAL_LOCKS_LOCK = threading.Lock()


class OwnershipError(RuntimeError):
    """数据目录的排他所有权已被其他实例持有。"""


def child_creation_flags(flags: int = 0) -> int:
    """在调用方已有 creationflags 上追加 CREATE_NO_WINDOW。"""

    return int(flags) | CREATE_NO_WINDOW


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


def _lock_handle(handle) -> bool:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    import fcntl

    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def _unlock_handle(handle) -> None:
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


class JobOwnerRegistry:
    """以操作系统文件锁持续持有的所有权；只有持有者可释放记录。

    锁文件放在数据根（``worker.lock``），持有记录放在日志目录；两者都由
    同一个实例持有与释放，调度器和本地服务共享这一份生命周期。
    """

    def __init__(self, paths: DataPaths, *, lock_name: str = "worker.lock",
                 record_name: str = OWNER_FILENAME) -> None:
        self.paths = paths
        self.path = Path(paths.logs) / record_name
        self.lock_path = Path(paths.root) / lock_name
        self._lock_key = os.path.normcase(str(self.lock_path.resolve()))
        self._handle = None
        self._token: str | None = None

    @property
    def held(self) -> bool:
        """本对象当前是否持有这把锁（不判断记录内容）。"""

        return self._handle is not None

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
        if self._handle is not None:
            return self.read() or {}

        with _LOCAL_LOCKS_LOCK:
            if self._lock_key in _LOCAL_LOCKS:
                raise OwnershipError("数据目录已被本进程中的实例占用")
            _LOCAL_LOCKS.add(self._lock_key)

        handle = None
        locked = False
        try:
            handle = self.lock_path.open("a+b")
            if not _lock_handle(handle):
                raise OwnershipError("数据目录已被其他实例占用")
            locked = True
            token = new_id()
            record = {
                "pid": int(pid if pid is not None else os.getpid()),
                "run_id": run_id,
                "token": token,
                "started_at": utc_now(),
            }
            atomic_write_json(self.path, record)
            self._handle = handle
            self._token = token
            return record
        except BaseException:
            if handle is not None:
                if locked:
                    _unlock_handle(handle)
                try:
                    handle.close()
                except OSError:
                    pass
            with _LOCAL_LOCKS_LOCK:
                _LOCAL_LOCKS.discard(self._lock_key)
            raise

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        token = self._token
        self._handle = None
        self._token = None
        try:
            record = self.read()
            if token and record and record.get("token") == token:
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
        finally:
            _unlock_handle(handle)
            try:
                handle.close()
            except OSError:
                pass
            with _LOCAL_LOCKS_LOCK:
                _LOCAL_LOCKS.discard(self._lock_key)


class _IoCounters(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _BasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _ExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class JobObject:
    """Windows Job Object：关闭句柄即结束所属进程树。"""

    def __init__(self) -> None:
        if os.name != "nt":
            raise OSError("Job Object 仅在 Windows 可用")
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = ctypes.c_void_p
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        kernel32.SetInformationJobObject.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32
        ]
        kernel32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        kernel32.TerminateJobObject.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        self._kernel32 = kernel32

        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise OSError(ctypes.get_last_error(), "无法创建 Job Object")
        self._handle = handle

        info = _ExtendedLimitInformation()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            handle,
            JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(info),
            ctypes.sizeof(info),
        ):
            error = ctypes.get_last_error()
            kernel32.CloseHandle(handle)
            self._handle = None
            raise OSError(error, "无法设置 Job Object 限制")

    def assign(self, process: subprocess.Popen) -> None:
        handle = getattr(process, "_handle", None)
        if not handle:
            raise OSError("进程句柄不可用")
        if not self._kernel32.AssignProcessToJobObject(self._handle, handle):
            raise OSError(ctypes.get_last_error(), "无法关联子进程到 Job Object")

    def terminate(self) -> None:
        if self._handle:
            self._kernel32.TerminateJobObject(self._handle, 1)

    def wait(self, timeout_ms: int = 3000) -> bool:
        if not self._handle:
            return True
        return self._kernel32.WaitForSingleObject(self._handle, timeout_ms) == 0

    def close(self) -> None:
        if self._handle:
            self._kernel32.CloseHandle(self._handle)
            self._handle = None


class ManagedProcess:
    """已关联 Job Object 的子进程；停止时先终止进程树。"""

    def __init__(self, process: subprocess.Popen, job: JobObject) -> None:
        self.process = process
        self.job = job

    @property
    def pid(self) -> int:
        return self.process.pid

    def stop(self, timeout_s: float = 3.0) -> None:
        try:
            self.job.terminate()
            try:
                self.process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                self.process.kill()
                try:
                    self.process.wait(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            self.job.close()


def spawn_managed(command: list[str] | str, **kwargs) -> ManagedProcess:
    """启动子进程并关联 Job Object；关联失败时终止刚创建的子进程。"""

    flags = kwargs.pop("creationflags", 0)
    job = JobObject()
    process: subprocess.Popen | None = None
    try:
        process = subprocess.Popen(
            command,
            creationflags=child_creation_flags(flags),
            **kwargs,
        )
        job.assign(process)
    except BaseException:
        if process is not None:
            try:
                process.kill()
                process.wait(timeout=3)
            except Exception:
                pass
        job.close()
        raise
    return ManagedProcess(process, job)
