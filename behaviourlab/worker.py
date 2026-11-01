"""任务工作进程：读冻结任务、写进度、导出结果。

工作进程只做一次分析：不创建 Web/GUI，也不启动另一个调度器。父进程通过
``progress.json`` 读取进度，通过返回码判断成败，通过结果清单判断发布内容。
同一函数在后台子进程与测试内的同步执行中都使用，行为只有一份。
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping

from . import outputs
from .store import atomic_write_json, read_json_file

PROGRESS_INTERVAL_S = 0.2
CAPTURE_CEILING = 0.75


class _ProgressWriter:
    """节流写进度：最多每 .2 秒落一次原子 JSON。"""

    def __init__(self, path: Path, *, interval_s: float = PROGRESS_INTERVAL_S,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.path = Path(path)
        self.interval_s = float(interval_s)
        self._clock = clock
        self._last = 0.0
        self._last_fraction = -1.0
        self._closed = False

    def __call__(self, fraction: Any, message: str = "") -> None:
        if self._closed:
            return
        try:
            value = float(fraction)
        except (TypeError, ValueError):
            value = 0.0
        if value != value or value in (float("inf"), float("-inf")):
            value = 0.0
        value = min(1.0, max(0.0, value))
        now = self._clock()
        if value < 1.0 and now - self._last < self.interval_s:
            return
        self._last = now
        self._last_fraction = value
        self._write({"fraction": round(value, 6), "message": str(message or "")})

    def error(self, message: str) -> None:
        if self._closed:
            return
        self._write({"fraction": self._last_fraction if self._last_fraction > 0
                     else 0.0, "message": "分析失败", "error": str(message)})

    def close(self) -> None:
        self._closed = True

    def _write(self, payload: Mapping[str, Any]) -> None:
        try:
            atomic_write_json(self.path, dict(payload))
        except OSError:
            # 进度写不进去不能中断分析本身；失败原因随后仍在 worker.log。
            pass


def worker_command(job_path: str | os.PathLike[str],
                   stage_dir: str | os.PathLike[str],
                   progress_path: str | os.PathLike[str]) -> list[str]:
    """独立的子进程命令：每个路径都是单独参数，不经过 shell 拼接。"""

    return [sys.executable, "-m", "behaviourlab.worker",
            str(job_path), str(stage_dir), str(progress_path)]


def run_frozen_job(resolved: Mapping[str, Any], *,
                   progress: Callable[[float, str], None] | None = None,
                   cancel: Callable[[], bool] | None = None) -> Any:
    """执行一个已冻结任务；身份不符直接拒绝，不返回假结果。"""

    from . import engine

    return engine.run_job(resolved, progress=progress, cancel=cancel)


def _merge_manifest(manifest: Mapping[str, Any], frozen: Mapping[str, Any],
                    files: list[str]) -> dict:
    """清单与冻结任务去重合并：同一文件只登记一次。"""

    merged = dict(manifest)
    merged["run_id"] = frozen.get("run_id")
    merged["job_id"] = frozen.get("job_id")
    merged["assay"] = frozen.get("assay")
    merged["algorithm_version"] = frozen.get("algorithm_version")
    order: list[str] = []
    seen: set[str] = set()
    for name in list(manifest.get("files") or []) + list(files) + [
            "frozen_job.json", "result_manifest.json"]:
        if name and name not in seen:
            seen.add(name)
            order.append(name)
    merged["files"] = order
    return merged


def execute_job_file(job_path: str | os.PathLike[str],
                     stage_dir: str | os.PathLike[str],
                     progress_path: str | os.PathLike[str], *,
                     cancel: Callable[[], bool] | None = None,
                     clock: Callable[[], float] = time.monotonic) -> int:
    """读 job.json、写冻结副本与进度、导出结果；0 成功，1 失败。"""

    stage = Path(stage_dir)
    writer = _ProgressWriter(Path(progress_path), clock=clock)
    try:
        frozen = read_json_file(job_path)
        if not isinstance(frozen, Mapping):
            raise ValueError("job.json 不是对象")
        stage.mkdir(parents=True, exist_ok=True)
        atomic_write_json(stage / "frozen_job.json", dict(frozen))
        writer(0.0, "开始分析")
        result = run_frozen_job(frozen, progress=writer, cancel=cancel)
        if cancel is not None and cancel():
            raise RuntimeError("分析已取消")
        manifest = outputs.export_result(stage, frozen, result,
                                         progress=writer)
        merged = _merge_manifest(manifest, frozen, list(manifest.get("files") or []))
        atomic_write_json(stage / "result_manifest.json", merged)
        writer(1.0, "已完成")
        return 0
    except BaseException as error:  # 任何失败都必须让父进程看到原因
        traceback.print_exc()
        writer.error(f"{type(error).__name__}: {error}")
        return 1
    finally:
        writer.close()


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 3:
        print("用法：python -m behaviourlab.worker JOB STAGE PROGRESS",
              file=sys.stderr)
        return 2
    return execute_job_file(args[0], args[1], args[2])


if __name__ == "__main__":  # pragma: no cover - 子进程入口
    sys.exit(main())
