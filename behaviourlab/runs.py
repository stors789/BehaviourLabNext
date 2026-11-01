"""已发布运行的读取边界：清单、声明文件与打包。

只引用该 run 自己声明过的文件；路径解析后必须仍在本 run 目录内。历史 run 的
嵌套布局（``runs/{run_id}/{job_id}``）按其自身清单读取，不改写、不合并。
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any, Mapping

from .config import DataPaths
from .store import Store, read_json_file

MANIFEST_NAME = "result_manifest.json"
FROZEN_NAME = "frozen_job.json"


def _run_dir_candidates(paths: DataPaths, run_id: str) -> list[Path]:
    root = Path(paths.runs) / run_id
    return [root]


def _legacy_dir(store: Store, paths: DataPaths, run_id: str) -> Path | None:
    """旧记录：run_id 下有 per-job 子目录，清单在子目录里。"""

    base = Path(paths.runs) / run_id
    if not base.is_dir():
        return None
    for job in store.list_jobs(run_id):
        candidate = base / str(job.get("id"))
        if (candidate / MANIFEST_NAME).is_file():
            return candidate
    for child in sorted(item for item in base.iterdir() if item.is_dir()):
        if (child / MANIFEST_NAME).is_file():
            return child
    return None


def manifest_issues(directory: Path, manifest: Mapping[str, Any],
                    run_id: str, *, require_status: bool = True,
                    job_id: str | None = None) -> list[str]:
    """清单有效性：身份一致、显式成功状态、声明文件都在本目录内。

    ``require_status`` 对**新布局**为真：失败或缺状态的清单不能因为目录存在
    就被当成已发布结果。旧嵌套布局由可信记录核对时允许缺状态（见调用方），
    但显式 ``failed`` 永远不是成功。
    """

    issues: list[str] = []
    if manifest.get("run_id") != run_id:
        issues.append("清单 run_id 与目录不一致")
    manifest_job = manifest.get("job_id")
    if not isinstance(manifest_job, str) or not manifest_job:
        issues.append("清单缺少 job_id")
    elif job_id is not None and manifest_job != job_id:
        issues.append("清单 job_id 与记录不一致")
    status = manifest.get("status")
    if status is not None and status != "completed":
        issues.append(f"清单状态不是 completed：{status}")
    elif status is None and require_status:
        issues.append("清单没有声明完成状态")
    declared_files = manifest.get("files")
    if not isinstance(declared_files, list) or not declared_files:
        issues.append("清单没有声明任何文件")
        return issues
    for name in declared_files:
        if not isinstance(name, str) or not name:
            issues.append("清单包含非法文件名")
            continue
        if safe_file(directory, name) is None:
            issues.append(f"声明文件缺失或不在本 run 目录内：{name}")
    return issues


def _record_job_id(store: Store, run_id: str, directory: Path) -> str | None:
    """可信记录里的任务 ID；旧嵌套布局用它核对目录归属。"""

    for job in store.list_jobs(run_id):
        if directory.name == str(job.get("id")):
            return str(job.get("id"))
    for batch in store.list_batches():
        for video in batch.get("videos") or []:
            if video.get("run_id") == run_id and video.get("job_id"):
                return str(video["job_id"])
    return None


def resolve_run(store: Store, paths: DataPaths, run_id: str) -> dict | None:
    """找到并校验已发布的 run：扁平优先，旧嵌套其次。"""

    directory: Path | None = None
    manifest = None
    for candidate in _run_dir_candidates(paths, run_id):
        if (candidate / MANIFEST_NAME).is_file():
            directory = candidate
            manifest = read_json_file(candidate / MANIFEST_NAME, default=None)
            break
    legacy = False
    if not isinstance(manifest, Mapping):
        directory = _legacy_dir(store, paths, run_id)
        manifest = (read_json_file(directory / MANIFEST_NAME, default=None)
                    if directory is not None else None)
        legacy = directory is not None
    if directory is None or not isinstance(manifest, Mapping):
        return None
    record_job = _record_job_id(store, run_id, directory)
    issues = manifest_issues(directory, manifest, run_id,
                             require_status=not legacy, job_id=record_job)
    if issues:
        # 未通过有效性检查：当作未发布，页面与下载都不引用它。
        return None
    frozen = read_json_file(directory / FROZEN_NAME, default=None)
    if not isinstance(frozen, Mapping):
        frozen = {}
    files = [str(item) for item in manifest.get("files") or []
             if isinstance(item, str) and item]
    return {
        "dir": directory,
        "manifest": dict(manifest),
        "job": dict(frozen),
        "files": files,
        "summary": [dict(item) for item in manifest.get("units") or []],
        "run_id": run_id,
        "job_id": manifest.get("job_id"),
        "assay": manifest.get("assay") or frozen.get("assay"),
        "status": manifest.get("status") or "completed",
        "asset": manifest.get("asset") or (frozen.get("asset") or {}),
        "coverage": manifest.get("coverage"),
        "algorithm_version": manifest.get("algorithm_version")
        or frozen.get("algorithm_version"),
        "warnings": list(manifest.get("warnings") or []),
    }


def spec_of(assay: Any) -> dict:
    from . import experiments

    if not experiments.exists(assay):
        return {}
    return dict(experiments.get(assay).SPEC or {})


def declared(manifest: Mapping[str, Any], relative: str) -> bool:
    """只认清单里精确声明的名字。"""

    if not isinstance(relative, str) or not relative:
        return False
    return relative in {str(item) for item in manifest.get("files") or []}


def safe_file(directory: Path, relative: str) -> Path | None:
    """规范解析后仍须位于本 run 目录且是普通文件。"""

    if not relative or relative.startswith(("/", "\\")) or ".." in Path(relative).parts:
        return None
    root = Path(directory).resolve()
    candidate = (root / relative).resolve()
    if candidate == root or root not in candidate.parents:
        return None
    if candidate.is_symlink() or not candidate.is_file():
        return None
    return candidate


def bundle_name(asset_name: Any) -> str:
    raw = str(asset_name or "result")
    stem = Path(raw).stem or "result"
    cleaned = "".join(ch for ch in stem if ch.isalnum() or ch in "-_")[:60]
    return f"BehaviourLabNext-{cleaned or 'result'}.zip"


def write_bundle(target: Path, run: Mapping[str, Any]) -> Path:
    """只打包声明文件，成员名保留相对路径。"""

    directory = Path(run["dir"])
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative in run.get("files") or []:
            path = safe_file(directory, str(relative))
            if path is None:
                continue
            archive.write(path, arcname=str(relative))
    return target
