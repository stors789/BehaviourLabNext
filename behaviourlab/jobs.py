"""本地唯一所有者顺序任务调度。

一个批次（job_id）按草稿顺序逐个视频执行，每个视频有独立 ``run_id``；任何
视频失败立即结束整批，后续视频保持未开始。任务只在完整成功后才原子发布到
``runs/{run_id}``，临时目录不会当作结果。进程内的调度线程只是所有权持有者
的一部分：没有 ``worker.lock`` 的实例不碰别人的记录，也不释放别人的锁。
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import engine, worker
from .config import ALGORITHM_VERSION, DataPaths
from .processes import JobOwnerRegistry, OwnershipError, spawn_managed
from .schema import ValidationError, is_valid_id, new_id
from .store import (
    NotFoundError,
    Store,
    atomic_write_json,
    read_json_file,
    utc_now,
)

LOGGER = logging.getLogger("behaviourlab.jobs")

LOCK_FILENAME = "worker.lock"
POLL_INTERVAL_S = 0.1
PROGRESS_WRITE_INTERVAL_S = 1.0

QUEUED = "queued"
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"
CANCELLED = "cancelled"
INTERRUPTED = "interrupted"
NOT_STARTED = "not_started"

TERMINAL = (COMPLETED, FAILED, CANCELLED, INTERRUPTED)
ACTIVE = (QUEUED, RUNNING)

QUEUED_MESSAGE = "排队中"
INTERRUPTED_MESSAGE = "上次运行被中断"
CANCELLED_MESSAGE = "已停止"

_SAFE_NAME = re.compile(r"[A-Za-z0-9_-]+")


class JobError(RuntimeError):
    """任务操作被拒绝；``fields`` 可直接给界面定位。"""

    def __init__(self, message: str, *, status: int = 400,
                 fields: list[dict] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.fields = fields or []


class JobConflict(JobError):
    """同一任务的并发操作冲突。"""

    def __init__(self, message: str, *, fields: list[dict] | None = None) -> None:
        super().__init__(message, status=409, fields=fields)


def _now() -> str:
    return utc_now()


def _fraction(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number != number or number in (float("inf"), float("-inf")):
        return 0.0
    return min(1.0, max(0.0, number))


def _video_status(value: Any) -> str:
    return str(value) if value in (QUEUED, RUNNING, COMPLETED, FAILED,
                                   CANCELLED, INTERRUPTED, NOT_STARTED) else QUEUED


def _batch_progress(videos: Sequence[Mapping[str, Any]], index: int,
                    fraction: float, message: str) -> dict:
    total = max(1, len(videos))
    current = index + 1 if videos else 0
    return {
        "fraction": round(_fraction((index + _fraction(fraction)) / total), 6),
        "current": current,
        "total": len(videos),
        "message": message,
    }


def _manifest_ok(directory: Path, job_id: str, run_id: str, *,
                 require_status: bool = True,
                 tolerate_missing: bool = False) -> dict | None:
    """已发布清单：身份一致、显式成功、声明文件都在本目录内。

    新布局与恢复共用这一判据（``require_status=True``）；只有旧嵌套记录经
    可信 job 记录核对时才允许缺状态，显式 failed 永远不算成功。
    ``tolerate_missing`` 只用于删除重试：容许本流程先前已删掉的声明文件
    缺失，仍然核对身份、状态与越界路径，不放宽发布/恢复判据。
    """

    from . import runs as run_service

    try:
        manifest = read_json_file(directory / "result_manifest.json", default=None)
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, Mapping):
        return None
    issues = run_service.manifest_issues(
        directory, manifest, run_id, require_status=require_status,
        job_id=job_id, tolerate_missing=tolerate_missing)
    if issues:
        return None
    return dict(manifest)


def _under(root: Path, candidate: Path) -> bool:
    try:
        root_resolved = root.resolve()
        candidate_resolved = candidate.resolve()
    except OSError:
        return False
    if candidate.is_symlink():
        return False
    return root_resolved == candidate_resolved or root_resolved in candidate_resolved.parents


class JobScheduler:
    """顺序执行器；构造不启动线程，``start`` 才取得所有权。"""

    def __init__(self, store: Store, paths: DataPaths | None = None, *,
                 owner: JobOwnerRegistry | None = None,
                 inline: bool = False,
                 runner: Callable[[Mapping[str, Any]], int] | None = None,
                 poll_interval: float = POLL_INTERVAL_S,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.store = store
        self.paths = paths or store.paths
        self.owner = owner or JobOwnerRegistry(self.paths,
                                               lock_name=LOCK_FILENAME)
        self.inline = bool(inline)
        self._runner = runner
        self._poll = max(0.01, float(poll_interval))
        self._clock = clock
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._stopping = threading.Event()
        self._cancel = threading.Event()
        self._wake = threading.Event()
        self._process: Any = None
        self._started = False
        self._owns_lock = False
        self._release_pending = False
        self._active_batch: str | None = None
        self._active_index = 0
        self._last_progress_write = 0.0

    # -- 生命周期 -------------------------------------------------------
    @property
    def running(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive())

    def start(self) -> bool:
        """取得数据根所有权后恢复旧运行并启动单一调度线程；失败返回 False。"""

        with self._lock:
            if self._started and self.running:
                return True
            if self._started and not self.running:
                # 上一次关闭超时后线程已退出（并可能已释放锁）：先重置状态，
                # 再真正重新持锁启动，不能空返 True 却没有调度线程。
                self._started = False
                self._release_pending = False
                if not self.owner.held:
                    self._owns_lock = False
                LOGGER.warning("调度器状态已重置：上次关闭未完成但线程已退出")
            held_before = self.owner.held
            try:
                self.owner.acquire(pid=os.getpid())
            except OwnershipError:
                return False
            self._owns_lock = self.owner.held and not held_before
            self._started = True
            self._stopping.clear()
            self._cancel.clear()
        try:
            self._recover()
        except Exception:  # 恢复失败不能挡住新的调度
            LOGGER.exception("恢复上次运行状态失败")
        thread = threading.Thread(target=self._loop, name="behaviourlab-jobs",
                                  daemon=True)
        self._thread = thread
        thread.start()
        return True

    def shutdown(self, *, timeout_s: float = 5.0) -> bool:
        """停止调度并把在跑的视频标为中断。

        返回是否**真正完成**（线程退出、锁已释放）；超时只表示尚在关闭，
        此时继续持有数据根所有权，绝不能让旧线程在放锁后继续使用数据库。
        未持锁对象重复调用无影响。
        """

        with self._lock:
            if not self._started and self._thread is None:
                return True
            self._stopping.set()
        self._wake.set()
        self._cancel.set()
        self._stop_active_process()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout_s)
        still_alive = thread is not None and thread.is_alive()
        with self._lock:
            if still_alive:
                # 线程仍在用数据库：先不放锁，等它自己退出后再释放。
                self._release_pending = True
                LOGGER.warning("调度线程在 %.1f 秒内未退出，暂不释放数据根所有权",
                               timeout_s)
                return False
            self._thread = None
            self._started = False
            self._release_lock()
        self._stopping.clear()
        self._cancel.clear()
        return True

    def _release_lock(self) -> None:
        self._release_pending = False
        if self._owns_lock:
            self.owner.release()
            self._owns_lock = False

    # -- 查询 -----------------------------------------------------------
    def get(self, batch_id: str) -> dict:
        batch = self.store.get_batch(batch_id)
        if batch is not None:
            return self._decorate(batch)
        legacy = self._legacy_entry(batch_id)
        if legacy is not None:
            return legacy
        raise NotFoundError("任务不存在")

    def list(self, project_id: str | None = None) -> list[dict]:
        rows = [self._decorate(batch)
                for batch in self.store.list_batches(project_id)]
        rows.extend(self._legacy_entries(project_id))
        rows.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
        return rows

    def _decorate(self, batch: dict) -> dict:
        item = dict(batch)
        item["job_id"] = batch.get("id")
        item["legacy"] = False
        videos = []
        for video in batch.get("videos") or []:
            entry = dict(video)
            entry["job_id"] = batch.get("id")
            entry.setdefault("progress", 0.0)
            videos.append(entry)
        item["videos"] = videos
        item["run_id"] = next((video.get("run_id") for video in videos
                               if video.get("status") == COMPLETED), None)
        item["assay_label"] = self._label(batch.get("assay"))
        return item

    @staticmethod
    def _label(assay: Any) -> str:
        from . import experiments

        if not experiments.exists(assay):
            return str(assay or "").upper()
        return str(experiments.get(assay).SPEC.get("label") or assay)

    def _legacy_entries(self, project_id: str | None = None) -> list[dict]:
        groups: dict[str, list[dict]] = {}
        for row in self.store.list_legacy_jobs():
            if project_id is not None and row.get("project_id") != project_id:
                continue
            groups.setdefault(str(row.get("run_id")), []).append(row)
        entries = [self._legacy_entry(run_id) for run_id in groups]
        return [entry for entry in entries if entry is not None]

    def _legacy_entry(self, run_id: str) -> dict | None:
        rows = [row for row in self.store.list_legacy_jobs()
                if row.get("run_id") == run_id]
        if not rows:
            return None
        videos = []
        for row in sorted(rows, key=lambda item: str(item.get("created_at"))):
            job_id = row.get("id")
            manifest = _manifest_ok(self.paths.runs / run_id / job_id,
                                    job_id, run_id, require_status=False)
            status = {COMPLETED: COMPLETED, "done": COMPLETED,
                      "error": FAILED}.get(str(row.get("status")), INTERRUPTED)
            if status == COMPLETED and manifest is None:
                # 旧状态 done 但清单不可核对：不能当作已完成。
                status = FAILED
            frozen = (row.get("config") or {}).get("job") or {}
            asset = frozen.get("asset") or {}
            videos.append({
                "asset_id": asset.get("id"),
                "name": asset.get("name") or row.get("assay"),
                "run_id": run_id,
                "job_id": job_id,
                "status": status,
                "progress": 1.0 if status == COMPLETED else 0.0,
                "job": frozen or None,
                "result_manifest": manifest,
                "error": None if status == COMPLETED else (
                    "结果不完整" if row.get("status") == "done" else "分析失败"),
                "output_dir": str(self.paths.runs / run_id / job_id)
                if status == COMPLETED else None,
            })
        if all(video["status"] == COMPLETED for video in videos):
            status = COMPLETED
        elif any(video["status"] == RUNNING for video in videos):
            status = INTERRUPTED
        else:
            status = FAILED
        created = min((str(row.get("created_at")) for row in rows), default=None)
        updated = max((str(row.get("updated_at")) for row in rows), default=None)
        return {
            "id": run_id,
            "job_id": run_id,
            "legacy": True,
            "draft_id": rows[0].get("draft_id"),
            "project_id": rows[0].get("project_id"),
            "assay": rows[0].get("assay"),
            "assay_label": self._label(rows[0].get("assay")),
            "status": status,
            "videos": videos,
            "progress": {"fraction": 1.0 if status == COMPLETED else 0.0,
                         "current": len(videos), "total": len(videos),
                         "message": "已完成" if status == COMPLETED else "失败"},
            "cancel_requested": False,
            "owned_run_ids": [run_id],
            "error": None if status == COMPLETED else "旧记录结果不完整",
            "started_at": created,
            "finished_at": updated,
            "created_at": created,
            "updated_at": updated,
            "run_id": next((video["run_id"] for video in videos
                            if video["status"] == COMPLETED), None),
        }

    # -- 提交 -----------------------------------------------------------
    def _project_animals(self, draft: Mapping[str, Any]) -> set[str] | None:
        project_id = draft.get("project_id")
        if project_id is None:
            return None
        return {item["id"] for item in self.store.list_animals(project_id)}

    def submit(self, draft_id: str) -> dict:
        """校验整份草稿与每个视频的冻结任务，然后排入批次。"""

        if not is_valid_id(draft_id):
            raise JobError("草稿 ID 无效", fields=[
                {"field": "draft_id", "message": "ID 无效"}])
        draft = self.store.get_draft(draft_id)
        if draft is None:
            raise NotFoundError("草稿不存在")
        videos = draft.get("videos")
        if not isinstance(videos, list) or not videos:
            raise JobError("草稿还没有视频", fields=[
                {"field": "videos", "message": "草稿还没有视频"}])

        animals = self._project_animals(draft)
        issues = engine.validate_draft(draft, complete=True, animals=animals,
                                       settings={})
        batch_id = new_id()
        prepared: list[dict] = []
        resolved_jobs: list[dict] = []
        for index, video in enumerate(videos):
            if not isinstance(video, Mapping):
                continue
            asset_id = video.get("asset_id")
            run_id = new_id()
            asset = (self.store.get_asset(asset_id)
                     if is_valid_id(asset_id) else None)
            if asset is None:
                issues.append({"asset_id": asset_id, "field": "asset_id",
                               "message": "素材不存在"})
                continue
            try:
                resolved = engine.build_resolved_job(
                    draft=draft, video=video, asset=asset, job_id=batch_id,
                    run_id=run_id, settings={})
            except ValidationError as error:
                entry = error.as_issue()
                entry.setdefault("asset_id", asset_id)
                issues.append(entry)
                continue
            job_issues = engine.validate_job(resolved, animals=animals)
            if job_issues:
                for issue in job_issues:
                    entry = dict(issue)
                    entry.setdefault("asset_id", asset_id)
                    entry["message"] = f"{asset.get('name') or '视频'}：{entry['message']}"
                    issues.append(entry)
                continue
            resolved_jobs.append(resolved)
            prepared.append({
                "asset_id": asset.get("id"),
                "name": asset.get("name"),
                "run_id": run_id,
                "job_id": batch_id,
                "status": QUEUED,
                "progress": 0.0,
                "job": resolved,
                "result_manifest": None,
                "error": None,
            })

        if issues:
            raise JobError("草稿未通过校验", fields=issues)
        if not prepared:
            raise JobError("草稿还没有视频", fields=[
                {"field": "videos", "message": "草稿还没有视频"}])

        batch = self.store.create_batch(
            assay=draft.get("assay"),
            videos=prepared,
            draft_id=draft_id,
            project_id=draft.get("project_id"),
            batch_id=batch_id,
        )
        self._wake.set()
        if self.inline:
            # 测试内的同步执行仍走真实的批次状态：先进入 running 再跑。
            self._start_batch(batch_id)
            self._run_batch(batch_id)
        return self.get(batch_id)

    # -- 取消 -----------------------------------------------------------
    def cancel(self, batch_id: str) -> dict:
        with self.store.mutate() as connection:
            record = self.store.batch_in(connection, batch_id)
            if record is None:
                raise NotFoundError("任务不存在")
            status = record.get("status")
            if status in TERMINAL:
                return self._decorate(record)
            if status == QUEUED:
                videos = [dict(video) for video in record.get("videos") or []]
                for video in videos:
                    if video.get("status") in (QUEUED, RUNNING):
                        video["status"] = NOT_STARTED
                record = self.store.patch_batch(connection, batch_id, {
                    "status": CANCELLED,
                    "videos": videos,
                    "cancel_requested": True,
                    "error": CANCELLED_MESSAGE,
                    "finished_at": _now(),
                    "progress": {**(record.get("progress") or {}),
                                 "message": CANCELLED_MESSAGE},
                })
                return self._decorate(record)
            # running：交给调度线程终止当前进程，不在这里伪造终态。
            record = self.store.patch_batch(connection, batch_id, {
                "cancel_requested": True,
                "progress": {**(record.get("progress") or {}),
                             "message": "正在停止"},
            })
        # 只有正在跑的就是这个批次时，才动当前进程；排队批次已在上面的
        # 分支直接置为取消，不影响别的批次。
        if self._active_batch == batch_id:
            self._cancel.set()
            self._stop_active_process()
        return self._decorate(record)

    # -- 重跑 -----------------------------------------------------------
    def replay(self, batch_id: str) -> dict:
        current = self.store.get_batch(batch_id)
        legacy = None if current is not None else self._legacy_entry(batch_id)
        if current is None and legacy is not None:
            current = {"id": legacy["id"], "status": legacy["status"],
                       "videos": [{"job": video.get("job"), "name": video.get("name"),
                                   "asset_id": video.get("asset_id")}
                                  for video in legacy["videos"]],
                       "draft_id": legacy.get("draft_id"),
                       "project_id": legacy.get("project_id"),
                       "assay": legacy.get("assay"), "legacy": True}
        if current is None:
            raise NotFoundError("任务不存在")
        status = current.get("status")
        if status in ACTIVE:
            raise JobConflict("任务正在排队或运行中，不能重复分析")
        if status not in TERMINAL:
            raise JobConflict("任务状态不支持重新分析")

        prepared: list[dict] = []
        issues: list[dict] = []
        project_id = current.get("project_id")
        animals = None
        if project_id is not None:
            animals = {item["id"] for item in self.store.list_animals(project_id)}
        for video in current.get("videos") or []:
            frozen = video.get("job")
            if not isinstance(frozen, Mapping):
                issues.append({"asset_id": video.get("asset_id"), "field": "job",
                               "message": "缺少冻结任务，无法按原配置重跑"})
                continue
            refreshed, video_issues = self._refresh_frozen(frozen, animals=animals)
            if video_issues:
                for issue in video_issues:
                    entry = dict(issue)
                    entry.setdefault("asset_id", video.get("asset_id"))
                    entry["message"] = f"{video.get('name') or '视频'}：{entry['message']}"
                    issues.append(entry)
                continue
            prepared.append({
                "asset_id": refreshed["asset"].get("id"),
                "name": refreshed["asset"].get("name") or video.get("name"),
                "run_id": refreshed["run_id"],
                "job_id": batch_id,
                "status": QUEUED,
                "progress": 0.0,
                "job": refreshed,
                "result_manifest": None,
                "error": None,
            })
        if issues:
            raise JobError("原配置无法重新验证", fields=issues)

        # 两个分支都要在最终写事务内重读来源与状态：删除已提交则不存在，
        # 另一个请求改了状态则明确冲突，不能基于事务前快照派生新批次。
        with self.store.mutate() as connection:
            record = self.store.batch_in(connection, batch_id)
            if record is None:
                raise NotFoundError("任务不存在")
            if record.get("status") != status:
                raise JobConflict("任务状态已变化，未排队")
            if status == COMPLETED:
                new_batch_id = new_id()
                for video in prepared:
                    video["job"] = {**video["job"], "job_id": new_batch_id}
                    video["job_id"] = new_batch_id
                batch = self.store.create_batch(
                    assay=record.get("assay"),
                    videos=prepared,
                    draft_id=record.get("draft_id"),
                    project_id=record.get("project_id"),
                    batch_id=new_batch_id,
                    connection=connection,
                )
            else:
                owned = list(record.get("owned_run_ids") or [])
                for video in prepared:
                    if video["run_id"] not in owned:
                        owned.append(video["run_id"])
                batch = self.store.patch_batch(connection, batch_id, {
                    "status": QUEUED,
                    "videos": prepared,
                    "progress": {"fraction": 0.0, "current": 0,
                                 "total": len(prepared), "message": QUEUED_MESSAGE},
                    "cancel_requested": False,
                    "error": None,
                    "started_at": None,
                    "finished_at": None,
                    "owned_run_ids": owned,
                })
        self._wake.set()
        if self.inline:
            self._start_batch(batch["id"])
            self._run_batch(batch["id"])
        return self.get(batch["id"])

    def _refresh_frozen(self, frozen: Mapping[str, Any], *,
                        animals: set[str] | None = None
                        ) -> tuple[dict, list[dict]]:
        """按原配置重新取身份与媒体事实；不读取草稿或当前默认值。

        ``animals`` 来自冻结 job 的 project_id 现有动物；项目单位必须仍绑定
        本项目动物，缺失/跨项目一律拒绝重跑。
        """

        item = deepcopy(dict(frozen))
        asset = dict(item.get("asset") or {})
        path = asset.get("path")
        issues: list[dict] = []
        if not isinstance(path, str) or not path.strip():
            return item, [{"field": "asset.path", "message": "缺少源视频路径"}]
        try:
            identity = engine._stat_identity(path)
        except ValidationError as error:
            return item, [error.as_issue()]
        from . import media

        stored = asset.get("metadata")
        stored = dict(stored) if isinstance(stored, Mapping) else {}
        try:
            probed = media.probe_video(identity["path"])
        except media.MediaError as error:
            return item, [{"field": "asset.path", "message": str(error)}]
        asset["path"] = identity["path"]
        asset["identity"] = identity
        asset["metadata"] = {**stored, **probed}
        item["asset"] = asset
        item["algorithm_version"] = ALGORITHM_VERSION
        item["run_id"] = new_id()
        issues = engine.validate_job(item, animals=animals)
        return item, issues

    # -- 删除 -----------------------------------------------------------
    def delete(self, batch_id: str) -> None:
        if not is_valid_id(batch_id):
            raise NotFoundError("任务不存在")
        with self.store.mutate() as connection:
            record = self.store.batch_in(connection, batch_id)
            if record is None:
                self._delete_legacy(batch_id, connection)
                return
            status = record.get("status")
            if status not in TERMINAL:
                raise JobConflict("任务尚未结束，不能删除")
            plan = self._removal_plan(record)
            protected = self._protected_paths()
            blocked = [item for item in plan if item[0] == "file"
                       and str(item[1].resolve()) in protected]
            if not blocked:
                # 目录里可能放着未登记却已被其他任务冻结的输入：整树删除前
                # 逐个核对受保护文件，不能因为“没声明”就带走原片。
                for kind, path in plan:
                    if kind != "dir":
                        continue
                    for item in path.rglob("*"):
                        if item.is_file() and str(item.resolve()) in protected:
                            blocked.append(("file", item))
                            break
                    if blocked:
                        break
            if not blocked:
                # 目录里可能放着未登记却已被其他任务冻结的输入：整树删除前
                # 逐个核对受保护文件，不能因为“没声明”就带走原片。
                for kind, path in plan:
                    if kind != "dir":
                        continue
                    for item in path.rglob("*"):
                        if item.is_file() and str(item.resolve()) in protected:
                            blocked.append(("file", item))
                            break
                    if blocked:
                        break
            if blocked:
                # 已登记或已冻结的输入不能被任务清理带走：拒绝并保留记录。
                raise JobError(
                    f"结果目录内有受保护的输入文件，未删除：{blocked[0][1].name}",
                    status=409)
            try:
                for kind, path in plan:
                    if kind == "file":
                        if str(path.resolve()) in protected:
                            continue
                        path.unlink(missing_ok=True)
                for kind, path in plan:
                    if kind != "dir":
                        continue
                    try:
                        path.rmdir()     # 只删空根目录
                    except OSError:
                        pass             # 还有不能认领的子项：原样保留
            except OSError as error:
                raise JobError(f"清理任务文件失败：{error}", status=409)
            self.store.delete_batch_record(batch_id, connection)

    def _delete_legacy(self, run_id: str, connection: Any) -> None:
        rows = [row for row in self.store.list_legacy_jobs()
                if row.get("run_id") == run_id]
        if not rows:
            raise NotFoundError("任务不存在")
        dead = [str(row.get("status")) for row in rows
                if str(row.get("status")) in ("running", "queued")]
        if dead:
            raise JobConflict("任务尚未结束，不能删除")
        protected = self._protected_paths()
        for row in rows:
            directory = self.paths.runs / run_id / row["id"]
            # 旧记录按可信 job 行核对，允许缺状态的旧清单。
            manifest = _manifest_ok(directory, row["id"], run_id,
                                    require_status=False)
            if manifest is None:
                continue
            if not _under(self.paths.runs, directory):
                raise JobError("任务目录不在数据根内，已拒绝删除", status=409)
            for name in manifest.get("files") or []:
                candidate = directory / str(name)
                if candidate.is_file() and str(candidate.resolve()) in protected:
                    raise JobError("结果目录内有受保护的输入文件，未删除："
                                   + candidate.name, status=409)
            for name in manifest.get("files") or []:
                candidate = directory / str(name)
                if candidate.is_file() and not candidate.is_symlink():
                    candidate.unlink(missing_ok=True)
            try:
                directory.rmdir()
            except OSError:
                pass
        connection.execute("DELETE FROM jobs WHERE run_id = ?", (run_id,))

    def _protected_paths(self) -> set[str]:
        """已登记资产与其他冻结任务引用的源文件：清理时一律保护。"""

        protected: set[str] = set()
        for asset in self.store.list_assets():
            path = asset.get("path")
            if isinstance(path, str) and path:
                protected.add(str(Path(path).resolve()))
        for batch in self.store.list_batches():
            for video in batch.get("videos") or []:
                job = video.get("job") or {}
                asset = job.get("asset") or {}
                for value in (asset.get("path"),
                              (asset.get("identity") or {}).get("path")):
                    if isinstance(value, str) and value:
                        protected.add(str(Path(value).resolve()))
        return protected

    def _removal_plan(self, record: Mapping[str, Any]) -> list[tuple[str, Path]]:
        """按可信材料列清理计划：只认本任务能证明归属的文件。

        ``owned_run_ids`` 不是压倒性授权：stage/task/result 各自的 job.json 与
        清单必须自称同一 job_id/run_id，冲突或不能认领的子项原样保留。
        """

        batch_id = str(record.get("id"))
        plan: list[tuple[str, Path]] = []
        run_ids = [str(item) for item in record.get("owned_run_ids") or []
                   if isinstance(item, str) and _SAFE_NAME.fullmatch(item)]
        tasks_root = self.paths.tasks / batch_id
        if not run_ids and tasks_root.is_dir() and _under(self.paths.tasks,
                                                          tasks_root):
            # 旧记录缺 owned_run_ids：只从本任务自己的 job.json 恢复归属。
            for child in sorted(tasks_root.iterdir()):
                job_file = child / "job.json"
                if not job_file.is_file():
                    continue
                try:
                    frozen = read_json_file(job_file, default=None)
                except (OSError, ValueError):
                    continue
                if not isinstance(frozen, Mapping):
                    continue
                if frozen.get("job_id") != batch_id:
                    continue
                run_id = frozen.get("run_id")
                if isinstance(run_id, str) and _SAFE_NAME.fullmatch(run_id):
                    run_ids.append(run_id)

        for run_id in run_ids:
            stage = self.paths.runs / f".{run_id}.tmp"
            if stage.exists() and _under(self.paths.runs, stage):
                if self._stage_belongs(stage, batch_id, run_id):
                    plan.extend(("file", item) for item in stage.rglob("*")
                                if item.is_file())
                    plan.append(("dir", stage))
            destination = self.paths.runs / run_id
            if not destination.exists() or not _under(self.paths.runs,
                                                      destination):
                continue
            # 删除可重试：本流程先前已成功删掉的声明文件允许缺失，只要
            # 清单仍自称本任务、状态为 completed 且没有越界声明，剩余
            # owned 文件继续清理。发布/恢复的判据不放宽。
            manifest = _manifest_ok(destination, batch_id, run_id,
                                    tolerate_missing=True)
            if manifest is None:
                # 清单自称呼别的任务或不可核对：不按名字猜归属。
                continue
            for name in manifest.get("files") or []:
                candidate = destination / str(name)
                if candidate.is_file() and not candidate.is_symlink():
                    plan.append(("file", candidate))
            plan.append(("dir", destination))

        if tasks_root.is_dir() and _under(self.paths.tasks, tasks_root):
            for child in tasks_root.iterdir():
                if child.is_file() and child.name in ("job.json", "progress.json",
                                                      "worker.log"):
                    if child.name != "job.json" or self._task_belongs(
                            child, batch_id):
                        plan.append(("file", child))
                elif child.is_dir() and child.name in run_ids:
                    if self._task_belongs(child / "job.json", batch_id):
                        # 只清本任务自己的三个文件；其他子项原样保留。
                        for name in ("job.json", "progress.json", "worker.log"):
                            candidate = child / name
                            if candidate.is_file():
                                plan.append(("file", candidate))
                        plan.append(("dir", child))
            plan.append(("dir", tasks_root))
        return plan

    @staticmethod
    def _task_belongs(job_file: Path, batch_id: str) -> bool:
        """task 内 job.json 必须自称本任务；缺失或冲突都不认领。"""

        if not job_file.is_file():
            return False
        try:
            frozen = read_json_file(job_file, default=None)
        except (OSError, ValueError):
            return False
        return isinstance(frozen, Mapping) and frozen.get("job_id") == batch_id

    @staticmethod
    def _stage_belongs(stage: Path, batch_id: str, run_id: str) -> bool:
        """stage 内 frozen_job.json 必须自称同一任务与运行。"""

        frozen_path = stage / "frozen_job.json"
        if not frozen_path.is_file():
            return False
        try:
            frozen = read_json_file(frozen_path, default=None)
        except (OSError, ValueError):
            return False
        if not isinstance(frozen, Mapping):
            return False
        return (frozen.get("job_id") == batch_id
                and frozen.get("run_id") == run_id)

    @staticmethod
    def _remove(path: Path) -> None:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=False)
        else:
            path.unlink(missing_ok=True)

    # -- 恢复 -----------------------------------------------------------
    def _recover(self) -> None:
        for batch in self.store.list_batches():
            if batch.get("status") != RUNNING:
                continue
            videos = [dict(video) for video in batch.get("videos") or []]
            for video in videos:
                run_id = video.get("run_id")
                if video.get("status") == RUNNING:
                    manifest = (_manifest_ok(self.paths.runs / str(run_id),
                                             str(batch.get("id")), str(run_id))
                                if isinstance(run_id, str) else None)
                    if manifest is not None:
                        video.update({"status": COMPLETED, "progress": 1.0,
                                      "result_manifest": manifest,
                                      "output_dir": str(self.paths.runs / run_id),
                                      "error": None})
                    else:
                        video.update({"status": INTERRUPTED,
                                      "error": INTERRUPTED_MESSAGE})
                elif video.get("status") == QUEUED:
                    video["status"] = NOT_STARTED
                if isinstance(run_id, str) and _SAFE_NAME.fullmatch(run_id):
                    stage = self.paths.runs / f".{run_id}.tmp"
                    if stage.is_dir() and _under(self.paths.runs, stage):
                        shutil.rmtree(stage, ignore_errors=True)
            completed = videos and all(video.get("status") == COMPLETED
                                       for video in videos)
            self.store.update_batch(batch["id"], {
                "status": COMPLETED if completed else INTERRUPTED,
                "videos": videos,
                "progress": {
                    "fraction": 1.0 if completed else _batch_progress(
                        videos, 0, 0.0, INTERRUPTED_MESSAGE)["fraction"],
                    "current": len([video for video in videos
                                    if video.get("status") == COMPLETED]),
                    "total": len(videos),
                    "message": "已完成" if completed else INTERRUPTED_MESSAGE,
                },
                "error": None if completed else INTERRUPTED_MESSAGE,
                "finished_at": _now(),
            })

    # -- 调度 -----------------------------------------------------------
    def _loop(self) -> None:
        while not self._stopping.is_set():
            try:
                batch = self.store.claim_next_batch()
            except Exception:
                LOGGER.exception("取下一个批次失败")
                batch = None
            if batch is None:
                self._wake.wait(0.25)
                self._wake.clear()
                continue
            try:
                self._run_batch(batch["id"])
            except Exception:  # 调度线程不能因单个批次退出
                LOGGER.exception("执行批次失败：%s", batch.get("id"))
        with self._lock:
            if self._release_pending:
                self._release_lock()

    def _start_batch(self, batch_id: str) -> None:
        """把批次置为 running；正常情况下由事务 claim 完成。"""

        self.store.update_batch(batch_id, {"status": RUNNING,
                                           "started_at": _now()})

    def _cancel_requested(self, batch_id: str) -> bool:
        record = self.store.get_batch(batch_id)
        return bool(record and record.get("cancel_requested"))

    def _run_batch(self, batch_id: str) -> None:
        batch = self.store.get_batch(batch_id)
        if batch is None:
            return
        videos = [dict(video) for video in batch.get("videos") or []]
        self._active_batch = batch_id
        self._cancel.clear()
        for index, video in enumerate(videos):
            if self._stopping.is_set():
                self._finish_batch(batch_id, videos, index, INTERRUPTED,
                                   INTERRUPTED_MESSAGE)
                return
            if self._cancel_requested(batch_id):
                self._finish_batch(batch_id, videos, index, CANCELLED,
                                   CANCELLED_MESSAGE)
                return
            videos[index]["status"] = RUNNING
            videos[index]["error"] = None
            videos[index]["progress"] = 0.0
            self._active_index = index
            self._store_videos(batch_id, videos, {
                "progress": _batch_progress(
                    videos, index, 0.0,
                    f"分析中：{video.get('name') or '视频'}")})
            outcome = self._execute_video(batch_id, videos, index)
            status = outcome.get("status")
            videos[index].update({
                "status": status,
                "progress": 1.0 if status == COMPLETED else
                videos[index].get("progress", 0.0),
            })
            for key in ("result_manifest", "output_dir", "log", "error"):
                if key in outcome:
                    videos[index][key] = outcome[key]
                elif key == "result_manifest":
                    videos[index]["result_manifest"] = None
            if status != COMPLETED:
                self._finish_batch(batch_id, videos, index,
                                   status if status in (CANCELLED, INTERRUPTED)
                                   else FAILED,
                                   outcome.get("error") or "分析失败")
                return
            self._store_videos(batch_id, videos, {
                "progress": _batch_progress(videos, index, 1.0,
                                            f"已完成：{video.get('name') or '视频'}")})
        self._finish_batch(batch_id, videos, len(videos), COMPLETED, None)

    def _finish_batch(self, batch_id: str, videos: list[dict], from_index: int,
                      status: str, error: str | None) -> None:
        for video in videos[from_index:]:
            if video.get("status") in (QUEUED, RUNNING):
                video["status"] = NOT_STARTED
        patch: dict[str, Any] = {
            "status": status,
            "videos": videos,
            "finished_at": _now(),
            "error": error,
            "progress": {
                "fraction": round(
                    len([video for video in videos
                         if video.get("status") == COMPLETED]) / max(1, len(videos)),
                    6),
                "current": len([video for video in videos
                                if video.get("status") == COMPLETED]),
                "total": len(videos),
                "message": ("已完成" if status == COMPLETED
                            else error or "分析失败"),
            },
        }
        if status == CANCELLED:
            patch["cancel_requested"] = True
        self._store_videos(batch_id, videos, patch)

    def _store_videos(self, batch_id: str, videos: list[dict],
                      patch: Mapping[str, Any]) -> None:
        payload = {**patch, "videos": videos}
        self.store.update_batch(batch_id, payload)
        self._last_progress_write = self._clock()

    # -- 单个视频 -------------------------------------------------------
    def _execute_video(self, batch_id: str, videos: list[dict],
                       index: int) -> dict:
        video = videos[index]
        job = video.get("job")
        if not isinstance(job, Mapping):
            return {"status": FAILED, "error": "缺少冻结任务"}
        run_id = str(video.get("run_id"))
        task_dir = self.paths.tasks / batch_id / run_id
        stage = self.paths.runs / f".{run_id}.tmp"
        destination = self.paths.runs / run_id
        try:
            task_dir.mkdir(parents=True, exist_ok=True)
            if stage.exists() and _under(self.paths.runs, stage):
                shutil.rmtree(stage, ignore_errors=True)
            job_path = task_dir / "job.json"
            progress_path = task_dir / "progress.json"
            log_path = task_dir / "worker.log"
            atomic_write_json(job_path, dict(job))
        except OSError as error:
            return {"status": FAILED, "error": f"无法创建任务目录：{error}"}

        context = {
            "batch": batch_id,
            "video": video,
            "index": index,
            "job": job,
            "task_dir": task_dir,
            "stage": stage,
            "destination": destination,
            "progress_path": progress_path,
            "log": log_path,
            "cancel": self._cancel,
            "stopping": self._stopping,
        }
        try:
            if self._runner is not None:
                code = self._run_inline_runner(context)
            elif self.inline:
                code = worker.execute_job_file(job_path, stage, progress_path,
                                               cancel=self._cancel.is_set)
            else:
                code = self._run_process(context)
        except BaseException as error:
            # 执行接口本身出错：先处理仍活的所属子进程，再给真实终态。
            self._stop_active_process()
            self._cleanup_stage(stage)
            video["log"] = str(log_path)
            return {"status": FAILED,
                    "error": f"执行失败：{type(error).__name__}: {error}"}
        video["log"] = str(log_path)

        if self._stopping.is_set():
            self._cleanup_stage(stage)
            return {"status": INTERRUPTED, "error": INTERRUPTED_MESSAGE}
        if self._cancel.is_set() or self._cancel_requested(batch_id):
            self._cleanup_stage(stage)
            return {"status": CANCELLED, "error": CANCELLED_MESSAGE}
        if code != 0:
            self._cleanup_stage(stage)
            return {"status": FAILED,
                    "error": self._failure_text(progress_path, code)}
        if destination.exists():
            self._cleanup_stage(stage)
            return {"status": FAILED, "error": "目标目录已存在，未覆盖旧结果"}
        manifest = _manifest_ok(stage, batch_id, run_id)
        if manifest is None:
            self._cleanup_stage(stage)
            return {"status": FAILED, "error": "结果清单缺失或不完整"}
        try:
            os.replace(stage, destination)
        except OSError as error:
            self._cleanup_stage(stage)
            return {"status": FAILED, "error": f"发布结果失败：{error}"}
        return {"status": COMPLETED, "result_manifest": manifest,
                "output_dir": str(destination), "error": None}

    def _run_inline_runner(self, context: Mapping[str, Any]) -> int:
        """测试替身：与真实子进程共用发布与状态逻辑，只替换执行者。"""

        return int(self._runner(context))  # type: ignore[misc]

    def _run_process(self, context: Mapping[str, Any]) -> int:
        job_path = context["task_dir"] / "job.json"
        command = worker.worker_command(job_path, context["stage"],
                                        context["progress_path"])
        root = str(Path(__file__).resolve().parents[1])
        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"]
                                    if env.get("PYTHONPATH") else "")
        log_path = context["log"]
        process = None
        try:
            with open(log_path, "ab") as log:
                process = self._spawn(command, log, env, root)
                self._process = process
                return self._wait(process, context)
        except OSError as error:
            if process is not None:
                # spawn 已成功、等待中出错：错误发生在已创建进程的所属
                # 路径上，必须先终止并等待确认退出，再写错误返回；判断
                # 依据是引用是否存在，不依赖 finally 里是否还有活动异常。
                self._terminate(process)
            self._write_error(context["progress_path"],
                              f"无法启动工作进程：{error}")
            return 1
        except BaseException:
            # 其他异常向上传播前同样先收尾已创建的进程；spawn 尚未产生
            # 进程时不动任何句柄。
            if process is not None:
                self._terminate(process)
            raise
        finally:
            # 正常返回时 _wait 已确认进程退出，这里只释放并清引用；
            # 异常路径的终止已在 except 分支完成。释放多次调用安全，
            # 外层 _stop_active_process 是重复调用的安全兜底。
            self._process = None
            if process is not None:
                release = getattr(process, "release", None)
                if callable(release):
                    try:
                        release()
                    except Exception:
                        LOGGER.exception("释放工作进程句柄失败")

    def _spawn(self, command: list[str], log: Any, env: dict, root: str) -> Any:
        if os.name == "nt":
            return spawn_managed(command, stdout=log, stderr=subprocess.STDOUT,
                                 env=env, cwd=root, stdin=subprocess.DEVNULL)
        return subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                env=env, cwd=root, stdin=subprocess.DEVNULL)

    def _wait(self, process: Any, context: Mapping[str, Any]) -> int:
        batch_id = str(context["batch"])
        index = int(context["index"])
        name = context["video"].get("name") or "视频"
        last_fraction = -1.0
        last_message = ""
        last_write = self._clock()
        while True:
            code = process.poll()
            record = self.store.get_batch(batch_id)
            videos = [dict(video) for video in (record or {}).get("videos") or []]
            progress = self._read_progress(context["progress_path"])
            if 0 <= index < len(videos) and progress is not None:
                fraction = _fraction(progress.get("fraction"))
                message = str(progress.get("message") or "")
                videos[index]["progress"] = fraction
                now = self._clock()
                due = (now - last_write >= PROGRESS_WRITE_INTERVAL_S)
                if code is None and (due or fraction - last_fraction >= 0.01
                                     or message != last_message):
                    last_fraction = fraction
                    last_message = message
                    last_write = now
                    self.store.update_batch(batch_id, {
                        "videos": videos,
                        "progress": _batch_progress(
                            videos, index, fraction,
                            message or f"分析中：{name}"),
                    })
            if self._stopping.is_set():
                self._terminate(process)
                return 1
            if self._cancel.is_set() or self._cancel_requested(batch_id):
                self._terminate(process)
                return 1
            if code is not None:
                return int(code)
            time.sleep(self._poll)

    def _terminate(self, process: Any) -> None:
        try:
            if hasattr(process, "stop"):
                process.stop()
            else:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except Exception:
                    process.kill()
        except Exception:
            LOGGER.exception("终止工作进程失败")

    def _stop_active_process(self) -> None:
        process = self._process
        if process is not None:
            self._terminate(process)

    @staticmethod
    def _read_progress(path: Path) -> dict | None:
        try:
            payload = read_json_file(path, default=None)
        except (OSError, ValueError):
            return None
        return dict(payload) if isinstance(payload, Mapping) else None

    def _failure_text(self, progress_path: Path, code: int) -> str:
        payload = self._read_progress(progress_path)
        error = (payload or {}).get("error")
        if isinstance(error, str) and error.strip():
            return error.strip()
        return f"分析进程退出码 {code}"

    @staticmethod
    def _write_error(progress_path: Path, message: str) -> None:
        try:
            atomic_write_json(progress_path, {"fraction": 0.0,
                                              "message": "分析失败",
                                              "error": message})
        except OSError:
            pass

    @staticmethod
    def _cleanup_stage(stage: Path) -> None:
        if stage.is_dir():
            shutil.rmtree(stage, ignore_errors=True)
        elif stage.exists():
            try:
                stage.unlink()
            except OSError:
                pass
