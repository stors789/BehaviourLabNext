"""第 22 块七组有限修正的定向自检。

只覆盖已确认缺口：manifest 有效性、关闭屏障、进程接口、删除归属、
重跑事务与项目动物、最终校验边界、任务中心草稿语义。
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest

from behaviourlab import engine, runs as run_service
from behaviourlab.config import DataPaths
from behaviourlab.jobs import (
    COMPLETED,
    FAILED,
    INTERRUPTED,
    NOT_STARTED,
    QUEUED,
    RUNNING,
    JobConflict,
    JobError,
    JobScheduler,
    _manifest_ok,
)
from behaviourlab.schema import AnalysisResult, ValidationError
from behaviourlab.store import NotFoundError, Store, atomic_write_json, read_json_file


# -- 共用夹具 ------------------------------------------------------------


@pytest.fixture
def paths(tmp_path, monkeypatch):
    root = tmp_path / "block22fix"
    paths = DataPaths.from_dir(root)
    paths.ensure()
    monkeypatch.setattr(
        engine.media, "probe_video",
        lambda path: {"width": 160, "height": 120, "duration_s": 1.0,
                      "fps": 10.0, "frame_count": 10})
    return paths


@pytest.fixture
def store(paths):
    return Store(paths)


def _asset(store: Store, paths: DataPaths, name: str = "a.mp4") -> dict:
    source = paths.root / "media" / name
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"placeholder-" + name.encode())
    return store.create_asset(name=name, path=str(source), project_id=None,
                              metadata={"width": 160, "height": 120})


def _draft(store: Store, asset: dict) -> dict:
    return store.create_draft("oft", output={"annotated_video": False}, videos=[{
        "asset_id": asset["id"], "window": {"start_s": 0.0, "end_s": 1.0},
        "orientation": 0, "fps_override": None, "sample_fps": None,
        "units": [{"id": "u" * 32, "name": "小鼠 1", "window": None,
                   "geometry": {"arena": {"type": "quad", "points": [
                       [10, 10], [150, 10], [150, 110], [10, 110]]}},
                   "options": {"arena_width_cm": 33.0}, "detection": {},
                   "manual": None}],
    }])


def _runner(*, fail: bool = False):
    def run(context):
        job = context["job"]
        stage = context["stage"]
        if fail:
            atomic_write_json(context["progress_path"],
                              {"fraction": 0.1, "message": "失败",
                               "error": "源视频大小已变化"})
            return 1
        stage.mkdir(parents=True, exist_ok=True)
        (stage / "summary.csv").write_text("unit_id\n", encoding="utf-8")
        atomic_write_json(stage / "results.json", {"units": []})
        atomic_write_json(stage / "frozen_job.json", dict(job))
        atomic_write_json(stage / "result_manifest.json", {
            "status": "completed", "run_id": job["run_id"],
            "job_id": job["job_id"], "assay": job["assay"],
            "files": ["summary.csv", "results.json", "frozen_job.json",
                      "result_manifest.json"],
            "units": [], "warnings": [], "metadata": {},
        })
        return 0
    return run


def _completed_batch(store: Store, paths: DataPaths, asset: dict) -> dict:
    draft = _draft(store, asset)
    scheduler = JobScheduler(store, paths, inline=True, runner=_runner())
    return scheduler.submit(draft["id"])


# -- 组 1：manifest 有效性贯通 -------------------------------------------


def _write_manifest(directory: Path, payload: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    return atomic_write_json(directory / "result_manifest.json", payload)


@pytest.mark.parametrize("variant", ["completed", "failed", "no_status",
                                     "empty_files", "missing_file", "escape"])
def test_manifest_validity_gate(paths, store, variant):
    run_id, job_id = "r" * 32, "j" * 32
    directory = paths.runs / run_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.csv").write_text("unit_id\n", encoding="utf-8")
    payload = {"run_id": run_id, "job_id": job_id,
               "files": ["summary.csv"], "units": [], "warnings": []}
    if variant == "completed":
        payload["status"] = "completed"
    elif variant == "failed":
        payload["status"] = "failed"
    elif variant == "empty_files":
        payload["status"] = "completed"
        payload["files"] = []
    elif variant == "missing_file":
        payload["status"] = "completed"
        payload["files"] = ["summary.csv", "gone.csv"]
    elif variant == "escape":
        payload["status"] = "completed"
        outside = paths.root / "outside.csv"
        outside.write_text("x", encoding="utf-8")
        payload["files"] = ["../outside.csv"]
    _write_manifest(directory, payload)

    manifest = read_json_file(directory / "result_manifest.json")
    accepted = _manifest_ok(directory, job_id, run_id)
    resolved = run_service.resolve_run(store, paths, run_id)
    if variant == "completed":
        assert accepted is not None
        assert resolved is not None
    else:
        assert accepted is None, variant
        assert resolved is None, variant


def test_recovery_rejects_incomplete_manifest(paths, store):
    asset = _asset(store, paths)
    draft = _draft(store, asset)
    scheduler = JobScheduler(store, paths, inline=False)
    batch = scheduler.submit(draft["id"])
    run_id = batch["videos"][0]["run_id"]
    directory = paths.runs / run_id
    _write_manifest(directory, {
        "status": "failed", "run_id": run_id, "job_id": batch["id"],
        "files": ["summary.csv"], "units": []})
    (directory / "summary.csv").write_text("x", encoding="utf-8")
    store.update_batch(batch["id"], {
        "status": "running",
        "videos": [{**batch["videos"][0], "status": "running"}]})
    stage = paths.runs / f".{run_id}.tmp"
    stage.mkdir(parents=True, exist_ok=True)
    (stage / "summary.csv").write_text("x", encoding="utf-8")

    recovered = JobScheduler(store, paths)
    recovered._recover()
    record = store.get_batch(batch["id"])
    assert record["status"] == INTERRUPTED
    assert record["videos"][0]["status"] == INTERRUPTED
    assert not stage.exists(), "只清本轮 stage"
    assert directory.is_dir(), "已发布目录不能因恢复被删"


def test_failed_manifest_is_not_downloadable(paths, store):
    from behaviourlab.web.app import create_app

    asset = _asset(store, paths)
    draft = _draft(store, asset)
    scheduler = JobScheduler(store, paths, inline=False)
    batch = scheduler.submit(draft["id"])
    run_id = batch["videos"][0]["run_id"]
    directory = paths.runs / run_id
    _write_manifest(directory, {
        "status": "failed", "run_id": run_id, "job_id": batch["id"],
        "files": ["summary.csv"], "units": []})
    (directory / "summary.csv").write_text("x", encoding="utf-8")

    app = create_app(data_dir=str(paths.root), store=store, testing=True)
    client = app.test_client()
    assert client.get(f"/api/runs/{run_id}").status_code == 404
    assert client.get(f"/api/runs/{run_id}/files/summary.csv").status_code == 404
    assert client.get(f"/runs/{run_id}/{batch['id']}/summary.csv").status_code == 404
    assert draft is not None


def test_legacy_nested_without_status_stays_readable(paths, store):
    run_id, job_id = "l" * 32, "m" * 32
    directory = paths.runs / run_id / job_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.csv").write_text("unit_id\n", encoding="utf-8")
    _write_manifest(directory, {"run_id": run_id, "job_id": job_id,
                                "files": ["summary.csv"], "units": []})
    store.create_job(assay="oft", run_id=run_id, job_id=job_id, status="done",
                     config={"job": {"assay": "oft"}})
    manifest = _manifest_ok(directory, job_id, run_id, require_status=False)
    assert manifest is not None, "旧嵌套清单可按可信记录读取"
    assert run_service.resolve_run(store, paths, run_id) is not None


# -- 组 2：关闭屏障与独立生命周期 ----------------------------------------


def test_service_stop_keeps_owner_until_jobs_finish(paths, store, monkeypatch):
    from behaviourlab.service import LocalService
    from behaviourlab.web.app import create_app

    events: list[str] = []
    started = threading.Event()
    release = threading.Event()

    def blocking_runner(context):
        started.set()
        release.wait(timeout=10.0)
        return 1

    app = create_app(data_dir=str(paths.root), store=store, testing=True)
    jobs = app.config["JOBS"]
    jobs.inline = False          # 这一组要真实后台线程，不用同步执行变体
    jobs._runner = blocking_runner
    asset = _asset(store, paths)
    draft = _draft(store, asset)
    batch = jobs.submit(draft["id"])
    assert jobs.start() is True
    assert started.wait(timeout=5.0)

    real_shutdown = jobs.shutdown

    def wrapped_shutdown(**kwargs):
        events.append("jobs")
        return real_shutdown(**kwargs)

    monkeypatch.setattr(jobs, "shutdown", wrapped_shutdown)

    class Server:
        def run(self):
            time.sleep(0.05)

        def close(self):
            events.append("http")

    service = LocalService(
        app, paths, port=0,
        server_factory=lambda *args, **kwargs: Server(),
        ready_probe=lambda url, root: (True, ""),
        owner_registry=app.config["JOB_OWNER"],
        stop_callbacks=[jobs.shutdown],
        final_callbacks=[lambda: events.append("store")],
    )
    service.start()
    owner_record = paths.logs / "job_owner.json"

    # 关闭时 runner 还卡着：不能当作完成，也不能放锁。
    jobs.shutdown = lambda timeout_s=0.2: wrapped_shutdown(timeout_s=0.2)
    service.stop()
    assert owner_record.exists(), "关闭未完成必须继续持有所有权"
    second = JobScheduler(store, paths)
    assert second.start() is False

    release.set()
    for _ in range(100):
        if not jobs.running:
            break
        time.sleep(0.05)
    assert not jobs.running
    assert store.get_batch(batch["id"])["status"] == INTERRUPTED
    assert events[:2] == ["jobs", "http"]
    # 线程真正退出后才释放所有权：关闭期间第二实例拿不到锁，正是这个原因。
    assert second.owner.held is False


def test_completed_shutdown_order(paths, store):
    from behaviourlab.service import LocalService
    from behaviourlab.web.app import create_app

    events: list[str] = []
    app = create_app(data_dir=str(paths.root), store=store, testing=True)
    jobs = app.config["JOBS"]

    class Server:
        def run(self):
            time.sleep(0.02)

        def close(self):
            events.append("http")

    service = LocalService(
        app, paths, port=0,
        server_factory=lambda *args, **kwargs: Server(),
        ready_probe=lambda url, root: (True, ""),
        owner_registry=app.config["JOB_OWNER"],
        stop_callbacks=[lambda: events.append("jobs")],
        final_callbacks=[lambda: events.append("store")],
    )
    service.start()
    jobs.start()
    service.stop()
    assert events == ["jobs", "http", "store"]
    assert not (paths.logs / "job_owner.json").exists(), "最后才释放所有权"


def test_independent_jobs_restarts_after_timeout(paths, store):
    started = threading.Event()
    release = threading.Event()

    def blocking_runner(context):
        started.set()
        release.wait(timeout=10.0)
        return 1

    scheduler = JobScheduler(store, paths, inline=False, runner=blocking_runner)
    asset = _asset(store, paths)
    draft = _draft(store, asset)
    batch = scheduler.submit(draft["id"])
    assert scheduler.start() is True
    assert started.wait(timeout=5.0)
    # 超时返回 False：尚未完成，锁仍在自己手里。
    assert scheduler.shutdown(timeout_s=0.05) is False
    release.set()
    for _ in range(100):
        if not scheduler.running:
            break
        time.sleep(0.05)
    assert store.get_batch(batch["id"])["status"] == INTERRUPTED
    # 线程退出后再次 start 必须真正重新持锁并启动，不能空返 True。
    assert scheduler.start() is True
    assert scheduler.running is True
    assert scheduler.owner.held is True
    assert scheduler.shutdown() is True


def test_unheld_second_instance_shutdown_is_inert(paths, store):
    first = JobScheduler(store, paths)
    assert first.start() is True
    second = JobScheduler(store, paths)
    second.shutdown()
    assert first.owner.held is True
    assert first.running is True
    first.shutdown()
    assert first.owner.held is False


# -- 组 3：管理进程接口与句柄释放 ----------------------------------------


class _Child:
    def __init__(self, code=0):
        self.code = code
        self.killed = False

    def poll(self):
        return self.code

    def wait(self, timeout=None):
        return self.code

    def kill(self):
        self.killed = True
        self.code = -9


class _Job:
    def __init__(self):
        self.closed = 0
        self.terminated = False

    def terminate(self):
        self.terminated = True

    def close(self):
        self.closed += 1


def test_managed_process_interface_and_release():
    from behaviourlab.processes import ManagedProcess

    child, job = _Child(0), _Job()
    managed = ManagedProcess(child, job)
    assert managed.poll() == 0
    managed.release()
    managed.release()
    assert job.closed == 1, "正常结束只关闭自己的 Job Object，且幂等"

    child2, job2 = _Child(None), _Job()
    stopped = ManagedProcess(child2, job2)
    stopped.stop(timeout_s=0.1)
    assert job2.terminated is True and job2.closed == 1


def test_process_exception_marks_terminal_and_clears_reference(paths, store,
                                                               monkeypatch):
    asset = _asset(store, paths)
    draft = _draft(store, asset)
    scheduler = JobScheduler(store, paths, inline=False)
    batch = scheduler.submit(draft["id"])

    class Broken:
        def poll(self):
            raise AttributeError("poll 不可用")

        def release(self):
            self.released = True

    broken = Broken()
    monkeypatch.setattr(scheduler, "_spawn", lambda *args, **kwargs: broken)
    monkeypatch.setattr(scheduler, "_wait", lambda process, context: process.poll())
    store.update_batch(batch["id"], {"status": RUNNING})

    scheduler._run_batch(batch["id"])
    record = store.get_batch(batch["id"])
    assert record["status"] == FAILED
    assert record["videos"][0]["status"] == FAILED
    assert "执行失败" in str(record["videos"][0]["error"])
    assert scheduler._process is None
    assert getattr(broken, "released", False) is True
    assert not (paths.runs / f".{batch['videos'][0]['run_id']}.tmp").exists()


# -- 组 4：删除的输入保护、归属核对与旧历史 ------------------------------


def test_delete_refuses_registered_input_inside_result(paths, store):
    asset = _asset(store, paths)
    done = _completed_batch(store, paths, asset)
    run_id = done["videos"][0]["run_id"]
    input_path = paths.runs / run_id / "source.mp4"
    input_path.write_text("original", encoding="utf-8")
    registered = store.create_asset(name="source.mp4", path=str(input_path),
                                    project_id=None, metadata={})
    scheduler = JobScheduler(store, paths, inline=False)
    with pytest.raises(JobError):
        scheduler.delete(done["id"])
    assert input_path.is_file(), "已登记输入不能被任务清理带走"
    assert store.get_asset(registered["id"]) is not None
    assert store.get_batch(done["id"]) is not None, "记录保留可重试"


def test_delete_refuses_frozen_input_from_other_batch(paths, store):
    asset = _asset(store, paths)
    done = _completed_batch(store, paths, asset)
    run_id = done["videos"][0]["run_id"]
    shared = paths.runs / run_id / "shared.mp4"
    shared.write_text("shared", encoding="utf-8")
    # 另一批次的冻结 job 指向同一物理文件（未登记为资产）。
    other = store.create_batch(assay="oft", videos=[{
        "asset_id": "z" * 32, "name": "shared.mp4", "run_id": "s" * 32,
        "status": QUEUED,
        "job": {"assay": "oft", "run_id": "s" * 32, "job_id": "t" * 32,
                "asset": {"id": "z" * 32, "path": str(shared),
                          "identity": {"path": str(shared)}}},
    }])
    scheduler = JobScheduler(store, paths, inline=False)
    with pytest.raises(JobError):
        scheduler.delete(done["id"])
    assert shared.is_file()
    assert store.get_batch(other["id"]) is not None


def test_delete_keeps_conflicting_task_and_stage(paths, store):
    asset = _asset(store, paths)
    done = _completed_batch(store, paths, asset)
    run_id = done["videos"][0]["run_id"]
    # 篡改 task 的 job.json 与 stage 的 frozen_job.json：自称别的任务。
    task_job = paths.tasks / done["id"] / run_id / "job.json"
    atomic_write_json(task_job, {"job_id": "x" * 32, "run_id": run_id})
    stage = paths.runs / f".{run_id}.tmp"
    stage.mkdir(parents=True, exist_ok=True)
    atomic_write_json(stage / "frozen_job.json",
                      {"job_id": "x" * 32, "run_id": run_id})
    (stage / "summary.csv").write_text("x", encoding="utf-8")

    scheduler = JobScheduler(store, paths, inline=False)
    scheduler.delete(done["id"])
    assert task_job.is_file(), "冲突的 task 材料原样保留"
    assert (stage / "frozen_job.json").is_file(), "冲突的 stage 材料保留"
    assert store.get_batch(done["id"]) is None


def test_delete_recovers_legacy_owned_runs_from_tasks(paths, store):
    batch_id, run_id = "b" * 32, "c" * 32
    stage_dir = paths.tasks / batch_id / run_id
    stage_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(stage_dir / "job.json",
                      {"job_id": batch_id, "run_id": run_id})
    atomic_write_json(stage_dir / "progress.json", {"fraction": 1.0})
    (stage_dir / "notes.txt").write_text("keep", encoding="utf-8")
    result_dir = paths.runs / run_id
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "summary.csv").write_text("x", encoding="utf-8")
    _write_manifest(result_dir, {"status": "completed", "run_id": run_id,
                                 "job_id": batch_id,
                                 "files": ["summary.csv"], "units": []})
    batch = store.create_batch(assay="oft", batch_id=batch_id, videos=[{
        "asset_id": "a" * 32, "name": "a.mp4", "run_id": run_id,
        "job_id": batch_id, "status": COMPLETED, "progress": 1.0,
        "job": {"assay": "oft", "run_id": run_id, "job_id": batch_id,
                "asset": {"id": "a" * 32, "path": str(paths.root / "a.mp4")}},
        "result_manifest": None, "error": None}])
    store.update_batch(batch["id"], {"owned_run_ids": [],
                                     "status": COMPLETED,
                                     "finished_at": "2026-01-01T00:00:00Z"})

    scheduler = JobScheduler(store, paths, inline=False)
    scheduler.delete(batch_id)
    assert store.get_batch(batch_id) is None
    assert not (result_dir / "summary.csv").exists(), "可信历史结果被清理"
    assert (stage_dir / "notes.txt").is_file(), "不能认领的子项保留"
    assert not (paths.runs / f".{run_id}.tmp").exists()


# -- 组 5：重跑事务与项目动物 --------------------------------------------


def test_completed_replay_after_concurrent_delete_is_not_found(paths, store,
                                                               monkeypatch):
    asset = _asset(store, paths)
    done = _completed_batch(store, paths, asset)
    paused = threading.Event()
    resume = threading.Event()

    replaying = JobScheduler(store, paths, inline=False)
    real_refresh = replaying._refresh_frozen

    def paused_refresh(frozen, **kwargs):
        paused.set()
        resume.wait(timeout=5.0)
        return real_refresh(frozen, **kwargs)

    monkeypatch.setattr(replaying, "_refresh_frozen", paused_refresh)
    outcome: dict = {}

    def run():
        try:
            replaying.replay(done["id"])
        except Exception as error:  # 记录给主线程断言
            outcome["error"] = error

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert paused.wait(timeout=5.0)
    other = JobScheduler(store, paths, inline=False)
    other.delete(done["id"])
    resume.set()
    thread.join(timeout=5.0)
    assert isinstance(outcome.get("error"), NotFoundError), outcome
    assert store.list_batches() == [], "删除已提交后不能派生新批次"


def test_replay_requires_existing_project_animal(paths, store):
    project = store.create_project("项目甲", "")
    animal = store.create_animal(project["id"], "小鼠 1")
    asset = _asset(store, paths)
    draft = store.create_draft("oft", project_id=project["id"],
                               output={"annotated_video": False}, videos=[{
        "asset_id": asset["id"],
        "window": {"start_s": 0.0, "end_s": 1.0},
        "orientation": 0, "fps_override": None, "sample_fps": None,
        "units": [{"id": "u" * 32, "name": "小鼠 1",
                   "animal_id": animal["id"], "window": None,
                   "geometry": {"arena": {"type": "quad", "points": [
                       [10, 10], [150, 10], [150, 110], [10, 110]]}},
                   "options": {}, "detection": {}, "manual": None}],
    }])
    scheduler = JobScheduler(store, paths, inline=True,
                             runner=_runner(fail=True))
    failed = scheduler.submit(draft["id"])
    assert failed["status"] == FAILED
    store.delete_animal(animal["id"])

    replaying = JobScheduler(store, paths, inline=False)
    with pytest.raises(JobError) as captured:
        replaying.replay(failed["id"])
    fields = captured.value.fields
    assert any(item.get("field", "").endswith("animal_id") for item in fields)
    assert any(item.get("asset_id") for item in fields)
    assert store.get_batch(failed["id"])["status"] == FAILED, "原终态保持"


# -- 组 6：最终提交校验边界 ----------------------------------------------


def test_orientation_bool_and_float_rejected(paths, store):
    from behaviourlab.web.app import create_app

    app = create_app(data_dir=str(paths.root), store=store, testing=True)
    client = app.test_client()
    asset = _asset(store, paths)
    for value in (False, 0.0):
        draft = _draft(store, asset)
        response = client.put(f"/api/drafts/{draft['id']}",
                              json={"videos": [{**draft["videos"][0],
                                                "orientation": value}]})
        assert response.status_code == 400, (value, response.get_json())
        fields = response.get_json()["error"]["fields"]
        assert any(item["field"].endswith("orientation") for item in fields)
        assert client.get("/api/jobs").get_json()["data"]["batches"] == []
    # 合法方向仍然通过。
    draft = _draft(store, asset)
    ok = client.put(f"/api/drafts/{draft['id']}",
                    json={"videos": [{**draft["videos"][0],
                                      "orientation": 90}]})
    assert ok.status_code == 200, ok.get_json()


def test_identity_mtime_mismatch_is_reported(paths, store):
    asset = _asset(store, paths)
    draft = _draft(store, asset)
    scheduler = JobScheduler(store, paths, inline=False)
    batch = scheduler.submit(draft["id"])
    job = dict(batch["videos"][0]["job"])
    job["asset"] = {**job["asset"],
                    "identity": {**job["asset"]["identity"],
                                 "mtime": job["asset"]["identity"]["mtime"] + 1.0}}
    issues = engine.validate_job(job)
    assert any(item["field"] == "asset.identity" for item in issues)
    # 未改动的 identity 仍然合法。
    assert not [item for item in engine.validate_job(batch["videos"][0]["job"])
                if item["field"] == "asset.identity"]


# -- 组 7：任务中心与项目草稿语义 ----------------------------------------


def test_tasks_and_project_draft_semantics(paths, store):
    from behaviourlab.web.app import create_app

    app = create_app(data_dir=str(paths.root), store=store, testing=True)
    client = app.test_client()
    asset = _asset(store, paths)
    project = store.create_project("项目乙", "")

    independent = _draft(store, asset)                       # 有视频、未提交
    empty_project = store.create_draft("oft", project_id=project["id"])
    project_draft = store.create_draft("oft", project_id=project["id"],
                                       videos=independent["videos"])
    submitted = _draft(store, asset)
    JobScheduler(store, paths, inline=True, runner=_runner()).submit(
        submitted["id"])

    tasks = client.get("/tasks").get_data(as_text=True)
    assert f"draft_id={independent['id']}" in tasks, "有视频的未提交草稿应可继续"
    assert f"draft_id={empty_project['id']}" not in tasks, "空草稿不是继续设置"
    assert f"draft_id={project_draft['id']}" not in tasks, "项目草稿不进独立入口"
    assert f"draft_id={submitted['id']}" not in tasks, "已提交草稿不再列继续设置"

    page = client.get(f"/projects/{project['id']}").get_data(as_text=True)
    assert "草稿（可继续编辑）" in page
    assert f"draft_id={project_draft['id']}" in page
    assert f"draft_id={independent['id']}" not in page, "独立草稿不进项目页"
