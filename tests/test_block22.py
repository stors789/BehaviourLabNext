"""第 22 块：顺序作业冻结、取消与重跑的三项固定自检。

用专用 Store 与受控 worker 替身，原片只用临时占位文件；不生成真实视频。
"""

from __future__ import annotations

import json
import os
import threading
import time
from copy import deepcopy
from pathlib import Path

import pytest

from behaviourlab import engine
from behaviourlab.config import ALGORITHM_VERSION, DataPaths
from behaviourlab.jobs import (
    CANCELLED,
    COMPLETED,
    FAILED,
    INTERRUPTED,
    NOT_STARTED,
    QUEUED,
    JobConflict,
    JobError,
    JobScheduler,
)
from behaviourlab.schema import ValidationError
from behaviourlab.store import NotFoundError, Store, atomic_write_json, read_json_file

WIDTH = 33.0


# -- 夹具 -----------------------------------------------------------------


def _video(name: str, tmp_path: Path, unit_id: str) -> dict:
    source = tmp_path / name
    source.write_bytes(b"placeholder-" + name.encode("utf-8"))
    return {"name": name, "path": str(source), "unit_id": unit_id}


def _draft(store: Store, videos: list[dict], *, width: float = WIDTH,
           project_id: str | None = None) -> dict:
    payload = []
    for item in videos:
        asset = store.create_asset(
            name=item["name"], path=item["path"], project_id=project_id,
            metadata={"width": 160, "height": 120, "duration_s": 1.0,
                      "fps": 10.0, "frame_count": 10})
        payload.append({
            "asset_id": asset["id"],
            "window": {"start_s": 0.0, "end_s": 1.0},
            "orientation": 0, "fps_override": None, "sample_fps": None,
            "units": [{
                "id": item["unit_id"], "name": "小鼠 1", "animal_id": None,
                "window": None,
                "geometry": {"arena": {"type": "quad", "points": [
                    [10, 10], [140, 10], [140, 100], [10, 100]]}},
                "options": {"arena_width_cm": width, "arena_height_cm": 40.0},
                "detection": {}, "manual": None,
            }],
        })
    return store.create_draft("oft", project_id=project_id, videos=payload)


def _fake_result():
    from behaviourlab.schema import AnalysisResult

    return AnalysisResult(
        observations=[{"unit_id": "u", "frame": 0, "time_s": 0.0,
                       "dt_s": 1.0}],
        summary=[{"unit_id": "u", "unit_name": "小鼠 1", "total_time_s": 1.0}],
        events=[], bins=[], warnings=[],
        metadata={"units": {"u": {"coordinate_unit": "cm"}}},
    )


def _runner_factory(store: Store, *, fail_index: int | None = None,
                    started: threading.Event | None = None,
                    release: threading.Event | None = None,
                    writes_progress: bool = True):
    """受控执行者：与真实子进程一样写 stage 清单，只替换执行本身。"""

    calls: list[dict] = []

    def runner(context):
        calls.append({"index": int(context["index"]),
                      "batch": str(context["batch"])})
        if writes_progress:
            atomic_write_json(context["progress_path"],
                              {"fraction": 0.5, "message": "分析中"})
        if started is not None:
            started.set()
        if release is not None:
            release.wait(timeout=5.0)
        if context["cancel"].is_set():
            return 1
        if fail_index is not None and int(context["index"]) == fail_index:
            atomic_write_json(context["progress_path"],
                              {"fraction": 0.2, "message": "分析失败",
                               "error": "源视频大小已变化"})
            return 1
        job = context["job"]
        stage = context["stage"]
        stage.mkdir(parents=True, exist_ok=True)
        (stage / "summary.csv").write_text("unit_id\n", encoding="utf-8")
        atomic_write_json(stage / "results.json", {"units": []})
        atomic_write_json(stage / "frozen_job.json", dict(job))
        atomic_write_json(stage / "result_manifest.json", {
            "status": "completed",
            "run_id": job["run_id"], "job_id": job["job_id"],
            "assay": job["assay"],
            "files": ["summary.csv", "results.json", "frozen_job.json",
                      "result_manifest.json"],
            "units": [], "warnings": [], "metadata": {},
        })
        return 0

    runner.calls = calls  # type: ignore[attr-defined]
    return runner


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    paths = DataPaths.from_dir(tmp_path / "block22")
    store = Store(paths)
    monkeypatch.setattr(
        engine.media, "probe_video",
        lambda path: {"width": 160, "height": 120, "duration_s": 1.0,
                      "fps": 10.0, "frame_count": 10})
    yield store, paths, tmp_path
    store.close()


# -- 自检一：顺序执行、失败即停、只发布完整结果 --------------------------


def test_batch_stops_at_first_failure(sandbox):
    store, paths, tmp_path = sandbox
    videos = [_video("a.mp4", tmp_path, "a" * 32),
              _video("b.mp4", tmp_path, "b" * 32),
              _video("c.mp4", tmp_path, "c" * 32)]
    draft = _draft(store, videos)
    runner = _runner_factory(store, fail_index=1)
    scheduler = JobScheduler(store, paths, inline=True, runner=runner)

    batch = scheduler.submit(draft["id"])
    assert [video["status"] for video in batch["videos"]] == [
        COMPLETED, FAILED, NOT_STARTED]
    assert batch["status"] == FAILED
    assert batch["videos"][1]["error"] == "源视频大小已变化"
    assert [call["index"] for call in runner.calls] == [0, 1], "失败后不再启动"

    # 冻结值来自草稿快照，不是当前默认值（单位层覆盖视频层）。
    unit_job = batch["videos"][0]["job"]["units"][0]
    assert unit_job["options"]["arena_width_cm"] == WIDTH
    assert batch["videos"][0]["job"]["options"]["arena_width_cm"] == 50.0
    assert batch["videos"][0]["job"]["algorithm_version"] == ALGORITHM_VERSION

    # 只有第一个视频有已发布清单；失败视频的 stage 已清理。
    published = batch["videos"][0]["run_id"]
    assert (paths.runs / published / "result_manifest.json").is_file()
    failed_run = batch["videos"][1]["run_id"]
    assert not (paths.runs / failed_run).exists()
    assert not (paths.runs / f".{failed_run}.tmp").exists()
    assert not (paths.runs / batch["videos"][2]["run_id"]).exists()

    # 逐视频记录与批次状态一致，进度不掩盖失败。
    assert batch["progress"]["current"] == 1
    assert batch["progress"]["total"] == 3
    assert batch["progress"]["fraction"] == pytest.approx(1 / 3, abs=1e-6)


# -- 自检二：原位重跑与已完成任务重跑 ------------------------------------


def test_replay_failed_batch_keeps_id_and_reuses_config(sandbox):
    store, paths, tmp_path = sandbox
    videos = [_video("a.mp4", tmp_path, "a" * 32),
              _video("b.mp4", tmp_path, "b" * 32),
              _video("c.mp4", tmp_path, "c" * 32)]
    draft = _draft(store, videos)
    scheduler = JobScheduler(store, paths, inline=True,
                             runner=_runner_factory(store, fail_index=1))
    failed = scheduler.submit(draft["id"])
    old_ids = [video["run_id"] for video in failed["videos"]]
    old_owned = list(failed["owned_run_ids"])

    # 提交之后改草稿与默认值，重跑都不能读它们。
    store.update_draft(draft["id"], {"options": {"arena_width_cm": 88.0}})
    store.set_setting("settings.defaults", {"oft": {"arena_width_cm": 99.0}})
    # 源文件身份变化：重跑必须记录新的快照。
    for video in failed["videos"]:
        path = Path(video["job"]["asset"]["path"])
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    replaying = JobScheduler(store, paths, inline=False)
    queued = replaying.replay(failed["id"])
    assert queued["id"] == failed["id"], "失败批次沿用原 job_id"
    new_ids = [video["run_id"] for video in queued["videos"]]
    assert len(set(new_ids) & set(old_ids)) == 0, "全部视频换新 run_id"
    assert [video["name"] for video in queued["videos"]] == [
        "a.mp4", "b.mp4", "c.mp4"]
    for video in queued["videos"]:
        assert video["job"]["units"][0]["options"]["arena_width_cm"] == WIDTH
        assert video["job"]["run_id"] == video["run_id"]
        assert video["status"] == QUEUED
    identity = queued["videos"][0]["job"]["asset"]["identity"]
    assert identity["mtime_ns"] == Path(
        queued["videos"][0]["job"]["asset"]["path"]).stat().st_mtime_ns
    assert set(old_owned) <= set(queued["owned_run_ids"]), "旧轮次仍归本任务"
    assert set(new_ids) <= set(queued["owned_run_ids"])

    # 另一个 Store 先排队后，删除必须被拒绝且文件不动。
    other = JobScheduler(store, paths, inline=False)
    with pytest.raises(JobConflict):
        other.delete(failed["id"])
    assert (paths.runs / old_ids[0]).exists(), "排队中删除不能动文件"


def test_replay_completed_batch_creates_new_record(sandbox):
    store, paths, tmp_path = sandbox
    videos = [_video("a.mp4", tmp_path, "a" * 32)]
    draft = _draft(store, videos)
    scheduler = JobScheduler(store, paths, inline=True,
                             runner=_runner_factory(store))
    done = scheduler.submit(draft["id"])
    assert done["status"] == COMPLETED
    old_video = deepcopy(done["videos"][0])

    second = JobScheduler(store, paths, inline=False)
    fresh = second.replay(done["id"])
    assert fresh["id"] != done["id"], "已完成任务重跑建立新批次"
    assert fresh["videos"][0]["run_id"] != old_video["run_id"]
    assert fresh["owned_run_ids"] == [fresh["videos"][0]["run_id"]]

    # 新任务深层修改不影响旧冻结文件与旧结果。
    fresh["videos"][0]["job"]["units"][0]["options"]["arena_width_cm"] = 77.0
    frozen = read_json_file(
        paths.runs / old_video["run_id"] / "frozen_job.json", default={})
    assert frozen["units"][0]["options"]["arena_width_cm"] == WIDTH
    original = store.get_batch(done["id"])
    assert original["videos"][0]["job"]["units"][0]["options"][
        "arena_width_cm"] == WIDTH
    assert (paths.runs / old_video["run_id"] / "summary.csv").is_file()

    # 选中终态后删除，再重跑必须是不存在，不能复活记录。
    second.delete(done["id"])
    with pytest.raises(NotFoundError):
        second.replay(done["id"])
    assert not (paths.runs / old_video["run_id"]).exists()
    assert store.get_batch(done["id"]) is None


# -- 自检三：取消、关闭与唯一所有者 --------------------------------------


def test_cancel_queued_batch_never_starts_worker(sandbox):
    store, paths, tmp_path = sandbox
    videos = [_video("a.mp4", tmp_path, "a" * 32)]
    draft = _draft(store, videos)
    runner = _runner_factory(store)
    scheduler = JobScheduler(store, paths, inline=False, runner=runner)
    batch = scheduler.submit(draft["id"])
    assert batch["status"] == QUEUED

    stopped = scheduler.cancel(batch["id"])
    assert stopped["status"] == CANCELLED
    assert [video["status"] for video in stopped["videos"]] == [NOT_STARTED]
    assert runner.calls == [], "排队批次取消后不得启动工作进程"


def test_running_cancel_terminates_and_cleans_stage(sandbox):
    store, paths, tmp_path = sandbox
    videos = [_video("a.mp4", tmp_path, "a" * 32),
              _video("b.mp4", tmp_path, "b" * 32)]
    draft = _draft(store, videos)
    started = threading.Event()
    release = threading.Event()
    runner = _runner_factory(store, started=started, release=release)
    scheduler = JobScheduler(store, paths, inline=False, runner=runner)
    batch = scheduler.submit(draft["id"])

    thread = threading.Thread(target=scheduler._run_batch, args=(batch["id"],),
                              daemon=True)
    thread.start()
    assert started.wait(timeout=5.0), "工作替身应已启动"
    stopping = scheduler.cancel(batch["id"])
    assert stopping["cancel_requested"] is True
    release.set()
    thread.join(timeout=5.0)
    assert not thread.is_alive()

    final = scheduler.get(batch["id"])
    assert final["status"] == CANCELLED
    assert [video["status"] for video in final["videos"]] == [
        CANCELLED, NOT_STARTED]
    run_id = final["videos"][0]["run_id"]
    assert not (paths.runs / run_id).exists()
    assert not (paths.runs / f".{run_id}.tmp").exists()


def test_shutdown_marks_interrupted_and_keeps_lock_for_owner(sandbox):
    store, paths, tmp_path = sandbox
    videos = [_video("a.mp4", tmp_path, "a" * 32)]
    draft = _draft(store, videos)
    started = threading.Event()
    release = threading.Event()
    runner = _runner_factory(store, started=started, release=release)
    owner = JobScheduler(store, paths, inline=False, runner=runner)
    batch = owner.submit(draft["id"])
    assert owner.start() is True, "空数据根应能取得所有权"

    assert started.wait(timeout=5.0), "调度线程应已开始执行第一个视频"

    # 第二个实例拿不到数据根所有权：不改记录，也不中止第一个实例。
    second = JobScheduler(store, paths)
    assert second.start() is False
    second.shutdown()
    assert store.get_batch(batch["id"])["status"] == "running"
    assert owner.owner.held is True

    owner.shutdown(timeout_s=0.5)   # 线程还卡在执行者里，先不放锁
    assert owner.owner.held is True, "旧线程仍可能用数据库时不能提前放锁"
    release.set()
    for _ in range(100):
        if not owner.running:
            break
        time.sleep(0.05)
    assert not owner.running
    final = store.get_batch(batch["id"])
    assert final["status"] == INTERRUPTED
    assert final["videos"][0]["status"] == INTERRUPTED
    assert final["error"] == "上次运行被中断"
    assert owner.owner.held is False, "线程退出后释放数据根所有权"


def test_second_instance_does_not_release_owner_lock(sandbox):
    store, paths, tmp_path = sandbox
    first = JobScheduler(store, paths)
    assert first.start() is True
    second = JobScheduler(store, paths)
    assert second.start() is False
    second.shutdown()          # 未持锁对象不得解锁别人的数据根
    assert first.owner.held is True
    first.shutdown()
    assert first.owner.held is False
    # 释放后新实例可以取得所有权。
    third = JobScheduler(store, paths)
    assert third.start() is True
    third.shutdown()
    assert tmp_path is not None


def test_recovery_marks_unpublished_running_video_interrupted(sandbox):
    store, paths, tmp_path = sandbox
    videos = [_video("a.mp4", tmp_path, "a" * 32)]
    draft = _draft(store, videos)
    scheduler = JobScheduler(store, paths, inline=False)
    batch = scheduler.submit(draft["id"])
    # 模拟上次运行中断：批次与视频都停在 running，没有已发布清单。
    store.update_batch(batch["id"], {
        "status": "running", "started_at": "2026-01-01T00:00:00Z",
        "videos": [{**batch["videos"][0], "status": "running"}],
    })
    stage = paths.runs / f".{batch['videos'][0]['run_id']}.tmp"
    stage.mkdir(parents=True, exist_ok=True)
    (stage / "summary.csv").write_text("x", encoding="utf-8")

    recovered = JobScheduler(store, paths)
    recovered._recover()
    record = store.get_batch(batch["id"])
    assert record["status"] == INTERRUPTED
    assert record["videos"][0]["status"] == INTERRUPTED
    assert not stage.exists(), "未发布的临时目录要清理"


def test_recovery_keeps_verified_completed_video(sandbox):
    store, paths, tmp_path = sandbox
    videos = [_video("a.mp4", tmp_path, "a" * 32)]
    draft = _draft(store, videos)
    scheduler = JobScheduler(store, paths, inline=True,
                             runner=_runner_factory(store))
    done = scheduler.submit(draft["id"])
    run_id = done["videos"][0]["run_id"]
    store.update_batch(done["id"], {
        "status": "running",
        "videos": [{**done["videos"][0], "status": "running"}],
    })
    recovered = JobScheduler(store, paths)
    recovered._recover()
    record = store.get_batch(done["id"])
    assert record["status"] == COMPLETED
    assert record["videos"][0]["status"] == COMPLETED
    assert record["videos"][0]["output_dir"].endswith(run_id)


# -- 提交校验与删除边界 --------------------------------------------------


def test_submit_reports_fields_without_creating_batch(sandbox):
    store, paths, tmp_path = sandbox
    draft = _draft(store, [_video("a.mp4", tmp_path, "a" * 32)])
    scheduler = JobScheduler(store, paths, inline=False)
    # 草稿还没有视频时不能创建半个批次。
    empty = store.create_draft("oft")
    with pytest.raises(JobError) as captured:
        scheduler.submit(empty["id"])
    assert captured.value.fields[0]["field"] == "videos"
    assert store.list_batches() == []

    # 素材不存在时问题指向该视频。
    broken = deepcopy(draft)
    broken["videos"][0]["asset_id"] = "z" * 32
    store.update_draft(draft["id"], {"videos": broken["videos"]})
    with pytest.raises(JobError) as missing:
        scheduler.submit(draft["id"])
    assert missing.value.fields[0]["asset_id"] == "z" * 32
    assert store.list_batches() == []


def test_delete_refuses_active_batch(sandbox):
    store, paths, tmp_path = sandbox
    draft = _draft(store, [_video("a.mp4", tmp_path, "a" * 32)])
    scheduler = JobScheduler(store, paths, inline=False)
    batch = scheduler.submit(draft["id"])
    with pytest.raises(JobConflict):
        scheduler.delete(batch["id"])
    assert store.get_batch(batch["id"]) is not None
    assert not list(paths.runs.iterdir())


def test_delete_keeps_unowned_material(sandbox):
    store, paths, tmp_path = sandbox
    draft = _draft(store, [_video("a.mp4", tmp_path, "a" * 32)])
    scheduler = JobScheduler(store, paths, inline=True,
                             runner=_runner_factory(store))
    done = scheduler.submit(draft["id"])
    run_id = done["videos"][0]["run_id"]
    # 结果目录里出现不属于本任务的子项：删除只清本任务能证明的部分。
    foreign = paths.runs / run_id / "someone-else.txt"
    foreign.write_text("keep", encoding="utf-8")

    scheduler.delete(done["id"])
    assert store.get_batch(done["id"]) is None
    # 本任务声明的文件被清理；不能认领的旁文件原样保留，目录因此不空。
    assert not (paths.runs / run_id / "summary.csv").exists()
    assert foreign.is_file(), "无归属材料不能跟着任务一起删"


def test_replay_rejects_running_batch(sandbox):
    store, paths, tmp_path = sandbox
    draft = _draft(store, [_video("a.mp4", tmp_path, "a" * 32)])
    scheduler = JobScheduler(store, paths, inline=False)
    batch = scheduler.submit(draft["id"])
    store.update_batch(batch["id"], {"status": "running"})
    with pytest.raises(JobConflict):
        scheduler.replay(batch["id"])
    assert tmp_path is not None


def test_local_service_shares_owner_and_stops_scheduler(sandbox):
    """启动器路径：服务先持锁，调度器复用同一份所有权，关闭按序释放。"""

    store, paths, tmp_path = sandbox
    from behaviourlab.service import LocalService
    from behaviourlab.web import create_app

    app = create_app(data_dir=str(paths.root), store=store, testing=True)
    jobs = app.config["JOBS"]

    class Server:
        def __init__(self):
            self.closed = False

        def run(self):
            time.sleep(0.05)

        def close(self):
            self.closed = True

    server = Server()
    service = LocalService(
        app, paths, port=0,
        server_factory=lambda *args, **kwargs: server,
        ready_probe=lambda url, root: (True, ""),
        owner_registry=app.config["JOB_OWNER"],
        stop_callbacks=[jobs.shutdown],
    )
    service.start()
    try:
        assert jobs.start() is True
        assert jobs.running is True
    finally:
        service.stop()
    assert server.closed is True
    assert jobs.running is False

    # 关闭后所有权真的释放：新实例可以取得。
    again = JobScheduler(store, paths)
    assert again.start() is True
    again.shutdown()
    assert again.owner.held is False


def test_local_service_shares_owner_and_stops_scheduler(sandbox):
    """启动器路径：服务先持锁，调度器复用同一份所有权，关闭按序释放。"""

    store, paths, tmp_path = sandbox
    from behaviourlab.service import LocalService
    from behaviourlab.web import create_app

    app = create_app(data_dir=str(paths.root), store=store, testing=True)
    jobs = app.config["JOBS"]

    class Server:
        def __init__(self):
            self.closed = False

        def run(self):
            time.sleep(0.05)

        def close(self):
            self.closed = True

    server = Server()
    service = LocalService(
        app, paths, port=0,
        server_factory=lambda *args, **kwargs: server,
        ready_probe=lambda url, root: (True, ""),
        owner_registry=app.config["JOB_OWNER"],
        stop_callbacks=[jobs.shutdown],
    )
    service.start()
    try:
        assert jobs.start() is True
        assert jobs.running is True
    finally:
        service.stop()
    assert server.closed is True
    assert jobs.running is False

    # 关闭后所有权真的释放：新实例可以取得。
    again = JobScheduler(store, paths)
    assert again.start() is True
    again.shutdown()
    assert again.owner.held is False
