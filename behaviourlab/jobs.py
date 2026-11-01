"""顺序任务调度骨架。

构造调度器不会启动线程或工作进程；批量执行在任务块接入。
"""

from __future__ import annotations

from typing import Any, Mapping

from . import engine
from .schema import new_id
from .store import Store


class JobScheduler:
    def __init__(self, store: Store) -> None:
        self.store = store
        self._started = False

    @property
    def running(self) -> bool:
        return False

    def freeze_batch(
        self, *, draft: Mapping[str, Any], assets: Mapping[str, Mapping]
    ) -> list[dict]:
        """为草稿内每个视频生成冻结任务，不执行。"""

        run_id = new_id()
        resolved: list[dict] = []
        for video in draft.get("videos", []):
            asset = assets.get(video.get("asset_id"))
            if asset is None:
                continue
            resolved.append(
                engine.build_resolved_job(
                    draft=draft,
                    video=video,
                    asset=asset,
                    job_id=new_id(),
                    run_id=run_id,
                    # 完整默认值已在草稿创建时快照；提交不再读取新的设置。
                    settings={},
                )
            )
        return resolved

    def run(self, *, draft: Mapping[str, Any],
            resolved: list[dict]) -> list[dict]:
        """顺序执行冻结任务并登记状态；失败如实记录，不伪装成功。"""

        from . import worker

        outcomes: list[dict] = []
        for job in resolved:
            self.store.create_job(
                assay=job.get("assay"),
                run_id=job.get("run_id"),
                job_id=job.get("job_id"),
                draft_id=draft.get("id"),
                project_id=job.get("project_id"),
                status="running",
                config={"job": job},
            )
            try:
                result = worker.run_frozen_job(job)
            except Exception as error:  # 任务失败要能被结果页看到
                self.store.update_job_status(job["job_id"], "error")
                outcomes.append({"job": job, "status": "error",
                                 "error": str(error)})
                continue
            self.store.update_job_status(job["job_id"], "done")
            outcomes.append({"job": job, "status": "done", "result": result})
        return outcomes

    def start(self) -> None:
        raise NotImplementedError("后台调度将在后续块接入")

    def stop(self) -> None:
        self._started = False
