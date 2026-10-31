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
                )
            )
        return resolved

    def start(self) -> None:
        raise NotImplementedError("任务调度将在后续块接入")

    def stop(self) -> None:
        self._started = False
