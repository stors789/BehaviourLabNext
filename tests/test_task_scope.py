"""项目页任务列表范围：真实渲染上下文 + 生产 tasks.js。

假 DOM 只提供 tasks.js 需要的节点与 fetch 记录，不代表浏览器渲染；
真实浏览器点击/滚动不在本文件覆盖范围。
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def make_batch(store, project_id, name, status="queued"):
    return store.create_batch(
        assay="oft",
        project_id=project_id,
        status=status,
        videos=[{"asset_id": "a" * 32, "name": name, "run_id": None,
                 "status": status}],
    )


@pytest.fixture
def scoped(client, store):
    project_a = store.create_project("项目 A")
    project_b = store.create_project("项目 B")
    a_batch = make_batch(store, project_a["id"], "a.mp4", "running")
    b_batch = make_batch(store, project_b["id"], "b.mp4", "running")
    solo = make_batch(store, None, "solo.mp4", "completed")
    return {
        "a": project_a, "b": project_b,
        "a_batch": a_batch, "b_batch": b_batch, "solo": solo,
    }


def seed_of(html: str) -> list:
    match = re.search(
        r'<script id="task-seed" type="application/json">(.*?)</script>',
        html, re.S)
    assert match, "页面必须给出任务种子"
    return json.loads(match.group(1))


def project_scope_of(html: str) -> str:
    match = re.search(r'id="task-list"[^>]*data-project="([^"]*)"', html)
    assert match, "项目页任务列表必须带 project_id"
    return match.group(1)


def test_api_list_is_project_scoped(client, scoped):
    project_a = scoped["a"]["id"]
    response = client.get(f"/api/jobs?project_id={project_a}")
    assert response.status_code == 200
    batches = response.get_json()["data"]["batches"]
    assert [item["id"] for item in batches] == [scoped["a_batch"]["id"]]

    every = client.get("/api/jobs").get_json()["data"]["batches"]
    assert {item["id"] for item in every} == {
        scoped["a_batch"]["id"], scoped["b_batch"]["id"], scoped["solo"]["id"]}


def test_project_page_hands_its_scope_to_tasks_js(client, scoped):
    html = client.get(f"/projects/{scoped['a']['id']}").get_data(as_text=True)
    assert project_scope_of(html) == scoped["a"]["id"]
    ids = {item["id"] for item in seed_of(html)}
    assert ids == {scoped["a_batch"]["id"]}, "项目页种子只含本项目批次"
    assert 'id="task-read-error"' in html, "项目页要能就地报读取失败"


def test_tasks_scope_runtime(client, scoped):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node 不可用")
    project_html = client.get(
        f"/projects/{scoped['a']['id']}").get_data(as_text=True)
    global_html = client.get("/tasks").get_data(as_text=True)

    payload = {
        "projectId": project_scope_of(project_html),
        "seed": seed_of(project_html),
        "aBatch": scoped["a_batch"]["id"],
        "bBatch": scoped["b_batch"]["id"],
        "independentBatch": scoped["solo"]["id"],
        "otherProjectId": scoped["b"]["id"],
    }
    # 全局页也必须是真实页面上下文：确认它确实带三个批次。
    assert len(seed_of(global_html)) == 3

    script = Path(__file__).parent / "js" / "test_task_scope.cjs"
    result = subprocess.run(
        [node, str(script)], input=json.dumps(payload), capture_output=True,
        text=True, timeout=60, encoding="utf-8",
    )
    assert result.returncode == 0, result.stderr
    assert "task-scope OK" in result.stdout
