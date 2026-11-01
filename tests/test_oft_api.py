"""OFT 分析接口：冻结、执行、结果文件与结果页。"""

from __future__ import annotations

from behaviourlab import worker
from behaviourlab.experiments import oft
from behaviourlab.schema import AnalysisResult

ARENA = {"type": "quad", "points": [[0, 0], [199, 0], [199, 199], [0, 199]]}


def seed(store, tmp_path, options=None, window=None):
    source = tmp_path / "oft.mp4"
    source.write_bytes(b"placeholder")
    asset = store.create_asset(
        name="旷场.mp4", path=str(source), project_id=None,
        metadata={"width": 200, "height": 200, "duration_s": 1.0,
                  "fps": 5.0, "frame_count": 5},
    )
    draft = store.create_draft("oft", output={"annotated_video": False}, videos=[{
        "asset_id": asset["id"],
        "window": window or {"start_s": 0.0, "end_s": 1.0},
        "orientation": 0, "fps_override": None, "sample_fps": None,
        "units": [{
            "id": "d" * 32, "name": "小鼠 1", "window": None,
            "geometry": {"arena": ARENA},
            "options": dict(oft.DEFAULTS if options is None else options),
            "detection": {}, "manual": None,
        }],
    }])
    return asset, draft


def fake_result():
    return AnalysisResult(
        observations=[],
        summary=[{
            "unit_id": "d" * 32, "unit_name": "小鼠 1",
            "center_percent": 50.0, "outer_percent": 50.0, "wall_percent": 0.0,
            "total_distance_cm": 3.0, "mean_speed_cm_s": 1.5,
            "center_entries": 1,
            "analysis_window_start_s": 0.0, "analysis_window_end_s": 1.0,
        }],
        events=[], bins=[], warnings=["示例警告"],
        metadata={"units": {"d" * 32: {"coordinate_unit": "cm"}}},
    )


def test_analyze_endpoint_writes_results(client, store, tmp_path, monkeypatch):
    asset, draft = seed(store, tmp_path)
    monkeypatch.setattr(
        "behaviourlab.engine.media.probe_video",
        lambda path: {"width": 200, "height": 200, "duration_s": 1.0,
                      "fps": 5.0, "frame_count": 5})
    monkeypatch.setattr(worker, "run_frozen_job",
                        lambda job, **kwargs: fake_result())

    response = client.post(f"/api/drafts/{draft['id']}/analyze")
    assert response.status_code == 202, response.get_json()
    data = response.get_json()["data"]
    run_id = data["run_id"]
    assert data["jobs"] and data["jobs"][0]["status"] == "completed"
    job_id = data["jobs"][0]["job_id"]

    page = client.get(f"/results/{run_id}")
    assert page.status_code == 200
    body = page.get_data(as_text=True)
    assert "小鼠 1" in body
    assert "示例警告" in body

    summary_csv = client.get(f"/runs/{run_id}/{job_id}/summary.csv")
    assert summary_csv.status_code == 200
    assert "center_percent" in summary_csv.get_data(as_text=True)
    report = client.get(f"/runs/{run_id}/{job_id}/report.md")
    assert report.status_code == 200
    assert client.get(f"/runs/{run_id}/{job_id}/secret.txt").status_code == 404

    tasks = client.get("/tasks").get_data(as_text=True)
    assert run_id in tasks


def test_analyze_endpoint_rejects_invalid_options(client, store, tmp_path):
    values = dict(oft.DEFAULTS)
    values["arena_width_cm"] = 0.0
    asset, draft = seed(store, tmp_path, options=values)
    response = client.post(f"/api/drafts/{draft['id']}/analyze")
    assert response.status_code == 400
    fields = [issue["field"] for issue in response.get_json()["error"]["fields"]]
    assert "videos[0].units[0].options.arena_width_cm" in fields


def test_analyze_endpoint_rejects_crossed_arena(client, store, tmp_path,
                                                monkeypatch):
    asset, draft = seed(store, tmp_path)
    videos = draft["videos"]
    videos[0]["units"][0]["geometry"] = {
        "arena": {"type": "quad",
                  "points": [[0, 0], [199, 199], [199, 0], [0, 199]]}}
    # 已保存的坏四角：保存入口已拒绝，这里绕过它以检查最终分析入口。
    store.update_draft(draft["id"], {"videos": videos})
    monkeypatch.setattr(
        "behaviourlab.engine.media.probe_video",
        lambda path: {"width": 200, "height": 200, "duration_s": 1.0,
                      "fps": 5.0, "frame_count": 5})
    response = client.post(f"/api/drafts/{draft['id']}/analyze")
    assert response.status_code == 400
    messages = [issue["message"] for issue in response.get_json()["error"]["fields"]]
    assert "场地四角必须凸且不交叉" in messages


def test_analyze_requires_arena_before_creating_job(client, store, tmp_path,
                                                   monkeypatch):
    asset, draft = seed(store, tmp_path)
    videos = draft["videos"]
    videos[0]["units"][0]["geometry"] = {}
    # 未画完的草稿仍允许保存；只是不能开始任务。
    store.update_draft(draft["id"], {"videos": videos})
    assert client.put(f"/api/drafts/{draft['id']}",
                      json={"videos": videos}).status_code == 200
    monkeypatch.setattr(
        "behaviourlab.engine.media.probe_video",
        lambda path: {"width": 200, "height": 200, "duration_s": 1.0,
                      "fps": 5.0, "frame_count": 5})
    response = client.post(f"/api/drafts/{draft['id']}/analyze")
    assert response.status_code == 400, response.get_json()
    issues = response.get_json()["error"]["fields"]
    assert any(issue["field"] == "videos[0].units[0].geometry.arena"
               for issue in issues)
    assert any(issue["message"] == "缺少场地四角" for issue in issues)
    assert store.list_jobs() == []

    checked = client.get(f"/api/drafts/{draft['id']}/validate").get_json()["data"]
    assert any(issue["field"] == "videos[0].units[0].geometry.arena"
               for issue in checked["issues"])

    # 草稿仍在，补上四角即可继续。
    kept = client.get(f"/api/drafts/{draft['id']}").get_json()["data"]["draft"]
    assert kept["videos"][0]["units"][0]["geometry"] == {}
    fixed = client.get(f"/api/drafts/{draft['id']}").get_json()["data"]["draft"]
    fixed["videos"][0]["units"][0]["geometry"] = {"arena": ARENA}
    assert client.put(f"/api/drafts/{draft['id']}",
                      json={"videos": fixed["videos"]}).status_code == 200


def test_width_zero_save_reason_and_recovery(client, store, tmp_path):
    asset, draft = seed(store, tmp_path)
    videos = draft["videos"]
    videos[0]["units"][0]["options"]["arena_width_cm"] = 40.0
    assert client.put(f"/api/drafts/{draft['id']}",
                      json={"videos": videos}).status_code == 200

    videos[0]["units"][0]["options"]["arena_width_cm"] = 0.0
    bad = client.put(f"/api/drafts/{draft['id']}", json={"videos": videos})
    assert bad.status_code == 400
    issues = bad.get_json()["error"]["fields"]
    assert any(issue["field"].endswith("options.arena_width_cm")
               and issue["message"] == "场地宽度必须 > 0"
               for issue in issues), issues
    # 保存失败不回写：草稿仍是上一次成功的 40，界面保留用户输入的 0。
    stored = client.get(f"/api/drafts/{draft['id']}").get_json()["data"]["draft"]
    assert stored["videos"][0]["units"][0]["options"]["arena_width_cm"] == 40.0
    assert store.list_jobs() == []

    videos[0]["units"][0]["options"]["arena_width_cm"] = 40.0
    fixed = client.put(f"/api/drafts/{draft['id']}", json={"videos": videos})
    assert fixed.status_code == 200, fixed.get_json()
    kept = client.get(f"/api/drafts/{draft['id']}").get_json()["data"]["draft"]
    assert kept["videos"][0]["units"][0]["options"]["arena_width_cm"] == 40.0
