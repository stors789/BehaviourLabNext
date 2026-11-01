"""Barnes 人工洞序列与完整窗预览：受控观测源与固定输入夹具。"""

from __future__ import annotations

import math

import pytest

from behaviourlab import timeline
from behaviourlab.experiments import barnes
from behaviourlab.schema import ValidationError

UNIT = "u" * 32
DISK = {"type": "ellipse", "center": [100.0, 100.0],
        "radii": [80.0, 80.0], "angle": 0.0}
THREE_HOLES = [
    {"id": "1", "center": [160.0, 100.0]},
    {"id": "2", "center": [70.0, 100.0 + 30.0 * math.sqrt(3.0)]},
    {"id": "3", "center": [70.0, 100.0 - 30.0 * math.sqrt(3.0)]},
]


def unit(*, manual=None, options=None, window=None):
    merged = dict(barnes.DEFAULTS, hole_count=3, target_hole="1")
    if options:
        merged.update(options)
    return {"id": UNIT, "name": "小鼠 1", "animal_id": None,
            "window": window,
            "geometry": {"arena": DISK, "holes": THREE_HOLES},
            "options": merged, "detection": {}, "manual": manual}


def steady_rows(count=20, step=0.2, body=(100.0, 100.0), nose=None,
                quality=0.9):
    """固定节拍的身体观测；默认没有合格鼻尖。"""

    rows = []
    for index in range(count):
        start = round(step * index, 10)
        end = round(start + step, 10)
        has_nose = nose is not None
        rows.append({
            "unit_id": UNIT, "unit_name": "小鼠 1", "animal_id": None,
            "frame": index, "time_s": start, "dt_s": end - start,
            "source_dt_s": end - start, "elapsed_s": start,
            "analysis_fps": 1.0 / step,
            "source_interval_start_s": start,
            "source_interval_end_s": end,
            "interval_start_s": start, "interval_end_s": end,
            "in_window": True, "detected": True,
            "x_px": float(body[0]), "y_px": float(body[1]),
            "quality": quality, "contour": [], "missing_reason": None,
            "head_angle": None,
            "head_quality": 0.9 if has_nose else 0.0,
            "nose_x_px": float(nose[0]) if has_nose else None,
            "nose_y_px": float(nose[1]) if has_nose else None,
            "tail_x_px": None, "tail_y_px": None,
            "left_ear_x_px": None, "left_ear_y_px": None,
            "right_ear_x_px": None, "right_ear_y_px": None,
        })
    return rows


def process(payload, rows, window=(0.0, 4.0)):
    return barnes.process_unit(payload, rows, timeline.Window(*window), None)


MANUAL = {"hole_visits": [{"time_s": 0.4, "hole_id": "2"},
                          {"time_s": 1.2, "hole_id": "2"},
                          {"time_s": 2.4, "hole_id": "1"}],
          "escape_time_s": 3.2}


# -- 夹具 1：人工序列决定主访问与入箱 -----------------------------------

def test_manual_visits_drive_primary_and_entry():
    facts = process(unit(manual=dict(MANUAL)), steady_rows())
    summary = facts["summary"]

    assert summary["visit_source"] == "manual"
    assert summary["visits"] == 3
    assert summary["unscored_visits"] == 0
    assert summary["post_escape_visits"] == 0
    assert summary["primary_censored"] is False
    assert summary["primary_errors"] == 2
    assert summary["primary_latency_s"] == pytest.approx(2.4, abs=1e-8)
    assert summary["primary_path_cm"] == pytest.approx(0.0, abs=1e-8)
    assert summary["total_errors"] == 2
    assert summary["total_latency_s"] == pytest.approx(3.2, abs=1e-8)
    assert summary["total_path_cm"] == pytest.approx(0.0, abs=1e-8)
    assert summary["total_censored"] is False
    assert summary["escape_candidates"] == 0

    visits = [item for item in facts["events"] if item["kind"] == "hole_visit"]
    assert [(item["time_s"], item["confirmed_s"], item["hole_id"],
             item["source"], item["scored"]) for item in visits] == [
        (0.4, 0.4, "2", "manual", True),
        (1.2, 1.2, "2", "manual", True),
        (2.4, 2.4, "1", "manual", True),
    ]
    assert len({item["event_id"] for item in facts["events"]}) == len(
        facts["events"])

    confirmed = [item for item in facts["events"]
                 if item["kind"] == "escape_confirmed"]
    assert len(confirmed) == 1
    assert confirmed[0]["time_s"] == pytest.approx(3.2, abs=1e-8)
    assert confirmed[0]["confirmed_s"] == pytest.approx(3.2, abs=1e-8)
    assert confirmed[0]["source"] == "manual"
    assert confirmed[0]["target"] == "1"

    holes = {item["hole_id"]: item for item in facts["tables"]["holes"]}
    assert holes["2"]["visits"] == 2
    assert holes["2"]["automatic_visits"] == 0
    assert holes["1"]["visits"] == 1
    assert facts["tables"]["automatic_hole_visits"] == []
    assert facts["metadata"]["manual_empty_means"] == "zero visits"
    assert facts["metadata"]["visit_source"] == "manual"
    # 人工模式不套用自动鼻尖覆盖警示；身体覆盖仍按真实捕获统计。
    assert facts["warning"] is None
    assert summary["nose_coverage_percent"] == pytest.approx(0.0, abs=1e-8)
    assert summary["body_coverage_percent"] == pytest.approx(100.0, abs=1e-8)


def test_manual_empty_list_is_zero_visits():
    manual = {"hole_visits": [], "escape_time_s": 3.2}
    facts = process(unit(manual=manual), steady_rows())
    summary = facts["summary"]
    assert summary["visit_source"] == "manual"
    assert summary["visits"] == 0
    assert summary["primary_censored"] is True
    assert summary["primary_latency_s"] is None
    assert summary["primary_path_cm"] is None
    assert summary["primary_errors"] == 0
    assert summary["total_errors"] == 0
    assert summary["total_latency_s"] == pytest.approx(3.2, abs=1e-8)
    assert [item for item in facts["events"]
            if item["kind"] == "hole_visit"] == []


def test_manual_null_uses_automatic_visits():
    nose = (160.0, 100.0)
    rows = steady_rows(count=6, step=0.2, nose=nose)
    automatic = process(unit(manual=None), rows, window=(0.0, 1.2))
    assert automatic["summary"]["visit_source"] == "automatic"
    assert len([item for item in automatic["events"]
                if item["kind"] == "hole_visit"]) == 1
    assert automatic["warning"] is None, "鼻尖覆盖 100% 时没有自动警示"

    manual = process(unit(manual={"hole_visits": []}), rows, window=(0.0, 1.2))
    assert manual["summary"]["visit_source"] == "manual"
    assert manual["summary"]["visits"] == 0
    assert manual["tables"]["automatic_hole_visits"], "自动审计记录仍保留"


# -- 夹具 3：短序列的半开窗与事后评分 -----------------------------------

def test_manual_short_sequence_scores_after_escape():
    manual = {"hole_visits": [{"time_s": 0.1, "hole_id": "2"},
                              {"time_s": 0.4, "hole_id": "1"}],
              "escape_time_s": 0.3}
    facts = process(unit(manual=manual), steady_rows(count=6, step=0.1),
                    window=(0.0, 0.6))
    summary = facts["summary"]
    assert summary["visits"] == 1
    assert summary["unscored_visits"] == 1
    assert summary["post_escape_visits"] == 1
    assert summary["primary_censored"] is True
    assert summary["primary_latency_s"] is None
    assert summary["primary_errors"] == 1
    assert summary["total_errors"] == 1
    assert summary["total_latency_s"] == pytest.approx(0.3, abs=1e-8)
    visits = [item for item in facts["events"] if item["kind"] == "hole_visit"]
    assert [(item["time_s"], item["scored"]) for item in visits] == [
        (0.1, True), (0.4, False)]


def test_automatic_visit_after_escape_stays_unscored():
    manual = {"escape_time_s": 0.3}
    rows = steady_rows(count=2, step=0.6, nose=(160.0, 100.0))
    facts = process(unit(manual=manual,
                         options={"visit_confirm_s": 0.4}),
                    rows, window=(0.0, 1.2))
    summary = facts["summary"]
    assert summary["visit_source"] == "automatic"
    assert summary["visits"] == 0
    assert summary["unscored_visits"] == 1
    assert summary["post_escape_visits"] == 0
    assert summary["primary_censored"] is True
    assert summary["total_errors"] == 0
    assert summary["total_latency_s"] == pytest.approx(0.3, abs=1e-8)
    visits = [item for item in facts["events"] if item["kind"] == "hole_visit"]
    assert [(item["time_s"], item["confirmed_s"], item["scored"])
            for item in visits] == [(0.0, 0.6, False)]


# -- 窗口边界与基础校验 -------------------------------------------------

def test_manual_times_must_fit_animal_window():
    inside = {"hole_visits": [{"time_s": 0.4, "hole_id": "2"}],
              "escape_time_s": 4.0}
    facts = process(unit(manual=inside), steady_rows())
    assert facts["summary"]["total_latency_s"] == pytest.approx(4.0, abs=1e-8)

    late = {"hole_visits": [{"time_s": 4.0, "hole_id": "2"}]}
    with pytest.raises(ValidationError) as captured:
        process(unit(manual=late), steady_rows())
    assert captured.value.field == "manual.hole_visits[0].time_s"

    beyond = {"escape_time_s": 4.2}
    with pytest.raises(ValidationError) as captured:
        process(unit(manual=beyond), steady_rows())
    assert captured.value.field == "manual.escape_time_s"

    # 没有有效动物窗（相对窗无锚点）时人工记录无法校验。
    relative = unit(manual={"hole_visits": [{"time_s": 0.4, "hole_id": "2"}]},
                    window={"mode": "relative", "offset_s": 0.0,
                            "duration_s": 1.0})
    with pytest.raises(ValidationError) as captured:
        barnes.process_unit(relative, [], timeline.Window(0.0, 4.0), None)
    assert captured.value.field == "manual.hole_visits"


def test_manual_input_structure_validation():
    options = dict(barnes.DEFAULTS, hole_count=3, target_hole="1")

    def issues(manual):
        return barnes.validate_unit_manual(unit(manual=manual), options=options)

    assert issues(None) == []
    assert issues({}) == []
    assert issues({"hole_visits": [], "escape_time_s": None}) == []
    assert issues(dict(MANUAL)) == []

    unknown = {"hole_visits": [{"time_s": 0.4, "hole_id": "9"}]}
    assert [item["field"] for item in issues(unknown)] == [
        "manual.hole_visits[0].hole_id"]

    unordered = {"hole_visits": [{"time_s": 1.0, "hole_id": "1"},
                                 {"time_s": 0.5, "hole_id": "2"}]}
    assert [item["field"] for item in issues(unordered)] == [
        "manual.hole_visits[1].time_s"]

    same_moment = {"hole_visits": [{"time_s": 1.0, "hole_id": "1"},
                                   {"time_s": 1.0, "hole_id": "1"}]}
    assert issues(same_moment) == []

    missing_time = {"hole_visits": [{"hole_id": "1"}]}
    assert [item["field"] for item in issues(missing_time)] == [
        "manual.hole_visits[0].time_s"]

    negative = {"hole_visits": [{"time_s": -1, "hole_id": "1"}]}
    assert [item["field"] for item in issues(negative)] == [
        "manual.hole_visits[0].time_s"]

    boolean = {"hole_visits": [{"time_s": True, "hole_id": "1"}]}
    assert [item["field"] for item in issues(boolean)] == [
        "manual.hole_visits[0].time_s"]

    empty_hole = {"hole_visits": [{"time_s": 0.4, "hole_id": " "}]}
    assert [item["field"] for item in issues(empty_hole)] == [
        "manual.hole_visits[0].hole_id"]

    assert [item["field"] for item in issues({"hole_visits": "x"})] == [
        "manual.hole_visits"]
    assert [item["field"] for item in issues({"escape_time_s": -1})] == [
        "manual.escape_time_s"]
    assert [item["field"] for item in issues({"escape_time_s": "3"})] == [
        "manual.escape_time_s"]

    probe = dict(barnes.DEFAULTS, hole_count=3, target_hole="1",
                 trial_type="probe")
    assert [item["field"] for item in
            barnes.validate_unit_manual(unit(manual={"escape_time_s": 1.0}),
                                        options=probe)] == [
        "manual.escape_time_s"]


# -- 夹具 2：完整配置窗预览与保存前后一致性 -----------------------------

def seed(store, tmp_path, manual, *, window=None, duration=4.0):
    window = window or {"start_s": 0.0, "end_s": duration}
    source = tmp_path / "barnes.mp4"
    source.write_bytes(b"placeholder")
    asset = store.create_asset(
        name="barnes.mp4", path=str(source), project_id=None,
        metadata={"width": 200, "height": 200, "duration_s": duration,
                  "fps": 5.0, "frame_count": int(duration * 5)})
    merged = dict(barnes.DEFAULTS, hole_count=3, target_hole="1")
    draft = store.create_draft("barnes", videos=[{
        "asset_id": asset["id"],
        "window": dict(window),
        "orientation": 0, "fps_override": None, "sample_fps": None,
        "units": [{"id": UNIT, "name": "小鼠 1", "animal_id": None,
                   "window": None,
                   "geometry": {"arena": DISK, "holes": THREE_HOLES},
                   "options": merged, "detection": {}, "manual": manual}],
    }])
    return asset, draft


def patch_preview(monkeypatch):
    import numpy as np

    from behaviourlab import media
    from behaviourlab import preview as preview_service
    from behaviourlab.media import FrameInfo
    from behaviourlab.schema import Detection, Frame

    class Frames:
        def __init__(self, path=None, *, start_s=0.0, end_s=None, **kwargs):
            begin = float(start_s or 0.0)
            stop = float(end_s if end_s is not None else begin + 4.0)
            count = max(1, int(round((stop - begin) / 0.2)))
            self._frames = []
            for index in range(count):
                moment = round(begin + 0.2 * index, 10)
                self._frames.append(Frame(
                    image=np.full((200, 200), 240, dtype=np.uint8),
                    index=index, time_s=moment, dt_s=0.2, elapsed_s=moment,
                    analysis_fps=5.0, interval_start_s=moment,
                    interval_end_s=moment + 0.2))

        def iter_frames(self):
            return iter(self._frames)

    def fake_frame(*args, **kwargs):
        return FrameInfo(image=np.full((200, 200), 240, dtype=np.uint8),
                         index=0, time_s=0.0, pts=0, time_base=None,
                         clock_source="pts")

    def track(run_key, unit_id, **kwargs):
        return Detection(center=(100.0, 100.0), quality=0.9, contour=[],
                         mask=None)

    monkeypatch.setattr(media, "VideoSource", Frames)
    monkeypatch.setattr(preview_service.media, "read_frame_at_time", fake_frame)
    monkeypatch.setattr(preview_service.media, "read_frame_info", fake_frame)
    monkeypatch.setattr(barnes.observation, "track", track)


def preview(client, asset, draft):
    return client.post(
        f"/api/drafts/{draft['id']}/preview",
        json={"asset_id": asset["id"], "unit_id": UNIT,
              "frame_index": 0, "kind": "behavior"})


def test_preview_keeps_full_configured_window(client, store, tmp_path,
                                              monkeypatch):
    asset, draft = seed(store, tmp_path, dict(MANUAL))
    patch_preview(monkeypatch)
    before = client.get(f"/api/drafts/{draft['id']}").get_json()["data"]["draft"]

    response = preview(client, asset, draft)
    assert response.status_code == 201, response.get_json()
    data = response.get_json()["data"]
    assert data["frame_index"] == 0
    assert data["preview_scope"]["mode"] == "configured"
    assert data["preview_scope"]["start_s"] == pytest.approx(0.0, abs=1e-8)
    assert data["preview_scope"]["end_s"] == pytest.approx(4.0, abs=1e-8)

    behavior = data["summary"]["behavior"]
    assert behavior["mode"] == "barnes"
    assert behavior["visit_source"] == "manual"
    assert behavior["visits"] == 3
    assert behavior["primary_latency_s"] == pytest.approx(2.4, abs=1e-8)
    assert behavior["total_latency_s"] == pytest.approx(3.2, abs=1e-8)
    assert behavior["window_start_s"] == pytest.approx(0.0, abs=1e-8)
    assert behavior["window_end_s"] == pytest.approx(4.0, abs=1e-8)
    assert any("人工洞访问" in text for text in data["warnings"])
    assert any("动物窗" in text for text in data["warnings"])
    # 人工只换访问序列：单帧检测仍运行，不能用纯人工 Y 的无检测路径。
    assert data["summary"]["detection"]["region"] == "arena"
    assert not any("不运行自动检测" in text for text in data["warnings"])

    after = client.get(f"/api/drafts/{draft['id']}").get_json()["data"]["draft"]
    assert after["videos"][0]["units"][0]["manual"] == \
        before["videos"][0]["units"][0]["manual"]
    assert after["videos"][0]["window"] == before["videos"][0]["window"]


def test_preview_empty_manual_is_zero_visits_and_null_is_automatic(
        client, store, tmp_path, monkeypatch):
    patch_preview(monkeypatch)
    asset, draft = seed(store, tmp_path,
                        {"hole_visits": [], "escape_time_s": 3.2})
    response = preview(client, asset, draft)
    assert response.status_code == 201, response.get_json()
    behavior = response.get_json()["data"]["summary"]["behavior"]
    assert behavior["visit_source"] == "manual"
    assert behavior["visits"] == 0
    assert behavior["primary_censored"] is True
    assert behavior["primary_latency_s"] is None
    assert behavior["primary_errors"] == 0
    assert behavior["total_latency_s"] == pytest.approx(3.2, abs=1e-8)

    asset2, draft2 = seed(store, tmp_path, None)
    response = preview(client, asset2, draft2)
    assert response.status_code == 201, response.get_json()
    behavior = response.get_json()["data"]["summary"]["behavior"]
    assert behavior["visit_source"] == "automatic"


def test_preview_enforces_manual_time_bounds(client, store, tmp_path,
                                             monkeypatch):
    patch_preview(monkeypatch)

    asset, draft = seed(store, tmp_path,
                        {"hole_visits": [{"time_s": 4.0, "hole_id": "2"}]})
    response = preview(client, asset, draft)
    assert response.status_code == 400, response.get_json()
    fields = [item["field"] for item in response.get_json()["error"]["fields"]]
    assert any(field.endswith("manual.hole_visits[0].time_s")
               for field in fields), fields

    asset, draft = seed(store, tmp_path, {"escape_time_s": 4.2})
    response = preview(client, asset, draft)
    assert response.status_code == 400, response.get_json()
    issues = response.get_json()["error"]["fields"]
    assert any(item["field"].endswith("manual.escape_time_s")
               for item in issues), issues
    assert all(item.get("unit_id") == UNIT for item in issues), issues

    # 右端点等于窗尾是合法的入箱时间。
    asset, draft = seed(store, tmp_path, {"escape_time_s": 4.0})
    response = preview(client, asset, draft)
    assert response.status_code == 201, response.get_json()
    assert response.get_json()["data"]["summary"]["behavior"]["total_latency_s"] \
        == pytest.approx(4.0, abs=1e-8)


def test_probe_rejects_saved_manual_entry_time(client, store, tmp_path):
    asset, draft = seed(store, tmp_path, {"escape_time_s": 1.0})
    unit_payload = dict(draft["videos"][0]["units"][0])
    unit_payload["options"] = dict(unit_payload["options"], trial_type="probe")
    response = client.put(
        f"/api/drafts/{draft['id']}",
        json={"videos": [dict(draft["videos"][0], units=[unit_payload])]})
    assert response.status_code == 400, response.get_json()
    fields = [item["field"] for item in response.get_json()["error"]["fields"]]
    assert any(field.endswith("manual.escape_time_s") for field in fields), fields

    # 界面切到 probe 时会写入 null：同一草稿随即可以保存。
    unit_payload = dict(unit_payload, manual={"escape_time_s": None})
    response = client.put(
        f"/api/drafts/{draft['id']}",
        json={"videos": [dict(draft["videos"][0], units=[unit_payload])]})
    assert response.status_code == 200, response.get_json()


def test_manual_result_page_marks_confirmed_entry(client, store):
    import json
    from pathlib import Path

    facts = process(unit(manual=dict(MANUAL)), steady_rows())
    run_id = "c" * 32
    job_dir = Path(store.paths.runs) / run_id / ("d" * 32)
    job_dir.mkdir(parents=True, exist_ok=True)
    summary = dict(facts["summary"], mode="barnes")
    result = {"assay": "barnes", "summary": [summary],
              "events": facts["events"], "bins": facts["bins"],
              "observations": [], "tables": facts["tables"],
              "warnings": [],
              "metadata": {"assay": "barnes", "coordinate_unit": "cm",
                           "units": {}, "warnings": []}}
    (job_dir / "results.json").write_text(json.dumps(result),
                                          encoding="utf-8")
    manifest = {"status": "completed", "run_id": run_id, "job_id": "d" * 32,
                "units": [{"unit_id": UNIT, "unit_name": "小鼠 1",
                           "summary": summary,
                           "figures": {"stem": "u"}}],
                "files": ["results.json"]}
    (job_dir / "result_manifest.json").write_text(json.dumps(manifest),
                                                  encoding="utf-8")

    page = client.get(f"/results/{run_id}").get_data(as_text=True)
    assert "人工确认" in page
    assert "计入" in page
    assert "未计入" not in page
    assert "暂无图表" in page
    # 来源说明跟随单位模式：人工主访问不再声称来自鼻尖证据。
    assert "主洞访问来自人工记录" in page
    assert "主洞访问来自鼻尖进入洞半径的自动证据" not in page


def test_automatic_result_page_keeps_nose_evidence(client, store):
    import json
    from pathlib import Path

    facts = process(unit(manual=None),
                    steady_rows(count=6, step=0.2, nose=(160.0, 100.0)),
                    window=(0.0, 1.2))
    run_id = "e" * 32
    job_dir = Path(store.paths.runs) / run_id / ("f" * 32)
    job_dir.mkdir(parents=True, exist_ok=True)
    summary = dict(facts["summary"], mode="barnes")
    result = {"assay": "barnes", "summary": [summary],
              "events": facts["events"], "bins": facts["bins"],
              "observations": [], "tables": facts["tables"],
              "warnings": [facts["warning"]] if facts["warning"] else [],
              "metadata": {"assay": "barnes", "coordinate_unit": "cm",
                           "units": {}, "warnings": []}}
    (job_dir / "results.json").write_text(json.dumps(result),
                                          encoding="utf-8")
    manifest = {"status": "completed", "run_id": run_id, "job_id": "f" * 32,
                "units": [{"unit_id": UNIT, "unit_name": "小鼠 1",
                           "summary": summary, "figures": {"stem": "u"}}],
                "files": ["results.json"]}
    (job_dir / "result_manifest.json").write_text(json.dumps(manifest),
                                                  encoding="utf-8")

    page = client.get(f"/results/{run_id}").get_data(as_text=True)
    assert "主洞访问来自鼻尖进入洞半径的自动证据" in page
    assert "主洞访问来自人工记录" not in page
    assert "未确认入箱不产生总逃避潜伏期" in page


# -- 三组修正：窗口时长、派生时刻与预览校验 -----------------------------

def test_zero_duration_window_cannot_confirm_entry():
    rows = steady_rows()
    zero_window = {"mode": "relative", "offset_s": 4.0, "duration_s": 1.0}
    payload = unit(manual={"hole_visits": [], "escape_time_s": 4.0},
                   window=zero_window)
    with pytest.raises(ValidationError) as captured:
        barnes.process_unit(payload, rows, timeline.Window(0.0, 4.0), 0.0)
    assert captured.value.field == "manual.escape_time_s"

    # [] + null 仍表示零访问：零时长窗也不生成确认事件或总逃避指标。
    quiet = barnes.process_unit(
        unit(manual={"hole_visits": []}, window=zero_window),
        rows, timeline.Window(0.0, 4.0), 0.0)
    assert quiet["summary"]["visits"] == 0
    assert not [item for item in quiet["events"]
                if item["kind"] == "escape_confirmed"]


def test_manual_confirmed_time_derives_from_time():
    manual = {"hole_visits": [{"time_s": 0.4, "hole_id": "1",
                               "confirmed_s": 2.0}],
              "escape_time_s": 1.0}
    facts = process(unit(manual=manual), steady_rows())
    visits = [item for item in facts["events"] if item["kind"] == "hole_visit"]
    assert [(item["time_s"], item["confirmed_s"], item["scored"])
            for item in visits] == [(0.4, 0.4, True)]
    summary = facts["summary"]
    assert summary["visits"] == 1
    assert summary["primary_latency_s"] == pytest.approx(0.4, abs=1e-8)

    # 自动访问仍保留真实的后续确认时刻，不跟着起点一起改。
    auto = process(unit(manual=None),
                   steady_rows(count=6, step=0.2, nose=(160.0, 100.0)),
                   window=(0.0, 1.2))
    auto_visit = [item for item in auto["events"]
                  if item["kind"] == "hole_visit"][0]
    assert auto_visit["time_s"] == 0.0
    assert auto_visit["confirmed_s"] == pytest.approx(0.2, abs=1e-8)


def test_preview_rejects_persisted_unordered_manual(client, store, tmp_path,
                                                    monkeypatch):
    patch_preview(monkeypatch)
    asset, draft = seed(
        store, tmp_path,
        {"hole_visits": [{"time_s": 6.2, "hole_id": "2"},
                         {"time_s": 5.4, "hole_id": "1"}]},
        window={"start_s": 5.0, "end_s": 10.0}, duration=10.0)
    before = client.get(f"/api/drafts/{draft['id']}").get_json()["data"]["draft"]

    response = preview(client, asset, draft)
    assert response.status_code == 400, response.get_json()
    issues = response.get_json()["error"]["fields"]
    assert any(item["field"].endswith("manual.hole_visits[1].time_s")
               for item in issues), issues
    assert all(item.get("unit_id") == UNIT for item in issues), issues

    after = client.get(f"/api/drafts/{draft['id']}").get_json()["data"]["draft"]
    assert after["videos"][0]["units"][0]["manual"] == \
        before["videos"][0]["units"][0]["manual"], "预览不得排序修复或写回"


def test_preview_reports_manual_entry_with_automatic_visits(client, store,
                                                            tmp_path,
                                                            monkeypatch):
    patch_preview(monkeypatch)
    asset, draft = seed(store, tmp_path, {"escape_time_s": 8.2},
                        window={"start_s": 5.0, "end_s": 10.0}, duration=10.0)
    response = preview(client, asset, draft)
    assert response.status_code == 201, response.get_json()
    data = response.get_json()["data"]
    behavior = data["summary"]["behavior"]
    assert data["preview_scope"]["mode"] == "configured"
    assert behavior["visit_source"] == "automatic"
    assert behavior["total_latency_s"] == pytest.approx(3.2, abs=1e-8)
    assert behavior["manual_entry_time_s"] == pytest.approx(8.2, abs=1e-8)
    warnings = data["warnings"]
    assert any("入箱端点由人工确认" in text for text in warnings), warnings
    assert any("动物窗" in text for text in warnings), warnings
    assert not any("人工洞访问" in text for text in warnings), warnings
