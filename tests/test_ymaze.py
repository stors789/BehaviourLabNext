"""Y 迷宫自动评分：固定序列夹具、短序列规则、候选关联与页面接线。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from behaviourlab import detection, experiments, figures, media, outputs, preview
from behaviourlab.experiments import ymaze
from behaviourlab.media import FrameInfo
from behaviourlab.schema import AnalysisContext

TOL = 1e-9
UNIT_ID = "u" * 32
RUN_ID = "r" * 32
JOB_ID = "j" * 32

ARM_A = (100, 30)
CENTER = (100, 97)
ARM_B = (170, 105)
ARM_C = (30, 105)

# 固定夹具：A×3、中心×3、A×3、中心×3、B×3、中心×3、C×3、缺失×1、C×1。
FIXTURE = ([ARM_A] * 3 + [CENTER] * 3 + [ARM_A] * 3 + [CENTER] * 3
           + [ARM_B] * 3 + [CENTER] * 3 + [ARM_C] * 3 + [None] + [ARM_C])


def rect(x1: float, y1: float, x2: float, y2: float) -> dict:
    return {"type": "quad",
            "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]}


def geometry() -> dict:
    return {"zones": {
        "center": {"type": "polygon",
                   "points": [[90, 90], [110, 90], [100, 110]]},
        "A": rect(90, 10, 110, 85),
        "B": rect(115, 95, 185, 115),
        "C": rect(15, 95, 85, 115),
    }}


def options(**overrides) -> dict:
    return dict(ymaze.DEFAULTS, **overrides)


def unit(values: dict | None = None) -> dict:
    return {"id": UNIT_ID, "name": "Y 鼠", "animal_id": None, "window": None,
            "geometry": geometry(), "options": values or options(), "detection": {}}


def rows(pattern) -> list[dict]:
    """源观测：第 i 条 time_s=.1i、曝光 [.1i, .1(i+1))；None 表示该帧缺失。"""

    result = []
    for index, point in enumerate(pattern):
        start = round(0.1 * index, 10)
        end = round(0.1 * (index + 1), 10)
        seen = point is not None
        result.append({
            "unit_id": UNIT_ID, "unit_name": "Y 鼠", "frame": index,
            "time_s": float(start), "source_dt_s": float(end - start),
            "elapsed_s": float(start), "analysis_fps": 10.0,
            "source_interval_start_s": float(start),
            "source_interval_end_s": float(end),
            "interval_start_s": float(start), "interval_end_s": float(end),
            "dt_s": float(end - start), "in_window": True,
            "detected": bool(seen),
            "x_px": None if not seen else float(point[0]),
            "y_px": None if not seen else float(point[1]),
            "quality": 1.0, "contour": [], "area_px": 200.0,
            "missing_reason": None if seen else "no_candidate",
            "head_angle": None, "head_quality": 0.0,
            "nose_x_px": None, "nose_y_px": None,
            "tail_x_px": None, "tail_y_px": None,
            "left_ear_x_px": None, "left_ear_y_px": None,
            "right_ear_x_px": None, "right_ear_y_px": None,
            "detection_method": "threshold",
        })
    return result


def candidates(items) -> list[dict]:
    """每帧一个真实候选（面积 200 px²）；缺失帧没有候选。"""

    return [{"unit_id": UNIT_ID, "unit_name": "Y 鼠", "frame": row["frame"],
             "time_s": row["time_s"], "candidate_index": 1,
             "x_px": row["x_px"], "y_px": row["y_px"], "quality": 1.0,
             "area_px": 200.0, "contour": [], "selected": True}
            for row in items if row["detected"]]


def run(pattern, *, values: dict | None = None, monkeypatch=None):
    """用直接给定的候选集合驱动 analyze：捕获层用替身，其余走真实核心。"""

    unit_value = unit(values)
    source = rows(pattern)
    facts = {"rows": source, "candidates": candidates(source)}
    job = {"assay": "ymaze", "run_id": RUN_ID, "job_id": JOB_ID,
           "asset": {"id": "asset", "name": "y.mp4", "path": "memory",
                     "metadata": {"duration_s": 0.1 * len(pattern)}},
           "window": {"start_s": 0.0, "end_s": 0.1 * len(pattern)},
           "units": [unit_value], "style": {}, "orientation": 0}
    if monkeypatch is not None:
        monkeypatch.setattr(ymaze, "capture",
                            lambda context, units: {UNIT_ID: facts})
    return ymaze.analyze(AnalysisContext(job=job))


# -- 1. 固定序列夹具 ---------------------------------------------------

def test_y_fixture_sequence_and_numbers(monkeypatch, tmp_path):
    result = run(FIXTURE, monkeypatch=monkeypatch)
    summary = result.summary[0]

    assert [event["target"] for event in result.events] == ["A", "A", "B", "C"]
    assert [event["entry_index"] for event in result.events] == [1, 2, 3, 4]
    assert [event["recent_sequence"] for event in result.events] == [
        "A", "A-A", "A-A-B", "A-B-C"]
    assert result.events[0]["time_s"] == pytest.approx(0.0, abs=TOL)
    assert result.events[0]["confirmed_s"] == pytest.approx(0.2, abs=TOL)
    assert [event["source"] for event in result.events] == ["automatic"] * 4

    assert summary["total_arm_entries"] == 4
    assert (summary["arm_A_entries"], summary["arm_B_entries"],
            summary["arm_C_entries"]) == (2, 1, 1)
    assert summary["alternation_count"] == 1
    assert summary["alternation_opportunities"] == 2
    assert summary["alternation_percent"] == pytest.approx(50.0, abs=TOL)

    assert summary["total_time_s"] == pytest.approx(2.3, abs=TOL)
    assert summary["center_time_s"] == pytest.approx(0.9, abs=TOL)
    assert summary["arm_A_time_s"] == pytest.approx(0.6, abs=TOL)
    assert summary["arm_B_time_s"] == pytest.approx(0.3, abs=TOL)
    assert summary["arm_C_time_s"] == pytest.approx(0.4, abs=TOL)
    assert summary["missing_time_s"] == pytest.approx(0.1, abs=TOL)
    assert summary["cm_per_px_est"] > 0

    # 候选表保留全部 22 条真实候选；入臂表与事件同源。
    assert len(result.tables["candidates"]) == 22
    assert result.tables["arm_entries"] == result.events

    # 恢复首帧没有已完成跨缺口速度。
    assert result.observations[-1]["frame"] == 22
    assert result.observations[-1]["step_distance_cm"] is None
    assert result.observations[-1]["speed_cm_s"] is None

    assert len(result.bins) == 1
    bin_row = result.bins[0]
    assert bin_row["bin_index"] == 1
    assert bin_row["total_arm_entries"] == 4
    assert bin_row["alternation_count"] == 1
    assert bin_row["alternation_opportunities"] == 2
    assert bin_row["alternation_percent"] == pytest.approx(50.0, abs=TOL)

    metadata = result.metadata["units"][UNIT_ID]
    assert metadata["mode"] == "automatic"
    assert metadata["coordinate_unit"] == "cm_estimated"
    assert metadata["cm_per_px_est"] == pytest.approx(summary["cm_per_px_est"],
                                                      abs=TOL)

    # 序列图数据来自同一 events：四次进入、重复 A 不合并、时刻与确认时刻一致。
    points = figures.ymaze_sequence_points(result.events)
    assert [(item["entry_index"], item["arm"], item["time_s"],
             item["confirmed_s"]) for item in points] == [
        (1, "A", 0.0, 0.2), (2, "A", 0.6, 0.8),
        (3, "B", 1.2, 1.4), (4, "C", 1.8, 2.0)]
    png = tmp_path / "sequence.png"
    assert figures.ymaze_sequence_png(png, result.events)
    assert png.read_bytes()[:8] == bytes.fromhex("89504e470d0a1a0a")

    # 注册顺序与模板键（界面概览取前六项）。
    entry = [item for item in experiments.catalog()
             if item["key"] == "ymaze"][0]
    assert entry["label"] == "Y 迷宫"
    assert [item["label"] for item in entry["geometry"]] == [
        "中心区三点", "A 臂四角", "B 臂四角", "C 臂四角", "检测遮罩"]
    assert [item["key"] for item in entry["metrics"]] == [
        "total_arm_entries", "alternation_count",
        "alternation_opportunities", "alternation_percent",
        "center_time_s", "arm_A_time_s", "arm_B_time_s", "arm_C_time_s",
        "total_distance_cm", "mean_speed_cm_s"]
    assert entry["template_keys"] == ["zones", "arm_length_cm", "arm_width_cm"]


# -- 2. 短序列、重复臂、中心要求与候选关联 -----------------------------

def test_y_sequence_rules_and_candidate_link(monkeypatch):
    # 两次进入：机会 0、比例保持空，不写 0%。
    two = run([ARM_A] * 3 + [CENTER] * 3 + [ARM_B] * 3, monkeypatch=monkeypatch)
    summary = two.summary[0]
    assert summary["total_arm_entries"] == 2
    assert summary["alternation_opportunities"] == 0
    assert summary["alternation_percent"] is None
    assert two.bins[0]["alternation_percent"] is None

    # 重复 A 不合并：三次 A 就是三次事件，机会 1、交替 0。
    repeat = run([ARM_A] * 3 + [CENTER] * 3 + [ARM_A] * 3 + [CENTER] * 3
                 + [ARM_A] * 3, monkeypatch=monkeypatch)
    repeated = repeat.summary[0]
    assert repeated["total_arm_entries"] == 3
    assert repeated["arm_A_entries"] == 3
    assert repeated["alternation_opportunities"] == 1
    assert repeated["alternation_count"] == 0
    assert repeated["alternation_percent"] == pytest.approx(0.0, abs=TOL)

    # 不要求经中心时，另一臂可直接形成下一次进入。
    free = run([ARM_A] * 3 + [ARM_B] * 3 + [ARM_B] * 3,
               values=options(require_center=False), monkeypatch=monkeypatch)
    assert [event["target"] for event in free.events] == ["A", "B"]

    # 参数与几何校验：深度比例范围、布尔、三角形点数、四角交叉。
    bad = ymaze.validate_unit(dict(ymaze.DEFAULTS, entry_depth_ratio=1.5,
                                   require_center="yes"),
                              geometry(), complete=True)
    fields = [issue["field"] for issue in bad]
    assert fields == ["options.entry_depth_ratio", "options.require_center"]
    triangle = {"zones": dict(geometry()["zones"],
                              center={"type": "polygon",
                                      "points": [[90, 90], [110, 90]]})}
    assert [issue["field"] for issue in ymaze.validate_unit(
        dict(ymaze.DEFAULTS), triangle, complete=True)] == [
        "geometry.zones.center"]
    crossed = {"zones": dict(geometry()["zones"], A={
        "type": "quad", "points": [[90, 10], [110, 85], [90, 85], [110, 10]]})}
    assert [issue["field"] for issue in ymaze.validate_unit(
        dict(ymaze.DEFAULTS), crossed, complete=True)] == ["geometry.zones.A"]

    # 候选关联：预测门限、短缺口接续、长缺口重新获取与面积双参考。
    link = ymaze.CandidateLink({"max_jump_cm": 12.0, "tracking_hold_s": 0.4},
                               0.5)      # 0.5 cm/px → 12 cm = 24 px
    far = {"center": (200.0, 50.0), "area": 200.0, "quality": 1.0}
    near = {"center": (105.0, 50.0), "area": 200.0, "quality": 1.0}
    first = link.pick([{"center": (100.0, 50.0), "area": 200.0,
                        "quality": 1.0}, far], 0.0)
    assert first["center"] == (100.0, 50.0)
    link.accept(first, 0.0)
    assert link.pick([far], 0.1) is None, "超出预测偏差的候选不接受"
    assert link.pick([near], 0.1) is not None
    link.miss()
    assert link.pick([{"center": (108.0, 50.0), "area": 200.0,
                       "quality": 1.0}], 0.3) is not None, "短缺口保留证据"
    link.miss()
    resumed = link.pick([{"center": (400.0, 400.0), "area": 200.0,
                          "quality": 1.0}], 2.0)
    assert resumed is not None and resumed["center"] == (400.0, 400.0)
    stable = ymaze.CandidateLink({"max_jump_cm": 12.0,
                                  "tracking_hold_s": 0.4}, 0.5)
    for step in range(9):
        stable.accept({"center": (100.0, 50.0), "area": 200.0,
                       "quality": 1.0}, 0.1 * step)
    assert stable.pick([{"center": (100.0, 50.0), "area": 800.0,
                         "quality": 1.0}], 1.0) is not None
    assert stable.pick([{"center": (100.0, 50.0), "area": 1000.0,
                         "quality": 1.0}], 1.0) is None, "墙级面积被拒绝"
    assert stable.pick([{"center": (100.0, 50.0), "area": 0.0,
                         "quality": 1.0}], 1.0) is not None


# -- 3. 行为预览与结果页 -----------------------------------------------

class PreviewSource:
    """内存解码头：小鼠停在 B 臂内。"""

    def __init__(self, path, *, start_s=0.0, end_s=None, **kwargs):
        from behaviourlab.schema import Frame

        self._frames = []
        begin = float(start_s or 0.0)
        stop = float(end_s if end_s is not None else begin + 4.0)
        step = 0.1
        count = max(1, int(round((stop - begin) / step)))
        for index in range(count):
            moment = round(begin + index * step, 6)
            image = np.full((240, 240, 3), 255, dtype=np.uint8)
            cv2.ellipse(image, (150, 105), (20, 9), 0, 0, 360, (0, 0, 0), -1)
            self._frames.append(Frame(
                image=image, index=index, time_s=moment, dt_s=step,
                elapsed_s=moment - begin, analysis_fps=1.0 / step,
                interval_start_s=moment, interval_end_s=moment + step))

    def iter_frames(self):
        return iter(self._frames)


def test_y_preview_and_result_page(client, store, tmp_path, monkeypatch):
    source = tmp_path / "y.mp4"
    source.write_bytes(b"placeholder")
    asset = store.create_asset(
        name="y.mp4", path=str(source), project_id=None,
        metadata={"width": 240, "height": 240, "duration_s": 4.0,
                  "fps": 10.0, "frame_count": 40})
    draft = store.create_draft("ymaze", videos=[{
        "asset_id": asset["id"],
        "window": {"start_s": 0.0, "end_s": 4.0},
        "orientation": 0, "fps_override": None, "sample_fps": 10,
        "units": [{"id": "e" * 32, "name": "Y 鼠", "window": None,
                   "geometry": geometry(), "options": {}, "detection": {},
                   "manual": None}],
    }])
    monkeypatch.setattr(media, "VideoSource", PreviewSource)
    monkeypatch.setattr(
        "behaviourlab.engine.media.probe_video",
        lambda path: {"width": 240, "height": 240, "duration_s": 4.0,
                      "fps": 10.0, "frame_count": 40})
    monkeypatch.setattr(
        preview.media, "read_frame_at_time",
        lambda *args, **kwargs: FrameInfo(
            image=np.full((240, 240, 3), 255, dtype=np.uint8), index=15,
            time_s=1.5, pts=None, time_base=None, clock_source="pts"))

    behavior = client.post(
        f"/api/drafts/{draft['id']}/preview",
        json={"asset_id": asset["id"], "unit_id": "e" * 32,
              "time_s": 1.5, "kind": "behavior"})
    assert behavior.status_code == 201, behavior.get_json()
    data = behavior.get_json()["data"]
    assert data["summary"]["missing_steps"] == []
    assert data["summary"]["drawn_steps"] == ["center", "A", "B", "C"]
    core = data["summary"]["behavior"]
    assert core["mode"] == "ymaze"
    assert core["total_arm_entries"] == 1
    assert core["arm_B_entries"] == 1
    assert core["alternation_percent"] is None

    started = client.post(f"/api/drafts/{draft['id']}/analyze")
    assert started.status_code == 201, started.get_json()
    payload = started.get_json()["data"]
    job = payload["jobs"][0]
    assert job["status"] == "done"
    summary = job["manifest"]["units"][0]["summary"]
    assert summary["total_arm_entries"] == 1
    assert summary["arm_B_entries"] == 1
    assert summary["arm_A_entries"] == 0
    assert summary["alternation_opportunities"] == 0
    assert summary["alternation_percent"] is None

    unit_figures = job["manifest"]["units"][0]["figures"]
    assert unit_figures["sequence"], "一次 B 入臂也要有序列图"
    assert not [key for key in ("trajectory", "heatmap", "activity", "zones")
                if unit_figures.get(key)]
    run_dir = Path(store.paths.runs) / payload["run_id"] / job["job_id"]
    sequence = unit_figures["sequence"]
    assert sequence in job["manifest"]["files"]
    stored = (run_dir / sequence).read_bytes()
    assert stored[:8] == bytes.fromhex("89504e470d0a1a0a")

    inline = client.get(
        f"/runs/{payload['run_id']}/{job['job_id']}/{sequence}")
    assert inline.status_code == 200
    assert inline.mimetype == "image/png"
    assert inline.data == stored
    download = client.get(
        f"/runs/{payload['run_id']}/{job['job_id']}/{sequence}?download=1")
    assert download.status_code == 200
    assert "attachment" in download.headers.get("Content-Disposition", "")
    assert download.data == stored

    report = (run_dir / "report.md").read_text("utf-8")
    assert "入臂序列图" in report
    assert f"]({sequence})" in report
    assert "不生成图形" not in report

    page = client.get(f"/results/{payload['run_id']}").get_data(as_text=True)
    for label in ("总入臂次数", "交替次数", "交替机会数", "自发交替率",
                  "中心区时间", "A 臂时间", "区域时间", "自动追踪",
                  "最近序列", "候选（", "入臂序列（"):
        assert label in page, label
    assert "B 4.000" in page, "B 臂时间来自同一汇总"
    assert f'src="/runs/{payload["run_id"]}/{job["job_id"]}/{sequence}"' in page
    # 机会 0 时交替率是 —，不是 0%；有确认事件时不写空态。
    assert "自发交替率 — %" in page
    assert "没有确认的入臂事件" not in page
    assert "未生成图形" not in page
    assert "没有可用的场地尺寸" not in page

    # 零进入：不写序列图，只给真实空态原因。
    empty = outputs._unit_figures(tmp_path, {"unit_name": "Y 鼠",
                                             "mode": "ymaze"}, [], "unit-empty",
                                  {}, events=[])
    assert empty["sequence"] is None
    assert "序列图" in empty["note"]
    assert not (tmp_path / "unit-empty_sequence.png").exists()


def tiny_source(count: int):
    """16×16 内存帧；10 Hz、每条曝光 .1 秒。"""

    from behaviourlab.schema import Frame

    class TinySource:
        def __init__(self, path=None, *, start_s=0.0, end_s=None, **kwargs):
            self._frames = [
                Frame(image=np.zeros((16, 16, 3), dtype=np.uint8), index=index,
                      time_s=round(0.1 * index, 6), dt_s=0.1,
                      elapsed_s=round(0.1 * index, 6), analysis_fps=10.0,
                      interval_start_s=round(0.1 * index, 6),
                      interval_end_s=round(0.1 * (index + 1), 6))
                for index in range(count)]

        def iter_frames(self):
            return iter(self._frames)

    return TinySource


def capture_facts(monkeypatch, source, values: dict | None = None) -> dict:
    """真实 capture 路径：解码用内存帧，候选由替身直接给定。"""

    unit_value = unit(values)
    monkeypatch.setattr(media, "VideoSource", source)
    job = {"assay": "ymaze", "run_id": "c" * 32, "job_id": "d" * 32,
           "asset": {"id": "asset", "name": "y.mp4", "path": "memory",
                     "metadata": {"duration_s": 1.0}},
           "window": {"start_s": 0.0, "end_s": 1.0}, "orientation": 0,
           "fps_override": None, "sample_fps": None,
           "units": [unit_value], "style": {}}
    return ymaze.capture(AnalysisContext(job=job), [unit_value])[UNIT_ID]


def test_y_association_failure_reason(monkeypatch):
    """自动 Y 的关联失败统一报 no_candidate_within_gate，底层原因另存。"""

    settings = dict(detection.common_detection_defaults(),
                    erosion_iter=0, dilation_iter=0)
    empty_foreground = np.zeros((16, 16), dtype=np.uint8)
    roi = np.full((16, 16), 255, dtype=np.uint8)
    candidates, reason = detection.find_candidates(empty_foreground, roi,
                                                   settings)
    assert candidates == [] and reason == "no_candidate", "固定反例的前提"

    monkeypatch.setattr(detection, "find_candidates",
                        lambda *args, **kwargs: (list(candidates), reason))
    facts = capture_facts(monkeypatch, tiny_source(1))
    row = facts["rows"][0]
    assert row["detected"] is False
    assert row["missing_reason"] == "no_candidate_within_gate"
    assert row["detection_reason"] == "no_candidate"
    assert row["x_px"] is None and row["nose_x_px"] is None
    assert facts["candidates"] == []

    # 有候选但被预测门限拒绝：同样是关联失败原因，候选与 selected 证据仍在。
    pool = [([{"center": (8.0, 8.0), "area": 200.0, "quality": 1.0,
               "contour": [], "mask": None, "bbox": [6.0, 6.0, 4.0, 4.0]}], None),
            ([{"center": (15.5, 15.5), "area": 200.0, "quality": 1.0,
               "contour": [], "mask": None, "bbox": [14.0, 14.0, 3.0, 3.0]}],
             None)]
    calls = {"count": 0}

    def fake_find(*args, **kwargs):
        index = min(calls["count"], len(pool) - 1)
        calls["count"] += 1
        return pool[index]

    monkeypatch.setattr(detection, "find_candidates", fake_find)
    facts = capture_facts(monkeypatch, tiny_source(2),
                          options(max_jump_cm=0.001))
    rows = facts["rows"]
    assert rows[0]["detected"] is True
    assert rows[1]["detected"] is False
    assert rows[1]["missing_reason"] == "no_candidate_within_gate"
    assert rows[1]["x_px"] is None and rows[1]["nose_x_px"] is None
    assert [item["selected"] for item in facts["candidates"]] == [True, False]
