"""第 14 块收口：预览计数、尺度显示、字段名、分母元数据与报告合同。"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from behaviourlab import outputs, timeline
from behaviourlab.experiments import epm
from behaviourlab.schema import AnalysisResult

TOL = 1e-9
UNIT_ID = "u" * 32
RUN_ID = "1" * 32


def rect(x1: float, y1: float, x2: float, y2: float) -> dict:
    return {"type": "quad",
            "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]}


def geometry() -> dict:
    return {"zones": {
        "center": rect(90, 90, 110, 110),
        "open_1": rect(111, 90, 190, 110),
        "open_2": rect(10, 90, 89, 110),
        "closed_1": rect(90, 10, 110, 89),
        "closed_2": rect(90, 111, 110, 190),
    }}


def unit() -> dict:
    return {"id": UNIT_ID, "name": "EPM 鼠", "animal_id": None, "window": None,
            "geometry": geometry(), "options": dict(epm.DEFAULTS),
            "detection": {}}


def source_rows(points, spans=None, detected=None) -> list[dict]:
    """源观测：索引 i 默认 time_s=.1i、曝光 .1 秒。"""

    result = []
    for index, point in enumerate(points):
        start, end = ((spans[index]) if spans
                      else (round(0.1 * index, 10), round(0.1 * (index + 1), 10)))
        seen = detected[index] if detected else point is not None
        result.append({
            "unit_id": UNIT_ID, "unit_name": "EPM 鼠", "frame": index,
            "time_s": float(start), "source_dt_s": float(end - start),
            "elapsed_s": float(start), "analysis_fps": 10.0,
            "source_interval_start_s": float(start),
            "source_interval_end_s": float(end),
            "interval_start_s": float(start), "interval_end_s": float(end),
            "dt_s": float(end - start), "in_window": True,
            "detected": bool(seen),
            "x_px": None if not seen else float(point[0]),
            "y_px": None if not seen else float(point[1]),
            "quality": 1.0, "contour": [],
            "missing_reason": None if seen else "no_candidate",
            "head_angle": None, "head_quality": 0.0,
            "nose_x_px": None, "nose_y_px": None,
            "tail_x_px": None, "tail_y_px": None,
            "left_ear_x_px": None, "left_ear_y_px": None,
            "right_ear_x_px": None, "right_ear_y_px": None,
            "detection_method": "threshold",
        })
    return result


# -- 1. 行为预览总入臂次数 ---------------------------------------------

def preview(monkeypatch, points, spans=None, detected=None, end_s=None):
    """用同一捕获核心的替身行驱动 preview_analysis，隔离真实解码。"""

    rows = source_rows(points, spans, detected)
    monkeypatch.setattr(epm.media, "VideoSource",
                        lambda *args, **kwargs: object())
    monkeypatch.setattr(epm, "capture", lambda context, units: {UNIT_ID: rows})
    return epm.preview_analysis(
        asset={"path": "memory", "metadata": {}},
        video={"orientation": 0, "fps_override": None, "sample_fps": None},
        unit=unit(), start_s=0.0, end_s=end_s or rows[-1]["interval_end_s"],
    )


def test_preview_returns_core_total_arm_entries(monkeypatch):
    still = preview(monkeypatch, [(100, 100), (100, 100), (100, 100)])
    assert still["mode"] == "epm"
    assert still["open_entries"] == 0
    assert still["closed_entries"] == 0
    assert still["total_arm_entries"] == 0, "零入臂必须返回 0 而不是缺失"

    closed = preview(monkeypatch, [(100, 100), (100, 40), (100, 40)])
    assert closed["open_entries"] == 0
    assert closed["closed_entries"] == 1
    assert closed["total_arm_entries"] == 1, "总入臂与核心 summary 同源"

    # 与核心 process_unit 的同一 summary 对齐，界面不再自己加一遍。
    facts = epm.process_unit(unit(), source_rows(
        [(100, 100), (100, 40), (100, 40)]),
        timeline.Window(0.0, 0.3), None)
    assert closed["total_arm_entries"] == facts["summary"]["total_arm_entries"]

    # 重复“更新行为预览”仍是 behavior，计数稳定。
    again = preview(monkeypatch, [(100, 100), (100, 40), (100, 40)])
    assert again["mode"] == "epm"
    assert again["total_arm_entries"] == closed["total_arm_entries"]


# -- 2. 尺度显示至多 3 位小数 ------------------------------------------

def summary_row(name: str, scale) -> dict:
    return {
        "unit_id": UNIT_ID + name, "unit_name": name, "mode": "epm",
        "coordinate_unit": "cm_estimated",
        "total_time_s": 0.2, "observed_time_s": 0.1, "missing_time_s": 0.1,
        "detected_frames": 1, "window_frames": 2,
        "open_time_s": 0.1, "closed_time_s": 0.0, "center_time_s": 0.0,
        "unknown_time_s": 0.0,
        "zone_seconds": {"center": 0.0, "open": 0.1, "closed": 0.0,
                         "unknown": 0.0},
        "open_time_percent_total": 50.0, "open_time_percent_arms_only": 100.0,
        "open_entries": 0, "closed_entries": 0, "total_arm_entries": 0,
        "open_entry_percent": None, "first_open_entry_latency_s": None,
        "total_distance_cm": 0.0, "measurable_motion_time_s": 0.0,
        "moving_time_s": 0.0, "immobility_time_s": 0.0,
        "mean_speed_cm_s": None, "max_speed_cm_s": None,
        "cm_per_px_est": scale,
        "analysis_window_start_s": 0.0, "analysis_window_end_s": 0.2,
    }


def test_scale_display_rounds_to_three_digits(client, store, tmp_path):
    run_dir = Path(store.paths.runs) / RUN_ID / ("2" * 32)
    run_dir.mkdir(parents=True)
    raw = 0.0778396775950181
    rows = [summary_row("小数", raw), summary_row("缺失", None),
            summary_row("零", 0.0)]
    result = AnalysisResult(
        observations=[], summary=rows, events=[], bins=[], warnings=[],
        metadata={"assay": "epm", "coordinate_unit": "cm_estimated"},
    )
    job = {"run_id": RUN_ID, "job_id": "2" * 32, "assay": "epm",
           "asset": {"id": "asset", "name": "epm.mp4", "path": "memory",
                     "metadata": {}},
           "window": {"start_s": 0.0, "end_s": 0.2}, "style": {}}
    outputs.export_result(run_dir, job, result)

    page = client.get(f"/results/{RUN_ID}").get_data(as_text=True)
    assert "0.078 cm_estimated/px" in page, "尺度最多 3 位小数"
    assert "0.0778" not in page, "不得显示 4 位小数"
    assert "— cm_estimated/px" in page, "null 仍显示 —"
    assert "0.000 cm_estimated/px" in page, "合法 0 不能变成缺失"

    # 导出仍保留原始精度。
    stored = json.loads((run_dir / "results.json").read_text("utf-8"))
    assert stored["summary"][0]["cm_per_px_est"] == raw
    summary_csv = (run_dir / "summary.csv").read_text("utf-8-sig")
    assert repr(raw) in summary_csv


# -- 3. 占用分母元数据 -------------------------------------------------

def test_occupancy_denominators_match_formulas():
    facts = epm.process_unit(
        unit(),
        source_rows([(150, 100), None], detected=[True, False]),
        timeline.Window(0.0, 0.2), None)
    summary = facts["summary"]
    assert summary["total_time_s"] == pytest.approx(0.2, abs=TOL)
    assert summary["missing_time_s"] == pytest.approx(0.1, abs=TOL)
    assert summary["open_time_s"] == pytest.approx(0.1, abs=TOL)
    assert summary["closed_time_s"] == pytest.approx(0.0, abs=TOL)
    # 开放 0.1 / 总时长 0.2（含缺失）= 50%；开放 0.1 / 臂区 0.1 = 100%。
    assert summary["open_time_percent_total"] == pytest.approx(50.0, abs=TOL)
    assert summary["open_time_percent_arms_only"] == pytest.approx(100.0,
                                                                   abs=TOL)

    denominators = facts["metadata"]["occupancy_denominator"]
    assert "total_time_s" in denominators["open_time_percent_total"]
    assert "missing" in denominators["open_time_percent_total"]
    assert denominators["open_time_percent_arms_only"] == \
        "open_time_s + closed_time_s"

    # 零分母仍为 null，文案修复不改数值。
    empty = epm.process_unit(
        unit(), source_rows([(100, 100)]), timeline.Window(0.0, 0.1), None)
    assert empty["summary"]["open_time_percent_arms_only"] is None


# -- 4. 导出报告 -------------------------------------------------------

def zebra_report(tmp_path, summary_rows, assay="epm") -> tuple[dict, str]:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    result = AnalysisResult(
        observations=[], summary=summary_rows, events=[], bins=[],
        warnings=[], metadata={"assay": assay, "coordinate_unit": "cm"},
    )
    job = {"run_id": RUN_ID, "job_id": "3" * 32, "assay": assay,
           "asset": {"id": "asset", "name": "v.mp4", "path": "memory",
                     "metadata": {}},
           "window": {"start_s": 0.0, "end_s": 0.2}, "style": {}}
    manifest = outputs.export_result(run_dir, job, result)
    return manifest, (run_dir / "report.md").read_text("utf-8")


def test_epm_report_uses_actual_zones_figure(tmp_path):
    manifest, report = zebra_report(
        tmp_path, [summary_row("小鼠 A", 0.1), summary_row("小鼠 B", 0.2)])
    zones = [unit["figures"]["zones"] for unit in manifest["units"]]
    assert all(zones), "EPM 必须生成区域时长图"
    for name in zones:
        assert name in manifest["files"]
        assert (tmp_path / "run" / name).is_file()

    assert "区域时长图" in report
    assert "未生成图形" not in report, "有图时不得声称没有图"
    assert "壁区" not in report, "EPM 不得夹带 OFT 壁区归属文案"
    refs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", report)
    assert refs, "报告必须给出可访问的图引用"
    for name in refs:
        assert name in manifest["files"]
        assert (tmp_path / "run" / name).is_file()
        assert name in report
    # 两个分母说明随 EPM 报告一起给出，不改变数值。
    assert "total_time_s" not in report  # 报告用人话说明，不抛字段名
    assert "含缺失曝光" in report


def test_epm_report_without_figures_has_no_fake_links(tmp_path):
    row = summary_row("小鼠 A", 0.1)
    row.pop("zone_seconds")
    manifest, report = zebra_report(tmp_path, [row])
    assert manifest["units"][0]["figures"]["zones"] is None
    assert not [name for name in manifest["files"] if name.endswith(".png")]
    assert "本次运行没有生成图形" in report
    assert "![" not in report
    assert "壁区" not in report


def test_oft_report_keeps_wall_note(tmp_path):
    row = summary_row("小鼠 A", 0.1)
    row.pop("zone_seconds")
    row.update({"mode": "oft", "arena_width_cm": 50.0,
                "arena_height_cm": 50.0, "center_ratio_x": 0.5,
                "center_ratio_y": 0.5, "center_percent": 40.0,
                "outer_percent": 60.0, "wall_percent": 20.0})
    manifest, report = zebra_report(tmp_path, [row], assay="oft")
    assert "壁区包含在外围内" in report, "OFT 原有归属说明必须保留"
    assert "臂区" not in report, "EPM 的臂区说明不能串到 OFT"
    assert "中心区占比" in report
    assert manifest["units"][0]["figures"]["zones"] is None
