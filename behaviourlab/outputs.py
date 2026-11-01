"""从保存的事实生成导出文件。

导出层把计算产生的非有限浮点归一为 null；CSV 用空单元格表示缺失。
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import annotated, figures
from .schema import sanitize_json
from .store import atomic_write_json

SAFE_STEM = re.compile(r"[^A-Za-z0-9_-]+")
# 只有七类明细是文件契约；其余表仍留在 results.json 里。
EXPORTED_TABLES = ("zones",)


def figure_stem(unit: Mapping[str, Any], used: set[str]) -> str:
    """安全、唯一、可读的每动物文件 stem；重名时追加序号。"""

    raw = SAFE_STEM.sub("-", str(unit.get("unit_name") or "")).strip("-_")
    raw = raw[:32] or "unit"
    identity = str(unit.get("unit_id") or "")
    suffix = identity[:6] if re.fullmatch(r"[A-Za-z0-9_-]+", identity) else ""
    base = f"{raw}-{suffix}" if suffix else raw
    stem = base
    index = 2
    while stem in used:
        stem = f"{base}-{index}"
        index += 1
    used.add(stem)
    return stem


def write_json_export(path: str | Path, payload: Any) -> Path:
    return atomic_write_json(path, sanitize_json(payload))


def _columns_from(rows: Sequence[Mapping[str, Any]], columns: Iterable[str] | None) -> list[str]:
    if columns is not None:
        return list(columns)
    ordered: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                ordered.append(key)
    return ordered


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]],
              columns: Iterable[str] | None = None) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _columns_from(rows, columns)
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            sanitized = sanitize_json(dict(row))
            writer.writerow(
                {key: "" if sanitized.get(key) is None else sanitized.get(key)
                 for key in fieldnames}
            )
    return target


def write_observations(path: str | Path, observations: Sequence[Mapping[str, Any]],
                       columns: Iterable[str] | None = None) -> Path:
    return write_csv(path, observations, columns)


def _report_metrics(assay: Any) -> list[dict]:
    from . import experiments

    if not experiments.exists(assay):
        return []
    return list(experiments.get(assay).SPEC.get("metrics") or [])[:5]


# 报告图形按清单里真实存在的每动物图形生成，名称与结果页一致。
FIGURE_LABELS = (
    ("zones", "区域时长图"),
    ("sequence", "入臂序列图"),
    ("trajectory", "轨迹图"),
    ("heatmap", "热图"),
    ("objects", "物体探索图"),
    ("activity", "活动曲线"),
)

# 归属说明只在实验适用范围成立时写：EPM 没有壁区，不能套用 OFT 文案。
ASSAY_NOTES = {
    "oft": "中心、外围、场外三类互斥；壁区包含在外围内。",
    "epm": "开放臂与封闭臂时间互斥；开放臂时间占总时长以窗口总时长为分母"
           "（含缺失曝光），开放臂时间占臂区时长以开放臂时间+封闭臂时间为分母。",
    "ldb": "明室、暗室、门区与未知互斥，比例分母为窗口总时长；不可见暗室下"
           "失检曝光按推断暗室计入暗室时间，同时计入缺失，二者不相加。",
    "barnes": "自动模式的洞访问来自鼻尖进入洞半径的证据，人工模式以人工记录为"
              "主序列；覆盖率分母为完整动物窗曝光；逃避候选只是目标附近身体消失"
              "的未确认记录，不产生总逃避潜伏期。",
    "ezm": "开放、封闭、边界、区外与未知五类互斥；开放时间占总时长以窗内全部"
           "曝光为分母（含缺失），开放时间占开放+封闭只算两段跑道；进入需真实"
           "更晚观测确认且只在环上相邻区之间成立，七类明细见 zones.csv。",
}


def _assay_of(job: Mapping[str, Any], result: Any) -> str:
    metadata = getattr(result, "metadata", None) or {}
    return str(job.get("assay") or metadata.get("assay") or "").strip()


def _manifest_figures(manifest: Mapping[str, Any]) -> tuple[list[tuple], list[tuple]]:
    """按清单列出 (动物, [(图名, 文件名)]) 与没有图形的 (动物, 原因)。"""

    listed: list[tuple] = []
    empty: list[tuple] = []
    for unit in manifest.get("units", []):
        info = unit.get("figures") or {}
        label = unit.get("unit_name") or unit.get("unit_id")
        names = [(caption, info[key]) for key, caption in FIGURE_LABELS
                 if isinstance(info.get(key), str) and info.get(key)]
        video = unit.get("annotated_video")
        if isinstance(video, str) and video:
            names.append(("标注视频", video))
        if names:
            listed.append((label, names))
        else:
            empty.append((label, str(info.get("note") or "未生成图形")))
    return listed, empty


def _report_markdown(job: Mapping[str, Any], result: Any,
                     manifest: Mapping[str, Any]) -> str:
    lines = [f"# {job.get('assay', '')} 分析报告", ""]
    lines.append(f"- run: {job.get('run_id')}")
    lines.append(f"- job: {job.get('job_id')}")
    lines.append(f"- unit 数: {len(result.summary)}")
    metrics = _report_metrics(job.get("assay"))
    if not metrics:
        metrics = [{"key": "center_percent", "label": "中心占比", "unit": "%"},
                   {"key": "total_distance_cm", "label": "场地内路程", "unit": "cm"},
                   {"key": "mean_speed_cm_s", "label": "平均速度", "unit": "cm/s"}]
    # 只保留当前结果真正有的指标列（人工 Y 没有占用与速度）。
    metrics = [item for item in metrics
               if any(item["key"] in row for row in result.summary)]
    headers = ["动物"] + [
        item["label"] + (f"（{item['unit']}）" if item.get("unit") else "")
        for item in metrics]
    lines.extend(["", "| " + " | ".join(headers) + " |",
                  "| " + " | ".join(["---"] * len(headers)) + " |"])
    for row in result.summary:
        cells = [str(row.get("unit_name") or row.get("unit_id"))]
        cells.extend(_fmt(row.get(item["key"])) for item in metrics)
        lines.append("| " + " | ".join(cells) + " |")
    note = ASSAY_NOTES.get(_assay_of(job, result))
    if note:
        lines.extend(["", note])
    if result.warnings:
        lines.extend(["", "## 警告", ""])
        lines.extend(f"- {text}" for text in result.warnings)

    listed, empty = _manifest_figures(manifest)
    lines.extend(["", "## 图形", ""])
    if not listed:
        lines.append("本次运行没有生成图形。")
    for label, names in listed:
        lines.append(f"- {label}：" + "、".join(
            f"{caption}（{name}）" for caption, name in names))
    for label, reason in empty:
        lines.append(f"- {label}：{reason}")
    if listed:
        # 引用同目录的实际文件，图片地址与清单条目一一对应。
        lines.extend(["", "图片（与本报告同一目录）：", ""])
        for label, names in listed:
            for caption, name in names:
                if str(name).lower().endswith(".png"):
                    lines.append(f"![{label} · {caption}]({name})")

    lines.extend(["", "## 文件", ""])
    lines.extend(f"- {name}" for name in manifest.get("files", []))
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _unit_figures(target: Path, summary: Mapping[str, Any],
                  rows: Sequence[Mapping[str, Any]], stem: str,
                  style: Mapping[str, Any] | None, *,
                  events: Sequence[Mapping[str, Any]] | None = None) -> dict:
    """每动物可用图形：活动曲线不依赖场地宽高，轨迹/热图才需要。"""

    missing_size_note = False
    figures_info: dict[str, Any] = {"stem": stem, "trajectory": None,
                                    "heatmap": None, "activity": None,
                                    "objects": None, "zones": None,
                                    "sequence": None}
    name = str(summary.get("unit_name") or summary.get("unit_id") or "")
    notes: list[str] = []

    # 只有真的带最终状态的实验才画活动曲线，不按汇总字段猜。
    state_based = any(row.get("state") for row in rows)
    if state_based:
        activity = f"{stem}_activity.png"
        try:
            if figures.activity_png(target / activity, rows, title=name,
                                    style=style,
                                    mode=str(summary.get("mode") or "classic")):
                figures_info["activity"] = activity
            else:
                notes.append("没有有效状态曝光，未生成活动曲线")
        except (OSError, ValueError) as error:
            notes.append(f"活动曲线生成失败：{error}")

    width = summary.get("arena_width_cm")
    height = summary.get("arena_height_cm")
    has_size = (isinstance(width, (int, float))
                and isinstance(height, (int, float))
                and width > 0 and height > 0)
    mode = str(summary.get("mode") or "classic")
    pool_radius = summary.get("pool_radius_cm")
    is_mwm = (mode == "mwm" and isinstance(pool_radius, (int, float))
              and pool_radius > 0)
    if is_mwm:
        radius = float(pool_radius)
        platform = None
        px = summary.get("platform_x_cm")
        py = summary.get("platform_y_cm")
        if isinstance(px, (int, float)) and isinstance(py, (int, float)):
            platform = (float(px), float(py))
        start = None
        sx = summary.get("start_x_cm")
        sy = summary.get("start_y_cm")
        if isinstance(sx, (int, float)) and isinstance(sy, (int, float)):
            start = (float(sx), float(sy))
        platform_radius = summary.get("platform_radius_cm")
        platform_radius = (float(platform_radius)
                           if isinstance(platform_radius, (int, float)) else 0.0)
        trajectory = f"{stem}_trajectory.png"
        heatmap = f"{stem}_heatmap.png"
        try:
            if figures.mwm_trajectory_png(
                    target / trajectory, rows, radius, platform=platform,
                    platform_radius=platform_radius, start=start,
                    title=name, style=style):
                figures_info["trajectory"] = trajectory
            else:
                notes.append("没有完成的连续运动段，未生成轨迹图")
            if figures.mwm_heatmap_png(
                    target / heatmap, rows, radius, platform=platform,
                    platform_radius=platform_radius, title=name, style=style):
                figures_info["heatmap"] = heatmap
            else:
                notes.append("没有有效位置，未生成热图")
        except (OSError, ValueError) as error:
            notes.append(f"图形生成失败：{error}")
            for key, filename in (("trajectory", trajectory),
                                  ("heatmap", heatmap)):
                if not (target / filename).is_file():
                    figures_info[key] = None
    elif mode == "ezm":
        diameter = summary.get("outer_diameter_cm")
        ring = summary.get("ring_width_cm")
        angle = summary.get("boundary_angle_deg")
        usable = (isinstance(diameter, (int, float)) and diameter > 0
                  and isinstance(ring, (int, float)) and ring >= 0
                  and isinstance(angle, (int, float)))
        if usable:
            trajectory = f"{stem}_trajectory.png"
            heatmap = f"{stem}_heatmap.png"
            try:
                if figures.ezm_trajectory_png(
                        target / trajectory, rows,
                        diameter_cm=float(diameter),
                        ring_width_cm=float(ring),
                        boundary_angle_deg=float(angle), title=name,
                        style=style):
                    figures_info["trajectory"] = trajectory
                else:
                    notes.append("没有完成的连续运动段，未生成轨迹图")
                if figures.ezm_heatmap_png(
                        target / heatmap, rows, diameter_cm=float(diameter),
                        ring_width_cm=float(ring),
                        boundary_angle_deg=float(angle), title=name,
                        style=style):
                    figures_info["heatmap"] = heatmap
                else:
                    notes.append("没有有效位置，未生成热图")
            except (OSError, ValueError) as error:
                notes.append(f"图形生成失败：{error}")
                for key, filename in (("trajectory", trajectory),
                                      ("heatmap", heatmap)):
                    if not (target / filename).is_file():
                        figures_info[key] = None
        else:
            missing_size_note = True
    elif has_size:
        center = figures.center_rect_for(summary)
        trajectory = f"{stem}_trajectory.png"
        heatmap = f"{stem}_heatmap.png"
        try:
            if figures.trajectory_png(target / trajectory, rows, float(width),
                                      float(height), center_rect=center,
                                      title=name, style=style):
                figures_info["trajectory"] = trajectory
            else:
                notes.append("没有完成的连续运动段，未生成轨迹图")
            if figures.heatmap_png(target / heatmap, rows, float(width),
                                   float(height), center_rect=center,
                                   title=name, style=style):
                figures_info["heatmap"] = heatmap
            else:
                notes.append("没有有效位置，未生成热图")
        except (OSError, ValueError) as error:
            # 图形失败要如实登记，不把没有落盘的文件写进清单。
            notes.append(f"图形生成失败：{error}")
            for key, filename in (("trajectory", trajectory),
                                  ("heatmap", heatmap)):
                if not (target / filename).is_file():
                    figures_info[key] = None
    elif figures_info["activity"] is None:
        # 尺寸缺失只是“没有轨迹/热图”的原因；若专属图形已生成就不刷噪音。
        missing_size_note = True

    if str(summary.get("mode")) == "epm":
        zones_figure = f"{stem}_zones.png"
        try:
            if figures.epm_zones_png(target / zones_figure, summary,
                                     title=name, style=style):
                figures_info["zones"] = zones_figure
        except (OSError, ValueError) as error:
            notes.append(f"区域时长图生成失败：{error}")

    if str(summary.get("mode")) == "ymaze":
        # 序列图直接读最终 arm_entries/events，不重跑检测也不重算交替。
        sequence_figure = f"{stem}_sequence.png"
        try:
            if figures.ymaze_sequence_png(target / sequence_figure, events,
                                          title=name, style=style):
                figures_info["sequence"] = sequence_figure
            else:
                notes.append("没有确认的入臂事件，未生成序列图")
        except (OSError, ValueError) as error:
            notes.append(f"入臂序列图生成失败：{error}")

    if missing_size_note and not any(
            figures_info.get(key) for key in ("zones", "objects", "sequence")):
        # Y 迷宫 / LDB / Barnes / EZM 不用场地宽高描述图形空态，是否出图由各自逻辑说明。
        if str(summary.get("mode")) not in ("ymaze", "ldb", "barnes", "ezm"):
            notes.append("没有可用的场地尺寸，未生成图形")

    if isinstance(summary.get("objects"), list) and summary["objects"]:
        objects_figure = f"{stem}_objects.png"
        try:
            if figures.nor_objects_png(target / objects_figure, summary,
                                       title=name, style=style):
                figures_info["objects"] = objects_figure
        except (OSError, ValueError) as error:
            notes.append(f"物体探索图生成失败：{error}")

    if notes:
        figures_info["note"] = "；".join(notes)
    return figures_info


def _annotated_video(target: Path, job: Mapping[str, Any], result: Any,
                     summary: Mapping[str, Any], stem: str,
                     events: Sequence[Mapping[str, Any]]
                     ) -> tuple[str, float] | None:
    """人工 Y 的源图标注视频：开关关闭或无人工单位时不生成。"""

    output = job.get("output")
    if not isinstance(output, Mapping) or not output.get("annotated_video"):
        return None
    units = (getattr(result, "metadata", None) or {}).get("units") or {}
    meta = units.get(summary.get("unit_id")) or {}
    if meta.get("mode") != "manual":
        return None

    name = f"{stem}_annotated.mp4"
    try:
        _, seconds = annotated.write_manual_y_video(
            target / name, job=job, summary=summary, events=events)
    except OSError:
        # 编码失败让异常冒到导出层：作业标 error，不留半成品清单。
        raise
    except Exception as error:  # pragma: no cover - 防御未知编码错误
        raise OSError(f"标注视频编码失败：{error}") from error
    return name, float(seconds)


def export_result(directory: str | Path, job: Mapping[str, Any],
                  result: Any) -> dict:
    """把一份 AnalysisResult 写成清单里的文件，返回 manifest。

    每动物另出轨迹/热图 PNG 并登记映射；没有有效位置时省图并说明。
    CSV 的 null 写成空单元格。
    """

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    files: list[str] = []

    write_json_export(target / "frozen_job.json", job)
    files.append("frozen_job.json")
    write_observations(target / "observations.csv", result.observations)
    files.append("observations.csv")
    write_csv(target / "summary.csv", result.summary)
    files.append("summary.csv")
    write_csv(target / "events.csv", result.events)
    files.append("events.csv")
    write_csv(target / "bins.csv", result.bins)
    files.append("bins.csv")
    for name in EXPORTED_TABLES:
        rows = result.tables.get(name) or []
        if not rows:
            continue
        write_csv(target / f"{name}.csv", rows)
        files.append(f"{name}.csv")
    payload = result.as_dict()
    write_json_export(target / "results.json", payload)
    files.append("results.json")

    style = job.get("style") if isinstance(job.get("style"), Mapping) else None
    units: list[dict] = []
    used: set[str] = set()
    for row in result.summary:
        unit_id = row.get("unit_id")
        rows = [item for item in result.observations
                if item.get("unit_id") == unit_id]
        events = [item for item in result.events
                  if item.get("unit_id") == unit_id]
        stem = figure_stem(row, used)
        info = _unit_figures(target, row, rows, stem, style, events=events)
        for key in ("trajectory", "heatmap", "activity", "objects",
                    "zones", "sequence"):
            if info.get(key):
                files.append(info[key])
        video = _annotated_video(target, job, result, row, stem, events)
        entry = {"unit_id": unit_id, "unit_name": row.get("unit_name"),
                 "summary": row, "figures": info,
                 "annotated_video": video[0] if video else None}
        if video:
            files.append(video[0])
            entry["annotated_video_duration_s"] = round(video[1], 6)
            entry["annotated_video_time_base"] = annotated.TIMEBASE_LABEL
        units.append(entry)

    manifest = {
        "run_id": job.get("run_id"),
        "job_id": job.get("job_id"),
        "assay": job.get("assay"),
        "units": units,
        "files": files + ["result_manifest.json", "report.md"],
        "warnings": list(result.warnings),
        "metadata": result.metadata,
    }
    write_json_export(target / "result_manifest.json", manifest)
    (target / "report.md").write_text(
        _report_markdown(job, result, manifest), encoding="utf-8")
    return manifest
