"""从保存的事实生成导出文件。

导出层把计算产生的非有限浮点归一为 null；CSV 用空单元格表示缺失。
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from . import annotated, figures
from .schema import sanitize_json, ValidationError
from .store import atomic_write_json

SAFE_STEM = re.compile(r"[^A-Za-z0-9_-]+")
TABLE_NAME = re.compile(r"[A-Za-z0-9_-]+")

# 四个基础表的固定表头；其余字段按首次出现顺序追加，不丢未知但合法的事实。
BASE_COLUMNS = {
    "observations": ("unit_id", "unit_name", "frame", "time_s", "source_dt_s",
                     "interval_start_s", "interval_end_s", "dt_s", "detected",
                     "x_px", "y_px", "quality", "contour"),
    "summary": ("unit_id", "unit_name"),
    "events": ("event_id", "unit_id", "kind", "time_s", "confirmed_s",
               "target", "source"),
    "bins": ("unit_id", "unit_name", "bin_index", "start_s", "end_s",
             "duration_s"),
}
BASE_TABLES = tuple(BASE_COLUMNS)


def unit_stem(index: int) -> str:
    """每动物文件 stem：unit_01 起，不用动物名称做文件名。"""

    return f"unit_{index:02d}"


def table_filename(name: str) -> str:
    """合法附加表名才能成为文件；非法名直接拒绝，不做猜测性改名。"""

    if not isinstance(name, str) or not TABLE_NAME.fullmatch(name):
        raise ValueError(f"表名不合法：{name!r}")
    if name in BASE_TABLES:
        raise ValueError(f"表名与基础表冲突：{name}")
    return f"{name}.csv"


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


def _csv_cell(value: Any) -> Any:
    """CSV 单元格格式：None→空、bool→True/False、数组/字典→合法 JSON。

    JSON 用 ensure_ascii=false，中文/逗号单元回读 json.loads 成立；
    数值保持数值（写出后再由 csv 模块转文本），不把 0 与缺失混写。
    """

    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]],
              columns: Iterable[str] | None = None) -> Path:
    """写 CSV：先固定表头，再按首次出现顺序追加其余字段。

    空表也保留表头，不能因为“没有事件”写出 0 字节文件。
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ordered = list(columns or ())
    for key in _columns_from(rows, None):
        if key not in ordered:
            ordered.append(key)
    fieldnames = ordered
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            sanitized = sanitize_json(dict(row))
            writer.writerow(
                {key: _csv_cell(sanitized.get(key)) for key in fieldnames}
            )
    return target


def write_observations(path: str | Path, observations: Sequence[Mapping[str, Any]],
                       columns: Iterable[str] | None = None) -> Path:
    return write_csv(path, observations,
                     columns if columns is not None
                     else BASE_COLUMNS["observations"])


def _with_unit_names(rows: Sequence[Mapping[str, Any]],
                     summary: Sequence[Mapping[str, Any]]) -> list[dict]:
    """按 unit_id 补上显示名；行里已有的值优先。"""

    names = {row.get("unit_id"): row.get("unit_name") for row in summary}
    prepared = []
    for row in rows:
        item = dict(row)
        if not item.get("unit_name") and item.get("unit_id") in names:
            item["unit_name"] = names[item["unit_id"]]
        prepared.append(item)
    return prepared


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
    from . import experiments

    assay = _assay_of(job, result)
    label = assay
    if experiments.exists(assay):
        label = str(experiments.get(assay).SPEC.get("label") or assay)
    lines = [f"# {label} 分析报告", ""]
    lines.append(f"- run: {job.get('run_id')}")
    lines.append(f"- job: {job.get('job_id')}")
    asset = job.get("asset") if isinstance(job.get("asset"), Mapping) else {}
    lines.append(f"- 源文件：{asset.get('name') or asset.get('id') or '未知'}"
                 f"（id {asset.get('id') or '未知'}）")
    lines.append(f"- unit 数: {len(result.summary)}")

    lines.extend(["", "## 源时间与缺失", ""])
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    coverage = metadata.get("coverage")
    if isinstance(coverage, Mapping) and coverage:
        parts = [f"{key}={_fmt(coverage.get(key))}" for key in
                 ("start_s", "end_s", "duration_s", "frame_count")
                 if key in coverage]
        lines.append(f"- 实际覆盖：{'、'.join(parts) if parts else '（无字段）'}")
    else:
        lines.append("- 实际覆盖：没有覆盖记录，无法说明本次实际读取范围。")
    for row in result.summary:
        label_cell = str(row.get("unit_name") or row.get("unit_id"))
        window = ""
        start = row.get("analysis_window_start_s")
        end = row.get("analysis_window_end_s")
        if start is not None and end is not None:
            window = f"窗口 {_fmt(start)}–{_fmt(end)} s"
        else:
            window = "窗口未确定"
        parts = [window]
        for key, text in (("missing_time_s", "检测缺失"),
                          ("unobserved_time_s", "未观测")):
            if key in row and row.get(key) is not None:
                parts.append(f"{text} {_fmt(row.get(key))} s")
        lines.append(f"- {label_cell}：{'；'.join(parts)}")

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

    # 计量 metadata：只写标量事实（单位、来源、分母、抑制标记等），
    # 不覆盖实验自己的字段，也不把嵌套结构压平成猜测性文字。
    meta_lines = _measurement_metadata_lines(metadata)
    if meta_lines:
        lines.extend(["", "## 计量 metadata", ""])
        lines.extend(meta_lines)
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


_METADATA_SKIP_KEYS = ("units", "coverage", "video", "warnings", "assay")


def _scalar_metadata_entries(source: Mapping[str, Any]) -> list[str]:
    """metadata 里的标量事实键值；嵌套结构不展开也不猜测。"""

    entries = []
    for key in sorted(source):
        if key in _METADATA_SKIP_KEYS:
            continue
        value = source[key]
        if value is None or isinstance(value, (str, bool, int, float)):
            entries.append(f"{key}={_fmt(value) if value is not None else 'null'}")
    return entries


def _measurement_metadata_lines(metadata: Mapping[str, Any]) -> list[str]:
    """报告的计量 metadata 段：全局标量 + 每单位标量（按 unit_id）。"""

    lines = [f"- 全局：{'、'.join(_scalar_metadata_entries(metadata)) or '（无）'}"]
    units = metadata.get("units")
    if isinstance(units, Mapping):
        for unit_id in sorted(units):
            unit_meta = units[unit_id]
            if not isinstance(unit_meta, Mapping):
                continue
            name = str(unit_meta.get("unit_name") or unit_id)
            entries = _scalar_metadata_entries(unit_meta)
            lines.append(f"- {name}（{unit_id}）："
                         f"{'、'.join(entries) or '（无标量计量字段）'}")
    return lines


def _number(value: Any) -> float | None:
    """有限数值或 None；布尔不当作数字。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _quad_outline(shape: Any) -> list[tuple[float, float]] | None:
    """quad 形状的轮廓点；非法形状返回 None。"""

    from .experiments.ldb import _polygon

    return _polygon(shape)


def _scale_outline(points: Sequence[Sequence[float]],
                   scale: float) -> list[tuple[float, float]]:
    """像素轮廓按同一有效尺度换算为估计厘米；只换算不重拟合。"""

    return [(float(x) * scale, float(y) * scale) for x, y in points]


def _space_pair(target: Path, rows: Sequence[Mapping[str, Any]], stem: str,
                style: Mapping[str, Any] | None, figures_info: dict,
                notes: list[str], name: str, *, segments, points, outlines,
                extent, xlabel: str, ylabel: str) -> dict:
    """共用出图：轨迹用完成段端点，热图用 dt_s 加权合格位置。"""

    trajectory = f"{stem}_trajectory.png"
    heatmap = f"{stem}_heatmap.png"
    try:
        if figures.space_trajectory_png(
                target / trajectory, segments=segments, outlines=outlines,
                extent=extent, xlabel=xlabel, ylabel=ylabel, title=name,
                style=style):
            figures_info["trajectory"] = trajectory
        else:
            notes.append("没有完成的连续运动段，未生成轨迹图")
        if figures.space_heatmap_png(
                target / heatmap, points=points, outlines=outlines,
                extent=extent, xlabel=xlabel, ylabel=ylabel, title=name,
                style=style):
            figures_info["heatmap"] = heatmap
        else:
            notes.append("没有合格位置，未生成热图")
    except (OSError, ValueError) as error:
        notes.append(f"图形生成失败：{error}")
        for key, filename in (("trajectory", trajectory),
                              ("heatmap", heatmap)):
            if not (target / filename).is_file():
                figures_info[key] = None
    return figures_info


def _estimate_cm_space_figures(target: Path, rows, stem, style,
                               figures_info: dict, notes: list[str], name: str,
                               *, mode: str, unit_metadata: Mapping[str, Any],
                               geometry: Mapping[str, Any] | None) -> dict:
    """EPM/Y 自动：观测 x_cm/y_cm + 仪器轮廓按同一有效尺度换算为
    估计厘米；不捏造场地宽高，尺度缺失就不出图。"""

    if unit_metadata.get("mode") == "manual":
        notes.append("纯人工记录，未生成空间图")
        return figures_info
    scale = _number(unit_metadata.get("cm_per_px_est"))
    zones = geometry.get("zones") if isinstance(geometry, Mapping) else None
    if not (scale is not None and scale > 0):
        notes.append("估计厘米尺度缺失，未生成空间图")
        return figures_info
    if not isinstance(zones, Mapping) or not zones:
        notes.append("缺少仪器几何，未生成空间图")
        return figures_info
    outlines: list[list[tuple[float, float]]] = []
    for shape in zones.values():
        points = _quad_outline(shape)
        if points:
            outlines.append(_scale_outline(points, scale))
    if not outlines:
        notes.append("仪器几何不是可绘制的轮廓，未生成空间图")
        return figures_info
    qualified = figures.qualified_points(rows, x_key="x_cm", y_key="y_cm")
    segments = figures.motion_segments(rows)
    extent = figures.space_extent(outlines, qualified)
    if extent is None:
        notes.append("没有可用的空间范围，未生成空间图")
        return figures_info
    return _space_pair(target, rows, stem, style, figures_info, notes, name,
                       segments=segments, points=qualified, outlines=outlines,
                       extent=extent, xlabel="x (估计 cm)",
                       ylabel="y (估计 cm)")


def _ldb_cm_segments_to_px(
        segments: Sequence[tuple[float, float, float, float]],
        mapping: Mapping[str, Any]) -> tuple[list[tuple[float, float, float,
                                                         float]], int]:
    """厘米完成段 → 源像素：明室 (+y)/暗室 (−y) 按 y_cm 符号选室，逆
    homography 后用 ldb 前向映射往返校验；依据不足的段丢弃并计数，
    不猜比例。"""

    import numpy as np

    from .experiments import ldb as ldb_module

    converted: list[tuple[float, float, float, float]] = []
    insufficient = 0
    for x0, y0, x1, y1 in segments:
        pixels: list[tuple[float, float]] = []
        ok = True
        for cx, cy in ((x0, y0), (x1, y1)):
            zone = "light" if cy >= 0 else "dark"
            if zone == "light":
                local = (cx + mapping["along_light"] / 2.0, cy)
                matrix = mapping["light"]
            else:
                local = (cx - mapping["offset"] + mapping["along_dark"] / 2.0,
                         -cy)
                matrix = mapping["dark"]
            try:
                inverse = np.linalg.inv(
                    np.asarray(matrix, dtype="float64"))
            except Exception:
                ok = False
                break
            vector = inverse @ np.array([local[0], local[1], 1.0])
            if abs(float(vector[2])) < 1e-12:
                ok = False
                break
            px = (float(vector[0] / vector[2]), float(vector[1] / vector[2]))
            back = ldb_module._apply_mapping(px, zone, mapping)
            if back is None or math.hypot(back[0] - cx, back[1] - cy) > 1e-6:
                ok = False
                break
            pixels.append(px)
        if ok:
            converted.append((pixels[0][0], pixels[0][1],
                              pixels[1][0], pixels[1][1]))
        else:
            insufficient += 1
    return converted, insufficient


def _ldb_space_figures(target: Path, rows, stem, style, figures_info: dict,
                       notes: list[str], name: str, *,
                       unit_metadata: Mapping[str, Any],
                       geometry: Mapping[str, Any] | None,
                       options: Mapping[str, Any]) -> dict:
    """LDB：暗室可见且位置可用时的全装置源像素空间图。像素轴不标成
    厘米；厘米完成段必须经可校验的逆映射回到源像素。"""

    from .experiments import ldb as ldb_module

    zones = geometry.get("zones") if isinstance(geometry, Mapping) else None
    outlines: list[list[tuple[float, float]]] = []
    if isinstance(zones, Mapping):
        for key in ("light", "dark", "doorway"):
            points = _quad_outline(zones.get(key))
            if points:
                outlines.append(points)
    qualified = figures.qualified_points(rows, x_key="x_px", y_key="y_px")
    raw_segments = figures.motion_segments(rows)
    coordinate_unit = unit_metadata.get("coordinate_unit")
    segments: list[tuple[float, float, float, float]] = []
    if coordinate_unit == "px":
        segments = raw_segments
    elif coordinate_unit == "cm":
        light_quad = _quad_outline(zones.get("light")) \
            if isinstance(zones, Mapping) else None
        dark_quad = _quad_outline(zones.get("dark")) \
            if isinstance(zones, Mapping) else None
        mapping = None
        if light_quad and dark_quad:
            try:
                mapping = ldb_module.chamber_mapping(light_quad, dark_quad,
                                                     options)
            except ValidationError as error:
                notes.append(f"完成段转换依据不足：{error}")
        if mapping is None:
            if raw_segments:
                notes.append("完成段转换依据不足，未生成轨迹图")
        else:
            segments, insufficient = _ldb_cm_segments_to_px(
                raw_segments, mapping)
            if insufficient:
                notes.append(
                    f"{insufficient} 段完成段转换依据不足，未绘制")
    elif raw_segments:
        notes.append("完成段转换依据不足，未生成轨迹图")
    extent = figures.space_extent(outlines, qualified)
    if extent is None:
        notes.append("没有可用的空间范围，未生成空间图")
        return figures_info
    return _space_pair(target, rows, stem, style, figures_info, notes, name,
                       segments=segments, points=qualified, outlines=outlines,
                       extent=extent, xlabel="x (px)", ylabel="y (px)")


def _barnes_space_figures(target: Path, rows, stem, style, figures_info: dict,
                          notes: list[str], name: str, *,
                          unit_metadata: Mapping[str, Any],
                          geometry: Mapping[str, Any] | None,
                          options: Mapping[str, Any]) -> dict:
    """Barnes：圆盘物理坐标（cm，y 轴向下），范围固定 ±1.1R；几何
    叠画与物理坐标使用同一既有仿射变换。"""

    from .experiments import barnes as barnes_module

    diameter = _number(unit_metadata.get("disc_diameter_cm"))
    arena = geometry.get("arena") if isinstance(geometry, Mapping) else None
    if not (diameter is not None and diameter > 0) or arena is None:
        notes.append("圆盘直径或几何缺失，未生成空间图")
        return figures_info
    radius = diameter / 2.0
    try:
        ellipse, holes = barnes_module.require_geometry(geometry, options)
    except ValidationError as error:
        notes.append(f"仪器几何不可用：{error}")
        return figures_info
    matrix = barnes_module.affine_matrix(ellipse, diameter)
    outlines: list[list[tuple[float, float]]] = [
        [(radius * math.cos(2.0 * math.pi * index / 36),
          radius * math.sin(2.0 * math.pi * index / 36))
         for index in range(36)]]
    hole_radius = _number(unit_metadata.get("hole_radius_cm")) or 0.0
    for hole in holes:
        center_cm = barnes_module.to_cm(matrix, ellipse[0], hole["center"])
        if hole_radius > 0:
            outlines.append([
                (center_cm[0] + hole_radius * math.cos(
                    2.0 * math.pi * index / 24),
                 center_cm[1] + hole_radius * math.sin(
                    2.0 * math.pi * index / 24))
                for index in range(24)])
    qualified = figures.qualified_points(rows, x_key="x_cm", y_key="y_cm")
    segments = figures.motion_segments(rows)
    extent = (-1.1 * radius, 1.1 * radius, -1.1 * radius, 1.1 * radius)
    return _space_pair(target, rows, stem, style, figures_info, notes, name,
                       segments=segments, points=qualified, outlines=outlines,
                       extent=extent, xlabel="x (cm)", ylabel="y (cm)")


def _unit_figures(target: Path, summary: Mapping[str, Any],
                  rows: Sequence[Mapping[str, Any]], stem: str,
                  style: Mapping[str, Any] | None, *,
                  events: Sequence[Mapping[str, Any]] | None = None,
                  bins: Sequence[Mapping[str, Any]] | None = None,
                  suppress_positions: bool = False,
                  unit_metadata: Mapping[str, Any] | None = None,
                  geometry: Mapping[str, Any] | None = None,
                  options: Mapping[str, Any] | None = None) -> dict:
    """每动物可用图形：活动曲线不依赖场地宽高，轨迹/热图才需要。

    ``suppress_positions``（tracking_reliable=false / position_available=
    false / 暗室不可见）抑制位置与运动图：轨迹/热图不生成并如实说明，
    其余不依赖位置的图不受影响。
    """

    missing_size_note = False
    figures_info: dict[str, Any] = {"stem": stem, "trajectory": None,
                                    "heatmap": None, "activity": None,
                                    "objects": None, "exploration": None,
                                    "zones": None, "sequence": None}
    name = str(summary.get("unit_name") or summary.get("unit_id") or "")
    notes: list[str] = []
    if suppress_positions:
        notes.append("位置不可靠或不可用，未生成轨迹与热图")

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
    if suppress_positions:
        pass  # 位置/运动图被抑制：上面已写说明，不再尝试任何轨迹/热图分支
    elif is_mwm:
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
    elif mode in ("epm", "ymaze") and not suppress_positions:
        info = _estimate_cm_space_figures(
            target, rows, stem, style, figures_info, notes, name,
            mode=mode, unit_metadata=unit_metadata or {}, geometry=geometry)
        if info is not None:
            figures_info = info
    elif mode == "ldb" and not suppress_positions:
        info = _ldb_space_figures(
            target, rows, stem, style, figures_info, notes, name,
            unit_metadata=unit_metadata or {}, geometry=geometry,
            options=options or {})
        if info is not None:
            figures_info = info
    elif mode == "barnes" and not suppress_positions:
        info = _barnes_space_figures(
            target, rows, stem, style, figures_info, notes, name,
            unit_metadata=unit_metadata or {}, geometry=geometry,
            options=options or {})
        if info is not None:
            figures_info = info
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
        # 序列图直接读最终 arm_entries/events 与分箱计数，不重跑检测也不重算交替。
        sequence_figure = f"{stem}_sequence.png"
        try:
            if figures.ymaze_sequence_png(
                    target / sequence_figure, events, bins=bins,
                    window_start_s=summary.get("analysis_window_start_s"),
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

    # NOR 探索时序：逐对象按保存观测的 dt_s 累计，缺失/无速度段断开；
    # 不是重新计算头向规则，也没有有效速度或探索观测就不出图。
    if str(summary.get("mode")) == "nor":
        exploration_figure = f"{stem}_exploration.png"
        try:
            if figures.nor_exploration_png(target / exploration_figure,
                                           rows, summary, title=name,
                                           style=style):
                figures_info["exploration"] = exploration_figure
        except (OSError, ValueError) as error:
            notes.append(f"物体探索时序图生成失败：{error}")

    if notes:
        figures_info["note"] = "；".join(notes)
    return figures_info


def _unit_payloads(job: Mapping[str, Any], result: Any,
                   summary_rows: Sequence[Mapping[str, Any]],
                   observations: Sequence[Mapping[str, Any]]
                   ) -> list[dict]:
    """叠画视频需要的每动物事实：窗口、几何、来源与位置抑制。"""

    from . import engine

    frozen_units = {unit.get("id"): unit for unit in job.get("units") or []
                    if isinstance(unit, Mapping)}
    meta_units = (getattr(result, "metadata", None) or {}).get("units") or {}
    assay = str(job.get("assay") or "")
    payloads = []
    for row in summary_rows:
        unit_id = row.get("unit_id")
        frozen = frozen_units.get(unit_id) or {}
        meta = meta_units.get(unit_id) or {}
        manual = engine.is_manual_path(assay, frozen) or meta.get("mode") == "manual"
        reliable = meta.get("tracking_reliable")
        available = meta.get("position_available")
        suppress = reliable is False or available is False
        if assay == "ldb" and meta.get("dark_box_invisible"):
            # 暗室不可见：不画整装置轨迹，也不假装有位置。
            suppress = True
        payloads.append({
            "unit_id": unit_id,
            "unit_name": row.get("unit_name"),
            "window_start_s": row.get("analysis_window_start_s"),
            "window_end_s": row.get("analysis_window_end_s"),
            "geometry": frozen.get("geometry") or {},
            "manual": bool(manual),
            "suppress_positions": bool(suppress),
            "observations": [item for item in observations
                             if item.get("unit_id") == unit_id],
        })
    return payloads


def _run_video(target: Path, job: Mapping[str, Any],
               units: Sequence[Mapping[str, Any]],
               events: Sequence[Mapping[str, Any]]) -> tuple[str, float] | None:
    """按 run 生成一份叠画视频；关闭开关时既不生成也不声明。"""

    output = job.get("output")
    # 默认值已在冻结解析时给成 true；这里只认显式 true，缺省不生成。
    if not isinstance(output, Mapping) or output.get("annotated_video") is not True:
        return None
    name = "annotated.mp4"
    _, seconds = annotated.write_run_video(
        target / name, job=job, units=units, events=events)
    return name, float(seconds)


def write_outputs(directory: str | Path, job: Mapping[str, Any],
                  result: Any, *,
                  progress: Callable[[float, str], None] | None = None
                  ) -> dict:
    """把一份 AnalysisResult 写成声明清单；唯一导出实现。

    返回 ``{files, figures, warnings, status, run_id, job_id, assay, asset,
    coverage, algorithm_version}``。``figures`` 以 unit_id 为键，即使该动物
    没有图也保留空数组。CSV 的 None 写成空单元格，非有限计算值转 null。
    """

    def report(fraction: float, message: str = "") -> None:
        if progress is not None:
            progress(min(0.99, max(0.75, float(fraction))), message)

    report(0.75, "正在写出结果文件")
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    files: list[str] = []

    summary_rows = _with_unit_names(result.summary, result.summary)
    observation_rows = _with_unit_names(result.observations, result.summary)
    event_rows = _with_unit_names(result.events, result.summary)
    bin_rows = _with_unit_names(result.bins, result.summary)

    write_json_export(target / "frozen_job.json", job)
    files.append("frozen_job.json")
    write_observations(target / "observations.csv", observation_rows,
                       BASE_COLUMNS["observations"])
    files.append("observations.csv")
    write_csv(target / "summary.csv", summary_rows, BASE_COLUMNS["summary"])
    files.append("summary.csv")
    write_csv(target / "events.csv", event_rows, BASE_COLUMNS["events"])
    files.append("events.csv")
    write_csv(target / "bins.csv", bin_rows, BASE_COLUMNS["bins"])
    files.append("bins.csv")

    # 每个 tables 合法表都导出；非法表名直接报错，不悄悄丢事实。
    for name in sorted(result.tables):
        filename = table_filename(name)
        write_csv(target / filename, result.tables.get(name) or [],
                  ("unit_id", "unit_name"))
        files.append(filename)

    write_json_export(target / "results.json", result.as_dict())
    files.append("results.json")

    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    assay_of = _assay_of(job, result)
    style = job.get("style") if isinstance(job.get("style"), Mapping) else None
    # 文件 stem 跟随冻结 job.units 的顺序，不随 summary 行序漂移；
    # summary 里多出的 unit 追加在后，不跨动物补图。
    frozen_order = [unit.get("id")
                    for unit in (job.get("units") or [])
                    if isinstance(unit, Mapping) and unit.get("id")]
    by_unit = {row.get("unit_id"): row for row in summary_rows}
    ordered_rows = [by_unit[uid] for uid in frozen_order if uid in by_unit]
    ordered_rows += [row for row in summary_rows
                     if row.get("unit_id") not in set(frozen_order)]
    metadata_units = metadata.get("units") if isinstance(metadata, Mapping) \
        else None
    frozen_units = {unit.get("id"): unit
                    for unit in (job.get("units") or [])
                    if isinstance(unit, Mapping) and unit.get("id")}
    units: list[dict] = []
    figure_map: dict[str, list[str]] = {}
    for index, row in enumerate(ordered_rows, start=1):
        unit_id = row.get("unit_id")
        rows = [item for item in observation_rows
                if item.get("unit_id") == unit_id]
        events = [item for item in event_rows if item.get("unit_id") == unit_id]
        stem = unit_stem(index)
        unit_bins = [item for item in bin_rows
                     if item.get("unit_id") == unit_id]
        unit_meta = (metadata_units or {}).get(unit_id) \
            if isinstance(metadata_units, Mapping) else None
        unit_meta = unit_meta if isinstance(unit_meta, Mapping) else {}
        frozen = frozen_units.get(unit_id) or {}
        suppress = (unit_meta.get("tracking_reliable") is False
                    or unit_meta.get("position_available") is False)
        if assay_of == "ldb" and unit_meta.get("dark_box_invisible"):
            # 暗室不可见：不画整装置轨迹，也不假装有位置。
            suppress = True
        info = _unit_figures(target, row, rows, stem, style, events=events,
                             bins=unit_bins,
                             suppress_positions=bool(suppress),
                             unit_metadata=unit_meta,
                             geometry=frozen.get("geometry") or {},
                             options=frozen.get("options") or {})
        names = [info[key] for key in ("trajectory", "heatmap", "activity",
                                       "objects", "exploration", "zones",
                                       "sequence")
                 if info.get(key)]
        files.extend(names)
        figure_map[str(unit_id)] = names
        units.append({"unit_id": unit_id, "unit_name": row.get("unit_name"),
                      "summary": row, "figures": info, "figure_files": names,
                      "annotated_video": None})
        report(0.75 + 0.2 * len(units) / max(1, len(ordered_rows)),
               "正在导出结果")

    # 一份 run 视频：所有动物共用一个流，按各自窗口叠各自几何与轨迹层。
    video = _run_video(
        target, job,
        _unit_payloads(job, result, summary_rows, observation_rows),
        event_rows)
    if video:
        files.append(video[0])
        for entry in units:
            entry["annotated_video"] = video[0]
            entry["annotated_video_duration_s"] = round(video[1], 6)
            entry["annotated_video_time_base"] = annotated.TIMEBASE_LABEL

    coverage = metadata.get("coverage")
    asset = job.get("asset") if isinstance(job.get("asset"), Mapping) else {}
    manifest = {
        "status": "completed",
        "run_id": job.get("run_id"),
        "job_id": job.get("job_id"),
        "assay": job.get("assay"),
        "asset": {"id": asset.get("id"), "name": asset.get("name"),
                  "path": asset.get("path")},
        "coverage": dict(coverage) if isinstance(coverage, Mapping) else None,
        "algorithm_version": job.get("algorithm_version"),
        "units": units,
        "figures": figure_map,
        "annotated_video": video[0] if video else None,
        "files": files + ["report.md", "result_manifest.json"],
        "warnings": list(result.warnings),
        "metadata": metadata,
    }
    (target / "report.md").write_text(
        _report_markdown(job, result, manifest), encoding="utf-8")
    files.append("report.md")
    write_json_export(target / "result_manifest.json", manifest)
    report(0.99, "已写出结果清单")
    return manifest


# 旧名保留为同一实现的别名：不并存两套导出。
export_result = write_outputs
