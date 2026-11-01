"""结果图形：只用保存的完成运动段与有效位置，不重新检测也不补线。

轨迹逐段绘制，缺失处自然断开；热图按有效曝光 dt_s 加权，单位是秒/箱。
厘米坐标、等比例轴、y 向下（左上是物理原点）。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")

from matplotlib import colormaps  # noqa: E402
from matplotlib import patches  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

HEATMAP_BINS = 40

matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _motion_missing(row: Mapping[str, Any]) -> bool:
    """行里带 motion_x 键但为空：位置事实不完整，空间图一律剔除。"""

    return "motion_x" in row and row.get("motion_x") is None


def valid_positions(rows: Iterable[Mapping[str, Any]],
                    width_cm: float, height_cm: float) -> list[tuple[float, float, float]]:
    """窗内有效位置的 (x, y, dt)；jump / position_valid=false 不算有效点。"""

    points: list[tuple[float, float, float]] = []
    for row in rows:
        if not row.get("in_window") or not row.get("detected"):
            continue
        if row.get("position_valid") is False or row.get("jump") is True:
            continue
        if _motion_missing(row):
            continue
        span = _finite(row.get("dt_s"))
        x = _finite(row.get("x_cm"))
        y = _finite(row.get("y_cm"))
        if span is None or span <= 0 or x is None or y is None:
            continue
        if not (0.0 <= x <= width_cm and 0.0 <= y <= height_cm):
            # 场地外的真实位置不进入场地热图，也不夹到边缘。
            continue
        points.append((x, y, span))
    return points


def motion_segments(rows: Iterable[Mapping[str, Any]]) -> list[tuple[float, float, float, float]]:
    """保存的完成运动段端点；没有段的行不产生任何线。"""

    segments: list[tuple[float, float, float, float]] = []
    for row in rows:
        x0 = _finite(row.get("motion_x0"))
        y0 = _finite(row.get("motion_y0"))
        x1 = _finite(row.get("motion_x1"))
        y1 = _finite(row.get("motion_y1"))
        if None in (x0, y0, x1, y1):
            continue
        segments.append((x0, y0, x1, y1))
    return segments


def _style(style: Mapping[str, Any] | None) -> tuple[str, str, float, int]:
    values = style or {}
    color = str(values.get("color") or "#B86245")
    colormap = str(values.get("colormap") or "viridis")
    try:
        colormaps[colormap]
    except KeyError:
        colormap = "viridis"
    width = _finite(values.get("line_width")) or 2.0
    dpi = int(_finite(values.get("dpi")) or 150)
    return color, colormap, max(0.5, width), max(50, min(dpi, 600))


def _base_figure(width_cm: float, height_cm: float) -> tuple[Figure, Any]:
    # 空间图固定 6×5 英寸；等比例轴在画布内自适应，画布尺寸不随场地比例变。
    figure = Figure(figsize=(6.0, 5.0))
    FigureCanvasAgg(figure)
    axes = figure.add_subplot(111)
    axes.set_xlim(0.0, width_cm)
    axes.set_ylim(height_cm, 0.0)
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlabel("x (cm)")
    axes.set_ylabel("y (cm)")
    return figure, axes


def _draw_arena(axes: Any, width_cm: float, height_cm: float,
                center_rect: tuple[float, float, float, float] | None) -> None:
    axes.add_patch(patches.Rectangle(
        (0.0, 0.0), width_cm, height_cm, fill=False,
        edgecolor="#8a8a8a", linewidth=0.8))
    if center_rect:
        x0, y0, x1, y1 = center_rect
        axes.add_patch(patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0, fill=False,
            edgecolor="#b0b0b0", linewidth=0.8, linestyle="--"))


def trajectory_png(path: str | Path, rows: Iterable[Mapping[str, Any]],
                   width_cm: float, height_cm: float, *,
                   center_rect: tuple[float, float, float, float] | None = None,
                   title: str = "", style: Mapping[str, Any] | None = None) -> bool:
    """逐段画已完成运动段；没有段就不出图。"""

    segments = motion_segments(rows)
    if not segments:
        return False
    color, _, line_width, dpi = _style(style)
    figure, axes = _base_figure(width_cm, height_cm)
    _draw_arena(axes, width_cm, height_cm, center_rect)
    for x0, y0, x1, y1 in segments:
        axes.plot([x0, x1], [y0, y1], color=color, linewidth=line_width,
                  solid_capstyle="round")
    if title:
        axes.set_title(title, fontsize=9, color="#333333")
    figure.tight_layout()
    figure.savefig(path, format="png", dpi=dpi)
    return True


def heat_grid(rows: Iterable[Mapping[str, Any]], width_cm: float,
              height_cm: float, bins: int = HEATMAP_BINS
              ) -> tuple[list[list[float]], int]:
    """40×40 空间箱的秒数网格与有效点数；在场外或无效的点不进入。"""

    points = valid_positions(rows, width_cm, height_cm)
    grid = [[0.0] * bins for _ in range(bins)]
    for x, y, span in points:
        column = min(bins - 1, int(x / width_cm * bins))
        row = min(bins - 1, int(y / height_cm * bins))
        grid[row][column] += span
    return grid, len(points)


def heatmap_png(path: str | Path, rows: Iterable[Mapping[str, Any]],
                width_cm: float, height_cm: float, *,
                center_rect: tuple[float, float, float, float] | None = None,
                bins: int = HEATMAP_BINS, title: str = "",
                style: Mapping[str, Any] | None = None) -> bool:
    """40×40 空间箱按 dt_s 加权；量纲秒/箱，没有有效位置不出图。"""

    grid, count = heat_grid(rows, width_cm, height_cm, bins)
    if not count:
        return False
    _, colormap, _, dpi = _style(style)
    figure, axes = _base_figure(width_cm, height_cm)
    _draw_arena(axes, width_cm, height_cm, center_rect)
    image = axes.imshow(grid, cmap=colormap, origin="upper", aspect="equal",
                        extent=(0.0, width_cm, height_cm, 0.0),
                        interpolation="nearest")
    figure.colorbar(image, ax=axes, fraction=0.046, pad=0.03,
                    label="s")
    if title:
        axes.set_title(title, fontsize=9, color="#333333")
    figure.tight_layout()
    figure.savefig(path, format="png", dpi=dpi)
    return True


def mwm_points(rows: Iterable[Mapping[str, Any]], radius_cm: float
               ) -> list[tuple[float, float, float]]:
    """池内有效位置的 (x, y, dt)；池外或缺失不进入轨迹与热图。"""

    points: list[tuple[float, float, float]] = []
    for row in rows:
        if not row.get("in_window") or not row.get("detected"):
            continue
        if row.get("in_pool") is not True:
            continue
        if _motion_missing(row):
            continue
        span = _finite(row.get("dt_s"))
        x = _finite(row.get("x_cm"))
        y = _finite(row.get("y_cm"))
        if span is None or span <= 0 or x is None or y is None:
            continue
        if math.hypot(x, y) > float(radius_cm) + 1e-8:
            continue
        points.append((x, y, span))
    return points


def _mwm_figure(radius_cm: float) -> tuple[Figure, Any]:
    figure = Figure(figsize=(6.0, 5.0))
    FigureCanvasAgg(figure)
    axes = figure.add_subplot(111)
    axes.set_xlim(-radius_cm, radius_cm)
    axes.set_ylim(-radius_cm, radius_cm)
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlabel("x (cm)")
    axes.set_ylabel("y (cm)")
    axes.add_patch(patches.Circle((0.0, 0.0), radius_cm, fill=False,
                                  edgecolor="#8a8a8a", linewidth=0.8))
    axes.axhline(0.0, color="#c9c9c9", linewidth=0.6)
    axes.axvline(0.0, color="#c9c9c9", linewidth=0.6)
    return figure, axes


def mwm_trajectory_png(path: str | Path,
                       rows: Iterable[Mapping[str, Any]], radius_cm: float, *,
                       platform: tuple[float, float] | None = None,
                       platform_radius: float = 0.0,
                       start: tuple[float, float] | None = None,
                       title: str = "",
                       style: Mapping[str, Any] | None = None) -> bool:
    """池内轨迹：只连接完成的连续运动段，附平台与释放点。"""

    segments = motion_segments(rows)
    if not segments:
        return False
    color, _, line_width, dpi = _style(style)
    figure, axes = _mwm_figure(float(radius_cm))
    if platform is not None and float(platform_radius) > 0.0:
        # 平台圈就是评分区：按配置的实际厘米半径画，不放大物理边界。
        axes.add_patch(patches.Circle(platform, float(platform_radius),
                                      fill=False, edgecolor="#b45f45",
                                      linewidth=1.2))
    if start is not None:
        axes.plot([start[0]], [start[1]], marker="o", markersize=4,
                  color="#1c5f9e")
    for x0, y0, x1, y1 in segments:
        axes.plot([x0, x1], [y0, y1], color=color, linewidth=line_width,
                  solid_capstyle="round")
    if title:
        axes.set_title(title, fontsize=9, color="#333333")
    figure.tight_layout()
    figure.savefig(path, format="png", dpi=dpi)
    return True


def mwm_heatmap_png(path: str | Path,
                    rows: Iterable[Mapping[str, Any]], radius_cm: float, *,
                    platform: tuple[float, float] | None = None,
                    platform_radius: float = 0.0,
                    bins: int = HEATMAP_BINS, title: str = "",
                    style: Mapping[str, Any] | None = None) -> bool:
    """以池心为原点的 40×40 秒/箱热图；池外点不进入。"""

    radius = float(radius_cm)
    points = mwm_points(rows, radius)
    if not points:
        return False
    grid = [[0.0] * bins for _ in range(bins)]
    for x, y, span in points:
        column = min(bins - 1, max(0, int((x + radius) / (2.0 * radius) * bins)))
        row_index = min(bins - 1, max(
            0, int((y + radius) / (2.0 * radius) * bins)))
        grid[row_index][column] += span
    _, colormap, _, dpi = _style(style)
    figure, axes = _mwm_figure(radius)
    image = axes.imshow(grid, cmap=colormap, origin="lower",
                        extent=(-radius, radius, -radius, radius),
                        aspect="equal", interpolation="nearest")
    figure.colorbar(image, ax=axes, fraction=0.046, pad=0.03, label="s")
    axes.add_patch(patches.Circle((0.0, 0.0), radius, fill=False,
                                  edgecolor="#8a8a8a", linewidth=0.8))
    if platform is not None and float(platform_radius) > 0.0:
        # 平台圈就是评分区：按配置的实际厘米半径画，不放大物理边界。
        axes.add_patch(patches.Circle(platform, float(platform_radius),
                                      fill=False, edgecolor="#b45f45",
                                      linewidth=1.2))
    if title:
        axes.set_title(title, fontsize=9, color="#333333")
    figure.tight_layout()
    figure.savefig(path, format="png", dpi=dpi)
    return True


EZM_OPEN_COLOR = "#16684e"
EZM_CLOSED_COLOR = "#1c5f9e"
EZM_ZONE_ORDER = ("O1", "C1", "O2", "C2")


def ezm_ring_points(rows: Iterable[Mapping[str, Any]], radius_cm: float
                    ) -> list[tuple[float, float, float]]:
    """环带内有效位置的 (x, y, dt)：只有开放/封闭/边界的合格身体位置。"""

    points: list[tuple[float, float, float]] = []
    limit = float(radius_cm) + 1e-8
    for row in rows:
        if not row.get("in_window") or not row.get("detected"):
            continue
        if row.get("position_valid") is False or row.get("jump") is True:
            continue
        span = _finite(row.get("dt_s"))
        x = _finite(row.get("motion_x"))
        y = _finite(row.get("motion_y"))
        if span is None or span <= 0 or x is None or y is None:
            continue
        if math.hypot(x, y) > limit:
            continue
        points.append((x, y, span))
    return points


def _ezm_figure(diameter_cm: float, ring_width_cm: float,
                boundary_angle_deg: float) -> tuple[Figure, Any, float]:
    """以外圈为半径 D/2 的圆画场地：内外圈、四条分界线与四个区名。"""

    radius = max(0.1, float(diameter_cm) / 2.0)
    inner = max(0.0, radius - max(0.0, float(ring_width_cm)))
    figure = Figure(figsize=(6.0, 5.0))
    FigureCanvasAgg(figure)
    axes = figure.add_subplot(111)
    axes.set_xlim(-radius, radius)
    axes.set_ylim(radius, -radius)
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlabel("x (cm)")
    axes.set_ylabel("y (cm)")
    axes.add_patch(patches.Circle((0.0, 0.0), radius, fill=False,
                                  edgecolor="#8a8a8a", linewidth=0.9))
    if inner > 0:
        axes.add_patch(patches.Circle((0.0, 0.0), inner, fill=False,
                                      edgecolor="#b0b0b0", linewidth=0.8,
                                      linestyle="--"))
    theta0 = math.radians(float(boundary_angle_deg))
    for index in range(4):
        angle = theta0 + index * math.pi / 2.0
        axes.plot([inner * math.cos(angle), radius * math.cos(angle)],
                  [inner * math.sin(angle), radius * math.sin(angle)],
                  color="#8a8a8a", linewidth=0.8)
    middle = inner + (radius - inner) / 2.0
    for index, key in enumerate(EZM_ZONE_ORDER):
        angle = theta0 + (index + 0.5) * math.pi / 2.0
        color = EZM_OPEN_COLOR if index % 2 == 0 else EZM_CLOSED_COLOR
        axes.text(middle * math.cos(angle), middle * math.sin(angle), key,
                  fontsize=8, color=color, ha="center", va="center", zorder=5,
                  bbox={"boxstyle": "round,pad=0.15", "facecolor": "white",
                        "edgecolor": "none", "alpha": 0.72})
    return figure, axes, radius


def ezm_trajectory_png(path: str | Path,
                       rows: Iterable[Mapping[str, Any]], *,
                       diameter_cm: float, ring_width_cm: float,
                       boundary_angle_deg: float, title: str = "",
                       style: Mapping[str, Any] | None = None) -> bool:
    """环带内轨迹：只连接完成运动段，缺失与区外自然断开。"""

    segments = motion_segments(rows)
    if not segments:
        return False
    color, _, line_width, dpi = _style(style)
    figure, axes, _ = _ezm_figure(diameter_cm, ring_width_cm,
                                  boundary_angle_deg)
    for x0, y0, x1, y1 in segments:
        axes.plot([x0, x1], [y0, y1], color=color, linewidth=line_width,
                  solid_capstyle="round")
    if title:
        axes.set_title(title, fontsize=9, color="#333333")
    figure.tight_layout()
    figure.savefig(path, format="png", dpi=dpi)
    return True


def ezm_heatmap_png(path: str | Path,
                    rows: Iterable[Mapping[str, Any]], *,
                    diameter_cm: float, ring_width_cm: float,
                    boundary_angle_deg: float, bins: int = HEATMAP_BINS,
                    title: str = "",
                    style: Mapping[str, Any] | None = None) -> bool:
    """环带内 40×40 空间箱按秒加权；没有有效位置不出图。"""

    radius = max(0.1, float(diameter_cm) / 2.0)
    points = ezm_ring_points(rows, radius)
    if not points:
        return False
    grid = [[0.0] * bins for _ in range(bins)]
    for x, y, span in points:
        column = min(bins - 1, max(0, int((x + radius) / (2.0 * radius) * bins)))
        row_index = min(bins - 1, max(
            0, int((y + radius) / (2.0 * radius) * bins)))
        grid[row_index][column] += span
    _, colormap, _, dpi = _style(style)
    figure, axes, _ = _ezm_figure(diameter_cm, ring_width_cm,
                                  boundary_angle_deg)
    image = axes.imshow(grid, cmap=colormap, origin="upper",
                        extent=(-radius, radius, radius, -radius),
                        aspect="equal", interpolation="nearest")
    figure.colorbar(image, ax=axes, fraction=0.046, pad=0.03, label="s")
    if title:
        axes.set_title(title, fontsize=9, color="#333333")
    figure.tight_layout()
    figure.savefig(path, format="png", dpi=dpi)
    return True


def nor_objects_png(path: str | Path, summary: Mapping[str, Any], *,
                    title: str = "",
                    style: Mapping[str, Any] | None = None) -> bool:
    """每个物体的探索与接近范围时长（秒）；没有物体数据不出图。"""

    objects = summary.get("objects")
    if not isinstance(objects, list) or not objects:
        return False
    color, _, _, dpi = _style(style)
    labels: list[str] = []
    explore: list[float] = []
    zone: list[float] = []
    for item in objects:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name") or item.get("id") or "")
        role = ("新奇" if item.get("role") == "novel"
                else "熟悉" if item.get("role") == "familiar" else "")
        labels.append(f"{name}\n{role}" if role else name)
        explore.append(_finite(item.get("exploration_s")) or 0.0)
        zone.append(_finite(item.get("zone_time_s")) or 0.0)
    if not labels:
        return False
    figure = Figure(figsize=(7.0, 4.0))
    FigureCanvasAgg(figure)
    axes = figure.add_subplot(111)
    positions = list(range(len(labels)))
    breadth = 0.34 if len(labels) > 1 else 0.26
    axes.bar([value - breadth / 2 for value in positions], explore, breadth,
             color=color, label="探索")
    axes.bar([value + breadth / 2 for value in positions], zone, breadth,
             color="#6F8F7C", label="接近范围")
    axes.set_xticks(positions)
    axes.set_xticklabels(labels, fontsize=8)
    axes.set_ylabel("s")
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.legend(fontsize=8, frameon=False)
    if title:
        axes.set_title(title, fontsize=9, color="#333333")
    figure.tight_layout()
    figure.savefig(path, format="png", dpi=dpi)
    return True


EPM_ZONE_LABELS = (("open", "开放臂"), ("closed", "封闭臂"),
                   ("center", "中心区"), ("unknown", "区外"))


def nor_exploration_points(rows: Iterable[Mapping[str, Any]]
                           ) -> tuple[list[list[tuple[float, float]]], bool]:
    """NOR 探索时序数据：逐对象按 dt_s 累计探索秒，缺失/无速度断开。

    返回 (每个物体的折线段列表, 是否存在有效速度)。累计值只在
    ``object_N_exploring`` 为真的合格行上增加；断开指该处插入 NaN，
    不把缺口前后的累计值连成线。
    """

    ordered = [row for row in rows
               if row.get("in_window") and _finite(row.get("dt_s")) is not None
               and float(row.get("dt_s")) > 0]
    ordered.sort(key=lambda row: float(row.get("time_s") or 0.0))
    has_speed = any(_finite(row.get("speed_cm_s")) is not None
                    for row in ordered)
    series: list[list[tuple[float, float]]] = [[], []]
    current: list[list[float]] = [[], []]
    cumulative = [0.0, 0.0]
    for row in ordered:
        exploring = (row.get("object_1_exploring"),
                     row.get("object_2_exploring"))
        broken = (not row.get("detected")
                  or _finite(row.get("speed_cm_s")) is None)
        moment = float(row.get("time_s")) + float(row.get("dt_s"))
        for index in (0, 1):
            if broken:
                # 缺失/无速度：当前折线在此断开；累计值保留，缺口不连线。
                if current[index]:
                    series[index].append(tuple(current[index]))
                    series[index].append((moment, float("nan")))
                    current[index] = []
                continue
            if exploring[index]:
                cumulative[index] += float(row.get("dt_s"))
                current[index] = [moment, cumulative[index]]
            elif current[index]:
                series[index].append(tuple(current[index]))
                current[index] = []
    for index in (0, 1):
        if current[index]:
            series[index].append(tuple(current[index]))
    return series, has_speed


def nor_exploration_png(path: str | Path, rows: Iterable[Mapping[str, Any]],
                        summary: Mapping[str, Any] | None = None, *,
                        title: str = "",
                        style: Mapping[str, Any] | None = None) -> bool:
    """NOR 探索时序图（7×4）：逐对象累计探索秒，缺失/无速度段断开。

    不是重新计算头向规则：只消费保存的 ``object_N_exploring`` 标记与
    真实曝光时长。没有任何探索观测或没有任何有效速度就不出图。
    """

    series, has_speed = nor_exploration_points(rows)
    if not has_speed or not any(points for points in series):
        return False
    color, _, line_width, dpi = _style(style)
    labels = ["物体 1", "物体 2"]
    objects = (summary or {}).get("objects")
    if isinstance(objects, list):
        names = []
        for item in objects[:2]:
            names.append(str(item.get("name") or item.get("id"))
                         if isinstance(item, Mapping) else None)
        labels = [(names[index] if names[index] else labels[index])
                  for index in (0, 1)]
    figure = Figure(figsize=(7.0, 4.0))
    FigureCanvasAgg(figure)
    axes = figure.add_subplot(111)
    colors = (color, "#1c5f9e")
    for index, points in enumerate(series):
        if not points:
            continue
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        axes.plot(xs, ys, color=colors[index], linewidth=line_width,
                  label=labels[index], solid_capstyle="round")
    axes.set_xlabel("源时间（s）")
    axes.set_ylabel("累计探索（s）")
    axes.set_xlim(left=0.0)
    axes.set_ylim(bottom=0.0)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    handles, labels_ = axes.get_legend_handles_labels()
    if handles:
        axes.legend(handles, labels_, fontsize=8, frameon=False)
    if title:
        axes.set_title(title, fontsize=9, color="#333333")
    figure.tight_layout()
    figure.savefig(path, format="png", dpi=dpi)
    return True


def epm_zones_png(path: str | Path, summary: Mapping[str, Any], *,
                  title: str = "",
                  style: Mapping[str, Any] | None = None) -> bool:
    """EPM 四类区域占用时长（秒）；没有区域数据不出图。"""

    seconds = summary.get("zone_seconds")
    if not isinstance(seconds, Mapping) or not seconds:
        return False
    color, _, _, dpi = _style(style)
    labels = [label for _, label in EPM_ZONE_LABELS]
    values = [float(_finite(seconds.get(key)) or 0.0)
              for key, _ in EPM_ZONE_LABELS]
    figure = Figure(figsize=(4.2, 2.6))
    FigureCanvasAgg(figure)
    axes = figure.add_subplot(111)
    positions = list(range(len(labels)))
    axes.bar(positions, values, 0.55, color=color)
    axes.set_xticks(positions)
    axes.set_xticklabels(labels, fontsize=8)
    axes.set_ylabel("s")
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    if title:
        axes.set_title(title, fontsize=9, color="#333333")
    figure.tight_layout()
    figure.savefig(path, format="png", dpi=dpi)
    return True


YMAZE_ARMS = ("A", "B", "C")


def ymaze_sequence_points(events: Iterable[Mapping[str, Any]]) -> list[dict]:
    """序列图数据：按最终事件顺序，重复臂全部保留，不重算交替。"""

    points: list[dict] = []
    for item in events or ():
        if not isinstance(item, Mapping):
            continue
        arm = item.get("target")
        if arm not in YMAZE_ARMS:
            continue
        moment = _finite(item.get("time_s"))
        if moment is None:
            continue
        index = item.get("entry_index")
        points.append({
            "entry_index": (int(index)
                            if isinstance(index, int)
                            and not isinstance(index, bool) and index > 0
                            else len(points) + 1),
            "arm": arm,
            "time_s": moment,
            "confirmed_s": _finite(item.get("confirmed_s")),
        })
    return points


def ymaze_sequence_png(path: str | Path,
                       events: Iterable[Mapping[str, Any]], *,
                       bins: Iterable[Mapping[str, Any]] | None = None,
                       window_start_s: float | None = None,
                       title: str = "",
                       style: Mapping[str, Any] | None = None) -> bool:
    """Y 迷宫入臂序列（臂 A/B/C × 绝对源秒）与每箱进入数。

    事件按最终顺序画在绝对源时刻；bins 的 start_s/end_s 是相对单位窗
    起点的偏移，须与窗口起点相加才是绝对时刻。没有确认事件不出图；
    没有 bins 或窗口起点时只画序列，不伪造每箱计数。
    """

    points = ymaze_sequence_points(events)
    if not points:
        return False
    bin_edges: list[tuple[float, float, int]] = []
    if window_start_s is not None:
        origin = _finite(window_start_s)
        if origin is not None:
            for item in bins or ():
                if not isinstance(item, Mapping):
                    continue
                low = _finite(item.get("start_s"))
                high = _finite(item.get("end_s"))
                if low is None or high is None or high <= low:
                    continue
                count = sum(1 for point in points
                            if origin + low <= point["time_s"] < origin + high)
                bin_edges.append((origin + low, origin + high, count))
    color, _, width, dpi = _style(style)
    figure = Figure(figsize=(7.0, 5.0))
    FigureCanvasAgg(figure)
    if bin_edges:
        seq_axes, bin_axes = figure.subplots(
            2, 1, sharex=True, gridspec_kw={"height_ratios": [2.6, 1.0]})
    else:
        seq_axes = figure.add_subplot(111)
        bin_axes = None
    rows = {arm: index for index, arm in enumerate(YMAZE_ARMS)}
    xs = [item["time_s"] for item in points]
    ys = [rows[item["arm"]] for item in points]
    seq_axes.plot(xs, ys, color=color, linewidth=max(1.0, width * 0.7),
                  marker="o", markersize=4, zorder=2)
    for item, x, y in zip(points, xs, ys):
        seq_axes.annotate(f"{item['entry_index']} {item['arm']}", (x, y),
                          textcoords="offset points", xytext=(0, 7),
                          ha="center", fontsize=7, color="#333333")
    seq_axes.set_yticks(range(len(YMAZE_ARMS)))
    seq_axes.set_yticklabels(list(YMAZE_ARMS), fontsize=8)
    seq_axes.set_ylim(len(YMAZE_ARMS) - 0.4, -0.6)
    low, high = min(xs), max(xs)
    if bin_edges:
        low = min(low, min(edge[0] for edge in bin_edges))
        high = max(high, max(edge[1] for edge in bin_edges))
    padding = max(0.1, (high - low) * 0.08)
    seq_axes.set_xlim(low - padding, high + padding)
    seq_axes.set_ylabel("臂", fontsize=8)
    seq_axes.tick_params(labelsize=8)
    seq_axes.spines["top"].set_visible(False)
    seq_axes.spines["right"].set_visible(False)
    if bin_axes is not None:
        centers = [(edge[0] + edge[1]) / 2.0 for edge in bin_edges]
        counts = [edge[2] for edge in bin_edges]
        span = min((edge[1] - edge[0] for edge in bin_edges), default=1.0)
        bin_axes.bar(centers, counts, width=span * 0.7, color="#6F8F7C")
        bin_axes.set_ylabel("进入（次）", fontsize=8)
        bin_axes.set_xlabel("源时间（s）", fontsize=8)
        bin_axes.set_ylim(0.0, max(1.0, max(counts) + 0.5))
        bin_axes.spines["top"].set_visible(False)
        bin_axes.spines["right"].set_visible(False)
        bin_axes.tick_params(labelsize=8)
    else:
        seq_axes.set_xlabel("s", fontsize=8)
    if title:
        seq_axes.set_title(title, fontsize=9, color="#333333")
    figure.tight_layout()
    figure.savefig(path, format="png", dpi=dpi)
    return True


def center_rect_for(summary: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    """场地中心矩形；只由保存的宽高与比例推得。"""

    width = _finite(summary.get("arena_width_cm"))
    height = _finite(summary.get("arena_height_cm"))
    rx = _finite(summary.get("center_ratio_x"))
    ry = _finite(summary.get("center_ratio_y"))
    if None in (width, height, rx, ry):
        return None
    return (width * (1.0 - rx) / 2.0, height * (1.0 - ry) / 2.0,
            width * (1.0 + rx) / 2.0, height * (1.0 + ry) / 2.0)


STYLE_PREVIEW_DPI_CAP = 240


def style_preview_png(path: str | Path, style: Mapping[str, Any] | None = None,
                      *, dpi_cap: int = STYLE_PREVIEW_DPI_CAP) -> bool:
    """图形样式的两面板预览（7×3 英寸）；预览 DPI 上限不回写正式 style。"""

    color, colormap, line_width, dpi = _style(style)
    dpi = max(50, min(int(dpi), int(dpi_cap)))
    figure = Figure(figsize=(7.0, 3.0))
    FigureCanvasAgg(figure)
    track_axes, heat_axes = figure.subplots(1, 2)

    track_axes.set_xlim(0.0, 30.0)
    track_axes.set_ylim(30.0, 0.0)
    track_axes.set_aspect("equal", adjustable="box")
    track_axes.add_patch(patches.Rectangle((0.0, 0.0), 30.0, 30.0, fill=False,
                                           edgecolor="#8a8a8a", linewidth=0.8))
    xs = [4.0, 9.0, 15.5, 21.0, 26.0]
    ys = [22.0, 12.0, 20.0, 8.0, 17.0]
    track_axes.plot(xs, ys, color=color, linewidth=line_width,
                    solid_capstyle="round")
    track_axes.set_xlabel("x (cm)", fontsize=8)
    track_axes.set_ylabel("y (cm)", fontsize=8)
    track_axes.tick_params(labelsize=8)
    track_axes.set_title("轨迹", fontsize=9, color="#333333")

    import numpy as np

    grid_x, grid_y = np.meshgrid(np.linspace(-3.0, 3.0, 48),
                                 np.linspace(-3.0, 3.0, 48))
    density = np.exp(-(grid_x ** 2 + (grid_y * 1.2) ** 2) / 1.6)
    density += 0.6 * np.exp(-((grid_x - 1.4) ** 2 + (grid_y + 1.2) ** 2) / 0.8)
    image = heat_axes.imshow(density, cmap=colormap, origin="upper",
                             extent=(-15.0, 15.0, 15.0, -15.0),
                             interpolation="bilinear")
    figure.colorbar(image, ax=heat_axes, fraction=0.046, pad=0.03, label="s")
    heat_axes.set_xlabel("x (cm)", fontsize=8)
    heat_axes.set_ylabel("y (cm)", fontsize=8)
    heat_axes.tick_params(labelsize=8)
    heat_axes.set_title("热图", fontsize=9, color="#333333")

    figure.tight_layout()
    figure.savefig(path, format="png", dpi=dpi)
    return True


STATE_COLORS = {
    "active": "#8C6BB1",

    "struggling": "#B45F45",
    "immobile": "#6F8F7C",
    "swing": "#4E7FA6",
    "unknown": "#CFC8BE",
}

# 活动特征曲线：决策 F 与速度波动 S，单位都是 mm/s。
CURVE_KEYS = {
    "decision_flow_p90_mm_s": ("#1c5f9e", "决策光流 p90"),
    "speed_std_mm_s": ("#16684e", "速度波动"),
}

# lis 的实际 F/S 来自残差模式自己的滚动量，不套用经典阈值。
LIS_CURVE_KEYS = {
    "lis_f": ("#1c5f9e", "光流 F"),
    "lis_s": ("#16684e", "速度波动 S"),
}

# FST 的分数曲线与最终时间线；阈值线用行内实际生效值。
FST_CURVE_KEYS = {
    "motion_score": ("#1c5f9e", "运动分数"),
    "smoothed_motion_score": ("#b86245", "平滑分数"),
}

FST_THRESHOLD_LINES = (
    ("threshold_low", "--", "不动阈值"),
    ("threshold_high", ":", "活动阈值"),
)

# 决策特征 F 使用的实际阈值参考线（虚线）：key、线型、图例标签。
FLOW_THRESHOLD_LINES = (
    ("flow_thr_hi_used", "--", "光流高阈值"),
    ("flow_thr_lo_used", ":", "光流低阈值"),
)


def activity_series(rows: Iterable[Mapping[str, Any]], key: str
                    ) -> tuple[list[float], list[float]]:
    """特征 key 的真实曝光阶梯序列 (x, y)。

    每行追加 (start, value) 与 (end, value) 两点：值在自己的曝光区间
    [start, end) 内成立，区间起点垂直跳变到新值；曝光不连续处用 NaN
    断开。feature_ready 为 False 的行不算证据，它的特征值是派生层的
    缺省 0，画出来就是伪造。
    """

    xs: list[float] = []
    ys: list[float] = []
    previous_end: float | None = None
    for row in rows:
        if not row.get("in_window"):
            continue
        if row.get("feature_ready") is False:
            continue
        start = _finite(row.get("interval_start_s"))
        end = _finite(row.get("interval_end_s"))
        value = _finite(row.get(key))
        if start is None or end is None or end <= start or value is None:
            continue
        if previous_end is not None and abs(start - previous_end) <= 1e-7:
            # 与上一行曝光相接：在同一 x 垂直跳变到本行的值。
            xs.append(start)
            ys.append(value)
        else:
            if xs:
                xs.append(float("nan"))
                ys.append(float("nan"))
            xs.append(start)
            ys.append(value)
        xs.append(end)
        ys.append(value)
        previous_end = end
    return xs, ys


def threshold_values(rows: Iterable[Mapping[str, Any]], key: str) -> list[float]:
    """行内实际使用的阈值；正常整段统一，取出现过的去重值。"""

    values: list[float] = []
    for row in rows:
        value = _finite(row.get(key))
        if value is not None and not any(abs(value - seen) <= 1e-12
                                         for seen in values):
            values.append(value)
    return values


def activity_png(path: str | Path, rows: Iterable[Mapping[str, Any]], *,
                 style: Mapping[str, Any] | None = None,
                 title: str = "", mode: str = "classic") -> bool:
    """上轴画 F/S 活动特征曲线，下轴按最终状态画时间线。

    逐行用真实曝光 [start,end)，缺口留白；没有可用状态不出图，
    没有配对证据时退回只画状态色带。lis 用 lis_f/lis_s 两条实际曲线，
    fst 用原始/平滑运动分数与实际阈值线；两者都不套用经典阈值。
    """

    items = []
    for row in rows:
        if not row.get("in_window"):
            continue
        state = row.get("state")
        span = _finite(row.get("dt_s"))
        start = _finite(row.get("interval_start_s"))
        end = _finite(row.get("interval_end_s"))
        if state not in STATE_COLORS or span is None or span <= 0 \
                or start is None or end is None or end <= start:
            continue
        items.append((start, end, state))
    if not items:
        return False

    curve_keys = LIS_CURVE_KEYS if mode == "lis" else CURVE_KEYS
    lines = () if mode == "lis" else FLOW_THRESHOLD_LINES
    if mode == "fst":
        curve_keys = FST_CURVE_KEYS
        lines = FST_THRESHOLD_LINES
    series = {key: activity_series(rows, key) for key in curve_keys}
    has_curves = any(xs for xs, _ in series.values())
    _, _, line_width, dpi = _style(style)

    begin = min(item[0] for item in items)
    finish = max(item[1] for item in items)
    for xs, _ in series.values():
        finite = [x for x in xs if math.isfinite(x)]
        if finite:
            begin = min(begin, finite[0])
            finish = max(finish, finite[-1])

    figure = Figure(figsize=(8.0, 4.0))
    FigureCanvasAgg(figure)
    if has_curves:
        curve_axes, band_axes = figure.subplots(
            2, 1, sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]})
    else:
        curve_axes = None
        band_axes = figure.add_subplot(111)

    if curve_axes is not None:
        for key, (color, label) in curve_keys.items():
            xs, ys = series[key]
            if not xs:
                continue
            # 序列已含每行区间起点与终点的阶梯坐标，直接直线连接。
            curve_axes.plot(xs, ys, color=color, linewidth=line_width,
                            label=label)
        if mode != "lis":
            for key, dash, label in lines:
                for value in threshold_values(rows, key):
                    curve_axes.axhline(value, color="#8a8a8a", linestyle=dash,
                                       linewidth=max(0.8, line_width * 0.6),
                                       label=label)
        curve_axes.set_ylabel("分数" if mode == "fst" else "mm/s")
        handles, labels = curve_axes.get_legend_handles_labels()
        if handles:
            curve_axes.legend(handles, labels, loc="upper right", fontsize=7,
                              frameon=False, ncol=2)
        curve_axes.spines["top"].set_visible(False)
        curve_axes.spines["right"].set_visible(False)
        if title:
            curve_axes.set_title(title, fontsize=9, color="#333333")

    band_axes.set_xlim(begin, finish)
    band_axes.set_ylim(0.0, 1.0)
    band_axes.set_yticks([])
    band_axes.set_xlabel("time (s)")
    for start, end, state in items:
        band_axes.add_patch(patches.Rectangle(
            (start, 0.0), end - start, 1.0,
            facecolor=STATE_COLORS[state], edgecolor="none"))
    present = []
    for state in ("active", "struggling", "immobile", "swing", "unknown"):
        if any(item[2] == state for item in items):
            present.append(patches.Patch(color=STATE_COLORS[state], label=state))
    if present:
        band_axes.legend(handles=present, loc="upper right", fontsize=7,
                         frameon=False, ncol=len(present))
    band_axes.spines["top"].set_visible(False)
    band_axes.spines["right"].set_visible(False)
    band_axes.spines["left"].set_linewidth(min(1.0, line_width))
    if not has_curves and title:
        band_axes.set_title(title, fontsize=9, color="#333333")
    figure.tight_layout()
    figure.savefig(path, format="png", dpi=dpi)
    return True
