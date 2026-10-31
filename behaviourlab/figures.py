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


def valid_positions(rows: Iterable[Mapping[str, Any]],
                    width_cm: float, height_cm: float) -> list[tuple[float, float, float]]:
    """窗内有效位置的 (x, y, dt)；jump / position_valid=false 不算有效点。"""

    points: list[tuple[float, float, float]] = []
    for row in rows:
        if not row.get("in_window") or not row.get("detected"):
            continue
        if row.get("position_valid") is False or row.get("jump") is True:
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
    ratio = height_cm / width_cm if width_cm > 0 else 1.0
    height = 4.2
    figure = Figure(figsize=(4.2, max(2.6, min(6.0, height * ratio))))
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


def center_rect_for(summary: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    width = _finite(summary.get("arena_width_cm"))
    height = _finite(summary.get("arena_height_cm"))
    rx = _finite(summary.get("center_ratio_x"))
    ry = _finite(summary.get("center_ratio_y"))
    if None in (width, height, rx, ry):
        return None
    return (width * (1.0 - rx) / 2.0, height * (1.0 - ry) / 2.0,
            width * (1.0 + rx) / 2.0, height * (1.0 + ry) / 2.0)


STATE_COLORS = {
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
                 title: str = "") -> bool:
    """上轴画 F/S 活动特征曲线，下轴按最终状态画时间线。

    逐行用真实曝光 [start,end)，缺口留白；没有可用状态不出图，
    没有配对证据时退回只画状态色带。
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

    series = {key: activity_series(rows, key) for key in CURVE_KEYS}
    has_curves = any(xs for xs, _ in series.values())
    _, _, line_width, dpi = _style(style)

    begin = min(item[0] for item in items)
    finish = max(item[1] for item in items)
    for xs, _ in series.values():
        finite = [x for x in xs if math.isfinite(x)]
        if finite:
            begin = min(begin, finite[0])
            finish = max(finish, finite[-1])

    figure = Figure(figsize=(6.4, 3.4 if has_curves else 1.9))
    FigureCanvasAgg(figure)
    if has_curves:
        curve_axes, band_axes = figure.subplots(
            2, 1, sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]})
    else:
        curve_axes = None
        band_axes = figure.add_subplot(111)

    if curve_axes is not None:
        for key, (color, label) in CURVE_KEYS.items():
            xs, ys = series[key]
            if not xs:
                continue
            # 序列已含每行区间起点与终点的阶梯坐标，直接直线连接。
            curve_axes.plot(xs, ys, color=color, linewidth=line_width,
                            label=label)
        for key, dash, label in FLOW_THRESHOLD_LINES:
            for value in threshold_values(rows, key):
                curve_axes.axhline(value, color="#8a8a8a", linestyle=dash,
                                   linewidth=max(0.8, line_width * 0.6),
                                   label=label)
        curve_axes.set_ylabel("mm/s")
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
    for state in ("struggling", "immobile", "swing", "unknown"):
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
