"""公共时间事实：统一窗口、真实曝光裁切、完成运动段与分箱。

实验只消费这里保存的观测与运动段，不各自用“帧数÷FPS”重算时间，也不在
绘图时重新接续缺失轨迹。窗口未确定时只返回空窗口与零曝光，不伪造时长。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from . import experiments
from . import geometry as geo
from .schema import ValidationError, is_number, make_observation

TIME_EPS = 1e-9
EXPOSURE_EPS = 1e-6
SPLIT_EPS = 1e-9
EXACT_MODES = ("exact", "absolute")
UNRESOLVED_WARNINGS = ("oft", "nor")
MOTION_FIELDS = ("step", "speed", "motion_start_s", "motion_end_s",
                 "motion_x0", "motion_y0", "motion_x1", "motion_y1")


@dataclass(frozen=True)
class Window:
    """动物窗口；start/end 为 None 表示未确定（相对窗没有真实起点）。"""

    start_s: float | None = None
    end_s: float | None = None

    @property
    def known(self) -> bool:
        return self.start_s is not None and self.end_s is not None

    @property
    def duration_s(self) -> float:
        if not self.known:
            return 0.0
        return max(0.0, float(self.end_s) - float(self.start_s))


UNKNOWN_WINDOW = Window()


def _finite(value: Any, field: str, message: str = "必须是有限数值") -> float:
    if not is_number(value) or not math.isfinite(float(value)):
        raise ValidationError(message, field)
    return float(value)


def resolve_video_window(window: Mapping[str, Any] | None,
                         rows: Iterable[Mapping[str, Any]] = (),
                         metadata_duration: Any = None) -> Window:
    """视频窗口：显式终点不超过实际可用末端；无观测才用元数据时长回退。"""

    config = window if isinstance(window, Mapping) else {}
    start = _finite(config.get("start_s", 0.0), "window.start_s",
                    "窗口起点必须是非负有限数")
    if start < 0:
        raise ValidationError("窗口起点必须是非负有限数", "window.start_s")

    actual_end: float | None = None
    has_rows = False
    for row in rows:
        has_rows = True
        value = row.get("source_interval_end_s")
        if is_number(value) and math.isfinite(float(value)):
            actual_end = (float(value) if actual_end is None
                          else max(actual_end, float(value)))
    if not has_rows and is_number(metadata_duration) \
            and math.isfinite(float(metadata_duration)):
        # 没有观测时元数据时长就是实际可用末端，显式终点同样不能超过它。
        actual_end = float(metadata_duration)

    declared = config.get("end_s")
    if declared is not None:
        end = _finite(declared, "window.end_s", "窗口终点必须是有限数值或 null")
        if actual_end is not None:
            end = min(end, actual_end)
    elif actual_end is not None:
        end = actual_end
    else:
        end = start
    return Window(start, max(start, end))


def resolve_unit_window(window: Mapping[str, Any] | None, video: Window,
                        first_seen: Any = None) -> Window:
    """单位窗口：global 沿用视频窗；exact 取绝对秒；relative 需真实起点。"""

    config = window if isinstance(window, Mapping) else {}
    mode = config.get("mode") or "global"
    if mode == "global":
        return video
    if mode in EXACT_MODES:
        start = _finite(config.get("start_s"), "window.start_s",
                        "窗口起点必须是非负有限数")
        end = _finite(config.get("end_s"), "window.end_s",
                      "窗口终点必须是有限数值")
        if start < 0:
            raise ValidationError("窗口起点必须是非负有限数", "window.start_s")
        if end <= start:
            raise ValidationError("窗口终点必须大于起点", "window.end_s")
        if video.known and min(end, video.end_s) <= max(start, video.start_s):
            raise ValidationError("窗口与视频范围没有重叠", "window.end_s")
        return Window(start, end)
    if mode == "relative":
        offset = _finite(config.get("offset_s", 0.0), "window.offset_s",
                         "偏移必须是有限数值")
        duration = _finite(config.get("duration_s"), "window.duration_s",
                           "持续时间必须是有限正数")
        if duration <= 0:
            raise ValidationError("持续时间必须大于 0", "window.duration_s")
        if first_seen is None:
            return UNKNOWN_WINDOW
        # 零秒是有效起点，不用真假判断把它丢掉。
        anchor = _finite(first_seen, "first_seen", "首次检测时间必须是有限数值")
        low = anchor + offset
        high = low + duration
        if not video.known:
            return Window(low, high)
        left = max(float(video.start_s), low)
        right = max(left, min(float(video.end_s), high))
        return Window(left, right)
    raise ValidationError("未知窗口模式", "window.mode")


def _first_seen_rule(assay: str) -> str:
    """首检规则由各实验模块声明；没有声明时用中心存在即算。"""

    if not experiments.exists(assay):
        return "center"
    return getattr(experiments.get(assay), "FIRST_SEEN", "center")


def first_seen_time(assay: str, detection: Any, time_s: Any,
                    geometry: Mapping[str, Any] | None = None) -> float | None:
    """各实验的首次检测规则；声明 arena 的实验还要求中心在场地内。"""

    if detection is None or not getattr(detection, "detected", False):
        return None
    moment = _finite(time_s, "time_s", "检测时间必须是有限数值")
    if _first_seen_rule(assay) == "arena":
        arena = (geometry or {}).get("arena")
        if not geo.shape_contains(arena, detection.center):
            return None
    return moment


def first_seen_in_window(assay: str, rows: Iterable[Mapping[str, Any]],
                         window: Window,
                         geometry: Mapping[str, Any] | None = None) -> float | None:
    """视频统计半开窗 [start,end) 内的首个真实检测；尾上下文不算锚点。"""

    if not window.known:
        return None
    start = float(window.start_s)
    end = float(window.end_s)
    for row in sort_rows(rows):
        moment = float(row["time_s"])
        if moment < start:
            continue
        if moment >= end:
            break
        if not row.get("detected"):
            continue
        if _first_seen_rule(assay) == "arena":
            arena = (geometry or {}).get("arena")
            if not geo.shape_contains(arena, (row.get("x_px"), row.get("y_px"))):
                continue
        return moment
    return None


def base_row(unit: Mapping[str, Any], frame: Any,
             detection: Any = None, *, extra: Mapping[str, Any] | None = None) -> dict:
    """源观测行：源区间取 Frame 显式值，缺失时才回退 time_s/time_s+dt_s。"""

    start = frame.interval_start_s
    if not is_number(start) or not math.isfinite(float(start)):
        start = frame.time_s
    end = frame.interval_end_s
    if not is_number(end) or not math.isfinite(float(end)):
        end = frame.time_s + frame.dt_s
    start = float(start)
    end = max(start, float(end))
    return make_observation(
        unit_id=unit.get("id"),
        unit_name=unit.get("name"),
        animal_id=unit.get("animal_id"),
        frame=frame.index,
        time_s=frame.time_s,
        source_dt_s=frame.dt_s,
        elapsed_s=frame.elapsed_s,
        analysis_fps=frame.analysis_fps,
        source_interval_start_s=start,
        source_interval_end_s=end,
        interval_start_s=start,
        interval_end_s=end,
        detection=detection,
        in_window=end > start,
        extra=extra,
    )


def clip_rows(rows: Iterable[Mapping[str, Any]], window: Window) -> list[dict]:
    """按动物窗口裁切曝光；保留原观测与源区间身份，不删行。"""

    clipped: list[dict] = []
    for row in rows:
        item = dict(row)
        source_start = float(item["source_interval_start_s"])
        source_end = float(item["source_interval_end_s"])
        item["window_start_s"] = window.start_s if window.known else None
        item["window_end_s"] = window.end_s if window.known else None
        if not window.known:
            item["interval_start_s"] = source_start
            item["interval_end_s"] = source_start
            item["dt_s"] = 0.0
            item["in_window"] = False
        else:
            low = max(float(window.start_s), source_start)
            high = min(float(window.end_s), source_end)
            if high < low:
                high = low
            item["interval_start_s"] = low
            item["interval_end_s"] = high
            item["dt_s"] = max(0.0, high - low)
            item["in_window"] = high > low
        clipped.append(item)
    return clipped


def sort_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict]:
    """按动物分组，再按 (time_s, frame) 升序。"""

    return sorted(rows, key=lambda row: (row["unit_id"], row["time_s"],
                                         row["frame"]))


def time_summary(rows: Iterable[Mapping[str, Any]]) -> dict:
    """窗口总曝光 / 观察占用 / 缺失；缺失只进总曝光，不进观察占用。"""

    total = 0.0
    observed = 0.0
    for row in rows:
        duration = float(row.get("dt_s") or 0.0)
        if duration <= 0:
            continue
        total += duration
        if row.get("detected"):
            observed += duration
    return {
        "total_time_s": total,
        "observed_time_s": observed,
        "missing_time_s": max(0.0, total - observed),
    }


def _valid_point(row: Mapping[str, Any], x_key: str,
                 y_key: str) -> tuple[float, float] | None:
    if not row.get("detected"):
        return None
    if row.get("position_valid") is False or row.get("jump") is True:
        return None
    x, y = row.get(x_key), row.get(y_key)
    if not is_number(x) or not is_number(y):
        return None
    x, y = float(x), float(y)
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return x, y


def _lerp(start: Sequence[float], end: Sequence[float],
          ratio: float) -> list[float]:
    return [float(start[0]) + (float(end[0]) - float(start[0])) * ratio,
            float(start[1]) + (float(end[1]) - float(start[1])) * ratio]


def exposures_continuous(previous: Mapping[str, Any] | None,
                         row: Mapping[str, Any]) -> bool:
    """前条真实曝光末端与本条起点相接；缺口或缺键都不算连续。

    优先比较源曝光（窗口裁切前的真实区间），这样窗口端点仍能借用相邻
    上下文插值；没有源区间时退回裁切后的区间。
    """

    if previous is None:
        return False
    for start_key, end_key in (
            ("source_interval_start_s", "source_interval_end_s"),
            ("interval_start_s", "interval_end_s")):
        if not is_number(previous.get(end_key)) \
                or not is_number(row.get(start_key)):
            continue
        return abs(float(previous[end_key])
                   - float(row[start_key])) <= EXPOSURE_EPS
    return False


def motion_segments(rows: Iterable[Mapping[str, Any]], window: Window, *,
                    x_key: str = "x_px", y_key: str = "y_px",
                    require_continuity: bool = False) -> list[dict]:
    """只用连续真实点构造已完成段；缺失或无效行立即清空前点。

    ``require_continuity`` 为真时还要求前条曝光末端与本条起点相接，
    中间没有真实曝光的两个点不连成段；窗口端点仍按时间插值裁剪。
    """

    if not window.known:
        return []
    segments: list[dict] = []
    previous: tuple[float, tuple[float, float], Mapping[str, Any]] | None = None
    previous_unit: Any = None
    for row in sort_rows(rows):
        if row["unit_id"] != previous_unit:
            # 换动物即断开：不让上一只鼠的末点连到这一只的首点。
            previous = None
            previous_unit = row["unit_id"]
        point = _valid_point(row, x_key, y_key)
        if point is None:
            previous = None
            continue
        moment = float(row["time_s"])
        if previous is None:
            previous = (moment, point, row)
            continue
        last_time, last_point, last_row = previous
        if moment <= last_time:
            previous = (moment, point, row)
            continue
        if require_continuity and not exposures_continuous(last_row, row):
            # 曝光被缺口或零曝光打断：只把当前点当新起点，不跨越缺口。
            previous = (moment, point, row)
            continue
        low = max(last_time, float(window.start_s))
        high = min(moment, float(window.end_s))
        if high > low:
            span = moment - last_time
            start_ratio = (low - last_time) / span
            end_ratio = (high - last_time) / span
            p0 = _lerp(last_point, point, start_ratio)
            p1 = _lerp(last_point, point, end_ratio)
            distance = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            segments.append({
                "unit_id": row["unit_id"],
                "start_s": low,
                "end_s": high,
                "p0": p0,
                "p1": p1,
                "distance": distance,
                "speed": math.hypot(point[0] - last_point[0],
                                    point[1] - last_point[1]) / span,
                "frame": row["frame"],
            })
        previous = (moment, point, row)
    return segments


def attach_motion(rows: Iterable[Mapping[str, Any]],
                  segments: Iterable[Mapping[str, Any]]) -> None:
    """行级运动事实：先全部置空，再按 (unit_id, frame) 写入已完成段。"""

    keyed = {(segment["unit_id"], segment["frame"]): segment
             for segment in segments}
    for row in rows:
        for field in MOTION_FIELDS:
            row[field] = None
        segment = keyed.get((row["unit_id"], row["frame"]))
        if segment is None:
            continue
        row["step"] = float(segment["distance"])
        row["speed"] = float(segment["speed"])
        row["motion_start_s"] = float(segment["start_s"])
        row["motion_end_s"] = float(segment["end_s"])
        row["motion_x0"] = float(segment["p0"][0])
        row["motion_y0"] = float(segment["p0"][1])
        row["motion_x1"] = float(segment["p1"][0])
        row["motion_y1"] = float(segment["p1"][1])


def split_interval(start_s: float, end_s: float, origin: float,
                   width: float) -> list[tuple[int, float, float]]:
    """按 origin+k·width 拆分；右端半开，返回 (0 基索引, 起点, 终点)。"""

    if not is_number(width) or not math.isfinite(float(width)) or float(width) <= 0:
        raise ValidationError("箱宽必须是有限正数", "width")
    width = float(width)
    start_s, end_s, origin = float(start_s), float(end_s), float(origin)
    if end_s - start_s <= TIME_EPS:
        return []
    pieces: list[tuple[int, float, float]] = []
    cursor = start_s
    while cursor < end_s - TIME_EPS:
        index = int(math.floor((cursor - origin) / width + SPLIT_EPS))
        boundary = origin + (index + 1) * width
        stop = min(end_s, boundary)
        if stop <= cursor + TIME_EPS:
            index += 1
            stop = min(end_s, origin + (index + 1) * width)
            if stop <= cursor + TIME_EPS:
                break
        pieces.append((index, cursor, stop))
        cursor = stop
    return pieces


def _segment_distance(segment: Mapping[str, Any], low: float,
                      high: float) -> float:
    start = float(segment["start_s"])
    end = float(segment["end_s"])
    span = end - start
    if span <= 0:
        return 0.0
    p0 = _lerp(segment["p0"], segment["p1"], (low - start) / span)
    p1 = _lerp(segment["p0"], segment["p1"], (high - start) / span)
    return math.hypot(p1[0] - p0[0], p1[1] - p0[1])


def build_bins(rows: Iterable[Mapping[str, Any]],
               segments: Iterable[Mapping[str, Any]], window: Window,
               width: Any) -> list[dict]:
    """以动物窗起点为 origin 的时间箱：曝光按交叠、距离按插值分配。"""

    if not window.known or window.duration_s <= 0:
        return []
    if not is_number(width) or not math.isfinite(float(width)) or float(width) <= 0:
        raise ValidationError("箱宽必须是有限正数", "width")
    width = float(width)
    origin = float(window.start_s)
    duration = window.duration_s
    count = max(1, int(math.ceil((duration - TIME_EPS) / width)))

    exposure: dict[int, float] = {}
    for row in rows:
        for index, low, high in split_interval(row["interval_start_s"],
                                               row["interval_end_s"],
                                               origin, width):
            if 0 <= index < count:
                exposure[index] = exposure.get(index, 0.0) + (high - low)

    distance: dict[int, float] = {}
    for segment in segments:
        for index, low, high in split_interval(segment["start_s"],
                                               segment["end_s"],
                                               origin, width):
            if 0 <= index < count:
                distance[index] = distance.get(index, 0.0) + _segment_distance(
                    segment, low, high)

    bins: list[dict] = []
    for index in range(count):
        start = index * width
        end = min(start + width, duration)
        bins.append({
            "bin_index": index + 1,
            "start_s": start,
            "end_s": end,
            "duration_s": end - start,
            "exposure_s": exposure.get(index, 0.0),
            "distance": distance.get(index, 0.0),
        })
    return bins


def heatmap_weights(rows: Iterable[Mapping[str, Any]], *,
                    x_key: str = "x_px",
                    y_key: str = "y_px") -> list[tuple[float, float, float]]:
    """热图点：仅窗口内真实位置，按 dt_s（秒）加权。"""

    points: list[tuple[float, float, float]] = []
    for row in rows:
        if not row.get("in_window"):
            continue
        duration = float(row.get("dt_s") or 0.0)
        if duration <= 0:
            continue
        point = _valid_point(row, x_key, y_key)
        if point is None:
            continue
        points.append((point[0], point[1], duration))
    return points


def ratio(numerator: Any, denominator: Any) -> float | None:
    """分母为零或非有限时返回 null，不写 0%、NaN 或 Infinity。"""

    if not is_number(numerator) or not is_number(denominator):
        return None
    numerator, denominator = float(numerator), float(denominator)
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return None
    if denominator <= 0:
        return None
    return numerator / denominator


def window_warning(assay: str, window: Window) -> str | None:
    """相对窗未确定时，只有 OFT/NOR 按各自契约追加警告。"""

    if window.known or assay not in UNRESOLVED_WARNINGS:
        return None
    return "相对窗口未确定：没有真实检测起点，时间与比例保持为空。"
