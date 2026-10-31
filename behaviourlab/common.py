"""窗口、曝光与运动段的公共时间工具。

相机的曝光/时钟推导在媒体块接入；本模块只保留纯区间运算。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Interval:
    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        if self.end_s < self.start_s:
            raise ValueError("区间终点不能早于起点")

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def intersect(a: Interval, b: Interval) -> Interval | None:
    start = max(a.start_s, b.start_s)
    end = min(a.end_s, b.end_s)
    if end <= start:
        return None
    return Interval(start, end)


def clip_to(start_s: float, end_s: float, window: Interval) -> Interval | None:
    return intersect(Interval(start_s, end_s), window)


def merge_segments(segments: list[Interval], gap_s: float = 0.0) -> list[Interval]:
    """合并重叠或间隔不超过 gap_s 的区间。"""

    if not segments:
        return []
    ordered = sorted(segments, key=lambda item: item.start_s)
    merged = [ordered[0]]
    for segment in ordered[1:]:
        last = merged[-1]
        if segment.start_s - last.end_s <= gap_s:
            merged[-1] = Interval(last.start_s, max(last.end_s, segment.end_s))
        else:
            merged.append(segment)
    return merged
