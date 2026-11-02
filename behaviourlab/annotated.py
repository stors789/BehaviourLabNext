"""叠画视频：源像素画面 + 每动物几何/位置/状态与已确认事件。

只重新解码原视频用于叠画，不重新检测或评分。每帧的显示时刻取事件
``confirmed_s``（有限时），否则用 ``time_s`` 并标 Unconfirmed。时间轴按量化
绝对相对端点写 PTS/duration，避免逐帧舍入累积。
"""

from __future__ import annotations

import math
import os
import time
from bisect import bisect_left, bisect_right
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import av
import numpy as np

from . import media

TIMEBASE = Fraction(1, 1_000_000)
TIMEBASE_LABEL = "1/1000000"
DECODE_TOLERANCE_S = 0.002
QUANTIZE_TOLERANCE_S = 1e-6


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def clock_text(seconds: Any) -> str:
    """源绝对秒 → mm:ss.sss，与界面帧标签同一写法。"""

    moment = _finite(seconds)
    if moment is None:
        return "--:--.---"
    total = max(0.0, moment)
    minutes = int(total // 60)
    return f"{minutes:02d}:{total - minutes * 60:06.3f}"


def _ascii_label(text: Any) -> str:
    """cv2 只能可靠画 ASCII；非 ASCII 名称退回通用标签。"""

    value = str(text or "")
    return value if value and value.isascii() else "unit"


def _even_canvas(canvas: Any) -> Any:
    """奇数宽/高各在右/下补 1 个黑像素到偶数；不拉伸也不复制边缘。"""

    import cv2

    height, width = canvas.shape[:2]
    pad_h = height % 2
    pad_w = width % 2
    if not pad_h and not pad_w:
        return canvas
    return cv2.copyMakeBorder(canvas, 0, pad_h, 0, pad_w,
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))


def _to_bgr(image: Any):
    import cv2

    if image is None:
        return None
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


# -- 几何叠画 -------------------------------------------------------------


def _points(shape: Mapping[str, Any]) -> list[tuple[int, int]]:
    raw = shape.get("points") or []
    output = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            x, y = _finite(item[0]), _finite(item[1])
            if x is not None and y is not None:
                output.append((int(round(x)), int(round(y))))
    return output


def _draw_shape(canvas: Any, shape: Any, color: tuple[int, int, int],
                thickness: int = 1) -> None:
    import cv2

    if not isinstance(shape, Mapping):
        return
    kind = shape.get("type")
    if kind == "ellipse":
        center = shape.get("center") or []
        radii = shape.get("radii") or []
        if len(center) < 2 or len(radii) < 2:
            return
        cx, cy = _finite(center[0]), _finite(center[1])
        rx, ry = _finite(radii[0]), _finite(radii[1])
        if None in (cx, cy, rx, ry) or rx <= 0 or ry <= 0:
            return
        cv2.ellipse(canvas, (int(round(cx)), int(round(cy))),
                    (int(round(rx)), int(round(ry))),
                    float(shape.get("angle") or 0.0), 0, 360, color,
                    thickness, cv2.LINE_AA)
        return
    if kind == "circle":
        center = shape.get("center") or []
        radius = _finite(shape.get("radius"))
        if len(center) < 2 or radius is None or radius <= 0:
            return
        cx, cy = _finite(center[0]), _finite(center[1])
        if cx is None or cy is None:
            return
        cv2.circle(canvas, (int(round(cx)), int(round(cy))),
                   int(round(radius)), color, thickness, cv2.LINE_AA)
        return
    points = _points(shape)
    if kind == "point" and points:
        x, y = points[0]
        cv2.drawMarker(canvas, (x, y), color, cv2.MARKER_CROSS, 10,
                       thickness, cv2.LINE_AA)
        return
    if len(points) >= 2:
        array = __import__("numpy").array(points, dtype="int32")
        closed = kind != "line"
        cv2.polylines(canvas, [array], closed, color, thickness, cv2.LINE_AA)


GEOMETRY_KEYS = ("arena", "body_line", "diameter_line", "waterline",
                 "platform", "open_boundary")
GEOMETRY_COLORS = {
    "arena": (111, 210, 255),
    "body_line": (246, 170, 82),
    "diameter_line": (246, 170, 82),
    "waterline": (200, 190, 120),
    "platform": (180, 160, 240),
    "open_boundary": (255, 210, 111),
}


def _draw_geometry(canvas: Any, geometry: Mapping[str, Any],
                   thickness: int) -> None:
    for key in GEOMETRY_KEYS:
        _draw_shape(canvas, geometry.get(key), GEOMETRY_COLORS[key], thickness)
    zones = geometry.get("zones")
    if isinstance(zones, Mapping):
        for value in zones.values():
            _draw_shape(canvas, value, (196, 168, 96), thickness)
    elif isinstance(zones, list):
        for value in zones:
            _draw_shape(canvas, value, (196, 168, 96), thickness)
    objects = geometry.get("objects")
    if isinstance(objects, list):
        for item in objects:
            if not isinstance(item, Mapping):
                continue
            _draw_shape(canvas, item.get("zone"), (196, 168, 96), thickness)
            center = item.get("center")
            if isinstance(center, (list, tuple)) and len(center) >= 2:
                x, y = _finite(center[0]), _finite(center[1])
                if x is not None and y is not None:
                    import cv2

                    cv2.circle(canvas, (int(round(x)), int(round(y))), 4,
                               (196, 168, 96), thickness, cv2.LINE_AA)
    holes = geometry.get("holes")
    if isinstance(holes, list):
        for item in holes:
            if isinstance(item, Mapping):
                _draw_shape(canvas, item, (196, 168, 96), thickness)


def _draw_label(canvas: Any, lines: Sequence[str]) -> Any:
    """左上角半透明底栏：源时间、帧号、动物状态与事件。"""

    import cv2

    lines = [line for line in lines if line]
    if not lines:
        return canvas
    height, width = canvas.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.3, min(0.55, width / 720.0))
    sizes = [cv2.getTextSize(text, font, scale, 1)[0] for text in lines]
    widest = max(size[0] for size in sizes)
    limit = max(20, width - 20)
    if widest > limit:
        scale *= limit / widest
        sizes = [cv2.getTextSize(text, font, scale, 1)[0] for text in lines]
    line_height = max(size[1] for size in sizes) + 6
    band = min(height, 8 + line_height * len(lines))
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (width, band), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.62, canvas, 0.38, 0.0, canvas)
    y = 5
    for text in lines:
        y += line_height
        cv2.putText(canvas, text, (10, y - 4), font, scale, (246, 246, 246),
                    1, cv2.LINE_AA)
    return canvas


# -- 时间轴 ---------------------------------------------------------------


def _unit_windows(units: Sequence[Mapping[str, Any]]) -> list[dict]:
    prepared = []
    for unit in units:
        start = _finite(unit.get("window_start_s"))
        end = _finite(unit.get("window_end_s"))
        if start is None:
            continue
        prepared.append({
            "unit_id": unit.get("unit_id"),
            "unit_name": unit.get("unit_name"),
            "start": float(start),
            "end": float(end) if end is not None else math.inf,
            "geometry": unit.get("geometry") or {},
            "manual": bool(unit.get("manual")),
            "observations": list(unit.get("observations") or []),
            "suppress_positions": bool(unit.get("suppress_positions")),
        })
    return prepared


def _display_time(event: Mapping[str, Any]) -> tuple[float | None, bool]:
    """事件显示时刻：confirmed_s 优先，否则 time_s 且未确认。"""

    confirmed = _finite(event.get("confirmed_s"))
    if confirmed is not None:
        return confirmed, True
    moment = _finite(event.get("time_s"))
    return moment, False


def _segment_bounds(frames: Sequence[Mapping[str, Any]],
                    units: Sequence[Mapping[str, Any]],
                    events: Sequence[Mapping[str, Any]]) -> list[float]:
    points: set[float] = set()
    for frame in frames:
        points.add(float(frame["start"]))
        points.add(float(frame["end"]))
    for unit in units:
        points.add(float(unit["start"]))
        points.add(float(unit["end"]))
    for event in events:
        moment = event.get("display_s")
        if moment is not None:
            points.add(float(moment))
    return sorted(points)


def _split(start: float, end: float, points: Sequence[float]) -> list[tuple]:
    """区间内切分：二分定位落在 (start, end) 内部的切点，不再逐点扫全表。"""

    if end - start <= 1e-12:
        return []
    low = bisect_right(points, start + 1e-9)
    high = bisect_left(points, end - 1e-9)
    cuts = [start, *points[low:high], end]
    return [(cuts[index], cuts[index + 1])
            for index in range(len(cuts) - 1)
            if cuts[index + 1] - cuts[index] > 1e-12]


def _clip_count(start: float, end: float, points: Sequence[float]) -> int:
    """该曝光会切成几片；只用于预估进度，不构造片段。"""

    if end - start <= 1e-12:
        return 0
    low = bisect_right(points, start + 1e-9)
    high = bisect_left(points, end - 1e-9)
    return max(0, high - low) + 1


def _active_units(units: Sequence[Mapping[str, Any]], start: float,
                  end: float) -> list[Mapping[str, Any]]:
    middle = (start + end) / 2.0
    return [unit for unit in units
            if unit["start"] - 1e-9 <= middle < unit["end"] - 1e-9]


def _cm_to_px_mapper(rows: Sequence[Mapping[str, Any]]) -> Any:
    """从本 run 保存的成对坐标恢复测量坐标→旋转后源像素的映射。

    完成段（motion_x0..y1）保存的是测量坐标（cm 或估计 cm），不是像素。
    这里用同一 run 中 ``(x_px, y_px)`` 与 ``(x_cm, y_cm)`` 同时有效的
    观测恢复二者之间的线性映射，再把完成段端点映射回源像素。数据声明
    就是像素、或没有任何坐标声明的历史数据按原样直画；声明为 cm 而成
    对坐标恢复不出一致线性映射（如分区透视）时返回 None——宁可少画
    一层轨迹，也不能把物理/估计单位当像素画。
    """

    declared: str | None = None
    pairs: list[tuple[float, float, float, float]] = []
    for row in rows:
        value = row.get("coordinate_unit")
        if isinstance(value, str) and value:
            declared = value
        if row.get("position_valid") is False or row.get("jump"):
            continue
        x_px, y_px = _finite(row.get("x_px")), _finite(row.get("y_px"))
        x_cm, y_cm = _finite(row.get("x_cm")), _finite(row.get("y_cm"))
        if None in (x_px, y_px, x_cm, y_cm):
            continue
        pairs.append((x_cm, y_cm, x_px, y_px))
    if declared in (None, "px"):
        return lambda point: (float(point[0]), float(point[1]))
    if declared not in ("cm", "cm_estimated") or len(pairs) < 3:
        return None
    np = __import__("numpy")
    cm = np.array([(item[0], item[1]) for item in pairs], dtype="float64")
    px = np.array([(item[2], item[3]) for item in pairs], dtype="float64")
    design = np.column_stack([cm, np.ones(len(cm))])
    try:
        solution, _, _, _ = np.linalg.lstsq(design, px, rcond=None)
    except Exception:
        return None
    if not np.all(np.isfinite(solution)):
        return None
    if float(np.max(np.abs(design @ solution - px))) > 0.5:
        return None
    def mapper(point):
        vector = np.array([float(point[0]), float(point[1]), 1.0])
        mapped = vector @ solution
        return float(mapped[0]), float(mapped[1])
    return mapper


def _unit_cm_mapper(unit: Mapping[str, Any]) -> Any:
    """单位级缓存：一次渲染只恢复一次坐标映射。"""

    if "_cm_px_mapper" in unit:
        return unit["_cm_px_mapper"]
    mapper = _cm_to_px_mapper(unit.get("observations") or [])
    unit["_cm_px_mapper"] = mapper
    return mapper


def _trajectory_segments(unit: Mapping[str, Any],
                         upto: float) -> list[list[tuple[int, int]]]:
    """只画保存的完成运动段（motion_x0..y1），逐段时间延长。

    完成段由分析管线按真实连续性保存：缺失帧没有跨缺口的段，这里
    不从位置行重新推导连线，因此缺口自然断开，也不连到 held 点。
    坐标是测量单位（cm/估计 cm）时先按本 run 冻结的成对坐标映射回
    旋转后源像素；映射建立不起来就不画轨迹层。位置被抑制或纯人工
    的单位不画。
    """

    return _segments_upto(_segment_plan(unit), upto)


def _segment_plan(unit: Mapping[str, Any]) -> list[tuple[float, Any]]:
    """一次性预处理：把每个完成段换算成像素端点，并记下它的完成时刻。

    逐帧只按时间游标追加新段，不再每帧重算整段历史。换算是线性的，
    因此一次换算与逐帧换算结果相同；行顺序即观测顺序。
    """

    if unit.get("suppress_positions") or unit.get("manual"):
        return []
    mapper = _unit_cm_mapper(unit)
    if mapper is None:
        return []
    plan: list[tuple[float, Any]] = []
    for row in unit.get("observations") or []:
        moment = _finite(row.get("time_s"))
        if moment is None:
            break
        if row.get("position_valid") is False or row.get("jump"):
            continue
        x0 = _finite(row.get("motion_x0"))
        y0 = _finite(row.get("motion_y0"))
        x1 = _finite(row.get("motion_x1"))
        y1 = _finite(row.get("motion_y1"))
        if None in (x0, y0, x1, y1):
            continue
        start = mapper((x0, y0))
        end = mapper((x1, y1))
        plan.append((float(moment),
                     [(int(round(start[0])), int(round(start[1]))),
                      (int(round(end[0])), int(round(end[1])))]))
    return plan


def _segments_upto(plan: Sequence[tuple[float, Any]],
                   upto: float) -> list[Any]:
    """已完成段：沿用“遇到第一个超出 upto 的段就停”的原有顺序语义。"""

    limit = upto + 1e-9
    segments: list[Any] = []
    for moment, segment in plan:
        if moment > limit:
            break
        segments.append(segment)
    return segments


def _row_for_frame(unit: Mapping[str, Any],
                   frame_index: int) -> Mapping[str, Any] | None:
    """当前源帧自身保存的观测事实；按 frame 索引匹配，不借用相邻帧。"""

    for row in unit.get("observations") or []:
        row_frame = row.get("frame")
        if isinstance(row_frame, int) and not isinstance(row_frame, bool) \
                and row_frame == frame_index:
            return row
    return None


def _body_point(row: Mapping[str, Any] | None) -> tuple[int, int] | None:
    """当前帧自己的身体中心；缺失或无效位置不给点。"""

    if not row:
        return None
    if row.get("position_valid") is False or row.get("jump"):
        return None
    x, y = _finite(row.get("x_px")), _finite(row.get("y_px"))
    if x is None or y is None:
        return None
    return int(round(x)), int(round(y))


def _row_state(row: Mapping[str, Any] | None) -> str:
    if not row:
        return "missing"
    for key in ("state", "zone"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    if row.get("detected") is False:
        return "missing"
    return "detected"


TRAJECTORY_COLOR = (95, 98, 184)


class _TraceLayer:
    """单色轨迹的增量层：只累积覆盖度，每帧只合成“真正画过的像素”。

    逐段直接画在画布上时，抗锯齿把线色与**当前帧**像素混合，所以画好的像素
    不能整块搬用；但同一只动物的完成段同色，顺序叠加后的结果只取决于覆盖度
    的累积乘积 Q=Π(1-aᵢ)：``out = C + (base - C)·Q``。

    于是把 Q 记在“画过的像素”列表里（每段只更新该段包围盒内的像素），每帧
    只对这份像素列表做一次合成：每帧成本与已画像素数有关，不随历史段数增长，
    也不为空白画布区域付代价。栅格化仍由同一个 cv2 抗锯齿调用产生，叠放顺序
    仍按观测顺序。
    """

    __slots__ = ("color", "thickness", "shape", "inverse", "pixels", "slots",
                 "count", "segments")

    def __init__(self, shape: tuple[int, int], color, thickness: int) -> None:
        height, width = shape
        self.shape = (height, width)
        self.color = np.asarray(color, dtype=np.float32).reshape(1, 3)
        self.thickness = int(thickness)
        capacity = 4096
        # Q=Π(1-a)：未被任何段覆盖时为 1（合成后与底色相同）。
        self.inverse = np.ones(capacity, dtype=np.float32)
        self.pixels = np.zeros(capacity, dtype=np.int32)
        self.slots = np.full(height * width, -1, dtype=np.int32)
        self.count = 0
        self.segments = 0

    def _grow(self, needed: int) -> None:
        capacity = len(self.inverse)
        if needed <= capacity:
            return
        while capacity < needed:
            capacity = int(capacity * 1.5) + 1
        grown = np.ones(capacity, dtype=np.float32)
        grown[:self.count] = self.inverse[:self.count]
        pixels = np.zeros(capacity, dtype=np.int32)
        pixels[:self.count] = self.pixels[:self.count]
        self.inverse = grown
        self.pixels = pixels

    def add(self, start, end) -> None:
        import cv2

        pad = self.thickness + 2
        height, width = self.shape
        x0 = max(0, min(start[0], end[0]) - pad)
        x1 = min(width, max(start[0], end[0]) + pad + 1)
        y0 = max(0, min(start[1], end[1]) - pad)
        y1 = min(height, max(start[1], end[1]) + pad + 1)
        if x1 <= x0 or y1 <= y0:
            return
        mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        points = np.array([[start[0] - x0, start[1] - y0],
                           [end[0] - x0, end[1] - y0]], dtype="int32")
        cv2.polylines(mask, [points], False, 255, self.thickness, cv2.LINE_AA)
        rows, cols = np.nonzero(mask)
        if not len(rows):
            return
        coverage = mask[rows, cols].astype(np.float32) * (1.0 / 255.0)
        flat = (rows + y0) * width + (cols + x0)
        slots = self.slots[flat]
        known = slots >= 0
        if known.any():
            self.inverse[slots[known]] *= (1.0 - coverage[known])
        fresh = ~known
        if fresh.any():
            extra = flat[fresh]
            start_slot = self.count
            self._grow(start_slot + len(extra))
            self.slots[extra] = np.arange(start_slot, start_slot + len(extra),
                                          dtype=np.int32)
            self.inverse[start_slot:start_slot + len(extra)] = \
                1.0 - coverage[fresh]
            self.pixels[start_slot:start_slot + len(extra)] = extra
            self.count = start_slot + len(extra)
        self.segments += 1

    def compose(self, canvas) -> None:
        count = self.count
        if not count:
            return
        index = self.pixels[:count]
        flat = canvas.reshape(-1, 3)
        base = flat[index].astype(np.float32)
        base -= self.color
        base *= self.inverse[:count][:, None]
        base += self.color
        np.rint(base, out=base)
        np.clip(base, 0, 255, out=base)
        flat[index] = base.astype(np.uint8)


class _UnitTrack:
    """一只动物的逐帧状态：一次预处理 + 单调游标 + 增量轨迹层。"""

    __slots__ = ("unit_id", "label", "start", "end", "geometry", "rows",
                 "plan", "cursor", "layer")

    def __init__(self, unit: Mapping[str, Any], shape: tuple[int, int],
                 thickness: int) -> None:
        self.unit_id = unit.get("unit_id")
        self.label = _ascii_label(unit.get("unit_name"))
        self.start = float(unit["start"])
        self.end = float(unit["end"])
        self.geometry = unit.get("geometry") or {}
        # 按帧预建观测索引：重复 frame 取首条，不借相邻帧。
        self.rows: dict[int, Mapping[str, Any]] = {}
        for row in unit.get("observations") or []:
            index = row.get("frame")
            if isinstance(index, int) and not isinstance(index, bool):
                self.rows.setdefault(index, row)
        self.plan = _segment_plan(unit)
        self.cursor = 0
        self.layer = (_TraceLayer(shape, TRAJECTORY_COLOR, thickness)
                      if self.plan else None)

    def active(self, start: float, end: float) -> bool:
        middle = (start + end) / 2.0
        return self.start - 1e-9 <= middle < self.end - 1e-9

    def advance(self, upto: float) -> None:
        """把已完成的新段追加进轨迹层；只做一次栅格化。"""

        layer = self.layer
        if layer is None:
            return
        limit = upto + 1e-9
        plan = self.plan
        while self.cursor < len(plan) and plan[self.cursor][0] <= limit:
            start, end = plan[self.cursor][1]
            layer.add(start, end)
            self.cursor += 1

    def state(self, frame_index: int) -> str:
        return _row_state(self.rows.get(frame_index))

    def body(self, frame_index: int):
        return _body_point(self.rows.get(frame_index))


# -- 编码 ---------------------------------------------------------------


def _open_writer(target: Path, width: int, height: int, fps: float | None):
    container = av.open(str(target), "w", format="mp4",
                        options={"movflags": "+faststart"})
    rate = Fraction(str(fps)).limit_denominator(1001) if fps else None
    stream = container.add_stream("libx264", rate=rate)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.time_base = TIMEBASE
    stream.codec_context.time_base = TIMEBASE
    stream.codec_context.options = {
        "crf": "20", "preset": "veryfast", "bf": "0",
        "profile": "main", "level": "4.0",
    }
    return container, stream


def _mux(container: Any, packet: Any, queued: list[int]) -> None:
    """按帧顺序补上该帧的持续时间（无 B 帧时包序与帧序一致）。"""

    if queued:
        packet.duration = queued.pop(0)
    container.mux(packet)


def _ticks(seconds: float) -> int:
    return int(round(seconds * TIMEBASE.denominator))


def verify_video(path: str | Path, expected: Sequence[tuple[int, int]], *,
                 cancel: Callable[[], bool] | None = None,
                 progress: Callable[[float], None] | None = None
                 ) -> list[str]:
    """流式逐帧核对数量、PTS 与 duration（容差 0.002 秒）。

    逐帧解码、逐帧释放：不把整段像素驻留内存，也不抽查代替全查。
    """

    issues: list[str] = []
    total = len(expected)
    count = 0
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            time_base = stream.time_base or TIMEBASE
            scale = float(time_base) / float(TIMEBASE)
            for frame in container.decode(stream):
                if cancel is not None and cancel():
                    issues.append("校验已取消")
                    return issues
                index = count
                count += 1
                if index >= total:
                    # 额外帧仍要计数：数量不符必须如实报出实际值。
                    continue
                want_pts, want_duration = expected[index]
                if frame.pts is None:
                    issues.append(f"帧 {index} 缺少 PTS")
                else:
                    pts_s = float(frame.pts) * scale / TIMEBASE.denominator
                    want_s = want_pts / TIMEBASE.denominator
                    if abs(pts_s - want_s) > DECODE_TOLERANCE_S:
                        issues.append(
                            f"帧 {index} PTS 偏差 {pts_s - want_s:+.4f}s")
                    if frame.duration:
                        duration_s = float(frame.duration) * scale \
                            / TIMEBASE.denominator
                        want_duration_s = want_duration / TIMEBASE.denominator
                        if abs(duration_s - want_duration_s) \
                                > DECODE_TOLERANCE_S:
                            issues.append(
                                f"帧 {index} 时长偏差 "
                                f"{duration_s - want_duration_s:+.4f}s")
                    else:
                        issues.append(f"帧 {index} 缺少时长")
                if progress is not None and total:
                    progress(min(1.0, count / total))
    except Exception as error:  # 解码失败同样是验证失败
        return [f"无法解码输出视频：{error}"]
    if count != total:
        issues.append(f"帧数不一致：期望 {total}，实际 {count}")
    return issues


def write_run_video(path: str | Path, *, job: Mapping[str, Any],
                    units: Sequence[Mapping[str, Any]],
                    events: Sequence[Mapping[str, Any]] = (),
                    cancel: Callable[[], bool] | None = None,
                    progress: Callable[[float, str], None] | None = None
                    ) -> tuple[int, float]:
    """按 run 的源窗口写叠画视频；返回 (帧数, 总时长秒)。

    ``progress`` 收到本阶段内部的 0–1 进度；未知总数时只报阶段与已处理数，
    不伪造百分比。``cancel`` 在定位、编码与校验三个长循环里都检查。
    """

    asset = job.get("asset") or {}
    source_path = asset.get("path")
    if not isinstance(source_path, str) or not source_path:
        raise OSError("叠画视频缺少源文件路径")
    metadata = asset.get("metadata") or {}
    window = job.get("window") or {}
    start = _finite(window.get("start_s")) or 0.0
    end = _finite(window.get("end_s"))
    orientation = int(job.get("orientation", 0) or 0)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    prepared_units = _unit_windows(units)
    prepared_events = []
    for event in events or ():
        if not isinstance(event, Mapping):
            continue
        moment, confirmed = _display_time(event)
        if moment is None:
            continue
        prepared_events.append({
            "display_s": float(moment),
            "confirmed": confirmed,
            "target": event.get("target"),
            "kind": event.get("kind"),
            "unit_id": event.get("unit_id"),
        })
    prepared_events.sort(key=lambda item: item["display_s"])

    def build_source(decode: bool) -> media.VideoSource:
        return media.VideoSource(
            source_path,
            start_s=float(start),
            end_s=float(end) if end is not None else None,
            orientation=orientation,
            fps_override=job.get("fps_override"),
            sample_fps=None,
            source_fps=metadata.get("fps"),
            decode_images=decode,
        )

    window_start = float(start)
    window_end = float(end) if end is not None else math.inf

    def clipped(frame: Any) -> tuple[float, float] | None:
        """只保留与该视频窗有正交叠的曝光；两端各自裁剪。"""

        low = _finite(frame.interval_start_s)
        high = _finite(frame.interval_end_s)
        if low is None or high is None:
            return None
        low = max(float(low), window_start)
        high = min(float(high), window_end)
        if high - low <= 1e-12:
            return None
        return low, high

    pitch = 0.2
    last_report = 0.0

    def report(fraction: float, message: str, *, force: bool = False) -> None:
        nonlocal last_report
        if progress is None:
            return
        now = time.monotonic()
        if not force and now - last_report < pitch:
            return
        last_report = now
        progress(fraction, message)

    # 第一遍只取时间事实（不解码像素），得到每帧的正曝光区间。
    timing: list[dict] = []
    timing_source = build_source(False)
    scanned = 0
    try:
        for frame in timing_source.iter_frames():
            if cancel is not None and cancel():
                raise OSError("叠画视频已取消")
            bounds_pair = clipped(frame)
            if bounds_pair is None:
                continue
            scanned += 1
            timing.append({"index": int(frame.index),
                           "start": bounds_pair[0], "end": bounds_pair[1],
                           "time_s": float(frame.time_s)})
            report(0.05, f"正在定位源帧：已处理 {scanned} 帧",
                   force=scanned == 1)
    finally:
        timing_source.coverage = {"start_s": None, "end_s": None,
                                  "duration_s": 0.0, "frame_count": 0}
    if not timing:
        raise OSError("分析范围内没有可写入的源帧，未生成标注视频")
    first_start = timing[0]["start"]
    bounds = _segment_bounds(timing, prepared_units, prepared_events)
    total_clips = sum(_clip_count(item["start"], item["end"], bounds)
                      for item in timing)

    # 事件按显示时刻排序后预生成文字前缀：每帧只按游标取已显示的那些。
    event_times: list[float] = []
    event_lines: list[str] = []
    for event in prepared_events:
        label = "Confirmed" if event["confirmed"] else "Unconfirmed"
        event_times.append(float(event["display_s"]))
        event_lines.append(f"{label} {event.get('kind') or 'event'} "
                           f"{event.get('target') or ''}".strip())

    container = None
    stream = None
    written = 0
    durations: dict[int, int] = {}
    queued: list[int] = []
    expected: list[tuple[int, int]] = []
    tracks: list[_UnitTrack] | None = None
    thickness = 1
    try:
        decode_source = build_source(True)
        try:
            for frame in decode_source.iter_frames():
                if cancel is not None and cancel():
                    raise OSError("叠画视频已取消")
                canvas_base = _to_bgr(frame.image)
                if canvas_base is None:
                    continue
                bounds_pair = clipped(frame)
                if bounds_pair is None:
                    continue
                if tracks is None:
                    shape = canvas_base.shape[:2]
                    thickness = max(1, int(round(min(shape) / 320)))
                    tracks = [_UnitTrack(unit, shape, thickness)
                              for unit in prepared_units]
                low, high = bounds_pair
                for clip_start, clip_end in _split(low, high, bounds):
                    if cancel is not None and cancel():
                        raise OSError("叠画视频已取消")
                    canvas = canvas_base.copy()
                    lines = [f"t={clock_text(clip_start)}  frame={frame.index}"]
                    for track in tracks:
                        if not track.active(clip_start, clip_end):
                            continue
                        _draw_geometry(canvas, track.geometry, thickness)
                        # 当前源帧只消费自己的保存事实：轨迹按完成段分开
                        # 画且只延长到本片段终点，不跨缺失，也不把未来
                        # 帧的身份画到当前帧。历史段已在增量层里，这里只
                        # 追加新完成的段并合成一次。
                        track.advance(clip_end)
                        if track.layer is not None:
                            track.layer.compose(canvas)
                        point = track.body(int(frame.index))
                        if point is not None:
                            import cv2

                            cv2.circle(canvas, point, 4 + thickness,
                                       (78, 104, 22), -1, cv2.LINE_AA)
                        lines.append(f"{track.label} {track.state(int(frame.index))}")
                    shown = bisect_right(event_times, clip_start + 1e-9)
                    lines.extend(event_lines[:shown])
                    canvas = _draw_label(canvas, lines)
                    canvas = _even_canvas(canvas)
                    if container is None:
                        fps = _finite(metadata.get("fps")) or 25.0
                        container, stream = _open_writer(target,
                                                         canvas.shape[1],
                                                         canvas.shape[0], fps)
                    video_frame = av.VideoFrame.from_ndarray(canvas,
                                                             format="bgr24")
                    pts = _ticks(clip_start - first_start)
                    end_ticks = _ticks(clip_end - first_start)
                    if end_ticks <= pts:
                        end_ticks = pts + 1
                    video_frame.pts = pts
                    video_frame.time_base = TIMEBASE
                    duration_ticks = end_ticks - pts
                    durations[pts] = duration_ticks
                    expected.append((pts, duration_ticks))
                    queued.append(duration_ticks)
                    for packet in stream.encode(video_frame):
                        _mux(container, packet, queued)
                    written += 1
                    report(0.10 + 0.85 * min(1.0, written / max(1, total_clips)),
                           f"正在写叠画视频：{written}/{total_clips} 段",
                           force=written == 1 or written == total_clips)
        finally:
            decode_source.coverage = {"start_s": None, "end_s": None,
                                      "duration_s": 0.0, "frame_count": 0}
        if container is None:
            raise OSError("分析范围内没有可写入的源帧，未生成标注视频")
        tail = expected[-1][1] if expected else 0
        if not queued:
            queued.append(tail)
        for packet in stream.encode(None):
            _mux(container, packet, queued)
        container.close()
        container = None
        stream = None
    except BaseException as error:
        if container is not None:
            try:
                container.close()
            except Exception:
                pass
        try:
            os.unlink(target)
        except OSError:
            pass
        if isinstance(error, OSError):
            raise
        raise OSError(f"标注视频编码失败：{error}") from error

    report(0.95, f"正在校验叠画视频：0/{len(expected)} 帧", force=True)
    issues = verify_video(
        target, expected, cancel=cancel,
        progress=lambda portion: report(
            0.95 + 0.05 * portion,
            f"正在校验叠画视频：{int(round(portion * len(expected)))}"
            f"/{len(expected)} 帧",
            force=portion >= 1.0))
    if issues:
        try:
            os.unlink(target)
        except OSError:
            pass
        raise OSError("叠画视频校验失败：" + "；".join(issues))
    total = float(_ticks(timing[-1]["end"] - first_start)) / TIMEBASE.denominator
    return written, total


def write_manual_y_video(path: str | Path, *, job: Mapping[str, Any],
                         summary: Mapping[str, Any],
                         events: Sequence[Mapping[str, Any]]
                         ) -> tuple[int, float]:
    """纯人工 Y 的兼容入口：同一实现，位置层由单位设置抑制。"""

    unit = {
        "unit_id": summary.get("unit_id"),
        "unit_name": summary.get("unit_name"),
        "window_start_s": summary.get("analysis_window_start_s"),
        "window_end_s": summary.get("analysis_window_end_s"),
        "geometry": {},
        "manual": True,
        "observations": [],
    }
    return write_run_video(path, job=job, units=[unit], events=events)
