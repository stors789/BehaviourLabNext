"""逐单位关联证据与保守缺失：只在真实候选上更新，不保持也不预测。

本模块接第 05 块的候选列表，输出仍是同一个 Detection 契约。体型证据按
unit 独立，缺失帧只保留证据供下一次选择，不把保留位置当成观测。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from . import detection
from .schema import Detection

EVIDENCE_SAMPLES = 9
AREA_RATIO_LOW = 0.25
AREA_RATIO_HIGH = 4.0
GATE_BASE = 3.0
GATE_RATE = 12.0

HEAD_ASSAYS = ("nor", "barnes")
HEAD_CORE_RATIO = 0.45
HEAD_MIN_RADIUS = 2.0
HEAD_CORE_MIN_PIXELS = 10
HEAD_AXIS_RATIO = 1.25
HEAD_TAIL_LONG_RATIO = 1.5
HEAD_TAIL_GAP_RATIO = 0.6
HEAD_THIN_LIMIT = 1.5
HEAD_EXTREME_PIXELS = 1.0

TRACKER_COUNT = 8
UNIT_COUNT = 16


def signature_of(geometry: Mapping[str, Any] | None,
                 settings: Mapping[str, Any] | None,
                 dark_box_visible: bool = False) -> str:
    """配置签名：unit 几何、完整合并检测参数与暗箱可见性。"""

    payload = {
        "geometry": geometry or {},
        "detection": settings or {},
        "dark_box_visible": bool(dark_box_visible),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), default=str)


def spatial_gate(span: float, elapsed_s: float) -> float:
    """跳跃半径 L = S×(3+12Δt)；只由已接受体型与真实时间差决定。"""

    return float(span) * (GATE_BASE + GATE_RATE * max(0.0, float(elapsed_s)))


def candidate_span(candidate: Mapping[str, Any]) -> float:
    """footprint span=max(1, min(√(w²+h²), 3d))；对方向不敏感，限制细尾。"""

    area = max(0.0, float(candidate.get("area") or 0.0))
    diameter = math.sqrt(4.0 * area / math.pi) if area > 0 else 0.0
    width = height = 0.0
    contour = candidate.get("contour")
    if contour is not None and len(contour) >= 3:
        points = np.array(contour, dtype=np.float32).reshape(-1, 1, 2)
        _, (w, h), _ = cv2.minAreaRect(points)
        width, height = float(w), float(h)
    else:
        bbox = candidate.get("bbox") or []
        if len(bbox) == 4:
            width, height = float(bbox[2]), float(bbox[3])
    diagonal = math.hypot(width, height)
    return max(1.0, min(diagonal, 3.0 * diameter))


def _median(values: Sequence[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


@dataclass
class Evidence:
    """一个 unit 在一条视频上的关联证据；缺失不写入，重置即清空。"""

    signature: str | None = None
    frame_index: int | None = None
    time_s: float | None = None
    center: tuple[float, float] | None = None
    observed_s: float | None = None
    reference_areas: list[float] = field(default_factory=list)
    areas: list[float] = field(default_factory=list)
    spans: list[float] = field(default_factory=list)

    def reset(self, signature: str | None = None) -> None:
        self.signature = signature
        self.frame_index = None
        self.time_s = None
        self.center = None
        self.observed_s = None
        self.reference_areas = []
        self.areas = []
        self.spans = []

    def accepts(self, area: float) -> bool:
        """候选面积必须同时贴合滚动中位数与最初固定的参考中位数。"""

        if not self.areas or not self.reference_areas:
            return True
        local = _median(self.areas)
        reference = _median(self.reference_areas)
        if local <= 0 or reference <= 0:
            return True
        return (AREA_RATIO_LOW <= area / local <= AREA_RATIO_HIGH
                and AREA_RATIO_LOW <= area / reference <= AREA_RATIO_HIGH)

    def remember(self, area: float, span: float, time_s: float) -> None:
        """只由被接受的真实候选更新；同一源时刻不重复取样。"""

        repeated = (self.observed_s is not None
                    and abs(float(time_s) - self.observed_s) <= 1e-9)
        if not repeated:
            if len(self.reference_areas) < EVIDENCE_SAMPLES:
                self.reference_areas.append(float(area))
            self.areas.append(float(area))
            self.spans.append(float(span))
            del self.areas[:-EVIDENCE_SAMPLES]
            del self.spans[:-EVIDENCE_SAMPLES]


def _distance(point: tuple[float, float], center: Sequence[float]) -> float:
    return math.hypot(point[0] - float(center[0]), point[1] - float(center[1]))


def _empty(center: Sequence[float]) -> tuple[float, float]:
    return (float(center[0]), float(center[1]))


def _missing(reason: str | None,
             candidates: Sequence[Mapping[str, Any]]) -> Detection:
    """缺失：不附带保持位置、掩膜或头点。"""

    return Detection(center=None, quality=0.0, contour=[], mask=None,
                     missing_reason=reason, candidates=list(candidates))


def _head_unknown() -> dict:
    return {"nose": None, "tail_base": None, "head_angle": None,
            "head_quality": 0.0}


def head_evidence(image: Any, region: np.ndarray,
                  candidate: Mapping[str, Any],
                  settings: Mapping[str, Any]) -> dict:
    """厚身体主轴 + 细长尾非对称；证据不足即未知，不沿运动方向补鼻点。"""

    candidate_mask = candidate.get("mask")
    if candidate_mask is None or image is None or region is None:
        return _head_unknown()

    gray = detection.to_gray(image)
    threshold = float(settings.get("body_gray_threshold", 95.0))
    if settings.get("bg_type", "light") == "light":
        body = gray <= threshold
    else:
        body = gray > threshold
    body = np.logical_and(body, region > 0)
    if not bool(body.any()):
        return _head_unknown()

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        body.astype(np.uint8), connectivity=8)
    overlap = np.bincount(labels[candidate_mask > 0], minlength=count)
    overlap[0] = 0
    label = int(np.argmax(overlap))
    if label == 0 or overlap[label] == 0:
        return _head_unknown()
    if overlap[label] < 0.5 * float(stats[label, cv2.CC_STAT_AREA]):
        return _head_unknown()

    component = (labels == label).astype(np.uint8)
    distances = cv2.distanceTransform(component, cv2.DIST_L2, 5)
    radius = float(distances.max())
    if radius < HEAD_MIN_RADIUS:
        return _head_unknown()

    core = distances >= HEAD_CORE_RATIO * radius
    core_points = np.argwhere(core)
    if core_points.shape[0] < HEAD_CORE_MIN_PIXELS:
        return _head_unknown()

    core_xy = core_points[:, ::-1].astype(np.float64)      # (x, y)
    center = core_xy.mean(axis=0)
    covariance = np.cov(core_xy.T, bias=True)
    values, vectors = np.linalg.eigh(covariance)
    if values[0] <= 0 or values[1] / values[0] < HEAD_AXIS_RATIO:
        return _head_unknown()
    axis = vectors[:, 1]

    core_rel = (core_xy - center) @ axis
    body_xy = np.argwhere(component)[:, ::-1].astype(np.float64)
    body_rel = (body_xy - center) @ axis
    low, high = float(core_rel.min()), float(core_rel.max())
    low_full, high_full = float(body_rel.min()), float(body_rel.max())
    extend_low = low - low_full
    extend_high = high_full - high
    long_extent = max(extend_low, extend_high)
    short_extent = min(extend_low, extend_high)
    if (long_extent < radius
            or long_extent < HEAD_TAIL_LONG_RATIO * short_extent
            or long_extent - short_extent < HEAD_TAIL_GAP_RATIO * radius):
        return _head_unknown()

    tail_low = extend_low >= extend_high
    if tail_low:
        outside = body_rel < low
        tail_end, nose_end = low, high_full
    else:
        outside = body_rel > high
        tail_end, nose_end = high, low_full
    # 细尾而非宽凸出：长端 core 外像素密度不超过 1.5R。
    if float(np.count_nonzero(outside)) / long_extent > HEAD_THIN_LIMIT * radius:
        return _head_unknown()

    if tail_low:
        nose_pixels = body_rel >= nose_end - HEAD_EXTREME_PIXELS
    else:
        nose_pixels = body_rel <= nose_end + HEAD_EXTREME_PIXELS
    if not bool(np.any(nose_pixels)):
        return _head_unknown()

    tail_base = center + axis * tail_end
    nose = body_xy[nose_pixels].mean(axis=0)
    angle = math.degrees(math.atan2(float(nose[1] - tail_base[1]),
                                    float(nose[0] - tail_base[0])))
    quality = min(1.0, 0.5 + 0.5 * (
        (long_extent - short_extent) / long_extent))
    return {
        "nose": [float(nose[0]), float(nose[1])],
        "tail_base": [float(tail_base[0]), float(tail_base[1])],
        "head_angle": angle,
        "head_quality": max(0.0, quality),
    }


def _detection_from(candidate: Mapping[str, Any], head: dict,
                    candidates: Sequence[Mapping[str, Any]]) -> Detection:
    return Detection(
        center=_empty(candidate["center"]),
        quality=float(candidate.get("quality") or 0.0),
        contour=candidate.get("contour") or [],
        mask=candidate.get("mask"),
        area=float(candidate.get("area") or 0.0),
        bbox=candidate.get("bbox"),
        nose=head["nose"], tail_base=head["tail_base"],
        head_angle=head["head_angle"], head_quality=head["head_quality"],
        candidates=list(candidates),
    )


class ObservationTracker:
    """一条视频内的关联证据；每个 unit 独立，缺失只保留证据。"""

    def __init__(self, key: tuple | None = None) -> None:
        self.key = key
        self._evidence: dict[Any, Evidence] = {}

    def state(self, unit_key: Any) -> Evidence:
        evidence = self._evidence.get(unit_key)
        if evidence is None:
            evidence = Evidence()
            self._evidence[unit_key] = evidence
            while len(self._evidence) > UNIT_COUNT:
                self._evidence.pop(next(iter(self._evidence)))
        return evidence

    def observe(self, unit_key: Any, *,
                candidates: Sequence[Mapping[str, Any]],
                reason: str | None,
                frame_index: int,
                time_s: float,
                signature: str,
                image: Any = None,
                region: np.ndarray | None = None,
                settings: Mapping[str, Any] | None = None,
                assay: str = "",
                previous: Sequence[float] | None = None) -> Detection:
        """选一个真实候选或返回缺失；缺失不会把保留状态写成观测。

        unit_key 为不透明单位 id；内部临时调用可传对象本身，按身份隔离。
        """

        evidence = self.state(unit_key)
        regressed = (
            (evidence.frame_index is not None and frame_index < evidence.frame_index)
            or (evidence.time_s is not None and time_s < evidence.time_s - 1e-9)
        )
        if evidence.signature != signature or regressed:
            evidence.reset(signature)
        evidence.frame_index = frame_index
        evidence.time_s = float(time_s)

        pool = list(candidates)
        if not pool:
            return _missing(reason or "no_candidate", [])

        compatible = [item for item in pool
                      if evidence.accepts(float(item["area"]))]
        if not compatible:
            return _missing("inconsistent_candidate_size", pool)

        selected = self._select(evidence, compatible, time_s, previous)
        if selected is None:
            return _missing("no_nearby_candidate", pool)

        span = candidate_span(selected)
        evidence.remember(float(selected["area"]), span, float(time_s))
        evidence.center = _empty(selected["center"])
        evidence.observed_s = float(time_s)

        head = _head_unknown()
        if assay in HEAD_ASSAYS and settings is not None:
            head = head_evidence(image, region, selected, settings)
        return _detection_from(selected, head, pool)

    @staticmethod
    def _select(evidence: Evidence, candidates: Sequence[Mapping[str, Any]],
                time_s: float,
                previous: Sequence[float] | None) -> Mapping[str, Any] | None:
        """按距离/尺寸打分；无参考位置时取面积最大的合格候选。"""

        reference = evidence.center
        if reference is None and previous is not None:
            reference = _empty(previous)
        if reference is None:
            return candidates[0]

        spans = [candidate_span(item) for item in candidates]
        span = _median(evidence.spans) if evidence.spans else min(spans)
        elapsed = (max(0.0, float(time_s) - evidence.observed_s)
                   if evidence.observed_s is not None else 0.0)
        limit = spatial_gate(span, elapsed)
        local = _median(evidence.areas) if evidence.areas else None

        best = None
        best_score = float("inf")
        for item in candidates:
            distance = _distance(_empty(item["center"]), reference)
            if distance > limit:
                continue
            score = distance / span
            if local:
                score += abs(math.log(float(item["area"]) / local))
            if score < best_score - 1e-12:
                best, best_score = item, score
        return best


_TRACKERS: dict[tuple, ObservationTracker] = {}


def tracker_for(key: tuple) -> ObservationTracker:
    """不同视频各用自己的关联证据。"""

    tracker = _TRACKERS.get(key)
    if tracker is None:
        tracker = ObservationTracker(key)
        _TRACKERS[key] = tracker
        if len(_TRACKERS) > TRACKER_COUNT:
            _TRACKERS.pop(next(iter(_TRACKERS)))
    return tracker


def track(key: tuple, unit_key: Any, *,
          image: Any, region: np.ndarray,
          settings: Mapping[str, Any],
          signature: str, assay: str, frame_index: int, time_s: float,
          foreground: np.ndarray,
          previous: Sequence[float] | None = None) -> Detection:
    """分割候选 → 关联；不接受保持位置，也不产生中间轨迹。"""

    candidates, reason = detection.find_candidates(
        foreground, region, settings)
    return tracker_for(key).observe(
        unit_key, candidates=candidates, reason=reason,
        frame_index=frame_index, time_s=time_s, signature=signature,
        image=image, region=region, settings=settings, assay=assay,
        previous=previous,
    )
