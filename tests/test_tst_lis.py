"""第 10 块 lis 自检：配准残差夹具、10 条证据评分与接口管线冒烟。

固定夹具来自任务书：96×96 白底 255、黑椭圆半轴 (9,18)；A 中心 (40,48)、
B/C 中心 (48,48)、C 在 B 上加黑矩形 (56,44)–(62,52)。残差比较对齐后图像。
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from behaviourlab import activity
from behaviourlab.experiments import tst

ARENA = {"type": "polygon",
         "points": [[5, 5], [120, 5], [120, 120], [5, 120]]}
BODY_LINE = {"type": "line", "points": [[10, 10], [70, 10]]}


def _frame_with_ellipse(center: tuple[int, int],
                        extra_rect: bool = False) -> tuple[np.ndarray, np.ndarray]:
    image = np.full((96, 96, 3), 255, dtype=np.uint8)
    cv2.ellipse(image, center, (9, 18), 0, 0, 360, (0, 0, 0), -1)
    if extra_rect:
        cv2.rectangle(image, (56, 44), (62, 52), (0, 0, 0), -1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = ((gray < 127).astype(np.uint8)) * 255
    return gray, mask


def test_lis_registration_residual_fixture():
    gray_a, mask_a = _frame_with_ellipse((40, 48))
    gray_b, mask_b = _frame_with_ellipse((48, 48))
    gray_c, mask_c = _frame_with_ellipse((48, 48), extra_rect=True)

    def residuals(cur_gray, cur_mask, rect=False):
        # A→B/C 只平移 8px：提供中心 A=(40,48)、B/C=(48,48)。
        aligned_gray, aligned_mask, delta = activity.lis_register(
            gray_a, cur_gray, mask_a, cur_mask, (40, 48), (48, 48),
            activity.lis_principal_angle(mask_a),
            activity.lis_principal_angle(cur_mask), rotate_align=True)
        residual = activity.lis_residual(aligned_gray, aligned_mask,
                                         cur_gray, cur_mask)
        core = activity.lis_core_peripheral(cur_mask,
                                            activity.lis_body_length(
                                                cur_mask, (39, 30, 18, 36)),
                                            residual["diff"])
        return residual, core, delta

    residual_ab, core_ab, delta_ab = residuals(gray_b, mask_b)
    assert abs(delta_ab) <= 30.0
    # A→B 仅平移，对齐残差应小于 .001。
    assert residual_ab["r"] < 0.001
    assert core_ab["peripheral_mean"] < 0.001

    residual_ac, core_ac, _ = residuals(gray_c, mask_c)
    # A→C 有附加形状：残差大于 .02，且周边变化高于 A→B。
    assert residual_ac["r"] > 0.02
    assert core_ac["peripheral_mean"] > core_ab["peripheral_mean"]


def lis_rows() -> list[dict]:
    rows = []
    for index in range(10):
        start = 0.1 * index
        rows.append({
            "unit_id": "f" * 32, "frame": index, "time_s": start,
            "interval_start_s": start, "interval_end_s": start + 0.1,
            "dt_s": 0.1, "in_window": True, "feature_ready": True,
            "lis_r": index / 100, "lis_peripheral_d": index / 100,
            "lis_z": index / 100, "lis_h": index / 100,
            "lis_f": 10.0 * index, "lis_s": 5.0 * index,
            "lis_core_ratio": 1.0, "lis_g": 0.0, "lis_b": 0.0,
            "lis_body_length_px": 36.0, "lis_aspect_ratio": 2.0,
            "lis_segmentation_bad": False,
            "flow_p90_mm_s": 10.0 * index, "speed_nominal_mm_s": 5.0 * index,
        })
    return rows


def test_lis_score_fixture_quantiles_and_motion_score():
    rows = lis_rows()
    info = tst.lis_score_rows(rows)
    r = info["thresholds"]["r"]
    assert r["q50"] == pytest.approx(0.045)
    assert r["q80"] == pytest.approx(0.072)
    assert r["q25"] == pytest.approx(0.0225)
    assert r["q90"] == pytest.approx(0.081)
    # 最大证据行：六个归一量都截为 1，M=1（不是按最大值除）。
    last = rows[-1]
    assert last["lis_motion_score"] == pytest.approx(1.0)
    for key in tst.LIS_SCORE_FEATURES:
        assert last[f"lis_score_{key}"] == pytest.approx(1.0)
    # p = max(.5×[1+22.5/25], .375×[1+40.5/40]) = .95 → 评分分支。
    assert last["lis_motion_prior"] == pytest.approx(0.95)
    assert last["lis_score_mode"] == "score"


def test_lis_pipeline_smoke(client, store, tmp_path, monkeypatch):
    from behaviourlab import media
    from behaviourlab.schema import Frame

    class MovingSource:
        """两帧平移运动的解码头，接口与 VideoSource 相同。"""

        def __init__(self, path, *, start_s=0.0, end_s=None, **kwargs):
            self._frames = []
            count = 10
            for index in range(count):
                moment = start_s + index * 0.1
                image = np.full((96, 96, 3), 255, dtype=np.uint8)
                cv2.ellipse(image, (40 + 4 * index, 48), (9, 18), 0, 0, 360,
                            (0, 0, 0), -1)
                self._frames.append(Frame(
                    image=image, index=index, time_s=moment, dt_s=0.1,
                    elapsed_s=moment, analysis_fps=10.0,
                    interval_start_s=moment, interval_end_s=moment + 0.1))

        def iter_frames(self):
            return iter(self._frames)

    source = tmp_path / "lis.mp4"
    source.write_bytes(b"placeholder")
    asset = store.create_asset(
        name="lis.mp4", path=str(source), project_id=None,
        metadata={"width": 96, "height": 96, "duration_s": 1.0,
                  "fps": 10.0, "frame_count": 10})
    options = dict(tst.DEFAULTS)
    options["mode"] = "lis"
    draft = store.create_draft("tst", videos=[{
        "asset_id": asset["id"],
        "window": {"start_s": 0.0, "end_s": 1.0},
        "orientation": 0, "fps_override": None, "sample_fps": None,
        "units": [{
            "id": "f" * 32, "name": "lis 鼠", "window": None,
            "geometry": {"arena": ARENA, "body_line": BODY_LINE},
            "options": options, "detection": {}, "manual": None,
        }],
    }])
    monkeypatch.setattr(media, "VideoSource", MovingSource)
    monkeypatch.setattr(
        media, "probe_video",
        lambda path: {"width": 96, "height": 96, "duration_s": 1.0,
                      "fps": 10.0, "frame_count": 10})

    response = client.post(f"/api/drafts/{draft['id']}/analyze")
    assert response.status_code == 202, response.get_json()
    payload = response.get_json()["data"]
    job = payload["jobs"][0]
    assert job["status"] == "completed"
    manifest = job["manifest"]
    unit_meta = manifest["metadata"]["units"]["f" * 32]
    assert unit_meta["mode"] == "lis"
    assert set(unit_meta["thresholds"]) == {"r", "p", "z", "h", "f", "s",
                                            "g", "b"}

    results = client.get(
        f"/runs/{payload['run_id']}/{job['job_id']}/results.json")
    assert results.status_code == 200
    data = results.get_json()
    summary = data["summary"][0]
    assert summary["mode"] == "lis"
    assert summary["swing_time_s"] == 0.0
    assert summary["latency_to_immobility_s"] is None
    # 未知 + 挣扎 + 非挣扎 = 有效曝光。
    assert summary["total_time_s"] == pytest.approx(
        summary["immobility_time_s"] + summary["struggling_time_s"]
        + summary["uncertain_time_s"])
    assert set(summary["thresholds"]) == {"r", "p", "z", "h", "f", "s",
                                          "g", "b"}

    rows = data["observations"]
    assert rows, "lis 管线应产出观测行"
    for row in rows:
        if not row.get("feature_ready"):
            continue
        assert row["lis_motion_score"] is not None
        assert row["lis_confidence"] in ("high", "medium", "low")
        assert row["lis_reason"]
        assert row["state"] in ("struggling", "immobile")
        assert row["state_group"] == row["state"]
        assert row["is_swinging"] is False
        assert row["is_struggling"] == (row["state"] == "struggling")

    # 结果页：lis 阈值按分位数块显示，不访问 classic 字段。
    page = client.get(f"/results/{payload['run_id']}").get_data(as_text=True)
    assert "尾悬实验 / TST" in page
    assert "分位数估计" in page
    assert "微动残差（先测量后评分）" in page
