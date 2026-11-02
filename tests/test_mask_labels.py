"""U04：OFT/LDB 的可选遮罩改称“检测范围”，圈内允许、圈外排除不变。"""

from __future__ import annotations

import cv2
import numpy as np

from behaviourlab import detection, experiments
from behaviourlab.experiments import ldb, oft

MASK = {"type": "polygon", "points": [[40, 40], [160, 40], [160, 160], [40, 160]]}
HELP = "仅在圈内检测；圈外不参与检测"


def entries() -> dict:
    return {entry["key"]: entry for entry in experiments.catalog()}


def mask_step(assay: str) -> dict:
    return [step for step in entries()[assay]["geometry"]
            if step["key"] == "mask"][0]


def reference_region(width: int = 200, height: int = 200) -> np.ndarray:
    canvas = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(canvas, [np.array(MASK["points"], dtype=np.int32)], 255)
    return canvas


def frame_with(dot, size: int = 200) -> np.ndarray:
    image = np.full((size, size, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, dot, (dot[0] + 20, dot[1] + 20), (0, 0, 0), -1)
    return image


def settings() -> dict:
    values = dict(detection.common_detection_defaults())
    values.update({"min_area": 50.0, "max_area": 10000.0})
    return values


# -- 文案 ---------------------------------------------------------------
def test_catalog_copy_changed_only_for_oft_and_ldb():
    for assay in ("oft", "ldb"):
        step = mask_step(assay)
        assert step["label"] == "检测范围"
        assert step["help"] == HELP
        assert step["required"] is False, "仍是可选步骤"
        assert step["type"] == "polygon", "字段类型与形态不变"
        assert step["min_points"] == 3

    for assay in ("tst", "fst", "mwm", "nor", "epm", "ymaze", "barnes", "ezm"):
        step = mask_step(assay)
        assert step["label"] == "检测遮罩", assay
        assert step.get("help") is None, assay


def test_catalog_api_and_workspace_use_the_same_text(client):
    response = client.get("/api/catalog")
    assert response.status_code == 200
    payload = {entry["key"]: entry for entry in response.get_json()["data"]["assays"]}
    for assay in ("oft", "ldb"):
        step = [item for item in payload[assay]["geometry"]
                if item["key"] == "mask"][0]
        assert step["label"] == "检测范围"
        assert step["help"] == HELP

    source = (__import__("pathlib").Path(__file__).resolve().parents[1]
              / "behaviourlab/web/static/workspace.js").read_text(encoding="utf-8")
    # 步骤行与工具标题仍读目录 label；说明只在当前步骤出现。
    assert "label.textContent = step.label;" in source
    assert '"绘制：" + step.label' in source
    assert "stepHelp.help" in source
    assert "step-help" in source


# -- 算法事实 -----------------------------------------------------------
def test_oft_mask_keeps_inside_and_excludes_outside():
    region, source = detection.geometry_region({"mask": MASK}, "oft", {}, 200, 200)
    assert source == "mask"
    assert np.array_equal(region, reference_region()), "圈内允许、圈外排除逐值不变"

    values = settings()
    inside = detection.detect_frame(frame_with((60, 60)), values, region)
    assert inside.center is not None
    assert 40 <= inside.center[0] <= 160 and 40 <= inside.center[1] <= 160

    outside = detection.detect_frame(frame_with((5, 5)), values, region)
    assert outside.center is None
    assert outside.candidates == []


def test_ldb_mask_still_wins_over_chambers():
    geometry = {
        "light": {"type": "quad", "points": [[0, 0], [90, 0], [90, 200], [0, 200]]},
        "dark": {"type": "quad",
                 "points": [[110, 0], [200, 0], [200, 200], [110, 200]]},
        "mask": MASK,
    }
    region, source = detection.geometry_region(
        geometry, "ldb", {"dark_box_visible": True}, 200, 200)
    assert source == "mask"
    assert np.array_equal(region, reference_region())

    values = settings()
    assert detection.detect_frame(frame_with((60, 60)), values, region).center
    assert detection.detect_frame(frame_with((10, 10)), values, region).center is None
    assert detection.detect_frame(frame_with((170, 60)), values,
                                  region).center is None


def test_validation_and_required_state_unchanged():
    # 遮罩仍是可选：缺遮罩不报错；非法类型仍按原字段报错。
    arena = {"type": "quad", "points": [[0, 0], [10, 0], [10, 10], [0, 10]]}
    assert oft.validate_unit(dict(oft.DEFAULTS), {"arena": arena},
                             complete=True) == []
    ldb_geometry = {"light": arena, "dark": arena}
    values = dict(ldb.DEFAULTS)
    assert ldb.validate_unit(values, ldb_geometry) == []
    issues = ldb.validate_unit(values, dict(ldb_geometry, mask="bad"))
    assert issues and issues[0]["field"] == "geometry.mask"
    assert ldb.step_drawn({"mask": MASK}, "mask") is True
    assert ldb.step_drawn({}, "mask") is False


# -- 旧草稿 -------------------------------------------------------------
def test_old_draft_geometry_reads_back_unchanged(client, store, tmp_path):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"placeholder")
    asset = store.create_asset(
        name="clip.mp4", path=str(source), project_id=None,
        metadata={"width": 200, "height": 200, "fps": 25.0,
                  "frame_count": 50, "duration_s": 2.0})
    geometry = {
        "arena": {"type": "quad", "points": [[5, 5], [195, 5], [195, 195], [5, 195]]},
        "mask": MASK,
    }
    draft = store.create_draft("oft", videos=[{
        "asset_id": asset["id"],
        "window": {"start_s": 0, "end_s": None},
        "orientation": 0, "fps_override": None, "sample_fps": None,
        "units": [{"id": "a" * 32, "name": "小鼠 1", "window": None,
                   "geometry": geometry, "options": {}, "detection": {}}],
    }])

    stored = client.get(f"/api/drafts/{draft['id']}").get_json()["data"]["draft"]
    unit = stored["videos"][0]["units"][0]
    assert unit["geometry"] == geometry, "旧草稿几何原样读回，无迁移"
    assert unit["geometry"]["mask"] == MASK
    region, source = detection.geometry_region(
        unit["geometry"], "oft", unit.get("options") or {}, 200, 200)
    assert source == "mask"
    assert np.array_equal(region, reference_region())
