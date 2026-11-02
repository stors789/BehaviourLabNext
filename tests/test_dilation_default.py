"""膨胀次数默认：公共与 FST 为 1、TST 仍为 3，且默认真的进到候选提取。"""

from __future__ import annotations

import cv2
import numpy as np

from behaviourlab import detection, engine, experiments
from behaviourlab.web import descriptors

BASE = {"min_area": 20.0, "max_area": 10000.0, "erosion_iter": 0}


def catalog_entries() -> dict:
    return {entry["key"]: entry for entry in experiments.catalog()}


def test_shared_and_fst_defaults_are_one():
    assert detection.DETECTION_DEFAULTS["dilation_iter"] == 1
    assert detection.common_detection_defaults()["dilation_iter"] == 1

    entries = catalog_entries()
    assert entries["fst"]["detection_defaults"]["dilation_iter"] == 1
    assert entries["tst"]["detection_defaults"]["dilation_iter"] == 3
    for key in ("oft", "nor", "epm", "ymaze", "ldb", "barnes", "ezm", "mwm"):
        assert entries[key]["detection_defaults"]["dilation_iter"] == 1, key


def test_other_detection_overrides_untouched():
    entries = catalog_entries()
    # 核大小、腐蚀、阈值与面积等实验专用覆盖都不随本次改动变化。
    assert entries["fst"]["detection_defaults"]["dilation_kernel"] == 5
    assert entries["fst"]["detection_defaults"]["erosion_iter"] == 1
    assert entries["fst"]["detection_defaults"]["min_area"] == 80.0
    assert entries["tst"]["detection_defaults"]["erosion_iter"] == 2
    assert entries["tst"]["detection_defaults"]["dilation_kernel"] == 5
    assert entries["tst"]["detection_defaults"]["min_area"] == 100.0
    shared = detection.common_detection_defaults()
    assert shared["erosion_kernel"] == 3
    assert shared["erosion_iter"] == 1
    assert shared["dilation_kernel"] == 5
    assert shared["gray_threshold"] == 127.0
    assert shared["min_area"] == 120.0


def test_new_draft_and_effective_resolution_use_factory_default():
    assert engine.snapshot_draft("oft")["detection"]["dilation_iter"] == 1
    assert engine.snapshot_draft("fst")["detection"]["dilation_iter"] == 1
    assert engine.snapshot_draft("tst")["detection"]["dilation_iter"] == 3
    # 预览与分析共用的有效解析，与设置页描述符同源。
    assert engine.resolve_detection("oft")["dilation_iter"] == 1
    assert engine.resolve_detection("fst")["dilation_iter"] == 1
    assert engine.resolve_detection("tst")["dilation_iter"] == 3
    assert descriptors.detection_defaults("oft")["dilation_iter"] == 1
    assert descriptors.detection_defaults("tst")["dilation_iter"] == 3
    assert descriptors.detection_defaults()["dilation_iter"] == 1


def test_explicit_values_are_preserved():
    for value in (0, 2):
        draft = {"detection": {"dilation_iter": value}}
        assert engine.resolve_detection("oft", draft=draft)["dilation_iter"] == value
        snapshot = engine.snapshot_draft("oft", detection={"dilation_iter": value})
        assert snapshot["detection"]["dilation_iter"] == value
        unit = {"detection": {"dilation_iter": value}}
        assert engine.resolve_detection("tst", unit=unit)["dilation_iter"] == value


def test_saved_setting_beats_factory_default():
    # 用户已保存 4：新草稿按既有覆盖优先级取 4，不回落到工厂 1。
    settings = {"detection": {"dilation_iter": 4}}
    assert engine.snapshot_draft("oft", settings=settings)["detection"][
        "dilation_iter"] == 4
    assert engine.resolve_detection("oft", settings=settings)[
        "dilation_iter"] == 4
    entry = [item for item in experiments.catalog() if item["key"] == "oft"][0]
    assert entry["detection_defaults"]["dilation_iter"] == 1


def test_existing_draft_snapshot_is_not_migrated(store):
    # 既有草稿里存的是旧值 2：读取时照旧取 2，不按新工厂默认重写。
    draft = store.create_draft("oft", detection={"dilation_iter": 2})
    stored = store.get_draft(draft["id"])
    assert stored["detection"]["dilation_iter"] == 2
    assert engine.resolve_detection("oft", draft=stored)["dilation_iter"] == 2
    assert engine.resolve_detection("oft")["dilation_iter"] == 1

    # TST 的三次仍是三次：显式 3 与默认为 3 都不被共享默认覆盖。
    tst = store.create_draft("tst")
    assert engine.resolve_detection("tst", draft=store.get_draft(tst["id"]))[
        "dilation_iter"] == 3


def square_frame() -> np.ndarray:
    image = np.full((120, 160), 255, dtype=np.uint8)
    cv2.rectangle(image, (40, 40), (60, 60), 0, -1)
    return image


def test_missing_key_defaults_to_single_dilation():
    mask = np.zeros((60, 60), dtype=np.uint8)
    mask[30, 30] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # 完全不写 dilation_iter：取默认值必须等于膨胀一次。
    resolved = detection.apply_morphology(
        mask, {"erosion_iter": 0, "erosion_kernel": 3, "dilation_kernel": 5})
    assert np.array_equal(resolved, cv2.dilate(mask, kernel, iterations=1))
    assert not np.array_equal(resolved, cv2.dilate(mask, kernel, iterations=2))


def test_default_reaches_candidate_extraction():
    image = square_frame()
    region = np.full((120, 160), 255, dtype=np.uint8)

    one = dict(BASE, dilation_iter=1)
    two = dict(BASE, dilation_iter=2)
    once, _ = detection.find_candidates(
        detection.threshold_foreground(image, one), region, one)
    twice, _ = detection.find_candidates(
        detection.threshold_foreground(image, two), region, two)
    assert once and twice
    assert once[0]["area"] < twice[0]["area"]

    # 有效解析结果（仅覆盖本次关心的面积）给出与显式 1 相同的候选。
    resolved = engine.resolve_detection(
        "oft", draft={"detection": dict(BASE)})
    assert resolved["dilation_iter"] == 1
    effective, _ = detection.find_candidates(
        detection.threshold_foreground(image, resolved), region, resolved)
    assert effective[0]["area"] == once[0]["area"]
    assert effective[0]["bbox"] == once[0]["bbox"]
