"""自动 Y 迷宫中心点入臂规则：固定序列、边界、隔离与兼容回归。

夹具复用 test_ymaze 的真实 analyze 驱动（capture 替身 + 真实
process_unit/_track* 核心），保证行为预览与正式分析同一判定路径。
"""

from __future__ import annotations

import pytest

from behaviourlab.experiments import ymaze
from behaviourlab.schema import AnalysisContext
from test_ymaze import (ARM_A, ARM_B, ARM_C, CENTER, FIXTURE, JOB_ID, RUN_ID,
                        UNIT_ID, TOL, geometry, options, rows, unit)

UNKNOWN_POINT = (100, 120)          # 三臂与中心三角形之外


def without_rule(values: dict) -> dict:
    """模拟旧草稿/旧冻结任务：选项里没有 arm_entry_rule 字段。"""

    return {key: value for key, value in values.items()
            if key != "arm_entry_rule"}


def targets(result) -> list[str]:
    return [event["target"] for event in result.events]


def times(result) -> list[float]:
    return [event["time_s"] for event in result.events]


# -- 一、原规则不受新字段影响 -------------------------------------------

def test_missing_rule_field_keeps_depth_semantics(monkeypatch):
    """旧草稿/旧冻结任务没有新字段：行为与显式 depth 完全一致。"""

    result = run_y(FIXTURE, values=without_rule(options()),
                   monkeypatch=monkeypatch)
    explicit = run_y(FIXTURE, values=options(arm_entry_rule="depth"),
                     monkeypatch=monkeypatch)
    assert targets(result) == ["A", "A", "B", "C"]
    assert [event["confirmed_s"] for event in result.events] == [
        event["confirmed_s"] for event in explicit.events]
    assert times(result) == times(explicit)
    assert result.summary[0]["arm_entry_rule"] == "depth", \
        "缺字段记录实际生效规则，不凭空改变含义"
    # 与既有 test_ymaze 固定断言一致（深度规则未被扰动）。
    assert result.summary[0]["total_arm_entries"] == 4


def test_explicit_depth_matches_documented_rule(monkeypatch):
    result = run_y([ARM_A] * 3 + [CENTER] * 3 + [ARM_B] * 3,
                   values=options(arm_entry_rule="depth"),
                   monkeypatch=monkeypatch)
    assert targets(result) == ["A", "B"]
    # 深度规则：确认发生在起点后达到 entry_confirm_s 的首个有效观测
    # （0.1s 采样、阈值 0.12s → 确认时刻为起点后 0.2s）。
    assert result.events[0]["confirmed_s"] - result.events[0]["time_s"] == \
        pytest.approx(0.2, abs=TOL)
    assert result.events[0]["confirmed_s"] - result.events[0]["time_s"] >= \
        ymaze.DEFAULTS["entry_confirm_s"] - TOL


# -- 二、中心点规则：固定序列 -------------------------------------------

def run_y(pattern, *, values=None, monkeypatch=None, window=None,
          extra_units=()):
    """test_ymaze.run 的本地包装：可覆盖冻结窗口并追加更多单位。"""

    unit_value = unit(values)
    source = rows(pattern)
    facts = {"rows": source, "candidates": []}
    units = [unit_value, *extra_units]
    job = {"assay": "ymaze", "run_id": RUN_ID, "job_id": JOB_ID,
           "asset": {"id": "asset", "name": "y.mp4", "path": "memory",
                     "metadata": {"duration_s": 0.1 * len(pattern)}},
           "window": window or {"start_s": 0.0, "end_s": 0.1 * len(pattern)},
           "units": units, "style": {}, "orientation": 0}
    if monkeypatch is not None:
        def fake_capture(context, units_arg):
            return {u["id"]: facts for u in units_arg}
        monkeypatch.setattr(ymaze, "capture", fake_capture)
    return ymaze.analyze(AnalysisContext(job=job))


def test_center_rule_fixture_sequence(monkeypatch):
    result = run_y(FIXTURE, values=options(arm_entry_rule="center"),
                   monkeypatch=monkeypatch)
    # A → 中心 → A 计两次 A；A→B、B→C 直接转换也计。
    assert targets(result) == ["A", "A", "B", "C"]
    assert times(result) == pytest.approx([0.0, 0.6, 1.2, 1.8], abs=TOL)
    for event in result.events:
        assert event["confirmed_s"] == pytest.approx(event["time_s"], abs=TOL), \
            "中心点规则：事件时间与确认时间同为该次有效观测的源时间"
    summary = result.summary[0]
    assert summary["arm_entry_rule"] == "center"
    assert summary["total_arm_entries"] == 4
    assert summary["alternation_count"] == 1
    assert summary["alternation_opportunities"] == 2
    assert summary["alternation_percent"] == pytest.approx(50.0, abs=TOL)
    # 末段缺失后回到 C：不重复计数（最近有效区域仍是 C）。
    assert result.events[-1]["frame"] == 18


def test_center_rule_warning_and_metadata(monkeypatch):
    result = run_y([ARM_A] * 2, values=options(arm_entry_rule="center"),
                   monkeypatch=monkeypatch)
    unit_meta = result.metadata["units"][UNIT_ID]
    assert unit_meta["arm_entry_rule"] == "center"
    assert any("中心点入臂规则" in w and "无时间防抖" in w
               for w in result.warnings)
    # 深度规则不产生该警告。
    plain = run_y([ARM_A] * 2, values=options(arm_entry_rule="depth"),
                  monkeypatch=monkeypatch)
    assert all("中心点入臂规则" not in w for w in plain.warnings)


def test_center_rule_first_observation_in_arm_counts(monkeypatch):
    result = run_y([ARM_A] * 3, values=options(arm_entry_rule="center"),
                   monkeypatch=monkeypatch)
    assert targets(result) == ["A"], "首个有效观测已在臂内记一次"
    assert times(result) == pytest.approx([0.0], abs=TOL)


def test_center_rule_same_arm_stay_counts_once(monkeypatch):
    result = run_y([ARM_A] * 8, values=options(arm_entry_rule="center"),
                   monkeypatch=monkeypatch)
    assert targets(result) == ["A"]
    assert len(result.events) == 1


def test_center_rule_direct_arm_transition(monkeypatch):
    result = run_y([ARM_A, ARM_A, ARM_B, ARM_C],
                   values=options(arm_entry_rule="center"),
                   monkeypatch=monkeypatch)
    assert targets(result) == ["A", "B", "C"], \
        "有效区域直接转换也计，不强制经过中心"


def test_center_rule_boundary_flip_counts_every_transition(monkeypatch):
    """臂/中心边界抖动：本规则无时间防抖，按观测转换计数（文档边界）。"""

    pattern = [ARM_A, CENTER, ARM_A, CENTER, ARM_A]
    result = run_y(pattern, values=options(arm_entry_rule="center"),
                   monkeypatch=monkeypatch)
    assert targets(result) == ["A", "A", "A"]
    assert times(result) == pytest.approx([0.0, 0.2, 0.4], abs=TOL)


def test_center_rule_gap_back_to_same_arm(monkeypatch):
    result = run_y([ARM_A, None, None, ARM_A, ARM_A],
                   values=options(arm_entry_rule="center"),
                   monkeypatch=monkeypatch)
    assert targets(result) == ["A"], "缺失不构成离臂，恢复后仍在原臂不重复计数"


def test_center_rule_gap_then_other_region(monkeypatch):
    result = run_y([ARM_A, None, CENTER, ARM_B],
                   values=options(arm_entry_rule="center"),
                   monkeypatch=monkeypatch)
    assert targets(result) == ["A", "B"], "恢复后观察到其他有效区域按转换计数"


def test_center_rule_unknown_zone_is_neutral(monkeypatch):
    result = run_y([ARM_A, UNKNOWN_POINT, ARM_A, UNKNOWN_POINT, CENTER, ARM_A],
                   values=options(arm_entry_rule="center"),
                   monkeypatch=monkeypatch)
    # 未知区域不计数、不更新上一有效区域：A 后接未知再回 A 仍是 1 次。
    assert targets(result) == ["A", "A"]
    assert [event["frame"] for event in result.events] == [0, 5]


def test_center_rule_depth_params_do_not_apply(monkeypatch):
    """中心点规则不读深度/确认/重置参数：改到极端值结果不变。"""

    normal = run_y([ARM_A, CENTER, ARM_B],
                   values=options(arm_entry_rule="center"),
                   monkeypatch=monkeypatch)
    extreme = run_y(
        [ARM_A, CENTER, ARM_B],
        values=options(arm_entry_rule="center", entry_depth_ratio=1.0,
                       entry_confirm_s=60.0, entry_reset_center_s=60.0,
                       require_center=True),
        monkeypatch=monkeypatch)
    assert targets(normal) == targets(extreme) == ["A", "B"]
    assert times(normal) == times(extreme)


# -- 三、窗口边界与多动物隔离 -------------------------------------------

def test_center_rule_window_boundary(monkeypatch):
    """裁切前行不产生事件：0–0.2s 的 A 不计入 0.2s 起的冻结窗口。"""

    pattern = [ARM_A, ARM_A, CENTER, ARM_A]
    values = options(arm_entry_rule="center")
    result = run_y(pattern, values=values, monkeypatch=monkeypatch,
                   window={"start_s": 0.2, "end_s": 0.4})
    assert targets(result) == ["A"], "窗外上下文不初始化状态"
    assert times(result) == pytest.approx([0.3], abs=TOL)


def test_center_rule_units_are_independent(monkeypatch):
    """同视频两只鼠：规则与序列各自独立，不串用。"""

    unit_a = unit(options(arm_entry_rule="center"))
    unit_b = unit(options(arm_entry_rule="depth"))
    unit_b["id"] = "f" * 32
    unit_b["name"] = "第二只"

    # 每只鼠自己的观测行副本（同一观测事实，不同归属）。
    pattern = [ARM_A, CENTER, ARM_B] * 3
    captured = {}
    for u in (unit_a, unit_b):
        source = rows(pattern)
        for row in source:
            row["unit_id"] = u["id"]
            row["unit_name"] = u["name"]
        captured[u["id"]] = {"rows": source, "candidates": []}
    job = {"assay": "ymaze", "run_id": RUN_ID, "job_id": JOB_ID,
           "asset": {"id": "asset", "name": "y.mp4", "path": "memory",
                     "metadata": {"duration_s": 0.9}},
           "window": {"start_s": 0.0, "end_s": 0.9},
           "units": [unit_a, unit_b], "style": {}, "orientation": 0}
    monkeypatch.setattr(
        ymaze, "capture",
        lambda context, units_arg: {u["id"]: captured[u["id"]]
                                    for u in units_arg})
    result = ymaze.analyze(AnalysisContext(job=job))

    def seq(unit_id):
        return [event["target"] for event in result.events
                if event["unit_id"] == unit_id]

    # 中心点规则：A→中心→B→中心→A→…每次区域转换都计（6 次）；
    # 深度规则下每段停留不足确认时间，为 0 次——差异来自规则本身。
    assert seq(UNIT_ID) == ["A", "B", "A", "B", "A", "B"]
    assert seq("f" * 32) == []
    rules = {row["unit_id"]: row["arm_entry_rule"] for row in result.summary}
    assert rules == {UNIT_ID: "center", "f" * 32: "depth"}


# -- 四、深比对：同观测事实下两规则只差预期字段 -------------------------

EXPECTED_ONLY_KEYS = {
    "total_arm_entries", "arm_A_entries", "arm_B_entries", "arm_C_entries",
    "alternation_count", "alternation_opportunities", "alternation_percent",
    "arm_entry_rule",
}


def test_center_vs_depth_differ_only_in_entry_statistics(monkeypatch):
    pattern = ([ARM_A] * 3 + [CENTER] * 2 + [ARM_B] * 3 + [None] * 2
               + [ARM_C] * 3 + [CENTER] * 2 + [ARM_A] * 3)
    depth = run_y(pattern, values=options(arm_entry_rule="depth"),
                  monkeypatch=monkeypatch)
    center = run_y(pattern, values=options(arm_entry_rule="center"),
                   monkeypatch=monkeypatch)

    depth_summary = dict(depth.summary[0])
    center_summary = dict(center.summary[0])
    diff = {key for key in set(depth_summary) | set(center_summary)
            if depth_summary.get(key) != center_summary.get(key)}
    assert diff <= EXPECTED_ONLY_KEYS, f"意外变化的汇总字段：{diff}"
    assert "arm_entry_rule" in diff

    # 观测行：位置/分区/占用/运动逐行一致；差异只允许规则驱动的入臂
    # 标记列与 current_arm（两规则计入臂的时刻不同，展示状态随之不同）。
    row_only = {"arm_entry_event", "arm_entry_name", "alternation_event",
                "recent_entry_sequence", "current_arm"}
    for old, new in zip(depth.observations, center.observations):
        assert old["zone"] == new["zone"]
        assert old["x_cm"] == new["x_cm"] and old["y_cm"] == new["y_cm"]
        assert old["step_distance_cm"] == new["step_distance_cm"]
        assert old["cumulative_distance_cm"] == new["cumulative_distance_cm"]
        for key in row_only:
            del old[key], new[key]
    assert depth.observations == center.observations

    # 事件差异集中在时间与次数；交替率语义（N−2、三异臂）两侧同构。
    assert depth.events and center.events
    for events in (depth.events, center.events):
        assert [e["entry_index"] for e in events] == list(
            range(1, len(events) + 1))


# -- 五、校验与兼容 -----------------------------------------------------

@pytest.mark.parametrize("rule", ["depth", "center"])
def test_validate_accepts_known_rules(rule):
    assert ymaze.validate_unit(options(arm_entry_rule=rule),
                               geometry(),
                               complete=True) == []


def test_validate_rejects_unknown_rule():
    issues = ymaze.validate_unit(options(arm_entry_rule="impc"),
                                 geometry(),
                                 complete=True)
    assert any(issue["field"] == "options.arm_entry_rule" for issue in issues)


def test_validate_allows_missing_rule_field():
    values = without_rule(options())
    assert ymaze.validate_unit(values, geometry(),
                               complete=True) == []


def test_analyze_rejects_invalid_rule(monkeypatch):
    with pytest.raises(Exception) as exc_info:
        run_y([ARM_A], values=options(arm_entry_rule="impc"),
              monkeypatch=monkeypatch)
    assert "入臂规则" in str(exc_info.value)


def test_depth_params_still_validated_for_depth_rule():
    issues = ymaze.validate_unit(
        options(arm_entry_rule="depth", entry_depth_ratio=1.5),
        geometry(), complete=True)
    assert any(issue["field"] == "options.entry_depth_ratio" for issue in issues)


# -- 六、原有必需字段的校验语义保持 -------------------------------------

def test_missing_legacy_required_fields_fail_validation():
    """不完整的原有配置：校验阶段给字段级错误，不在分析中 KeyError。"""

    incomplete = {"arm_length_cm": 30.0, "arm_width_cm": 10.0,
                  "max_jump_cm": 12.0, "arm_entry_rule": "center"}
    issues = ymaze.validate_unit(incomplete, geometry(), complete=True)
    fields = {issue["field"] for issue in issues}
    for field in ("options.entry_depth_ratio", "options.tracking_hold_s",
                  "options.entry_confirm_s", "options.entry_reset_center_s",
                  "options.movement_threshold_cm_s", "options.bin_seconds"):
        assert field in fields, field
    messages = {issue["field"]: issue["message"] for issue in issues}
    assert "必须是有限数值" in messages["options.entry_confirm_s"]


def test_analyze_rejects_incomplete_options_at_validation(monkeypatch):
    """缺 entry_confirm_s 的冻结配置：校验阶段 ValidationError，非 KeyError。"""

    from behaviourlab.schema import ValidationError

    incomplete = {"arm_length_cm": 30.0, "arm_width_cm": 10.0,
                  "max_jump_cm": 12.0, "arm_entry_rule": "depth"}
    with pytest.raises(ValidationError) as exc_info:
        run_y([ARM_A], values=incomplete, monkeypatch=monkeypatch)
    assert "必须是有限数值" in str(exc_info.value)
    assert not isinstance(exc_info.value, KeyError)


def test_new_rule_field_compat_unchanged():
    """同一次改动内的两种语义并存：原必需字段严格、新字段按实际存在。"""

    legacy = without_rule(options())
    assert ymaze.validate_unit(legacy, geometry(), complete=True) == [], \
        "旧配置仅缺 arm_entry_rule：原深度规则，不报错"
    values = dict(legacy)
    values["arm_entry_rule"] = None
    issues = ymaze.validate_unit(values, geometry(), complete=True)
    assert [issue["field"] for issue in issues] == ["options.arm_entry_rule"]


def test_factory_defaults_center_rule_and_temporal_background():
    """新配置默认：中心点入臂规则 + 时序背景（仅 Y 迷宫覆盖）。"""

    assert ymaze.DEFAULTS["arm_entry_rule"] == "center"
    assert ymaze.DETECTION_DEFAULTS == {"bg_method": "temporal"}
    from behaviourlab import detection
    from behaviourlab.experiments import catalog

    spec = [item for item in catalog() if item["key"] == "ymaze"][0]
    assert spec["detection_defaults"]["bg_method"] == "temporal"
    rule = [p for p in spec["parameters"] if p["key"] == "arm_entry_rule"][0]
    assert rule["default"] == "center"
    assert rule["choices"][0]["value"] == "center"
    # 其他实验的检测默认不受影响。
    others = {item["key"]: item["detection_defaults"]["bg_method"]
              for item in catalog() if item["key"] != "ymaze"}
    assert "temporal" not in others.values(), "时序默认只落在 Y 迷宫"
    # 兼容语义不变：缺字段仍按原深度规则。
    legacy = {key: value for key, value in ymaze.DEFAULTS.items()
              if key != "arm_entry_rule"}
    assert ymaze.validate_unit(legacy, geometry(), complete=True) == []
