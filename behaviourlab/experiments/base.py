"""实验模块的统一骨架：参数/几何/指标描述与未实现异常。"""

from __future__ import annotations

from typing import Any


class ExperimentNotImplemented(NotImplementedError):
    """实验分析尚未实现；绝不返回虚构统计。"""

    def __init__(self, key: str, label: str = "") -> None:
        self.key = key
        detail = f"实验 {key}" + (f"（{label}）" if label else "")
        super().__init__(f"{detail} 的分析尚未实现，本版本不返回统计结果。")


def parameter(
    key: str,
    label: str,
    type: str,
    default: Any,
    *,
    unit: str | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
    step: float | str | None = None,
    choices: list[dict] | None = None,
    section: str | None = None,
    help: str | None = None,
) -> dict:
    item: dict[str, Any] = {
        "key": key,
        "label": label,
        "type": type,
        "default": default,
    }
    if unit is not None:
        item["unit"] = unit
    if minimum is not None:
        item["min"] = minimum
    if maximum is not None:
        item["max"] = maximum
    if step is not None:
        item["step"] = step
    if choices is not None:
        item["choices"] = choices
    if section is not None:
        item["section"] = section
    if help is not None:
        item["help"] = help
    return item


def geometry_field(
    key: str,
    label: str,
    type: str,
    count: int = 0,
    *,
    min_points: int = 0,
    optional: bool = False,
    count_key: str | None = None,
    count_field: str | None = None,
    auto_commit: bool = False,
    fixed: bool = False,
    modes: list[str] | None = None,
    role: str | None = None,
) -> dict:
    """一步几何的绘制契约；count 为 0 表示点数可变，min_points 为下限。"""

    item: dict[str, Any] = {
        "key": key,
        "label": label,
        "type": type,
        "count": count,
        "min_points": min_points,
        "required": not optional,
    }
    if optional:
        item["optional"] = True
    if count_key is not None:
        item["count_key"] = count_key
    if count_field is not None:
        item["count_field"] = count_field
    if auto_commit:
        item["auto_commit"] = True
    if fixed:
        item["fixed"] = True
    if modes is not None:
        item["modes"] = list(modes)
    if role is not None:
        item["role"] = role
    return item


def metric(key: str, label: str, unit: str | None = None) -> dict:
    item: dict[str, Any] = {"key": key, "label": label}
    if unit is not None:
        item["unit"] = unit
    return item


def no_issues(job: Any) -> list[dict]:
    return []
