"""场地模板：只保存场地几何与允许的尺寸参数。

模板不携带运行身份、窗口、人工输入、角色/目标或统计阈值；应用时用深拷贝
替换当前几何、只合并允许的尺寸，其他参数保持不变，任何一侧都不会反向
污染另一侧。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from . import experiments
from .schema import ValidationError

# 这些键属于某次试验/某只动物，不得进入模板。
FORBIDDEN_KEYS = frozenset({
    "role", "target", "target_id", "target_hole", "target_hole_id",
    "target_quadrant", "platform", "animal_id", "manual", "window",
})

# 应用模板时仍需保留的当前几何键：平台是当次试验的目标，模板里没有它。
PRESERVED_GEOMETRY_KEYS = ("platform",)


def strip_forbidden(value: Any) -> Any:
    """递归去掉禁止键；只处理对象与数组，其他值原样深拷贝。"""

    if isinstance(value, Mapping):
        return {
            key: strip_forbidden(item)
            for key, item in value.items()
            if key not in FORBIDDEN_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [strip_forbidden(item) for item in value]
    return deepcopy(value)


def sanitize_geometry(geometry: Mapping[str, Any] | None) -> dict:
    if geometry is None:
        return {}
    if not isinstance(geometry, Mapping):
        raise ValidationError("几何必须是对象", "geometry")
    return strip_forbidden(geometry)


def template_keys(assay: str) -> tuple[str, ...]:
    """该实验允许进入模板的参数键；未知实验没有可保存的尺寸。"""

    if not experiments.exists(assay):
        return ()
    spec = experiments.get(assay).SPEC
    return tuple(spec.get("template_keys") or ())


def allowed_options(assay: str, options: Mapping[str, Any] | None) -> dict:
    """只保留 SPEC.template_keys 中实际存在的值。"""

    values = options if isinstance(options, Mapping) else {}
    allowed = template_keys(assay)
    return {key: deepcopy(values[key]) for key in allowed if key in values}


def build_template(assay: str, unit: Mapping[str, Any],
                   options: Mapping[str, Any] | None = None) -> dict:
    """从当前单位抽取模板内容：几何去掉禁止键，尺寸只留白名单。"""

    # 保存时传入的 options 是界面已解析的有效值；没给就退回单位自己的配置。
    source = options if options is not None else (unit.get("options") or {})
    return {
        "geometry": sanitize_geometry(unit.get("geometry") or {}),
        "options": allowed_options(assay, source),
    }


def _restore_nor_roles(assay: str, geometry: dict,
                       current: Mapping[str, Any] | None) -> None:
    """NOR 的角色被剥离后按对象顺序补回。

    优先沿用当前单位同一序号的角色；没有时第一只熟悉、第二只新奇。
    """

    if assay != "nor":
        return
    objects = geometry.get("objects")
    if not isinstance(objects, list) or not objects:
        return
    existing = (current or {}).get("objects")
    previous: list[str | None] = []
    if isinstance(existing, list):
        for item in existing:
            previous.append(item.get("role") if isinstance(item, Mapping) else None)
    defaults = ("familiar", "novel")
    roles: list[str] = []
    for index, item in enumerate(objects):
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        if role not in ("familiar", "novel"):
            role = previous[index] if index < len(previous) else None
        if role not in ("familiar", "novel"):
            role = defaults[index % len(defaults)]
        item["role"] = role
        roles.append(role)
    if len(roles) == 2 and roles[0] == roles[1]:
        # 两个物体不能同角色：把第二只翻到相反一侧，与工作台规则一致。
        objects[1]["role"] = ("novel" if roles[0] == "familiar" else "familiar")


def apply_template(assay: str, template: Mapping[str, Any],
                   unit: Mapping[str, Any]) -> dict:
    """返回应用后的 {geometry, options}；不修改模板与单位本身。"""

    template_geometry = template.get("geometry")
    if not isinstance(template_geometry, Mapping):
        raise ValidationError("模板缺少几何", "geometry")
    # 先裁剪模板自带的禁止键，再补回必须保留的当前几何。
    geometry = sanitize_geometry(template_geometry)
    current = unit.get("geometry") or {}
    for key in PRESERVED_GEOMETRY_KEYS:
        if key not in geometry and isinstance(current.get(key), Mapping):
            geometry[key] = deepcopy(current[key])
    _restore_nor_roles(assay, geometry, current)

    options = deepcopy(dict(unit.get("options") or {}))
    options.update(allowed_options(assay, template.get("options")))
    return {"geometry": geometry, "options": options}


def validate_template_payload(template: Mapping[str, Any]) -> list[dict]:
    """保存前的结构检查：几何必须是对象，选项必须是对象。"""

    issues: list[dict] = []
    geometry = template.get("geometry", {})
    options = template.get("options", {})
    if not isinstance(geometry, Mapping):
        issues.append({"field": "geometry", "message": "必须是对象"})
    if not isinstance(options, Mapping):
        issues.append({"field": "options", "message": "必须是对象"})
    return issues
