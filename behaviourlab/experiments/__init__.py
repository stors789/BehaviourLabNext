"""十个实验模块的注册表与 catalog 描述。

注册顺序即界面顺序：tst、fst、mwm、oft、nor、epm、ymaze、ldb、barnes、ezm。
TST 的 classic/lis 是同一实验内的两种模式，不额外占用注册键。
"""

from __future__ import annotations

from copy import deepcopy
from types import ModuleType

from ..detection import common_detection_defaults
from . import barnes, epm, ezm, fst, ldb, mwm, nor, oft, tst, ymaze

ORDER: tuple[str, ...] = (
    "tst",
    "fst",
    "mwm",
    "oft",
    "nor",
    "epm",
    "ymaze",
    "ldb",
    "barnes",
    "ezm",
)

REGISTRY: dict[str, ModuleType] = {
    "tst": tst,
    "fst": fst,
    "mwm": mwm,
    "oft": oft,
    "nor": nor,
    "epm": epm,
    "ymaze": ymaze,
    "ldb": ldb,
    "barnes": barnes,
    "ezm": ezm,
}


def get(assay: str) -> ModuleType:
    if not isinstance(assay, str) or assay not in REGISTRY:
        raise KeyError(f"未知实验：{assay!r}")
    return REGISTRY[assay]


def exists(assay: object) -> bool:
    return isinstance(assay, str) and assay in REGISTRY


def _catalog_parameters(module: ModuleType) -> list[dict]:
    parameters = []
    defaults = module.DEFAULTS
    for item in module.SPEC.get("parameters", []):
        entry = deepcopy(item)
        key = entry.get("key")
        if key in defaults:
            entry["default"] = deepcopy(defaults[key])
        parameters.append(entry)
    return parameters


def catalog() -> list[dict]:
    """返回十项实验描述，参数默认值与工厂 DEFAULTS 对齐。"""

    entries = []
    for key in ORDER:
        module = REGISTRY[key]
        spec = module.SPEC
        entries.append(
            {
                "key": key,
                "label": spec["label"],
                "short": spec["short"],
                "parameters": _catalog_parameters(module),
                "geometry": deepcopy(spec.get("geometry", [])),
                "metrics": deepcopy(spec.get("metrics", [])),
                "template_keys": list(spec.get("template_keys", [])),
                "defaults": deepcopy(module.DEFAULTS),
                "detection_defaults": {
                    **common_detection_defaults(),
                    **deepcopy(getattr(module, "DETECTION_DEFAULTS", {})),
                },
            }
        )
    return entries
