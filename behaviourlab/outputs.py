"""从保存的事实生成导出文件。

导出层把计算产生的非有限浮点归一为 null；CSV 用空单元格表示缺失。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .schema import sanitize_json
from .store import atomic_write_json


def write_json_export(path: str | Path, payload: Any) -> Path:
    return atomic_write_json(path, sanitize_json(payload))


def _columns_from(rows: Sequence[Mapping[str, Any]], columns: Iterable[str] | None) -> list[str]:
    if columns is not None:
        return list(columns)
    ordered: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                ordered.append(key)
    return ordered


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]],
              columns: Iterable[str] | None = None) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _columns_from(rows, columns)
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            sanitized = sanitize_json(dict(row))
            writer.writerow(
                {key: "" if sanitized.get(key) is None else sanitized.get(key)
                 for key in fieldnames}
            )
    return target


def write_observations(path: str | Path, observations: Sequence[Mapping[str, Any]],
                       columns: Iterable[str] | None = None) -> Path:
    return write_csv(path, observations, columns)
