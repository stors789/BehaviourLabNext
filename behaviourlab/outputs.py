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


def _report_markdown(job: Mapping[str, Any], result: Any,
                     manifest: Mapping[str, Any]) -> str:
    lines = [f"# {job.get('assay', '')} 分析报告", ""]
    lines.append(f"- run: {job.get('run_id')}")
    lines.append(f"- job: {job.get('job_id')}")
    lines.append(f"- unit 数: {len(result.summary)}")
    lines.extend(["", "| 动物 | 中心占比 | 场地内路程 (cm) | 平均速度 (cm/s) |",
                  "| --- | --- | --- | --- |"])
    for row in result.summary:
        lines.append(
            "| {name} | {center} | {distance} | {speed} |".format(
                name=row.get("unit_name") or row.get("unit_id"),
                center=_fmt(row.get("center_percent")),
                distance=_fmt(row.get("total_distance_cm")),
                speed=_fmt(row.get("mean_speed_cm_s")),
            ))
    if result.warnings:
        lines.extend(["", "## 警告", ""])
        lines.extend(f"- {text}" for text in result.warnings)
    lines.extend(["", "## 文件", ""])
    lines.extend(f"- {name}" for name in manifest.get("files", []))
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def export_result(directory: str | Path, job: Mapping[str, Any],
                  result: Any) -> dict:
    """把一份 AnalysisResult 写成清单里的文件，返回 manifest。

    没有有效位置时省掉图形；CSV 的 null 写成空单元格。
    """

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    files: list[str] = []

    write_json_export(target / "frozen_job.json", job)
    files.append("frozen_job.json")
    write_observations(target / "observations.csv", result.observations)
    files.append("observations.csv")
    write_csv(target / "summary.csv", result.summary)
    files.append("summary.csv")
    write_csv(target / "events.csv", result.events)
    files.append("events.csv")
    write_csv(target / "bins.csv", result.bins)
    files.append("bins.csv")
    payload = result.as_dict()
    write_json_export(target / "results.json", payload)
    files.append("results.json")

    manifest = {
        "run_id": job.get("run_id"),
        "job_id": job.get("job_id"),
        "assay": job.get("assay"),
        "units": [{"unit_id": row.get("unit_id"),
                   "unit_name": row.get("unit_name"),
                   "summary": row} for row in result.summary],
        "files": files + ["result_manifest.json", "report.md"],
        "warnings": list(result.warnings),
        "metadata": result.metadata,
    }
    write_json_export(target / "result_manifest.json", manifest)
    (target / "report.md").write_text(
        _report_markdown(job, result, manifest), encoding="utf-8")
    return manifest
