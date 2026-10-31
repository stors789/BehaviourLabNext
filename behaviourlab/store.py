"""SQLite 持久化与原子 JSON 写入边界。

数据库使用 WAL、每连接外键约束和显式事务；关系列、ID 与时间为权威字段，
数据体按严格 JSON 存放。导入本模块不创建数据库。
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .config import SCHEMA_VERSION, DataPaths
from .schema import (
    ValidationError,
    check_json_value,
    dumps_strict,
    new_id,
    require_id,
)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS groups (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_groups_project ON groups(project_id);

CREATE TABLE IF NOT EXISTS animals (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    group_id TEXT REFERENCES groups(id) ON DELETE SET NULL,
    name TEXT NOT NULL,
    sex TEXT,
    birth_date TEXT,
    notes TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_animals_project ON animals(project_id);
CREATE INDEX IF NOT EXISTS idx_animals_group ON animals(group_id);

CREATE TABLE IF NOT EXISTS assets (
    id TEXT PRIMARY KEY,
    project_id TEXT REFERENCES projects(id) ON DELETE SET NULL,
    name TEXT NOT NULL,
    path TEXT NOT NULL,
    ownership TEXT NOT NULL CHECK (ownership IN ('linked', 'owned')),
    metadata TEXT NOT NULL DEFAULT '{}',
    upload_sha256 TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_assets_project ON assets(project_id);

CREATE TABLE IF NOT EXISTS drafts (
    id TEXT PRIMARY KEY,
    project_id TEXT REFERENCES projects(id) ON DELETE SET NULL,
    assay TEXT NOT NULL,
    videos TEXT NOT NULL DEFAULT '[]',
    options TEXT NOT NULL DEFAULT '{}',
    detection TEXT NOT NULL DEFAULT '{}',
    output TEXT NOT NULL DEFAULT '{}',
    style TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_drafts_project ON drafts(project_id);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    draft_id TEXT REFERENCES drafts(id) ON DELETE SET NULL,
    project_id TEXT REFERENCES projects(id) ON DELETE SET NULL,
    assay TEXT NOT NULL,
    status TEXT NOT NULL,
    config TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_draft ON jobs(draft_id);
CREATE INDEX IF NOT EXISTS idx_jobs_run ON jobs(run_id);

CREATE TABLE IF NOT EXISTS templates (
    id TEXT PRIMARY KEY,
    key TEXT NOT NULL,
    assay TEXT NOT NULL,
    name TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_templates_assay ON templates(assay);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS schema_info (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class NotFoundError(LookupError):
    """请求的持久对象不存在。"""


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _text(value: Any, field: str, *, max_length: int = 200,
          allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValidationError("必须是文本", field)
    text = value.strip()
    if not text and not allow_empty:
        raise ValidationError("不能为空", field)
    if len(text) > max_length:
        raise ValidationError(f"不能超过 {max_length} 个字符", field)
    return text


def _json_text(value: Any, field: str) -> str:
    check_json_value(value, field)
    return dumps_strict(value)


def _decode_row(row: sqlite3.Row, json_fields: Sequence[str] = ()) -> dict:
    data = dict(row)
    for name in json_fields:
        raw = data.get(name)
        data[name] = json.loads(raw) if raw not in (None, "") else None
    return data


def _row_or_none(row: sqlite3.Row | None, json_fields: Sequence[str] = ()) -> dict | None:
    return None if row is None else _decode_row(row, json_fields)


def atomic_write_json(path: str | os.PathLike[str], payload: Any, *, indent: int = 2) -> Path:
    """同目录唯一临时文件写入、fsync 后原子替换。"""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = dumps_strict(payload, indent=indent)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise
    return target


def read_json_file(path: str | os.PathLike[str], default: Any = None) -> Any:
    file_path = Path(path)
    if not file_path.exists():
        return deepcopy(default)
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class Store:
    """专用数据根上的 SQLite 存储；构造时才创建目录与数据库。"""

    def __init__(self, paths: DataPaths, *, create: bool = True) -> None:
        self.paths = paths.ensure() if create else paths
        self.db_path = self.paths.database
        self._initialize()

    # -- 连接与事务 -----------------------------------------------------
    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.db_path, timeout=30.0, isolation_level=None
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    @contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        connection = self.connect()
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.executescript(SCHEMA_SQL)
            connection.execute(
                "INSERT OR REPLACE INTO schema_info(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
        finally:
            connection.close()

    def schema_version(self) -> int:
        with self.read() as connection:
            row = connection.execute(
                "SELECT value FROM schema_info WHERE key = 'schema_version'"
            ).fetchone()
        return int(row["value"]) if row else 0

    # -- 项目 -----------------------------------------------------------
    def create_project(self, name: str, description: str = "") -> dict:
        project_id = new_id()
        now = utc_now()
        clean_name = _text(name, "name")
        clean_description = _text(description or "", "description", max_length=2000,
                                  allow_empty=True)
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO projects(id, name, description, created_at, updated_at)"
                " VALUES(?, ?, ?, ?, ?)",
                (project_id, clean_name, clean_description, now, now),
            )
        record = self.get_project(project_id)
        assert record is not None
        return record

    def list_projects(self) -> list[dict]:
        with self.read() as connection:
            rows = connection.execute(
                """
                SELECT p.*,
                       (SELECT COUNT(*) FROM groups g WHERE g.project_id = p.id) AS group_count,
                       (SELECT COUNT(*) FROM animals a WHERE a.project_id = p.id) AS animal_count
                FROM projects p
                ORDER BY p.updated_at DESC, p.created_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_project(self, project_id: str) -> dict | None:
        require_id(project_id, "project_id")
        with self.read() as connection:
            row = connection.execute(
                """
                SELECT p.*,
                       (SELECT COUNT(*) FROM groups g WHERE g.project_id = p.id) AS group_count,
                       (SELECT COUNT(*) FROM animals a WHERE a.project_id = p.id) AS animal_count
                FROM projects p WHERE p.id = ?
                """,
                (project_id,),
            ).fetchone()
        return _row_or_none(row)

    def update_project(self, project_id: str, changes: Mapping[str, Any]) -> dict:
        require_id(project_id, "project_id")
        if not isinstance(changes, Mapping):
            raise ValidationError("更新内容必须是对象", "")
        allowed = {"name", "description"}
        for key in changes:
            if key in ("id", "created_at", "updated_at"):
                raise ValidationError("不允许修改该字段", key)
            if key not in allowed:
                raise ValidationError(f"不支持的字段：{key}", key)
        if not changes:
            raise ValidationError("没有可更新的字段", "")

        assignments: list[str] = []
        values: list[Any] = []
        if "name" in changes:
            assignments.append("name = ?")
            values.append(_text(changes["name"], "name"))
        if "description" in changes:
            assignments.append("description = ?")
            values.append(_text(changes["description"], "description",
                                max_length=2000, allow_empty=True))
        assignments.append("updated_at = ?")
        values.append(utc_now())
        values.append(project_id)

        with self.transaction() as connection:
            cursor = connection.execute(
                f"UPDATE projects SET {', '.join(assignments)} WHERE id = ?", values
            )
            if cursor.rowcount == 0:
                raise NotFoundError("项目不存在")
        record = self.get_project(project_id)
        assert record is not None
        return record

    def delete_project(self, project_id: str) -> None:
        require_id(project_id, "project_id")
        with self.transaction() as connection:
            cursor = connection.execute("DELETE FROM projects WHERE id = ?", (project_id,))
            if cursor.rowcount == 0:
                raise NotFoundError("项目不存在")

    # -- 分组与动物 -----------------------------------------------------
    def _require_project(self, connection: sqlite3.Connection, project_id: str) -> None:
        row = connection.execute(
            "SELECT 1 FROM projects WHERE id = ?", (project_id,)
        ).fetchone()
        if row is None:
            raise NotFoundError("项目不存在")

    def _insert_groups(
        self, project_id: str, names: Sequence[Any], fields: Sequence[str]
    ) -> list[dict]:
        require_id(project_id, "project_id")
        now = utc_now()
        prepared = [
            {"id": new_id(), "name": _text(name, field)}
            for name, field in zip(names, fields)
        ]
        with self.transaction() as connection:
            self._require_project(connection, project_id)
            for entry in prepared:
                connection.execute(
                    "INSERT INTO groups(id, project_id, name, created_at, updated_at)"
                    " VALUES(?, ?, ?, ?, ?)",
                    (entry["id"], project_id, entry["name"], now, now),
                )
        return [
            {
                "id": entry["id"],
                "project_id": project_id,
                "name": entry["name"],
                "created_at": now,
                "updated_at": now,
            }
            for entry in prepared
        ]

    def create_group(self, project_id: str, name: str) -> dict:
        return self._insert_groups(project_id, [name], ["name"])[0]

    def create_groups(self, project_id: str, names: Sequence[Any]) -> list[dict]:
        """批量新增分组；任一名称无效则整批失败，不写入任何一行。"""

        if not isinstance(names, (list, tuple)) or not names:
            raise ValidationError("groups 必须是非空数组", "groups")
        fields = [f"groups[{index}].name" for index in range(len(names))]
        return self._insert_groups(project_id, names, fields)

    def list_groups(self, project_id: str) -> list[dict]:
        require_id(project_id, "project_id")
        with self.read() as connection:
            rows = connection.execute(
                "SELECT * FROM groups WHERE project_id = ?"
                " ORDER BY name COLLATE NOCASE, created_at",
                (project_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def create_animals(self, project_id: str, items: Sequence[Mapping[str, Any]]) -> list[dict]:
        require_id(project_id, "project_id")
        if not isinstance(items, (list, tuple)) or not items:
            raise ValidationError("至少提供一只动物", "animals")
        now = utc_now()
        prepared: list[dict] = []
        seen: set[str] = set()
        for index, item in enumerate(items):
            if not isinstance(item, Mapping):
                raise ValidationError("动物条目必须是对象", f"animals[{index}]")
            animal_id = item.get("id") or new_id()
            require_id(animal_id, f"animals[{index}].id")
            if animal_id in seen:
                raise ValidationError("动物 ID 重复", f"animals[{index}].id")
            seen.add(animal_id)
            name = _text(item.get("name"), f"animals[{index}].name", max_length=120)
            group_id = item.get("group_id")
            if group_id is not None:
                require_id(group_id, f"animals[{index}].group_id")
            sex = item.get("sex")
            if sex is not None:
                sex = _text(sex, f"animals[{index}].sex", max_length=32)
            birth_date = item.get("birth_date")
            if birth_date is not None:
                birth_date = _text(birth_date, f"animals[{index}].birth_date",
                                   max_length=32)
            notes = item.get("notes")
            if notes is not None:
                notes = _text(notes, f"animals[{index}].notes", max_length=2000,
                              allow_empty=True)
            prepared.append(
                {
                    "id": animal_id,
                    "name": name,
                    "group_id": group_id,
                    "sex": sex,
                    "birth_date": birth_date,
                    "notes": notes,
                }
            )

        with self.transaction() as connection:
            self._require_project(connection, project_id)
            for entry in prepared:
                group_id = entry["group_id"]
                if group_id is not None:
                    row = connection.execute(
                        "SELECT project_id FROM groups WHERE id = ?", (group_id,)
                    ).fetchone()
                    if row is None or row["project_id"] != project_id:
                        raise ValidationError("分组不属于该项目", "group_id")
                connection.execute(
                    "INSERT INTO animals(id, project_id, group_id, name, sex,"
                    " birth_date, notes, created_at, updated_at)"
                    " VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        entry["id"],
                        project_id,
                        group_id,
                        entry["name"],
                        entry["sex"],
                        entry["birth_date"],
                        entry["notes"],
                        now,
                        now,
                    ),
                )
        return self.list_animals(project_id)

    def create_animal(self, project_id: str, name: str, **fields: Any) -> dict:
        animal_id = fields.pop("id", None) or new_id()
        self.create_animals(project_id, [{"id": animal_id, "name": name, **fields}])
        return next(
            item for item in self.list_animals(project_id) if item["id"] == animal_id
        )

    def list_animals(self, project_id: str) -> list[dict]:
        require_id(project_id, "project_id")
        with self.read() as connection:
            rows = connection.execute(
                """
                SELECT a.*, g.name AS group_name
                FROM animals a LEFT JOIN groups g ON g.id = a.group_id
                WHERE a.project_id = ?
                ORDER BY a.created_at, a.name COLLATE NOCASE
                """,
                (project_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    # -- 素材 -----------------------------------------------------------
    def create_asset(
        self,
        *,
        name: str,
        path: str,
        ownership: str = "linked",
        project_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        upload_sha256: str | None = None,
    ) -> dict:
        asset_id = new_id()
        now = utc_now()
        clean_name = _text(name, "name")
        if ownership not in ("linked", "owned"):
            raise ValidationError("ownership 只能是 linked 或 owned", "ownership")
        if not isinstance(path, str) or not path.strip():
            raise ValidationError("路径不能为空", "path")
        if project_id is not None:
            require_id(project_id, "project_id")
        metadata = dict(metadata or {})
        check_json_value(metadata, "metadata")
        if upload_sha256 is not None:
            if not isinstance(upload_sha256, str) or len(upload_sha256) != 64:
                raise ValidationError("sha256 必须是 64 位十六进制", "upload_sha256")

        with self.transaction() as connection:
            if project_id is not None:
                self._require_project(connection, project_id)
            connection.execute(
                "INSERT INTO assets(id, project_id, name, path, ownership, metadata,"
                " upload_sha256, created_at, updated_at)"
                " VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    asset_id,
                    project_id,
                    clean_name,
                    str(Path(path)),
                    ownership,
                    _json_text(metadata, "metadata"),
                    upload_sha256,
                    now,
                    now,
                ),
            )
        record = self.get_asset(asset_id)
        assert record is not None
        return record

    def get_asset(self, asset_id: str) -> dict | None:
        require_id(asset_id, "asset_id")
        with self.read() as connection:
            row = connection.execute(
                "SELECT * FROM assets WHERE id = ?", (asset_id,)
            ).fetchone()
        return _row_or_none(row, ("metadata",))

    def list_assets(self, project_id: str | None = None) -> list[dict]:
        with self.read() as connection:
            if project_id is None:
                rows = connection.execute(
                    "SELECT * FROM assets ORDER BY created_at DESC"
                ).fetchall()
            else:
                require_id(project_id, "project_id")
                rows = connection.execute(
                    "SELECT * FROM assets WHERE project_id = ? ORDER BY created_at DESC",
                    (project_id,),
                ).fetchall()
        return [_decode_row(row, ("metadata",)) for row in rows]

    # -- 草稿 -----------------------------------------------------------
    def create_draft(
        self,
        assay: str,
        *,
        project_id: str | None = None,
        videos: Sequence[Mapping[str, Any]] | None = None,
        options: Mapping[str, Any] | None = None,
        detection: Mapping[str, Any] | None = None,
        output: Mapping[str, Any] | None = None,
        style: Mapping[str, Any] | None = None,
        draft_id: str | None = None,
    ) -> dict:
        if draft_id is not None:
            require_id(draft_id, "draft_id")
        draft_id = draft_id or new_id()
        now = utc_now()
        clean_assay = _text(assay, "assay", max_length=40)
        if project_id is not None:
            require_id(project_id, "project_id")
        payload = {
            "videos": list(videos or []),
            "options": dict(options or {}),
            "detection": dict(detection or {}),
            "output": dict(output or {}),
            "style": dict(style or {}),
        }
        encoded = {key: _json_text(value, key) for key, value in payload.items()}

        with self.transaction() as connection:
            if project_id is not None:
                self._require_project(connection, project_id)
            connection.execute(
                "INSERT INTO drafts(id, project_id, assay, videos, options, detection,"
                " output, style, created_at, updated_at)"
                " VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    draft_id,
                    project_id,
                    clean_assay,
                    encoded["videos"],
                    encoded["options"],
                    encoded["detection"],
                    encoded["output"],
                    encoded["style"],
                    now,
                    now,
                ),
            )
        record = self.get_draft(draft_id)
        assert record is not None
        return record

    def get_draft(self, draft_id: str) -> dict | None:
        require_id(draft_id, "draft_id")
        with self.read() as connection:
            row = connection.execute(
                "SELECT * FROM drafts WHERE id = ?", (draft_id,)
            ).fetchone()
        return _row_or_none(
            row, ("videos", "options", "detection", "output", "style")
        )

    def list_drafts(self, project_id: str | None = None) -> list[dict]:
        with self.read() as connection:
            if project_id is None:
                rows = connection.execute(
                    "SELECT * FROM drafts ORDER BY updated_at DESC"
                ).fetchall()
            else:
                require_id(project_id, "project_id")
                rows = connection.execute(
                    "SELECT * FROM drafts WHERE project_id = ? ORDER BY updated_at DESC",
                    (project_id,),
                ).fetchall()
        return [
            _decode_row(row, ("videos", "options", "detection", "output", "style"))
            for row in rows
        ]

    def update_draft(self, draft_id: str, changes: Mapping[str, Any]) -> dict:
        require_id(draft_id, "draft_id")
        if not isinstance(changes, Mapping):
            raise ValidationError("更新内容必须是对象", "")
        allowed = {"assay", "project_id", "videos", "options", "detection",
                   "output", "style"}
        for key in changes:
            if key in ("id", "created_at", "updated_at"):
                raise ValidationError("不允许修改该字段", key)
            if key not in allowed:
                raise ValidationError(f"不支持的字段：{key}", key)
        if not changes:
            raise ValidationError("没有可更新的字段", "")

        assignments: list[str] = []
        values: list[Any] = []
        if "assay" in changes:
            assignments.append("assay = ?")
            values.append(_text(changes["assay"], "assay", max_length=40))
        if "project_id" in changes:
            project_id = changes["project_id"]
            if project_id is not None:
                require_id(project_id, "project_id")
            assignments.append("project_id = ?")
            values.append(project_id)
        if "videos" in changes:
            if not isinstance(changes["videos"], (list, tuple)):
                raise ValidationError("videos 必须是数组", "videos")
            assignments.append("videos = ?")
            values.append(_json_text(list(changes["videos"]), "videos"))
        for field in ("options", "detection", "output", "style"):
            if field in changes:
                value = changes[field]
                if not isinstance(value, Mapping):
                    raise ValidationError(f"{field} 必须是对象", field)
                assignments.append(f"{field} = ?")
                values.append(_json_text(dict(value), field))

        assignments.append("updated_at = ?")
        values.append(utc_now())
        values.append(draft_id)

        with self.transaction() as connection:
            if "project_id" in changes and changes["project_id"] is not None:
                self._require_project(connection, changes["project_id"])
            cursor = connection.execute(
                f"UPDATE drafts SET {', '.join(assignments)} WHERE id = ?", values
            )
            if cursor.rowcount == 0:
                raise NotFoundError("草稿不存在")
        record = self.get_draft(draft_id)
        assert record is not None
        return record

    def delete_draft(self, draft_id: str) -> None:
        require_id(draft_id, "draft_id")
        with self.transaction() as connection:
            cursor = connection.execute("DELETE FROM drafts WHERE id = ?", (draft_id,))
            if cursor.rowcount == 0:
                raise NotFoundError("草稿不存在")

    # -- 任务 -----------------------------------------------------------
    def create_job(
        self,
        *,
        assay: str,
        run_id: str | None = None,
        job_id: str | None = None,
        draft_id: str | None = None,
        project_id: str | None = None,
        status: str = "queued",
        config: Mapping[str, Any] | None = None,
    ) -> dict:
        job_id = job_id or new_id()
        run_id = run_id or new_id()
        require_id(job_id, "job_id")
        require_id(run_id, "run_id")
        now = utc_now()
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO jobs(id, run_id, draft_id, project_id, assay, status,"
                " config, created_at, updated_at)"
                " VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job_id,
                    run_id,
                    draft_id,
                    project_id,
                    _text(assay, "assay", max_length=40),
                    _text(status, "status", max_length=32),
                    _json_text(dict(config or {}), "config"),
                    now,
                    now,
                ),
            )
        record = self.get_job(job_id)
        assert record is not None
        return record

    def get_job(self, job_id: str) -> dict | None:
        require_id(job_id, "job_id")
        with self.read() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return _row_or_none(row, ("config",))

    def list_jobs(self, run_id: str | None = None) -> list[dict]:
        with self.read() as connection:
            if run_id is None:
                rows = connection.execute(
                    "SELECT * FROM jobs ORDER BY created_at DESC"
                ).fetchall()
            else:
                require_id(run_id, "run_id")
                rows = connection.execute(
                    "SELECT * FROM jobs WHERE run_id = ? ORDER BY created_at",
                    (run_id,),
                ).fetchall()
        return [_decode_row(row, ("config",)) for row in rows]

    def update_job_status(self, job_id: str, status: str) -> dict:
        require_id(job_id, "job_id")
        with self.transaction() as connection:
            cursor = connection.execute(
                "UPDATE jobs SET status = ?, updated_at = ? WHERE id = ?",
                (_text(status, "status", max_length=32), utc_now(), job_id),
            )
            if cursor.rowcount == 0:
                raise NotFoundError("任务不存在")
        record = self.get_job(job_id)
        assert record is not None
        return record

    # -- 模板 -----------------------------------------------------------
    def create_template(self, *, key: str, assay: str, name: str,
                        payload: Mapping[str, Any] | None = None) -> dict:
        template_id = new_id()
        now = utc_now()
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO templates(id, key, assay, name, payload, created_at,"
                " updated_at) VALUES(?, ?, ?, ?, ?, ?, ?)",
                (
                    template_id,
                    _text(key, "key", max_length=80),
                    _text(assay, "assay", max_length=40),
                    _text(name, "name", max_length=120),
                    _json_text(dict(payload or {}), "payload"),
                    now,
                    now,
                ),
            )
        return self.get_template(template_id)  # type: ignore[return-value]

    def get_template(self, template_id: str) -> dict | None:
        require_id(template_id, "template_id")
        with self.read() as connection:
            row = connection.execute(
                "SELECT * FROM templates WHERE id = ?", (template_id,)
            ).fetchone()
        return _row_or_none(row, ("payload",))

    def list_templates(self, assay: str | None = None) -> list[dict]:
        with self.read() as connection:
            if assay is None:
                rows = connection.execute(
                    "SELECT * FROM templates ORDER BY assay, name"
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM templates WHERE assay = ? ORDER BY name",
                    (assay,),
                ).fetchall()
        return [_decode_row(row, ("payload",)) for row in rows]

    # -- 设置 -----------------------------------------------------------
    def get_setting(self, key: str, default: Any = None) -> Any:
        with self.read() as connection:
            row = connection.execute(
                "SELECT value FROM settings WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return deepcopy(default)
        return json.loads(row["value"])

    def set_setting(self, key: str, value: Any) -> Any:
        clean_key = _text(key, "key", max_length=64)
        encoded = _json_text(value, "value")
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO settings(key, value, updated_at) VALUES(?, ?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value,"
                " updated_at = excluded.updated_at",
                (clean_key, encoded, utc_now()),
            )
        return json.loads(encoded)

    def list_settings(self) -> dict:
        with self.read() as connection:
            rows = connection.execute("SELECT key, value FROM settings").fetchall()
        return {row["key"]: json.loads(row["value"]) for row in rows}
