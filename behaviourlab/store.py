"""SQLite 持久化与原子 JSON 写入边界。

数据库使用 WAL、每连接外键约束和显式事务；关系列、ID 与时间为权威字段，
数据体按严格 JSON 存放。导入本模块不创建数据库。旧库通过幂等迁移补齐新列，
不重建表、不清空用户数据。
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import tempfile
import threading
import time
import traceback
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from .config import APP_NAME, SCHEMA_VERSION, DataPaths
from .schema import (
    ValidationError,
    check_json_value,
    dumps_strict,
    new_id,
    require_id,
)

STORAGE_LOGGER = logging.getLogger("behaviourlab.storage")

# Windows 上 os.replace 偶尔遇到短暂共享冲突：有限退避后仍失败就如实报错。
ATOMIC_REPLACE_ATTEMPTS = 5
ATOMIC_REPLACE_BACKOFF_S = 0.02

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
    code TEXT,
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
    geometry TEXT NOT NULL DEFAULT '{}',
    options TEXT NOT NULL DEFAULT '{}',
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


# 旧库补列：只做 ADD COLUMN，可重复执行，不动已有记录。
MIGRATION_COLUMNS = (
    ("animals", "code", "TEXT"),
    ("templates", "geometry", "TEXT NOT NULL DEFAULT '{}'"),
    ("templates", "options", "TEXT NOT NULL DEFAULT '{}'"),
)


class NotFoundError(LookupError):
    """请求的持久对象不存在。"""


class StoreClosedError(RuntimeError):
    """Store 已经关闭，不再接受新的读写。"""


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


def storage_log_path(paths: DataPaths) -> Path:
    """存储失败日志的实际可写位置；主日志目录不可写时回退到临时目录。"""

    candidate = Path(paths.logs) / "storage.log"
    try:
        candidate.parent.mkdir(parents=True, exist_ok=True)
        with candidate.open("a", encoding="utf-8"):
            pass
        return candidate
    except OSError:
        fallback = Path(tempfile.gettempdir()) / APP_NAME / "logs"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback / "storage.log"


def log_storage_failure(paths: DataPaths, phase: str, error: BaseException,
                        *, operation: str = "",
                        database: str | None = None) -> Path:
    """把存储失败写到独立于数据库的日志；返回实际使用的路径。

    记录真实数据库路径、阶段、操作与异常堆栈，以及可用的 SQLite 扩展错误码。
    """

    target = storage_log_path(paths)
    details = []
    if database:
        details.append(f"database={database}")
    details.append(f"phase={phase}")
    if operation:
        details.append(f"operation={operation}")
    details.append(f"error={error.__class__.__name__}: {error}")
    name = getattr(error, "sqlite_errorname", None)
    if name:
        details.append(f"sqlite_errorname={name}")
    code = getattr(error, "sqlite_errorcode", None)
    if code is not None:
        details.append(f"sqlite_errorcode={code}")
    text = (
        f"[{utc_now()}] " + " ".join(details) + "\n"
        + "".join(traceback.format_exception(type(error), error,
                                             error.__traceback__))
        + "\n"
    )
    try:
        with target.open("a", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError:
        STORAGE_LOGGER.error("存储失败（日志写入也失败）：%s", text)
    else:
        STORAGE_LOGGER.error("存储失败已记录：%s", target)
    return target


def atomic_write_json(path: str | os.PathLike[str], payload: Any, *,
                      indent: int | None = 2) -> Path:
    """同目录唯一临时文件写入、fsync 后原子替换。

    Windows 上替换偶发共享冲突时做有限退避重试；成功或失败都清理自己的
    临时文件，重试耗尽仍抛出原异常，不把失败伪装成已保存。
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = dumps_strict(payload, indent=indent)
    last_error: BaseException | None = None
    for attempt in range(ATOMIC_REPLACE_ATTEMPTS):
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, target)
            return target
        except BaseException as error:  # noqa: BLE001 - 统一清理后再抛出
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            except OSError:
                pass
            last_error = error
            if isinstance(error, PermissionError) \
                    and attempt + 1 < ATOMIC_REPLACE_ATTEMPTS:
                time.sleep(ATOMIC_REPLACE_BACKOFF_S * (attempt + 1))
                continue
            raise
    raise last_error if last_error is not None else OSError("无法写入 JSON")


def read_json_file(path: str | os.PathLike[str], default: Any = None) -> Any:
    file_path = Path(path)
    if not file_path.exists():
        return deepcopy(default)
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class Store:
    """专用数据根上的 SQLite 存储；构造时才创建目录与数据库。"""

    def __init__(self, paths: DataPaths, *, create: bool = True,
                 keeper: bool = False) -> None:
        self.paths = paths.ensure() if create else paths
        self.db_path = self.paths.database
        self._lock = threading.RLock()
        self._keeper: sqlite3.Connection | None = None
        self._closed = False
        self._initialize()
        if keeper:
            # 初始化读取已经结束：keeper 之后不持有任何读/写事务。
            self.keeper()

    # -- 连接与事务 -----------------------------------------------------
    def connect(self, *, shared: bool = False) -> sqlite3.Connection:
        if self._closed:
            raise StoreClosedError("存储已关闭，不再接受新的读写")
        connection = sqlite3.connect(
            self.db_path, timeout=30.0, isolation_level=None,
            check_same_thread=not shared,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def keeper(self) -> sqlite3.Connection:
        """可选的空闲连接：不持事务，可跨线程关闭；仅初始化后创建。"""

        with self._lock:
            if self._closed:
                raise StoreClosedError("存储已关闭，不再接受新的读写")
            if self._keeper is None:
                self._keeper = self.connect(shared=True)
            return self._keeper

    def close(self) -> None:
        """幂等关闭：只释放 keeper；短连接各自关闭，文件仍可被新 Store 打开。"""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            keeper, self._keeper = self._keeper, None
        if keeper is not None:
            try:
                keeper.close()
            except sqlite3.Error:  # 已关闭或不可用都不影响幂等
                pass

    @property
    def closed(self) -> bool:
        return self._closed

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
        except BaseException as error:
            try:
                connection.execute("ROLLBACK")
            finally:
                connection.close()
            if isinstance(error, sqlite3.Error):
                log_storage_failure(self.paths, "transaction", error,
                                    database=str(self.db_path))
            raise
        else:
            connection.close()

    @contextmanager
    def mutate(self) -> Iterator[sqlite3.Connection]:
        """读改写边界：BEGIN IMMEDIATE 内完成整段读—改—写，异常回滚。"""

        with self.transaction() as connection:
            yield connection

    def _initialize(self) -> None:
        connection = self.connect()
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.executescript(SCHEMA_SQL)
            self._migrate(connection)
            connection.execute(
                "INSERT OR REPLACE INTO schema_info(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
        except BaseException:
            connection.close()
            raise
        connection.close()

    def _migrate(self, connection: sqlite3.Connection) -> None:
        """幂等补列：已有旧库只加列，不动记录，也不重建表。"""

        for table, column, definition in MIGRATION_COLUMNS:
            columns = {
                row["name"] for row in
                connection.execute(f"PRAGMA table_info({table})").fetchall()
            }
            if column not in columns:
                connection.execute(
                    f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        # 旧模板把 geometry/options 塞在 payload 里：新列为空时按值搬一次。
        rows = connection.execute(
            "SELECT id, payload, geometry, options FROM templates"
        ).fetchall()
        for row in rows:
            geometry = json.loads(row["geometry"] or "{}")
            options = json.loads(row["options"] or "{}")
            if geometry and options:
                continue
            try:
                payload = json.loads(row["payload"] or "{}")
            except ValueError:
                continue
            if not isinstance(payload, Mapping):
                continue
            moved_geometry = geometry or payload.get("geometry") or {}
            moved_options = options or payload.get("options") or {}
            if not isinstance(moved_geometry, Mapping):
                moved_geometry = {}
            if not isinstance(moved_options, Mapping):
                moved_options = {}
            if moved_geometry or moved_options:
                connection.execute(
                    "UPDATE templates SET geometry = ?, options = ? WHERE id = ?",
                    (dumps_strict(dict(moved_geometry)),
                     dumps_strict(dict(moved_options)), row["id"]),
                )

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

    def get_group(self, group_id: str) -> dict | None:
        require_id(group_id, "group_id")
        with self.read() as connection:
            row = connection.execute(
                "SELECT * FROM groups WHERE id = ?", (group_id,)
            ).fetchone()
        return _row_or_none(row)

    def update_group(self, group_id: str, changes: Mapping[str, Any]) -> dict:
        """顶层 patch：只改提供的字段，未提供的保留。"""

        require_id(group_id, "group_id")
        if not isinstance(changes, Mapping):
            raise ValidationError("更新内容必须是对象", "")
        for key in changes:
            if key in ("id", "created_at", "updated_at", "project_id"):
                raise ValidationError("不允许修改该字段", key)
            if key != "name":
                raise ValidationError(f"不支持的字段：{key}", key)
        if not changes:
            raise ValidationError("没有可更新的字段", "")
        with self.transaction() as connection:
            cursor = connection.execute(
                "UPDATE groups SET name = ?, updated_at = ? WHERE id = ?",
                (_text(changes["name"], "name"), utc_now(), group_id),
            )
            if cursor.rowcount == 0:
                raise NotFoundError("分组不存在")
        record = self.get_group(group_id)
        assert record is not None
        return record

    def delete_group(self, group_id: str) -> None:
        """删除分组只解绑动物：动物记录保留，group_id 置空。"""

        require_id(group_id, "group_id")
        with self.transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM groups WHERE id = ?", (group_id,))
            if cursor.rowcount == 0:
                raise NotFoundError("分组不存在")

    def _validate_group(self, connection: sqlite3.Connection, project_id: str,
                        group_id: Any, field: str) -> str | None:
        """动物所属分组必须存在且与动物同项目。"""

        if group_id is None:
            return None
        require_id(group_id, field)
        row = connection.execute(
            "SELECT project_id FROM groups WHERE id = ?", (group_id,)
        ).fetchone()
        if row is None or row["project_id"] != project_id:
            raise ValidationError("分组不属于该项目", field)
        return group_id

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
            code = item.get("code")
            if code is not None:
                code = _text(code, f"animals[{index}].code", max_length=64,
                             allow_empty=True)
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
                    "code": code,
                    "group_id": group_id,
                    "sex": sex,
                    "birth_date": birth_date,
                    "notes": notes,
                }
            )

        with self.transaction() as connection:
            self._require_project(connection, project_id)
            for entry in prepared:
                self._validate_group(connection, project_id, entry["group_id"],
                                      "group_id")
                connection.execute(
                    "INSERT INTO animals(id, project_id, group_id, name, code, sex,"
                    " birth_date, notes, created_at, updated_at)"
                    " VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        entry["id"],
                        project_id,
                        entry["group_id"],
                        entry["name"],
                        entry["code"],
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

    def get_animal(self, animal_id: str) -> dict | None:
        require_id(animal_id, "animal_id")
        with self.read() as connection:
            row = connection.execute(
                """
                SELECT a.*, g.name AS group_name
                FROM animals a LEFT JOIN groups g ON g.id = a.group_id
                WHERE a.id = ?
                """,
                (animal_id,),
            ).fetchone()
        return _row_or_none(row)

    def update_animal(self, animal_id: str, changes: Mapping[str, Any]) -> dict:
        """顶层 patch；改组时同样检查同项目归属。"""

        require_id(animal_id, "animal_id")
        if not isinstance(changes, Mapping):
            raise ValidationError("更新内容必须是对象", "")
        allowed = {"name", "code", "group_id", "sex", "birth_date", "notes"}
        for key in changes:
            if key in ("id", "created_at", "updated_at", "project_id"):
                raise ValidationError("不允许修改该字段", key)
            if key not in allowed:
                raise ValidationError(f"不支持的字段：{key}", key)
        if not changes:
            raise ValidationError("没有可更新的字段", "")

        assignments: list[str] = []
        values: list[Any] = []
        if "name" in changes:
            assignments.append("name = ?")
            values.append(_text(changes["name"], "name", max_length=120))
        if "code" in changes:
            code = changes["code"]
            assignments.append("code = ?")
            values.append(None if code is None
                          else _text(code, "code", max_length=64,
                                     allow_empty=True))
        if "sex" in changes:
            sex = changes["sex"]
            assignments.append("sex = ?")
            values.append(None if sex is None
                          else _text(sex, "sex", max_length=32))
        if "birth_date" in changes:
            birth_date = changes["birth_date"]
            assignments.append("birth_date = ?")
            values.append(None if birth_date is None
                          else _text(birth_date, "birth_date", max_length=32))
        if "notes" in changes:
            notes = changes["notes"]
            assignments.append("notes = ?")
            values.append(None if notes is None
                          else _text(notes, "notes", max_length=2000,
                                     allow_empty=True))
        if "group_id" in changes:
            assignments.append("group_id = ?")
            values.append(changes["group_id"])

        assignments.append("updated_at = ?")
        values.append(utc_now())
        values.append(animal_id)

        with self.transaction() as connection:
            row = connection.execute(
                "SELECT project_id FROM animals WHERE id = ?", (animal_id,)
            ).fetchone()
            if row is None:
                raise NotFoundError("动物不存在")
            if "group_id" in changes:
                self._validate_group(connection, row["project_id"],
                                     changes["group_id"], "group_id")
            connection.execute(
                f"UPDATE animals SET {', '.join(assignments)} WHERE id = ?",
                values,
            )
        record = self.get_animal(animal_id)
        assert record is not None
        return record

    def delete_animal(self, animal_id: str) -> None:
        require_id(animal_id, "animal_id")
        with self.transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM animals WHERE id = ?", (animal_id,))
            if cursor.rowcount == 0:
                raise NotFoundError("动物不存在")

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
        asset_id: str | None = None,
    ) -> dict:
        asset_id = asset_id or new_id()
        require_id(asset_id, "asset_id")
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

    def list_assets_for_scope(self, project_id: str | None) -> list[dict]:
        """仅返回该归属下的资产；project_id 为 None 时只返回未绑定资产的记录。"""

        with self.read() as connection:
            if project_id is None:
                rows = connection.execute(
                    "SELECT * FROM assets WHERE project_id IS NULL ORDER BY created_at DESC"
                ).fetchall()
            else:
                require_id(project_id, "project_id")
                rows = connection.execute(
                    "SELECT * FROM assets WHERE project_id = ? ORDER BY created_at DESC",
                    (project_id,),
                ).fetchall()
        return [_decode_row(row, ("metadata",)) for row in rows]

    def find_asset_by_path(self, project_id: str | None, path: str) -> dict | None:
        folded = os.path.normcase(os.path.abspath(path)).casefold()
        for asset in self.list_assets_for_scope(project_id):
            if os.path.normcase(os.path.abspath(asset["path"])).casefold() == folded:
                return asset
        return None

    def find_asset_by_sha256(self, project_id: str | None, digest: str) -> dict | None:
        with self.read() as connection:
            if project_id is None:
                row = connection.execute(
                    "SELECT * FROM assets WHERE project_id IS NULL AND upload_sha256 = ?"
                    " ORDER BY created_at DESC LIMIT 1",
                    (digest,),
                ).fetchone()
            else:
                require_id(project_id, "project_id")
                row = connection.execute(
                    "SELECT * FROM assets WHERE project_id = ? AND upload_sha256 = ?"
                    " ORDER BY created_at DESC LIMIT 1",
                    (project_id, digest),
                ).fetchone()
        return _row_or_none(row, ("metadata",))

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

    DRAFT_JSON_FIELDS = ("videos", "options", "detection", "output", "style")

    def _draft_in(self, connection: sqlite3.Connection,
                  draft_id: str) -> dict | None:
        row = connection.execute(
            "SELECT * FROM drafts WHERE id = ?", (draft_id,)
        ).fetchone()
        return _row_or_none(row, self.DRAFT_JSON_FIELDS)

    def _draft_patch(self, changes: Mapping[str, Any]
                     ) -> tuple[list[str], list[Any]]:
        """顶层 patch 的列赋值；嵌套字典整体替换，不做引擎深合并。"""

        allowed = {"assay", "project_id", *self.DRAFT_JSON_FIELDS}
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
        for field in self.DRAFT_JSON_FIELDS[1:]:
            if field in changes:
                value = changes[field]
                if not isinstance(value, Mapping):
                    raise ValidationError(f"{field} 必须是对象", field)
                assignments.append(f"{field} = ?")
                values.append(_json_text(dict(value), field))
        return assignments, values

    def update_draft(self, draft_id: str, changes: Mapping[str, Any]) -> dict:
        require_id(draft_id, "draft_id")
        if not isinstance(changes, Mapping):
            raise ValidationError("更新内容必须是对象", "")
        assignments, values = self._draft_patch(changes)
        assignments.append("updated_at = ?")
        values.append(utc_now())
        values.append(draft_id)
        with self.transaction() as connection:
            if changes.get("project_id") is not None:
                self._require_project(connection, changes["project_id"])
            cursor = connection.execute(
                f"UPDATE drafts SET {', '.join(assignments)} WHERE id = ?", values
            )
            if cursor.rowcount == 0:
                raise NotFoundError("草稿不存在")
        record = self.get_draft(draft_id)
        assert record is not None
        return record

    def mutate_draft(self, draft_id: str,
                     updater: Callable[[dict], Mapping[str, Any] | None]) -> dict:
        """在 BEGIN IMMEDIATE 内读—改—写草稿，避免两次操作之间被插队。"""

        require_id(draft_id, "draft_id")
        with self.mutate() as connection:
            record = self._draft_in(connection, draft_id)
            if record is None:
                raise NotFoundError("草稿不存在")
            patch = updater(deepcopy(record))
            if patch is None:
                return record
            assignments, values = self._draft_patch(patch)
            assignments.append("updated_at = ?")
            values.append(utc_now())
            values.append(draft_id)
            if patch.get("project_id") is not None:
                self._require_project(connection, patch["project_id"])
            connection.execute(
                f"UPDATE drafts SET {', '.join(assignments)} WHERE id = ?", values
            )
            updated = self._draft_in(connection, draft_id)
            assert updated is not None
            return updated

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

    def list_jobs_for_project(self, project_id: str) -> list[dict]:
        require_id(project_id, "project_id")
        with self.read() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs WHERE project_id = ?"
                " ORDER BY created_at DESC",
                (project_id,),
            ).fetchall()
        return [_decode_row(row, ("config",)) for row in rows]

    # -- 模板 -----------------------------------------------------------
    def create_template(self, *, assay: str, name: str,
                        geometry: Mapping[str, Any] | None = None,
                        options: Mapping[str, Any] | None = None,
                        key: str | None = None) -> dict:
        """模板只保存场地几何与允许的尺寸，不带运行 ID 或媒体路径。"""

        template_id = new_id()
        now = utc_now()
        clean_assay = _text(assay, "assay", max_length=40)
        clean_geometry = dict(geometry or {})
        clean_options = dict(options or {})
        check_json_value(clean_geometry, "geometry")
        check_json_value(clean_options, "options")
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO templates(id, key, assay, name, payload, geometry,"
                " options, created_at, updated_at) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    template_id,
                    _text(key or clean_assay, "key", max_length=80),
                    clean_assay,
                    _text(name, "name", max_length=120),
                    "{}",
                    dumps_strict(clean_geometry),
                    dumps_strict(clean_options),
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
        record = _row_or_none(row, ("payload", "geometry", "options"))
        if record is not None:
            record.pop("payload", None)
        return record

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
        records = [_decode_row(row, ("payload", "geometry", "options"))
                   for row in rows]
        for record in records:
            record.pop("payload", None)
        return records

    def delete_template(self, template_id: str) -> None:
        require_id(template_id, "template_id")
        with self.transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM templates WHERE id = ?", (template_id,))
            if cursor.rowcount == 0:
                raise NotFoundError("模板不存在")

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
