"""运行配置与数据根解析。

本模块只描述路径与常量；导入不创建目录、数据库、窗口或后台资源。
数据根默认位于 Windows ``LOCALAPPDATA/BehaviourLabNext``，代码目录与原片目录
都不作为数据库或输出的默认位置。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

APP_NAME = "BehaviourLabNext"
APP_VERSION = "0.1.0"
SCHEMA_VERSION = 1
ALGORITHM_VERSION = "0.1.0"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

DATABASE_FILENAME = "behaviourlab.sqlite3"
SUBDIRECTORIES = ("videos", "backgrounds", "previews", "runs", "logs", "tasks")

# 指定依赖契约，不是“最新版本”声明。
PINNED_DEPENDENCIES = {
    "Flask": "3.1.3",
    "numpy": "2.4.6",
    "opencv-python": "5.0.0.93",
    "av": "18.1.0",
    "matplotlib": "3.11.2",
    "pandas": "3.1.0",
    "pywebview": "6.2.1",
    "waitress": "3.0.2",
    "imageio-ffmpeg": "0.7.0",
    "pytest": "9.2.0",
}


def default_data_root() -> Path:
    """返回未显式指定 data-dir 时使用的独立数据目录。"""

    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        return Path(local_appdata) / APP_NAME
    return Path.home() / "AppData" / "Local" / APP_NAME


def resolve_data_root(data_dir: str | os.PathLike[str] | None = None) -> Path:
    """解析数据根；显式路径优先，否则回退到本地应用数据目录。"""

    if data_dir is None or str(data_dir).strip() == "":
        return default_data_root()
    return Path(data_dir).expanduser().resolve()


@dataclass(frozen=True)
class DataPaths:
    """一个数据根下的固定目录布局。"""

    root: Path

    @classmethod
    def from_dir(cls, data_dir: str | os.PathLike[str] | None = None) -> "DataPaths":
        return cls(resolve_data_root(data_dir))

    @property
    def database(self) -> Path:
        return self.root / DATABASE_FILENAME

    @property
    def videos(self) -> Path:
        return self.root / "videos"

    @property
    def backgrounds(self) -> Path:
        return self.root / "backgrounds"

    @property
    def previews(self) -> Path:
        return self.root / "previews"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def tasks(self) -> Path:
        return self.root / "tasks"

    def ensure(self) -> "DataPaths":
        self.root.mkdir(parents=True, exist_ok=True)
        for name in SUBDIRECTORIES:
            (self.root / name).mkdir(parents=True, exist_ok=True)
        return self

    def as_dict(self) -> dict:
        return {
            "data_root": str(self.root),
            "database": str(self.database),
            "videos": str(self.videos),
            "backgrounds": str(self.backgrounds),
            "previews": str(self.previews),
            "runs": str(self.runs),
            "logs": str(self.logs),
            "tasks": str(self.tasks),
        }
