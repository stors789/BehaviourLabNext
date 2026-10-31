import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from behaviourlab.config import DataPaths
from behaviourlab.store import Store
from behaviourlab.web.app import create_app


@pytest.fixture
def store(tmp_path):
    return Store(DataPaths.from_dir(tmp_path / "data"))


@pytest.fixture
def app(store):
    return create_app(data_dir=str(store.paths.root), store=store, testing=True)


@pytest.fixture
def client(app):
    with app.test_client() as test_client:
        yield test_client


@pytest.fixture
def project(store):
    return store.create_project("示例项目", "用于接口检查")
