"""自检 1–2：注册表、导入边界、健康检查与数字/JSON 边界。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from behaviourlab import experiments
from behaviourlab.config import DataPaths, resolve_data_root
from behaviourlab.engine import COMMON_OUTPUT_DEFAULTS, COMMON_STYLE_DEFAULTS
from behaviourlab.experiments.base import ExperimentNotImplemented
from behaviourlab.schema import (
    AnalysisContext,
    Detection,
    ValidationError,
    check_json_value,
    dumps_strict,
    ensure_finite_tree,
    is_valid_id,
    make_observation,
    new_id,
    sanitize_json,
)
from behaviourlab.store import Store
from behaviourlab.web.app import create_app

EXPECTED_ORDER = ["tst", "fst", "mwm", "oft", "nor", "epm", "ymaze", "ldb",
                  "barnes", "ezm"]


def test_catalog_registers_ten_assays_once():
    entries = experiments.catalog()
    keys = [entry["key"] for entry in entries]
    assert keys == EXPECTED_ORDER
    assert keys.count("tst") == 1

    required = {"key", "label", "short", "parameters", "geometry", "metrics",
                "template_keys", "defaults", "detection_defaults"}
    for entry in entries:
        assert required.issubset(entry.keys())
        assert isinstance(entry["label"], str) and entry["label"]
        for parameter in entry["parameters"]:
            assert {"key", "label", "type", "default"}.issubset(parameter.keys())
            assert parameter["default"] == entry["defaults"][parameter["key"]]

    tst = next(entry for entry in entries if entry["key"] == "tst")
    assert tst["defaults"]["mode"] == "classic"
    mode = next(item for item in tst["parameters"] if item["key"] == "mode")
    assert [choice["value"] for choice in mode["choices"]] == ["classic", "lis"]


def test_import_creates_no_database_or_threads(tmp_path):
    local_appdata = tmp_path / "LocalAppData"
    code = """
import os, threading
before = threading.active_count()
import behaviourlab
from behaviourlab import schema, config, store, engine, media, geometry, common
from behaviourlab import detection, jobs, worker, processes, outputs, desktop
from behaviourlab import launcher, experiments
from behaviourlab.web import create_app
assert threading.active_count() == before, threading.active_count()
base = os.environ["LOCALAPPDATA"]
assert not os.path.exists(base), "import created local app data"
print("clean")
"""
    env = {**os.environ, "LOCALAPPDATA": str(local_appdata),
           "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout
    assert not local_appdata.exists()


def test_health_reports_explicit_data_root_and_no_owner(tmp_path):
    data_root = tmp_path / "custom-root"
    app = create_app(data_dir=str(data_root), testing=True)
    client = app.test_client()

    response = client.get("/api/health")
    payload = response.get_json()
    assert response.status_code == 200
    assert payload["ok"] is True
    data = payload["data"]
    assert data["status"] == "ok"
    assert Path(data["data_root"]) == data_root.resolve()
    assert Path(data["database"]).exists()
    assert data["job_owner"] is False
    assert (data_root / "videos").is_dir()
    assert (data_root / "logs").is_dir()


def test_unfinished_experiment_reports_not_implemented():
    # TST、FST、MWM、NOR、EPM、Y 迷宫与明暗箱已接入；用仍未实现的 Barnes 守住契约。
    context = AnalysisContext(job={"assay": "barnes"})
    with pytest.raises(ExperimentNotImplemented) as captured:
        experiments.get("barnes").analyze(context)
    assert "尚未实现" in str(captured.value)


def test_numeric_boundary_keeps_source_time_and_nulls():
    detection = Detection(center=None, missing_reason="质量不足")
    row = make_observation(
        unit_id=new_id(),
        unit_name="小鼠 1",
        animal_id=None,
        frame=4,
        time_s=0.4,
        source_dt_s=0.1,
        elapsed_s=0.4,
        analysis_fps=10.0,
        source_interval_start_s=0.4,
        source_interval_end_s=0.5,
        interval_start_s=0.45,
        interval_end_s=0.5,
        detection=detection,
    )

    assert abs(row["dt_s"] - 0.05) <= 1e-12
    assert row["frame"] == 4
    assert row["time_s"] == 0.4
    assert row["interval_start_s"] == 0.45
    assert row["detected"] is False
    assert row["x_px"] is None and row["y_px"] is None
    assert row["quality"] == 0
    assert row["nose_x_px"] is None and row["head_angle"] is None
    assert row["in_window"] is True

    encoded = json.dumps(row, ensure_ascii=False, allow_nan=False)
    assert '"x_px": null' in encoded


def test_nan_rejected_in_inputs_but_sanitized_in_results():
    with pytest.raises(ValidationError):
        ensure_finite_tree({"threshold": float("nan")}, "options")
    with pytest.raises(ValidationError):
        check_json_value({"threshold": float("inf")}, "detection")

    # 导出层清理：计算产生的非有限浮点归一为 null，而不是抛错。
    assert dumps_strict({"value": float("nan")}) == '{"value":null}'
    assert sanitize_json({"a": float("nan"), "b": [1.0, float("inf")]}) == {
        "a": None,
        "b": [1.0, None],
    }
    assert dumps_strict({"a": float("nan")}) == '{"a":null}'


def test_ids_are_opaque_and_bounded():
    assert len(new_id()) == 32
    assert is_valid_id("a-b_C-9")
    assert not is_valid_id("")
    assert not is_valid_id("a" * 101)
    assert not is_valid_id("../escape")
    assert not is_valid_id("路径")


def test_settings_reject_nan(tmp_path):
    store = Store(DataPaths.from_dir(tmp_path / "data"))
    with pytest.raises(ValidationError):
        store.set_setting("settings.style", {"dpi": float("nan")})


def test_common_defaults_are_stable():
    assert COMMON_OUTPUT_DEFAULTS == {"annotated_video": True}
    assert COMMON_STYLE_DEFAULTS == {
        "color": "#B86245",
        "colormap": "viridis",
        "line_width": 2,
        "dpi": 150,
    }


def test_localappdata_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))
    assert resolve_data_root(None) == tmp_path / "Local" / "BehaviourLabNext"
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    assert resolve_data_root(None) == (
        Path.home() / "AppData" / "Local" / "BehaviourLabNext"
    )
