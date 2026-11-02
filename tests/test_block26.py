"""第 26 块小型自检：注册键清单、交接状态样例与文档/路径一致性。

只做清单与一致性检查：不执行真实构建，不产生截图或大型验收报告。
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from behaviourlab.experiments import ORDER, REGISTRY

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
STATUS_PATH = DOCS / "BUILD_VALIDATION_STATUS.json"

EXPECTED_KEYS = ["tst", "fst", "mwm", "oft", "nor",
                 "epm", "ymaze", "ldb", "barnes", "ezm"]


def test_registry_keys_exactly_ten():
    """注册键恰等于固定清单且无重复；模式不占用注册键。"""

    keys = list(REGISTRY)
    assert keys == EXPECTED_KEYS == list(ORDER), keys
    assert len(keys) == len(set(keys)) == 10, "恰好十项且无重复"

    # TST 两模式、LDB 两可见性：是模式/参数，不是新注册键。
    from behaviourlab.experiments import ldb, tst

    assert tst.MODES == ("classic", "lis"), "TST 两种行为算法同属 tst"
    assert ldb.DEFAULTS["dark_box_visible"] is True, "LDB 可见/不可见是参数"
    assert len(REGISTRY) == 10

    # 实验模块文件与注册键一一对应（base 为共享模块）。
    module_files = sorted(path.stem for path in
                          (ROOT / "behaviourlab" / "experiments").glob("*.py"))
    assert module_files == sorted([*EXPECTED_KEYS, "base", "__init__"])


def test_status_json_not_run_scopes_stay_not_run():
    """交接状态样例：只有源码级检查通过时，不得推导其他范围通过。"""

    payload = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["product"] == "BehaviourLabNext"
    assert payload["target"] == "Windows"

    build = payload["build"]
    assert build["status"] == "not_run"
    assert build["artifact_count"] == 0, "EXE 数量为 0"
    assert build["artifact_path"] is None
    assert build["exit_code"] is None

    checks = payload["checks"]
    scopes = {check["scope"]: check["status"] for check in checks}
    assert scopes.get("unit_numeric") == "passed", "源码级检查已通过"
    for scope in ("real_browser_ui", "windows_native",
                  "independent_animal_accuracy"):
        assert scopes.get(scope) == "not_run", \
            f"{scope} 不能从 unit 通过推导"

    # 每个检查项都有完整结构；not_run 的 observed 明确写未执行。
    for check in checks:
        for field in ("id", "scope", "status", "environment",
                      "command_or_action", "expected", "observed",
                      "source_revision", "limitations"):
            assert field in check, (check.get("id"), field)
        if check["status"] == "not_run":
            assert "未执行" in check["observed"], check["id"]

    feature_keys = {item["assay"] for item in payload["feature_inventory"]
                    if item["assay"] != "(ui)"}
    assert feature_keys == set(EXPECTED_KEYS), "功能清单恰好十实验"


def test_docs_paths_and_sha_consistency():
    """文档可读、入口/资源真实存在、参数名与实现一致、SHA 核对一致。"""

    # 文档可读且非空。
    for name in ("README.md", "docs/ARCHITECTURE.md",
                 "docs/RUN_AND_BUILD.md", "docs/ACCEPTANCE_STATUS.md",
                 "docs/HANDOFF.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert len(text.strip()) > 100, name
    run_doc = (ROOT / "docs" / "RUN_AND_BUILD.md").read_text(
        encoding="utf-8")

    # 启动入口/worker/spec/static 资源真实存在。
    for relative in ("launch.py", "behaviourlab/__main__.py",
                     "behaviourlab/worker.py", "behaviourlab/runtime.py",
                     "requirements.txt", "requirements-build.txt",
                     "packaging/behaviorlab_windows.spec"):
        assert (ROOT / relative).is_file(), relative
    static_root = ROOT / "behaviourlab" / "web" / "static"
    for name in ("app.css", "app.js", "shell.js", "editor-core.js",
                 "frameview.js", "workspace.js", "pickers.js"):
        assert (static_root / name).is_file(), name

    # 文档参数名与实现一致。
    from behaviourlab import runtime
    from behaviourlab.launcher import build_parser

    assert runtime.WORKER_FLAG == "--behaviorlab-worker"
    assert runtime.WORKER_FLAG in run_doc
    options = vars(build_parser().parse_args([]))
    for flag in ("browser", "data_dir", "port"):
        assert flag in options
        assert f"--{flag.replace('_', '-')}" in run_doc or flag == "port"
    assert "--port" in run_doc

    spec_text = (ROOT / "packaging" / "behaviorlab_windows.spec").read_text(
        encoding="utf-8")
    assert "av.libs" in spec_text
    assert 'name="BehaviourLabNext"' in spec_text
    assert "dist/windows/BehaviourLabNext" in run_doc

    # 交接 SHA：记录的交付提交一致、是当前 HEAD 的祖先；远端 main 与
    # 本地 HEAD 实时核对一致才写发布完成（网络失败显式跳过，不编造）。
    payload = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    handoff = payload["handoff"]
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True,
                          check=True).stdout.strip()
    recorded = handoff["local_head_sha"]
    assert recorded != "PENDING_FINAL_COMMIT", "交付 SHA 已回填"
    assert handoff["remote_branch_sha"] == recorded, "本地与远端记录一致"
    assert handoff["consistent"] is True
    subprocess.run(["git", "merge-base", "--is-ancestor", recorded, head],
                   cwd=ROOT, check=True)
    assert payload["source_revision"] != recorded, \
        "source_revision 描述代码修订，交付 SHA 是其后的文档提交"
    remote = subprocess.run(
        ["git", "ls-remote", "origin", "refs/heads/main"],
        cwd=ROOT, capture_output=True, text=True, timeout=30)
    if remote.returncode == 0 and remote.stdout.strip():
        remote_sha = remote.stdout.split()[0]
        assert remote_sha == head, "远端 main 与本地 HEAD 一致"
