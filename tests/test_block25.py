"""第 25 块小型自检：私有分派、windowed worker 日志与构建配方声明。

不执行真实构建：产物数量为 0；spec 用假 PyInstaller 收集 API 做声明
检查并编译语法。真实 Windows 构建/DLL 加载/原生启动留待独立验收。
"""

from __future__ import annotations

import json
import re
import sys
import types
from pathlib import Path

import pytest

from behaviourlab import launcher as launcher_module
from behaviourlab import runtime
from behaviourlab.desktop import DesktopRuntimeError
from behaviourlab.experiments import REGISTRY
from behaviourlab.launcher import main
from behaviourlab.runtime import WORKER_FLAG

# -- 夹具 1：worker_command 与私有分派 ------------------------------------


def test_worker_command_source_and_frozen(monkeypatch):
    """源码命令 6 项、冻结 5 项；中文/空格路径逐参数不变。"""

    job = "C:/数据 一/job.json"
    stage = "C:/结果 二/.run.tmp"
    progress = "C:/数据 一/progress.json"

    monkeypatch.setattr(sys, "executable", "C:/程序 空间/python.exe")
    monkeypatch.delattr(sys, "frozen", raising=False)
    command = runtime.worker_command(job, stage, progress)
    assert command == ["C:/程序 空间/python.exe", "-m", "behaviourlab.worker",
                       job, stage, progress]
    assert len(command) == 6
    assert all(isinstance(item, str) for item in command)

    monkeypatch.setattr(sys, "executable", "C:/程序 空间/BehaviourLabNext.exe")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    frozen = runtime.worker_command(job, stage, progress)
    assert frozen == ["C:/程序 空间/BehaviourLabNext.exe", WORKER_FLAG,
                      job, stage, progress]
    assert len(frozen) == 5, "冻结模式不把 EXE 当 Python 使用 -m"
    assert not any(item == "-m" for item in frozen)


def test_private_flag_routes_to_worker_without_app(monkeypatch, capsys):
    """替身 worker 返回 7 → 入口返回 7；GUI/数据库/HTTP 调用 0 次。"""

    calls = {"app": 0, "service": 0, "worker": 0, "box": 0}

    def fake_create_app(*args, **kwargs):
        calls["app"] += 1
        return {}

    def fake_service(*args, **kwargs):
        calls["service"] += 1
        raise AssertionError("私有 worker 不允许构造 LocalService")

    def fake_worker_main(job, stage, progress):
        calls["worker"] += 1
        assert job == "C:/数据 一/job.json"
        assert stage == "C:/结果 二/.run.tmp"
        assert progress == "C:/数据 一/progress.json"
        return 7

    monkeypatch.setattr("behaviourlab.web.create_app", fake_create_app)
    monkeypatch.setattr(launcher_module, "LocalService", fake_service)
    monkeypatch.setattr("behaviourlab.worker.main", fake_worker_main)
    monkeypatch.setattr(launcher_module, "show_native_error",
                        lambda message: calls.__setitem__("box", calls["box"] + 1))

    code = main([WORKER_FLAG, "C:/数据 一/job.json",
                 "C:/结果 二/.run.tmp", "C:/数据 一/progress.json"])
    assert code == 7, "worker 退出码原样返回"
    assert calls == {"app": 0, "service": 0, "worker": 1, "box": 0}
    assert capsys.readouterr().out == ""


def test_private_flag_rejects_wrong_path_count(monkeypatch, capsys):
    """私有 flag 后仅 2 个路径：返回 2、worker 调用 0 次、stderr 短错。"""

    calls = {"worker": 0}

    def fake_worker_main(job, stage, progress):
        calls["worker"] += 1
        return 0

    monkeypatch.setattr("behaviourlab.worker.main", fake_worker_main)
    code = main([WORKER_FLAG, "C:/数据 一/job.json", "C:/结果 二/.run.tmp"])
    assert code == 2
    assert calls["worker"] == 0
    captured = capsys.readouterr()
    assert "WORKER_FLAG" in captured.err or WORKER_FLAG in captured.err
    assert captured.out == ""


# -- 夹具 2：windowed worker 日志与异常 -----------------------------------


def test_frozen_worker_logging_and_recovery(monkeypatch, tmp_path):
    """stdout/stderr 缺失时以任务 worker.log 补流：追加、恢复、可读错误。"""

    job_dir = tmp_path / "任务 一"
    job_dir.mkdir()
    log = job_dir / "worker.log"
    log.write_text("预存一行\n", encoding="utf-8")
    job = job_dir / "job.json"
    job.write_text("{}", encoding="utf-8")
    stage = tmp_path / "结果 二" / ".run.tmp"
    progress = tmp_path / "数据 一" / "progress.json"

    seen = {}

    def fake_worker_main(job_arg, stage_arg, progress_arg):
        seen["args"] = (job_arg, stage_arg, progress_arg)
        raise RuntimeError("分析前的中文失败：无法读取任务")

    boxes: list[str] = []
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    monkeypatch.setattr("behaviourlab.worker.main", fake_worker_main)
    monkeypatch.setattr(launcher_module, "show_native_error",
                        lambda message: boxes.append(message))

    code = main([WORKER_FLAG, str(job), str(stage), str(progress)])
    assert code == 1
    text = log.read_text(encoding="utf-8")
    assert text.splitlines()[0] == "预存一行", "预存日志前行保留"
    assert "Traceback" in text and "RuntimeError" in text
    assert "分析前的中文失败" in text, "traceback 以 UTF-8 追加到 worker.log"
    payload = json.loads(progress.read_text(encoding="utf-8"))
    assert "分析前的中文失败" in payload["error"], "progress.error 可读"
    assert sys.stdout is None and sys.stderr is None, "标准流恢复为 None"
    assert boxes == [], "私有 worker 不触发桌面错误框"
    assert seen["args"] == (str(job), str(stage), str(progress))


def test_frozen_worker_success_and_unwritable_log(monkeypatch, tmp_path):
    """worker 正常返回 0 原样返回；日志不可写也不把失败标成功。"""

    # 正常路径：worker 返回 0 → 入口返回 0，progress 不被改写。
    ok_dir = tmp_path / "正常 任务"
    ok_dir.mkdir()
    ok_progress = tmp_path / "ok-progress.json"
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    monkeypatch.setattr("behaviourlab.worker.main",
                        lambda job, stage, progress: 0)
    assert main([WORKER_FLAG, str(ok_dir / "job.json"),
                 str(tmp_path / "stage"), str(ok_progress)]) == 0
    assert not ok_progress.exists()
    assert (ok_dir / "worker.log").exists()

    # 日志不可写（父目录是文件）：worker 失败仍返回 1，进度错误可读。
    monkeypatch.setattr(
        "behaviourlab.worker.main",
        lambda job, stage, progress: (_ for _ in ()).throw(
            RuntimeError("依赖导入失败")))
    blocked = tmp_path / "blocked"
    blocked.write_text("占用", encoding="utf-8")
    progress2 = tmp_path / "progress2.json"
    code = main([WORKER_FLAG, str(blocked / "job.json"),
                 str(tmp_path / "stage2"), str(progress2)])
    assert code == 1, "日志失败不得把执行标成功"
    payload = json.loads(progress2.read_text(encoding="utf-8"))
    assert "依赖导入失败" in payload["error"]
    assert sys.stdout is None and sys.stderr is None


def test_frozen_worker_import_failure_reaches_log_and_progress(
        monkeypatch, tmp_path):
    """精确在 worker 导入处注入失败：必须进入日志/进度边界。

    worker 顶层导入 outputs 与科学依赖，导入本身的 ImportError/DLL
    失败不能逃出 main——要先补齐缺失标准流，再在同一异常边界内导入
    并调用；失败写 traceback 与 progress.error 后返回 1，不弹错误框。
    """

    job_dir = tmp_path / "任务 一"
    job_dir.mkdir()
    log = job_dir / "worker.log"
    log.write_text("预存一行\n", encoding="utf-8")
    job = job_dir / "job.json"
    job.write_text("{}", encoding="utf-8")
    progress = tmp_path / "progress.json"

    class _BoomFinder:
        """worker 导入即抛受控 ImportError，模拟真实 DLL/依赖失败。"""

        def find_spec(self, fullname, path=None, target=None):
            if fullname == "behaviourlab.worker":
                raise ImportError("受控依赖 DLL 导入失败")
            return None

    boxes: list[str] = []
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    # from-import 会优先命中父包属性：必须同时清除模块缓存与属性，
    # 让导入真正走 meta_path，在 worker 导入处精确注入失败。
    monkeypatch.delitem(sys.modules, "behaviourlab.worker")
    monkeypatch.delattr(sys.modules["behaviourlab"], "worker", raising=False)
    monkeypatch.setattr(sys, "meta_path",
                        [_BoomFinder(), *sys.meta_path])
    monkeypatch.setattr(launcher_module, "show_native_error",
                        lambda message: boxes.append(message))

    code = main([WORKER_FLAG, str(job), str(tmp_path / "stage"),
                 str(progress)])
    assert code == 1, "导入失败返回 1，不逃出 main"
    text = log.read_text(encoding="utf-8")
    assert text.splitlines()[0] == "预存一行", "预存日志前行保留"
    assert "Traceback" in text
    assert "受控依赖 DLL 导入失败" in text, "导入失败进入 worker.log"
    payload = json.loads(progress.read_text(encoding="utf-8"))
    assert payload["error"].startswith("ImportError")
    assert "受控依赖 DLL 导入失败" in payload["error"], "progress.error 可读"
    assert sys.stdout is None and sys.stderr is None, "标准流恢复为 None"
    assert boxes == [], "不触发桌面错误框"


# -- 夹具 3：构建配方声明检查 ---------------------------------------------

SPEC_PATH = Path(launcher_module.__file__).resolve().parents[1] / (
    "packaging/behaviorlab_windows.spec")


def _spec_globals(recorder):
    spec_dir = SPEC_PATH.parent

    def fake_analysis(scripts, **kwargs):
        # 约束官方 6.22.3 路径语义：相对脚本相对 spec 所在目录解析，
        # 且文件必须存在（pathex 不影响脚本路径）。
        resolved = []
        for script in scripts:
            candidate = Path(script)
            if not candidate.is_absolute():
                candidate = spec_dir / candidate
            if not candidate.is_file():
                raise SystemExit(f"ERROR: script '{candidate}' not found")
            resolved.append(str(candidate))
        recorder["analysis_scripts"] = resolved
        recorder["analysis_kwargs"] = kwargs
        return types.SimpleNamespace(pure=object(), scripts=[], binaries=[],
                                     datas=[])

    fake_hooks = types.ModuleType("PyInstaller.utils.hooks")

    def collect_submodules(package):
        recorder["submodules"].append(package)
        return [f"{package}.{name}" for name in sorted(REGISTRY)]

    def collect_data_files(package, includes=None, **kwargs):
        recorder["data_files"].append((package, list(includes or [])))
        return []

    def copy_metadata(distribution):
        recorder["metadata"].append(distribution)
        return []

    # 官方签名 collect_dynamic_libs(package, destdir=None,
    # search_patterns=...)：一次只接受一个包；多包位置参数必须 TypeError。
    def collect_dynamic_libs(package, destdir=None, search_patterns=None):
        recorder["dynamic_libs"].append(package)
        return []

    fake_hooks.collect_submodules = collect_submodules
    fake_hooks.collect_data_files = collect_data_files
    fake_hooks.copy_metadata = copy_metadata
    fake_hooks.collect_dynamic_libs = collect_dynamic_libs
    fake_utils = types.ModuleType("PyInstaller.utils")
    fake_utils.hooks = fake_hooks
    fake_pyinstaller = types.ModuleType("PyInstaller")
    fake_pyinstaller.utils = fake_utils
    spec_globals = {
        "SPECPATH": str(spec_dir),
        "Analysis": fake_analysis,
        "PYZ": lambda *args, **kwargs: object(),
        "EXE": lambda *args, **kwargs: recorder["exe"].update(
            args=args, kwargs=kwargs) or object(),
        "COLLECT": lambda *args, **kwargs: recorder["collect"].update(
            args=args, kwargs=kwargs) or object(),
    }
    return spec_globals, fake_pyinstaller, fake_utils, fake_hooks


def _run_spec(monkeypatch, recorder, *, platform="win32",
              version=(3, 11, 9, "final", 0), maxsize=None,
              av_file=None):
    import av as av_module

    fake_globals, fake_pyinstaller, fake_utils, fake_hooks = _spec_globals(
        recorder)
    monkeypatch.setitem(sys.modules, "PyInstaller", fake_pyinstaller)
    monkeypatch.setitem(sys.modules, "PyInstaller.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "PyInstaller.utils.hooks", fake_hooks)
    if av_file is not None:
        monkeypatch.setattr(av_module, "__file__", av_file)
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(sys, "version_info", version)
    if maxsize is not None:
        monkeypatch.setattr(sys, "maxsize", maxsize)
    source = SPEC_PATH.read_text(encoding="utf-8")
    exec(compile(source, str(SPEC_PATH), "exec"), fake_globals)


def test_spec_rejects_wrong_platforms(monkeypatch):
    """Linux、32 位或 Python 3.12 一律拒绝，不伪装交叉构建。"""

    for kwargs in ({"platform": "linux"},
                   {"version": (3, 12, 0, "final", 0)},
                   {"maxsize": 2 ** 31 - 1}):
        recorder = _fresh_recorder()
        with pytest.raises(SystemExit, match="拒绝执行"):
            _run_spec(monkeypatch, recorder, **kwargs)
        assert recorder["analysis_scripts"] is None, "拒绝时不进入收集"


def _fresh_recorder():
    return {"submodules": [], "data_files": [], "metadata": [],
            "dynamic_libs": [], "analysis_scripts": None,
            "analysis_kwargs": {}, "exe": {}, "collect": {}}


def test_spec_accepts_windows_64_python311(monkeypatch):
    """Windows 64 位 3.11 条件接受；资源/动态模块/EXE/COLLECT 声明齐全。"""

    recorder = _fresh_recorder()
    _run_spec(monkeypatch, recorder)

    project_root = SPEC_PATH.parents[1]
    expected_entry = str(project_root / "launch.py")
    # 入口按官方语义解析为根目录 launch.py 的绝对路径且真实存在。
    assert recorder["analysis_scripts"] == [expected_entry]
    assert Path(expected_entry).is_file()

    analysis_kwargs = recorder["analysis_kwargs"]
    assert str(project_root) in analysis_kwargs["pathex"]

    datas = analysis_kwargs["datas"]
    destinations = {destination for _, destination in datas}
    assert "behaviourlab/web/templates" in destinations
    assert "behaviourlab/web/static" in destinations
    assert "av.libs" in destinations
    av_lib_entries = [source for source, destination in datas
                      if destination == "av.libs"]
    assert av_lib_entries, "av.libs 的 FFmpeg DLL 按原布局进入 bundle"

    hidden = analysis_kwargs["hiddenimports"]
    assert "behaviourlab.worker" in hidden
    for key in REGISTRY:
        assert f"behaviourlab.experiments.{key}" in hidden
    for required in ("clr", "pythonnet", "clr_loader",
                     "webview.platforms.winforms", "webview.platforms.win32",
                     "webview.platforms.edgechromium",
                     "matplotlib.backends.backend_agg"):
        assert required in hidden, required

    assert recorder["submodules"] == ["behaviourlab.experiments"]
    data_calls = {}
    for package, includes in recorder["data_files"]:
        data_calls.setdefault(package, []).append(includes)
    assert ["js/**/*", "lib/**/*"] in data_calls["webview"]
    assert ["mpl-data/**/*"] in data_calls["matplotlib"]
    assert ["runtime/**/*"] in data_calls["pythonnet"]
    assert [] in data_calls["clr_loader"]
    assert set(recorder["metadata"]) == {"pywebview", "pythonnet",
                                         "clr_loader", "av",
                                         "opencv-python", "numpy",
                                         "matplotlib"}
    assert recorder["dynamic_libs"] == ["webview", "pythonnet",
                                        "clr_loader", "av", "cv2"], \
        "5 包各调用一次官方单包签名"
    assert analysis_kwargs["hooksconfig"] == {
        "matplotlib": {"backends": ["agg"]}}

    exe_kwargs = recorder["exe"]["kwargs"]
    assert exe_kwargs["exclude_binaries"] is True
    assert exe_kwargs["name"] == "BehaviourLabNext"
    assert exe_kwargs["console"] is False
    assert exe_kwargs["debug"] is False
    assert exe_kwargs["upx"] is False
    assert exe_kwargs["strip"] is False
    collect_args = recorder["collect"]["args"]
    collect_kwargs = recorder["collect"]["kwargs"]
    assert collect_kwargs["name"] == "BehaviourLabNext"
    assert len(collect_args) == 3, "COLLECT 收 EXE、binaries、datas"


def test_spec_rejects_missing_av_libs(monkeypatch, tmp_path):
    """av.libs 缺失时明确构建失败，不安静跳过。"""

    recorder = _fresh_recorder()
    fake_av = str(tmp_path / "nope" / "av" / "__init__.py")
    with pytest.raises(SystemExit, match="av.libs"):
        _run_spec(monkeypatch, recorder, av_file=fake_av)
    assert recorder["analysis_scripts"] is None, "失败时不产出任何声明"


def test_spec_compiles_and_template_assets_covered():
    """spec 语法可编译；真实模板引用的本地脚本/样式都在 static 可提供。"""

    compile(SPEC_PATH.read_text(encoding="utf-8"), str(SPEC_PATH), "exec")

    root = Path(launcher_module.__file__).resolve().parents[1]
    web = root / "behaviourlab" / "web"
    static_root = web / "static"
    static_refs: set[str] = set()
    for template in (web / "templates").rglob("*.html"):
        text = template.read_text(encoding="utf-8")
        static_refs.update(re.findall(
            r"url_for\(['\"]static['\"],\s*filename=['\"]([^'\"]+)['\"]\)",
            text))
        static_refs.update(re.findall(
            r"(?:src|href)=\"/static/([^\"]+)\"", text))
    assert static_refs, "真实模板确实引用本地静态资源"
    for reference in sorted(static_refs):
        assert (static_root / reference).is_file(), reference

    # frameview 必须先于 workspace 可用。
    for template in (web / "templates").rglob("*.html"):
        text = template.read_text(encoding="utf-8")
        if "workspace.js" in text:
            assert "frameview.js" in text, template
            assert text.index("frameview.js") < text.index("workspace.js"), \
                f"{template}: frameview 应先于 workspace 加载"


def test_entry_points_delegate_with_freeze_support():
    """launch.py 与 python -m 入口先 freeze_support 再进同一 main。"""

    root = Path(launcher_module.__file__).resolve().parents[1]
    for entry in (root / "launch.py", root / "behaviourlab" / "__main__.py"):
        source = entry.read_text(encoding="utf-8")
        assert "multiprocessing.freeze_support()" in source, entry
        assert "sys.exit(main())" in source, entry
    compile((root / "launch.py").read_text(encoding="utf-8"),
            str(root / "launch.py"), "exec")
