# -*- mode: python ; coding: utf-8 -*-
"""BehaviourLabNext Windows 64 位 CPython 3.11 onedir 构建配方。

目标：dist/windows/BehaviourLabNext/BehaviourLabNext.exe 及整个同目录
（单目录才是候选交付物，单个 EXE 不是完整包）。构建只读项目源代码与
明确依赖，不递归复制仓库/虚拟环境/用户目录；用户媒体、数据库、密钥、
任务输出一律不进包。系统 Microsoft WebView2 与 .NET 是外部前提，不
捆绑、不自动安装。没有真实 Windows 构建与启动证据前，产物数量为 0。
"""

import sys
from pathlib import Path

def _fail(message: str) -> None:
    raise SystemExit(f"BehaviourLabNext 构建配方拒绝执行：{message}")

def _require_build_environment() -> None:
    """Windows 64 位 CPython 3.11 是必要条件；不伪装交叉构建。"""

    if sys.platform != "win32":
        _fail(f"需要 Windows（sys.platform=win32），当前 {sys.platform}")
    if sys.version_info[:2] != (3, 11):
        _fail("需要 CPython 3.11，当前 "
              f"{sys.version_info[0]}.{sys.version_info[1]}")
    if sys.maxsize <= 2 ** 31:
        _fail("需要 64 位解释器，当前为 32 位")

_require_build_environment()

try:
    _SPEC_DIR = Path(SPECPATH).resolve()  # PyInstaller 注入
except NameError:  # 静态检查/测试直接执行本文件时
    _SPEC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SPEC_DIR.parent

from PyInstaller.utils.hooks import (  # noqa: E402
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)

import av  # noqa: E402

# -- av.libs：PyAV Windows wheel 的 FFmpeg DLL 在包旁，缺失即失败 -------

AV_LIBS_DIR = Path(av.__file__).resolve().parent.parent / "av.libs"
if not AV_LIBS_DIR.is_dir():
    _fail(f"PyAV Windows wheel 缺少 FFmpeg DLL 目录：{AV_LIBS_DIR}")
av_libs_count = sum(1 for item in AV_LIBS_DIR.iterdir() if item.is_file())
if av_libs_count == 0:
    _fail(f"av.libs 目录为空，无法完成 PyAV 解码/编码：{AV_LIBS_DIR}")

# -- 应用 data：本地页面、脚本与样式，保持包内相对路径 ------------------

datas: list[tuple[str, str]] = []
for relative in ("behaviourlab/web/templates", "behaviourlab/web/static"):
    source = PROJECT_ROOT / relative
    if not source.is_dir():
        _fail(f"缺少应用资源目录：{source}")
    datas.append((str(source), relative))
# 不收测试素材/.evidence；不依据运行数据扩张 datas。

# -- 动态模块：实验注册按名称 import，必须显式收集 ----------------------

hiddenimports: list[str] = list(collect_submodules("behaviourlab.experiments"))
hiddenimports += [
    "behaviourlab",
    "behaviourlab.worker",
    "behaviourlab.runtime",
    "behaviourlab.launcher",
    "behaviourlab.web",
    "behaviourlab.desktop",
    # WebView2/Python.NET 依赖的隐藏导入
    "clr",
    "pythonnet",
    "clr_loader",
    "webview.platforms.winforms",
    "webview.platforms.win32",
    "webview.platforms.edgechromium",
    # 图形后端：只使用 Agg
    "matplotlib.backends.backend_agg",
]

# -- 官方 hooks 辅助 API：数据、元数据与动态库 --------------------------

datas += collect_data_files("webview", includes=["js/**/*", "lib/**/*"])
datas += collect_data_files("matplotlib", includes=["mpl-data/**/*"])
datas += collect_data_files("pythonnet", includes=["runtime/**/*"])
datas += collect_data_files("clr_loader")
for distribution in ("pywebview", "pythonnet", "clr_loader", "av",
                     "opencv-python", "numpy", "matplotlib"):
    datas += copy_metadata(distribution)
# 官方 6.22.3 API 为 collect_dynamic_libs(package, destdir=None,
# search_patterns=...)：一次一个包，逐包调用后合并。
binaries: list = []
for _package in ("webview", "pythonnet", "clr_loader", "av", "cv2"):
    binaries += collect_dynamic_libs(_package)

# PyAV 的 FFmpeg DLL 按包旁 av.libs 原布局进入 bundle。
for item in sorted(AV_LIBS_DIR.iterdir()):
    if item.is_file():
        datas.append((str(item), "av.libs"))

a = Analysis(
    # 官方 Analysis 把相对脚本相对 spec 所在目录解析：必须给绝对路径，
    # 否则会去找 packaging/launch.py。pathex 只影响模块搜索。
    [str(PROJECT_ROOT / "launch.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={"matplotlib": {"backends": ["agg"]}},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="BehaviourLabNext",
    debug=False,
    console=False,
    upx=False,
    strip=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="BehaviourLabNext",
)
