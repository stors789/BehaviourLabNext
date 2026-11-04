# 运行与构建说明

## 源码运行（Windows/Linux 开发均可）

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m behaviourlab --data-dir "D:\BehaviourLabData 专用目录"
.venv\Scripts\python launch.py --data-dir "D:\BehaviourLabData 专用目录"
```

- 两个入口委托同一 `behaviourlab.launcher.main`；`--browser` 只起服务、`--port` 指定端口（默认自动空闲端口，仅绑定 127.0.0.1）。
- 正式桌面窗口需要 **Microsoft Edge WebView2 Evergreen Runtime** 及对应 .NET（pywebview 6.2.1 要求 .NET Framework 4.6.2+）。缺失/损坏时程序给出含官方修复入口 `https://developer.microsoft.com/microsoft-edge/webview2/` 的可行动错误；不自动下载或安装系统运行时。
- 数据根独立：`--data-dir` 专用目录（默认 `%LOCALAPPDATA%\BehaviourLabNext`）。同一数据根同时只允许一个实例。

## 日志分工

| 文件 | 内容 |
| --- | --- |
| `<数据根>/logs/application.log` | 服务运行日志（UTF-8） |
| `<数据根>/logs/startup.log` | 启动失败的完整 cause/traceback |
| `<数据根>/tasks/<job>/<run>/worker.log` | 单个后台分析进程日志（含冻结 windowed 下缺失标准流的输出） |
| `<数据根>/server.json` | 就绪后的 `{url, data_root, pid}` 记录 |

私有参数 `--behaviorlab-worker JOB STAGE PROGRESS` 只作冻结 EXE 的后台工作进程路由：源码模式下调度器用 `python -m behaviourlab.worker` 分派，冻结模式复用同一 EXE；它不创建数据根、不弹启动错误框，不是用户入口。

## Windows 单目录构建（64 位 CPython 3.11）

前置：Windows x64 + CPython 3.11；系统 WebView2/.NET 是运行前提，不随包分发。

```powershell
py -3.11 -m venv .venv-build
.\.venv-build\Scripts\python.exe -m pip install -r requirements-build.txt -r requirements-dev.txt
.\.venv-build\Scripts\python.exe -m PyInstaller --distpath dist/windows --workpath build/windows packaging/behaviorlab_windows.spec
& .\dist\windows\BehaviourLabNext\BehaviourLabNext.exe --data-dir "$env:TEMP\BehaviourLabNext-Windows-QA"
```

- 配方：`packaging/behaviorlab_windows.spec`。平台核对（win32/CPython 3.11/64 位）不满足即拒绝；PyAV 的 FFmpeg DLL 来自包旁 `av.libs`（缺失即构建失败）。
- 包内资源：`behaviourlab/web/templates` 与 `behaviourlab/web/static`（全部本地 HTML/JS/CSS）、matplotlib `mpl-data`（Agg）、webview/js/lib、pythonnet runtime、clr_loader 数据；依赖元数据与动态库按官方 hooks API 收集。
- 产物：`dist/windows/BehaviourLabNext/` **整个目录**才是候选交付物；`BehaviourLabNext.exe` 单独取出不是完整包。
- `build/`、`dist/`、`.venv-build/` 不入 Git；已有输出目录不擅自清除。

## 当前构建状态

**最近构建（2026-11-04，fff5958）：成功。** 上述配方在本机实际执行
（`.venv-build` 基于 CPython 3.11.13，PyInstaller 6.22.3，exit 0），
产物 `dist/windows/BehaviourLabNext/`（≈288 MB，`BehaviourLabNext.exe` +
`_internal`）。冒烟：`--browser --data-dir <临时目录>` 启动后
`/api/health` ok（version 1.0.0）、首页正常渲染；本次为预览背景异常分类
修复后的重建，打包为真 ZIP（≈122 MB）分发，**未更新** Release 附件——
已发布的是 [v1.0.0](https://github.com/stors789/BehaviourLabNext/releases/tag/v1.0.0)
（22da64e 构建）。
`build/`、`dist/`、`.venv-build/` 不入
Git。完整 Windows 桌面验收（WebView2 窗口、原生选择器、冻结 worker
路由）仍未执行，构建与冒烟记录见
`docs/BUILD_VALIDATION_STATUS.json`（check `exe-build-smoke`）。
