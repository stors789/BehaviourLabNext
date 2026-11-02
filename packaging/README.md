# BehaviourLabNext Windows 构建（onedir）

目标平台：Windows 64 位 + CPython 3.11 + Microsoft Edge WebView2
Evergreen Runtime（系统级前提，不捆绑、不自动安装）。构建配方是
`packaging/behaviorlab_windows.spec`；Python 依赖版本由根目录
`requirements.txt` 锁定，构建工具由 `requirements-build.txt` 锁定。

视频导出使用 PyAV 进程内 FFmpeg 库（`av.libs` 下的 DLL 随包分发），
不附带任何外部命令行编码器。中文优先使用系统字体，Matplotlib 只带
自带 `mpl-data`，不复制再分发系统 Microsoft 字体。

## 构建命令（PowerShell，按顺序）

```powershell
py -3.11 -m venv .venv-build
.\.venv-build\Scripts\python.exe -m pip install -r requirements-build.txt -r requirements-dev.txt
.\.venv-build\Scripts\python.exe -m PyInstaller --distpath dist/windows --workpath build/windows packaging/behaviorlab_windows.spec
& .\dist\windows\BehaviourLabNext\BehaviourLabNext.exe --data-dir "$env:TEMP\BehaviourLabNext-Windows-QA"
```

说明：

- `dist/windows/BehaviourLabNext/` 整个目录才是候选交付物；单个 EXE
  从目录里取走不是完整包。
- 中间产物在 `build/windows`；`build/`、`dist/`、`.venv-build/` 均已
  排除 Git。已有输出目录不擅自清除；失败保留日志，明确所缺
  wheel/DLL/资源，而不是把部分目录称为完成包。
- 构建状态应记录：依赖版本、命令、退出码、产物路径与是否存在。
  完整包分发的许可证/第三方声明在实际发行前复核，配方本身不构成
  许可结论。

## 交付边界

- 私有后台 worker 参数固定为 `--behaviorlab-worker JOB STAGE
  PROGRESS`；EXE 后台 worker 不会递归启动 GUI/数据库/HTTP。
- 公开参数仍为 `--browser`、`--data-dir`、`--port`。
- 在没有真实 Windows 构建与启动证据前，本仓库不包含任何 EXE 产物；
  原生启动核验（页面/桥/解码编码/字体/中文路径/无控制台/关闭回收/
  下载另存为）按独立验收清单执行。
