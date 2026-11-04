# 交接摘要（第 7 组 1–26 块）

> **历史快照**：本文记录第 7 组 1–26 块交接时的状态，以下“未构建 Windows EXE”等描述仅对应当时阶段，保留作为历史记录，不代表当前构建状态。截至 2026-11-04，v1.0.1 已完成 Windows EXE 构建与 `--browser` 冒烟；完整 Windows 桌面验收仍未执行。当前构建记录见 [`RUN_AND_BUILD.md`](RUN_AND_BUILD.md)“当前构建状态”，逐项验收状态见 [`BUILD_VALIDATION_STATUS.json`](BUILD_VALIDATION_STATUS.json)。

本批交付：源码 + 构建配方 + 交接资料。**未构建 Windows EXE，可交付 EXE 数量为 0。**

## 运行 / 构建 / 故障入口

- 启动：`python -m behaviourlab --data-dir <专用目录>` 或 `python launch.py --data-dir <专用目录>`（同一主入口；`--browser`/`--port` 可选）。
- 构建：见 [`RUN_AND_BUILD.md`](RUN_AND_BUILD.md)（Windows x64 + CPython 3.11；spec = `packaging/behaviorlab_windows.spec`）。
- 故障：`logs/application.log` / `logs/startup.log` / `tasks/<job>/<run>/worker.log`（分工见 README 与 RUN_AND_BUILD）。
- 每类实验入口、参数与人工路径：[`ARCHITECTURE.md`](ARCHITECTURE.md)；页面功能清单与状态：[`ACCEPTANCE_STATUS.md`](ACCEPTANCE_STATUS.md)。

## 本批（24–26 阶段）实现变更

- **22**：manifest 有效性、关闭屏障与 HTTP 排空、进程异常收尾（terminate→wait→release）、删除归属核对。
- **23**：CSV/报告上下文、图形事实与缺失过滤、NOR 探索时序曲线、叠画完成段 cm→源像素映射、关键点容差整体失效、结果页（动物切换/文件栏/声明下载）。
- **24**：WebView2 运行时门槛（initialized 回调严格 bool + abort 语义）、数据根所有权与锁最终释放责任、stop=False 语义、Waitress 工厂所属资源显式收齐、选择器触发守卫、原生背景可解码确认、Bridge open_directory 业务错误、startup.log/错误框。
- **25**：`runtime.py` 私有分派（`--behaviorlab-worker`）、windowed worker 日志、根 `launch.py`、requirements-build.txt、`packaging/behaviorlab_windows.spec`（官方签名/入口绝对路径）。
- **26**：本批交接资料（README、ARCHITECTURE、RUN_AND_BUILD、ACCEPTANCE_STATUS、BUILD_VALIDATION_STATUS.json、HANDOFF）。

## 轻量检查结果（本批实际执行）

1. 全量 `pytest -q` 通过（实验数值、生命周期、发布边界、前端假 DOM、spec 声明）。
2. 13 个 Node 假 DOM 脚本通过。
3. 真实入口轻量检查：`--browser` 启动 health ok/data_root 一致/job_owner 就绪；私有 worker 路由（缺任务→1、数量不符→2）。

## 待验证 / 失败 / 阻塞

- **阻塞**：未构建 Windows EXE（缺真实 Windows x64 CPython 3.11 构建环境执行记录）；最小下一步见 RUN_AND_BUILD。
- **未验证**：真实浏览器 UI、Windows 原生（选择器/另存为/关闭回收/中文路径）、独立动物精度、真实素材导出——均 not_run，路线见 ACCEPTANCE_STATUS。
- 无 failed 项遗留：历史验收反馈（22/23/24/25 各单）均已修复并带定向自检。

## 提交与远端核对

- 本状态描述的代码修订：`311fe764df776ebf5f285dc485faf6fc4c13458b`（1–25 块）。
- 交接文档提交与最终 HEAD/远端 SHA：见 `BUILD_VALIDATION_STATUS.json` 的 `handoff` 字段（提交后回填并经 `git ls-remote` 核对一致）。
