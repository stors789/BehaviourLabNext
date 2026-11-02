# BehaviourLabNext

Windows 本地经典视觉动物行为分析应用（不含深度学习依赖）。本地服务与桌面窗口共用同一套服务与前端；正式桌面交付目标为 Windows 64 位 + CPython 3.11 + Microsoft Edge WebView2（系统级前提）。

- 当前构建状态：**未构建 Windows EXE，可交付 EXE 数量为 0**。构建方法与边界见 [`docs/RUN_AND_BUILD.md`](docs/RUN_AND_BUILD.md)；结构与逐项证据见 [`docs/BUILD_VALIDATION_STATUS.json`](docs/BUILD_VALIDATION_STATUS.json) 与 [`docs/ACCEPTANCE_STATUS.md`](docs/ACCEPTANCE_STATUS.md)。
- 当前已知阻塞：真实 Windows 64 位 CPython 3.11 构建环境未就绪；真实浏览器 UI 验收与独立动物精度证据未执行（详见状态文档，均不阻塞源码使用）。

## 环境与安装

- Python 3.11；依赖锁定见 `requirements.txt`，测试依赖见 `requirements-dev.txt`，构建工具见 `requirements-build.txt`。

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
```

## 启动（两个源码入口，同一主入口）

```bash
.venv\Scripts\python -m behaviourlab --data-dir "D:\BehaviourLabData"
.venv\Scripts\python launch.py --data-dir "D:\BehaviourLabData"
```

- 普通启动创建桌面窗口（需要 WebView2）；`--browser` 只运行同一服务等待用户中断。
- 未指定 `--port` 时自动申请空闲端口；服务只绑定 `127.0.0.1`。
- 未指定 `--data-dir` 时使用 `%LOCALAPPDATA%\BehaviourLabNext`；同一数据根同一时刻只允许一个实例（含排他锁）。
- `--behaviorlab-worker` 是冻结 EXE 的私有后台分派参数，不作为用户入口。

## 界面入口

首页选择**项目**（项目/组/动物上下文）或**独立分析**（项目外单次分析）。分析页顺序：选择视频 → 时间/方向/帧率 → 场地绘制或人工记录（依实验而定） → 检测设置与预览 → 开始分析；完成后从任务中心或项目页进入结果页，取回 CSV/PNG/MP4/ZIP。各实验的入口与人工路径差异见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

## 数据只读原则

原始视频文件只读；分析使用冻结配置写入数据根的 `runs/`、`tasks/`、`database/` 等专用目录。用户媒体、数据库、密钥与任务输出不进入版本库，也不进入构建包。

## 故障定位

- `logs/application.log`：服务运行日志。
- `logs/startup.log`：启动失败 cause/traceback（配 Windows 错误框）。
- `tasks/<job>/<run>/worker.log`：单个后台分析进程日志。

构建失败与 Windows 原生验收待办见 [`docs/ACCEPTANCE_STATUS.md`](docs/ACCEPTANCE_STATUS.md)。
