# BehaviourLabNext

Windows 本地经典视觉动物行为分析应用。本地服务与桌面窗口共用同一套服务与前端。

## 环境

- Python 3.11
- 依赖锁定见 `requirements.txt`，测试依赖见 `requirements-dev.txt`
- `imageio-ffmpeg` 安装版本记录为 0.7.0

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
```

## 启动

```bash
.venv\Scripts\python -m behaviourlab --data-dir "D:\BehaviourLabData"
```

普通启动创建桌面窗口；`--browser` 只运行同一服务，等待用户中断。
未指定 `--port` 时自动申请空闲端口；服务只绑定 `127.0.0.1`。
未指定 `--data-dir` 时使用 `%LOCALAPPDATA%\BehaviourLabNext`。

## 结构

- `behaviourlab/schema.py`：跨层数据契约
- `behaviourlab/config.py`：数据根与运行常量
- `behaviourlab/store.py`：SQLite 持久化与原子 JSON 写入
- `behaviourlab/assets.py`：路径导入、上传副本、背景图
- `behaviourlab/media.py`：视频探测、源帧读取、解释时钟与采样
- `behaviourlab/experiments/`：十项实验注册与描述
- `behaviourlab/engine.py`：配置合并、冻结与校验
- `behaviourlab/service.py`：本地服务所有权、就绪检查与清理
- `behaviourlab/desktop.py`：窗口桥与原生文件选择
- `behaviourlab/web/`：本地 API 与页面
- `tests/`：轻量契约检查
