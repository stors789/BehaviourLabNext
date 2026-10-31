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
- `behaviourlab/geometry.py`：画布映射、单应、形状包含与方向变换
- `behaviourlab/detection.py`：经典前景、检测区域与候选质量
- `behaviourlab/preview.py`：真实源帧叠画预览
- `behaviourlab/experiments/`：十项实验注册、参数与几何步骤
- `behaviourlab/engine.py`：配置合并、冻结与校验
- `behaviourlab/service.py`：本地服务所有权、就绪检查与清理
- `behaviourlab/desktop.py`：窗口桥与原生文件选择
- `behaviourlab/web/`：本地 API 与页面
- `tests/`：轻量契约检查

## 分析流程

`/analyze?mode=project|independent&draft_id=…&step=select|range|arena&view=geometry|detection`

1. **视频**：选择实验与项目，导入或勾选素材，创建草稿。
2. **场地**：逐视频设置起止时间、方向、解释帧率与采样帧率，可应用到其他视频。
3. **场地/检测**：画布逐帧绘制几何，检查器列出该实验必需的步骤；编辑后 600 ms 防抖保存。
4. 检测阶段生成预览图，确认后进入分析校验。

几何保存坐标是当前方向下的原分辨率像素；画布缩放、留白与 DPR 只影响显示。
预览接口为 `POST /api/drafts/<id>/preview`，返回真实源帧叠画的 PNG 与范围说明。
