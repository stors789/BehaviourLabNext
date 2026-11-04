# 状态与验收记录（可读摘要）

机读版本：[`BUILD_VALIDATION_STATUS.json`](BUILD_VALIDATION_STATUS.json)。规则：**implemented** = 代码存在并连通；**verified** = 有对应检查证据；失败保持 failed；未执行保持 not_run/未验证。不同证据层级不合并。

## 证据层级现状（修订 311fe764df776ebf5f285dc485faf6fc4c13458b 及 v1.0.1 实机补充）

| 层级 | 状态 | 证据 |
| --- | --- | --- |
| 单元/数值（unit_numeric） | verified（源码级） | 全量 pytest 通过；13 个 Node 假 DOM 脚本通过；spec 声明检查通过 |
| 进程/服务（service_process） | verified（源码级） | `--browser` 启动 + `/api/health` 一致；私有 worker 真实路由（缺任务→1、数量不符→2）；生命周期/关闭链测试 |
| 真实引擎导出（real_engine_export） | verified | PyAV 编码-解码往返（帧数/PTS/时长）通过；并在 Windows 11 实机完成旷场与 Y 迷宫真实录像 `annotated.mp4` 导出与结果页回放验证 |
| 真实桌面窗口工作流（desktop_e2e_workflow） | **verified（实机主流程）** | Windows 11 WebView2 原生桌面窗口实跑通过：项目管理、导入真实双鼠录像（`旷场适应_前3分钟.mp4`）、四角透视标定与模板保存、时序中位数预览、后台 Worker 调度、结果页图表与叠画视频回放全链路跑通（全页面自动化遍历矩阵仍标 `not_run`） |
| Windows 原生构建与冒烟（windows_native） | **verified（构建+冒烟+桌面实跑）** | PyInstaller 6.22.3 / CPython 3.11.13 构建成功，产物 `dist/windows/BehaviourLabNext/`（≈288 MB，exit 0）；`--browser` 冒烟与 WebView2 桌面主流程实测通过（极端中文/空格 cwd 等全矩阵边界项在机读 JSON 中保留 `not_run`） |
| 独立动物精度（independent_animal_accuracy） | **not_run** | 无独立标注金标准；人工记录不代表更准确（标准可能漂移，有时漏和多），不改变本项状态 |
| 真实素材人工对照（user_manual_comparison） | **verified（用户实跑）** | Y 迷宫中心点规则 2 run（80da94b，用户自有录像）：小鼠 1 人工 vs 自动入臂序列相似率 92.68% / 95.24%，差异全部为自动多出单次入臂（经核实为人工漏记）；旷场 10 分钟双鼠录像检测率 95.7%–99.8%；详见 [`screenshots/ymaze-manual-verified-README.md`](screenshots/ymaze-manual-verified-README.md) 与 [`demo/`](demo/README.md) |

## 功能清单（逐页：入口 / 主要操作 / 空·加载·失败·成功状态）

接通判定基于源码与假 DOM/接线检查（**implemented**）；Windows 11 原生桌面窗口下已完成项目管理、旷场（OFT）与 Y 迷宫主流程实跑验证。

| 页面（模板/脚本） | 入口与主要操作 | 空 | 加载 | 失败 | 成功 |
| --- | --- | --- | --- | --- | --- |
| 首页（home.html） | `/`：创建项目、进入项目列表 | "还没有项目" | 服务端渲染，无异步加载 | 表单/接口错误走错误提示与 error.html | 创建后跳转 |
| 项目列表（projects.html） | `/projects`：项目卡片、进入 | "还没有项目" | 服务端渲染 | 删除/接口错误提示 | 操作后刷新列表 |
| 项目三tab（project.html/project.js） | 组织（分组/小鼠）、媒体（导入）、分析（草稿/批次） | 分组/小鼠/视频/草稿/批次各有"还没有…"空态 | 导入显示"正在导入…"（importStatus） | 逐文件错误列表（errorList）+ 创建分析失败 showErrors 且按钮恢复 | 导入成功刷新；分析跳转 analyze |
| 选择视频（analyze step=select） | 原生桥/浏览器回退/拖放；多选与文件夹 | `selected-empty` / `asset-empty` 内联空态 | `selectionPending` 与按钮禁用 | 逐文件 assets/errors 更新选中列表 | 选中计数更新，可继续追加 |
| 视频设置/场地/检测（workspace.js + editor-core.js） | 时间/方向/帧率、逐鼠几何、检测参数、两种预览 | 画布空态（未载入帧不可标注） | 预览进行中标记（previewAwaiting）、600ms 防抖保存 | `status(text,tone)` 与预览说明（真实 missing_reason/warnings） | "已保存"/预览图与说明呈现 |
| 人工Y记录/复核（ymaze 纯人工路径） | 视频→记录→复核→结果；无几何/检测门槛 | 空序列保留空比率与提示，不冒充数据 | 复核进行中状态 | 校验失败给出可定位原因 | 复核确认后进入分析 |
| Barnes 人工（barnes manual） | 探洞 null/[] 语义、首次目标与人工入箱分开 | 未记录探洞保留空态语义 | 同工作台预览链路 | 人工输入校验错误（与纯人工Y规则各自独立） | 预览/分析正常消费 |
| 默认设置/模板/样式（settings.html/settings.js） | 查看/修改出厂默认、保存为模板 | 首次为出厂默认值 | 载入现有设置 | 保存失败提示 | 保存成功提示 |
| 任务中心（tasks.html/tasks.js） | 批次进度、取消、原位重跑、终态删除、进入结果 | `task-empty` 空态 | 进度行轮询刷新 | 失败原因逐视频显示 | 完成态与结果入口 |
| 结果页（result.html/result.js） | 动物切换、图/视频、文件栏下载、ZIP、打开目录 | `data-empty="figures/video/bins/events"` 与"还没有写入结果"空态 | 图/视频加载中占位 | 图 onerror 转空态；打开目录失败可选中文本复制 | 声明文件全部带下载链接 |
| 关键点会话（landmarks.html/landmarks.js） | 新建会话、逐帧标注、容差验证 | `data-empty-sessions`、"还没有带场地几何的草稿"、`canvas-empty` | 新帧加载完成前禁点击（imageReady） | 保存/验证失败保留编辑并提示 | 验证完成表与预测图层（改容差即整体失效） |

横向能力已实现：多视频/多单位绑定、方向/解释FPS/采样FPS、草稿与历史、顺序批次/取消/失败/完整重跑/终态删除、CSV/JSON/PNG/MP4/ZIP、经典关键点标注。**未实现/未连通项：无已知占位控件**；如发现，按 feature_inventory 规则记 not_implemented。

## 生命周期要点（记录，不重复验收）

- 失败/取消/中断可原位重跑；成功重跑产生新 run；每轮独立 run 目录。
- 终态删除同事务重读状态；`owned_run_ids` 跨轮归属；旧无可信归属孤立文件保留。
- 取消 ≠ 关闭。关闭顺序：停调度/工作进程 → 排空 HTTP → 关 Store → 释放数据根所有权（service.py）。
- 错误日志给可操作路径（README「故障定位」）；SQLite 扩展码只作诊断事实。

## 时序背景取样（现状记录）

当前实现（detection.py `temporal_background`，ALGORITHM_VERSION=0.1.0）：已知有效终点时按源时钟均匀布目标，取首个起点到达目标时刻的真实曝光帧；零曝光帧不参与；样本不足只对实际有效帧求中位数；零样本拒绝；未知终点取前 N 帧。提示词中的"取目标时刻所在的真实源曝光帧（at or before）"表述**未接入**本工程代码。参考实现报告的两类边界（未知终点只取一帧、越界仍生成背景）与本工程当前实现行为不符：本工程只取正有效曝光的真实帧且零样本即报错；不把参考实现问题记为本工程故障，也不据此重开取样规则。

## 真实素材与身份记录

实验室自有旷场与 Y 迷宫素材已完成桌面实跑（演示输出见 `docs/demo/`）。其余 8 范式公开素材线索（仅后续扩展基准用，未核实文件前不填作者/许可/SHA）：`https://zenodo.org/records/3608658`（OFT_11/EPM_15/FST_8）、`https://zenodo.org/records/5078719`（Barnes Krill.mp4）。固定人工夹具一律标注"功能测试评分，非动物行为观察"。

## 后续扩展验收路线

1. **其余 8 范式真实素材扩展**：在已完成旷场（OFT）与 Y 迷宫真实录像实跑基础上，引入合法公开数据集验证其余 8 种范式的真实录像表现。
2. **Windows 极端路径矩阵**：补充不同 cwd、特殊字符路径与取消对话框边界的自动化回归。
3. **导出复核**：只比较本 run 声明文件、CSV 值/空值与 results、PNG 可读与图单位、MP4 实际解码 PTS/持续曝光、原片前后 SHA；证明集成一致，不自动证明生物学正确。

## 已确认完成的修复（保留行为，不重复返工）

第 22–25 块的各组有限修正（进程异常收尾、NOR 时序曲线、叠画坐标映射、容差清评估、结果枚举、运行时门槛、锁最终释放责任、选择器守卫、官方收集签名、入口绝对路径、worker 导入异常边界等）均已修复并带定向自检；详见 git 提交记录与本目录状态 JSON 的 checks。
