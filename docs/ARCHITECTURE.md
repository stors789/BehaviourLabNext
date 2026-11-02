# 架构与接口说明

本文只引用本仓库实际存在的文件。接收者应能据此定位每类实验的参数、几何与指标出处，以及本地 API 的成功/错误约定。

## 分层

```
web/templates + web/static（本地页面，ES modules/CSS，无外部CDN）
        │  fetch /api/*（同源 JSON）+ 声明文件同源下载
        ▼
web/app.py（Flask 路由，52 个 GET/POST）── service.py（Waitress 127.0.0.1、所有权、就绪、统一清理）
        │                                   desktop.py（pywebview 桥、私有四方法）
        ▼
assets.py（导入/上传/背景） ─ media.py（探测、源帧、解释时钟、采样）
        ▼
detection.py（经典前景、检测区域、时序背景） ─ geometry.py（画布映射/单应/形状/方向变换）
        ▼
engine.py（默认值合并、校验、作业冻结） ─ experiments/（十项实验评分）
        ▼
timeline.py / outputs.py（观测行、CSV/JSON/PNG/MP4/ZIP 统一导出与校验）
        ▼
jobs.py（Owned 顺序调度、私有子进程分派） ─ worker.py（后台单次分析） ─ processes.py（Windows Job Object、所有者记录）
```

## 核心数据契约（behaviourlab/schema.py）

- `Frame`：一个分析采样帧。`image` 为 BGR 数组（纯人工路径可为 `None`）；`index` 是源帧号；`time_s/dt_s/elapsed_s/analysis_fps` 与 `interval_start_s/interval_end_s`（该帧在本窗口内的有效曝光区间）共同构成显式时钟。
- `Detection`：单帧检测结果。`center` 为空即 `detected=False`；`missing_reason` 说明缺席原因；`nose/tail_base/left_ear/right_ear` 可为 `None`；质量与头向角度是独立字段，不互相冒充。
- 观测行 `OBSERVATION_FIELDS`：`x_px/y_px` 是**当前方向下旋转后源像素**；实验层再派生 `x_cm/y_cm`（`coordinate_unit` 标注 `cm/cm_estimated/px/mm`）；`motion_x0..y1` 是测量单位的完成运动段。
- `AnalysisContext/AnalysisResult`（schema.py:365 起）：作业冻结输入与统一输出；结果清单（`result_manifest.json`）与 `results.json` 见下"Result"。

### 关键结构的最小字段（出处均为本仓库实现文件）

- **Draft**（store.py `create_draft`，store.py:1070）：`id`（主键）、`assay`、`project_id`（可空=独立分析）、`videos`（每视频：`asset_id`、`window{start_s,end_s}`、`orientation`、`fps_override`、`sample_fps`、`units`）、`options/detection/output/style` 快照（草稿创建时并入默认值，提交不再重读设置）、`created_at/updated_at`。
- **Video**（Draft.videos 条目）：`asset_id` 关联素材（含 `ownership: linked|owned`，assets.py）；`window/orientation/fps_override/sample_fps` 为该视频的时间与采样语义；`units` 为逐鼠配置列表。
- **Unit**（草稿内与冻结后同形，engine.py `build_resolved_job` units 段）：`id/name/animal_id`、`window`（每鼠分析窗）、`geometry`（当前方向源像素）、`manual`（纯人工记录；Y-迷宫与 Barnes 人工语义各自独立）、`options`（该实验参数，深拷贝）、`detection`（检测参数）。
- **ResolvedJob**（engine.py `build_resolved_job`，engine.py:650）：`assay`、`job_id`、`run_id`、`project_id`、`asset{name,path,metadata,identity{path,size,mtime_ns}}`（`metadata` 为 probe 结果：宽高/时长/fps/frame_count，位于 asset 段内；源身份分析前后核对）、`window`、`orientation`、`units`（上述完整深拷贝）——冻结为 `frozen_job.json`，worker 只消费它。
- **Batch**（store.py `create_batch`，store.py:1336）：`id`（主键）、`assay`、`draft_id/project_id`、`videos`（每视频：`asset_id/name/run_id/status/progress` 及失败原因）、`status`（queued/running/completed/failed/cancelled/interrupted）、`owned_run_ids`（跨轮归属）、时间戳。
- **Result**：`result_manifest.json`（`status/run_id/job_id/assay/files` 声明清单、`units[{unit_id,unit_name,summary,figures,annotated_video}]`、`warnings/metadata`）+ `results.json`（`AnalysisResult.as_dict`：summary/events/bins/observations/tables/warnings/metadata）。页面与下载只消费清单声明文件。

## 时间/坐标/空值约定

- 源时钟：显式 PTS/曝光区间优先，`fps_override` 只建立解释时钟，不宣称修复源 PTS（media.py）。无覆盖且取不到有效源帧率时，名义源帧率取默认 20（`source_fps_source=default`、`source_fps_note` 说明），时钟仍是真实 PTS。
- 按时间取帧：先定位真实源索引再按索引读（`fps_override` 下 `index=⌊(t+ε)·fps⌋`；可靠 PTS 下只用已验证的顺序时间/缓存证据找目标，不用名义帧率猜帧号），缓存淘汰后按已验证的关键帧锚点（索引+PTS+时间）做一次有界 seek 恢复、失败才顺序重读；超过真实 EOF 仍返回最后可解码帧（media.py `FrameReader.frame_at_time`）。
- seek 未获证实（越过锚点、探测耗尽、抛错）即作废当前位置：关闭解码器、清零索引计数与时钟状态，再从头顺序读到目标；连续多次失败停用 seek，剩余目标只顺序前进（media.py `_invalidate_position`）。不猜位置、不返回错帧像素。
- 索引证据（锚点+首帧绝对 PTS）绑定产生它的**源文件身份**与解释帧率：换文件、换时钟参数时目标读取器拒绝采纳，退回顺序读取；规划→取像素之间源身份变化时报错放弃本次背景，不把旧索引/时间嫁接到新读取器（media.py `IndexEvidence`、detection `_build_temporal_background`）。
- 叠画视频（annotated.py `write_run_video`）：切分点用二分只取当前曝光内部的切点；每只动物一次性把完成段端点换算成像素并按完成时刻建游标，轨迹只把**新完成段**追加进同色增量层（覆盖度乘积）后整层合成一次，每帧成本随已画像素/画布增长而不随历史段数重放；观测按 `frame` 预建索引（重复 frame 取首条）。编码前画面与逐段重放相比，差异只出现在抗锯齿部分覆盖像素（实测 ≤2 灰阶），完全覆盖与未画像素逐值相同。
- 输出校验与进度：`verify_video` 流式逐帧核对数量/PTS/duration（不把整段像素驻留内存）；定位、编码、校验三个阶段都有取消检查与真实进度（未知总数只报阶段+已处理数），取消或失败关闭容器并删除未完成 MP4。
- 缺失语义：缺席帧保留行、`detected=False`、`x_px=None`；窗口外上下文帧 `in_window=False`；缺失/未知/零值分开，不用 0 冒充。
- 坐标：几何保存为当前方向下的原分辨率像素（analyze 页画布）；物理坐标只出现在 `x_cm/y_cm` 等测量字段并带 `coordinate_unit`；EPM/Y-迷宫的物理尺度是估计值（`cm_estimated`）。
- 冻结：`engine.py` 把草稿+默认快照冻结为 `frozen_job.json`，含源身份（路径/大小/mtime）；分析只消费冻结件，重跑产生新 `run_id`，发布以 `result_manifest.json` 声明文件清单为准（`runs.py`/`web/app.py` 下载边界）。

## 十项实验（注册键与出处，注册表见 experiments/__init__.py）

| 键 | 模式 | 参数/几何/指标出处 |
| --- | --- | --- |
| tst | classic / lis（体长按标准 60 mm 换算，假定写在参数说明） | experiments/tst.py |
| fst | 自动运动观测/状态/丢失 | experiments/fst.py |
| mwm | training / reversal / probe（真实平台确认） | experiments/mwm.py |
| oft | 区域时间/路径 | experiments/oft.py |
| nor | 对象角色、头向辅助 + 明确距离回退（低头向不冒称鼻尖验证） | experiments/nor.py |
| epm | 入臂/深度（物理尺度为估计值） | experiments/epm.py |
| ymaze | 自动交替 + 纯人工Y（空序列保留空比率/提示） | experiments/ymaze.py |
| ldb | 暗室可见/不可见（观察与缺席推断分开，不可靠轨迹抑制） | experiments/ldb.py |
| barnes | 自动 + 人工 null/[]（探洞重复访问、首次目标与人工入箱分开，probe 无入箱） | experiments/barnes.py |
| ezm | 四段顺时针区域、边界/区外/未知、首次观察与确认相邻进入分开 | experiments/ezm.py |

共享模块：experiments/base.py、timeline.py。注册键恰好十个；TST 两模式与 LDB 两可见性不占用额外注册键。

## 本地 API 约定（web/app.py）

- 成功：`{"ok": true, "data": {...}}`；错误：`{"ok": false, "error": {"message", "fields": [...]}}`，HTTP 状态码区分 400/404/409 等。
- 二进制下载只走声明文件：`/api/runs/<run_id>/files/<相对路径>`（`?download=1` 附件）与 `/api/runs/<run_id>/bundle`（ZIP）；清单未声明的文件 404。背景图只认 32 位十六进制 `.png` 名。
- 桌面桥（desktop.py）公开成员只有 `choose_videos/choose_folder/choose_image/open_directory` 四个；`open_directory` 对缺失/非目录/系统失败抛可读业务错误。下载另存为由 WebView2 `DownloadStarting` 契约处理（确认写所选路径、取消不保存）。
- 结果页打开目录用 `POST /api/runs/<run_id>/open`（服务端实现，与桥的 `open_directory` 互不影响）。

## 默认值、模板与样式

- 公共默认：web/app.py 的 `COMMON_OUTPUT_DEFAULTS/COMMON_STYLE_DEFAULTS`；每实验默认在其模块 `DEFAULTS`。
- 覆盖顺序：出厂默认 → 数据根保存设置（settings 表） → 草稿值；冻结时全部并入 frozen job。
- 模板白名单：field_templates.py 定义可保存为模板的字段集合；模板应用于草稿时不越界。

## 生命周期

- 失败/取消/中断的批次可**原位重跑**（同 job 重置），成功重跑产生新 run；每轮分析写入独立 run 目录。
- 终态删除在同一事务内重读状态后执行；`owned_run_ids` 跨轮归属核对，无可信归属的旧孤立文件保留不删（jobs.py 删除计划）。
- 取消（单批次）与关闭（整实例）不同：关闭先停调度与工作进程、再排空 HTTP、再关 Store、最后释放数据根所有权（service.py `stop`）。
- SQLite 扩展码（如 disk I/O）只作为诊断事实写入日志，不宣称已在所有机器解决。

## 时序背景取样（现状）

`detection.py temporal_background`：已知有效终点时按源时钟在 [start,end] 均匀布目标，取**首个起点到达目标时刻的真实曝光帧**（目标后首帧），零曝光的窗口外帧不参与，样本不足时只对实际取得的有效帧求中位数，零样本拒绝；未知终点取前 N 帧。`ALGORITHM_VERSION="0.1.0"`（config.py）。提示词中"取目标时刻所在的真实源曝光帧（at or before）"表述**未接入**当前代码；参考实现报告的"未知终点只取一帧/越界仍生成背景"两类边界在本工程代码中不存在（只取有效真实帧、零样本即报错）。

实际调用路径是两段式（`prepare_background` → `temporal_frame_plan` → `VideoSource.frames_at`）：先用同一采样流只读时刻/曝光定出目标源帧（`decode_images=False`），再只对这些源帧取方向像素，像素读取器用 `defer_images` 只对目标帧做 `to_ndarray`，中间帧不生成 BGR/不做方向复制/不转灰。背景按“源身份+方向+解释FPS+采样规则+窗口/推导终点+样本数+算法版本”做有界复用（条目与字节双上限，同键并发只建一份，取消/失败不发布），`TEMPORAL_BG_VERSION` 参与键。
