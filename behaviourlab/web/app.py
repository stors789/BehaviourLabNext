"""本地 Flask 服务：API 骨架与页面路由。"""

from __future__ import annotations

import logging
import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from flask import (Flask, Response, after_this_request, jsonify,
                   render_template, request, send_file)
from werkzeug.exceptions import HTTPException

from .. import assets as asset_service
from .. import experiments
from .. import field_templates
from .. import landmarks as landmark_service
from .. import figures as figure_service
from .. import media as media_service
from .. import preview as preview_service
from .. import runs as run_service
from ..assets import BackgroundError
from ..config import APP_NAME, APP_VERSION, SCHEMA_VERSION, DataPaths
from ..detection import common_detection_defaults
from ..engine import (
    COMMON_OUTPUT_DEFAULTS,
    COMMON_STYLE_DEFAULTS,
    snapshot_draft,
    validate_draft,
    validate_settings_value,
    validate_style,
)
from ..jobs import (
    ACTIVE,
    CANCELLED,
    COMPLETED,
    FAILED,
    INTERRUPTED,
    NOT_STARTED,
    QUEUED,
    RUNNING,
    JobConflict,
    JobError,
    JobScheduler,
)
from .. import outputs as output_service
from ..processes import JobOwnerRegistry, child_creation_flags, child_creation_flags
from ..schema import ValidationError, is_number, is_valid_id, new_id
from ..store import (
    NotFoundError,
    Store,
    read_json_file,
    storage_log_path,
)
from . import descriptors as web_descriptors

LOGGER = logging.getLogger(__name__)

SETTINGS_SECTIONS = ("defaults", "detection", "output", "style")
BACKGROUND_NAME = re.compile(r"[0-9a-f]{32}\.png")
PREVIEW_NAME = re.compile(r"[0-9a-f]{32}\.(?:png|jpg|jpeg)")
RUN_FILES = ("observations.csv", "summary.csv", "events.csv", "bins.csv",
             "zones.csv", "frozen_job.json", "results.json",
             "result_manifest.json", "report.md")
RUN_FIGURE = re.compile(
    r"[A-Za-z0-9_-]+_(?:trajectory|heatmap|activity|objects|zones|sequence"
    r"|exploration)\.png")
RUN_VIDEO = re.compile(r"(?:[A-Za-z0-9_-]+_)?annotated\.mp4")


def _is_run_file(name: str) -> bool:
    """运行产物白名单：固定文件加上每动物图形与标注视频。"""

    return (name in RUN_FILES or bool(RUN_FIGURE.fullmatch(name))
            or bool(RUN_VIDEO.fullmatch(name)))


class ApiError(Exception):
    def __init__(self, message: str, status: int = 400,
                 fields: list[dict] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.fields = fields or []


def success(data: Any, status: int = 200):
    return jsonify({"ok": True, "data": data}), status


def failure(message: str, status: int = 400, fields: list[dict] | None = None):
    return jsonify({"ok": False, "error": {"message": message,
                                           "fields": fields or []}}), status


def _json_body() -> dict:
    if not request.is_json:
        raise ApiError("请求体必须是 JSON", 400)
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ApiError("请求 JSON 顶层必须是对象", 400)
    return payload


def _mapping(value: Any, field: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ApiError("必须是对象", 400, [{"field": field, "message": "必须是对象"}])
    return dict(value)


def _video_defaults(video: Mapping[str, Any], index: int) -> dict:
    item = dict(video)
    item.setdefault("window", {"start_s": 0, "end_s": None})
    item.setdefault("orientation", 0)
    item.setdefault("fps_override", None)
    item.setdefault("sample_fps", None)
    item.setdefault("options", {})
    item.setdefault("detection", {})
    if "units" in item:
        units = item["units"]
        if units is None:
            raise ApiError(
                "units 不能为 null", 400,
                [{"field": f"videos[{index}].units",
                  "message": "不能用 null 清空；空数组表示没有单位"}],
            )
    else:
        units = []
    if not isinstance(units, list):
        raise ApiError("units 必须是数组", 400,
                       [{"field": f"videos[{index}].units", "message": "必须是数组"}])
    normalized_units = []
    for unit_index, unit in enumerate(units):
        if not isinstance(unit, Mapping):
            raise ApiError("单位条目必须是对象", 400,
                           [{"field": f"videos[{index}].units[{unit_index}]",
                             "message": "必须是对象"}])
        entry = dict(unit)
        entry.setdefault("id", new_id())
        if not entry.get("name"):
            entry["name"] = f"小鼠 {unit_index + 1}"
        entry.setdefault("animal_id", None)
        entry.setdefault("window", None)
        entry.setdefault("geometry", {})
        entry.setdefault("options", {})
        entry.setdefault("detection", {})
        entry.setdefault("manual", None)
        normalized_units.append(entry)
    item["units"] = normalized_units
    return item


def _normalize_videos(videos: Any) -> list[dict]:
    if videos is None:
        raise ApiError("videos 不能为 null", 400,
                       [{"field": "videos",
                         "message": "不能用 null 清空；空数组表示没有视频"}])
    if not isinstance(videos, list):
        raise ApiError("videos 必须是数组", 400,
                       [{"field": "videos", "message": "必须是数组"}])
    normalized = []
    for index, video in enumerate(videos):
        if not isinstance(video, Mapping):
            raise ApiError("视频条目必须是对象", 400,
                           [{"field": f"videos[{index}]", "message": "必须是对象"}])
        normalized.append(_video_defaults(video, index))
    return normalized


def _assay_metrics(assay: Any) -> list[dict]:
    if not experiments.exists(assay):
        return []
    return deepcopy(experiments.get(assay).SPEC.get("metrics") or [])


def _assay_label(assay: Any) -> str:
    if not experiments.exists(assay):
        return str(assay or "").upper()
    return str(experiments.get(assay).SPEC.get("label") or assay)


def _filled_setting(store: Store, section: str, factory: Mapping[str, Any]) -> dict:
    """输出/样式：保存值补全 factory 字段后返回，未保存时就是 factory。"""

    value = store.get_setting(f"settings.{section}", default=None)
    merged = deepcopy(dict(factory))
    if isinstance(value, Mapping):
        merged.update(deepcopy(dict(value)))
    return merged


def _settings_snapshot(store: Store, assay: str | None) -> dict:
    """两个分区各自的合同形状：defaults 按实验键、detection 是当前对象。

    通用解析对 defaults 仍会按实验名取一层，对检测则直接使用传入对象；服务
    必须按同一形状提供，草稿创建、预览与最终校验才共用一套规则。
    """

    defaults = store.get_setting("settings.defaults", default={})
    detection = store.get_setting("settings.detection", default={})
    defaults = defaults if isinstance(defaults, Mapping) else {}
    detection = detection if isinstance(detection, Mapping) else {}
    saved_detection = detection.get(assay) if assay else None
    return {
        "defaults": deepcopy(dict(defaults)),
        "detection": deepcopy(dict(saved_detection))
        if isinstance(saved_detection, Mapping) else {},
        "output": _filled_setting(store, "output", COMMON_OUTPUT_DEFAULTS),
        "style": _filled_setting(store, "style", COMMON_STYLE_DEFAULTS),
    }


def _assay_defaults_factory() -> dict:
    """每个实验的工厂默认参数；设置页与 API 缺省值都从这里取。"""

    return {entry["key"]: deepcopy(entry["defaults"])
            for entry in experiments.catalog()}


def _assay_detection_factory() -> dict:
    """每个实验的完整检测默认（公共默认 + 实验覆盖）。"""

    factory = {}
    for entry in experiments.catalog():
        values = deepcopy(entry["detection_defaults"])
        saved = entry.get("detection_overrides")
        if isinstance(saved, Mapping):
            values.update(deepcopy(dict(saved)))
        factory[entry["key"]] = values
    return factory


def _settings_value(store: Store, section: str) -> Any:
    """单个设置分区的有效值：无保存值时给工厂值，不是空对象。"""

    if section in ("output", "style"):
        factory = (COMMON_OUTPUT_DEFAULTS if section == "output"
                   else COMMON_STYLE_DEFAULTS)
        return _filled_setting(store, section, factory)
    factory = (_assay_defaults_factory() if section == "defaults"
               else _assay_detection_factory())
    saved = store.get_setting(f"settings.{section}", default={})
    if not isinstance(saved, Mapping):
        return factory
    for key, value in saved.items():
        if isinstance(value, Mapping) and isinstance(factory.get(key), Mapping):
            factory[key] = {**factory[key], **deepcopy(dict(value))}
        elif isinstance(value, Mapping):
            factory[key] = deepcopy(dict(value))
    return factory


def _run_payloads(directory: Path) -> list[dict]:
    """读取一次运行的结果；新布局是 runs/{run_id}，旧布局是嵌套 job 目录。

    目录本身就是一次已解析的发布（直接含 results.json/result_manifest.json）
    时只展示它自己的结果：figures/ 等声明文件子目录不再次枚举，也不
    变成“结果文件缺失”的错误卡片。目录只是容器时才枚举子目录，保留
    真实旧布局（嵌套 job 目录）的合法兼容。
    """

    if not directory.is_dir():
        return []
    if any((directory / name).is_file()
           for name in ("results.json", "result_manifest.json")):
        return [_run_payload(directory)]
    entries = sorted(item for item in directory.iterdir() if item.is_dir())
    return [_run_payload(entry) for entry in entries]


def _rail_groups(files: list[str]) -> dict:
    """结果页文件栏分组：图、视频、表格、报告与配置。"""

    lower = [(name, str(name).lower()) for name in files]
    return {
        "figures": [name for name, low in lower if low.endswith(".png")],
        "videos": [name for name, low in lower if low.endswith(".mp4")],
        "tables": [name for name, low in lower if low.endswith(".csv")],
        "docs": [name for name, low in lower
                 if not low.endswith((".png", ".mp4", ".csv"))],
    }


def _run_payload(job_dir: Path) -> dict:
    """单个视频的产物；缺失或损坏时如实标记，不假装成功。"""

    entry: dict = {"job_id": job_dir.name, "files": []}
    result_file = job_dir / "results.json"
    if not result_file.is_file():
        entry["error"] = "结果文件缺失"
        return entry
    try:
        entry["result"] = json.loads(result_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        entry["error"] = f"结果文件不可读取：{error}"
        return entry
    entry["files"] = sorted(
        path.name for path in job_dir.iterdir()
        if path.is_file() and _is_run_file(path.name))
    try:
        manifest_file = job_dir / "result_manifest.json"
        if manifest_file.is_file():
            entry["manifest"] = json.loads(
                manifest_file.read_text(encoding="utf-8"))
        frozen = job_dir / "frozen_job.json"
        if frozen.is_file():
            entry["job"] = json.loads(frozen.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        entry["error"] = f"清单文件不可读取：{error}"
    # 新布局把整个视频发布在 runs/{run_id}：任务 ID 以清单/冻结任务为准。
    for source in (entry.get("manifest"), entry.get("job")):
        if isinstance(source, Mapping) and source.get("job_id"):
            entry["job_id"] = source["job_id"]
            break
    # 文件入口以清单声明为准：声明里合法的嵌套相对路径（新图种所在
    # 子目录）也在这里出现；目录扫描只兜底极旧的裸 results.json。
    declared_files = (entry.get("manifest") or {}).get("files")
    if isinstance(declared_files, list) and declared_files:
        merged = [str(name) for name in declared_files
                  if isinstance(name, str) and name]
        for name in entry["files"]:
            if name not in merged:
                merged.append(name)
        entry["files"] = merged
    units = []
    for unit in (entry.get("manifest") or {}).get("units") or []:
        units.append({"unit_id": unit.get("unit_id"),
                      "unit_name": unit.get("unit_name"),
                      "summary": unit.get("summary") or {},
                      "figures": unit.get("figures") or {},
                      "annotated_video": unit.get("annotated_video")})
    if not units:
        for row in entry["result"].get("summary") or []:
            units.append({"unit_id": row.get("unit_id"),
                          "unit_name": row.get("unit_name"),
                          "summary": row, "figures": {}})
    entry["units"] = units
    figures_map = (entry.get("manifest") or {}).get("figures")
    if isinstance(figures_map, Mapping):
        entry["figures"] = {str(key): list(value) if isinstance(value, list) else []
                            for key, value in figures_map.items()}
    else:
        entry["figures"] = {}
    for unit in units:
        names = entry["figures"].get(str(unit.get("unit_id")))
        if names is None:
            info = unit.get("figures") or {}
            names = [value for value in info.values()
                     if isinstance(value, str) and value.endswith(".png")]
        unit["figure_files"] = names
        files = entry.get("files") or []
        for name in names:
            if name not in files:
                files.append(name)
        entry["files"] = files
    assay = ((entry.get("result") or {}).get("metadata") or {}).get("assay") \
        or (entry.get("job") or {}).get("assay")
    entry["assay"] = assay
    entry["metrics"] = _assay_metrics(assay)
    entry["assay_label"] = _assay_label(assay)
    entry["rail"] = _rail_groups([str(name) for name in entry["files"]])
    return entry


def create_app(*, data_dir: Any = None, store: Store | None = None,
               testing: bool = False) -> Flask:
    paths = DataPaths.from_dir(data_dir)
    store = store or Store(paths)
    # 调度器不在这里抢所有权：create_app 只创建，应用真正启动时才 start。
    # testing 用同一服务的同步执行变体，避免测试里另开一条分析路径。
    jobs = JobScheduler(store, paths, inline=bool(testing))
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config.update(
        TESTING=testing,
        DATA_PATHS=paths,
        STORE=store,
        JOBS=jobs,
        JOB_OWNER=jobs.owner,
        JSON_SORT_KEYS=False,
    )

    app.json.ensure_ascii = False
    app.json.sort_keys = False

    @app.context_processor
    def inject_globals() -> dict:
        return {"app_name": APP_NAME, "version": APP_VERSION}

    # -- 错误处理 -------------------------------------------------------
    @app.errorhandler(ApiError)
    def handle_api_error(error: ApiError):
        return failure(error.message, error.status, error.fields)

    @app.errorhandler(ValidationError)
    def handle_validation(error: ValidationError):
        return failure(error.message, 400, [error.as_issue()])

    @app.errorhandler(NotFoundError)
    def handle_not_found(error: NotFoundError):
        if request.path.startswith("/api/"):
            return failure(str(error), 404)
        return render_template("not_found.html"), 404

    @app.errorhandler(404)
    def handle_404(error):
        if request.path.startswith("/api/"):
            return failure("接口不存在", 404)
        return render_template("not_found.html"), 404

    @app.errorhandler(HTTPException)
    def handle_http_error(error: HTTPException):
        status = error.code or 500
        if request.path.startswith("/api/"):
            return failure(error.description or "请求失败", status)
        template = "not_found.html" if status == 404 else "error.html"
        return render_template(template), status

    @app.errorhandler(Exception)
    def handle_unexpected(error: Exception):
        LOGGER.exception("未处理的异常")
        if request.path.startswith("/api/"):
            return failure("服务器内部错误", 500)
        return render_template("error.html"), 500

    @app.context_processor
    def inject_shell():
        project_id = request.args.get("project")
        project = store.get_project(project_id) if is_valid_id(project_id) else None
        mode = request.args.get("mode", "")
        return {
            "nav_projects": store.list_projects(),
            "current_project": project,
            "current_mode": "independent" if mode == "independent" else "",
        }

    # -- 健康与目录 -----------------------------------------------------
    @app.get("/api/health")
    def health():
        registry = app.config["JOB_OWNER"]
        return success(
            {
                "status": "ok",
                "version": APP_VERSION,
                "schema_version": SCHEMA_VERSION,
                "data_root": str(paths.root),
                "database": str(paths.database),
                "storage_log": str(storage_log_path(paths)),
                "job_owner": registry.is_active(),
            }
        )

    @app.get("/api/catalog")
    def catalog():
        return success(
            {
                "schema_version": SCHEMA_VERSION,
                "assays": experiments.catalog(),
                "detection_defaults": common_detection_defaults(),
                "output_defaults": deepcopy(COMMON_OUTPUT_DEFAULTS),
                "style_defaults": deepcopy(COMMON_STYLE_DEFAULTS),
            }
        )

    # -- 素材导入与媒体 -------------------------------------------------
    def _require_project(project_id: str) -> dict:
        if not is_valid_id(project_id):
            raise ApiError("项目 ID 无效", 400,
                           [{"field": "project_id", "message": "ID 无效"}])
        project = store.get_project(project_id)
        if project is None:
            raise ApiError("项目不存在", 404)
        return project

    @app.get("/api/assets")
    def list_assets():
        project_id = request.args.get("project_id")
        if project_id is None:
            return success({"assets": store.list_assets()})
        _require_project(project_id)
        return success({"assets": store.list_assets_for_scope(project_id)})

    @app.post("/api/assets/import")
    def import_assets():
        body = _json_body()
        paths = body.get("paths")
        if not isinstance(paths, list):
            raise ApiError("paths 必须是数组", 400,
                           [{"field": "paths", "message": "必须是数组"}])
        for index, value in enumerate(paths):
            if not isinstance(value, str):
                raise ApiError("paths 必须是字符串数组", 400,
                               [{"field": f"paths[{index}]", "message": "必须是字符串"}])
        project_id = body.get("project_id")
        if project_id is not None:
            _require_project(project_id)
        return success(asset_service.import_paths(store, paths, project_id))

    @app.post("/api/assets/upload")
    def upload_assets():
        files = request.files.getlist("files")
        if not files:
            raise ApiError("缺少上传文件", 400,
                           [{"field": "files", "message": "至少选择一个文件"}])
        project_id = request.form.get("project_id") or None
        if project_id is not None:
            _require_project(project_id)
        return success(asset_service.import_uploads(store, files, project_id))

    @app.get("/api/assets/<asset_id>/stream")
    def stream_asset(asset_id: str):
        asset = store.get_asset(asset_id)
        if asset is None:
            raise ApiError("素材不存在", 404)
        path = Path(asset["path"])
        if not path.is_file():
            raise ApiError("源文件不可读取", 404)
        return send_file(path, conditional=True)

    @app.get("/api/assets/<asset_id>/frame")
    def asset_frame(asset_id: str):
        asset = store.get_asset(asset_id)
        if asset is None:
            raise ApiError("素材不存在", 404)

        raw_index = request.args.get("index")
        if raw_index is None:
            raise ApiError("缺少帧索引", 400,
                           [{"field": "index", "message": "必须提供帧索引"}])
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            raise ApiError("帧索引必须是整数", 400,
                           [{"field": "index", "message": "必须是整数"}])

        raw_orientation = request.args.get("orientation", "0")
        try:
            orientation = int(raw_orientation)
        except (TypeError, ValueError):
            raise ApiError("方向必须是 0/90/180/270", 400,
                           [{"field": "orientation", "message": "必须是整数"}])

        raw_fps = request.args.get("fps_override")
        fps_override = None
        if raw_fps not in (None, ""):
            try:
                fps_override = float(raw_fps)
            except (TypeError, ValueError):
                raise ApiError("解释帧率必须是有限正数", 400,
                               [{"field": "fps_override", "message": "必须是数值"}])

        try:
            info = media_service.read_frame_info(
                asset["path"], index, orientation, fps_override
            )
        except ValidationError as error:
            raise ApiError(error.message, 400, [error.as_issue()])
        except media_service.MediaError as error:
            raise ApiError(str(error), 422, [{"field": "index",
                                              "message": str(error)}])

        import cv2

        ok, encoded = cv2.imencode(
            ".jpg", info.image, [cv2.IMWRITE_JPEG_QUALITY, 90]
        )
        if not ok:
            raise ApiError("无法编码源帧", 422,
                           [{"field": "index", "message": "JPEG 编码失败"}])
        response = Response(encoded.tobytes(), mimetype="image/jpeg")
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Source-Time"] = str(float(info.time_s))
        response.headers["X-Source-Frame"] = str(info.index)
        response.headers["X-Clock-Source"] = info.clock_source
        return response

    @app.post("/api/backgrounds/upload")
    def upload_background():
        upload = request.files.get("file")
        if upload is None:
            raise ApiError("缺少背景文件", 400,
                           [{"field": "file", "message": "请选择一个文件"}])
        try:
            result = asset_service.save_background(paths, upload.read())
        except BackgroundError as error:
            raise ApiError(str(error), 400,
                           [{"field": "file", "message": str(error)}])
        return success(result, 201)

    @app.get("/api/backgrounds/<filename>")
    def get_background(filename: str):
        if not BACKGROUND_NAME.fullmatch(filename):
            raise ApiError("背景不存在", 404)
        path = Path(paths.backgrounds) / filename
        if not path.is_file():
            raise ApiError("背景不存在", 404)
        return send_file(path, mimetype="image/png", conditional=True)

    @app.post("/api/backgrounds/validate")
    def validate_background():
        """原生 linked 背景的可解码确认：失败不替换旧有效背景。"""

        payload = request.get_json(silent=True) or {}
        path = payload.get("path")
        if not isinstance(path, str) or not path.strip():
            raise ApiError("缺少背景路径", 400,
                           [{"field": "path", "message": "请提供背景文件路径"}])
        try:
            asset_service.validate_background_path(path)
        except BackgroundError as error:
            raise ApiError(str(error), 400,
                           [{"field": "path", "message": str(error)}])
        return success({"ok": True})

    @app.get("/runs/<run_id>/<job_id>/<filename>")
    def get_run_file(run_id: str, job_id: str, filename: str):
        if not (is_valid_id(run_id) and is_valid_id(job_id)):
            raise ApiError("结果不存在", 404)
        if not _is_run_file(filename):
            raise ApiError("结果文件不存在", 404)
        # 与结果 API 共用同一份归属/状态/声明检查：失败清单不能从这里取文件。
        run = run_service.resolve_run(store, paths, run_id)
        if run is None or str(run.get("job_id")) != job_id:
            raise ApiError("结果文件不存在", 404)
        if not run_service.declared(run["manifest"], filename):
            raise ApiError("结果文件不存在", 404)
        path = run_service.safe_file(run["dir"], filename)
        if path is None:
            raise ApiError("结果文件不存在", 404)
        if RUN_FIGURE.fullmatch(filename) or RUN_VIDEO.fullmatch(filename):
            # 图形与标注视频默认在结果页内联显示，带 download=1 时按附件下载。
            if request.args.get("download"):
                return send_file(path, as_attachment=True,
                                 download_name=filename)
            mimetype = ("video/mp4" if RUN_VIDEO.fullmatch(filename)
                        else "image/png")
            return send_file(path, mimetype=mimetype, conditional=True)
        return send_file(path, as_attachment=True, download_name=filename)

    @app.get("/api/previews/<filename>")
    def get_preview(filename: str):
        if not PREVIEW_NAME.fullmatch(filename):
            raise ApiError("预览不存在", 404)
        path = Path(paths.previews) / filename
        if not path.is_file():
            raise ApiError("预览不存在", 404)
        return send_file(path, conditional=True)

    # -- 设置 -----------------------------------------------------------
    @app.get("/api/settings/<section>")
    def get_settings(section: str):
        if section not in SETTINGS_SECTIONS:
            raise ApiError("未知设置分区", 404)
        return success({"key": section, "value": _settings_value(store, section)})

    @app.put("/api/settings/<section>")
    def put_settings(section: str):
        if section not in SETTINGS_SECTIONS:
            raise ApiError("未知设置分区", 404)
        body = _json_body()
        if "value" not in body:
            raise ApiError("缺少 value", 400,
                           [{"field": "value", "message": "必须提供 value"}])
        value = body["value"]
        if not isinstance(value, Mapping):
            raise ApiError("设置值必须是对象", 400,
                           [{"field": "value", "message": "必须是对象"}])
        value = dict(value)
        if section in ("output", "style"):
            # 保存时补全 factory 字段，读回不再依赖缺省逻辑。
            factory = (COMMON_OUTPUT_DEFAULTS if section == "output"
                       else COMMON_STYLE_DEFAULTS)
            filled = deepcopy(dict(factory))
            filled.update(value)
            value = filled
        issues = validate_settings_value(section, value)
        if issues:
            raise ApiError("设置未通过校验", 400, issues)
        saved = store.set_setting(f"settings.{section}", value)
        return success({"key": section, "value": saved})

    @app.post("/api/settings/style/preview")
    def preview_style():
        body = _json_body()
        value = body.get("value")
        if not isinstance(value, Mapping):
            raise ApiError("样式必须是对象", 400,
                           [{"field": "value", "message": "必须是对象"}])
        merged = deepcopy(dict(COMMON_STYLE_DEFAULTS))
        merged.update(dict(value))
        issues = validate_style(merged)
        if issues:
            raise ApiError("样式未通过校验", 400, issues)
        filename = f"{new_id()}.png"
        target = Path(paths.previews) / filename
        try:
            figure_service.style_preview_png(target, merged)
        except (OSError, ValueError) as error:
            raise ApiError(f"无法生成样式预览：{error}", 422) from error
        # 预览只渲染到 240 DPI；返回值说明实际渲染值，正式设置不被改写。
        render_dpi = min(float(merged.get("dpi") or 150), 240.0)
        preview_style = {**merged, "dpi": render_dpi}
        return success({"image_url": f"/api/previews/{filename}",
                        "style": preview_style})

    # -- 项目 -----------------------------------------------------------
    @app.get("/api/projects")
    def list_projects():
        return success({"projects": store.list_projects()})

    @app.post("/api/projects")
    def create_project():
        body = _json_body()
        project = store.create_project(
            body.get("name"), body.get("description", "")
        )
        return success({"project": project}, 201)

    @app.get("/api/projects/<project_id>")
    def get_project(project_id: str):
        project = store.get_project(project_id)
        if project is None:
            raise ApiError("项目不存在", 404)
        return success({"project": project})

    @app.put("/api/projects/<project_id>")
    def update_project(project_id: str):
        body = _json_body()
        project = store.update_project(project_id, body)
        return success({"project": project})

    @app.delete("/api/projects/<project_id>")
    def delete_project(project_id: str):
        store.delete_project(project_id)
        return success({"deleted": True})

    @app.get("/api/projects/<project_id>/groups")
    def list_groups(project_id: str):
        if store.get_project(project_id) is None:
            raise ApiError("项目不存在", 404)
        return success({"groups": store.list_groups(project_id)})

    @app.post("/api/projects/<project_id>/groups")
    def create_groups(project_id: str):
        body = _json_body()
        if "groups" in body:
            entries = body["groups"]
            if not isinstance(entries, list) or not entries:
                raise ApiError("groups 必须是非空数组", 400,
                               [{"field": "groups", "message": "必须是非空数组"}])
            names = []
            for index, entry in enumerate(entries):
                if not isinstance(entry, Mapping):
                    raise ApiError("分组条目必须是对象", 400,
                                   [{"field": f"groups[{index}]", "message": "必须是对象"}])
                names.append(entry.get("name"))
            created = store.create_groups(project_id, names)
        elif "names" in body:
            names = body["names"]
            if not isinstance(names, list) or not names:
                raise ApiError("names 必须是非空数组", 400,
                               [{"field": "names", "message": "必须是非空数组"}])
            created = store.create_groups(project_id, names)
        else:
            created = [store.create_group(project_id, body.get("name"))]
        return success({"groups": created}, 201)

    def _batch_animals(body: Mapping[str, Any]) -> list[dict]:
        """{group_id, prefix, start, count} → 连续命名的动物条目。"""

        prefix = body.get("prefix", "M")
        if not isinstance(prefix, str) or not prefix.strip():
            raise ApiError("prefix 必须是非空文本", 400,
                           [{"field": "prefix", "message": "必须是非空文本"}])
        start = body.get("start", 1)
        if not isinstance(start, int) or isinstance(start, bool):
            raise ApiError("start 必须是整数", 400,
                           [{"field": "start", "message": "必须是整数"}])
        if start < 0:
            raise ApiError("start 必须是非负整数", 400,
                           [{"field": "start", "message": "必须是非负整数"}])
        count = body.get("count")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ApiError("count 必须是正整数", 400,
                           [{"field": "count", "message": "必须是正整数"}])
        group_id = body.get("group_id")
        return [
            {"name": f"{prefix.strip()}{start + index}", "group_id": group_id}
            for index in range(count)
        ]

    @app.get("/api/projects/<project_id>/animals")
    def list_animals(project_id: str):
        if store.get_project(project_id) is None:
            raise ApiError("项目不存在", 404)
        return success({"animals": store.list_animals(project_id)})

    @app.post("/api/projects/<project_id>/animals")
    def create_animals(project_id: str):
        body = _json_body()
        if "animals" in body:
            items = body["animals"]
            if not isinstance(items, list) or not items:
                raise ApiError("animals 必须是非空数组", 400,
                               [{"field": "animals", "message": "必须是非空数组"}])
        elif "count" in body or "prefix" in body:
            items = _batch_animals(body)
        else:
            items = [body]
        created = store.create_animals(project_id, items)
        return success({"animals": created}, 201)

    # -- 草稿 -----------------------------------------------------------
    @app.get("/api/drafts")
    def list_drafts():
        project_id = request.args.get("project_id")
        if project_id is not None and not is_valid_id(project_id):
            raise ApiError("项目 ID 无效", 400,
                           [{"field": "project_id", "message": "ID 无效"}])
        return success({"drafts": store.list_drafts(project_id)})

    @app.post("/api/drafts")
    def create_draft():
        body = _json_body()
        # 服务缺省是 oft；界面初始选择（注册表首项 TST）是另一条契约。
        assay = body.get("assay")
        if assay is None:
            assay = "oft"
        if not isinstance(assay, str) or not experiments.exists(assay):
            raise ApiError("未知实验", 400, [{"field": "assay", "message": "未知实验"}])
        project_id = body.get("project_id")
        if project_id is not None and not is_valid_id(project_id):
            raise ApiError("项目 ID 无效", 400,
                           [{"field": "project_id", "message": "ID 无效"}])
        if project_id is not None and store.get_project(project_id) is None:
            raise ApiError("项目不存在", 404)

        # 只复制当时的设置快照：草稿不引用 settings 对象。
        snapshot = snapshot_draft(assay, settings=_settings_snapshot(store, assay))
        for key in ("options", "detection", "output", "style"):
            if key not in body:
                continue
            value = body[key]
            if not isinstance(value, Mapping):
                raise ApiError(f"{key} 必须是对象", 400,
                               [{"field": key, "message": "必须是对象"}])
            for field, item in value.items():
                snapshot[key][field] = deepcopy(item)

        videos = _normalize_videos(body["videos"]) if "videos" in body else []
        draft = {
            "assay": assay,
            "project_id": project_id,
            "videos": videos,
            "options": snapshot["options"],
            "detection": snapshot["detection"],
            "output": snapshot["output"],
            "style": snapshot["style"],
        }
        issues = validate_draft(draft, settings=_settings_snapshot(
            store, draft.get("assay")))
        if issues:
            raise ApiError("草稿未通过校验", 400, issues)

        record = store.create_draft(
            assay,
            project_id=project_id,
            videos=videos,
            options=draft["options"],
            detection=draft["detection"],
            output=draft["output"],
            style=draft["style"],
        )
        return success({"draft": record}, 201)

    def _require_draft(draft_id: str) -> dict:
        draft = store.get_draft(draft_id)
        if draft is None:
            raise ApiError("草稿不存在", 404)
        return draft

    @app.get("/api/drafts/<draft_id>")
    def get_draft(draft_id: str):
        return success({"draft": _require_draft(draft_id)})

    @app.put("/api/drafts/<draft_id>")
    def update_draft(draft_id: str):
        existing = _require_draft(draft_id)
        body = _json_body()
        changes = dict(body)
        if "videos" in changes:
            changes["videos"] = _normalize_videos(changes["videos"])
        candidate = {**existing, **changes}
        issues = validate_draft(
            candidate, settings=_settings_snapshot(store, candidate.get("assay")))
        if issues:
            raise ApiError("草稿未通过校验", 400, issues)
        record = store.update_draft(draft_id, changes)
        return success({"draft": record})

    def _project_animals(draft: Mapping[str, Any]) -> set[str] | None:
        """当前项目可绑定的动物 ID；独立分析没有项目上下文时不校验归属。"""

        project_id = draft.get("project_id")
        if not is_valid_id(project_id):
            return None
        return {item["id"] for item in store.list_animals(project_id)}

    @app.get("/api/drafts/<draft_id>/validate")
    def validate_draft_get(draft_id: str):
        draft = _require_draft(draft_id)
        # 最终校验：允许未画完的草稿保存，但不允许带着缺四角开始任务。
        return success({"issues": validate_draft(
            draft, complete=True, animals=_project_animals(draft),
            settings=_settings_snapshot(store, draft.get("assay")))})

    @app.post("/api/drafts/<draft_id>/validate")
    def validate_draft_post(draft_id: str):
        draft = _require_draft(draft_id)
        return success({"issues": validate_draft(
            draft, complete=True, animals=_project_animals(draft),
            settings=_settings_snapshot(store, draft.get("assay")))})

    def _scheduler() -> JobScheduler:
        return app.config["JOBS"]

    def _batch_error(error: Exception):
        if isinstance(error, NotFoundError):
            raise ApiError("任务不存在", 404)
        if isinstance(error, JobError):
            raise ApiError(error.message, error.status, error.fields)
        raise error

    def _batch_payload(batch: dict) -> dict:
        """批次 + 逐视频行；状态只有一份，不做第二套映射。"""

        payload = dict(batch)
        payload["job_id"] = batch.get("id")
        videos = list(batch.get("videos") or [])
        rows = []
        for video in videos:
            row = {
                "job_id": video.get("job_id"),
                "run_id": video.get("run_id"),
                "name": video.get("name"),
                "status": video.get("status"),
                "progress": video.get("progress"),
                "error": video.get("error"),
            }
            # 只有真的发布了结果才有清单；失败任务不能带一个空清单冒充。
            if isinstance(video.get("result_manifest"), Mapping):
                row["manifest"] = video["result_manifest"]
            if video.get("output_dir"):
                row["output_dir"] = video["output_dir"]
            if video.get("log"):
                row["log"] = video["log"]
            rows.append(row)
        payload["jobs"] = rows
        payload["run_id"] = next(
            (video.get("run_id") for video in videos
             if video.get("status") == COMPLETED), None)
        return payload

    def _submit_draft(draft_id: str):
        try:
            batch = _scheduler().submit(draft_id)
        except Exception as error:  # 统一转成带字段的接口错误
            _batch_error(error)
        return _batch_payload(batch)

    @app.post("/api/drafts/<draft_id>/analyze")
    def analyze_draft(draft_id: str):
        """旧入口保留，但只做同一件事：校验、冻结、排队。"""

        _require_draft(draft_id)
        return success(_submit_draft(draft_id), 202)

    @app.post("/api/jobs")
    def create_job():
        body = _json_body()
        draft_id = body.get("draft_id")
        if not is_valid_id(draft_id):
            raise ApiError("草稿 ID 无效", 400,
                           [{"field": "draft_id", "message": "ID 无效"}])
        _require_draft(draft_id)
        return success(_submit_draft(draft_id), 202)

    @app.get("/api/jobs")
    def list_jobs():
        project_id = request.args.get("project_id")
        if project_id is not None and not is_valid_id(project_id):
            raise ApiError("项目 ID 无效", 400,
                           [{"field": "project_id", "message": "ID 无效"}])
        rows = _scheduler().list(project_id)
        return success({"batches": rows})

    @app.get("/api/jobs/<batch_id>")
    def get_job(batch_id: str):
        if not is_valid_id(batch_id):
            raise ApiError("任务不存在", 404)
        try:
            batch = _scheduler().get(batch_id)
        except Exception as error:
            _batch_error(error)
        return success({"batch": _batch_payload(batch)})

    @app.post("/api/jobs/<batch_id>/cancel")
    def cancel_job(batch_id: str):
        if not is_valid_id(batch_id):
            raise ApiError("任务不存在", 404)
        try:
            batch = _scheduler().cancel(batch_id)
        except Exception as error:
            _batch_error(error)
        return success({"batch": _batch_payload(batch)}, 202)

    @app.post("/api/jobs/<batch_id>/replay")
    def replay_job(batch_id: str):
        if not is_valid_id(batch_id):
            raise ApiError("任务不存在", 404)
        try:
            batch = _scheduler().replay(batch_id)
        except Exception as error:
            _batch_error(error)
        return success({"batch": _batch_payload(batch)}, 202)

    @app.delete("/api/jobs/<batch_id>")
    def delete_job(batch_id: str):
        if not is_valid_id(batch_id):
            raise ApiError("任务不存在", 404)
        try:
            _scheduler().delete(batch_id)
        except Exception as error:
            _batch_error(error)
        return success({"deleted": True})

    @app.post("/api/drafts/<draft_id>/preview")
    def create_preview(draft_id: str):
        draft = _require_draft(draft_id)
        body = _json_body()
        unit_id = body.get("unit_id")
        try:
            result = preview_service.create_preview(
                store, paths, draft_id, body,
                settings=_settings_snapshot(store, draft.get("assay")))
        except ValidationError as error:
            # 字段问题带上单位身份，界面才能定位到具体动物的人工输入。
            raise ApiError(error.message, 400, [error.as_issue(
                unit_id=unit_id if is_valid_id(unit_id) else None)])
        except media_service.MediaError as error:
            raise ApiError(str(error), 422,
                           [{"field": "frame_index", "message": str(error)}])
        return success(result, 201)

    # -- 模板 -----------------------------------------------------------
    @app.get("/api/templates")
    def list_templates():
        assay = request.args.get("assay")
        if assay is not None and not experiments.exists(assay):
            raise ApiError("未知实验", 400,
                           [{"field": "assay", "message": "未知实验"}])
        return success({"templates": store.list_templates(assay)})

    @app.post("/api/templates")
    def create_template():
        body = _json_body()
        assay = body.get("assay")
        if not experiments.exists(assay):
            raise ApiError("未知实验", 400,
                           [{"field": "assay", "message": "未知实验"}])
        unit = body.get("unit")
        if unit is None:
            unit = {"geometry": body.get("geometry"),
                    "options": body.get("options")}
        if not isinstance(unit, Mapping):
            raise ApiError("unit 必须是对象", 400,
                           [{"field": "unit", "message": "必须是对象"}])
        geometry = unit.get("geometry")
        if not isinstance(geometry, Mapping) or not geometry:
            raise ApiError("模板缺少场地", 400,
                           [{"field": "geometry", "message": "至少需要一个场地步骤"}])
        options = unit.get("options")
        if options is not None and not isinstance(options, Mapping):
            raise ApiError("options 必须是对象", 400,
                           [{"field": "options", "message": "必须是对象"}])
        payload = field_templates.build_template(assay, unit, options)
        issues = field_templates.validate_template_payload(payload)
        if issues:
            raise ApiError("模板未通过校验", 400, issues)
        template = store.create_template(
            assay=assay, name=body.get("name"),
            geometry=payload["geometry"], options=payload["options"],
        )
        return success({"template": template}, 201)

    @app.get("/api/templates/<template_id>")
    def get_template(template_id: str):
        template = store.get_template(template_id)
        if template is None:
            raise ApiError("模板不存在", 404)
        return success({"template": template})

    @app.delete("/api/templates/<template_id>")
    def delete_template(template_id: str):
        store.delete_template(template_id)
        return success({"deleted": True})

    # -- 页面 -----------------------------------------------------------
    @app.get("/")
    def page_home():
        return render_template(
            "home.html", active="home", projects=store.list_projects()[:4]
        )

    @app.get("/projects")
    def page_projects():
        return render_template(
            "projects.html", active="projects", projects=store.list_projects()
        )

    def _project_jobs(project_id: str) -> list[dict]:
        """项目批次：与任务中心同一份信息，只按项目过滤。"""

        return app.config["JOBS"].list(project_id)

    @app.get("/projects/<project_id>")
    def page_project(project_id: str):
        project = store.get_project(project_id)
        if project is None:
            return render_template("not_found.html"), 404
        return render_template(
            "project.html",
            active="projects",
            project=project,
            groups=store.list_groups(project_id),
            animals=store.list_animals(project_id),
            assets=store.list_assets_for_scope(project_id),
            drafts=store.list_drafts(project_id),
            jobs=_project_jobs(project_id),
            catalog=experiments.catalog(),
        )

    @app.get("/analyze")
    def page_analyze():
        mode = request.args.get("mode", "project")
        if mode not in ("project", "independent"):
            mode = "project"
        project_id = request.args.get("project")
        if project_id is not None and store.get_project(project_id) is None:
            project_id = None
        draft_id = request.args.get("draft_id") or ""
        draft = store.get_draft(draft_id) if is_valid_id(draft_id) else None
        if draft is None:
            draft_id = ""
        step = request.args.get("step", "select")
        if step not in ("select", "range", "arena", "review"):
            step = "select"
        view = "detection" if request.args.get("view") == "detection" else "geometry"
        assay = draft["assay"] if draft else ""
        return render_template(
            "analyze.html",
            active="analyze",
            mode=mode,
            project_id=project_id,
            projects=store.list_projects(),
            draft_id=draft_id,
            step=step,
            view=view,
            assay=assay,
            catalog=experiments.catalog(),
        )

    @app.get("/tasks")
    def page_tasks():
        selected = request.args.get("job")
        batches = app.config["JOBS"].list()
        # “继续设置”是未提交的独立草稿：有视频、且没有任何批次引用它；
        # 项目草稿走项目页，不混进独立入口。
        submitted = {batch.get("draft_id") for batch in store.list_batches()
                     if batch.get("draft_id")}
        drafts = [draft for draft in store.list_drafts()
                  if draft.get("project_id") is None
                  and draft.get("videos")
                  and draft.get("id") not in submitted]
        return render_template(
            "tasks.html", active="tasks", batches=batches, drafts=drafts,
            selected=selected if is_valid_id(selected) else None,
        )

    @app.get("/results/<run_id>")
    def page_result(run_id: str):
        if not is_valid_id(run_id):
            return render_template("not_found.html"), 404
        run = run_service.resolve_run(store, paths, run_id)
        if run is None:
            # 页面与下载共用同一发布/状态校验：失败或未发布的清单不进
            # fallback，不能因为 results.json 可读就渲染成正常结果。
            return render_template("not_found.html"), 404
        results = _run_payloads(run["dir"])
        return render_template(
            "result.html", active="tasks", run_id=run_id, results=results,
            run=run,
            run_spec=run_service.spec_of(run.get("assay")) if run else {},
        )

    # -- 已发布运行：清单、声明文件、打包与打开目录 ---------------------
    def _require_run(run_id: str) -> dict:
        if not is_valid_id(run_id):
            raise ApiError("结果不存在", 404)
        run = run_service.resolve_run(store, paths, run_id)
        if run is None:
            raise ApiError("结果不存在或尚未发布", 404)
        return run

    @app.get("/api/runs/<run_id>")
    def api_run(run_id: str):
        run = _require_run(run_id)
        return success({
            "job": run["job"],
            "manifest": run["manifest"],
            "summary": [unit.get("summary") or {} for unit in run["summary"]],
            "files": run["files"],
            "spec": run_service.spec_of(run["assay"]),
            "status": run["status"],
        })

    @app.get("/api/runs/<run_id>/files/<path:relative>")
    def api_run_file(run_id: str, relative: str):
        run = _require_run(run_id)
        if not run_service.declared(run["manifest"], relative):
            raise ApiError("文件未在清单中声明", 404)
        path = run_service.safe_file(run["dir"], relative)
        if path is None:
            raise ApiError("结果文件不存在", 404)
        if request.args.get("download"):
            return send_file(path, as_attachment=True,
                             download_name=Path(relative).name)
        mimetype = None
        if relative.lower().endswith(".mp4"):
            mimetype = "video/mp4"
        elif relative.lower().endswith(".png"):
            mimetype = "image/png"
        return send_file(path, mimetype=mimetype, conditional=True)

    @app.get("/api/runs/<run_id>/bundle")
    def api_run_bundle(run_id: str):
        import tempfile

        run = _require_run(run_id)
        asset = run.get("asset") or {}
        name = run_service.bundle_name(asset.get("name"))
        handle = tempfile.NamedTemporaryFile(prefix="bln-bundle-", suffix=".zip",
                                             delete=False)
        handle.close()
        target = Path(handle.name)
        try:
            run_service.write_bundle(target, run)
        except OSError as error:
            target.unlink(missing_ok=True)
            raise ApiError(f"无法打包结果：{error}", 500)

        @after_this_request
        def _cleanup(response):
            try:
                target.unlink(missing_ok=True)
            except OSError:
                pass
            return response

        return send_file(target, mimetype="application/zip",
                         as_attachment=True, download_name=name)

    @app.post("/api/runs/<run_id>/open")
    def api_run_open(run_id: str):
        import subprocess

        run = _require_run(run_id)
        directory = Path(run["dir"]).resolve()
        if os.name != "nt":
            return success({"path": str(directory), "opened": False})
        try:
            subprocess.Popen(["explorer", str(directory)],
                             creationflags=child_creation_flags())
        except OSError as error:
            raise ApiError(f"无法打开结果目录：{error}", 500)
        return success({"path": str(directory), "opened": True})

    # -- 经典关键点会话 -------------------------------------------------
    @app.get("/api/landmarks")
    def list_landmarks():
        return success({"sessions": landmark_service.list_sessions(paths)})

    @app.post("/api/landmarks")
    def create_landmark():
        body = _json_body()
        try:
            session = landmark_service.create_session(store, paths, body)
        except ValidationError as error:
            raise ApiError(error.message, 400, [error.as_issue()])
        return success({"session": session}, 201)

    @app.get("/api/landmarks/<session_id>")
    def get_landmark(session_id: str):
        session = landmark_service.get_session(paths, session_id)
        if session is None:
            raise ApiError("会话不存在", 404)
        return success({"session": session})

    @app.put("/api/landmarks/<session_id>")
    def update_landmark(session_id: str):
        body = _json_body()
        try:
            session = landmark_service.update_session(paths, session_id, body)
        except ValidationError as error:
            status = 404 if error.message == "会话不存在" else 400
            raise ApiError(error.message, status, [error.as_issue()])
        return success({"session": session})

    @app.post("/api/landmarks/<session_id>/evaluate")
    def evaluate_landmark(session_id: str):
        session = landmark_service.get_session(paths, session_id)
        if session is None:
            raise ApiError("会话不存在", 404)
        try:
            evaluation = landmark_service.evaluate(paths, session)
            session = landmark_service.save_evaluation(paths, session,
                                                       evaluation)
        except ValidationError as error:
            raise ApiError(error.message, 400, [error.as_issue()])
        except media_service.MediaError as error:
            raise ApiError(str(error), 422, [{"field": "frames",
                                              "message": str(error)}])
        return success({"session": session, "evaluation": evaluation})

    @app.get("/landmarks")
    def page_landmarks():
        """只从已有草稿的几何里选视频/动物；不在这里新建分析配置。"""

        by_asset: dict[str, dict] = {}
        for draft in store.list_drafts():
            for video in draft.get("videos") or []:
                asset_id = video.get("asset_id")
                if not is_valid_id(asset_id):
                    continue
                entry = by_asset.setdefault(str(asset_id), {"units": [],
                                                            "asset": None})
                for unit in video.get("units") or []:
                    if not isinstance(unit.get("geometry"), Mapping):
                        continue
                    if not unit.get("geometry"):
                        continue
                    entry["units"].append({
                        "id": unit.get("id"),
                        "name": unit.get("name"),
                        "assay": draft.get("assay"),
                        "draft_id": draft.get("id"),
                        "geometry": unit.get("geometry"),
                        "options": unit.get("options") or {},
                        "detection": unit.get("detection") or {},
                    })
        media = []
        for asset_id, entry in by_asset.items():
            if not entry["units"]:
                continue
            asset = store.get_asset(asset_id)
            if asset is None:
                continue
            entry["asset"] = asset
            media.append(entry)
        return render_template("landmarks.html", active="settings",
                               sessions=landmark_service.list_sessions(paths),
                               media=media)

    @app.get("/settings")
    def page_settings():
        values = {section: _settings_value(store, section)
                  for section in SETTINGS_SECTIONS}
        catalog = experiments.catalog()
        assay_defaults = []
        assay_detection = []
        for entry in catalog:
            merged = deepcopy(entry["defaults"])
            saved = values["defaults"].get(entry["key"])
            if isinstance(saved, Mapping):
                merged.update(deepcopy(dict(saved)))
            assay_defaults.append({
                "key": entry["key"], "label": entry["label"],
                "short": entry["short"], "parameters": entry["parameters"],
                "current": merged,
            })
            detection_values = web_descriptors.detection_defaults(entry["key"])
            saved_detection = values["detection"].get(entry["key"])
            if isinstance(saved_detection, Mapping):
                detection_values.update(deepcopy(dict(saved_detection)))
            assay_detection.append({
                "key": entry["key"], "label": entry["label"],
                "short": entry["short"], "current": detection_values,
                "overrides": deepcopy(
                    getattr(experiments.get(entry["key"]),
                            "DETECTION_DEFAULTS", {})),
            })
        return render_template(
            "settings.html", active="settings",
            assay_defaults=assay_defaults,
            assay_detection=assay_detection,
            output_values=values["output"],
            style_values=values["style"],
            detection_fields=web_descriptors.DETECTION_FIELDS,
            style_fields=web_descriptors.STYLE_FIELDS,
            output_fields=web_descriptors.OUTPUT_FIELDS,
        )

    return app
