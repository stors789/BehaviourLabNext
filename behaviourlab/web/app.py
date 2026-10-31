"""本地 Flask 服务：API 骨架与页面路由。"""

from __future__ import annotations

import logging
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from flask import Flask, Response, jsonify, render_template, request, send_file
from werkzeug.exceptions import HTTPException

from .. import assets as asset_service
from .. import experiments
from .. import media as media_service
from ..assets import BackgroundError
from ..config import APP_NAME, APP_VERSION, SCHEMA_VERSION, DataPaths
from ..detection import common_detection_defaults
from ..engine import (
    COMMON_OUTPUT_DEFAULTS,
    COMMON_STYLE_DEFAULTS,
    snapshot_draft,
    validate_draft,
    validate_settings_value,
)
from ..processes import JobOwnerRegistry
from ..schema import ValidationError, is_valid_id, new_id
from ..store import NotFoundError, Store

LOGGER = logging.getLogger(__name__)

SETTINGS_SECTIONS = ("defaults", "detection", "output", "style")
BACKGROUND_NAME = re.compile(r"[0-9a-f]{32}\.png")
PREVIEW_NAME = re.compile(r"[0-9a-f]{32}\.(?:png|jpg|jpeg)")


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


def create_app(*, data_dir: Any = None, store: Store | None = None,
               testing: bool = False) -> Flask:
    paths = DataPaths.from_dir(data_dir)
    store = store or Store(paths)
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config.update(
        TESTING=testing,
        DATA_PATHS=paths,
        STORE=store,
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

    # -- 健康与目录 -----------------------------------------------------
    @app.get("/api/health")
    def health():
        registry = JobOwnerRegistry(paths)
        return success(
            {
                "status": "ok",
                "version": APP_VERSION,
                "schema_version": SCHEMA_VERSION,
                "data_root": str(paths.root),
                "database": str(paths.database),
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

    @app.get("/api/previews/<filename>")
    def get_preview(filename: str):
        if not PREVIEW_NAME.fullmatch(filename):
            raise ApiError("预览不存在", 404)
        path = Path(paths.previews) / filename
        if not path.is_file():
            raise ApiError("预览不存在", 404)
        return send_file(path, conditional=True)

    # -- 设置 -----------------------------------------------------------
    def _defaults_for(section: str) -> dict:
        if section == "output":
            return deepcopy(COMMON_OUTPUT_DEFAULTS)
        if section == "style":
            return deepcopy(COMMON_STYLE_DEFAULTS)
        return {}

    @app.get("/api/settings/<section>")
    def get_settings(section: str):
        if section not in SETTINGS_SECTIONS:
            raise ApiError("未知设置分区", 404)
        value = store.get_setting(f"settings.{section}", default=_defaults_for(section))
        return success({"value": value})

    @app.put("/api/settings/<section>")
    def put_settings(section: str):
        if section not in SETTINGS_SECTIONS:
            raise ApiError("未知设置分区", 404)
        body = _json_body()
        if "value" not in body:
            raise ApiError("缺少 value", 400,
                           [{"field": "value", "message": "必须提供 value"}])
        value = body["value"]
        issues = validate_settings_value(section, value)
        if issues:
            raise ApiError("设置未通过校验", 400, issues)
        saved = store.set_setting(f"settings.{section}", value)
        return success({"value": saved})

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
        created = []
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
        else:
            created = [store.create_group(project_id, body.get("name"))]
        return success({"groups": created}, 201)

    @app.get("/api/projects/<project_id>/animals")
    def list_animals(project_id: str):
        if store.get_project(project_id) is None:
            raise ApiError("项目不存在", 404)
        return success({"animals": store.list_animals(project_id)})

    @app.post("/api/projects/<project_id>/animals")
    def create_animals(project_id: str):
        body = _json_body()
        items = body.get("animals")
        if items is None:
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
        assay = body.get("assay")
        if not isinstance(assay, str) or not experiments.exists(assay):
            raise ApiError("未知实验", 400, [{"field": "assay", "message": "未知实验"}])
        project_id = body.get("project_id")
        if project_id is not None and not is_valid_id(project_id):
            raise ApiError("项目 ID 无效", 400,
                           [{"field": "project_id", "message": "ID 无效"}])
        if project_id is not None and store.get_project(project_id) is None:
            raise ApiError("项目不存在", 404)

        saved_defaults = store.get_setting("settings.defaults", default={}) or {}
        saved_detection = store.get_setting("settings.detection", default={}) or {}
        if not isinstance(saved_defaults, Mapping):
            saved_defaults = {}
        if not isinstance(saved_detection, Mapping):
            saved_detection = {}

        snapshot = snapshot_draft(
            assay,
            defaults=saved_defaults.get(assay),
            detection=saved_detection.get(assay),
            output=store.get_setting("settings.output", default=None),
            style=store.get_setting("settings.style", default=None),
        )

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
        issues = validate_draft(draft)
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
        issues = validate_draft(candidate)
        if issues:
            raise ApiError("草稿未通过校验", 400, issues)
        record = store.update_draft(draft_id, changes)
        return success({"draft": record})

    @app.get("/api/drafts/<draft_id>/validate")
    def validate_draft_get(draft_id: str):
        draft = _require_draft(draft_id)
        return success({"issues": validate_draft(draft)})

    @app.post("/api/drafts/<draft_id>/validate")
    def validate_draft_post(draft_id: str):
        draft = _require_draft(draft_id)
        return success({"issues": validate_draft(draft)})

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
        )

    @app.get("/analyze")
    def page_analyze():
        mode = request.args.get("mode", "project")
        if mode not in ("project", "independent"):
            mode = "project"
        project_id = request.args.get("project")
        if project_id is not None and store.get_project(project_id) is None:
            project_id = None
        return render_template(
            "analyze.html",
            active="analyze",
            mode=mode,
            project_id=project_id,
            projects=store.list_projects(),
        )

    @app.get("/tasks")
    def page_tasks():
        return render_template("tasks.html", active="tasks")

    @app.get("/results/<run_id>")
    def page_result(run_id: str):
        return render_template("result.html", active="tasks", run_id=run_id)

    @app.get("/settings")
    def page_settings():
        values = {
            section: store.get_setting(
                f"settings.{section}", default=_defaults_for(section)
            )
            for section in SETTINGS_SECTIONS
        }
        return render_template("settings.html", active="settings", values=values)

    @app.get("/landmarks")
    def page_landmarks():
        return render_template("landmarks.html", active="landmarks")

    return app
