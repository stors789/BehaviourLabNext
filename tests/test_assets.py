"""真实媒体导入、上传去重、背景图与媒体读取。"""

from __future__ import annotations

import io
import os
import re
from pathlib import Path

import pytest
from werkzeug.datastructures import FileStorage

from behaviourlab import assets as asset_service
from behaviourlab import media


def fake_probe(path: str) -> dict:
    return {
        "width": 640,
        "height": 480,
        "fps": 30.0,
        "frame_count": 300,
        "duration_s": 10.0,
        "codec": "h264",
    }


def video_file(tmp_path: Path, name: str, content: bytes = b"fake-video-bytes") -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def upload(filename: str, content: bytes) -> FileStorage:
    return FileStorage(stream=io.BytesIO(content), filename=filename)


def test_import_links_and_dedupes_batch(tmp_path, store, client, monkeypatch):
    root = tmp_path / "媒体"
    first = video_file(root, "动物 甲.mp4")
    second = video_file(root, "动物 乙.mp4")
    broken = video_file(root, "broken.avi")

    def probe(path: str) -> dict:
        if Path(path).name == "broken.avi":
            raise media.MediaError("没有视频流")
        return fake_probe(path)

    monkeypatch.setattr(asset_service.media, "probe_video", probe)

    response = client.post(
        "/api/assets/import",
        json={"paths": [str(first), str(first), str(broken), str(second)]},
    )
    assert response.status_code == 200
    data = response.get_json()["data"]
    assert len(data["assets"]) == 2
    assert len(data["errors"]) == 1
    assert data["errors"][0]["name"] == "broken.avi"
    assert "没有视频流" in data["errors"][0]["message"]
    assert {asset["ownership"] for asset in data["assets"]} == {"linked"}
    assert {asset["name"] for asset in data["assets"]} == {"动物 甲.mp4", "动物 乙.mp4"}

    first_id = next(
        asset["id"] for asset in data["assets"] if asset["name"] == "动物 甲.mp4"
    )
    again = client.post("/api/assets/import", json={"paths": [str(first)]})
    assert again.status_code == 200
    assert again.get_json()["data"]["assets"][0]["id"] == first_id
    assert len(client.get("/api/assets").get_json()["data"]["assets"]) == 2


def test_import_dedupes_case_folded_paths(tmp_path, store):
    if os.name != "nt":
        pytest.skip("大小写折叠仅 Windows 路径需要")
    clip = video_file(tmp_path, "Clip.MP4")
    result = asset_service.import_paths(
        store, [str(clip), str(clip).lower()], None, probe=fake_probe
    )
    assert len(result["assets"]) == 1
    assert result["errors"] == []


def test_import_same_name_different_directories(tmp_path, store, client, monkeypatch):
    monkeypatch.setattr(asset_service.media, "probe_video", fake_probe)
    first = video_file(tmp_path / "组一", "clip.mp4")
    second = video_file(tmp_path / "组二", "clip.mp4")

    response = client.post(
        "/api/assets/import", json={"paths": [str(first), str(second)]}
    )
    assets = response.get_json()["data"]["assets"]
    assert len(assets) == 2
    assert len({asset["id"] for asset in assets}) == 2
    assert {asset["name"] for asset in assets} == {"clip.mp4"}


def test_import_directory_sorts_and_filters(tmp_path, store, client, monkeypatch):
    monkeypatch.setattr(asset_service.media, "probe_video", fake_probe)
    folder = tmp_path / "录像"
    video_file(folder, "b.MP4")
    video_file(folder, "a.avi")
    (folder / "note.txt").write_text("not a video", encoding="utf-8")
    video_file(folder / "sub", "c.mkv")

    response = client.post("/api/assets/import", json={"paths": [str(folder)]})
    data = response.get_json()["data"]
    assert data["errors"] == []
    assert [asset["name"] for asset in data["assets"]] == ["a.avi", "b.MP4", "c.mkv"]


def test_import_project_scope_and_validation(tmp_path, store, client, monkeypatch):
    monkeypatch.setattr(asset_service.media, "probe_video", fake_probe)
    project = store.create_project("项目甲")
    clip = video_file(tmp_path, "clip.mp4")

    scoped = client.post(
        "/api/assets/import",
        json={"paths": [str(clip)], "project_id": project["id"]},
    )
    assert scoped.status_code == 200
    scoped_asset = scoped.get_json()["data"]["assets"][0]
    assert scoped_asset["project_id"] == project["id"]

    unbound = client.post("/api/assets/import", json={"paths": [str(clip)]})
    assert unbound.status_code == 200
    assert unbound.get_json()["data"]["assets"][0]["project_id"] is None

    filtered = client.get(f"/api/assets?project_id={project['id']}").get_json()["data"]
    assert [asset["id"] for asset in filtered["assets"]] == [scoped_asset["id"]]

    missing = client.post(
        "/api/assets/import", json={"paths": [str(clip)], "project_id": "missing"}
    )
    assert missing.status_code == 404

    invalid = client.post("/api/assets/import", json={"paths": "clip.mp4"})
    assert invalid.status_code == 400
    assert invalid.get_json()["error"]["fields"][0]["field"] == "paths"


def test_upload_creates_owned_copy_and_reuses_duplicate(client, store, monkeypatch):
    monkeypatch.setattr(asset_service.media, "probe_video", fake_probe)
    payload = b"identical-video-bytes"

    first = client.post(
        "/api/assets/upload",
        data={"files": (io.BytesIO(payload), "动物 甲.MP4")},
        content_type="multipart/form-data",
    )
    assert first.status_code == 200
    asset = first.get_json()["data"]["assets"][0]
    assert asset["name"] == "动物 甲.MP4"
    assert asset["ownership"] == "owned"
    assert asset["upload_sha256"]
    target = Path(asset["path"])
    assert target.parent == store.paths.videos
    assert re.fullmatch(r"[0-9a-f]{32}\.mp4", target.name)
    assert target.exists()

    second = client.post(
        "/api/assets/upload",
        data={"files": (io.BytesIO(payload), "another-name.mp4")},
        content_type="multipart/form-data",
    )
    assert second.status_code == 200
    reused = second.get_json()["data"]["assets"][0]
    assert reused["id"] == asset["id"]
    assert len(list(store.paths.videos.glob("*.mp4"))) == 1
    assert not list(store.paths.videos.glob("*.partial"))


def test_upload_display_name_strips_windows_directories(store):
    result = asset_service.import_uploads(
        store,
        [FileStorage(stream=io.BytesIO(b"x"), filename="C:\\实验数据\\动物 甲.MP4")],
        None,
        probe=fake_probe,
    )
    assert result["errors"] == []
    assert result["assets"][0]["name"] == "动物 甲.MP4"
    assert Path(result["assets"][0]["path"]).suffix == ".mp4"


def test_upload_same_bytes_in_other_project_is_separate(client, store, monkeypatch):
    monkeypatch.setattr(asset_service.media, "probe_video", fake_probe)
    project = store.create_project("项目乙")
    payload = b"shared-bytes"

    global_upload = client.post(
        "/api/assets/upload",
        data={"files": (io.BytesIO(payload), "clip.mp4")},
        content_type="multipart/form-data",
    ).get_json()["data"]["assets"][0]
    project_upload = client.post(
        "/api/assets/upload",
        data={
            "files": (io.BytesIO(payload), "clip.mp4"),
            "project_id": project["id"],
        },
        content_type="multipart/form-data",
    ).get_json()["data"]["assets"][0]

    assert global_upload["id"] != project_upload["id"]
    assert global_upload["project_id"] is None
    assert project_upload["project_id"] == project["id"]


def test_upload_failure_cleans_only_new_copy(store, monkeypatch):
    monkeypatch.setattr(asset_service.media, "probe_video", fake_probe)
    good = asset_service.import_uploads(
        store, [upload("good.mp4", b"good")], None
    )["assets"][0]
    before = sorted(path.name for path in store.paths.videos.iterdir())

    def fail(path: str) -> dict:
        raise media.MediaError("没有视频流")

    result = asset_service.import_uploads(
        store, [upload("bad.avi", b"bad")], None, probe=fail
    )
    assert result["assets"] == []
    assert result["errors"][0]["name"] == "bad.avi"
    assert sorted(path.name for path in store.paths.videos.iterdir()) == before
    assert Path(good["path"]).exists()
    assert not list(store.paths.videos.glob("*.partial"))


def test_background_upload_reencodes_png(client, store):
    import cv2
    import numpy as np

    image = np.zeros((8, 6, 3), dtype=np.uint8)
    image[:, :, 2] = 255
    ok, encoded = cv2.imencode(".png", image)
    assert ok

    response = client.post(
        "/api/backgrounds/upload",
        data={"file": (io.BytesIO(encoded.tobytes()), "bg.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 201
    data = response.get_json()["data"]
    assert Path(data["path"]).parent == store.paths.backgrounds
    assert re.fullmatch(r"/api/backgrounds/[0-9a-f]{32}\.png", data["url"])

    served = client.get(data["url"])
    assert served.status_code == 200
    assert served.mimetype == "image/png"
    assert served.data.startswith(b"\x89PNG")

    invalid = client.post(
        "/api/backgrounds/upload",
        data={"file": (io.BytesIO(b"not-an-image"), "bad.png")},
        content_type="multipart/form-data",
    )
    assert invalid.status_code == 400
    assert invalid.get_json()["error"]["fields"][0]["field"] == "file"


def test_background_validate_decodable_before_replace(client, store, tmp_path):
    """原生背景确认合同：可解码 200，缺失/坏图 400 且字段可读。"""
    import cv2
    import numpy as np

    image = np.zeros((6, 8, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    good = tmp_path / "good.png"
    good.write_bytes(encoded.tobytes())

    response = client.post("/api/backgrounds/validate",
                           json={"path": str(good)})
    assert response.status_code == 200
    assert response.get_json()["data"]["ok"] is True

    missing = client.post("/api/backgrounds/validate",
                          json={"path": str(tmp_path / "不存在.png")})
    assert missing.status_code == 400
    assert "背景文件不存在" in missing.get_json()["error"]["message"]

    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not-an-image")
    broken = client.post("/api/backgrounds/validate", json={"path": str(bad)})
    assert broken.status_code == 400
    assert "背景无法解码" in broken.get_json()["error"]["message"]

    empty = client.post("/api/backgrounds/validate", json={"path": "   "})
    assert empty.status_code == 400
    assert empty.get_json()["error"]["fields"][0]["field"] == "path"

    # 0 字节 PNG 不再冒泡成服务器内部错误，而是可理解的背景业务错误。
    zero_byte = tmp_path / "zero.png"
    zero_byte.write_bytes(b"")
    zero = client.post("/api/backgrounds/validate",
                       json={"path": str(zero_byte)})
    assert zero.status_code == 400
    assert "背景文件为空" in zero.get_json()["error"]["message"]


def test_media_routes_only_accept_registered_names(client):
    assert client.get("/api/backgrounds/not-a-name.png").status_code == 404
    assert client.get("/api/backgrounds/../behaviourlab.sqlite3").status_code == 404
    assert client.get("/api/previews/random.png").status_code == 404


def test_asset_stream_supports_range(client, store, tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"0123456789abcdef")
    asset = store.create_asset(name="clip.mp4", path=str(source), metadata=fake_probe(""))

    response = client.get(
        f"/api/assets/{asset['id']}/stream", headers={"Range": "bytes=0-3"}
    )
    assert response.status_code == 206
    assert response.data == b"0123"
    assert response.headers["Accept-Ranges"] == "bytes"

    missing = client.get("/api/assets/missing/stream")
    assert missing.status_code == 404
