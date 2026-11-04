"""预览背景构建失败的处理：降级、原因标注与行为预览的明确报错。

数据说明：
- 带 fake 视频的用例使用假字节文件 + 注入的 FrameInfo（合成数据），
  用于隔离"背景构建失败"本身；
- 静态背景用例中"尺寸不符"写入真实 PNG（真实解码路径）；
- `test_temporal_preview_real_video_reports_detection`、身份变化、等待
  超时三个用例使用 PyAV 编码的真实 H.264 MP4（身份/等待走真实分支；
  超时阈值缩短到 0.003 s，取消用例在 build 抛出点注入，见各用例说明）。
"""

from __future__ import annotations

import threading

import cv2
import numpy as np
import pytest

from behaviourlab import detection, preview
from behaviourlab.media import FrameInfo


@pytest.fixture(autouse=True)
def _isolated_background_cache():
    """背景缓存/在途注册表是模块级全局：用例间互不串用。"""

    detection.clear_background_cache()
    yield
    detection.clear_background_cache()


def _preview(client, draft, asset, *, kind, frame_index=7, time_s=None):
    payload = {"asset_id": asset["id"], "unit_id": "a" * 32, "kind": kind}
    if time_s is not None:
        payload["time_s"] = time_s
    else:
        payload["frame_index"] = frame_index
    return client.post(f"/api/drafts/{draft['id']}/preview", json=payload)


def test_auto_temporal_undecodable_video_degrades_with_clear_reason(
        client, store, tmp_path, monkeypatch):
    """自动检测预览：时序背景解码失败 → 降级 + 明确原因，不冒充人工路径。"""

    asset, draft = _seed_oft(store, tmp_path)
    monkeypatch.setattr(
        preview.media, "read_frame_info",
        lambda *args, **kwargs: FrameInfo(
            image=np.zeros((200, 200, 3), dtype=np.uint8), index=7,
            time_s=0.28, pts=700, time_base=None, clock_source="pts"))

    response = _preview(client, draft, asset, kind="detection")
    assert response.status_code == 201, response.get_json()
    data = response.get_json()["data"]
    summary = data["summary"]
    assert summary["detection_skipped_reason"] == "background_unavailable"
    assert summary["detection_available"] is False
    assert summary["detection"] is None
    assert any("背景构建失败" in item for item in data["warnings"])
    assert not any("人工记录路径" in item for item in data["warnings"]), \
        "自动路径不得显示人工记录提示"


def test_static_background_missing_file_degrades(
        client, store, tmp_path, monkeypatch):
    """静态背景文件不存在：同类预期失败，同样降级并写明原因。"""

    missing = tmp_path / "missing-bg.png"
    asset, draft = _seed_oft(
        store, tmp_path,
        detection={"bg_method": "static",
                   "static_background": str(missing)})
    monkeypatch.setattr(
        preview.media, "read_frame_info",
        lambda *args, **kwargs: FrameInfo(
            image=np.zeros((200, 200, 3), dtype=np.uint8), index=7,
            time_s=0.28, pts=700, time_base=None, clock_source="pts"))

    response = _preview(client, draft, asset, kind="detection")
    assert response.status_code == 201, response.get_json()
    data = response.get_json()["data"]
    assert data["summary"]["detection_skipped_reason"] == \
        "background_unavailable"
    assert any("静态背景不可读取" in item for item in data["warnings"])


def test_static_background_undecodable_degrades(
        client, store, tmp_path, monkeypatch):
    """静态背景无法解码（坏字节）：真实解码路径失败后降级。"""

    bad = tmp_path / "bad-bg.png"
    bad.write_bytes(b"not-an-image")
    asset, draft = _seed_oft(
        store, tmp_path,
        detection={"bg_method": "static", "static_background": str(bad)})
    monkeypatch.setattr(
        preview.media, "read_frame_info",
        lambda *args, **kwargs: FrameInfo(
            image=np.zeros((200, 200, 3), dtype=np.uint8), index=7,
            time_s=0.28, pts=700, time_base=None, clock_source="pts"))

    response = _preview(client, draft, asset, kind="detection")
    assert response.status_code == 201, response.get_json()
    data = response.get_json()["data"]
    assert any("静态背景无法解码" in item for item in data["warnings"])


def test_static_background_size_mismatch_degrades(
        client, store, tmp_path, monkeypatch):
    """静态背景尺寸与源帧不一致：真实 PNG 解码后按尺寸拒绝并降级。"""

    wrong = tmp_path / "wrong-size-bg.png"
    ok, buffer = cv2.imencode(".png", np.full((60, 80, 3), 255, np.uint8))
    assert ok
    wrong.write_bytes(buffer.tobytes())
    asset, draft = _seed_oft(
        store, tmp_path,
        detection={"bg_method": "static", "static_background": str(wrong)})
    monkeypatch.setattr(
        preview.media, "read_frame_info",
        lambda *args, **kwargs: FrameInfo(
            image=np.zeros((200, 200, 3), dtype=np.uint8), index=7,
            time_s=0.28, pts=700, time_base=None, clock_source="pts"))

    response = _preview(client, draft, asset, kind="detection")
    assert response.status_code == 201, response.get_json()
    data = response.get_json()["data"]
    assert any("静态背景尺寸与源帧不一致" in item for item in data["warnings"])


def test_behavior_preview_background_failure_returns_clear_error(
        client, store, tmp_path, monkeypatch):
    """行为预览遇背景失败：不再二次构建后漏出 422，改为明确报错。"""

    asset, draft = _seed_oft(store, tmp_path)
    monkeypatch.setattr(
        preview.media, "read_frame_at_time",
        lambda *args, **kwargs: FrameInfo(
            image=np.zeros((200, 200, 3), dtype=np.uint8), index=7,
            time_s=0.28, pts=700, time_base=None, clock_source="pts"))

    response = _preview(client, draft, asset, kind="behavior", time_s=0.28)
    assert response.status_code == 400, response.get_json()
    message = response.get_json()["error"]["message"]
    assert "背景构建失败" in message
    assert "行为预览" in message
    assert "无法降级为有效" in message


def test_temporal_preview_real_video_reports_detection(
        client, store, tmp_path):
    """真实 H.264 视频 + 时序默认背景：正常检测预览不受降级逻辑影响。"""

    import av

    source = tmp_path / "real.mp4"
    frames = []
    for index in range(30):
        image = np.full((96, 128, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (10 + 3 * index, 40), (40 + 3 * index, 70),
                      (0, 0, 0), -1)
        frames.append(image)
    container = av.open(str(source), mode="w")
    stream = container.add_stream("libx264", rate=10)
    stream.width = 128
    stream.height = 96
    stream.pix_fmt = "yuv420p"
    for image in frames:
        for packet in stream.encode(
                av.VideoFrame.from_ndarray(image, format="bgr24")):
            container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()

    asset = store.create_asset(
        name="real.mp4", path=str(source), project_id=None,
        metadata={"width": 128, "height": 96, "fps": 10.0,
                  "frame_count": 30, "duration_s": 3.0},
    )
    draft = store.create_draft("oft", videos=[{
        "asset_id": asset["id"],
        "window": {"start_s": 0, "end_s": 3.0},
        "orientation": 0, "fps_override": None, "sample_fps": None,
        "units": [{
            "id": "a" * 32, "name": "小鼠 1", "window": None,
            "geometry": {"arena": {"type": "quad",
                                   "points": [[0, 0], [127, 0],
                                              [127, 95], [0, 95]]}},
            "options": {},
            # 不写 detection：显式验证工厂默认 temporal 本身可用。
            "detection": {},
            "manual": None,
        }],
    }])

    response = _preview(client, draft, asset, kind="detection", frame_index=15)
    assert response.status_code == 201, response.get_json()
    data = response.get_json()["data"]
    summary = data["summary"]
    assert summary["detection_skipped_reason"] is None
    assert summary["detection_available"] is True
    assert summary["detection"] is not None
    assert summary["detection"]["candidates"] >= 1
    assert not any("背景构建失败" in item for item in data["warnings"])


def test_manual_ymaze_behavior_preview_still_reports_manual_reason(
        client, store, tmp_path, monkeypatch):
    """人工 Y 行为预览回归：跳过原因标为 manual，提示保持原文。"""

    from test_preview_api import ClockSource

    asset, draft = _seed_ymaze_manual(store, tmp_path)
    monkeypatch.setattr(preview.media, "VideoSource", ClockSource)
    monkeypatch.setattr(
        preview.media, "read_frame_at_time",
        lambda *args, **kwargs: FrameInfo(
            image=np.zeros((200, 200, 3), dtype=np.uint8), index=90,
            time_s=2.99997, pts=900, time_base=None, clock_source="pts"))

    response = _preview(client, draft, asset, kind="behavior", time_s=2.99997)
    assert response.status_code == 201, response.get_json()
    data = response.get_json()["data"]
    assert data["summary"]["detection_skipped_reason"] == "manual"
    assert data["summary"]["detection_available"] is False
    assert any("人工记录路径不运行自动检测" in item
               for item in data["warnings"])
    assert not any("背景构建失败" in item for item in data["warnings"])


def test_geometry_error_is_not_swallowed_as_background_failure(
        client, store, tmp_path, monkeypatch):
    """非背景类检测异常（缺检测区域）仍然 400，不被降级逻辑吞掉。"""

    asset, draft = _seed_oft(store, tmp_path, geometry={})
    monkeypatch.setattr(
        preview.media, "read_frame_info",
        lambda *args, **kwargs: FrameInfo(
            image=np.zeros((200, 200, 3), dtype=np.uint8), index=7,
            time_s=0.28, pts=700, time_base=None, clock_source="pts"))

    response = _preview(client, draft, asset, kind="detection")
    assert response.status_code == 400, response.get_json()
    message = response.get_json()["error"]["message"]
    assert "背景构建失败" not in message


def _write_real_mp4(path, *, count=30):
    """PyAV 编码真实 H.264：白底黑方块逐帧右移。"""

    import av

    frames = []
    for index in range(count):
        image = np.full((96, 128, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (10 + 3 * index, 40), (40 + 3 * index, 70),
                      (0, 0, 0), -1)
        frames.append(image)
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=10)
    stream.width = 128
    stream.height = 96
    stream.pix_fmt = "yuv420p"
    for image in frames:
        for packet in stream.encode(
                av.VideoFrame.from_ndarray(image, format="bgr24")):
            container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()


def _seed_real_video_draft(store, tmp_path):
    source = tmp_path / "real.mp4"
    _write_real_mp4(source)
    asset = store.create_asset(
        name="real.mp4", path=str(source), project_id=None,
        metadata={"width": 128, "height": 96, "fps": 10.0,
                  "frame_count": 30, "duration_s": 3.0},
    )
    draft = store.create_draft("oft", videos=[{
        "asset_id": asset["id"],
        "window": {"start_s": 0, "end_s": 3.0},
        "orientation": 0, "fps_override": None, "sample_fps": None,
        "units": [{
            "id": "a" * 32, "name": "小鼠 1", "window": None,
            "geometry": {"arena": {"type": "quad",
                                   "points": [[0, 0], [127, 0],
                                              [127, 95], [0, 95]]}},
            "options": {},
            # 不写 detection：显式验证工厂默认 temporal 本身可用。
            "detection": {},
            "manual": None,
        }],
    }])
    return asset, draft


def test_identity_change_keeps_failure_semantics(
        client, store, tmp_path, monkeypatch):
    """取缓存键后源身份变化：明确失败（400 + 原因），不降级为 201。"""

    asset, draft = _seed_real_video_draft(store, tmp_path)
    monkeypatch.setattr(
        preview.media, "read_frame_info",
        lambda *args, **kwargs: FrameInfo(
            image=np.zeros((96, 128, 3), dtype=np.uint8), index=15,
            time_s=1.5, pts=150, time_base=None, clock_source="pts"))

    identities = [
        {"device": 1, "inode": 1, "size": 10, "mtime_ns": 1, "ctime_ns": 1},
        {"device": 1, "inode": 1, "size": 20, "mtime_ns": 2, "ctime_ns": 2},
    ]

    def fake_file_identity(path):
        # 第 1 次调用（取缓存键）用身份 A；之后的核对一律是身份 B。
        index = fake_file_identity.calls
        fake_file_identity.calls += 1
        return identities[min(index, len(identities) - 1)]

    fake_file_identity.calls = 0
    monkeypatch.setattr(preview.media, "file_identity", fake_file_identity)

    response = _preview(client, draft, asset, kind="detection", frame_index=15)
    assert response.status_code == 400, response.get_json()
    body = response.get_json()
    message = body["error"]["message"]
    assert "身份不同" in message, message
    assert "发生变化" in message, message
    warnings = body["error"].get("fields") or []
    assert isinstance(warnings, list)
    # 失败语义：不出现降级成功的任何标记。
    assert "background_unavailable" not in str(body)


def test_shared_build_wait_timeout_keeps_failure_semantics(
        client, store, tmp_path, monkeypatch):
    """同键共享构建等待超时：等待方 400 + "等待背景构建超时"，不降级。

    超时阈值从 900 s 缩短到 0.003 s（轮询间隔 0.05 s，首轮等待即超时），
    仅影响本用例；真实阈值与轮询常量不动。
    """

    asset, draft = _seed_real_video_draft(store, tmp_path)
    monkeypatch.setattr(detection, "BACKGROUND_WAIT_SECONDS", 0.003)

    release = threading.Event()
    started = threading.Event()

    def slow_build(*args, **kwargs):
        started.set()
        release.wait(10)
        return np.full((96, 128), 255, dtype=np.uint8)

    monkeypatch.setattr(detection, "_build_temporal_background", slow_build)

    results = {}

    def owner_request():
        owner_client = client.application.test_client()
        response = _preview(owner_client, draft, asset, kind="detection",
                            frame_index=15)
        results["owner"] = response.status_code

    owner_thread = threading.Thread(target=owner_request)
    owner_thread.start()
    assert started.wait(10), "共享构建未开始"

    response = _preview(client, draft, asset, kind="detection", frame_index=15)
    assert response.status_code == 400, response.get_json()
    message = response.get_json()["error"]["message"]
    assert "等待背景构建超时" in message, message

    release.set()
    owner_thread.join(15)
    assert not owner_thread.is_alive()
    # 等待方超时不得影响持有者：构建完成后仍正常返回并检出候选。
    assert results["owner"] == 201, results


def test_build_cancelled_keeps_cancel_semantics(
        client, store, tmp_path, monkeypatch):
    """共享构建被取消：保持取消语义（400 + "已取消"），不降级为 201。

    注入点与生产一致：取消异常在真实 ``_cached_background`` 的 build()
    内部抛出（生产中由 temporal_frame_plan 的 cancel 检查触发）。
    """

    asset, draft = _seed_real_video_draft(store, tmp_path)

    def cancelled_build(*args, **kwargs):
        raise detection.DetectionError("已取消")

    monkeypatch.setattr(detection, "_build_temporal_background",
                        cancelled_build)

    response = _preview(client, draft, asset, kind="detection", frame_index=15)
    assert response.status_code == 400, response.get_json()
    body = response.get_json()
    assert "已取消" in body["error"]["message"]
    assert "background_unavailable" not in str(body)


def test_delayed_frame_read_error_degrades(client, store, tmp_path,
                                           monkeypatch):
    """取像素阶段（生成器消费时）抛 FrameReadError 也要归入可降级内容失败。

    真实视频 + 真实规划/缓存链路；只有像素读取器返回的生成器在首次
    迭代时抛 FrameReadError（模拟解码器在实际读取时才失败）。
    """

    asset, draft = _seed_real_video_draft(store, tmp_path)
    media_module = preview.media

    def delayed_failure(self, indices, orientation, evidence=None):
        def generator():
            raise media_module.FrameReadError("源帧不可读")
            yield  # pragma: no cover - 只为让它成为生成器

        return generator()

    monkeypatch.setattr(media_module.VideoSource, "frames_at",
                        delayed_failure)

    response = _preview(client, draft, asset, kind="detection", frame_index=15)
    assert response.status_code == 201, response.get_json()
    data = response.get_json()["data"]
    assert data["summary"]["detection_skipped_reason"] == \
        "background_unavailable"
    warnings = data["warnings"]
    assert any("背景视频无法解码（取像素）" in item for item in warnings), \
        warnings
    assert any("源帧不可读" in item for item in warnings), warnings
    assert not any("人工记录路径" in item for item in warnings)


def test_empty_static_background_degrades(client, store, tmp_path,
                                          monkeypatch):
    """0 字节静态背景：明确报告"为空文件"，不再 500。"""

    empty = tmp_path / "empty-bg.png"
    empty.write_bytes(b"")
    asset, draft = _seed_oft(
        store, tmp_path,
        detection={"bg_method": "static", "static_background": str(empty)})
    monkeypatch.setattr(
        preview.media, "read_frame_info",
        lambda *args, **kwargs: FrameInfo(
            image=np.zeros((200, 200, 3), dtype=np.uint8), index=7,
            time_s=0.28, pts=700, time_base=None, clock_source="pts"))

    response = _preview(client, draft, asset, kind="detection")
    assert response.status_code == 201, response.get_json()
    data = response.get_json()["data"]
    assert data["summary"]["detection_skipped_reason"] == \
        "background_unavailable"
    assert any("静态背景为空文件" in item for item in data["warnings"])


def test_exclusive_locked_video_is_access_failure(client, store, tmp_path,
                                                  monkeypatch):
    """Windows 独占文件：属性可读但解码打不开 → 明确失败，不降级。

    真实独占：ctypes CreateFileW 以 dwShareMode=0 持有真实 MP4；只有
    单帧读取用注入帧（否则会在读帧阶段就失败，测不到背景构建的判定）。
    环境无法独占时显式跳过，不用 mock 冒充。
    """

    import ctypes
    from ctypes import wintypes

    asset, draft = _seed_real_video_draft(store, tmp_path)
    source_path = str(draft and asset["path"])
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD,
                                     wintypes.DWORD, wintypes.LPVOID,
                                     wintypes.DWORD, wintypes.DWORD,
                                     wintypes.HANDLE]
    GENERIC_READ, OPEN_EXISTING = 0x80000000, 3
    handle = kernel32.CreateFileW(source_path, GENERIC_READ, 0, None,
                                  OPEN_EXISTING, 0, None)
    invalid = wintypes.HANDLE(-1).value
    if handle in (None, invalid):
        pytest.skip("当前环境无法以独占方式打开文件（无法构造真实拒绝访问）")
    try:
        monkeypatch.setattr(
            preview.media, "read_frame_info",
            lambda *args, **kwargs: FrameInfo(
                image=np.zeros((96, 128, 3), dtype=np.uint8), index=15,
                time_s=1.5, pts=150, time_base=None, clock_source="pts"))
        response = _preview(client, draft, asset, kind="detection",
                            frame_index=15)
    finally:
        kernel32.CloseHandle(handle)
        detection.clear_background_cache()

    assert response.status_code == 422, response.get_json()
    body = response.get_json()
    assert "background_unavailable" not in str(body)
    assert "无法打开视频" in body["error"]["message"] or \
        "不可再访问" in body["error"]["message"], body


def test_formal_job_background_failure_propagates(tmp_path, monkeypatch):
    """正式任务：背景不可用必须失败，不能被预览的降级逻辑吞掉。

    边界注入：``detection.prepare_background`` 抛 BackgroundUnavailableError
    （正式分析的调用点）；真实冻结任务结构走 engine.run_job。
    """

    from behaviourlab import engine
    from behaviourlab.config import ALGORITHM_VERSION, SCHEMA_VERSION
    from behaviourlab.experiments import oft

    source = tmp_path / "job.mp4"
    _write_real_mp4(source)
    stat = source.stat()

    def fail_background(*args, **kwargs):
        raise detection.BackgroundUnavailableError("静态背景为空文件")

    monkeypatch.setattr(detection, "prepare_background", fail_background)

    settings = dict(detection.common_detection_defaults())
    settings.update({"bg_method": "temporal", "temporal_bg_samples": 3})
    resolved = {
        "schema_version": SCHEMA_VERSION,
        "run_id": "r" * 32, "job_id": "j" * 32, "project_id": None,
        "assay": "oft",
        "asset": {"id": "x" * 32, "name": "job.mp4",
                  "path": str(source.resolve()),
                  "metadata": {"width": 128, "height": 96, "fps": 10.0,
                               "frame_count": 30, "duration_s": 3.0},
                  "identity": {"path": str(source.resolve()),
                               "size": int(stat.st_size),
                               "mtime": float(stat.st_mtime),
                               "mtime_ns": int(stat.st_mtime_ns)}},
        "window": {"start_s": 0.0, "end_s": 3.0},
        "orientation": 0, "fps_override": None, "sample_fps": None,
        "options": dict(oft.DEFAULTS), "detection": dict(settings),
        "units": [{
            "id": "a" * 32, "name": "小鼠 1", "animal_id": None,
            "window": {"mode": "relative", "offset_s": 0.0,
                       "duration_s": 3.0},
            "geometry": {"arena": {"type": "quad",
                                   "points": [[0, 0], [127, 0],
                                              [127, 95], [0, 95]]}},
            "options": dict(oft.DEFAULTS), "detection": dict(settings),
            "manual": None,
        }],
        "output": {"annotated_video": False},
        "style": {"color": "#B86245", "colormap": "viridis",
                  "line_width": 2, "dpi": 100},
        "algorithm_version": ALGORITHM_VERSION,
    }

    with pytest.raises(detection.BackgroundUnavailableError):
        engine.run_job(resolved)


# -- 夹具助手 ------------------------------------------------------------


def _seed_oft(store, tmp_path, *, detection=None, geometry=None):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"controlled-not-real-video")
    asset = store.create_asset(
        name="示例.mp4", path=str(source), project_id=None,
        metadata={"width": 200, "height": 200, "fps": 25.0,
                  "frame_count": 250, "duration_s": 10.0},
    )
    draft = store.create_draft("oft", videos=[{
        "asset_id": asset["id"],
        "window": {"start_s": 0, "end_s": None},
        "orientation": 0, "fps_override": 12.5, "sample_fps": 2,
        "units": [{
            "id": "a" * 32, "name": "小鼠 1", "window": None,
            "geometry": geometry if geometry is not None else {
                "arena": {"type": "quad",
                          "points": [[0, 0], [199, 0],
                                     [199, 199], [0, 199]]}},
            "options": {},
            # 不写 bg_method：依赖工厂默认（当前为 temporal）。
            "detection": dict(detection or {}),
            "manual": None,
        }],
    }])
    return asset, draft


def _seed_ymaze_manual(store, tmp_path):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"controlled-not-real-video")
    asset = store.create_asset(
        name="示例.mp4", path=str(source), project_id=None,
        metadata={"width": 200, "height": 200, "fps": 25.0,
                  "frame_count": 500, "duration_s": 20.0},
    )
    draft = store.create_draft("ymaze", videos=[{
        "asset_id": asset["id"],
        "window": {"start_s": 1.0, "end_s": 20.0},
        "orientation": 0, "fps_override": None, "sample_fps": None,
        "units": [{
            "id": "a" * 32, "name": "小鼠 1", "window": None,
            "geometry": {},
            "options": {},
            "detection": {},
            "manual": {"mode": "manual",
                       "entries": [{"time_s": 2.5, "arm": "A"}]},
        }],
    }])
    return asset, draft
