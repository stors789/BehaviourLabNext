"""人工 Y 窗口约束与源身份校验。"""

from __future__ import annotations

import os

from behaviourlab import engine
from behaviourlab.schema import new_id


def test_ymaze_manual_only_allows_global_or_exact_window(client):
    response = client.post(
        "/api/drafts",
        json={
            "assay": "ymaze",
            "videos": [
                {
                    "asset_id": new_id(),
                    "units": [
                        {
                            "name": "小鼠 1",
                            "window": {"mode": "relative", "offset_s": 0,
                                       "duration_s": 60},
                            "manual": {"entries": [{"time_s": 1.0, "arm": "A"}]},
                        }
                    ],
                }
            ],
        },
    )
    assert response.status_code == 400
    issues = response.get_json()["error"]["fields"]
    assert any(issue["field"].endswith("window.mode") for issue in issues)
    assert all(issue.get("unit_id") for issue in issues if "unit_id" in issue)


def test_identity_mismatch_is_reported(tmp_path):
    video = tmp_path / "clip.bin"
    video.write_bytes(b"abcd")
    stat = os.stat(video)
    asset = {
        "path": str(video),
        "identity": {
            "path": str(video.resolve()),
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "mtime_ns": stat.st_mtime_ns,
        },
    }
    assert engine.verify_identity(asset) == []

    asset["identity"]["size"] = stat.st_size + 1
    issues = engine.verify_identity(asset)
    assert issues and issues[0]["field"] == "asset.identity"
