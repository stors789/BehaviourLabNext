"""自检 3：合并规则、结果独立性、Store 顶层覆盖与草稿快照。"""

from __future__ import annotations

import copy

import pytest

from behaviourlab.engine import resolve_option_layers, snapshot_draft
from behaviourlab.schema import ValidationError
from behaviourlab.store import Store


def test_option_resolution_follows_layer_rules():
    factory = {"a": 1, "nested": {"x": 1, "y": 2}, "list": [1, 2]}
    saved = {"a": 2}
    draft = {"nested": {"x": 3}}
    video = {"list": [9]}
    unit = {"nested": {"y": 4}, "a": None}

    resolved = resolve_option_layers(
        factory, saved=saved, draft={"options": draft},
        video={"options": video}, unit={"options": unit},
    )

    assert resolved == {"a": None, "nested": {"x": 3, "y": 4}, "list": [9]}


def test_resolution_does_not_mutate_inputs():
    factory = {"nested": {"x": 1, "y": 2}, "list": [1, 2]}
    saved = {"a": 2}
    draft = {"options": {"nested": {"x": 3}}}
    video = {"options": {"list": [9]}}
    unit = {"options": {"nested": {"y": 4}, "a": None}}
    snapshot = copy.deepcopy((factory, saved, draft, video, unit))

    resolved = resolve_option_layers(
        factory, saved=saved, draft=draft, video=video, unit=unit
    )
    resolved["nested"]["x"] = 99
    resolved["list"].append(7)

    assert (factory, saved, draft, video, unit) == snapshot
    assert factory["nested"]["x"] == 1
    assert draft["options"]["nested"]["x"] == 3
    assert video["options"]["list"] == [9]


def test_store_update_replaces_top_level_field(store: Store):
    draft = store.create_draft(
        "oft",
        options={"a": 1, "nested": {"x": 1}, "list": [1, 2]},
        detection={"threshold": 0.5},
    )

    updated = store.update_draft(draft["id"], {"options": {"a": 7}})

    assert updated["options"] == {"a": 7}
    assert updated["detection"] == {"threshold": 0.5}
    assert updated["id"] == draft["id"]
    assert updated["created_at"] == draft["created_at"]


def test_store_update_rejects_unknown_or_identity_fields(store: Store):
    draft = store.create_draft("oft")
    for payload in ({"id": "x"}, {"created_at": "x"}, {"name": "x"}):
        with pytest.raises(ValidationError):
            store.update_draft(draft["id"], payload)


def test_draft_snapshot_is_independent_of_later_settings(store: Store):
    store.set_setting("settings.defaults", {"tst": {"mode": "lis"}})
    snapshot = snapshot_draft(
        "tst", defaults=store.get_setting("settings.defaults")["tst"]
    )
    assert snapshot["options"]["mode"] == "lis"

    store.set_setting("settings.defaults", {"tst": {"mode": "classic"}})
    assert snapshot["options"]["mode"] == "lis"
    assert snapshot["output"] == {"annotated_video": True}
    assert snapshot["style"]["color"] == "#B86245"
