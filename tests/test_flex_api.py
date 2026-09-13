"""柔性跨段修订治理 API 测试：跨段重大修改必须另存修订并保留旧参数，
确认版只读，旧点锚请求保持兼容。
"""
from __future__ import annotations

import copy

import pytest
from fastapi.testclient import TestClient

from app.db import Store
from app.main import app, get_store

from .scenarios import (flex_passable_payload, mixed_anchor_span_payload,
                        passable_payload)


@pytest.fixture()
def client(tmp_path):
    store = Store(str(tmp_path / "test.db"))
    app.dependency_overrides[get_store] = lambda: store
    yield TestClient(app)
    app.dependency_overrides.clear()


def _span_change(p, *, pretension=6.5, max_users=3):
    p = copy.deepcopy(p)
    p["route"]["spans"] = [
        {**s, "pretension_kn": pretension, "max_users": max_users}
        for s in p["route"]["spans"]]
    return p


def test_span_change_must_use_new_revision_with_reason(client):
    r = client.post("/plans", json={
        "name": "跨段一", "note": "", "payload": flex_passable_payload()})
    assert r.status_code == 201, r.text
    plan_id = r.json()["plan_id"]
    changed = _span_change(flex_passable_payload())

    # 直接覆盖草稿：即便附理由也 409，旧参数必须保留
    r = client.put(f"/plans/{plan_id}/revisions/1",
                   json={"note": "改跨", "payload": changed,
                         "changes": [{"kind": "span_change",
                                      "reason": "现场复测预张力不符"}]})
    assert r.status_code == 409

    # 旧参数未被覆盖：回看 r1 仍为原预张力 5.0、容许 2 人
    r = client.get(f"/plans/{plan_id}/revisions/1")
    sp = r.json()["payload"]["route"]["spans"][0]
    assert sp["pretension_kn"] == 5.0 and sp["max_users"] == 2

    # 另存修订但不给理由：422
    r = client.post(f"/plans/{plan_id}/revisions",
                    json={"note": "改跨", "payload": changed})
    assert r.status_code == 422

    # 另存修订并附 span_change 理由：成功，生成新修订
    r = client.post(f"/plans/{plan_id}/revisions",
                    json={"note": "复测后调预张力", "payload": changed,
                          "changes": [{"kind": "span_change",
                                       "reason": "现场复测预张力为 6.5 kN，"
                                                 "容许人数增至 3 人"}]})
    assert r.status_code == 201, r.text
    rev2 = r.json()
    assert rev2["rev_no"] == 2
    assert rev2["changes"][0]["kind"] == "span_change"
    assert rev2["payload"]["route"]["spans"][0]["pretension_kn"] == 6.5

    # 旧版按冻结参数复算
    r = client.get(f"/plans/{plan_id}/revisions/1")
    assert r.json()["payload"]["route"]["spans"][0]["pretension_kn"] == 5.0


def test_shuttle_change_requires_reason_and_new_revision(client):
    r = client.post("/plans", json={
        "name": "s", "payload": flex_passable_payload()})
    plan_id = r.json()["plan_id"]

    changed = copy.deepcopy(flex_passable_payload())
    changed["route"]["shuttles"][0]["can_pass"] = False

    # 覆盖被拒
    assert client.put(f"/plans/{plan_id}/revisions/1",
                      json={"note": "x", "payload": changed,
                            "changes": [{"kind": "shuttle_change",
                                         "reason": "换断开式滑梭"}]}
                      ).status_code == 409
    # 旧版滑梭 can_pass 仍为 True
    r = client.get(f"/plans/{plan_id}/revisions/1")
    assert r.json()["payload"]["route"]["shuttles"][0]["can_pass"] is True
    # 另存但理由类型错误（只给 span_change）→ 422
    assert client.post(f"/plans/{plan_id}/revisions",
                       json={"note": "x", "payload": changed,
                             "changes": [{"kind": "span_change",
                                          "reason": "改滑梭"}]}
                       ).status_code == 422
    # 正确理由 → 201
    r = client.post(f"/plans/{plan_id}/revisions",
                    json={"note": "换滑梭", "payload": changed,
                          "changes": [{"kind": "shuttle_change",
                                       "reason": "更换为不可过支座滑梭"}]})
    assert r.status_code == 201
    assert r.json()["rev_no"] == 2


def test_non_structural_draft_update_still_allowed(client):
    """点锚/装备/人员等非结构参数仍可覆盖草稿（旧行为兼容）。"""
    r = client.post("/plans", json={
        "name": "p", "payload": flex_passable_payload()})
    plan_id = r.json()["plan_id"]
    updated = flex_passable_payload()
    updated["persons"][0]["weight_kg"] = 90.0
    r = client.put(f"/plans/{plan_id}/revisions/1",
                   json={"note": "改体重", "payload": updated})
    assert r.status_code == 200
    assert r.json()["payload"]["persons"][0]["weight_kg"] == 90.0


def test_confirmed_revision_remains_read_only(client):
    r = client.post("/plans", json={
        "name": "p", "payload": flex_passable_payload()})
    plan_id = r.json()["plan_id"]
    assert client.post(
        f"/plans/{plan_id}/revisions/1/confirm").json()["status"] == "confirmed"
    # 即便仅改非结构参数，确认稿仍 409
    changed = flex_passable_payload()
    changed["persons"][0]["weight_kg"] = 99.0
    assert client.put(f"/plans/{plan_id}/revisions/1",
                      json={"note": "x", "payload": changed}).status_code == 409
    # 确认稿回看按冻结参数复算
    r = client.get(f"/plans/{plan_id}/revisions/1/analysis")
    assert r.json()["passable"] is True


def test_conservative_bounds_requires_reason(client):
    p = flex_passable_payload()
    p["conservative_bounds"] = {"enabled": True, "reason": ""}
    r = client.post("/plans", json={"name": "c", "payload": p})
    assert r.status_code == 422
    p["conservative_bounds"]["reason"] = "现场无法确认钢索 EA，授权保守边界"
    r = client.post("/plans", json={
        "name": "c", "payload": p,
        "changes": [{"kind": "conservative_bounds",
                     "reason": "现场无法确认钢索 EA，授权保守边界"}]})
    assert r.status_code == 201, r.text


def test_legacy_anchor_only_request_compatible(client):
    """原有点锚请求（无 spans/supports/shuttles 字段）保持兼容。"""
    r = client.post("/plans", json={
        "name": "旧点锚方案", "payload": passable_payload()})
    assert r.status_code == 201, r.text
    a = r.json()["analysis"]
    assert a["conclusive"] is True and a["passable"] is True
    assert a["cable_results"] == []
    assert a["profile"][0]["controlling_anchor"] is not None


def test_mixed_route_results_attached_shuttles(client):
    r = client.post("/plans", json={
        "name": "混用", "payload": mixed_anchor_span_payload()})
    assert r.status_code == 201, r.text
    a = r.json()["analysis"]
    assert a["conclusive"] and a["passable"]
    seq = a["sequence"]
    # 点锚段先行，进入跨段后换到具体滑梭（动作序列引用具体滑梭 id）
    shuttle_events = [e for e in seq if e.get("to_shuttle")]
    assert shuttle_events
    assert all(e["to_shuttle"] in {"T1", "T2"} for e in shuttle_events)
