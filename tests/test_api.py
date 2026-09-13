"""API 测试：方案/修订的保存、确认稿不可覆盖、按版参数重算。"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.db import Store
from app.main import app, get_store

from .scenarios import (clearance_payload, manual_order_payload,
                        passable_payload, single_overload_payload)


@pytest.fixture()
def client(tmp_path):
    store = Store(str(tmp_path / "test.db"))
    app.dependency_overrides[get_store] = lambda: store
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_plan_lifecycle_and_revision_isolation(client):
    # 创建方案（第 1 版可通行）
    r = client.post("/plans", json={
        "name": "管廊一号线", "note": "初稿",
        "payload": passable_payload()})
    assert r.status_code == 201, r.text
    rev1 = r.json()
    assert rev1["rev_no"] == 1 and rev1["status"] == "draft"
    assert rev1["analysis"]["passable"] is True
    plan_id = rev1["plan_id"]

    # 换装备/调参数另存修订（第 2 版：下层管线侵入 → 不可通行）
    r = client.post(f"/plans/{plan_id}/revisions", json={
        "note": "补充下层管线", "payload": clearance_payload()})
    assert r.status_code == 201
    rev2 = r.json()
    assert rev2["rev_no"] == 2
    assert rev2["analysis"]["passable"] is False
    assert rev2["analysis"]["first_failure"]["check"] == "clearance"
    # 失败响应包含位置、动作、计算分量
    ff = rev2["analysis"]["first_failure"]
    assert ff["action"] == "traverse"
    assert ff["position"]["x"] is not None
    assert ff["components"]["margin_m"] < 0

    # 回看第 1 版：仍按该版参数重算为可通行（修订互不影响）
    r = client.get(f"/plans/{plan_id}/revisions/1")
    assert r.json()["analysis"]["passable"] is True
    # 动作序列与剖面标注随版本重算
    assert len(r.json()["analysis"]["sequence"]) > 0
    assert len(r.json()["analysis"]["profile"]) == \
        r.json()["analysis"]["station_count"]


def test_draft_update_and_confirm_freezes(client):
    r = client.post("/plans", json={
        "name": "p", "note": "", "payload": clearance_payload()})
    plan_id = r.json()["plan_id"]
    assert r.json()["analysis"]["passable"] is False

    # 草稿可覆盖：移除障碍后变为可通行
    r = client.put(f"/plans/{plan_id}/revisions/1",
                   json={"note": "改", "payload": passable_payload()})
    assert r.status_code == 200
    assert r.json()["analysis"]["passable"] is True

    # 确认定稿
    r = client.post(f"/plans/{plan_id}/revisions/1/confirm")
    assert r.status_code == 200 and r.json()["status"] == "confirmed"

    # 确认稿不可覆盖
    r = client.put(f"/plans/{plan_id}/revisions/1",
                   json={"note": "改", "payload": clearance_payload()})
    assert r.status_code == 409

    # 定稿后回看仍按该版参数重算
    r = client.get(f"/plans/{plan_id}/revisions/1/analysis")
    assert r.json()["passable"] is True


def test_plan_listing_and_404(client):
    r = client.post("/plans", json={
        "name": "p1", "note": "", "payload": passable_payload()})
    plan_id = r.json()["plan_id"]

    r = client.get("/plans")
    assert any(p["plan_id"] == plan_id for p in r.json())

    r = client.get(f"/plans/{plan_id}")
    assert r.json()["revisions"][0]["rev_no"] == 1

    assert client.get("/plans/nope").status_code == 404
    assert client.get(f"/plans/{plan_id}/revisions/99").status_code == 404
    assert client.post("/plans/nope/revisions", json={
        "note": "", "payload": passable_payload()}).status_code == 404


def test_invalid_payload_rejected(client):
    bad = passable_payload()
    bad["persons"][0]["equipment_id"] = "ghost"
    r = client.post("/plans", json={"name": "x", "note": "", "payload": bad})
    assert r.status_code == 422

    bad2 = passable_payload()
    bad2["hook_order"] = [
        {"person_id": "p1", "hook": "A", "action": "attach",
         "anchor": "ghost", "station_index": 0}]
    r = client.post("/plans", json={"name": "x", "note": "", "payload": bad2})
    assert r.status_code == 422


def test_single_person_overload_api(client):
    r = client.post("/plans", json={
        "name": "o", "note": "", "payload": single_overload_payload()})
    assert r.status_code == 201
    a = r.json()["analysis"]
    assert a["passable"] is False
    ff = a["first_failure"]
    assert ff["check"] == "anchor_overload"
    assert ff["action"] == "traverse"
    assert ff["components"]["user_count"] == 1
    assert ff["components"]["combined_force_kn"] > \
        ff["components"]["rated_load_kn"]


def test_manual_order_revisions_recomputed(client):
    # rev1：无人工次序，自动算法在站点 2 首次换钩
    r = client.post("/plans", json={
        "name": "m", "note": "", "payload": passable_payload()})
    assert r.status_code == 201
    plan_id = r.json()["plan_id"]
    auto_sw = next(e for e in r.json()["analysis"]["sequence"]
                   if e["action"] == "switch")
    assert auto_sw["station_index"] == 2

    # rev2：人工调序，首次换钩提前到站点 1，另存修订
    r = client.post(f"/plans/{plan_id}/revisions", json={
        "note": "人工调序", "payload": manual_order_payload()})
    assert r.status_code == 201
    assert r.json()["analysis"]["passable"] is True
    seq2 = r.json()["analysis"]["sequence"]
    sw2 = next(e for e in seq2 if e["action"] == "switch")
    assert sw2["station_index"] == 1 and sw2["to_anchor"] == "A1"

    # 回看 rev1：仍按该版（自动次序）重算
    r = client.get(f"/plans/{plan_id}/revisions/1")
    sw1 = next(e for e in r.json()["analysis"]["sequence"]
               if e["action"] == "switch")
    assert sw1["station_index"] == 2

    # 确认 rev2 后不可覆盖
    r = client.post(f"/plans/{plan_id}/revisions/2/confirm")
    assert r.json()["status"] == "confirmed"
    r = client.put(f"/plans/{plan_id}/revisions/2",
                   json={"note": "x", "payload": passable_payload()})
    assert r.status_code == 409

    # 定稿后回看仍按人工次序重算，剖面标注齐全
    r = client.get(f"/plans/{plan_id}/revisions/2/analysis")
    a = r.json()
    sw = next(e for e in a["sequence"] if e["action"] == "switch")
    assert sw["station_index"] == 1
    assert len(a["profile"]) == a["station_count"]
