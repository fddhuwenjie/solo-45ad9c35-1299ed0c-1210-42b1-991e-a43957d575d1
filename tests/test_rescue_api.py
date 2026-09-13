"""坠落后救援推演 API 测试：独立修订、来源复用、改入口/换器材理由、
确认版锁定来源与待复核、作业包保留步骤/分量/人工决定。"""
from __future__ import annotations

import copy

import pytest
from fastapi.testclient import TestClient

from app.db import Store
from app.main import app, get_store

from .scenarios import (flex_missing_params_payload, passable_payload,
                        rescue_payload)


@pytest.fixture()
def client(tmp_path):
    store = Store(str(tmp_path / "test.db"))
    app.dependency_overrides[get_store] = lambda: store
    yield TestClient(app)
    app.dependency_overrides.clear()


def _plan(client, payload=None, confirm=True):
    r = client.post("/plans", json={
        "name": "管廊", "note": "", "payload": payload or passable_payload()})
    assert r.status_code == 201, r.text
    plan_id = r.json()["plan_id"]
    if confirm:
        client.post(f"/plans/{plan_id}/revisions/1/confirm")
    return plan_id


def _rescue(client, plan_id, rev_no=1, **kw):
    body = {"name": "救援方案甲", "note": "初稿",
            "payload": rescue_payload(**kw)}
    return client.post(f"/plans/{plan_id}/revisions/{rev_no}/rescue",
                       json=body)


# ---------------------------------------------------------------- 基本流程

def test_create_rescue_reuses_frozen_source_and_steps(client):
    plan_id = _plan(client, confirm=True)
    r = _rescue(client, plan_id)
    assert r.status_code == 201, r.text
    out = r.json()
    assert out["source_rev_no"] == 1
    assert out["status"] == "draft" and out["source_locked"] is False
    res = out["result"]
    assert res["executable"] is True
    assert res["suspension_station_count"] == 21
    s0 = res["simulations"][0]
    codes = [x["code"] for x in s0["steps"]]
    assert codes == ["approach", "secondary_protection", "haul_unload",
                     "detach_original", "transfer"]
    haul = next(x for x in s0["steps"] if x["code"] == "haul_unload")
    assert haul["pull_force_kn"] is not None
    assert haul["anchor_resultant_kn"] > 0
    assert haul["components"]["mechanical_advantage"] > 0


def test_inconclusive_source_never_executable(client):
    plan_id = _plan(client, flex_missing_params_payload())
    r = _rescue(client, plan_id, anchor_zs=[2.6, 2.6])
    assert r.status_code == 201
    res = r.json()["result"]
    assert res["executable"] is False
    assert res["earliest_block"]["code"] == "source_inconclusive"
    assert res["suspension_station_count"] == 0


def test_blocked_plan_steps_and_components_in_package(client):
    plan_id = _plan(client)
    r = _rescue(client, plan_id, rope_main_len=3.0)
    out = r.json()
    assert out["result"]["executable"] is False
    b = out["result"]["earliest_block"]
    assert b["code"] == "rope_too_short"
    assert b["components"]["required_m"] > \
        b["components"]["rope_length_m"]
    # 作业包同样保留受阻步骤与分量
    rid = out["rescue_id"]
    r2 = client.get(f"/rescue/{rid}/revisions/1/job-package")
    assert r2.json()["result"]["earliest_block"]["code"] == "rope_too_short"


def test_manual_choice_requires_reason_api(client):
    plan_id = _plan(client)
    r = _rescue(client, plan_id, primary="R0a")
    assert r.status_code == 422


def test_rescue_requires_confirmed_source(client):
    # 来源修订未冻结：409
    pid_draft = _plan(client, confirm=False)
    r = _rescue(client, pid_draft)
    assert r.status_code == 409

    # 确认后可登记
    client.post(f"/plans/{pid_draft}/revisions/1/confirm")
    r = _rescue(client, pid_draft)
    assert r.status_code == 201, r.text


def test_rescue_404(client):
    r = _rescue(client, "nope")
    assert r.status_code == 404
    assert client.get("/rescue/nope").status_code == 404


# ---------------------------------------------------------------- 修订治理

def test_change_entry_or_equipment_requires_reason_and_new_revision(client):
    plan_id = _plan(client)
    r = _rescue(client, plan_id)
    rid = r.json()["rescue_id"]

    # 草稿覆盖：换绳组（结构性修改）→ 409，必须另存
    changed = rescue_payload(desc_limit=1.5)
    r = client.put(f"/rescue/{rid}/revisions/1",
                   json={"note": "换下降器", "payload": changed})
    assert r.status_code == 409

    # 另存但无理由 → 422
    r = client.post(f"/rescue/{rid}/revisions",
                    json={"note": "换下降器", "payload": changed})
    assert r.status_code == 422

    # 改入口同理（保持上一版的下降器 1.5 kN，仅入口变化）
    changed_entry = rescue_payload(
        desc_limit=1.5,
        entry={"id": "E2", "position": {"x": 10, "y": 0, "z": 0}})
    r = client.post(f"/rescue/{rid}/revisions",
                    json={"note": "改入口", "payload": changed_entry})
    assert r.status_code == 422

    # 附理由另存修订成功，旧版参数保留
    r = client.post(f"/rescue/{rid}/revisions", json={
        "note": "换小限载下降器", "payload": changed,
        "changes": [{"kind": "equipment_change",
                     "reason": "现场仅有 1.5kN 下降器"}]})
    assert r.status_code == 201, r.text
    assert r.json()["rev_no"] == 2
    r1 = client.get(f"/rescue/{rid}/revisions/1").json()
    assert r1["payload"]["rope_teams"][0]["descender_limit_kn"] == 2.5

    # 改入口附 entry_change 理由
    r = client.post(f"/rescue/{rid}/revisions", json={
        "note": "改从另一端进入", "payload": changed_entry,
        "changes": [{"kind": "entry_change", "reason": "原入口被占用"}]})
    assert r.status_code == 201 and r.json()["rev_no"] == 3


def test_non_structural_draft_update_allowed(client):
    plan_id = _plan(client)
    r = _rescue(client, plan_id)
    rid = r.json()["rescue_id"]
    same = rescue_payload()
    same["note_dummy"] = 1
    # 仅改说明（note）：可覆盖草稿
    r = client.put(f"/rescue/{rid}/revisions/1",
                   json={"note": "更新说明", "payload": rescue_payload()})
    assert r.status_code == 200
    assert r.json()["note"] == "更新说明"


def test_confirmed_rescue_locked_and_source_snapshot(client):
    plan_id = _plan(client, confirm=True)
    r = _rescue(client, plan_id)
    rid = r.json()["rescue_id"]

    r = client.post(f"/rescue/{rid}/revisions/1/confirm")
    assert r.status_code == 200
    out = r.json()
    assert out["status"] == "confirmed"
    assert out["source_locked"] is True
    assert out["source"]["status"] == "confirmed"

    # 确认稿不可覆盖
    r = client.put(f"/rescue/{rid}/revisions/1",
                   json={"note": "x", "payload": rescue_payload()})
    assert r.status_code == 409

    # 原路线新增修订（无关参数：人员/装备/锚点/跨段签名内不变 → 不复核）
    changed_plan = copy.deepcopy(passable_payload())
    changed_plan["params"]["cable_max_iter"] = 300   # 不在相关性签名内
    r = client.post(f"/plans/{plan_id}/revisions",
                    json={"note": "无关参数", "payload": changed_plan})
    assert r.status_code == 201
    out2 = client.get(f"/rescue/{rid}/revisions/1").json()
    assert out2["recheck_required"] is False
    # 仍按冻结来源（r1）计算
    assert out2["source_rev_no"] == 1


def test_related_route_change_marks_only_recheck(client):
    plan_id = _plan(client, confirm=True)
    # 方案甲（默认）
    r = _rescue(client, plan_id)
    rid = r.json()["rescue_id"]
    client.post(f"/rescue/{rid}/revisions/1/confirm")

    # 方案乙（不同器材），先不确认：草稿始终不复核
    r2 = _rescue(client, plan_id, max_time=25.0)
    rid2 = r2.json()["rescue_id"]

    # 原路线相关变化：锚点额定载荷改变（签名包含 used anchors）
    changed = copy.deepcopy(passable_payload())
    for a in changed["route"]["anchors"]:
        a["rated_load_kn"] = 9.0
    r = client.post(f"/plans/{plan_id}/revisions", json={
        "note": "锚点复测降载", "payload": changed})
    assert r.status_code == 201

    out = client.get(f"/rescue/{rid}/revisions/1").json()
    assert out["recheck_required"] is True
    assert "r2" in out["recheck_reason"]
    # 计算仍锁定 r1 冻结快照，结果不随新路线变（executable 保持）
    assert out["result"]["executable"] is True
    assert out["source_rev_no"] == 1

    # 未确认的方案乙不标记复核（草稿本就跟随最新来源）
    out2 = client.get(f"/rescue/{rid2}/revisions/1").json()
    assert out2["recheck_required"] is False


def test_stretcher_only_obstacle_marks_confirmed_recheck(client):
    """回归：相关路线修订新增只影响担架走廊的障碍物、最新结果出现
    casualty_path_blocked 时，已确认方案必须待复核，不再提供过期的
    可执行结论（确认结果仍锁定 r1 冻结快照）。"""
    plan_id = _plan(client, confirm=True)
    # 向侧向落点转运，担架走廊沿 +y 展开
    landing = {"x": 0, "y": 3, "z": 0}
    r = _rescue(client, plan_id, landing=landing)
    rid = r.json()["rescue_id"]
    assert r.json()["result"]["executable"] is True
    client.post(f"/rescue/{rid}/revisions/1/confirm")

    # 原路线新增修订：障碍物只横在担架走廊（y∈[1,3]），
    # 不影响 y≈0 的人员坠落/摆坠，来源路线仍可通行
    changed = copy.deepcopy(passable_payload())
    changed["route"]["obstacles"] = [
        {"kind": "box", "id": "stretcher_gate",
         "min": {"x": -1.0, "y": 1.0, "z": -1.2},
         "max": {"x": 1.0, "y": 3.0, "z": -0.5}}]
    r = client.post(f"/plans/{plan_id}/revisions",
                    json={"note": "走廊增设管线", "payload": changed})
    assert r.status_code == 201, r.text
    assert r.json()["analysis"]["passable"] is True

    out = client.get(f"/rescue/{rid}/revisions/1").json()
    assert out["recheck_required"] is True
    assert "r2" in out["recheck_reason"]
    # 冻结快照结果仍按 r1（无障碍）计算：确认结论不被静默改写
    assert out["source_rev_no"] == 1
    assert out["result"]["executable"] is True

    # 若直接按最新路线（确认 r2 后）登记方案，最新结果确为担架路径受阻
    client.post(f"/plans/{plan_id}/revisions/2/confirm")
    rid2 = client.post(
        f"/plans/{plan_id}/revisions/2/rescue",
        json={"name": "新", "note": "",
              "payload": rescue_payload(landing=landing)})
    assert rid2.status_code == 201
    assert rid2.json()["result"]["earliest_block"]["code"] \
        == "casualty_path_blocked"


def test_rescue_plan_listing(client):
    plan_id = _plan(client)
    _rescue(client, plan_id)
    r = client.get("/rescue")
    assert r.status_code == 200
    assert len(r.json()) == 1
    rid = r.json()[0]["rescue_id"]
    r = client.get(f"/rescue/{rid}")
    assert r.json()["revisions"][0]["rev_no"] == 1


def test_manual_decisions_persisted_in_job_package(client):
    plan_id = _plan(client)
    # 人工指定绳组（附理由）另存确认
    body = {"name": "m", "note": "",
            "payload": rescue_payload(rope_team="RT1",
                                      manual_reason="RT1 为专用提拉组")}
    r = client.post(f"/plans/{plan_id}/revisions/1/rescue", json=body)
    rid = r.json()["rescue_id"]
    client.post(f"/rescue/{rid}/revisions/1/confirm")
    out = client.get(f"/rescue/{rid}/revisions/1/job-package").json()
    dec = out["result"]["manual_decisions"]
    assert any(d["field"] == "rope_team_id" and d["value"] == "RT1"
               for d in dec)
    # 每步分量齐全
    for sim in out["result"]["simulations"][:3]:
        for st in sim["steps"]:
            assert "cumulative_minutes" in st
