"""Y 型双腿系绳 API / 修订治理 / SQLite 回归测试。

覆盖：双腿响应经 HTTP 完整返回；腿长/曲线/钩腿绑定变化必须另存修订，
草稿原地覆盖返回 409，确认版只读；旧请求与既有修订结果保持不变。
"""
from __future__ import annotations

import copy

import pytest
from fastapi.testclient import TestClient

from app.db import Store
from app.main import app, get_store

from .scenarios import (flex_passable_payload, passable_payload,
                        twin_angle_payload, twin_buffer_capacity_payload,
                        twin_flex_payload,
                        twin_mixed_anchor_shuttle_payload,
                        twin_passable_payload)


@pytest.fixture()
def client(tmp_path):
    store = Store(str(tmp_path / "test.db"))
    app.dependency_overrides[get_store] = lambda: store
    yield TestClient(app)
    app.dependency_overrides.clear()


def _create(client, payload, name="双腿方案"):
    return client.post("/plans", json={"name": name, "payload": payload})


# ---------------------------------------------------------------- 响应

def test_twin_response_over_http(client):
    r = _create(client, twin_passable_payload())
    assert r.status_code == 201, r.text
    a = r.json()["analysis"]
    assert a["conclusive"] and a["passable"]
    assert a["twin_leg_results"]
    t = a["twin_leg_results"][0]
    assert t["engagement_order"] == ["LA", "LB"]
    assert {"leg_id", "tension_kn", "vertical_component_kn",
            "horizontal_component_kn", "connector_side_load_kn",
            "target_id", "taut"} <= set(t["legs"][0])
    assert "buffer_deployment_m" in t
    assert "buffer_force_kn" in t       # 人体峰值制动力
    assert t["adopted_params"]["leg_LA_hook"] == "A"
    # 序列事件携带钩腿绑定
    attach = [e for e in a["sequence"] if e["action"] == "attach"]
    assert {e["leg"] for e in attach} == {"LA", "LB"}


def test_twin_failure_blocks_passable(client):
    r = _create(client, twin_angle_payload())
    assert r.status_code == 201
    a = r.json()["analysis"]
    assert a["passable"] is False
    assert a["first_failure"]["check"] == "included_angle"
    assert a["first_failure"]["station_index"] == 0


def test_twin_flexible_span_response(client):
    r = _create(client, twin_flex_payload())
    assert r.status_code == 201, r.text
    a = r.json()["analysis"]
    assert a["conclusive"], [o["code"] for o in a["open_items"]]
    assert a["passable"], [f["check"] for f in a["failures"]]
    assert a["twin_leg_results"]
    # 滑梭挂点含动态下移（anchor_position.z 低于静态 2.6）
    mid = next(t for t in a["twin_leg_results"]
               if t["station_index"] == len(a["twin_leg_results"]) // 2)
    assert all(l["target_kind"] == "shuttle" for l in mid["legs"])
    assert any(l["anchor_position"]["z"] < 2.6 for l in mid["legs"])


# ---------------------------------------------------------------- 修订治理

def _change_leg_length(p):
    p = copy.deepcopy(p)
    for leg in p["equipment"][0]["twin_leg"]["legs"]:
        leg["leg_length_m"] = 2.2
    return p


def _change_curve(p):
    p = copy.deepcopy(p)
    p["equipment"][0]["twin_leg"]["buffer_curve"] = [
        {"travel_m": 0.0, "force_kn": 3.5},
        {"travel_m": 1.2, "force_kn": 3.5}]
    return p


def _change_binding(p):
    p = copy.deepcopy(p)
    # 钩腿绑定交换（LA->B, LB->A）
    legs = p["equipment"][0]["twin_leg"]["legs"]
    legs[0]["hook"], legs[1]["hook"] = legs[1]["hook"], legs[0]["hook"]
    return p


@pytest.mark.parametrize("mutator,reason", [
    (_change_leg_length, "lanyard_change"),
    (_change_curve, "lanyard_change"),
    (_change_binding, "lanyard_change"),
])
def test_lanyard_change_requires_new_revision(client, mutator, reason):
    r = _create(client, twin_passable_payload())
    plan_id = r.json()["plan_id"]
    changed = mutator(twin_passable_payload())

    # 草稿原地覆盖：409（即便附理由）
    r = client.put(f"/plans/{plan_id}/revisions/1",
                   json={"note": "改系绳", "payload": changed,
                         "changes": [{"kind": reason, "reason": "x"}]})
    assert r.status_code == 409

    # 另存修订但无理由：422
    r = client.post(f"/plans/{plan_id}/revisions",
                    json={"note": "改系绳", "payload": changed})
    assert r.status_code == 422

    # 附 lanyard_change 理由：201
    r = client.post(f"/plans/{plan_id}/revisions",
                    json={"note": "改系绳", "payload": changed,
                          "changes": [{"kind": reason,
                                       "reason": "更换 Y 型系绳参数"}]})
    assert r.status_code == 201, r.text
    assert r.json()["rev_no"] == 2

    # 旧版冻结参数不变
    r = client.get(f"/plans/{plan_id}/revisions/1")
    legs = r.json()["payload"]["equipment"][0]["twin_leg"]["legs"]
    assert legs[0]["leg_length_m"] == 2.5


def test_confirmed_twin_revision_read_only(client):
    r = _create(client, twin_passable_payload())
    plan_id = r.json()["plan_id"]
    assert client.post(f"/plans/{plan_id}/revisions/1/confirm"
                       ).json()["status"] == "confirmed"
    changed = _change_leg_length(twin_passable_payload())
    r = client.put(f"/plans/{plan_id}/revisions/1",
                   json={"note": "x", "payload": changed})
    assert r.status_code == 409


# ---------------------------------------------------------------- 误判回归

def test_mixed_anchor_shuttle_anchor_load_once_over_http(client):
    """HTTP/SQLite 路径：固定锚腿独承 3.0 kN 时 4.5 kN 锚点不误判过载。"""
    r = _create(client, twin_mixed_anchor_shuttle_payload(4.5),
                name="混挂防重复计数")
    assert r.status_code == 201, r.text
    a = r.json()["analysis"]
    assert a["conclusive"] and a["passable"], \
        [(f["check"], f["station_index"]) for f in a["failures"]]
    assert not any(f["check"] == "anchor_overload" for f in a["failures"])
    # 存在固定锚腿独承 3.0 kN、滑梭腿松弛的站
    solo = [t for t in a["twin_leg_results"]
            if any(not l["taut"] for l in t["legs"])
            and any(l["target_id"].startswith("AX") and l["taut"]
                    for l in t["legs"])]
    assert solo
    fixed = [l for l in solo[0]["legs"]
             if l["target_id"].startswith("AX") and l["taut"]]
    assert abs(fixed[0]["tension_kn"] - 3.0) < 1e-6


def test_buffer_capacity_compares_only_curve_energy_over_http(client):
    """HTTP 路径：曲线吸收 < 容量 < 曲线+腿部弹性能时不误判 buffer_energy。"""
    r = _create(client, twin_buffer_capacity_payload(),
                name="缓冲容量只比曲线")
    assert r.status_code == 201, r.text
    a = r.json()["analysis"]
    t = a["twin_leg_results"][0]
    assert t["buffer_energy_absorbed_j"] < t["energy_capacity_j"]
    assert (t["buffer_energy_absorbed_j"] + t["elastic_energy_j"]) \
        > t["energy_capacity_j"]
    assert not any(f["check"] == "buffer_energy" for f in a["failures"])
    assert a["conclusive"] and a["passable"]


# ---------------------------------------------------------------- 兼容 / SQLite

def test_legacy_request_compatible_and_persisted(client):
    r = _create(client, passable_payload(), name="旧方案")
    assert r.status_code == 201
    a = r.json()["analysis"]
    assert a["passable"] and a["twin_leg_results"] == []
    # 重新 GET（从 SQLite payload 反序列化重算）结果一致
    plan_id = r.json()["plan_id"]
    r2 = client.get(f"/plans/{plan_id}/revisions/1/analysis")
    assert r2.json()["passable"] is True
    assert r2.json()["twin_leg_results"] == []


def test_legacy_revision_result_unchanged_with_twin_feature(client):
    """既有等长模型修订在加入双腿特性后，关键结果分量保持不变。"""
    r = _create(client, flex_passable_payload(), name="旧跨段")
    a1 = r.json()["analysis"]
    plan_id = r.json()["plan_id"]
    # 回看重算
    a2 = client.get(f"/plans/{plan_id}/revisions/1/analysis").json()
    assert a2["conclusive"] == a1["conclusive"]
    assert a2["passable"] == a1["passable"]
    assert len(a2["cable_results"]) == len(a1["cable_results"])
    assert a2["twin_leg_results"] == []
    # 旧 CableResult 分量不被双腿逻辑改写
    assert a2["cable_results"][0]["loads_kn"] == \
        a1["cable_results"][0]["loads_kn"]


def test_twin_result_recomputed_from_sqlite_each_get(client):
    r = _create(client, twin_passable_payload())
    plan_id = r.json()["plan_id"]
    a1 = r.json()["analysis"]
    a2 = client.get(f"/plans/{plan_id}/revisions/1").json()["analysis"]
    assert len(a2["twin_leg_results"]) == len(a1["twin_leg_results"])
    t1, t2 = a1["twin_leg_results"][0], a2["twin_leg_results"][0]
    assert t2["engagement_order"] == t1["engagement_order"]
    assert t2["buffer_force_kn"] == t1["buffer_force_kn"]
    assert t2["legs"][0]["tension_kn"] == t1["legs"][0]["tension_kn"]


def test_curve_validation_rejects_short_curve(client):
    p = twin_passable_payload()
    # 曲线未覆盖最大行程 → 422
    p["equipment"][0]["twin_leg"]["buffer_curve"] = [
        {"travel_m": 0.0, "force_kn": 3.0},
        {"travel_m": 0.5, "force_kn": 3.0}]
    r = _create(client, p)
    assert r.status_code == 422
