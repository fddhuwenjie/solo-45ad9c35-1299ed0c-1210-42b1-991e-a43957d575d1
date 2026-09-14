"""Y 型双腿系绳载荷分配引擎测试。

覆盖：接载次序、缓冲展开量、人体峰值制动力、逐腿张力与锚点合力、
能量不重复计入、夹角/侧载/能量/腿长越限阻断、求解分量齐全；
旧等长模型结果保持兼容。
"""
from __future__ import annotations

import copy
import math

import pydantic
import pytest

from app.engine import analyze, twinleg
from app.models import BufferCurvePoint, PlanPayload

from .scenarios import (passable_payload, twin_angle_payload,
                        twin_energy_payload, twin_manual_payload,
                        twin_passable_payload, twin_side_load_payload,
                        twin_unequal_legs_payload)


def run(payload):
    return analyze(PlanPayload.model_validate(payload))


# ---------------------------------------------------------------- 响应分量

def test_twin_passable_basic_components():
    r = run(twin_passable_payload())
    assert r.conclusive and r.passable
    assert r.twin_leg_results, "应逐站返回双腿结果"
    t = r.twin_leg_results[0]
    # 接载次序为两条实体腿
    assert t.engagement_order == ["LA", "LB"]
    # 对称几何：两腿张力相等、竖直分量之和等于人体峰值制动力（一个缓冲）
    legs = {l.leg_id: l for l in t.legs}
    assert legs["LA"].target_id == "a0"
    assert legs["LB"].target_id == "b0"
    assert abs(legs["LA"].tension_kn - legs["LB"].tension_kn) < 1e-6
    vsum = legs["LA"].vertical_component_kn + legs["LB"].vertical_component_kn
    assert abs(vsum - t.buffer_force_kn) < 1e-3
    # 实际夹角约 74°
    assert 60.0 < t.included_angle_deg < 90.0
    # 采用参数随响应返回
    assert t.adopted_params["leg_LA_length_m"] == 2.5
    assert t.adopted_params["curve_0_force_kn"] == 3.0
    # 缓冲能量 + 腿弹性能 = 需吸收能量（能量守恒），且缓冲只计一次
    assert abs(t.buffer_energy_absorbed_j + t.elastic_energy_j
               - t.energy_demand_j) < 2.0


def test_twin_sequence_binds_hook_to_leg():
    r = run(twin_passable_payload())
    attach = [e for e in r.sequence if e.action == "attach"]
    by_hook = {e.hook: e for e in attach}
    assert by_hook["A"].leg == "LA" and by_hook["B"].leg == "LB"
    assert by_hook["A"].to_anchor.startswith("a")
    assert by_hook["B"].to_anchor.startswith("b")
    # 两腿不得挂同一实体
    assert by_hook["A"].to_anchor != by_hook["B"].to_anchor
    for e in r.sequence:
        assert e.leg in ("LA", "LB")


def test_twin_anchor_resultant_is_vector_sum():
    """锚点合力 = 该锚所连承拉腿张力；不同锚各自承受对应绳腿。"""
    r = run(twin_passable_payload())
    t = r.twin_leg_results[0]
    for l in t.legs:
        mag = math.hypot(l.vertical_component_kn, l.horizontal_component_kn)
        assert abs(mag - l.tension_kn) < 1e-3


# ---------------------------------------------------------------- 越限阻断

def test_included_angle_exceeded_blocks():
    r = run(twin_angle_payload())
    assert not r.passable
    assert r.first_failure.check == "included_angle"
    assert r.first_failure.station_index == 0
    assert r.first_failure.components["included_angle_deg"] > \
        r.first_failure.components["max_included_angle_deg"]


def test_buffer_energy_insufficient_blocks():
    r = run(twin_energy_payload())
    assert not r.passable and r.conclusive
    checks = {f.check for f in r.failures}
    assert "buffer_energy" in checks
    ff = next(f for f in r.failures if f.check == "buffer_energy")
    assert ff.components["energy_shortfall_j"] > 0
    # 完全展开
    assert ff.components["buffer_deployment_m"] >= 0.2 - 1e-9


def test_energy_capacity_field_independently_enforced():
    # 曲线可吸收能量足够，但声明能量容量很小：仍判 buffer_energy
    p = twin_energy_payload()
    eq = p["equipment"][0]
    eq["twin_leg"]["buffer_curve"] = [
        {"travel_m": 0.0, "force_kn": 3.0},
        {"travel_m": 1.2, "force_kn": 3.0}]
    eq["twin_leg"]["max_travel_m"] = 1.2
    eq["twin_leg"]["energy_capacity_j"] = 100.0
    r = run(p)
    assert not r.passable
    assert any(f.check == "buffer_energy" for f in r.failures)


def test_peak_arrest_force_cap_enforced():
    p = twin_energy_payload()
    eq = p["equipment"][0]
    # 强曲线（峰值高）+ 很小的装备最大止坠力
    eq["max_arrest_force_kn"] = 1.0
    r = run(p)
    assert not r.passable
    assert any(f.check == "arrest_force" for f in r.failures)
    ff = next(f for f in r.failures if f.check == "arrest_force")
    assert ff.components["buffer_force_kn"] > 1.0


def test_connector_side_load_exceeded_blocks():
    r = run(twin_side_load_payload(limit=0.1))
    assert not r.passable
    assert r.first_failure.check == "connector_side_load"
    assert r.first_failure.components["connector_side_load_kn"] > \
        r.first_failure.components["connector_side_load_limit_kn"]
    assert r.first_failure.components["failing_leg"] in ("LA", "LB")


def test_short_leg_insufficient_blocks():
    r = run(twin_unequal_legs_payload())
    assert not r.passable and r.conclusive
    assert r.first_failure.check == "leg_length_insufficient"
    assert r.first_failure.components["failing_hook"] == "B"


def test_first_failure_locates_station_and_action():
    r = run(twin_angle_payload())
    ff = r.first_failure
    assert ff.station_index == 0
    assert ff.action in ("attach", "switch", "traverse")
    assert ff.position is not None


# ---------------------------------------------------------------- 求解器直测

def _curve(f=3.0, t=1.2):
    return [BufferCurvePoint(travel_m=0.0, force_kn=f),
            BufferCurvePoint(travel_m=t, force_kn=f)]


def test_solver_single_overhead_anchor_matches_energy_formula():
    # 双腿共点于头顶，等价单连接；恒力缓冲，W·h=(F−W)·s
    d = (0.0, 0.0, 1.4)
    legs = [
        twinleg.LegTarget("LA", "A", "anchor", "X", (0, 0, 3.4),
                          2.0, 200.0, 16.0),
        twinleg.LegTarget("LB", "B", "anchor", "X", (0, 0, 3.4),
                          2.0, 200.0, 16.0)]
    sol = twinleg.solve_twin_legs(80.0, 9.81, d, legs, _curve(), 1.2)
    assert sol.converged
    # 站立即张紧、无坠距：静态悬挂，峰值 = 体重
    assert abs(sol.buffer_force_kn - 0.7848) < 1e-3
    assert sol.deployment_m == 0.0


def test_solver_low_anchor_deployment_energy_balance():
    d = (0.0, 0.0, 1.4)
    legs = [
        twinleg.LegTarget("LA", "A", "anchor", "X", (0, 0, 0.4),
                          2.0, 20000.0, 16.0),
        twinleg.LegTarget("LB", "B", "anchor", "X", (0, 0, 0.4),
                          2.0, 20000.0, 16.0)]
    sol = twinleg.solve_twin_legs(80.0, 9.81, d, legs, _curve(3.0), 1.2)
    assert sol.converged and sol.reason is None
    # 解析：s = W·h/(F−W)，h=3 → ≈1.063
    assert abs(sol.deployment_m - 1.063) < 0.01
    assert abs(sol.buffer_force_kn - 3.0) < 1e-6
    # 缓冲能力只计一次（不是两腿各算一份）
    assert abs(sol.buffer_energy_j - 3.0 * sol.deployment_m * 1000.0) < 1.0


def test_solver_energy_insufficient_flag():
    d = (0.0, 0.0, 1.4)
    legs = [
        twinleg.LegTarget("LA", "A", "anchor", "X", (0, 0, 0.4),
                          2.0, 20000.0, 16.0),
        twinleg.LegTarget("LB", "B", "anchor", "X", (0.3, 0, 0.4),
                          2.0, 20000.0, 16.0)]
    sol = twinleg.solve_twin_legs(120.0, 9.81, d, legs, _curve(2.0, 0.2), 0.2)
    assert sol.reason == "buffer_energy_insufficient"
    assert sol.deployment_m == 0.2


# ---------------------------------------------------------------- 人工绑定

def test_manual_hook_leg_binding_replayed():
    r = run(twin_manual_payload())
    assert r.conclusive and r.passable
    ev = {(e.hook, e.action): e for e in r.sequence}
    assert ev[("A", "attach")].leg == "LA"
    assert ev[("A", "attach")].to_anchor == "a0"
    assert ev[("B", "attach")].leg == "LB"
    assert ev[("B", "attach")].to_anchor == "b0"


def test_manual_cross_binding_rejected_at_validation():
    p = twin_manual_payload()
    p["hook_order"][0]["leg"] = "LB"   # A 钩不得绑到 LB 腿
    with pytest.raises(pydantic.ValidationError):
        PlanPayload.model_validate(p)


# ---------------------------------------------------------------- 旧模型兼容

def test_legacy_equipment_result_unchanged():
    """旧等长模型：无 twin_leg_results，点锚分量与判定保持原样。"""
    r = run(passable_payload())
    assert r.twin_leg_results == []
    assert r.passable and r.conclusive
    # 旧响应字段仍在
    assert r.profile and r.profile[0].controlling_anchor
    for ev in r.sequence:
        assert ev.leg is None


def test_legacy_payload_without_twin_field_validates():
    p = passable_payload()
    assert "twin_leg" not in p["equipment"][0]
    r = run(copy.deepcopy(p))
    assert r.passable
