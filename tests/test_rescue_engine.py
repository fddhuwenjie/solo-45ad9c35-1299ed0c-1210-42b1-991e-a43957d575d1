"""坠落后救援推演引擎测试：逐步分量、最早受阻步骤、来源冻结与相关性。"""
from __future__ import annotations

import pytest

from app.engine import build_context
from app.engine.rescue import simulate_rescue, source_relevance_signature
from app.models import PlanPayload, RescuePayload

from .scenarios import (clearance_payload, flex_missing_params_payload,
                        flex_passable_payload, hook_chain_break_payload,
                        passable_payload, rescue_payload)


def _ctx(plan):
    return build_context(PlanPayload.model_validate(plan))


def _rp(**kw):
    return RescuePayload.model_validate(rescue_payload(**kw))


def _earliest(plan, **kw):
    res = simulate_rescue(_ctx(plan), _rp(**kw))
    return res


# ---------------------------------------------------------------- 可执行基准

def test_executable_baseline_steps_and_components():
    res = simulate_rescue(_ctx(passable_payload()), _rp())
    assert res.source_conclusive is True
    assert res.executable is True
    assert res.earliest_block is None
    assert res.suspension_station_count == 21        # 每个站点都可悬吊
    s0 = res.simulations[0]
    assert [x.code for x in s0.steps] == [
        "approach", "secondary_protection", "haul_unload",
        "detach_original", "transfer"]
    # 逐步分量：绳程、牵引力、锚点合力、累计耗时均齐备且非负
    for st in s0.steps:
        assert st.cumulative_minutes >= st.elapsed_minutes >= 0
        assert st.rope_travel_m >= 0 and st.rope_required_m >= 0
    haul = next(x for x in s0.steps if x.code == "haul_unload")
    # 80kg+5kg 载荷 ≈ 0.834 kN，5:1、η=0.8 → 有效倍率 4
    assert haul.pull_force_kn == pytest.approx(0.834 / 4, abs=1e-3)
    assert haul.rope_travel_m == pytest.approx(
        5 * haul.components["lift_m"], abs=1e-6)
    assert haul.anchor_resultant_kn > haul.components["load_kn"]
    # 伤员悬挂脚底标高 = 站面 0 − 总坠距 1.5
    assert s0.feet_z == pytest.approx(-1.5, abs=1e-6)
    # 脱离原系统步骤列出全部原连接
    det = next(x for x in s0.steps if x.code == "detach_original")
    assert det.components["connection_count"] >= 1
    assert det.components["released_connections"].startswith("anchor:")


def test_flexible_span_rescue_uses_cable_sag():
    """柔性跨段：悬挂脚底含动态下挠分量，原系统连接为滑梭。"""
    res = simulate_rescue(_ctx(flex_passable_payload()),
                          _rp(anchor_zs=[2.6, 2.6]))
    assert res.executable is True
    s = next(x for x in res.simulations if x.station_index == 10)
    # 总坠距比点锚大（钢索下挠），脚底低于 -1.5
    assert s.feet_z < -1.5
    det = next(x for x in s.steps if x.code == "detach_original")
    assert "shuttle:T" in det.components["released_connections"]
    appr = s.steps[0]
    assert appr.components["cable_sag_m"] > 0


# ---------------------------------------------------------------- 最早受阻

def test_source_inconclusive_blocks_all_simulation():
    res = simulate_rescue(_ctx(flex_missing_params_payload()),
                          _rp(anchor_zs=[2.6, 2.6]))
    assert res.source_conclusive is False
    assert res.executable is False
    assert res.suspension_station_count == 0
    b = res.earliest_block
    assert b.code == "source_inconclusive"
    assert b.step_code == "approach" and b.step_no == 1


def test_rescue_anchor_unreachable_returns_blocked_step():
    res = _earliest(passable_payload(), anchors=[
        {"id": "X1", "position": {"x": 50, "y": 0, "z": 3.4},
         "rated_load_kn": 22.0},
        {"id": "X2", "position": {"x": 50, "y": 5, "z": 3.4},
         "rated_load_kn": 22.0}])
    assert res.executable is False
    b = res.earliest_block
    assert b.code == "rescue_anchor_unreachable"
    assert b.step_code == "secondary_protection"
    assert b.components["nearest_distance_m"] > 3.5
    # 受阻方案不得标为可执行，且步骤停在受阻处
    sim = res.simulations[b.station_index]
    assert sim.executable is False
    assert [x.code for x in sim.steps] == ["approach"]


def test_rope_too_short_blocks():
    res = _earliest(passable_payload(), rope_main_len=3.0)
    b = res.earliest_block
    assert b.code == "rope_too_short"
    assert b.step_code == "haul_unload"
    assert b.components["required_m"] > b.components["rope_length_m"]


def test_descender_overload_blocks():
    res = _earliest(passable_payload(), desc_limit=0.2)
    b = res.earliest_block
    assert b.code == "descender_overload"
    assert b.components["descender_limit_kn"] == 0.2


def test_manual_pull_exceeded_blocks():
    # 1:1、效率 1 → 牵引 0.834 kN > 单人持续 0.5 kN
    res = _earliest(passable_payload(), ratio=1, eff=1.0)
    b = res.earliest_block
    assert b.code == "manual_pull_exceeded"
    assert b.components["rescuer_pull_kn"] == 0.5
    assert b.components["pull_force_kn"] > 0.5


def test_anchor_overload_blocks():
    res = _earliest(passable_payload(), rated=0.3)
    b = res.earliest_block
    assert b.code == "anchor_overload"
    assert b.components["resultant_kn"] > b.components["rated_load_kn"]


def test_time_limit_blocks_at_farthest_station():
    res = _earliest(passable_payload(), max_time=0.05)
    assert res.executable is False
    b = res.earliest_block
    assert b.code == "time_limit"
    assert b.components["cumulative_minutes"] > 0.05
    # 站 0 仅 13+ min 的固定作业耗时；0.05 min 在二次保护步骤即超时
    assert b.step_code in {"secondary_protection", "approach"}


def test_entry_off_route_blocks():
    res = _earliest(
        passable_payload(),
        entry={"id": "E", "position": {"x": 0, "y": 9, "z": 0}})
    b = res.earliest_block
    assert b.code == "entry_off_route"
    assert b.components["offset_m"] == 9.0


def test_equipment_reuse_same_primary_backup_rejected():
    with pytest.raises(ValueError):
        _rp(primary="R0a", backup="R0a", manual_reason="现场要求")


def test_rope_team_reuse_rejected():
    with pytest.raises(ValueError):
        _rp(rope_team="RT2", manual_reason="现场要求")


def test_manual_choice_requires_reason():
    with pytest.raises(ValueError):
        _rp(primary="R0a")


def test_manual_choice_recorded_in_result():
    # 短路线 + 加大挂接距离：人工指定 x=0 处的双锚在全部站点可达
    import copy
    plan = copy.deepcopy(passable_payload())
    plan["route"]["walk_polyline"] = [
        {"x": 0, "y": 0, "z": 0}, {"x": 1.5, "y": 0, "z": 0}]
    res = simulate_rescue(
        _ctx(plan),
        _rp(primary="R0a", backup="R0b",
            manual_reason="结构梁避开锐边",
            params={"anchor_rig_reach_m": 4.0}))
    assert res.executable is True
    fields = {d.field: d.value for d in res.manual_decisions}
    assert fields["primary_anchor_id"] == "R0a"
    assert all(d.reason == "结构梁避开锐边" for d in res.manual_decisions)
    # 各站确实使用了人工指定锚点（自动选择不会恒为 R0a/R0b）
    assert all(s.primary_anchor_id == "R0a" and s.backup_anchor_id == "R0b"
               for s in res.simulations)


def test_impact_station_not_simulated():
    """净空不足（撞击）的站点不产生悬吊推演，但登记为 impact。"""
    res = simulate_rescue(_ctx(clearance_payload()), _rp())
    assert res.executable is False
    # 所有站点均撞击：无悬吊推演，impact_stations 非空
    assert res.suspension_station_count == 0
    assert len(res.impact_stations) > 0
    assert res.impact_stations[0]["reason"] == "clearance_impact"


def test_hook_chain_break_has_no_suspension_after_failure():
    """换挂链断开：断开站之后序列状态为 None，不生成悬吊推演。"""
    res = simulate_rescue(_ctx(hook_chain_break_payload()), _rp())
    # 前段站点仍可悬吊；断开站之后不推演
    assert res.suspension_station_count < 21
    assert all(s.station_index < 21 for s in res.simulations)


# ---------------------------------------------------------------- 相关性签名

def test_relevance_signature_changes_only_with_related_route():
    import copy
    plan = PlanPayload.model_validate(passable_payload())
    res = simulate_rescue(build_context(plan), _rp())
    sig0 = source_relevance_signature(plan, res)

    # 仅改步距（站点重编号）：用各自推演结果计算的悬挂几何集合不变
    plan2 = copy.deepcopy(plan)
    plan2.params.station_spacing_m = 1.0
    res2 = simulate_rescue(build_context(plan2), _rp())
    geom0 = sig0[-1]
    geom2 = source_relevance_signature(plan2, res2)[-1]
    # 0.5 m 步距站点集是 1.0 m 步距的超集；共有站点悬挂几何一致
    assert set(geom2).issubset(set(geom0))

    # 相关修改：锚点额定载荷变化 → 签名变化
    changed = copy.deepcopy(plan)
    changed.route.anchors[0].rated_load_kn = 99.0
    assert source_relevance_signature(changed, res) != sig0
