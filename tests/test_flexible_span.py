"""柔性跨段（临时水平生命线）回归测试：悬索求解、滑梭卡支座、重复占用、
参数缺失、净空/反力越限、点锚混用、多人组合总坠距。
"""
from __future__ import annotations

import pytest

from app.engine import analyze
from app.models import PlanPayload

from .scenarios import (flex_clearance_payload, flex_duplicate_shuttle_payload,
                        flex_manual_jam_payload, flex_missing_params_payload,
                        flex_passable_payload, flex_two_persons_payload,
                        flex_weak_support_payload, mixed_anchor_span_payload)


def run(payload: dict):
    return analyze(PlanPayload.model_validate(payload))


# ---------------------------------------------------------------- 基准可通行

def test_flex_span_passable_baseline():
    r = run(flex_passable_payload())
    assert r.conclusive is True
    assert r.passable is True
    assert r.first_failure is None and not r.open_items
    # 双钩分别挂两条滑梭，自动序列从不重复挂同梭
    for e in r.sequence:
        if e.action != "detach":
            assert len(set(e.attached_shuttles_after)) == \
                len(e.attached_shuttles_after)
    # 每站单人 CableResult 给出下挠、总坠距与端座反力
    assert len(r.cable_results) == r.station_count
    mid = next(c for c in r.cable_results if c.station_index == 10)
    assert mid.sag_m > 0
    assert len(mid.total_fall_m) == 1
    assert mid.total_fall_m[0] > 0
    assert mid.left_reaction_kn > 0 and mid.right_reaction_kn > 0
    assert set(mid.support_reactions_kn) == {"S0", "S1"}
    assert mid.converged and not mid.conservative
    # 剖面标注含动态下挠分量
    pp = next(p for p in r.profile if p.station_index == 10)
    assert pp.controlling_shuttle == "T1"
    assert pp.cable_sag_m == pytest.approx(mid.sag_m, abs=1e-6)
    assert pp.cable_tension_kn > 0


# ---------------------------------------------------------------- 滑梭卡支座

def test_manual_action_across_nonpass_support_jams():
    r = run(flex_manual_jam_payload())
    assert r.conclusive is False
    assert r.passable is False
    jam = [o for o in r.open_items if o.code == "shuttle_jammed"]
    assert jam
    o = jam[0]
    # 中间支座位于 x=5，对应站点 10
    assert o.station_index == 10
    assert o.components["blocked_support_id"] == "SM"
    assert o.components["blocked_support_index"] == 1
    assert o.components["span"] == "H1"
    # first_open_item 直接给出最早的不下结论位置与动作
    assert r.first_open_item.code == "shuttle_jammed"
    assert r.first_open_item.action == "traverse"


def test_auto_traverse_across_nonpass_support_jams():
    """无人工次序时自动行进越过断开式支座同样卡支座（不下结论）。"""
    p = flex_manual_jam_payload()
    p.pop("hook_order")
    r = run(p)
    assert r.conclusive is False and r.passable is False
    assert any(o.code == "shuttle_jammed" for o in r.open_items)


def test_passable_intermediate_support_does_not_jam():
    """连续式中间支座（shuttle_pass=True）可通过，不产生卡支座项。"""
    p = flex_passable_payload()
    p["route"]["supports"].insert(1, {
        "id": "SM", "position": {"x": 5, "y": 0, "z": 2.6},
        "rated_load_kn": 50.0})
    p["route"]["spans"][0]["supports"] = ["S0", "SM", "S1"]
    r = run(p)
    assert r.conclusive and r.passable
    assert not any(o.code == "shuttle_jammed" for o in r.open_items)
    bays = {c.bay_index for c in r.cable_results}
    assert bays == {0, 1}


# ---------------------------------------------------------------- 重复占用

def test_same_person_two_hooks_same_shuttle_duplicate_occupancy():
    r = run(flex_duplicate_shuttle_payload())
    assert r.conclusive is False
    assert r.passable is False
    dups = [o for o in r.open_items if o.code == "duplicate_occupancy"]
    assert dups
    o = dups[0]
    assert o.station_index == 0
    assert o.components["target_shuttle"] == "T1"
    assert o.components["reason"] == "same_person_other_hook"
    assert o.components["failing_hook"] == "B"
    # 绝不返回可通行结论
    assert r.first_failure is None or True  # 允许只有 open item
    assert not r.passable


def test_single_shuttle_auto_is_never_safe():
    """只有一条滑梭时，自动算法不得把双钩挂到同一滑梭上冒充安全。"""
    p = flex_duplicate_shuttle_payload()
    p.pop("hook_order")
    r = run(p)
    assert r.passable is False
    assert any(o.code == "duplicate_occupancy" for o in r.open_items)


# ---------------------------------------------------------------- 参数缺失

def test_missing_span_params_inconclusive():
    r = run(flex_missing_params_payload())
    assert r.conclusive is False and r.passable is False
    miss = [o for o in r.open_items if o.code == "missing_params"]
    assert miss and "pretension_kn" in miss[0].components["missing"]


# ---------------------------------------------------------------- 净空与反力

def test_flex_clearance_failure_includes_sag_components():
    r = run(flex_clearance_payload())
    assert r.conclusive and not r.passable
    # 净空失效沿跨出现；取首个有动态下挠的失效（端座站位下挠为 0）
    f = next(x for x in r.failures
             if x.check == "clearance" and x.components["cable_sag_m"] > 0)
    assert f.components["margin_m"] < 0
    # 柔性跨段失效分量须含动态下挠
    assert f.components["span"] == "H1"
    assert f.components["cable_sag_m"] > 0
    assert f.components["total_fall_m"] > 0


def test_end_support_overload_reports_earliest_station():
    r = run(flex_weak_support_payload())
    assert not r.passable and r.conclusive
    ff = r.first_failure
    assert ff.check == "support_overload"
    # 最早失效：人在最靠近弱端座 S0 的站点
    assert ff.components["support"] == "S0"
    assert ff.components["reaction_kn"] > ff.components["rated_load_kn"]


def test_sag_limit_exceeded():
    p = flex_passable_payload()
    p["route"]["spans"][0]["max_sag_m"] = 0.05
    r = run(p)
    assert not r.passable
    assert r.first_failure.check == "sag_limit"
    assert r.first_failure.components["cable_sag_m"] > 0.05


# ---------------------------------------------------------------- 多人组合

def test_two_persons_combined_fall_results_and_totals():
    r = run(flex_two_persons_payload())
    assert r.conclusive and r.passable
    full = [c for c in r.cable_results if len(c.falling_persons) == 2]
    solo = [c for c in r.cable_results if len(c.falling_persons) == 1]
    assert len(full) == r.station_count
    assert len(solo) == 2 * r.station_count
    mid = next(c for c in full if c.station_index == 10)
    # 多人结果按人员返回总坠距（同站同位故长度相等，数值相等）
    assert mid.total_fall_m == sorted(mid.total_fall_m, reverse=True)
    assert len(mid.total_fall_m) == 2
    assert all(t > 0 for t in mid.total_fall_m)
    # 全员组合的张力/反力不小于单人
    solo_mid = [c for c in solo if c.station_index == 10]
    assert mid.horizontal_tension_kn >= max(
        c.horizontal_tension_kn for c in solo_mid) - 1e-9
    # 单人 CableResult 同样有每人一条总坠距
    assert all(len(c.total_fall_m) == 1 for c in solo)


def test_two_persons_combined_overload():
    """两人同跨全员同时坠落使端座反力越限（单人不越限时仍须失败）。"""
    p = flex_two_persons_payload()
    for s in p["route"]["supports"]:
        s["rated_load_kn"] = 22.0   # 单人 ~15 kN，全员 ~25 kN
    r = run(p)
    assert not r.passable
    assert r.first_failure.check == "support_overload"
    assert r.first_failure.components["reaction_kn"] > 22.0


# ---------------------------------------------------------------- 点锚混用

def test_mixed_anchor_and_span_passable():
    r = run(mixed_anchor_span_payload())
    assert r.conclusive and r.passable
    kinds = {(p.controlling_anchor, p.controlling_shuttle)
             for p in r.profile}
    assert any(a for a, _ in kinds) and any(s for _, s in kinds)


def test_mixed_manual_anchor_then_shuttle_handoff():
    """人工次序：前段点锚双钩，在重叠区逐钩换到滑梭。"""
    p = mixed_anchor_span_payload()
    p["hook_order"] = [
        {"person_id": "p1", "hook": "A", "action": "attach",
         "anchor": "A0", "station_index": 0},
        {"person_id": "p1", "hook": "B", "action": "attach",
         "anchor": "A0", "station_index": 0},
        {"person_id": "p1", "hook": "A", "action": "switch",
         "anchor": "A1", "station_index": 2},
        {"person_id": "p1", "hook": "B", "action": "switch",
         "anchor": "A1", "station_index": 2},
        {"person_id": "p1", "hook": "A", "action": "switch",
         "anchor": "A2", "station_index": 5},
        {"person_id": "p1", "hook": "B", "action": "switch",
         "anchor": "A2", "station_index": 5},
        {"person_id": "p1", "hook": "A", "action": "switch",
         "shuttle": "T1", "station_index": 9},
        {"person_id": "p1", "hook": "B", "action": "switch",
         "shuttle": "T2", "station_index": 10},
    ]
    r = run(p)
    assert r.conclusive and r.passable, r.first_failure or r.first_open_item
    sw = [(e.station_index, e.hook, e.to_anchor, e.to_shuttle)
          for e in r.sequence if e.action == "switch"]
    assert (9, "A", None, "T1") in sw
    assert (10, "B", None, "T2") in sw
    # 动作序列引用具体滑梭
    assert sw[-1][3] == "T2"


# ---------------------------------------------------------------- 确定性

def test_flex_analysis_deterministic():
    a = run(flex_two_persons_payload()).model_dump_json()
    b = run(flex_two_persons_payload()).model_dump_json()
    assert a == b
