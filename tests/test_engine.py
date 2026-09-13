"""引擎单元测试：几何、坠落计算分量、序列生成、各类失效判定。"""
from __future__ import annotations

import math

import pytest

from app.engine import analyze
from app.engine import geometry as g
from app.engine.calc import free_fall_distance
from app.models import PlanPayload

from .scenarios import (anchor_direction_payload, anchor_overload_payload,
                        clearance_payload, hook_chain_break_payload,
                        manual_order_payload, passable_payload,
                        sharp_edge_payload, sharing_limit_payload,
                        single_overload_payload, sweep_payload)


def run(payload: dict):
    return analyze(PlanPayload.model_validate(payload))


# ---------------------------------------------------------------- 几何

def test_discretize_keeps_vertices_and_spacing():
    pts = [(0, 0, 0), (2, 0, 0), (2, 3, 0)]
    st = g.discretize_polyline(pts, 0.5)
    assert st[0] == (0, 0, 0)
    assert (2, 0, 0) in st and st[-1] == (2, 3, 0)
    step = max(g.dist3(a, b) for a, b in zip(st, st[1:]))
    assert step <= 0.5 + 1e-9


def test_free_fall_distance_clamped():
    # 锚点与 D 环同高：FFD = L；锚点高于 D 环 2L 以上：FFD = 0；低于：最多 2L
    class E:  # 轻量替身
        lanyard_length_m = 2.0
    assert free_fall_distance(E, 1.4, 1.4) == 2.0
    assert free_fall_distance(E, 1.4, 10.0) == 0.0
    assert free_fall_distance(E, 1.4, -5.0) == 4.0


# ---------------------------------------------------------------- 可通行

def test_passable_route():
    r = run(passable_payload())
    assert r.passable, r.first_failure
    assert r.first_failure is None and not r.failures
    actions = [e.action for e in r.sequence]
    assert "attach" in actions and "switch" in actions and "detach" in actions
    # 每个事件后至少保留一个有效连接（终点全部解钩除外）
    for e in r.sequence:
        if e.action != "detach":
            assert len(e.attached_after) >= 1
    # 终点解钩逐钩进行，第一次解钩后仍有一钩连接
    detaches = [e for e in r.sequence if e.action == "detach"]
    assert detaches[0].attached_after != []
    assert detaches[-1].attached_after == []
    # 剖面标注覆盖所有站点
    assert len(r.profile) == r.station_count
    assert all(p.required_clearance_m is not None for p in r.profile)


# ---------------------------------------------------------------- 换挂链断开

def test_hook_chain_break_reports_first_failure():
    r = run(hook_chain_break_payload())
    assert not r.passable
    ff = r.first_failure
    assert ff is not None and ff.check == "hook_chain"
    assert ff.action == "switch"
    assert ff.person_id == "p1"
    # 失败位置在首锚点触及边界附近（x≈1.0）
    assert 0.5 <= ff.position.x <= 1.5
    assert "current_anchor" in ff.components
    # 失败前的每个状态仍至少一钩连接
    for e in r.sequence:
        if e.action != "detach":
            assert len(e.attached_after) >= 1


# ---------------------------------------------------------------- 净空不足

def test_clearance_negative_margin():
    r = run(clearance_payload())
    assert not r.passable
    ff = r.first_failure
    assert ff.check == "clearance" and ff.action == "traverse"
    c = ff.components
    # 计算分量齐全且余量为负
    assert c["margin_m"] < 0
    assert c["required_clearance_m"] > c["available_clearance_m"]
    assert c["free_fall_m"] == 0.0          # 锚点在 D 环上方 2 m
    assert c["total_fall_m"] == 1.5         # 0 + 0.2 伸长 + 1.0 缓冲 + 0.3 安全带
    assert c["required_clearance_m"] == 2.1  # + 0.6 安全余量
    assert c["available_clearance_m"] == 1.6


# ---------------------------------------------------------------- 摆坠扫掠

def test_sweep_hits_obstacle():
    r = run(sweep_payload())
    assert not r.passable
    ff = r.first_failure
    assert ff.check == "sweep"
    assert ff.components["obstacle_id"] == "side1"
    assert ff.components["horizontal_offset_m"] > 0.3
    assert ff.components["swing_radius_m"] == 4.2  # 3.0 + 0.2 + 1.0


# ---------------------------------------------------------------- 锐边

def test_sharp_edge_mismatch():
    r = run(sharp_edge_payload())
    assert not r.passable
    ff = r.first_failure
    assert ff.check == "sharp_edge"
    assert ff.components["edge_id"] == "E1"
    assert ff.components["edge_sharpness_class"] == 2
    assert ff.components["equipment_sharp_edge_rating"] == 1
    # 只发生在边缘绳长范围内
    assert abs(ff.position.x - 5.0) <= 2.0 + 1e-6


def test_sharp_edge_ok_when_rated():
    p = sharp_edge_payload()
    p["equipment"] = p["equipment"][:1]
    p["equipment"][0]["sharp_edge_rating"] = 2
    assert run(p).passable


# ---------------------------------------------------------------- 锚点方向

def test_anchor_direction_violation():
    r = run(anchor_direction_payload())
    assert not r.passable
    ff = r.first_failure
    assert ff.check == "anchor_direction"
    assert ff.components["force_angle_deg"] > 30.0
    assert ff.components["allowed_half_angle_deg"] == 30.0


# ---------------------------------------------------------------- 共用锚点

def test_shared_anchor_overload():
    r = run(anchor_overload_payload())
    assert not r.passable
    ff = r.first_failure
    assert ff.check == "anchor_overload"
    assert ff.components["user_count"] == 2
    assert ff.components["combined_force_kn"] > ff.components["rated_load_kn"]


def test_sharing_limit_blocks_second_person():
    r = run(sharing_limit_payload())
    assert not r.passable
    ff = r.first_failure
    assert ff.check == "hook_chain" and ff.action == "attach"
    assert ff.person_id == "p2" and ff.station_index == 0


# ---------------------------------------------------------------- 单人过载

def test_single_person_anchor_overload():
    r = run(single_overload_payload())
    assert not r.passable
    ff = r.first_failure
    assert ff.check == "anchor_overload" and ff.action == "traverse"
    assert ff.station_index == 0
    c = ff.components
    assert c["user_count"] == 1
    assert c["combined_force_kn"] == pytest.approx(0.7848, abs=1e-3)
    assert c["rated_load_kn"] == 0.1
    assert "共用" not in ff.message


def test_single_person_under_rated_passes():
    # 单人止坠合力 0.7848 kN < 1.0 kN 额定值，可通行
    assert run(single_overload_payload(rated=1.0)).passable


# ---------------------------------------------------------------- 人工调序

def test_manual_order_replayed_and_passable():
    r = run(manual_order_payload())
    assert r.passable, r.first_failure
    seq = r.sequence
    # 人工指定的换钩站点 1,4,7,...,19 被严格遵循（自动算法在站点 2 换钩）
    switches = [e for e in seq if e.action == "switch"]
    assert [e.station_index for e in switches] == \
        [s for s in (3 * k - 2 for k in range(1, 8)) for _ in (0, 1)]
    # 锚点使用次序与人工输入一致
    attaches = [e for e in seq if e.action == "attach"]
    assert [e.to_anchor for e in attaches] == ["A0", "A0"]
    assert [e.to_anchor for e in switches[:4]] == ["A1", "A1", "A2", "A2"]
    # 终点自动补记解钩
    assert seq[-1].action == "detach" and seq[-1].attached_after == []
    # 每个非终点状态至少保留一个有效连接
    for e in seq:
        if e.action != "detach":
            assert len(e.attached_after) >= 1


def test_manual_order_input_sequence_preserved():
    p = manual_order_payload()
    # 颠倒站点 4 两个换钩动作的输入次序，事件次序应跟随输入
    idx = [k for k, a in enumerate(p["hook_order"])
           if a["station_index"] == 4]
    p["hook_order"][idx[0]], p["hook_order"][idx[1]] = \
        p["hook_order"][idx[1]], p["hook_order"][idx[0]]
    r = run(p)
    at4 = [e for e in r.sequence
           if e.station_index == 4 and e.action == "switch"]
    assert [e.hook for e in at4] == ["B", "A"]


def test_manual_switch_target_out_of_reach():
    p = manual_order_payload()
    # 站点 1 换向 x=4.5 的 A3，超出连接器触及范围
    p["hook_order"][2]["anchor"] = "A3"
    r = run(p)
    assert not r.passable
    ff = r.first_failure
    assert ff.check == "hook_chain" and ff.action == "switch"
    assert ff.station_index == 1
    assert ff.components["target_anchor"] == "A3"


def test_manual_detach_without_backup_rejected():
    p = passable_payload()
    p["hook_order"] = [
        {"person_id": "p1", "hook": "A", "action": "attach",
         "anchor": "A0", "station_index": 0},
        {"person_id": "p1", "hook": "B", "action": "attach",
         "anchor": "A0", "station_index": 0},
        {"person_id": "p1", "hook": "A", "action": "detach",
         "station_index": 1},
        {"person_id": "p1", "hook": "B", "action": "detach",
         "station_index": 1},
    ]
    r = run(p)
    assert not r.passable
    ff = r.first_failure
    assert ff.check == "hook_chain" and ff.action == "detach"
    assert ff.station_index == 1


def test_manual_order_gap_breaks_chain():
    p = passable_payload()
    # 直接跳到站点 10 换 A4：A0 在站点 3 已出触及范围，行进中断链
    p["hook_order"] = [
        {"person_id": "p1", "hook": "A", "action": "attach",
         "anchor": "A0", "station_index": 0},
        {"person_id": "p1", "hook": "B", "action": "attach",
         "anchor": "A0", "station_index": 0},
        {"person_id": "p1", "hook": "A", "action": "switch",
         "anchor": "A4", "station_index": 10},
    ]
    r = run(p)
    assert not r.passable
    ff = r.first_failure
    assert ff.check == "hook_chain" and ff.action == "traverse"
    assert ff.station_index == 3


# ---------------------------------------------------------------- 确定性

def test_analysis_deterministic():
    a = run(passable_payload()).model_dump_json()
    b = run(passable_payload()).model_dump_json()
    assert a == b
