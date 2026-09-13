"""测试场景构造：各类典型失效与可通行场景的载荷。"""
from __future__ import annotations


def v(x, y, z):
    return {"x": x, "y": y, "z": z}


def base_equipment(**kw):
    eq = {
        "id": "eq1",
        "lanyard_length_m": 2.0,
        "elongation_m": 0.2,
        "buffer_travel_m": 1.0,
        "sharp_edge_rating": 1,
        "connector_reach_m": 0.3,
        "max_arrest_force_kn": 6.0,
    }
    eq.update(kw)
    return eq


def base_person(pid="p1", **kw):
    p = {"id": pid, "weight_kg": 80.0, "equipment_id": "eq1",
         "d_ring_height_m": 1.4, "body_radius_m": 0.35}
    p.update(kw)
    return p


def overhead_anchors(xs, z=3.4, **kw):
    """沿 x 均布的头顶锚点（D 环上方 2 m）。"""
    a = {"rated_load_kn": 12.0, "max_users": 1}
    a.update(kw)
    return [{"id": f"A{k}", "position": v(x, 0.0, z), **a}
            for k, x in enumerate(xs)]


def passable_payload():
    """可通行基准：直线 10 m，头顶锚点密排，无障碍。"""
    return {
        "route": {
            "walk_polyline": [v(0, 0, 0), v(10, 0, 0)],
            "drop_edges": [],
            "obstacles": [],
            "anchors": overhead_anchors(
                [0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.0]),
        },
        "persons": [base_person()],
        "equipment": [base_equipment()],
        "params": {"station_spacing_m": 0.5, "safety_margin_m": 0.6,
                   "harness_stretch_m": 0.3, "swing_threshold_m": 0.3,
                   "gravity": 9.81},
    }


def hook_chain_break_payload():
    """锚点间距过大：中途无锚可换挂，双钩将同时解开。"""
    p = passable_payload()
    p["route"]["anchors"] = overhead_anchors([0.0, 9.0])
    return p


def clearance_payload():
    """下层管线侵入净空：缓冲包展开后撞上。"""
    p = passable_payload()
    p["route"]["obstacles"] = [
        {"kind": "box", "id": "pipe1",
         "min": v(0, -1, -2.0), "max": v(10, 1, -1.6)}]
    return p


def sweep_payload():
    """锚点侧向偏移，摆坠扫掠体撞侧面障碍物。"""
    p = passable_payload()
    p["equipment"] = [base_equipment(lanyard_length_m=3.0)]
    anchors = []
    for k, x in enumerate([0, 2, 4, 6, 8, 10]):
        anchors.append({"id": f"A{k}", "position": v(x, 1.5, 3.4),
                        "rated_load_kn": 12.0, "max_users": 1})
    p["route"]["anchors"] = anchors
    p["route"]["obstacles"] = [
        {"kind": "box", "id": "side1",
         "min": v(0, 0.8, -2.0), "max": v(10, 2.0, -1.0)}]
    return p


def sharp_edge_payload():
    """落差边缘锐边等级超过装备适配等级。"""
    p = passable_payload()
    p["route"]["drop_edges"] = [
        {"id": "E1", "point": v(5, 0, 0), "sharpness_class": 2,
         "drop_depth_m": 3.0}]
    return p


def anchor_direction_payload():
    """锚点允许受力方向锥不允许向下拉。"""
    p = passable_payload()
    for a in p["route"]["anchors"]:
        a["allowed_axis"] = v(0, 0, 1)
        a["allowed_half_angle_deg"] = 30.0
    return p


def anchor_overload_payload():
    """两人共用锚点，合力超过额定载荷。"""
    p = passable_payload()
    for a in p["route"]["anchors"]:
        a["max_users"] = 2
        a["rated_load_kn"] = 1.5
    p["persons"] = [base_person("p1"), base_person("p2")]
    return p


def sharing_limit_payload():
    """单锚点限 1 人，第二人无法建立连接。"""
    p = passable_payload()
    p["route"]["walk_polyline"] = [v(0, 0, 0), v(1, 0, 0)]
    p["route"]["anchors"] = [
        {"id": "A0", "position": v(0, 0, 3.4),
         "rated_load_kn": 12.0, "max_users": 1}]
    p["persons"] = [base_person("p1"), base_person("p2")]
    return p


def single_overload_payload(rated=0.1):
    """单人止坠合力 0.7848 kN 超过锚点额定值。"""
    p = passable_payload()
    for a in p["route"]["anchors"]:
        a["rated_load_kn"] = rated
    return p


def manual_order_payload():
    """人工挂接次序：站点 0 双钩挂 A0，随后在站点 3k-2 依次换挂 A1..A7。

    自动算法在站点 2 才换钩，人工次序把首次换钩提前到站点 1，可据此区分。
    """
    p = passable_payload()
    acts = [
        {"person_id": "p1", "hook": "A", "action": "attach",
         "anchor": "A0", "station_index": 0},
        {"person_id": "p1", "hook": "B", "action": "attach",
         "anchor": "A0", "station_index": 0},
    ]
    for k in range(1, 8):
        s = 3 * k - 2
        acts.append({"person_id": "p1", "hook": "A", "action": "switch",
                     "anchor": f"A{k}", "station_index": s})
        acts.append({"person_id": "p1", "hook": "B", "action": "switch",
                     "anchor": f"A{k}", "station_index": s})
    p["hook_order"] = acts
    return p


# ---------------------------------------------------------------- 柔性跨段

def flex_equipment(**kw):
    """滑梭短绳装备：绳长 1.2 m，连接器余量 0.3 m。"""
    eq = {
        "id": "eqf",
        "lanyard_length_m": 1.2,
        "elongation_m": 0.2,
        "buffer_travel_m": 1.0,
        "sharp_edge_rating": 1,
        "connector_reach_m": 0.3,
        "max_arrest_force_kn": 6.0,
    }
    eq.update(kw)
    return eq


def flex_span_route(supports=None, span_kw=None, shuttles=None,
                    shuttle_kw=None, anchors=None, drop_edges=None,
                    obstacles=None):
    """0~10 m 直线管廊 + 头顶柔性跨段（默认端座 z=2.6）。

    默认给两条滑梭（双钩各一），跨段容许 2 人。
    """
    if supports is None:
        supports = [
            {"id": "S0", "position": v(0, 0, 2.6), "rated_load_kn": 50.0},
            {"id": "S1", "position": v(10, 0, 2.6), "rated_load_kn": 50.0},
        ]
    span = {
        "id": "H1",
        "supports": [s["id"] for s in supports],
        "pretension_kn": 5.0,
        "line_density_kg_m": 0.3,
        "axial_stiffness_kn": 20000.0,
        "max_sag_m": 1.5,
        "shuttle_pass": True,
        "max_users": 2,
    }
    if span_kw:
        span.update(span_kw)
    if shuttles is None:
        kw = {"connector_reach_m": 0.3, "max_users": 2}
        if shuttle_kw:
            kw.update(shuttle_kw)
        shuttles = [
            {"id": "T1", "span_id": "H1", **kw},
            {"id": "T2", "span_id": "H1", **kw},
        ]
    return {
        "walk_polyline": [v(0, 0, 0), v(10, 0, 0)],
        "drop_edges": drop_edges or [],
        "obstacles": obstacles or [],
        "anchors": anchors or [],
        "supports": supports,
        "spans": [span],
        "shuttles": shuttles,
    }


def flex_payload(persons=1, **route_kw):
    return {
        "route": flex_span_route(**route_kw),
        "persons": [{"id": f"p{k + 1}", "weight_kg": 80.0,
                     "equipment_id": "eqf"}
                    for k in range(persons)],
        "equipment": [flex_equipment()],
        "params": {"station_spacing_m": 0.5},
    }


def flex_passable_payload():
    """双滑梭单作业者，自动序列可通行。"""
    return flex_payload()


def flex_manual_jam_payload():
    """人工把滑梭挂在跨起点，随后越过不可通过的中间支座（卡支座）。"""
    p = flex_payload(
        supports=[
            {"id": "S0", "position": v(0, 0, 2.6), "rated_load_kn": 50.0},
            {"id": "SM", "position": v(5, 0, 2.6), "rated_load_kn": 50.0},
            {"id": "S1", "position": v(10, 0, 2.6), "rated_load_kn": 50.0},
        ],
        span_kw={"supports": ["S0", "SM", "S1"], "shuttle_pass": False})
    p["hook_order"] = [
        {"person_id": "p1", "hook": "A", "action": "attach",
         "shuttle": "T1", "station_index": 0},
        {"person_id": "p1", "hook": "B", "action": "attach",
         "shuttle": "T2", "station_index": 0},
    ]
    return p


def flex_duplicate_shuttle_payload():
    """同一人的 A、B 钩在同站重复挂到同一滑梭 T1（重复占用，不下结论）。"""
    p = flex_payload(shuttles=[{"id": "T1", "span_id": "H1",
                                "connector_reach_m": 0.3, "max_users": 2}])
    p["hook_order"] = [
        {"person_id": "p1", "hook": "A", "action": "attach",
         "shuttle": "T1", "station_index": 0},
        {"person_id": "p1", "hook": "B", "action": "attach",
         "shuttle": "T1", "station_index": 0},
    ]
    return p


def flex_missing_params_payload():
    """缺少预张力：悬索无法求解，不下结论。"""
    return flex_payload(span_kw={"pretension_kn": None})


def flex_weak_support_payload():
    """端座结构容许反力过低：同跨组合反力越限。"""
    return flex_payload(
        supports=[
            {"id": "S0", "position": v(0, 0, 2.6), "rated_load_kn": 5.0},
            {"id": "S1", "position": v(10, 0, 2.6), "rated_load_kn": 50.0},
        ])


def flex_two_persons_payload():
    """两人同跨：四条滑梭（每人双钩各一条，滑梭限 1 人），跨段容许 2 人。

    同站同 bay 时既给出每人单人 CableResult，也给出全员同时坠落组合。
    """
    shuttles = [
        {"id": f"T{k}", "span_id": "H1", "connector_reach_m": 0.3,
         "max_users": 1}
        for k in range(1, 5)
    ]
    return flex_payload(persons=2, shuttles=shuttles)


def flex_clearance_payload():
    """下层障碍侵入净空：钢索动态下挠占用脚下净空后余量为负。"""
    return flex_payload(obstacles=[
        {"kind": "box", "id": "b1",
         "min": v(0, -1, -2.5), "max": v(10, 1, -2.0)}])


def mixed_anchor_span_payload():
    """点锚 + 柔性跨段混用：x∈[0,6] 点锚（与端座同高 z=2.6，配 2 m 绳装备），
    x∈[0,10] 柔性跨段，重叠区可点锚↔滑梭换挂，双钩分别用点锚/滑梭。"""
    anchors = [
        {"id": f"A{k}", "position": v(x, 0, 2.6),
         "rated_load_kn": 12.0, "max_users": 1}
        for k, x in enumerate([0, 1.5, 3.0, 4.5, 6.0])
    ]
    p = flex_payload(anchors=anchors)
    # 两套装备：点锚段用 2 m 绳（余量 0.3），人员只有一名时引用点锚装备
    p["equipment"].append({
        "id": "eqa", "lanyard_length_m": 2.0, "elongation_m": 0.2,
        "buffer_travel_m": 1.0, "sharp_edge_rating": 1,
        "connector_reach_m": 0.3, "max_arrest_force_kn": 6.0})
    # 人员装备取两套可达包络的较大者（2.0 m），滑梭同样可挂
    p["persons"][0]["equipment_id"] = "eqa"
    return p


# ---------------------------------------------------------------- 救援推演

def rescue_payload(*, entry=None, anchors=None, rope_main_len=60.0,
                   rope_sec_len=20.0, ratio=5, eff=0.8,
                   desc_limit=2.5, rated=22.0, max_time=30.0,
                   rescuers=2, stretcher_kg=5.0, landing=None,
                   primary=None, backup=None, rope_team=None,
                   manual_reason="", anchor_zs=None, params=None):
    """救援方案载荷：头顶 z=3.4 的两个救援锚点（x=0 与 x=10），
    5:1 主绳组 + 1:1 二次保护绳组。"""
    if anchor_zs is None:
        anchor_zs = [3.4, 3.4]
    if anchors is None:
        # 沿路成对密布主锚（y=0）/ 备份锚（y=0.3），任意站 3.5 m
        # 挂接距离内均有两个不同锚点（竖直 3.4 m + 水平 ≤0.9 m）
        anchors = []
        for k, x in enumerate([0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.0]):
            anchors.append({"id": f"R{k}a", "position": v(x, 0.0, anchor_zs[0]),
                            "rated_load_kn": rated})
            anchors.append({"id": f"R{k}b", "position": v(x, 0.3, anchor_zs[1]),
                            "rated_load_kn": rated})
    body = {
        "entry": entry or {"id": "E1", "position": v(0, 0, 0)},
        "rescue_anchors": anchors,
        "rescuers": [{"id": f"rr{k + 1}", "weight_kg": 80.0}
                     for k in range(rescuers)],
        "rope_teams": [
            {"id": "RT1", "rope_length_m": rope_main_len,
             "pulley_ratio": ratio, "pulley_efficiency": eff,
             "descender_limit_kn": desc_limit},
            {"id": "RT2", "rope_length_m": rope_sec_len,
             "pulley_ratio": 1, "pulley_efficiency": 1.0,
             "descender_limit_kn": desc_limit},
        ],
        "secondary_rope_team_id": "RT2",
        "stretcher": {"id": "ST1", "length_m": 2.0, "width_m": 0.6,
                      "height_m": 0.4, "weight_kg": stretcher_kg},
        "max_suspension_minutes": max_time,
        "landing_point": landing,
        "manual_reason": manual_reason,
    }
    if primary is not None:
        body["primary_anchor_id"] = primary
    if backup is not None:
        body["backup_anchor_id"] = backup
    if rope_team is not None:
        body["rope_team_id"] = rope_team
    if params is not None:
        body["params"] = params
    return body
