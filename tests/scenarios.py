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


# ---------------------------------------------------------------- Y 型双腿系绳

def twin_equipment(**kw):
    """默认 Y 型双腿系绳装备：两腿 2.5 m 等长、刚度 2000 kN/m、恒力 3 kN
    缓冲曲线（行程 1.2 m）、能量容量 5000 J、允许夹角 120°。"""
    eq = {
        "id": "eqt",
        "lanyard_length_m": 2.5,
        "elongation_m": 0.2,
        "buffer_travel_m": 1.0,
        "sharp_edge_rating": 1,
        "connector_reach_m": 0.3,
        "max_arrest_force_kn": 6.0,
        "twin_leg": {
            "legs": [
                {"id": "LA", "hook": "A", "leg_length_m": 2.5,
                 "axial_stiffness_kn": 2000.0,
                 "connector_side_load_limit_kn": 16.0},
                {"id": "LB", "hook": "B", "leg_length_m": 2.5,
                 "axial_stiffness_kn": 2000.0,
                 "connector_side_load_limit_kn": 16.0},
            ],
            "buffer_curve": [
                {"travel_m": 0.0, "force_kn": 3.0},
                {"travel_m": 1.2, "force_kn": 3.0},
            ],
            "max_travel_m": 1.2,
            "energy_capacity_j": 5000.0,
            "max_included_angle_deg": 120.0,
        },
    }
    eq.update(kw)
    return eq


def twin_side_anchors(xs=(0.0, 0.5, 1.0), y=1.5, z=3.4, rated=12.0):
    """左右两列（y=±）头顶锚点，便于两腿分挂产生夹角。"""
    anchors = [
        {"id": f"a{k}", "position": v(x, -y, z),
         "rated_load_kn": rated, "max_users": 2}
        for k, x in enumerate(xs)]
    anchors += [
        {"id": f"b{k}", "position": v(x, y, z),
         "rated_load_kn": rated, "max_users": 2}
        for k, x in enumerate(xs)]
    return anchors


def twin_passable_payload():
    """双腿分挂左右锚点的可通行基准（自动序列）。"""
    return {
        "route": {
            "walk_polyline": [v(0, 0, 0), v(1, 0, 0)],
            "drop_edges": [], "obstacles": [],
            "anchors": twin_side_anchors(),
            "supports": [], "spans": [], "shuttles": [],
        },
        "persons": [base_person(equipment_id="eqt")],
        "equipment": [twin_equipment()],
        "params": {"station_spacing_m": 0.5},
    }


def twin_angle_payload(max_angle=40.0):
    """允许夹角收紧到 40°：实际夹角约 74°，夹角越限。"""
    p = twin_passable_payload()
    p["equipment"] = [twin_equipment()]
    p["equipment"][0]["twin_leg"]["max_included_angle_deg"] = max_angle
    return p


def twin_energy_payload():
    """缓冲曲线能量容量不足：重人 + 短行程低力曲线 + 低位锚点。"""
    p = twin_passable_payload()
    p["route"]["anchors"] = [
        {"id": "A1", "position": v(0, 0, 0.4), "rated_load_kn": 12.0,
         "max_users": 2},
        {"id": "A2", "position": v(0.3, 0, 0.4), "rated_load_kn": 12.0,
         "max_users": 2},
    ]
    p["persons"] = [base_person("p1", weight_kg=120.0, equipment_id="eqt")]
    eq = twin_equipment()
    eq["twin_leg"]["buffer_curve"] = [
        {"travel_m": 0.0, "force_kn": 2.0},
        {"travel_m": 0.2, "force_kn": 2.0}]
    eq["twin_leg"]["max_travel_m"] = 0.2
    eq["twin_leg"]["energy_capacity_j"] = 400.0
    p["equipment"] = [eq]
    return p


def twin_side_load_payload(limit=0.1):
    """连接器侧载限值收紧：分挂侧锚时水平分量超限。"""
    p = twin_passable_payload()
    eq = twin_equipment()
    for leg in eq["twin_leg"]["legs"]:
        leg["connector_side_load_limit_kn"] = limit
    p["equipment"] = [eq]
    return p


def twin_unequal_legs_payload(la=2.5, lb=1.4):
    """两腿不等长：短腿（B）够不到左右侧锚点 → 腿长不足。"""
    p = twin_passable_payload()
    eq = twin_equipment()
    eq["twin_leg"]["legs"][0]["leg_length_m"] = la
    eq["twin_leg"]["legs"][1]["leg_length_m"] = lb
    p["equipment"] = [eq]
    return p


def twin_manual_payload():
    """人工次序：A 钩（LA 腿）挂左列 a0，B 钩（LB 腿）挂右列 b0。"""
    p = twin_passable_payload()
    p["hook_order"] = [
        {"person_id": "p1", "hook": "A", "action": "attach",
         "anchor": "a0", "station_index": 0, "leg": "LA"},
        {"person_id": "p1", "hook": "B", "action": "attach",
         "anchor": "b0", "station_index": 0, "leg": "LB"},
    ]
    return p


def twin_flex_payload():
    """双腿各挂一条滑梭（同跨不同滑梭）的柔性跨段场景。"""
    route = flex_span_route(
        shuttles=[
            {"id": "T1", "span_id": "H1", "connector_reach_m": 0.3,
             "max_users": 1, "can_pass": True},
            {"id": "T2", "span_id": "H1", "connector_reach_m": 0.3,
             "max_users": 1, "can_pass": True},
        ])
    return {
        "route": route,
        "persons": [base_person(equipment_id="eqt", d_ring_height_m=1.4)],
        "equipment": [{
            "id": "eqt",
            "lanyard_length_m": 1.6, "elongation_m": 0.2,
            "buffer_travel_m": 1.0, "sharp_edge_rating": 1,
            "connector_reach_m": 0.3, "max_arrest_force_kn": 6.0,
            "twin_leg": {
                "legs": [
                    {"id": "LA", "hook": "A", "leg_length_m": 1.6,
                     "axial_stiffness_kn": 2000.0,
                     "connector_side_load_limit_kn": 16.0},
                    {"id": "LB", "hook": "B", "leg_length_m": 1.6,
                     "axial_stiffness_kn": 2000.0,
                     "connector_side_load_limit_kn": 16.0},
                ],
                "buffer_curve": [
                    {"travel_m": 0.0, "force_kn": 3.0},
                    {"travel_m": 1.2, "force_kn": 3.0},
                ],
                "max_travel_m": 1.2, "energy_capacity_j": 5000.0,
                "max_included_angle_deg": 120.0,
            }}],
        "params": {"station_spacing_m": 0.5},
    }


def _mixed_anchor_shuttle_equipment(rated_anchor=4.5):
    """固定锚 AX + 滑梭 T1 混挂：固定锚额定 rated_anchor kN（介于
    单腿 3.0 与虚增 6.0 之间），曲线恒力 3 kN。"""
    return {
        "id": "eqt", "lanyard_length_m": 1.6, "elongation_m": 0.2,
        "buffer_travel_m": 1.0, "sharp_edge_rating": 1,
        "connector_reach_m": 0.3, "max_arrest_force_kn": 6.0,
        "twin_leg": {
            "legs": [
                {"id": "LA", "hook": "A", "leg_length_m": 1.6,
                 "axial_stiffness_kn": 20000.0,
                 "connector_side_load_limit_kn": 16.0},
                {"id": "LB", "hook": "B", "leg_length_m": 1.6,
                 "axial_stiffness_kn": 20000.0,
                 "connector_side_load_limit_kn": 16.0},
            ],
            "buffer_curve": [
                {"travel_m": 0.0, "force_kn": 3.0},
                {"travel_m": 1.2, "force_kn": 3.0},
            ],
            "max_travel_m": 1.2, "energy_capacity_j": 5000.0,
            "max_included_angle_deg": 120.0,
        }}


def twin_mixed_anchor_shuttle_payload(rated_anchor=4.5):
    """固定锚腿 + 滑梭腿混挂（弱跨段低预张力/低 EA，单人加载下钢索下挠
    约 0.25 m）：下挠后滑梭腿在峰值时刻松弛，固定锚腿独承 3.0 kN。

    固定锚额定 4.5 kN（介于真实 3.0 与静态+动态虚增的 6.0 之间）：若同一
    固定锚腿负荷在静态、动态两阶段被重复登记，会误判 anchor_overload。
    """
    supports = [
        {"id": "S0", "position": v(0, 0, 2.6), "rated_load_kn": 50.0},
        {"id": "S1", "position": v(10, 0, 2.6), "rated_load_kn": 50.0},
    ]
    route = flex_span_route(
        supports=supports, span_kw={
            "pretension_kn": 1.0, "axial_stiffness_kn": 800.0,
            "max_sag_m": 3.0},
        anchors=[
            {"id": "AX", "position": v(0, 0, 2.6),
             "rated_load_kn": rated_anchor, "max_users": 2},
            {"id": "AX2", "position": v(1, 0, 2.6),
             "rated_load_kn": rated_anchor, "max_users": 2},
        ],
        shuttles=[
            {"id": "T1", "span_id": "H1", "connector_reach_m": 0.3,
             "max_users": 1, "can_pass": True},
            {"id": "T2", "span_id": "H1", "connector_reach_m": 0.3,
             "max_users": 1, "can_pass": True},
        ])
    p = {
        "route": route,
        "persons": [base_person(equipment_id="eqt", d_ring_height_m=1.4)],
        "equipment": [_mixed_anchor_shuttle_equipment(rated_anchor)],
        "params": {"station_spacing_m": 0.5},
        # 人工：A 钩(LA) 挂固定锚 AX，B 钩(LB) 挂滑梭 T1
        "hook_order": [
            {"person_id": "p1", "hook": "A", "action": "attach",
             "anchor": "AX", "station_index": 0, "leg": "LA"},
            {"person_id": "p1", "hook": "B", "action": "attach",
             "shuttle": "T1", "station_index": 0, "leg": "LB"},
        ],
    }
    # 行走线限制在 0~0.5 m：固定锚 AX 与滑梭 T1 在两站均可达，
    # 不需换钩即可出现下挠后滑梭腿松弛、固定锚腿独承的站
    p["route"]["walk_polyline"] = [v(0, 0, 0), v(0.5, 0, 0)]
    return p


def twin_buffer_capacity_payload():
    """缓冲曲线实际吸收 ≈4635 J（< 5000 J 容量），但腿部弹性能 ≈2083 J，
    二者合计 ≈6718 J。容量只应与曲线吸收比较，不得合计后误判 buffer_energy。
    """
    return {
        "route": {
            "walk_polyline": [v(0, 0, 0), v(0.5, 0, 0)],
            "drop_edges": [], "obstacles": [],
            "anchors": [
                {"id": "A1", "position": v(-0.01, 0, -2.0),
                 "rated_load_kn": 50.0, "max_users": 2},
                {"id": "A2", "position": v(0.01, 0, -2.0),
                 "rated_load_kn": 50.0, "max_users": 2},
            ],
            "supports": [], "spans": [], "shuttles": [],
        },
        "persons": [base_person(equipment_id="eqt", d_ring_height_m=1.4)],
        "equipment": [{
            "id": "eqt", "lanyard_length_m": 3.4, "elongation_m": 0.2,
            "buffer_travel_m": 1.0, "sharp_edge_rating": 3,
            "connector_reach_m": 0.3, "max_arrest_force_kn": 8.0,
            "twin_leg": {
                "legs": [
                    {"id": "LA", "hook": "A", "leg_length_m": 3.4,
                     "axial_stiffness_kn": 3.0,
                     "connector_side_load_limit_kn": 16.0},
                    {"id": "LB", "hook": "B", "leg_length_m": 3.4,
                     "axial_stiffness_kn": 3.0,
                     "connector_side_load_limit_kn": 16.0},
                ],
                "buffer_curve": [
                    {"travel_m": 0.0, "force_kn": 5.0},
                    {"travel_m": 2.0, "force_kn": 5.0},
                ],
                "max_travel_m": 2.0, "energy_capacity_j": 5000.0,
                "max_included_angle_deg": 180.0,
            }}],
        "params": {"station_spacing_m": 0.5},
    }


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
