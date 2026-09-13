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
