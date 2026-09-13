"""生命线通行核算引擎：离散路线 → 生成挂接序列 → 逐站坠落核算 → 汇总失败。"""
from __future__ import annotations

from ..models import (AnalysisResult, CheckFailure, Person, PlanPayload,
                      ProfilePoint, Vec3)
from . import calc, geometry as g
from .sequence import build_sequence, replay_sequence

# 同站失败排序优先级（数值小者优先，作为“最先失败”）
_CHECK_PRIORITY = {
    "hook_chain": 0,
    "sharp_edge": 1,
    "anchor_direction": 2,
    "anchor_overload": 3,
    "clearance": 4,
    "sweep": 5,
}


def analyze(payload: PlanPayload) -> AnalysisResult:
    route = payload.route
    params = payload.params
    anchors = {a.id: a for a in route.anchors}
    equipment = {e.id: e for e in payload.equipment}
    person_idx = {p.id: k for k, p in enumerate(payload.persons)}

    stations = g.discretize_polyline(
        [p.as_tuple() for p in route.walk_polyline], params.station_spacing_m)
    n = len(stations)

    def pos(i: int) -> Vec3:
        s = stations[i]
        return Vec3(x=s[0], y=s[1], z=s[2])

    failures: list[CheckFailure] = []

    # ---- 1. 挂接/换钩/解钩序列（多人共享锚点容量） ----------------------
    # 该版给出人工挂接动作次序的人员按次序回放校验，其余人员自动生成
    manual: dict[str, list] = {}
    for act in payload.hook_order:
        manual.setdefault(act.person_id, []).append(act)

    capacity: list[dict[str, int]] = [dict() for _ in range(n)]
    sequences = []
    for person in payload.persons:
        eq = equipment[person.equipment_id]
        if person.id in manual:
            seq = replay_sequence(person, eq, anchors, stations, capacity,
                                  manual[person.id])
        else:
            seq = build_sequence(person, eq, anchors, stations, capacity)
        sequences.append(seq)
        failures.extend(seq.failures)

    # ---- 2. 逐站坠落核算 ----------------------------------------------
    # per-station/per-anchor 合力累计（多人共用锚点过载）
    anchor_load: list[dict[str, float]] = [dict() for _ in range(n)]
    anchor_users: list[dict[str, list[str]]] = [dict() for _ in range(n)]

    for seq in sequences:
        person = next(p for p in payload.persons if p.id == seq.person_id)
        eq = equipment[person.equipment_id]
        L = eq.lanyard_length_m
        for i, state in enumerate(seq.states):
            if state is None:
                break
            for aid in sorted({a for a in state if a}):
                anchor = anchors[aid]
                fc = calc.fall_calc(stations[i], person, eq, anchor,
                                    route, params)
                comp = {
                    "anchor": aid,
                    "free_fall_m": fc.free_fall_m,
                    "total_fall_m": fc.total_fall_m,
                    "deployed_length_m": fc.deployed_length_m,
                    "required_clearance_m": fc.required_clearance_m,
                    "available_clearance_m": fc.available_clearance_m,
                    "margin_m": fc.margin_m,
                    "arrest_force_kn": fc.arrest_force_kn,
                    "horizontal_offset_m": fc.horizontal_offset_m,
                    "swing_radius_m": fc.swing_radius_m,
                }
                # 锐边适配
                for e in route.drop_edges:
                    if (g.dist2d(stations[i], e.point.as_tuple()) <= L
                            and e.sharpness_class > eq.sharp_edge_rating):
                        failures.append(CheckFailure(
                            station_index=i, position=pos(i),
                            person_id=person.id, action="traverse",
                            check="sharp_edge",
                            message=(f"落差边缘 {e.id} 锐边等级 "
                                     f"{e.sharpness_class} 超过装备适配等级 "
                                     f"{eq.sharp_edge_rating}"),
                            components={**comp, "edge_id": e.id,
                                        "edge_sharpness_class": e.sharpness_class,
                                        "equipment_sharp_edge_rating":
                                            eq.sharp_edge_rating}))
                # 锚点允许受力方向
                if (fc.force_angle_deg is not None
                        and fc.force_angle_deg > anchor.allowed_half_angle_deg):
                    failures.append(CheckFailure(
                        station_index=i, position=pos(i),
                        person_id=person.id, action="traverse",
                        check="anchor_direction",
                        message=(f"锚点 {aid} 受力方向 {fc.force_angle_deg}° "
                                 f"超出允许锥半角 {anchor.allowed_half_angle_deg}°"),
                        components={**comp,
                                    "force_angle_deg": fc.force_angle_deg,
                                    "allowed_half_angle_deg":
                                        anchor.allowed_half_angle_deg}))
                # 总净空余量
                if fc.margin_m is not None and fc.margin_m < 0:
                    failures.append(CheckFailure(
                        station_index=i, position=pos(i),
                        person_id=person.id, action="traverse",
                        check="clearance",
                        message=(f"总净空余量为负（{fc.margin_m} m）：缓冲包完全"
                                 f"展开后将撞上下层障碍物/楼面"),
                        components=comp))
                # 摆坠扫掠体
                hit = calc.sweep_hit_obstacle(stations[i], person, eq, anchor,
                                              route, params)
                if hit is not None:
                    failures.append(CheckFailure(
                        station_index=i, position=pos(i),
                        person_id=person.id, action="traverse",
                        check="sweep",
                        message=f"摆坠扫掠体与障碍物 {hit} 相交",
                        components={**comp, "obstacle_id": hit}))
                # 合力累计
                anchor_load[i][aid] = anchor_load[i].get(aid, 0.0) \
                    + fc.arrest_force_kn
                anchor_users[i].setdefault(aid, [])
                if person.id not in anchor_users[i][aid]:
                    anchor_users[i][aid].append(person.id)

    # ---- 3. 锚点过载：每个在用锚点核对合力与额定载荷 --------------------
    for i in range(n):
        for aid, total in anchor_load[i].items():
            users = anchor_users[i][aid]
            rated = anchors[aid].rated_load_kn
            if users and total > rated + 1e-9:
                if len(users) > 1:
                    msg = (f"共用锚点 {aid} 过载：{len(users)} 人合力 "
                           f"{round(total, 3)} kN 超过额定载荷 {rated} kN")
                else:
                    msg = (f"锚点 {aid} 过载：止坠合力 {round(total, 3)} kN "
                           f"超过额定载荷 {rated} kN")
                failures.append(CheckFailure(
                    station_index=i, position=pos(i),
                    person_id=None, action="traverse",
                    check="anchor_overload",
                    message=msg,
                    components={
                        "anchor": aid,
                        "user_count": len(users),
                        "users": ",".join(sorted(users)),
                        "combined_force_kn": round(total, 4),
                        "rated_load_kn": rated,
                    }))

    # ---- 4. 剖面标注（以首名人员的控制性锚点为准） -----------------------
    profile: list[ProfilePoint] = []
    seq0 = sequences[0]
    person0 = payload.persons[0]
    eq0 = equipment[person0.equipment_id]
    for i in range(n):
        state = seq0.states[i] if i < len(seq0.states) else None
        pp = ProfilePoint(station_index=i, position=pos(i),
                          walk_z=stations[i][2])
        if state is not None:
            best = None
            for aid in sorted({a for a in state if a}):
                fc = calc.fall_calc(stations[i], person0, eq0, anchors[aid],
                                    route, params)
                if best is None or fc.required_clearance_m > best[1].required_clearance_m:
                    best = (aid, fc)
            if best is not None:
                aid, fc = best
                pp.controlling_anchor = aid
                pp.anchor_z = anchors[aid].position.z
                pp.free_fall_m = fc.free_fall_m
                pp.required_clearance_m = fc.required_clearance_m
                pp.clearance_floor_z = round(
                    stations[i][2] - fc.required_clearance_m, 4)
                pp.available_clearance_m = fc.available_clearance_m
                pp.margin_m = fc.margin_m
        profile.append(pp)

    # ---- 5. 汇总：最先失败 ----------------------------------------------
    failures.sort(key=lambda f: (f.station_index,
                                 _CHECK_PRIORITY.get(f.check, 99),
                                 person_idx.get(f.person_id or "", 999)))
    return AnalysisResult(
        passable=not failures,
        first_failure=failures[0] if failures else None,
        failures=failures,
        sequence=[ev for seq in sequences for ev in seq.events],
        profile=profile,
        station_count=n,
    )
