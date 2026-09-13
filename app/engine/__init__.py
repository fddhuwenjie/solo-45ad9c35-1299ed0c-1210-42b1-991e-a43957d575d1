"""生命线通行核算引擎：离散路线 → 生成挂接序列 → 逐站坠落核算 → 汇总失败。

固定点锚按点锚模型核算；柔性跨段（临时水平生命线）按悬索模型：
- 挂接序列由 sequence.ReachProvider 统一处理点锚与滑梭；
- 坠落后按“同 bay 同跨”的单人 / 多人组合枚举，悬索弹性迭代求解，
  给出动态下挠、总坠距、端座与中间支座反力；
- 连续性断开、滑梭卡支座、重复占用、参数缺失、求解不收敛 → open item，
  conclusive=False 且不作可通行结论；
- 净空不足、下挠越限、反力越限 → failure，返回最早失效位置与动作。
"""
from __future__ import annotations

import math

from ..models import (AnalysisResult, CableResult, CheckFailure, OpenItem,
                      Person, PlanPayload, ProfilePoint, Vec3)
from . import calc, cable, geometry as g
from .sequence import (ReachProvider, Target, build_sequence,
                       replay_sequence)
from .shuttle_path import SpanPath, unloaded_bay_sags

# 同站失败排序优先级（数值小者优先，作为“最先失败”）
_CHECK_PRIORITY = {
    "hook_chain": 0,
    "sharp_edge": 1,
    "anchor_direction": 2,
    "anchor_overload": 3,
    "support_overload": 3,
    "clearance": 4,
    "sag_limit": 4,
    "sweep": 5,
}


def analyze(payload: PlanPayload) -> AnalysisResult:
    route = payload.route
    params = payload.params
    anchors = {a.id: a for a in route.anchors}
    supports = {s.id: s for s in route.supports}
    spans = {s.id: s for s in route.spans}
    shuttles = {s.id: s for s in route.shuttles}
    equipment = {e.id: e for e in payload.equipment}
    person_idx = {p.id: k for k, p in enumerate(payload.persons)}
    person_map = {p.id: p for p in payload.persons}
    grav = params.gravity

    stations = g.discretize_polyline(
        [p.as_tuple() for p in route.walk_polyline], params.station_spacing_m)
    n = len(stations)

    def pos(i: int) -> Vec3:
        s = stations[i]
        return Vec3(x=s[0], y=s[1], z=s[2])

    failures: list[CheckFailure] = []
    open_items: list[OpenItem] = []

    # ---- 0. 柔性跨段几何与每站滑梭站位 --------------------------------
    span_paths: dict[str, SpanPath] = {}
    for sp in route.spans:
        sags0 = unloaded_bay_sags(sp, supports, grav)
        span_paths[sp.id] = SpanPath(sp, supports, stations, sags0)

    # 缺失参数的跨段：只要该跨被任一有效序列使用，即记 open item（后述）
    missing_by_span = {sp.id: sp.missing_params() for sp in route.spans}

    # ---- 1. 可达性 / 共用容量 / 滑梭跨支座 -----------------------------
    # 点锚：每目标每站独立可达；滑梭：站位有效且 D 环到该站滑梭位置可达
    def shuttle_reachable(sid: str, i: int, person: Person, eq) -> bool:
        t = span_paths[shuttles[sid].span_id].table[i]
        if not t.valid:
            return False
        d = calc.d_ring_pos(stations[i], person)
        reach = eq.lanyard_length_m + shuttles[sid].connector_reach_m
        return g.dist3(d, t.point) <= reach + 1e-9

    # 容量计数：点锚/滑梭按“占用钩数（人去重）”记录，用于他人容量上限；
    # 同一人 A/B 钩重复挂同一滑梭属 duplicate_occupancy（双钩不独立，不下结论）。
    anchor_count: list[dict[str, set[str]]] = [dict() for _ in range(n)]
    shuttle_count: list[dict[str, set[str]]] = [dict() for _ in range(n)]
    span_users: list[dict[str, set[str]]] = [dict() for _ in range(n)]

    def make_provider(person: Person) -> ReachProvider:
        eq = equipment[person.equipment_id]
        targets: list[Target] = [("anchor", a.id) for a in route.anchors] \
            + [("shuttle", s.id) for s in route.shuttles]

        def reachable(i: int, t: Target) -> bool:
            kind, tid = t
            if kind == "anchor":
                return calc.reachable(stations[i], person, eq, anchors[tid])
            return shuttle_reachable(tid, i, person, eq)

        # 前向连续可达末站
        last_cache: dict[Target, list[int]] = {}

        def last_reach(t: Target, i: int) -> int:
            if t not in last_cache:
                arr = [-1] * n
                if reachable(n - 1, t):
                    arr[n - 1] = n - 1
                for k in range(n - 2, -1, -1):
                    arr[k] = k if not reachable(k + 1, t) else arr[k + 1] \
                        if reachable(k, t) else -1
                last_cache[t] = arr
            return last_cache[t][i]

        def attach_block(i: int, t: Target, other: Target | None):
            """返回 None 表示可挂；否则 (code, components)。

            点锚：他人占用达上限 → capacity_full（旧语义失效）。
            滑梭：本人另一钩已挂同一滑梭 → duplicate_occupancy；
            滑梭或跨段的他人占用达上限 → duplicate_occupancy（不下结论）。
            """
            kind, tid = t
            if kind == "anchor":
                others = anchor_count[i].get(tid, set()) - {person.id}
                if len(others) >= anchors[tid].max_users:
                    return "capacity_full", {
                        "target_anchor": tid,
                        "max_users": anchors[tid].max_users,
                        "occupants": ",".join(sorted(others))}
                return None
            sh = shuttles[tid]
            spid = sh.span_id
            if other == t:
                return "duplicate_occupancy", {
                    "target_shuttle": tid, "span": spid,
                    "reason": "same_person_other_hook"}
            oth_s = shuttle_count[i].get(tid, set()) - {person.id}
            if len(oth_s) >= sh.max_users:
                return "duplicate_occupancy", {
                    "target_shuttle": tid, "span": spid,
                    "max_users": sh.max_users,
                    "occupants": ",".join(sorted(oth_s))}
            oth_span = span_users[i].get(spid, set()) - {person.id}
            if len(oth_span) >= spans[spid].max_users:
                return "duplicate_occupancy", {
                    "target_shuttle": tid, "span": spid,
                    "span_max_users": spans[spid].max_users,
                    "occupants": ",".join(sorted(oth_span))}
            return None

        def occupy(i: int, t: Target, hook: str) -> None:
            kind, tid = t
            if kind == "anchor":
                anchor_count[i].setdefault(tid, set()).add(person.id)
            else:
                shuttle_count[i].setdefault(tid, set()).add(person.id)
                sp = shuttles[tid].span_id
                span_users[i].setdefault(sp, set()).add(person.id)

        def target_pos(i: int, t: Target) -> tuple[float, float, float]:
            if t[0] == "anchor":
                return anchors[t[1]].position.as_tuple()
            return span_paths[shuttles[t[1]].span_id].table[i].point

        def crosses_blocked(sid: str, i0: int, i1: int) -> int | None:
            sh = shuttles[sid]
            path = span_paths[shuttles[sid].span_id]
            k = path.crosses_intermediate_support(i0, i1)
            if k is None:
                return None
            span = spans[sh.span_id]
            if not span.shuttle_pass or not sh.can_pass:
                return k
            return None

        return ReachProvider(
            n=n, stations=stations, reachable=reachable,
            last_reach=last_reach, attach_block=attach_block, occupy=occupy,
            targets=targets, pos_fn=pos, target_pos=target_pos,
            anchor_max_users=lambda aid: anchors[aid].max_users,
            shuttle_span=lambda sid: shuttles[sid].span_id,
            span_supports=lambda spid: spans[spid].supports,
            crosses_blocked=crosses_blocked)

    # ---- 2. 挂接序列（人工次序回放 / 自动生成） -------------------------
    manual: dict[str, list] = {}
    for act in payload.hook_order:
        manual.setdefault(act.person_id, []).append(act)

    sequences = []
    for person in payload.persons:
        eq = equipment[person.equipment_id]
        prov = make_provider(person)
        if person.id in manual:
            seq = replay_sequence(person, eq, prov, manual[person.id])
        else:
            seq = build_sequence(person, eq, prov)
        sequences.append(seq)
        failures.extend(seq.failures)
        open_items.extend(seq.open_items)
    seq_by_person = {s.person_id: s for s in sequences}

    # ---- 3. 参数缺失：被使用到的跨段才报 -------------------------------
    used_spans: set[str] = set()
    used_shuttles: set[str] = set()
    for seq in sequences:
        for state in seq.states:
            if state is None:
                continue
            for t in state:
                if t is not None and t[0] == "shuttle":
                    used_shuttles.add(t[1])
                    used_spans.add(shuttles[t[1]].span_id)
    for spid in sorted(used_spans):
        miss = missing_by_span[spid]
        if miss:
            open_items.append(OpenItem(
                code="missing_params",
                message=f"柔性跨段 {spid} 缺少必要参数 {', '.join(miss)}，"
                        f"无法进行悬索求解，本版不下结论",
                components={"span": spid,
                            "missing": ",".join(miss)}))

    # ---- 4. 点锚坠落核算（保持旧版分量与判定） --------------------------
    point_load: list[dict[str, float]] = [dict() for _ in range(n)]
    point_users: list[dict[str, list[str]]] = [dict() for _ in range(n)]

    def d_ring(i: int, person: Person):
        return calc.d_ring_pos(stations[i], person)

    for seq in sequences:
        person = person_map[seq.person_id]
        eq = equipment[person.equipment_id]
        L = eq.lanyard_length_m
        for i, state in enumerate(seq.states):
            if state is None:
                break
            anchor_targets = sorted({t for t in state
                                     if t and t[0] == "anchor"})
            for (_k, aid) in anchor_targets:
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
                if fc.margin_m is not None and fc.margin_m < 0:
                    failures.append(CheckFailure(
                        station_index=i, position=pos(i),
                        person_id=person.id, action="traverse",
                        check="clearance",
                        message=(f"总净空余量为负（{fc.margin_m} m）：缓冲包完全"
                                 f"展开后将撞上下层障碍物/楼面"),
                        components=comp))
                hit = calc.sweep_hit_obstacle(stations[i], person, eq, anchor,
                                              route, params)
                if hit is not None:
                    failures.append(CheckFailure(
                        station_index=i, position=pos(i),
                        person_id=person.id, action="traverse",
                        check="sweep",
                        message=f"摆坠扫掠体与障碍物 {hit} 相交",
                        components={**comp, "obstacle_id": hit}))
                point_load[i][aid] = point_load[i].get(aid, 0.0) \
                    + fc.arrest_force_kn
                point_users[i].setdefault(aid, [])
                if person.id not in point_users[i][aid]:
                    point_users[i][aid].append(person.id)

    # ---- 5. 柔性跨段坠落组合：按站/跨/bay 枚举同 bay 子集 ---------------
    cable_results: list[CableResult] = []
    conservative_spans: set[str] = set()

    def shuttle_state_at(i, person):
        """该人在站 i 挂接的滑梭目标（取唯一；双钩同梭按一个计）。"""
        state = seq_by_person[person.id].states[i] \
            if i < len(seq_by_person[person.id].states) else None
        if state is None:
            return None
        sh = [t[1] for t in state if t and t[0] == "shuttle"]
        return sh[0] if sh else None

    def shuttle_fall_components(i: int, person: Person, eq, sid: str,
                                sag: float):
        """给定该人坠落点动态下挠，返回 (锚点静标高, FFD, 总坠距, 所需净空,
        可用净空, 余量, 止坠力)。止坠力由调用方在组合层迭代。"""
        t = span_paths[shuttles[sid].span_id].table[i]
        anchor_z = t.point[2]
        dz = stations[i][2] + person.d_ring_height_m
        ffd = calc.free_fall_with_sag(eq, dz, anchor_z, sag)
        total = ffd + eq.elongation_m + eq.buffer_travel_m \
            + params.harness_stretch_m
        required = total + params.safety_margin_m
        surfaces = []
        top = calc.obstacle_top_below(stations[i], person.body_radius_m,
                                      route.obstacles)
        if top is not None:
            surfaces.append(top)
        low = calc.lower_level_below(stations[i], eq.lanyard_length_m,
                                     route.drop_edges)
        if low is not None:
            surfaces.append(low)
        if surfaces:
            avail = stations[i][2] - max(surfaces)
            margin = avail - required
        else:
            avail = margin = None
        return {"anchor_z": anchor_z, "ffd": ffd, "total": total,
                "required": required, "avail": avail, "margin": margin}

    def solve_combo(i: int, span_id: str, bay_index: int,
                    members: list[tuple[str, str]]):
        """同 bay 坠落组合的悬索迭代求解，封装于 _solve_iterated。"""
        return _solve_iterated(
            i, span_id, bay_index, members, person_map, equipment,
            params, stations, span_paths, shuttles, spans,
            payload.conservative_bounds.enabled, open_items, pos)

    for i in range(n):
        # 站 i：span -> bay -> [(pid, sid)]
        groups: dict[str, dict[int, list[tuple[str, str]]]] = {}
        for person in payload.persons:
            sid = shuttle_state_at(i, person)
            if sid is None:
                continue
            spid = shuttles[sid].span_id
            t = span_paths[spid].table[i]
            if not t.valid:
                continue
            groups.setdefault(spid, {}).setdefault(t.bay, []).append(
                (person.id, sid))

        for spid in sorted(groups):
            sp = spans[spid]
            path = span_paths[spid]
            # 参数缺失的跨段已记 open item，不下结论、不做悬索求解
            if missing_by_span[spid]:
                continue
            for bay_index in sorted(groups[spid]):
                members = sorted(groups[spid][bay_index])
                # 单人组合（逐人净空/锐边/摆坠）+ 同 bay 全员同时坠落组合
                # （挠度限值与支座反力的保守包络）
                combos = [[m] for m in members]
                if len(members) > 1:
                    combos.append(members)
                full_idx = len(combos) - 1
                full_result = None
                solo_results: dict[str, tuple] = {}
                for ci, combo in enumerate(combos):
                    sol, force_map = solve_combo(i, spid, bay_index, combo)
                    if sol is None:
                        continue
                    if sol.conservative:
                        conservative_spans.add(spid)
                    if len(combo) > 1:
                        full_result = (sol, combo, force_map)
                    else:
                        solo_results[combo[0][0]] = (sol, combo, force_map)
                    _evaluate_combo_failures(
                        i, spid, bay_index, combo, sol, force_map, person_map,
                        equipment, route, params, stations, pos,
                        shuttle_fall_components, span_paths, shuttles,
                        spans, supports, failures,
                        structural=(ci == full_idx))
                # 结果记录：多人 bay 记全员组合（结构包络）+ 每人单人组合；
                # 单人 bay 仅记一条单人结果
                if full_result is None:
                    record = [solo_results[pid]
                              for pid, _sid in members if pid in solo_results]
                else:
                    record = [full_result] + [
                        solo_results[pid]
                        for pid, _sid in members if pid in solo_results]
                for sol, combo, force_map in record:
                    _append_cable_result(
                        cable_results, i, spid, bay_index, combo, sol,
                        force_map, spans, supports, span_paths, grav,
                        person_map, equipment, params, stations)

    # ---- 6. 点锚合力过载 ----------------------------------------------
    for i in range(n):
        for aid, total in point_load[i].items():
            users = point_users[i][aid]
            rated = anchors[aid].rated_load_kn
            if users and total > rated + 1e-9:
                if len(users) > 1:
                    msg = (f"共用锚点 {aid} 过载：{len(users)} 人合力 "
                           f"{round(total, 3)} kN 超过额定载荷 {rated} kN")
                else:
                    msg = (f"锚点 {aid} 过载：止坠合力 {round(total, 3)} kN "
                           f"超过额定载荷 {rated} kN")
                failures.append(CheckFailure(
                    station_index=i, position=pos(i), person_id=None,
                    action="traverse", check="anchor_overload",
                    message=msg,
                    components={"anchor": aid, "user_count": len(users),
                                "users": ",".join(sorted(users)),
                                "combined_force_kn": round(total, 4),
                                "rated_load_kn": rated}))

    # ---- 7. 剖面标注（首名人员；点锚取控制锚，滑梭取悬索分量） ----------
    profile: list[ProfilePoint] = []
    p0 = payload.persons[0]
    eq0 = equipment[p0.equipment_id]
    # 站 i -> 该站首人滑梭的已求悬索分量：优先取其单人结果（个人净空），
    # 无单人结果时退回结构包络组合。
    cable_by_station_span: dict[tuple[int, str], CableResult] = {}
    for cr in cable_results:
        key = (cr.station_index, cr.span_id)
        if key not in cable_by_station_span:
            cable_by_station_span[key] = cr
        if cr.falling_persons == [p0.id]:
            cable_by_station_span[key] = cr
    for i in range(n):
        pp = ProfilePoint(station_index=i, position=pos(i),
                          walk_z=stations[i][2])
        state = seq_by_person[p0.id].states[i] \
            if i < len(seq_by_person[p0.id].states) else None
        if state is None:
            profile.append(pp)
            continue
        anchor_ids = sorted({t[1] for t in state if t and t[0] == "anchor"})
        shuttle_ids = sorted({t[1] for t in state if t and t[0] == "shuttle"})
        if anchor_ids:
            best = None
            for aid in anchor_ids:
                fc = calc.fall_calc(stations[i], p0, eq0, anchors[aid],
                                    route, params)
                if best is None or fc.required_clearance_m > best[1].required_clearance_m:
                    best = (aid, fc)
            aid, fc = best
            pp.controlling_anchor = aid
            pp.anchor_z = anchors[aid].position.z
            pp.free_fall_m = fc.free_fall_m
            pp.required_clearance_m = fc.required_clearance_m
            pp.total_fall_m = fc.total_fall_m
            pp.clearance_floor_z = round(
                stations[i][2] - fc.required_clearance_m, 4)
            pp.available_clearance_m = fc.available_clearance_m
            pp.margin_m = fc.margin_m
        if shuttle_ids:
            sid0 = shuttle_ids[0]
            spid = shuttles[sid0].span_id
            cr = cable_by_station_span.get((i, spid))
            t = span_paths[spid].table[i]
            if cr is not None and p0.id in cr.falling_persons:
                # 剖面取保守的 bay 最大动态下挠
                sag = cr.sag_m
                comp = shuttle_fall_components(i, p0, eq0, sid0, sag)
                pp.controlling_shuttle = sid0
                pp.span_id = spid
                pp.anchor_z = round(t.point[2], 4)
                pp.cable_sag_m = round(sag, 4)
                pp.free_fall_m = round(comp["ffd"], 4)
                pp.total_fall_m = round(comp["total"], 4)
                pp.required_clearance_m = round(comp["required"], 4)
                pp.clearance_floor_z = round(
                    stations[i][2] - comp["required"], 4)
                pp.available_clearance_m = None if comp["avail"] is None \
                    else round(comp["avail"], 4)
                pp.margin_m = None if comp["margin"] is None \
                    else round(comp["margin"], 4)
                pp.cable_tension_kn = round(cr.horizontal_tension_kn, 4)
            else:
                pp.controlling_shuttle = sid0
                pp.span_id = spid
                pp.anchor_z = round(t.point[2], 4)
        profile.append(pp)

    # ---- 8. 汇总 --------------------------------------------------------
    failures.sort(key=lambda f: (f.station_index,
                                 _CHECK_PRIORITY.get(f.check, 99),
                                 person_idx.get(f.person_id or "", 999)))
    open_items.sort(key=lambda o: ((o.station_index if o.station_index is not None else 10**9),
                                   o.code,
                                   person_idx.get(o.person_id or "", 999)))
    conclusive = not open_items
    return AnalysisResult(
        conclusive=conclusive,
        passable=conclusive and not failures,
        first_failure=failures[0] if failures else None,
        failures=failures,
        first_open_item=open_items[0] if open_items else None,
        open_items=open_items,
        sequence=[ev for seq in sequences for ev in seq.events],
        profile=profile,
        cable_results=cable_results,
        conservative_used=sorted(conservative_spans),
        station_count=n,
    )


# ---------------------------------------------------------------- 悬索组合求解

def _arrest_force(person, eq, ffd, params):
    return calc.arrest_force_kn(person, eq, ffd, params)


def _solve_iterated(i, span_id, bay_index, combo, person_map, equipment,
                    params, stations, span_paths, shuttles, spans,
                    allow_conservative, open_items, pos_fn):
    """止坠力 ↔ 下挠定点迭代（各成员在自身坠落点的下挠决定 FFD）。

    返回 (BaySolution|None, {pid: force})；参数缺失由调用处预先过滤，
    不收敛且未授权保守边界时记 open item 并返回 None。
    """
    sp = spans[span_id]
    path = span_paths[span_id]
    bay = path.bays[bay_index]
    w = sp.line_density_kg_m * params.gravity / 1000.0

    # 各成员在本 bay 的分数、静挂点标高与 D 环标高
    info = {}
    for pid, sid in combo:
        person = person_map[pid]
        eq = equipment[person.equipment_id]
        t = path.table[i]
        info[pid] = (person, eq, t.frac, t.point[2],
                     stations[i][2] + person.d_ring_height_m)

    forces = {}
    for pid, _sid in combo:
        person, eq, frac, az, dz = info[pid]
        forces[pid] = _arrest_force(
            person, eq, calc.free_fall_distance(eq, dz, az), params)

    sol = None
    for _ in range(25):
        loads = [cable.BayLoad(pid, info[pid][2], forces[pid])
                 for pid, _sid in combo]
        sol = cable.solve_bay(
            L=bay.horiz, rise=bay.rise, loads=loads,
            h0=sp.pretension_kn, w=w, ea=sp.ea_kn(),
            allow_conservative=allow_conservative,
            max_iter=params.cable_max_iter, tol=params.cable_tol_kn)
        if not sol.converged:
            break
        new_forces = {}
        max_df = 0.0
        for k, (pid, _sid) in enumerate(combo):
            person, eq, frac, az, dz = info[pid]
            ffd = calc.free_fall_with_sag(eq, dz, az, sol.load_sags_m[k])
            f = _arrest_force(person, eq, ffd, params)
            new_forces[pid] = f
            max_df = max(max_df, abs(f - forces[pid]))
        forces = new_forces
        if max_df < 1e-4:
            break

    if not sol.converged:
        open_items.append(OpenItem(
            station_index=i, position=pos_fn(i),
            person_id=combo[0][0], action="traverse",
            code="solver_nonconvergence",
            message=f"柔性跨段 {span_id} bay {bay_index} 坠落组合 "
                    f"{[pid for pid, _ in combo]} 悬索迭代不收敛，"
                    f"本版不下结论"
                    + ("（已按授权采用保守边界）" if allow_conservative else ""),
            components={"span": span_id, "bay_index": bay_index,
                        "falling_persons": ",".join(pid for pid, _ in combo),
                        "conservative_allowed": int(allow_conservative)}))
        if not (allow_conservative and sol.conservative):
            return None, forces

    if sol.conservative:
        open_items.append(OpenItem(
            station_index=i, position=pos_fn(i),
            person_id=combo[0][0], action="traverse",
            code="solver_conservative_bounds",
            message=f"柔性跨段 {span_id} 弹性迭代不收敛，已按授权采用保守边界："
                    f"反力按刚性索（上界）、下挠按预张力几何索形（上界）。"
                    f"改跨或重新授权须另存修订并说明理由。",
            components={"span": span_id, "bay_index": bay_index,
                        "falling_persons": ",".join(pid for pid, _ in combo)}))
    return sol, forces


def _evaluate_combo_failures(i, span_id, bay_index, combo, sol, force_map,
                             person_map, equipment, route, params, stations,
                             pos_fn, shuttle_fall_components, span_paths,
                             shuttles, spans, supports, failures,
                             structural: bool):
    """对一个已求解组合：净空、锐边、摆坠；structural 组合另核挠度限值与
    端座/中间支座反力。"""
    sp = spans[span_id]
    path = span_paths[span_id]
    # ---- 各人净空 / 锐边 / 摆坠 ---------------------------------------
    for k, (pid, sid) in enumerate(combo):
        person = person_map[pid]
        eq = equipment[person.equipment_id]
        sag_k = sol.load_sags_m[k]
        c = shuttle_fall_components(i, person, eq, sid, sag_k)
        t = path.table[i]
        comp = {
            "span": span_id, "shuttle": sid, "bay_index": bay_index,
            "anchor_z": round(t.point[2], 4),
            "cable_sag_m": round(sag_k, 4),
            "free_fall_m": round(c["ffd"], 4),
            "total_fall_m": round(c["total"], 4),
            "required_clearance_m": round(c["required"], 4),
            "available_clearance_m": None if c["avail"] is None
                else round(c["avail"], 4),
            "margin_m": None if c["margin"] is None
                else round(c["margin"], 4),
            "arrest_force_kn": round(force_map[pid], 4),
            "horizontal_tension_kn": round(sol.h_kn, 4),
            "falling_persons": ",".join(p for p, _ in combo),
        }

        if c["margin"] is not None and c["margin"] < 0:
            failures.append(CheckFailure(
                station_index=i, position=pos_fn(i), person_id=pid,
                action="traverse", check="clearance",
                message=(f"柔性跨段 {span_id} 滑梭 {sid} 坠落总净空余量为负"
                         f"（{round(c['margin'], 3)} m）：钢索动态下挠 "
                         f"{round(sag_k, 3)} m 占用脚下净空，缓冲展开后"
                         f"撞上下层障碍物/楼面"),
                components=comp))
        # 锐边：滑梭的连接器/绳跨越落差边缘
        L = eq.lanyard_length_m
        for e in route.drop_edges:
            if (g.dist2d(stations[i], e.point.as_tuple()) <= L
                    and e.sharpness_class > eq.sharp_edge_rating):
                failures.append(CheckFailure(
                    station_index=i, position=pos_fn(i), person_id=pid,
                    action="traverse", check="sharp_edge",
                    message=(f"落差边缘 {e.id} 锐边等级 {e.sharpness_class} "
                             f"超过装备适配等级 {eq.sharp_edge_rating}"),
                    components={**comp, "edge_id": e.id,
                                "edge_sharpness_class": e.sharpness_class,
                                "equipment_sharp_edge_rating":
                                    eq.sharp_edge_rating}))
        # 摆坠扫掠：以滑梭动态位置为等效锚点
        anchor_like = _DynamicAnchor(sid, t, sag_k, path, bay_index)
        hit = calc.sweep_hit_obstacle(stations[i], person, eq, anchor_like,
                                      route, params)
        if hit is not None:
            failures.append(CheckFailure(
                station_index=i, position=pos_fn(i), person_id=pid,
                action="traverse", check="sweep",
                message=f"滑梭 {sid} 摆坠扫掠体与障碍物 {hit} 相交",
                components={**comp, "obstacle_id": hit}))

    # ---- 跨段挠度限值（仅结构包络组合） --------------------------------
    if structural and sp.max_sag_m is not None \
            and sol.sag_m > sp.max_sag_m + 1e-9:
        failures.append(CheckFailure(
            station_index=i, position=pos_fn(i),
            person_id=combo[0][0], action="traverse",
            check="sag_limit",
            message=(f"柔性跨段 {span_id} 动态下挠 {round(sol.sag_m, 3)} m "
                     f"超过挠度限值 {sp.max_sag_m} m"),
            components={"span": span_id, "bay_index": bay_index,
                        "cable_sag_m": round(sol.sag_m, 4),
                        "max_sag_m": sp.max_sag_m,
                        "falling_persons":
                            ",".join(p for p, _ in combo)}))

    # ---- 端座 / 支座反力（本组合下全跨叠加）与方向锥（仅结构包络组合） --
    if not structural:
        return
    reactions = _support_reactions(sol, span_id, spans, span_paths,
                                   params.gravity, loaded_bay=bay_index)
    for supid, (vec, mag) in reactions.items():
        sup = supports[supid]
        if mag > sup.rated_load_kn + 1e-9:
            failures.append(CheckFailure(
                station_index=i, position=pos_fn(i), person_id=None,
                action="traverse", check="support_overload",
                message=(f"柔性跨段 {span_id} 支座 {supid} 反力合力 "
                         f"{round(mag, 3)} kN 超过结构容许 "
                         f"{sup.rated_load_kn} kN"
                         + ("（保守边界上界）" if sol.conservative else "")),
                components={"span": span_id, "support": supid,
                            "bay_index": bay_index,
                            "reaction_kn": round(mag, 4),
                            "rated_load_kn": sup.rated_load_kn,
                            "rx_kn": round(vec[0], 4),
                            "ry_kn": round(vec[1], 4),
                            "rz_kn": round(vec[2], 4),
                            "conservative": int(sol.conservative),
                            "falling_persons":
                                ",".join(p for p, _ in combo)}))
        if sup.allowed_axis is not None:
            # 受力方向：支座受到的拉力（与支座对索的支撑力反向），角度同
            ang = g.angle_deg((-vec[0], -vec[1], -vec[2]),
                              sup.allowed_axis.as_tuple())
            if ang > sup.allowed_half_angle_deg + 1e-9:
                failures.append(CheckFailure(
                    station_index=i, position=pos_fn(i), person_id=None,
                    action="traverse", check="anchor_direction",
                    message=(f"支座 {supid} 受力方向 {round(ang, 2)}° 超出"
                             f"允许锥半角 {sup.allowed_half_angle_deg}°"),
                    components={"support": supid, "span": span_id,
                                "force_angle_deg": round(ang, 3),
                                "allowed_half_angle_deg":
                                    sup.allowed_half_angle_deg}))


def _append_cable_result(cable_results, i, span_id, bay_index, combo, sol,
                         force_map, spans, supports, span_paths, gravity,
                         person_map, equipment, params, stations):
    reactions = _support_reactions(sol, span_id, spans, span_paths, gravity,
                                   loaded_bay=bay_index)
    path = span_paths[span_id]
    t = path.table[i]
    total_falls = []
    for k, (pid, _sid) in enumerate(combo):
        person = person_map[pid]
        eq = equipment[person.equipment_id]
        ffd = calc.free_fall_with_sag(
            eq, stations[i][2] + person.d_ring_height_m,
            t.point[2], sol.load_sags_m[k])
        total_falls.append(round(
            ffd + eq.elongation_m + eq.buffer_travel_m
            + params.harness_stretch_m, 4))
    c = sol.chord_rise / max(sol.horiz_span, 1e-9)
    cable_results.append(CableResult(
        span_id=span_id, station_index=i, bay_index=bay_index,
        falling_persons=[pid for pid, _ in combo],
        loads_kn=[round(force_map[pid], 4) for pid, _ in combo],
        load_fractions=[round(f, 4) for f in sol.load_fractions],
        total_fall_m=total_falls,
        sag_m=round(sol.sag_m, 4),
        max_sag_m=spans[span_id].max_sag_m,
        horizontal_tension_kn=round(sol.h_kn, 4),
        left_reaction_kn=round(math.hypot(sol.h_kn, sol.vl_kn - sol.h_kn * c), 4),
        right_reaction_kn=round(math.hypot(sol.h_kn, sol.vr_kn + sol.h_kn * c), 4),
        left_support_id=span_paths[span_id].bays[bay_index].left,
        right_support_id=span_paths[span_id].bays[bay_index].right,
        support_reactions_kn={k: round(v[1], 4)
                              for k, v in sorted(reactions.items())},
        conservative=sol.conservative,
        converged=sol.converged))


class _DynamicAnchor:
    """滑梭动态位置的轻量锚点替身（供 sweep_hit_obstacle 复用）。"""

    class _Pos:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

        def as_tuple(self):
            return (self.x, self.y, self.z)

    def __init__(self, sid, table_entry, sag, path, bay_index):
        bay = path.bays[bay_index]
        f = table_entry.frac
        x = bay.p0[0] + (bay.p1[0] - bay.p0[0]) * f
        y = bay.p0[1] + (bay.p1[1] - bay.p0[1]) * f
        z = bay.p0[2] + bay.rise * f - sag
        self.id = sid
        self.position = self._Pos(x, y, z)
        self.allowed_axis = None
        self.allowed_half_angle_deg = 180.0


def _support_reactions(sol, span_id, spans, span_paths,
                       gravity, loaded_bay: int) -> dict[str, list]:
    """该组合下全跨各支座合力：加载 bay 用解，其余 bay 用预张力 + 自重。

    返回 {support_id: ((rx,ry,rz), magnitude)}。
    """
    sp = spans[span_id]
    path = span_paths[span_id]
    w = (sp.line_density_kg_m or 0.0) * gravity / 1000.0
    h0 = sp.pretension_kn or 0.0
    acc: dict[str, list] = {sid: [0.0, 0.0, 0.0] for sid in sp.supports}

    for b_idx, bay in enumerate(path.bays):
        Lh = bay.horiz
        ux = (bay.p1[0] - bay.p0[0]) / max(Lh, 1e-12)
        uy = (bay.p1[1] - bay.p0[1]) / max(Lh, 1e-12)
        c = bay.rise / max(Lh, 1e-12)
        if b_idx == loaded_bay:
            H, VL, VR = sol.h_kn, sol.vl_kn, sol.vr_kn
            # 端 bay 坠落点落在端座上：集中力直接作用于端座
            extra = {bay.left: 0.0, bay.right: 0.0}
            for fr_j, p_j in zip(sol.load_fractions,
                                 getattr(sol, "point_loads_kn", [])):
                if fr_j <= 1e-9:
                    extra[bay.left] += p_j
                elif fr_j >= 1.0 - 1e-9:
                    extra[bay.right] += p_j
        else:
            H, VL, VR = h0, w * Lh / 2.0, w * Lh / 2.0
            extra = {bay.left: 0.0, bay.right: 0.0}
        # 左端：索对支座的拉力（水平向左；竖直分量 cH−VL），另加端座直接受荷
        vL = H * c - VL - extra[bay.left]
        vR = H * c + VR + extra[bay.right]
        lvec = (-H * ux, -H * uy, vL)
        rvec = (H * ux, H * uy, vR)
        for kk in range(3):
            acc[bay.left][kk] += lvec[kk]
            acc[bay.right][kk] += rvec[kk]
    return {sid: ((v[0], v[1], v[2]),
                  math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2))
            for sid, v in acc.items()}
