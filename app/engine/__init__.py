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
from dataclasses import dataclass, field

from ..models import (AnalysisResult, CableResult, CheckFailure, OpenItem,
                      Person, PlanPayload, ProfilePoint, Vec3)
from . import calc, cable, geometry as g
from . import twinleg
from .sequence import (ReachProvider, Target, build_sequence,
                       replay_sequence)
from .shuttle_path import SpanPath, unloaded_bay_sags

# 同站失败排序优先级（数值小者优先，作为“最先失败”）
_CHECK_PRIORITY = {
    "hook_chain": 0,
    "leg_length_insufficient": 0,
    "sharp_edge": 1,
    "anchor_direction": 2,
    "included_angle": 2,
    "connector_side_load": 2,
    "anchor_overload": 3,
    "support_overload": 3,
    "arrest_force": 3,
    "buffer_energy": 3,
    "clearance": 4,
    "sag_limit": 4,
    "sweep": 5,
}


@dataclass
class AnalysisContext:
    """一次通行核算的完整上下文：结果之外保留站点、序列状态与悬索解，
    供坠落后救援推演复用（冻结修订的动作序列、坠距与支座反力）。"""
    payload: PlanPayload
    stations: list[tuple[float, float, float]]
    span_paths: dict[str, SpanPath]
    missing_by_span: dict[str, list[str]]
    seq_by_person: dict[str, object]
    cable_results: list[CableResult]
    point_falls: dict
    result: AnalysisResult


def analyze(payload: PlanPayload) -> AnalysisResult:
    return build_context(payload).result


def build_context(payload: PlanPayload) -> AnalysisContext:
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
    # (站, 锚点) -> FallCalc：救援推演复用冻结修订的点锚坠距/止坠力
    point_falls: dict[tuple[int, str], object] = {}

    # ---- 0. 柔性跨段几何与每站滑梭站位 --------------------------------
    span_paths: dict[str, SpanPath] = {}
    for sp in route.spans:
        sags0 = unloaded_bay_sags(sp, supports, grav)
        span_paths[sp.id] = SpanPath(sp, supports, stations, sags0)

    # 缺失参数的跨段：只要该跨被任一有效序列使用，即记 open item（后述）
    missing_by_span = {sp.id: sp.missing_params() for sp in route.spans}

    # ---- 1. 可达性 / 共用容量 / 滑梭跨支座 -----------------------------
    # 点锚：每目标每站独立可达；滑梭：站位有效且 D 环到该站滑梭位置可达。
    # Y 型双腿系绳：可达按该钩绑定实体腿的原长 + 装备连接器余量判定。
    def anchor_reach(eq, hook: str | None) -> float:
        if eq.twin_leg is not None and hook is not None:
            return eq.twin_leg.leg_for_hook(hook).leg_length_m \
                + eq.connector_reach_m
        return eq.lanyard_length_m + eq.connector_reach_m

    def shuttle_reach(eq, sid: str, hook: str | None) -> float:
        if eq.twin_leg is not None and hook is not None:
            return eq.twin_leg.leg_for_hook(hook).leg_length_m \
                + eq.connector_reach_m
        return eq.lanyard_length_m + shuttles[sid].connector_reach_m

    def shuttle_reachable(sid: str, i: int, person: Person, eq,
                          hook: str | None) -> bool:
        t = span_paths[shuttles[sid].span_id].table[i]
        if not t.valid:
            return False
        d = calc.d_ring_pos(stations[i], person)
        return g.dist3(d, t.point) <= shuttle_reach(eq, sid, hook) + 1e-9

    # 容量计数：点锚/滑梭按“占用钩数（人去重）”记录，用于他人容量上限；
    # 同一人 A/B 钩重复挂同一滑梭属 duplicate_occupancy（双钩不独立，不下结论）。
    anchor_count: list[dict[str, set[str]]] = [dict() for _ in range(n)]
    shuttle_count: list[dict[str, set[str]]] = [dict() for _ in range(n)]
    span_users: list[dict[str, set[str]]] = [dict() for _ in range(n)]

    def make_provider(person: Person) -> ReachProvider:
        eq = equipment[person.equipment_id]
        targets: list[Target] = [("anchor", a.id) for a in route.anchors] \
            + [("shuttle", s.id) for s in route.shuttles]

        def reachable(i: int, t: Target, hook: str | None = None) -> bool:
            kind, tid = t
            if kind == "anchor":
                a = anchors[tid].position.as_tuple()
                return g.dist3(calc.d_ring_pos(stations[i], person), a) \
                    <= anchor_reach(eq, hook) + 1e-9
            return shuttle_reachable(tid, i, person, eq, hook)

        def leg_shortfall(i: int, t: Target, hook: str) -> dict | None:
            """Y 型双腿系绳：该钩绑定腿的触及包络（含装备连接器余量）是否
            恰好差在腿长上——目标在另一钩（更长）腿的包络内或本可换挂，
            仅因绑定腿过短而不可达时，返回超差分量；其他原因返回 None。"""
            if eq.twin_leg is None:
                return None
            leg = eq.twin_leg.leg_for_hook(hook)
            reach = leg.leg_length_m + eq.connector_reach_m
            kind, tid = t
            if kind == "anchor":
                tp = anchors[tid].position.as_tuple()
            else:
                te = span_paths[shuttles[tid].span_id].table[i]
                if not te.valid:
                    return None
                tp = te.point
            d_ring = calc.d_ring_pos(stations[i], person)
            d = g.dist3(d_ring, tp)
            if d <= reach + 1e-9:
                return None                      # 本腿可达，非腿长问题
            # 目标在另一钩腿的包络内：可证明确系该绑定腿长度不足
            other = "B" if hook == "A" else "A"
            other_len = eq.twin_leg.leg_for_hook(other).leg_length_m
            other_reach = other_len + eq.connector_reach_m
            if d > other_reach + 1e-9:
                return None                      # 两腿都够不着，不属本判定
            return {"target": f"{'滑梭' if kind == 'shuttle' else '锚点'} {tid}",
                    "target_kind": kind, "target_id": tid,
                    "bound_leg": leg.id, "hook": hook,
                    "leg_length_m": leg.leg_length_m,
                    "connector_reach_m": eq.connector_reach_m,
                    "required_m": round(d, 4),
                    "shortfall_m": round(d - reach, 4)}

        # 前向连续可达末站（按目标 + 钩分别缓存：双腿系绳两腿长度不同）
        last_cache: dict[tuple[Target, str | None], list[int]] = {}

        def last_reach(t: Target, i: int, hook: str | None = None) -> int:
            key = (t, hook)
            if key not in last_cache:
                arr = [-1] * n
                if reachable(n - 1, t, hook):
                    arr[n - 1] = n - 1
                for k in range(n - 2, -1, -1):
                    arr[k] = k if not reachable(k + 1, t, hook) else arr[k + 1] \
                        if reachable(k, t, hook) else -1
                last_cache[key] = arr
            return last_cache[key][i]

        def attach_block(i: int, t: Target, other: Target | None):
            """返回 None 表示可挂；否则 (code, components)。

            点锚：他人占用达上限 → capacity_full（旧语义失效）。
            双腿 Y 型系绳：本人另一钩已挂同一目标（点锚或滑梭）→
            duplicate_occupancy（两腿必须独立接载，否则载荷分配无意义）。
            滑梭：滑梭或跨段的他人占用达上限 → duplicate_occupancy（不下结论）。
            """
            kind, tid = t
            # Y 型双腿系绳：两腿挂同一实体即重复占用（不下结论）
            if eq.twin_leg is not None and other == t:
                return "duplicate_occupancy", {
                    ("target_shuttle" if kind == "shuttle"
                     else "target_anchor"): tid,
                    "reason": "same_person_other_hook"}
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
            crosses_blocked=crosses_blocked,
            leg_length_shortfall=leg_shortfall)

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

    # ---- 4. 坠落核算 ----------------------------------------------------
    # 旧等长模型：点锚逐连接核算（分量/判定保持旧版）；
    # Y 型双腿系绳：按整人（两腿 + 一个共享缓冲包）做能量 + 变形协调求解。
    point_load: list[dict[str, float]] = [dict() for _ in range(n)]
    point_users: list[dict[str, list[str]]] = [dict() for _ in range(n)]
    twin_results: list = []
    # (i, pid) -> TwinSolution：第 5 节悬索迭代复用/精化，剖面与救援复用
    twin_solutions: dict[tuple[int, str], object] = {}
    # 双腿系绳佩戴者在站 i 的钩→(kind,id) 映射（用于跨段迭代与结果构建）
    twin_hooks: dict[tuple[int, str], dict[str, tuple[str, str]]] = {}

    def d_ring(i: int, person: Person):
        return calc.d_ring_pos(stations[i], person)

    def eval_twin_station(person, eq, state, i, sag_by_shuttle):
        """对一站一名双腿系绳佩戴者求解载荷分配（挂点坐标可随悬索下挠）。"""
        by_hook = {h: t for h, t in zip(("A", "B"), state)
                   if t is not None}
        legs = twinleg.build_leg_targets(
            eq, by_hook, person, stations[i], i, anchors, shuttles,
            span_paths, sag_by_shuttle)
        dpos = calc.d_ring_pos(stations[i], person)
        sol = twinleg.solve_twin_legs(
            person.weight_kg, params.gravity, dpos, legs,
            eq.twin_leg.buffer_curve, eq.twin_leg.max_travel_m)
        return sol, by_hook

    for seq in sequences:
        person = person_map[seq.person_id]
        eq = equipment[person.equipment_id]
        L = eq.lanyard_length_m
        for i, state in enumerate(seq.states):
            if state is None:
                break
            if eq.twin_leg is not None:
                # ---- Y 型双腿系绳：整人一个共享缓冲包，只求解一次 --------
                by_hook = {h: t for h, t in zip(("A", "B"), state)
                           if t is not None}
                # 两腿挂点分属不同跨段：不同悬索解耦合，本版不下结论
                span_of = {}
                for h, t in by_hook.items():
                    if t[0] == "shuttle":
                        span_of[h] = shuttles[t[1]].span_id
                if len(set(span_of.values())) > 1:
                    open_items.append(OpenItem(
                        station_index=i, position=pos(i),
                        person_id=person.id, action="traverse",
                        code="twin_leg_cross_span",
                        message="Y 型系绳两腿挂在不同柔性跨段的滑梭上，"
                                "悬索下挠耦合无法由本核算分别判定，本版不下结论",
                        components={"hook_A": _tid_target(by_hook.get("A")),
                                    "hook_B": _tid_target(by_hook.get("B"))}))
                    continue
                twin_hooks[(i, person.id)] = by_hook
                sol, by_hook = eval_twin_station(person, eq, state, i, {})
                twin_solutions[(i, person.id)] = sol
                _evaluate_twin_solution(
                    i, person, eq, sol, by_hook, stations, pos, anchors,
                    shuttles, span_paths, route, params, failures,
                    open_items, point_load, point_users,
                    twin_results, twin_solutions, dynamic=False,
                    point_falls=point_falls)
                continue

            anchor_targets = sorted({t for t in state
                                     if t and t[0] == "anchor"})
            for (_k, aid) in anchor_targets:
                anchor = anchors[aid]
                fc = calc.fall_calc(stations[i], person, eq, anchor,
                                    route, params)
                point_falls[(i, aid)] = fc
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
        # Y 型双腿系绳人员的滑梭载荷由第 5b 节按整人（一个共享缓冲包）求解，
        # 此处排除，避免同一缓冲/止坠力被旧单连接路径重复计入。
        groups: dict[str, dict[int, list[tuple[str, str]]]] = {}
        for person in payload.persons:
            if equipment[person.equipment_id].twin_leg is not None:
                continue
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

    # ---- 5b. Y 型双腿系绳 × 柔性跨段：动态下挠耦合 ----------------------
    # 按 (站, 动作最早发生站) 索引序列事件，供双腿结果定位 attach/switch 动作
    seq_events_by_station: dict[int, list] = {}
    for seq0 in sequences:
        for ev in seq0.events:
            seq_events_by_station.setdefault(ev.station_index, []).append(ev)

    def _state_from_byhook(by_hook):
        return (by_hook.get("A"), by_hook.get("B"))

    # 含滑梭腿的双腿佩戴者：以人体峰值制动力（单人一个竖直合力）加载 bay，
    # 迭代时按各滑梭腿当前下挠重跑 solve_twin_legs；同一缓冲能力只计一次。
    twin_persons = {p.id for p in payload.persons
                    if equipment[p.equipment_id].twin_leg is not None}
    for i in range(n):
        for pid in sorted(twin_persons):
            person = person_map[pid]
            eq = equipment[person.equipment_id]
            by_hook = twin_hooks.get((i, pid))
            if not by_hook:
                continue
            shuttle_hooks = {h: t for h, t in by_hook.items()
                             if t[0] == "shuttle"}
            if not shuttle_hooks:
                continue                          # 纯点锚，第 4 节已评估
            spid0 = shuttles[next(iter(shuttle_hooks.values()))[1]].span_id
            # 两腿滑梭分属不同跨段已在第 4 节记 open item，跳过耦合
            span_ids = {shuttles[t[1]].span_id for t in shuttle_hooks.values()}
            if len(span_ids) > 1 or missing_by_span.get(spid0):
                continue
            spid = next(iter(span_ids))
            path = span_paths[spid]
            te = path.table[i]
            if not te.valid:
                continue
            bay = path.bays[te.bay]
            w = (spans[spid].line_density_kg_m or 0.0) * grav / 1000.0

            def run_twin(sag_by_sid):
                sol, _ = eval_twin_station(person, eq,
                                           _state_from_byhook(by_hook),
                                           i, sag_by_sid)
                return sol

            # 初解（静态挂点）
            sol = run_twin({})
            force0 = sol.buffer_force_kn if sol.converged \
                else eq.max_arrest_force_kn
            bay_sol = None
            sag_prev = 0.0
            for _it in range(25):
                load = cable.BayLoad(pid, te.frac, force0)
                bay_sol = cable.solve_bay(
                    L=bay.horiz, rise=bay.rise, loads=[load],
                    h0=spans[spid].pretension_kn or 0.0, w=w,
                    ea=spans[spid].ea_kn(),
                    allow_conservative=payload.conservative_bounds.enabled,
                    max_iter=params.cable_max_iter, tol=params.cable_tol_kn)
                if not bay_sol.converged:
                    break
                sag_k = bay_sol.load_sags_m[0]
                sag_by_sid = {t[1]: sag_k for t in shuttle_hooks.values()}
                sol = run_twin(sag_by_sid)
                f_new = sol.buffer_force_kn if sol.converged else force0
                if abs(f_new - force0) < 1e-4 and abs(sag_k - sag_prev) < 1e-6:
                    force0 = f_new
                    sag_prev = sag_k
                    break
                force0, sag_prev = f_new, sag_k

            final_sags = {t[1]: sag_prev for t in shuttle_hooks.values()}
            sol = run_twin(final_sags)

            # 悬索不收敛：不下结论
            if bay_sol is None or not bay_sol.converged:
                open_items.append(OpenItem(
                    station_index=i, position=pos(i), person_id=pid,
                    action="traverse", code="solver_nonconvergence",
                    message=f"柔性跨段 {spid} 上 Y 型双腿系绳悬索迭代不收敛，"
                            f"本版不下结论"
                            + ("（已按授权采用保守边界）"
                               if payload.conservative_bounds.enabled else ""),
                    components={"span": spid, "bay_index": te.bay}))
                continue
            if bay_sol.conservative:
                conservative_spans.add(spid)

            # 用动态解重评双腿结果（替换第 4 节静态结果，缓冲不重复计）
            _evaluate_twin_solution(
                i, person, eq, sol, by_hook, stations, pos, anchors,
                shuttles, span_paths, route, params, failures, open_items,
                point_load, point_users, twin_results, twin_solutions,
                dynamic=True, point_falls=point_falls,
                seq_events_by_station=seq_events_by_station)

            # 挠度限值 + 端座/支座反力（单人组合即结构包络）
            sp_obj = spans[spid]
            if sp_obj.max_sag_m is not None \
                    and bay_sol.sag_m > sp_obj.max_sag_m + 1e-9:
                failures.append(CheckFailure(
                    station_index=i, position=pos(i), person_id=pid,
                    action="traverse", check="sag_limit",
                    message=(f"柔性跨段 {spid} 动态下挠 "
                             f"{round(bay_sol.sag_m, 3)} m 超过挠度限值 "
                             f"{sp_obj.max_sag_m} m"),
                    components={"span": spid, "bay_index": te.bay,
                                "cable_sag_m": round(bay_sol.sag_m, 4),
                                "max_sag_m": sp_obj.max_sag_m}))
            reactions = _support_reactions(bay_sol, spid, spans, span_paths,
                                           grav, loaded_bay=te.bay)
            for supid, (vec, mag) in reactions.items():
                sup = supports[supid]
                if mag > sup.rated_load_kn + 1e-9:
                    failures.append(CheckFailure(
                        station_index=i, position=pos(i), person_id=None,
                        action="traverse", check="support_overload",
                        message=(f"柔性跨段 {spid} 支座 {supid} 反力合力 "
                                 f"{round(mag, 3)} kN 超过结构容许 "
                                 f"{sup.rated_load_kn} kN"),
                        components={"span": spid, "support": supid,
                                    "bay_index": te.bay,
                                    "reaction_kn": round(mag, 4),
                                    "rated_load_kn": sup.rated_load_kn}))
                if sup.allowed_axis is not None:
                    ang = g.angle_deg((-vec[0], -vec[1], -vec[2]),
                                      sup.allowed_axis.as_tuple())
                    if ang > sup.allowed_half_angle_deg + 1e-9:
                        failures.append(CheckFailure(
                            station_index=i, position=pos(i), person_id=None,
                            action="traverse", check="anchor_direction",
                            message=(f"支座 {supid} 受力方向 {round(ang, 2)}° "
                                     f"超出允许锥半角 "
                                     f"{sup.allowed_half_angle_deg}°"),
                            components={"support": supid, "span": spid,
                                        "force_angle_deg": round(ang, 3),
                                        "allowed_half_angle_deg":
                                            sup.allowed_half_angle_deg}))
            # 记一条 CableResult（保留跨段结构分量；总坠距取双腿人体解）
            twin_res = next((r for r in twin_results
                             if r.station_index == i and r.person_id == pid),
                            None)
            twin_total = {pid: twin_res.total_fall_m} if twin_res else None
            _append_cable_result(
                cable_results, i, spid, te.bay, [(pid, "twin")], bay_sol,
                {pid: force0}, spans, supports, span_paths, grav,
                person_map, equipment, params, stations,
                twin_dynamic=True, twin_total_fall=twin_total)

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
        # Y 型双腿系绳：剖面直接取双腿人体解（总坠距/净空/分量）
        if eq0.twin_leg is not None:
            tr = next((r for r in twin_results
                       if r.station_index == i and r.person_id == p0.id),
                      None)
            if tr is not None:
                taut = [l for l in tr.legs if l.taut]
                anc = next((l for l in reversed(taut)
                            if l.target_kind == "anchor"), None)
                shu = next((l for l in reversed(taut)
                            if l.target_kind == "shuttle"), None)
                if anc is not None:
                    pp.controlling_anchor = anc.target_id
                    pp.anchor_z = round(anc.anchor_position.z, 4)
                if shu is not None:
                    pp.controlling_shuttle = shu.target_id
                    pp.span_id = shuttles[shu.target_id].span_id
                    pp.anchor_z = round(shu.anchor_position.z, 4)
                    static_z = span_paths[shuttles[shu.target_id].span_id] \
                        .table[i].point[2]
                    pp.cable_sag_m = round(
                        max(0.0, static_z - shu.anchor_position.z), 4)
                pp.free_fall_m = tr.free_fall_m
                pp.total_fall_m = tr.total_fall_m
                pp.required_clearance_m = tr.required_clearance_m
                pp.clearance_floor_z = round(
                    stations[i][2] - tr.required_clearance_m, 4)
                pp.available_clearance_m = tr.available_clearance_m
                pp.margin_m = tr.margin_m
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
    result = AnalysisResult(
        conclusive=conclusive,
        passable=conclusive and not failures,
        first_failure=failures[0] if failures else None,
        failures=failures,
        first_open_item=open_items[0] if open_items else None,
        open_items=open_items,
        sequence=[ev for seq in sequences for ev in seq.events],
        profile=profile,
        cable_results=cable_results,
        twin_leg_results=twin_results,
        conservative_used=sorted(conservative_spans),
        station_count=n,
    )
    return AnalysisContext(
        payload=payload, stations=stations, span_paths=span_paths,
        missing_by_span=missing_by_span, seq_by_person=seq_by_person,
        cable_results=cable_results, point_falls=point_falls, result=result)


# ---------------------------------------------------------------- 悬索组合求解

def _arrest_force(person, eq, ffd, params):
    return calc.arrest_force_kn(person, eq, ffd, params)


def _tid_target(t) -> str:
    if t is None:
        return ""
    return ("anchor:" if t[0] == "anchor" else "shuttle:") + t[1]


def _adopted_params(eq) -> dict:
    """Y 型双腿系绳求解采用的参数（随响应返回，便于复核/版本对比）。"""
    tw = eq.twin_leg
    out: dict = {
        "connector_reach_m": eq.connector_reach_m,
        "max_arrest_force_kn": eq.max_arrest_force_kn,
        "max_travel_m": tw.max_travel_m,
        "energy_capacity_j": tw.energy_capacity_j,
        "max_included_angle_deg": tw.max_included_angle_deg,
    }
    for leg in tw.legs:
        out[f"leg_{leg.id}_hook"] = leg.hook
        out[f"leg_{leg.id}_length_m"] = leg.leg_length_m
        out[f"leg_{leg.id}_stiffness_kn_m"] = leg.axial_stiffness_kn
        out[f"leg_{leg.id}_side_limit_kn"] = leg.connector_side_load_limit_kn
    for k, pnt in enumerate(tw.buffer_curve):
        out[f"curve_{k}_travel_m"] = pnt.travel_m
        out[f"curve_{k}_force_kn"] = pnt.force_kn
    return out


def _build_twin_result(i, pos3, person, eq, sol, by_hook, anchors, shuttles,
                       span_paths, stations, params, route) -> object:
    """把物理解 TwinSolution 转成响应模型 TwinLegResult（含净空分量）。"""
    from ..models import (LanyardLegResult, TwinLegResult)
    leg_out = []
    for lo in sol.legs:
        leg_out.append(LanyardLegResult(
            leg_id=lo.leg_id, hook=lo.hook, target_kind=lo.target_kind,
            target_id=lo.target_id,
            anchor_position=Vec3(x=lo.anchor[0], y=lo.anchor[1],
                                 z=lo.anchor[2]),
            original_length_m=round(lo.length, 4),
            elastic_extension_m=round(lo.extension_m, 5),
            tension_kn=round(lo.tension_kn, 4),
            vertical_component_kn=round(lo.vertical_component_kn, 4),
            horizontal_component_kn=round(lo.horizontal_component_kn, 4),
            connector_side_load_kn=round(lo.side_load_kn, 4),
            connector_side_load_limit_kn=round(lo.side_limit_kn, 4),
            taut=lo.taut))
    station = stations[i]
    total_fall = sol.drop_m
    required = total_fall + params.safety_margin_m + params.harness_stretch_m
    surfaces = []
    top = calc.obstacle_top_below(station, person.body_radius_m,
                                  route.obstacles)
    if top is not None:
        surfaces.append(top)
    low = calc.lower_level_below(station, eq.lanyard_length_m,
                                 route.drop_edges)
    if low is not None:
        surfaces.append(low)
    if surfaces:
        avail = station[2] - max(surfaces)
        margin = avail - required
    else:
        avail = margin = None
    angle = twinleg.included_angle(sol)
    return TwinLegResult(
        station_index=i, position=pos3, person_id=person.id,
        engagement_order=list(sol.engagement_order), legs=leg_out,
        included_angle_deg=None if angle is None else round(angle, 3),
        buffer_deployment_m=round(sol.deployment_m, 4),
        buffer_max_travel_m=eq.twin_leg.max_travel_m,
        buffer_force_kn=round(sol.buffer_force_kn, 4),
        energy_demand_j=round(sol.energy_demand_j, 2),
        buffer_energy_absorbed_j=round(sol.buffer_energy_j, 2),
        elastic_energy_j=round(sol.elastic_energy_j, 3),
        energy_capacity_j=eq.twin_leg.energy_capacity_j,
        free_fall_m=round(sol.free_fall_m, 4),
        total_fall_m=round(total_fall, 4),
        required_clearance_m=round(required, 4),
        available_clearance_m=None if avail is None else round(avail, 4),
        margin_m=None if margin is None else round(margin, 4),
        junction_position=Vec3(x=sol.junction[0], y=sol.junction[1],
                               z=sol.junction[2]),
        converged=sol.converged,
        adopted_params=_adopted_params(eq))


def _evaluate_twin_solution(i, person, eq, sol, by_hook, stations, pos,
                            anchors, shuttles, span_paths, route, params,
                            failures, open_items, point_load, point_users,
                            twin_results, twin_solutions, *, dynamic,
                            point_falls=None,
                            seq_events_by_station=None):
    """把一个双腿物理解登记为结果并执行全部判定。

    dynamic=True 表示挂点已含悬索动态下挠（由第 5 节回写）；此时替换既有
    静态结果，避免同一缓冲能力被重复计入。
    """
    res = _build_twin_result(i, pos(i), person, eq, sol, by_hook, anchors,
                             shuttles, span_paths, stations, params, route)
    if dynamic:
        twin_results[:] = [r for r in twin_results
                           if not (r.station_index == i
                                   and r.person_id == person.id)]
    twin_results.append(res)

    action = "traverse"
    if seq_events_by_station is not None:
        evs = seq_events_by_station.get(i, [])
        if evs:
            action = evs[0].action

    base_comp = {
        "buffer_deployment_m": res.buffer_deployment_m,
        "buffer_force_kn": res.buffer_force_kn,
        "energy_demand_j": res.energy_demand_j,
        "buffer_energy_absorbed_j": res.buffer_energy_absorbed_j,
        "energy_capacity_j": res.energy_capacity_j,
        "free_fall_m": res.free_fall_m,
        "total_fall_m": res.total_fall_m,
        "engagement_order": ",".join(res.engagement_order),
        "leg_tensions_kn": ",".join(str(lg.tension_kn) for lg in res.legs),
    }

    def fail(check, msg, comp):
        failures.append(CheckFailure(
            station_index=i, position=pos(i), person_id=person.id,
            action=action, check=check, message=msg,
            components={**base_comp, **comp}))

    def openp(code, msg, comp):
        open_items.append(OpenItem(
            station_index=i, position=pos(i), person_id=person.id,
            action=action, code=code, message=msg,
            components={**base_comp, **comp}))

    # ---- 求解不收敛 / 无腿接载（不下结论） ------------------------------
    if not sol.converged or sol.reason == "equilibrium_solver_failed":
        openp("twin_leg_nonconvergence",
              "Y 型双腿系绳载荷分配求解不收敛，本版不下结论", {})
        return res
    if sol.reason == "no_leg_engages":
        fail("leg_length_insufficient",
             "两腿在竖直坠落路径上均无法张紧（挂点水平投影超过腿原长），"
             "腿长不足，不得标为可通行", {})
        return res

    # ---- 曲线能量不足（失效） ------------------------------------------
    if sol.reason == "buffer_energy_insufficient":
        fail("buffer_energy",
             f"共享缓冲包完全展开 {eq.twin_leg.max_travel_m} m 后仍有 "
             f"{round(max(0.0, sol.demand_at_full_j - sol.buffer_energy_j), 1)} J "
             f"能量亏缺，曲线能量不足，不得标为可通行",
             {"energy_shortfall_j":
                 round(max(0.0, sol.demand_at_full_j - sol.buffer_energy_j), 2)})
    elif res.energy_demand_j > eq.twin_leg.energy_capacity_j + 1e-6:
        fail("buffer_energy",
             f"坠落需吸收能量 {round(res.energy_demand_j, 1)} J 超过缓冲包"
             f"能量容量 {eq.twin_leg.energy_capacity_j} J", {})

    # ---- 人体峰值制动力超限 --------------------------------------------
    if res.buffer_force_kn > eq.max_arrest_force_kn + 1e-9:
        fail("arrest_force",
             f"人体峰值制动力 {res.buffer_force_kn} kN 超过装备最大止坠力 "
             f"{eq.max_arrest_force_kn} kN",
             {"max_arrest_force_kn": eq.max_arrest_force_kn})

    # ---- 夹角越限 ------------------------------------------------------
    if res.included_angle_deg is not None \
            and res.included_angle_deg > eq.twin_leg.max_included_angle_deg + 1e-9:
        fail("included_angle",
             f"两腿夹角 {res.included_angle_deg}° 超过允许夹角 "
             f"{eq.twin_leg.max_included_angle_deg}°",
             {"included_angle_deg": res.included_angle_deg,
              "max_included_angle_deg": eq.twin_leg.max_included_angle_deg})

    # ---- 逐腿：连接器侧载越限 + 锚点负荷收集 ----------------------------
    anchor_load_add: dict[str, float] = {}
    for lg in res.legs:
        if lg.taut and lg.connector_side_load_kn \
                > lg.connector_side_load_limit_kn + 1e-9:
            fail("connector_side_load",
                 f"实体腿 {lg.leg_id}（{lg.hook} 钩，{lg.target_id}）连接器"
                 f"侧载 {lg.connector_side_load_kn} kN 超过限值 "
                 f"{lg.connector_side_load_limit_kn} kN",
                 {"failing_leg": lg.leg_id, "failing_hook": lg.hook,
                  "target_id": lg.target_id,
                  "connector_side_load_kn": lg.connector_side_load_kn,
                  "connector_side_load_limit_kn":
                      lg.connector_side_load_limit_kn})
        if lg.target_kind == "anchor":
            anchor_load_add[lg.target_id] = \
                anchor_load_add.get(lg.target_id, 0.0) + lg.tension_kn

    # ---- 净空（人体总坠距 + 安全带伸长） --------------------------------
    if res.margin_m is not None and res.margin_m < 0:
        fail("clearance",
             f"Y 型系绳总净空余量为负（{res.margin_m} m）：缓冲展开后"
             f"撞上下层障碍物/楼面",
             {"required_clearance_m": res.required_clearance_m,
              "available_clearance_m": res.available_clearance_m,
              "margin_m": res.margin_m})

    # ---- 锚点方向锥（按绳腿受力方向） ----------------------------------
    for lg in res.legs:
        if lg.target_kind != "anchor" or not lg.taut:
            continue
        anchor = anchors[lg.target_id]
        if anchor.allowed_axis is not None:
            vj = (res.junction_position.x - anchor.position.x,
                  res.junction_position.y - anchor.position.y,
                  res.junction_position.z - anchor.position.z)
            ang = g.angle_deg(vj, anchor.allowed_axis.as_tuple())
            if ang > anchor.allowed_half_angle_deg + 1e-9:
                fail("anchor_direction",
                     f"锚点 {lg.target_id} 受力方向 {round(ang, 2)}° 超出"
                     f"允许锥半角 {anchor.allowed_half_angle_deg}°",
                     {"anchor": lg.target_id, "failing_leg": lg.leg_id,
                      "force_angle_deg": round(ang, 3),
                      "allowed_half_angle_deg":
                          anchor.allowed_half_angle_deg})

    # ---- 锐边（装备等级） ----------------------------------------------
    L = eq.lanyard_length_m
    for e in route.drop_edges:
        if (g.dist2d(stations[i], e.point.as_tuple()) <= L
                and e.sharpness_class > eq.sharp_edge_rating):
            fail("sharp_edge",
                 f"落差边缘 {e.id} 锐边等级 {e.sharpness_class} 超过装备"
                 f"适配等级 {eq.sharp_edge_rating}",
                 {"edge_id": e.id,
                  "edge_sharpness_class": e.sharpness_class,
                  "equipment_sharp_edge_rating": eq.sharp_edge_rating})

    # ---- 登记锚点负荷（合力 = 该锚所连各承拉腿张力之和） ----------------
    for aid, load in anchor_load_add.items():
        point_load[i][aid] = point_load[i].get(aid, 0.0) + load
        point_users[i].setdefault(aid, [])
        if person.id not in point_users[i][aid]:
            point_users[i][aid].append(person.id)

    # ---- 为承拉点锚腿合成 FallCalc，供坠落后救援推演复用冻结分量 ---------
    if point_falls is not None:
        from .calc import FallCalc
        for lg in res.legs:
            if lg.target_kind != "anchor" or not lg.taut:
                continue
            anchor = anchors[lg.target_id]
            station = stations[i]
            r_h = g.dist2d(anchor.position.as_tuple(), station)
            point_falls[(i, lg.target_id)] = FallCalc(
                free_fall_m=res.free_fall_m,
                total_fall_m=res.total_fall_m,
                deployed_length_m=round(
                    lg.original_length_m + lg.elastic_extension_m
                    + res.buffer_deployment_m, 4),
                required_clearance_m=res.required_clearance_m,
                available_clearance_m=res.available_clearance_m,
                margin_m=res.margin_m,
                arrest_force_kn=res.buffer_force_kn,
                force_angle_deg=None,
                horizontal_offset_m=round(r_h, 4),
                swing_radius_m=round(
                    lg.original_length_m + lg.elastic_extension_m
                    + res.buffer_deployment_m, 4))
    return res


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
                         person_map, equipment, params, stations,
                         twin_dynamic: bool = False,
                         twin_total_fall: dict | None = None):
    reactions = _support_reactions(sol, span_id, spans, span_paths, gravity,
                                   loaded_bay=bay_index)
    path = span_paths[span_id]
    t = path.table[i]
    total_falls = []
    for k, (pid, _sid) in enumerate(combo):
        person = person_map[pid]
        eq = equipment[person.equipment_id]
        if twin_dynamic and twin_total_fall and pid in twin_total_fall:
            total_falls.append(round(twin_total_fall[pid], 4))
            continue
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
