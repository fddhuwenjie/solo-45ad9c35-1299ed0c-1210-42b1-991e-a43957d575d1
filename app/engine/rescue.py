"""坠落后救援推演：在冻结修订（站点、动作序列、坠距、支座反力）之上，
逐步模拟接近 → 二次保护 → 提拉卸载 → 脱离原系统 → 升降转运。

每步给出绳程、人工牵引力、锚点合力、伤员轨迹净空与预计耗时；
来源分析无结论、救援锚点不可达、绳长不足、器材重复占用、
载荷越限（锚点额定 / 下降器限载 / 单人持续牵引力）或悬吊时间越限时，
在最早受阻步骤返回 block，方案不得标为可执行。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..models import (ManualDecision, Person, PlanPayload, RescueBlock,
                      RescuePayload, RescueResult, RescueStep,
                      StationRescue, Vec3)
from . import geometry as g

Vec = tuple[float, float, float]

_STEP_NAMES = {
    "approach": "接近",
    "secondary_protection": "二次保护",
    "haul_unload": "提拉卸载",
    "detach_original": "脱离原系统",
    "transfer": "升降转运",
}


@dataclass
class _Hang:
    """一名伤员在某站被止坠后的悬挂状态（取各有效连接中的控制性分量）。"""
    anchor_pos: Vec
    deploy: float              # 展开长度（L+伸长+缓冲）
    feet_z: float              # 悬挂脚底标高
    arrest_force_kn: float
    total_fall_m: float
    cable_sag_m: float
    target: str                # anchor:<id> / shuttle:<id>
    is_shuttle: bool
    original: list[str]        # 全部原系统连接（脱离步骤逐件解脱）


class _Block(Exception):
    def __init__(self, step_no: int, step_code: str, code: str,
                 message: str, components: dict | None = None):
        super().__init__(message)
        self.step_no = step_no
        self.step_code = step_code
        self.code = code
        self.message = message
        self.components = components or {}


# ---------------------------------------------------------------- 入口几何

def _project_polyline(point: Vec, polyline: list[Vec]):
    """点到行走折线的最近水平投影：返回 (水平偏差, 累积弧长参数 s, 投影点)。"""
    best = None
    acc = 0.0
    for p0, p1 in zip(polyline, polyline[1:]):
        seg = g.dist3(p0, p1)
        vx, vy = p1[0] - p0[0], p1[1] - p0[1]
        denom = vx * vx + vy * vy
        t = 0.0 if denom < 1e-12 else min(
            1.0, max(0.0, ((point[0] - p0[0]) * vx
                           + (point[1] - p0[1]) * vy) / denom))
        q = (p0[0] + vx * t, p0[1] + vy * t,
             p0[2] + (p1[2] - p0[2]) * t)
        key = (g.dist2d(point, q), acc + seg * t)
        if best is None or key < best[:2]:
            best = (key[0], key[1], q)
        acc += seg
    return best


def _walk_path(polyline: list[Vec], a: Vec, b: Vec) -> float:
    """救援人员从入口到站点的行进路程：入口偏离段（水平）+ 折线段弧长 +
    站点偏离段（水平，站点在折线上时为 0）。"""
    off_a, sa, _qa = _project_polyline(a, poly)
    off_b, sb, _qb = _project_polyline(b, poly)
    return off_a + abs(sb - sa) + off_b


# ---------------------------------------------------------------- 悬挂状态

def _hang_for(ctx, i: int, person: Person, eq) -> _Hang | None:
    """该人在站 i 被止坠后的悬挂状态；撞击下方实体（余量<0）返回 None。

    点锚与滑梭分别取冻结分析的 FallCalc / CableResult；
    多连接时取总坠距最大（挂得最低）的控制性连接。
    """
    seq = ctx.seq_by_person[person.id]
    if i >= len(seq.states):
        return None
    state = seq.states[i]
    if state is None:
        return None
    station = ctx.stations[i]
    cands: list[_Hang] = []
    targets = {t for t in state if t is not None}
    for kind, tid in sorted(targets):
        if kind == "anchor":
            fc = ctx.point_falls.get((i, tid))
            if fc is None:
                continue
            if fc.margin_m is not None and fc.margin_m < 0:
                return None                       # 撞击：不属于悬吊
            a = {x.id: x for x in ctx.payload.route.anchors}[tid]
            cands.append(_Hang(
                anchor_pos=a.position.as_tuple(),
                deploy=fc.deployed_length_m,
                feet_z=station[2] - fc.total_fall_m,
                arrest_force_kn=fc.arrest_force_kn,
                total_fall_m=fc.total_fall_m,
                cable_sag_m=0.0,
                target=f"anchor:{tid}", is_shuttle=False,
                original=[f"anchor:{x[1]}" for x in sorted(targets)]))
        else:
            sh = {s.id: s for s in ctx.payload.route.shuttles}[tid]
            spid = sh.span_id
            cr = None
            for c0 in ctx.cable_results:
                if c0.station_index == i and c0.span_id == spid \
                        and c0.falling_persons == [person.id]:
                    cr = c0
                    break
            if cr is None:
                # 退而取含本人的组合（结构包络），保证有冻结分量可用
                for c0 in ctx.cable_results:
                    if c0.station_index == i and c0.span_id == spid \
                            and person.id in c0.falling_persons:
                        cr = c0
                        break
            if cr is None:
                continue
            k = cr.falling_persons.index(person.id)
            total = cr.total_fall_m[k]
            t = ctx.span_paths[spid].table[i]
            dyn = (t.point[0], t.point[1], t.point[2] - cr.sag_m)
            # 净空判定复用通行口径分量
            comp = _shuttle_components(ctx, station, person, eq,
                                       t.point[2], cr.sag_m)
            if comp["margin"] is not None and comp["margin"] < 0:
                return None
            cands.append(_Hang(
                anchor_pos=dyn,
                deploy=eq.lanyard_length_m + eq.elongation_m
                + eq.buffer_travel_m,
                feet_z=station[2] - total,
                arrest_force_kn=cr.loads_kn[k],
                total_fall_m=total, cable_sag_m=cr.sag_m,
                target=f"shuttle:{tid}", is_shuttle=True,
                original=[f"{'shuttle' if x[0] == 'shuttle' else 'anchor'}:{x[1]}"
                          for x in sorted(targets)]))
    if not cands:
        return None
    return max(cands, key=lambda c: c.total_fall_m)


def _shuttle_components(ctx, station, person, eq, anchor_z, sag):
    from . import calc as calc_mod
    return calc_mod.shuttle_fall_components(
        station, person, eq, anchor_z, sag, ctx.payload.route,
        ctx.payload.params)


# ---------------------------------------------------------------- 轨迹净空

def _path_hits(p0: Vec, p1: Vec, radius: float, route) -> list[str]:
    """球体沿直线 p0→p1 移动时与之相交的全部障碍物 id（含担架走廊净空）。"""
    hits: list[str] = []
    d = g.dist3(p0, p1)
    n = max(2, math.ceil(d / 0.5))
    for k in range(n + 1):
        t = k / n
        c = (p0[0] + (p1[0] - p0[0]) * t,
             p0[1] + (p1[1] - p0[1]) * t,
             p0[2] + (p1[2] - p0[2]) * t)
        for ob in route.obstacles:
            if ob.id in hits:
                continue
            if ob.kind == "box":
                hit = g.sphere_box_hit(c, radius, ob.min.as_tuple(),
                                       ob.max.as_tuple())
            else:
                hit = g.sphere_cylinder_hit(c, radius, ob.base.as_tuple(),
                                            ob.radius, ob.height)
            if hit:
                hits.append(ob.id)
    return hits


def _path_obstructed(p0: Vec, p1: Vec, radius: float, route) -> str | None:
    """球体（担架包络外接球）沿直线 p0→p1 移动，返回首个相撞障碍物。"""
    hits = _path_hits(p0, p1, radius, route)
    return hits[0] if hits else None


# ---------------------------------------------------------------- 单站推演

def _simulate_station(ctx, rp: RescuePayload, i: int, person: Person, eq,
                      person_map) -> tuple[StationRescue | None, bool,
                                           set, set]:
    """返回 (StationRescue|None, is_impact, used_anchors, used_shuttles)。"""
    seq = ctx.seq_by_person[person.id]
    state = seq.states[i] if i < len(seq.states) else None
    used_a = {t[1] for t in (state or []) if t and t[0] == "anchor"}
    used_s = {t[1] for t in (state or []) if t and t[0] == "shuttle"}
    if state is None:
        return None, False, set(), set()
    hang = _hang_for(ctx, i, person, eq)
    if hang is None:
        return None, True, used_a, used_s                    # 撞击站（失效侧）

    route = ctx.payload.route
    params = rp.params
    poly = [p.as_tuple() for p in route.walk_polyline]
    station = ctx.stations[i]
    anchors = {a.id: a for a in rp.rescue_anchors}
    ropes = {r.id: r for r in rp.rope_teams}
    entry = rp.entry.position.as_tuple()
    landing = rp.landing_point.as_tuple() if rp.landing_point else entry
    steps: list[RescueStep] = []
    t_cum = 0.0
    primary = backup = main_rope = None

    def add(code, *, travel=0.0, rope_req=0.0, pull=None,
            aid=None, resultant=None, desc=None, clear=True,
            elapsed=0.0, components=None):
        nonlocal t_cum
        t_cum += elapsed
        st = RescueStep(
            step_no=len(steps) + 1, code=code, name=_STEP_NAMES[code],
            rope_travel_m=round(travel, 4), rope_required_m=round(rope_req, 4),
            pull_force_kn=None if pull is None else round(pull, 4),
            anchor_id=aid,
            anchor_resultant_kn=None if resultant is None
            else round(resultant, 4),
            descender_load_kn=None if desc is None else round(desc, 4),
            path_clearance_ok=clear, elapsed_minutes=round(elapsed, 3),
            cumulative_minutes=round(t_cum, 3),
            components=components or {})
        steps.append(st)
        return st

    def block(code_code, message, components, step_code=None):
        sc = step_code or (steps[-1].code if steps else "approach")
        no = len(steps) if steps else 1
        raise _Block(no, sc, code_code, message, components)

    def time_check(code_code="time_limit"):
        if t_cum > rp.max_suspension_minutes + 1e-9:
            block(code_code,
                  f"累计悬吊作业时间 {round(t_cum, 1)} min 超过最大悬吊时间 "
                  f"{rp.max_suspension_minutes} min",
                  {"cumulative_minutes": round(t_cum, 3),
                   "max_suspension_minutes": rp.max_suspension_minutes})

    # 伤员 + 担架工作载荷（kN）
    load = ((person.weight_kg + rp.stretcher.weight_kg)
            * ctx.payload.params.gravity / 1000.0)
    hang_pt = hang.anchor_pos
    # 伤员 D 环悬挂位置：动态锚点正下方、展开绳长处
    d_hang = (hang_pt[0], hang_pt[1], hang_pt[2] - hang.deploy)
    feet0 = hang.feet_z
    # 提拉/转运轨迹走廊内影响担架净空的障碍物（半径 0：走廊几何，
    # 担架包络半径在逐步净空检查中另计）
    corridor: set[str] = set()

    try:
        # ---- 步骤 1：接近 ----------------------------------------------
        off, entry_s, entry_pt = _project_polyline(entry, poly)
        if off > params.entry_max_offset_m + 1e-9:
            raise _Block(
                1, "approach", "entry_off_route",
                f"救援入口距行走折线 {round(off, 2)} m 超过容许 "
                f"{params.entry_max_offset_m} m，无法进入路线接近伤员",
                {"offset_m": round(off, 3),
                 "entry_max_offset_m": params.entry_max_offset_m})
        # 入口至折线投影点的三维直线段 + 沿折线到站点的路程
        entry_diag = g.dist3(entry, entry_pt)
        _st_off, st_s, _st_pt = _project_polyline(station, poly)
        approach_dist = entry_diag + abs(st_s - entry_s)
        add("approach", travel=approach_dist,
            elapsed=approach_dist / params.approach_speed_m_min,
            components={
                "entry_id": rp.entry.id,
                "approach_distance_m": round(approach_dist, 3),
                "approach_speed_m_min": params.approach_speed_m_min,
                "feet_z": round(feet0, 3),
                "original_arrest_force_kn": round(hang.arrest_force_kn, 3),
                "cable_sag_m": round(hang.cable_sag_m, 4),
                "original_system": hang.target})
        time_check()

        # ---- 主救援锚点选择（自动 / 人工） ------------------------------
        def reachable_ra(ra):
            d = g.dist3(station, ra.position.as_tuple())
            return d <= params.anchor_rig_reach_m + 1e-9, d

        def choose_primary():
            if rp.primary_anchor_id is not None:
                ra = anchors[rp.primary_anchor_id]
                d = g.dist3(station, ra.position.as_tuple())
                if d > params.anchor_rig_reach_m + 1e-9:
                    raise _Block(
                        2, "secondary_protection", "rescue_anchor_unreachable",
                        f"人工指定的主救援锚点 {ra.id} 距站点 {round(d, 2)} m，"
                        f"超出可挂接距离 {params.anchor_rig_reach_m} m",
                        {"anchor": ra.id, "distance_m": round(d, 3),
                           "anchor_rig_reach_m": params.anchor_rig_reach_m})
                return ra
            cand = [(g.dist3(station, ra.position.as_tuple()), ra)
                    for ra in rp.rescue_anchors
                    if reachable_ra(ra)[0]]
            if not cand:
                dmin = min(g.dist3(station, ra.position.as_tuple())
                           for ra in rp.rescue_anchors)
                raise _Block(
                    2, "secondary_protection", "rescue_anchor_unreachable",
                    f"站点 {i} 附近 {params.anchor_rig_reach_m} m 内无可用"
                    f"救援锚点（最近 {round(dmin, 2)} m）",
                    {"nearest_distance_m": round(dmin, 3),
                     "anchor_rig_reach_m": params.anchor_rig_reach_m})
            cand.sort(key=lambda x: x[0])
            return cand[0][1]

        primary = choose_primary()
        # 二次保护锚点：自动取另一最近可达点锚；与主锚不得相同
        def choose_backup():
            if rp.backup_anchor_id is not None:
                ra = anchors[rp.backup_anchor_id]
                if ra.id == primary.id:
                    raise _Block(
                        2, "secondary_protection", "equipment_reuse",
                        "二次保护锚点与主救援锚点为同一锚点（器材重复占用）",
                        {"anchor": ra.id})
                d = g.dist3(station, ra.position.as_tuple())
                if d > params.anchor_rig_reach_m + 1e-9:
                    raise _Block(
                        2, "secondary_protection", "rescue_anchor_unreachable",
                        f"人工指定的二次保护锚点 {ra.id} 距站点 "
                        f"{round(d, 2)} m，超出可挂接距离",
                        {"anchor": ra.id, "distance_m": round(d, 3),
                         "anchor_rig_reach_m": params.anchor_rig_reach_m})
                return ra
            cand = [(g.dist3(station, ra.position.as_tuple()), ra)
                    for ra in rp.rescue_anchors
                    if ra.id != primary.id and reachable_ra(ra)[0]]
            if not cand:
                raise _Block(
                    2, "secondary_protection", "rescue_anchor_unreachable",
                    f"主锚点 {primary.id} 之外无第二个可达救援锚点"
                    f"建立二次保护",
                    {"primary_anchor": primary.id,
                     "anchor_rig_reach_m": params.anchor_rig_reach_m})
            cand.sort(key=lambda x: x[0])
            return cand[0][1]

        backup = choose_backup()

        # ---- 步骤 2：二次保护 -------------------------------------------
        sec_rope = ropes[rp.secondary_rope_team_id]
        # 二次保护路径：备份锚点 → 伤员悬挂位置
        sec_need = g.dist3(backup.position.as_tuple(), d_hang) \
            + params.rope_tail_m
        if sec_rope.rope_length_m < sec_need - 1e-9:
            raise _Block(
                2, "secondary_protection", "rope_too_short",
                f"二次保护绳组 {sec_rope.id} 长 {sec_rope.rope_length_m} m，"
                f"不足以从备份锚点 {backup.id} 连接伤员（需 "
                f"{round(sec_need, 2)} m，含尾绳 {params.rope_tail_m} m）",
                {"rope_team": sec_rope.id,
                 "rope_length_m": sec_rope.rope_length_m,
                 "required_m": round(sec_need, 3),
                 "backup_anchor": backup.id})
        # 备份锚点合力：预张力
        if params.secondary_pretension_kn > backup.rated_load_kn + 1e-9:
            raise _Block(
                2, "secondary_protection", "anchor_overload",
                f"二次保护锚点 {backup.id} 额定 {backup.rated_load_kn} kN "
                f"低于预张力 {params.secondary_pretension_kn} kN",
                {"anchor": backup.id,
                 "rated_load_kn": backup.rated_load_kn,
                 "resultant_kn": params.secondary_pretension_kn})
        add("secondary_protection", travel=sec_need - params.rope_tail_m,
            rope_req=sec_need, resultant=params.secondary_pretension_kn,
            aid=backup.id,
            elapsed=params.rig_secondary_min,
            components={"backup_anchor": backup.id,
                        "secondary_rope_team": sec_rope.id,
                        "rope_length_m": sec_rope.rope_length_m,
                        "required_m": round(sec_need, 3),
                        "pretension_kn": params.secondary_pretension_kn,
                        "rated_load_kn": backup.rated_load_kn})
        time_check()

        # ---- 主绳组选择（自动：各候选按自身倍率/效率核算牵引力，
        #      可行者取绳长最短；人工指定须写理由） --------------------
        def eff_ma(rt):
            return rt.pulley_ratio * rt.pulley_efficiency

        def candidate_pull(rt):
            # 该绳组自身滑轮倍率与效率下的人工牵引力
            return load / max(eff_ma(rt), 1e-9)

        def choose_rope():
            if rp.rope_team_id is not None:
                rt = ropes[rp.rope_team_id]
                if rt.id == sec_rope.id:
                    raise _Block(
                        3, "haul_unload", "equipment_reuse",
                        f"主绳组与二次保护绳组同为 {rt.id}（器材重复占用）",
                        {"rope_team": rt.id})
                return rt
            eligible = [rt for rt in rp.rope_teams
                        if rt.id != sec_rope.id]
            cand = [rt for rt in eligible
                    if candidate_pull(rt) <= params.rescuer_pull_kn + 1e-9]
            if not cand:
                return None
            cand.sort(key=lambda rt: (rt.rope_length_m, -eff_ma(rt)))
            return cand[0]

        # ---- 步骤 3：提拉卸载 -------------------------------------------
        # 提升量：把伤员从悬挂脚底提至站面（+担架半高进入包络）
        stretcher_half = rp.stretcher.height_m / 2.0
        lift = max(0.0, station[2] - feet0) + stretcher_half
        rope_travel = lift
        main_rope = choose_rope()
        if main_rope is None:
            eligible = [rt for rt in rp.rope_teams
                        if rt.id != sec_rope.id]
            pulls = {rt.id: round(candidate_pull(rt), 3)
                     for rt in eligible}
            best_rt = min(eligible, key=candidate_pull)
            raise _Block(
                3, "haul_unload", "manual_pull_exceeded",
                f"伤员+担架载荷 {round(load, 2)} kN，各候选绳组按自身倍率"
                f"计算的人工牵引力均超过 {params.rescuer_pull_kn} kN，"
                f"无法人工提拉",
                {"load_kn": round(load, 3),
                 "rescuer_pull_kn": params.rescuer_pull_kn,
                 "candidate_pulls_kn": ",".join(
                     f"{k}:{v}" for k, v in sorted(pulls.items())),
                 "best_mechanical_advantage": round(eff_ma(best_rt), 3),
                 "pull_force_kn": round(candidate_pull(best_rt), 3),
                 "rescuer_count": len(rp.rescuers)})
        ma = eff_ma(main_rope)
        pull = load / ma
        if pull > params.rescuer_pull_kn + 1e-9:
            raise _Block(
                3, "haul_unload", "manual_pull_exceeded",
                f"绳组 {main_rope.id} 有效倍率 {round(ma, 2)}，牵引力 "
                f"{round(pull, 2)} kN 超过单人持续牵引力 "
                f"{params.rescuer_pull_kn} kN",
                {"rope_team": main_rope.id,
                 "pulley_ratio": main_rope.pulley_ratio,
                 "pulley_efficiency": main_rope.pulley_efficiency,
                 "mechanical_advantage": round(ma, 3),
                 "pull_force_kn": round(pull, 3),
                 "rescuer_pull_kn": params.rescuer_pull_kn,
                 "load_kn": round(load, 3)})
        if pull > main_rope.descender_limit_kn + 1e-9:
            raise _Block(
                3, "haul_unload", "descender_overload",
                f"绳组 {main_rope.id} 下降器工作载荷 {round(pull, 2)} kN "
                f"超过限载 {main_rope.descender_limit_kn} kN",
                {"rope_team": main_rope.id,
                 "working_load_kn": round(pull, 3),
                 "descender_limit_kn": main_rope.descender_limit_kn})
        # 绳程：n 倍提升量 + 尾绳
        haul_rope = main_rope.pulley_ratio * lift + params.rope_tail_m
        if main_rope.rope_length_m < haul_rope - 1e-9:
            raise _Block(
                3, "haul_unload", "rope_too_short",
                f"主绳组 {main_rope.id} 长 {main_rope.rope_length_m} m，"
                f"提拉 {round(lift, 2)} m 需要绳程 {round(haul_rope, 2)} m"
                f"（{main_rope.pulley_ratio} 倍行程 + 尾绳），绳长不足",
                {"rope_team": main_rope.id,
                 "rope_length_m": main_rope.rope_length_m,
                 "lift_m": round(lift, 3),
                 "required_m": round(haul_rope, 3)})
        # 主锚合力：提拉荷载 + 尾绳牵引力（保守同向下加）
        anchor_f = load + pull
        if anchor_f > primary.rated_load_kn + 1e-9:
            raise _Block(
                3, "haul_unload", "anchor_overload",
                f"主救援锚点 {primary.id} 合力 {round(anchor_f, 2)} kN "
                f"超过额定载荷 {primary.rated_load_kn} kN",
                {"anchor": primary.id, "resultant_kn": round(anchor_f, 3),
                 "rated_load_kn": primary.rated_load_kn,
                 "load_kn": round(load, 3),
                 "pull_force_kn": round(pull, 3)})
        # 伤员轨迹净空：悬挂点 → 提拉至锚点下方可转运高度
        top_pt = (d_hang[0], d_hang[1], feet0 + lift)
        corridor.update(_path_hits(d_hang, top_pt,
                                   person.body_radius_m, route))
        hit = _path_obstructed(d_hang, top_pt,
                               person.body_radius_m, route)
        if hit is not None:
            raise _Block(
                3, "haul_unload", "casualty_path_blocked",
                f"提拉路径上伤员与障碍物 {hit} 净空冲突，无法垂直提拉卸载",
                {"obstacle_id": hit, "lift_m": round(lift, 3),
                 "body_radius_m": person.body_radius_m})
        add("haul_unload", travel=haul_rope - params.rope_tail_m,
            rope_req=haul_rope, pull=pull, aid=primary.id,
            resultant=anchor_f, desc=pull, clear=hit is None,
            elapsed=params.rig_haul_min
            + main_rope.pulley_ratio * lift / params.haul_rate_m_min,
            components={
                "primary_anchor": primary.id,
                "rope_team": main_rope.id,
                "pulley_ratio": main_rope.pulley_ratio,
                "pulley_efficiency": main_rope.pulley_efficiency,
                "mechanical_advantage": round(ma, 3),
                "load_kn": round(load, 3),
                "lift_m": round(lift, 3),
                "rope_travel_m":
                    round(main_rope.pulley_ratio * lift, 3),
                "rated_load_kn": primary.rated_load_kn})
        time_check()

        # ---- 步骤 4：脱离原系统 ------------------------------------------
        n_conn = len(hang.original)
        det_elapsed = params.detach_min_per_connection * n_conn
        add("detach_original",
            elapsed=det_elapsed, aid=primary.id, resultant=anchor_f,
            components={"released_connections": ",".join(hang.original),
                        "connection_count": n_conn,
                        "load_transferred_to": f"rescue_anchor:{primary.id}",
                        "primary_anchor": primary.id})
        time_check()

        # ---- 步骤 5：升降转运 --------------------------------------------
        # 下降器工作载荷即全载荷；绳程 = 锚点到落点三维距离 + 尾绳
        lower_dist = g.dist3(primary.position.as_tuple(), landing)
        # 实际移送：伤员自锚点正下方转运至 landing（以担架外接球扫掠）
        move_from = (d_hang[0], d_hang[1], feet0 + lift)
        move_to = landing
        move_dist = g.dist3(move_from, move_to)
        if main_rope.descender_limit_kn < load + 1e-9:
            raise _Block(
                5, "transfer", "descender_overload",
                f"转运时下降器承受全载荷 {round(load, 2)} kN，超过绳组 "
                f"{main_rope.id} 下降器限载 {main_rope.descender_limit_kn} kN",
                {"rope_team": main_rope.id, "load_kn": round(load, 3),
                 "descender_limit_kn": main_rope.descender_limit_kn})
        trans_rope = lower_dist + params.rope_tail_m
        if main_rope.rope_length_m < trans_rope - 1e-9:
            raise _Block(
                5, "transfer", "rope_too_short",
                f"转运到落点需绳程 {round(trans_rope, 2)} m，主绳组 "
                f"{main_rope.id} 长 {main_rope.rope_length_m} m，绳长不足",
                {"rope_team": main_rope.id,
                 "rope_length_m": main_rope.rope_length_m,
                 "lower_distance_m": round(lower_dist, 3),
                 "required_m": round(trans_rope, 3)})
        # 锚点合力：荷载与尾绳持力的向量合成（保守同向下界用勾股）
        trans_f = math.hypot(load, params.descender_hold_kn)
        if trans_f > primary.rated_load_kn + 1e-9:
            raise _Block(
                5, "transfer", "anchor_overload",
                f"转运时主锚点 {primary.id} 合力 {round(trans_f, 2)} kN "
                f"超过额定载荷 {primary.rated_load_kn} kN",
                {"anchor": primary.id, "resultant_kn": round(trans_f, 3),
                 "rated_load_kn": primary.rated_load_kn})
        radius = rp.stretcher.bounding_radius_m()
        corridor.update(_path_hits(move_from, move_to, radius, route))
        hit = _path_obstructed(move_from, move_to, radius, route)
        if hit is not None:
            raise _Block(
                5, "transfer", "casualty_path_blocked",
                f"担架（外接球半径 {round(radius, 2)} m）转运轨迹与障碍物 "
                f"{hit} 净空冲突",
                {"obstacle_id": hit, "stretcher_envelope_radius_m":
                    round(radius, 3),
                 "move_distance_m": round(move_dist, 3)})
        # 落点必须低于锚点（可下放）或等于站面
        if landing[2] > primary.position.z + 1e-6:
            raise _Block(
                5, "transfer", "landing_unreachable",
                f"落点标高 {landing[2]} m 高于主锚点 "
                f"{primary.position.z} m，无法下放转运",
                {"landing_z": landing[2],
                 "anchor_z": primary.position.z})
        add("transfer", travel=lower_dist, rope_req=trans_rope,
            pull=params.descender_hold_kn, aid=primary.id,
            resultant=trans_f, desc=load,
            elapsed=params.rig_transfer_min
            + max(move_dist, 0.0) / params.lower_rate_m_min,
            components={
                "primary_anchor": primary.id,
                "rope_team": main_rope.id,
                "lower_distance_m": round(lower_dist, 3),
                "move_distance_m": round(move_dist, 3),
                "landing_point": f"({landing[0]},{landing[1]},{landing[2]})",
                "stretcher_envelope_m":
                    f"{rp.stretcher.length_m}x{rp.stretcher.width_m}"
                    f"x{rp.stretcher.height_m}",
                "descender_load_kn": round(load, 3),
                "descender_limit_kn": main_rope.descender_limit_kn,
                "descender_hold_kn": params.descender_hold_kn})
        time_check()

    except _Block as b:
        res = StationRescue(
            station_index=i,
            position=Vec3(x=station[0], y=station[1], z=station[2]),
            person_id=person.id,
            hang_position=Vec3(x=d_hang[0], y=d_hang[1], z=d_hang[2]),
            feet_z=round(feet0, 4),
            original_connections=hang.original,
            primary_anchor_id=primary.id if primary else None,
            backup_anchor_id=backup.id if backup else None,
            rope_team_id=main_rope.id if main_rope else None,
            secondary_rope_team_id=rp.secondary_rope_team_id,
            executable=False,
            block=RescueBlock(
                station_index=i, person_id=person.id,
                step_no=b.step_no, step_code=b.step_code,
                code=b.code, message=b.message, components=b.components),
            steps=steps, corridor_obstacles=sorted(corridor),
            total_elapsed_minutes=round(t_cum, 3))
        return res, False, used_a, used_s

    res = StationRescue(
        station_index=i,
        position=Vec3(x=station[0], y=station[1], z=station[2]),
        person_id=person.id,
        hang_position=Vec3(x=d_hang[0], y=d_hang[1], z=d_hang[2]),
        feet_z=round(feet0, 4),
        original_connections=hang.original,
        primary_anchor_id=primary.id,
        backup_anchor_id=backup.id,
        rope_team_id=main_rope.id,
        secondary_rope_team_id=rp.secondary_rope_team_id,
        executable=True, steps=steps,
        corridor_obstacles=sorted(corridor),
        total_elapsed_minutes=round(t_cum, 3))
    return res, False, used_a, used_s


# ---------------------------------------------------------------- 整条路线

def simulate_rescue(ctx, rp: RescuePayload) -> RescueResult:
    """对每个可导致悬吊的站点逐步推演；任一受阻即方案不可执行。"""
    payload = ctx.payload
    equipment = {e.id: e for e in payload.equipment}
    person_map = {p.id: p for p in payload.persons}

    # 来源分析无结论：坠距/反力未冻结，不得开展逐站推演，
    # 直接在“接近”步骤整体受阻，方案不得标为可执行。
    if not ctx.result.conclusive:
        oi = ctx.result.first_open_item
        block = RescueBlock(
            station_index=oi.station_index if oi and oi.station_index is not None
            else 0,
            person_id=oi.person_id if oi else None,
            step_no=1, step_code="approach",
            code="source_inconclusive",
            message=f"来源修订分析无结论（{oi.code if oi else 'open_item'}："
                    f"{oi.message if oi else ''}），坠距/反力未冻结，"
                    f"不得开展救援推演并给出可执行结论",
            components={"open_item": oi.code if oi else "",
                        "open_item_station":
                            oi.station_index if oi else None})
        return RescueResult(
            source_conclusive=False, source_passable=ctx.result.passable,
            executable=False, earliest_block=block, simulations=[],
            suspension_station_count=0, impact_stations=[],
            manual_decisions=_manual_decisions(rp))

    sims: list[StationRescue] = []
    impacts = []
    used_anchors: set[str] = set()
    used_shuttles: set[str] = set()
    used_persons: set[str] = set()

    # 每个可导致悬吊的站点（每人）生成一次推演
    for person in payload.persons:
        eq = equipment[person.equipment_id]
        for i in range(len(ctx.stations)):
            sr, is_impact, ua, us = _simulate_station(
                ctx, rp, i, person, eq, person_map)
            used_anchors |= ua
            used_shuttles |= us
            if sr is not None:
                used_persons.add(person.id)
                sims.append(sr)
            if is_impact:
                fc_total = None
                state = ctx.seq_by_person[person.id].states[i]
                zs = []
                for t in state:
                    if t and t[0] == "anchor":
                        f0 = ctx.point_falls.get((i, t[1]))
                        if f0 is not None:
                            zs.append(f0.total_fall_m)
                if zs:
                    impacts.append({"station_index": i,
                                    "person_id": person.id,
                                    "total_fall_m": round(max(zs), 4),
                                    "reason": "clearance_impact"})

    # 最早受阻步骤：站点序号 → 步骤序号 → 人员
    blocked = [s for s in sims if not s.executable]
    earliest = None
    if blocked:
        blocked.sort(key=lambda s: (s.station_index, s.block.step_no,
                                    s.person_id))
        earliest = blocked[0].block
    elif impacts:
        # 存在撞击站点（坠落后直接撞及下方实体，伤员不处于可悬吊转移状态）：
        # 该来源路线下本方案不得标为可执行
        im = min(impacts, key=lambda x: (x["station_index"],
                                         str(x["person_id"])))
        earliest = RescueBlock(
            station_index=int(im["station_index"]),
            person_id=str(im["person_id"]), step_no=1,
            step_code="approach", code="casualty_impact",
            message=f"站 {im['station_index']} 伤员 {im['person_id']} "
                    f"坠落总坠距 {im['total_fall_m']} m 已撞及下方实体，"
                    f"不存在可悬吊转移状态，救援方案不可执行",
            components={"total_fall_m": float(im["total_fall_m"]),
                        "reason": str(im["reason"])})

    return RescueResult(
        source_conclusive=ctx.result.conclusive,
        source_passable=ctx.result.passable,
        executable=earliest is None,
        earliest_block=earliest,
        simulations=sims,
        suspension_station_count=len(sims),
        impact_stations=impacts,
        manual_decisions=_manual_decisions(rp),
        used_source_anchors=sorted(used_anchors),
        used_source_shuttles=sorted(used_shuttles),
        used_persons=sorted(used_persons))


def _manual_decisions(rp: RescuePayload) -> list[ManualDecision]:
    manual: list[ManualDecision] = []
    if rp.primary_anchor_id:
        manual.append(ManualDecision(field="primary_anchor_id",
                                     value=rp.primary_anchor_id,
                                     reason=rp.manual_reason))
    if rp.backup_anchor_id:
        manual.append(ManualDecision(field="backup_anchor_id",
                                     value=rp.backup_anchor_id,
                                     reason=rp.manual_reason))
    if rp.rope_team_id:
        manual.append(ManualDecision(field="rope_team_id",
                                     value=rp.rope_team_id,
                                     reason=rp.manual_reason))
    return manual


# ---------------------------------------------------------------- 相关性签名

def source_relevance_signature(payload: PlanPayload,
                               result: RescueResult,
                               rp: RescuePayload | None = None) -> tuple:
    """来源修订中与本救援方案相关的部分：方案所用人/锚/滑梭/跨段参数、
    悬挂点几何与担架轨迹走廊内障碍物。仅改步距（站点重新编号、悬挂几何
    不变）不触发复核；走廊内新增/修改障碍物（含只影响担架包络者）触发。"""
    pids = set(result.used_persons)
    aids = set(result.used_source_anchors)
    sids = set(result.used_source_shuttles)
    span_ids = {s.span_id for s in payload.route.shuttles
                if s.id in sids}
    persons = [(p.id, p.weight_kg, p.equipment_id, p.d_ring_height_m,
                p.body_radius_m)
               for p in payload.persons if p.id in pids]
    eq_ids = {p.equipment_id for p in payload.persons if p.id in pids}
    def _eq_sig(e):
        if e.twin_leg is None:
            return (e.id, e.lanyard_length_m, e.elongation_m,
                    e.buffer_travel_m, e.max_arrest_force_kn, None)
        t = e.twin_leg
        return (e.id, e.lanyard_length_m, e.elongation_m,
                e.buffer_travel_m, e.max_arrest_force_kn,
                (tuple((l.id, l.hook, l.leg_length_m, l.axial_stiffness_kn,
                        l.connector_side_load_limit_kn) for l in t.legs),
                 tuple((p.travel_m, p.force_kn) for p in t.buffer_curve),
                 t.max_travel_m, t.energy_capacity_j,
                 t.max_included_angle_deg))
    equip = [_eq_sig(e) for e in payload.equipment if e.id in eq_ids]
    anchors = [(a.id, a.position.model_dump(), a.rated_load_kn)
               for a in payload.route.anchors if a.id in aids]
    shuttles = [(s.id, s.span_id, s.connector_reach_m, s.can_pass)
                for s in payload.route.shuttles if s.id in sids]
    spans = [(s.id, s.supports, s.pretension_kn, s.line_density_kg_m,
              s.axial_stiffness_kn, s.max_sag_m)
             for s in payload.route.spans if s.id in span_ids]
    supports = [(s.id, s.position.model_dump(), s.rated_load_kn)
                for s in payload.route.supports
                if any(s.id in sp.supports for sp in payload.route.spans
                       if sp.id in span_ids)]
    # 悬挂点与脚底标高的几何集合（不含站号）：步距变化但几何覆盖不变时
    # 集合相同（取整到 1 mm 抵消离散误差）
    hang_geom = tuple(sorted({(
        round(s.hang_position.x, 3), round(s.hang_position.y, 3),
        round(s.hang_position.z, 3), round(s.feet_z, 3))
        for s in result.simulations}))
    # 担架轨迹走廊障碍物：以推演收集到的走廊 id 集合为准（两侧分别按各自
    # 路线/方案重算），纳入这些障碍物的几何与额定参数。新增只影响担架
    # 包络（不影响人员坠落）的障碍物即在此处改变签名。
    corridor_ids = {oid for s in result.simulations
                    for oid in s.corridor_obstacles}
    obstacles = tuple(sorted(
        _obstacle_sig(ob) for ob in payload.route.obstacles
        if ob.id in corridor_ids))
    return (persons, equip, anchors, shuttles, spans, supports, hang_geom,
            obstacles)


def _obstacle_sig(ob) -> tuple:
    if ob.kind == "box":
        return (ob.id, "box", ob.min.model_dump(), ob.max.model_dump())
    return (ob.id, "cylinder", ob.base.model_dump(), ob.radius, ob.height)
