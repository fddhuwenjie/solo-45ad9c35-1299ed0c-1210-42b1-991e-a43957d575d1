"""Y 型双腿系绳载荷分配：能量法 + 变形协调求解。

物理模型（质量仅集中于人体；汇接器/绳腿无质量）：

- 两条实体腿从各自挂点（点锚或滑梭动态位置）汇至汇接器 J，J 下方经**一个**
  共享缓冲包（展开量 s、展开阻力 F(s)，由力—行程曲线分段线性给出）连接人体；
- 人体沿竖直方向坠落：人体 B = J + (0,0,−s)；
- 每腿张力 T_i = k_i·max(|J−A_i| − L0_i, 0)（只拉不压）；
- 汇接器三向平衡：Σ T_i·u_i = (0,0,F)；
- 能量（同一缓冲能力只计一次）：

      m·g·h(s) = ∫₀^s F(u) du  +  Σ T_i(s)²/(2k_i)

  外层求能量方程的根 s*（缓冲展开量），内层对给定 s 求汇接器平衡（主动集
  Newton）。先张紧的腿单独承拉（解析解），第二腿张紧后切换为双腿 Newton
  解；接载次序按各腿竖直坠落路径上的张紧时刻排序。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..models import BufferCurvePoint, LanyardLeg

Vec = tuple[float, float, float]


# ---------------------------------------------------------------- 数据结构

@dataclass
class LegTarget:
    """一条实体腿的求解输入（含其绑定挂点的当前坐标）。"""
    leg_id: str
    hook: str
    target_kind: str          # 'anchor' | 'shuttle'
    target_id: str
    anchor: Vec
    length: float
    stiffness: float
    side_limit: float


@dataclass
class LegOutcome:
    leg_id: str
    hook: str
    target_kind: str
    target_id: str
    anchor: Vec
    taut: bool
    tension_kn: float
    extension_m: float
    vertical_component_kn: float
    horizontal_component_kn: float
    side_load_kn: float
    side_limit_kn: float


@dataclass
class TwinSolution:
    converged: bool
    engagement_order: list[str] = field(default_factory=list)
    legs: list[LegOutcome] = field(default_factory=list)
    junction: Vec = (0.0, 0.0, 0.0)
    deployment_m: float = 0.0
    buffer_force_kn: float = 0.0
    drop_m: float = 0.0                 # 人体自站位到峰值制动力时刻的总坠落 h*
    free_fall_m: float = 0.0            # 首腿张紧前的自由坠距 h1
    energy_demand_j: float = 0.0
    buffer_energy_j: float = 0.0
    elastic_energy_j: float = 0.0
    demand_at_full_j: float = 0.0       # 完全展开时的需吸收能量（能量不足上报）
    reason: str | None = None           # nonconvergence 时的原因说明


# ---------------------------------------------------------------- 曲线

def curve_force(curve: list[BufferCurvePoint], s: float) -> float:
    """分段线性插值缓冲阻力（kN）；s 钳到曲线覆盖范围。"""
    if s <= curve[0].travel_m:
        return curve[0].force_kn
    if s >= curve[-1].travel_m:
        return curve[-1].force_kn
    for k in range(len(curve) - 1):
        s0, s1 = curve[k].travel_m, curve[k + 1].travel_m
        if s0 <= s <= s1:
            t = (s - s0) / (s1 - s0)
            f0, f1 = curve[k].force_kn, curve[k + 1].force_kn
            return f0 + (f1 - f0) * t
    return curve[-1].force_kn


def curve_energy_j(curve: list[BufferCurvePoint], s: float) -> float:
    """0→s 曲线下面积（梯形积分，分段线性精确），单位 J（kN·m×1000）。"""
    s = min(max(s, 0.0), curve[-1].travel_m)
    area = 0.0
    for k in range(len(curve) - 1):
        s0, s1 = curve[k].travel_m, curve[k + 1].travel_m
        f0, f1 = curve[k].force_kn, curve[k + 1].force_kn
        if s <= s0:
            break
        seg = min(s, s1) - s0
        if seg > 0:
            area += 0.5 * (f0 + (f0 + (f1 - f0) * seg / (s1 - s0))) * seg
    return area * 1000.0


# ---------------------------------------------------------------- 几何工具

def _dist(a: Vec, b: Vec) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def _unit(v: tuple[float, float, float]):
    n = math.sqrt(sum(x * x for x in v))
    if n < 1e-12:
        return None
    return tuple(x / n for x in v)


def _horizontal(a: Vec, b: Vec) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def engagement_drop(leg: LegTarget, d0: Vec, dz0: float) -> float | None:
    """人体沿竖直路径坠落时该腿的张紧坠落量；水平投影已超过腿长则永不张紧。"""
    H = _horizontal(leg.anchor, d0)
    if H > leg.length + 1e-9:
        return None
    g = math.sqrt(max(0.0, leg.length ** 2 - H ** 2))
    # 站立时已处于张拉/超程（借助连接器余量挂上）：立即接载
    if _dist(leg.anchor, d0) >= leg.length - 1e-9:
        return 0.0
    # 张紧时人体标高 = anchor_z − g，坠距 = dz0 − (anchor_z − g)
    return dz0 - leg.anchor[2] + g


# ---------------------------------------------------------------- 单腿解析

def _single_leg_state(leg: LegTarget, d0: Vec, dz0: float, s: float,
                      F: float) -> tuple[Vec, float, float] | None:
    """仅一条承拉腿时的解析状态：返回 (J, h, δ)。

    缓冲与绳腿共线，T=F；人体在挂点竖直面内，水平投影固定为 H。
    """
    H = _horizontal(leg.anchor, d0)
    delta = F / leg.stiffness
    L = leg.length + delta
    if L + 1e-9 < H:
        return None
    g = math.sqrt(max(0.0, L ** 2 - H ** 2))
    J = (d0[0], d0[1], leg.anchor[2] - g)
    # 人体 B = J − (0,0,s)，坠距 h = dz0 − B_z = dz0 − anchor_z + g + s
    h = dz0 - leg.anchor[2] + g + s
    return J, h, delta


def _leg_outcome(leg: LegTarget, J: Vec, taut: bool) -> LegOutcome:
    d = _dist(leg.anchor, J)
    ext = max(0.0, d - leg.length) if taut else 0.0
    T = leg.stiffness * ext if taut else 0.0
    u = _unit((J[0] - leg.anchor[0], J[1] - leg.anchor[1],
               J[2] - leg.anchor[2]))
    if u is None:
        u = (0.0, 0.0, -1.0)
    Tv = T * abs(u[2])          # 腿张力的竖直分量（对锚点的竖直负荷）
    Th = T * math.hypot(u[0], u[1])
    # 连接器侧载约定：取绳腿张力的水平分量（横向加载），与侧载限值比较
    return LegOutcome(
        leg_id=leg.leg_id, hook=leg.hook, target_kind=leg.target_kind,
        target_id=leg.target_id, anchor=leg.anchor, taut=taut,
        tension_kn=T, extension_m=ext, vertical_component_kn=Tv,
        horizontal_component_kn=Th, side_load_kn=Th,
        side_limit_kn=leg.side_limit)


# ---------------------------------------------------------------- 双腿平衡 Newton

def _residual(x, s: float, F: float, d0: Vec, dz0: float,
              active: list[LegTarget]):
    jx, jy, jz, h = x
    J = (jx, jy, jz)
    rx = ry = rz = 0.0
    for leg in active:
        v = (leg.anchor[0] - jx, leg.anchor[1] - jy, leg.anchor[2] - jz)
        d = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        if d < 1e-12:
            return None
        T = leg.stiffness * max(0.0, d - leg.length)
        rx += T * v[0] / d
        ry += T * v[1] / d
        rz += T * v[2] / d
    rz -= F                      # 缓冲包对汇接器的向下拉力
    # 人体 B = J − (0,0,s)，坠距 h = dz0 − B_z
    r4 = dz0 + s - jz - h
    return (rx, ry, rz, r4)


def _solve4(s: float, F: float, d0: Vec, dz0: float,
            active: list[LegTarget], guess, *, max_iter: int = 40,
            tol: float = 1e-9) -> tuple[Vec, float] | None:
    """给定主动集解 4×4 非线性平衡（数值雅可比 + 选主元高斯消元）。"""
    x = list(guess)
    eps = 1e-7
    for _ in range(max_iter):
        r0 = _residual(x, s, F, d0, dz0, active)
        if r0 is None:
            return None
        norm = math.sqrt(sum(v * v for v in r0))
        if norm <= tol:
            return (x[0], x[1], x[2]), x[3]
        J = [[0.0] * 4 for _ in range(4)]
        for j in range(4):
            xp = list(x)
            step = eps * max(1.0, abs(xp[j]))
            xp[j] += step
            rp = _residual(xp, s, F, d0, dz0, active)
            if rp is None:
                return None
            for i in range(4):
                J[i][j] = (rp[i] - r0[i]) / step
        # 选主元高斯消元解 J·dx = −r
        b = [-v for v in r0]
        for col in range(4):
            piv = max(range(col, 4), key=lambda r: abs(J[r][col]))
            if abs(J[piv][col]) < 1e-14:
                return None
            if piv != col:
                J[col], J[piv] = J[piv], J[col]
                b[col], b[piv] = b[piv], b[col]
            for r in range(col + 1, 4):
                fct = J[r][col] / J[col][col]
                for c in range(col, 4):
                    J[r][c] -= fct * J[col][c]
                b[r] -= fct * b[col]
        dx = [0.0] * 4
        for r in range(3, -1, -1):
            dx[r] = (b[r] - sum(J[r][c] * dx[c] for c in range(r + 1, 4))) \
                / J[r][r]
        # 阻尼线搜索，避免步长跨过锚点
        alpha = 1.0
        for _ls in range(12):
            xn = [x[k] + alpha * dx[k] for k in range(4)]
            rn = _residual(xn, s, F, d0, dz0, active)
            if rn is not None and math.sqrt(sum(v * v for v in rn)) < norm:
                x = xn
                break
            alpha *= 0.5
        else:
            return None
    r = _residual(x, s, F, d0, dz0, active)
    if r is None or math.sqrt(sum(v * v for v in r)) > 1e-6:
        return None
    return (x[0], x[1], x[2]), x[3]


# ---------------------------------------------------------------- 主求解

def solve_twin_legs(person_weight_kg: float, gravity: float,
                    d_ring: Vec, legs: list[LegTarget],
                    curve: list[BufferCurvePoint], max_travel: float
                    ) -> TwinSolution:
    """能量 + 变形协调求解双腿系绳在一次竖直坠落中的峰值分量。

    遍历缓冲展开量：先按接载次序的首腿解析解推进，第二腿张紧后切换双腿
    Newton；能量方程变号即二分求根。完全展开仍为正能量亏缺 → 曲线能量不足。
    """
    W = person_weight_kg * gravity                 # N
    dz0 = d_ring[2]

    # ---- 接载次序：竖直坠落路径上的张紧时刻 ----------------------------
    engage: list[tuple[float, LegTarget]] = []
    for leg in legs:
        h_e = engagement_drop(leg, d_ring, dz0)
        if h_e is None:
            # 水平投影超过腿长：该腿在竖直坠落中无法张紧（腿长不足，
            # 由调用方定位站点/动作并判失效）
            continue
        engage.append((h_e, leg))
    engage.sort(key=lambda z: (z[0], z[1].hook))
    if not engage:
        return TwinSolution(converged=False,
                            reason="no_leg_engages")
    h1, first = engage[0]
    order = [lg.leg_id for _h, lg in engage]

    def energy_at(s: float, J, h, active: list[LegTarget],
                  tensions: dict[str, float]):
        e_buf = curve_energy_j(curve, s)
        e_el = sum(T ** 2 / (2.0 * lg.stiffness) * 1000.0
                   for lg in active
                   for T in [tensions[lg.leg_id]])
        return W * h - e_buf - e_el, e_buf, e_el

    def state_at(s: float, guess, active_ids: set[str] | None = None,
                 force_override: float | None = None):
        """给定展开量求平衡状态（主动集自动调整）。

        两条承拉腿挂点重合时合并为一条等效力学腿（刚度相加），张力按刚度
        分配——避免共线导致的 Newton 奇异。
        """
        F = curve_force(curve, s) if force_override is None else force_override
        cur_active = [lg for lg in legs if lg.leg_id in active_ids] \
            if active_ids is not None else [first]
        J = h = None

        def solve_current(cur, warm):
            # 合并挂点重合的承拉腿
            groups: list[list[LegTarget]] = []
            for lg in cur:
                for gp in groups:
                    if _dist(gp[0].anchor, lg.anchor) < 1e-7:
                        gp.append(lg)
                        break
                else:
                    groups.append([lg])
            eff: list[LegTarget] = []
            members: dict[str, list[LegTarget]] = {}
            for gp in groups:
                if len(gp) == 1:
                    eff.append(gp[0])
                    members[gp[0].leg_id] = gp
                else:
                    key = gp[0].leg_id
                    eff.append(LegTarget(
                        leg_id=key, hook=gp[0].hook,
                        target_kind=gp[0].target_kind,
                        target_id=gp[0].target_id, anchor=gp[0].anchor,
                        length=gp[0].length,
                        stiffness=sum(x.stiffness for x in gp),
                        side_limit=min(x.side_limit for x in gp)))
                    members[key] = gp
            if len(eff) == 1:
                st = _single_leg_state(eff[0], d_ring, dz0, s, F)
                if st is None:
                    return None
                Jj, hh, _d = st
            else:
                if warm is None:
                    st0 = _single_leg_state(eff[0], d_ring, dz0, s, F)
                    if st0 is None:
                        return None
                    J0, h0, _d0 = st0
                    warm = (J0[0], J0[1], J0[2], h0)
                sol = _solve4(s, F, d_ring, dz0, eff, warm)
                if sol is None:
                    return None
                Jj, hh = sol
            return Jj, hh, eff, members

        warm = guess
        for _as in range(6):
            got = solve_current(cur_active, warm)
            if got is None:
                return None
            J, h, eff, members = got
            warm = (J[0], J[1], J[2], h)
            # 主动集调整：松弛腿距离超过原长 → 加入；张力腿缩回原长内 → 移除
            changed = False
            cur_ids = {lg.leg_id for lg in cur_active}
            for lg in legs:
                d = _dist(lg.anchor, J)
                if lg.leg_id not in cur_ids and d > lg.length + 1e-7:
                    cur_active.append(lg)
                    cur_ids.add(lg.leg_id)
                    changed = True
                elif lg.leg_id in cur_ids and len(cur_active) > 1 \
                        and d <= lg.length + 1e-7:
                    cur_active = [x for x in cur_active
                                  if x.leg_id != lg.leg_id]
                    changed = True
            if not changed:
                break
        # 展开每腿张力（共点合并腿按刚度分配）
        tensions: dict[str, float] = {}
        got = solve_current(cur_active, warm)
        if got is None:
            return None
        J, h, eff, members = got
        for eg in eff:
            d = _dist(eg.anchor, J)
            T_eff = eg.stiffness * max(0.0, d - eg.length)
            gp = members[eg.leg_id]
            if len(gp) == 1:
                tensions[gp[0].leg_id] = T_eff
            else:
                ksum = sum(x.stiffness for x in gp)
                for x in gp:
                    tensions[x.leg_id] = T_eff * x.stiffness / ksum
        for lg in legs:
            tensions.setdefault(lg.leg_id, 0.0)
        return J, h, F, cur_active, tensions

    # ---- 沿展开量推进，寻找能量根 --------------------------------------
    # 站立即张紧（h1=0）时 R(0)=0 是平凡根：需先越过正亏缺段再回到零点，
    # 因此记录“已转正”，仅对正→非正的穿越二分；全程未转正则为静态悬挂
    # （缓冲启动力 ≥ 体重，峰值取体重）。
    n_steps = 120
    ds = max_travel / n_steps
    s_prev, warm_prev = 0.0, None
    active_prev = {first.leg_id}
    st0 = state_at(0.0, None, {first.leg_id})
    r_prev = W * max(0.0, h1)
    if st0 is None:
        return TwinSolution(converged=False, engagement_order=order,
                            reason="equilibrium_solver_failed")
    seen_positive = r_prev > 1e-9
    root = None
    static_hang = False
    for k in range(1, n_steps + 1):
        s = k * ds
        st = state_at(s, warm_prev, set(active_prev))
        if st is None:
            return TwinSolution(converged=False, engagement_order=order,
                                reason="equilibrium_solver_failed")
        J, h, F, active, tensions = st
        R, e_buf, e_el = energy_at(s, J, h, active, tensions)
        if not math.isfinite(R) or not math.isfinite(h):
            return TwinSolution(converged=False, engagement_order=order,
                                reason="equilibrium_solver_failed")
        if R > 1e-9:
            seen_positive = True
        if seen_positive and R <= 0.0:
            # 二分精化（主动集以变号区间右端为准）
            lo, hi, rlo = s_prev, s, r_prev
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                stm = state_at(mid, (J[0], J[1], J[2], h),
                               {x.leg_id for x in active})
                if stm is None:
                    return TwinSolution(
                        converged=False, engagement_order=order,
                        reason="equilibrium_solver_failed")
                Jm, hm, Fm, am, tm = stm
                Rm, ebm, eem = energy_at(mid, Jm, hm, am, tm)
                if abs(Rm) <= 1e-7 or (hi - lo) / max_travel < 1e-9:
                    root = (mid, Jm, hm, Fm, am, tm, ebm, eem)
                    break
                if (rlo > 0.0) == (Rm > 0.0):
                    lo, rlo = mid, Rm
                else:
                    hi = mid
            if root is None:
                root = (mid, Jm, hm, Fm, am, tm, ebm, eem)
            break
        s_prev, warm_prev, active_prev, r_prev = s, \
            (J[0], J[1], J[2], h), {x.leg_id for x in active}, R
    else:
        if not seen_positive:
            # 静态悬挂：缓冲包不展开，峰值制动力取体重（汇接器按 W 平衡）
            st = state_at(0.0, None, {first.leg_id}, force_override=W / 1000.0)
            if st is None:
                return TwinSolution(converged=False, engagement_order=order,
                                    reason="equilibrium_solver_failed")
            J, h, F, active, tensions = st
            return _assemble(legs, order, J, 0.0, 0.0, 0.0, W / 1000.0,
                             0.0, 0.0, 0.0, tensions=tensions,
                             active={x.leg_id for x in active},
                             drop_h=0.0)
        # 缓冲包完全展开仍不能吸收全部能量
        st = state_at(max_travel, warm_prev, set(active_prev))
        if st is None:
            return TwinSolution(converged=False, engagement_order=order,
                                reason="equilibrium_solver_failed")
        J, h, F, active, tensions = st
        R, e_buf, e_el = energy_at(max_travel, J, h, active, tensions)
        return TwinSolution(
            converged=True, engagement_order=order,
            legs=[_leg_outcome(lg, J, lg.leg_id in {x.leg_id for x in active})
                  for lg in legs],
            junction=J, deployment_m=max_travel, buffer_force_kn=F,
            drop_m=h, free_fall_m=max(0.0, h1),
            energy_demand_j=W * h, buffer_energy_j=e_buf,
            elastic_energy_j=e_el, demand_at_full_j=W * h,
            reason="buffer_energy_insufficient")

    s_star, J, h, F, active, tensions, e_buf, e_el = root
    return _assemble(legs, order, J, s_star, h, h1, F,
                     W * h, e_buf, e_el, tensions=tensions,
                     active={x.leg_id for x in active})


def first_anchor_J(first: LegTarget, d0: Vec, dz0: float) -> Vec:
    H = _horizontal(first.anchor, d0)
    g = math.sqrt(max(0.0, first.length ** 2 - H ** 2))
    return (d0[0], d0[1], first.anchor[2] - g)


def _assemble(legs, order, J, s, h, h1, F, demand, e_buf, e_el,
              tensions=None, active=None, drop_h=None) -> TwinSolution:
    outcomes = []
    for lg in legs:
        taut = active is not None and lg.leg_id in active
        outcomes.append(_leg_outcome(lg, J, taut))
    return TwinSolution(
        converged=True, engagement_order=order, legs=outcomes,
        junction=J, deployment_m=s, buffer_force_kn=F,
        drop_m=h if drop_h is None else drop_h,
        free_fall_m=max(0.0, h1), energy_demand_j=demand,
        buffer_energy_j=e_buf, elastic_energy_j=e_el)


# ---------------------------------------------------------------- 评估辅助

def included_angle(sol: TwinSolution) -> float | None:
    """峰值时刻两条承拉腿在汇接器处的夹角（度）。"""
    taut = [l for l in sol.legs if l.taut]
    if len(taut) < 2:
        return None
    import math as _m
    J = sol.junction
    us = []
    for l in taut:
        v = (l.anchor[0] - J[0], l.anchor[1] - J[1], l.anchor[2] - J[2])
        n = _m.sqrt(sum(x * x for x in v))
        if n < 1e-12:
            return None
        us.append(tuple(x / n for x in v))
    c = sum(us[0][k] * us[1][k] for k in range(3))
    return _m.degrees(_m.acos(max(-1.0, min(1.0, c))))


def build_leg_targets(eq, by_hook, person, station, i, anchors, shuttles,
                      span_paths, sag_by_shuttle, twinleg_mod=None):
    """由钩→(kind,id) 映射构造双腿求解输入（滑梭挂点可随下挠下移）。"""
    out = []
    for leg in eq.twin_leg.legs:
        kind, tid = by_hook[leg.hook]
        if kind == "anchor":
            ap = anchors[tid].position.as_tuple()
        else:
            te = span_paths[shuttles[tid].span_id].table[i]
            sag = sag_by_shuttle.get(tid, 0.0)
            ap = (te.point[0], te.point[1], te.point[2] - max(0.0, sag))
        out.append(LegTarget(
            leg_id=leg.id, hook=leg.hook, target_kind=kind, target_id=tid,
            anchor=ap, length=leg.leg_length_m,
            stiffness=leg.axial_stiffness_kn,
            side_limit=leg.connector_side_load_limit_kn))
    return out


def leg_length_ok(sol: TwinSolution) -> tuple[bool, str | None]:
    """求解后检查：是否所有承拉腿都在几何允许内（无超程硬拉）。

    单连接挂接可达由序列层用 L+触及余量判定；这里捕捉“挂上了但竖直
    坠落路径上水平投影超过腿原长”的几何不可行（no_leg_engages）。
    """
    if sol.reason == "no_leg_engages":
        return False, "no_leg_engages"
    return True, None
