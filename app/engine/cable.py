"""柔性跨段悬索求解：浅索近似（水平张力分量恒定），弹性相容迭代。

模型：bay（相邻支座间）承受均布自重 w（kN/m，沿水平跨距）与若干坠落点
集中力 P（止坠力，kN）。索形 = 弦线 + 弦下附加挠曲 y(x)，y 由简支梁
剪力除以水平张力 H 给出；相容条件

    L_loaded(H) − L0(H0) = ΔL(H)

（几何伸长 = 弹性伸长，ΔL 分段按 (T−T0)·ℓ0/EA 累加），以二分法求 H。
求解不收敛且修订许可保守边界时，取包络：
- 反力用不计弹性（ΔL=0）的刚性索解（H 偏大 → 反力偏大，结构侧保守）；
- 下挠用 H=H0 的几何索形（H 最小可能值 → 下挠最大，净空侧保守）。
二者分别是真实解的两侧界。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class BayLoad:
    person_id: str
    frac: float           # bay 内水平分数 [0,1]
    load_kn: float


@dataclass
class BaySolution:
    converged: bool
    conservative: bool
    h_kn: float                       # 水平张力（反力用；保守时为刚性索值）
    vl_kn: float                      # 左端弦下剪力 VL = ΣP(1−f) + wL/2
    vr_kn: float                      # 右端弦下剪力 VR = ΣP f   + wL/2
    sag_m: float                      # 最大动态下挠（弦下，m）
    load_sags_m: list[float]          # 与 loads 同序的各坠落点弦下挠
    load_fractions: list[float]
    chord_rise: float                 # 右−左高差（带符号）
    horiz_span: float
    self_weight_kn_m: float
    pretension_kn: float
    ea_kn: float | None
    point_loads_kn: list[float] = field(default_factory=list)


def _shape(h: float, L: float, rise: float, w: float,
           fr: list[float], pp: list[float]):
    """给定水平张力 h，返回（节点弦下挠 y、分段长度、分段端弦下剪力）。

    节点 x: 0, fr·L, ..., L；分段内均布 w 使剪力线性变化，
    节点集中力使剪力阶跃。y 由分段平均剪力积分，右端自然回零。
    """
    xs = [0.0] + [f * L for f in fr] + [L]
    # 节点左侧/右侧剪力（弦下，向下为正）；简支端 VL, VR
    VL = w * L / 2.0 + sum(p * (1.0 - f) for f, p in zip(fr, pp))
    VR = w * L / 2.0 + sum(p * f for f, p in zip(fr, pp))
    s_left, s_right = [], []
    y = [0.0]
    seg_len = []
    for j in range(len(xs) - 1):
        xa, xb = xs[j], xs[j + 1]
        # 分段内无集中力（集中力全在节点），剪力从 S(xa+) 线性到 S(xb−)
        loads_before_a = sum(p for f, p in zip(fr, pp) if f * L <= xa + 1e-9)
        sa = VL - w * xa - loads_before_a
        loads_before_b = sum(p for f, p in zip(fr, pp) if f * L < xb - 1e-9)
        sb = VL - w * xb - loads_before_b
        s_left.append(sa)
        s_right.append(sb)
        dy = (sa + sb) / 2.0 * (xb - xa) / h
        y.append(y[-1] + dy)
        c = rise / L
        dz = c * (xb - xa) - dy
        seg_len.append(math.hypot(xb - xa, dz))
    # 数值上把右端钳回零并修正最后一段（误差量级 1e-12）
    y[-1] = 0.0
    return y, seg_len, (VL, VR), s_left, s_right, xs


def _geom_length(h: float, L, rise, w, fr, pp) -> float:
    return sum(_shape(h, L, rise, w, fr, pp)[1])


def solve_bay(L: float, rise: float, loads: list[BayLoad],
              h0: float, w: float, ea: float | None,
              allow_conservative: bool,
              max_iter: int = 200, tol: float = 1e-5) -> BaySolution:
    """求 bay 在给定坠落组合下的张力与下挠。

    弹性相容迭代不收敛时：默认返回 converged=False（不下结论）；
    allow_conservative=True 时回退保守包络——反力按刚性（不计弹性，必要时
    钳至 H_RIGID_CAP 声明的刚性下界）、下挠按 H=H0 几何索形（上界）。
    """
    fr = [ld.frac for ld in loads]
    pp = [ld.load_kn for ld in loads]
    p_tot = sum(pp) + w * L
    # 坠落点恰在支座上（frac≈0/1）：不经索传力，索保持预张力 + 自重态
    at_support = all(f <= 1e-9 or f >= 1.0 - 1e-9 for f in fr)
    if at_support:
        _y, _sl, (VL, VR), _e, _r, _x = _shape(
            max(h0, 1e-9), L, rise, w,
            [f for f in fr if 1e-9 < f < 1.0 - 1e-9],
            [p for f, p in zip(fr, pp) if 1e-9 < f < 1.0 - 1e-9])
        sags0 = []
        for f in fr:
            sags0.append(0.0 if f <= 1e-9 or f >= 1.0 - 1e-9 else 0.0)
        return BaySolution(
            converged=True, conservative=False, h_kn=max(h0, 1e-9),
            vl_kn=VL, vr_kn=VR, sag_m=0.0, load_sags_m=[0.0] * len(loads),
            load_fractions=fr, point_loads_kn=pp,
            chord_rise=rise, horiz_span=L,
            self_weight_kn_m=w, pretension_kn=h0, ea_kn=ea)
    # 刚性张力声明下界：足以让任何有限额定支座判越限，同时数值可控
    h_cap = max(1000.0 * h0, 1000.0 * p_tot, 1.0)

    # 初态（仅自重，H0）索形：作为无应力基准的近似
    L0 = _geom_length(max(h0, 1e-9), L, rise, w, [], [])
    c = rise / L

    def elong(h: float, seg_xs, seg_sl, seg_sr) -> float:
        if ea is None:
            return 0.0
        total = 0.0
        for j, (xa, xb) in enumerate(zip(seg_xs, seg_xs[1:])):
            a = xb - xa
            t_l = math.hypot(h, h * c - seg_sl[j])
            t_r = math.hypot(h, h * c - seg_sr[j])
            # 预张力态分段张力（自重抛物线，线性化取段中剪力）
            xm = (xa + xb) / 2.0
            s0m = w * (L / 2.0 - xm)
            t0 = math.hypot(h0, h0 * c - s0m)
            l0 = a  # 基准分段水平投影（弹性项量级小，弦长近似足够）
            total += ((t_l + t_r) / 2.0 - t0) * l0 / ea
        return total

    def residual(h: float) -> tuple[float, object]:
        _y, sl, _ends, sle, sre, xs = _shape(h, L, rise, w, fr, pp)
        return sum(sl) - L0 - elong(h, xs, sle, sre), \
            (sl, sle, sre, xs)

    def bisect(include_elastic: bool):
        lo = max(h0, 1e-6)
        hi = max(lo * 2.0, 2.0 * (sum(pp) + w * L + 1.0))

        def f(h):
            v = residual(h)[0]
            return v if include_elastic or ea is None else \
                _geom_length(h, L, rise, w, fr, pp) - L0

        flo = f(lo)
        for _ in range(60):           # 指数扩展上界
            fh = f(hi)
            if flo * fh <= 0:
                break
            hi *= 2.0
        else:
            return None
        for _ in range(max_iter):
            mid = (lo + hi) / 2.0
            fm = f(mid)
            if abs(fm) <= tol or (hi - lo) / hi < 1e-10:
                return mid
            if flo * fm <= 0:
                hi = mid
            else:
                lo, flo = mid, fm
        return (lo + hi) / 2.0

    h_solved = bisect(True)
    converged = h_solved is not None
    conservative = False

    if converged:
        h_use = h_solved
    elif allow_conservative:
        h_rigid = bisect(False)                 # 刚性索（反力上界）
        if h_rigid is None or h_rigid > h_cap:
            h_rigid = h_cap                     # 声明刚性下界
        h_use = h_rigid
        conservative = True
    else:
        h_use = None

    def assemble(h: float) -> dict:
        y, _sl, (VL, VR), _sle, _sre, _xs = _shape(h, L, rise, w, fr, pp)
        # 最大下挠：节点值与分段内（剪力过零处）抛物线峰值
        sag = max(y) if y else 0.0
        xs = [0.0] + [f * L for f in fr] + [L]
        for j in range(len(xs) - 1):
            sa = VL - w * xs[j] - sum(p for f, p in zip(fr, pp) if f * L <= xs[j] + 1e-9)
            sb = VL - w * xs[j + 1] - sum(p for f, p in zip(fr, pp) if f * L < xs[j + 1] - 1e-9)
            if sa > 0 > sb and w > 1e-12:
                # 分段内剪力线性过零（距 xa 为 sa/w），抛物线极值点
                xstar = xs[j] + sa / w
                if xs[j] <= xstar <= xs[j + 1]:
                    ym = y[j] + (sa + (sa - w * (xstar - xs[j]))) / 2.0 \
                        * (xstar - xs[j]) / h
                    sag = max(sag, ym)
        return {"y": y, "VL": VL, "VR": VR, "sag": sag}

    if h_use is None:
        return BaySolution(
            converged=False, conservative=False, h_kn=0.0,
            vl_kn=0.0, vr_kn=0.0, sag_m=0.0, load_sags_m=[0.0] * len(loads),
            load_fractions=fr, point_loads_kn=pp,
            chord_rise=rise, horiz_span=L,
            self_weight_kn_m=w, pretension_kn=h0, ea_kn=ea)

    d = assemble(h_use)
    # 净空保守包络：H=H0 几何索形的各坠落点下挠（下挠上界）
    if conservative:
        y_env, _, (VL_e, VR_e), _, _, _ = _shape(
            max(h0, 1e-6), L, rise, w, fr, pp)
        load_sags = [y_env[k + 1] for k in range(len(loads))]
        sag_env = max([d["sag"]] + load_sags + [0.0])
        # 下挠以包络为准；剪力/反力仍取刚性索解
        vl, vr, sag = d["VL"], d["VR"], max(d["sag"], sag_env)
    else:
        load_sags = [d["y"][k + 1] for k in range(len(loads))]
        vl, vr, sag = d["VL"], d["VR"], d["sag"]

    return BaySolution(
        converged=True, conservative=conservative, h_kn=h_use,
        vl_kn=vl, vr_kn=vr, sag_m=sag, load_sags_m=load_sags,
        load_fractions=fr, point_loads_kn=pp,
        chord_rise=rise, horiz_span=L,
        self_weight_kn_m=w, pretension_kn=h0, ea_kn=ea)


def pretension_end_components(bay, span, gravity: float) -> tuple[float, float, float]:
    """未加载相邻 bay 在支座端的 (水平张力 H0, 弦坡度 c, 端弦下剪力 wL/2)。"""
    w = (span.line_density_kg_m or 0.0) * gravity / 1000.0
    h0 = span.pretension_kn or 0.0
    return h0, bay.rise / max(bay.horiz, 1e-9), w * bay.horiz / 2.0
