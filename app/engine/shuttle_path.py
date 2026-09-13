"""滑梭沿柔性跨段的站位几何：投影到各 bay 弦线，给出位置、索形高度与过支座信息。

钢索未加载时按各 bay 均布自重抛物线近似（仅用于站站位与触及范围；
坠落后的动态下挠由 cable.py 迭代求解）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..models import FlexibleSpan, Support
from . import geometry as g

Vec = tuple[float, float, float]


@dataclass
class BayGeom:
    index: int
    left: str
    right: str
    p0: Vec
    p1: Vec
    horiz: float          # 水平跨距
    chord3d: float        # 弦长（三维）
    rise: float           # 右端 - 左端 高差（带符号）
    sag0: float = 0.0     # 未加载自重下挠（相对弦线中点，m）

    def cable_z_unloaded(self, frac: float) -> float:
        """均布自重抛物线：相对弦线中点下垂 sag0，形状 4·f·(1−f)。"""
        chord_z = self.p0[2] + self.rise * frac
        return chord_z - 4.0 * self.sag0 * frac * (1.0 - frac)


@dataclass
class ShuttleStation:
    valid: bool                       # 滑索覆盖到该站
    bay: int                          # 所在 bay 序号
    frac: float                       # bay 内水平分数 [0,1]
    point: Vec                        # 滑梭在索上的位置
    arc_s: float                      # 沿全跨三维累积长度（用于过支座判定）
    left_id: str
    right_id: str


def unloaded_bay_sags(span: FlexibleSpan, supports: dict[str, Support],
                      gravity: float) -> list[float]:
    """各 bay 未加载自重抛物线中点下挠 q·L_h²/(8·H0)（均布沿水平跨距近似）。

    参数缺失时返回 0（几何回退；动态求解阶段另行报参数缺失 open item）。
    """
    sags = []
    q = None
    if span.pretension_kn is not None and span.line_density_kg_m is not None:
        q = span.line_density_kg_m * gravity / 1000.0  # kN/m
    for s0, s1 in zip(span.supports, span.supports[1:]):
        p0, p1 = supports[s0].position, supports[s1].position
        Lh = math.hypot(p1.x - p0.x, p1.y - p0.y)
        if q is not None and span.pretension_kn and span.pretension_kn > 1e-9:
            sags.append(q * Lh * Lh / (8.0 * span.pretension_kn))
        else:
            sags.append(0.0)
    return sags


class SpanPath:
    """一条柔性跨段的静态几何与每站滑梭站位表。"""

    def __init__(self, span: FlexibleSpan,
                 supports: dict[str, Support],
                 stations: list[Vec],
                 sags0: list[float] | None = None):
        self.span = span
        self.bays: list[BayGeom] = []
        pts = [supports[sid].position.as_tuple() for sid in span.supports]
        for k, (s0, s1) in enumerate(zip(span.supports, span.supports[1:])):
            p0, p1 = pts[k], pts[k + 1]
            self.bays.append(BayGeom(
                index=k, left=s0, right=s1, p0=p0, p1=p1,
                horiz=g.dist2d(p0, p1), chord3d=g.dist3(p0, p1),
                rise=p1[2] - p0[2],
                sag0=(sags0[k] if sags0 is not None else 0.0)))
        # bay 在全跨三维累积弧长上的起点
        self.bay_s0: list[float] = []
        s = 0.0
        for b in self.bays:
            self.bay_s0.append(s)
            s += b.chord3d
        self.total_len = s
        self.table: list[ShuttleStation] = [self._project(st) for st in stations]

    def _project(self, st: Vec) -> ShuttleStation:
        # 先在跨内（弦线参数 t∈[0,1]）选水平残差最小 bay；
        # 无跨内 bay 时（站在延长线上）取残差最小者并标记 valid=False。
        # 残差相同时（如站恰在中间支座上）取较小 bay（frac→右端点），
        # 保证沿 x 行进时 bay 序号单调不减，不会在支座处来回跳变。
        best_in = None
        best_out = None
        for b in self.bays:
            vx, vy = b.p1[0] - b.p0[0], b.p1[1] - b.p0[1]
            denom = vx * vx + vy * vy
            t = ((st[0] - b.p0[0]) * vx + (st[1] - b.p0[1]) * vy) / denom
            tc = min(1.0, max(0.0, t))
            px = b.p0[0] + vx * tc
            py = b.p0[1] + vy * tc
            resid = math.hypot(st[0] - px, st[1] - py)
            key = (round(resid, 9), b.index, b, tc)
            if -1e-6 <= t <= 1.0 + 1e-6:
                if best_in is None or key[:2] < best_in[:2]:
                    best_in = key
            elif best_out is None or resid < best_out[0]:
                best_out = (resid, b, tc)
        if best_in is not None:
            valid = True
            _r, _bi, b, tc = best_in
        else:
            valid = False
            _r, b, tc = best_out
        z = b.cable_z_unloaded(tc)
        point = (b.p0[0] + (b.p1[0] - b.p0[0]) * tc,
                 b.p0[1] + (b.p1[1] - b.p0[1]) * tc, z)
        arc_s = self.bay_s0[b.index] + b.chord3d * tc
        return ShuttleStation(
            valid=valid, bay=b.index, frac=tc, point=point, arc_s=arc_s,
            left_id=b.left, right_id=b.right)

    def crosses_intermediate_support(self, i0: int, i1: int) -> int | None:
        """从站 i0 行至 i1 是否越过任一中间支座；返回被越过的分界序号
        （k=1..len(bays)-1，对应 span.supports[k]），未越过为 None。
        """
        t0, t1 = self.table[i0], self.table[i1]
        lo, hi = sorted((t0.arc_s, t1.arc_s))
        for k in range(1, len(self.bays)):
            sk = self.bay_s0[k]
            if lo < sk - 1e-9 < hi + 1e-9:
                return k
        return None

    def support_id_at_boundary(self, k: int) -> str:
        return self.span.supports[k]
