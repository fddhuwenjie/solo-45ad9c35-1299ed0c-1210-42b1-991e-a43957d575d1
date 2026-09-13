"""三维几何工具：折线离散、距离、盒/柱相交、摆坠弧线采样。"""
from __future__ import annotations

import math

Vec = tuple[float, float, float]


def v_sub(a: Vec, b: Vec) -> Vec:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def v_add(a: Vec, b: Vec) -> Vec:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def v_scale(a: Vec, s: float) -> Vec:
    return (a[0] * s, a[1] * s, a[2] * s)


def v_dot(a: Vec, b: Vec) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def v_norm(a: Vec) -> float:
    return math.sqrt(v_dot(a, a))


def v_unit(a: Vec) -> Vec:
    n = v_norm(a)
    if n < 1e-12:
        return (0.0, 0.0, 0.0)
    return (a[0] / n, a[1] / n, a[2] / n)


def dist3(a: Vec, b: Vec) -> float:
    return v_norm(v_sub(a, b))


def dist2d(a: Vec, b: Vec) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def angle_deg(a: Vec, b: Vec) -> float:
    """两向量夹角（度）。"""
    na, nb = v_norm(a), v_norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    c = max(-1.0, min(1.0, v_dot(a, b) / (na * nb)))
    return math.degrees(math.acos(c))


def discretize_polyline(points: list[Vec], spacing: float) -> list[Vec]:
    """把行走折线离散为站点：保留全部顶点，并按 spacing 加密。"""
    if len(points) < 2:
        return list(points)
    stations: list[Vec] = [points[0]]
    for p0, p1 in zip(points, points[1:]):
        seg_len = dist3(p0, p1)
        if seg_len < 1e-9:
            continue
        n = max(1, math.ceil(seg_len / spacing))
        for k in range(1, n + 1):
            t = k / n
            stations.append((
                p0[0] + (p1[0] - p0[0]) * t,
                p0[1] + (p1[1] - p0[1]) * t,
                p0[2] + (p1[2] - p0[2]) * t,
            ))
    return stations


# ---------------------------------------------------------------- 相交测试

def sphere_box_hit(c: Vec, r: float, bmin: Vec, bmax: Vec) -> bool:
    d = 0.0
    for i in range(3):
        if c[i] < bmin[i]:
            d += (bmin[i] - c[i]) ** 2
        elif c[i] > bmax[i]:
            d += (c[i] - bmax[i]) ** 2
    return d <= r * r


def sphere_cylinder_hit(c: Vec, r: float, base: Vec, radius: float, height: float) -> bool:
    """竖直圆柱：底面圆心 base，沿 +z 高 height。"""
    if c[2] + r < base[2] or c[2] - r > base[2] + height:
        return False
    return dist2d(c, base) <= radius + r


def point_in_box_xy(p: Vec, margin: float, bmin: Vec, bmax: Vec) -> bool:
    return (bmin[0] - margin <= p[0] <= bmax[0] + margin
            and bmin[1] - margin <= p[1] <= bmax[1] + margin)


def swing_arc_samples(anchor: Vec, fall_xy: Vec, radius: float,
                      n: int = 16) -> list[Vec]:
    """摆坠扫掠弧：以锚点为圆心、radius 为半径，从初始摆角摆到锚点正下方。

    初始摆角 theta0 = asin(min(水平偏移, R) / R)，在锚点与坠落点所在竖直面内。
    """
    r_h = dist2d(anchor, fall_xy)
    if r_h < 1e-9:
        return [(anchor[0], anchor[1], anchor[2] - radius)]
    theta0 = math.asin(min(r_h, radius) / radius)
    # 水平单位向量：锚点 -> 坠落点
    ux, uy = (fall_xy[0] - anchor[0]) / r_h, (fall_xy[1] - anchor[1]) / r_h
    pts = []
    for k in range(n + 1):
        th = theta0 * (1.0 - k / n)
        pts.append((
            anchor[0] + radius * math.sin(th) * ux,
            anchor[1] + radius * math.sin(th) * uy,
            anchor[2] - radius * math.cos(th),
        ))
    return pts
