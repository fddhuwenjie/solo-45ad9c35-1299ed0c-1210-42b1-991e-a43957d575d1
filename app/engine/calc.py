"""坠落核算：自由坠距、总净空、摆坠扫掠体、锚点方向与合力。"""
from __future__ import annotations

from dataclasses import dataclass

from ..models import Anchor, CalcParams, Equipment, Person, Vec3
from . import geometry as g

Vec = tuple[float, float, float]


@dataclass
class FallCalc:
    """单站、单人、单锚点的坠落计算分量。"""
    free_fall_m: float
    total_fall_m: float
    deployed_length_m: float
    required_clearance_m: float
    available_clearance_m: float | None   # None 表示下方无实体（无限净空）
    margin_m: float | None
    arrest_force_kn: float
    force_angle_deg: float | None
    horizontal_offset_m: float
    swing_radius_m: float


def d_ring_pos(station: Vec, person: Person) -> Vec:
    return (station[0], station[1], station[2] + person.d_ring_height_m)


def reachable(station: Vec, person: Person, eq: Equipment, anchor: Anchor) -> bool:
    """连接器触及范围：D 环到锚点距离 ≤ 绳长 + 连接器触及余量。"""
    return g.dist3(d_ring_pos(station, person), anchor.position.as_tuple()) \
        <= eq.lanyard_length_m + eq.connector_reach_m + 1e-9


def free_fall_distance(eq: Equipment, d_z: float, anchor_z: float) -> float:
    """自由坠距：锚点低于 D 环时增加，高于 D 环时减小，限制在 [0, 2L]。"""
    L = eq.lanyard_length_m
    return max(0.0, min(2.0 * L, L + (d_z - anchor_z)))


def free_fall_with_sag(eq: Equipment, d_z: float, anchor_z_static: float,
                       sag: float) -> float:
    """滑梭挂在柔性跨段上：坠落后滑梭随钢索下挠 sag（m），挂点下移。"""
    return free_fall_distance(eq, d_z, anchor_z_static - max(0.0, sag))


def arrest_force_kn(person: Person, eq: Equipment, ffd: float,
                    params: CalcParams) -> float:
    """能量法估算止坠力：W·g·(FFD+缓冲行程)/缓冲行程，封顶于装备最大止坠力。"""
    if eq.buffer_travel_m <= 1e-9:
        return eq.max_arrest_force_kn
    f = person.weight_kg * params.gravity * (ffd + eq.buffer_travel_m) \
        / eq.buffer_travel_m / 1000.0
    return min(eq.max_arrest_force_kn, f)


def obstacle_top_below(station: Vec, body_radius: float,
                       obstacles) -> float | None:
    """站点正下方（水平 body_radius 范围内）最高障碍物顶面 z。"""
    top = None
    for ob in obstacles:
        if ob.kind == "box":
            bmin, bmax = ob.min.as_tuple(), ob.max.as_tuple()
            if g.point_in_box_xy(station, body_radius, bmin, bmax):
                z = bmax[2]
                if z < station[2] - 1e-9 and (top is None or z > top):
                    top = z
        else:
            base = ob.base.as_tuple()
            if g.dist2d(station, base) <= ob.radius + body_radius:
                z = base[2] + ob.height
                if z < station[2] - 1e-9 and (top is None or z > top):
                    top = z
    return top


def lower_level_below(station: Vec, lanyard_len: float,
                      drop_edges) -> float | None:
    """落差边缘下层楼面：站点水平距离边缘不超过绳长时，坠体可能荡过边缘落到下层。"""
    low = None
    for e in drop_edges:
        if g.dist2d(station, e.point.as_tuple()) <= lanyard_len:
            z = e.point.z - e.drop_depth_m
            if z < station[2] - 1e-9 and (low is None or z > low):
                low = z
    return low


def fall_calc(station: Vec, person: Person, eq: Equipment, anchor: Anchor,
              route, params: CalcParams) -> FallCalc:
    a = anchor.position.as_tuple()
    d = d_ring_pos(station, person)
    L = eq.lanyard_length_m

    ffd = free_fall_distance(eq, d[2], a[2])
    deployed = L + eq.elongation_m + eq.buffer_travel_m
    total_fall = ffd + eq.elongation_m + eq.buffer_travel_m + params.harness_stretch_m
    required = total_fall + params.safety_margin_m

    # 可用净空：障碍物顶面与落差边缘下层楼面取较高者（先撞到的那个）
    surfaces = []
    top = obstacle_top_below(station, person.body_radius_m, route.obstacles)
    if top is not None:
        surfaces.append(top)
    low = lower_level_below(station, L, route.drop_edges)
    if low is not None:
        surfaces.append(low)
    if surfaces:
        avail = station[2] - max(surfaces)
        margin = avail - required
    else:
        avail = None
        margin = None

    force = arrest_force_kn(person, eq, ffd, params)

    angle = None
    if anchor.allowed_axis is not None:
        angle = g.angle_deg(g.v_sub(d, a), anchor.allowed_axis.as_tuple())

    r_h = g.dist2d(a, station)
    return FallCalc(
        free_fall_m=round(ffd, 4),
        total_fall_m=round(total_fall, 4),
        deployed_length_m=round(deployed, 4),
        required_clearance_m=round(required, 4),
        available_clearance_m=None if avail is None else round(avail, 4),
        margin_m=None if margin is None else round(margin, 4),
        arrest_force_kn=round(force, 4),
        force_angle_deg=None if angle is None else round(angle, 2),
        horizontal_offset_m=round(r_h, 4),
        swing_radius_m=round(deployed, 4),
    )


def sweep_hit_obstacle(station: Vec, person: Person, eq: Equipment,
                       anchor: Anchor, route, params: CalcParams) -> str | None:
    """摆坠扫掠体与障碍物相交检查，返回撞到的障碍物 id 或 None。

    扫掠体：沿摆坠弧线采样的 D 环与脚底两串球体（半径 body_radius）。
    """
    a = anchor.position.as_tuple()
    r_h = g.dist2d(a, station)
    if r_h <= params.swing_threshold_m:
        return None
    R = eq.lanyard_length_m + eq.elongation_m + eq.buffer_travel_m
    arc = g.swing_arc_samples(a, station, R)
    r = person.body_radius_m
    centers: list[Vec] = []
    for p in arc:
        centers.append(p)                                            # D 环
        centers.append((p[0], p[1], p[2] - person.d_ring_height_m))  # 脚底
    for ob in route.obstacles:
        hit = False
        if ob.kind == "box":
            bmin, bmax = ob.min.as_tuple(), ob.max.as_tuple()
            hit = any(g.sphere_box_hit(c, r, bmin, bmax) for c in centers)
        else:
            base = ob.base.as_tuple()
            hit = any(g.sphere_cylinder_hit(c, r, base, ob.radius, ob.height)
                      for c in centers)
        if hit:
            return ob.id
    return None
