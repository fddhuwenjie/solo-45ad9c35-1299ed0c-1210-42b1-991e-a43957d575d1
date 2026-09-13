"""Pydantic 请求/响应模型：线路方案、人员装备、核算结果。

路线要素可以混用两类挂接对象：
- 固定点锚（Anchor）：挂点不动，按点锚核算；
- 柔性跨段（FlexibleSpan + Support + Shuttle）：临时水平生命线，端座之间
  张拉钢索，中间可有连续/断开式支座，滑梭沿索随作业者移动。
"""
from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------- 基础几何

class Vec3(BaseModel):
    x: float
    y: float
    z: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


# ---------------------------------------------------------------- 障碍物

class ObstacleBox(BaseModel):
    """轴对齐盒（如下层管线桥架、设备箱体）。"""
    kind: Literal["box"] = "box"
    id: str
    min: Vec3
    max: Vec3


class ObstacleCylinder(BaseModel):
    """竖直圆柱（如立柱、立管），base 为底面圆心。"""
    kind: Literal["cylinder"] = "cylinder"
    id: str
    base: Vec3
    radius: float = Field(gt=0)
    height: float = Field(gt=0)


Obstacle = Annotated[Union[ObstacleBox, ObstacleCylinder], Field(discriminator="kind")]


# ---------------------------------------------------------------- 路线要素

class DropEdge(BaseModel):
    """落差边缘：行走面上的开口/平台边沿，下方为 lower_level_z = z - drop_depth_m。"""
    id: str
    point: Vec3
    sharpness_class: int = Field(ge=0, le=3, description="0=圆角 1=一般 2=锐利 3=极锐利")
    drop_depth_m: float = Field(ge=0)


class Anchor(BaseModel):
    """固定点锚：坐标、额定载荷、允许受力方向锥、多人共用限制。"""
    id: str
    position: Vec3
    rated_load_kn: float = Field(gt=0)
    allowed_axis: Optional[Vec3] = Field(
        default=None, description="允许受力方向锥轴线（单位向量），None 表示全向"
    )
    allowed_half_angle_deg: float = Field(default=180.0, ge=0, le=180)
    max_users: int = Field(default=1, ge=1, description="同时共用人数上限")


class Support(BaseModel):
    """水平生命线端座 / 中间支座：含结构额定反力与允许受力方向锥。"""
    id: str
    position: Vec3
    rated_load_kn: float = Field(gt=0, description="结构容许反力（合力，kN）")
    allowed_axis: Optional[Vec3] = Field(
        default=None, description="允许受力方向锥轴线（单位向量），None 表示全向"
    )
    allowed_half_angle_deg: float = Field(default=180.0, ge=0, le=180)


class FlexibleSpan(BaseModel):
    """柔性跨段：两个端座之间张拉的水平钢索，可含中间支座。

    supports 依次为 端座、中间支座…、端座；相邻支座构成一个计算 bay。
    数值参数允许 None：参数缺失时本跨不下结论（conclusive=False）。
    """
    id: str
    supports: list[str] = Field(min_length=2, description="端座与中间支座 id（按线路顺序）")
    pretension_kn: Optional[float] = Field(default=None, ge=0, description="安装预张力 H0（kN）")
    line_density_kg_m: Optional[float] = Field(default=None, ge=0, description="钢索线密度（kg/m）")
    axial_stiffness_kn: Optional[float] = Field(
        default=None, gt=0, description="轴向刚度 EA（kN）；与 cross_section_m2 至少给一个")
    cross_section_m2: Optional[float] = Field(
        default=None, gt=0, description="截面积（m²）；与弹性模量配合推导 EA")
    elastic_modulus_kn_m2: Optional[float] = Field(
        default=None, gt=0, description="弹性模量 E（kN/m²）")
    max_sag_m: Optional[float] = Field(default=None, gt=0, description="动态下挠限值（m）")
    shuttle_pass: bool = Field(
        default=True, description="中间支座是否允许滑梭连续通过（断开式支座为 False）")
    max_users: int = Field(default=1, ge=1, description="本跨容许同时作业人数")

    def ea_kn(self) -> Optional[float]:
        """轴向刚度 EA（kN）：优先用显式 EA，其次 E·A。"""
        if self.axial_stiffness_kn is not None:
            return self.axial_stiffness_kn
        if self.cross_section_m2 is not None and self.elastic_modulus_kn_m2 is not None:
            return self.cross_section_m2 * self.elastic_modulus_kn_m2
        return None

    def missing_params(self) -> list[str]:
        """返回缺失的必要参数名。"""
        miss = []
        if self.pretension_kn is None:
            miss.append("pretension_kn")
        if self.line_density_kg_m is None:
            miss.append("line_density_kg_m")
        if self.ea_kn() is None:
            miss.append("axial_stiffness_kn(或 E·A)")
        if self.max_sag_m is None:
            miss.append("max_sag_m")
        return miss


class Shuttle(BaseModel):
    """滑梭：套在某条柔性跨段钢索上、可沿索移动并通过中间支座。"""
    id: str
    span_id: str
    connector_reach_m: float = Field(
        default=0.3, ge=0, description="连接器超出人员绳端的触及余量")
    max_users: int = Field(default=1, ge=1, description="同一滑梭容许同时挂接人数")
    can_pass: bool = Field(
        default=True,
        description="滑梭自身能否通过断开式中间支座"
                    "（须与跨段 shuttle_pass 同时满足）")


class ConservativeBounds(BaseModel):
    """采用保守边界（不计钢索弹性）求解的许可说明。

    仅当弹性迭代不收敛时作为回退；采用时必须在修订 changes 中给出理由。
    """
    enabled: bool = False
    reason: str = ""


class Equipment(BaseModel):
    """防坠装备：绳长、伸长量、缓冲行程、锐边等级、连接器触及范围。"""
    id: str
    lanyard_length_m: float = Field(gt=0)
    elongation_m: float = Field(ge=0, description="止坠时绳体伸长量")
    buffer_travel_m: float = Field(ge=0, description="缓冲包展开行程")
    sharp_edge_rating: int = Field(ge=0, le=3, description="可承受的锐边等级")
    connector_reach_m: float = Field(ge=0, description="连接器超出绳端的触及余量")
    max_arrest_force_kn: float = Field(default=6.0, gt=0)


class Person(BaseModel):
    id: str
    weight_kg: float = Field(gt=0)
    equipment_id: str
    d_ring_height_m: float = Field(default=1.4, gt=0, description="D 环距脚底高度")
    body_radius_m: float = Field(default=0.35, gt=0)


class CalcParams(BaseModel):
    station_spacing_m: float = Field(default=0.5, gt=0.05, le=5.0)
    safety_margin_m: float = Field(default=0.6, ge=0)
    harness_stretch_m: float = Field(default=0.3, ge=0)
    swing_threshold_m: float = Field(default=0.3, ge=0, description="水平偏移小于该值不计摆坠")
    gravity: float = Field(default=9.81, gt=0)
    cable_max_iter: int = Field(default=200, ge=10, description="悬索迭代最大次数")
    cable_tol_kn: float = Field(default=1e-5, gt=0, description="悬索水平张力收敛容差（kN）")


class RouteSpec(BaseModel):
    walk_polyline: list[Vec3] = Field(min_length=2)
    drop_edges: list[DropEdge] = []
    obstacles: list[Obstacle] = []
    anchors: list[Anchor] = []
    supports: list[Support] = []
    spans: list[FlexibleSpan] = []
    shuttles: list[Shuttle] = []


class ManualHookAction(BaseModel):
    """人工挂接动作：在指定站点对某钩执行挂接/换钩/解钩。

    目标为 anchor（固定点锚）或 shuttle（柔性跨段滑梭）二选一；
    detach 二者均省略。
    """
    person_id: str
    hook: Literal["A", "B"]
    action: Literal["attach", "switch", "detach"]
    anchor: Optional[str] = Field(
        default=None, description="目标固定点锚 id（attach/switch 与 shuttle 二选一）")
    shuttle: Optional[str] = Field(
        default=None, description="目标滑梭 id（attach/switch 与 anchor 二选一）")
    station_index: int = Field(ge=0, description="动作发生的站点序号")

    @model_validator(mode="after")
    def _check_target(self) -> "ManualHookAction":
        if self.action in ("attach", "switch"):
            n = (1 if self.anchor else 0) + (1 if self.shuttle else 0)
            if n != 1:
                raise ValueError("attach/switch 必须且只能给出目标 anchor 或 shuttle 之一")
        if self.action == "detach" and (self.anchor or self.shuttle):
            raise ValueError("detach 不应给出目标 anchor/shuttle")
        return self


class PlanPayload(BaseModel):
    """一版修订的全部参数：路线 + 人员 + 装备 + 计算参数。"""
    route: RouteSpec
    persons: list[Person] = Field(min_length=1)
    equipment: list[Equipment] = Field(min_length=1)
    params: CalcParams = Field(default_factory=CalcParams)
    hook_order: list[ManualHookAction] = Field(
        default_factory=list,
        description="人工挂接动作次序；为空则按自动算法生成序列")
    conservative_bounds: ConservativeBounds = Field(default_factory=ConservativeBounds)

    @model_validator(mode="after")
    def _check_refs(self) -> "PlanPayload":
        route = self.route
        if not route.anchors and not route.spans:
            raise ValueError("路线必须至少包含一个固定点锚或一条柔性跨段")

        eq_ids = {e.id for e in self.equipment}
        for p in self.persons:
            if p.equipment_id not in eq_ids:
                raise ValueError(f"人员 {p.id} 引用了不存在的装备 {p.equipment_id}")

        anchor_ids = [a.id for a in route.anchors]
        if len(anchor_ids) != len(set(anchor_ids)):
            raise ValueError("锚点 id 重复")
        support_ids = [s.id for s in route.supports]
        if len(support_ids) != len(set(support_ids)):
            raise ValueError("支座 id 重复")
        span_ids = [s.id for s in route.spans]
        if len(span_ids) != len(set(span_ids)):
            raise ValueError("跨段 id 重复")
        shuttle_ids = [s.id for s in route.shuttles]
        if len(shuttle_ids) != len(set(shuttle_ids)):
            raise ValueError("滑梭 id 重复")
        if set(anchor_ids) & set(shuttle_ids):
            raise ValueError("锚点与滑梭 id 不得重名")

        support_map = {s.id: s for s in route.supports}
        for sp in route.spans:
            if len(sp.supports) < 2:
                raise ValueError(f"跨段 {sp.id} 至少需要两个端座")
            if len(set(sp.supports)) != len(sp.supports):
                raise ValueError(f"跨段 {sp.id} 的支座序列存在重复")
            for sid in sp.supports:
                if sid not in support_map:
                    raise ValueError(f"跨段 {sp.id} 引用了不存在的支座 {sid}")
            # 相邻支座不得为同一点（需要非零水平跨距）
            for s0, s1 in zip(sp.supports, sp.supports[1:]):
                p0, p1 = support_map[s0].position, support_map[s1].position
                if ((p0.x - p1.x) ** 2 + (p0.y - p1.y) ** 2) < 1e-12:
                    raise ValueError(
                        f"跨段 {sp.id} 相邻支座 {s0}->{s1} 无水平跨距")

        span_map = {s.id: s for s in route.spans}
        for sh in route.shuttles:
            if sh.span_id not in span_map:
                raise ValueError(f"滑梭 {sh.id} 引用了不存在的跨段 {sh.span_id}")

        person_ids = {p.id for p in self.persons}
        anchor_id_set = set(anchor_ids)
        shuttle_id_set = set(shuttle_ids)
        for act in self.hook_order:
            if act.person_id not in person_ids:
                raise ValueError(
                    f"人工动作引用了不存在的人员 {act.person_id}")
            if act.anchor is not None and act.anchor not in anchor_id_set:
                raise ValueError(
                    f"人工动作引用了不存在的锚点 {act.anchor}")
            if act.shuttle is not None and act.shuttle not in shuttle_id_set:
                raise ValueError(
                    f"人工动作引用了不存在的滑梭 {act.shuttle}")

        if self.conservative_bounds.enabled and not self.conservative_bounds.reason.strip():
            raise ValueError("启用保守边界必须在 conservative_bounds.reason 中说明理由")
        return self


# ---------------------------------------------------------------- 核算结果

Action = Literal["attach", "switch", "detach", "traverse"]


class SequenceEvent(BaseModel):
    """挂接 / 换钩 / 解钩 事件（目标可为固定点锚或滑梭）。"""
    station_index: int
    position: Vec3
    person_id: str
    action: Literal["attach", "switch", "detach"]
    hook: Literal["A", "B"]
    from_anchor: Optional[str] = None
    to_anchor: Optional[str] = None
    from_shuttle: Optional[str] = None
    to_shuttle: Optional[str] = None
    attached_after: list[str] = []
    attached_shuttles_after: list[str] = []


class CheckFailure(BaseModel):
    """一次核算失败：位置、动作、检查项与计算分量。"""
    station_index: int
    position: Vec3
    person_id: Optional[str] = None
    action: Action
    check: str
    message: str
    components: dict[str, float | int | str | None] = {}


class OpenItem(BaseModel):
    """不下结论项：连续性断开、滑梭卡支座、重复占用、参数缺失、求解不收敛等。

    出现 open item 时该版分析 conclusive=False、passable=False，
    须人工复核，不得以任何计算分量冒充可通行结论。
    """
    station_index: Optional[int] = None
    position: Optional[Vec3] = None
    person_id: Optional[str] = None
    action: Optional[Action] = None
    code: str
    message: str
    components: dict[str, float | int | str | None] = {}


class CableResult(BaseModel):
    """一次悬索坠落组合的求解分量（单 bay、给定坠落人员组合）。"""
    span_id: str
    station_index: int
    bay_index: int
    falling_persons: list[str]
    loads_kn: list[float]
    load_fractions: list[float]
    total_fall_m: list[float] = Field(
        default_factory=list,
        description="与 falling_persons 同序的各人总坠距（m，含动态下挠）")
    sag_m: float = Field(description="相对最高支座连线的动态下挠（m）")
    max_sag_m: Optional[float] = Field(
        default=None, description="本跨挠度限值（m），缺失为 None")
    horizontal_tension_kn: float
    left_reaction_kn: float
    right_reaction_kn: float
    left_support_id: str
    right_support_id: str
    support_reactions_kn: dict[str, float] = Field(
        description="该组合下本跨全部支座的合力反力（含中间支座相邻 bay 叠加）")
    conservative: bool = Field(description="是否采用保守边界（不计弹性）")
    converged: bool


class ProfilePoint(BaseModel):
    """剖面标注：沿程每站的净空剖面数据。"""
    station_index: int
    position: Vec3
    walk_z: float
    controlling_anchor: Optional[str] = None
    anchor_z: Optional[float] = None
    free_fall_m: Optional[float] = None
    required_clearance_m: Optional[float] = None
    clearance_floor_z: Optional[float] = None
    available_clearance_m: Optional[float] = None
    margin_m: Optional[float] = None
    # 柔性跨段：控制性滑梭/跨段与动态下挠分量
    controlling_shuttle: Optional[str] = None
    span_id: Optional[str] = None
    cable_sag_m: Optional[float] = None
    total_fall_m: Optional[float] = None
    cable_tension_kn: Optional[float] = None


class AnalysisResult(BaseModel):
    conclusive: bool = Field(
        default=True, description="False 表示存在不下结论项，须人工复核")
    passable: bool
    first_failure: Optional[CheckFailure] = None
    failures: list[CheckFailure] = []
    first_open_item: Optional[OpenItem] = None
    open_items: list[OpenItem] = []
    sequence: list[SequenceEvent] = []
    profile: list[ProfilePoint] = []
    cable_results: list[CableResult] = []
    conservative_used: list[str] = Field(
        default_factory=list, description="实际采用了保守边界的跨段 id")
    station_count: int = 0


# ---------------------------------------------------------------- 方案与修订

class ChangeRecord(BaseModel):
    """修订变更说明：改跨、换滑梭或采用保守边界时必须给出理由。"""
    kind: Literal["span_change", "shuttle_change", "conservative_bounds"]
    reason: str = Field(min_length=1)
    detail: str = ""


class PlanCreate(BaseModel):
    name: str = Field(min_length=1)
    note: str = ""
    payload: PlanPayload
    changes: list[ChangeRecord] = []


class RevisionCreate(BaseModel):
    note: str = ""
    payload: PlanPayload
    changes: list[ChangeRecord] = []


class RevisionMeta(BaseModel):
    rev_no: int
    status: Literal["draft", "confirmed"]
    note: str
    created_at: str


class RevisionOut(BaseModel):
    plan_id: str
    rev_no: int
    status: Literal["draft", "confirmed"]
    note: str
    created_at: str
    payload: PlanPayload
    changes: list[ChangeRecord] = []
    analysis: AnalysisResult


class PlanOut(BaseModel):
    plan_id: str
    name: str
    created_at: str
    revisions: list[RevisionMeta]


# ---------------------------------------------------------------- 坠落后救援推演

class RescueEntry(BaseModel):
    """救援入口：救援人员由该位置进入路线接近伤员。"""
    id: str
    position: Vec3


class RescueAnchor(BaseModel):
    """可用救援锚点：位置、额定载荷（合力，kN）。"""
    id: str
    position: Vec3
    rated_load_kn: float = Field(gt=0)


class Rescuer(BaseModel):
    """救援人员：重量与持续牵引力由推演参数统一取值。"""
    id: str
    weight_kg: float = Field(gt=0)


class RopeTeam(BaseModel):
    """绳组：绳长、滑轮倍率 n（n:1）、滑轮效率、下降器限载（kN）。"""
    id: str
    rope_length_m: float = Field(gt=0)
    pulley_ratio: int = Field(default=3, ge=1, description="滑轮倍率 n（n:1）")
    pulley_efficiency: float = Field(default=0.8, gt=0, le=1)
    descender_limit_kn: float = Field(gt=0, description="下降器容许工作载荷")


class Stretcher(BaseModel):
    """担架包络：长×宽×高（m）与自重（kg）。"""
    id: str
    length_m: float = Field(gt=0)
    width_m: float = Field(gt=0)
    height_m: float = Field(gt=0)
    weight_kg: float = Field(default=0.0, ge=0)

    def bounding_radius_m(self) -> float:
        """保守外接球半径（半对角线）：轨迹净空按球体扫掠。"""
        return 0.5 * (self.length_m ** 2 + self.width_m ** 2
                      + self.height_m ** 2) ** 0.5


class RescueParams(BaseModel):
    """救援推演的作业参数（均给出工程缺省值，结果中逐步列明）。"""
    approach_speed_m_min: float = Field(default=25.0, gt=0)
    rig_secondary_min: float = Field(default=3.0, ge=0)
    rig_haul_min: float = Field(default=4.0, ge=0)
    detach_min_per_connection: float = Field(default=2.0, ge=0)
    rig_transfer_min: float = Field(default=3.0, ge=0)
    haul_rate_m_min: float = Field(default=6.0, gt=0,
                                   description="每分钟可拉过的绳程")
    lower_rate_m_min: float = Field(default=12.0, gt=0)
    anchor_rig_reach_m: float = Field(
        default=3.5, gt=0,
        description="救援人员立于站点时可挂接救援锚点的最大三维距离")
    entry_max_offset_m: float = Field(
        default=2.0, gt=0, description="救援入口偏离行走折线的容许距离")
    rope_tail_m: float = Field(default=2.0, ge=0, description="绳组操作尾绳余量")
    secondary_pretension_kn: float = Field(
        default=0.5, ge=0, description="二次保护绳张紧力")
    rescuer_pull_kn: float = Field(
        default=0.5, gt=0, description="单人可持续牵引力（kN）")
    descender_hold_kn: float = Field(
        default=0.3, ge=0, description="下降器操作尾绳持力（kN）")


class RescuePayload(BaseModel):
    """一版救援方案：入口、救援锚点、救援人员、绳组、担架与悬吊时限。"""
    entry: RescueEntry
    rescue_anchors: list[RescueAnchor] = Field(min_length=1)
    rescuers: list[Rescuer] = Field(min_length=1)
    rope_teams: list[RopeTeam] = Field(min_length=2,
                                       description="主绳组与二次保护绳组至少各一")
    secondary_rope_team_id: str = Field(description="二次保护专用绳组 id")
    stretcher: Stretcher
    max_suspension_minutes: float = Field(gt=0)
    landing_point: Optional[Vec3] = Field(
        default=None,
        description="转运目的地；缺省取救援入口位置（升降转运终点）")
    # 人工指定（缺省由推演按最近可达/最短够用自动选择），给出即须写理由
    primary_anchor_id: Optional[str] = None
    backup_anchor_id: Optional[str] = None
    rope_team_id: Optional[str] = None
    manual_reason: str = Field(
        default="", description="人工指定锚点/绳组的理由（人工决定，随作业包保留）")
    params: RescueParams = Field(default_factory=RescueParams)

    @model_validator(mode="after")
    def _check_refs(self) -> "RescuePayload":
        anchor_ids = [a.id for a in self.rescue_anchors]
        if len(anchor_ids) != len(set(anchor_ids)):
            raise ValueError("救援锚点 id 重复")
        rope_ids = [r.id for r in self.rope_teams]
        if len(rope_ids) != len(set(rope_ids)):
            raise ValueError("绳组 id 重复")
        rescuer_ids = [r.id for r in self.rescuers]
        if len(rescuer_ids) != len(set(rescuer_ids)):
            raise ValueError("救援人员 id 重复")
        amap = {a.id for a in self.rescue_anchors}
        rmap = set(rope_ids)
        if self.secondary_rope_team_id not in rmap:
            raise ValueError(
                f"二次保护绳组 {self.secondary_rope_team_id} 不存在")
        for fid in (self.primary_anchor_id, self.backup_anchor_id):
            if fid is not None and fid not in amap:
                raise ValueError(f"人工指定的救援锚点 {fid} 不存在")
        if self.rope_team_id is not None:
            if self.rope_team_id not in rmap:
                raise ValueError(f"人工指定的绳组 {self.rope_team_id} 不存在")
            if self.rope_team_id == self.secondary_rope_team_id:
                raise ValueError("主绳组与二次保护绳组不得为同一绳组（器材重复占用）")
        if self.primary_anchor_id is not None \
                and self.primary_anchor_id == self.backup_anchor_id:
            raise ValueError("主锚点与二次保护锚点不得为同一锚点（器材重复占用）")
        if (self.primary_anchor_id or self.backup_anchor_id or self.rope_team_id) \
                and not self.manual_reason.strip():
            raise ValueError("人工指定锚点或绳组必须在 manual_reason 中写明理由")
        return self


# ---------------------------------------------------------------- 救援推演结果

RescueStepCode = Literal["approach", "secondary_protection", "haul_unload",
                        "detach_original", "transfer"]


class RescueBlock(BaseModel):
    """最早受阻步骤：步骤、受阻代码、说明与全部计算分量。"""
    station_index: int
    person_id: Optional[str] = None
    step_no: int
    step_code: RescueStepCode
    code: str
    message: str
    components: dict[str, float | int | str | None] = {}


class RescueStep(BaseModel):
    """一个救援步骤的逐步计算分量。"""
    step_no: int
    code: RescueStepCode
    name: str
    rope_travel_m: float = 0.0
    rope_required_m: float = 0.0
    pull_force_kn: Optional[float] = None
    anchor_id: Optional[str] = None
    anchor_resultant_kn: Optional[float] = None
    descender_load_kn: Optional[float] = None
    path_clearance_ok: bool = True
    elapsed_minutes: float = 0.0
    cumulative_minutes: float = 0.0
    components: dict[str, float | int | str | None] = {}


class StationRescue(BaseModel):
    """单个悬吊站点（单名伤员）的救援推演。"""
    station_index: int
    position: Vec3
    person_id: str
    hang_position: Vec3
    feet_z: float
    original_connections: list[str] = []
    primary_anchor_id: Optional[str] = None
    backup_anchor_id: Optional[str] = None
    rope_team_id: Optional[str] = None
    secondary_rope_team_id: Optional[str] = None
    executable: bool
    block: Optional[RescueBlock] = None
    steps: list[RescueStep] = []
    total_elapsed_minutes: float = 0.0


class ManualDecision(BaseModel):
    field: str
    value: str
    reason: str


class RescueResult(BaseModel):
    """整条路线（全部悬吊站点）的救援推演结论。"""
    source_conclusive: bool
    source_passable: bool
    executable: bool = Field(description="任一悬吊站点受阻即为 False")
    earliest_block: Optional[RescueBlock] = None
    simulations: list[StationRescue] = []
    suspension_station_count: int = 0
    impact_stations: list[dict[str, float | int | str]] = []
    manual_decisions: list[ManualDecision] = []
    used_source_anchors: list[str] = []
    used_source_shuttles: list[str] = []
    used_persons: list[str] = []


# ---------------------------------------------------------------- 救援方案与修订

class RescueChangeRecord(BaseModel):
    """救援修订变更说明：改入口或换器材必须给出理由。"""
    kind: Literal["entry_change", "equipment_change"]
    reason: str = Field(min_length=1)
    detail: str = ""


class RescueCreate(BaseModel):
    name: str = Field(min_length=1)
    note: str = ""
    payload: RescuePayload
    changes: list[RescueChangeRecord] = []


class RescueRevisionCreate(BaseModel):
    note: str = ""
    payload: RescuePayload
    changes: list[RescueChangeRecord] = []


class RescuePlanOut(BaseModel):
    rescue_id: str
    plan_id: str
    name: str
    created_at: str
    revisions: list[RevisionMeta] = []


class RescueRevisionOut(BaseModel):
    rescue_id: str
    plan_id: str
    rev_no: int
    status: Literal["draft", "confirmed"]
    note: str
    created_at: str
    source_rev_no: int
    source_locked: bool = Field(description="确认版已冻结来源修订快照")
    recheck_required: bool = Field(
        description="原路线已有更新且影响本方案相关站点/器材，待复核")
    recheck_reason: str = ""
    payload: RescuePayload
    changes: list[RescueChangeRecord] = []
    source: dict
    result: RescueResult
