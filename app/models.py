"""Pydantic 请求/响应模型：线路方案、人员装备、核算结果。"""
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
    """锚点：坐标、额定载荷、允许受力方向锥、多人共用限制。"""
    id: str
    position: Vec3
    rated_load_kn: float = Field(gt=0)
    allowed_axis: Optional[Vec3] = Field(
        default=None, description="允许受力方向锥轴线（单位向量），None 表示全向"
    )
    allowed_half_angle_deg: float = Field(default=180.0, ge=0, le=180)
    max_users: int = Field(default=1, ge=1, description="同时共用人数上限")


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


class RouteSpec(BaseModel):
    walk_polyline: list[Vec3] = Field(min_length=2)
    drop_edges: list[DropEdge] = []
    obstacles: list[Obstacle] = []
    anchors: list[Anchor] = Field(min_length=1)


class ManualHookAction(BaseModel):
    """人工挂接动作：在指定站点对某钩执行挂接/换钩/解钩。"""
    person_id: str
    hook: Literal["A", "B"]
    action: Literal["attach", "switch", "detach"]
    anchor: Optional[str] = Field(
        default=None, description="目标锚点 id（attach/switch 必填，detach 省略）")
    station_index: int = Field(ge=0, description="动作发生的站点序号")

    @model_validator(mode="after")
    def _check_anchor(self) -> "ManualHookAction":
        if self.action in ("attach", "switch") and not self.anchor:
            raise ValueError("attach/switch 必须给出目标锚点 anchor")
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

    @model_validator(mode="after")
    def _check_refs(self) -> "PlanPayload":
        eq_ids = {e.id for e in self.equipment}
        for p in self.persons:
            if p.equipment_id not in eq_ids:
                raise ValueError(f"人员 {p.id} 引用了不存在的装备 {p.equipment_id}")
        anchor_ids = [a.id for a in self.route.anchors]
        if len(anchor_ids) != len(set(anchor_ids)):
            raise ValueError("锚点 id 重复")
        person_ids = {p.id for p in self.persons}
        anchor_id_set = set(anchor_ids)
        for act in self.hook_order:
            if act.person_id not in person_ids:
                raise ValueError(
                    f"人工动作引用了不存在的人员 {act.person_id}")
            if act.anchor is not None and act.anchor not in anchor_id_set:
                raise ValueError(
                    f"人工动作引用了不存在的锚点 {act.anchor}")
        return self


# ---------------------------------------------------------------- 核算结果

Action = Literal["attach", "switch", "detach", "traverse"]


class SequenceEvent(BaseModel):
    """挂接 / 换钩 / 解钩 事件。"""
    station_index: int
    position: Vec3
    person_id: str
    action: Literal["attach", "switch", "detach"]
    hook: Literal["A", "B"]
    from_anchor: Optional[str] = None
    to_anchor: Optional[str] = None
    attached_after: list[str] = []


class CheckFailure(BaseModel):
    """一次核算失败：位置、动作、检查项与计算分量。"""
    station_index: int
    position: Vec3
    person_id: Optional[str] = None
    action: Action
    check: str
    message: str
    components: dict[str, float | int | str | None] = {}


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


class AnalysisResult(BaseModel):
    passable: bool
    first_failure: Optional[CheckFailure] = None
    failures: list[CheckFailure] = []
    sequence: list[SequenceEvent] = []
    profile: list[ProfilePoint] = []
    station_count: int = 0


# ---------------------------------------------------------------- 方案与修订

class PlanCreate(BaseModel):
    name: str = Field(min_length=1)
    note: str = ""
    payload: PlanPayload


class RevisionCreate(BaseModel):
    note: str = ""
    payload: PlanPayload


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
    analysis: AnalysisResult


class PlanOut(BaseModel):
    plan_id: str
    name: str
    created_at: str
    revisions: list[RevisionMeta]
