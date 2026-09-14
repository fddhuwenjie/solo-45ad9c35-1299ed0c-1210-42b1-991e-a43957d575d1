"""FastAPI 路由：线路方案与修订的保存、确认、回看（按版参数重算）。"""
from __future__ import annotations

import json

from fastapi import Depends, FastAPI, HTTPException

from .db import Store
from .engine import analyze, build_context
from .engine.rescue import simulate_rescue, source_relevance_signature
from .models import (AnalysisResult, ChangeRecord, PlanCreate, PlanOut,
                     PlanPayload, RescueChangeRecord, RescueCreate,
                     RescuePlanOut, RescuePayload, RescueRevisionCreate,
                     RescueRevisionOut, RevisionCreate, RevisionMeta,
                     RevisionOut)

app = FastAPI(title="生命线通行核算接口", version="1.3.0")

_store: Store | None = None


def get_store() -> Store:
    global _store
    if _store is None:
        _store = Store()
    return _store


def _load_revision(store: Store, plan_id: str, rev_no: int):
    row = store.get_revision(plan_id, rev_no)
    if row is None:
        raise HTTPException(404, f"修订不存在: {plan_id}/r{rev_no}")
    return row


def _changes(row) -> list[ChangeRecord]:
    raw = row["changes"] if "changes" in row.keys() else "[]"
    if not raw:
        return []
    try:
        return [ChangeRecord.model_validate(x) for x in json.loads(raw)]
    except (ValueError, TypeError):
        return []


def _revision_out(store: Store, plan_id: str, rev_no: int) -> RevisionOut:
    row = _load_revision(store, plan_id, rev_no)
    payload = PlanPayload.model_validate_json(row["payload"])
    return RevisionOut(
        plan_id=plan_id, rev_no=row["rev_no"], status=row["status"],
        note=row["note"], created_at=row["created_at"],
        payload=payload, changes=_changes(row), analysis=analyze(payload))


def _lanyard_signature(p: PlanPayload):
    """Y 型双腿系绳的结构性参数：腿长、轴向刚度、侧载限值、钩腿绑定、
    缓冲力—行程曲线、最大行程、能量容量、允许夹角。变化即重大修改。"""
    def tw(e):
        if e.twin_leg is None:
            return None
        t = e.twin_leg
        return (
            [(l.id, l.hook, l.leg_length_m, l.axial_stiffness_kn,
              l.connector_side_load_limit_kn) for l in t.legs],
            [(pt.travel_m, pt.force_kn) for pt in t.buffer_curve],
            t.max_travel_m, t.energy_capacity_j, t.max_included_angle_deg,
        )
    return [(e.id, tw(e)) for e in p.equipment]


def _structural_route_signature(p: PlanPayload):
    """跨段 / 滑梭 / 双腿系绳的结构性参数（不含说明性字段）；变化即重大修改。"""
    return (
        [(s.id, s.supports, s.pretension_kn, s.line_density_kg_m,
          s.axial_stiffness_kn, s.cross_section_m2, s.elastic_modulus_kn_m2,
          s.max_sag_m, s.shuttle_pass, s.max_users)
         for s in p.route.spans],
        [(s.id, s.span_id, s.connector_reach_m, s.can_pass, s.max_users)
         for s in p.route.shuttles],
        [(s.id, s.position.model_dump(), s.rated_load_kn,
          s.allowed_axis.model_dump() if s.allowed_axis else None,
          s.allowed_half_angle_deg) for s in p.route.supports],
        _lanyard_signature(p),
    )


def _require_change_reason(body: RevisionCreate | PlanCreate,
                           payload: PlanPayload,
                           prev_payload: PlanPayload | None = None) -> None:
    """改跨、换滑梭或采用保守边界，必须在 changes 中给出理由（新修订）。"""
    kinds = {c.kind for c in body.changes}
    if payload.conservative_bounds.enabled \
            and "conservative_bounds" not in kinds:
        raise HTTPException(
            422, "采用保守边界必须在 changes 中给出 "
                 "kind=conservative_bounds 的理由")
    if prev_payload is not None:
        sig_prev = _structural_route_signature(prev_payload)
        sig_new = _structural_route_signature(payload)
        if sig_prev != sig_new:
            span_changed = sig_prev[0] != sig_new[0] or sig_prev[2] != sig_new[2]
            shuttle_changed = sig_prev[1] != sig_new[1]
            lanyard_changed = sig_prev[3] != sig_new[3]
            if span_changed and "span_change" not in kinds:
                raise HTTPException(
                    422, "对柔性跨段/端座/中间支座的重大修改"
                         "（端座、预张力、线密度、弹性参数、限值、"
                         "滑梭过支座能力、容许人数等）必须在 changes 中给出 "
                         "kind=span_change 的理由")
            if shuttle_changed and "shuttle_change" not in kinds:
                raise HTTPException(
                    422, "更换/修改滑梭必须在 changes 中给出 "
                         "kind=shuttle_change 的理由")
            if lanyard_changed and "lanyard_change" not in kinds:
                raise HTTPException(
                    422, "修改 Y 型双腿系绳（腿原长、轴向刚度、缓冲力—行程"
                         "曲线、最大行程/能量容量、允许夹角、连接器侧载限值"
                         "或钩腿绑定）必须在 changes 中给出 "
                         "kind=lanyard_change 的理由")


# ---------------------------------------------------------------- 方案

@app.post("/plans", status_code=201)
def create_plan(body: PlanCreate, store: Store = Depends(get_store)):
    """创建线路方案，同时保存第 1 版修订（草稿）。"""
    _require_change_reason(body, body.payload)
    plan_id, rev_no = store.create_plan(
        body.name, body.note, body.payload.model_dump_json(),
        json.dumps([c.model_dump() for c in body.changes], ensure_ascii=False))
    return _revision_out(store, plan_id, rev_no)


@app.get("/plans")
def list_plans(store: Store = Depends(get_store)):
    out = []
    for p in store.list_plans():
        revs = [RevisionMeta(rev_no=r["rev_no"], status=r["status"],
                             note=r["note"], created_at=r["created_at"])
                for r in store.list_revisions(p["id"])]
        out.append(PlanOut(plan_id=p["id"], name=p["name"],
                           created_at=p["created_at"], revisions=revs))
    return out


@app.get("/plans/{plan_id}", response_model=PlanOut)
def get_plan(plan_id: str, store: Store = Depends(get_store)):
    p = store.get_plan(plan_id)
    if p is None:
        raise HTTPException(404, f"方案不存在: {plan_id}")
    revs = [RevisionMeta(rev_no=r["rev_no"], status=r["status"],
                         note=r["note"], created_at=r["created_at"])
            for r in store.list_revisions(plan_id)]
    return PlanOut(plan_id=p["id"], name=p["name"],
                   created_at=p["created_at"], revisions=revs)


# ---------------------------------------------------------------- 修订

@app.post("/plans/{plan_id}/revisions", status_code=201)
def add_revision(plan_id: str, body: RevisionCreate,
                 store: Store = Depends(get_store)):
    """另存修订：改跨、换滑梭、换装备或采用保守边界后，以新修订保存
    （不改动既有版本）；相应改动须在 changes 中说明理由。"""
    if store.get_plan(plan_id) is None:
        raise HTTPException(404, f"方案不存在: {plan_id}")
    prev_row = store.get_latest_revision(plan_id)
    prev_payload = (PlanPayload.model_validate_json(prev_row["payload"])
                    if prev_row is not None else None)
    _require_change_reason(body, body.payload, prev_payload)
    rev_no = store.add_revision(
        plan_id, body.note, body.payload.model_dump_json(),
        json.dumps([c.model_dump() for c in body.changes], ensure_ascii=False))
    return _revision_out(store, plan_id, rev_no)


@app.get("/plans/{plan_id}/revisions/{rev_no}", response_model=RevisionOut)
def get_revision(plan_id: str, rev_no: int, store: Store = Depends(get_store)):
    """回看任一修订：动作序列、失败依据、剖面标注均由**该版参数重算**。"""
    return _revision_out(store, plan_id, rev_no)


@app.get("/plans/{plan_id}/revisions/{rev_no}/analysis",
         response_model=AnalysisResult)
def get_analysis(plan_id: str, rev_no: int, store: Store = Depends(get_store)):
    row = _load_revision(store, plan_id, rev_no)
    payload = PlanPayload.model_validate_json(row["payload"])
    return analyze(payload)


@app.put("/plans/{plan_id}/revisions/{rev_no}", response_model=RevisionOut)
def update_revision(plan_id: str, rev_no: int, body: RevisionCreate,
                    store: Store = Depends(get_store)):
    """覆盖草稿参数；确认稿不可覆盖（409）。

    柔性跨段/支座/滑梭的重大结构性修改不得覆盖既有草稿（即便已附
    span_change 理由）：必须 POST 另存修订，保留旧版冻结参数。
    点锚、人员、装备、动作次序等非结构参数仍可覆盖草稿。
    """
    row = _load_revision(store, plan_id, rev_no)
    if row["status"] == "confirmed":
        raise HTTPException(409, "确认稿不可覆盖，请另存修订")
    prev_payload = PlanPayload.model_validate_json(row["payload"])
    sig_prev = _structural_route_signature(prev_payload)
    sig_new = _structural_route_signature(body.payload)
    if sig_prev != sig_new:
        if sig_prev[0] != sig_new[0] or sig_prev[2] != sig_new[2]:
            what = "柔性跨段/端座/中间支座"
        elif sig_prev[1] != sig_new[1]:
            what = "滑梭"
        else:
            what = "Y 型双腿系绳（腿长/缓冲曲线/钩腿绑定）"
        raise HTTPException(
            409, f"{what}的重大修改必须另存修订（POST .../revisions）以保留"
                 f"旧版参数，不允许覆盖既有草稿；请在新修订 changes 中"
                 f"附 span_change/shuttle_change/lanyard_change 理由")
    _require_change_reason(body, body.payload, prev_payload)
    ok = store.update_draft_payload(
        plan_id, rev_no, body.payload.model_dump_json(),
        json.dumps([c.model_dump() for c in body.changes], ensure_ascii=False))
    if not ok:
        raise HTTPException(409, "确认稿不可覆盖，请另存修订")
    return _revision_out(store, plan_id, rev_no)


@app.post("/plans/{plan_id}/revisions/{rev_no}/confirm",
          response_model=RevisionOut)
def confirm_revision(plan_id: str, rev_no: int,
                     store: Store = Depends(get_store)):
    """确认定稿：定稿后该修订冻结，不可再覆盖。"""
    _load_revision(store, plan_id, rev_no)
    store.confirm_revision(plan_id, rev_no)
    return _revision_out(store, plan_id, rev_no)


# ================================================================ 救援推演

def _rescue_changes(row) -> list[RescueChangeRecord]:
    raw = row["changes"] if "changes" in row.keys() else "[]"
    if not raw:
        return []
    try:
        return [RescueChangeRecord.model_validate(x)
                for x in json.loads(raw)]
    except (ValueError, TypeError):
        return []


def _rescue_structural_signature(rp: RescuePayload):
    """入口/器材的结构性参数（不含说明性字段）；变化即重大修改。"""
    return (
        (rp.entry.id, rp.entry.position.model_dump()),
        [(a.id, a.position.model_dump(), a.rated_load_kn)
         for a in rp.rescue_anchors],
        [(r.id, r.rope_length_m, r.pulley_ratio, r.pulley_efficiency,
          r.descender_limit_kn) for r in rp.rope_teams],
        rp.secondary_rope_team_id,
        (rp.stretcher.id, rp.stretcher.length_m, rp.stretcher.width_m,
         rp.stretcher.height_m, rp.stretcher.weight_kg),
        rp.max_suspension_minutes,
        rp.landing_point.model_dump() if rp.landing_point else None,
    )


def _require_rescue_change_reason(body, rp: RescuePayload,
                                  prev: RescuePayload | None) -> None:
    """改入口或换器材必须在 changes 中给出理由（另存新修订）。"""
    kinds = {c.kind for c in body.changes}
    if prev is not None:
        sig_prev = _rescue_structural_signature(prev)
        sig_new = _rescue_structural_signature(rp)
        if sig_prev != sig_new:
            entry_changed = sig_prev[0] != sig_new[0]
            equip_changed = sig_prev[1:6] != sig_new[1:6]
            if entry_changed and "entry_change" not in kinds:
                raise HTTPException(
                    422, "修改救援入口必须在 changes 中给出 "
                         "kind=entry_change 的理由")
            if equip_changed and "equipment_change" not in kinds:
                raise HTTPException(
                    422, "更换救援锚点/绳组/担架或调整悬吊时限必须在 changes"
                         " 中给出 kind=equipment_change 的理由")


def _load_source(store: Store, plan_id: str, source_rev_no: int):
    row = store.get_revision(plan_id, source_rev_no)
    if row is None:
        raise HTTPException(
            404, f"来源修订不存在: {plan_id}/r{source_rev_no}")
    payload = PlanPayload.model_validate_json(row["payload"])
    return row, payload


def _recompute_rescue(store: Store, rp_row, rescue_row, source_row,
                      source_payload: PlanPayload) -> RescueRevisionOut:
    """按来源修订（或确认时冻结快照）重算救援推演，并判断是否待复核。"""
    rp = RescuePayload.model_validate_json(rescue_row["payload"])
    use_snapshot = (rescue_row["status"] == "confirmed"
                    and rescue_row["source_snapshot"])
    recheck = False
    recheck_reason = ""
    source_rev_no = rp_row["source_rev_no"]
    if use_snapshot:
        snap = json.loads(rescue_row["source_snapshot"])
        calc_payload = PlanPayload.model_validate(snap["payload"])
        source_locked = True
        source_status = snap.get("source_status", "confirmed")
        source_note = snap.get("source_note", "")
        source_created = snap.get("source_created_at",
                                  source_row["created_at"])
        source_rev_no = snap["rev_no"]
    else:
        calc_payload = source_payload
        source_locked = False
        source_status = source_row["status"]
        source_note = source_row["note"]
        source_created = source_row["created_at"]
    ctx = build_context(calc_payload)
    result = simulate_rescue(ctx, rp)

    # 原路线变化：只让相关救援方案待复核（按最新路线重算推演后比对签名）
    if use_snapshot and rescue_row["source_sig"]:
        frozen_sig = json.loads(rescue_row["source_sig"])
        latest = store.get_latest_revision(rp_row["plan_id"])
        if latest is not None and latest["rev_no"] != source_rev_no:
            latest_payload = PlanPayload.model_validate_json(
                latest["payload"])
            latest_result = simulate_rescue(
                build_context(latest_payload), rp)
            latest_sig = json.loads(json.dumps(
                source_relevance_signature(latest_payload, latest_result, rp),
                ensure_ascii=False, default=list))
            if frozen_sig != latest_sig:
                recheck = True
                recheck_reason = (
                    f"原路线已由 r{source_rev_no} 更新至 r{latest['rev_no']}，"
                    f"且本方案相关的站点/人员/锚点/跨段参数发生变化，"
                    f"确认结果待复核")
    return RescueRevisionOut(
        rescue_id=rp_row["id"], plan_id=rp_row["plan_id"],
        rev_no=rescue_row["rev_no"], status=rescue_row["status"],
        note=rescue_row["note"], created_at=rescue_row["created_at"],
        source_rev_no=source_rev_no,
        source_locked=source_locked,
        recheck_required=recheck, recheck_reason=recheck_reason,
        payload=rp, changes=_rescue_changes(rescue_row),
        source={"status": source_status, "note": source_note,
                "created_at": source_created,
                "conclusive": ctx.result.conclusive,
                "passable": ctx.result.passable},
        result=result)


@app.post("/plans/{plan_id}/revisions/{rev_no}/rescue",
          status_code=201, response_model=RescueRevisionOut)
def create_rescue_plan(plan_id: str, rev_no: int, body: RescueCreate,
                       store: Store = Depends(get_store)):
    """登记救援方案：复用已冻结（confirmed）的来源修订，保存为独立救援
    修订（草稿）。来源修订未确认时 409——坠距/反力尚未冻结，不得登记。"""
    source_row, source_payload = _load_source(store, plan_id, rev_no)
    if source_row["status"] != "confirmed":
        raise HTTPException(
            409, f"来源修订 {plan_id}/r{rev_no} 尚未确认冻结，请先确认该"
                 f"修订后再登记救援方案")
    _require_rescue_change_reason(body, body.payload, None)
    rescue_id, rrev = store.create_rescue_plan(
        plan_id, rev_no, body.name, body.note,
        body.payload.model_dump_json(),
        json.dumps([c.model_dump() for c in body.changes],
                   ensure_ascii=False))
    rr = store.get_rescue_revision(rescue_id, rrev)
    rp_row2 = store.get_rescue_plan(rescue_id)
    return _recompute_rescue(store, rp_row2, rr, source_row, source_payload)


@app.get("/rescue", response_model=list[RescuePlanOut])
def list_rescue_plans(plan_id: str | None = None,
                      store: Store = Depends(get_store)):
    out = []
    for rp_row in store.list_rescue_plans(plan_id):
        revs = [RevisionMeta(rev_no=r["rev_no"], status=r["status"],
                             note=r["note"], created_at=r["created_at"])
                for r in store.list_rescue_revisions(rp_row["id"])]
        out.append(RescuePlanOut(rescue_id=rp_row["id"],
                                 plan_id=rp_row["plan_id"],
                                 name=rp_row["name"],
                                 created_at=rp_row["created_at"],
                                 revisions=revs))
    return out


@app.get("/rescue/{rescue_id}", response_model=RescuePlanOut)
def get_rescue_plan(rescue_id: str, store: Store = Depends(get_store)):
    rp_row = store.get_rescue_plan(rescue_id)
    if rp_row is None:
        raise HTTPException(404, f"救援方案不存在: {rescue_id}")
    revs = [RevisionMeta(rev_no=r["rev_no"], status=r["status"],
                         note=r["note"], created_at=r["created_at"])
            for r in store.list_rescue_revisions(rescue_id)]
    return RescuePlanOut(rescue_id=rp_row["id"], plan_id=rp_row["plan_id"],
                         name=rp_row["name"], created_at=rp_row["created_at"],
                         revisions=revs)


def _load_rr(store: Store, rescue_id: str, rrev: int):
    rp_row = store.get_rescue_plan(rescue_id)
    if rp_row is None:
        raise HTTPException(404, f"救援方案不存在: {rescue_id}")
    rr = store.get_rescue_revision(rescue_id, rrev)
    if rr is None:
        raise HTTPException(404, f"救援修订不存在: {rescue_id}/r{rrev}")
    source_row = store.get_revision(rp_row["plan_id"],
                                    rp_row["source_rev_no"])
    source_payload = PlanPayload.model_validate_json(source_row["payload"])
    return rp_row, rr, source_row, source_payload


@app.post("/rescue/{rescue_id}/revisions", status_code=201,
          response_model=RescueRevisionOut)
def add_rescue_revision(rescue_id: str, body: RescueRevisionCreate,
                        store: Store = Depends(get_store)):
    """另存救援修订：人工改入口或换器材后以新版本保存，须在 changes 写理由。"""
    rp_row = store.get_rescue_plan(rescue_id)
    if rp_row is None:
        raise HTTPException(404, f"救援方案不存在: {rescue_id}")
    prev_row = store.get_rescue_revision(
        rescue_id,
        store.list_rescue_revisions(rescue_id)[-1]["rev_no"])
    prev_payload = RescuePayload.model_validate_json(prev_row["payload"])
    _require_rescue_change_reason(body, body.payload, prev_payload)
    rrev = store.add_rescue_revision(
        rescue_id, body.note, body.payload.model_dump_json(),
        json.dumps([c.model_dump() for c in body.changes],
                   ensure_ascii=False))
    source_row, source_payload = _load_source(
        store, rp_row["plan_id"], rp_row["source_rev_no"])
    rr = store.get_rescue_revision(rescue_id, rrev)
    return _recompute_rescue(store, rp_row, rr, source_row, source_payload)


@app.get("/rescue/{rescue_id}/revisions/{rrev}",
         response_model=RescueRevisionOut)
def get_rescue_revision(rescue_id: str, rrev: int,
                        store: Store = Depends(get_store)):
    rp_row, rr, source_row, source_payload = _load_rr(
        store, rescue_id, rrev)
    return _recompute_rescue(store, rp_row, rr, source_row, source_payload)


@app.put("/rescue/{rescue_id}/revisions/{rrev}",
         response_model=RescueRevisionOut)
def update_rescue_revision(rescue_id: str, rrev: int,
                           body: RescueRevisionCreate,
                           store: Store = Depends(get_store)):
    """覆盖救援草稿；确认稿 409。改入口/换器材的重大修改必须另存修订。"""
    rp_row, rr, source_row, source_payload = _load_rr(
        store, rescue_id, rrev)
    if rr["status"] == "confirmed":
        raise HTTPException(409, "确认稿不可覆盖，请另存救援修订")
    prev_payload = RescuePayload.model_validate_json(rr["payload"])
    if _rescue_structural_signature(prev_payload) \
            != _rescue_structural_signature(body.payload):
        raise HTTPException(
            409, "改救援入口或更换救援锚点/绳组/担架/悬吊时限必须另存救援修订"
                 "（POST .../revisions）并在 changes 中附 entry_change/"
                 "equipment_change 理由，不允许覆盖既有草稿")
    _require_rescue_change_reason(body, body.payload, prev_payload)
    ok = store.update_draft_rescue(
        rescue_id, rrev, body.payload.model_dump_json(),
        json.dumps([c.model_dump() for c in body.changes],
                   ensure_ascii=False),
        note=body.note)
    if not ok:
        raise HTTPException(409, "确认稿不可覆盖，请另存救援修订")
    rr = store.get_rescue_revision(rescue_id, rrev)
    return _recompute_rescue(store, rp_row, rr, source_row, source_payload)


@app.post("/rescue/{rescue_id}/revisions/{rrev}/confirm",
          response_model=RescueRevisionOut)
def confirm_rescue_revision(rescue_id: str, rrev: int,
                            store: Store = Depends(get_store)):
    """确认救援方案：锁定来源修订快照与计算结果（即便推演受阻也可确认登记，
    但 executable=False 绝不标为可执行）。"""
    rp_row, rr, source_row, source_payload = _load_rr(
        store, rescue_id, rrev)
    if rr["status"] != "draft":
        raise HTTPException(409, "该救援修订已确认，不可重复确认")
    rp_payload = RescuePayload.model_validate_json(rr["payload"])
    ctx = build_context(source_payload)
    result = simulate_rescue(ctx, rp_payload)
    snapshot = json.dumps({
        "rev_no": rp_row["source_rev_no"],
        "payload": json.loads(source_payload.model_dump_json()),
        "source_status": source_row["status"],
        "source_note": source_row["note"],
        "source_created_at": source_row["created_at"],
    }, ensure_ascii=False)
    sig = json.dumps(
        source_relevance_signature(source_payload, result, rp_payload),
        ensure_ascii=False, default=list)
    if not store.confirm_rescue_revision(rescue_id, rrev, snapshot, sig):
        raise HTTPException(409, "确认失败（仅草稿可确认）")
    rr = store.get_rescue_revision(rescue_id, rrev)
    return _recompute_rescue(store, rp_row, rr, source_row, source_payload)


@app.get("/rescue/{rescue_id}/revisions/{rrev}/job-package",
         response_model=RescueRevisionOut)
def rescue_job_package(rescue_id: str, rrev: int,
                       store: Store = Depends(get_store)):
    """JSON 作业包：逐步步骤、计算分量与人工决定（确认版按冻结来源重算）。"""
    rp_row, rr, source_row, source_payload = _load_rr(
        store, rescue_id, rrev)
    return _recompute_rescue(store, rp_row, rr, source_row, source_payload)
