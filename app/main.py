"""FastAPI 路由：线路方案与修订的保存、确认、回看（按版参数重算）。"""
from __future__ import annotations

import json

from fastapi import Depends, FastAPI, HTTPException

from .db import Store
from .engine import analyze
from .models import (AnalysisResult, ChangeRecord, PlanCreate, PlanOut,
                     PlanPayload, RevisionCreate, RevisionMeta, RevisionOut)

app = FastAPI(title="生命线通行核算接口", version="1.1.0")

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


def _structural_route_signature(p: PlanPayload):
    """跨段 / 滑梭的结构性参数（不含说明性字段）；变化即重大修改。"""
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
        what = "柔性跨段/端座/中间支座" if sig_prev[0] != sig_new[0] \
            or sig_prev[2] != sig_new[2] else "滑梭"
        raise HTTPException(
            409, f"{what}的重大修改必须另存修订（POST .../revisions）以保留"
                 f"旧版参数，不允许覆盖既有草稿；请在新修订 changes 中"
                 f"附 span_change/shuttle_change 理由")
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
