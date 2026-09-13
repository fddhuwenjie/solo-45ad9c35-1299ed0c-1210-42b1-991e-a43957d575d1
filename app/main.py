"""FastAPI 路由：线路方案与修订的保存、确认、回看（按版参数重算）。"""
from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException

from .db import Store
from .engine import analyze
from .models import (AnalysisResult, PlanCreate, PlanOut, PlanPayload,
                     RevisionCreate, RevisionMeta, RevisionOut)

app = FastAPI(title="生命线通行核算接口", version="1.0.0")

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


def _revision_out(store: Store, plan_id: str, rev_no: int) -> RevisionOut:
    row = _load_revision(store, plan_id, rev_no)
    payload = PlanPayload.model_validate_json(row["payload"])
    return RevisionOut(
        plan_id=plan_id, rev_no=row["rev_no"], status=row["status"],
        note=row["note"], created_at=row["created_at"],
        payload=payload, analysis=analyze(payload))


# ---------------------------------------------------------------- 方案

@app.post("/plans", status_code=201)
def create_plan(body: PlanCreate, store: Store = Depends(get_store)):
    """创建线路方案，同时保存第 1 版修订（草稿）。"""
    plan_id, rev_no = store.create_plan(
        body.name, body.note, body.payload.model_dump_json())
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
    """另存修订：人工调序或换装备后，以新修订保存（不改动既有版本）。"""
    if store.get_plan(plan_id) is None:
        raise HTTPException(404, f"方案不存在: {plan_id}")
    rev_no = store.add_revision(plan_id, body.note,
                                body.payload.model_dump_json())
    return _revision_out(store, plan_id, rev_no)


@app.get("/plans/{plan_id}/revisions/{rev_no}", response_model=RevisionOut)
def get_revision(plan_id: str, rev_no: int, store: Store = Depends(get_store)):
    """回看任一修订：动作序列、失败依据、剖面标注均由该版参数重算。"""
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
    """覆盖草稿参数；确认稿不可覆盖，返回 409。"""
    row = _load_revision(store, plan_id, rev_no)
    if row["status"] == "confirmed":
        raise HTTPException(409, "确认稿不可覆盖，请另存修订")
    store.update_draft_payload(plan_id, rev_no, body.payload.model_dump_json())
    return _revision_out(store, plan_id, rev_no)


@app.post("/plans/{plan_id}/revisions/{rev_no}/confirm",
          response_model=RevisionOut)
def confirm_revision(plan_id: str, rev_no: int,
                     store: Store = Depends(get_store)):
    """确认定稿：定稿后该修订冻结，不可再覆盖。"""
    _load_revision(store, plan_id, rev_no)
    store.confirm_revision(plan_id, rev_no)
    return _revision_out(store, plan_id, rev_no)
