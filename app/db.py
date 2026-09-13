"""SQLite 持久层：方案与修订。确认稿不可覆盖，分析不落库、按版参数重算。"""
from __future__ import annotations

import os
import sqlite3
import uuid
from datetime import datetime, timezone

_SCHEMA = """
CREATE TABLE IF NOT EXISTS plans (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS revisions (
    plan_id TEXT NOT NULL REFERENCES plans(id),
    rev_no INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'draft',
    note TEXT NOT NULL DEFAULT '',
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (plan_id, rev_no)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path: str | None = None):
        self.path = path or os.environ.get("LIFELINE_DB", "lifeline.db")
        self._init()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._conn() as c:
            c.executescript(_SCHEMA)

    # ------------------------------------------------------------- 方案
    def create_plan(self, name: str, note: str, payload_json: str) -> tuple[str, int]:
        plan_id = uuid.uuid4().hex[:12]
        with self._conn() as c:
            c.execute("INSERT INTO plans(id, name, created_at) VALUES (?,?,?)",
                      (plan_id, name, _now()))
            self._insert_revision(c, plan_id, 1, note, payload_json)
        return plan_id, 1

    def get_plan(self, plan_id: str) -> sqlite3.Row | None:
        with self._conn() as c:
            return c.execute("SELECT * FROM plans WHERE id=?",
                             (plan_id,)).fetchone()

    def list_plans(self) -> list[sqlite3.Row]:
        with self._conn() as c:
            return c.execute("SELECT * FROM plans ORDER BY created_at").fetchall()

    # ------------------------------------------------------------- 修订
    def _insert_revision(self, c: sqlite3.Connection, plan_id: str,
                         rev_no: int, note: str, payload_json: str) -> None:
        c.execute(
            "INSERT INTO revisions(plan_id, rev_no, status, note, payload,"
            " created_at) VALUES (?,?,?,?,?,?)",
            (plan_id, rev_no, "draft", note, payload_json, _now()))

    def add_revision(self, plan_id: str, note: str, payload_json: str) -> int:
        with self._conn() as c:
            row = c.execute(
                "SELECT COALESCE(MAX(rev_no), 0) AS m FROM revisions"
                " WHERE plan_id=?", (plan_id,)).fetchone()
            rev_no = row["m"] + 1
            self._insert_revision(c, plan_id, rev_no, note, payload_json)
            return rev_no

    def get_revision(self, plan_id: str, rev_no: int) -> sqlite3.Row | None:
        with self._conn() as c:
            return c.execute(
                "SELECT * FROM revisions WHERE plan_id=? AND rev_no=?",
                (plan_id, rev_no)).fetchone()

    def list_revisions(self, plan_id: str) -> list[sqlite3.Row]:
        with self._conn() as c:
            return c.execute(
                "SELECT rev_no, status, note, created_at FROM revisions"
                " WHERE plan_id=? ORDER BY rev_no", (plan_id,)).fetchall()

    def update_draft_payload(self, plan_id: str, rev_no: int,
                             payload_json: str) -> bool:
        """仅草稿可改；确认稿返回 False（不可覆盖）。"""
        with self._conn() as c:
            cur = c.execute(
                "UPDATE revisions SET payload=? WHERE plan_id=? AND rev_no=?"
                " AND status='draft'",
                (payload_json, plan_id, rev_no))
            return cur.rowcount == 1

    def confirm_revision(self, plan_id: str, rev_no: int) -> bool:
        with self._conn() as c:
            cur = c.execute(
                "UPDATE revisions SET status='confirmed' WHERE plan_id=?"
                " AND rev_no=? AND status='draft'", (plan_id, rev_no))
            return cur.rowcount == 1
