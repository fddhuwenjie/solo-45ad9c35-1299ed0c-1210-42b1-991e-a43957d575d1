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
    changes TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    PRIMARY KEY (plan_id, rev_no)
);
CREATE TABLE IF NOT EXISTS rescue_plans (
    id TEXT PRIMARY KEY,
    plan_id TEXT NOT NULL REFERENCES plans(id),
    source_rev_no INTEGER NOT NULL,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rescue_revisions (
    rescue_id TEXT NOT NULL REFERENCES rescue_plans(id),
    rev_no INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'draft',
    note TEXT NOT NULL DEFAULT '',
    payload TEXT NOT NULL,
    changes TEXT NOT NULL DEFAULT '[]',
    source_snapshot TEXT,
    source_sig TEXT,
    recheck_flag INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    PRIMARY KEY (rescue_id, rev_no)
);
CREATE INDEX IF NOT EXISTS idx_rescue_by_plan ON rescue_revisions(rescue_id);
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
            # 既有数据库（旧 schema）补齐变更说明列；确认版仍只读
            cols = {r["name"] for r in c.execute(
                "PRAGMA table_info(revisions)").fetchall()}
            if "changes" not in cols:
                c.execute("ALTER TABLE revisions ADD COLUMN changes"
                          " TEXT NOT NULL DEFAULT '[]'")

    # ------------------------------------------------------------- 方案
    def create_plan(self, name: str, note: str, payload_json: str,
                    changes_json: str = "[]") -> tuple[str, int]:
        plan_id = uuid.uuid4().hex[:12]
        with self._conn() as c:
            c.execute("INSERT INTO plans(id, name, created_at) VALUES (?,?,?)",
                      (plan_id, name, _now()))
            self._insert_revision(c, plan_id, 1, note, payload_json,
                                  changes_json)
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
                         rev_no: int, note: str, payload_json: str,
                         changes_json: str = "[]") -> None:
        c.execute(
            "INSERT INTO revisions(plan_id, rev_no, status, note, payload,"
            " changes, created_at) VALUES (?,?,?,?,?,?,?)",
            (plan_id, rev_no, "draft", note, payload_json,
             changes_json, _now()))

    def add_revision(self, plan_id: str, note: str, payload_json: str,
                     changes_json: str = "[]") -> int:
        with self._conn() as c:
            row = c.execute(
                "SELECT COALESCE(MAX(rev_no), 0) AS m FROM revisions"
                " WHERE plan_id=?", (plan_id,)).fetchone()
            rev_no = row["m"] + 1
            self._insert_revision(c, plan_id, rev_no, note, payload_json,
                                  changes_json)
            return rev_no

    def get_revision(self, plan_id: str, rev_no: int) -> sqlite3.Row | None:
        with self._conn() as c:
            return c.execute(
                "SELECT * FROM revisions WHERE plan_id=? AND rev_no=?",
                (plan_id, rev_no)).fetchone()

    def get_latest_revision(self, plan_id: str) -> sqlite3.Row | None:
        with self._conn() as c:
            return c.execute(
                "SELECT * FROM revisions WHERE plan_id=?"
                " ORDER BY rev_no DESC LIMIT 1", (plan_id,)).fetchone()

    def list_revisions(self, plan_id: str) -> list[sqlite3.Row]:
        with self._conn() as c:
            return c.execute(
                "SELECT rev_no, status, note, created_at FROM revisions"
                " WHERE plan_id=? ORDER BY rev_no", (plan_id,)).fetchall()

    def update_draft_payload(self, plan_id: str, rev_no: int,
                             payload_json: str,
                             changes_json: str | None = None) -> bool:
        """仅草稿可改；确认稿返回 False（不可覆盖）。"""
        with self._conn() as c:
            if changes_json is None:
                cur = c.execute(
                    "UPDATE revisions SET payload=? WHERE plan_id=? AND rev_no=?"
                    " AND status='draft'",
                    (payload_json, plan_id, rev_no))
            else:
                cur = c.execute(
                    "UPDATE revisions SET payload=?, changes=?"
                    " WHERE plan_id=? AND rev_no=? AND status='draft'",
                    (payload_json, changes_json, plan_id, rev_no))
            return cur.rowcount == 1

    def confirm_revision(self, plan_id: str, rev_no: int) -> bool:
        with self._conn() as c:
            cur = c.execute(
                "UPDATE revisions SET status='confirmed' WHERE plan_id=?"
                " AND rev_no=? AND status='draft'", (plan_id, rev_no))
            return cur.rowcount == 1

    # ------------------------------------------------------------- 救援方案
    def create_rescue_plan(self, plan_id: str, source_rev_no: int,
                           name: str, note: str, payload_json: str,
                           changes_json: str = "[]") -> tuple[str, int]:
        rescue_id = uuid.uuid4().hex[:12]
        with self._conn() as c:
            c.execute(
                "INSERT INTO rescue_plans(id, plan_id, source_rev_no, name,"
                " created_at) VALUES (?,?,?,?,?)",
                (rescue_id, plan_id, source_rev_no, name, _now()))
            c.execute(
                "INSERT INTO rescue_revisions(rescue_id, rev_no, status,"
                " note, payload, changes, created_at) VALUES "
                "(?,1,'draft',?,?,?,?)",
                (rescue_id, note, payload_json, changes_json, _now()))
        return rescue_id, 1

    def get_rescue_plan(self, rescue_id: str) -> sqlite3.Row | None:
        with self._conn() as c:
            return c.execute("SELECT * FROM rescue_plans WHERE id=?",
                             (rescue_id,)).fetchone()

    def list_rescue_plans(self, plan_id: str | None = None
                          ) -> list[sqlite3.Row]:
        with self._conn() as c:
            if plan_id is None:
                return c.execute(
                    "SELECT * FROM rescue_plans ORDER BY created_at").fetchall()
            return c.execute(
                "SELECT * FROM rescue_plans WHERE plan_id=? ORDER BY created_at",
                (plan_id,)).fetchall()

    def add_rescue_revision(self, rescue_id: str, note: str,
                            payload_json: str,
                            changes_json: str = "[]") -> int:
        with self._conn() as c:
            row = c.execute(
                "SELECT COALESCE(MAX(rev_no), 0) AS m FROM rescue_revisions"
                " WHERE rescue_id=?", (rescue_id,)).fetchone()
            rev_no = row["m"] + 1
            c.execute(
                "INSERT INTO rescue_revisions(rescue_id, rev_no, status,"
                " note, payload, changes, created_at) VALUES "
                "(?,?,'draft',?,?,?,?)",
                (rescue_id, rev_no, note, payload_json, changes_json, _now()))
            return rev_no

    def get_rescue_revision(self, rescue_id: str, rev_no: int
                            ) -> sqlite3.Row | None:
        with self._conn() as c:
            return c.execute(
                "SELECT * FROM rescue_revisions WHERE rescue_id=? AND rev_no=?",
                (rescue_id, rev_no)).fetchone()

    def list_rescue_revisions(self, rescue_id: str) -> list[sqlite3.Row]:
        with self._conn() as c:
            return c.execute(
                "SELECT rev_no, status, note, created_at FROM rescue_revisions"
                " WHERE rescue_id=? ORDER BY rev_no", (rescue_id,)).fetchall()

    def update_draft_rescue(self, rescue_id: str, rev_no: int,
                            payload_json: str,
                            changes_json: str | None = None,
                            note: str | None = None) -> bool:
        """仅草稿可改；确认稿返回 False。"""
        with self._conn() as c:
            sets = ["payload=?"]
            args: list = [payload_json]
            if changes_json is not None:
                sets.append("changes=?")
                args.append(changes_json)
            if note is not None:
                sets.append("note=?")
                args.append(note)
            args += [rescue_id, rev_no]
            cur = c.execute(
                f"UPDATE rescue_revisions SET {', '.join(sets)}"
                f" WHERE rescue_id=? AND rev_no=? AND status='draft'",
                tuple(args))
            return cur.rowcount == 1

    def confirm_rescue_revision(self, rescue_id: str, rev_no: int,
                                source_snapshot_json: str,
                                source_sig: str) -> bool:
        """确认救援修订：冻结状态并写入来源修订快照与相关性签名。"""
        with self._conn() as c:
            cur = c.execute(
                "UPDATE rescue_revisions SET status='confirmed',"
                " source_snapshot=?, source_sig=? WHERE rescue_id=?"
                " AND rev_no=? AND status='draft'",
                (source_snapshot_json, source_sig, rescue_id, rev_no))
            return cur.rowcount == 1

    def list_confirmed_rescue_revisions(self) -> list[sqlite3.Row]:
        with self._conn() as c:
            return c.execute(
                "SELECT * FROM rescue_revisions WHERE status='confirmed'"
            ).fetchall()

    def set_rescue_recheck(self, rescue_id: str, rev_no: int,
                           flag: bool) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE rescue_revisions SET recheck_flag=? WHERE rescue_id=?"
                " AND rev_no=?", (1 if flag else 0, rescue_id, rev_no))
