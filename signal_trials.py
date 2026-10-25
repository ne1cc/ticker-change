"""Persisted trial count ledger for multiple testing and Deflated Sharpe Ratio.

Part of Epic 5 (issue #135):
- Records every signal, variation, and parameter trial evaluated in Signal Lab
- Guarantees honest multiple-testing adjustment in Deflated Sharpe Ratio
- Persisted in SQLite; surviving restarts and multi-worker execution
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import sqlite3
import db


def init_trials_db():
    with db.get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signal_trials (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_name TEXT NOT NULL,
                variant TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL
            );
        """)


def record_signal_trial(signal_name: str, variant: str = "", metadata: dict | None = None) -> int:
    """Record an evaluation trial and return the cumulative total trials count."""
    init_trials_db()
    meta_str = json.dumps(metadata) if metadata else ""
    created = datetime.now(timezone.utc).isoformat()
    with db.get_conn() as conn:
        conn.execute(
            "INSERT INTO signal_trials (signal_name, variant, metadata, created_at) "
            "VALUES (?, ?, ?, ?)",
            (signal_name, variant, meta_str, created),
        )
        row = conn.execute("SELECT COUNT(*) as cnt FROM signal_trials").fetchone()
        return int(row["cnt"]) if row is not None else 1


def get_total_trial_count() -> int:
    """Return the current total trial count, defaulting to at least 1."""
    init_trials_db()
    with db.get_conn() as conn:
        row = conn.execute("SELECT COUNT(*) as cnt FROM signal_trials").fetchone()
        count = int(row["cnt"]) if row is not None else 0
        return max(1, count)


def get_trials_ledger(limit: int = 50) -> list[dict]:
    """Return recent trial ledger entries."""
    init_trials_db()
    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT id, signal_name, variant, metadata, created_at "
            "FROM signal_trials ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    results = []
    for r in rows:
        results.append({
            "id": r["id"],
            "signal_name": r["signal_name"],
            "variant": r["variant"],
            "metadata": json.loads(r["metadata"]) if r["metadata"] else {},
            "created_at": r["created_at"],
        })
    return results
