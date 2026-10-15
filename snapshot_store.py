"""Shared snapshot-publication store: payload, pointer (last), retention.

A publisher stores its validated payload under `snapshot:<id>`, then moves
its 'current' pointer, then prunes to the newest two payloads — pointer
write last, so a failed publish leaves the last-good snapshot serving.
Publisher-generic: payload shape and provider name belong to the caller;
only publication itself lives here.
"""
from __future__ import annotations

import itertools

from datetime import datetime

import db

_TTL_HOURS = 24 * 7
_RETENTION = 2
_SEQ = itertools.count()


def new_snapshot_id() -> str:
    """UTC stamp to the second plus a per-session counter: chronologically
    sortable and collision-free within a session, same second included."""
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{next(_SEQ):04d}"


def publish(provider: str, snapshot_id: str, payload) -> None:
    """Store the payload, move the pointer (last), prune. Call only after
    build/validation succeed, so the pointer never leads the payload."""
    db.cache_set(provider, f"snapshot:{snapshot_id}", payload)
    db.cache_set(provider, "current", {"snapshot_id": snapshot_id})
    _prune(provider)


def current(provider: str):
    """The published payload, or None when absent/invalid."""
    pointer = db.cache_get(provider, "current", _TTL_HOURS)
    if not isinstance(pointer, dict) or not pointer.get("snapshot_id"):
        return None
    payload = db.cache_get(
        provider, f"snapshot:{pointer['snapshot_id']}", _TTL_HOURS)
    return payload if isinstance(payload, dict) else None


def _prune(provider: str, keep: int = _RETENTION):
    """Keep only the newest `keep` snapshot payloads; 'current' is a separate
    key and never matches the snapshot: prefix."""
    with db.get_conn() as conn:
        stored = conn.execute(
            "SELECT key FROM api_cache WHERE provider = ? AND key LIKE 'snapshot:%' "
            "ORDER BY fetched_at DESC, rowid DESC", (provider,)).fetchall()
        for row in stored[keep:]:
            conn.execute(
                "DELETE FROM api_cache WHERE provider = ? AND key = ?",
                (provider, row["key"]))
