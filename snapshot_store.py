"""Shared snapshot-publication store: payload, pointer (last), retention.

A publisher stores its validated payload under `snapshot:<id>`, then moves
its 'current' pointer, then prunes to the newest two payloads — pointer
write last, so a failed publish leaves the last-good snapshot serving.
Publisher-generic: payload shape and provider name belong to the caller;
only publication itself lives here.
"""
from __future__ import annotations

import itertools
import json
import sqlite3

from datetime import datetime, timedelta

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


def current_all(provider_prefix: str) -> dict[str, dict]:
    """Scan twin of current(): every live published payload whose provider
    starts with the prefix, keyed by the provider suffix. Same TTL rule as
    current(); one query for pointers and payloads, joined in memory."""
    try:
        with db.get_conn() as conn:
            rows = conn.execute(
                "SELECT provider, key, payload, fetched_at FROM api_cache "
                "WHERE provider LIKE ? AND (key = 'current' OR key LIKE 'snapshot:%')",
                (provider_prefix + "%",)).fetchall()
    except sqlite3.OperationalError:
        return {}
    cutoff = datetime.utcnow() - timedelta(hours=_TTL_HOURS)
    pointers: dict[str, str] = {}
    payloads: dict[tuple[str, str], dict] = {}
    for row in rows:
        try:
            if datetime.fromisoformat(row["fetched_at"]) < cutoff:
                continue
            payload = json.loads(row["payload"])
        except (ValueError, TypeError):
            continue
        if row["key"] == "current":
            if isinstance(payload, dict) and payload.get("snapshot_id"):
                pointers[row["provider"]] = payload["snapshot_id"]
        elif isinstance(payload, dict):
            payloads[(row["provider"], row["key"])] = payload
    out: dict[str, dict] = {}
    for provider, snapshot_id in pointers.items():
        payload = payloads.get((provider, f"snapshot:{snapshot_id}"))
        if payload is not None:
            out[provider[len(provider_prefix):]] = payload
    return out


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
