"""Radar: universe scanners over warmed caches (drift / value / squeeze).

Scan builders (Task 14) read ONLY api_cache + SQLite — never fetch. Tunables
are persisted display thresholds edited inline on /radar; they are not secrets,
so /api/radar/settings follows the /api/active-tickers auth posture:
Bearer-checked only when WARM_CACHE_TOKEN is configured.
"""
from __future__ import annotations

import math

import db

TUNABLES: dict[str, dict] = {
    "radar_peer_cap": {"default": 8, "min": 3, "max": 20, "type": int},
    "radar_liquidity_floor_usd": {
        "default": 5_000_000.0, "min": 100_000.0,
        "max": 100_000_000.0, "type": float},
}


def get_tunable(key: str):
    """Persisted tunable value; default when unset or unparseable; clamped."""
    spec = TUNABLES.get(key)
    if spec is None:
        raise KeyError(key)
    try:
        val = spec["type"](db.get_setting(key, ""))
    except (TypeError, ValueError):
        return spec["default"]
    if isinstance(val, float) and not math.isfinite(val):
        return spec["default"]
    return spec["type"](min(max(val, spec["min"]), spec["max"]))


def clamp_tunable(key: str, value):
    """Coerce, reject non-finite, and clamp; no persistence."""
    spec = TUNABLES.get(key)
    if spec is None:
        raise ValueError(f"unknown tunable: {key}")
    val = spec["type"](value)
    if isinstance(val, float) and not math.isfinite(val):
        raise ValueError(f"non-finite value for {key}: {val}")
    return spec["type"](min(max(val, spec["min"]), spec["max"]))


def set_tunable(key: str, value):
    """Clamp, persist, and return the stored value. ValueError on unknown key."""
    val = clamp_tunable(key, value)
    db.set_setting(key, str(val))
    return val
