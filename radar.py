"""Radar: universe scanners over warmed caches (drift / value / squeeze).

Scan builders (Task 14) read ONLY api_cache + SQLite — never fetch. Tunables
are persisted display thresholds edited inline on /radar; they are not secrets,
so /api/radar/settings follows the /api/active-tickers auth posture:
Bearer-checked only when WARM_CACHE_TOKEN is configured.
"""
from __future__ import annotations

from datetime import date  # noqa: E402

import math

import pandas as pd  # noqa: E402

import db
import event_study  # noqa: E402
import microstructure  # noqa: E402
import momentum_engine  # noqa: E402
import peers  # noqa: E402

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


DRIFT_LOOKBACK_DAYS = 45
DRIFT_HORIZON_DAYS = 30
SCAN_PAYLOAD_TTL_HOURS = 24


def universe_symbols(min_bars: int = 253) -> list[str]:
    """Symbols with enough cached price history to score."""
    try:
        with db.get_conn() as conn:
            rows = conn.execute(
                "SELECT symbol FROM daily_prices GROUP BY symbol HAVING COUNT(*) >= ?",
                (min_bars,),
            ).fetchall()
        return sorted(r["symbol"] for r in rows)
    except Exception:
        return []


def _spy_df():
    return db.get_prices("SPY")


def build_drift_scan(lookback_days: int = DRIFT_LOOKBACK_DAYS,
                     drift_horizon: int = DRIFT_HORIZON_DAYS) -> dict:
    """Recent prints whose post-earnings drift may still be running.

    Reads only cached earnings payloads + SQLite prices. Rows carry the
    event-study CAR when the windows allow, and remaining = max(0, horizon −
    days since print).
    """
    payloads = db.cache_scan("yfinance", "earnings:")
    spy_df = _spy_df()
    today = date.today()
    rows: list[dict] = []
    priced = {}
    for symbol, events in payloads.items():
        past = [e for e in events or [] if not e.get("is_upcoming")]
        if not past:
            continue
        last = max(past, key=lambda e: e["date"])
        try:
            days_since = (today - pd.to_datetime(last["date"]).date()).days
        except Exception:
            continue
        if not 0 <= days_since <= lookback_days:
            continue
        if symbol not in priced:
            priced[symbol] = db.get_prices(symbol)
        stock_df = priced[symbol]
        if stock_df is None or getattr(stock_df, "empty", True):
            continue
        res = (event_study.run_event_study(stock_df, spy_df if spy_df is not None
                                           else stock_df, last["date"],
                                           event_type="EARNINGS", ticker=symbol)
               if spy_df is not None else None)
        rows.append({
            "symbol": symbol,
            "date": last["date"],
            "days_since": days_since,
            "remaining": max(0, drift_horizon - days_since),
            "surprise_pct": last.get("surprise_pct"),
            "car": round(res.car, 4) if res else None,
            "is_significant_95": bool(res.is_significant_95) if res else False,
        })
    rows.sort(key=lambda r: (not r["is_significant_95"], -r["remaining"]))
    return {"rows": rows, "coverage": len(payloads)}


def build_value_scan() -> dict:
    """Cheapest quartile per industry (fallback sector) with positive momentum.

    Value score and percentiles reuse peers.py math over warmed fundamentals
    payloads; momentum from SQLite closes.
    """
    classifications = peers.universe_classifications()
    scored: list[dict] = []
    for symbol, cls in classifications.items():
        payload = db.cache_get("yfinance", f"fundamentals:{symbol}",
                               SCAN_PAYLOAD_TTL_HOURS)
        if not payload:
            continue
        px = db.get_prices(symbol)
        mom = None
        if px is not None and not px.empty:
            score = momentum_engine.score_series(px["close"], symbol=symbol)
            mom = round(score.mom_12_1, 4) if score else None
        percs = {
            "forward_pe": payload.get("Forward P/E"),
            "ps": payload.get("Price / Sales"),
            "peg": payload.get("PEG Ratio"),
        }
        scored.append({
            "symbol": symbol,
            "industry": cls.get("industry") or cls.get("sector") or "Unknown",
            "mom_12_1": mom,
            "forward_pe": payload.get("Forward P/E"),
            "_raw": {k: v for k, v in percs.items()},
        })
    rows: list[dict] = []
    by_group: dict[str, list[dict]] = {}
    for row in scored:
        by_group.setdefault(row["industry"], []).append(row)
    for group, members in by_group.items():
        if len(members) < 2:
            continue
        ranks: dict[str, list] = {k: [m["_raw"][k] for m in members]
                                  for k in members[0]["_raw"]}
        for m in members:
            cheap = [peers.percentile_rank(ranks[k], m["_raw"][k], invert=True)
                     for k in m["_raw"]]
            m["value_score"] = peers.value_score(dict(
                zip(m["_raw"].keys(), cheap)))
        eligible = [m for m in members
                    if m.get("value_score") is not None and (m.get("mom_12_1") or 0) > 0]
        if not eligible:
            continue
        scores = sorted(m["value_score"] for m in eligible)
        # top quartile of eligible scores; small groups collapse to the best
        cutoff = scores[max(0, int(len(scores) * 0.75) - 1)]
        for m in eligible:
            if m["value_score"] >= cutoff:
                rows.append({k: m[k] for k in
                             ("symbol", "industry", "value_score", "mom_12_1", "forward_pe")})
    rows.sort(key=lambda r: -(r["value_score"] or 0))
    rows = rows[:50]
    return {"rows": rows, "coverage": len(scored)}


def build_squeeze_scan() -> dict:
    """Universe squeeze screen: real SI composite x momentum, above the
    liquidity floor. is_negative_gex stays False (no chain data here)."""
    floor = float(get_tunable("radar_liquidity_floor_usd"))
    payloads = db.cache_scan("yfinance", "short:")
    rows: list[dict] = []
    for symbol, si in payloads.items():
        if not si or si.get("short_pct_float") is None or si.get("days_to_cover") is None:
            continue
        px = db.get_prices(symbol)
        if px is None or getattr(px, "empty", True):
            continue
        try:
            recent = px.tail(20)
            dollar_volume = float((recent["close"] * recent["volume"]).mean())
        except Exception:
            continue
        if dollar_volume < floor:
            continue
        score, level = microstructure.compute_squeeze_risk_index(
            si["short_pct_float"] * 100.0, si["days_to_cover"])
        mom = None
        try:
            ms = momentum_engine.score_series(px["close"], symbol=symbol)
            mom = round(ms.mom_12_1, 4) if ms else None
        except Exception:
            pass
        rows.append({
            "symbol": symbol,
            "short_pct_float": si["short_pct_float"],
            "days_to_cover": si["days_to_cover"],
            "si_mom_change": si.get("si_mom_change"),
            "squeeze_score": score,
            "squeeze_level": level,
            "mom_12_1": mom,
            "dollar_volume": round(dollar_volume),
        })
    rows.sort(key=lambda r: (-r["squeeze_score"], -(r["mom_12_1"] or 0)))
    return {"rows": rows, "coverage": len(payloads)}
