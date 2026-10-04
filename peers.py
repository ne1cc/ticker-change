"""Peer Comparison: peer-set resolution (Finnhub primary, warmed-universe fallback).

Resolution order:
  1. Finnhub /stock/peer (keyed, 24h cache).
  2. Same GICS industry among tickers whose fundamentals payload is already in
     api_cache (no fetch); sector from the cached S&P 500 constituents as the
     cohort when the focus ticker has no industry.

Metric tables are built by build_peer_comparison (Task 10) — this module never
blocks on network beyond the single Finnhub call.
"""
from __future__ import annotations

import db
import providers

import pandas as pd  # noqa: E402

import momentum_engine  # noqa: E402

FUNDAMENTALS_TTL_HOURS = 24
CONSTITUENTS_TTL_HOURS = 24

# metric key -> (fundamentals key, invert). invert=True: higher raw = pricier.
_VALUE_METRICS = {
    "forward_pe": ("Forward P/E", True),
    "trailing_pe": ("Trailing P/E", True),
    "peg": ("PEG Ratio", True),
    "ps": ("Price / Sales", True),
    "ev_ebitda": ("EV / EBITDA", True),
}
_GROWTH_METRICS = {
    "revenue_growth": ("Revenue Growth", False),
    "earnings_growth": ("Earnings Growth", False),
}


def universe_classifications() -> dict[str, dict]:
    """{symbol: {"industry": …, "sector": …}} from every cached fundamentals
    payload, sector falling back to the cached constituents payload."""
    out: dict[str, dict] = {}
    for sym, payload in db.cache_scan("yfinance", "fundamentals:").items():
        if not isinstance(payload, dict):
            continue
        out[sym.upper()] = {
            "industry": payload.get("Industry"),
            "sector": payload.get("Sector"),
        }
    for row in db.cache_get("sp500", "constituents", CONSTITUENTS_TTL_HOURS) or []:
        if not isinstance(row, dict):
            continue
        sym = str(row.get("symbol", "")).upper()
        if sym and sym not in out:
            out[sym] = {"industry": None, "sector": row.get("sector")}
    return out


def resolve_peers(ticker: str, cap: int = 8) -> list[str] | None:
    """Peer symbols for ticker, focus excluded, capped. None when unresolvable."""
    ticker = ticker.upper()
    finnhub_peers = providers.finnhub_peer(ticker)
    if finnhub_peers:
        peers = [p for p in finnhub_peers if p != ticker]
        return peers[:cap] or None

    classifications = universe_classifications()
    focus = classifications.get(ticker) or {}
    industry, sector = focus.get("industry"), focus.get("sector")
    if not industry and not sector:
        return None
    candidates = []
    for sym, cls in classifications.items():
        if sym == ticker:
            continue
        if industry and cls.get("industry") == industry:
            candidates.append(sym)
        elif not industry and sector and cls.get("sector") == sector:
            candidates.append(sym)
    candidates.sort()
    return candidates[:cap] or None


def percentile_rank(values: list, x, invert: bool = False) -> float | None:
    """Percentile of x within values (0-100). invert=True scores 'higher raw is
    worse' (multiples) so cheaper names read higher. None with <2 comparable values."""
    vals = [float(v) for v in values if v is not None]
    if x is None or len(vals) < 2:
        return None
    below = sum(1 for v in vals if v < float(x))
    pct = 100.0 * below / (len(vals) - 1)
    return round(100.0 - pct, 1) if invert else round(pct, 1)


def value_score(percs: dict) -> float | None:
    """Mean percentile across available metrics, ignoring None."""
    vals = [v for v in percs.values() if v is not None]
    return round(sum(vals) / len(vals), 1) if vals else None


def quadrant(valuation_pct: float | None, mom_12_1: float | None) -> str | None:
    """Valuation (cheap vs rich at the 50th pct) x momentum (positive vs not)."""
    if valuation_pct is None or mom_12_1 is None:
        return None
    cheap = valuation_pct >= 50.0
    strong = mom_12_1 > 0.0
    if cheap and strong:
        return "Cheap & Strengthening"
    if cheap:
        return "Cheap & Weakening"
    return "Rich & Strengthening" if strong else "Rich & Weakening"


def _row_for(sym: str) -> dict | None:
    payload = db.cache_get("yfinance", f"fundamentals:{sym}", FUNDAMENTALS_TTL_HOURS)
    if not payload:
        return None
    row = {"symbol": sym}
    for key, (src, _inv) in {**_VALUE_METRICS, **_GROWTH_METRICS}.items():
        row[key] = payload.get(src)
    si = db.cache_get("yfinance", f"short:{sym}", FUNDAMENTALS_TTL_HOURS)
    row["short_pct_float"] = (si or {}).get("short_pct_float")
    try:
        px = db.get_prices(sym)
        if px is not None and not px.empty:
            score = momentum_engine.score_series(px["close"], symbol=sym)
            row["mom_12_1"] = round(score.mom_12_1, 4) if score else None
        else:
            row["mom_12_1"] = None
    except Exception:
        row["mom_12_1"] = None
    return row


def build_peer_comparison(ticker: str, cap: int | None = None) -> dict | None:
    """Cross-sectional peer table with percentile ranks + quadrants.

    Reads only api_cache/SQLite — never fetches. None without a peer set or
    focus fundamentals.
    """
    ticker = ticker.upper()
    peers_syms = resolve_peers(ticker, cap=cap if cap is not None else 8)
    if not peers_syms:
        return None
    focus = _row_for(ticker)
    if not focus:
        return None
    rows = [focus] + [r for r in (_row_for(s) for s in peers_syms) if r]
    if len(rows) < 2:
        return None

    for key, (_src, invert) in {**_VALUE_METRICS, **_GROWTH_METRICS}.items():
        col = [r.get(key) for r in rows]
        for r in rows:
            r[f"{key}_pct"] = percentile_rank(col, r.get(key), invert=invert)
    mom_col = [r.get("mom_12_1") for r in rows]
    for r in rows:
        r["mom_pct"] = percentile_rank(mom_col, r.get("mom_12_1"))

    cheap_keys = [f"{k}_pct" for k in _VALUE_METRICS]
    growth_keys = [f"{k}_pct" for k in _GROWTH_METRICS]
    for r in rows:
        r["valuation_pct"] = value_score({k: r.get(k) for k in cheap_keys})
        r["growth_pct"] = value_score({k: r.get(k) for k in growth_keys})
        r["value_score"] = value_score({
            "cheap": r["valuation_pct"],
            "growth": r["growth_pct"],
            "momentum": r.get("mom_pct"),
        })
        r["quadrant"] = quadrant(r["valuation_pct"], r.get("mom_12_1"))

    return {
        "focus": focus,
        "peers": [r for r in rows if r["symbol"] != ticker],
        "n_compared": len(rows),
        "metrics": [{"key": k, "label": src, "invert": inv}
                    for k, (src, inv) in {**_VALUE_METRICS, **_GROWTH_METRICS}.items()],
    }
