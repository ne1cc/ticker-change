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

FUNDAMENTALS_TTL_HOURS = 24
CONSTITUENTS_TTL_HOURS = 24


def universe_classifications() -> dict[str, dict]:
    """{symbol: {"industry": …, "sector": …}} from every cached fundamentals
    payload, sector falling back to the cached constituents payload."""
    out: dict[str, dict] = {}
    for sym, payload in db.cache_scan("yfinance", "fundamentals:").items():
        out[sym.upper()] = {
            "industry": payload.get("Industry"),
            "sector": payload.get("Sector"),
        }
    for row in db.cache_get("sp500", "constituents", CONSTITUENTS_TTL_HOURS) or []:
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
