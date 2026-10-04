"""Earnings Event Study: history fetch, post-print CAR/drift, upcoming-print block.

All fetchers return None on failure (provider contract). app.py imports this
module at load, so `app` itself is imported lazily inside functions only.
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf

import db
import providers

EARNINGS_TTL_HOURS = 24


def fetch_earnings_history(ticker: str) -> list[dict] | None:
    """Normalized earnings prints (past + upcoming) from yfinance .earnings_dates."""
    try:
        frame = yf.Ticker(ticker.upper()).earnings_dates
    except Exception as e:
        print(f"[earnings] fetch failed for {ticker}: {e}")
        return None
    if frame is None or getattr(frame, "empty", True):
        return None
    rows: list[dict] = []
    for idx, row in frame.iterrows():
        try:
            dt = pd.to_datetime(idx)
            if pd.isna(dt):
                continue
            date_str = dt.date().isoformat()
        except Exception:
            continue  # yfinance occasionally returns junk index rows
        est = row.get("EPS Estimate")
        act = row.get("Reported EPS")
        surp = row.get("Surprise(%)")
        rows.append({
            "date": date_str,
            "eps_estimate": float(est) if est is not None and pd.notna(est) else None,
            "eps_actual": float(act) if act is not None and pd.notna(act) else None,
            "surprise_pct": float(surp) if surp is not None and pd.notna(surp) else None,
            "is_upcoming": act is None or pd.isna(act),
        })
    rows.sort(key=lambda r: r["date"])
    return rows or None


def get_earnings_history(ticker: str) -> list[dict] | None:
    """Cached earnings history (24h TTL, provider "yfinance")."""
    return providers._cached(
        "yfinance", f"earnings:{ticker.upper()}",
        lambda: fetch_earnings_history(ticker),
        ttl_hours=EARNINGS_TTL_HOURS,
    )
