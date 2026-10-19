"""Delayed options-flow views: unusual chain activity + Alpaca indicative prints.

Both are pure functions over data the caller already has, so they are testable
without network. Neither can tell buyer-initiated from seller-initiated trades
(no real NBBO on free feeds), so nothing here labels flow bullish/bearish.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

CONTRACT_MULTIPLIER = 100


def occ_symbol(ticker: str, expiration: str, cp: str, strike: float) -> str:
    """OCC contract symbol as Alpaca spells it, e.g. AAPL240621C00190000."""
    yymmdd = datetime.strptime(expiration, "%Y-%m-%d").strftime("%y%m%d")
    return f"{ticker.upper()}{yymmdd}{cp}{round(strike * 1000):08d}"


def unusual_activity(chain_df: pd.DataFrame, ticker: str, top: int = 15,
                     min_volume: int = 100, min_premium: float = 25_000) -> list[dict]:
    """Contracts trading more than their open interest, ranked by premium.

    Volume > OI means the day's trading is mostly opening new positions.
    Premium is volume x last x 100 (last, not mid: it is what actually traded).
    """
    if chain_df is None or chain_df.empty:
        return []
    df = chain_df[chain_df["last_price"] > 0].copy()
    df["premium"] = df["volume"] * df["last_price"] * CONTRACT_MULTIPLIER
    df = df[(df["volume"] >= min_volume) & (df["premium"] >= min_premium)
            & (df["volume"] > df["open_interest"])]
    rows = []
    for r in df.sort_values("premium", ascending=False).head(top).itertuples():
        rows.append({
            "symbol": occ_symbol(ticker, r.expiration, r.cp, r.strike),
            "expiration": r.expiration, "dte": int(r.dte), "cp": r.cp,
            "strike": float(r.strike), "last": float(r.last_price),
            "volume": int(r.volume), "open_interest": int(r.open_interest),
            "vol_oi": round(r.volume / r.open_interest, 1) if r.open_interest else None,
            "premium": float(r.premium),
        })
    return rows


def largest_prints(trades_by_symbol: dict, top: int = 20) -> list[dict]:
    """Flatten Alpaca's {symbol: [trade]} into the biggest prints by premium."""
    rows = []
    for symbol, trades in (trades_by_symbol or {}).items():
        for t in trades:
            price, size = t.get("p"), t.get("s")
            if not price or not size:
                continue
            rows.append({"symbol": symbol, "time": t.get("t"), "price": price,
                         "size": size, "exchange": t.get("x"),
                         "premium": price * size * CONTRACT_MULTIPLIER})
    return sorted(rows, key=lambda r: r["premium"], reverse=True)[:top]
