"""Decide layer: pre-trade checklist.

Repackages existing app metrics into pass/warn/fail flags — no new indicators.
"""
from __future__ import annotations

from datetime import datetime

import db
import ml
import pandas as pd

MOM_MIN_BARS = 253
TRADE_TYPES = frozenset({"long_stock", "long_call", "short_put", "other"})


def _cached_universe_symbols(min_bars: int = MOM_MIN_BARS) -> list[str]:
    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT symbol FROM daily_prices GROUP BY symbol HAVING COUNT(*) >= ?",
            (min_bars,),
        ).fetchall()
    return [r["symbol"] for r in rows]


def compute_universe_momentum_ranks() -> tuple[dict[str, int], int]:
    """12-1 momentum rank for each cached symbol (1 = highest)."""
    scores: dict[str, float] = {}
    for symbol in _cached_universe_symbols():
        df = db.get_prices(symbol)
        if df is None or len(df) < MOM_MIN_BARS:
            continue
        p_past = float(df["close"].iloc[-253])
        p_recent = float(df["close"].iloc[-22])
        if p_past > 0:
            scores[symbol] = (p_recent - p_past) / p_past
    sorted_univ = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ranks = {s: idx + 1 for idx, (s, _) in enumerate(sorted_univ)}
    return ranks, len(sorted_univ)


def days_until_earnings(fundamentals: dict | None) -> int | None:
    if not fundamentals:
        return None
    raw = fundamentals.get("Next Earnings")
    if not raw:
        return None
    try:
        dt = pd.to_datetime(raw).date()
        return (dt - datetime.now().date()).days
    except Exception:
        return None


def _check_row(key: str, label: str, status: str, reason: str) -> dict:
    return {"key": key, "label": label, "status": status, "reason": reason}


def build_checklist(
    ticker: str,
    data: dict,
    trade_type: str = "long_stock",
) -> dict:
    """Pre-trade checklist from analytics page data + lightweight fetches."""
    from app import compute_options_analysis, get_fundamentals

    ticker = ticker.upper()
    checks: list[dict] = []
    ranks, universe_size = compute_universe_momentum_ranks()
    rank = ranks.get(ticker)
    fundamentals = data.get("fundamentals") or get_fundamentals(ticker)
    current = data.get("current_price")
    gex = data.get("gex") or {}
    df = db.get_prices(ticker)

    # Trend
    feat = None
    try:
        if df is not None and len(df) >= 220:
            row = ml.build_features(df).iloc[-1]
            if not row.isna().any():
                feat = row
    except Exception:
        feat = None

    if feat is not None:
        d20, d50 = float(feat["dist_sma20"]), float(feat["dist_sma50"])
        if d50 > 0:
            st, reason = "pass", f"Above 50d MA ({d50:+.1%})"
        elif d50 <= 0 and d20 > 0:
            st, reason = "warn", "Below 50d MA but above 20d — mixed trend"
        else:
            st, reason = "fail", f"Below 50d MA ({d50:+.1%})"
        checks.append(_check_row("trend", "Trend", st, reason))
    else:
        checks.append(_check_row("trend", "Trend", "skip", "Insufficient price history"))

    # Momentum rank
    if rank is not None and universe_size > 0:
        pct = rank / universe_size
        if pct <= 0.50:
            st, reason = "pass", f"Rank #{rank} of {universe_size} (top half)"
        elif pct >= 0.90:
            st, reason = "fail", f"Rank #{rank} of {universe_size} (bottom decile)"
        elif pct >= 0.75:
            st, reason = "warn", f"Rank #{rank} of {universe_size} (bottom quartile)"
        else:
            st, reason = "pass", f"Rank #{rank} of {universe_size}"
        checks.append(_check_row("momentum", "Momentum", st, reason))
    else:
        checks.append(_check_row("momentum", "Momentum", "skip", "Not in cached universe"))

    # IV / premium
    opts = None
    try:
        opts = compute_options_analysis(ticker)
    except Exception:
        pass
    iv_rank = opts.get("iv_rank") if opts else None
    if iv_rank is not None:
        if iv_rank <= 50:
            st, reason = "pass", f"IV rank {iv_rank:.0f} — moderate"
        elif iv_rank <= 70:
            st, reason = "warn", f"IV rank {iv_rank:.0f} — elevated premium"
        else:
            st, reason = "fail", f"IV rank {iv_rank:.0f} — expensive options"
        checks.append(_check_row("iv", "IV / premium", st, reason))
    else:
        checks.append(_check_row("iv", "IV / premium", "skip", "Options data unavailable"))

    # Earnings
    earn_days = days_until_earnings(fundamentals)
    if earn_days is not None:
        if earn_days > 5:
            st, reason = "pass", f"Earnings in {earn_days}d"
        elif earn_days >= 3:
            st, reason = "warn", f"Earnings in {earn_days}d — event risk"
        else:
            st, reason = "fail", f"Earnings in {earn_days}d — too close"
        checks.append(_check_row("earnings", "Earnings", st, reason))
    else:
        checks.append(_check_row("earnings", "Earnings", "skip", "No earnings date on file"))

    # GEX regime
    flip = gex.get("gamma_flip")
    if flip is not None and current:
        if current >= flip:
            st, reason = "pass", f"Spot ${current:.2f} ≥ γ-flip ${flip:.2f}"
        else:
            st, reason = "fail", f"Spot ${current:.2f} below γ-flip ${flip:.2f}"
        checks.append(_check_row("gex", "GEX regime", st, reason))
    else:
        checks.append(_check_row("gex", "GEX regime", "warn", "GEX levels unavailable"))

    # Extension
    if current and feat is not None:
        from app import get_price_ranges

        ranges = get_price_ranges(ticker, current_price=current)
        atr = ranges.get("atr_30d") if ranges else None
        sma20 = float(df["close"].tail(20).mean()) if df is not None and len(df) >= 20 else None
        if atr and sma20 and atr > 0:
            dist = abs(current - sma20)
            ratio = dist / atr
            if ratio <= 2.0:
                st, reason = "pass", f"{ratio:.1f}× ATR from 20d mean"
            elif ratio <= 2.5:
                st, reason = "warn", f"{ratio:.1f}× ATR from 20d — stretched"
            else:
                st, reason = "fail", f"{ratio:.1f}× ATR from 20d — extended"
            checks.append(_check_row("extension", "Extension", st, reason))
        else:
            checks.append(_check_row("extension", "Extension", "skip", "ATR unavailable"))
    else:
        checks.append(_check_row("extension", "Extension", "skip", "Insufficient data"))

    passed = sum(1 for c in checks if c["status"] == "pass")
    warned = sum(1 for c in checks if c["status"] == "warn")
    failed = sum(1 for c in checks if c["status"] == "fail")
    actionable = [c for c in checks if c["status"] in ("warn", "fail")]
    risk_bits = [c["label"].lower() for c in actionable[:3]]
    summary = f"{passed}/{len(checks)} pass"
    if risk_bits:
        summary += f" — main risks: {', '.join(risk_bits)}"
    elif passed == len([c for c in checks if c["status"] != "skip"]):
        summary += " — no major flags"

    trade_labels = {
        "long_stock": "Long stock swing",
        "long_call": "Long call",
        "short_put": "Short put / premium",
        "other": "Other",
    }

    return {
        "ticker": ticker,
        "trade_type": trade_type,
        "trade_label": trade_labels.get(trade_type, trade_type),
        "checks": checks,
        "passed": passed,
        "warned": warned,
        "failed": failed,
        "total": len(checks),
        "summary": summary,
    }
