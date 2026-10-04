"""Earnings Event Study: history fetch, post-print CAR/drift, upcoming-print block.

All fetchers return None on failure (provider contract). app.py imports this
module at load, so `app` itself is imported lazily inside functions only.
"""
from __future__ import annotations

from datetime import date

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


import event_study  # noqa: E402


def post_drift(stock_df: pd.DataFrame, date_str: str, n_days: int) -> float | None:
    """Raw close-to-close return from the first session on/after date_str to
    the first session on/after date_str + n calendar days. None when either
    endpoint is missing (recent print, thin history)."""
    try:
        if stock_df is None or getattr(stock_df, "empty", True):
            return None
        close = stock_df["close"] if "close" in stock_df.columns else stock_df["Close"]
        idx = close.index
        ts = pd.to_datetime(date_str)
        at_or_after = idx[idx >= ts]
        if len(at_or_after) == 0:
            return None
        start = at_or_after[0]
        later = idx[idx >= start + pd.Timedelta(days=n_days)]
        if len(later) == 0:
            return None
        p0, p1 = float(close.loc[start]), float(close.loc[later[0]])
        if not p0:
            return None
        return round((p1 - p0) / p0, 4)
    except Exception:
        return None


def compute_earnings_events(stock_df, benchmark_df, events, ticker="UNKNOWN",
                            max_events=8) -> list[dict]:
    """Market-model CAR (existing event_study engine) for past earnings prints,
    plus raw 5/10/20-calendar-day post-drift. Upcoming prints are skipped."""
    out: list[dict] = []
    for ev in events or []:
        if ev.get("is_upcoming"):
            continue
        res = event_study.run_event_study(
            stock_df, benchmark_df, ev["date"],
            event_type="EARNINGS", ticker=ticker,
        )
        row = dict(ev)
        row.update({
            "car": res.car if res else None,
            "car_t_stat": res.car_t_stat if res else None,
            "car_p_value": res.car_p_value if res else None,
            "is_significant_95": bool(res.is_significant_95) if res else False,
            "drift_5d": post_drift(stock_df, ev["date"], 5),
            "drift_10d": post_drift(stock_df, ev["date"], 10),
            "drift_20d": post_drift(stock_df, ev["date"], 20),
        })
        out.append(row)
        if len(out) >= max_events:
            break
    out.sort(key=lambda r: r["date"], reverse=True)
    return out


def summarize_earnings_drift(events: list[dict]) -> dict:
    """Aggregate drift stats: beat-vs-miss average CAR and significance rate."""
    cars = [(e.get("surprise_pct"), e["car"])
            for e in events or [] if e.get("car") is not None]

    def _avg(pairs, beat: bool):
        vals = ([c for s, c in pairs if (s or 0) > 0] if beat
                else [c for s, c in pairs if (s or 0) < 0])
        return round(sum(vals) / len(vals), 4) if vals else None

    sig = [e for e in events or [] if e.get("is_significant_95")]
    return {
        "n_events": len(events or []),
        "n_significant": len(sig),
        "significance_rate": round(len(sig) / len(events), 3) if events else None,
        "avg_car_beat": _avg(cars, beat=True),
        "avg_car_miss": _avg(cars, beat=False),
    }


def realized_earnings_moves(stock_df: pd.DataFrame, events: list[dict],
                            n: int = 8) -> float | None:
    """Median absolute close-to-close return across the most recent n print
    days (the print session vs the prior close). Baseline for the event-vol flag."""
    try:
        if stock_df is None or getattr(stock_df, "empty", True):
            return None
        close = stock_df["close"] if "close" in stock_df.columns else stock_df["Close"]
        rets = close.pct_change()
        moves = []
        for ev in (events or [])[:n]:
            if ev.get("is_upcoming"):
                continue
            idx = rets.index[rets.index > pd.to_datetime(ev["date"])]
            if len(idx) == 0:
                continue
            r = rets.loc[idx[0]]
            if r is not None and pd.notna(r):
                moves.append(abs(float(r)))
        return round(float(pd.Series(moves).median()), 4) if moves else None
    except Exception:
        return None


def implied_event_move(options_analysis: dict | None, earnings_date: str) -> dict | None:
    """Scale the front-expiration straddle expected move out to the earnings
    date: EM_earnings ~= EM_straddle * sqrt(DTE_earnings / DTE_expiry)."""
    try:
        if not options_analysis:
            return None
        em = options_analysis.get("expected_move_straddle")
        dte_exp = options_analysis.get("days_to_expiration")
        dte_earn = (pd.to_datetime(earnings_date).date() - date.today()).days
        if not em or not dte_exp or dte_exp <= 0 or dte_earn is None or dte_earn < 0:
            return None
        return {
            "implied_move": round(float(em) * (dte_earn / dte_exp) ** 0.5, 2),
            "days_to_earnings": dte_earn,
            "basis_days_to_exp": dte_exp,
        }
    except Exception:
        return None


def event_vol_flag(implied_move: float | None, realized_move: float | None) -> str | None:
    """Rich (>1.25x realized), cheap (<0.75x), fair — None without both sides."""
    if not implied_move or not realized_move:
        return None
    ratio = implied_move / realized_move
    if ratio > 1.25:
        return "rich"
    if ratio < 0.75:
        return "cheap"
    return "fair"


def build_earnings_section(ticker: str) -> dict | None:
    """Everything the /analytics embed renders. None when no earnings data."""
    from app import compute_options_analysis, get_or_fetch_prices  # late import: app imports earnings at module load

    ticker = ticker.upper()
    events = get_earnings_history(ticker)
    if not events:
        return None
    stock_df = get_or_fetch_prices(ticker, period="5y")
    if stock_df is None or getattr(stock_df, "empty", True):
        return None
    benchmark_df = get_or_fetch_prices("SPY", period="5y")
    if benchmark_df is None or getattr(benchmark_df, "empty", True):
        benchmark_df = stock_df

    past = compute_earnings_events(stock_df, benchmark_df, events, ticker=ticker)
    upcoming = [e for e in events if e.get("is_upcoming")]
    next_print = upcoming[0] if upcoming else None
    implied = implied_event_move(
        compute_options_analysis(ticker),
        next_print["date"],
    ) if next_print else None
    realized = realized_earnings_moves(stock_df, events)
    return {
        "events": past,
        "summary": summarize_earnings_drift(past),
        "next_print": next_print,
        "implied_move": implied,
        "realized_move": realized,
        "event_vol": event_vol_flag(
            implied["implied_move"] if implied else None, realized),
    }
