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
    """Deprecated: calendar-day drift, superseded by event_metrics (sessions,
    completed-window states). Kept for the touch-and-reversal engine's
    compatibility; removed in P2 if nothing calls it.

    Raw close-to-close return from the first session on/after date_str to
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


def reaction_session(stock_df: pd.DataFrame, date_str: str):
    """(t0, alignment): first trading session on/after the announcement date.

    yfinance publishes no before-open/after-close flag, so alignment is
    'unverified' in v1. (None, 'no-data') when the date precedes all history.
    """
    try:
        if stock_df is None or getattr(stock_df, "empty", True):
            return None, "no-data"
        idx = stock_df.index
        if len(idx) == 0:
            return None, "no-data"
        ts = pd.to_datetime(date_str)
        if ts < idx[0]:
            return None, "no-data"
        at_or_after = idx[idx >= ts]
        if len(at_or_after) == 0:
            return None, "no-data"
        return at_or_after[0], "unverified"
    except Exception:
        return None, "no-data"


def _close_at(stock_df, pos: int) -> float | None:
    close = stock_df["close"] if "close" in stock_df.columns else stock_df["Close"]
    if pos < 0 or pos >= len(close):
        return None
    return float(close.iloc[pos])


def event_metrics(stock_df: pd.DataFrame, date_str: str) -> dict:
    """Reaction-session metrics per the drift contract (sessions, not calendar
    days; drift excludes the reaction day; incomplete windows are None)."""
    out = {"t0": None, "alignment": "no-data", "sessions_since": None,
           "gap_pct": None, "reaction_ret": None, "runup_5d": None,
           "rel_volume": None, "drift_5d": None, "drift_10d": None,
           "drift_20d": None, "drift_to_date": None, "drift_to_date_sessions": None}
    try:
        t0, alignment = reaction_session(stock_df, date_str)
        if t0 is None:
            return out
        out["t0"], out["alignment"] = t0, alignment
        pos = stock_df.index.get_loc(t0)
        if isinstance(pos, slice):
            return out
        out["sessions_since"] = len(stock_df.index) - 1 - pos
        last = len(stock_df.index) - 1

        c = lambda p: _close_at(stock_df, p)
        if pos >= 1 and c(pos - 1):
            prev = c(pos - 1)
            out["reaction_ret"] = round(c(pos) / prev - 1, 6) if prev else None
            if "open" in stock_df.columns:
                op = float(stock_df["open"].iloc[pos])
                out["gap_pct"] = round(op / prev - 1, 6) if prev else None
        if pos >= 6 and c(pos - 1) and c(pos - 6):
            out["runup_5d"] = round(c(pos - 1) / c(pos - 6) - 1, 6)

        if "volume" in stock_df.columns and pos >= 1:
            vol = stock_df["volume"].iloc[max(0, pos - 20): pos]  # 20 sessions ending t0-1
            base = float(vol.mean())
            out["rel_volume"] = round(float(stock_df["volume"].iloc[pos]) / base, 2) if base else None

        for h in (5, 10, 20):
            if pos + h <= last:
                cc = c(pos + h)
                out[f"drift_{h}d"] = round(cc / c(pos) - 1, 6) if (cc and c(pos)) else None
        if out["sessions_since"] and out["sessions_since"] > 0 and c(last) and c(pos):
            out["drift_to_date"] = round(c(last) / c(pos) - 1, 6)
            out["drift_to_date_sessions"] = out["sessions_since"]
        return out
    except Exception as e:
        print(f"[earnings] event_metrics failed for {date_str}: {e}")
        return out


def own_event_history(stock_df: pd.DataFrame, events: list[dict],
                      horizon: int = 10) -> dict:
    """Beat/miss 10-session drift summaries from completed windows only.

    Future events and unfinished horizons are excluded. n < 3 per side sets
    the top-level state to 'limited history'.
    """
    import numpy as np
    out = {"beat": {"median": None, "mean": None, "positive": 0, "n": 0},
           "miss": {"median": None, "mean": None, "positive": 0, "n": 0},
           "state": "limited history", "horizon_sessions": horizon}
    try:
        beats, misses = [], []
        for ev in events or []:
            if ev.get("is_upcoming") or ev.get("surprise_pct") is None:
                continue
            m = event_metrics(stock_df, ev["date"])
            val = m.get(f"drift_{horizon}d")
            if val is None:
                continue  # unfinished window
            (beats if ev["surprise_pct"] > 0 else misses).append(val)

        def _fill(bucket, vals):
            bucket["n"] = len(vals)
            if vals:
                bucket["median"] = round(float(np.median(vals)), 4)
                bucket["mean"] = round(float(np.mean(vals)), 4)
                bucket["positive"] = sum(1 for v in vals if v > 0)

        _fill(out["beat"], beats)
        _fill(out["miss"], misses)
        if out["beat"]["n"] >= 3 and out["miss"]["n"] >= 3:
            out["state"] = "ok"
        return out
    except Exception as e:
        print(f"[earnings] own_event_history failed: {e}")
        return out


def compute_earnings_events(stock_df, benchmark_df, events, ticker="UNKNOWN",
                            max_events=8) -> list[dict]:
    """Market-model CAR (existing event_study engine) for past earnings prints,
    plus session-based reaction/drift metrics (event_metrics; completed windows
    only). Upcoming prints are skipped."""
    out: list[dict] = []
    for ev in events or []:
        if ev.get("is_upcoming"):
            continue
        res = event_study.run_event_study(
            stock_df, benchmark_df, ev["date"],
            event_type="EARNINGS", ticker=ticker,
        )
        metrics = event_metrics(stock_df, ev["date"])
        row = dict(ev)
        row.update({
            "sessions_since": metrics["sessions_since"],
            "alignment": metrics["alignment"],
            "gap_pct": metrics["gap_pct"],
            "reaction_ret": metrics["reaction_ret"],
            "runup_5d": metrics["runup_5d"],
            "rel_volume": metrics["rel_volume"],
            "drift_5d": metrics["drift_5d"],
            "drift_10d": metrics["drift_10d"],
            "drift_20d": metrics["drift_20d"],
            "car": res.car if res else None,
            "car_t_stat": res.car_t_stat if res else None,
            "car_p_value": res.car_p_value if res else None,
            "is_significant_95": bool(res.is_significant_95) if res else False,
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
        past = [e for e in (events or []) if not e.get("is_upcoming")]
        moves = []
        for ev in past[-n:]:
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
