"""Options-Implied Earnings Move vs Realized Move Engine.

Part of Epic 4 (issue #128):
- Calculates ATM straddle implied move for expiration immediately following next earnings
- Compares implied move to mean and median realized absolute move on past prints
- Flags options pricing as 'rich', 'cheap', or 'fair' with explicit sample size n
- Declares explicit state when upcoming earnings or options chain is absent
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

import db
import earnings


def compute_atm_straddle_move(
    chain_df: pd.DataFrame,
    spot: float,
) -> Optional[Dict[str, Any]]:
    """Compute implied move % from the nearest ATM call and put mid prices.

    Implied Move % = ((Call_mid + Put_mid) / Spot) * 100.
    """
    if chain_df is None or chain_df.empty or spot <= 0:
        return None

    calls = chain_df[chain_df["cp"] == "C"]
    puts = chain_df[chain_df["cp"] == "P"]
    if calls.empty or puts.empty:
        return None

    # Find ATM strike
    all_strikes = np.unique(chain_df["strike"].values)
    atm_strike = float(all_strikes[np.argmin(np.abs(all_strikes - spot))])

    atm_call = calls[calls["strike"] == atm_strike]
    atm_put = puts[puts["strike"] == atm_strike]
    if atm_call.empty or atm_put.empty:
        return None

    c_row = atm_call.iloc[0]
    p_row = atm_put.iloc[0]

    # Mid price calculation with quote validity checks
    def get_mid(row):
        bid = float(row.get("bid", 0.0)) if pd.notna(row.get("bid")) else 0.0
        ask = float(row.get("ask", 0.0)) if pd.notna(row.get("ask")) else 0.0
        last = float(row.get("lastPrice", row.get("last_price", 0.0))) if pd.notna(row.get("lastPrice", row.get("last_price"))) else 0.0
        if bid > 0 and ask >= bid:
            return (bid + ask) / 2.0
        if last > 0:
            return last
        return 0.0

    c_mid = get_mid(c_row)
    p_mid = get_mid(p_row)

    if c_mid <= 0 or p_mid <= 0:
        return None

    straddle_price = c_mid + p_mid
    implied_move_pct = (straddle_price / spot) * 100.0

    return {
        "atm_strike": atm_strike,
        "call_mid": round(c_mid, 2),
        "put_mid": round(p_mid, 2),
        "straddle_price": round(straddle_price, 2),
        "implied_move_pct": round(implied_move_pct, 2),
    }


def compute_historical_earnings_moves(
    ticker: str,
    prices_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Calculate mean and median absolute realized 1-day moves across historical earnings prints."""
    past_prints = earnings.get_earnings_history(ticker) or []
    historical = [p for p in past_prints if not p.get("is_upcoming") and p.get("date")]

    if not historical:
        return {
            "n_prints": 0,
            "mean_move_pct": 0.0,
            "median_move_pct": 0.0,
            "moves": [],
        }

    if prices_df is None:
        prices_df = db.get_prices(ticker)

    if prices_df is None or prices_df.empty:
        return {
            "n_prints": 0,
            "mean_move_pct": 0.0,
            "median_move_pct": 0.0,
            "moves": [],
        }

    c_col = {c.lower(): c for c in prices_df.columns}.get("close", "Close")
    closes = prices_df[c_col].astype(float)
    daily_returns = closes.pct_change().abs() * 100.0

    valid_moves: List[Dict[str, Any]] = []

    for print_row in historical:
        dt_str = print_row["date"]
        try:
            target_dt = pd.to_datetime(dt_str)
        except Exception:
            continue

        # Look up price action on or next trading day
        idx_pos = closes.index.searchsorted(target_dt)
        if idx_pos < len(closes):
            actual_dt = closes.index[idx_pos]
            if actual_dt in daily_returns.index:
                move_val = daily_returns.loc[actual_dt]
                if pd.notna(move_val) and isinstance(move_val, (int, float)):
                    valid_moves.append({
                        "date": dt_str,
                        "abs_move_pct": round(float(move_val), 2),
                        "eps_actual": print_row.get("eps_actual"),
                        "surprise_pct": print_row.get("surprise_pct"),
                    })

    if not valid_moves:
        return {
            "n_prints": 0,
            "mean_move_pct": 0.0,
            "median_move_pct": 0.0,
            "moves": [],
        }

    move_values = [m["abs_move_pct"] for m in valid_moves]
    mean_move = float(np.mean(move_values))
    median_move = float(np.median(move_values))

    return {
        "n_prints": len(valid_moves),
        "mean_move_pct": round(mean_move, 2),
        "median_move_pct": round(median_move, 2),
        "moves": valid_moves,
    }


def get_earnings_implied_vs_realized(
    ticker: str,
    spot: float,
    chains_by_exp: Dict[str, pd.DataFrame],
    prices_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Full analysis comparing upcoming earnings implied move to past realized prints."""
    past_and_upcoming = earnings.get_earnings_history(ticker) or []
    upcoming = [p for p in past_and_upcoming if p.get("is_upcoming")]

    if not upcoming:
        return {
            "available": False,
            "reason": "No upcoming earnings date scheduled or published",
        }

    next_earn_date_str = upcoming[0]["date"]
    try:
        next_earn_dt = pd.to_datetime(next_earn_date_str)
    except Exception:
        return {
            "available": False,
            "reason": f"Unparseable next earnings date: {next_earn_date_str}",
        }

    # Find earliest option expiration strictly on or after earnings date
    sorted_expiries = sorted(chains_by_exp.keys())
    target_exp = None
    for exp in sorted_expiries:
        try:
            exp_dt = pd.to_datetime(exp)
            if exp_dt >= next_earn_dt:
                target_exp = exp
                break
        except Exception:
            continue

    if not target_exp or target_exp not in chains_by_exp:
        return {
            "available": False,
            "next_earnings_date": next_earn_date_str,
            "reason": f"No options expiration available after earnings date ({next_earn_date_str})",
        }

    target_chain = chains_by_exp[target_exp]
    straddle = compute_atm_straddle_move(target_chain, spot)
    if not straddle:
        return {
            "available": False,
            "next_earnings_date": next_earn_date_str,
            "target_expiration": target_exp,
            "reason": "ATM straddle quote invalid or missing on front post-earnings chain",
        }

    hist = compute_historical_earnings_moves(ticker, prices_df=prices_df)

    implied = straddle["implied_move_pct"]
    median_realized = hist["median_move_pct"]
    n_hist = hist["n_prints"]

    # Pricing valuation regime
    if n_hist >= 3 and median_realized > 0:
        ratio = implied / median_realized
        if ratio > 1.25:
            regime = "rich"
            regime_label = f"Rich (Implied {implied:.1f}% vs Median Realized {median_realized:.1f}%)"
        elif ratio < 0.75:
            regime = "cheap"
            regime_label = f"Cheap (Implied {implied:.1f}% vs Median Realized {median_realized:.1f}%)"
        else:
            regime = "fair"
            regime_label = f"Fairly priced (within 25% of median historical move)"
    else:
        regime = "insufficient_history"
        regime_label = f"Implied move {implied:.1f}% (historical base rate n={n_hist} insufficient)"

    return {
        "available": True,
        "next_earnings_date": next_earn_date_str,
        "target_expiration": target_exp,
        "implied_move_pct": implied,
        "straddle_price": straddle["straddle_price"],
        "atm_strike": straddle["atm_strike"],
        "historical_n": n_hist,
        "mean_realized_move_pct": hist["mean_move_pct"],
        "median_realized_move_pct": median_realized,
        "valuation_regime": regime,
        "regime_label": regime_label,
    }
