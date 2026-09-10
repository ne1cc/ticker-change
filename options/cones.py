"""Realized Volatility Cones, Yang-Zhang (1998) estimator, and Expected Move Coverage Calibration."""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

from .models import RealizedVolCones


def compute_close_to_close_hv(returns: np.ndarray, window: int) -> float:
    """Annualized close-to-close realized volatility."""
    if len(returns) < max(window, 5):
        return 0.0
    sub = returns[-window:]
    std = float(np.std(sub, ddof=1))
    return round(std * math.sqrt(252) * 100.0, 2)


def compute_yang_zhang_vol(df: pd.DataFrame, window: int = 30) -> float:
    """Compute Yang-Zhang (1998) drift-independent, jump-adjusted volatility estimator.

    Combines overnight jumps, continuous open-to-close drift, and Rogers-Satchell high-low intraday components.
    """
    if len(df) < window + 1:
        return 0.0

    # Ensure lowercase columns
    sub = df.iloc[-(window + 1):].copy()
    c_col = "close" if "close" in sub.columns else "Close"
    o_col = "open" if "open" in sub.columns else "Open"
    h_col = "high" if "high" in sub.columns else "High"
    l_col = "low" if "low" in sub.columns else "Low"

    close = sub[c_col].to_numpy(dtype=np.float64)
    open_p = sub[o_col].to_numpy(dtype=np.float64)
    high = sub[h_col].to_numpy(dtype=np.float64)
    low = sub[l_col].to_numpy(dtype=np.float64)

    # 1. Overnight jump variance (open_t vs close_{t-1})
    log_oc = np.log(open_p[1:] / close[:-1])
    var_o = float(np.var(log_oc, ddof=1))

    # 2. Open-to-close variance
    log_co = np.log(close[1:] / open_p[1:])
    var_c = float(np.var(log_co, ddof=1))

    # 3. Rogers-Satchell intraday variance
    h = high[1:]
    l = low[1:]
    c = close[1:]
    o = open_p[1:]
    rs = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
    var_rs = float(np.mean(rs))

    # Yang-Zhang weighting factor k
    n = float(window)
    k = 0.34 / (1.34 + (n + 1.0) / (n - 1.0))

    var_yz = var_o + k * var_c + (1.0 - k) * var_rs
    if var_yz <= 0:
        return 0.0

    return round(math.sqrt(var_yz * 252.0) * 100.0, 2)


def compute_expected_move_coverage(
    df: pd.DataFrame,
    current_iv: float,
    window: int = 30,
) -> float:
    """Measure empirical coverage of theoretical 1-sigma expected moves over past N days.

    Under Black-Scholes normal assumptions, 1-sigma daily moves should bracket ~68.2% of sessions.
    """
    if len(df) < window + 1 or current_iv <= 0:
        return 68.0

    c_col = "close" if "close" in df.columns else "Close"
    closes = df[c_col].iloc[-(window + 1):].to_numpy(dtype=np.float64)

    daily_sigma = (current_iv / 100.0) * math.sqrt(1.0 / 252.0)
    expected_moves = closes[:-1] * daily_sigma
    actual_moves = np.abs(closes[1:] - closes[:-1])

    within = np.sum(actual_moves <= expected_moves)
    return round(float(within / len(actual_moves) * 100.0), 1)


def compute_vol_cones_and_vrp(
    daily_df: pd.DataFrame,
    current_iv: float,
    history_days: int = 252,
) -> RealizedVolCones:
    """Compile multi-window realized volatility cones, VRP, and IV rank."""
    if daily_df is None or daily_df.empty:
        return RealizedVolCones(
            current_iv=current_iv,
            hv10=0.0, hv20=0.0, hv30=0.0, hv60=0.0, hv90=0.0,
            yang_zhang_30=0.0, yang_zhang_90=0.0,
            vrp=0.0, vrp_regime="Data Unavailable",
            iv_rank=0.0, iv_percentile=0.0,
            iv_min_52w=current_iv, iv_max_52w=current_iv,
            expected_move_1d=0.0, expected_move_straddle=0.0,
            coverage_68_pct=68.0, is_accumulating=True, history_days=0,
        )

    c_col = "close" if "close" in daily_df.columns else "Close"
    prices = daily_df[c_col].dropna().to_numpy(dtype=np.float64)
    returns = np.diff(np.log(prices)) if len(prices) > 1 else np.array([])

    hv10 = compute_close_to_close_hv(returns, 10)
    hv20 = compute_close_to_close_hv(returns, 20)
    hv30 = compute_close_to_close_hv(returns, 30)
    hv60 = compute_close_to_close_hv(returns, 60)
    hv90 = compute_close_to_close_hv(returns, 90)

    yz30 = compute_yang_zhang_vol(daily_df, 30)
    yz90 = compute_yang_zhang_vol(daily_df, 90)

    # Volatility Risk Premium (VRP) = ATM IV - HV30
    vrp = round(current_iv - (hv30 or yz30), 2)
    if vrp > 5.0:
        vrp_regime = "Positive VRP (Expensive Vol · Net Short Premium Favored)"
    elif vrp < -5.0:
        vrp_regime = "Negative VRP (Underpriced Vol · Net Long Gamma Favored)"
    else:
        vrp_regime = "Neutral / Fairly Priced Volatility"

    # 52-week IV range proxy from realized volatility bands + current IV
    # Rolling 30D volatility over past 252 days as proxy for annual IV distribution
    if len(returns) >= 30:
        roll_30 = []
        for i in range(30, len(returns) + 1):
            w = returns[i - 30:i]
            roll_30.append(float(np.std(w, ddof=1) * math.sqrt(252) * 100.0))
        iv_min = round(float(min(roll_30)), 2)
        iv_max = round(float(max(roll_30)), 2)
        
        # IV Rank
        if iv_max > iv_min:
            iv_rank = round(max(0.0, min(100.0, ((current_iv - iv_min) / (iv_max - iv_min)) * 100.0)), 1)
        else:
            iv_rank = 50.0

        # IV Percentile
        pct_count = sum(1 for v in roll_30 if v < current_iv)
        iv_percentile = round((pct_count / len(roll_30)) * 100.0, 1)
        is_accumulating = len(roll_30) < 180
    else:
        iv_min = round(current_iv * 0.7, 2)
        iv_max = round(current_iv * 1.4, 2)
        iv_rank = 50.0
        iv_percentile = 50.0
        is_accumulating = True

    spot = float(prices[-1]) if len(prices) > 0 else 100.0
    expected_move_1d = round(spot * (current_iv / 100.0) * math.sqrt(1.0 / 252.0), 2)
    expected_move_straddle = round(spot * (current_iv / 100.0) * math.sqrt(30.0 / 365.0) * 0.8, 2)

    coverage_68 = compute_expected_move_coverage(daily_df, current_iv, window=30)

    return RealizedVolCones(
        current_iv=current_iv,
        hv10=hv10,
        hv20=hv20,
        hv30=hv30,
        hv60=hv60,
        hv90=hv90,
        yang_zhang_30=yz30,
        yang_zhang_90=yz90,
        vrp=vrp,
        vrp_regime=vrp_regime,
        iv_rank=iv_rank,
        iv_percentile=iv_percentile,
        iv_min_52w=iv_min,
        iv_max_52w=iv_max,
        expected_move_1d=expected_move_1d,
        expected_move_straddle=expected_move_straddle,
        coverage_68_pct=coverage_68,
        is_accumulating=is_accumulating,
        history_days=len(returns),
    )
