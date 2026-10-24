"""Realized Volatility Cones, Yang-Zhang (1998) estimator, and Expected Move Coverage Calibration."""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

from .models import RealizedVolCones
from .storage import compute_iv_rank_and_percentile


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

    # Overnight jump variance (close[t-1] to open[t])
    o_ret = np.log(open_p[1:] / close[:-1])
    var_o = float(np.var(o_ret, ddof=1))

    # Open-to-close continuous variance (open[t] to close[t])
    c_ret = np.log(close[1:] / open_p[1:])
    var_c = float(np.var(c_ret, ddof=1))

    # Rogers-Satchell intraday drift-independent variance
    u = np.log(high[1:] / open_p[1:])
    d = np.log(low[1:] / open_p[1:])
    c = np.log(close[1:] / open_p[1:])
    rs = u * (u - c) + d * (d - c)
    var_rs = float(np.mean(rs))

    # Yang-Zhang weighting factor k
    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    # Total Yang-Zhang annualized variance
    var_yz = var_o + k * var_c + (1.0 - k) * var_rs
    yz_vol = math.sqrt(max(0.0, var_yz)) * math.sqrt(252) * 100.0

    return round(yz_vol, 2)


def compute_expected_move_coverage(
    df: pd.DataFrame,
    current_iv: float,
    window: int = 30,
) -> float:
    """Measure empirical % of days the stock's absolute 1D return stayed inside implied 1-sigma cone."""
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
    ticker: Optional[str] = None,
    history_days: int = 252,
) -> RealizedVolCones:
    """Compile multi-window realized volatility cones, VRP, and IV rank.
    
    IV rank and percentile are strictly evaluated against stored IV history (never proxying
    with historical realized volatility). When history is under 63 days, is_accumulating is True
    and iv_rank/iv_percentile are None.
    """
    if daily_df is None or daily_df.empty:
        return RealizedVolCones(
            current_iv=current_iv,
            hv10=0.0, hv20=0.0, hv30=0.0, hv60=0.0, hv90=0.0,
            yang_zhang_30=0.0, yang_zhang_90=0.0,
            vrp=0.0, vrp_regime="Data Unavailable",
            iv_rank=None, iv_percentile=None,
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
        vrp_regime = "Positive VRP (Expensive Vol \u00b7 Net Short Premium Favored)"
    elif vrp < -5.0:
        vrp_regime = "Negative VRP (Underpriced Vol \u00b7 Net Long Gamma Favored)"
    else:
        vrp_regime = "Neutral / Fairly Priced Volatility"

    # Compute real IV rank/percentile from stored IV history
    iv_rank = None
    iv_percentile = None
    iv_min = round(current_iv * 0.7, 2)
    iv_max = round(current_iv * 1.4, 2)
    is_accumulating = True
    hist_days_count = 0

    if ticker:
        iv_stats = compute_iv_rank_and_percentile(ticker, current_iv, min_obs=63)
        hist_days_count = iv_stats["n_obs"]
        if iv_stats["available"]:
            iv_rank = iv_stats["iv_rank"]
            iv_percentile = iv_stats["iv_percentile"]
            iv_min = iv_stats["min_iv"]
            iv_max = iv_stats["max_iv"]
            is_accumulating = False

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
        history_days=hist_days_count,
    )
