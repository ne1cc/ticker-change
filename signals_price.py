"""Price-based quantitative alpha signals.

Part of Epic 5 (issue #136):
- 12-1 Momentum: 12-month momentum skipping the most recent 1 month (21 days)
- Short-Term Reversal: Trailing 21-day return, negated
- Residual Reversal: Trailing 21-day idiosyncratic return from factor model, negated
- Low-Volatility Anomaly: Trailing 63-day realized volatility, negated
"""
from __future__ import annotations

import math
from typing import Sequence
import numpy as np
import pandas as pd


def compute_momentum_12_1(prices: pd.Series) -> pd.Series:
    """Classic 12-1 month price momentum (t-252 to t-21), skipping the last month."""
    s = prices.astype(float)
    p_t21 = s.shift(21)
    p_t252 = s.shift(252)
    valid = p_t252 > 0
    mom = pd.Series(np.nan, index=s.index)
    mom[valid] = (p_t21[valid] - p_t252[valid]) / p_t252[valid]
    return mom


def compute_short_term_reversal(prices: pd.Series, window: int = 21) -> pd.Series:
    """1-month (21-day) return negated. Losers receive positive scores."""
    s = prices.astype(float)
    p_prev = s.shift(window)
    valid = p_prev > 0
    ret = pd.Series(np.nan, index=s.index)
    ret[valid] = (s[valid] - p_prev[valid]) / p_prev[valid]
    # Negate return: reversal expects negative relationship with past performance
    return -ret


def compute_residual_reversal(
    stock_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 21,
    reg_window: int = 252,
) -> pd.Series:
    """Short-term reversal on idiosyncratic residuals, NOT raw returns.

    Regresses stock returns on benchmark (or factors) to extract residuals u_t.
    Signal is the negative of the trailing `window` cumulative residual return.
    """
    df = pd.DataFrame({"stock": stock_returns, "bm": benchmark_returns}).dropna()
    if len(df) < window:
        return pd.Series(np.nan, index=stock_returns.index)

    # Compute rolling or expanding OLS residuals
    residuals = pd.Series(np.nan, index=df.index, dtype=float)
    y = df["stock"].values
    x = df["bm"].values
    n = len(df)

    # For fast, stable residuals: rolling regression or overall OLS if < reg_window
    if n <= reg_window:
        X = np.column_stack([np.ones(n), x])
        coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
        u = y - X @ coeffs
        residuals.iloc[:] = u
    else:
        # Rolling regression over reg_window
        res_arr = np.full(n, np.nan)
        for i in range(reg_window, n + 1):
            w_y = y[i - reg_window:i]
            w_x = x[i - reg_window:i]
            X = np.column_stack([np.ones(reg_window), w_x])
            coeffs, *_ = np.linalg.lstsq(X, w_y, rcond=None)
            res_arr[i - 1] = w_y[-1] - (coeffs[0] + coeffs[1] * w_x[-1])
        residuals.iloc[:] = res_arr

    # Trailing window sum of residuals, negated
    trailing_resid = residuals.rolling(window, min_periods=window).sum()
    sig = -trailing_resid
    return sig.reindex(stock_returns.index)


def compute_low_volatility(prices: pd.Series, window: int = 63) -> pd.Series:
    """Realized volatility over trailing `window` days, annualised and negated."""
    s = prices.astype(float)
    ret = s.pct_change()
    rolling_std = ret.rolling(window, min_periods=window).std(ddof=1)
    ann_vol = rolling_std * math.sqrt(252.0)
    # Negate: low-volatility anomaly favors lower-vol stocks
    return -ann_vol


PRICE_SIGNAL_NAMES = [
    "momentum_12_1",
    "short_term_reversal",
    "residual_reversal",
    "low_volatility",
]


def add_price_signals_to_panel(
    panel: pd.DataFrame,
    spy_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute and append the 4 price-based signals to each ticker in the panel."""
    if panel.empty:
        return panel

    spy_ret = None
    if spy_df is not None and not spy_df.empty:
        c_col = {c.lower(): c for c in spy_df.columns}.get("close", "Close")
        spy_ret = spy_df[c_col].pct_change().dropna()

    out_dfs = []
    for ticker, sub in panel.groupby("ticker"):
        sub = sub.sort_index().copy()
        closes = sub["close"]
        returns = closes.pct_change()

        # 1. Momentum 12-1
        sub["momentum_12_1"] = compute_momentum_12_1(closes)

        # 2. Short-term reversal
        sub["short_term_reversal"] = compute_short_term_reversal(closes, window=21)

        # 3. Residual reversal
        if spy_ret is not None and not spy_ret.empty:
            sub_spy = spy_ret.reindex(returns.index).fillna(0.0)
            sub["residual_reversal"] = compute_residual_reversal(
                returns, sub_spy, window=21, reg_window=min(252, len(returns))
            )
        else:
            # Fall back to self-demeaned reversal if benchmark unavailable
            sub["residual_reversal"] = sub["short_term_reversal"]

        # 4. Low-volatility anomaly
        sub["low_volatility"] = compute_low_volatility(closes, window=63)

        out_dfs.append(sub)

    res = pd.concat(out_dfs).sort_index()
    return res
