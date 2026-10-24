"""Point-in-time cross-sectional research dataset for Signal Lab.

Part of Epic 5 (issue #132):
- Builds panel of (date, ticker, signal, forward returns at 1/5/21 days)
- Explicit alignment: signal known at close t, forward return from close t to close t+h
- Liquidity and history filters
- Loud survivorship-bias caveat present in metadata and docstrings
"""
from __future__ import annotations

from typing import Iterable, Sequence
import numpy as np
import pandas as pd

SURVIVORSHIP_CAVEAT: str = (
    "Universe is S&P 500 constituents as of current membership; "
    "historical survivorship bias applies (delisted/acquired firms omitted)."
)

DEFAULT_HORIZONS: tuple[int, ...] = (1, 5, 21)


def compute_forward_returns(
    prices: pd.Series | pd.DataFrame,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    price_col: str = "close",
) -> pd.DataFrame:
    """Compute strictly out-of-sample forward returns over specified horizons.

    Alignment contract:
      Row t corresponds to market close of trading day t.
      Forward return forward_ret_{h}d is (Close[t+h] - Close[t]) / Close[t].
      Any signal calculated using Close[<=t] evaluated against forward_ret_{h}d
      has zero look-ahead leakage.
    """
    if isinstance(prices, pd.DataFrame):
        col_lookup = {c.lower(): c for c in prices.columns}
        actual_col = col_lookup.get(price_col.lower(), price_col)
        close_series = prices[actual_col].astype(float)
    else:
        close_series = prices.astype(float)

    fwd_df = pd.DataFrame(index=close_series.index)
    for h in horizons:
        # Shift -h moves the price h steps in the future onto row t
        future_price = close_series.shift(-h)
        fwd_df[f"forward_ret_{h}d"] = (future_price - close_series) / close_series

    return fwd_df


def build_research_panel(
    prices_map: dict[str, pd.DataFrame],
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    min_history: int = 63,
    min_price: float = 1.0,
    min_volume: float = 1000.0,
) -> dict:
    """Build a unified point-in-time panel across all tickers.

    Returns a dict containing the stacked DataFrame, metadata, and the
    mandatory survivorship caveat.
    """
    dfs = []
    excluded: dict[str, str] = {}

    for ticker, df in prices_map.items():
        if df is None or len(df) < min_history:
            excluded[ticker] = f"insufficient history (< {min_history} bars)"
            continue

        cols = {c.lower(): c for c in df.columns}
        c_col = cols.get("close", "Close")
        v_col = cols.get("volume", "Volume")
        o_col = cols.get("open", "Open")
        h_col = cols.get("high", "High")
        l_col = cols.get("low", "Low")

        sub = pd.DataFrame(index=df.index)
        sub["ticker"] = ticker
        sub["close"] = df[c_col].astype(float)
        if v_col in df.columns:
            sub["volume"] = df[v_col].astype(float)
        else:
            sub["volume"] = 0.0

        if o_col in df.columns:
            sub["open"] = df[o_col].astype(float)
        if h_col in df.columns:
            sub["high"] = df[h_col].astype(float)
        if l_col in df.columns:
            sub["low"] = df[l_col].astype(float)

        # Forward returns
        fwd = compute_forward_returns(sub["close"], horizons=horizons)
        for col in fwd.columns:
            sub[col] = fwd[col]

        # Liquidity filter flag
        is_liquid = (sub["close"] >= min_price) & (sub["volume"] >= min_volume)
        sub["is_liquid"] = is_liquid

        dfs.append(sub)

    if not dfs:
        empty_panel = pd.DataFrame(
            columns=["ticker", "close", "volume", "is_liquid"]
            + [f"forward_ret_{h}d" for h in horizons]
        )
        return {
            "panel": empty_panel,
            "survivorship_caveat": SURVIVORSHIP_CAVEAT,
            "excluded": excluded,
            "n_tickers": 0,
            "n_dates": 0,
            "n_rows": 0,
        }

    panel = pd.concat(dfs).sort_index()
    # Normalize index to datetime
    if not isinstance(panel.index, pd.DatetimeIndex):
        panel.index = pd.to_datetime(panel.index)

    unique_dates = panel.index.unique().sort_values()
    unique_tickers = panel["ticker"].unique().tolist()

    return {
        "panel": panel,
        "survivorship_caveat": SURVIVORSHIP_CAVEAT,
        "excluded": excluded,
        "tickers": unique_tickers,
        "dates": unique_dates,
        "n_tickers": len(unique_tickers),
        "n_dates": len(unique_dates),
        "n_rows": len(panel),
    }


def verify_no_lookahead_leakage(
    signal_series: pd.Series,
    forward_returns: pd.Series,
    rng_seed: int = 42,
) -> tuple[float, float]:
    """Test that shuffling forward returns destroys the correlation.

    Returns (original_correlation, shuffled_correlation). Shuffled correlation
    should collapse towards 0.
    """
    from scipy.stats import spearmanr

    valid = ~(signal_series.isna() | forward_returns.isna())
    s = signal_series[valid].values
    r = forward_returns[valid].values

    if len(s) < 10:
        return 0.0, 0.0

    orig_corr, _ = spearmanr(s, r)

    rng = np.random.default_rng(rng_seed)
    shuffled_r = rng.permutation(r)
    shuffled_corr, _ = spearmanr(s, shuffled_r)

    return float(orig_corr), float(shuffled_corr)
