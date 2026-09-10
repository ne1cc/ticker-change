"""Shared scoring + backtest engine for the /strategies page.

Three cross-sectional/trend strategies share this module: 12-1 relative-
strength momentum (`relative_strength`), dual momentum (`dual_momentum` —
relative-strength ranking gated by an absolute return hurdle), and a 50/200
SMA golden-cross trend filter (`sma_trend`). All three share the same
253-bar (MOMENTUM_LOOKBACK + 1) warmup so their equity curves are directly
comparable.

No Flask, no Plotly, no `app` import — this module is imported by `app.py`,
never the reverse.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

MOMENTUM_LOOKBACK = 252
MOMENTUM_EXCLUDE = 21
SMA_FAST = 50
SMA_SLOW = 200
DEFAULT_COST_BPS = 7.5
RF_ANNUAL = 0.04
TOP_N = 5
REBALANCE_FREQ = 21

STRATEGIES = {
    "relative_strength": "Relative Strength (12-1 Momentum)",
    "dual_momentum": "Dual Momentum (Relative + Absolute)",
    "sma_trend": "SMA 50/200 Golden Cross",
}


def absolute_hurdle(strategy_id: str) -> float | None:
    """Annualised risk-free hurdle scaled to the 11-month momentum window
    (t-252 to t-21), or None for strategies with no absolute-return gate."""
    if strategy_id != "dual_momentum":
        return None
    return (1 + RF_ANNUAL) ** ((MOMENTUM_LOOKBACK - MOMENTUM_EXCLUDE) / 252) - 1


@dataclass
class MomentumScore:
    symbol: str
    mom_12_1: float      # raw fraction, NOT percent
    mom_6m: float
    mom_3m: float
    mom_1m: float
    ann_vol_1y: float    # raw fraction
    risk_adj: float
    passes_absolute: bool


def rolling_score(close: pd.Series) -> pd.Series:
    """12-1 momentum score at every day: close.shift(21) / close.shift(252) - 1."""
    return close.shift(MOMENTUM_EXCLUDE) / close.shift(MOMENTUM_LOOKBACK) - 1


def score_series(close: pd.Series, symbol: str = "",
                  hurdle: float | None = None) -> MomentumScore | None:
    """Scalar momentum scores from a single close-price series.

    Returns None if there isn't enough history (MOMENTUM_LOOKBACK +
    MOMENTUM_EXCLUDE = 273 bars minimum).
    """
    if len(close) < MOMENTUM_LOOKBACK + MOMENTUM_EXCLUDE:
        return None

    p_latest = close.iloc[-1]
    p_21 = close.iloc[-22]
    p_63 = close.iloc[-64]
    p_126 = close.iloc[-127]
    p_252 = close.iloc[-253]

    mom_12_1 = (p_21 - p_252) / p_252
    mom_6m = (p_latest - p_126) / p_126
    mom_3m = (p_latest - p_63) / p_63
    mom_1m = (p_latest - p_21) / p_21

    vol = close.pct_change(fill_method=None).iloc[-MOMENTUM_LOOKBACK:].std() * math.sqrt(252)
    ann_vol_1y = float(vol) if pd.notna(vol) and vol > 0 else 0.0
    risk_adj = round(mom_12_1 / ann_vol_1y, 2) if ann_vol_1y > 0 else 0.0
    passes_absolute = hurdle is None or mom_12_1 > hurdle

    return MomentumScore(
        symbol=symbol,
        mom_12_1=mom_12_1,
        mom_6m=mom_6m,
        mom_3m=mom_3m,
        mom_1m=mom_1m,
        ann_vol_1y=ann_vol_1y,
        risk_adj=risk_adj,
        passes_absolute=passes_absolute,
    )


def score_universe(price_df: pd.DataFrame, symbols: list[str],
                    as_of: int | None = None,
                    hurdle: float | None = None) -> dict[str, MomentumScore]:
    """Momentum scores for every symbol in `symbols` present in `price_df`,
    evaluated at integer row position `as_of` (default: last row).

    Symbols that can't be scored (missing column, insufficient history,
    NaN or non-positive anchor price) are absent from the result. When
    `as_of` is left at its default (the latest row), the 6m/3m/1m horizons
    are computed the same way `score_series` computes them; for an
    arbitrary historical `as_of` (as used by the rotation backtest) they
    are left at 0.0 since only the leaderboard/screener consume them.
    """
    at_latest = as_of is None
    if as_of is None:
        as_of = len(price_df) - 1
    if as_of < MOMENTUM_LOOKBACK:
        return {}

    daily_rets = price_df.pct_change(fill_method=None)
    result: dict[str, MomentumScore] = {}

    for sym in symbols:
        if sym not in price_df.columns:
            continue
        col = price_df[sym]
        if as_of >= len(col):
            continue

        p_past = col.iloc[as_of - MOMENTUM_LOOKBACK]
        p_recent = col.iloc[as_of - MOMENTUM_EXCLUDE]
        if pd.isna(p_past) or pd.isna(p_recent) or p_past <= 0:
            continue
        mom_12_1 = (p_recent - p_past) / p_past

        vol_window = daily_rets[sym].iloc[as_of - MOMENTUM_LOOKBACK + 1: as_of + 1]
        vol = vol_window.std() * math.sqrt(252)
        ann_vol_1y = float(vol) if pd.notna(vol) and vol > 0 else 0.0
        risk_adj = round(mom_12_1 / ann_vol_1y, 2) if ann_vol_1y > 0 else 0.0
        passes_absolute = hurdle is None or mom_12_1 > hurdle

        mom_6m = mom_3m = mom_1m = 0.0
        if at_latest:
            p_latest = col.iloc[as_of]
            p_63 = col.iloc[as_of - 63]
            p_126 = col.iloc[as_of - 126]
            mom_6m = (p_latest - p_126) / p_126
            mom_3m = (p_latest - p_63) / p_63
            mom_1m = (p_latest - p_recent) / p_recent

        result[sym] = MomentumScore(
            symbol=sym,
            mom_12_1=mom_12_1,
            mom_6m=mom_6m,
            mom_3m=mom_3m,
            mom_1m=mom_1m,
            ann_vol_1y=ann_vol_1y,
            risk_adj=risk_adj,
            passes_absolute=passes_absolute,
        )

    return result


def sma_spread(close: pd.Series, fast: int = SMA_FAST,
               slow: int = SMA_SLOW) -> pd.Series:
    """Rolling (SMA_fast - SMA_slow) / SMA_slow, full series, NaN during warmup."""
    sma_fast = close.rolling(fast).mean()
    sma_slow = close.rolling(slow).mean()
    return (sma_fast - sma_slow) / sma_slow


def sma_score_universe(price_df: pd.DataFrame, symbols: list[str],
                        as_of: int | None = None) -> dict[str, float]:
    """SMA spread at row `as_of` per symbol; unscorable symbols absent."""
    if as_of is None:
        as_of = len(price_df) - 1

    result: dict[str, float] = {}
    for sym in symbols:
        if sym not in price_df.columns:
            continue
        spread = sma_spread(price_df[sym])
        if as_of >= len(spread):
            continue
        val = spread.iloc[as_of]
        if pd.notna(val):
            result[sym] = float(val)
    return result


def perf_stats(series, rf_annual: float = RF_ANNUAL) -> dict:
    """Geometric, risk-adjusted performance stats for a daily return series.

    Uses CAGR (not the arithmetic-mean annualisation, which overstates
    returns for volatile series) and the standard Sharpe
    (sqrt(252) * mean excess / std). Ported verbatim from app.py's
    `_perf_stats`.
    """
    series = pd.Series(series).dropna()
    n = len(series)
    empty = {"total_return": 0.0, "annual_return": 0.0, "volatility": 0.0,
             "sharpe": 0.0, "sortino": 0.0, "max_dd": 0.0, "calmar": 0.0}
    if n == 0:
        return empty

    cum = (1 + series).prod() - 1
    cagr = (1 + cum) ** (252.0 / n) - 1 if (1 + cum) > 0 else -1.0

    std = series.std(ddof=1) if n > 1 else 0.0
    ann_vol = std * np.sqrt(252)
    rf_daily = rf_annual / 252.0
    excess = series - rf_daily
    sharpe = (excess.mean() / std * np.sqrt(252)) if std > 0 else 0.0

    downside = excess[excess < 0]
    dd_std = downside.std(ddof=1) if len(downside) > 1 else 0.0
    sortino = (excess.mean() / dd_std * np.sqrt(252)) if dd_std > 0 else 0.0

    cum_prod = (1 + series).cumprod()
    running_max = cum_prod.cummax()
    drawdown = (cum_prod - running_max) / running_max
    max_dd = drawdown.min()
    calmar = (cagr / abs(max_dd)) if max_dd < 0 else 0.0

    return {
        "total_return": round(cum * 100, 1),
        "annual_return": round(cagr * 100, 1),
        "volatility": round(ann_vol * 100, 1),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "max_dd": round(max_dd * 100, 1),
        "calmar": round(calmar, 2),
    }


def relative_stats(strat, bench, rf_annual: float = RF_ANNUAL) -> dict:
    """Benchmark-relative stats: annualised Jensen alpha, beta, info ratio,
    corr. Ported verbatim from app.py's `_relative_stats`."""
    df = pd.concat([pd.Series(strat), pd.Series(bench)], axis=1).dropna()
    df.columns = ["s", "b"]
    empty = {"alpha": 0.0, "beta": 0.0, "info_ratio": 0.0, "corr": 0.0}
    if len(df) < 2 or df["b"].var() == 0:
        return empty

    beta = df["s"].cov(df["b"]) / df["b"].var()
    rf_daily = rf_annual / 252.0
    alpha_daily = (df["s"].mean() - rf_daily) - beta * (df["b"].mean() - rf_daily)
    alpha_ann = ((1 + alpha_daily) ** 252 - 1) * 100

    active = df["s"] - df["b"]
    act_std = active.std(ddof=1)
    info = (active.mean() / act_std * np.sqrt(252)) if act_std > 0 else 0.0

    return {
        "alpha": round(alpha_ann, 1),
        "beta": round(beta, 2),
        "info_ratio": round(info, 2),
        "corr": round(df["s"].corr(df["b"]), 2),
    }


@dataclass
class BacktestResult:
    dates: pd.DatetimeIndex
    strategy_returns: pd.Series
    trade_signal: pd.Series | None = None   # timeseries only
    avg_turnover_pct: float | None = None   # rotation only


def backtest_rotation(price_df: pd.DataFrame, symbols: list[str],
                       strategy_id: str = "relative_strength",
                       top_n: int = TOP_N,
                       rebalance_freq: int = REBALANCE_FREQ,
                       cost_bps: float = DEFAULT_COST_BPS) -> BacktestResult:
    """Equal-weight top-N cross-sectional rotation, rebalanced every
    `rebalance_freq` bars, with turnover-scaled transaction costs.

    `strategy_id == "sma_trend"` ranks by SMA spread; everything else ranks
    by 12-1 momentum. `dual_momentum` additionally drops (to cash) any
    top-N name that fails its absolute-return hurdle rather than
    backfilling from rank N+1 — see the module-level ruling on cash drag.
    """
    daily_rets = price_df.pct_change(fill_method=None)
    start_idx = MOMENTUM_LOOKBACK + 1
    hurdle = absolute_hurdle(strategy_id)
    is_sma = strategy_id == "sma_trend"

    active_portfolio: list[str] = []
    prev_weights: dict[str, float] = {}
    total_turnover = 0.0
    rebalance_count = 0
    portfolio_returns = []
    backtest_dates = price_df.index[start_idx:]

    for i in range(start_idx, len(price_df)):
        is_rebalance = (i - start_idx) % rebalance_freq == 0
        cost_today = 0.0

        if is_rebalance:
            if is_sma:
                spreads = sma_score_universe(price_df, symbols, as_of=i)
                ranked = sorted(spreads.items(), key=lambda kv: kv[1], reverse=True)
                active_portfolio = [sym for sym, _ in ranked[:top_n]]
            else:
                scores = score_universe(price_df, symbols, as_of=i, hurdle=hurdle)
                ranked = sorted(scores.items(), key=lambda kv: kv[1].mom_12_1, reverse=True)
                candidates = [sym for sym, _ in ranked[:top_n]]
                if hurdle is not None:
                    active_portfolio = [sym for sym in candidates if scores[sym].passes_absolute]
                else:
                    active_portfolio = candidates

            if hurdle is not None:
                new_weights = {t: 1.0 / top_n for t in active_portfolio}
            else:
                new_weights = {t: 1.0 / len(active_portfolio) for t in active_portfolio} if active_portfolio else {}

            names = set(new_weights) | set(prev_weights)
            turnover = sum(abs(new_weights.get(t, 0.0) - prev_weights.get(t, 0.0)) for t in names)
            if i > start_idx:
                cost_today = (turnover / 2.0) * (cost_bps / 1e4)
                total_turnover += turnover / 2.0
                rebalance_count += 1
            prev_weights = new_weights

        if active_portfolio:
            if hurdle is not None:
                daily_ret = daily_rets[active_portfolio].iloc[i].sum() / top_n
            else:
                daily_ret = daily_rets[active_portfolio].iloc[i].mean()
        else:
            daily_ret = 0.0

        daily_ret -= cost_today
        portfolio_returns.append(daily_ret)

    strategy_returns = pd.Series(portfolio_returns, index=backtest_dates)
    avg_turnover_pct = (
        round((total_turnover / rebalance_count) * (252.0 / rebalance_freq) * 100, 0)
        if rebalance_count else 0.0
    )

    return BacktestResult(
        dates=backtest_dates,
        strategy_returns=strategy_returns,
        trade_signal=None,
        avg_turnover_pct=avg_turnover_pct,
    )


def backtest_timeseries(close: pd.Series, strategy_id: str = "relative_strength",
                         cost_bps: float = DEFAULT_COST_BPS) -> BacktestResult:
    """Single-ticker long/cash trend-following backtest.

    `sma_trend` goes long when the SMA spread is positive (golden cross);
    everything else goes long when the 12-1 momentum score clears its
    absolute hurdle (0.0 for strategies with no hurdle).
    """
    daily_rets = close.pct_change(fill_method=None)

    if strategy_id == "sma_trend":
        raw_signal = sma_spread(close) > 0
    else:
        hurdle = absolute_hurdle(strategy_id) or 0.0
        raw_signal = rolling_score(close) > hurdle

    signal = raw_signal.astype(float).shift(1).fillna(0.0)
    trades = signal.diff().abs().fillna(0.0)
    strat_rets = signal * daily_rets - trades * (cost_bps / 1e4)

    warmup = MOMENTUM_LOOKBACK + 1
    return BacktestResult(
        dates=close.index[warmup:],
        strategy_returns=strat_rets.iloc[warmup:],
        trade_signal=signal.iloc[warmup:],
        avg_turnover_pct=None,
    )
