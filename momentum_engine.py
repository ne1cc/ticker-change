"""Shared scoring + backtest engine for the /strategies page.

Five cross-sectional/trend/mean-reversion quantitative strategies share this module:
1. 12-1 relative-strength momentum (`relative_strength`)
2. Dual momentum (`dual_momentum` — relative-strength ranking gated by an absolute return hurdle)
3. 50/200 SMA golden-cross trend filter (`sma_trend`)
4. Connors RSI-2 Mean Reversion (`rsi_reversion` — buying deep pullbacks in macro uptrends)
5. Multi-Factor Composite (`buyer_signal` — confluence across trend, momentum, volatility)

All strategies share the same 253-bar (MOMENTUM_LOOKBACK + 1) warmup so their equity curves
are directly comparable.

No Flask, no Plotly, no `app` import — this module is imported by `app.py`, never the reverse.
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
    "relative_strength": {
        "label": "12-1 Relative Strength (Top-5 Rotation)",
        "family": "momentum",
        "hurdle": None,
    },
    "dual_momentum": {
        "label": "Dual Momentum (Relative + Absolute Filter)",
        "family": "momentum",
        "hurdle": RF_ANNUAL,
    },
    "sma_trend": {
        "label": "SMA Trend Following (50/200 Golden Cross)",
        "family": "sma",
        "hurdle": None,
    },
    "rsi_reversion": {
        "label": "Mean Reversion (RSI-2 Connors Pullback)",
        "family": "rsi",
        "hurdle": None,
    },
    "buyer_signal": {
        "label": "Multi-Factor Composite (Buyer Signals Alpha)",
        "family": "composite",
        "hurdle": None,
    },
}


def absolute_hurdle(strategy_id: str) -> float | None:
    """Annualised risk-free hurdle scaled to the 11-month momentum window
    (t-252 to t-21), or None for strategies with no absolute-return gate
    (including an unrecognized `strategy_id`)."""
    raw = STRATEGIES.get(strategy_id, {}).get("hurdle")
    if raw is None:
        return None
    return (1 + raw) ** ((MOMENTUM_LOOKBACK - MOMENTUM_EXCLUDE) / 252) - 1


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

    def _ret(now, then):
        if pd.isna(now) or pd.isna(then) or then <= 0:
            return 0.0
        return (now - then) / then

    mom_12_1 = _ret(p_21, p_252)
    mom_6m = _ret(p_latest, p_126)
    mom_3m = _ret(p_latest, p_63)
    mom_1m = _ret(p_latest, p_21)

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


def calc_rsi(series: pd.Series, period: int = 2) -> pd.Series:
    """Standard Wilder's RSI calculation."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def rsi_signal(close: pd.Series) -> pd.Series:
    """Stateful Larry Connors RSI-2 strategy simulation.
    Long entry: Close > 200-day SMA and RSI(2) < 15.
    Exit: RSI(2) > 70 or Close > 5-day SMA.
    """
    sma200 = close.rolling(200).mean()
    sma5 = close.rolling(5).mean()
    rsi2 = calc_rsi(close, 2)

    n = len(close)
    sig = np.zeros(n, dtype=float)
    in_pos = False

    for i in range(200, n):
        c = close.iloc[i]
        s200 = sma200.iloc[i]
        s5 = sma5.iloc[i]
        r = rsi2.iloc[i]

        if pd.isna(s200) or pd.isna(r):
            continue

        if not in_pos:
            if c > s200 and r < 15.0:
                in_pos = True
        else:
            if r > 70.0 or c > s5:
                in_pos = False

        sig[i] = 1.0 if in_pos else 0.0

    return pd.Series(sig, index=close.index)


def rsi_score_universe(price_df: pd.DataFrame, symbols: list[str],
                       as_of: int | None = None) -> dict[str, float]:
    """Score for RSI mean reversion:
    For stocks above 200 SMA, score is (100 - RSI2) so the most oversold ranks highest.
    For stocks below 200 SMA, score is negative (-RSI2) to penalize downtrends.
    """
    if as_of is None:
        as_of = len(price_df) - 1
    result: dict[str, float] = {}
    for sym in symbols:
        if sym not in price_df.columns:
            continue
        col = price_df[sym]
        if as_of < 200 or as_of >= len(col):
            continue
        c = col.iloc[as_of]
        sma200 = col.iloc[max(0, as_of - 199):as_of + 1].mean()
        sub = col.iloc[max(0, as_of - 50):as_of + 1]
        r = calc_rsi(sub, 2).iloc[-1]
        if pd.notna(c) and pd.notna(sma200) and pd.notna(r):
            if c > sma200:
                score = float(100.0 - r)  # Higher is more oversold in uptrend
            else:
                score = float(-r)         # Downtrend penalty
            result[sym] = score
    return result


def composite_score_series(close: pd.Series) -> pd.Series:
    """Multi-factor quantitative composite score (0 to 100) combining:
    1. Trend (40%): Price vs 200 SMA, 50 SMA, and EMA 8/21 cross.
    2. Momentum (35%): 14-day RSI and 63-day relative return.
    3. Volatility / Risk (25%): 20-day volatility normalized against 1-year median.
    Normalized into a 0 to 100 score where 50 is neutral, >60 is bullish, <40 is defensive.
    """
    n = len(close)
    if n < 50:
        return pd.Series(50.0, index=close.index)

    sma200 = close.rolling(200, min_periods=50).mean()
    sma50 = close.rolling(50, min_periods=20).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema8 = close.ewm(span=8, adjust=False).mean()

    # Trend component (-1 to +1)
    t1 = np.clip((close / sma200 - 1.0) / 0.15, -1.0, 1.0)
    t2 = np.clip((close / sma50 - 1.0) / 0.10, -1.0, 1.0)
    t3 = np.clip((ema8 / ema21 - 1.0) / 0.05, -1.0, 1.0)
    trend_comp = 0.45 * t1 + 0.35 * t2 + 0.20 * t3

    # Momentum component (-1 to +1)
    rsi14 = calc_rsi(close, 14)
    m1 = (rsi14 - 50.0) / 25.0
    m2 = np.clip(close.pct_change(63, fill_method=None) / 0.20, -1.0, 1.0).fillna(0.0)
    mom_comp = 0.50 * np.clip(m1, -1.0, 1.0) + 0.50 * m2

    # Volatility / Calmness component (-1 to +1)
    daily_rets = close.pct_change(fill_method=None)
    vol20 = daily_rets.rolling(20, min_periods=10).std() * np.sqrt(252)
    vol252 = daily_rets.rolling(252, min_periods=50).median() * np.sqrt(252)
    vol_ratio = vol20 / (vol252.replace(0.0, np.nan) + 1e-8)
    vol_comp = np.clip(1.5 - vol_ratio, -1.0, 1.0).fillna(0.0)

    raw = 0.40 * trend_comp + 0.35 * mom_comp + 0.25 * vol_comp
    score = 50.0 + 45.0 * raw
    return score.clip(0.0, 100.0).fillna(50.0)


def composite_signal(close: pd.Series) -> pd.Series:
    """Stateful trade signal for Multi-Factor Composite:
    Enters Long when score >= 55.0.
    Exits to Cash when score < 45.0.
    Maintains current state in the 45-55 neutral zone (hysteresis).
    """
    scores = composite_score_series(close)
    n = len(close)
    sig = np.zeros(n, dtype=float)
    in_pos = False

    for i in range(50, n):
        sc = scores.iloc[i]
        if pd.isna(sc):
            continue
        if not in_pos and sc >= 55.0:
            in_pos = True
        elif in_pos and sc < 45.0:
            in_pos = False
        sig[i] = 1.0 if in_pos else 0.0

    return pd.Series(sig, index=close.index)


def composite_score_universe(price_df: pd.DataFrame, symbols: list[str],
                             as_of: int | None = None) -> dict[str, float]:
    """Composite score (0-100) at row `as_of` per symbol."""
    if as_of is None:
        as_of = len(price_df) - 1

    result: dict[str, float] = {}
    for sym in symbols:
        if sym not in price_df.columns:
            continue
        col = price_df[sym]
        if as_of >= len(col):
            continue
        scores = composite_score_series(col)
        val = scores.iloc[as_of]
        if pd.notna(val):
            result[sym] = float(val)
    return result


def perf_stats(series, rf_annual: float = RF_ANNUAL) -> dict:
    """Geometric, risk-adjusted performance stats for a daily return series."""
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
    """Benchmark-relative stats: annualised Jensen alpha, beta, info ratio, corr."""
    df = pd.DataFrame({"strat": strat, "bench": bench}).dropna()
    empty = {"alpha": 0.0, "beta": 0.0, "info_ratio": 0.0, "corr": 0.0}
    if len(df) < 5:
        return empty

    cov = df["strat"].cov(df["bench"])
    bench_var = df["bench"].var()
    beta = (cov / bench_var) if bench_var > 0 else 0.0

    strat_cum = (1 + df["strat"]).prod() - 1
    bench_cum = (1 + df["bench"]).prod() - 1
    n = len(df)
    strat_cagr = (1 + strat_cum) ** (252.0 / n) - 1 if (1 + strat_cum) > 0 else -1.0
    bench_cagr = (1 + bench_cum) ** (252.0 / n) - 1 if (1 + bench_cum) > 0 else -1.0
    alpha = strat_cagr - (rf_annual + beta * (bench_cagr - rf_annual))

    diff = df["strat"] - df["bench"]
    te = diff.std(ddof=1) * np.sqrt(252)
    ir = (diff.mean() * np.sqrt(252) / te) if te > 0 else 0.0

    corr = df["strat"].corr(df["bench"])
    corr_val = float(corr) if pd.notna(corr) else 0.0

    return {
        "alpha": round(alpha * 100, 1),
        "beta": round(beta, 2),
        "info_ratio": round(ir, 2),
        "corr": round(corr_val, 2),
    }


@dataclass
class BacktestResult:
    dates: pd.DatetimeIndex
    strategy_returns: pd.Series
    trade_signal: pd.Series | None
    avg_turnover_pct: float | None


def backtest_rotation(price_df: pd.DataFrame, symbols: list[str],
                       strategy_id: str = "relative_strength",
                       top_n: int = TOP_N,
                       rebalance_freq: int = REBALANCE_FREQ,
                       cost_bps: float = DEFAULT_COST_BPS) -> BacktestResult:
    """Equal-weight top-N cross-sectional rotation, rebalanced every
    `rebalance_freq` bars, with turnover-scaled transaction costs.
    """
    daily_rets = price_df.pct_change(fill_method=None)
    start_idx = MOMENTUM_LOOKBACK + 1
    hurdle = absolute_hurdle(strategy_id)
    family = STRATEGIES.get(strategy_id, {}).get("family", "momentum")

    active_portfolio: list[str] = []
    prev_weights: dict[str, float] = {}
    total_turnover = 0.0
    rebalance_count = 0
    portfolio_returns = []
    backtest_dates = price_df.index[start_idx:]

    # Precompute indicator score series once per symbol for massive speedup
    indicator_series: dict[str, pd.Series] = {}
    if family == "sma":
        for sym in symbols:
            if sym in price_df.columns:
                indicator_series[sym] = sma_spread(price_df[sym])
    elif family == "rsi":
        for sym in symbols:
            if sym in price_df.columns:
                c = price_df[sym]
                s200 = c.rolling(200).mean()
                r2 = calc_rsi(c, 2)
                sc = pd.Series(np.where(c > s200, 100.0 - r2, -r2), index=c.index)
                sc[s200.isna() | r2.isna()] = np.nan
                indicator_series[sym] = sc
    elif family == "composite":
        for sym in symbols:
            if sym in price_df.columns:
                indicator_series[sym] = composite_score_series(price_df[sym])

    for i in range(start_idx, len(price_df)):
        is_rebalance = (i - start_idx) % rebalance_freq == 0
        cost_today = 0.0

        if is_rebalance:
            if family in ("sma", "rsi", "composite"):
                curr_scores = {}
                for sym, s_series in indicator_series.items():
                    if i < len(s_series):
                        val = s_series.iloc[i]
                        if pd.notna(val):
                            curr_scores[sym] = float(val)
                ranked = sorted(curr_scores.items(), key=lambda kv: kv[1], reverse=True)
                if family == "sma":
                    active_portfolio = [sym for sym, _ in ranked[:top_n]]
                elif family == "rsi":
                    active_portfolio = [sym for sym, s in ranked if s > 0][:top_n]
                else:  # composite
                    active_portfolio = [sym for sym, s in ranked if s >= 50.0][:top_n]
            else:
                scores = score_universe(price_df, symbols, as_of=i, hurdle=hurdle)
                ranked = sorted(scores.items(), key=lambda kv: kv[1].mom_12_1, reverse=True)
                candidates = [sym for sym, _ in ranked[:top_n]]
                if hurdle is not None:
                    active_portfolio = [sym for sym in candidates if scores[sym].passes_absolute]
                else:
                    active_portfolio = candidates

            if hurdle is not None or family in ("rsi", "composite"):
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
            if hurdle is not None or family in ("rsi", "composite"):
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
    """Single-ticker long/cash trend-following or mean-reversion backtest."""
    daily_rets = close.pct_change(fill_method=None)
    family = STRATEGIES.get(strategy_id, {}).get("family", "momentum")

    if family == "sma":
        raw_signal = sma_spread(close) > 0
    elif family == "rsi":
        raw_signal = rsi_signal(close) > 0
    elif family == "composite":
        raw_signal = composite_signal(close) > 0
    else:
        hurdle = absolute_hurdle(strategy_id)
        if hurdle is None:
            hurdle = 0.0
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
