"""Quantitative Signal Simulation & Backtesting Engine.

Runs vectorised, walk-forward simulations on the multi-factor Buyer Signal score
with slippage, commission friction, drawdown tracking, and performance attribution.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

import signals
import ml


@dataclass
class BacktestSummary:
    total_return_pct: float
    cagr_pct: float
    benchmark_return_pct: float
    alpha_pct: float
    annualized_sharpe: float
    annualized_sortino: float
    max_drawdown_pct: float
    calmar_ratio: float
    win_rate_pct: float
    profit_factor: float
    total_trades: int
    factors_used: List[str] = field(default_factory=list)
    n_folds: int = 0
    oos_sharpe_mean: float = 0.0
    oos_sharpe_std: float = 0.0
    fold_returns: List[float] = field(default_factory=list)


def _historical_composite_signal(df: pd.DataFrame) -> Tuple[pd.Series, List[str]]:
    """Per-bar 0-100 composite score, restricted to the factors that are
    honestly and cheaply computable from OHLCV history alone: trend,
    momentum, volatility, tail_risk, ml. Reuses signals.py's own factor
    functions directly (not reimplemented) so this is provably the same
    calculation the live /analytics page uses for these factors.

    gex, monte_carlo, and valuation are excluded -- see spec section
    'Constraints discovered during design' for why (no historical options
    data, simulation cost/fidelity tradeoff, and no historical fundamentals
    integration yet, respectively).
    """
    df_lc = df.rename(columns=str.lower)
    feat_df = ml.build_features(df_lc)

    returns = df_lc["close"].pct_change()
    rolling_vol_pct = feat_df["vol_20"] * math.sqrt(252) * 100.0
    rolling_var95_pct = returns.rolling(252).quantile(0.05) * 100.0
    rolling_var99_pct = returns.rolling(252).quantile(0.01) * 100.0

    valid = ~feat_df.isna().any(axis=1)
    valid_positions = [i for i in range(len(df)) if valid.iloc[i]]

    # Batch the model call once across all valid rows -- never call
    # predict_proba inside the per-bar loop below (see Global Constraints).
    ml_actions: Dict[int, Tuple[str, float]] = {}
    bundle = ml._load()
    if bundle is not None and bundle.get("features") == ml.FEATURES and valid_positions:
        model = bundle["model"]
        classes = list(model.classes_)
        valid_feat = feat_df.iloc[valid_positions]
        proba_matrix = model.predict_proba(valid_feat)
        for pos, proba in zip(valid_positions, proba_matrix):
            p = {int(c): float(pr) for c, pr in zip(classes, proba)}
            buy, hold, sell = p.get(1, 0.0), p.get(0, 0.0), p.get(-1, 0.0)
            top = int(max(p, key=p.get))
            action = {1: "Buy", 0: "Hold", -1: "Sell"}[top]
            ml_actions[pos] = (action, max(buy, hold, sell))

    composite = pd.Series(np.nan, index=df.index)
    factors_used: set = set()

    for i in range(len(df)):
        if not valid.iloc[i]:
            continue
        row = feat_df.iloc[i]
        candidates = [
            signals._trend(row),
            signals._momentum(row),
            signals._volatility(row, {"annualised_vol": rolling_vol_pct.iloc[i]}),
            signals._tail_risk({"var_95": rolling_var95_pct.iloc[i], "var_99": rolling_var99_pct.iloc[i]}),
        ]
        if i in ml_actions:
            action, conf = ml_actions[i]
            candidates.append(signals._ml({"ml": {"action": action, "confidence": conf}}))

        factors = [f for f in candidates if f]
        if not factors:
            continue
        wsum = sum(f["weight"] for f in factors)
        blended = sum(f["score"] * f["weight"] for f in factors) / wsum
        composite.iloc[i] = round(50 + 50 * blended)
        factors_used.update(f["key"] for f in factors)

    return composite, sorted(factors_used)


def run_signals_backtest(
    prices_df: pd.DataFrame,
    entry_threshold: float = 65.0,
    exit_threshold: float = 45.0,
    slippage_bps: float = 5.0,
    commission_bps: float = 1.0,
) -> Tuple[BacktestSummary, pd.DataFrame]:
    """Execute walk-forward trading simulation driven by quantitative momentum/signals."""
    if len(prices_df) < 50:
        empty_res = BacktestSummary(
            total_return_pct=0.0,
            cagr_pct=0.0,
            benchmark_return_pct=0.0,
            alpha_pct=0.0,
            annualized_sharpe=0.0,
            annualized_sortino=0.0,
            max_drawdown_pct=0.0,
            calmar_ratio=0.0,
            win_rate_pct=0.0,
            profit_factor=0.0,
            total_trades=0,
            factors_used=[],
        )
        return empty_res, pd.DataFrame()

    df = prices_df.copy()
    c_col = "close" if "close" in df.columns else "Close"
    df["ret"] = df[c_col].pct_change().fillna(0.0)

    signal, factors_used = _historical_composite_signal(df)
    df["signal"] = signal

    # Position simulation with slippage
    friction = (slippage_bps + commission_bps) / 10000.0

    in_pos = False
    positions = []
    strat_returns = []
    trades = []
    entry_p = 0.0

    for i in range(len(df)):
        sig = df["signal"].iloc[i]
        curr_p = df[c_col].iloc[i]
        ret = df["ret"].iloc[i]

        if not in_pos and sig >= entry_threshold:
            in_pos = True
            entry_p = curr_p * (1.0 + friction)
            positions.append(1)
            strat_returns.append(-friction)
        elif in_pos and sig <= exit_threshold:
            in_pos = False
            exit_p = curr_p * (1.0 - friction)
            trade_ret = (exit_p - entry_p) / entry_p
            trades.append(trade_ret)
            positions.append(0)
            strat_returns.append(ret - friction)
        elif in_pos:
            positions.append(1)
            strat_returns.append(ret)
        else:
            positions.append(0)
            strat_returns.append(0.0)

    df["strat_ret"] = strat_returns
    df["position"] = positions
    df["cum_strat"] = (1.0 + df["strat_ret"]).cumprod()
    df["cum_bench"] = (1.0 + df["ret"]).cumprod()

    # Performance metrics
    total_strat_ret = float(df["cum_strat"].iloc[-1] - 1.0) * 100.0
    total_bench_ret = float(df["cum_bench"].iloc[-1] - 1.0) * 100.0
    alpha = total_strat_ret - total_bench_ret

    n_days = len(df)
    n_years = max(n_days / 252.0, 0.1)
    cagr = (float(df["cum_strat"].iloc[-1]) ** (1.0 / n_years) - 1.0) * 100.0

    # Drawdown
    rolling_peak = df["cum_strat"].cummax()
    drawdown = (df["cum_strat"] - rolling_peak) / rolling_peak
    max_dd = abs(float(drawdown.min())) * 100.0

    # Sharpe & Sortino
    mean_ret = df["strat_ret"].mean() * 252.0
    std_ret = df["strat_ret"].std() * math.sqrt(252.0)
    sharpe = float(mean_ret / std_ret) if std_ret > 0 else 0.0

    downside_ret = df[df["strat_ret"] < 0]["strat_ret"]
    downside_std = downside_ret.std() * math.sqrt(252.0) if len(downside_ret) > 1 else std_ret
    sortino = float(mean_ret / downside_std) if downside_std > 0 else 0.0

    calmar = float(cagr / max_dd) if max_dd > 0 else 0.0

    win_trades = [t for t in trades if t > 0]
    loss_trades = [t for t in trades if t <= 0]
    win_rate = (len(win_trades) / len(trades) * 100.0) if trades else 0.0
    gross_profits = sum(win_trades)
    gross_losses = abs(sum(loss_trades))
    profit_factor = (gross_profits / gross_losses) if gross_losses > 0 else 2.5

    summary = BacktestSummary(
        total_return_pct=round(total_strat_ret, 2),
        cagr_pct=round(cagr, 2),
        benchmark_return_pct=round(total_bench_ret, 2),
        alpha_pct=round(alpha, 2),
        annualized_sharpe=round(sharpe, 2),
        annualized_sortino=round(sortino, 2),
        max_drawdown_pct=round(max_dd, 2),
        calmar_ratio=round(calmar, 2),
        win_rate_pct=round(win_rate, 1),
        profit_factor=round(profit_factor, 2),
        total_trades=len(trades),
        factors_used=factors_used,
    )

    return summary, df


def run_walkforward_backtest(
    prices_df: pd.DataFrame,
    train_days: int = 252,
    test_days: int = 63,
    step_days: int = 63,
    entry_threshold: float = 65.0,
    exit_threshold: float = 45.0,
    slippage_bps: float = 5.0,
    commission_bps: float = 1.0,
) -> Tuple[BacktestSummary, pd.DataFrame]:
    """Rolling walk-forward OOS evaluation. entry_threshold/exit_threshold
    are FIXED constants, not fit per fold -- this app doesn't grid-search
    thresholds today (see spec). train_days is warm-up burn-in only: each
    fold runs run_signals_backtest on a (train_days + test_days) window and
    keeps only the last test_days rows for scoring, so indicators and
    position state are realistic entering the OOS period without any
    parameter being fit on it.

    Falls back to a single run_signals_backtest call on the whole series if
    there isn't enough history for even one fold.
    """
    if len(prices_df) < train_days + test_days:
        summary, df = run_signals_backtest(
            prices_df, entry_threshold, exit_threshold, slippage_bps, commission_bps
        )
        summary.n_folds = 0
        summary.oos_sharpe_mean = summary.annualized_sharpe
        summary.oos_sharpe_std = 0.0
        summary.fold_returns = []
        return summary, df

    fold_sharpes: List[float] = []
    fold_return_pcts: List[float] = []
    oos_frames: List[pd.DataFrame] = []
    fold_factors_used: Set[str] = set()
    trades: List[float] = []

    start = 0
    while start + train_days + test_days <= len(prices_df):
        window = prices_df.iloc[start: start + train_days + test_days]
        fold_summary, fold_df = run_signals_backtest(
            window, entry_threshold, exit_threshold, slippage_bps, commission_bps
        )
        if not fold_df.empty:
            oos = fold_df.iloc[train_days:]
            if not oos.empty:
                oos_ret = oos["strat_ret"]
                mean_ret = oos_ret.mean() * 252.0
                std_ret = oos_ret.std() * math.sqrt(252.0)
                fold_sharpe = float(mean_ret / std_ret) if std_ret > 0 else 0.0
                fold_total_ret = float((1.0 + oos_ret).prod() - 1.0) * 100.0
                fold_sharpes.append(fold_sharpe)
                fold_return_pcts.append(round(fold_total_ret, 2))
                oos_frames.append(oos)
                fold_factors_used.update(fold_summary.factors_used)

                # Reconstruct closed-trade returns from this fold's own OOS
                # slice only (Task 1's position column) so a position still
                # open at the end of one fold's OOS window is never spliced
                # onto the next fold's unrelated first trade -- each fold's
                # trade-reconstruction state starts fresh.
                fold_pos = oos["position"].to_numpy()
                fold_strat_ret = oos["strat_ret"].to_numpy()
                current_trade: List[float] = []
                prev = 0
                for p, r in zip(fold_pos, fold_strat_ret):
                    if p == 1 and prev == 0:
                        current_trade = [float(r)]
                    elif p == 1 and prev == 1:
                        current_trade.append(float(r))
                    elif p == 0 and prev == 1:
                        current_trade.append(float(r))
                        trades.append(float(np.prod([1.0 + x for x in current_trade]) - 1.0))
                        current_trade = []
                    prev = p
        start += step_days

    if not fold_sharpes:
        summary, df = run_signals_backtest(
            prices_df, entry_threshold, exit_threshold, slippage_bps, commission_bps
        )
        summary.n_folds = 0
        summary.oos_sharpe_mean = summary.annualized_sharpe
        summary.oos_sharpe_std = 0.0
        summary.fold_returns = []
        return summary, df

    combined = pd.concat(oos_frames)

    total_strat_ret = float((1.0 + combined["strat_ret"]).prod() - 1.0) * 100.0
    total_bench_ret = float((1.0 + combined["ret"]).prod() - 1.0) * 100.0
    alpha = total_strat_ret - total_bench_ret

    n_days = len(combined)
    n_years = max(n_days / 252.0, 0.1)
    cum_strat = (1.0 + combined["strat_ret"]).cumprod()
    cagr = (float(cum_strat.iloc[-1]) ** (1.0 / n_years) - 1.0) * 100.0

    rolling_peak = cum_strat.cummax()
    drawdown = (cum_strat - rolling_peak) / rolling_peak
    max_dd = abs(float(drawdown.min())) * 100.0

    mean_ret = combined["strat_ret"].mean() * 252.0
    std_ret = combined["strat_ret"].std() * math.sqrt(252.0)
    sharpe = float(mean_ret / std_ret) if std_ret > 0 else 0.0

    downside_ret = combined[combined["strat_ret"] < 0]["strat_ret"]
    downside_std = downside_ret.std() * math.sqrt(252.0) if len(downside_ret) > 1 else std_ret
    sortino = float(mean_ret / downside_std) if downside_std > 0 else 0.0

    calmar = float(cagr / max_dd) if max_dd > 0 else 0.0

    win_trades = [t for t in trades if t > 0]
    loss_trades = [t for t in trades if t <= 0]
    win_rate = (len(win_trades) / len(trades) * 100.0) if trades else 0.0
    gross_profits = sum(win_trades)
    gross_losses = abs(sum(loss_trades))
    profit_factor = (gross_profits / gross_losses) if gross_losses > 0 else 2.5

    summary = BacktestSummary(
        total_return_pct=round(total_strat_ret, 2),
        cagr_pct=round(cagr, 2),
        benchmark_return_pct=round(total_bench_ret, 2),
        alpha_pct=round(alpha, 2),
        annualized_sharpe=round(sharpe, 2),
        annualized_sortino=round(sortino, 2),
        max_drawdown_pct=round(max_dd, 2),
        calmar_ratio=round(calmar, 2),
        win_rate_pct=round(win_rate, 1),
        profit_factor=round(profit_factor, 2),
        total_trades=len(trades),
    )
    summary.n_folds = len(fold_sharpes)
    summary.oos_sharpe_mean = round(float(np.mean(fold_sharpes)), 3)
    summary.oos_sharpe_std = round(float(np.std(fold_sharpes)), 3)
    summary.fold_returns = fold_return_pcts
    summary.factors_used = sorted(fold_factors_used)
    return summary, combined
