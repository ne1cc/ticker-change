"""Quantitative signal evaluation engine for Signal Lab.

Part of Epic 5 (issues #133, #134, #135):
- Rank IC engine with Newey-West HAC t-statistics and decay curve (1d, 5d, 21d)
- Decile portfolio sorts, long-short spreads, monotonicity checks
- Portfolio turnover, transaction costs (bps), cost-adjusted Sharpe
- Deflated Sharpe Ratio (DSR) using persisted trial count
"""
from __future__ import annotations

import math
from typing import Sequence
import numpy as np
import pandas as pd
from scipy import stats

import factors

# Euler-Mascheroni constant for expected maximum of Gaussian variables
_EULER_MASCHERONI = 0.5772156649015328606


def _nw_lags(n: int) -> int:
    """Standard rule-of-thumb Newey-West Bartlett kernel bandwidth."""
    return max(1, int(4 * (n / 100.0) ** (2.0 / 9.0)))


def newey_west_mean_se(series: pd.Series | np.ndarray, lags: int | None = None) -> float:
    """HAC standard error of the sample mean using Bartlett kernel."""
    x = np.asarray(series, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return 0.0

    if lags is None:
        lags = _nw_lags(n)

    lags = min(lags, n - 1)
    u = x - np.mean(x)
    # Variance of mean: gamma_0 + 2 * sum_{l=1}^L (1 - l/(L+1)) * gamma_l, divided by n
    gamma_0 = float(np.sum(u ** 2))
    gamma_sum = 0.0
    for l in range(1, lags + 1):
        w = 1.0 - (l / (lags + 1.0))
        g = float(np.sum(u[l:] * u[:-l]))
        gamma_sum += 2.0 * w * g

    var_sum = max(0.0, (gamma_0 + gamma_sum) / (n * n))
    return math.sqrt(var_sum)


def compute_deflated_sharpe_ratio(
    returns: pd.Series | np.ndarray,
    n_trials: int = 1,
) -> float:
    """Bailey & Lopez de Prado (2014) closed-form Deflated Sharpe Ratio.

    Evaluates the probability that the population Sharpe ratio exceeds zero,
    correcting for non-normality (skew, kurtosis) and multiple-testing
    overfitting across `n_trials` independent or correlated strategy attempts.
    """
    s = pd.Series(returns).dropna()
    n = len(s)
    if n < 4:
        return 0.0
    std = float(s.std(ddof=1))
    if std <= 1e-12:
        return 0.0

    sr = float(s.mean() / std)
    skew = float(s.skew())
    kurt = float(s.kurtosis()) + 3.0  # Pearson raw kurtosis

    var_term = (1.0 - skew * sr + ((kurt - 1.0) / 4.0) * (sr ** 2)) / (n - 1.0)
    sr_std = math.sqrt(max(var_term, 1e-12))

    if n_trials <= 1:
        expected_max_sr = 0.0
    else:
        # Bailey & Lopez de Prado (2014) eq. 8: Expected maximum Sharpe under null
        p1 = 1.0 - (1.0 / n_trials)
        p2 = 1.0 - (1.0 / (n_trials * math.e))
        # Quantiles of standard normal distribution
        z1 = float(stats.norm.ppf(p1))
        z2 = float(stats.norm.ppf(p2))
        expected_max_sr = ((1.0 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2) * sr_std

    z = (sr - expected_max_sr) / sr_std
    dsr = float(stats.norm.cdf(z))
    return round(dsr, 4)


def compute_rank_ic(
    panel: pd.DataFrame,
    signal_col: str,
    return_col: str = "forward_ret_1d",
    min_cross_section: int = 10,
) -> dict:
    """Compute time-series of cross-sectional Spearman rank ICs and HAC t-stat."""
    if panel.empty or signal_col not in panel.columns or return_col not in panel.columns:
        return {
            "mean_ic": 0.0,
            "std_ic": 0.0,
            "icir": 0.0,
            "annualized_icir": 0.0,
            "t_stat": 0.0,
            "p_value": 1.0,
            "n_dates": 0,
            "ic_series": pd.Series(dtype=float),
        }

    # Group by date index
    dates = panel.index.unique()
    daily_ics: dict[pd.Timestamp, float] = {}

    for dt in dates:
        sub = panel.loc[[dt]] if isinstance(dt, pd.Timestamp) else panel.loc[dt]
        if isinstance(sub, pd.Series):
            continue
        # Filter valid liquid observations
        valid_mask = ~(sub[signal_col].isna() | sub[return_col].isna())
        if "is_liquid" in sub.columns:
            valid_mask = valid_mask & (sub["is_liquid"] == True)

        sub_valid = sub[valid_mask]
        if len(sub_valid) < min_cross_section:
            continue

        sig_vals = sub_valid[signal_col].values
        ret_vals = sub_valid[return_col].values

        # If constant signal, correlation is 0
        if np.all(sig_vals == sig_vals[0]) or np.all(ret_vals == ret_vals[0]):
            daily_ics[dt] = 0.0
            continue

        corr, _ = stats.spearmanr(sig_vals, ret_vals)
        if not np.isnan(corr):
            daily_ics[dt] = float(corr)

    if not daily_ics:
        return {
            "mean_ic": 0.0,
            "std_ic": 0.0,
            "icir": 0.0,
            "annualized_icir": 0.0,
            "t_stat": 0.0,
            "p_value": 1.0,
            "n_dates": 0,
            "ic_series": pd.Series(dtype=float),
        }

    ic_s = pd.Series(daily_ics).sort_index()
    n = len(ic_s)
    mean_ic = float(ic_s.mean())
    std_ic = float(ic_s.std(ddof=1)) if n > 1 else 0.0
    icir = (mean_ic / std_ic) if std_ic > 1e-12 else 0.0
    ann_icir = icir * math.sqrt(252.0)

    se = newey_west_mean_se(ic_s)
    t_stat = (mean_ic / se) if se > 1e-12 else 0.0
    p_val = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=max(1, n - 1))))

    return {
        "mean_ic": round(mean_ic, 4),
        "std_ic": round(std_ic, 4),
        "icir": round(icir, 4),
        "annualized_icir": round(ann_icir, 4),
        "t_stat": round(t_stat, 2),
        "p_value": round(p_val, 4),
        "n_dates": n,
        "ic_series": ic_s,
    }


def compute_ic_decay(
    panel: pd.DataFrame,
    signal_col: str,
    horizons: Sequence[int] = (1, 5, 21),
    min_cross_section: int = 10,
) -> dict[int, dict]:
    """Compute rank IC and decay metrics across 1d, 5d, 21d horizons."""
    decay: dict[int, dict] = {}
    for h in horizons:
        col = f"forward_ret_{h}d"
        res = compute_rank_ic(
            panel, signal_col=signal_col, return_col=col, min_cross_section=min_cross_section
        )
        decay[h] = {
            "mean_ic": res["mean_ic"],
            "icir": res["icir"],
            "t_stat": res["t_stat"],
            "p_value": res["p_value"],
            "n_dates": res["n_dates"],
        }
    return decay


def compute_decile_spreads(
    panel: pd.DataFrame,
    signal_col: str,
    return_col: str = "forward_ret_1d",
    n_deciles: int = 10,
    min_cross_section: int = 20,
) -> dict:
    """Perform cross-sectional decile portfolio sorts and compute long-short spread."""
    if panel.empty or signal_col not in panel.columns or return_col not in panel.columns:
        return {
            "decile_means": [0.0] * n_deciles,
            "is_strictly_monotonic": False,
            "monotonicity_spearman": 0.0,
            "spread_mean_daily": 0.0,
            "spread_annualised_return": 0.0,
            "spread_annualised_vol": 0.0,
            "gross_sharpe": 0.0,
            "spread_t_stat": 0.0,
            "spread_p_value": 1.0,
            "n_dates": 0,
            "spread_series": pd.Series(dtype=float),
            "weights_history": {},
        }

    dates = panel.index.unique()
    daily_deciles: dict[pd.Timestamp, dict[int, float]] = {}
    daily_spreads: dict[pd.Timestamp, float] = {}
    weights_history: dict[pd.Timestamp, dict[str, float]] = {}

    for dt in dates:
        sub = panel.loc[[dt]] if isinstance(dt, pd.Timestamp) else panel.loc[dt]
        if isinstance(sub, pd.Series):
            continue

        valid_mask = ~(sub[signal_col].isna() | sub[return_col].isna())
        if "is_liquid" in sub.columns:
            valid_mask = valid_mask & (sub["is_liquid"] == True)

        sub_valid = sub[valid_mask]
        if len(sub_valid) < min_cross_section:
            continue

        sigs = sub_valid[signal_col]
        if sigs.nunique() < n_deciles:
            continue

        try:
            ranks = sigs.rank(method="first")
            buckets = pd.qcut(ranks, q=n_deciles, labels=False) + 1
        except Exception:
            continue

        sub_assigned = sub_valid.copy()
        sub_assigned["decile"] = buckets

        means = sub_assigned.groupby("decile")[return_col].mean().to_dict()
        daily_deciles[dt] = means

        d_low = means.get(1, 0.0)
        d_high = means.get(n_deciles, 0.0)
        daily_spreads[dt] = d_high - d_low

        d_top_tickers = sub_assigned[sub_assigned["decile"] == n_deciles]["ticker"].tolist()
        d_bot_tickers = sub_assigned[sub_assigned["decile"] == 1]["ticker"].tolist()

        w_dict: dict[str, float] = {}
        if d_top_tickers:
            w_top = 1.0 / len(d_top_tickers)
            for t in d_top_tickers:
                w_dict[t] = w_top
        if d_bot_tickers:
            w_bot = -1.0 / len(d_bot_tickers)
            for t in d_bot_tickers:
                w_dict[t] = w_bot

        weights_history[dt] = w_dict

    if not daily_spreads:
        return {
            "decile_means": [0.0] * n_deciles,
            "is_strictly_monotonic": False,
            "monotonicity_spearman": 0.0,
            "spread_mean_daily": 0.0,
            "spread_annualised_return": 0.0,
            "spread_annualised_vol": 0.0,
            "gross_sharpe": 0.0,
            "spread_t_stat": 0.0,
            "spread_p_value": 1.0,
            "n_dates": 0,
            "spread_series": pd.Series(dtype=float),
            "weights_history": {},
        }

    decile_matrix = pd.DataFrame.from_dict(daily_deciles, orient="index")
    decile_means = [float(decile_matrix[d].mean()) if d in decile_matrix.columns else 0.0
                    for d in range(1, n_deciles + 1)]

    is_strictly_monotonic = all(
        decile_means[i] < decile_means[i + 1] for i in range(len(decile_means) - 1)
    )
    spearman_mono, _ = stats.spearmanr(range(1, n_deciles + 1), decile_means)

    spread_s = pd.Series(daily_spreads).sort_index()
    n = len(spread_s)
    mean_spread = float(spread_s.mean())
    ann_return = mean_spread * 252.0
    daily_vol = float(spread_s.std(ddof=1)) if n > 1 else 0.0
    ann_vol = daily_vol * math.sqrt(252.0)

    gross_sharpe = (ann_return / ann_vol) if ann_vol > 1e-12 else 0.0

    se_spread = newey_west_mean_se(spread_s)
    t_stat = (mean_spread / se_spread) if se_spread > 1e-12 else 0.0
    p_val = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=max(1, n - 1))))

    return {
        "decile_means": [round(m, 6) for m in decile_means],
        "is_strictly_monotonic": is_strictly_monotonic,
        "monotonicity_spearman": round(float(spearman_mono), 3),
        "spread_mean_daily": round(mean_spread, 6),
        "spread_annualised_return": round(ann_return, 4),
        "spread_annualised_vol": round(ann_vol, 4),
        "gross_sharpe": round(gross_sharpe, 2),
        "spread_t_stat": round(t_stat, 2),
        "spread_p_value": round(p_val, 4),
        "n_dates": n,
        "spread_series": spread_s,
        "weights_history": weights_history,
    }


def compute_turnover_and_costs(
    weights_history: dict[pd.Timestamp, dict[str, float]],
    spread_series: pd.Series,
    cost_bps: float = 10.0,
    n_trials: int = 1,
) -> dict:
    """Compute portfolio turnover, transaction cost drag, cost-adjusted Sharpe, and DSR."""
    if not weights_history or spread_series.empty:
        return {
            "mean_turnover": 0.0,
            "cost_bps": cost_bps,
            "net_annualised_return": 0.0,
            "cost_adjusted_sharpe": 0.0,
            "deflated_sharpe": 0.0,
            "n_trials": n_trials,
            "net_spread_series": pd.Series(dtype=float),
        }

    sorted_dates = sorted(weights_history.keys())
    turnovers: dict[pd.Timestamp, float] = {}
    prev_w: dict[str, float] = {}

    for dt in sorted_dates:
        curr_w = weights_history[dt]
        all_tickers = set(curr_w.keys()).union(prev_w.keys())
        w_delta_sum = sum(abs(curr_w.get(t, 0.0) - prev_w.get(t, 0.0)) for t in all_tickers)
        turnovers[dt] = w_delta_sum / 2.0
        prev_w = curr_w

    if isinstance(spread_series.index, pd.DatetimeIndex) and any(d in turnovers for d in spread_series.index):
        turnover_s = pd.Series(turnovers).reindex(spread_series.index).fillna(0.0)
    else:
        # Align by chronological position
        t_vals = [turnovers[d] for d in sorted_dates]
        if len(t_vals) >= len(spread_series):
            turnover_s = pd.Series(t_vals[:len(spread_series)], index=spread_series.index)
        else:
            turnover_s = pd.Series(t_vals + [0.0] * (len(spread_series) - len(t_vals)), index=spread_series.index)

    mean_turnover = float(turnover_s.mean())

    cost_series = turnover_s * 2.0 * (cost_bps / 10000.0)
    net_spread = spread_series - cost_series

    n = len(net_spread)
    net_mean = float(net_spread.mean())
    net_ann_return = net_mean * 252.0
    net_vol = float(net_spread.std(ddof=1)) * math.sqrt(252.0) if n > 1 else 0.0

    cost_adjusted_sharpe = (net_ann_return / net_vol) if net_vol > 1e-12 else 0.0

    # Deflated Sharpe ratio accounting for multiple trials
    dsr = compute_deflated_sharpe_ratio(net_spread, n_trials=n_trials)

    return {
        "mean_turnover": round(mean_turnover, 4),
        "cost_bps": cost_bps,
        "net_annualised_return": round(net_ann_return, 4),
        "cost_adjusted_sharpe": round(cost_adjusted_sharpe, 2),
        "deflated_sharpe": round(dsr, 4),
        "n_trials": n_trials,
        "net_spread_series": net_spread,
    }
