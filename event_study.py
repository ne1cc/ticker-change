"""Cumulative Abnormal Return (CAR) Event Study Analytics.

Calculates abnormal and cumulative abnormal returns around corporate events
(splits, mergers, ticker changes, SEC 8-K filings) using CAPM / Market Model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class EventStudyResult:
    ticker: str
    event_date: str
    event_type: str
    estimation_window: Tuple[int, int]  # e.g., (-120, -21)
    event_window: Tuple[int, int]  # e.g., (-10, 30)
    alpha: float
    beta: float
    r_squared: float
    residual_variance: float
    car: float  # Cumulative Abnormal Return over event window
    car_t_stat: float
    car_p_value: float
    is_significant_95: bool
    daily_abnormal_returns: List[Dict[str, Any]] = field(default_factory=list)


def run_event_study(
    stock_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    event_date: str,
    event_type: str = "CORPORATE_ACTION",
    ticker: str = "UNKNOWN",
    estimation_window: Tuple[int, int] = (-120, -21),
    event_window: Tuple[int, int] = (-10, 30),
) -> Optional[EventStudyResult]:
    """Execute a single-firm Event Study on daily return series using Market Model.

    stock_df: DataFrame with DatetimeIndex and 'close' (or 'Close') column
    benchmark_df: Market proxy DataFrame (e.g. SPY) with DatetimeIndex and 'close'
    event_date: 'YYYY-MM-DD'
    estimation_window: Trading days before event (start, end) e.g. (-120, -21)
    event_window: Relative trading days around event (start, end) e.g. (-10, 30)
    """
    # Normalize column names
    s_col = "close" if "close" in stock_df.columns else "Close"
    b_col = "close" if "close" in benchmark_df.columns else "Close"

    # Merge aligned return series
    s_ret = stock_df[s_col].pct_change().dropna().rename("stock_ret")
    b_ret = benchmark_df[b_col].pct_change().dropna().rename("bench_ret")

    df = pd.concat([s_ret, b_ret], axis=1).dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)

    target_dt = pd.to_datetime(event_date)
    # Find closest trading day on or before target_dt
    available_dates = df.index
    if target_dt not in available_dates:
        valid_dates = available_dates[available_dates <= target_dt]
        if valid_dates.empty:
            return None
        target_dt = valid_dates[-1]

    event_idx = df.index.get_loc(target_dt)

    est_start_idx = event_idx + estimation_window[0]
    est_end_idx = event_idx + estimation_window[1]

    ev_start_idx = event_idx + event_window[0]
    ev_end_idx = event_idx + event_window[1]

    # Validate window boundaries
    if est_start_idx < 0 or ev_end_idx >= len(df):
        return None
    if est_end_idx >= ev_start_idx:
        return None

    est_data = df.iloc[est_start_idx : est_end_idx + 1]
    if len(est_data) < 30:
        return None

    # Fit Market Model: R_it = alpha_i + beta_i * R_mt + eps_it
    x = est_data["bench_ret"].values
    y = est_data["stock_ret"].values

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    cov_xy = np.sum((x - x_mean) * (y - y_mean))
    var_x = np.sum((x - x_mean) ** 2)

    if var_x == 0:
        beta = 1.0
        alpha = 0.0
    else:
        beta = float(cov_xy / var_x)
        alpha = float(y_mean - beta * x_mean)

    est_pred = alpha + beta * x
    est_resid = y - est_pred
    n_est = len(est_data)
    sigma_eps_sq = float(np.sum(est_resid ** 2) / max(n_est - 2, 1))

    corr = np.corrcoef(x, y)[0, 1] if var_x > 0 else 0.0
    r_squared = float(corr ** 2) if not np.isnan(corr) else 0.0

    # Compute Abnormal Returns over event window
    ev_data = df.iloc[ev_start_idx : ev_end_idx + 1]
    daily_abnormal = []
    cumulative_ar = 0.0
    var_car_sum = 0.0

    for i, (dt, row) in enumerate(ev_data.iterrows()):
        rel_day = event_window[0] + i
        actual_ret = float(row["stock_ret"])
        bench_ret = float(row["bench_ret"])
        expected_ret = alpha + beta * bench_ret
        ar = actual_ret - expected_ret
        cumulative_ar += ar

        # Variance adjustment per Salinger (1992)
        var_ar = sigma_eps_sq * (1.0 + 1.0 / n_est + ((bench_ret - x_mean) ** 2) / max(var_x, 1e-8))
        var_car_sum += var_ar

        daily_abnormal.append({
            "relative_day": int(rel_day),
            "date": dt.strftime("%Y-%m-%d"),
            "actual_return": round(actual_ret, 6),
            "expected_return": round(expected_ret, 6),
            "abnormal_return": round(ar, 6),
            "cumulative_abnormal_return": round(cumulative_ar, 6),
        })

    car_std = math.sqrt(max(var_car_sum, 1e-12))
    car_t_stat = float(cumulative_ar / car_std)

    # Two-tailed p-value approximation via standard normal (for large N)
    from scipy import stats
    try:
        p_val = float(2 * (1 - stats.norm.cdf(abs(car_t_stat))))
    except Exception:
        # Fallback approximation if scipy is not present
        p_val = float(math.erfc(abs(car_t_stat) / math.sqrt(2)))

    return EventStudyResult(
        ticker=ticker.upper(),
        event_date=target_dt.strftime("%Y-%m-%d"),
        event_type=event_type,
        estimation_window=estimation_window,
        event_window=event_window,
        alpha=round(alpha, 6),
        beta=round(beta, 4),
        r_squared=round(r_squared, 4),
        residual_variance=round(sigma_eps_sq, 8),
        car=round(cumulative_ar, 6),
        car_t_stat=round(car_t_stat, 4),
        car_p_value=round(p_val, 5),
        is_significant_95=p_val < 0.05,
        daily_abnormal_returns=daily_abnormal,
    )


@dataclass
class GexLevelEventStudyResult:
    ticker: str
    level_type: str  # "call_wall", "put_wall", "gamma_flip"
    level_price: float
    total_touches: int
    reversals: int
    reversal_rate: float  # e.g. 71.4%
    baseline_rate: float  # Monte Carlo random level baseline, e.g. 50.2%
    z_score: float
    p_value: float
    is_significant: bool  # p < 0.05
    avg_reversal_return_pct: float
    summary: str


def run_gex_touch_and_reversal_study(
    daily_df: pd.DataFrame,
    level_price: float,
    level_type: Literal["call_wall", "put_wall", "gamma_flip"] = "put_wall",
    ticker: str = "UNKNOWN",
    tolerance_pct: float = 0.01,
) -> Optional[GexLevelEventStudyResult]:
    """Empirical event study testing whether price respects a structural GEX level.

    Measures touch-and-reversal frequency at the structural level (Put Wall support,
    Call Wall resistance, or Gamma Flip boundary) compared against a random-level null hypothesis.
    """
    if daily_df is None or daily_df.empty or len(daily_df) < 30 or level_price <= 0:
        return None

    c_col = "close" if "close" in daily_df.columns else "Close"
    h_col = "high" if "high" in daily_df.columns else "High"
    l_col = "low" if "low" in daily_df.columns else "Low"

    closes = daily_df[c_col].to_numpy(dtype=np.float64)
    highs = daily_df[h_col].to_numpy(dtype=np.float64) if h_col in daily_df.columns else closes
    lows = daily_df[l_col].to_numpy(dtype=np.float64) if l_col in daily_df.columns else closes

    n = len(closes)
    touches = 0
    reversals = 0
    reversal_returns = []

    for i in range(1, n - 1):
        prev_c = closes[i - 1]
        curr_h = highs[i]
        curr_l = lows[i]
        curr_c = closes[i]
        next_c = closes[i + 1]

        if level_type == "call_wall":
            # Approaching resistance from below
            touched = (curr_h >= level_price * (1.0 - tolerance_pct)) and (prev_c < level_price)
            if touched:
                touches += 1
                # Reversal means price was rejected downward
                ret = (next_c - curr_c) / curr_c
                if ret < 0:
                    reversals += 1
                reversal_returns.append(ret)

        elif level_type == "put_wall":
            # Approaching support from above
            touched = (curr_l <= level_price * (1.0 + tolerance_pct)) and (prev_c > level_price)
            if touched:
                touches += 1
                # Reversal means price bounced upward
                ret = (next_c - curr_c) / curr_c
                if ret > 0:
                    reversals += 1
                reversal_returns.append(ret)

        elif level_type == "gamma_flip":
            # Crossed the gamma flip boundary
            crossed = (prev_c - level_price) * (curr_c - level_price) < 0
            if crossed:
                touches += 1
                # If crossed below into negative gamma, check if volatility expanded
                if i + 5 < n and i >= 5:
                    pre_vol = float(np.std(closes[i - 5:i]))
                    post_vol = float(np.std(closes[i:i + 5]))
                    if post_vol > pre_vol:
                        reversals += 1
                    reversal_returns.append((post_vol - pre_vol) / max(pre_vol, 1e-4))

    # Monte Carlo random baseline (null hypothesis)
    np.random.seed(42)
    p_min, p_max = float(np.min(closes)), float(np.max(closes))
    random_rates = []
    for _ in range(25):
        rnd_level = np.random.uniform(p_min * 1.05, p_max * 0.95)
        rnd_touches = 0
        rnd_revs = 0
        for i in range(1, n - 1):
            if level_type == "call_wall":
                if highs[i] >= rnd_level * (1.0 - tolerance_pct) and closes[i - 1] < rnd_level:
                    rnd_touches += 1
                    if closes[i + 1] < closes[i]:
                        rnd_revs += 1
            else:
                if lows[i] <= rnd_level * (1.0 + tolerance_pct) and closes[i - 1] > rnd_level:
                    rnd_touches += 1
                    if closes[i + 1] > closes[i]:
                        rnd_revs += 1
        if rnd_touches >= 3:
            random_rates.append(rnd_revs / rnd_touches)

    baseline = float(np.mean(random_rates)) if random_rates else 0.50
    baseline_std = float(np.std(random_rates)) if len(random_rates) > 1 else 0.08

    reversal_rate = round((reversals / touches * 100.0), 1) if touches > 0 else 0.0
    baseline_rate = round(baseline * 100.0, 1)

    # Standard one-tailed z-test
    if touches >= 3 and baseline_std > 0:
        z = (reversal_rate - baseline_rate) / (baseline_std * 100.0)
        from scipy import stats
        try:
            p_val = round(float(1.0 - stats.norm.cdf(z)), 4)
        except Exception:
            p_val = 0.05
    else:
        z = 0.0
        p_val = 0.50

    is_sig = bool(p_val < 0.05 and reversal_rate > baseline_rate)
    avg_ret = round(float(np.mean(reversal_returns) * 100.0), 2) if reversal_returns else 0.0

    role_desc = "Support bounce" if level_type == "put_wall" else (
        "Resistance rejection" if level_type == "call_wall" else "Volatility expansion"
    )
    if touches > 0:
        summary = (
            f"{role_desc} win rate of {reversal_rate}% across {touches} touches "
            f"vs {baseline_rate}% random baseline (p={p_val}, {'Statistically Significant' if is_sig else 'Not Significant'})."
        )
    else:
        summary = f"No historical touches recorded for {role_desc.lower()} test within ±{tolerance_pct*100:.1f}% tolerance."

    return GexLevelEventStudyResult(
        ticker=ticker.upper(),
        level_type=level_type,
        level_price=round(level_price, 2),
        total_touches=touches,
        reversals=reversals,
        reversal_rate=reversal_rate,
        baseline_rate=baseline_rate,
        z_score=round(z, 2),
        p_value=p_val,
        is_significant=is_sig,
        avg_reversal_return_pct=avg_ret,
        summary=summary,
    )

