"""Credible risk: tail-honest VaR/ES, volatility forecasts, and backtests.

1. Tail honesty (issue #140): historical VaR/ES with tail-observation counts,
   bootstrap confidence intervals, and a low-confidence flag when the quantile
   rests on too few tail points.
2. Volatility forecasts (#141): EWMA (lambda 0.94) and GARCH(1,1) MLE,
   evaluated out-of-sample against realised vol with MSE and QLIKE against a
   trailing-window naive benchmark.
3. VaR backtests (#142): Kupiec proportion-of-failures and Christoffersen
   independence tests on rolling VaR exceedances, with a traffic-light verdict.
4. Horizon scaling (#143): empirical multi-day VaR on overlapping windows
   (sqrt-time shown as a caveat, not used) and dollar scaling for a position.

Conventions match the page: VaR/ES are RETURN quantiles (negative numbers =
losses); vols are daily decimals, annualised for display.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
from scipy import optimize, stats

MIN_TAIL_OBS = 10  # below this a 99% quantile is noise


# --- 1. Tail-honest VaR/ES (issue #140) ----------------------------------------

def tail_honest_var(returns: np.ndarray, levels=(0.95, 0.99),
                    bootstrap: int = 500, seed: int = 0) -> dict:
    """Historical VaR/ES with tail counts and bootstrap CIs per level.

    `low_confidence` is True when fewer than MIN_TAIL_OBS observations lie in
    the tail — the number shown is an order-of-magnitude estimate, not risk
    management. Returns {} for degenerate input."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 100:
        return {}
    rng = np.random.default_rng(seed)
    out = {}
    for level in levels:
        q = float(np.quantile(r, 1.0 - level))
        tail = r[r <= q]
        es = float(tail.mean()) if len(tail) else q
        boots = np.quantile(rng.choice(r, size=(bootstrap, len(r)), replace=True),
                            1.0 - level, axis=1)
        out[f"var_{int(level * 100)}"] = {
            "var": round(q, 4),
            "expected_shortfall": round(es, 4),
            "tail_obs": int(len(tail)),
            "ci95": [round(float(np.quantile(boots, 0.025)), 4),
                     round(float(np.quantile(boots, 0.975)), 4)],
            "low_confidence": len(tail) < MIN_TAIL_OBS,
        }
    return out


# --- 2. Volatility forecasts (issue #141) ---------------------------------------

def ewma_vol_forecast(returns: np.ndarray, lam: float = 0.94) -> float:
    """RiskMetrics EWMA next-day variance from daily returns (decimal)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return float("nan")
    var = float(r[:60].var()) if len(r) >= 60 else float(r.var())
    for x in r:
        var = lam * var + (1.0 - lam) * x * x
    return math.sqrt(var)


def fit_garch11(returns: np.ndarray) -> Optional[dict]:
    """Gaussian GARCH(1,1) MLE via scipy. Returns params, log-likelihood, the
    filtered variance path, and the next-day forecast — None when the optimiser
    fails (declared state, never a silent fallback)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 250:
        return None
    sample_var = float(r.var())
    floor = sample_var * 1e-6

    def filtered(omega, alpha, beta):
        h = np.empty(len(r) + 1)
        h[0] = sample_var
        for t in range(1, len(r) + 1):
            h[t] = omega + alpha * r[t - 1] ** 2 + beta * h[t - 1]
        return np.clip(h, floor, None)

    def nll(params):
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.9999:
            return 1e12
        h = filtered(omega, alpha, beta)
        ll = -0.5 * np.sum(np.log(2 * math.pi) + np.log(h[:-1]) + r ** 2 / h[:-1])
        return -ll

    best, best_val = None, np.inf
    for a0 in (0.05, 0.10):
        for b0 in (0.80, 0.90):
            res = optimize.minimize(
                nll, x0=[sample_var * (1 - a0 - b0), a0, b0], method="Nelder-Mead",
                options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-8})
            if res.fun < best_val:
                best_val, best = float(res.fun), res.x
    if best is None or best_val >= 1e12:
        return None
    omega, alpha, beta = (float(best[0]), float(best[1]), float(best[2]))
    if omega <= 0 or alpha + beta >= 0.9999:
        return None
    h = filtered(omega, alpha, beta)
    return {
        "omega": omega, "alpha": alpha, "beta": beta,
        "log_likelihood": -best_val,
        "filtered_var": h[1:],           # h_t uses info up to t-1
        "next_var": float(h[-1]),
        "persistence": alpha + beta,
    }


def forecast_evaluation(returns: np.ndarray, forecast_var: np.ndarray) -> dict:
    """Out-of-sample forecast quality vs realised squared returns: MSE on
    variance and QLIKE, compared with a trailing-60d naive benchmark."""
    r2 = np.asarray(returns, dtype=float) ** 2
    f = np.asarray(forecast_var, dtype=float)
    mse = float(np.mean((f - r2) ** 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        qlike = np.mean(r2 / f - np.log(f / r2) - 1.0)
    naive = trailing_mean_var(r2, 60)
    valid = np.isfinite(naive)
    if valid.sum() > 20:
        mse_naive = float(np.mean((naive[valid] - r2[valid]) ** 2))
        with np.errstate(divide="ignore", invalid="ignore"):
            ql_naive = float(np.mean(r2[valid] / naive[valid]
                                     - np.log(naive[valid] / r2[valid]) - 1.0))
    else:
        mse_naive, ql_naive = float("nan"), float("nan")
    return {"mse": mse, "qlike": float(qlike),
            "mse_naive": mse_naive, "qlike_naive": ql_naive,
            "beats_naive_qlike": bool(np.isfinite(ql_naive) and qlike < ql_naive)}


def trailing_mean_var(r2: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(r2), np.nan)
    for i in range(window, len(r2)):
        out[i] = r2[i - window:i].mean()
    return out


# --- 3. VaR backtests (issue #142) ----------------------------------------------

def kupiec_pof(exceedances: np.ndarray, level: float) -> dict:
    """Kupiec proportion-of-failures likelihood-ratio test. p < 0.05 rejects
    the VaR's claimed coverage."""
    x = int(np.sum(exceedances))
    n = int(len(exceedances))
    if n == 0:
        return {"lr": None, "p_value": None, "n": 0, "violations": 0}
    p_hat = x / n
    alpha = 1.0 - level
    if 0.0 < p_hat < 1.0:
        lr = -2.0 * math.log(
            ((1 - alpha) ** (n - x) * alpha ** x)
            / ((1 - p_hat) ** (n - x) * p_hat ** x))
    elif p_hat == alpha or n == 0:
        lr = 0.0
    else:
        lr = float("inf")
    p_value = float(1.0 - stats.chi2.cdf(lr, 1)) if math.isfinite(lr) else 0.0
    return {"lr": round(lr, 3), "p_value": round(p_value, 4),
            "n": n, "violations": x}


def christoffersen_ind(exceedances: np.ndarray) -> dict:
    """Christoffersen independence test: first-order Markov vs independent
    exceedances. Clustered violations (vol clustering the VaR missed) reject."""
    e = np.asarray(exceedances, dtype=int)
    if len(e) < 2 or e.sum() == 0:
        return {"lr": None, "p_value": None}
    n00 = int(np.sum((e[:-1] == 0) & (e[1:] == 0)))
    n01 = int(np.sum((e[:-1] == 0) & (e[1:] == 1)))
    n10 = int(np.sum((e[:-1] == 1) & (e[1:] == 0)))
    n11 = int(np.sum((e[:-1] == 1) & (e[1:] == 1)))
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    def _log(p, k, n):
        if n == 0:
            return 0.0
        p = min(max(p, 1e-12), 1 - 1e-12)
        return k * math.log(p) + (n - k) * math.log(1 - p)
    ll_ind = _log(pi, n01, n00 + n01) + _log(pi, n11, n10 + n11)
    ll_markov = _log(pi01, n01, n00 + n01) + _log(pi11, n11, n10 + n11)
    lr = -2.0 * (ll_ind - ll_markov)
    lr = max(lr, 0.0)
    p_value = float(1.0 - stats.chi2.cdf(lr, 1))
    return {"lr": round(lr, 3), "p_value": round(p_value, 4)}


def var_backtest(returns: np.ndarray, level: float, window: int = 250,
                 step: int = 1) -> dict:
    """Roll out a rolling-window historical VaR over the series and backtest
    its exceedances. Returns Kupiec + Christoffersen + a traffic light."""
    r = np.asarray(returns, dtype=float)
    if len(r) < window + 50:
        return {"sufficient": False, "n": len(r)}
    exceed = np.zeros(len(r), dtype=bool)
    for t in range(window, len(r)):
        q = np.quantile(r[t - window:t], 1.0 - level)
        exceed[t] = r[t] < q
    exceed = exceed[window:]
    kup = kupiec_pof(exceed, level)
    ind = christoffersen_ind(exceed)
    exp_rate = round((1.0 - level) * 100, 2)
    rate = round(100.0 * kup["violations"] / kup["n"], 2) if kup.get("n") else None
    if kup["p_value"] is None:
        traffic = "grey"
    elif kup["p_value"] >= 0.05 and (ind["p_value"] is None or ind["p_value"] >= 0.05):
        traffic = "green"
    elif kup["p_value"] >= 0.05 or (ind["p_value"] is not None and ind["p_value"] >= 0.05):
        traffic = "amber"
    else:
        traffic = "red"
    return {"sufficient": True, "level": level, "expected_violation_rate_pct": exp_rate,
            "violation_rate_pct": rate, "kupiec": kup, "christoffersen": ind,
            "traffic": traffic}


# --- 4. Horizon scaling (issue #143) ---------------------------------------------

def horizon_var(returns: np.ndarray, levels=(0.95, 0.99),
                horizons=(1, 10, 21)) -> dict:
    """Empirical multi-day VaR from OVERLAPPING n-day sums. sqrt-time scaling
    is a crude approximation — the overlap method is the estimate, the caveat
    ships with it."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 252:
        return {}
    out = {}
    for h in horizons:
        agg = np.convolve(r, np.ones(h), mode="valid")  # overlapping sums
        if len(agg) < 100:
            continue
        for level in levels:
            q = float(np.quantile(agg, 1.0 - level))
            tail = agg[agg <= q]
            out[f"h{h}_var_{int(level * 100)}"] = {
                "var": round(q, 4), "expected_shortfall": round(float(tail.mean()), 4) if len(tail) else q,
                "windows": int(len(agg)), "overlap": True,
            }
    return out


def scale_to_position(var_ret: float, es_ret: float, position_usd: float) -> dict:
    """Convert return-denominated VaR/ES (negative = loss) to dollars."""
    if position_usd is None or position_usd <= 0:
        return {}
    return {"position_usd": round(float(position_usd), 2),
            "var_usd": round(float(var_ret) / 100.0 * position_usd, 2),
            "es_usd": round(float(es_ret) / 100.0 * position_usd, 2)}


# --- 5. Liquidity-adjusted exit (issue #145) -------------------------------------

def liquidity_adjusted_exit(stock_df: pd.DataFrame,
                            position_usd: float | None = None,
                            spread_pct: float | None = None) -> dict:
    """Compute 20d/60d ADV, days-to-exit at 1% and 5% participation, and spread exit cost.

    Flags an illiquid warning when days to exit at 5% exceeds 5 days or 20d dollar ADV < M.
    """
    if stock_df is None or getattr(stock_df, "empty", True) or len(stock_df) < 5:
        return {}
    close = stock_df["close"] if "close" in stock_df.columns else stock_df["Close"]
    vol = stock_df["volume"] if "volume" in stock_df.columns else stock_df["Volume"]
    dollar_vol = close * vol

    adv_20_shares = float(vol.iloc[-20:].mean()) if len(vol) >= 20 else float(vol.mean())
    adv_60_shares = float(vol.iloc[-60:].mean()) if len(vol) >= 60 else float(vol.mean())
    adv_20_usd = float(dollar_vol.iloc[-20:].mean()) if len(dollar_vol) >= 20 else float(dollar_vol.mean())
    adv_60_usd = float(dollar_vol.iloc[-60:].mean()) if len(dollar_vol) >= 60 else float(dollar_vol.mean())

    pos_usd = float(position_usd) if position_usd is not None and position_usd > 0 else 100_000.0
    has_custom_position = position_usd is not None and position_usd > 0

    d_exit_1pct = round(pos_usd / (adv_20_usd * 0.01), 2) if adv_20_usd > 0 else None
    d_exit_5pct = round(pos_usd / (adv_20_usd * 0.05), 2) if adv_20_usd > 0 else None

    # Spread cost (half-spread for one-way liquidation)
    spread_cost_usd = round(pos_usd * ((spread_pct or 0.0) / 2.0 / 100.0), 2) if spread_pct is not None else None

    # Amihud illiquidity over trailing 60d: 1e6 * mean(|r| / dollar_vol)
    rets = close.pct_change().dropna()
    valid_dvol = dollar_vol.loc[rets.index]
    pos_mask = valid_dvol > 0
    if pos_mask.sum() >= 10:
        amihud = float((rets[pos_mask].abs() / valid_dvol[pos_mask]).iloc[-60:].mean() * 1e6)
    else:
        amihud = None

    is_illiquid = bool((d_exit_5pct is not None and d_exit_5pct > 5.0) or (adv_20_usd < 1_000_000.0))
    warning = None
    if is_illiquid:
        if d_exit_5pct is not None and d_exit_5pct > 5.0:
            warning = f"High liquidation risk: exiting ${pos_usd:,.0f} at 5% daily participation takes {d_exit_5pct:.1f} trading days."
        else:
            warning = f"Thin trading liquidity: 20-day dollar ADV is ${adv_20_usd:,.0f} (< $1M)."

    return {
        "adv_20_shares": round(adv_20_shares, 0),
        "adv_60_shares": round(adv_60_shares, 0),
        "adv_20_usd": round(adv_20_usd, 2),
        "adv_60_usd": round(adv_60_usd, 2),
        "position_usd": pos_usd,
        "position_custom": has_custom_position,
        "days_to_exit_1pct": d_exit_1pct,
        "days_to_exit_5pct": d_exit_5pct,
        "spread_pct": round(spread_pct, 4) if spread_pct is not None else None,
        "spread_cost_usd": spread_cost_usd,
        "amihud_illiquidity": round(amihud, 4) if amihud is not None else None,
        "is_illiquid": is_illiquid,
        "warning": warning,
    }


# --- 6. Stress scenarios (issue #144) --------------------------------------------

STRESS_SCENARIOS_CONFIG = [
    {
        "id": "covid_crash_2020",
        "name": "COVID-19 Crash",
        "start": "2020-02-19",
        "trough": "2020-03-23",
        "end": "2020-08-18",
        "description": "Global pandemic outbreak; SPY -33.9% in 23 trading sessions.",
        "benchmark_drop_pct": -33.9,
        "factor_shocks": {"Mkt-RF": -0.34, "SMB": -0.05, "HML": -0.12, "RMW": 0.02, "CMA": 0.01, "Mom": 0.08},
    },
    {
        "id": "rate_shock_2022",
        "name": "2022 Fed Rate Shock",
        "start": "2022-01-03",
        "trough": "2022-10-12",
        "end": "2022-12-30",
        "description": "Fastest Fed tightening in 40y; duration and growth multiple compression (SPY -25.4%).",
        "benchmark_drop_pct": -25.4,
        "factor_shocks": {"Mkt-RF": -0.25, "SMB": -0.04, "HML": 0.15, "RMW": 0.08, "CMA": 0.12, "Mom": -0.14},
    },
    {
        "id": "volmageddon_2018",
        "name": "Q4 2018 Fed Tightening",
        "start": "2018-09-20",
        "trough": "2018-12-24",
        "end": "2019-04-23",
        "description": "Powell 'long way from neutral'; growth scare and vol spike (SPY -19.8%).",
        "benchmark_drop_pct": -19.8,
        "factor_shocks": {"Mkt-RF": -0.19, "SMB": -0.06, "HML": -0.03, "RMW": 0.01, "CMA": 0.01, "Mom": 0.02},
    },
    {
        "id": "gfc_2008",
        "name": "2008 Lehman / GFC",
        "start": "2008-09-02",
        "trough": "2009-03-09",
        "end": "2009-06-01",
        "description": "Lehman Brothers collapse and systemic bank solvency crisis (SPY -50.8%).",
        "benchmark_drop_pct": -50.8,
        "factor_shocks": {"Mkt-RF": -0.50, "SMB": -0.08, "HML": -0.10, "RMW": -0.05, "CMA": 0.02, "Mom": 0.15},
    },
]


def stress_scenarios(stock_df: pd.DataFrame,
                     factor_loadings: dict | None = None) -> list[dict]:
    """Replay ticker through named historical stress windows.

    When history covers the window, computes exact peak-to-trough drop and recovery.
    When history is missing (e.g. IPO after event), uses factor-implied shocks
    and labels the result explicitly as 'factor-implied (no direct history)'.
    """
    if stock_df is None or getattr(stock_df, "empty", True):
        return []
    close = stock_df["close"] if "close" in stock_df.columns else stock_df["Close"]
    idx = pd.to_datetime(close.index)

    results = []
    for sc in STRESS_SCENARIOS_CONFIG:
        start_ts = pd.to_datetime(sc["start"])
        trough_ts = pd.to_datetime(sc["trough"])
        end_ts = pd.to_datetime(sc["end"])

        covers = (idx.min() <= start_ts + pd.Timedelta(days=7)) and (idx.max() >= trough_ts - pd.Timedelta(days=7))

        if covers:
            sub_window = close[(idx >= start_ts - pd.Timedelta(days=5)) & (idx <= trough_ts + pd.Timedelta(days=5))]
            if len(sub_window) >= 2:
                p_peak = float(sub_window.max())
                p_trough = float(sub_window.min())
                drop_pct = round((p_trough - p_peak) / p_peak * 100.0, 2)

                post_window = close[(idx >= trough_ts) & (idx <= end_ts + pd.Timedelta(days=7))]
                if len(post_window) >= 2 and idx.max() >= end_ts - pd.Timedelta(days=7):
                    p_end = float(post_window.iloc[-1])
                    rec_pct = round((p_end - p_trough) / p_trough * 100.0, 2)
                else:
                    rec_pct = None

                results.append({
                    "id": sc["id"],
                    "name": sc["name"],
                    "dates": f"{sc['start']} to {sc['trough']}",
                    "description": sc["description"],
                    "drop_pct": drop_pct,
                    "recovery_pct": rec_pct,
                    "benchmark_drop_pct": sc["benchmark_drop_pct"],
                    "method": "historical replay",
                })
                continue

        if factor_loadings:
            shocks = sc["factor_shocks"]
            implied_ret = 0.0
            matched = 0
            for f_name, beta_val in factor_loadings.items():
                if f_name in shocks and beta_val is not None and np.isfinite(beta_val):
                    implied_ret += float(beta_val) * float(shocks[f_name])
                    matched += 1
            if matched >= 1:
                results.append({
                    "id": sc["id"],
                    "name": sc["name"],
                    "dates": f"{sc['start']} to {sc['trough']}",
                    "description": sc["description"],
                    "drop_pct": round(implied_ret * 100.0, 2),
                    "recovery_pct": None,
                    "benchmark_drop_pct": sc["benchmark_drop_pct"],
                    "method": "factor-implied (no direct history)",
                })
                continue

        results.append({
            "id": sc["id"],
            "name": sc["name"],
            "dates": f"{sc['start']} to {sc['trough']}",
            "description": sc["description"],
            "drop_pct": None,
            "recovery_pct": None,
            "benchmark_drop_pct": sc["benchmark_drop_pct"],
            "method": "insufficient history",
        })

    return results


# --- 7. Rolling Sharpe with Lo (2002) SE & PSR (issue #146) -----------------------

def rolling_sharpe_with_lo_se(returns: pd.Series,
                              rf_daily: float | pd.Series = 0.0,
                              window: int = 252) -> dict:
    """Compute rolling Sharpe on excess returns with Lo (2002) autocorrelation-adjusted
    SE confidence band and Probabilistic Sharpe Ratio (PSR).
    """
    r = pd.Series(returns).dropna()
    if len(r) < window:
        return {"dates": [], "sharpe": [], "se": [], "ci_upper": [], "ci_lower": [],
                "latest_sharpe": None, "latest_se": None, "psr": None,
                "autocorrelation_rho1": None, "rf_source": "none"}

    if isinstance(rf_daily, pd.Series):
        rf_aligned = rf_daily.reindex(r.index).fillna(0.0)
        excess = r - rf_aligned
        rf_source = "aligned series"
    elif isinstance(rf_daily, (int, float)) and rf_daily != 0.0:
        excess = r - rf_daily
        rf_source = f"{rf_daily * 252 * 100:.2f}% annualized"
    else:
        excess = r
        rf_source = "zero"

    rolling_mean = excess.rolling(window).mean()
    rolling_std = excess.rolling(window).std()

    sharpe_daily = rolling_mean / rolling_std
    sharpe_annual = sharpe_daily * math.sqrt(252)

    n_pts = len(r)
    ses = [np.nan] * (window - 1)
    ci_u = [np.nan] * (window - 1)
    ci_l = [np.nan] * (window - 1)
    excess_vals = excess.values

    for i in range(window - 1, n_pts):
        win_x = excess_vals[i - window + 1 : i + 1]
        sr_ann = sharpe_annual.iloc[i]
        if np.isnan(sr_ann) or np.isinf(sr_ann):
            ses.append(np.nan)
            ci_u.append(np.nan)
            ci_l.append(np.nan)
            continue

        w_std = win_x.std()
        if w_std > 0:
            rho1 = float(np.corrcoef(win_x[1:], win_x[:-1])[0, 1])
            if not np.isfinite(rho1):
                rho1 = 0.0
        else:
            rho1 = 0.0
        rho1 = max(min(rho1, 0.9), -0.9)

        var_sr = (252.0 / window) * (1.0 + 2.0 * rho1) + (1.0 / (2.0 * window)) * (sr_ann ** 2)
        var_sr = max(var_sr, 1e-4)
        se = math.sqrt(var_sr)

        ses.append(round(se, 3))
        ci_u.append(round(sr_ann + 1.96 * se, 3))
        ci_l.append(round(sr_ann - 1.96 * se, 3))

    latest_win = excess.iloc[-window:]
    latest_sr_ann = float(sharpe_annual.iloc[-1]) if not np.isnan(sharpe_annual.iloc[-1]) else None
    latest_se = ses[-1] if ses and not np.isnan(ses[-1]) else None

    psr = None
    latest_rho1 = None
    if len(latest_win) >= 4 and latest_win.std() > 0 and latest_sr_ann is not None:
        sr_d = latest_win.mean() / latest_win.std()
        skew = float(latest_win.skew())
        kurt = float(latest_win.kurtosis()) + 3.0
        denom = (1.0 - skew * sr_d + (kurt - 1.0) / 4.0 * (sr_d ** 2)) / (len(latest_win) - 1.0)
        if denom > 0:
            z = sr_d / math.sqrt(denom)
            psr = round(0.5 * (1.0 + math.erf(z / math.sqrt(2.0))), 4)
        latest_rho1 = round(float(np.corrcoef(latest_win.values[1:], latest_win.values[:-1])[0, 1]), 3)

    return {
        "dates": list(r.index),
        "sharpe": [round(float(x), 3) if np.isfinite(x) else None for x in sharpe_annual],
        "se": [round(float(x), 3) if np.isfinite(x) else None for x in ses],
        "ci_upper": [round(float(x), 3) if np.isfinite(x) else None for x in ci_u],
        "ci_lower": [round(float(x), 3) if np.isfinite(x) else None for x in ci_l],
        "latest_sharpe": round(latest_sr_ann, 2) if latest_sr_ann is not None else None,
        "latest_se": latest_se,
        "psr": psr,
        "autocorrelation_rho1": latest_rho1,
        "rf_source": rf_source,
    }


# --- 8. Bar-range volume profile (issue #147) ------------------------------------

def bar_range_volume_profile(df: pd.DataFrame, n_bins: int = 50,
                             years: int = 2) -> dict:
    """Construct volume profile by distributing each bar's volume uniformly across its
    [low, high] price range rather than assigning all volume to close.
    """
    if df is None or getattr(df, "empty", True) or len(df) < 5:
        return {"bin_centers": [], "volumes": [], "poc_price": None}

    df = df.copy()
    cutoff = df.index[-1] - pd.DateOffset(years=years)
    sub = df[df.index >= cutoff]
    if len(sub) < 5:
        sub = df

    low = sub["low"] if "low" in sub.columns else sub["Low"]
    high = sub["high"] if "high" in sub.columns else sub["High"]
    vol = sub["volume"] if "volume" in sub.columns else sub["Volume"]

    price_min = float(low.min())
    price_max = float(high.max())
    if price_min >= price_max:
        return {"bin_centers": [price_min], "volumes": [float(vol.sum())], "poc_price": round(price_min, 2)}

    bins = np.linspace(price_min, price_max, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    vol_bins = np.zeros(n_bins, dtype=float)

    low_vals = low.values
    high_vals = high.values
    vol_vals = vol.values

    for l_val, h_val, v_val in zip(low_vals, high_vals, vol_vals):
        if not np.isfinite(v_val) or v_val <= 0:
            continue
        if h_val <= l_val:
            b_idx = int(np.digitize(l_val, bins) - 1)
            b_idx = min(max(b_idx, 0), n_bins - 1)
            vol_bins[b_idx] += v_val
        else:
            spread = h_val - l_val
            for j in range(n_bins):
                b_lo = bins[j]
                b_hi = bins[j + 1]
                overlap = max(0.0, min(h_val, b_hi) - max(l_val, b_lo))
                if overlap > 0:
                    vol_bins[j] += v_val * (overlap / spread)

    poc_idx = int(np.argmax(vol_bins))
    poc_price = round(float(bin_centers[poc_idx]), 2)

    return {
        "bin_centers": [round(float(c), 2) for c in bin_centers],
        "volumes": [round(float(v), 0) for v in vol_bins],
        "poc_price": poc_price,
        "n_bins": n_bins,
    }


historical_stress_scenarios = stress_scenarios
