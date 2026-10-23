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
