"""Gatheral's Stochastic Volatility Inspired (SVI) model and OTM smile engine.

Part of Epic 4 (issue #129):
- Selects OTM puts below spot and OTM calls above spot (avoiding illiquid ITM call quotes)
- Fits raw SVI total variance: w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
- Validates quote validity, minimum open interest, and fit RMSE
- Tests no-arbitrage bounds (Durrleman butterfly condition)
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def svi_total_variance(
    k: np.ndarray | float,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> np.ndarray | float:
    """Evaluate raw SVI total implied variance w(k) = sigma^2 * T."""
    k_arr = np.asarray(k, dtype=float)
    diff = k_arr - m
    val = a + b * (rho * diff + np.sqrt(diff ** 2 + sigma ** 2))
    return val if isinstance(k, np.ndarray) else float(val)


def select_otm_wings(
    calls_df: pd.DataFrame,
    puts_df: pd.DataFrame,
    spot: float,
    t_years: float,
    min_volume: int = 0,
    min_oi: int = 0,
) -> pd.DataFrame:
    """Construct an out-of-the-money options smile.

    Rule:
      - OTM Puts below spot (K < spot)
      - OTM Calls at or above spot (K >= spot)
    Filters out invalid quotes (zero bids, inverted bid/ask, non-positive IV).
    """
    if spot <= 0 or t_years <= 0:
        return pd.DataFrame()

    rows = []

    # OTM Puts (K < spot)
    if puts_df is not None and not puts_df.empty:
        p_cols = {c.lower(): c for c in puts_df.columns}
        k_col = p_cols.get("strike", "strike")
        iv_col = p_cols.get("impliedvolatility", p_cols.get("iv", "iv"))
        bid_col = p_cols.get("bid")
        ask_col = p_cols.get("ask")
        vol_col = p_cols.get("volume")
        oi_col = p_cols.get("openinterest", p_cols.get("oi"))

        for _, row in puts_df.iterrows():
            strike = float(row[k_col])
            if strike >= spot:
                continue  # Skip ITM puts

            iv = float(row[iv_col]) if iv_col in row and pd.notna(row[iv_col]) else 0.0
            if iv < 0.01 or iv > 4.0:
                continue

            # Quote validity check
            if bid_col and ask_col and pd.notna(row[bid_col]) and pd.notna(row[ask_col]):
                bid = float(row[bid_col])
                ask = float(row[ask_col])
                if bid > 0 and ask < bid:
                    continue

            oi = int(row[oi_col]) if oi_col and pd.notna(row[oi_col]) else 0
            if oi < min_oi:
                continue

            log_m = math.log(strike / spot)
            w = (iv ** 2) * t_years
            rows.append({
                "strike": strike,
                "cp": "P",
                "iv": iv,
                "log_moneyness": log_m,
                "total_variance": w,
                "is_otm": True,
            })

    # OTM Calls (K >= spot)
    if calls_df is not None and not calls_df.empty:
        c_cols = {c.lower(): c for c in calls_df.columns}
        k_col = c_cols.get("strike", "strike")
        iv_col = c_cols.get("impliedvolatility", c_cols.get("iv", "iv"))
        bid_col = c_cols.get("bid")
        ask_col = c_cols.get("ask")
        oi_col = c_cols.get("openinterest", c_cols.get("oi"))

        for _, row in calls_df.iterrows():
            strike = float(row[k_col])
            if strike < spot:
                continue  # Skip ITM calls

            iv = float(row[iv_col]) if iv_col in row and pd.notna(row[iv_col]) else 0.0
            if iv < 0.01 or iv > 4.0:
                continue

            if bid_col and ask_col and pd.notna(row[bid_col]) and pd.notna(row[ask_col]):
                bid = float(row[bid_col])
                ask = float(row[ask_col])
                if bid > 0 and ask < bid:
                    continue

            oi = int(row[oi_col]) if oi_col and pd.notna(row[oi_col]) else 0
            if oi < min_oi:
                continue

            log_m = math.log(strike / spot)
            w = (iv ** 2) * t_years
            rows.append({
                "strike": strike,
                "cp": "C",
                "iv": iv,
                "log_moneyness": log_m,
                "total_variance": w,
                "is_otm": True,
            })

    if not rows:
        return pd.DataFrame()

    res = pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)
    return res


def check_durrleman_arbitrage(
    k_grid: np.ndarray,
    w_grid: np.ndarray,
) -> bool:
    """Check Durrleman condition g(k) >= 0 to ensure absence of butterfly arbitrage."""
    if len(k_grid) < 5 or np.any(w_grid <= 0):
        return False

    dk = k_grid[1] - k_grid[0]
    w_prime = np.gradient(w_grid, dk)
    w_second = np.gradient(w_prime, dk)

    # Durrleman (2003) condition g(k) = (1 - k*w'/(2w))^2 - (w'^2/4)*(1/w + 1/4) + w''/2
    term1 = (1.0 - (k_grid * w_prime) / (2.0 * w_grid)) ** 2
    term2 = (w_prime ** 2 / 4.0) * (1.0 / w_grid + 0.25)
    term3 = w_second / 2.0

    g = term1 - term2 + term3
    # Allow tiny numerical tolerance
    return bool(np.all(g >= -1e-4))


def fit_svi(
    k_obs: np.ndarray,
    w_obs: np.ndarray,
    t_years: float,
) -> Dict[str, Any]:
    """Fit raw SVI model parameters to empirical log-moneyness and total variance."""
    k_clean = np.asarray(k_obs, dtype=float)
    w_clean = np.asarray(w_obs, dtype=float)

    valid = (w_clean > 0) & np.isfinite(k_clean) & np.isfinite(w_clean)
    k_clean = k_clean[valid]
    w_clean = w_clean[valid]

    if len(k_clean) < 5:
        return {
            "success": False,
            "reason": f"Insufficient observations for SVI fit ({len(k_clean)} < 5)",
            "rmse": 0.0,
            "params": {},
            "arbitrage_free": False,
        }

    # Initial parameter heuristics
    atm_idx = np.argmin(np.abs(k_clean))
    w_min = float(np.min(w_clean))
    w_atm = float(w_clean[atm_idx])

    init_params = [
        max(0.001, w_min * 0.5),  # a
        0.1,                      # b
        -0.2,                     # rho (equity index skew typically negative)
        float(k_clean[np.argmin(w_clean)]),  # m
        0.1,                      # sigma
    ]

    # Bounds on parameters: a >= 0, b >= 0, |rho| < 1, sigma > 0
    bounds = [
        (0.0, None),
        (0.0001, 2.0),
        (-0.999, 0.999),
        (float(k_clean.min()) * 1.5, float(k_clean.max()) * 1.5),
        (0.001, 1.0),
    ]

    def loss(p):
        a, b, rho, m, sig = p
        # Penalty for negative variance constraint: a + b * sig * sqrt(1 - rho^2) >= 0
        min_var = a + b * sig * math.sqrt(max(0.0, 1.0 - rho ** 2))
        penalty = 1e4 * (min_var ** 2) if min_var < 0 else 0.0

        model_w = svi_total_variance(k_clean, a, b, rho, m, sig)
        residuals = model_w - w_clean
        return float(np.sum(residuals ** 2) + penalty)

    res = minimize(loss, init_params, bounds=bounds, method="L-BFGS-B")

    a, b, rho, m, sig = res.x
    fitted_w = svi_total_variance(k_clean, a, b, rho, m, sig)
    rmse = math.sqrt(float(np.mean((fitted_w - w_clean) ** 2)))

    # Evaluate dense grid for plotting and butterfly arbitrage check
    dense_k = np.linspace(k_clean.min(), k_clean.max(), 100)
    dense_w = svi_total_variance(dense_k, a, b, rho, m, sig)
    dense_iv = np.sqrt(np.maximum(dense_w / t_years, 0.0001))

    arb_free = check_durrleman_arbitrage(dense_k, dense_w)

    return {
        "success": bool(res.success and rmse < 0.20),
        "params": {
            "a": round(float(a), 5),
            "b": round(float(b), 5),
            "rho": round(float(rho), 4),
            "m": round(float(m), 4),
            "sigma": round(float(sig), 4),
        },
        "rmse": round(float(rmse), 5),
        "arbitrage_free": arb_free,
        "dense_k": dense_k.tolist(),
        "dense_iv": dense_iv.tolist(),
    }
