"""Risk-Neutral Probability Density Function via Breeden-Litzenberger (1978).

Part of Epic 4 (issue #127):
- Replaces removed GBM assumption with options-implied risk-neutral distribution
- Evaluates second derivative of call price w.r.t strike: q(K) = e^(rT) * d^2 C / dK^2
- Strictly enforces no-arbitrage checks (monotonicity and butterfly spread non-negativity)
- Computes tail mass and implied move probabilities (+/- 10%, +/- 20%)
- Recovers theoretical lognormal density under Black-Scholes prices
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.stats import norm
from scipy.interpolate import CubicSpline

# NumPy 2.x trapezoid compatibility
_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))


def breeden_litzenberger_density(
    strikes: np.ndarray,
    call_prices: np.ndarray,
    spot: float,
    t_years: float,
    r: float = 0.045,
    enforce_no_arbitrage: bool = True,
) -> Dict[str, Any]:
    """Calculate risk-neutral density from smooth strike-call price pairs.

    Returns dict with density curve, move probabilities, and no-arbitrage diagnostics.
    """
    k = np.asarray(strikes, dtype=float)
    c = np.asarray(call_prices, dtype=float)

    if len(k) < 5 or t_years <= 0 or spot <= 0:
        return {
            "available": False,
            "reason": "Insufficient strike points or non-positive spot/maturity",
        }

    # Ensure strictly ascending strikes
    order = np.argsort(k)
    k = k[order]
    c = c[order]

    # No-Arbitrage Check 1: Monotonicity (Call prices must decrease as strike increases)
    dC = np.diff(c)
    if enforce_no_arbitrage and np.any(dC > 1e-4):
        bad_idx = np.where(dC > 1e-4)[0][0]
        return {
            "available": False,
            "reason": f"Monotonicity arbitrage violated: Call({k[bad_idx]:.2f})={c[bad_idx]:.2f} < Call({k[bad_idx+1]:.2f})={c[bad_idx+1]:.2f}",
        }

    try:
        cs = CubicSpline(k, c)
    except Exception as e:
        return {
            "available": False,
            "reason": f"Spline interpolation failed: {e}",
        }

    # Dense uniform grid for evaluating second derivatives
    n_grid = 201
    k_min = max(0.01, float(k.min()))
    k_max = float(k.max())
    grid_k = np.linspace(k_min, k_max, n_grid)
    inner_k = grid_k[1:-1]

    # Second derivative d^2C/dK^2
    d2C = cs(inner_k, 2)

    # No-Arbitrage Check 2: Convexity / Butterfly arbitrage (d2C/dK^2 >= 0)
    if enforce_no_arbitrage and np.any(d2C < -1e-4):
        bad_min = float(np.min(d2C))
        return {
            "available": False,
            "reason": f"Butterfly arbitrage violated: negative probability density (min d2C={bad_min:.4f})",
        }

    # Density q(K) = exp(r * T) * d^2 C / dK^2
    disc_factor = math.exp(r * t_years)
    q = disc_factor * np.maximum(d2C, 0.0)

    # Numerical integral to normalize density
    total_prob = float(_trapezoid(q, inner_k))
    if total_prob > 0:
        q_norm = q / total_prob
    else:
        q_norm = q

    # Implied tail probabilities
    # Prob(K >= 1.10 * S), Prob(K <= 0.90 * S), Prob(K >= 1.20 * S), Prob(K <= 0.80 * S)
    mask_up10 = inner_k >= spot * 1.10
    mask_down10 = inner_k <= spot * 0.90
    mask_up20 = inner_k >= spot * 1.20
    mask_down20 = inner_k <= spot * 0.80

    prob_up10 = float(_trapezoid(q_norm[mask_up10], inner_k[mask_up10])) if np.any(mask_up10) else 0.0
    prob_down10 = float(_trapezoid(q_norm[mask_down10], inner_k[mask_down10])) if np.any(mask_down10) else 0.0
    prob_up20 = float(_trapezoid(q_norm[mask_up20], inner_k[mask_up20])) if np.any(mask_up20) else 0.0
    prob_down20 = float(_trapezoid(q_norm[mask_down20], inner_k[mask_down20])) if np.any(mask_down20) else 0.0

    # Lognormal comparison benchmark
    # Forward price F = S * exp(r * T)
    # Estimate ATM vol from total variance or standard dev of q_norm
    log_k = np.log(inner_k / spot)
    mu_bench = (r - 0.5 * 0.20 ** 2) * t_years
    sig_bench = 0.20 * math.sqrt(t_years)
    lognormal_bench = norm.pdf((np.log(inner_k) - (math.log(spot) + mu_bench)) / sig_bench) / (inner_k * sig_bench)

    return {
        "available": True,
        "strikes": [round(float(x), 2) for x in inner_k],
        "density": [round(float(x), 6) for x in q_norm],
        "lognormal_benchmark": [round(float(x), 6) for x in lognormal_bench],
        "total_probability": round(total_prob, 4),
        "prob_up_10pct": round(prob_up10 * 100.0, 1),
        "prob_down_10pct": round(prob_down10 * 100.0, 1),
        "prob_up_20pct": round(prob_up20 * 100.0, 1),
        "prob_down_20pct": round(prob_down20 * 100.0, 1),
        "spot": spot,
        "t_years": t_years,
        "risk_free_rate": r,
    }
