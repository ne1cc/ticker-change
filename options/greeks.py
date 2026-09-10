"""Analytical and vectorized Black-Scholes Greeks engine including first and second-order Greeks."""
from __future__ import annotations

import math
from typing import Dict, Literal, Union
import numpy as np
from scipy.stats import norm


def calculate_greeks(
    s: float,
    k: float,
    t: float,
    v: float,
    r: float = 0.045,
    cp: Literal['call', 'put', 'C', 'P'] = 'call',
) -> Dict[str, float]:
    """Compute individual contract first and second-order Black-Scholes Greeks.

    Parameters:
        s: Current spot price
        k: Strike price
        t: Time to expiration in years
        v: Implied volatility (decimal, e.g. 0.25 for 25%)
        r: Annualized risk-free rate (e.g. 0.045 for 4.5%)
        cp: 'call'/'C' or 'put'/'P'
    """
    is_call = cp.lower() in ('call', 'c')

    # Defensive boundary protections
    t = max(float(t), 1.0 / 365.0)  # floor at 1 day
    v = max(float(v), 0.001)         # floor at 0.1% IV
    s = max(float(s), 0.01)
    k = max(float(k), 0.01)

    sq_t = math.sqrt(t)
    d1 = (math.log(s / k) + (r + 0.5 * v * v) * t) / (v * sq_t)
    d2 = d1 - v * sq_t

    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)

    # First-order Greeks
    if is_call:
        delta = float(cdf_d1)
        theta = float((- (s * pdf_d1 * v) / (2.0 * sq_t) - r * k * math.exp(-r * t) * cdf_d2) / 365.0)
        rho = float((k * t * math.exp(-r * t) * cdf_d2) * 0.01)
    else:
        delta = float(cdf_d1 - 1.0)
        theta = float((- (s * pdf_d1 * v) / (2.0 * sq_t) + r * k * math.exp(-r * t) * (1.0 - cdf_d2)) / 365.0)
        rho = float((-k * t * math.exp(-r * t) * (1.0 - cdf_d2)) * 0.01)

    gamma = float(pdf_d1 / (s * v * sq_t))
    vega = float(s * pdf_d1 * sq_t * 0.01)

    # Second-order Greeks
    # Vanna: dDelta / dSigma = dVega / dSpot = -n(d1) * d2 / v
    vanna = float(-pdf_d1 * (d2 / v))

    # Charm: -dDelta / dt (per day decay)
    charm = float(- (pdf_d1 * ((r / (v * sq_t)) - (d2 / (2.0 * t)))) / 365.0)

    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho,
        'vanna': vanna,
        'charm': charm,
        # Legacy compatibility keys
        'call_delta': float(cdf_d1),
        'call_theta': float((- (s * pdf_d1 * v) / (2.0 * sq_t) - r * k * math.exp(-r * t) * cdf_d2) / 365.0),
        'call_rho': float((k * t * math.exp(-r * t) * cdf_d2) * 0.01),
        'put_delta': float(cdf_d1 - 1.0),
        'put_theta': float((- (s * pdf_d1 * v) / (2.0 * sq_t) + r * k * math.exp(-r * t) * (1.0 - cdf_d2)) / 365.0),
        'put_rho': float((-k * t * math.exp(-r * t) * (1.0 - cdf_d2)) * 0.01),
    }


def vectorized_greeks(
    s: Union[float, np.ndarray],
    k: np.ndarray,
    t_years: np.ndarray,
    sigma: np.ndarray,
    r: float = 0.045,
    cp: Union[str, np.ndarray] = 'C',
) -> Dict[str, np.ndarray]:
    """Vectorized calculation of first and second-order Black-Scholes Greeks using NumPy."""
    s = np.asarray(s, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    t = np.maximum(np.asarray(t_years, dtype=np.float64), 1.0 / 365.0)
    v = np.clip(np.asarray(sigma, dtype=np.float64), 1e-3, 5.0)

    sq_t = np.sqrt(t)
    d1 = (np.log(s / k) + (r + 0.5 * v * v) * t) / (v * sq_t)
    d2 = d1 - v * sq_t

    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)

    is_call = (cp == 'C') if isinstance(cp, np.ndarray) else (cp.upper() in ('C', 'CALL'))

    delta = np.where(is_call, cdf_d1, cdf_d1 - 1.0)
    gamma = pdf_d1 / (s * v * sq_t)
    vega = s * pdf_d1 * sq_t * 0.01
    
    theta_call = (- (s * pdf_d1 * v) / (2.0 * sq_t) - r * k * np.exp(-r * t) * cdf_d2) / 365.0
    theta_put = (- (s * pdf_d1 * v) / (2.0 * sq_t) + r * k * np.exp(-r * t) * (1.0 - cdf_d2)) / 365.0
    theta = np.where(is_call, theta_call, theta_put)

    # Second-order Greeks
    vanna = -pdf_d1 * (d2 / v)
    charm = - (pdf_d1 * ((r / (v * sq_t)) - (d2 / (2.0 * t)))) / 365.0

    return {
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta,
        'vanna': vanna,
        'charm': charm,
    }
