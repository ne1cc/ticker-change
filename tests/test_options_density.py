"""Unit tests for Breeden-Litzenberger Risk-Neutral Density.

Verifies Epic 4 issue #127 acceptance criteria:
- Recovers a known lognormal density from Black-Scholes prices within tolerance
- Density integrates to ~1.0 and is non-negative
- Arbitrage-violating input yields a declared failure, not a chart
- Implied tail probabilities (+/- 10%, +/- 20%) are computed
"""
import math
import numpy as np
import pytest
from scipy.stats import norm

from options.density import breeden_litzenberger_density

_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))


def bs_call_price(s: float, k: float, t: float, v: float, r: float) -> float:
    sq_t = math.sqrt(t)
    d1 = (math.log(s / k) + (r + 0.5 * v * v) * t) / (v * sq_t)
    d2 = d1 - v * sq_t
    return float(s * norm.cdf(d1) - k * math.exp(-r * t) * norm.cdf(d2))


def test_breeden_litzenberger_recovers_lognormal_density():
    """Verify BL density recovers theoretical Black-Scholes lognormal density."""
    spot = 100.0
    t_years = 0.5
    vol = 0.20
    r = 0.045

    # Dense strike grid around spot
    strikes = np.linspace(60.0, 150.0, 91)
    call_prices = np.array([bs_call_price(spot, k, t_years, vol, r) for k in strikes])

    res = breeden_litzenberger_density(
        strikes, call_prices, spot=spot, t_years=t_years, r=r, enforce_no_arbitrage=True
    )

    assert res["available"] is True
    dens = np.array(res["density"])
    inner_k = np.array(res["strikes"])

    # Verify normalization (~1.0)
    assert 0.95 <= res["total_probability"] <= 1.05

    # Verify non-negativity
    assert np.all(dens >= 0.0)

    # Theoretical risk-neutral density
    # ln(S_T) ~ N(ln(S) + (r - 0.5*sigma^2)*T, sigma^2*T)
    # q(K) = (1 / (K * sigma * sqrt(T))) * n(d2)
    sig_t = vol * math.sqrt(t_years)
    d2 = (np.log(spot / inner_k) + (r - 0.5 * vol * vol) * t_years) / sig_t
    theoretical_q = norm.pdf(d2) / (inner_k * sig_t)
    theoretical_q_norm = theoretical_q / float(_trapezoid(theoretical_q, inner_k))

    # Mean absolute error between reconstructed BL density and theoretical density
    mae = float(np.mean(np.abs(dens - theoretical_q_norm)))
    assert mae < 0.001  # Accurate recovery of lognormal benchmark within 0.001

    # Implied move probabilities exist and are sensible for 20% vol over 6 months
    assert 5.0 <= res["prob_up_10pct"] <= 40.0
    assert 5.0 <= res["prob_down_10pct"] <= 40.0
    assert res["prob_up_20pct"] < res["prob_up_10pct"]
    assert res["prob_down_20pct"] < res["prob_down_10pct"]


def test_breeden_litzenberger_refuses_monotonicity_arbitrage():
    """Verify inverted call prices (C(K2) > C(K1) for K2 > K1) yields declared failure."""
    spot = 100.0
    t_years = 0.25
    strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
    # Strike 105 has price 6.0 > Strike 100 price 4.0 (clear monotonicity arbitrage)
    call_prices = np.array([12.0, 8.0, 4.0, 6.0, 1.0])

    res = breeden_litzenberger_density(
        strikes, call_prices, spot=spot, t_years=t_years, enforce_no_arbitrage=True
    )
    assert res["available"] is False
    assert "Monotonicity arbitrage violated" in res["reason"]


def test_breeden_litzenberger_refuses_butterfly_arbitrage():
    """Verify non-convex call prices (negative butterfly spread) yields declared failure."""
    spot = 100.0
    t_years = 0.25
    # Non-convex dip in call prices
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    # Butterfly at K=100: C(90) - 2*C(100) + C(110) = 15 - 2*10 + 2 = -3 < 0
    call_prices = np.array([22.0, 15.0, 10.0, 2.0, 1.0])

    res = breeden_litzenberger_density(
        strikes, call_prices, spot=spot, t_years=t_years, enforce_no_arbitrage=True
    )
    assert res["available"] is False
    assert "arbitrage violated" in res["reason"].lower()
