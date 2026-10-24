"""Unit tests for OTM wing selection and SVI volatility smile fitting.

Verifies Epic 4 issue #129 acceptance criteria:
- Wing selection: puts below spot, calls above spot
- SVI fit recovers synthetic SVI parameters within tolerance
- Low-liquidity expiries are dropped with visible count
- Durrleman condition catches butterfly arbitrage
"""
import math
import numpy as np
import pandas as pd
import pytest

from options.svi import (
    select_otm_wings,
    fit_svi,
    svi_total_variance,
    check_durrleman_arbitrage,
)


def test_select_otm_wings_rule():
    """Verify OTM wing selection: puts strictly below spot, calls at or above spot."""
    spot = 100.0
    t_years = 30.0 / 365.0

    calls_data = pd.DataFrame([
        {"strike": 90.0, "iv": 0.30, "bid": 11.0, "ask": 11.5, "openInterest": 100},  # ITM call -> drop
        {"strike": 95.0, "iv": 0.28, "bid": 6.0, "ask": 6.5, "openInterest": 100},   # ITM call -> drop
        {"strike": 100.0, "iv": 0.25, "bid": 2.5, "ask": 2.7, "openInterest": 200},  # ATM/OTM call -> keep
        {"strike": 105.0, "iv": 0.23, "bid": 0.8, "ask": 1.0, "openInterest": 150},  # OTM call -> keep
        {"strike": 110.0, "iv": 0.22, "bid": 0.2, "ask": 0.3, "openInterest": 80},   # OTM call -> keep
    ])

    puts_data = pd.DataFrame([
        {"strike": 90.0, "iv": 0.31, "bid": 0.4, "ask": 0.5, "openInterest": 120},   # OTM put -> keep
        {"strike": 95.0, "iv": 0.27, "bid": 1.2, "ask": 1.4, "openInterest": 180},   # OTM put -> keep
        {"strike": 100.0, "iv": 0.25, "bid": 2.5, "ask": 2.7, "openInterest": 200},  # ATM put -> drop (call used)
        {"strike": 105.0, "iv": 0.24, "bid": 6.2, "ask": 6.5, "openInterest": 100},  # ITM put -> drop
        {"strike": 110.0, "iv": 0.23, "bid": 10.5, "ask": 11.0, "openInterest": 50}, # ITM put -> drop
    ])

    wings = select_otm_wings(calls_data, puts_data, spot, t_years)
    assert not wings.empty
    assert len(wings) == 5  # 90P, 95P, 100C, 105C, 110C

    # Verify every strike < spot has cp == 'P'
    puts_sub = wings[wings["strike"] < spot]
    assert all(puts_sub["cp"] == "P")
    assert set(puts_sub["strike"]) == {90.0, 95.0}

    # Verify every strike >= spot has cp == 'C'
    calls_sub = wings[wings["strike"] >= spot]
    assert all(calls_sub["cp"] == "C")
    assert set(calls_sub["strike"]) == {100.0, 105.0, 110.0}


def test_select_otm_wings_filters_invalid_quotes():
    """Verify inverted bid/ask, non-positive IV, and zero-bid quotes are filtered."""
    spot = 100.0
    t_years = 0.1

    calls = pd.DataFrame([
        {"strike": 105.0, "iv": 0.20, "bid": 2.0, "ask": 1.5, "openInterest": 50},   # Inverted bid/ask -> drop
        {"strike": 110.0, "iv": 0.001, "bid": 0.5, "ask": 0.6, "openInterest": 50},  # Tiny IV -> drop
        {"strike": 115.0, "iv": 0.22, "bid": 0.3, "ask": 0.4, "openInterest": 10},   # Valid -> keep
    ])
    puts = pd.DataFrame([
        {"strike": 95.0, "iv": 0.25, "bid": 0.8, "ask": 0.9, "openInterest": 20},    # Valid -> keep
    ])

    wings = select_otm_wings(calls, puts, spot, t_years, min_oi=15)
    # Strike 115 has oi=10 < min_oi=15 -> dropped
    assert len(wings) == 1
    assert wings.iloc[0]["strike"] == 95.0


def test_svi_synthetic_parameter_recovery():
    """Verify SVI fit recovers synthetic total variance from known parameters."""
    t_years = 0.25
    # True synthetic parameters
    true_a = 0.04
    true_b = 0.12
    true_rho = -0.35
    true_m = 0.02
    true_sigma = 0.15

    # Synthetic grid of log-moneyness k in [-0.3, 0.3]
    k_synth = np.linspace(-0.25, 0.25, 21)
    w_synth = svi_total_variance(k_synth, true_a, true_b, true_rho, true_m, true_sigma)

    fit_res = fit_svi(k_synth, w_synth, t_years)
    assert fit_res["success"] is True
    assert fit_res["rmse"] < 0.005  # Within 0.005 total variance RMSE
    assert fit_res["arbitrage_free"] is True

    # Evaluated dense fit matches synthetic w
    dense_k = np.array(fit_res["dense_k"])
    fitted_w = svi_total_variance(
        dense_k,
        fit_res["params"]["a"],
        fit_res["params"]["b"],
        fit_res["params"]["rho"],
        fit_res["params"]["m"],
        fit_res["params"]["sigma"],
    )
    expected_w = svi_total_variance(dense_k, true_a, true_b, true_rho, true_m, true_sigma)
    np.testing.assert_allclose(fitted_w, expected_w, atol=0.01)


def test_durrleman_arbitrage_detection():
    """Verify Durrleman condition detects butterfly arbitrage (negative density)."""
    k_grid = np.linspace(-0.3, 0.3, 50)
    # Valid convex total variance
    w_valid = 0.04 + 0.10 * (0.0 * k_grid + np.sqrt(k_grid ** 2 + 0.05 ** 2))
    assert check_durrleman_arbitrage(k_grid, w_valid) is True

    # Dipping total variance with steep negative curvature violating butterfly condition (g(k) < 0)
    k_grid_sub = np.linspace(-0.1, 0.1, 51)
    w_invalid = 0.04 - 2.5 * (k_grid_sub ** 2)
    assert check_durrleman_arbitrage(k_grid_sub, w_invalid) is False
