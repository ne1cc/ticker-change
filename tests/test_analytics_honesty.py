"""Regressions for the /analytics honesty pass: no fabricated or mislabelled metrics."""
import dataclasses
from unittest.mock import patch

import numpy as np
import pandas as pd

import app as app_module
import glossary
import microstructure


def _prices(n=400, seed=1):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, n)))
    idx = pd.bdate_range(end="2026-10-20", periods=n)
    return pd.DataFrame({"open": close * 0.999, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "volume": rng.integers(1_000_000, 2_000_000, n)}, index=idx)


# --- Removed: VPIN (daily bars cannot produce it) -----------------------------
def test_vpin_is_gone():
    fields = {f.name for f in dataclasses.fields(microstructure.MicrostructureAnalytics)}
    assert "vpin_toxicity" not in fields and "toxicity_regime" not in fields
    assert not hasattr(microstructure, "calculate_vpin")
    assert "vpin" not in glossary.GLOSSARY


# --- Removed: GBM Monte Carlo forecast ----------------------------------------
def test_monte_carlo_forecast_is_gone():
    with patch.object(app_module, "get_or_fetch_prices", return_value=_prices()):
        data = app_module.compute_analytics("TEST")
    assert "forward_estimates" not in data
    assert "monte_carlo" not in data["charts"]
    assert not [k for k in data["stats"] if k.startswith("mc_")]
    assert "monte_carlo" not in glossary.GLOSSARY


# --- VRP / Greeks use a real contract, not a multiple of historical vol ------
def _chain(strikes, call_iv, put_iv):
    calls = pd.DataFrame({"strike": strikes, "impliedVolatility": call_iv})
    puts = pd.DataFrame({"strike": strikes, "impliedVolatility": put_iv})
    return calls, puts, 100.0


def _expiry(days):
    return (pd.Timestamp.now().normalize() + pd.Timedelta(days=days)).strftime("%Y-%m-%d")


def _mock_chains(chains):
    return (patch.object(app_module, "_get_yf_ticker", return_value=object()),
            patch.object(app_module, "get_cached_expirations", return_value=list(chains)),
            patch.object(app_module, "get_cached_chain",
                         side_effect=lambda t, exp, **kw: chains[exp]))


def test_atm_contract_picks_30dte_and_averages_call_put_iv():
    chains = {_expiry(10): _chain([100], [0.50], [0.50]),
              _expiry(29): _chain([95, 100, 105], [0.30, 0.28, 0.31], [0.34, 0.32, 0.35]),
              _expiry(90): _chain([100], [0.40], [0.40])}
    p1, p2, p3 = _mock_chains(chains)
    with p1, p2, p3:
        atm = app_module.get_atm_contract("TEST", 100.0)
    assert atm["expiration"] == _expiry(29) and atm["dte"] == 29
    assert atm["strike"] == 100
    assert abs(atm["iv"] - 0.30) < 1e-9


def test_atm_contract_none_without_valid_iv():
    chains = {_expiry(30): _chain([100], [0.0], [float("nan")])}
    p1, p2, p3 = _mock_chains(chains)
    with p1, p2, p3:
        assert app_module.get_atm_contract("TEST", 100.0) is None


def test_vrp_is_iv_minus_realized_not_a_fixed_multiple():
    atm = {"expiration": _expiry(30), "dte": 30, "strike": 100.0, "iv": 0.45}
    out = app_module._atm_derivatives(atm, _prices())
    rv = out["vrp"]["realized_vol"]
    assert out["vrp"]["atm_implied_vol"] == 45.0
    assert out["vrp"]["vrp_spread_pct"] == round(45.0 - rv, 2)
    assert out["higher_order_greeks"]["gamma"] > 0
    assert out["atm_contract"] == atm


def test_atm_derivatives_empty_without_contract_or_history():
    assert app_module._atm_derivatives(None, _prices()) is None
    atm = {"expiration": _expiry(30), "dte": 30, "strike": 100.0, "iv": 0.45}
    assert app_module._atm_derivatives(atm, _prices(n=10)) is None


def test_analytics_page_wires_gex_and_real_contract():
    atm = {"expiration": _expiry(30), "dte": 30, "strike": 100.0, "iv": 0.45}
    gex = {"chart": None, "stats": {"regime": "negative", "spot": 100.0}}
    seen = {}
    real = microstructure.get_microstructure_analytics

    def spy(df, **kw):
        seen.update(kw)
        return real(df, **kw)

    minimal = {"current_price": 100.0, "charts": {}, "stats": {}}
    with patch.object(app_module, "compute_analytics", return_value=minimal), \
         patch.object(app_module, "get_or_fetch_prices", return_value=_prices()), \
         patch.object(app_module, "get_gex_profile", return_value=gex), \
         patch.object(app_module, "get_atm_contract", return_value=atm), \
         patch.object(app_module.microstructure, "get_microstructure_analytics", side_effect=spy), \
         patch.object(app_module.sec_8k, "fetch_and_parse_8k_filings", return_value=[]), \
         patch.object(app_module.providers, "get_short_interest", return_value=None), \
         patch.object(app_module.ai, "generate_report", return_value=None):
        html = app_module.app.test_client().get("/analytics?ticker=TEST").get_data(as_text=True)
    assert seen["is_negative_gex"] is True
    assert "Priced on the real contract" in html and "100.00 call" in html
