"""Epic 3.4: factor panel on /analytics — wiring, declared states, template."""
from unittest.mock import patch

import numpy as np
import pandas as pd

import app as app_module
import factors


def _price_df(n=400, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2026-10-01", periods=n)
    ret = pd.Series(rng.normal(0.0004, 0.012, n), index=idx)
    close = 100 * (1 + ret).cumprod()
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "volume": 1_000_000}, index=idx)


def _factor_df(price_df):
    ret = price_df["close"].pct_change().iloc[1:]
    f = pd.DataFrame({name: ret * w + np.random.default_rng(1).normal(0, 0.005, len(ret))
                      for name, w in zip(factors.SIX_FACTORS, [1.0, 0.3, -0.2, 0.1, 0.2, 0.1])},
                     index=ret.index)
    f["RF"] = 0.02 / 252.0
    return f


def test_factor_block_builds_exposures_and_hygiene(monkeypatch):
    px = _price_df()
    monkeypatch.setattr(app_module, "get_or_fetch_prices",
                        lambda symbol, period="5y": px)
    with patch.object(factors, "load_factor_returns",
                      return_value=(_factor_df(px), "2026-09-30")):
        out = app_module._build_factor_block("TEST")
    assert out["available"] is True
    assert out["factor_as_of"] == "2026-09-30"
    assert set(out["exposures"]) == set(factors.SIX_FACTORS)
    assert "alpha_t_stat" in out and "residual_vol_annualised_pct" in out
    assert out["beta_hygiene"]["n"] >= 60
    assert {"ols_beta", "dimson_beta", "blume_beta", "vasicek_beta"} <= set(out["beta_hygiene"])
    assert len(out["rolling_beta"]) >= 3


def test_factor_block_declared_states(monkeypatch):
    px = _price_df()
    monkeypatch.setattr(app_module, "get_or_fetch_prices",
                        lambda symbol, period="5y": px)
    # Factor file missing entirely
    with patch.object(factors, "load_factor_returns", return_value=(None, None)):
        out = app_module._build_factor_block("TEST")
    assert out == {"available": False,
                   "state": "Factor data unavailable (French download failed and nothing cached)."}

    # Too little overlap -> declared insufficient, not a blank
    short = px.iloc[-80:]
    with patch.object(factors, "load_factor_returns",
                      return_value=(_factor_df(short), "2026-09-30")):
        out = app_module._build_factor_block("TEST")
    assert out["available"] is False and "Insufficient" in out["state"]


def test_factor_panel_renders_and_declares_unavailable():
    from flask import render_template
    factors_data = {
        "available": True, "n": 750, "start": "2023-10-01", "end": "2026-09-30",
        "factor_as_of": "2026-09-30",
        "exposures": {"Mkt-RF": {"beta": 1.1, "t_stat": 8.0, "ci95": [0.83, 1.37]},
                      "SMB": {"beta": 0.2, "t_stat": 1.1, "ci95": [-0.16, 0.56]}},
        "alpha_annualised_pct": 3.4, "alpha_t_stat": 1.2, "r_squared": 0.42,
        "residual_vol_annualised_pct": 18.3, "residual_return_annualised_pct": 2.9,
        "beta_hygiene": {"n": 1250, "ols_beta": 1.15, "ols_beta_se": 0.08,
                         "dimson_beta": 1.21, "blume_beta": 1.1, "vasicek_beta": 1.13},
    }
    data = {"ticker": "T", "current_price": 1.0, "stats": {}, "charts": {}, "factors": factors_data}
    with app_module.app.test_request_context("/analytics?ticker=T"):
        html = render_template("analytics.html", data=data, ticker="T")
    assert "Factor Exposure & Residual Alpha" in html
    assert "Mkt-RF" in html and "Newey-West" in html
    assert "Dimson" in html and "Vasicek" in html
    assert "not significant" in html  # alpha t-stat 1.2 must not be sold as edge

    unavailable = {"available": False, "state": "Factor data unavailable (French download failed and nothing cached)."}
    data["factors"] = unavailable
    with app_module.app.test_request_context("/analytics?ticker=T"):
        html2 = render_template("analytics.html", data=data, ticker="T")
    assert "Factor model unavailable" in html2


def test_analytics_page_carries_factor_section(monkeypatch):
    px = _price_df()
    monkeypatch.setattr(app_module, "get_or_fetch_prices",
                        lambda symbol, period="5y": px)
    with patch.object(factors, "load_factor_returns",
                      return_value=(_factor_df(px), "2026-09-30")), \
         patch.object(app_module, "get_options_smile", return_value=None), \
         patch.object(app_module, "get_gex_profile", return_value=None), \
         patch.object(app_module.ai, "generate_report", return_value=None), \
         patch.object(app_module.providers, "get_short_interest", return_value=None):
        html = app_module.app.test_client().get("/analytics/fragment/options?ticker=TEST").get_data(as_text=True)
    assert "Factor Exposure & Residual Alpha" in html
    assert "Rolling 1y beta" in html  # CI-band chart mounted
    assert "insufficient" not in html.split("Factor Exposure")[1][:200]
