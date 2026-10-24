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

    with patch.object(app_module, "get_or_fetch_prices", return_value=_prices()), \
         patch.object(app_module, "get_options_smile", return_value=None), \
         patch.object(app_module.ml, "predict", return_value=None), \
         patch.object(app_module, "get_gex_profile", return_value=gex), \
         patch.object(app_module, "get_atm_contract", return_value=atm), \
         patch.object(app_module.microstructure, "get_microstructure_analytics", side_effect=spy), \
         patch.object(app_module.sec_8k, "fetch_and_parse_8k_filings", return_value=[]), \
         patch.object(app_module.providers, "get_short_interest", return_value=None):
        html = app_module.app.test_client().get("/analytics/fragment/options?ticker=TEST").get_data(as_text=True)
    assert seen["is_negative_gex"] is True
    assert "Priced on the real contract" in html and "100.00 call" in html


# --- FRED unavailable -> None, never a stale 2024 number ---------------------
def test_macro_report_has_no_invented_values_without_fred():
    import macro_engine
    px = _prices()
    with patch.object(macro_engine, "fetch_fred_series", return_value=[]):
        rep = macro_engine.get_macro_financial_report(px, px)
    assert rep.fed_funds_rate is None and rep.yield_curve_2s10s_spread is None
    assert rep.cpi_inflation_yoy is None and rep.is_yield_curve_inverted is None
    assert rep.current_regime is None


def test_macro_report_uses_fred_when_present():
    import macro_engine
    px = _prices()
    obs = {"DFF": [{"value": "4.0"}], "T10Y2Y": [{"value": "-0.2"}],
           "CPIAUCSL": [{"value": str(100 + i)} for i in range(13)][::-1]}
    with patch.object(macro_engine, "fetch_fred_series", side_effect=lambda sid, limit=0: obs[sid]):
        rep = macro_engine.get_macro_financial_report(px, px)
    assert rep.fed_funds_rate == 4.0 and rep.is_yield_curve_inverted is True
    assert rep.current_regime == "CURVE_INVERTED_EASING"
    assert rep.cpi_inflation_yoy == 12.0


# --- Seasonality is monthly, not daily ---------------------------------------
def test_seasonality_is_compounded_monthly_return():
    idx = pd.bdate_range("2020-01-01", "2025-07-15")
    daily = pd.Series(0.0, index=idx)
    jan = idx.month == 1
    daily[jan] = 1.02 ** (1.0 / jan[idx.year == 2021].sum()) - 1  # ~+2% per January
    close = 100 * (1 + daily).cumprod()
    avg, counts = app_module._monthly_seasonality(close)
    assert abs(avg[1] - 2.0) < 0.15          # was ~0.1 when daily returns were averaged
    assert abs(avg[3]) < 1e-6
    assert counts[1] == 5          # Jan 2020 has no prior month-end close
    assert counts[6] == 6          # Jun 2025 is complete
    assert counts[7] == 5          # Jul 2025 is in progress and excluded


# --- Spreads are already in percent; the template must not x100 again --------
def test_spread_displayed_once_not_times_100():
    from flask import render_template
    micro = {"corwin_schultz_spread_pct": 0.42, "roll_effective_spread_pct": 0.31,
             "amihud_illiquidity": 0.1, "squeeze_risk_score": 10.0, "squeeze_risk_level": "LOW"}
    data = {"ticker": "T", "current_price": 1.0, "stats": {}, "charts": {},
            "institutional": {"microstructure": micro}}
    with app_module.app.test_request_context("/analytics?ticker=T"):
        html = render_template("analytics.html", data=data, ticker="T")
    assert "0.420%" in html and "0.310%" in html and "42.000%" not in html


def test_macro_panel_flags_fred_unavailable():
    from flask import render_template
    macro = {"regime": None, "fed_funds_rate": None, "yield_curve_2s10s": None, "is_inverted": None,
             "cpi_yoy": None, "betas": {"bull_beta": 1.1, "bear_beta": 0.9}, "upside_capture": 100.0,
             "downside_capture": 100.0}
    data = {"ticker": "T", "current_price": 1.0, "stats": {}, "charts": {},
            "institutional": {"macro_conditioning": macro}}
    with app_module.app.test_request_context("/analytics?ticker=T"):
        html = render_template("analytics.html", data=data, ticker="T")
    assert "FRED unavailable" in html and "5.33" not in html


# --- Squeeze index never scores an invented cost-to-borrow --------------------
def test_squeeze_index_excludes_ctb_when_unreported():
    # Weights renormalise over the inputs actually used (SI 40, DTC 30, GEX 15).
    score, _ = microstructure.compute_squeeze_risk_index(3.0, 2.0)
    assert abs(score - (4.8 + 30 * 2 / 7 + 5) / 85 * 100) < 0.05
    # A reported CTB still contributes its term on the full 100 weight sum.
    score_ctb, _ = microstructure.compute_squeeze_risk_index(3.0, 2.0, cost_to_borrow_pct=2.5)
    assert score_ctb != score


def test_squeeze_score_stays_in_range_with_renormalised_weights():
    assert microstructure.compute_squeeze_risk_index(25.0, 7.0, is_negative_gex=True)[0] == 100.0
    assert microstructure.compute_squeeze_risk_index(0.0, 0.0)[0] == round(5 / 85 * 100, 1)
    assert microstructure.compute_squeeze_risk_index(
        200.0, 20.0, cost_to_borrow_pct=50.0, is_negative_gex=True)[0] == 100.0


def test_squeeze_badge_lists_assumed_inputs_not_just_si():
    from flask import render_template
    micro = {"corwin_schultz_spread_pct": 0.4, "roll_effective_spread_pct": 0.3,
             "amihud_illiquidity": 0.1, "squeeze_risk_score": 21.6,
             "squeeze_risk_level": "LOW", "squeeze_inputs_assumed": ["SI % float", "DTC"]}
    data = {"ticker": "T", "current_price": 1.0, "stats": {}, "charts": {},
            "institutional": {"microstructure": micro,
                              "short_interest": {"si_source": "assumed"}}}
    with app_module.app.test_request_context("/analytics?ticker=T"):
        html = render_template("analytics.html", data=data, ticker="T")
    assert "assumed si % float · dtc" in html
    assert "Cost-to-borrow excluded" in html


# --- period= is honoured: shallow caches trigger a deeper fetch ----------------
def _shallow_prices(n=43, freq="D"):
    idx = pd.date_range(end=pd.Timestamp.now().normalize(), periods=n, freq=freq)
    close = pd.Series(100.0, index=idx)
    return pd.DataFrame({"Open": close, "High": close, "Low": close,
                         "Close": close, "Volume": pd.Series(1_000_000, index=idx)})


def test_shallow_cache_triggers_deeper_fetch_when_5y_requested(tmp_path, monkeypatch):
    import db as db_module
    monkeypatch.setattr(db_module, "DB_PATH", str(tmp_path / "t.db"))
    db_module.init_db()
    app_module._PRICE_MEMO.clear()
    app_module._PRICE_DEPTH_FETCH_ATTEMPTS.clear()
    db_module.store_prices("SHAL", _shallow_prices())

    deep = _shallow_prices(1260)  # ~5y of calendar rows
    fetch = patch.object(app_module, "_fetch_yfinance_with_retry", return_value=deep)
    with fetch as m:
        out = app_module.get_or_fetch_prices("SHAL", period="5y")
    assert m.call_args.args[1] == "5y" or m.call_args.kwargs.get("period") == "5y"
    assert len(out) == 1260  # 43 rows would have starved the page


def test_period_2y_slices_cached_full_history(tmp_path, monkeypatch):
    import db as db_module
    monkeypatch.setattr(db_module, "DB_PATH", str(tmp_path / "t.db"))
    db_module.init_db()
    app_module._PRICE_MEMO.clear()
    app_module._PRICE_DEPTH_FETCH_ATTEMPTS.clear()
    db_module.store_prices("TWOY", _shallow_prices(1260))
    with patch.object(app_module, "_fetch_yfinance_with_retry") as m:
        out = app_module.get_or_fetch_prices("TWOY", period="2y")
    assert m.call_count == 0  # fresh and deep enough: no network
    assert 700 <= len(out) <= 740  # ~2y of calendar rows, not the full 5y


def test_depth_refetch_is_throttled_for_short_history_symbols(tmp_path, monkeypatch):
    import db as db_module
    monkeypatch.setattr(db_module, "DB_PATH", str(tmp_path / "t.db"))
    db_module.init_db()
    app_module._PRICE_MEMO.clear()
    app_module._PRICE_DEPTH_FETCH_ATTEMPTS.clear()
    db_module.store_prices("IPOX", _shallow_prices(20))  # young IPO, can never cover 5y

    with patch.object(app_module, "_fetch_yfinance_with_retry", return_value=None) as m:
        app_module.get_or_fetch_prices("IPOX", period="5y")
        app_module.get_or_fetch_prices("IPOX", period="5y")
    assert m.call_count == 1  # second call within the hour must not re-hit yfinance


# --- Capture ratios are geometric on monthly returns, or declared insufficient
def _monthly_spread(monthly_returns):
    """Expand a list of monthly simple returns into a daily return series whose
    per-month compound equals each entry (20 trading days per month)."""
    days = []
    idx = []
    for i, r in enumerate(monthly_returns):
        k = 20
        daily = (1.0 + r) ** (1.0 / k) - 1.0
        days += [daily] * k
        idx += pd.bdate_range(start=pd.Timestamp("2024-01-01") + pd.DateOffset(months=i), periods=k).tolist()
    return pd.Series(days, index=pd.DatetimeIndex(idx))


def test_capture_ratios_match_hand_computed_geometric_example():
    import macro_engine
    bench = _monthly_spread([0.10] * 6 + [-0.10] * 6)
    stock = _monthly_spread([0.05] * 6 + [-0.05] * 6)
    up, down = macro_engine.calculate_capture_ratios(stock, bench)
    assert abs(up - 1.05 / 1.10 * 100) < 0.01
    assert abs(down - 0.95 / 0.90 * 100) < 0.01


def test_capture_ratios_declared_insufficient_not_neutral_100():
    import macro_engine
    bench = _monthly_spread([0.10, -0.10, 0.10])
    stock = _monthly_spread([0.05, -0.05, 0.05])
    assert macro_engine.calculate_capture_ratios(stock, bench) == (None, None)


def test_capture_glossary_matches_geometric_monthly_implementation():
    assert "geometric mean" in glossary.GLOSSARY["upside_capture"]["long"]
    assert "monthly" in glossary.GLOSSARY["downside_capture"]["long"]


def test_capture_panel_declares_insufficient_history():
    from flask import render_template
    macro = {"regime": None, "fed_funds_rate": None, "yield_curve_2s10s": None,
             "is_inverted": None, "cpi_yoy": None,
             "betas": {"bull_beta": 1.1, "bear_beta": 0.9},
             "upside_capture": None, "downside_capture": None}
    data = {"ticker": "T", "current_price": 1.0, "stats": {}, "charts": {},
            "institutional": {"macro_conditioning": macro}}
    with app_module.app.test_request_context("/analytics?ticker=T"):
        html = render_template("analytics.html", data=data, ticker="T")
    assert "insufficient history" in html


# --- Up/down betas are labelled as market-direction, not macro regime ---------
def test_dual_betas_report_sample_sizes_and_are_relabelled():
    import macro_engine
    rng = np.random.default_rng(7)
    bench = pd.Series(rng.normal(0.0004, 0.01, 400))
    out = macro_engine.calculate_regime_conditional_betas(bench * 1.2, bench)
    assert out["bull_days"] + out["bear_days"] == 400
    assert glossary.GLOSSARY["dual_betas"]["term"] == "Up/Down-Market Betas"


def _fred_obs(dates, values):
    return [{"date": d.strftime("%Y-%m-%d"), "value": str(v)} for d, v in zip(dates, values)]


def test_macro_regime_betas_condition_on_actual_fred_regimes():
    import macro_engine
    idx = pd.bdate_range("2024-01-01", periods=400)
    bench = pd.Series(rng_example := np.random.default_rng(7).normal(0.0004, 0.01, 400), index=idx)
    dff = [3.0] * 200 + [round(3.0 + 0.01 * i, 4) for i in range(200)]   # tightening phase
    curve = [0.5] * 200 + [-0.3] * 200                                   # then inverted
    out = macro_engine.calculate_macro_regime_betas(
        bench * 1.2, bench, _fred_obs(idx, dff), _fred_obs(idx, curve))
    assert out is not None
    assert out["curve_inverted"]["n"] + out["curve_steep"]["n"] == 400
    assert out["curve_inverted"]["beta"] is not None and out["curve_steep"]["beta"] is not None
    assert out["rate_tightening"]["n"] > 100 and out["rate_tightening"]["beta"] is not None
    assert out["rate_easing"]["beta"] is None  # no easing phase in the fixture


def test_regime_beta_buckets_under_minimum_are_withheld_with_n():
    import macro_engine
    idx = pd.bdate_range("2024-01-01", periods=400)
    bench = pd.Series(np.random.default_rng(7).normal(0.0004, 0.01, 400), index=idx)
    dff = [3.0] * 380 + [3.5] * 20  # only 19 tightening days
    curve = [0.5] * 200 + [-0.3] * 200
    out = macro_engine.calculate_macro_regime_betas(
        bench * 1.2, bench, _fred_obs(idx, dff), _fred_obs(idx, curve), min_obs=60)
    assert out["rate_tightening"] == {"beta": None, "n": 20}
    assert out["curve_inverted"]["beta"] is not None


def test_regime_betas_declared_unavailable_without_fred():
    import macro_engine
    px = _prices()
    with patch.object(macro_engine, "fetch_fred_series", return_value=[]):
        rep = macro_engine.get_macro_financial_report(px, px)
    assert all(b["beta"] is None and b["n"] == 0 for b in rep.macro_regime_betas.values())


def test_regime_beta_panel_renders_buckets_and_declared_states():
    from flask import render_template
    macro = {"regime": None, "fed_funds_rate": None, "yield_curve_2s10s": None,
             "is_inverted": None, "cpi_yoy": None,
             "betas": {"bull_beta": 1.1, "bear_beta": 0.9, "bull_days": 210, "bear_days": 188},
             "regime_betas": {"curve_inverted": {"beta": 1.25, "n": 240},
                              "curve_steep": {"beta": None, "n": 12},
                              "rate_tightening": {"beta": None, "n": 0},
                              "rate_pause": {"beta": 0.9, "n": 300},
                              "rate_easing": {"beta": None, "n": 0}},
             "upside_capture": 100.0, "downside_capture": 100.0}
    data = {"ticker": "T", "current_price": 1.0, "stats": {}, "charts": {},
            "institutional": {"macro_conditioning": macro}}
    with app_module.app.test_request_context("/analytics?ticker=T"):
        html = render_template("analytics.html", data=data, ticker="T")
    assert "Up/Down-Market Betas" in html and "Up-market (SPY &gt; 0)" in html
    assert "insufficient sample (n=12)" in html and "FRED history unavailable" in html
