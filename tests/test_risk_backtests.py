"""Epic 6 core (6.1-6.4): tail-honest VaR/ES, EWMA/GARCH, backtests, horizons."""
import numpy as np
import pytest

import risk_backtests as rb


def _iid(n=1000, sigma=0.01, seed=2):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, sigma, n)


# --- 6.1 tail honesty -----------------------------------------------------------
def test_tail_counts_bootstrap_ci_and_low_confidence_flag():
    r = _iid(1000)
    out = rb.tail_honest_var(r)
    v95, v99 = out["var_95"], out["var_99"]
    assert abs(v95["var"] - _iid(1000).std() * -1.645) < 0.15 * v95["var"] + 0.01
    assert v95["tail_obs"] == pytest.approx(50, abs=15)
    assert v99["tail_obs"] == pytest.approx(10, abs=6)
    lo, hi = v95["ci95"]
    assert lo <= v95["var"] <= hi
    # 400 obs: the 99% quantile rests on ~4 tail points -> flagged
    assert rb.tail_honest_var(_iid(400))["var_99"]["low_confidence"] is True
    assert rb.tail_honest_var(r)["var_95"]["low_confidence"] is False
    # ES at least as bad as VaR
    assert v99["expected_shortfall"] <= v99["var"]
    assert rb.tail_honest_var(_iid(50)) == {}


# --- 6.2 EWMA / GARCH ------------------------------------------------------------
def test_ewma_formula_matches_hand_calculation():
    r = np.array([0.01, -0.02, 0.01])
    lam = 0.94
    var = float(r.var())  # implementation seeds from the sample variance
    for x in r:
        var = lam * var + (1 - lam) * x * x
    assert abs(rb.ewma_vol_forecast(r, lam) - np.sqrt(var)) < 1e-12


def test_garch_parameters_recovered_from_simulated_garch():
    rng = np.random.default_rng(8)
    omega, a, b = 2e-6, 0.08, 0.90
    n, burn = 1200, 300
    r = np.empty(n + burn)
    h = 1e-4
    for t in range(n + burn):
        r[t] = np.sqrt(h) * rng.standard_normal()
        h = omega + a * r[t] ** 2 + b * h
    fit = rb.fit_garch11(r[burn:])
    assert fit is not None
    assert 0.02 < fit["alpha"] < 0.25
    assert 0.70 < fit["beta"] <= 0.999
    assert fit["persistence"] < 1.0
    assert fit["next_var"] > 0


def test_garch_insufficient_history_declared():
    assert rb.fit_garch11(_iid(200)) is None


def test_forecast_evaluation_mechanics():
    r = _iid(300)
    r2 = r ** 2
    perfect = rb.forecast_evaluation(r, r2)
    assert perfect["mse"] == pytest.approx(0.0, abs=1e-20)
    assert perfect["qlike"] == pytest.approx(0.0, abs=1e-10)
    out = rb.forecast_evaluation(r, np.full(len(r), r2.var()))
    assert np.isfinite(out["qlike"]) and np.isfinite(out["qlike_naive"])


# --- 6.3 backtests ---------------------------------------------------------------
def test_kupiec_passes_calibrated_and_rejects_misspecified():
    # exactly 5% violations against a 95% claim -> perfect fit
    e = np.zeros(1000); e[np.linspace(0, 999, 50, dtype=int)] = 1
    ok = rb.kupiec_pof(e, 0.95)
    assert ok["p_value"] > 0.9 and ok["violations"] == 50
    # claim 99% coverage but violate 10% of the time -> rejected
    bad = np.zeros(500); bad[:50] = 1
    rej = rb.kupiec_pof(bad, 0.99)
    assert rej["p_value"] < 0.001


def test_christoffersen_rejects_clusters_not_even_spread():
    clustered = np.zeros(1000); clustered[950:] = 1
    assert rb.christoffersen_ind(clustered)["p_value"] < 0.05
    rng = np.random.default_rng(0)  # random, not periodic: periodic placement is itself dependence
    even = np.zeros(1000); even[rng.choice(1000, 50, replace=False)] = 1
    assert rb.christoffersen_ind(even)["p_value"] > 0.05
    assert rb.christoffersen_ind(np.zeros(100))["p_value"] is None


def test_var_backtest_traffic_light():
    r = _iid(1200, seed=5)
    out = rb.var_backtest(r, 0.95)
    assert out["sufficient"] is True
    assert out["traffic"] == "green"
    # regime shift: 99% VaR calibrated on calm data then vol doubles -> violations cluster
    shifted = np.concatenate([_iid(600, 0.008, 3), _iid(600, 0.02, 4)])
    bad = rb.var_backtest(shifted, 0.99)
    assert bad["traffic"] != "green"
    assert rb.var_backtest(_iid(100), 0.95)["sufficient"] is False


# --- 6.4 horizons ---------------------------------------------------------------
def test_horizon_var_on_overlapping_windows_matches_hand_calculation():
    r = np.full(260, -0.01)
    out = rb.horizon_var(r)
    assert out["h10_var_95"]["var"] == pytest.approx(-0.10)
    assert out["h21_var_99"]["windows"] == 240
    assert rb.horizon_var(_iid(100)) == {}


def test_position_scaling():
    assert rb.scale_to_position(-2.0, -3.5, 10_000) == {
        "position_usd": 10000.0, "var_usd": -200.0, "es_usd": -350.0}
    assert rb.scale_to_position(-2.0, -3.5, 0) == {}


# --- 6.5 Historical stress tests (#145) ------------------------------------------
def test_historical_stress_scenarios_direct_history_and_factor_implied():
    import pandas as pd
    # Stock with history covering COVID crash (2020-02 to 2020-03)
    idx = pd.bdate_range("2020-01-01", "2020-09-01")
    # Base price 100, drops to 65 during March 2020, recovers to 85 by June
    prices = pd.Series(100.0, index=idx)
    prices.loc["2020-03-23":] = 65.0
    prices.loc["2020-06-25":] = 85.0
    df = pd.DataFrame({"close": prices, "high": prices * 1.01, "low": prices * 0.99, "volume": 1000}, index=idx)

    # 1. Direct historical replay for COVID-19 Crash
    res = rb.historical_stress_scenarios(df, factor_loadings={"Mkt-RF": 1.1, "HML": -0.2})
    covid = next(s for s in res if s["id"] == "covid_crash_2020")
    assert covid["method"] == "historical replay"
    assert covid["drop_pct"] == pytest.approx(-35.0, abs=1.0)
    assert covid["recovery_pct"] is not None

    # 2. Factor-implied fallback for 2008 GFC (no 2008 price history)
    gfc = next(s for s in res if s["id"] == "gfc_2008")
    assert gfc["method"] == "factor-implied (no direct history)"
    assert gfc["drop_pct"] is not None
    assert gfc["drop_pct"] < 0  # Market shock is negative

    # 3. Insufficient history and no factor loadings
    res_no_factors = rb.historical_stress_scenarios(df, factor_loadings=None)
    gfc_empty = next(s for s in res_no_factors if s["id"] == "gfc_2008")
    assert gfc_empty["method"] == "insufficient history"
    assert gfc_empty["drop_pct"] is None


# --- 6.6 Rolling Sharpe with Lo (2002) SE & PSR (#146) ---------------------------
def test_rolling_sharpe_with_lo_se_and_psr():
    import pandas as pd
    idx = pd.bdate_range("2023-01-01", periods=400)
    rng = np.random.default_rng(42)
    # Autocorrelated returns
    e = rng.normal(0.0008, 0.01, 400)
    for i in range(1, 400):
        e[i] += 0.3 * e[i-1]
    returns = pd.Series(e, index=idx)

    out = rb.rolling_sharpe_with_lo_se(returns, window=252)
    assert len(out["dates"]) == 400
    valid_sharpes = [s for s in out["sharpe"] if s is not None]
    assert len(valid_sharpes) == 400 - 252 + 1
    assert out["latest_sharpe"] is not None
    assert out["latest_se"] is not None and out["latest_se"] > 0
    assert out["psr"] is not None and 0.0 <= out["psr"] <= 1.0
    assert out["autocorrelation_rho1"] is not None

    # Confidence intervals bracket the estimate
    latest_u = out["ci_upper"][-1]
    latest_l = out["ci_lower"][-1]
    assert latest_l < out["latest_sharpe"] < latest_u

    # Short history returns empty
    short_out = rb.rolling_sharpe_with_lo_se(returns.iloc[:100], window=252)
    assert short_out["latest_sharpe"] is None


# --- 6.7 Bar-range Volume Profile (#147) ----------------------------------------
def test_bar_range_volume_profile_distributes_across_spread():
    import pandas as pd
    idx = pd.bdate_range("2025-01-01", periods=50)
    # Bars oscillating around 100 with [95, 105] ranges, highest volume at center
    vols = np.concatenate([np.linspace(100, 1000, 25), np.linspace(1000, 100, 25)])
    df = pd.DataFrame({
        "open": np.full(50, 100.0),
        "high": np.full(50, 105.0),
        "low": np.full(50, 95.0),
        "close": np.full(50, 100.0),
        "volume": vols,
    }, index=idx)

    vp = rb.bar_range_volume_profile(df, n_bins=20, years=2)
    assert vp["n_bins"] == 20
    assert len(vp["bin_centers"]) == 20
    assert len(vp["volumes"]) == 20
    assert sum(vp["volumes"]) == pytest.approx(vols.sum(), rel=1e-2)
    # POC is within the spread
    assert 95.0 <= vp["poc_price"] <= 105.0
    # Multiple bins received volume (not just the single bin at close=100)
    non_zero_bins = sum(1 for v in vp["volumes"] if v > 0)
    assert non_zero_bins >= 15
