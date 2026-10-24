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
