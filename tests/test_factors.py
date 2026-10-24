"""Epic 3: French factor ingest, HAC regression, and beta hygiene."""
import io
import zipfile

import numpy as np
import pandas as pd
import pytest

import factors


def _zip_csv(text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("F-F_Whatever_Daily_CSV.CSV", text)
    return buf.getvalue()


FF5_TEXT = (
    "\r\n\r\n"
    "  Copyright 2026 Kenneth French\r\n"
    "\r\n"
    "  Date        Mkt-RF       SMB      HML       RMW       CMA        RF\r\n"
    "\r\n"
    "  20240102     1.25    -0.31     0.44     0.12    -0.07     0.38\r\n"
    "  20240103    -0.62     0.18    -0.21    -0.05     0.03     0.38\r\n"
    "  20240104     0.10     0.02     0.11     0.00     0.01     0.37\r\n"
    "\r\n\r\n"
    "  Copyright 2026 Kenneth French\r\n"
)

MOM_TEXT = (
    "\r\n\r\n"
    "  Copyright 2026 Kenneth French\r\n"
    "\r\n"
    "  Date        Mom        RF\r\n"
    "\r\n"
    "  20240102     0.71     0.38\r\n"
    "  20240103    -0.44     0.38\r\n"
    "  20240104     0.05     0.37\r\n"
    "\r\n\r\n"
    "  Copyright 2026 Kenneth French\r\n"
)


def _factor_frame(n=800, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    f = pd.DataFrame(rng.normal(0.0004, 0.01, (n, 6)), index=idx, columns=factors.SIX_FACTORS)
    f["RF"] = 0.02 / 252.0
    return f


# --- 3.1 parser ---------------------------------------------------------------
def test_parser_handles_real_french_layout():
    df = factors.parse_ff_daily_csv(_zip_csv(FF5_TEXT),
                                    ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"])
    assert len(df) == 3
    assert df.index[0] == pd.Timestamp("2024-01-02")
    assert abs(df["Mkt-RF"].iloc[0] - 0.0125) < 1e-12  # percent in file, decimal out
    assert abs(df["RF"].iloc[2] - 0.0037) < 1e-12


def test_load_serves_cache_then_stale_when_offline(tmp_path, monkeypatch):
    import db as db_module
    monkeypatch.setattr(db_module, "DB_PATH", str(tmp_path / "t.db"))
    db_module.init_db()
    monkeypatch.setattr(factors, "_MIN_MERGED_ROWS", 3)

    payloads = {"ff5": _zip_csv(FF5_TEXT), "mom": _zip_csv(MOM_TEXT)}
    monkeypatch.setattr(factors, "_download",
                        lambda url: payloads["ff5"] if "5_Factors" in url else payloads["mom"])
    df, as_of = factors.load_factor_returns()
    assert as_of == "2024-01-04" and list(df.columns) == factors.SIX_FACTORS + ["RF"]

    # Offline (download returns None), cache expired -> stale copy still serves
    monkeypatch.setattr(factors, "_download", lambda url: None)
    df2, as_of2 = factors.load_factor_returns(max_age_hours=0)
    assert as_of2 == "2024-01-04" and len(df2) == 3

    # Never cached and offline -> declared None
    monkeypatch.setattr(db_module, "DB_PATH", str(tmp_path / "empty.db"))
    db_module.init_db()
    assert factors.load_factor_returns(max_age_hours=0) == (None, None)


# --- 3.2 regression ------------------------------------------------------------
def test_regression_recovers_known_betas_and_alpha():
    f = _factor_frame()
    true_betas = np.array([1.0, 0.5, -0.3, 0.2, 0.4, 0.1])
    rng = np.random.default_rng(11)
    noise = rng.normal(0, 0.004, len(f))
    rets = f["RF"] + f[factors.SIX_FACTORS].values @ true_betas + 0.0003 + noise
    stock_df = pd.DataFrame({"close": 100 * (1 + rets).cumprod()}, index=f.index)

    out = factors.compute_factor_exposures(stock_df, f, factor_as_of="2025-01-02")
    assert out["insufficient"] is False
    for name, b in zip(factors.SIX_FACTORS, true_betas):
        assert abs(out["exposures"][name]["beta"] - b) < 0.15, name
        lo, hi = out["exposures"][name]["ci95"]
        assert lo < hi
    assert abs(out["alpha_annualised_pct"] - (1.0003 ** 252 - 1) * 100) < 4.0
    assert 0.7 < out["r_squared"] < 1.0
    assert abs(out["residual_vol_annualised_pct"] - 0.004 * np.sqrt(252) * 100) < 1.0
    assert out["factor_as_of"] == "2025-01-02"


def test_hac_se_differs_from_ols_on_autocorrelated_residuals():
    rng = np.random.default_rng(5)
    n = 400
    x = np.empty(n)  # autocorrelated regressor: cross terms give NW real weight
    x[0] = rng.normal(0, 1)
    for t in range(1, n):
        x[t] = 0.7 * x[t - 1] + rng.normal(0, 1)
    u = np.empty(n)
    u[0] = rng.normal(0, 1)
    for t in range(1, n):
        u[t] = 0.8 * u[t - 1] + rng.normal(0, 1)
    y = 0.5 * x + u
    X = np.column_stack([np.ones(n), x])
    nw = factors.newey_west_se(X, u, factors._nw_lags(n))
    ols = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) * u.var(ddof=2))
    assert nw[1] > ols[1] * 1.2  # autocorrelation inflates the HAC band


def test_insufficient_history_is_declared():
    f = _factor_frame(120)
    stock_df = pd.DataFrame({"close": 100 * (1 + f["Mkt-RF"]).cumprod()}, index=f.index)
    out = factors.compute_factor_exposures(stock_df, f)
    assert out["insufficient"] is True and out["n"] == 119


# --- 3.3 beta hygiene -----------------------------------------------------------
def _nonsynchronous_pair(n=750, true_beta=1.2, seed=9):
    rng = np.random.default_rng(seed)
    m = pd.Series(rng.normal(0.0003, 0.01, n))
    s = true_beta * (0.5 * m + 0.5 * m.shift(1).fillna(0)) + rng.normal(0, 0.002, n)
    return s.iloc[1:], m.iloc[1:]


def test_dimson_beta_recovers_truth_that_ols_misses():
    s, m = _nonsynchronous_pair()
    ols = float(np.cov(s, m)[0, 1] / m.var())
    dimson = factors.dimson_beta(s, m)
    assert abs(dimson - 1.2) < abs(ols - 1.2)
    assert abs(dimson - 1.2) < 0.2


def test_shrunk_betas_lie_between_raw_and_prior():
    raw, se = 1.6, 0.25
    assert 1.0 < factors.blume_shrink(raw) < raw
    tight = factors.vasicek_shrink(raw, se)
    loose = factors.vasicek_shrink(raw, 5.0)
    assert 1.0 < tight < raw
    assert abs(loose - 1.0) < abs(tight - 1.0)  # noisier estimate shrinks further


def test_rolling_beta_band_contains_true_beta():
    rng = np.random.default_rng(4)
    idx = pd.bdate_range("2022-01-03", periods=900)
    m = pd.Series(rng.normal(0, 0.01, 900), index=idx)
    s = pd.Series(1.1 * m.values + rng.normal(0, 0.002, 900), index=idx)
    rows = factors.rolling_beta(s, m)
    assert len(rows) >= 20
    inside = sum(1 for r in rows if r["lo"] <= 1.1 <= r["hi"])
    assert inside / len(rows) > 0.8
