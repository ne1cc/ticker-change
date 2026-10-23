"""Fama-French factor ingest, regression engine, and beta hygiene.

1. Ingest: daily Ken French FF5 + momentum factors (zipped CSVs), parsed and
   cached in SQLite; French data lags live markets by weeks, so every
   regression is displayed with the factor file's as-of date and an offline
   machine serves the last cached copy.
2. Regression: excess-return regression on MKT-RF, SMB, HML, RMW, CMA, MOM
   with Newey-West HAC standard errors — exposures with t-stats, R-squared,
   annualised alpha with t-stat, residual (idiosyncratic) return and vol.
3. Beta hygiene: Dimson beta for asynchronous trading, Blume/Vasicek
   shrinkage, and a rolling beta with a 95% confidence band.

All factors are decimal daily returns (French publishes percent); the
risk-free rate comes from the factor file, never from zero.
"""
from __future__ import annotations

import io
import re
import zipfile
from typing import Optional

import numpy as np
import pandas as pd

import db
import providers

FF5_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
MOM_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"
CACHE_PROVIDER = "french"
CACHE_KEY = "ff5_mom_daily"
_MIN_MERGED_ROWS = 500  # sanity floor for a real French download
REFRESH_TTL_HOURS = 24 * 7  # files update monthly; weekly refresh keeps as-of tight

SIX_FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]

_DATE_RE = re.compile(r"^\s*(\d{8})\s+")


def parse_ff_daily_csv(raw: bytes, columns: list[str]) -> pd.DataFrame:
    """Parse a Ken French daily-factor CSV (zipped): preamble lines, a header
    row, whitespace-separated rows of `yyyymmdd <pct>...`, and a footer.
    Returns decimal-return columns indexed by date."""
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        text = zf.read(name).decode("latin-1")

    rows = []
    for line in text.splitlines():
        m = _DATE_RE.match(line)
        if not m:
            if rows:
                break  # footer reached
            continue
        parts = line.split()
        if len(parts) < len(columns) + 1:
            continue
        date = pd.Timestamp(datetime_from_compact(parts[0]))
        values = [float(p) / 100.0 for p in parts[1:len(columns) + 1]]
        rows.append((date, *values))
    if not rows:
        raise ValueError("no factor rows found in French CSV")
    df = pd.DataFrame(rows, columns=["date", *columns]).set_index("date")
    return df


def datetime_from_compact(s: str):
    from datetime import datetime
    return datetime.strptime(s, "%Y%m%d")


def _download(url: str) -> Optional[bytes]:
    try:
        res = providers._SESSION.get(url, timeout=20)
        if res.status_code == 200 and res.content:
            return res.content
        print(f"French download {url} -> HTTP {res.status_code}")
    except Exception as e:
        print(f"French download failed for {url}: {e}")
    return None


def _stale_cache() -> Optional[dict]:
    """Last cached payload regardless of age — offline machines still work."""
    with db.get_conn() as conn:
        row = conn.execute(
            "SELECT payload FROM api_cache WHERE provider = ? AND key = ?",
            (CACHE_PROVIDER, CACHE_KEY),
        ).fetchone()
    if row is None:
        return None
    try:
        import json
        return json.loads(row["payload"])
    except Exception:
        return None


def _payload_to_df(payload: dict) -> pd.DataFrame:
    df = pd.DataFrame(payload["rows"], columns=["date", *SIX_FACTORS, "RF"])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def load_factor_returns(max_age_hours: float = REFRESH_TTL_HOURS):
    """Daily FF5+momentum factor returns (decimal) -> (df | None, as_of | None).

    Serves SQLite cache when fresh; otherwise downloads both files, merges,
    and caches. A failed download falls back to the last cached copy even if
    expired, and to None only when nothing was ever cached.
    """
    cached = db.cache_get(CACHE_PROVIDER, CACHE_KEY, max_age_hours)
    if cached is not None:
        return _payload_to_df(cached), cached.get("as_of")

    ff5_raw, mom_raw = _download(FF5_URL), _download(MOM_URL)
    if ff5_raw and mom_raw:
        try:
            ff5 = parse_ff_daily_csv(ff5_raw, ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"])
            mom = parse_ff_daily_csv(mom_raw, ["Mom", "RF_x"])
            mom = mom[["Mom"]]
            merged = ff5.join(mom, how="inner").dropna()
            if len(merged) >= _MIN_MERGED_ROWS:
                merged = merged[SIX_FACTORS + ["RF"]]
                as_of = merged.index.max().strftime("%Y-%m-%d")
                import json
                payload = {"as_of": as_of,
                           "rows": [[d.strftime("%Y-%m-%d"), *row] for d, row in
                                    zip(merged.index, merged.values)]}
                db.cache_set(CACHE_PROVIDER, CACHE_KEY, payload)
                return merged, as_of
        except Exception as e:
            print(f"French parse failed: {e}")

    stale = _stale_cache()
    if stale is not None:
        return _payload_to_df(stale), stale.get("as_of")
    return None, None


# --- Regression (issue 3.2) ---------------------------------------------------

def _nw_lags(n: int) -> int:
    return max(1, int(4 * (n / 100.0) ** (2.0 / 9.0)))


def newey_west_se(X: np.ndarray, u: np.ndarray, lags: int) -> np.ndarray:
    """Newey-West HAC standard errors for OLS coefficients (Bartlett kernel)."""
    n, k = X.shape
    bread = np.linalg.pinv(X.T @ X)
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        G = Xu[l:].T @ Xu[:-l]
        S += w * (G + G.T)
    meat = bread @ S @ bread
    return np.sqrt(np.clip(np.diag(meat), 0.0, None))


def compute_factor_exposures(
    stock_df: pd.DataFrame,
    factors_df: pd.DataFrame,
    factor_as_of: Optional[str] = None,
    min_obs: int = 250,
) -> dict:
    """OLS + Newey-West regression of the stock's daily excess return on the
    six factors. Returns an insufficient-state dict below `min_obs`."""
    col = "close" if "close" in stock_df.columns else "Close"
    ret = stock_df[col].pct_change().dropna()
    joined = ret.rename("ret").to_frame().join(factors_df, how="inner").dropna()
    n = len(joined)
    if n < min_obs:
        return {"insufficient": True, "n": n, "min_obs": min_obs,
                "factor_as_of": factor_as_of}

    y = joined["ret"].values - joined["RF"].values
    X = np.column_stack([np.ones(n), joined[SIX_FACTORS].values])
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    u = y - X @ coeffs
    se = newey_west_se(X, u, _nw_lags(n))

    from math import sqrt
    r_squared = float(1.0 - u.var() / y.var()) if y.var() > 0 else 0.0
    alpha_daily = float(coeffs[0])
    alpha_ann_pct = ((1.0 + alpha_daily) ** 252.0 - 1.0) * 100.0
    resid_series = pd.Series(u, index=joined.index)
    resid_vol_ann = float(u.std(ddof=1) * sqrt(252.0))
    growth = float((1.0 + resid_series).prod())
    resid_ret_ann_pct = (np.sign(growth) * (abs(growth) ** (252.0 / n) - 1.0)) * 100.0

    exposures = {}
    for i, name in enumerate(SIX_FACTORS, start=1):
        t = float(coeffs[i] / se[i]) if se[i] > 0 else 0.0
        exposures[name] = {
            "beta": round(float(coeffs[i]), 4),
            "t_stat": round(t, 2),
            "ci95": [round(float(coeffs[i] - 1.96 * se[i]), 4),
                     round(float(coeffs[i] + 1.96 * se[i]), 4)],
        }

    return {
        "insufficient": False,
        "n": n,
        "start": joined.index.min().strftime("%Y-%m-%d"),
        "end": joined.index.max().strftime("%Y-%m-%d"),
        "factor_as_of": factor_as_of or (factors_df.index.max().strftime("%Y-%m-%d") if len(factors_df) else None),
        "exposures": exposures,
        "alpha_annualised_pct": round(alpha_ann_pct, 2),
        "alpha_t_stat": round(float(coeffs[0] / se[0]), 2) if se[0] > 0 else 0.0,
        "r_squared": round(r_squared, 4),
        "residual_vol_annualised_pct": round(resid_vol_ann * 100.0, 2),
        "residual_return_annualised_pct": round(float(resid_ret_ann_pct), 2),
    }


# --- Beta hygiene (issue 3.3) --------------------------------------------------

def dimson_beta(stock_ret: pd.Series, bench_ret: pd.Series, lag: int = 1) -> float:
    """Dimson (1979): regress on contemporaneous, lagged and led benchmark
    returns and sum the coefficients, recovering the true beta when the stock
    trades asynchronously."""
    df = pd.concat([stock_ret.rename("s"), bench_ret.rename("m")], axis=1).dropna()
    if len(df) < 30:
        return float("nan")
    cols = {"m0": df["m"]}
    for l in range(1, lag + 1):
        cols[f"m_lag{l}"] = df["m"].shift(l)
        cols[f"m_lead{l}"] = df["m"].shift(-l)
    X = pd.DataFrame(cols).dropna()
    y = df.loc[X.index, "s"]
    Xv = np.column_stack([np.ones(len(X)), X.values])
    coeffs, *_ = np.linalg.lstsq(Xv, y.values, rcond=None)
    return float(coeffs[1:].sum())


def blume_shrink(beta: float, a: float = 1.0 / 3.0, b: float = 2.0 / 3.0) -> float:
    """Blume (1971) adjustment toward the market prior of 1."""
    return b * beta + a * 1.0


def vasicek_shrink(beta: float, beta_se: float,
                   prior_mean: float = 1.0, prior_sd: float = 0.5) -> float:
    """Vasicek (1973) Bayesian shrinkage of a beta estimate toward a prior;
    the estimate is weighted by its own precision."""
    if not beta_se or beta_se <= 0:
        return float("nan")
    post = (beta / beta_se ** 2 + prior_mean / prior_sd ** 2) / (1.0 / beta_se ** 2 + 1.0 / prior_sd ** 2)
    return float(post)


def rolling_beta(stock_ret: pd.Series, bench_ret: pd.Series,
                 window: int = 252, step: int = 21) -> list[dict]:
    """Rolling OLS beta with 95% CI — [{end, beta, lo, hi}] at `step` spacing."""
    df = pd.concat([stock_ret.rename("s"), bench_ret.rename("m")], axis=1).dropna()
    out = []
    s_arr, m_arr = df["s"].values, df["m"].values
    idx = df.index
    for end in range(window, len(df) + 1, step):
        y = s_arr[end - window:end]
        x = m_arr[end - window:end]
        var_x = x.var()
        if var_x <= 0:
            continue
        cov = np.cov(y, x)[0, 1]
        beta = cov / var_x
        resid = y - beta * x
        se = float(np.sqrt(resid.var(ddof=2) / (window * var_x))) if window > 2 else float("nan")
        out.append({
            "end": pd.Timestamp(idx[end - 1]).strftime("%Y-%m-%d"),
            "beta": round(float(beta), 3),
            "lo": round(float(beta - 1.96 * se), 3),
            "hi": round(float(beta + 1.96 * se), 3),
        })
    return out
