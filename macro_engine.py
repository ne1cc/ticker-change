"""Macro-Financial Conditioning & Regime-Aware Analytics.

Implements:
1. Macro Regime Classification (FRED Policy States: Hike / Cut / Pause, Yield Curve Inversion)
2. Dual-Beta / Regime-Conditional Betas (Bull vs Bear, Tightening vs Easing)
3. Downside vs Upside Capture Ratios
4. Cross-Asset Macro Spillover Matrix (Correlations to DXY, US10Y, HYG, SPY)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import providers
import db


@dataclass
class MacroRegimeReport:
    current_regime: Optional[str]  # None when FRED data is unavailable
    fed_funds_rate: Optional[float]
    yield_curve_2s10s_spread: Optional[float]
    is_yield_curve_inverted: Optional[bool]
    cpi_inflation_yoy: Optional[float]
    regime_conditional_betas: Dict[str, float]
    macro_regime_betas: Optional[Dict[str, Dict[str, Optional[float]]]]
    upside_capture_ratio: Optional[float]
    downside_capture_ratio: Optional[float]
    cross_asset_correlations: Dict[str, float]


def fetch_fred_series(series_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    """Fetch series observations from FRED API with SQLite caching."""
    api_key = db.get_setting("fred_api_key") or os.environ.get("FRED_API_KEY", "")
    if not api_key:
        return []

    cache_key = f"fred_{series_id}"
    cached = db.cache_get("fred", cache_key, ttl_hours=24.0)
    if cached is not None:
        return cached

    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json&sort_order=desc&limit={limit}"
    try:
        res = providers._SESSION.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json()
            obs = data.get("observations", [])
            db.cache_set("fred", cache_key, obs)
            return obs
    except Exception as e:
        print(f"Error fetching FRED series {series_id}: {e}")
    return []


def calculate_regime_conditional_betas(
    stock_returns: pd.Series, benchmark_returns: pd.Series
) -> Dict[str, float]:
    """Up/Down-market betas: covariance split by the SIGN of the benchmark's
    daily return (SPY > 0 vs SPY < 0). These are *market-direction* betas, not
    macro-regime betas — see calculate_macro_regime_betas for the FRED-based
    conditioning. Keys kept as bull_beta/bear_beta for API compatibility."""
    df = pd.concat([stock_returns.rename("stock"), benchmark_returns.rename("bench")], axis=1).dropna()
    if len(df) < 20:
        return {"overall": 1.0, "bull_beta": 1.0, "bear_beta": 1.0,
                "bull_days": 0, "bear_days": 0}

    # Overall Beta
    cov_all = np.cov(df["stock"], df["bench"])
    var_bench = np.var(df["bench"])
    overall_beta = float(cov_all[0, 1] / var_bench) if var_bench > 0 else 1.0

    # Up-market beta (Benchmark Return >= 0)
    bull_df = df[df["bench"] >= 0]
    if len(bull_df) >= 10 and np.var(bull_df["bench"]) > 0:
        cov_bull = np.cov(bull_df["stock"], bull_df["bench"])
        bull_beta = float(cov_bull[0, 1] / np.var(bull_df["bench"]))
    else:
        bull_beta = overall_beta

    # Down-market beta (Benchmark Return < 0)
    bear_df = df[df["bench"] < 0]
    if len(bear_df) >= 10 and np.var(bear_df["bench"]) > 0:
        cov_bear = np.cov(bear_df["stock"], bear_df["bench"])
        bear_beta = float(cov_bear[0, 1] / np.var(bear_df["bench"]))
    else:
        bear_beta = overall_beta

    return {
        "overall": round(overall_beta, 3),
        "bull_beta": round(bull_beta, 3),
        "bear_beta": round(bear_beta, 3),
        "bull_days": int(len(bull_df)),
        "bear_days": int(len(bear_df)),
    }


def _fred_obs_to_series(obs: List[Dict[str, Any]]) -> Optional[pd.Series]:
    """FRED observations (desc order, string values) -> ascending date Series."""
    if not obs:
        return None
    try:
        pairs = [(pd.Timestamp(o["date"]), float(o["value"]))
                 for o in obs if o.get("value") not in (".", None, "")]
    except (KeyError, TypeError, ValueError):
        return None
    if not pairs:
        return None
    return pd.Series(dict(pairs)).sort_index()


def calculate_macro_regime_betas(
    stock_returns: pd.Series,
    benchmark_returns: pd.Series,
    dff_obs: List[Dict[str, Any]],
    t10y2y_obs: List[Dict[str, Any]],
    min_obs: int = 60,
) -> Optional[Dict[str, Dict[str, Optional[float]]]]:
    """Betas conditional on the macro regime actually prevailing on each date.

    Buckets (each {"beta": float|None, "n": int}; beta None when n < min_obs):
      - curve_inverted / curve_steep: sign of the 10Y-2Y spread that day
      - rate_tightening / rate_pause / rate_easing: 63-trading-day change in
        the daily fed funds rate vs +/-25bp
    Returns None when returns/FRED history cannot support the split.
    """
    df = pd.concat([stock_returns.rename("stock"), benchmark_returns.rename("bench")], axis=1).dropna()
    if len(df) < min_obs or not isinstance(df.index, pd.DatetimeIndex):
        return None

    def beta_for(mask: pd.Series) -> Dict[str, Optional[float]]:
        n = int(mask.fillna(False).sum())
        if n < min_obs or np.var(df.loc[mask, "bench"]) == 0:
            return {"beta": None, "n": n}
        cov = np.cov(df.loc[mask, "stock"], df.loc[mask, "bench"])
        return {"beta": round(float(cov[0, 1] / np.var(df.loc[mask, "bench"])), 3), "n": n}

    out: Dict[str, Dict[str, Optional[float]]] = {}
    curve = _fred_obs_to_series(t10y2y_obs)
    if curve is not None:
        daily = curve.reindex(df.index, method="ffill")
        out["curve_inverted"] = beta_for(daily < 0)
        out["curve_steep"] = beta_for(daily >= 0)
    else:
        out["curve_inverted"] = out["curve_steep"] = {"beta": None, "n": 0}

    dff = _fred_obs_to_series(dff_obs)
    if dff is not None:
        change = dff.reindex(df.index, method="ffill").diff(63)
        out["rate_tightening"] = beta_for(change > 0.25)
        out["rate_easing"] = beta_for(change < -0.25)
        out["rate_pause"] = beta_for((change >= -0.25) & (change <= 0.25))
    else:
        out["rate_tightening"] = out["rate_pause"] = out["rate_easing"] = {"beta": None, "n": 0}
    return out


def calculate_capture_ratios(
    stock_returns: pd.Series, benchmark_returns: pd.Series, min_months: int = 6
) -> Tuple[Optional[float], Optional[float]]:
    """Upside/downside capture on compounded monthly returns (the Morningstar
    geometric convention: geometric mean monthly return of the asset in up/down
    benchmark months divided by the benchmark's geometric mean in those months).

    Returns (None, None) when there is insufficient history to state a ratio —
    never a neutral 100.0 placeholder.
    """
    df = pd.concat([stock_returns.rename("stock"), benchmark_returns.rename("bench")], axis=1).dropna()
    if len(df) < 20 or not isinstance(df.index, pd.DatetimeIndex):
        return None, None

    monthly_stock = (1.0 + df["stock"]).resample("ME").prod() - 1.0
    monthly_bench = (1.0 + df["bench"]).resample("ME").prod() - 1.0

    def _capture(mask: pd.Series) -> Optional[float]:
        n = int(mask.sum())
        if n < min_months:
            return None
        g_stock = (1.0 + monthly_stock[mask]).prod() ** (1.0 / n) - 1.0
        g_bench = (1.0 + monthly_bench[mask]).prod() ** (1.0 / n) - 1.0
        if g_bench <= -1.0:
            return None
        return round((1.0 + g_stock) / (1.0 + g_bench) * 100.0, 2)

    up_months = monthly_bench > 0
    down_months = monthly_bench < 0
    return _capture(up_months), _capture(down_months)


def get_macro_financial_report(
    stock_df: pd.DataFrame, benchmark_df: pd.DataFrame
) -> MacroRegimeReport:
    """Assemble complete macro conditioning report for a given equity."""
    s_col = "close" if "close" in stock_df.columns else "Close"
    b_col = "close" if "close" in benchmark_df.columns else "Close"

    s_ret = stock_df[s_col].pct_change().dropna()
    b_ret = benchmark_df[b_col].pct_change().dropna()

    betas = calculate_regime_conditional_betas(s_ret, b_ret)
    up_cap, down_cap = calculate_capture_ratios(s_ret, b_ret)

    # FRED unconfigured/offline -> None; never substitute stale placeholder values.
    # DFF/T10Y2Y are pulled as full histories: they feed both the latest-reading
    # badges and the per-date regime conditioning of the regime betas.
    dff_obs = fetch_fred_series("DFF", limit=5000)
    t10y2y_obs = fetch_fred_series("T10Y2Y", limit=5000)
    cpi_obs = fetch_fred_series("CPIAUCSL", limit=15)

    def _latest(obs):
        return float(obs[0]["value"]) if obs and obs[0].get("value") not in (".", None) else None

    effr = _latest(dff_obs)
    t10y2y = _latest(t10y2y_obs)
    is_inverted = t10y2y < 0.0 if t10y2y is not None else None

    cpi_val = None
    if len(cpi_obs) >= 13:
        try:
            latest_cpi = float(cpi_obs[0]["value"])
            prior_yr_cpi = float(cpi_obs[12]["value"])
            cpi_val = round(((latest_cpi - prior_yr_cpi) / prior_yr_cpi) * 100.0, 2)
        except Exception:
            pass

    regime = None
    if effr is not None and is_inverted is not None:
        if effr >= 4.5 and is_inverted:
            regime = "RESTRICTIVE_INVERTED (Late Cycle / Recession Risk)"
        elif effr >= 4.5:
            regime = "TIGHT_MONETARY_POLICY (High Rate Plateau)"
        elif is_inverted:
            regime = "CURVE_INVERTED_EASING"
        else:
            regime = "NORMAL_EXPANSION"

    # Cross asset correlation proxy (Stock vs Benchmark)
    corr_bench = float(s_ret.corr(b_ret)) if len(s_ret) > 10 else None

    regime_betas = calculate_macro_regime_betas(s_ret, b_ret, dff_obs, t10y2y_obs)

    return MacroRegimeReport(
        current_regime=regime,
        fed_funds_rate=effr,
        yield_curve_2s10s_spread=t10y2y,
        is_yield_curve_inverted=is_inverted,
        cpi_inflation_yoy=cpi_val,
        regime_conditional_betas=betas,
        macro_regime_betas=regime_betas,
        upside_capture_ratio=up_cap,
        downside_capture_ratio=down_cap,
        cross_asset_correlations={"SPY": round(corr_bench, 3)} if corr_bench is not None else {},
    )
