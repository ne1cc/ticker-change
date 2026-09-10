"""Volatility Term Structure, 25-Delta Skew, and 3D Volatility Surface Engine."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .models import VolTermStructure


def compute_vol_term_structure(
    chains_df: pd.DataFrame,
    spot: float,
) -> VolTermStructure:
    """Compute ATM IV term structure across all available expirations.

    Calculates term structure slope and tags Contango vs Backwardation regime.
    """
    if chains_df is None or chains_df.empty:
        return VolTermStructure()

    df = chains_df.copy()
    if "expiration" not in df.columns:
        df["expiration"] = df["dte"].astype(str)

    expirations: List[str] = []
    dtes: List[int] = []
    atm_ivs: List[float] = []

    for exp, group in df.groupby("expiration", sort=False):
        valid = group[group["iv"].between(0.005, 4.0)]
        if valid.empty:
            continue

        dte = int(valid["dte"].iloc[0])
        # Find calls and puts closest to spot
        calls = valid[valid["cp"] == "C"]
        puts = valid[valid["cp"] == "P"]

        c_iv = None
        p_iv = None
        if not calls.empty:
            c_idx = (calls["strike"] - spot).abs().idxmin()
            c_iv = float(calls.loc[c_idx, "iv"])
        if not puts.empty:
            p_idx = (puts["strike"] - spot).abs().idxmin()
            p_iv = float(puts.loc[p_idx, "iv"])

        if c_iv is not None and p_iv is not None:
            atm_iv = round(float((c_iv + p_iv) / 2.0 * 100.0), 2)
        elif c_iv is not None:
            atm_iv = round(float(c_iv * 100.0), 2)
        elif p_iv is not None:
            atm_iv = round(float(p_iv * 100.0), 2)
        else:
            continue

        expirations.append(str(exp))
        dtes.append(dte)
        atm_ivs.append(atm_iv)

    # Sort chronologically by DTE
    if dtes:
        sorted_pairs = sorted(zip(dtes, expirations, atm_ivs), key=lambda x: x[0])
        dtes = [x[0] for x in sorted_pairs]
        expirations = [x[1] for x in sorted_pairs]
        atm_ivs = [x[2] for x in sorted_pairs]

    # Calculate Slope & Term Structure Regime
    slope = 0.0
    regime: Any = "flat"
    if len(dtes) >= 2:
        # Linear fit of ATM IV vs sqrt(DTE)
        sq_dtes = np.sqrt(np.array(dtes, dtype=np.float64))
        poly = np.polyfit(sq_dtes, np.array(atm_ivs), 1)
        slope = round(float(poly[0]), 3)
        if slope > 0.35:
            regime = "contango"
        elif slope < -0.35:
            regime = "backwardation"
        else:
            regime = "flat"

    # Compute 25-Delta Skew & Risk Reversal (on front/second expiration)
    skew_25d = 1.0
    risk_reversal = 0.0
    if len(expirations) > 0:
        front_exp = expirations[0] if dtes[0] >= 3 or len(expirations) == 1 else expirations[1]
        front_group = df[df["expiration"] == front_exp]
        skew_25d, risk_reversal = compute_25d_skew(front_group, spot)

    # Build 3D Surface Grid
    surface_grid = build_surface_grid(df, spot)

    return VolTermStructure(
        expirations=expirations,
        dtes=dtes,
        atm_ivs=atm_ivs,
        slope=slope,
        regime=regime,
        skew_25d=skew_25d,
        risk_reversal=risk_reversal,
        surface_grid=surface_grid,
    )


def compute_25d_skew(df: pd.DataFrame, spot: float) -> Tuple[float, float]:
    """Calculate 25-Delta Put/Call Skew ratio and Risk Reversal spread."""
    if df.empty:
        return 1.0, 0.0

    calls = df[(df["cp"] == "C") & (df["iv"] > 0.01)].copy()
    puts = df[(df["cp"] == "P") & (df["iv"] > 0.01)].copy()

    if calls.empty or puts.empty:
        return 1.0, 0.0

    # Approximate 25-delta strike if delta column not present
    # 25-delta call is ~spot * exp(0.67 * iv * sqrt(t))
    c_iv_25 = None
    p_iv_25 = None

    if "delta" in calls.columns and calls["delta"].notna().any():
        c_25 = calls.iloc[(calls["delta"] - 0.25).abs().argsort()[:1]]
        if not c_25.empty:
            c_iv_25 = float(c_25["iv"].iloc[0])
    if "delta" in puts.columns and puts["delta"].notna().any():
        p_25 = puts.iloc[(puts["delta"] - (-0.25)).abs().argsort()[:1]]
        if not p_25.empty:
            p_iv_25 = float(p_25["iv"].iloc[0])

    # Fallback to moneyness approximation (0.95 moneyness for put, 1.05 for call)
    if c_iv_25 is None:
        c_target = spot * 1.05
        c_25 = calls.iloc[(calls["strike"] - c_target).abs().argsort()[:1]]
        if not c_25.empty:
            c_iv_25 = float(c_25["iv"].iloc[0])

    if p_iv_25 is None:
        p_target = spot * 0.95
        p_25 = puts.iloc[(puts["strike"] - p_target).abs().argsort()[:1]]
        if not p_25.empty:
            p_iv_25 = float(p_25["iv"].iloc[0])

    if c_iv_25 and p_iv_25 and c_iv_25 > 0:
        skew_ratio = round(p_iv_25 / c_iv_25, 3)
        rr = round((c_iv_25 - p_iv_25) * 100.0, 2)
        return skew_ratio, rr

    return 1.0, 0.0


def build_surface_grid(df: pd.DataFrame, spot: float, max_expirations: int = 8) -> Optional[Dict[str, Any]]:
    """Build a regular 2D grid of (Moneyness, DTE) with interpolated IV for 3D Surface."""
    if df.empty or spot <= 0:
        return None

    exps = sorted(df["dte"].unique())[:max_expirations]
    if len(exps) < 2:
        return None

    moneyness_axis = np.linspace(0.80, 1.20, 21)
    grid_iv = []

    for dte in exps:
        sub = df[(df["dte"] == dte) & (df["iv"].between(0.01, 3.0))]
        if len(sub) < 3:
            grid_iv.append([np.nan] * len(moneyness_axis))
            continue

        # Sort by moneyness
        m = (sub["strike"] / spot).to_numpy()
        iv = (sub["iv"] * 100.0).to_numpy()
        order = np.argsort(m)
        m_sorted = m[order]
        iv_sorted = iv[order]

        # Interpolate along moneyness axis
        iv_interp = np.interp(moneyness_axis, m_sorted, iv_sorted, left=iv_sorted[0], right=iv_sorted[-1])
        grid_iv.append([round(float(x), 2) for x in iv_interp])

    return {
        "moneyness": [round(float(m), 2) for m in moneyness_axis],
        "dtes": [int(d) for d in exps],
        "iv_matrix": grid_iv,
    }
