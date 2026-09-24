"""Dealer Gamma Exposure (GEX), 0DTE volume weighting, dual conventions, and grid-revalued flip engine."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import norm

from .models import Convention, GexProfile

CONTRACT_MULTIPLIER = 100.0


def clean_chain(df: pd.DataFrame) -> pd.DataFrame:
    """Defensively filter and cast an options chain DataFrame.

    Filters out unquoted strikes where IV is uninformative or corrupts net GEX.
    Retains 0DTE strikes with zero open interest if volume > 0 (OCC OI is published overnight).
    Prefers rows with a live bid; when those are a small fraction of the chain
    (e.g. after-hours zero-bid quotes), keeps every row with valid IV plus
    open interest or volume instead of collapsing the strike ladder.
    """
    if df is None or df.empty:
        raise ValueError("Options chain is empty or None")

    req_cols = {"strike", "cp", "dte", "iv"}
    if not req_cols.issubset(set(df.columns)):
        raise ValueError(f"Chain missing required columns: {req_cols - set(df.columns)}")

    out = df.copy()
    out["strike"] = pd.to_numeric(out["strike"], errors="coerce")
    out["dte"] = pd.to_numeric(out["dte"], errors="coerce").fillna(0).astype(int)
    out["iv"] = pd.to_numeric(out["iv"], errors="coerce")
    out["cp"] = out["cp"].astype(str).str.upper().str.strip()

    for col in ["bid", "ask", "open_interest", "volume"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
        else:
            out[col] = 0.0

    # Defensive filtering
    quoted = out[
        out["cp"].isin(["C", "P"])
        & (out["strike"] > 0)
        & (out["dte"] >= 0)
        & (out["iv"].between(0.001, 5.0))
        & (out["bid"] > 0)
        & ((out["open_interest"] > 0) | (out["volume"] > 0))
    ].copy()

    # Fallback: after hours (or on sources that omit quotes) nearly every row
    # has bid=0, so the bid filter can collapse a full chain to a handful of
    # strikes. Prefer the quoted subset, but fall back to dropping only the
    # bid condition whenever that subset is a small fraction of what is
    # otherwise usable.
    usable = out[
        out["cp"].isin(["C", "P"])
        & (out["strike"] > 0)
        & (out["dte"] >= 0)
        & (out["iv"].between(0.001, 5.0))
        & ((out["open_interest"] > 0) | (out["volume"] > 0))
    ].copy()

    if quoted.empty or len(quoted) * 4 < len(usable):
        valid = usable
    else:
        valid = quoted

    if valid.empty:
        raise ValueError("Chain has no tradable rows after defensive cleaning")

    return valid


def _hedge_weight(df: pd.DataFrame) -> np.ndarray:
    """Hedge-weighting: OCC open interest, backfilled by volume where the
    source omits OI (0DTE contracts publish OI overnight; some payloads omit
    it entirely, which would otherwise zero out the whole GEX profile)."""
    volumes = df["volume"].to_numpy()
    ois = df["open_interest"].to_numpy()
    return np.maximum(volumes, ois).astype(np.float64)


def _dealer_sign(df: pd.DataFrame, convention: Convention, spot: float) -> np.ndarray:
    """Dealer positioning sign convention.

    - 'naive': Dealers long calls (+1.0), short puts (-1.0).
    - 'short_wings': Dealers short OTM options (puts below spot, calls above spot) (-1.0) and long ITM (+1.0).
    """
    cps = df["cp"].to_numpy()
    strikes = df["strike"].to_numpy()

    if convention == "naive":
        return np.where(cps == "C", 1.0, -1.0)
    elif convention == "short_wings":
        is_otm = ((cps == "P") & (strikes < spot)) | ((cps == "C") & (strikes > spot))
        return np.where(is_otm, -1.0, 1.0)
    else:
        raise ValueError(f"Unknown dealer sign convention: {convention}")


def _bs_gamma_matrix(
    s_grid: np.ndarray,
    k: np.ndarray,
    t_years: np.ndarray,
    sigma: np.ndarray,
    r: float = 0.045,
) -> np.ndarray:
    """Compute gamma across a 2D grid: s_grid (N, 1) vs contracts (1, M)."""
    s = s_grid[:, None]
    k = k[None, :]
    t = np.maximum(t_years[None, :], 1.0 / 365.0)
    sig = np.clip(sigma[None, :], 1e-3, 5.0)

    sq_t = np.sqrt(t)
    d1 = (np.log(s / k) + (r + 0.5 * sig * sig) * t) / (sig * sq_t)
    return norm.pdf(d1) / (s * sig * sq_t)


def find_gamma_flip(
    chain: pd.DataFrame,
    spot: float,
    *,
    convention: Convention = "naive",
    lo: float = 0.5,
    hi: float = 1.5,
    n: int = 201,
    r: float = 0.045,
) -> Optional[float]:
    """Find spot level where net GEX crosses zero via 201-point continuous grid revaluation.

    Revalues Black-Scholes gamma at every spot price rather than assuming static gamma.
    Returns None if net GEX maintains a single sign across the entire range (no flip).
    """
    df = clean_chain(chain)
    k = df["strike"].to_numpy()
    t = df["dte"].to_numpy().astype(np.float64) / 365.0
    iv = df["iv"].to_numpy()
    weights = _hedge_weight(df)
    signs = _dealer_sign(df, convention, spot)
    ws = weights * signs

    grid = spot * np.linspace(lo, hi, n)  # shape (n,)
    gammas = _bs_gamma_matrix(grid, k, t, iv, r)  # shape (n, M)

    # Dollar gamma per 1% move in $M
    gex_grid = (gammas * ws[None, :] * CONTRACT_MULTIPLIER * (grid[:, None] ** 2) * 0.01 / 1e6).sum(axis=1)

    # Search for zero-crossings
    crossings = []
    for i in range(n - 1):
        if gex_grid[i] == 0.0:
            crossings.append(float(grid[i]))
        elif gex_grid[i] * gex_grid[i + 1] < 0.0:
            # Linear interpolation for zero crossing
            frac = abs(gex_grid[i]) / (abs(gex_grid[i]) + abs(gex_grid[i + 1]))
            crossing = float(grid[i] + frac * (grid[i + 1] - grid[i]))
            crossings.append(crossing)

    if not crossings:
        return None

    # Pick the crossing closest to current spot
    return round(min(crossings, key=lambda c: abs(c - spot)), 2)


def compute_gex_profile(
    chain: pd.DataFrame,
    spot: float,
    *,
    convention: Convention = "naive",
    r: float = 0.045,
) -> GexProfile:
    """Compute complete dealer GEX profile across all strikes with walls and gamma flip."""
    df = clean_chain(chain)

    k = df["strike"].to_numpy()
    t = df["dte"].to_numpy().astype(np.float64) / 365.0
    iv = df["iv"].to_numpy()
    cps = df["cp"].to_numpy()
    weights = _hedge_weight(df)
    signs = _dealer_sign(df, convention, spot)

    # Analytical gammas at current spot
    t_safe = np.maximum(t, 1.0 / 365.0)
    sig_safe = np.clip(iv, 1e-3, 5.0)
    sq_t = np.sqrt(t_safe)
    d1 = (np.log(spot / k) + (r + 0.5 * sig_safe * sig_safe) * t_safe) / (sig_safe * sq_t)
    d2 = d1 - sig_safe * sq_t
    pdf_d1 = norm.pdf(d1)
    gamma = pdf_d1 / (spot * sig_safe * sq_t)

    # Second-order Greek exposures ($M per 1% vol move and per day)
    vanna = -pdf_d1 * (d2 / sig_safe)
    charm = - (pdf_d1 * ((r / (sig_safe * sq_t)) - (d2 / (2.0 * t_safe)))) / 365.0

    # Dollar GEX in $M
    dollar_gex = gamma * signs * weights * CONTRACT_MULTIPLIER * (spot ** 2) * 0.01 / 1e6
    vanna_exp = vanna * signs * weights * CONTRACT_MULTIPLIER * spot * 0.01 / 1e6
    charm_exp = charm * signs * weights * CONTRACT_MULTIPLIER / 1e6

    df["net_gex"] = dollar_gex
    df["call_gex"] = np.where(cps == "C", np.abs(dollar_gex), 0.0)
    df["put_gex"] = np.where(cps == "P", -np.abs(dollar_gex), 0.0)
    df["vanna_exp"] = vanna_exp
    df["charm_exp"] = charm_exp

    # Aggregate by strike
    agg = df.groupby("strike").agg(
        net_gex=("net_gex", "sum"),
        call_gex=("call_gex", "sum"),
        put_gex=("put_gex", "sum"),
        vanna=("vanna_exp", "sum"),
        charm=("charm_exp", "sum"),
    ).reset_index().sort_values("strike")

    strikes = agg["strike"].tolist()
    net_gex_series = [round(float(v), 3) for v in agg["net_gex"]]
    call_gex_series = [round(float(v), 3) for v in agg["call_gex"]]
    put_gex_series = [round(float(v), 3) for v in agg["put_gex"]]
    vanna_series = [round(float(v), 3) for v in agg["vanna"]]
    charm_series = [round(float(v), 3) for v in agg["charm"]]

    total_gex = round(float(sum(net_gex_series)), 2)
    regime: Literal["positive", "negative"] = "positive" if total_gex >= 0 else "negative"

    # Identify Key Structural Levels
    call_wall = None
    call_wall_val = 0.0
    calls_above = agg[agg["strike"] >= spot]
    if not calls_above.empty:
        idx_c = calls_above["call_gex"].idxmax()
        call_wall = float(agg.loc[idx_c, "strike"])
        call_wall_val = round(float(agg.loc[idx_c, "call_gex"]), 2)

    put_wall = None
    put_wall_val = 0.0
    puts_below = agg[agg["strike"] <= spot]
    if not puts_below.empty:
        idx_p = puts_below["put_gex"].idxmin()
        put_wall = float(agg.loc[idx_p, "strike"])
        put_wall_val = round(float(agg.loc[idx_p, "put_gex"]), 2)

    # HVL (High Volatility Level / Absolute Magnet)
    hvl = None
    hvl_val = 0.0
    if not agg.empty:
        idx_h = agg["net_gex"].abs().idxmax()
        hvl = float(agg.loc[idx_h, "strike"])
        hvl_val = round(float(agg.loc[idx_h, "net_gex"]), 2)

    # Compute continuous gamma flip level
    gamma_flip = find_gamma_flip(df, spot, convention=convention, r=r)

    # Top concentration nodes
    gross_gex = sum(abs(v) for v in net_gex_series) or 1.0
    top_nodes = []
    ranked = sorted(zip(strikes, net_gex_series, call_gex_series, put_gex_series), key=lambda x: abs(x[1]), reverse=True)[:5]
    for k_val, net_v, c_v, p_v in ranked:
        share = round((abs(net_v) / gross_gex) * 100, 1)
        role = "Resistance / Ceiling" if k_val > spot and net_v > 0 else (
            "Support / Floor" if k_val < spot and net_v < 0 else (
                "Pin / Vol Dampener" if net_v > 0 else "Volatility Accelerator"
            )
        )
        top_nodes.append({
            "strike": k_val,
            "net_gex": net_v,
            "call_gex": c_v,
            "put_gex": p_v,
            "share_pct": share,
            "role": role,
        })

    return GexProfile(
        total_gex=total_gex,
        call_gex=round(float(sum(call_gex_series)), 2),
        put_gex=round(float(sum(put_gex_series)), 2),
        regime=regime,
        gamma_flip=gamma_flip,
        call_wall=call_wall,
        call_wall_val=call_wall_val,
        put_wall=put_wall,
        put_wall_val=put_wall_val,
        hvl=hvl,
        hvl_val=hvl_val,
        convention=convention,
        strikes=strikes,
        net_gex_series=net_gex_series,
        call_gex_series=call_gex_series,
        put_gex_series=put_gex_series,
        vanna_series=vanna_series,
        charm_series=charm_series,
        top_nodes=top_nodes,
    )
