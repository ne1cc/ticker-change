"""Microstructure, Liquidity & Squeeze Vulnerability Engine.

Implements:
1. Corwin-Schultz (2012) High-Low Bid-Ask Spread Estimator
2. Roll (1984) Effective Spread from Serial Covariance of Price Changes
3. Amihud (2002) illiquidity
4. Squeeze Risk Index (Multi-factor: DTC, Short Interest % Float, CTB, Negative GEX)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class MicrostructureAnalytics:
    corwin_schultz_spread_pct: float
    roll_effective_spread_pct: float
    amihud_illiquidity: float
    squeeze_risk_score: float  # [0, 100]
    squeeze_risk_level: str  # LOW, ELEVATED, HIGH, EXTREME
    squeeze_inputs_assumed: list = field(default_factory=list)  # inputs the squeeze index assumed (e.g. ["SI % float", "DTC"])


def calculate_corwin_schultz_spread(df: pd.DataFrame, window: int = 2) -> float:
    """Calculate the Corwin & Schultz (2012) Bid-Ask Spread Estimator from High/Low prices.

    S = 2 * (e^alpha - 1) / (1 + e^alpha)
    where alpha = (sqrt(2 * beta) - sqrt(beta)) / (3 - 2 * sqrt(2)) - sqrt(gamma / (3 - 2 * sqrt(2)))
    """
    if len(df) < window + 1:
        return 0.0

    high = df["high"].values if "high" in df.columns else df["High"].values
    low = df["low"].values if "low" in df.columns else df["Low"].values

    # Clean zero / invalid entries
    valid_mask = (high > 0) & (low > 0) & (high >= low)
    if np.sum(valid_mask) < 2:
        return 0.0

    h = high[valid_mask]
    l = low[valid_mask]

    spreads = []
    const_sqrt2 = math.sqrt(2.0)
    denom = 3.0 - 2.0 * const_sqrt2

    for t in range(1, len(h)):
        h1, l1 = h[t - 1], l[t - 1]
        h2, l2 = h[t], l[t]

        # 2-day high and low
        h2d = max(h1, h2)
        l2d = min(l1, l2)

        beta = (math.log(h1 / l1) ** 2) + (math.log(h2 / l2) ** 2)
        gamma = math.log(h2d / l2d) ** 2

        alpha = (math.sqrt(2.0 * beta) - math.sqrt(beta)) / denom - math.sqrt(gamma / denom)
        if alpha < 0:
            s = 0.0
        else:
            e_alpha = math.exp(alpha)
            s = 2.0 * (e_alpha - 1.0) / (1.0 + e_alpha)
        spreads.append(s)

    if not spreads:
        return 0.0
    return float(np.clip(np.nanmean(spreads), 0.0, 0.20))


def calculate_roll_effective_spread(df: pd.DataFrame) -> float:
    """Calculate Roll (1984) Effective Spread from serial covariance of price changes.

    Roll Spread = 2 * sqrt(-Cov(Delta P_t, Delta P_{t-1})) / P_mean
    If covariance is positive, return 0.0.
    """
    if len(df) < 5:
        return 0.0

    close = df["close"].values if "close" in df.columns else df["Close"].values
    delta_p = np.diff(close)
    if len(delta_p) < 2:
        return 0.0

    cov_matrix = np.cov(delta_p[:-1], delta_p[1:])
    cov = cov_matrix[0, 1]

    if cov >= 0:
        return 0.0

    mean_price = np.mean(close)
    if mean_price <= 0:
        return 0.0

    roll_spread = 2.0 * math.sqrt(-cov) / mean_price
    return float(np.clip(roll_spread, 0.0, 0.20))


def calculate_amihud_illiquidity(df: pd.DataFrame) -> float:
    """Amihud (2002) Illiquidity Measure: Mean(|R_t| / Dollar_Volume_t) * 1e6."""
    if len(df) < 5:
        return 0.0

    close = df["close"].values if "close" in df.columns else df["Close"].values
    volume = df["volume"].values if "volume" in df.columns else df["Volume"].values

    returns = np.abs(np.diff(close) / close[:-1])
    dollar_vol = close[1:] * volume[1:]

    valid = dollar_vol > 0
    if not np.any(valid):
        return 0.0

    illiq = np.mean(returns[valid] / dollar_vol[valid]) * 1e6
    return float(illiq)


def compute_squeeze_risk_index(
    short_pct_float: float,
    days_to_cover: float,
    cost_to_borrow_pct: float | None = None,
    is_negative_gex: bool = False,
    realized_vol_30d: float = 0.35,
) -> Tuple[float, str]:
    """Multi-factor Short Squeeze Risk Composite (0 - 100).

    Components (weights renormalise over the inputs actually used):
    - Short Interest % Float (0-40 pts): >20% heavily penalized
    - Days to Cover (0-30 pts): >5 days heavily penalized
    - Negative Gamma Dealer Exposure (0-15 pts): Accelerates price upside cascades
    - Cost to Borrow (0-15 pts): >10% annualized fee — only when a reported CTB
      is supplied. There is no free CTB feed, so callers omit it (None) rather
      than scoring every ticker on an invented constant.
    """
    # 1. Short Float Score
    si = max(short_pct_float, 0.0)
    si_score = min(si / 25.0, 1.0) * 40.0

    # 2. Days to Cover Score
    dtc = max(days_to_cover, 0.0)
    dtc_score = min(dtc / 7.0, 1.0) * 30.0

    # 3. Negative GEX Accelerator
    gex_score = 15.0 if is_negative_gex else 5.0

    raw_score = si_score + dtc_score + gex_score
    weight_sum = 85.0
    if cost_to_borrow_pct is not None:
        ctb = max(cost_to_borrow_pct, 0.0)
        raw_score += min(ctb / 20.0, 1.0) * 15.0
        weight_sum += 15.0

    score = round(float(np.clip(raw_score / weight_sum * 100.0, 0.0, 100.0)), 1)

    if score >= 75.0:
        level = "EXTREME"
    elif score >= 55.0:
        level = "HIGH"
    elif score >= 35.0:
        level = "ELEVATED"
    else:
        level = "LOW"

    return score, level


def get_microstructure_analytics(
    df: pd.DataFrame,
    short_pct_float: float = 3.0,
    days_to_cover: float = 2.0,
    is_negative_gex: bool = False,
    assumed_si_inputs: list | None = None,
) -> MicrostructureAnalytics:
    """Unified entry-point for quantitative microstructure analysis.

    `assumed_si_inputs` names the squeeze inputs that fell back to defaults
    (e.g. ["SI % float", "DTC"] when no short-interest feed reported) so the UI
    can flag them; cost-to-borrow is always absent (no free source) and is
    never scored on an invented value.
    """
    cs_spread = calculate_corwin_schultz_spread(df)
    roll_spread = calculate_roll_effective_spread(df)
    amihud = calculate_amihud_illiquidity(df)

    sq_score, sq_level = compute_squeeze_risk_index(
        short_pct_float=short_pct_float,
        days_to_cover=days_to_cover,
        is_negative_gex=is_negative_gex,
    )

    return MicrostructureAnalytics(
        corwin_schultz_spread_pct=round(cs_spread * 100, 3),
        roll_effective_spread_pct=round(roll_spread * 100, 3),
        amihud_illiquidity=round(amihud, 4),
        squeeze_risk_score=sq_score,
        squeeze_risk_level=sq_level,
        squeeze_inputs_assumed=list(assumed_si_inputs or []),
    )
