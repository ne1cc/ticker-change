"""Percentile engine with peer-count guard and trend term.

Part of Epic 2 (Cross-sectional context, issue #118):
- Turns (metric, value) into exact percentiles vs S&P 500 universe and GICS sector
- Direction semantics: higher_is_riskier vs higher_is_better vs higher_is_more_expensive
- Minimum-peer guard: sector with fewer peers than guard renders as 'not meaningful'
- Trend term: compares current percentile with 3m and 12m prior snapshots;
  trend is None (not 0) when no prior snapshot exists.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import math
import numpy as np

import snapshot_store
import universe_metrics

MIN_PEERS_GUARD = 5

DEFAULT_DIRECTIONS: dict[str, str] = {
    # Higher is riskier
    "annualised_vol": "higher_is_riskier",
    "vol_21d": "higher_is_riskier",
    "vol_63d": "higher_is_riskier",
    "vol_252d": "higher_is_riskier",
    "max_drawdown": "higher_is_riskier",
    "beta": "higher_is_riskier",
    "kurtosis": "higher_is_riskier",
    "var_95": "higher_is_riskier",
    "var_99": "higher_is_riskier",
    "amihud": "higher_is_riskier",
    "short_float": "higher_is_riskier",
    "short_ratio": "higher_is_riskier",
    # Higher is better
    "sharpe": "higher_is_better",
    "sortino": "higher_is_better",
    "calmar": "higher_is_better",
    "momentum_12_1": "higher_is_better",
    "revenue_growth": "higher_is_better",
    "earnings_growth": "higher_is_better",
    "dividend_yield": "higher_is_better",
    "skewness": "higher_is_better",
    # Valuation multiples (higher is more expensive)
    "pe": "higher_is_more_expensive",
    "forward_pe": "higher_is_more_expensive",
    "price_book": "higher_is_more_expensive",
    "price_sales": "higher_is_more_expensive",
}


def _clean_numbers(values: list | np.ndarray) -> list[float]:
    """Filter out None, NaN, inf, and non-numeric values; return sorted floats."""
    cleaned = []
    for v in values:
        if v is None:
            continue
        try:
            f = float(v)
            if not (math.isnan(f) or math.isinf(f)):
                cleaned.append(f)
        except (ValueError, TypeError):
            continue
    return sorted(cleaned)


def percentile_rank(
    values: list | np.ndarray,
    target: float | int,
    method: str = "mean",
    min_peers: int = 1,
) -> float | None:
    """Exact empirical percentile rank of target within values (0.0 to 100.0).

    Handles NaN/None automatically.
    method='mean': (count(v < target) + 0.5 * count(v == target)) / N * 100
    method='rank': count(v <= target) / N * 100
    method='strict': count(v < target) / N * 100

    Returns None if valid peer count < min_peers or target is invalid.
    """
    if target is None:
        return None
    try:
        t_val = float(target)
        if math.isnan(t_val) or math.isinf(t_val):
            return None
    except (ValueError, TypeError):
        return None

    vals = _clean_numbers(values)
    n = len(vals)
    if n < min_peers:
        return None

    count_less = sum(1 for v in vals if v < t_val)
    count_eq = sum(1 for v in vals if v == t_val)

    if method == "mean":
        pct = 100.0 * (count_less + 0.5 * count_eq) / n
    elif method == "rank":
        pct = 100.0 * (count_less + count_eq) / n
    elif method == "strict":
        pct = 100.0 * count_less / n
    else:
        raise ValueError(f"Unknown percentile method: {method}")

    return round(float(pct), 1)


def get_metric_direction(metric_key: str) -> str:
    """Return declared direction semantics for metric."""
    if metric_key in universe_metrics.METRIC_DEFINITIONS:
        return universe_metrics.METRIC_DEFINITIONS[metric_key].get(
            "direction", DEFAULT_DIRECTIONS.get(metric_key, "neutral")
        )
    return DEFAULT_DIRECTIONS.get(metric_key, "neutral")


def compute_historical_trend(
    metric_key: str,
    target_value: float,
    current_pct: float | None,
    pool_type: str = "all",
    sector_name: str | None = None,
) -> dict | None:
    """Compare current percentile with 3m (~63d) and 12m (~252d) prior snapshots.

    Returns None (NOT 0) when no prior snapshot exists.
    """
    if current_pct is None:
        return None

    hist = snapshot_store.history(universe_metrics.PROVIDER)
    if len(hist) <= 1:
        # No prior snapshot exists
        return None

    # Current snapshot is hist[0], prior snapshots are hist[1:]
    now = datetime.utcnow()
    target_3m_days = 90
    target_12m_days = 365

    closest_3m = None
    min_dist_3m = float("inf")

    closest_12m = None
    min_dist_12m = float("inf")

    for sid, snap in hist[1:]:
        as_of_str = snap.get("as_of") or snap.get("created_at")
        if not as_of_str:
            continue
        try:
            snap_date = datetime.fromisoformat(as_of_str.split("T")[0])
            days_ago = (now - snap_date).days
        except Exception:
            continue

        dist_3m = abs(days_ago - target_3m_days)
        if dist_3m < min_dist_3m and 45 <= days_ago <= 150:
            min_dist_3m = dist_3m
            closest_3m = snap

        dist_12m = abs(days_ago - target_12m_days)
        if dist_12m < min_dist_12m and 200 <= days_ago <= 450:
            min_dist_12m = dist_12m
            closest_12m = snap

    delta_3m = None
    delta_12m = None

    if closest_3m:
        past_vals = _get_peer_values(closest_3m, metric_key, pool_type, sector_name)
        past_pct = percentile_rank(past_vals, target_value)
        if past_pct is not None:
            delta_3m = round(current_pct - past_pct, 1)

    if closest_12m:
        past_vals = _get_peer_values(closest_12m, metric_key, pool_type, sector_name)
        past_pct = percentile_rank(past_vals, target_value)
        if past_pct is not None:
            delta_12m = round(current_pct - past_pct, 1)

    if delta_3m is None and delta_12m is None:
        return None

    # Determine trend label
    ref_delta = delta_3m if delta_3m is not None else delta_12m
    if ref_delta >= 5.0:
        direction = "rising"
    elif ref_delta <= -5.0:
        direction = "falling"
    else:
        direction = "stable"

    return {
        "delta_3m": delta_3m,
        "delta_12m": delta_12m,
        "direction": direction,
    }


def _get_peer_values(
    snapshot: dict,
    metric_key: str,
    pool_type: str,
    sector_name: str | None = None,
) -> list[float]:
    """Extract peer values array from snapshot payload."""
    by_metric = snapshot.get("by_metric") or {}
    m_data = by_metric.get(metric_key) or {}
    if pool_type == "all":
        return m_data.get("all") or []
    elif pool_type == "sector" and sector_name:
        sectors = m_data.get("sectors") or {}
        return sectors.get(sector_name) or []
    return []


def calculate_metric_percentile(
    metric_key: str,
    value: float | int,
    sector: str | None = None,
    min_sector_peers: int = MIN_PEERS_GUARD,
    snapshot: dict | None = None,
) -> dict:
    """Calculate cross-sectional percentile vs S&P 500 universe and GICS sector.

    Returns complete context:
    - universe percentile rank, peer count, trend
    - sector percentile rank, peer count, guard status ('not meaningful' if < min_sector_peers)
    - direction semantics
    """
    direction = get_metric_direction(metric_key)

    if snapshot is None:
        snapshot = universe_metrics.current_universe_metrics()

    if not snapshot:
        return {
            "metric": metric_key,
            "value": value,
            "direction": direction,
            "universe": {
                "percentile": None,
                "peer_count": 0,
                "as_of": None,
                "trend": None,
                "status": "missing_snapshot",
            },
            "sector": {
                "name": sector,
                "percentile": None,
                "peer_count": 0,
                "status": "missing_snapshot",
                "trend": None,
            } if sector else None,
        }

    as_of = snapshot.get("as_of")

    # 1. Universe percentile
    univ_vals = _get_peer_values(snapshot, metric_key, "all")
    univ_pct = percentile_rank(univ_vals, value, min_peers=10)
    univ_trend = compute_historical_trend(metric_key, value, univ_pct, pool_type="all")

    univ_result = {
        "percentile": univ_pct,
        "peer_count": len(univ_vals),
        "as_of": as_of,
        "trend": univ_trend,
        "status": "ok" if univ_pct is not None else "insufficient_peers",
    }

    # 2. Sector percentile
    sector_result = None
    if sector:
        sec_vals = _get_peer_values(snapshot, metric_key, "sector", sector)
        n_sec = len(sec_vals)
        if n_sec < min_sector_peers:
            sector_result = {
                "name": sector,
                "percentile": None,
                "peer_count": n_sec,
                "status": "not meaningful",
                "trend": None,
            }
        else:
            sec_pct = percentile_rank(sec_vals, value, min_peers=min_sector_peers)
            sec_trend = compute_historical_trend(
                metric_key, value, sec_pct, pool_type="sector", sector_name=sector
            )
            sector_result = {
                "name": sector,
                "percentile": sec_pct,
                "peer_count": n_sec,
                "status": "ok" if sec_pct is not None else "error",
                "trend": sec_trend,
            }

    return {
        "metric": metric_key,
        "value": value,
        "direction": direction,
        "universe": univ_result,
        "sector": sector_result,
    }
