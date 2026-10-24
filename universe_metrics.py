"""Nightly universe metric snapshot engine for S&P 500 constituents.

Part of Epic 2 (Cross-sectional context, issue #116):
- Computes per-ticker metric vectors across the S&P 500 universe
- Published atomically via snapshot_store; failed run leaves last-good serving
- Each metric has a declared definition, window, and direction in METRIC_DEFINITIONS
- Constituents with insufficient history are excluded with an explicit count, not silently zero.
"""
from __future__ import annotations

from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats

import db
import gics
import snapshot_store

PROVIDER = "universe_metrics"
MIN_HISTORY_BARS = 63  # At least one quarter of daily trading bars required

METRIC_DEFINITIONS: dict[str, dict] = {
    "vol_21d": {
        "name": "21-Day Realized Volatility",
        "window_days": 21,
        "formula": "std(returns[-21:]) * sqrt(252) * 100",
        "direction": "higher_is_riskier",
        "min_bars": 21,
        "description": "Short-term 1-month annualised volatility.",
    },
    "vol_63d": {
        "name": "63-Day Realized Volatility",
        "window_days": 63,
        "formula": "std(returns[-63:]) * sqrt(252) * 100",
        "direction": "higher_is_riskier",
        "min_bars": 63,
        "description": "Quarterly 3-month annualised volatility.",
    },
    "vol_252d": {
        "name": "252-Day Realized Volatility",
        "window_days": 252,
        "formula": "std(returns[-252:]) * sqrt(252) * 100",
        "direction": "higher_is_riskier",
        "min_bars": 126,
        "description": "Trailing 1-year annualised volatility.",
    },
    "annualised_vol": {
        "name": "Annualised Volatility",
        "window_days": 252,
        "formula": "std(returns) * sqrt(252) * 100",
        "direction": "higher_is_riskier",
        "min_bars": 63,
        "description": "Standard annualised volatility matching /analytics headline stat.",
    },
    "max_drawdown": {
        "name": "Maximum Drawdown (1y)",
        "window_days": 252,
        "formula": "min((price - running_max) / running_max) * 100",
        "direction": "higher_is_riskier",
        "min_bars": 63,
        "description": "Worst peak-to-trough drop over the trailing year.",
    },
    "beta": {
        "name": "Beta vs SPY (1y)",
        "window_days": 252,
        "formula": "cov(R_stock, R_spy) / var(R_spy)",
        "direction": "higher_is_riskier",
        "min_bars": 63,
        "description": "Market sensitivity slope against SPY over 252 trading days.",
    },
    "momentum_12_1": {
        "name": "12-1 Month Momentum",
        "window_days": 252,
        "formula": "(price[t-21] - price[t-252]) / price[t-252] * 100",
        "direction": "higher_is_better",
        "min_bars": 252,
        "description": "Classic 12-1 month price momentum excluding the most recent month.",
    },
    "skewness": {
        "name": "Return Skewness (1y)",
        "window_days": 252,
        "formula": "skew(returns[-252:])",
        "direction": "higher_is_better",
        "min_bars": 63,
        "description": "Asymmetry of daily returns distribution.",
    },
    "kurtosis": {
        "name": "Excess Kurtosis (1y)",
        "window_days": 252,
        "formula": "kurtosis(returns[-252:], fisher=True)",
        "direction": "higher_is_riskier",
        "min_bars": 63,
        "description": "Tail-heaviness / extreme move frequency vs standard Gaussian.",
    },
    "var_95": {
        "name": "1-Day 95% Value-at-Risk",
        "window_days": 252,
        "formula": "-percentile(returns[-252:], 5) * 100",
        "direction": "higher_is_riskier",
        "min_bars": 63,
        "description": "Daily 95% historical Value-at-Risk loss percentage.",
    },
    "amihud": {
        "name": "Amihud Illiquidity (1y)",
        "window_days": 252,
        "formula": "mean(|return| / (close * volume)) * 1e6",
        "direction": "higher_is_riskier",
        "min_bars": 63,
        "description": "Price impact per dollar traded (Amihud illiquidity ratio x 1M).",
    },
}


def compute_ticker_metrics(
    df: pd.DataFrame, spy_df: pd.DataFrame | None = None
) -> dict[str, float]:
    """Compute standard metrics vector for a single ticker's price history.

    Raises ValueError if history is shorter than MIN_HISTORY_BARS.
    """
    if df is None or len(df) < MIN_HISTORY_BARS:
        n = len(df) if df is not None else 0
        raise ValueError(
            f"Insufficient history: {n} bars (minimum {MIN_HISTORY_BARS})"
        )

    # Normalize column names to lowercase for robust access
    cols = {c.lower(): c for c in df.columns}
    close_col = cols.get("close", "Close")
    volume_col = cols.get("volume", "Volume")

    closes = df[close_col].astype(float)
    returns = closes.pct_change().dropna()
    n_ret = len(returns)
    if n_ret < 20:
        raise ValueError("Insufficient returns data")

    metrics: dict[str, float] = {}

    # Realized volatilities
    std_all = float(returns.std())
    metrics["annualised_vol"] = round(std_all * np.sqrt(252) * 100.0, 2)

    ret_21 = returns.iloc[-21:] if n_ret >= 21 else returns
    metrics["vol_21d"] = round(float(ret_21.std()) * np.sqrt(252) * 100.0, 2)

    ret_63 = returns.iloc[-63:] if n_ret >= 63 else returns
    metrics["vol_63d"] = round(float(ret_63.std()) * np.sqrt(252) * 100.0, 2)

    ret_252 = returns.iloc[-252:] if n_ret >= 252 else returns
    metrics["vol_252d"] = round(float(ret_252.std()) * np.sqrt(252) * 100.0, 2)

    # Maximum Drawdown (trailing 1y / 252d)
    w_closes = closes.iloc[-252:] if len(closes) >= 252 else closes
    running_max = w_closes.cummax()
    dd = (w_closes - running_max) / running_max * 100.0
    metrics["max_drawdown"] = round(float(dd.min()), 2)

    # Skewness & Kurtosis
    ret_w = ret_252
    metrics["skewness"] = round(float(stats.skew(ret_w)), 3)
    metrics["kurtosis"] = round(float(stats.kurtosis(ret_w, fisher=True)), 3)

    # 1-day 95% Historical VaR (expressed as positive loss %)
    metrics["var_95"] = round(float(-np.percentile(ret_w, 5) * 100.0), 2)

    # 12-1 Momentum (requires >= 252 days)
    if len(closes) >= 252:
        p_t21 = float(closes.iloc[-21])
        p_t252 = float(closes.iloc[-252])
        if p_t252 > 0:
            metrics["momentum_12_1"] = round((p_t21 - p_t252) / p_t252 * 100.0, 2)
        else:
            metrics["momentum_12_1"] = None
    else:
        metrics["momentum_12_1"] = None

    # Beta vs SPY
    if spy_df is not None and not spy_df.empty:
        spy_close_col = {c.lower(): c for c in spy_df.columns}.get("close", "Close")
        spy_closes = spy_df[spy_close_col].astype(float)
        spy_ret = spy_closes.pct_change().dropna()
        combined = pd.DataFrame({"stock": returns, "spy": spy_ret}).dropna()
        if len(combined) >= 63:
            w_comb = combined.iloc[-252:] if len(combined) >= 252 else combined
            cov = float(np.cov(w_comb["stock"], w_comb["spy"])[0, 1])
            var_spy = float(np.var(w_comb["spy"], ddof=1))
            if var_spy > 1e-12:
                metrics["beta"] = round(cov / var_spy, 3)
            else:
                metrics["beta"] = None
        else:
            metrics["beta"] = None
    else:
        metrics["beta"] = None

    # Amihud Illiquidity
    if volume_col in df.columns:
        vols = df[volume_col].astype(float).reindex(returns.index)
        dollar_vol = closes.reindex(returns.index) * vols
        w_ret = returns.iloc[-252:] if len(returns) >= 252 else returns
        w_dvol = dollar_vol.iloc[-252:] if len(dollar_vol) >= 252 else dollar_vol
        valid = (w_dvol > 0) & (~w_ret.isna())
        if valid.sum() >= 63:
            ratio = (np.abs(w_ret[valid]) / w_dvol[valid]) * 1e6
            metrics["amihud"] = round(float(ratio.mean()), 4)
        else:
            metrics["amihud"] = None
    else:
        metrics["amihud"] = None

    return metrics


def build_universe_metrics_snapshot(
    tickers: list[str] | None = None,
    gics_map: dict[str, dict] | None = None,
    spy_df: pd.DataFrame | None = None,
) -> dict:
    """Build the full universe metrics snapshot payload.

    Extracts price history for all constituents, computes declared metric
    vectors, partitions distributions by universe and GICS sector, and
    records excluded tickers with explicit reasons.
    """
    if gics_map is None:
        gics_map = gics.fetch_gics_mapping()

    if tickers is None:
        tickers = sorted(gics_map.keys())

    if not tickers:
        raise RuntimeError("Universe tickers list is empty")

    if spy_df is None:
        spy_df = db.get_prices("SPY")

    # Fetch price data in batch
    prices_map = db.get_prices_batch(tickers)

    ticker_metrics: dict[str, dict] = {}
    excluded_tickers: dict[str, str] = {}

    for t in tickers:
        t_norm = gics.normalize_ticker(t)
        df = prices_map.get(t_norm) if t_norm in prices_map else prices_map.get(t)
        if df is None:
            excluded_tickers[t_norm] = "no price history in cache"
            continue
        try:
            m_vector = compute_ticker_metrics(df, spy_df=spy_df)
            sec = (gics_map.get(t_norm) or {}).get("sector")
            sub = (gics_map.get(t_norm) or {}).get("sub_industry")
            m_vector["_sector"] = sec
            m_vector["_sub_industry"] = sub
            ticker_metrics[t_norm] = m_vector
        except ValueError as e:
            excluded_tickers[t_norm] = str(e)
        except Exception as e:
            excluded_tickers[t_norm] = f"error: {e}"

    # Partition values by metric for O(1) percentile lookups
    by_metric: dict[str, dict] = {}
    for m_key in METRIC_DEFINITIONS:
        all_vals = []
        sectors_vals: dict[str, list[float]] = {}
        for t, m_data in ticker_metrics.items():
            val = m_data.get(m_key)
            if val is not None and not np.isnan(val):
                all_vals.append(float(val))
                sec = m_data.get("_sector")
                if sec:
                    sectors_vals.setdefault(sec, []).append(float(val))

        by_metric[m_key] = {
            "all": sorted(all_vals),
            "sectors": {k: sorted(v) for k, v in sectors_vals.items()},
        }

    snapshot_id = snapshot_store.new_snapshot_id()
    now_utc = datetime.utcnow()

    payload = {
        "snapshot_id": snapshot_id,
        "as_of": now_utc.strftime("%Y-%m-%d"),
        "created_at": now_utc.isoformat(),
        "total_constituents": len(tickers),
        "valid_count": len(ticker_metrics),
        "excluded_count": len(excluded_tickers),
        "excluded_tickers": excluded_tickers,
        "metrics": METRIC_DEFINITIONS,
        "tickers": ticker_metrics,
        "by_metric": by_metric,
    }
    return payload


def publish_universe_metrics_snapshot(payload: dict) -> str:
    """Validate and atomically publish universe metrics snapshot."""
    if not isinstance(payload, dict) or not payload.get("snapshot_id"):
        raise ValueError("Invalid payload: missing snapshot_id")
    if payload.get("valid_count", 0) <= 0:
        raise ValueError("Cannot publish snapshot with zero valid tickers")

    snapshot_id = payload["snapshot_id"]
    snapshot_store.publish(PROVIDER, snapshot_id, payload)
    return snapshot_id


def current_universe_metrics() -> dict | None:
    """Return the currently published universe metrics snapshot, or None."""
    return snapshot_store.current(PROVIDER)


def warm_universe_metrics(
    tickers: list[str] | None = None,
    gics_map: dict[str, dict] | None = None,
) -> dict | None:
    """Warmer worker job: build and atomically publish the universe metrics snapshot.

    If an error occurs, catches the exception and does NOT publish,
    leaving the last-good snapshot serving.
    """
    try:
        payload = build_universe_metrics_snapshot(tickers=tickers, gics_map=gics_map)
        publish_universe_metrics_snapshot(payload)
        return payload
    except Exception as e:
        print(f"[universe_metrics] build/publish failed: {e}")
        return None
