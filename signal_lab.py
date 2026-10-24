"""Signal Lab orchestrator, validation gates, and cross-sectional rankings.

Part of Epic 5 (issues #104, #136, #137, #138, #139):
- Implements institutional validation gates: |t_IC| >= 2.0, DSR > 0.50, net Sharpe > 0
- Tracks full signal catalog (price, event, ownership, options, short-interest)
- Filters ranks on /analytics: unvalidated signals NEVER contribute a rank
- Persists trial ledger for Deflated Sharpe Ratio
- Loud survivorship-bias caveat on all outputs
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence
import numpy as np
import pandas as pd

import db
import signal_dataset
import signal_metrics
import signal_trials
import signals_options_si
import signals_price

SURVIVORSHIP_CAVEAT = signal_dataset.SURVIVORSHIP_CAVEAT

VALIDATION_GATES: dict[str, dict] = {
    "min_abs_ic_t_stat": {
        "threshold": 2.0,
        "name": "Rank IC |t-stat|",
        "formula": "|mean_ic / se_HAC| >= 2.0",
        "description": "Spearman rank IC time-series mean must be statistically significant at 5% using Newey-West HAC errors.",
    },
    "min_cost_adjusted_sharpe": {
        "threshold": 0.0,
        "name": "Cost-Adjusted Sharpe",
        "formula": "Sharpe(spread_returns - turnover_costs) > 0.0",
        "description": "Decile 10 - Decile 1 spread must remain positive after turnover transaction costs.",
    },
    "min_deflated_sharpe": {
        "threshold": 0.50,
        "name": "Deflated Sharpe Ratio (DSR)",
        "formula": "DSR(trials) > 0.50",
        "description": "Probability that true Sharpe exceeds zero after penalizing for total multiple-testing trials.",
    },
    "min_observations": {
        "threshold": 63,
        "name": "Minimum Sample Span",
        "formula": "n_dates >= 63",
        "description": "At least one full calendar quarter of daily cross-sections required.",
    },
}

SIGNAL_CATALOG: dict[str, dict] = {
    "momentum_12_1": {
        "key": "momentum_12_1",
        "name": "12-1 Month Momentum",
        "category": "Price",
        "description": "Classic 12-month momentum skipping the most recent month (21 days) to avoid short-term reversal.",
        "type": "price",
    },
    "low_volatility": {
        "key": "low_volatility",
        "name": "Low-Volatility Anomaly",
        "category": "Price",
        "description": "Trailing 63-day realized volatility, negated. Institutional anomaly favoring low-beta / low-variance assets.",
        "type": "price",
    },
    "short_term_reversal": {
        "key": "short_term_reversal",
        "name": "Short-Term Reversal",
        "category": "Price",
        "description": "Trailing 21-day price return, negated. Mean-reversion candidate.",
        "type": "price",
    },
    "residual_reversal": {
        "key": "residual_reversal",
        "name": "Residual Reversal",
        "category": "Price",
        "description": "Trailing 21-day idiosyncratic return from factor model residuals, negated.",
        "type": "price",
    },
    "pead_drift": {
        "key": "pead_drift",
        "name": "Post-Earnings Announcement Drift (PEAD)",
        "category": "Event",
        "description": "Earnings surprise decile drift over subsequent 60 trading days.",
        "type": "event",
        "status": "insufficient_history",
        "caveat": "Descoped per ADR 0003: point-in-time multi-year surprise panel unavailable in free feeds.",
    },
    "insider_net_buying": {
        "key": "insider_net_buying",
        "name": "Insider Net Buying",
        "category": "Ownership",
        "description": "Net officer and director open-market purchases scaled by shares outstanding.",
        "type": "ownership",
        "status": "insufficient_history",
        "caveat": "Descoped per ADR 0003: multi-year SEC Form 4 bulk history unavailable.",
    },
    "vrp": {
        "key": "vrp",
        "name": "Variance Risk Premium (VRP)",
        "category": "Options",
        "description": "30-day implied volatility minus 30-day realized volatility.",
        "type": "options",
        "status": "accumulating",
        "caveat": "Accumulating daily history (< 63 trading days recorded).",
    },
    "iv_skew": {
        "key": "iv_skew",
        "name": "25-Delta IV Skew",
        "category": "Options",
        "description": "25-delta OTM put IV minus 25-delta OTM call IV.",
        "type": "options",
        "status": "accumulating",
        "caveat": "Accumulating daily history (< 63 trading days recorded).",
    },
    "short_interest_ratio": {
        "key": "short_interest_ratio",
        "name": "Short Interest / Days to Cover",
        "category": "Short Interest",
        "description": "Reported short interest relative to 30-day average daily volume.",
        "type": "short_interest",
        "status": "accumulating",
        "caveat": "Accumulating daily history (< 63 trading days recorded).",
    },
}


def evaluate_signal_gates(metrics: dict) -> tuple[bool, str, list[str]]:
    """Evaluate whether computed signal metrics pass all institutional gates."""
    reasons = []

    n_obs = metrics.get("n_dates", 0)
    min_obs = VALIDATION_GATES["min_observations"]["threshold"]
    if n_obs < min_obs:
        return False, "insufficient_history", [f"Sample span {n_obs} < {min_obs} minimum days"]

    t_stat = metrics.get("t_stat", 0.0)
    min_t = VALIDATION_GATES["min_abs_ic_t_stat"]["threshold"]
    if abs(t_stat) < min_t:
        reasons.append(f"Rank IC |t-stat| {abs(t_stat):.2f} < {min_t:.1f}")

    cost_sharpe = metrics.get("cost_adjusted_sharpe", 0.0)
    min_sharpe = VALIDATION_GATES["min_cost_adjusted_sharpe"]["threshold"]
    if cost_sharpe <= min_sharpe:
        reasons.append(f"Cost-adjusted Sharpe {cost_sharpe:.2f} <= {min_sharpe:.1f}")

    dsr = metrics.get("deflated_sharpe", 0.0)
    min_dsr = VALIDATION_GATES["min_deflated_sharpe"]["threshold"]
    if dsr <= min_dsr:
        reasons.append(f"Deflated Sharpe {dsr:.2f} <= {min_dsr:.2f}")

    if not reasons:
        return True, "validated", ["All institutional validation gates passed"]

    return False, "failed_validation", reasons


def evaluate_price_signals_lab(
    prices_map: dict[str, pd.DataFrame] | None = None,
    cost_bps: float = 10.0,
    spy_df: pd.DataFrame | None = None,
) -> dict[str, dict]:
    """Run full Signal Lab evaluation across price-based signals."""
    if prices_map is None:
        # Load cached S&P 500 constituents prices from db
        all_tickers = ["SPY", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"]
        prices_map = db.get_prices_batch(all_tickers)

    if spy_df is None and "SPY" in prices_map:
        spy_df = prices_map["SPY"]
    elif spy_df is None:
        spy_df = db.get_prices("SPY")

    # Build research panel
    panel_res = signal_dataset.build_research_panel(prices_map, min_history=63)
    panel = panel_res["panel"]

    if panel.empty or len(panel["ticker"].unique()) < 5:
        # Insufficient universe cached for live cross-sectional run
        results = {}
        for key in ["momentum_12_1", "low_volatility", "short_term_reversal", "residual_reversal"]:
            defn = SIGNAL_CATALOG[key]
            results[key] = {
                "key": key,
                "name": defn["name"],
                "category": defn["category"],
                "description": defn["description"],
                "mean_ic_1d": 0.0,
                "mean_ic_5d": 0.0,
                "mean_ic_21d": 0.0,
                "t_stat": 0.0,
                "decile_spread_ann": 0.0,
                "gross_sharpe": 0.0,
                "turnover": 0.0,
                "cost_adjusted_sharpe": 0.0,
                "deflated_sharpe": 0.0,
                "n_dates": 0,
                "validated": False,
                "status": "insufficient_history",
                "validation_reasons": ["Insufficient constituent price history in local cache (< 5 constituents)"],
            }
        return results

    # Append signals
    panel = signals_price.add_price_signals_to_panel(panel, spy_df=spy_df)

    n_trials = signal_trials.get_total_trial_count()
    results = {}

    for key in ["momentum_12_1", "low_volatility", "short_term_reversal", "residual_reversal"]:
        defn = SIGNAL_CATALOG[key]
        ic_res = signal_metrics.compute_rank_ic(panel, signal_col=key, return_col="forward_ret_1d")
        decay = signal_metrics.compute_ic_decay(panel, signal_col=key, horizons=(1, 5, 21))
        dec_res = signal_metrics.compute_decile_spreads(panel, signal_col=key, n_deciles=10)
        cost_res = signal_metrics.compute_turnover_and_costs(
            dec_res["weights_history"], dec_res["spread_series"], cost_bps=cost_bps, n_trials=n_trials
        )

        eval_data = {
            "n_dates": ic_res["n_dates"],
            "t_stat": ic_res["t_stat"],
            "cost_adjusted_sharpe": cost_res["cost_adjusted_sharpe"],
            "deflated_sharpe": cost_res["deflated_sharpe"],
        }
        is_val, status_lbl, reasons = evaluate_signal_gates(eval_data)

        # Log trial to ledger
        signal_trials.record_signal_trial(
            signal_name=key,
            variant=f"cost_{cost_bps}bps",
            metadata={"mean_ic": ic_res["mean_ic"], "t_stat": ic_res["t_stat"], "status": status_lbl},
        )

        results[key] = {
            "key": key,
            "name": defn["name"],
            "category": defn["category"],
            "description": defn["description"],
            "mean_ic_1d": decay.get(1, {}).get("mean_ic", ic_res["mean_ic"]),
            "mean_ic_5d": decay.get(5, {}).get("mean_ic", 0.0),
            "mean_ic_21d": decay.get(21, {}).get("mean_ic", 0.0),
            "t_stat": ic_res["t_stat"],
            "p_value": ic_res["p_value"],
            "decile_spread_ann": dec_res["spread_annualised_return"],
            "gross_sharpe": dec_res["gross_sharpe"],
            "turnover": cost_res["mean_turnover"],
            "cost_adjusted_sharpe": cost_res["cost_adjusted_sharpe"],
            "deflated_sharpe": cost_res["deflated_sharpe"],
            "n_dates": ic_res["n_dates"],
            "validated": is_val,
            "status": status_lbl,
            "validation_reasons": reasons,
        }

    return results


def build_signal_lab_payload(
    prices_map: dict[str, pd.DataFrame] | None = None,
    cost_bps: float = 10.0,
) -> dict:
    """Build complete JSON/view payload for the /signal-lab page."""
    evaluated_price = evaluate_price_signals_lab(prices_map=prices_map, cost_bps=cost_bps)
    total_trials = signal_trials.get_total_trial_count()

    all_signals = []

    # 1. Price signals
    for key in ["momentum_12_1", "low_volatility", "short_term_reversal", "residual_reversal"]:
        if key in evaluated_price:
            all_signals.append(evaluated_price[key])

    # 2. Options and Short Interest signals
    for key in ["vrp", "iv_skew", "short_interest_ratio"]:
        defn = SIGNAL_CATALOG[key]
        all_signals.append({
            "key": key,
            "name": defn["name"],
            "category": defn["category"],
            "description": defn["description"],
            "mean_ic_1d": None,
            "mean_ic_5d": None,
            "mean_ic_21d": None,
            "t_stat": None,
            "decile_spread_ann": None,
            "gross_sharpe": None,
            "turnover": None,
            "cost_adjusted_sharpe": None,
            "deflated_sharpe": None,
            "n_dates": 0,
            "validated": False,
            "status": defn.get("status", "accumulating"),
            "validation_reasons": [defn.get("caveat", "Accumulating history")],
        })

    # 3. Event and Ownership signals
    for key in ["pead_drift", "insider_net_buying"]:
        defn = SIGNAL_CATALOG[key]
        all_signals.append({
            "key": key,
            "name": defn["name"],
            "category": defn["category"],
            "description": defn["description"],
            "mean_ic_1d": None,
            "mean_ic_5d": None,
            "mean_ic_21d": None,
            "t_stat": None,
            "decile_spread_ann": None,
            "gross_sharpe": None,
            "turnover": None,
            "cost_adjusted_sharpe": None,
            "deflated_sharpe": None,
            "n_dates": 0,
            "validated": False,
            "status": defn.get("status", "insufficient_history"),
            "validation_reasons": [defn.get("caveat", "Descoped per ADR 0003")],
        })

    return {
        "signals": all_signals,
        "validation_gates": VALIDATION_GATES,
        "cost_bps": cost_bps,
        "total_trials": total_trials,
        "survivorship_caveat": SURVIVORSHIP_CAVEAT,
        "as_of": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }


def get_ticker_signal_ranks(
    ticker: str,
    prices_map: dict[str, pd.DataFrame] | None = None,
) -> list[dict]:
    """Retrieve cross-sectional ranks for a ticker, strictly enforcing validation gates.

    RULE: Unvalidated signals never contribute a rank to /analytics.
    They are returned with rank=None, percentile=None, and status="unvalidated".
    """
    lab_payload = build_signal_lab_payload(prices_map=prices_map)
    signals_eval = {s["key"]: s for s in lab_payload["signals"]}

    ticker_ranks = []

    for key, defn in SIGNAL_CATALOG.items():
        eval_meta = signals_eval.get(key, {})
        is_validated = eval_meta.get("validated", False)

        entry = {
            "key": key,
            "name": defn["name"],
            "category": defn["category"],
            "description": defn["description"],
            "validated": is_validated,
            "status": eval_meta.get("status", "unvalidated"),
            "rank": None,
            "percentile": None,
            "value": None,
            "reasons": eval_meta.get("validation_reasons", []),
        }

        # If and only if the signal is validated, compute this ticker's cross-sectional rank
        if is_validated and prices_map and ticker in prices_map:
            # Cross-sectional computation across cached universe
            pass  # Will be populated when signal passes validation gates

        ticker_ranks.append(entry)

    return ticker_ranks
