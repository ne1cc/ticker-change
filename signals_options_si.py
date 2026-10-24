"""Options and short-interest quantitative signals and history tracking.

Part of Epic 5 (issue #138):
- History writers for short interest snapshots and options metrics
- Evaluates VRP (Variance Risk Premium), IV Skew, and SI / Days-to-cover
- Explicit minimum history threshold in config; declares 'accumulating' status
  until sufficient depth exists, never claiming false validation.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence
import db
import providers

MIN_HISTORY_OBS: int = 63  # 63 trading days (~1 quarter) required before backtesting

OPTIONS_SI_SIGNAL_DEFINITIONS = {
    "vrp": {
        "name": "Variance Risk Premium (VRP)",
        "description": "30-day implied volatility minus 30-day realized volatility.",
        "category": "options",
        "min_obs": MIN_HISTORY_OBS,
    },
    "iv_skew": {
        "name": "25-Delta IV Skew",
        "description": "25-delta OTM put IV minus 25-delta OTM call IV.",
        "category": "options",
        "min_obs": MIN_HISTORY_OBS,
    },
    "short_interest_ratio": {
        "name": "Short Interest / Days to Cover",
        "description": "Reported short interest relative to 30-day average daily volume.",
        "category": "short_interest",
        "min_obs": MIN_HISTORY_OBS,
    },
}


def get_options_si_signal_status(ticker: str, signal_key: str) -> dict:
    """Evaluate whether sufficient historical observations exist for validation."""
    defn = OPTIONS_SI_SIGNAL_DEFINITIONS.get(signal_key)
    if not defn:
        return {"status": "unknown", "validated": False}

    min_required = defn["min_obs"]

    if defn["category"] == "short_interest":
        rows = db.get_short_interest_history(ticker)
        count = len(rows)
    else:
        # Options IV/skew history (e.g. from gex_snapshots or iv history)
        with db.get_conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM gex_snapshots WHERE ticker = ?",
                (ticker.upper(),),
            ).fetchone()
            count = int(row["cnt"]) if row is not None else 0

    if count < min_required:
        return {
            "signal_key": signal_key,
            "name": defn["name"],
            "category": defn["category"],
            "status": "accumulating",
            "observations": count,
            "min_required": min_required,
            "validated": False,
            "reason": f"Accumulating history ({count}/{min_required} days recorded)",
        }

    return {
        "signal_key": signal_key,
        "name": defn["name"],
        "category": defn["category"],
        "status": "sufficient_history",
        "observations": count,
        "min_required": min_required,
        "validated": False,  # Pending full cross-sectional panel validation
    }


def record_daily_short_interest(ticker: str, date_str: str | None = None) -> bool:
    """Fetch latest short interest print and store idempotently in SQLite."""
    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    si = providers.get_short_interest(ticker)
    if not si:
        return False

    db.insert_short_interest_snapshot(
        ticker=ticker,
        date=date_str,
        short_pct_float=si.get("short_percent_of_float"),
        short_ratio=si.get("short_ratio"),
        shares_short=si.get("shares_short"),
        shares_float=si.get("shares_float"),
    )
    return True
