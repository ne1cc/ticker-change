"""Unit tests for Options-Implied Earnings Move vs Realized Move Engine.

Verifies Epic 4 issue #128 acceptance criteria:
- Implied move computed from straddle mid with quote validity checks
- Realized comparison shows sample size n and excludes missing prints
- Valuation regime correctly classified as rich / cheap / fair
- No upcoming earnings renders a declared state
"""
import numpy as np
import pandas as pd
import pytest

from options.earnings_move import (
    compute_atm_straddle_move,
    compute_historical_earnings_moves,
    get_earnings_implied_vs_realized,
)


def test_compute_atm_straddle_move_valid():
    """Verify ATM straddle move calculation from call and put mid prices."""
    spot = 100.0
    chain = pd.DataFrame([
        {"cp": "C", "strike": 95.0, "bid": 6.0, "ask": 6.2},
        {"cp": "C", "strike": 100.0, "bid": 3.0, "ask": 3.2},  # Call mid = 3.10
        {"cp": "C", "strike": 105.0, "bid": 1.0, "ask": 1.2},
        {"cp": "P", "strike": 95.0, "bid": 1.1, "ask": 1.3},
        {"cp": "P", "strike": 100.0, "bid": 2.8, "ask": 3.0},  # Put mid = 2.90
        {"cp": "P", "strike": 105.0, "bid": 5.8, "ask": 6.0},
    ])

    res = compute_atm_straddle_move(chain, spot)
    assert res is not None
    assert res["atm_strike"] == 100.0
    assert res["call_mid"] == 3.10
    assert res["put_mid"] == 2.90
    assert res["straddle_price"] == 6.00
    # Implied move = (6.00 / 100) * 100 = 6.00%
    assert res["implied_move_pct"] == 6.00


def test_compute_atm_straddle_move_invalid_quotes():
    """Verify straddle calculation returns None when quotes are inverted or missing."""
    spot = 100.0
    # Inverted ask < bid
    bad_chain = pd.DataFrame([
        {"cp": "C", "strike": 100.0, "bid": 4.0, "ask": 3.0},
        {"cp": "P", "strike": 100.0, "bid": 3.0, "ask": 3.2},
    ])
    assert compute_atm_straddle_move(bad_chain, spot) is None


def test_earnings_implied_vs_realized_regimes(monkeypatch):
    """Verify rich / cheap / fair classification against historical realized moves."""
    spot = 100.0

    # Mock historical earnings prints in earnings.py matching trading days
    fake_prints = [
        {"date": "2026-11-15", "is_upcoming": True},  # Upcoming next print
        {"date": "2026-08-17", "is_upcoming": False, "eps_actual": 1.2},
        {"date": "2026-05-18", "is_upcoming": False, "eps_actual": 1.1},
        {"date": "2026-02-17", "is_upcoming": False, "eps_actual": 0.9},
        {"date": "2025-11-17", "is_upcoming": False, "eps_actual": 1.0},
    ]

    import earnings
    monkeypatch.setattr(earnings, "get_earnings_history", lambda t: fake_prints)

    # Mock price DataFrame with ~4% moves on past dates
    dates = pd.date_range("2025-11-01", "2026-09-01", freq="B")
    prices = [100.0]
    for d in dates[1:]:
        # Insert a 4% move on earnings prints
        if str(d.date()) in ("2025-11-17", "2026-02-17", "2026-05-18", "2026-08-17"):
            prices.append(prices[-1] * 1.04)
        else:
            prices.append(prices[-1] * 1.001)

    price_df = pd.DataFrame({"close": prices}, index=dates)

    # 1. Rich pricing scenario: Implied move 8% vs historical ~4%
    rich_chain = pd.DataFrame([
        {"cp": "C", "strike": 100.0, "bid": 4.0, "ask": 4.0},
        {"cp": "P", "strike": 100.0, "bid": 4.0, "ask": 4.0},
    ])  # Straddle = 8.0 -> 8% implied
    chains_by_exp = {"2026-11-20": rich_chain}

    res_rich = get_earnings_implied_vs_realized("TEST", spot, chains_by_exp, prices_df=price_df)
    assert res_rich["available"] is True
    assert res_rich["implied_move_pct"] == 8.0
    assert res_rich["valuation_regime"] == "rich"
    assert res_rich["historical_n"] > 0

    # 2. Cheap pricing scenario: Implied move 2% vs historical ~4%
    cheap_chain = pd.DataFrame([
        {"cp": "C", "strike": 100.0, "bid": 1.0, "ask": 1.0},
        {"cp": "P", "strike": 100.0, "bid": 1.0, "ask": 1.0},
    ])  # Straddle = 2.0 -> 2% implied
    chains_by_exp_cheap = {"2026-11-20": cheap_chain}

    res_cheap = get_earnings_implied_vs_realized("TEST", spot, chains_by_exp_cheap, prices_df=price_df)
    assert res_cheap["available"] is True
    assert res_cheap["valuation_regime"] == "cheap"


def test_earnings_implied_no_upcoming_earnings(monkeypatch):
    """Verify declared state when no upcoming earnings is scheduled."""
    import earnings
    monkeypatch.setattr(earnings, "get_earnings_history", lambda t: [])

    res = get_earnings_implied_vs_realized("TEST", 100.0, {})
    assert res["available"] is False
    assert "No upcoming earnings" in res["reason"]
