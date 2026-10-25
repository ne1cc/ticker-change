"""Unit tests for GEX hardening, risk-free rate lookup, IV models, and consolidation.

Verifies Epic 4 issue #131 acceptance criteria:
- Single GEX/BS implementation; old shadow in app.py removed
- Risk-free rate source shown in GEX profile
- Continuous gamma_flip differs between sticky-strike and sticky-moneyness on a skewed fixture
- Idempotent daily snapshot written to gex_snapshots
"""
import sqlite3
import numpy as np
import pandas as pd
import pytest

import db
import options
from options.gex import (
    find_gamma_flip,
    compute_gex_profile,
    get_risk_free_rate,
)


def test_single_bs_implementation():
    """Verify that app.py uses options.calculate_greeks with no duplicate shadow."""
    import app
    assert hasattr(app, "calculate_greeks")
    assert app.calculate_greeks is options.calculate_greeks


def test_rate_source_surfaced():
    """Verify rate source is included in GEX profile."""
    spot = 100.0
    chain = pd.DataFrame([
        {"strike": 95.0, "cp": "P", "dte": 30, "iv": 0.25, "open_interest": 1000, "volume": 100, "bid": 1.0},
        {"strike": 105.0, "cp": "C", "dte": 30, "iv": 0.22, "open_interest": 1000, "volume": 100, "bid": 1.0},
    ])
    profile = compute_gex_profile(chain, spot)
    assert profile.rate_source is not None
    assert "FRED" in profile.rate_source or "Fallback" in profile.rate_source
    assert 0.0 < profile.rf_rate < 0.25


def test_gamma_flip_differs_between_iv_models_on_skewed_fixture():
    """Verify gamma flip level differs between sticky-strike and sticky-moneyness under steep skew."""
    spot = 100.0
    # Create an asymmetric, heavily skewed put-call options ladder
    # OTM Puts have steep skew (IV rises as strike drops)
    rows = []
    strikes = [80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0]
    for k in strikes:
        # Steep equity put skew: IV = 0.25 - 0.5 * (k - 100)/100
        skew_iv = 0.25 - 0.40 * ((k - spot) / spot)
        skew_iv = max(0.10, min(1.0, skew_iv))
        rows.append({
            "strike": k,
            "cp": "P",
            "dte": 45,
            "iv": skew_iv,
            "open_interest": 1500 if k < spot else 500,
            "volume": 200,
            "bid": 1.0,
        })
        rows.append({
            "strike": k,
            "cp": "C",
            "dte": 45,
            "iv": max(0.12, skew_iv * 0.85),
            "open_interest": 1200 if k > spot else 400,
            "volume": 200,
            "bid": 1.0,
        })

    chain_df = pd.DataFrame(rows)

    flip_strike = find_gamma_flip(chain_df, spot, iv_model="sticky_strike", r=0.045)
    flip_moneyness = find_gamma_flip(chain_df, spot, iv_model="sticky_moneyness", r=0.045)

    assert flip_strike is not None
    assert flip_moneyness is not None
    # On a skewed surface, sticky-moneyness shifts the IV smile when spot moves,
    # resulting in a measurable difference from sticky-strike flip
    assert flip_strike != flip_moneyness

    profile = compute_gex_profile(chain_df, spot, r=0.045, rate_source="Test Rate")
    assert profile.gamma_flip == flip_strike
    assert profile.gamma_flip_sticky_moneyness == flip_moneyness
    assert profile.gamma_flip_diff is not None
    assert profile.gamma_flip_diff == pytest.approx(round(flip_moneyness - flip_strike, 2), abs=1e-3)


def test_idempotent_gex_snapshots_recording(tmp_path, monkeypatch):
    """Verify daily GEX snapshot is written to gex_snapshots idempotently."""
    db_file = str(tmp_path / "test_gex.db")
    monkeypatch.setattr(db, "DB_PATH", db_file)
    db.init_db()

    # Insert snapshot for AAPL on 2026-10-24
    db.insert_gex_snapshot("AAPL", "2026-10-24", gamma_flip=220.5, call_wall=230.0, put_wall=210.0, spot=222.0)

    # Re-insert on the same date with updated or duplicate values (should ignore without error)
    db.insert_gex_snapshot("AAPL", "2026-10-24", gamma_flip=220.5, call_wall=230.0, put_wall=210.0, spot=222.0)

    with db.get_conn() as conn:
        rows = conn.execute("SELECT * FROM gex_snapshots WHERE ticker = 'AAPL' AND date = '2026-10-24'").fetchall()
        assert len(rows) == 1
        assert rows[0]["gamma_flip"] == 220.5
        assert rows[0]["call_wall"] == 230.0
        assert rows[0]["put_wall"] == 210.0
