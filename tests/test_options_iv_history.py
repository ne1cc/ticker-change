"""Unit tests for IV history persistence, real IV rank/percentile, VRP history, and term slope.

Verifies Epic 4 issues #124, #125, #126, #130 acceptance criteria:
- #124: Re-running on same date doesn't duplicate rows; records source and as-of; queryable history status
- #125: IV rank / percentile computed on stored IV history; below 63 days renders accumulating (n/252) with no proxy numbers
- #126: Real VRP history chart with percentile bands and dynamic regime label based on percentile
- #130: Term-structure slope and 25-delta skew historical percentiles
"""
import sqlite3
from datetime import datetime, timedelta, timezone
import pytest

from options.storage import (
    init_options_db,
    record_snapshot,
    get_snapshot_history,
    get_history_count,
    get_history_status,
    compute_iv_rank_and_percentile,
    get_vrp_history_with_bands,
    get_term_structure_and_skew_history,
)


def test_record_snapshot_idempotency_and_metadata(tmp_path):
    """Verify that recording multiple snapshots on the same date replaces without duplicating (#124)."""
    db_file = str(tmp_path / "test_options.db")
    symbol = "NVDA"

    # 1. Record initial snapshot
    ok1 = record_snapshot(
        symbol=symbol,
        spot=120.0,
        atm_iv=45.0,
        hv30=38.0,
        vrp=7.0,
        call_wall=130.0,
        put_wall=110.0,
        gamma_flip=118.0,
        total_gex=25.0,
        skew_25d=1.12,
        term_slope=0.42,
        source="yfinance",
        convention="naive",
        as_of="2026-10-24T12:00:00Z",
        db_path=db_file,
    )
    assert ok1 is True
    assert get_history_count(symbol, db_path=db_file) == 1

    # 2. Re-record on the same date with updated IV
    ok2 = record_snapshot(
        symbol=symbol,
        spot=121.0,
        atm_iv=46.5,
        hv30=38.0,
        vrp=8.5,
        call_wall=130.0,
        put_wall=110.0,
        gamma_flip=118.5,
        total_gex=26.0,
        skew_25d=1.14,
        term_slope=0.45,
        source="yfinance",
        convention="naive",
        as_of="2026-10-24T16:00:00Z",
        db_path=db_file,
    )
    assert ok2 is True
    # Still exactly 1 row (no duplicate)
    assert get_history_count(symbol, db_path=db_file) == 1

    history = get_snapshot_history(symbol, db_path=db_file)
    assert len(history) == 1
    row = history[0]
    assert row["spot"] == 121.0
    assert row["atm_iv"] == 46.5
    assert row["vrp"] == 8.5
    assert row["term_slope"] == 0.45
    assert row["source"] == "yfinance"
    assert row["as_of"] == "2026-10-24T16:00:00Z"


def test_history_status_accumulating_state(tmp_path):
    """Verify history status correctly declares accumulating state (n/252) when below threshold (#124)."""
    db_file = str(tmp_path / "test_options.db")
    status_0 = get_history_status("AAPL", target=252, db_path=db_file)
    assert status_0["days_recorded"] == 0
    assert status_0["is_accumulating"] is True
    assert status_0["status"] == "accumulating (0/252)"

    # Add 5 snapshots
    init_options_db(db_file)
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    for i in range(5):
        d_str = (datetime(2026, 10, 1) + timedelta(days=i)).strftime("%Y-%m-%d")
        cur.execute(
            """INSERT INTO options_iv_history (symbol, timestamp, date, spot, atm_iv, hv30, vrp, total_gex, skew_25d, term_slope, source, convention, as_of)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("AAPL", f"{d_str}T00:00:00Z", d_str, 150.0, 25.0 + i, 20.0, 5.0 + i, 10.0, 1.05, 0.1, "yfinance", "naive", f"{d_str}T00:00:00Z"),
        )
    conn.commit()
    conn.close()

    status_5 = get_history_status("AAPL", target=252, db_path=db_file)
    assert status_5["days_recorded"] == 5
    assert status_5["is_accumulating"] is True
    assert status_5["status"] == "accumulating (5/252)"


def test_iv_rank_and_percentile_thresholds(tmp_path):
    """Verify IV rank/percentile requires minimum 63 days and never uses proxy numbers (#125)."""
    db_file = str(tmp_path / "test_options.db")
    init_options_db(db_file)

    # 1. Under minimum history (e.g. 10 days) -> returns accumulating and None numeric values
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    for i in range(10):
        d_str = (datetime(2026, 1, 1) + timedelta(days=i)).strftime("%Y-%m-%d")
        cur.execute(
            """INSERT INTO options_iv_history (symbol, timestamp, date, spot, atm_iv, hv30, vrp, total_gex, skew_25d, term_slope, source, convention, as_of)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("MSFT", f"{d_str}T00:00:00Z", d_str, 400.0, 20.0 + i, 18.0, 2.0 + i, 10.0, 1.05, 0.1, "yfinance", "naive", f"{d_str}T00:00:00Z"),
        )
    conn.commit()
    conn.close()

    short_res = compute_iv_rank_and_percentile("MSFT", current_iv=25.0, min_obs=63, db_path=db_file)
    assert short_res["available"] is False
    assert short_res["status"] == "accumulating (10/252)"
    assert short_res["iv_rank"] is None
    assert short_res["iv_percentile"] is None

    # 2. Populate 100 days (exceeding min_obs=63) -> returns real IV rank and percentile
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    # Populate IV from 20.0 to 40.0 across 100 days
    for i in range(10, 100):
        d_str = (datetime(2026, 1, 1) + timedelta(days=i)).strftime("%Y-%m-%d")
        iv_val = 20.0 + (i / 100.0) * 20.0  # Range: [20.0, 40.0]
        cur.execute(
            """INSERT INTO options_iv_history (symbol, timestamp, date, spot, atm_iv, hv30, vrp, total_gex, skew_25d, term_slope, source, convention, as_of)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("MSFT", f"{d_str}T00:00:00Z", d_str, 400.0, iv_val, 18.0, iv_val - 18.0, 10.0, 1.05, 0.1, "yfinance", "naive", f"{d_str}T00:00:00Z"),
        )
    conn.commit()
    conn.close()

    full_res = compute_iv_rank_and_percentile("MSFT", current_iv=30.0, min_obs=63, db_path=db_file)
    assert full_res["available"] is True
    assert full_res["status"] == "ready"
    assert full_res["n_obs"] == 100
    # Range is [20.0, ~40.0], current IV 30.0 is right in the middle (~50% rank)
    assert 45.0 <= full_res["iv_rank"] <= 55.0
    assert 45.0 <= full_res["iv_percentile"] <= 55.0


def test_vrp_history_with_percentile_bands(tmp_path):
    """Verify VRP history chart calculation, percentile bands, and dynamic regime (#126)."""
    db_file = str(tmp_path / "test_options.db")
    init_options_db(db_file)

    # 1. Under min_obs=21 -> accumulating
    short_vrp = get_vrp_history_with_bands("TSLA", min_obs=21, db_path=db_file)
    assert short_vrp["available"] is False
    assert short_vrp["status"] == "accumulating (0/252)"

    # 2. Populate 50 days of VRP
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    for i in range(50):
        d_str = (datetime(2026, 1, 1) + timedelta(days=i)).strftime("%Y-%m-%d")
        vrp_val = -2.0 + (i / 50.0) * 10.0  # VRP from -2.0 to +8.0
        cur.execute(
            """INSERT INTO options_iv_history (symbol, timestamp, date, spot, atm_iv, hv30, vrp, total_gex, skew_25d, term_slope, source, convention, as_of)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("TSLA", f"{d_str}T00:00:00Z", d_str, 200.0, 50.0, 45.0, vrp_val, 10.0, 1.1, 0.2, "yfinance", "naive", f"{d_str}T00:00:00Z"),
        )
    conn.commit()
    conn.close()

    vrp_res = get_vrp_history_with_bands("TSLA", min_obs=21, db_path=db_file)
    assert vrp_res["available"] is True
    assert vrp_res["n_obs"] == 50
    assert len(vrp_res["vrp_series"]) == 50
    assert vrp_res["p10"] < vrp_res["p25"] < vrp_res["median"] < vrp_res["p75"] < vrp_res["p90"]
    # Last reading was the maximum (+8.0) -> high regime
    assert vrp_res["regime"] == "high"
    assert "High VRP" in vrp_res["regime_label"]


def test_term_structure_and_skew_history(tmp_path):
    """Verify term structure slope and skew historical percentile calculation (#130)."""
    db_file = str(tmp_path / "test_options.db")
    init_options_db(db_file)

    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    for i in range(70):
        d_str = (datetime(2026, 1, 1) + timedelta(days=i)).strftime("%Y-%m-%d")
        cur.execute(
            """INSERT INTO options_iv_history (symbol, timestamp, date, spot, atm_iv, hv30, vrp, total_gex, skew_25d, term_slope, source, convention, as_of)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("AMZN", f"{d_str}T00:00:00Z", d_str, 180.0, 30.0, 25.0, 5.0, 10.0, 1.0 + i*0.005, 0.1 + i*0.01, "yfinance", "naive", f"{d_str}T00:00:00Z"),
        )
    conn.commit()
    conn.close()

    res = get_term_structure_and_skew_history("AMZN", current_slope=0.45, current_skew=1.20, min_obs=63, db_path=db_file)
    assert res["available"] is True
    assert res["n_obs"] == 70
    assert 0.0 <= res["slope_percentile"] <= 100.0
    assert 0.0 <= res["skew_percentile"] <= 100.0
