"""Radar builder v2s: contract-driven rows over seeded caches, never fetch."""
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import db

_fd, _DB_PATH = tempfile.mkstemp(suffix=".db")
os.close(_fd)
db.DB_PATH = _DB_PATH
db.init_db()

import radar  # noqa: E402
import radar_contracts  # noqa: E402


class _PinnedDB(unittest.TestCase):
    """Pin this module's temp DB around each test: other test modules in the
    suite reassign db.DB_PATH at import time and their cache writes would
    otherwise leak into these cache-emptiness assertions."""

    def setUp(self):
        self._orig_db = db.DB_PATH
        db.DB_PATH = _DB_PATH

    def tearDown(self):
        db.DB_PATH = self._orig_db

    def _wipe(self):
        with db.get_conn() as conn:
            conn.execute("DELETE FROM daily_prices")
            conn.execute("DELETE FROM api_cache WHERE provider = 'yfinance'")


def _store(sym, days=320, drift=0.001, volume=1_000_000):
    idx = pd.bdate_range("2024-06-03", periods=days)
    close = 100 * np.exp(np.cumsum(np.full(days, drift)))
    db.store_prices(sym, pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close,
         "Volume": np.full(days, volume)}, index=idx))


def _build():
    return radar.build_snapshot(radar.load_context())


class TestDriftRows(_PinnedDB):
    def test_empty_cache_yields_empty_rows_not_crash(self):
        self._wipe()
        snap = _build()
        self.assertEqual(snap["drift"]["rows"], [])
        self.assertEqual(snap["drift"]["excluded"], [])

    def test_missing_prices_land_in_exclusion_ledger(self):
        ctx = {"symbols": ["DRFTX"], "prices": {}, "fundamentals": {},
               "earnings": {"DRFTX": [{"date": "2024-09-06", "eps_estimate": 1,
                                       "eps_actual": 1.2, "surprise_pct": 10.0,
                                       "is_upcoming": False}]},
               "short": {}, "source_as_of": {}, "built_at": radar.now_iso()}
        rows, excluded = radar._drift_rows(ctx)
        self.assertEqual(rows, [])
        self.assertEqual(excluded, [{"symbol": "DRFTX", "reason": "no-prices"}])

    def test_rows_carry_full_contract_and_rank_by_remaining_window(self):
        _store("DRFTA", drift=0.002)
        db.cache_set("yfinance", "earnings:DRFTA", [
            {"date": "2024-09-06", "eps_estimate": 1, "eps_actual": 1.2,
             "surprise_pct": 10.0, "is_upcoming": False}])
        last_day = pd.bdate_range("2024-06-03", periods=320)[-1].date().isoformat()
        _store("DRFTB", drift=0.001)
        db.cache_set("yfinance", "earnings:DRFTB", [
            {"date": last_day, "eps_estimate": 1, "eps_actual": 0.8,
             "surprise_pct": -20.0, "is_upcoming": False}])
        rows = _build()["drift"]["rows"]
        self.assertEqual([r["symbol"] for r in rows], ["DRFTB", "DRFTA"])
        remaining = [r["sessions_to_20d"] for r in rows]
        self.assertEqual(remaining, sorted(remaining, reverse=True))
        for r in rows:
            for key in radar_contracts.METRIC_KEYS_BY_TAB["drift"]:
                self.assertIn(key, r)


class TestValueRows(_PinnedDB):
    def test_empty_cache_yields_empty_rows(self):
        self._wipe()
        snap = _build()
        self.assertEqual(snap["value"]["rows"], [])
        self.assertEqual(snap["value"]["excluded"], [])

    def test_cheapest_names_rank_first_with_full_contract(self):
        _store("VALA", drift=0.002)
        _store("VALB", drift=-0.001)
        db.cache_set("yfinance", "fundamentals:VALA", {
            "Sector": "Technology", "Industry": "Tech", "Market Cap": 1e9,
            "Forward P/E": 8.0, "Price / Sales": 1.0})
        db.cache_set("yfinance", "fundamentals:VALB", {
            "Sector": "Technology", "Industry": "Tech", "Market Cap": 2e9,
            "Forward P/E": 60.0, "Price / Sales": 9.0})
        rows = _build()["value"]["rows"]
        self.assertEqual([r["symbol"] for r in rows], ["VALA", "VALB"])
        self.assertGreater(rows[0]["value_score"], rows[1]["value_score"])
        for r in rows:
            for key in radar_contracts.METRIC_KEYS_BY_TAB["value"]:
                self.assertIn(key, r)

    def test_negative_multiple_flags_nm_and_skips_pool(self):
        _store("VALN", drift=0.001)
        db.cache_set("yfinance", "fundamentals:VALN", {
            "Sector": "Tech", "Industry": "Tech", "Forward P/E": -8.0})
        row = next(r for r in _build()["value"]["rows"]
                   if r["symbol"] == "VALN")
        self.assertTrue(row["fwd_pe_nm"])
        self.assertIsNone(row["fwd_pe_pct"])

    def test_thin_pool_marks_insufficient_peers(self):
        _store("VALS", drift=0.001)
        db.cache_set("yfinance", "fundamentals:VALS", {
            "Sector": "Utilities", "Industry": "Water", "Forward P/E": 9.0})
        row = next(r for r in _build()["value"]["rows"]
                   if r["symbol"] == "VALS")
        self.assertEqual(row["_state"], "insufficient-peers")
        self.assertIsNone(row["peer_percentile"])


class TestSqueezeRows(_PinnedDB):
    SI = {"short_pct_float": 0.22, "days_to_cover": 7.0, "si_mom_change": 0.3,
          "shares_short": 1, "shares_short_prior_month": 1, "as_of": 1,
          "source": "reported"}

    def test_empty_cache_yields_empty_rows(self):
        self._wipe()
        snap = _build()
        self.assertEqual(snap["squeeze"]["rows"], [])
        self.assertEqual(snap["squeeze"]["excluded"], [])

    def test_floor_and_ranking(self):
        _store("SQZA", drift=0.003, volume=5_000_000)
        _store("SQZB", drift=0.001, volume=10_000)  # below liquidity floor
        db.cache_set("yfinance", "short:SQZA", dict(self.SI))
        db.cache_set("yfinance", "short:SQZB",
                     {**self.SI, "si_mom_change": None,
                      "shares_short_prior_month": None})
        with patch.object(radar, "get_tunable", return_value=5_000_000.0):
            snap = _build()
        self.assertEqual([r["symbol"] for r in snap["squeeze"]["rows"]], ["SQZA"])
        reasons = {e["symbol"]: e["reason"] for e in snap["squeeze"]["excluded"]}
        self.assertEqual(reasons.get("SQZB"), "below-floor")
        row = snap["squeeze"]["rows"][0]
        self.assertIn("squeeze_score", row)
        self.assertIn("squeeze_level", row)
        for key in radar_contracts.METRIC_KEYS_BY_TAB["squeeze"]:
            self.assertIn(key, row)

    def test_squeeze_score_pinned_to_percent_scale(self):
        _store("SQZC", drift=0.003, volume=5_000_000)
        db.cache_set("yfinance", "short:SQZC",
                     {**self.SI, "si_mom_change": None})
        with patch.object(radar, "get_tunable", return_value=5_000_000.0):
            rows = _build()["squeeze"]["rows"]
        row = next(r for r in rows if r["symbol"] == "SQZC")
        expected = radar.microstructure.compute_squeeze_risk_index(22.0, 7.0)[0]
        self.assertEqual(row["squeeze_score"], expected)

    def test_missing_si_fields_excluded_as_assumed(self):
        _store("SQZD", drift=0.001, volume=5_000_000)
        db.cache_set("yfinance", "short:SQZD",
                     {"short_pct_float": 0.1, "days_to_cover": None})
        reasons = {e["symbol"]: e["reason"] for e in _build()["squeeze"]["excluded"]}
        self.assertEqual(reasons.get("SQZD"), "assumed-si")
