"""Radar scans: built entirely from seeded api_cache + SQLite, never fetch."""
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


class _PinnedDB(unittest.TestCase):
    """Pin this module's temp DB around each test: other test modules in the
    suite reassign db.DB_PATH at import time and their cache writes would
    otherwise leak into these cache-emptiness assertions."""

    def setUp(self):
        self._orig_db = db.DB_PATH
        db.DB_PATH = _DB_PATH

    def tearDown(self):
        db.DB_PATH = self._orig_db


def _store(sym, days=320, drift=0.001, volume=1_000_000):
    idx = pd.bdate_range("2024-06-03", periods=days)
    close = 100 * np.exp(np.cumsum(np.full(days, drift)))
    db.store_prices(sym, pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close,
         "Volume": np.full(days, volume)}, index=idx))


class TestDriftScan(_PinnedDB):
    def test_rows_ranked_with_remaining_window(self):
        _store("DRFTA", drift=0.002)
        db.cache_set("yfinance", "earnings:DRFTA", [
            {"date": pd.Timestamp.today().date().isoformat(), "eps_estimate": 1,
             "eps_actual": 1.2, "surprise_pct": 10.0, "is_upcoming": False}])
        db.cache_set("yfinance", "earnings:DRFTB", [
            {"date": (pd.Timestamp.today() - pd.Timedelta(days=40)).date().isoformat(),
             "eps_estimate": 1, "eps_actual": 0.8, "surprise_pct": -20.0,
             "is_upcoming": False}])
        out = radar.build_drift_scan()
        syms = [r["symbol"] for r in out["rows"]]
        self.assertIn("DRFTA", syms)
        for r in out["rows"]:
            self.assertIn("remaining", r)
            self.assertLessEqual(r["remaining"], 30)
            self.assertIn("days_since", r)

    def test_empty_cache_yields_empty_rows_not_crash(self):
        out = radar.build_drift_scan()
        self.assertEqual(out["rows"], [])
        self.assertEqual(out["coverage"], 0)


class TestValueScan(_PinnedDB):
    def test_cheapest_with_positive_momentum_shortlisted(self):
        _store("VALA", drift=0.002)
        _store("VALB", drift=-0.001)
        db.cache_set("yfinance", "fundamentals:VALA", {
            "Industry": "Tech", "Forward P/E": 8.0, "Price / Sales": 1.0})
        db.cache_set("yfinance", "fundamentals:VALB", {
            "Industry": "Tech", "Forward P/E": 60.0, "Price / Sales": 9.0})
        out = radar.build_value_scan()
        syms = [r["symbol"] for r in out["rows"]]
        self.assertIn("VALA", syms)
        self.assertNotIn("VALB", syms)  # expensive AND negative momentum

    def test_empty_cache_yields_empty_rows(self):
        out = radar.build_value_scan()
        self.assertIsInstance(out["rows"], list)


class TestSqueezeScan(_PinnedDB):
    def test_floor_and_ranking(self):
        _store("SQZA", drift=0.003, volume=5_000_000)
        _store("SQZB", drift=0.001, volume=10_000)  # below liquidity floor
        db.cache_set("yfinance", "short:SQZA", {
            "short_pct_float": 0.22, "days_to_cover": 7.0, "si_mom_change": 0.3,
            "shares_short": 1, "shares_short_prior_month": 1, "as_of": 1,
            "source": "reported"})
        db.cache_set("yfinance", "short:SQZB", {
            "short_pct_float": 0.30, "days_to_cover": 9.0, "si_mom_change": None,
            "shares_short": 1, "shares_short_prior_month": None, "as_of": 1,
            "source": "reported"})
        with patch.object(radar, "get_tunable", return_value=5_000_000.0):
            out = radar.build_squeeze_scan()
        syms = [r["symbol"] for r in out["rows"]]
        self.assertEqual(syms, ["SQZA"])
        row = out["rows"][0]
        self.assertIn("squeeze_score", row)
        self.assertIn("squeeze_level", row)

    def test_empty_cache_yields_empty_rows(self):
        out = radar.build_squeeze_scan()
        self.assertEqual(out["rows"], [])
