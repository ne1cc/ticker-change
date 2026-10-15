"""Radar warmer: fills the three payload caches, tolerates failures."""
import os
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd

import db

_fd, _DB_PATH = tempfile.mkstemp(suffix=".db")
os.close(_fd)
db.DB_PATH = _DB_PATH
db.init_db()

import app as app_module  # noqa: E402
import earnings  # noqa: E402


def _mk_df(days=300):
    idx = pd.bdate_range("2024-06-03", periods=days)
    close = 100 + np.arange(days)
    return pd.DataFrame({"Open": close, "High": close, "Low": close,
                         "Close": close, "Volume": np.full(days, 1000)}, index=idx)


class TestRadarWarmer(unittest.TestCase):
    def setUp(self):
        # Each warm claims a cross-worker lock; release it between tests.
        with db.get_conn() as conn:
            conn.execute("DELETE FROM api_cache WHERE provider = 'internal' "
                         "AND key = 'radar_warmer_lock'")

    def test_warm_populates_payload_caches(self):
        db.store_prices("WARM1", _mk_df())
        with patch.object(earnings, "fetch_earnings_history",
                          return_value=[{"date": "2025-01-30", "eps_estimate": 1.0,
                                         "eps_actual": 1.1, "surprise_pct": 10.0,
                                         "is_upcoming": False}]), \
             patch.object(app_module.heatmap, "refresh_universe",
                          return_value={"fetched": 0, "failed": []}), \
             patch.object(app_module, "get_fundamentals",
                          return_value={"Name": "Warm Co"}), \
             patch.object(app_module.providers, "get_short_interest",
                          return_value={"short_pct_float": 0.05,
                                        "days_to_cover": 2.0}):
            app_module._warm_radar_cache(symbols=["WARM1"], background=False)
        self.assertIsNotNone(db.cache_get("yfinance", "earnings:WARM1", 24))
        self.assertIsNotNone(db.cache_get("yfinance", "fundamentals:WARM1", 24))
        self.assertIsNotNone(db.cache_get("yfinance", "short:WARM1", 24))

    def test_full_warm_backfills_sp500_year_and_skips_etfs(self):
        # The heatmap warmer keeps only ~10 sessions per constituent, so the
        # radar warmer must backfill a year before universe_symbols(200)
        # can see anything beyond the sector ETFs.
        db.store_prices("XLK", _mk_df())
        db.store_prices("SPY", _mk_df())
        db.store_prices("SPX1", _mk_df())
        with patch.object(app_module.heatmap, "get_constituents",
                          return_value=[{"symbol": "SPX1"}]), \
             patch.object(app_module.heatmap, "refresh_universe",
                          return_value={"fetched": 0, "failed": []}) as refresh, \
             patch.object(earnings, "fetch_earnings_history", return_value=None), \
             patch.object(app_module, "get_fundamentals", return_value=None), \
             patch.object(app_module.providers, "get_short_interest",
                          return_value=None) as short, \
             patch.object(app_module.radar, "publish_snapshot"):
            app_module._warm_radar_cache(background=False)
        backfilled, kwargs = refresh.call_args
        self.assertEqual(kwargs.get("period"), "1y")
        self.assertIn("SPX1", backfilled[0])
        self.assertIn("SPY", backfilled[0])
        self.assertIn("XLK", backfilled[0])
        warmed = {c.args[0] for c in short.call_args_list}
        self.assertIn("SPX1", warmed)
        self.assertNotIn("XLK", warmed)
        self.assertNotIn("SPY", warmed)

    def test_universe_excludes_benchmark_etfs(self):
        db.store_prices("XLF", _mk_df())
        db.store_prices("SPY", _mk_df())
        db.store_prices("EQ1", _mk_df())
        universe = app_module.radar.universe_symbols(min_bars=200)
        self.assertIn("EQ1", universe)
        self.assertNotIn("XLF", universe)
        self.assertNotIn("SPY", universe)

    def test_lock_left_by_killed_warm_lapses_within_an_hour(self):
        # A redeploy kills the warm mid-run; its lock must not block the
        # next container's warm for hours.
        stale = (datetime.utcnow() - timedelta(hours=1)).isoformat()
        with db.get_conn() as conn:
            conn.execute("INSERT INTO api_cache (provider, key, fetched_at, payload) "
                         "VALUES ('internal', 'radar_warmer_lock', ?, '{}')", (stale,))
        with patch.object(app_module.heatmap, "refresh_universe",
                          return_value={"fetched": 0, "failed": []}) as refresh, \
             patch.object(app_module.radar, "universe_symbols", return_value=[]):
            app_module._warm_radar_cache(symbols=["WARM1"], background=False)
        refresh.assert_called_once()

    def test_warm_due_until_a_full_warm_completes(self):
        db.cache_set("radar", "warm_progress", {"done": 175, "total": 501})
        self.assertTrue(app_module._radar_warm_due())  # interrupted
        db.cache_set("radar", "warm_progress", {"done": 501, "total": 501})
        self.assertFalse(app_module._radar_warm_due())  # fresh and complete
        old = (datetime.utcnow() - timedelta(hours=13)).isoformat()
        with db.get_conn() as conn:
            conn.execute("UPDATE api_cache SET fetched_at = ? WHERE provider = 'radar' "
                         "AND key = 'warm_progress'", (old,))
        self.assertTrue(app_module._radar_warm_due())  # older than 12h
