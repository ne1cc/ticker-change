"""Radar warmer: fills the three payload caches, tolerates failures."""
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

import app as app_module  # noqa: E402
import earnings  # noqa: E402


def _mk_df(days=300):
    idx = pd.bdate_range("2024-06-03", periods=days)
    close = 100 + np.arange(days)
    return pd.DataFrame({"Open": close, "High": close, "Low": close,
                         "Close": close, "Volume": np.full(days, 1000)}, index=idx)


class TestRadarWarmer(unittest.TestCase):
    def setUp(self):
        # Each warm claims a 6h cross-worker lock; release it between tests.
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
