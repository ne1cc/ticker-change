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
    def test_warm_populates_payload_caches(self):
        db.store_prices("WARM1", _mk_df())
        with patch.object(earnings, "fetch_earnings_history",
                          return_value=[{"date": "2025-01-30", "eps_estimate": 1.0,
                                         "eps_actual": 1.1, "surprise_pct": 10.0,
                                         "is_upcoming": False}]), \
             patch.object(app_module, "get_fundamentals",
                          return_value={"Name": "Warm Co"}), \
             patch.object(app_module.providers, "get_short_interest",
                          return_value={"short_pct_float": 0.05,
                                        "days_to_cover": 2.0}):
            app_module._warm_radar_cache(symbols=["WARM1"], background=False)
        self.assertIsNotNone(db.cache_get("yfinance", "earnings:WARM1", 24))
        self.assertIsNotNone(db.cache_get("yfinance", "fundamentals:WARM1", 24))
        self.assertIsNotNone(db.cache_get("yfinance", "short:WARM1", 24))
