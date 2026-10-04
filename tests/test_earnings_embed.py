"""/analytics earnings embed: guarded attach, degrade-to-None, page render."""
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

FAKE_SECTION = {
    "events": [{"date": "2025-04-24", "surprise_pct": -12.5, "car": -0.05,
                "car_t_stat": -2.2, "car_p_value": 0.03, "is_significant_95": True,
                "drift_5d": -0.02, "drift_10d": -0.03, "drift_20d": -0.04}],
    "summary": {"n_events": 1, "n_significant": 1, "significance_rate": 1.0,
                "avg_car_beat": None, "avg_car_miss": -0.05},
    "next_print": {"date": "2026-10-22", "eps_estimate": 1.7, "is_upcoming": True},
    "implied_move": {"implied_move": 5.1, "days_to_earnings": 12,
                     "basis_days_to_exp": 40},
    "realized_move": 0.04,
    "event_vol": "fair",
}


class TestEarningsEmbed(unittest.TestCase):
    def test_guarded_attach_returns_section(self):
        with patch.object(app_module.earnings, "build_earnings_section",
                          return_value=FAKE_SECTION):
            section = app_module._guard_section(
                "earnings", lambda: app_module.earnings.build_earnings_section("AAPL"))
        self.assertEqual(section["summary"]["n_events"], 1)

    def test_boom_degrades_to_none(self):
        with patch.object(app_module.earnings, "build_earnings_section",
                          side_effect=RuntimeError("boom")):
            self.assertIsNone(app_module._guard_section(
                "earnings", lambda: app_module.earnings.build_earnings_section("AAPL")))

    def test_page_renders_section_without_500(self):
        # Seed prices so compute_analytics renders the full layout (with no
        # prices it renders the error page, which never includes the partial).
        idx = pd.bdate_range("2024-01-01", periods=300)
        close = 100 + np.arange(300)
        frame = pd.DataFrame({"Open": close, "High": close, "Low": close,
                              "Close": close, "Volume": np.full(300, 1000)},
                             index=idx)
        db.store_prices("AAPL", frame)
        db.store_prices("SPY", frame)
        client = app_module.app.test_client()
        with patch.object(app_module.earnings, "build_earnings_section",
                          return_value=FAKE_SECTION), \
             patch.object(app_module, "get_fundamentals", return_value=None):
            resp = client.get("/analytics?ticker=AAPL")
        html = resp.get_data(as_text=True)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("Earnings Event Study", html)
        # Partial-body-only marker: not satisfiable by the Row 9 HTML comment.
        self.assertIn("prints studied", html)

    def test_page_renders_nothing_when_no_earnings(self):
        idx = pd.bdate_range("2024-01-01", periods=300)
        close = 100 + np.arange(300)
        frame = pd.DataFrame({"Open": close, "High": close, "Low": close,
                              "Close": close, "Volume": np.full(300, 1000)},
                             index=idx)
        db.store_prices("AAPL", frame)
        db.store_prices("SPY", frame)
        client = app_module.app.test_client()
        with patch.object(app_module.earnings, "build_earnings_section",
                          return_value=None), \
             patch.object(app_module, "get_fundamentals", return_value=None):
            resp = client.get("/analytics?ticker=AAPL")
        html = resp.get_data(as_text=True)
        self.assertEqual(resp.status_code, 200)
        self.assertNotIn("prints studied", html)
