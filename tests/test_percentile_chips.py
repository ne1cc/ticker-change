"""Tests for percentile chips on analytics stat tiles and /api/analytics (Issue #119)."""
import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import numpy as np

import app as app_module
import db
import gics
import percentiles
import snapshot_store
import universe_metrics


class TestPercentileChips(unittest.TestCase):
    def setUp(self):
        self._orig_db = db.DB_PATH
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.temp_db_path = path
        db.DB_PATH = path
        db.init_db()
        self.client = app_module.app.test_client()
        self.client.testing = True

    def tearDown(self):
        db.DB_PATH = self._orig_db
        try:
            os.remove(self.temp_db_path)
        except OSError:
            pass

    def _seed_ticker(self, ticker: str = "AAPL", n_days: int = 260):
        dates = pd.bdate_range(end="2026-10-23", periods=n_days)
        np.random.seed(123)
        ret = np.random.normal(0.0005, 0.015, size=n_days)
        price_series = 150.0 * np.exp(np.cumsum(ret))
        df = pd.DataFrame(
            {
                "Open": price_series,
                "High": price_series * 1.01,
                "Low": price_series * 0.99,
                "Close": price_series,
                "Volume": [10_000_000] * n_days,
            },
            index=dates,
        )
        db.store_prices(ticker, df)
        return df

    def test_missing_snapshot_renders_nothing_misleading(self):
        """Missing snapshot renders nothing misleading (declared state), verified by template test."""
        # Ensure no snapshot is published
        current_snap = universe_metrics.current_universe_metrics()
        self.assertIsNone(current_snap)

        self._seed_ticker("AAPL")

        resp = self.client.get("/analytics?ticker=AAPL")
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode("utf-8")

        # Must not render bogus percentiles or misleading chips like P0 or NaN
        self.assertNotIn("percentile-chip", html)
        self.assertNotIn("P0 ·", html)
        self.assertNotIn("NaN", html)

    def test_api_analytics_includes_percentile_fields(self):
        """API response /api/analytics/<ticker> includes percentile fields."""
        self._seed_ticker("AAPL")

        # Publish a valid snapshot
        snapshot = {
            "snapshot_id": "snap-20261023",
            "as_of": "2026-10-23",
            "by_metric": {
                "annualised_vol": {
                    "all": [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 60.0, 70.0, 80.0],
                    "sectors": {
                        "Information Technology": [15.0, 20.0, 25.0, 30.0, 35.0, 40.0],
                    },
                },
                "max_drawdown": {
                    "all": [-30.0, -25.0, -20.0, -15.0, -10.0, -5.0, -4.0, -3.0, -2.0, -1.0],
                    "sectors": {
                        "Information Technology": [-25.0, -20.0, -15.0, -10.0, -5.0],
                    },
                },
            },
        }
        snapshot_store.publish("universe_metrics", "snap-20261023", snapshot)

        # Mock gics to return Information Technology for AAPL
        with patch.object(gics, "sector_of", return_value="Information Technology"):
            resp = self.client.get("/api/analytics/AAPL")
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()

            self.assertIn("percentiles", data)
            self.assertEqual(data["gics_sector"], "Information Technology")

            vol_pct = data["percentiles"].get("annualised_vol")
            self.assertIsNotNone(vol_pct)
            self.assertEqual(vol_pct["direction"], "higher_is_riskier")
            self.assertIn("universe", vol_pct)
            self.assertIsNotNone(vol_pct["universe"]["percentile"])
            self.assertEqual(vol_pct["universe"]["peer_count"], 11)
            self.assertEqual(vol_pct["universe"]["as_of"], "2026-10-23")

            self.assertIn("sector", vol_pct)
            self.assertEqual(vol_pct["sector"]["name"], "Information Technology")
            self.assertIsNotNone(vol_pct["sector"]["percentile"])
            self.assertEqual(vol_pct["sector"]["peer_count"], 6)

    def test_stat_tiles_show_universe_and_sector_percentile_when_available(self):
        """Every stat tile in the summary shows universe and sector percentile when available."""
        self._seed_ticker("AAPL")

        snapshot = {
            "snapshot_id": "snap-20261023",
            "as_of": "2026-10-23",
            "by_metric": {
                "annualised_vol": {
                    "all": [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 60.0, 70.0, 80.0],
                    "sectors": {
                        "Information Technology": [15.0, 20.0, 25.0, 30.0, 35.0, 40.0],
                    },
                },
                "max_drawdown": {
                    "all": [-30.0, -25.0, -20.0, -15.0, -10.0, -5.0, -4.0, -3.0, -2.0, -1.0],
                    "sectors": {
                        "Information Technology": [-25.0, -20.0, -15.0, -10.0, -5.0],
                    },
                },
            },
        }
        snapshot_store.publish("universe_metrics", "snap-20261023", snapshot)

        with patch.object(gics, "sector_of", return_value="Information Technology"):
            resp = self.client.get("/analytics?ticker=AAPL")
            self.assertEqual(resp.status_code, 200)
            html = resp.data.decode("utf-8")

            # Chips should render on the page
            self.assertIn("percentile-chip", html)
            self.assertIn("S&P 500:", html)
            self.assertIn("Sector (Information Technology):", html)
            self.assertIn("as of 2026-10-23", html)
