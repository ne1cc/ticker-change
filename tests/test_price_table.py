import os
import tempfile
import unittest
from unittest.mock import patch
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

import db
from app import (
    app,
    calculate_date_periods,
    _find_closest_trading_price,
    _close_n_sessions_ago,
    get_stock_data,
    SHORT_DAY_PERIODS,
)


class TestPriceTable(unittest.TestCase):
    def setUp(self):
        self.patch_yf_live = patch("app.get_current_price_yfinance", return_value=None)
        self.patch_chart = patch("app.generate_stock_chart", return_value=None)
        self.patch_yf_live.start()
        self.patch_chart.start()

        self.orig_db_path = db.DB_PATH
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        db.DB_PATH = self.temp_db.name
        db.init_db()
        self.client = app.test_client()
        self.client.testing = True

        # Construct 5+ years of synthetic business days history
        # Base anchor: today
        end_dt = pd.Timestamp.now().normalize()
        start_dt = end_dt - pd.Timedelta(days=365 * 6)
        dates = pd.date_range(start=start_dt, end=end_dt, freq="B")
        n = len(dates)

        np.random.seed(42)
        base_price = 50.0
        # Gentle upward trend with volatility
        daily_returns = np.random.normal(0.0005, 0.015, size=n)
        prices = base_price * np.exp(np.cumsum(daily_returns))

        self.raw_synth_df = pd.DataFrame(
            {
                "Open": prices * 0.995,
                "High": prices * 1.01,
                "Low": prices * 0.99,
                "Close": prices,
                "Volume": np.random.randint(100000, 5000000, size=n),
            },
            index=dates,
        )

        # Store into temporary test database and load standard db dataframe
        db.store_prices("TESTSTOCK", self.raw_synth_df)
        self.synth_df = db.get_prices("TESTSTOCK")

    def tearDown(self):
        self.patch_yf_live.stop()
        self.patch_chart.stop()
        db.DB_PATH = self.orig_db_path
        try:
            os.remove(self.temp_db.name)
        except OSError:
            pass

    def test_find_closest_trading_price_exact_match(self):
        """When target_date is a trading day, return exact close price."""
        mid_date = self.synth_df.index[100]
        price, matched_dt = _find_closest_trading_price(self.synth_df, mid_date)
        self.assertIsNotNone(price)
        self.assertAlmostEqual(price, float(self.synth_df.loc[mid_date, "close"]), places=4)
        self.assertEqual(matched_dt, mid_date.strftime("%Y-%m-%d"))

    def test_find_closest_trading_price_on_weekend(self):
        """When target_date is Saturday or Sunday, resolve to adjacent trading day within 2 days."""
        # Pick a Friday in index, target Saturday
        fri = [d for d in self.synth_df.index if d.weekday() == 4][10]
        sat = fri + timedelta(days=1)
        sun = fri + timedelta(days=2)

        price_sat, matched_sat = _find_closest_trading_price(self.synth_df, sat)
        self.assertIsNotNone(price_sat)
        self.assertIn(matched_sat, [fri.strftime("%Y-%m-%d"), (fri + timedelta(days=3)).strftime("%Y-%m-%d")])

        price_sun, matched_sun = _find_closest_trading_price(self.synth_df, sun)
        self.assertIsNotNone(price_sun)

    def test_find_closest_trading_price_exceeds_tolerance_returns_none(self):
        """When target_date is way before earliest available history, return (None, None)."""
        way_back = self.synth_df.index[0] - timedelta(days=365 * 10)
        price, matched_dt = _find_closest_trading_price(self.synth_df, way_back, max_tolerance_days=30)
        self.assertIsNone(price)
        self.assertIsNone(matched_dt)

    def test_all_periods_have_valid_values_no_dashes(self):
        """Verify that get_stock_data for a 5Y stock returns numeric values for all periods (no '-' placeholders)."""
        data = get_stock_data("TESTSTOCK")
        self.assertNotIn("error", data)

        pct_data = data["percentage_data"]
        net_data = data["net_change_data"]

        expected_periods = [
            "Current Price",
            "1D", "2D", "3D", "4D", "5D",
            "1W", "2W", "3W",
            "1M", "2M", "3M", "6M",
            "1Y", "2Y", "3Y", "4Y", "5Y",
            "YTD",
        ]

        found_periods_pct = [row["period"] for row in pct_data]
        found_periods_net = [row["period"] for row in net_data]

        for p in expected_periods:
            self.assertIn(p, found_periods_pct, f"Missing {p} in percentage_data")
            self.assertIn(p, found_periods_net, f"Missing {p} in net_change_data")

        for row in pct_data:
            if row["period"] == "Current Price":
                continue
            self.assertNotEqual(
                row["value"],
                "-",
                f"Period {row['period']} rendered as a dash '-' instead of a calculated value",
            )
            self.assertNotEqual(
                row["value"],
                "N/A",
                f"Period {row['period']} unexpectedly rendered as 'N/A' for full 5Y dataset",
            )
            self.assertIsInstance(
                row["value"],
                (float, int),
                f"Period {row['period']} value {row['value']} is not numeric",
            )
            # Matched date should be populated
            self.assertTrue(
                row.get("matched_date"),
                f"Period {row['period']} missing matched_date",
            )

    def test_short_history_stock_gracefully_reports_na_for_unavailable_years(self):
        """A stock with only 6 months of history should report numbers for <= 6M and 'N/A' for 1Y–5Y."""
        end_dt = pd.Timestamp.now().normalize()
        start_dt = end_dt - pd.Timedelta(days=180)
        dates = pd.date_range(start=start_dt, end=end_dt, freq="B")
        short_df = pd.DataFrame(
            {
                "Open": [100.0] * len(dates),
                "High": [105.0] * len(dates),
                "Low": [95.0] * len(dates),
                "Close": [102.0] * len(dates),
                "Volume": [500000] * len(dates),
            },
            index=dates,
        )
        db.store_prices("RECENT_IPO", short_df)

        data = get_stock_data("RECENT_IPO")
        self.assertNotIn("error", data)

        pct_map = {row["period"]: row["value"] for row in data["percentage_data"]}
        # 1M, 2M, 3M should be numeric
        self.assertIsInstance(pct_map["1M"], (float, int))
        self.assertIsInstance(pct_map["2M"], (float, int))
        self.assertIsInstance(pct_map["3M"], (float, int))

        # 1Y through 5Y should be 'N/A', NOT '-'
        for y_period in ["1Y", "2Y", "3Y", "4Y", "5Y"]:
            self.assertEqual(
                pct_map[y_period],
                "N/A",
                f"Period {y_period} should be 'N/A' for recent IPO, got {pct_map[y_period]}",
            )
