"""Earnings history normalization and caching."""
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import pandas as pd

import db

_fd, _DB_PATH = tempfile.mkstemp(suffix=".db")
os.close(_fd)
db.DB_PATH = _DB_PATH
db.init_db()

import earnings  # noqa: E402


def _frame():
    idx = pd.DatetimeIndex(["2025-01-30", "2025-04-24", "2026-07-30"])
    return pd.DataFrame(
        {"EPS Estimate": [1.5, 1.6, 1.7], "Reported EPS": [1.7, 1.4, None],
         "Surprise(%)": [13.3, -12.5, None]}, index=idx)


class TestEarningsHistory(unittest.TestCase):
    def test_normalization_and_upcoming_flag(self):
        with patch.object(earnings.yf, "Ticker", MagicMock()) as mock:
            mock.return_value.earnings_dates = _frame()
            rows = earnings.fetch_earnings_history("aapl")
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0], {"date": "2025-01-30", "eps_estimate": 1.5,
                                   "eps_actual": 1.7, "surprise_pct": 13.3,
                                   "is_upcoming": False})
        self.assertTrue(rows[2]["is_upcoming"])
        self.assertIsNone(rows[2]["eps_actual"])
        self.assertEqual([r["date"] for r in rows], sorted(r["date"] for r in rows))

    def test_malformed_rows_skipped(self):
        df = pd.concat([_frame(), pd.DataFrame(
            {"EPS Estimate": [None, None], "Reported EPS": [None, 1.0],
             "Surprise(%)": [None, None]},
            index=pd.DatetimeIndex(["2026-10-15", pd.NaT]))])
        with patch.object(earnings.yf, "Ticker", MagicMock()) as mock:
            mock.return_value.earnings_dates = df
            rows = earnings.fetch_earnings_history("AAPL")
        self.assertEqual([r["date"] for r in rows],
                         ["2025-01-30", "2025-04-24", "2026-07-30", "2026-10-15"])

    def test_none_on_empty_or_error(self):
        with patch.object(earnings.yf, "Ticker", MagicMock()) as mock:
            mock.return_value.earnings_dates = None
            self.assertIsNone(earnings.fetch_earnings_history("AAPL"))
        with patch.object(earnings.yf, "Ticker", side_effect=RuntimeError("x")):
            self.assertIsNone(earnings.fetch_earnings_history("AAPL"))

    def test_get_earnings_history_caches(self):
        with patch.object(earnings.yf, "Ticker", MagicMock()) as mock:
            mock.return_value.earnings_dates = _frame()
            first = earnings.get_earnings_history("NVDA")
            second = earnings.get_earnings_history("NVDA")
        self.assertEqual(first, second)
        self.assertEqual(mock.call_count, 1)


import numpy as np


def _price_frame(days=400, base=100.0, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=days)
    close = base * np.exp(np.cumsum(rng.normal(0.0005, 0.012, days)))
    return pd.DataFrame({"close": close}, index=idx)


class TestPostDrift(unittest.TestCase):
    def test_drift_over_known_window(self):
        df = _price_frame()
        start = df.index[100]
        val = earnings.post_drift(df, start.date().isoformat(), 10)
        self.assertIsNotNone(val)

    def test_drift_none_beyond_history(self):
        df = _price_frame(days=40)
        self.assertIsNone(earnings.post_drift(df, "2030-01-01", 5))


class TestComputeEarningsEvents(unittest.TestCase):
    def test_car_attached_and_upcoming_skipped(self):
        df = _price_frame()
        bench = _price_frame(days=400, base=500.0, seed=3)
        events = [
            {"date": "2024-06-03", "surprise_pct": 5.0, "is_upcoming": False},
            {"date": "2030-01-15", "surprise_pct": None, "is_upcoming": True},
        ]
        rows = earnings.compute_earnings_events(df, bench, events, ticker="TEST")
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertIn("car", row)
        self.assertIn("car_p_value", row)
        self.assertIn("drift_5d", row)
        self.assertIn("drift_20d", row)

    def test_summary_means(self):
        events = [
            {"surprise_pct": 5.0, "car": 0.10, "is_significant_95": True},
            {"surprise_pct": -4.0, "car": -0.05, "is_significant_95": False},
        ]
        s = earnings.summarize_earnings_drift(events)
        self.assertEqual(s["n_events"], 2)
        self.assertEqual(s["n_significant"], 1)
        self.assertAlmostEqual(s["avg_car_beat"], 0.10, places=6)
        self.assertAlmostEqual(s["avg_car_miss"], -0.05, places=6)

    def test_summary_empty_is_safe(self):
        s = earnings.summarize_earnings_drift([])
        self.assertEqual(s["n_events"], 0)
        self.assertIsNone(s["avg_car_beat"])


class TestImpliedMove(unittest.TestCase):
    def test_scales_straddle_to_earnings_dte(self):
        oa = {"expected_move_straddle": 10.0, "days_to_expiration": 40}
        out = earnings.implied_event_move(
            oa, (pd.Timestamp.today() + pd.Timedelta(days=10)).date().isoformat())
        self.assertAlmostEqual(out["implied_move"], 5.0, places=6)
        self.assertEqual(out["days_to_earnings"], 10)
        self.assertEqual(out["basis_days_to_exp"], 40)

    def test_none_when_missing_inputs(self):
        self.assertIsNone(earnings.implied_event_move(None, "2026-10-15"))
        self.assertIsNone(earnings.implied_event_move(
            {"expected_move_straddle": None, "days_to_expiration": 30}, "2026-10-15"))
        past = (pd.Timestamp.today() - pd.Timedelta(days=3)).date().isoformat()
        self.assertIsNone(earnings.implied_event_move(
            {"expected_move_straddle": 5, "days_to_expiration": 30}, past))


class TestEventVolFlag(unittest.TestCase):
    def test_rich_cheap_fair(self):
        self.assertEqual(earnings.event_vol_flag(2.0, 1.0), "rich")
        self.assertEqual(earnings.event_vol_flag(0.5, 1.0), "cheap")
        self.assertEqual(earnings.event_vol_flag(1.0, 1.0), "fair")
        self.assertIsNone(earnings.event_vol_flag(None, 1.0))
        self.assertIsNone(earnings.event_vol_flag(1.0, None))


class TestRealizedMoves(unittest.TestCase):
    def test_median_absolute_print_day_move(self):
        idx = pd.bdate_range("2024-01-01", periods=10)
        close = pd.Series([100, 100, 104, 104, 104, 104, 99, 99, 99, 99],
                          index=idx, name="close")
        df = pd.DataFrame({"close": close})
        events = [{"date": idx[1].date().isoformat()},   # +4.0%
                  {"date": idx[5].date().isoformat()}]   # -4.8%
        med = earnings.realized_earnings_moves(df, events)
        self.assertIsNotNone(med)
        self.assertGreater(med, 0.03)
        self.assertLess(med, 0.05)
