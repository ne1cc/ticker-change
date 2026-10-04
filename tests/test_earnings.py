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
