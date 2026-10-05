"""Insider fetch cache: one yfinance call per ticker per TTL, shared by
the summary card and the chart."""
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

import app as app_module  # noqa: E402


def _raw_insider_df():
    idx = pd.DatetimeIndex(["2025-08-15", "2025-09-10"])
    return pd.DataFrame({
        "insider": ["Jane Doe", "John Roe"],
        "position": ["CEO", "CFO"],
        "shares": [1000, 2000],
        "text": ["Buy", "Sale"],
        "value": [150000.0, 300000.0],
    }, index=idx)


def _price_df():
    idx = pd.bdate_range("2025-07-01", "2025-10-15")
    return pd.DataFrame({"close": 100 + (idx - idx[0]).days * 0.1}, index=idx)


class TestInsiderCache(unittest.TestCase):
    def setUp(self):
        with db.get_conn() as conn:
            conn.execute("DELETE FROM api_cache WHERE provider = 'yfinance' "
                         "AND key LIKE 'insider:%'")

    def _patch_ticker(self, raw=None, count=None):
        ticker = MagicMock()
        ticker.insider_transactions = raw if raw is not None else _raw_insider_df()
        mock = patch.object(app_module, "_get_yf_ticker", return_value=ticker)
        if count is not None:
            self._counts = count
        return mock

    def test_summary_second_call_served_from_cache(self):
        with patch.object(app_module, "_get_yf_ticker") as mock:
            mock.return_value.insider_transactions = _raw_insider_df()
            first = app_module.get_insider_summary("INS1")
            second = app_module.get_insider_summary("INS1")
        self.assertEqual(mock.call_count, 1)
        self.assertEqual(first, second)
        self.assertEqual(first["n_buys"], 1)
        self.assertEqual(first["n_sells"], 1)

    def test_chart_and_summary_share_cached_rows(self):
        price = _price_df()
        with patch.object(app_module, "_get_yf_ticker") as mock:
            mock.return_value.insider_transactions = _raw_insider_df()
            summary = app_module.get_insider_summary("INS2")
            chart = app_module.get_insider_chart("INS2", price)
        self.assertEqual(mock.call_count, 1)  # both consumers, one fetch
        self.assertIsNotNone(summary)
        self.assertIsNotNone(chart)

    def test_none_result_not_cached(self):
        with patch.object(app_module, "_get_yf_ticker") as mock:
            mock.return_value.insider_transactions = None
            self.assertIsNone(app_module.get_insider_summary("INS3"))
            self.assertIsNone(db.cache_get("yfinance", "insider:INS3", 24))
            self.assertIsNone(app_module.get_insider_summary("INS3"))
        self.assertEqual(
            db.cache_get("yfinance", "insider:INS3", 24), None)


if __name__ == "__main__":
    unittest.main()
