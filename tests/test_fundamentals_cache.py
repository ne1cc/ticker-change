"""get_fundamentals must be served from api_cache on the second call."""
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import db

_fd, _DB_PATH = tempfile.mkstemp(suffix=".db")
os.close(_fd)
db.DB_PATH = _DB_PATH
db.init_db()

import app as app_module  # noqa: E402


class TestFundamentalsCache(unittest.TestCase):
    def _ticker(self, info):
        t = MagicMock()
        t.info = info
        t.calendar = None
        return t

    def test_second_call_served_from_cache(self):
        info = {"longName": "Test Co", "marketCap": 1000, "sector": "Tech"}
        with patch.object(app_module.yf, "Ticker", return_value=self._ticker(info)) as mock:
            first = app_module.get_fundamentals("TESTCO")
            second = app_module.get_fundamentals("TESTCO")
        self.assertEqual(first["Name"], "Test Co")
        self.assertEqual(first, second)
        self.assertEqual(mock.call_count, 1)

    def test_none_result_is_not_cached(self):
        with patch.object(app_module.yf, "Ticker", side_effect=RuntimeError("boom")):
            self.assertIsNone(app_module.get_fundamentals("FAILCO"))
        self.assertIsNone(db.cache_get("yfinance", "fundamentals:FAILCO", 24))
