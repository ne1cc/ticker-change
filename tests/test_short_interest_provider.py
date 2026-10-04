"""Short-interest provider: normalization, fallback, caching."""
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import db

_fd, _DB_PATH = tempfile.mkstemp(suffix=".db")
os.close(_fd)
db.DB_PATH = _DB_PATH
db.init_db()

import providers  # noqa: E402


class TestFetchShortInterest(unittest.TestCase):
    def _ticker(self, info):
        t = MagicMock()
        t.info = info
        return t

    def test_normalizes_fields_and_mom_change(self):
        info = {"sharesShort": 12_000_000, "shortPercentOfFloat": 0.081,
                "shortRatio": 4.5, "sharesShortPriorMonth": 10_000_000,
                "dateShortInterest": 1_700_000_000}
        with patch.object(providers.yf, "Ticker", return_value=self._ticker(info)):
            row = providers.fetch_short_interest("aapl")
        self.assertEqual(row["shares_short"], 12_000_000)
        self.assertEqual(row["short_pct_float"], 0.081)
        self.assertEqual(row["days_to_cover"], 4.5)
        self.assertAlmostEqual(row["si_mom_change"], 0.2, places=6)
        self.assertEqual(row["source"], "reported")

    def test_returns_none_when_no_short_fields(self):
        with patch.object(providers.yf, "Ticker",
                          return_value=self._ticker({"regularMarketPrice": 1.0})):
            self.assertIsNone(providers.fetch_short_interest("AAPL"))

    def test_returns_none_on_exception(self):
        with patch.object(providers.yf, "Ticker", side_effect=RuntimeError("boom")):
            self.assertIsNone(providers.fetch_short_interest("AAPL"))

    def test_missing_prior_month_gives_none_mom(self):
        info = {"sharesShort": 100, "shortPercentOfFloat": 0.01, "shortRatio": 2.0}
        with patch.object(providers.yf, "Ticker", return_value=self._ticker(info)):
            row = providers.fetch_short_interest("AAPL")
        self.assertIsNone(row["si_mom_change"])

    def test_get_short_interest_caches(self):
        info = {"sharesShort": 5, "shortPercentOfFloat": 0.01, "shortRatio": 1.0}
        with patch.object(providers.yf, "Ticker", return_value=self._ticker(info)) as mock:
            first = providers.get_short_interest("MSFT")
            second = providers.get_short_interest("MSFT")
        self.assertEqual(first, second)
        self.assertEqual(mock.call_count, 1)  # second read served from api_cache
