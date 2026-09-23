"""Regression tests for the /ai-summary workbook VAL sheet.

The VAL sheet ("Valuation & consensus estimates") speaks Finnhub's label
names ('P/E (TTM)', 'P/B', ...), but Finnhub doesn't cover ETFs and may be
unconfigured entirely. The yfinance fundamentals that `get_fundamentals`
already fetched must be aliased into those labels (and ETFs fall back to
category / fund family for the Sector / Industry cells) instead of
rendering N/A or a literal `None`.

Providers are stubbed so the tests are fully offline; only the price
history comes from the seeded temp DB.
"""
import os
import tempfile
import unittest
from types import SimpleNamespace

import pandas as pd

import db

# app.py starts a background options-chain warmer at import time; point it
# at a throwaway DB before the first import (same pattern as
# test_strategies_route.py).
_fd, _DB_PATH = tempfile.mkstemp(suffix=".db")
os.close(_fd)
db.DB_PATH = _DB_PATH
db.init_db()

import app as app_module  # noqa: E402

END_DATE = "2024-06-28"


def _spy_closes(n=400, seed=100):
    import numpy as np
    rng = np.random.RandomState(seed)
    rets = rng.normal(0.08 / 252, 0.10 / (252 ** 0.5), size=n)
    return 100.0 * np.exp(np.cumsum(rets))


class TestAiSummaryValSheet(unittest.TestCase):

    def setUp(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.temp_db_path = path
        db.DB_PATH = path
        db.init_db()
        dates = pd.bdate_range(end=END_DATE, periods=400)
        rows = [
            ("SPY", d.strftime("%Y-%m-%d"), float(c), float(c), float(c), float(c), 1_000_000)
            for d, c in zip(dates, _spy_closes())
        ]
        with db.get_conn() as conn:
            conn.executemany(
                "INSERT INTO daily_prices (symbol, date, open, high, low, close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        self.client = app_module.app.test_client()
        self.client.testing = True

    def tearDown(self):
        try:
            os.remove(self.temp_db_path)
        except OSError:
            pass

    # ------------------------------------------------------------------ #
    # Stubbing helpers                                                    #
    # ------------------------------------------------------------------ #

    def _stub_providers(self, fundamentals, finnhub_valuation=None,
                        finnhub_recs=None):
        positioning = {
            "ticker": "SPY",
            "configured": {"finnhub": False, "fmp": False, "sec": True},
            "valuation": finnhub_valuation,
            "recommendations": finnhub_recs,
            "insider": {"sentiment": None, "transactions": None,
                        "sec_filings": None, "chart": None},
            "institutional": {"holders": None, "sec_filings": None, "chart": None},
        }
        orig = (
            app_module.get_fundamentals,
            app_module.compute_positioning,
            app_module.get_gex_profile,
            app_module.ml,
            app_module.ai,
            app_module.providers,
        )
        app_module.get_fundamentals = lambda t: fundamentals
        app_module.compute_positioning = lambda t: positioning
        app_module.get_gex_profile = lambda t, p: None
        app_module.ml = SimpleNamespace(predict=lambda t: None)
        app_module.ai = SimpleNamespace(
            generate_comprehensive_report=lambda t, p: (None, "stubbed"))
        # The workbook markup is gated on an AI key being configured; stub
        # it so the sheets render in this keyless test environment.
        app_module.providers = SimpleNamespace(ai_providers=lambda: [{"id": "stub"}])
        return orig

    def _restore_providers(self, orig):
        (app_module.get_fundamentals,
         app_module.compute_positioning,
         app_module.get_gex_profile,
         app_module.ml,
         app_module.ai,
         app_module.providers) = orig

    # ------------------------------------------------------------------ #
    # Tests                                                               #
    # ------------------------------------------------------------------ #

    def test_etf_val_sheet_aliases_yfinance_fundamentals(self):
        """ETF-style fundamentals (Finnhub empty) must populate the VAL
        sheet via yfinance aliases instead of N/A / literal None."""
        fundamentals = {
            "Name": "SPDR S&P 500 ETF Trust",
            "Category": "Large Blend",
            "Fund Family": "State Street Investment Management",
            "Trailing P/E": 24.807701,
            "Price / Book": 1.7888495,
            "Dividend Yield": 0.98,
            "Analyst Rating": "hold",
        }
        orig = self._stub_providers(fundamentals)
        try:
            resp = self.client.get("/ai-summary?ticker=SPY")
        finally:
            self._restore_providers(orig)

        self.assertEqual(resp.status_code, 200)
        body = resp.data
        # P/E / P/B aliased from yfinance (rounded to 2dp like finnhub).
        self.assertIn(b"24.81", body)
        self.assertIn(b"1.79", body)
        # Div yield aliased and percent-formatted (analytics-page style).
        self.assertIn(b"0.98%", body)
        # ETF classification instead of empty Sector / Industry cells.
        self.assertIn(b"Large Blend", body)
        self.assertIn(b"State Street Investment Management", body)
        # yfinance single rating backs the Consensus Reco cell.
        self.assertIn(b"Hold", body)
        # The old bug rendered a literal `None` in the Sector cell.
        self.assertNotIn(b">None<", body)

    def test_finnhub_values_take_precedence_over_yfinance_aliases(self):
        """When Finnhub answers, its values keep the VAL sheet slot; the
        yfinance alias must not overwrite them."""
        fundamentals = {
            "Name": "Apple Inc.",
            "Sector": "Technology",
            "Industry": "Consumer Electronics",
            "Trailing P/E": 24.807701,
            "Price / Book": 1.7888495,
            "Dividend Yield": 0.98,
        }
        finnhub_valuation = [
            {"label": "P/E (TTM)", "value": 31.4},
            {"label": "P/B", "value": 45.2},
        ]
        orig = self._stub_providers(fundamentals, finnhub_valuation=finnhub_valuation)
        try:
            resp = self.client.get("/ai-summary?ticker=SPY")
        finally:
            self._restore_providers(orig)

        self.assertEqual(resp.status_code, 200)
        body = resp.data
        self.assertIn(b"31.4", body)
        self.assertIn(b"45.2", body)
        self.assertNotIn(b"24.81", body)
        self.assertNotIn(b"1.79", body)
        # Equities keep their real sector / industry.
        self.assertIn(b"Technology", body)
        self.assertIn(b"Consumer Electronics", body)
        self.assertNotIn(b">None<", body)


if __name__ == "__main__":
    unittest.main()
