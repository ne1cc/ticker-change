"""/brief/<ticker> section evidence links: every section with underlying
depth links into its evidence page with the brief's own ticker; sections
without a natural evidence page get none; warming renders none.
"""
import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

import db

_fd, _DB_PATH = tempfile.mkstemp(suffix=".db")
os.close(_fd)
db.DB_PATH = _DB_PATH
db.init_db()

import app as app_module  # noqa: E402


def _seed(ticker="AAA"):
    idx = pd.bdate_range(end="2026-10-12", periods=30)
    db.store_prices(ticker, pd.DataFrame(
        {"Open": [100.0] * 30, "High": [100.0] * 30, "Low": [100.0] * 30,
         "Close": [101.0] * 30, "Volume": [1_000_000] * 30}, index=idx))
    db.cache_set("yfinance", f"fundamentals:{ticker}",
                 {"Sector": "Technology", "Market Cap": 1e9,
                  "Forward P/E": 10.0, "Price / Sales": 2.0})
    db.cache_set("yfinance", f"earnings:{ticker}", [
        {"date": "2026-09-08", "eps_estimate": 1.0, "eps_actual": 1.2,
         "surprise_pct": 20.0, "is_upcoming": False}])


class _IsolatedDB(unittest.TestCase):
    def setUp(self):
        self._orig_db = db.DB_PATH
        db.DB_PATH = _DB_PATH
        db.init_db()
        with db.get_conn() as conn:
            conn.execute("DELETE FROM api_cache WHERE provider LIKE 'brief:%'")
        self.client = app_module.app.test_client()

    def tearDown(self):
        db.DB_PATH = self._orig_db

    def _sections_html(self, ticker="AAA"):
        html = self.client.get(f"/brief/{ticker}").get_data(as_text=True)
        return [chunk.split("</section>")[0]
                for chunk in html.split("<section")[1:]]


class TestBriefEvidenceLinks(_IsolatedDB):
    def test_every_section_links_its_evidence_page_with_ticker(self):
        _seed("AAA")
        self.assertIsNotNone(app_module.publish_brief("AAA"))
        html = self.client.get("/brief/AAA").get_data(as_text=True)

        # price -> /stock, context -> /stock, earnings -> the events-bearing
        # page (/analytics hosts the earnings event study).
        self.assertEqual(html.count('href="/stock?ticker=AAA"'), 2)
        self.assertEqual(html.count('href="/analytics?ticker=AAA"'), 1)

    def test_links_carry_the_page_ticker_not_the_route_case(self):
        _seed("AAA")
        self.assertIsNotNone(app_module.publish_brief("AAA"))
        html = self.client.get("/brief/aaa").get_data(as_text=True)
        self.assertIn('href="/stock?ticker=AAA"', html)
        self.assertIn('href="/analytics?ticker=AAA"', html)
        self.assertNotIn("ticker=aaa", html)

    def test_evidence_links_land_in_the_matching_sections(self):
        _seed("AAA")
        self.assertIsNotNone(app_module.publish_brief("AAA"))
        price, earnings, context = self._sections_html("AAA")
        self.assertIn('href="/stock?ticker=AAA"', price)
        self.assertIn('href="/analytics?ticker=AAA"', earnings)
        self.assertIn('href="/stock?ticker=AAA"', context)
        self.assertNotIn("/analytics", price)
        self.assertNotIn("/analytics", context)

    def test_deltas_brief_links_only_its_present_sections(self):
        _seed("AAA")
        self.assertIsNotNone(app_module.publish_brief("AAA", depth="deltas"))
        price, earnings = self._sections_html("AAA")
        self.assertIn('href="/stock?ticker=AAA"', price)
        self.assertIn('href="/analytics?ticker=AAA"', earnings)
        html = self.client.get("/brief/AAA").get_data(as_text=True)
        self.assertEqual(html.count('href="/stock?ticker=AAA"'), 1)

    def test_warming_page_renders_no_evidence_links(self):
        with patch.object(app_module.brief, "load_brief_context",
                          side_effect=RuntimeError("boom")), \
             patch.object(app_module, "kick_brief_build"):
            html = self.client.get("/brief/ZZZ").get_data(as_text=True)
        self.assertNotIn('href="/stock?ticker=', html)
        self.assertNotIn('href="/analytics?ticker=', html)


if __name__ == "__main__":
    unittest.main()
