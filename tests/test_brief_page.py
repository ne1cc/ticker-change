"""/brief/<ticker>: published content renders; cold builds on demand and
falls back to warming on failure; JSON twin parity.

The render path is proven offline by sentinels on every network entry point:
the page serves the published artifact (or an on-demand build over the
cache-only context) and never fetches.
"""
import json
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


def _store(sym, closes, volumes=None, end="2026-10-12"):
    n = len(closes)
    idx = pd.bdate_range(end=end, periods=n)
    if volumes is None:
        volumes = [1_000_000] * n
    db.store_prices(sym, pd.DataFrame(
        {"Open": closes, "High": closes, "Low": closes, "Close": closes,
         "Volume": volumes}, index=idx))


def _seed(ticker="AAA"):
    closes = [100.0, 101.0] * 14 + [100.0, 105.0]
    _store(ticker, closes)
    db.cache_set("yfinance", f"fundamentals:{ticker}",
                 {"Sector": "Technology", "Market Cap": 1e9,
                  "Forward P/E": 10.0, "Price / Sales": 2.0})
    db.cache_set("yfinance", f"earnings:{ticker}", [
        {"date": "2026-09-08", "eps_estimate": 1.0, "eps_actual": 1.2,
         "surprise_pct": 20.0, "is_upcoming": False}])


class _IsolatedDB:
    def setUp(self):
        self._orig_db = db.DB_PATH
        db.DB_PATH = _DB_PATH
        db.init_db()
        with db.get_conn() as conn:
            conn.execute("DELETE FROM api_cache WHERE provider LIKE 'brief:%'")

    def tearDown(self):
        db.DB_PATH = self._orig_db

    def _client(self):
        return app_module.app.test_client()


class TestBriefPage(_IsolatedDB, unittest.TestCase):
    def test_published_brief_renders_200_with_content(self):
        _seed("AAA")
        published = app_module.publish_brief("AAA")
        self.assertIsNotNone(published)
        resp = self._client().get("/brief/AAA")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertIn("Daily brief", html)
        self.assertIn("AAA", html)
        self.assertIn("Session return", html)
        self.assertIn("+5.00%", html)             # published value, rendered
        self.assertIn("as of", html)              # receipts visible on numbers
        self.assertIn("prices", html)

    def test_cold_request_builds_publishes_and_serves(self):
        _seed("ZZZ")
        resp = self._client().get("/brief/ZZZ")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertIn("+5.00%", html)
        self.assertNotIn("warming", html.lower())
        current = app_module.current_brief("ZZZ")
        self.assertIsNotNone(current)
        self.assertEqual(current["meta"]["ticker"], "ZZZ")
        self.assertEqual(current["meta"]["validation_status"], "ok")

    def test_second_request_serves_published_without_rebuild(self):
        _seed("ZZZ")
        client = self._client()
        with patch.object(app_module.brief, "load_brief_context",
                          wraps=app_module.brief.load_brief_context) as load:
            first = client.get("/brief/ZZZ")
            second = client.get("/brief/ZZZ")
        self.assertEqual(load.call_count, 1)
        for resp in (first, second):
            self.assertEqual(resp.status_code, 200)
            self.assertIn("+5.00%", resp.get_data(as_text=True))

    def test_on_demand_build_touches_no_network(self):
        with patch("yfinance.Ticker", side_effect=AssertionError("network!")), \
             patch("requests.get", side_effect=AssertionError("network!"),
                   create=True), \
             patch.object(app_module, "get_or_fetch_prices",
                          side_effect=AssertionError("network!")), \
             patch.object(app_module.earnings, "get_earnings_history",
                          side_effect=AssertionError("network!")), \
             patch.object(app_module, "get_fundamentals",
                          side_effect=AssertionError("network!")), \
             patch.object(app_module.providers, "get_short_interest",
                          side_effect=AssertionError("network!")):
            resp = self._client().get("/brief/NOPE")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertIn("No cached data (never fetched)", html)
        self.assertIn("unknown — payload missing", html)
        self.assertNotIn("warming", html.lower())

    def test_build_failure_falls_back_to_warming_and_kick(self):
        with patch.object(app_module.brief, "load_brief_context",
                          side_effect=RuntimeError("boom")), \
             patch.object(app_module, "kick_brief_build") as kick:
            resp = self._client().get("/brief/ZZZ")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("warming", resp.get_data(as_text=True).lower())
        kick.assert_called_once_with("ZZZ")

    def test_in_flight_build_serves_warming_without_second_build(self):
        _seed("ZZZ")
        app_module._BRIEF_WARM_INFLIGHT.add("ZZZ")
        self.addCleanup(app_module._BRIEF_WARM_INFLIGHT.discard, "ZZZ")
        with patch.object(app_module, "publish_brief") as pub, \
             patch.object(app_module, "kick_brief_build") as kick:
            resp = self._client().get("/brief/ZZZ")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("warming", resp.get_data(as_text=True).lower())
        pub.assert_not_called()
        kick.assert_called_once_with("ZZZ")

    def test_on_demand_build_leaves_universe_publications_untouched(self):
        _seed("AAA")
        self.assertIsNotNone(app_module.publish_brief("AAA"))
        pointer_before = app_module.current_brief("AAA")["meta"]["snapshot_id"]
        with db.get_conn() as conn:
            rows_before = conn.execute(
                "SELECT key, payload FROM api_cache WHERE provider='brief:AAA' "
                "ORDER BY key").fetchall()
        _seed("BBB")
        resp = self._client().get("/brief/BBB")
        self.assertIn("+5.00%", resp.get_data(as_text=True))
        with db.get_conn() as conn:
            rows_after = conn.execute(
                "SELECT key, payload FROM api_cache WHERE provider='brief:AAA' "
                "ORDER BY key").fetchall()
        self.assertEqual([tuple(r) for r in rows_before],
                         [tuple(r) for r in rows_after])
        self.assertEqual(app_module.current_brief("AAA")["meta"]["snapshot_id"],
                         pointer_before)
        self.assertIsNotNone(app_module.current_brief("BBB"))

    def test_unknown_ticker_never_404s(self):
        with patch.object(app_module, "kick_brief_build"):
            for path in ("/brief/ZZZZ", "/brief/nope"):
                resp = self._client().get(path)
                self.assertEqual(resp.status_code, 200, path)

    def test_ticker_route_is_case_insensitive(self):
        _seed("AAA")
        app_module.publish_brief("AAA")
        resp = self._client().get("/brief/aaa")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("+5.00%", resp.get_data(as_text=True))

    def test_render_fetches_nothing_live(self):
        _seed("AAA")
        app_module.publish_brief("AAA")
        with patch("yfinance.Ticker", side_effect=AssertionError("network!")), \
             patch("requests.get", side_effect=AssertionError("network!"),
                   create=True), \
             patch.object(app_module, "get_or_fetch_prices",
                          side_effect=AssertionError("network!")), \
             patch.object(app_module.earnings, "get_earnings_history",
                          side_effect=AssertionError("network!")), \
             patch.object(app_module, "get_fundamentals",
                          side_effect=AssertionError("network!")), \
             patch.object(app_module.providers, "get_short_interest",
                          side_effect=AssertionError("network!")):
            resp = self._client().get("/brief/AAA")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("+5.00%", resp.get_data(as_text=True))

    def test_empty_context_renders_declared_missing_states(self):
        app_module.publish_brief("NOPE")
        resp = self._client().get("/brief/NOPE")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertIn("No cached data (never fetched)", html)
        self.assertIn("unknown — payload missing", html)
        self.assertNotIn("warming", html.lower())


class TestBriefTwin(_IsolatedDB, unittest.TestCase):
    def test_twin_returns_identical_published_payload(self):
        _seed("AAA")
        published = app_module.publish_brief("AAA")
        resp = self._client().get("/api/brief/AAA")
        self.assertEqual(resp.status_code, 200)
        twin = json.loads(resp.get_data(as_text=True))
        self.assertEqual(twin, published)
        self.assertEqual(twin["meta"]["snapshot_id"],
                         published["meta"]["snapshot_id"])
        self.assertIn("sections", twin)

    def test_twin_cold_returns_warming_stub(self):
        with patch.object(app_module, "kick_brief_build") as kick:
            resp = self._client().get("/api/brief/ZZZ")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(json.loads(resp.get_data(as_text=True)),
                         {"ticker": "ZZZ", "status": "warming"})
        kick.assert_called_once_with("ZZZ")

    def test_page_and_twin_serve_the_same_publication(self):
        _seed("AAA")
        app_module.publish_brief("AAA")
        page = self._client().get("/brief/AAA")
        twin = self._client().get("/api/brief/AAA")
        self.assertEqual(page.status_code, 200)
        self.assertEqual(twin.status_code, 200)
        html = page.get_data(as_text=True)
        payload = json.loads(twin.get_data(as_text=True))
        price = payload["sections"]["price"]
        self.assertEqual(price["state"], "ok")
        self.assertIn(str(price["session_date"]), html)
        self.assertIn(payload["meta"]["receipts"]["prices"][:10], html)


if __name__ == "__main__":
    unittest.main()
