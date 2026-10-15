"""Front-door dashboard at /: published briefs only, ranked, offline.

Covers the five acceptance criteria: cold 200 + warming skeleton (with the
universe warmer kicked), render assembled exclusively from published
snapshots (sentinels on every network entry point), notable deltas ranked
with per-ticker brief links, search navigating to the brief route, and
universe tickers without a publication rendering declared-pending rows.
"""
import os
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import pandas as pd

import db
import snapshot_store

_fd, _DB_PATH = tempfile.mkstemp(suffix=".db")
os.close(_fd)
db.DB_PATH = _DB_PATH
db.init_db()

import app as app_module  # noqa: E402


def _store(sym, closes, end="2026-10-12"):
    idx = pd.bdate_range(end=end, periods=len(closes))
    db.store_prices(sym, pd.DataFrame(
        {"Open": closes, "High": closes, "Low": closes, "Close": closes,
         "Volume": [1_000_000] * len(closes)}, index=idx))


def _seed(ticker, last_close=105.0):
    closes = [100.0, 101.0] * 14 + [100.0, last_close]
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
            for table in ("api_cache", "daily_prices", "tickers"):
                conn.execute(f"DELETE FROM {table}")

    def tearDown(self):
        db.DB_PATH = self._orig_db

    def _client(self):
        return app_module.app.test_client()


def _poison_network(stack):
    """Sentinel-patch every network entry point the render must never touch,
    including the cache-miss fallbacks (constituents fetch, yfinance)."""
    stack.enter_context(patch("yfinance.Ticker",
                              side_effect=AssertionError("network!")))
    stack.enter_context(patch("requests.get",
                              side_effect=AssertionError("network!"),
                              create=True))
    stack.enter_context(patch.object(app_module, "get_or_fetch_prices",
                                     side_effect=AssertionError("network!")))
    stack.enter_context(patch.object(app_module.earnings,
                                     "get_earnings_history",
                                     side_effect=AssertionError("network!")))
    stack.enter_context(patch.object(app_module, "get_fundamentals",
                                     side_effect=AssertionError("network!")))
    stack.enter_context(patch.object(app_module.providers,
                                     "get_short_interest",
                                     side_effect=AssertionError("network!")))
    stack.enter_context(patch.object(app_module.heatmap, "get_constituents",
                                     side_effect=AssertionError("network!")))
    stack.enter_context(patch("sp500.fetch_sp500_constituents",
                              side_effect=AssertionError("network!")))


class TestColdDashboard(_IsolatedDB, unittest.TestCase):
    def test_cold_renders_skeleton_200_and_kicks_warmer(self):
        with patch.object(app_module, "_warm_brief_universe") as warm:
            resp = self._client().get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("warming", resp.get_data(as_text=True).lower())
        warm.assert_called_once_with()

    def test_warm_pass_in_progress_shows_progress_without_briefs(self):
        db.cache_set("brief", "warm_progress",
                     {"done": 2, "total": 5, "published": 2, "skipped": 0,
                      "failed": 0, "updated": "2026-10-14T05:00:00"})
        with patch.object(app_module, "_warm_brief_universe"), \
             patch.object(app_module.radar, "universe_symbols",
                          return_value=[]):
            resp = self._client().get("/")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertIn("2/5", html)
        self.assertIn("No briefs published yet", html)
        self.assertNotIn("Check back shortly", html)


class TestPublicationsOnly(_IsolatedDB, unittest.TestCase):
    def test_render_touches_no_network(self):
        for t in ("AAA", "BBB"):
            _seed(t)
            app_module.publish_brief(t)
        with patch.object(app_module, "_warm_brief_universe") as warm, \
                ExitStack() as stack:
            _poison_network(stack)
            resp = self._client().get("/")
        html = resp.get_data(as_text=True)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(warm.call_count, 0)
        self.assertIn("+5.00%", html)
        self.assertIn("/brief/AAA", html)
        self.assertIn("/brief/BBB", html)

    def test_empty_universe_and_no_briefs_still_render_declared_states(self):
        with patch.object(app_module, "_warm_brief_universe"), \
             patch.object(app_module.radar, "universe_symbols",
                          return_value=[]):
            resp = self._client().get("/")
        html = resp.get_data(as_text=True)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("Warming", html)
        self.assertNotIn("Notable moves", html)


class TestRanking(_IsolatedDB, unittest.TestCase):
    def test_ranked_by_absolute_session_move_with_brief_links(self):
        for t, close in (("AAA", 105.0), ("BBB", 109.0), ("CCC", 93.0)):
            _seed(t, last_close=close)
            app_module.publish_brief(t)
        resp = self._client().get("/")
        html = resp.get_data(as_text=True)
        self.assertEqual(resp.status_code, 200)
        self.assertLess(html.find("/brief/BBB"), html.find("/brief/CCC"))
        self.assertLess(html.find("/brief/CCC"), html.find("/brief/AAA"))
        for href in ("/brief/BBB", "/brief/CCC", "/brief/AAA"):
            self.assertIn(f'href="{href}"', html)
        self.assertIn("+9.00%", html)
        self.assertIn("-7.00%", html)
        self.assertIn("+5.00%", html)

    def test_equal_absolute_moves_tie_break_by_ticker_ascending(self):
        for t, close in (("QQQ", 96.0), ("PPP", 104.0)):
            _seed(t, last_close=close)
            app_module.publish_brief(t)
        html = self._client().get("/").get_data(as_text=True)
        self.assertLess(html.find("/brief/PPP"), html.find("/brief/QQQ"))

    def test_published_without_session_data_rank_after_movers(self):
        app_module.publish_brief("NOPE")
        _seed("AAA")
        app_module.publish_brief("AAA")
        html = self._client().get("/").get_data(as_text=True)
        self.assertLess(html.find("/brief/AAA"), html.find("/brief/NOPE"))
        self.assertIn("absent under 2 cached bars", html)


class TestPending(_IsolatedDB, unittest.TestCase):
    def test_universe_tickers_without_publication_render_pending(self):
        db.cache_set("sp500", "constituents", [{"symbol": "BBB"}])
        _seed("AAA")
        app_module.publish_brief("AAA")
        with patch.object(app_module, "_warm_brief_universe"), \
             patch.object(app_module.radar, "universe_symbols",
                          return_value=["AAA", "ZZZ"]):
            data = app_module.compute_dashboard()
            resp = self._client().get("/")
        self.assertEqual([p["ticker"] for p in data["pending"]], ["BBB", "ZZZ"])
        html = resp.get_data(as_text=True)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("Not yet published", html)
        self.assertIn('title="not yet published"', html)
        self.assertIn('href="/brief/ZZZ"', html)
        self.assertIn('href="/brief/BBB"', html)

    def test_pending_rows_render_during_warming(self):
        db.cache_set("brief", "warm_progress",
                     {"done": 0, "total": 2, "published": 0, "skipped": 0,
                      "failed": 0, "updated": "2026-10-14T05:00:00"})
        with patch.object(app_module, "_warm_brief_universe"), \
             patch.object(app_module.radar, "universe_symbols",
                          return_value=["ZZZ"]):
            resp = self._client().get("/")
        html = resp.get_data(as_text=True)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("warming", html.lower())
        self.assertIn('href="/brief/ZZZ"', html)


class TestSearch(_IsolatedDB, unittest.TestCase):
    def test_search_form_targets_the_brief_route(self):
        _seed("AAA")
        app_module.publish_brief("AAA")
        html = self._client().get("/").get_data(as_text=True)
        self.assertIn('id="dashboard-search-form"', html)
        self.assertIn('id="dashboard-search-input"', html)
        self.assertIn("'/brief/'", html)
        self.assertIn("toUpperCase()", html)

    def test_entered_ticker_resolves_case_insensitively_via_brief_route(self):
        _seed("AAA")
        app_module.publish_brief("AAA")
        resp = self._client().get("/brief/aaa")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("+5.00%", resp.get_data(as_text=True))


class TestCurrentAll(_IsolatedDB, unittest.TestCase):
    def test_enumerates_live_publications_by_suffix(self):
        for t in ("AAA", "BBB"):
            _seed(t)
            app_module.publish_brief(t)
        out = snapshot_store.current_all("brief:")
        self.assertEqual(set(out), {"AAA", "BBB"})
        self.assertEqual(out["AAA"]["meta"]["ticker"], "AAA")
        self.assertIsNotNone(out["BBB"].get("sections"))

    def test_expired_rows_drop_out_like_current(self):
        _seed("AAA")
        app_module.publish_brief("AAA")
        with db.get_conn() as conn:
            conn.execute(
                "UPDATE api_cache SET fetched_at = datetime('now', '-8 days') "
                "WHERE provider LIKE 'brief:%'")
        self.assertEqual(snapshot_store.current_all("brief:"), {})
        self.assertIsNone(app_module.current_brief("AAA"))

    def test_foreign_providers_and_status_keys_excluded(self):
        db.cache_set("brief", "warm_progress", {"done": 1, "total": 1})
        db.cache_set("radar", "current", {"snapshot_id": "X"})
        db.cache_set("other:AAA", "current", {"snapshot_id": "Y"})
        db.cache_set("other:AAA", "snapshot:Y", {"ticker": "AAA"})
        self.assertEqual(snapshot_store.current_all("brief:"), {})

    def test_pointer_without_payload_excluded(self):
        _seed("AAA")
        app_module.publish_brief("AAA")
        with db.get_conn() as conn:
            conn.execute("DELETE FROM api_cache WHERE provider = 'brief:AAA' "
                         "AND key LIKE 'snapshot:%'")
        self.assertEqual(snapshot_store.current_all("brief:"), {})


if __name__ == "__main__":
    unittest.main()
