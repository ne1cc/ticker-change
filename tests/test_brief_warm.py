"""Tiered nightly brief universe warmer: watchlist full-depth, SPX deltas.

Covers the five acceptance criteria on a small synthetic universe: one pass
publishes every ticker (distinct ids, retention respected), tiering of the
two depth levels, all-sources-fail still publishes declared missing-states,
progress observable via the brief:warm_progress status key, and the pass
provably stops when the universe is complete (idempotent re-warm).
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


def _seed(ticker):
    closes = [100.0, 101.0] * 14 + [100.0, 105.0]
    idx = pd.bdate_range(end="2026-10-12", periods=len(closes))
    db.store_prices(ticker, pd.DataFrame(
        {"Open": closes, "High": closes, "Low": closes, "Close": closes,
         "Volume": [1_000_000] * len(closes)}, index=idx))
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
            for table in ("api_cache", "daily_prices", "tickers"):
                conn.execute(f"DELETE FROM {table}")

    def tearDown(self):
        db.DB_PATH = self._orig_db

    def _clear_warmer_lock(self):
        with db.get_conn() as conn:
            conn.execute("DELETE FROM api_cache WHERE provider='internal' "
                         "AND key='brief_warmer_lock'")


class TestUniversePass(_IsolatedDB):
    def test_one_pass_publishes_every_universe_ticker(self):
        _seed("AAA")
        _seed("BBB")
        app_module._warm_brief_universe(watchlist=["AAA"],
                                        sp500=[{"symbol": "BBB"}],
                                        background=False)
        a = app_module.current_brief("AAA")
        b = app_module.current_brief("BBB")
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        self.assertNotEqual(a["meta"]["snapshot_id"], b["meta"]["snapshot_id"])
        self.assertEqual(a["meta"]["validation_status"], "ok")
        self.assertEqual(b["meta"]["validation_status"], "ok")

    def test_rewarm_within_retention_keeps_two_distinct_snapshots(self):
        _seed("AAA")
        app_module._warm_brief_universe(watchlist=["AAA"], sp500=[],
                                        background=False)
        first = app_module.current_brief("AAA")["meta"]["snapshot_id"]
        self._clear_warmer_lock()
        app_module._warm_brief_universe(watchlist=["AAA"], sp500=[],
                                        background=False, fresh_hours=0)
        second = app_module.current_brief("AAA")["meta"]["snapshot_id"]
        self.assertNotEqual(first, second)
        with db.get_conn() as conn:
            rows = conn.execute(
                "SELECT key FROM api_cache WHERE provider='brief:AAA' "
                "AND key LIKE 'snapshot:%'").fetchall()
        self.assertEqual(len(rows), 2)
        self.assertEqual(app_module.current_brief("AAA")["meta"]["snapshot_id"],
                         second)


class TestTiering(_IsolatedDB):
    def test_watchlist_full_spx_deltas_overlap_wins_full(self):
        _seed("AAA")
        _seed("BBB")
        _seed("CCC")
        app_module._warm_brief_universe(watchlist=["AAA", "CCC"],
                                        sp500=[{"symbol": "BBB"},
                                               {"symbol": "CCC"}],
                                        background=False)
        full = app_module.current_brief("AAA")
        deltas = app_module.current_brief("BBB")
        overlap = app_module.current_brief("CCC")
        self.assertEqual(set(full["sections"]),
                         {"price", "earnings", "context"})
        self.assertEqual(full["meta"]["depth"], "full")
        self.assertEqual(set(deltas["sections"]), {"price", "earnings"})
        self.assertEqual(deltas["meta"]["depth"], "deltas")
        self.assertEqual(set(overlap["sections"]),
                         {"price", "earnings", "context"})
        self.assertEqual(overlap["meta"]["depth"], "full")


class TestSourceFailure(_IsolatedDB):
    def test_all_sources_missing_still_publishes_declared_missing(self):
        app_module._warm_brief_universe(watchlist=["COLD"], sp500=[],
                                        background=False)
        out = app_module.current_brief("COLD")
        self.assertIsNotNone(out)
        for sec in out["sections"].values():
            self.assertEqual(sec["state"], "missing")
        self.assertTrue(out["meta"]["validation_status"].startswith("ok"))

    def test_one_tickers_hard_failure_does_not_abort_the_pass(self):
        _seed("AAA")
        _seed("BBB")
        real = app_module.publish_brief

        def flaky(ticker, depth="full"):
            if ticker == "AAA":
                raise RuntimeError("context read exploded")
            return real(ticker, depth=depth)

        with patch.object(app_module, "publish_brief", side_effect=flaky):
            app_module._warm_brief_universe(watchlist=["AAA", "BBB"],
                                            sp500=[], background=False)
        self.assertIsNotNone(app_module.current_brief("BBB"))
        progress = db.cache_get("brief", "warm_progress", 24)
        self.assertEqual(progress["failed"], 1)
        self.assertEqual(progress["published"], 1)


class TestFreshnessGate(_IsolatedDB):
    def test_all_missing_brief_does_not_consume_freshness(self):
        app_module._warm_brief_universe(watchlist=["COLD"], sp500=[],
                                        background=False)
        first = app_module.current_brief("COLD")
        self.assertTrue(all(sec["state"] == "missing"
                            for sec in first["sections"].values()))
        self._clear_warmer_lock()
        app_module._warm_brief_universe(watchlist=["COLD"], sp500=[],
                                        background=False)
        second = app_module.current_brief("COLD")
        self.assertNotEqual(first["meta"]["snapshot_id"],
                            second["meta"]["snapshot_id"])
        progress = db.cache_get("brief", "warm_progress", 24)
        self.assertEqual((progress["published"], progress["skipped"]), (1, 0))


class TestLockGuard(_IsolatedDB):
    def test_empty_universes_do_not_claim_the_lock(self):
        _seed("AAA")
        app_module._warm_brief_universe(watchlist=[], sp500=[],
                                        background=False)
        with db.get_conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM api_cache WHERE provider='internal' "
                "AND key='brief_warmer_lock'").fetchone()
        self.assertIsNone(row)
        app_module._warm_brief_universe(watchlist=["AAA"], sp500=[],
                                        background=False)
        self.assertIsNotNone(app_module.current_brief("AAA"))


class TestProgress(_IsolatedDB):
    def test_progress_observable_during_and_after_the_pass(self):
        for t in ("AAA", "BBB", "CCC"):
            _seed(t)
        seen_mid = {}
        real = app_module.publish_brief

        def spy(ticker, depth="full"):
            out = real(ticker, depth=depth)
            if ticker == "BBB":
                seen_mid["progress"] = db.cache_get("brief", "warm_progress", 24)
            return out

        with patch.object(app_module, "publish_brief", side_effect=spy):
            app_module._warm_brief_universe(watchlist=["AAA", "BBB", "CCC"],
                                            sp500=[], background=False)
        mid = seen_mid["progress"]
        self.assertEqual((mid["done"], mid["total"]), (1, 3))
        self.assertEqual(mid["published"], 1)
        final = db.cache_get("brief", "warm_progress", 24)
        self.assertEqual(final["done"], 3)
        self.assertEqual(final["total"], 3)
        self.assertEqual(final["published"], 3)

    def test_progress_write_failure_does_not_abort_the_pass(self):
        _seed("AAA")
        _seed("BBB")
        real_cache_set = db.cache_set

        def flaky_cache_set(provider, key, payload):
            if provider == "brief" and key == "warm_progress":
                raise RuntimeError("sqlite disk hiccup")
            return real_cache_set(provider, key, payload)

        with patch.object(app_module.db, "cache_set",
                          side_effect=flaky_cache_set):
            app_module._warm_brief_universe(watchlist=["AAA", "BBB"],
                                            sp500=[], background=False)
        self.assertIsNotNone(app_module.current_brief("AAA"))
        self.assertIsNotNone(app_module.current_brief("BBB"))


class TestIdempotence(_IsolatedDB):
    def test_complete_universe_rewarm_does_no_work(self):
        _seed("AAA")
        _seed("BBB")
        app_module._warm_brief_universe(watchlist=["AAA", "BBB"], sp500=[],
                                        background=False)
        ids = {t: app_module.current_brief(t)["meta"]["snapshot_id"]
               for t in ("AAA", "BBB")}
        self._clear_warmer_lock()
        with patch.object(app_module.brief, "load_brief_context",
                          side_effect=AssertionError("re-warm built a brief")):
            app_module._warm_brief_universe(watchlist=["AAA", "BBB"],
                                            sp500=[], background=False)
        self.assertEqual(ids, {t: app_module.current_brief(t)["meta"]["snapshot_id"]
                               for t in ("AAA", "BBB")})
        progress = db.cache_get("brief", "warm_progress", 24)
        self.assertEqual(progress["skipped"], 2)
        self.assertEqual(progress["published"], 0)
        self.assertEqual(progress["failed"], 0)


class TestDefaultSources(_IsolatedDB):
    def test_watchlist_from_radar_universe_spx_from_constituents(self):
        _seed("AAA")
        _seed("BBB")
        with patch.object(app_module.radar, "universe_symbols",
                          return_value=["AAA"]) as uni, \
             patch.object(app_module.heatmap, "get_constituents",
                          return_value=[{"symbol": "BBB"}]) as constituents:
            app_module._warm_brief_universe(background=False)
        uni.assert_called_once_with(min_bars=200)
        constituents.assert_called_once()
        self.assertEqual(app_module.current_brief("AAA")["meta"]["depth"], "full")
        self.assertEqual(app_module.current_brief("BBB")["meta"]["depth"], "deltas")


if __name__ == "__main__":
    unittest.main()
