"""Brief publication: per-ticker pointer-last store, last-good, retention."""
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
import brief  # noqa: E402


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


class TestPublish(_IsolatedDB, unittest.TestCase):
    def test_publish_moves_pointer_and_serves_current(self):
        _seed("AAA")
        out = app_module.publish_brief("AAA")
        self.assertIsNotNone(out)
        self.assertEqual(out["meta"]["validation_status"], "ok")
        current = app_module.current_brief("AAA")
        self.assertIsNotNone(current)
        self.assertEqual(current["meta"]["snapshot_id"],
                         out["meta"]["snapshot_id"])
        self.assertEqual(current["meta"]["ticker"], "AAA")
        self.assertIn("receipts", current["meta"])

    def test_failed_build_never_replaces_pointer(self):
        _seed("AAA")
        app_module.publish_brief("AAA")
        first = app_module.current_brief("AAA")["meta"]["snapshot_id"]
        with patch.object(app_module.brief, "load_brief_context",
                          side_effect=RuntimeError("boom")):
            self.assertIsNone(app_module.publish_brief("AAA"))
        self.assertEqual(app_module.current_brief("AAA")["meta"]["snapshot_id"],
                         first)

    def test_invalid_build_never_replaces_pointer(self):
        _seed("AAA")
        app_module.publish_brief("AAA")
        first = app_module.current_brief("AAA")["meta"]["snapshot_id"]

        def _uncontracted(ticker, context):
            out = brief.build_brief(ticker, context)
            out["sections"]["price"]["mystery_alpha"] = 1.0
            return out

        with patch.object(app_module.brief, "build_brief",
                          side_effect=_uncontracted):
            self.assertIsNone(app_module.publish_brief("AAA"))
        self.assertEqual(app_module.current_brief("AAA")["meta"]["snapshot_id"],
                         first)

    def test_retention_keeps_last_two_with_distinct_ids(self):
        _seed("AAA")
        with patch.object(app_module.snapshot_store, "new_snapshot_id",
                          side_effect=[f"S{i}" for i in range(4)]):
            for _ in range(4):
                app_module.publish_brief("AAA")
        with db.get_conn() as conn:
            rows = conn.execute(
                "SELECT key FROM api_cache WHERE provider='brief:AAA' "
                "AND key LIKE 'snapshot:%'").fetchall()
        ids = {r["key"].split(":", 1)[1] for r in rows}
        self.assertEqual(len(rows), 2)
        self.assertEqual(ids, {"S2", "S3"})
        self.assertEqual(app_module.current_brief("AAA")["meta"]["snapshot_id"],
                         "S3")

    def test_tickers_publish_to_separate_namespaces(self):
        _seed("AAA")
        _seed("BBB")
        a = app_module.publish_brief("AAA")
        self.assertIsNotNone(app_module.current_brief("AAA"))
        self.assertIsNone(app_module.current_brief("BBB"))
        b = app_module.publish_brief("BBB")
        self.assertNotEqual(a["meta"]["snapshot_id"], b["meta"]["snapshot_id"])
        self.assertEqual(app_module.current_brief("BBB")["meta"]["ticker"], "BBB")
        self.assertEqual(app_module.current_brief("AAA")["meta"]["snapshot_id"],
                         a["meta"]["snapshot_id"])

    def test_kick_builds_and_publishes_synchronously_when_asked(self):
        _seed("AAA")
        app_module.kick_brief_build("AAA", background=False)
        current = app_module.current_brief("AAA")
        self.assertIsNotNone(current)
        self.assertEqual(current["meta"]["ticker"], "AAA")


class TestPublishNoNetwork(_IsolatedDB, unittest.TestCase):
    def test_publish_touches_no_network(self):
        _seed("AAA")
        with patch("yfinance.Ticker", side_effect=AssertionError("network!")), \
             patch("requests.get", side_effect=AssertionError("network!"),
                   create=True), \
             patch.object(app_module, "get_or_fetch_prices",
                          side_effect=AssertionError("network!")):
            out = app_module.publish_brief("AAA")
        self.assertIsNotNone(out)
        self.assertEqual(out["meta"]["validation_status"], "ok")

    def test_empty_context_publishes_declared_missing_states(self):
        out = app_module.publish_brief("NOPE")
        self.assertIsNotNone(out)
        for sec in out["sections"].values():
            self.assertEqual(sec["state"], "missing")
        self.assertTrue(out["meta"]["validation_status"].startswith("ok"))


if __name__ == "__main__":
    unittest.main()
