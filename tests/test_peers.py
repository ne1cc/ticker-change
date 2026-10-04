"""Peer resolution: Finnhub primary, warmed-classification fallback, cap."""
import os
import tempfile
import unittest
from unittest.mock import patch

import db

_fd, _DB_PATH = tempfile.mkstemp(suffix=".db")
os.close(_fd)
db.DB_PATH = _DB_PATH
db.init_db()

import providers  # noqa: E402
import peers  # noqa: E402


class TestCacheScan(unittest.TestCase):
    def test_scan_returns_suffix_keyed_payloads(self):
        db.cache_set("t", "fundamentals:AAA", {"x": 1})
        db.cache_set("t", "fundamentals:BBB", {"y": 2})
        db.cache_set("t", "other:CCC", {"z": 3})
        out = db.cache_scan("t", "fundamentals:")
        self.assertEqual(out, {"AAA": {"x": 1}, "BBB": {"y": 2}})


class TestResolvePeers(unittest.TestCase):
    def test_finnhub_primary_filters_self_and_caps(self):
        with patch.object(providers, "finnhub_peer",
                          return_value=["AAPL", "MSFT", "GOOG", "AMZN", "META",
                                        "NVDA", "TSLA", "CRM", "ORCL", "AMD"]):
            out = peers.resolve_peers("AAPL", cap=4)
        self.assertEqual(out, ["MSFT", "GOOG", "AMZN", "META"])

    def test_fallback_same_industry_from_cache(self):
        db.cache_set("yfinance", "fundamentals:AAPL",
                     {"Industry": "Tech", "Sector": "Technology"})
        for sym in ["MSFT", "GOOG", "BANK"]:
            db.cache_set("yfinance", f"fundamentals:{sym}",
                         {"Industry": "Tech" if sym != "BANK" else "Banks",
                          "Sector": "Technology" if sym != "BANK" else "Financials"})
        db.cache_set("sp500", "constituents",
                     [{"symbol": s, "name": s, "sector": "Technology"}
                      for s in ["AAPL", "MSFT", "GOOG", "BANK"]])
        with patch.object(providers, "finnhub_peer", return_value=None):
            out = peers.resolve_peers("AAPL", cap=8)
        self.assertEqual(out, ["GOOG", "MSFT"])

    def test_none_when_nothing_available(self):
        with patch.object(providers, "finnhub_peer", return_value=None), \
             patch.object(peers, "universe_classifications", return_value={}):
            self.assertIsNone(peers.resolve_peers("ZZZZ", cap=8))


class TestUniverseClassifications(unittest.TestCase):
    def test_malformed_cache_payloads_are_skipped(self):
        db.cache_set("yfinance", "fundamentals:GOOD",
                     {"Industry": "Tech", "Sector": "Technology"})
        db.cache_set("yfinance", "fundamentals:BAD", 42)
        db.cache_set("sp500", "constituents",
                     ["not-a-dict", 7,
                      {"symbol": "FROMIDX", "name": "F", "sector": "Technology"}])
        out = peers.universe_classifications()
        self.assertEqual(out["GOOD"], {"industry": "Tech", "sector": "Technology"})
        self.assertNotIn("BAD", out)
        self.assertEqual(out["FROMIDX"], {"industry": None, "sector": "Technology"})
