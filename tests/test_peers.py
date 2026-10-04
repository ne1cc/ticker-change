"""Peer resolution: Finnhub primary, warmed-classification fallback, cap."""
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

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


class TestPercentileAndQuadrant(unittest.TestCase):
    def test_percentile_rank_basic_and_invert(self):
        vals = [10.0, 20.0, 30.0, 40.0]
        self.assertEqual(peers.percentile_rank(vals, 30.0), 66.7)
        self.assertEqual(peers.percentile_rank(vals, 30.0, invert=True), 33.3)

    def test_percentile_edge_cases(self):
        self.assertIsNone(peers.percentile_rank([5.0], 5.0))
        self.assertIsNone(peers.percentile_rank([1.0, None, 2.0], None))

    def test_quadrant_labels(self):
        self.assertEqual(peers.quadrant(80.0, 0.25), "Cheap & Strengthening")
        self.assertEqual(peers.quadrant(80.0, -0.25), "Cheap & Weakening")
        self.assertEqual(peers.quadrant(10.0, 0.25), "Rich & Strengthening")
        self.assertEqual(peers.quadrant(10.0, -0.25), "Rich & Weakening")

    def test_value_score_ignores_none(self):
        self.assertEqual(peers.value_score({"pe": 80.0, "ps": None}), 80.0)
        self.assertIsNone(peers.value_score({"pe": None, "ps": None}))


class TestBuildPeerComparison(unittest.TestCase):
    def _seed(self):
        db.cache_set("yfinance", "fundamentals:FOCUS", {
            "Forward P/E": 10.0, "Price / Sales": 1.0, "Revenue Growth": 0.2})
        db.cache_set("yfinance", "fundamentals:PEER1", {
            "Forward P/E": 30.0, "Price / Sales": 5.0, "Revenue Growth": 0.1})
        db.cache_set("yfinance", "fundamentals:PEER2", {
            "Forward P/E": 20.0, "Price / Sales": 3.0, "Revenue Growth": 0.05})
        idx = pd.bdate_range("2024-01-01", periods=300)
        for sym, drift in [("FOCUS", 0.001), ("PEER1", -0.001), ("PEER2", 0.0)]:
            close = 100 * np.exp(np.cumsum(np.full(300, drift)))
            db.store_prices(sym, pd.DataFrame(
                {"Open": close, "High": close, "Low": close,
                 "Close": close, "Volume": np.full(300, 1_000_000)}, index=idx))

    def test_build_comparison(self):
        self._seed()
        with patch.object(peers, "resolve_peers", return_value=["PEER1", "PEER2"]):
            out = peers.build_peer_comparison("FOCUS")
        self.assertIsNotNone(out)
        self.assertEqual(len(out["peers"]), 2)
        focus = out["focus"]
        self.assertGreater(focus["valuation_pct"], 50.0)
        self.assertEqual(focus["quadrant"], "Cheap & Strengthening")
        self.assertIn("mom_12_1", focus)

    def test_none_without_peers(self):
        with patch.object(peers, "resolve_peers", return_value=None):
            self.assertIsNone(peers.build_peer_comparison("NOPE"))
