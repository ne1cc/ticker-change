"""Daily brief builder: context in, contract-validated brief out.

Covers the six acceptance criteria: validated build for a cached ticker,
weekend/holiday session alignment, explicitly incomplete windows, receipts
on every numeric field with declared missing states, sentinel-proven
zero-network build, and enforced contract completeness.
"""
import json
import os
import tempfile
import unittest
from datetime import date
from unittest.mock import patch

import pandas as pd

import db

_fd, _DB_PATH = tempfile.mkstemp(suffix=".db")
os.close(_fd)
db.DB_PATH = _DB_PATH
db.init_db()

import brief  # noqa: E402
import brief_contracts  # noqa: E402


def _store(sym, closes, volumes=None, end="2026-10-12", periods=None):
    n = len(closes)
    idx = pd.bdate_range(end=end, periods=periods or n)
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
                  "Forward P/E": 10.0, "Price / Sales": 2.0,
                  "Dividend Yield": 0.01})
    db.cache_set("yfinance", f"earnings:{ticker}", [
        {"date": "2026-09-08", "eps_estimate": 1.0, "eps_actual": 1.2,
         "surprise_pct": 20.0, "is_upcoming": False},
        {"date": (pd.Timestamp.today() + pd.Timedelta(days=3)).date().isoformat(),
         "eps_estimate": 1.3, "eps_actual": None, "is_upcoming": True}])
    db.cache_set("yfinance", f"short:{ticker}",
                 {"short_pct_float": 0.05, "days_to_cover": 2.0,
                  "shares_short": 1_000_000.0,
                  "shares_short_prior_month": 1_100_000.0})


class _IsolatedDB:
    """Pin this module's throwaway DB for each test: other suite modules
    leave db.DB_PATH pointing at deleted temp files after tearDown."""

    def setUp(self):
        self._orig_db = db.DB_PATH
        db.DB_PATH = _DB_PATH
        db.init_db()

    def tearDown(self):
        db.DB_PATH = self._orig_db


class TestBuild(_IsolatedDB, unittest.TestCase):
    def test_build_returns_contract_validated_brief(self):
        _seed()
        ctx = brief.load_brief_context("AAA")
        out = brief.build_brief("AAA", ctx)
        ok, problems = brief.validate_brief(out)
        self.assertTrue(ok, problems)
        self.assertEqual(out["meta"]["validation_status"], "ok")
        self.assertEqual(out["meta"]["ticker"], "AAA")
        for name in brief_contracts.METRIC_KEYS_BY_SECTION:
            self.assertIn(name, out["sections"])
        price = out["sections"]["price"]
        self.assertEqual(price["state"], "ok")
        self.assertAlmostEqual(price["session_return"], 0.05, places=6)
        self.assertEqual(price["significance_state"], "ok")
        self.assertEqual(price["significance"], "notable")
        self.assertGreaterEqual(price["z_score"], 2.0)
        self.assertAlmostEqual(price["volume_ratio"], 1.0, places=6)
        self.assertEqual(price["vol_window_sessions"], 20)
        json.dumps(out)  # store-ready: JSON-safe end to end

    def test_uncontracted_section_key_fails_validation(self):
        _seed()
        out = brief.build_brief("AAA", brief.load_brief_context("AAA"))
        out["sections"]["price"]["mystery_alpha"] = 1.0
        ok, problems = brief.validate_brief(out)
        self.assertFalse(ok)
        self.assertTrue(any("uncontracted" in p and "mystery_alpha" in p
                            for p in problems), problems)

    def test_incomplete_contract_entry_fails_validation(self):
        _seed()
        out = brief.build_brief("AAA", brief.load_brief_context("AAA"))
        broken = {**brief_contracts.contract("z_score"), "unit": ""}
        with patch.dict(brief_contracts._CONTRACTS, {"z_score": broken}):
            ok, problems = brief.validate_brief(out)
        self.assertFalse(ok)
        self.assertTrue(any("z_score" in p and "unit" in p for p in problems),
                        problems)

    def test_missing_contracted_key_fails_validation(self):
        _seed()
        out = brief.build_brief("AAA", brief.load_brief_context("AAA"))
        del out["sections"]["price"]["session_return"]
        ok, problems = brief.validate_brief(out)
        self.assertFalse(ok)
        self.assertTrue(any("session_return" in p for p in problems), problems)


class TestSessionAlignment(_IsolatedDB, unittest.TestCase):
    def test_friday_to_monday_is_one_session_step(self):
        closes = [100.0 + i for i in range(28)] + [100.0, 105.0]
        _store("WKND", closes)  # last two bars: Fri 2026-10-09, Mon 2026-10-12
        db.cache_set("yfinance", "earnings:WKND", [
            {"date": "2026-10-10", "eps_estimate": 1.0, "eps_actual": 1.4,
             "surprise_pct": 40.0, "is_upcoming": False}])
        out = brief.build_brief(
            "WKND", brief.load_brief_context("WKND"))
        price = out["sections"]["price"]
        earn = out["sections"]["earnings"]
        self.assertEqual(price["session_date"], "2026-10-12")
        self.assertAlmostEqual(price["session_return"], 0.05, places=6)
        # Saturday print reacts on Monday: zero sessions elapsed since
        self.assertEqual(earn["last_print_date"], "2026-10-10")
        self.assertEqual(earn["sessions_since_print"], 0)
        self.assertAlmostEqual(earn["reaction_ret"], 0.05, places=6)
        # zero sessions after the reaction: every drift window incomplete
        self.assertIsNone(earn["post_drift_5d"])
        self.assertIsNone(earn["post_drift_20d"])

    def test_holiday_gap_is_one_session_step(self):
        idx = pd.bdate_range(end="2026-10-12", periods=25)
        idx = idx[~idx.isin(pd.DatetimeIndex(["2026-10-09"]))]  # Friday holiday
        closes = [100.0 + i for i in range(len(idx) - 1)] + [110.0]
        db.store_prices("HOLI", pd.DataFrame(
            {"Open": closes, "High": closes, "Low": closes, "Close": closes,
             "Volume": [1_000_000] * len(closes)}, index=idx))
        out = brief.build_brief(
            "HOLI", brief.load_brief_context("HOLI"))
        price = out["sections"]["price"]
        self.assertEqual(str(idx[-2].date()), "2026-10-08")  # Thu before holiday
        self.assertEqual(price["session_date"], "2026-10-12")
        # one session step across the holiday: straight to the prior bar
        self.assertAlmostEqual(price["session_return"],
                               110.0 / closes[-2] - 1, places=6)
        self.assertEqual(price["vol_window_sessions"], 20)


class TestIncompleteWindows(_IsolatedDB, unittest.TestCase):
    def test_short_history_declares_incomplete_never_silently_short(self):
        closes = [100.0 + i for i in range(10)]
        _store("THIN", closes)
        idx = pd.bdate_range(end="2026-10-12", periods=10)
        db.cache_set("yfinance", "earnings:THIN", [
            {"date": idx[5].date().isoformat(), "eps_estimate": 1.0,
             "eps_actual": 1.1, "surprise_pct": 10.0, "is_upcoming": False}])
        out = brief.build_brief(
            "THIN", brief.load_brief_context("THIN"))
        price = out["sections"]["price"]
        self.assertEqual(price["significance_state"], "window incomplete")
        self.assertEqual(price["vol_window_sessions"], 8)
        self.assertIsNone(price["z_score"])
        self.assertIsNone(price["significance"])
        self.assertIsNone(price["volume_ratio"])
        earn = out["sections"]["earnings"]
        self.assertEqual(earn["sessions_since_print"], 4)
        self.assertIsNone(earn["post_drift_5d"])
        self.assertIsNone(earn["post_drift_20d"])
        ok, problems = brief.validate_brief(out)
        self.assertTrue(ok, problems)

    def test_degenerate_window_declared(self):
        _store("FLAT", [100.0] * 25)
        out = brief.build_brief(
            "FLAT", brief.load_brief_context("FLAT"))
        price = out["sections"]["price"]
        self.assertEqual(price["significance_state"], "degenerate window")
        self.assertIsNone(price["z_score"])


class TestReceipts(_IsolatedDB, unittest.TestCase):
    def test_every_numeric_carries_source_and_as_of(self):
        _seed()
        out = brief.build_brief("AAA", brief.load_brief_context("AAA"))
        receipts = out["meta"]["receipts"]
        self.assertEqual(set(receipts), set(brief_contracts.SOURCE_KEYS))
        for name, section in out["sections"].items():
            for key, value in section.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                source = brief_contracts.contract(key)["source"]
                self.assertIn(source, brief_contracts.SOURCE_KEYS,
                              f"{name}/{key}")
                self.assertIsNotNone(receipts.get(source),
                                     f"{name}/{key} unreceipted")
                self.assertTrue(receipts[source])

    def test_removing_earnings_yields_declared_missing_state(self):
        _seed()
        ctx = brief.load_brief_context("AAA")
        ctx["earnings"] = None
        out = brief.build_brief("AAA", ctx)
        earn = out["sections"]["earnings"]
        self.assertEqual(earn["state"], "missing")
        self.assertIsNone(out["meta"]["receipts"]["earnings"])
        for key in brief_contracts.METRIC_KEYS_BY_SECTION["earnings"]:
            self.assertIsNone(earn[key], key)
        ok, problems = brief.validate_brief(out)
        self.assertTrue(ok, problems)

    def test_removing_fundamentals_and_short_declares_states(self):
        _seed()
        ctx = brief.load_brief_context("AAA")
        ctx["fundamentals"] = None
        ctx["short"] = None
        out = brief.build_brief("AAA", ctx)
        context_sec = out["sections"]["context"]
        self.assertEqual(context_sec["state"], "missing")
        for key in brief_contracts.METRIC_KEYS_BY_SECTION["context"]:
            self.assertIsNone(context_sec[key], key)
        self.assertIsNone(out["meta"]["receipts"]["fundamentals"])
        self.assertIsNone(out["meta"]["receipts"]["short"])
        ok, problems = brief.validate_brief(out)
        self.assertTrue(ok, problems)

    def test_no_prices_declares_missing_price_section(self):
        _seed()
        ctx = brief.load_brief_context("AAA")
        ctx["prices"] = None
        ctx["source_as_of"]["prices"] = None
        out = brief.build_brief("AAA", ctx)
        self.assertEqual(out["sections"]["price"]["state"], "missing")
        self.assertIsNone(out["sections"]["price"]["session_return"])
        ok, problems = brief.validate_brief(out)
        self.assertTrue(ok, problems)


class TestDepths(_IsolatedDB, unittest.TestCase):
    def test_deltas_depth_carries_price_and_earnings_only(self):
        _seed()
        out = brief.build_brief("AAA", brief.load_brief_context("AAA"),
                                depth="deltas")
        self.assertEqual(out["meta"]["depth"], "deltas")
        self.assertEqual(set(out["sections"]), {"price", "earnings"})
        ok, problems = brief.validate_brief(out)
        self.assertTrue(ok, problems)

    def test_default_depth_is_the_full_three_sections(self):
        _seed()
        out = brief.build_brief("AAA", brief.load_brief_context("AAA"))
        self.assertEqual(out["meta"]["depth"], "full")
        self.assertEqual(set(out["sections"]),
                         {"price", "earnings", "context"})
        ok, _ = brief.validate_brief(out)
        self.assertTrue(ok)

    def test_legacy_brief_without_depth_still_validates_as_full(self):
        _seed()
        out = brief.build_brief("AAA", brief.load_brief_context("AAA"))
        del out["meta"]["depth"]
        ok, _ = brief.validate_brief(out)
        self.assertTrue(ok)

    def test_unknown_depth_is_rejected(self):
        with self.assertRaises(ValueError):
            brief.build_brief("AAA", {}, depth="bogus")


class TestNoNetwork(_IsolatedDB, unittest.TestCase):
    def test_build_touches_no_network_and_reads_no_cache(self):
        _seed()
        ctx = brief.load_brief_context("AAA")
        with patch("yfinance.Ticker", side_effect=AssertionError("network!")), \
             patch.object(db, "get_prices",
                          side_effect=AssertionError("network!")), \
             patch.object(db, "cache_get",
                          side_effect=AssertionError("network!")), \
             patch.object(db, "cache_get_with_age",
                          side_effect=AssertionError("network!")), \
             patch.object(db, "cache_scan",
                          side_effect=AssertionError("network!")):
            out = brief.build_brief("AAA", ctx)
        self.assertEqual(out["meta"]["validation_status"], "ok")

    def test_loader_touches_no_network(self):
        _seed()
        with patch("yfinance.Ticker", side_effect=AssertionError("network!")), \
             patch("requests.get", side_effect=AssertionError("network!"),
                   create=True):
            ctx = brief.load_brief_context("AAA")
        self.assertIsNotNone(ctx["prices"])
        self.assertIsNotNone(ctx["earnings"])
        self.assertIsNotNone(ctx["source_as_of"]["prices"])


if __name__ == "__main__":
    unittest.main()
