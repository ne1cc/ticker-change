"""Epic 9: /analytics request path (lazy fragments, async AI, data receipts)."""
import os
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd

import db

import ai  # noqa: E402
import analytics_receipts  # noqa: E402
import app as app_module  # noqa: E402


def setUpModule():
    # Other test modules also assign db.DB_PATH at import, so bind ours at run time.
    global _orig_db_path
    _orig_db_path = db.DB_PATH
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db.DB_PATH = path
    db.init_db()


def tearDownModule():
    db.DB_PATH = _orig_db_path


def _seed_prices(*tickers):
    idx = pd.bdate_range("2024-01-01", periods=300)
    close = 100 + np.arange(300) + np.sin(np.arange(300))
    frame = pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1,
                          "Close": close, "Volume": np.full(300, 1000)}, index=idx)
    for t in tickers:
        db.store_prices(t, frame)


class TestLazyFragments(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _seed_prices("AAPL", "SPY")
        cls.client = app_module.app.test_client()

    def test_core_page_renders_with_hanging_providers(self):
        release = threading.Event()
        hang = lambda *a, **k: release.wait(30)  # noqa: E731
        try:
            with patch.object(app_module, "get_fundamentals", hang), \
                 patch.object(app_module, "get_options_smile", hang), \
                 patch.object(app_module, "get_gex_profile", hang), \
                 patch.object(app_module.earnings, "build_earnings_section", hang), \
                 patch.object(ai, "generate_report", hang):
                started = time.perf_counter()
                resp = self.client.get("/analytics?ticker=AAPL")
                elapsed = time.perf_counter() - started
        finally:
            release.set()
        self.assertEqual(resp.status_code, 200)
        self.assertLess(elapsed, 5)
        html = resp.get_data(as_text=True)
        self.assertIn("/analytics/fragment/", html)  # slow sections are deferred, not inlined

    def test_hanging_section_errors_without_affecting_others(self):
        release = threading.Event()
        try:
            with patch.object(app_module, "SECTION_TIMEOUT_S", 0.2), \
                 patch.object(app_module, "get_fundamentals", lambda *a, **k: release.wait(30)), \
                 patch.object(app_module.earnings, "build_earnings_section",
                              return_value={"events": [], "summary": {"n_events": 0}, "next_print": None,
                                            "implied_move": None, "realized_move": None, "event_vol": None}):
                resp = self.client.get("/analytics/fragment/company?ticker=AAPL")
        finally:
            release.set()
        html = resp.get_data(as_text=True)
        self.assertEqual(resp.status_code, 200)
        self.assertIn('data-section-error="fundamentals"', html)
        self.assertIn("timed out", html)
        self.assertNotIn('data-section-error="earnings"', html)

    def test_unknown_fragment_and_missing_ticker_404(self):
        self.assertEqual(self.client.get("/analytics/fragment/nope?ticker=AAPL").status_code, 404)
        self.assertEqual(self.client.get("/analytics/fragment/company").status_code, 404)

    def test_guard_logs_timing_and_records_status(self):
        status = {}
        with patch("builtins.print") as printed:
            app_module._guard_section("demo", lambda: 1, status=status)
            app_module._guard_section("boom", lambda: 1 / 0, status=status)
            app_module._guard_section("slow", lambda: time.sleep(5), timeout=0.05, status=status)
        self.assertEqual(status, {"demo": "ok", "boom": "error", "slow": "timeout"})
        logged = " ".join(str(c.args[0]) for c in printed.call_args_list if c.args)
        for label, outcome in (("demo", "ok"), ("boom", "error"), ("slow", "timeout")):
            self.assertRegex(logged, rf"\[analytics:{label}\] {outcome} in \d+ms")


class TestAiOffRequestPath(unittest.TestCase):
    def setUp(self):
        self.keys = patch.object(ai.providers, "ai_providers", return_value=["anthropic"])
        self.keys.start()
        self.addCleanup(self.keys.stop)

    def test_unconfigured(self):
        with patch.object(ai.providers, "ai_providers", return_value=[]):
            self.assertEqual(ai.report_status("UNCF", lambda: {})["state"], "unconfigured")

    def test_hanging_client_never_blocks(self):
        release = threading.Event()
        try:
            with patch.object(ai, "generate_report", lambda *a, **k: release.wait(30)):
                started = time.perf_counter()
                state = ai.report_status("HANG", lambda: {})
                elapsed = time.perf_counter() - started
        finally:
            release.set()
        self.assertEqual(state["state"], "pending")
        self.assertLess(elapsed, 1)

    def test_cached_report_is_ready_without_generation(self):
        db.cache_set("ai_report", "CACHED", {"html": "<p>hi</p>", "provider": "x"})
        with patch.object(ai, "generate_report", side_effect=AssertionError("must not generate")):
            state = ai.report_status("CACHED", lambda: {})
        self.assertEqual(state["state"], "ready")
        self.assertEqual(state["html"], "<p>hi</p>")
        self.assertIsNotNone(state["generated_at"])

    def test_failed_generation_is_declared_then_not_restarted(self):
        done = threading.Event()

        def fail(*a, **k):
            done.set()
            return None

        with patch.object(ai, "generate_report", fail):
            self.assertEqual(ai.report_status("FAIL", lambda: {})["state"], "pending")
            self.assertTrue(done.wait(5))
            deadline = time.time() + 5
            while time.time() < deadline and ai.report_status("FAIL", lambda: {})["state"] != "failed":
                time.sleep(0.05)
        self.assertEqual(ai.report_status("FAIL", lambda: {})["state"], "failed")

    def test_fragment_renders_each_state(self):
        _seed_prices("AAPL")
        client = app_module.app.test_client()
        with patch.object(ai, "report_status", return_value={"state": "pending"}):
            self.assertIn('data-ai-state="pending"', client.get("/analytics/fragment/ai?ticker=AAPL").get_data(as_text=True))
        with patch.object(ai, "report_status", return_value={"state": "failed"}):
            self.assertIn('data-ai-state="failed"', client.get("/analytics/fragment/ai?ticker=AAPL").get_data(as_text=True))
        with patch.object(ai, "report_status", return_value={"state": "unconfigured"}):
            self.assertEqual(client.get("/analytics/fragment/ai?ticker=AAPL").get_data(as_text=True), "")


class TestReceipts(unittest.TestCase):
    NOW = datetime(2026, 10, 24, 12, 0)

    def test_unavailable_when_panel_absent(self):
        r = analytics_receipts.receipt("fundamentals", "AAPL", present=False)
        self.assertEqual(r["state"], "unavailable")
        self.assertTrue(r["source"])

    def test_live_and_stale_derive_from_as_of_and_ttl(self):
        live = analytics_receipts.receipt("fundamentals", "AAPL", present=True,
                                          as_of=self.NOW - timedelta(hours=2), now=self.NOW)
        stale = analytics_receipts.receipt("fundamentals", "AAPL", present=True,
                                           as_of=self.NOW - timedelta(hours=30), now=self.NOW)
        self.assertEqual((live["state"], live["age_text"]), ("live", "2h old"))
        self.assertEqual((stale["state"], stale["age_text"]), ("stale", "30h old"))

    def test_assumed_inputs_override_state(self):
        r = analytics_receipts.receipt("short_interest", "AAPL", present=True, as_of=self.NOW,
                                       assumed=["SI % float"], now=self.NOW)
        self.assertEqual(r["state"], "assumed")
        self.assertIn("SI % float", r["note"])

    def test_no_asof_when_cache_has_no_row(self):
        r = analytics_receipts.receipt("fundamentals", "NOROW", present=True)
        self.assertEqual(r["state"], "no_asof")

    def test_as_of_read_from_cache_row(self):
        db.cache_set("yfinance", "fundamentals:RCPT", {"x": 1})
        r = analytics_receipts.receipt("fundamentals", "RCPT", present=True)
        self.assertEqual(r["state"], "live")
        self.assertIsNotNone(db.cache_fetched_at("yfinance", "fundamentals:RCPT"))

    def test_unreadable_cache_degrades_to_no_asof(self):
        with patch.object(db, "cache_fetched_at", side_effect=RuntimeError("no table")):
            r = analytics_receipts.receipt("fundamentals", "AAPL", present=True)
        self.assertEqual(r["state"], "no_asof")


if __name__ == "__main__":
    unittest.main()
