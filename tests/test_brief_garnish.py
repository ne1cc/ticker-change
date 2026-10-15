"""Brief AI garnish: an optional, labeled "Interpretation" section on
/brief/<ticker>, attached at render time from ai.py's provider chain.

Contract under test:
- configured provider + success -> visible AI labeling and provenance
- no provider / failed call -> 200 and every deterministic section renders
  byte-identically without it (additive-only at the view layer)
- cached -> repeat requests never re-call the provider (call-count proven)
- the published payload and its JSON twin stay AI-free
- the provider call is time-bounded (timeout actually wired)
"""
import json
import os
import re
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

import db

_fd, _DB_PATH = tempfile.mkstemp(suffix=".db")
os.close(_fd)
db.DB_PATH = _DB_PATH
db.init_db()

import ai as ai_module  # noqa: E402
import app as app_module  # noqa: E402


STUB_PROVIDERS = [{"id": "stubprov", "label": "Stub", "model": "stub-model",
                   "base_url": "http://stub", "key": "stub-key"}]
CANNED = ("Price ground higher on the session's +5.00% move while the "
          "upcoming earnings print keeps event risk elevated.")

_GARNISH_RE = re.compile(
    r'<section id="brief-ai-garnish".*?</section>', re.DOTALL)


def _squash(html):
    return re.sub(r"\s+", " ", html)


def _store(sym, closes, end="2026-10-12"):
    idx = pd.bdate_range(end=end, periods=len(closes))
    db.store_prices(sym, pd.DataFrame(
        {"Open": closes, "High": closes, "Low": closes, "Close": closes,
         "Volume": [1_000_000] * len(closes)}, index=idx))


def _seed(ticker="AAA"):
    _store(ticker, [100.0, 101.0] * 14 + [100.0, 105.0])
    db.cache_set("yfinance", f"fundamentals:{ticker}",
                 {"Sector": "Technology", "Market Cap": 1e9,
                  "Forward P/E": 10.0, "Price / Sales": 2.0})
    db.cache_set("yfinance", f"earnings:{ticker}", [
        {"date": "2026-09-08", "eps_estimate": 1.0, "eps_actual": 1.2,
         "surprise_pct": 20.0, "is_upcoming": False}])


def _stub_report(calls):
    def _fake(base_url, key, model, user_msg, system_prompt, timeout=None):
        calls.append({"timeout": timeout})
        return CANNED
    return _fake


class _IsolatedDB(unittest.TestCase):
    def setUp(self):
        self._orig_db = db.DB_PATH
        db.DB_PATH = _DB_PATH
        db.init_db()
        with db.get_conn() as conn:
            conn.execute("DELETE FROM api_cache WHERE provider LIKE 'brief:%' "
                         "OR provider = 'ai_brief_garnish'")
        self.client = app_module.app.test_client()

    def tearDown(self):
        db.DB_PATH = self._orig_db

    def _published(self, ticker="AAA"):
        _seed(ticker)
        published = app_module.publish_brief(ticker)
        self.assertIsNotNone(published)
        return published

    def _with_stub(self):
        calls = []
        cm_providers = patch.object(ai_module.providers, "ai_providers",
                                    lambda: STUB_PROVIDERS)
        cm_report = patch.object(ai_module, "_openai_compatible_report",
                                 side_effect=_stub_report(calls))
        return cm_providers, cm_report, calls


class TestBriefGarnish(_IsolatedDB):
    def test_provider_configured_renders_labeled_section(self):
        self._published()
        provs, reports, calls = self._with_stub()
        with provs, reports:
            resp = self.client.get("/brief/AAA")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertEqual(len(calls), 1)
        self.assertIn('id="brief-ai-garnish"', html)
        self.assertIn("Interpretation", html)
        self.assertIn("AI-generated", html)
        self.assertIn(CANNED, html)

    def test_provenance_provider_and_generation_time_disclosed(self):
        self._published()
        provs, reports, _ = self._with_stub()
        with provs, reports:
            html = self.client.get("/brief/AAA").get_data(as_text=True)
        self.assertIn("via stubprov", html)
        self.assertIn("generated 20", html)
        self.assertIn("not financial advice", html)

    def test_no_provider_renders_200_without_garnish(self):
        self._published()
        with patch.object(ai_module.providers, "ai_providers", lambda: []):
            resp = self.client.get("/brief/AAA")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertNotIn("AI-generated", html)
        self.assertNotIn("brief-ai-garnish", html)
        self.assertIn("+5.00%", html)

    def test_failed_call_leaves_every_other_section_unchanged(self):
        self._published()
        with patch.object(ai_module.providers, "ai_providers", lambda: []):
            without = self.client.get("/brief/AAA").get_data(as_text=True)
        with patch.object(ai_module.providers, "ai_providers",
                          lambda: STUB_PROVIDERS), \
             patch.object(ai_module, "_openai_compatible_report",
                          side_effect=RuntimeError("provider down")):
            resp = self.client.get("/brief/AAA")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_data(as_text=True), without)

    def test_success_is_additive_only_over_deterministic_render(self):
        self._published()
        with patch.object(ai_module.providers, "ai_providers", lambda: []):
            without = self.client.get("/brief/AAA").get_data(as_text=True)
        provs, reports, _ = self._with_stub()
        with provs, reports:
            with_garnish = self.client.get("/brief/AAA").get_data(as_text=True)
        self.assertNotEqual(with_garnish, without)
        self.assertEqual(_squash(_GARNISH_RE.sub("", with_garnish)),
                         _squash(without))

    def test_cached_no_provider_recall_on_repeat_requests(self):
        self._published()
        provs, reports, calls = self._with_stub()
        with provs, reports:
            for _ in range(3):
                resp = self.client.get("/brief/AAA")
                self.assertEqual(resp.status_code, 200)
                self.assertIn(CANNED, resp.get_data(as_text=True))
        self.assertEqual(len(calls), 1)

    def test_published_payload_and_twin_stay_ai_free(self):
        published = self._published()
        provs, reports, _ = self._with_stub()
        with provs, reports:
            page = self.client.get("/brief/AAA")
            twin = self.client.get("/api/brief/AAA")
        self.assertEqual(page.status_code, 200)
        self.assertEqual(twin.status_code, 200)
        raw = twin.get_data(as_text=True)
        self.assertNotIn("stubprov", raw)
        self.assertNotIn("AI-generated", raw)
        payload = json.loads(raw)
        self.assertEqual(payload, published)
        self.assertNotIn("garnish", payload)
        self.assertNotIn("interpretation", payload)

    def test_provider_call_is_time_bounded(self):
        published = self._published()
        provs, reports, calls = self._with_stub()
        with provs, reports:
            out = ai_module.generate_brief_interpretation("AAA", published)
        self.assertIsNotNone(out)
        self.assertEqual(calls[0]["timeout"], ai_module.BRIEF_GARNISH_TIMEOUT)
        self.assertLessEqual(ai_module.BRIEF_GARNISH_TIMEOUT, 30)

    def test_garnish_failure_never_breaks_the_page(self):
        self._published()
        with patch.object(ai_module, "generate_brief_interpretation",
                          side_effect=RuntimeError("boom")):
            resp = self.client.get("/brief/AAA")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertNotIn("AI-generated", html)
        self.assertIn("+5.00%", html)

    def test_empty_context_brief_still_wires_the_garnish_seam(self):
        app_module.publish_brief("NOPE")
        with patch.object(ai_module, "generate_brief_interpretation") as gen:
            resp = self.client.get("/brief/NOPE")
        self.assertEqual(resp.status_code, 200)
        gen.assert_called_once()


if __name__ == "__main__":
    unittest.main()
