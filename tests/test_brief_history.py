"""Brief history per ticker: the retained-edition strip on /brief/<ticker>,
the read-only edition view at /brief/<ticker>/<snapshot_id>, and the declared
gone-state for editions beyond retention.

Contract under test:
- the history strip lists retained editions (newest first) with session dates
  and links to their edition URLs, and holds up at the common single-edition case
- a past edition renders read-only at its stable URL: no warmer kicks, no
  on-demand builds, no pointer or retention writes, no network
- an edition id beyond retention (or never stored, or TTL-expired) renders a
  declared gone-state — never a 500, never a redirect to current
- the current edition is badged "Current" wherever it renders; a historical
  edition carries the "Historical" badge and the archived banner
- the AI garnish attaches only to the live brief: historical editions render
  deterministic content alone and never call the provider chain
"""
import os
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

import pandas as pd

import db

_fd, _DB_PATH = tempfile.mkstemp(suffix=".db")
os.close(_fd)
db.DB_PATH = _DB_PATH
db.init_db()

import ai as ai_module  # noqa: E402
import app as app_module  # noqa: E402
import snapshot_store  # noqa: E402


STUB_PROVIDERS = [{"id": "stubprov", "label": "Stub", "model": "stub-model",
                   "base_url": "http://stub", "key": "stub-key"}]
CANNED = "Stub interpretation of the live brief only."


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


def _seed_next_session(ticker="AAA"):
    _store(ticker, [100.0, 101.0] * 14 + [100.0, 110.0], end="2026-10-13")


def _publish_two(ticker="AAA"):
    _seed(ticker)
    first = app_module.publish_brief(ticker)
    _seed_next_session(ticker)
    second = app_module.publish_brief(ticker)
    return first, second


def _stub_report(calls):
    def _fake(base_url, key, model, user_msg, system_prompt, timeout=None):
        calls.append({"timeout": timeout})
        return CANNED
    return _fake


class _IsolatedDB:
    def setUp(self):
        self._orig_db = db.DB_PATH
        db.DB_PATH = _DB_PATH
        db.init_db()
        with db.get_conn() as conn:
            conn.execute("DELETE FROM api_cache WHERE provider LIKE 'brief:%' "
                         "OR provider = 'ai_brief_garnish'")
            conn.execute("DELETE FROM daily_prices")

    def tearDown(self):
        db.DB_PATH = self._orig_db

    def _client(self):
        return app_module.app.test_client()


class TestSnapshotStoreHistory(unittest.TestCase):
    def setUp(self):
        self._orig_db = db.DB_PATH
        db.DB_PATH = _DB_PATH
        db.init_db()
        with db.get_conn() as conn:
            conn.execute("DELETE FROM api_cache WHERE provider IN ('p', 'p2')")

    def tearDown(self):
        db.DB_PATH = self._orig_db

    def test_history_returns_retained_payloads_newest_first(self):
        db.cache_set("p", "snapshot:S1", {"n": 1})
        db.cache_set("p", "snapshot:S2", {"n": 2})
        self.assertEqual(snapshot_store.history("p"),
                         [("S2", {"n": 2}), ("S1", {"n": 1})])

    def test_history_skips_pointer_and_foreign_providers(self):
        db.cache_set("p", "current", {"snapshot_id": "S1"})
        db.cache_set("p2", "snapshot:S9", {"n": 9})
        db.cache_set("p", "snapshot:S1", {"n": 1})
        self.assertEqual(snapshot_store.history("p"), [("S1", {"n": 1})])

    def test_history_treats_expired_rows_as_pruned(self):
        db.cache_set("p", "snapshot:S1", {"n": 1})
        old = (datetime.utcnow()
               - timedelta(hours=snapshot_store._TTL_HOURS + 1)).isoformat()
        with db.get_conn() as conn:
            conn.execute("UPDATE api_cache SET fetched_at = ? "
                         "WHERE provider = 'p'", (old,))
        self.assertEqual(snapshot_store.history("p"), [])

    def test_snapshot_reads_one_payload_or_none(self):
        db.cache_set("p", "snapshot:S1", {"n": 1})
        db.cache_set("p", "snapshot:bad", [1, 2])
        self.assertEqual(snapshot_store.snapshot("p", "S1"), {"n": 1})
        self.assertIsNone(snapshot_store.snapshot("p", "missing"))
        self.assertIsNone(snapshot_store.snapshot("p", "bad"))


class TestBriefHistoryPage(_IsolatedDB, unittest.TestCase):
    def test_history_strip_lists_retained_editions_with_session_dates(self):
        first, second = _publish_two()
        resp = self._client().get("/brief/AAA")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertIn("Past editions", html)
        self.assertIn(f"/brief/AAA/{first['meta']['snapshot_id']}", html)
        self.assertIn(f"/brief/AAA/{second['meta']['snapshot_id']}", html)
        self.assertIn("2026-10-12", html)
        self.assertIn("2026-10-13", html)
        self.assertLess(html.find(second["meta"]["snapshot_id"]),
                        html.find(first["meta"]["snapshot_id"]))

    def test_single_edition_history_strip_renders(self):
        _seed("AAA")
        published = app_module.publish_brief("AAA")
        self.assertIsNotNone(published)
        resp = self._client().get("/brief/AAA")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertIn("Past editions", html)
        self.assertIn(f"/brief/AAA/{published['meta']['snapshot_id']}", html)
        self.assertIn("2026-10-12", html)
        self.assertIn("Current", html)
        self.assertEqual(html.count("/brief/AAA/"), 1)

    def test_past_edition_renders_read_only_at_stable_url(self):
        first, _ = _publish_two()
        old_id = first["meta"]["snapshot_id"]
        with db.get_conn() as conn:
            rows_before = conn.execute(
                "SELECT key, payload FROM api_cache WHERE provider='brief:AAA' "
                "ORDER BY key").fetchall()
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
                          side_effect=AssertionError("network!")), \
             patch.object(app_module, "kick_brief_build",
                          side_effect=AssertionError("read-only!")), \
             patch.object(app_module, "publish_brief",
                          side_effect=AssertionError("read-only!")), \
             patch.object(app_module, "_build_brief_on_demand",
                          side_effect=AssertionError("read-only!")):
            resp = self._client().get(f"/brief/AAA/{old_id}")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertIn("+5.00%", html)
        self.assertNotIn("+10.00%", html)
        self.assertIn("Historical", html)
        self.assertIn("Archived edition", html)
        with db.get_conn() as conn:
            rows_after = conn.execute(
                "SELECT key, payload FROM api_cache WHERE provider='brief:AAA' "
                "ORDER BY key").fetchall()
        self.assertEqual([tuple(r) for r in rows_before],
                         [tuple(r) for r in rows_after])

    def test_edition_route_is_case_insensitive(self):
        first, _ = _publish_two()
        resp = self._client().get(f"/brief/aaa/{first['meta']['snapshot_id']}")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("+5.00%", resp.get_data(as_text=True))

    def test_current_edition_url_renders_as_current(self):
        _, second = _publish_two()
        resp = self._client().get(f"/brief/AAA/{second['meta']['snapshot_id']}")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertIn("+10.00%", html)
        self.assertIn("Current", html)
        self.assertNotIn("Historical", html)
        self.assertNotIn("Archived edition", html)

    def test_todays_brief_is_visually_distinguished_from_historical(self):
        first, _ = _publish_two()
        live = self._client().get("/brief/AAA").get_data(as_text=True)
        past = self._client().get(
            f"/brief/AAA/{first['meta']['snapshot_id']}").get_data(as_text=True)
        self.assertIn("Current", live)
        self.assertNotIn("Historical", live)
        self.assertNotIn("Archived edition", live)
        self.assertIn("Historical", past)
        self.assertIn("Archived edition", past)

    def test_unknown_edition_renders_declared_gone_state(self):
        _publish_two()
        for ticker, sid in (("AAA", "20990101T000000-9999"), ("NOPE", "nope")):
            resp = self._client().get(f"/brief/{ticker}/{sid}")
            self.assertEqual(resp.status_code, 200)
            html = resp.get_data(as_text=True)
            self.assertIn("no longer retained", html)
            self.assertIn(f'href="/brief/{ticker}"', html)
            self.assertNotIn("Archived edition", html)

    def test_ttl_expired_edition_renders_declared_gone_state(self):
        first, _ = _publish_two()
        old = (datetime.utcnow()
               - timedelta(hours=snapshot_store._TTL_HOURS + 1)).isoformat()
        with db.get_conn() as conn:
            conn.execute("UPDATE api_cache SET fetched_at = ? "
                         "WHERE provider = 'brief:AAA'", (old,))
        resp = self._client().get(f"/brief/AAA/{first['meta']['snapshot_id']}")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("no longer retained", resp.get_data(as_text=True))

    def test_historical_edition_renders_without_ai_garnish(self):
        first, _ = _publish_two()
        calls = []
        with patch.object(ai_module.providers, "ai_providers",
                          lambda: STUB_PROVIDERS), \
             patch.object(ai_module, "_openai_compatible_report",
                          side_effect=_stub_report(calls)):
            live = self._client().get("/brief/AAA")
            past = self._client().get(
                f"/brief/AAA/{first['meta']['snapshot_id']}")
        self.assertEqual(live.status_code, 200)
        self.assertEqual(past.status_code, 200)
        live_html = live.get_data(as_text=True)
        past_html = past.get_data(as_text=True)
        self.assertEqual(len(calls), 1)
        self.assertIn("brief-ai-garnish", live_html)
        self.assertIn("AI-generated", live_html)
        self.assertNotIn("brief-ai-garnish", past_html)
        self.assertNotIn("AI-generated", past_html)
        self.assertIn("+5.00%", past_html)
        self.assertIn("attaches only to the live brief", past_html)


if __name__ == "__main__":
    unittest.main()
