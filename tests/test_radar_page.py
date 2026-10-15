"""/radar: three tabs render; cold cache renders warming skeleton; tunables echoed."""
import os
import tempfile
import unittest
from unittest.mock import patch

import db

_fd, _DB_PATH = tempfile.mkstemp(suffix=".db")
os.close(_fd)
db.DB_PATH = _DB_PATH
db.init_db()

import app as app_module  # noqa: E402


class TestRadarPage(unittest.TestCase):
    def setUp(self):
        # The route serves scan payloads from api_cache (1h TTL); other test
        # modules share this temp DB file, so start each test cache-cold.
        with db.get_conn() as conn:
            conn.execute("DELETE FROM api_cache WHERE provider = 'radar'")

    def _get(self, tab="drift"):
        return app_module.app.test_client().get(f"/radar?tab={tab}")

    def test_three_tabs_render_200(self):
        for tab in ("drift", "value", "squeeze"):
            resp = self._get(tab)
            self.assertEqual(resp.status_code, 200)
            html = resp.get_data(as_text=True)
            self.assertIn("Radar", html)

    def test_cold_cache_shows_warming(self):
        resp = self._get("drift")
        self.assertIn("warming", resp.get_data(as_text=True).lower())

    def test_unknown_tab_falls_back(self):
        self.assertEqual(self._get("nope").status_code, 200)

    def test_renders_populated_snapshot(self):
        snap = {"meta": {"snapshot_id": "S1", "published_at": "2026-10-04T10:00",
                         "source_as_of": {}, "universe_count": 2,
                         "warmed_count": 2, "scored_count": 2,
                         "excluded_count_by_reason": {},
                         "validation_status": "ok",
                         "metric_contract_version": 2,
                         "value_score_version": 1},
                "drift": {"rows": [{"symbol": "AAA", "print_date": "2024-09-06",
                                    "sessions_since": 30, "sessions_to_20d": 0,
                                    "surprise_pct": 20.0, "car": 0.05,
                                    "drift_to_date": 0.03,
                                    "drift_to_date_sessions": 30}],
                          "upcoming": [], "excluded": []},
                "value": {"rows": [], "excluded": []},
                "squeeze": {"rows": [], "excluded": []}}
        with patch.object(app_module.radar, "current_snapshot", return_value=snap):
            resp = self._get("drift")
        html = resp.get_data(as_text=True)
        self.assertIn("AAA", html)
        self.assertIn("Updated Oct 4, 3:00 AM PDT", html)

    def test_tunables_echoed(self):
        payload = {"meta": {"warmed_count": 2},
                   "drift": {"rows": [{"symbol": "X", "car_30": 0.01,
                                       "surprise_pct": 5.0,
                                       "sessions_since": 3,
                                       "sessions_to_20d": 27}],
                             "upcoming": [], "excluded": []},
                   "value": {"rows": [], "excluded": []},
                   "squeeze": {"rows": [], "excluded": []}}
        with patch.object(app_module.radar, "current_snapshot",
                          return_value=payload):
            resp = self._get("drift")
        html = resp.get_data(as_text=True)
        self.assertIn("Radar", html)
        self.assertIn("+1.0%", html)
        self.assertNotIn("%.1f", html)

    def _snap(self, **tabs):
        snap = {"meta": {"warmed_count": 3, "universe_count": 5,
                         "published_at": "2026-10-15T03:12:00"},
                "drift": {"rows": [], "upcoming": [], "excluded": []},
                "value": {"rows": [], "excluded": []},
                "squeeze": {"rows": [], "excluded": []}}
        snap.update(tabs)
        return snap

    def _render(self, snap, tab):
        with patch.object(app_module.radar, "current_snapshot", return_value=snap):
            return self._get(tab).get_data(as_text=True)

    def test_header_reports_coverage_in_pacific_time(self):
        html = self._render(self._snap(), "drift")
        self.assertIn("Scanning 5 stocks", html)
        self.assertIn("Oct 14, 8:12 PM PDT", html)

    def test_drift_splits_open_and_closed_windows_and_shows_upcoming(self):
        blank = {"print_date": "2026-10-08", "surprise_pct": None,
                 "car_30": None, "car_significant": False,
                 "drift_to_date": None, "drift_to_date_sessions": None}
        drift = {"rows": [dict(blank, symbol="OPEN1", sessions_since=4,
                               sessions_to_20d=16),
                          dict(blank, symbol="DONE1", sessions_since=40,
                               sessions_to_20d=0)],
                 "upcoming": [{"symbol": "SOON1", "date": "2026-10-17",
                               "days_until": 3, "eps_estimate": 1.2}],
                 "excluded": [{"symbol": "Z", "reason": "no-prices"}]}
        html = self._render(self._snap(drift=drift), "drift")
        self.assertIn("SOON1", html)
        self.assertLess(html.index("OPEN1"), html.index("Window closed"))
        self.assertLess(html.index("Window closed"), html.index("DONE1"))
        self.assertIn("1 no price history", html)
        # 5 in universe - 2 rows - 1 excluded = 2 never had earnings data
        self.assertIn("2 no earnings data", html)

    def test_value_missing_score_renders_dash_not_none(self):
        value = {"rows": [{"symbol": "V1", "industry": "Widgets",
                           "value_score": None, "mom_12_1": 0.1,
                           "fwd_pe": 20.0}], "excluded": []}
        html = self._render(self._snap(value=value), "value")
        self.assertIn("V1", html)
        self.assertNotIn(">None<", html.replace(" ", "").replace("\n", ""))

    def test_value_tab_has_no_peer_cap_input(self):
        html = self._render(self._snap(), "value")
        self.assertNotIn("tunable-peer-cap", html)

    def test_empty_tab_explains_exclusions(self):
        squeeze = {"rows": [], "excluded": [
            {"symbol": "A", "reason": "below-floor"},
            {"symbol": "B", "reason": "below-floor"}]}
        html = self._render(self._snap(squeeze=squeeze), "squeeze")
        self.assertIn("2 below liquidity floor", html)
