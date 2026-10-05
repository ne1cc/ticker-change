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
        self.assertIn("Published", html)

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

    def test_breadth_strip_renders_with_denominator(self):
        breadth = {"advancers": 210, "decliners": 190, "unchanged": 14,
                   "covered": 414, "as_of": "2026-10-04T15:00",
                   "movers_up": [{"symbol": "AAA", "change_pct": 4.2}],
                   "movers_down": [{"symbol": "BBB", "change_pct": -3.1}]}
        with patch.object(app_module.heatmap, "get_heatmap_payload",
                          return_value={"tiles": []}):
            with patch.object(app_module, "_breadth_from_payload",
                              return_value=breadth):
                resp = self._get("drift")
        html = resp.get_data(as_text=True)
        self.assertIn("210", html)
        self.assertIn("of 414 covered names", html)
        self.assertIn("AAA", html)

    def test_breadth_hidden_when_heatmap_warming(self):
        with patch.object(app_module.heatmap, "get_heatmap_payload",
                          return_value=None):
            resp = self._get("drift")
        self.assertNotIn("covered names", resp.get_data(as_text=True))
