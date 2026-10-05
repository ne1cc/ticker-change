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

    def test_tunables_echoed(self):
        payload = {"meta": {"warmed_count": 2},
                   "rows": [{"symbol": "X", "car": 0.01, "surprise_pct": 5.0,
                             "days_since": 3, "remaining": 27}], "coverage": 1}
        with patch.object(app_module.radar, "current_snapshot",
                          return_value=payload):
            resp = self._get("drift")
        html = resp.get_data(as_text=True)
        self.assertIn("Radar", html)
        self.assertIn("+1.0%", html)
        self.assertNotIn("%.1f", html)
