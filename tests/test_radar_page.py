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


def _populated_snap():
    """Snapshot fixture with one drift row; shared by structure tests."""
    return {"meta": {"snapshot_id": "S1", "published_at": "2026-10-04T10:00",
                     "source_as_of": {}, "universe_count": 2,
                     "warmed_count": 2, "scored_count": 2,
                     "excluded_count_by_reason": {},
                     "validation_status": "ok",
                     "metric_contract_version": 2,
                     "value_score_version": 1},
            "drift": {"rows": [{"symbol": "AAA", "print_date": "2024-09-06",
                                "sessions_since": 30, "sessions_to_20d": 0,
                                "surprise_pct": 20.0, "car": 0.05,
                                "car_significant": True, "surprise_nm": True,
                                "drift_to_date": 0.03,
                                "drift_to_date_sessions": 30}],
                      "upcoming": [], "excluded": []},
            "value": {"rows": [{"symbol": "AAA", "industry": "Software",
                                "value_score": 81.0, "fwd_pe": 28.0,
                                "_state": "insufficient-peers", "_nm": True,
                                "low_coverage": True}],
                      "excluded": []},
            "squeeze": {"rows": [{"symbol": "AAA", "short_pct_float": 0.22}],
                        "excluded": []}}


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

    def test_warming_progress_strip_renders(self):
        db.cache_set("radar", "warm_progress",
                     {"done": 214, "total": 500,
                      "updated": "2026-10-04T14:32"})
        with patch.object(app_module.radar, "current_snapshot",
                          return_value=None):
            resp = self._get("drift")
        html = resp.get_data(as_text=True)
        self.assertIn("214/500", html)
        self.assertIn("14:32", html)

    def test_unknown_tab_falls_back(self):
        self.assertEqual(self._get("nope").status_code, 200)

    def test_renders_populated_snapshot(self):
        with patch.object(app_module.radar, "current_snapshot",
                          return_value=_populated_snap()):
            resp = self._get("drift")
        html = resp.get_data(as_text=True)
        self.assertIn("AAA", html)
        self.assertIn("Published", html)

    def test_drawer_shell_and_row_hooks(self):
        with patch.object(app_module.radar, "current_snapshot",
                          return_value=_populated_snap()):
            resp = self._get("drift")
        html = resp.get_data(as_text=True)
        self.assertIn('id="radar-drawer"', html)
        self.assertIn('data-symbol="AAA"', html)
        self.assertIn("/api/radar/detail/", html)

    def test_column_presets_and_filters_present(self):
        with patch.object(app_module.radar, "current_snapshot",
                          return_value=_populated_snap()):
            html = self._get("drift").get_data(as_text=True)
        for marker in ('data-preset="scan"', 'data-preset="research"',
                       'data-preset="audit"', 'data-sort-key',
                       'id="radar-sector-filter"', 'id="radar-search"',
                       'data-col=', 'th data-col="symbol" class="sticky'):
            self.assertIn(marker, html)

    def test_squeeze_si_cell_carries_percent_scale(self):
        """SI % float is stored as a fraction but filtered in percent, so the
        cell must declare data-scale="100" for the range filter to honor."""
        with patch.object(app_module.radar, "current_snapshot",
                          return_value=_populated_snap()):
            html = self._get("squeeze").get_data(as_text=True)
        self.assertIn('data-col="short_pct_float" data-v="0.22" '
                      'data-scale="100"', html)

    def test_watchlist_screens_export_and_snapshot_meta_hooks(self):
        with patch.object(app_module.radar, "current_snapshot",
                          return_value=_populated_snap()):
            html = self._get("drift").get_data(as_text=True)
        for marker in ('data-star-toggle', 'id="radar-star-only"',
                       'id="radar-export"', 'id="radar-screen-name"',
                       'id="radar-screen-save"', 'id="radar-screen-select"',
                       'id="radar-screen-apply"', 'id="radar-screen-delete"',
                       'data-snapshot-id="S1"', 'data-published=',
                       'data-contract-version="2"'):
            self.assertIn(marker, html)

    def test_nm_state_and_coverage_badges_replace_bare_dashes(self):
        with patch.object(app_module.radar, "current_snapshot",
                          return_value=_populated_snap()):
            drift = self._get("drift").get_data(as_text=True)
            value = self._get("value").get_data(as_text=True)
        self.assertIn('data-badge="nm"', drift)
        self.assertIn('data-badge="state"', value)
        self.assertIn("insufficient-peers", value)
        self.assertIn('data-badge="coverage"', value)
        self.assertIn("low coverage", value)

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
