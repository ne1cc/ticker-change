"""/positioning peers panel: attached, guarded, rendered; JSON twin strips chart."""
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

FAKE = {
    "focus": {"symbol": "FOCUS", "forward_pe": 10.0, "forward_pe_pct": 100.0,
              "valuation_pct": 90.0, "mom_12_1": 0.2, "mom_pct": 100.0,
              "value_score": 95.0, "quadrant": "Cheap & Strengthening"},
    "peers": [{"symbol": "PEER1", "forward_pe": 30.0, "forward_pe_pct": 0.0,
               "valuation_pct": 10.0, "mom_12_1": -0.1, "mom_pct": 0.0,
               "value_score": 5.0, "quadrant": "Rich & Weakening"}],
    "n_compared": 2,
    "metrics": [{"key": "forward_pe", "label": "Forward P/E", "invert": True}],
}


class TestPeersEmbed(unittest.TestCase):
    def test_positioning_page_renders_panel(self):
        client = app_module.app.test_client()
        with patch.object(app_module.peers, "build_peer_comparison", return_value=FAKE), \
             patch.object(app_module.providers, "sec_cik_for_ticker", return_value=None):
            resp = client.get("/positioning?ticker=FOCUS")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("Peer Comparison", resp.get_data(as_text=True))

    def test_page_renders_without_peers(self):
        client = app_module.app.test_client()
        with patch.object(app_module.peers, "build_peer_comparison",
                          side_effect=RuntimeError("boom")), \
             patch.object(app_module.providers, "sec_cik_for_ticker", return_value=None):
            resp = client.get("/positioning?ticker=FOCUS")
        self.assertEqual(resp.status_code, 200)

    def test_json_twin_has_no_chart_html(self):
        client = app_module.app.test_client()
        with patch.object(app_module.peers, "build_peer_comparison", return_value=FAKE), \
             patch.object(app_module.providers, "sec_cik_for_ticker", return_value=None):
            resp = client.get("/api/positioning/FOCUS")
        self.assertEqual(resp.status_code, 200)
        self.assertNotIn("plotly", resp.get_data(as_text=True).lower())
