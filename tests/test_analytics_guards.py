"""P0: analytics failures must degrade, not 500."""
import os
import tempfile
import unittest
from unittest.mock import patch

import db

# app.py starts a background cache-warmer at import time; point it at a
# throwaway DB before the first import (same pattern as
# test_ai_summary_val_sheet.py). The module-level import also backs the
# `app_module` references used inside the test methods.
_fd, _DB_PATH = tempfile.mkstemp(suffix=".db")
os.close(_fd)
db.DB_PATH = _DB_PATH
db.init_db()

import app as app_module  # noqa: E402


class TestAnalyticsGuards(unittest.TestCase):

    def test_compute_analytics_returns_none_on_internal_error(self):
        import app as app_module
        with patch.object(app_module, "get_or_fetch_prices",
                          side_effect=RuntimeError("boom")):
            self.assertIsNone(app_module.compute_analytics("AAPL"))

    def test_analytics_page_renders_error_not_500(self):
        from app import app
        client = app.test_client()
        with patch.object(app_module, "get_or_fetch_prices",
                          side_effect=RuntimeError("boom")):
            resp = client.get("/analytics?ticker=AAPL")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("Could not retrieve data", resp.get_data(as_text=True))

    def test_analytics_api_returns_404_json_not_500(self):
        from app import app
        client = app.test_client()
        with patch.object(app_module, "get_or_fetch_prices",
                          side_effect=RuntimeError("boom")):
            resp = client.get("/api/analytics/AAPL")
        self.assertEqual(resp.status_code, 404)
        self.assertIn("error", resp.get_json())

    def test_analytics_page_survives_attachment_failure(self):
        import app as app_module
        from app import app
        # `stats` is set unconditionally by the real compute_analytics and is
        # required by the template; everything attached afterwards is guarded.
        minimal = {"current_price": 100.0, "charts": {}, "stats": {}}
        client = app.test_client()
        with patch.object(app_module, "compute_analytics", return_value=minimal), \
             patch.object(app_module, "get_fundamentals", side_effect=RuntimeError("boom")), \
             patch.object(app_module, "get_options_smile", side_effect=RuntimeError("boom")), \
             patch.object(app_module, "get_gex_profile", side_effect=RuntimeError("boom")), \
             patch.object(app_module, "decide") as mock_decide, \
             patch.object(app_module, "get_or_fetch_prices", return_value=None), \
             patch.object(app_module, "get_price_target_chart", side_effect=RuntimeError("boom")), \
             patch.object(app_module, "ml") as mock_ml:
            mock_decide.build_checklist.side_effect = RuntimeError("boom")
            mock_decide.TRADE_TYPES = ("long_stock",)
            mock_ml.predict.side_effect = RuntimeError("boom")
            resp = client.get("/analytics?ticker=AAPL")
        self.assertEqual(resp.status_code, 200)

    def test_compute_momentum_returns_empty_on_internal_error(self):
        import app as app_module
        with patch.object(app_module.db, "get_prices",
                          side_effect=RuntimeError("boom")):
            self.assertEqual(app_module.compute_momentum("AAPL"), {})

    def test_ai_summary_renders_error_not_500(self):
        from app import app
        client = app.test_client()
        with patch.object(app_module, "get_or_fetch_prices",
                          side_effect=RuntimeError("boom")):
            resp = client.get("/ai-summary?ticker=AAPL")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("Could not retrieve data", resp.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main()
