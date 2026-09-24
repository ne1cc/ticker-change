"""P0: an unknown spot must surface as unavailable, never a fabricated $100."""
import unittest
from unittest.mock import patch

import app as app_module


class TestOptionsSpotGuard(unittest.TestCase):

    def test_terminal_api_503_when_no_spot(self):
        client = app_module.app.test_client()
        with patch.object(app_module, "get_current_price_yfinance", return_value=None), \
             patch.object(app_module, "_spot_price", return_value=None):
            resp = client.get("/api/options-terminal/SPY")
        self.assertEqual(resp.status_code, 503)
        self.assertIn("error", resp.get_json())

    def test_options_page_renders_unavailable_state(self):
        client = app_module.app.test_client()
        with patch.object(app_module, "get_current_price_yfinance", return_value=None), \
             patch.object(app_module, "_spot_price", return_value=None):
            resp = client.get("/options?ticker=SPY")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_data(as_text=True)
        self.assertIn("Could not determine a current price", body)


if __name__ == "__main__":
    unittest.main()
