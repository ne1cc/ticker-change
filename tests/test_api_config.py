"""P0: /api/config must never disclose raw Finnhub key material."""
import unittest
from unittest.mock import patch


class TestApiConfig(unittest.TestCase):

    def test_config_response_contains_no_key_material(self):
        from app import app
        client = app.test_client()
        with patch("app.providers.active_finnhub_key", return_value="super-secret-key"):
            resp = client.get("/api/config")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json(), {"has_finnhub": True})
        self.assertNotIn("super-secret-key", resp.get_data(as_text=True))

    def test_config_response_when_unconfigured(self):
        from app import app
        client = app.test_client()
        with patch("app.providers.active_finnhub_key", return_value=None):
            resp = client.get("/api/config")
        self.assertEqual(resp.get_json(), {"has_finnhub": False})


if __name__ == "__main__":
    unittest.main()
