"""P0: internal exception text must not leak to API clients."""
import unittest
from unittest.mock import patch


class TestSecFilingsError(unittest.TestCase):

    def test_filings_failure_returns_generic_error(self):
        from app import app
        client = app.test_client()
        with patch("app.providers.sec_recent_filings",
                   side_effect=RuntimeError("boom internal detail")):
            resp = client.get("/api/raw-sec-filings/AAPL")
        self.assertEqual(resp.status_code, 502)
        body = resp.get_data(as_text=True)
        self.assertNotIn("boom internal detail", body)
        self.assertIn("error", body)


if __name__ == "__main__":
    unittest.main()
