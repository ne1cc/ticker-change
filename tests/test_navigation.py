import os
import tempfile
import unittest
import db
from app import app


class TestNavigation(unittest.TestCase):
    def setUp(self):
        self.orig_db_path = db.DB_PATH
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        db.DB_PATH = self.temp_db.name
        db.init_db()
        self.client = app.test_client()
        self.client.testing = True

    def tearDown(self):
        db.DB_PATH = self.orig_db_path
        try:
            os.remove(self.temp_db.name)
        except OSError:
            pass

    def test_all_core_routes_return_ok(self):
        """Verify that all core application views return HTTP 200 or valid redirect."""
        routes = [
            "/",
            "/stock",
            "/analytics",
            "/positioning",
            "/live",
            "/momentum",
            "/strategies",
            "/ai-summary",
            "/glossary",
            "/api/docs",
        ]
        for route in routes:
            resp = self.client.get(route)
            self.assertIn(
                resp.status_code,
                [200, 302],
                f"Route {route} failed with status {resp.status_code}",
            )

        # /settings is gated by HTTP Basic Auth (returns 503 when unconfigured, 401 unauthorized, or 200)
        settings_resp = self.client.get("/settings")
        self.assertIn(
            settings_resp.status_code,
            [200, 401, 503],
            f"Route /settings returned unexpected status {settings_resp.status_code}",
        )

    def test_strategies_handles_ticker_parameter(self):
        """Verify that /strategies and /momentum accept ?ticker=AAPL without error."""
        for path in ["/momentum?ticker=AAPL", "/strategies?ticker=AAPL"]:
            resp = self.client.get(path)
            self.assertEqual(
                resp.status_code,
                200,
                f"Route {path} failed with status {resp.status_code}",
            )

    def test_desktop_navigation_links_present(self):
        """Verify that desktop navigation contains all primary and utility links."""
        resp = self.client.get("/glossary")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)

        expected_ids = [
            'id="nav-stock"',
            'id="nav-analytics"',
            'id="nav-positioning"',
            'id="nav-live"',
            'id="nav-momentum"',
            'id="nav-ai-summary"',
            'id="nav-glossary"',
            'id="nav-apidocs"',
            'id="nav-settings"',
        ]
        for link_id in expected_ids:
            self.assertIn(
                link_id,
                html,
                f"Missing desktop navigation element: {link_id}",
            )

    def test_ticker_sub_navbar_links_present(self):
        """Verify that the ticker sub-navbar has links for all 6 ticker views."""
        resp = self.client.get("/glossary")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)

        expected_sub_ids = [
            'id="sub-link-stock"',
            'id="sub-link-analytics"',
            'id="sub-link-positioning"',
            'id="sub-link-live"',
            'id="sub-link-momentum"',
            'id="sub-link-ai-summary"',
        ]
        for sub_id in expected_sub_ids:
            self.assertIn(
                sub_id,
                html,
                f"Missing ticker sub-navbar element: {sub_id}",
            )

    def test_mobile_navigation_links_present(self):
        """Verify that mobile navigation drawer contains all core view links."""
        resp = self.client.get("/glossary")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)

        expected_mobile_ids = [
            'id="mobile-nav-stock"',
            'id="mobile-nav-analytics"',
            'id="mobile-nav-positioning"',
            'id="mobile-nav-live"',
            'id="mobile-nav-momentum"',
            'id="mobile-nav-ai-summary"',
            'id="mobile-nav-glossary"',
            'id="mobile-nav-apidocs"',
            'id="mobile-nav-settings"',
        ]
        for m_id in expected_mobile_ids:
            self.assertIn(
                m_id,
                html,
                f"Missing mobile navigation element: {m_id}",
            )
