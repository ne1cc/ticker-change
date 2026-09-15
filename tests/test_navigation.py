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
            "/options",
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
        """Verify that /strategies and /momentum accept ?ticker=AAPL without error and retain query params."""
        for path in ["/momentum?ticker=AAPL", "/strategies?ticker=AAPL"]:
            resp = self.client.get(path)
            self.assertEqual(
                resp.status_code,
                200,
                f"Route {path} failed with status {resp.status_code}",
            )
            self.assertEqual(
                resp.request.args.get("ticker"),
                "AAPL",
                f"Query param ticker=AAPL not retained in {path}",
            )
            html = resp.get_data(as_text=True)
            self.assertIn(
                'id="nav-strategies"',
                html,
                f"Pillar nav-strategies not found in response for {path}",
            )

    def test_top_nav_search_bar_present_to_the_left_of_glossary(self):
        """Verify that a visible searchbar is present in top right navigation to the left of glossary."""
        resp = self.client.get("/glossary")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)

        self.assertIn('id="top-nav-search-form"', html)
        self.assertIn('id="top-nav-search-input"', html)

        pos_search = html.find('id="top-nav-search-form"')
        pos_glossary = html.find('id="nav-glossary"')
        self.assertTrue(pos_search < pos_glossary, "Top nav searchbar should be to the left of glossary")

    def test_desktop_navigation_links_present(self):
        """Verify that desktop navigation contains all 4 pillar links, utilities, and theme toggle, with no legacy IDs."""
        resp = self.client.get("/glossary")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)

        expected_ids = [
            'id="nav-markets"',
            'id="nav-options"',
            'id="nav-analytics"',
            'id="nav-strategies"',
            'id="nav-glossary"',
            'id="nav-apidocs"',
            'id="nav-settings"',
            'id="theme-toggle"',
        ]
        for link_id in expected_ids:
            self.assertIn(
                link_id,
                html,
                f"Missing desktop navigation element: {link_id}",
            )

        deprecated_legacy_ids = [
            'id="nav-stock"',
            'id="nav-live"',
            'id="nav-momentum"',
        ]
        for legacy_id in deprecated_legacy_ids:
            self.assertNotIn(
                legacy_id,
                html,
                f"Deprecated legacy topbar navigation element still present: {legacy_id}",
            )

    def test_ticker_sub_navbar_links_present(self):
        """Verify that the ticker sub-navbar structure contains all child tool link anchors and search/ticker badges."""
        resp = self.client.get("/stock?ticker=AAPL")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)

        expected_sub_ids = [
            'id="sub-nav-ticker"',
            'id="sub-nav-search-input"',
            'id="sub-link-stock"',
            'id="sub-link-live"',
            'id="sub-link-options"',
            'id="sub-link-analytics"',
            'id="sub-link-positioning"',
            'id="sub-link-strategies"',
            'id="sub-link-ai-summary"',
        ]
        for sub_id in expected_sub_ids:
            self.assertIn(
                sub_id,
                html,
                f"Missing ticker sub-navbar element: {sub_id}",
            )

    def test_mobile_navigation_links_present(self):
        """Verify that mobile navigation drawer contains all core view links organized by pillar."""
        resp = self.client.get("/glossary")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)

        expected_mobile_ids = [
            'id="mobile-nav-stock"',
            'id="mobile-nav-live"',
            'id="mobile-nav-options"',
            'id="mobile-nav-analytics"',
            'id="mobile-nav-positioning"',
            'id="mobile-nav-strategies"',
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

    def test_analytics_view_structure_and_ticker_preservation(self):
        """Verify that /analytics?ticker=AAPL renders proper structure, retains ticker, and includes nav elements."""
        resp = self.client.get("/analytics?ticker=AAPL")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)

        self.assertIn('id="nav-analytics"', html, "Missing id='nav-analytics' in /analytics response")
        self.assertIn('id="sub-link-analytics"', html, "Missing sub-link-analytics in /analytics response")
        self.assertIn('id="sub-link-positioning"', html, "Missing sub-link-positioning in /analytics response")
        self.assertIn('id="sub-nav-ticker"', html, "Missing sub-nav-ticker in /analytics response")
        self.assertIn('id="sub-nav-search-input"', html, "Missing sub-nav-search-input in /analytics response")
        self.assertIn("AAPL", html, "Ticker AAPL not preserved in /analytics response")

    def test_stock_view_renders_markets_sub_links(self):
        """Verify that visiting /stock?ticker=AAPL renders proper structure including sub-link-stock and sub-link-live."""
        resp = self.client.get("/stock?ticker=AAPL")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)

        self.assertIn('id="nav-markets"', html, "Missing id='nav-markets' in /stock response")
        self.assertIn('id="sub-link-stock"', html, "Missing sub-link-stock in /stock response")
        self.assertIn('id="sub-link-live"', html, "Missing sub-link-live in /stock response")
        self.assertIn('id="sub-nav-ticker"', html, "Missing sub-nav-ticker in /stock response")
        self.assertIn('id="sub-nav-search-input"', html, "Missing sub-nav-search-input in /stock response")
        self.assertIn("AAPL", html, "Ticker AAPL not rendered in /stock response")

    def test_excel_mode_stylesheet_rules(self):
        """Verify that excel-mode.css strictly hides .hidden elements in #ticker-sub-nav and styles nav links."""
        resp = self.client.get("/static/excel-mode.css")
        self.assertEqual(resp.status_code, 200)
        css = resp.get_data(as_text=True)

        self.assertIn(
            "html.excel #ticker-sub-nav .hidden",
            css,
            "Missing strict .hidden rule for #ticker-sub-nav in excel-mode.css",
        )
        self.assertIn(
            "display: none !important;",
            css,
            "Missing display: none !important for hidden elements in excel-mode.css",
        )
        self.assertIn(
            'html.excel nav a[id^="nav-"]',
            css,
            "Missing topbar nav styling in excel-mode.css",
        )
        self.assertIn(
            'html.excel nav a[id^="nav-"].border-amber-500',
            css,
            "Missing active topbar pillar styling in excel-mode.css",
        )

    def test_options_default_ticker_does_not_leak_database_symbols(self):
        """Verify that /options without ?ticker= defaults to SPY and does not leak tickers searched by other users."""
        with db.get_conn() as conn:
            conn.execute(
                "INSERT INTO daily_prices (symbol, date, open, high, low, close, volume) "
                "VALUES ('TSLA', '2026-09-01', 200.0, 210.0, 195.0, 205.0, 1000000)"
            )
        resp = self.client.get('/options')
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertNotIn("TSLA Options Terminal", html, "Leaked previously searched ticker TSLA on fresh /options load")
        self.assertIn("SPY", html)

    def test_brand_link_and_search_input_cleanliness(self):
        """Verify that brand-link points to / and search input has autocomplete=off."""
        resp = self.client.get('/')
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertIn('id="brand-link" href="/"', html)
        self.assertIn('id="sub-nav-search-input"', html)
        self.assertIn('autocomplete="off"', html)



