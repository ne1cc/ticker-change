"""Manifest + install meta tags: installable shell, no service worker (by decision)."""
import json
import os
import tempfile
import unittest

import db


class TestPwaManifest(unittest.TestCase):
    def setUp(self):
        self._orig = db.DB_PATH
        fd, db.DB_PATH = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        db.init_db()
        from app import app
        self.client = app.test_client()

    def tearDown(self):
        os.remove(db.DB_PATH)
        db.DB_PATH = self._orig

    def test_manifest_served_as_json_with_required_fields(self):
        r = self.client.get("/static/manifest.webmanifest")
        self.assertEqual(r.status_code, 200)
        self.assertIn("json", r.content_type)
        m = json.loads(r.data)
        for key in ("name", "short_name", "start_url", "display", "background_color", "theme_color"):
            self.assertIn(key, m)
        self.assertEqual(m["display"], "standalone")
        sizes = {i["sizes"] for i in m["icons"]}
        self.assertTrue({"192x192", "512x512"} <= sizes)
        for icon in m["icons"]:
            self.assertEqual(self.client.get(icon["src"]).status_code, 200, icon["src"])

    def test_base_links_manifest_and_icons(self):
        html = self.client.get("/glossary").get_data(as_text=True)
        self.assertRegex(html, r'<link rel="manifest" href="/static/manifest.webmanifest"')
        self.assertIn('name="theme-color"', html)
        self.assertRegex(html, r'rel="apple-touch-icon" href="/static/icons/apple-touch-icon.png"')
        self.assertNotIn("serviceWorker", html)
