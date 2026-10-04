"""Radar tunables: clamp, default, persistence, route auth posture."""
import os
import tempfile
import unittest
from unittest.mock import patch

import db

_fd, _DB_PATH = tempfile.mkstemp(suffix=".db")
os.close(_fd)
db.DB_PATH = _DB_PATH
db.init_db()

import radar  # noqa: E402
import app as app_module  # noqa: E402


class TestTunables(unittest.TestCase):
    def setUp(self):
        with db.get_conn() as conn:
            conn.execute("DELETE FROM app_settings WHERE key LIKE 'radar_%'")

    def test_defaults_when_unset(self):
        self.assertEqual(radar.get_tunable("radar_peer_cap"), 8)
        self.assertEqual(radar.get_tunable("radar_liquidity_floor_usd"), 5_000_000.0)

    def test_unparseable_falls_back(self):
        db.set_setting("radar_peer_cap", "banana")
        self.assertEqual(radar.get_tunable("radar_peer_cap"), 8)

    def test_clamped(self):
        radar.set_tunable("radar_peer_cap", 999)
        self.assertEqual(radar.get_tunable("radar_peer_cap"), 20)
        radar.set_tunable("radar_liquidity_floor_usd", 5)
        self.assertEqual(radar.get_tunable("radar_liquidity_floor_usd"), 100_000.0)

    def test_unknown_key_raises(self):
        with self.assertRaises(ValueError):
            radar.set_tunable("not_a_key", 1)


class TestSettingsRoute(unittest.TestCase):
    def test_post_persists_and_echoes(self):
        client = app_module.app.test_client()
        resp = client.post("/api/radar/settings", json={"radar_peer_cap": 12})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["radar_peer_cap"], 12)
        self.assertEqual(radar.get_tunable("radar_peer_cap"), 12)

    def test_token_enforced_only_when_configured(self):
        client = app_module.app.test_client()
        with patch.dict(os.environ, {"WARM_CACHE_TOKEN": "secret"}):
            self.assertEqual(client.post(
                "/api/radar/settings", json={"radar_peer_cap": 5}).status_code, 401)
            self.assertEqual(client.post(
                "/api/radar/settings", json={"radar_peer_cap": 5},
                headers={"Authorization": "Bearer secret"}).status_code, 200)

    def test_unknown_key_400(self):
        client = app_module.app.test_client()
        self.assertEqual(client.post(
            "/api/radar/settings", json={"nope": 1}).status_code, 400)
