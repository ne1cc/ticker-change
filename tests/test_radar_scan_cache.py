"""Radar scan cache: builders run once per TTL, requests serve the payload."""
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import db

_fd, _DB_PATH = tempfile.mkstemp(suffix=".db")
os.close(_fd)
db.DB_PATH = _DB_PATH
db.init_db()

import radar  # noqa: E402

PAYLOAD = {"rows": [{"symbol": "X"}], "coverage": 1}


class TestGetScan(unittest.TestCase):
    def setUp(self):
        with db.get_conn() as conn:
            conn.execute("DELETE FROM api_cache WHERE provider = 'radar'")

    def test_miss_builds_and_stores(self):
        fn = MagicMock(return_value=dict(PAYLOAD))
        first = radar.get_scan("drift", fn)
        second = radar.get_scan("drift", fn)
        self.assertEqual(fn.call_count, 1)  # second read served from api_cache
        self.assertEqual(first, second)
        self.assertEqual(first["coverage"], 1)

    def test_hit_serves_cache_without_building(self):
        db.cache_set("radar", "scan:value", {"rows": [{"symbol": "CACHED"}], "coverage": 9})
        fn = MagicMock()
        out = radar.get_scan("value", fn)
        self.assertEqual(out["coverage"], 9)
        fn.assert_not_called()

    def test_tabs_use_distinct_keys(self):
        a = MagicMock(return_value={"rows": [], "coverage": 1})
        b = MagicMock(return_value={"rows": [], "coverage": 2})
        radar.get_scan("drift", a)
        radar.get_scan("squeeze", b)
        self.assertEqual(a.call_count, 1)
        self.assertEqual(b.call_count, 1)

    def test_builder_failure_not_cached(self):
        fn = MagicMock(side_effect=RuntimeError("boom"))
        with self.assertRaises(RuntimeError):
            radar.get_scan("drift", fn)
        fn2 = MagicMock(return_value=dict(PAYLOAD))
        radar.get_scan("drift", fn2)  # cache untouched by the failure
        self.assertEqual(fn2.call_count, 1)


class TestRefreshScans(unittest.TestCase):
    def setUp(self):
        with db.get_conn() as conn:
            conn.execute("DELETE FROM api_cache WHERE provider = 'radar'")

    def test_refresh_writes_all_three(self):
        with patch.object(radar, "build_drift_scan", return_value={"rows": [], "coverage": 3}), \
             patch.object(radar, "build_value_scan", return_value={"rows": [], "coverage": 4}), \
             patch.object(radar, "build_squeeze_scan", return_value={"rows": [], "coverage": 5}):
            radar.refresh_scans()
        self.assertEqual(db.cache_get("radar", "scan:drift", 1)["coverage"], 3)
        self.assertEqual(db.cache_get("radar", "scan:value", 1)["coverage"], 4)
        self.assertEqual(db.cache_get("radar", "scan:squeeze", 1)["coverage"], 5)

    def test_refresh_never_raises(self):
        with patch.object(radar, "build_drift_scan", side_effect=RuntimeError("boom")), \
             patch.object(radar, "build_value_scan", return_value={"rows": [], "coverage": 0}), \
             patch.object(radar, "build_squeeze_scan", return_value={"rows": [], "coverage": 0}):
            radar.refresh_scans()  # must not raise; other scans still stored
        self.assertIsNotNone(db.cache_get("radar", "scan:value", 1))


if __name__ == "__main__":
    unittest.main()
