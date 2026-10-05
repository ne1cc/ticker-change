"""cache_get_with_age returns the payload plus its fetch timestamp."""
import os, tempfile, unittest
from datetime import datetime, timedelta

import db

_fd, _DB_PATH = tempfile.mkstemp(suffix=".db")
os.close(_fd)
db.DB_PATH = _DB_PATH
db.init_db()


class _PinnedDB(unittest.TestCase):
    """Pin this module's temp DB around each test: other test modules in the
    suite reassign db.DB_PATH at import time and their cache writes would
    otherwise leak into these assertions."""

    def setUp(self):
        self._orig_db = db.DB_PATH
        db.DB_PATH = _DB_PATH

    def tearDown(self):
        db.DB_PATH = self._orig_db


class TestCacheGetWithAge(_PinnedDB):
    def test_returns_payload_and_fetched_at(self):
        db.cache_set("t", "k", {"a": 1})
        payload, fetched = db.cache_get_with_age("t", "k", 24)
        self.assertEqual(payload, {"a": 1})
        self.assertIsNotNone(fetched)
        datetime.fromisoformat(fetched)  # parses

    def test_expired_reports_miss(self):
        db.cache_set("t", "old", [1, 2])
        with db.get_conn() as conn:
            conn.execute(
                "UPDATE api_cache SET fetched_at = ? WHERE provider='t' AND key='old'",
                ((datetime.utcnow() - timedelta(hours=48)).isoformat(),))
        self.assertEqual(db.cache_get_with_age("t", "old", 24), (None, None))

    def test_miss_is_none_none(self):
        self.assertEqual(db.cache_get_with_age("t", "nope", 24), (None, None))


if __name__ == "__main__":
    unittest.main()
