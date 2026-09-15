"""Unit tests for db.py schema additions and snapshot helpers."""
import os
import sys
import tempfile
import unittest

# Add parent directory to path to import db module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestGexSnapshots(unittest.TestCase):

    def setUp(self):
        # Isolated on-disk DB per test -- db.py reads DB_PATH at import time,
        # so DB_PATH must be set before the module is (re)imported.
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_gex.db")
        os.environ["DB_PATH"] = self.db_path
        import importlib
        import db as db_module
        importlib.reload(db_module)
        self.db = db_module
        self.db.init_db()

    def tearDown(self):
        os.environ.pop("DB_PATH", None)

    def test_insert_gex_snapshot_creates_row(self):
        self.db.insert_gex_snapshot("AAPL", "2026-09-14", 230.5, 235.0, 225.0, 231.2)
        with self.db.get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM gex_snapshots WHERE ticker = ? AND date = ?",
                ("AAPL", "2026-09-14"),
            ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["gamma_flip"], 230.5)
        self.assertEqual(row["call_wall"], 235.0)
        self.assertEqual(row["put_wall"], 225.0)
        self.assertEqual(row["spot"], 231.2)

    def test_insert_gex_snapshot_is_idempotent(self):
        """Calling twice for the same (ticker, date) must not raise or duplicate --
        _warm_options_cache() runs this multiple times per day."""
        self.db.insert_gex_snapshot("AAPL", "2026-09-14", 230.5, 235.0, 225.0, 231.2)
        self.db.insert_gex_snapshot("AAPL", "2026-09-14", 999.0, 999.0, 999.0, 999.0)
        with self.db.get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM gex_snapshots WHERE ticker = ? AND date = ?",
                ("AAPL", "2026-09-14"),
            ).fetchall()
        self.assertEqual(len(rows), 1)

    def test_insert_gex_snapshot_handles_none_values(self):
        """get_gex_profile can return None for individual levels; must not raise."""
        self.db.insert_gex_snapshot("MU", "2026-09-14", None, None, None, 95.0)
        with self.db.get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM gex_snapshots WHERE ticker = ?", ("MU",)
            ).fetchone()
        self.assertIsNotNone(row)
        self.assertIsNone(row["gamma_flip"])


if __name__ == "__main__":
    unittest.main()
