"""Tests for db.get_prices_batch() — the batched, N+1-avoiding read used by
the strategies page.

Verifies:
1. Per-symbol output is byte-for-byte identical to db.get_prices() output.
2. Symbols with no rows are absent from the returned dict (not None).
3. Empty input returns an empty dict.
4. Exactly one DB connection is opened regardless of symbol count, including
   past the 900-symbol chunk boundary.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

import pandas as pd
import pandas.testing as pdt

import db


class TestGetPricesBatch(unittest.TestCase):

    def setUp(self):
        self._old_db_path = db.DB_PATH
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        db.DB_PATH = self.temp_db.name
        db.init_db()

        self._seed("AAA", [
            ("2024-01-02", 10.0, 11.0, 9.5, 10.5, 1000),
            ("2024-01-03", 10.5, 11.5, 10.0, 11.0, 1500),
            ("2024-01-04", 11.0, 12.0, 10.5, 11.5, 2000),
        ])
        self._seed("BBB", [
            ("2024-01-02", 50.0, 51.0, 49.5, 50.5, 3000),
            ("2024-01-03", 50.5, 52.0, 50.0, 51.5, 3500),
        ])
        # "ZZZ" is intentionally left with no rows in daily_prices.

    def tearDown(self):
        db.DB_PATH = self._old_db_path
        try:
            os.remove(self.temp_db.name)
        except OSError:
            pass

    @staticmethod
    def _seed(symbol: str, rows: list[tuple]):
        with db.get_conn() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO daily_prices "
                "(symbol, date, open, high, low, close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [(symbol, *row) for row in rows],
            )

    def test_batch_output_matches_get_prices_per_symbol(self):
        """Batched per-symbol frames must be indistinguishable from get_prices()."""
        batch = db.get_prices_batch(["AAA", "BBB"])

        self.assertEqual(set(batch.keys()), {"AAA", "BBB"})
        pdt.assert_frame_equal(batch["AAA"], db.get_prices("AAA"))
        pdt.assert_frame_equal(batch["BBB"], db.get_prices("BBB"))

    def test_mixed_case_symbols_match_uppercased_get_prices(self):
        """Casing must not distinguish the batch path from get_prices()."""
        batch = db.get_prices_batch(["aaa", "Bbb"])

        self.assertEqual(set(batch.keys()), {"AAA", "BBB"})
        pdt.assert_frame_equal(batch["AAA"], db.get_prices("AAA"))
        pdt.assert_frame_equal(batch["BBB"], db.get_prices("bbb"))

    def test_symbol_with_no_rows_is_absent(self):
        """A symbol with no daily_prices rows must be missing, not None."""
        batch = db.get_prices_batch(["AAA", "ZZZ"])

        self.assertIn("AAA", batch)
        self.assertNotIn("ZZZ", batch)
        self.assertEqual(len(batch), 1)

    def test_empty_input_returns_empty_dict(self):
        self.assertEqual(db.get_prices_batch([]), {})

    def test_single_connection_regardless_of_symbol_count(self):
        """One get_conn() call for a small batch..."""
        with mock.patch.object(db, "get_conn", wraps=db.get_conn) as spy_conn:
            db.get_prices_batch(["AAA", "BBB", "ZZZ"])
            self.assertEqual(spy_conn.call_count, 1)

    def test_single_connection_past_chunk_boundary(self):
        """...and still exactly one get_conn() call past the 900-symbol chunk
        boundary, with results from both chunks correctly merged."""
        # AAA lands in the first 900-symbol chunk, BBB lands past it.
        padding = [f"PAD{i}" for i in range(1198)]
        symbols = ["AAA"] + padding + ["BBB"]
        self.assertGreater(len(symbols), 900)

        with mock.patch.object(db, "get_conn", wraps=db.get_conn) as spy_conn:
            batch = db.get_prices_batch(symbols)
            self.assertEqual(spy_conn.call_count, 1)

        self.assertEqual(set(batch.keys()), {"AAA", "BBB"})
        pdt.assert_frame_equal(batch["AAA"], db.get_prices("AAA"))
        pdt.assert_frame_equal(batch["BBB"], db.get_prices("BBB"))


if __name__ == "__main__":
    unittest.main()
