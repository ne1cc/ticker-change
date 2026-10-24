"""Tests for signals_options_si.py (issue #138)."""
import os
import tempfile
import unittest
from unittest.mock import patch

import db
import signals_options_si


class TestSignalsOptionsSI(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp_dir.name, "test_options_si.db")
        self.patch_db = patch.object(db, "DB_PATH", self.db_path)
        self.patch_db.start()
        db.init_db()

    def tearDown(self):
        self.patch_db.stop()
        self.tmp_dir.cleanup()

    def test_explicit_config_threshold(self):
        """Minimum history threshold is explicitly configured."""
        self.assertEqual(signals_options_si.MIN_HISTORY_OBS, 63)
        self.assertIn("short_interest_ratio", signals_options_si.OPTIONS_SI_SIGNAL_DEFINITIONS)
        self.assertEqual(
            signals_options_si.OPTIONS_SI_SIGNAL_DEFINITIONS["short_interest_ratio"]["min_obs"],
            63,
        )

    def test_si_snapshot_writer_idempotency(self):
        """Short interest snapshot writer stores rows idempotently without duplication."""
        mock_si = {
            "short_percent_of_float": 0.045,
            "short_ratio": 2.8,
            "shares_short": 15000000,
            "shares_float": 330000000,
        }
        with patch.object(signals_options_si.providers, "get_short_interest", return_value=mock_si):
            # First write
            res1 = signals_options_si.record_daily_short_interest("NVDA", date_str="2024-05-01")
            self.assertTrue(res1)

            # Second write on identical date (idempotent duplicate)
            res2 = signals_options_si.record_daily_short_interest("NVDA", date_str="2024-05-01")
            self.assertTrue(res2)

            history = db.get_short_interest_history("NVDA")
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["ticker"], "NVDA")
            self.assertEqual(history[0]["date"], "2024-05-01")
            self.assertAlmostEqual(history[0]["short_pct_float"], 0.045)

    def test_accumulating_status_below_history_threshold(self):
        """Signals are declared 'accumulating' and unvalidated when observations < threshold."""
        # 0 observations
        status = signals_options_si.get_options_si_signal_status("AAPL", "short_interest_ratio")
        self.assertEqual(status["status"], "accumulating")
        self.assertFalse(status["validated"])
        self.assertEqual(status["observations"], 0)
        self.assertEqual(status["min_required"], 63)

        # Insert 10 snapshots (still below 63)
        for i in range(10):
            db.insert_short_interest_snapshot("AAPL", f"2024-01-{i+1:02d}", short_pct_float=0.02)

        status_10 = signals_options_si.get_options_si_signal_status("AAPL", "short_interest_ratio")
        self.assertEqual(status_10["status"], "accumulating")
        self.assertFalse(status_10["validated"])
        self.assertEqual(status_10["observations"], 10)

    def test_vrp_and_skew_options_status_accumulating(self):
        """VRP and IV Skew options signals report accumulating when gex history is below threshold."""
        status_vrp = signals_options_si.get_options_si_signal_status("TSLA", "vrp")
        self.assertEqual(status_vrp["status"], "accumulating")
        self.assertFalse(status_vrp["validated"])

        status_skew = signals_options_si.get_options_si_signal_status("TSLA", "iv_skew")
        self.assertEqual(status_skew["status"], "accumulating")
        self.assertFalse(status_skew["validated"])


if __name__ == "__main__":
    unittest.main()
