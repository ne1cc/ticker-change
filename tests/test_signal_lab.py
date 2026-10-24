"""Tests for Signal Lab, validation gates, and /analytics integration (issue #139)."""
import os
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

import app as app_module
import db
import signal_lab
import signals


class TestSignalLab(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp_dir.name, "test_signal_lab.db")
        self.patch_db = patch.object(db, "DB_PATH", self.db_path)
        self.patch_db.start()
        db.init_db()

        self.client = app_module.app.test_client()

    def tearDown(self):
        self.patch_db.stop()
        self.tmp_dir.cleanup()

    def test_validation_gates_configured_and_displayed(self):
        """Validation gates are explicitly defined with required institutional thresholds."""
        gates = signal_lab.VALIDATION_GATES
        self.assertIn("min_abs_ic_t_stat", gates)
        self.assertIn("min_cost_adjusted_sharpe", gates)
        self.assertIn("min_deflated_sharpe", gates)
        self.assertIn("min_observations", gates)

        self.assertEqual(gates["min_abs_ic_t_stat"]["threshold"], 2.0)
        self.assertEqual(gates["min_cost_adjusted_sharpe"]["threshold"], 0.0)
        self.assertEqual(gates["min_deflated_sharpe"]["threshold"], 0.50)
        self.assertEqual(gates["min_observations"]["threshold"], 63)

    def test_unvalidated_signals_never_contribute_rank_to_analytics(self):
        """Unvalidated signals must NEVER contribute a rank or percentile to /analytics."""
        ranks = signal_lab.get_ticker_signal_ranks("AAPL")
        for s in ranks:
            if not s["validated"]:
                self.assertIsNone(
                    s["rank"],
                    f"Unvalidated signal '{s['name']}' illegally contributed a numeric rank!",
                )
                self.assertIsNone(
                    s["percentile"],
                    f"Unvalidated signal '{s['name']}' illegally contributed a numeric percentile!",
                )

    def test_hand_weighted_composite_removed_or_replaced_by_data_driven_weights(self):
        """Hand-weighted composite is removed and replaced by data-driven weights."""
        # signals.WEIGHTS should match signals.DATA_DRIVEN_WEIGHTS
        self.assertEqual(signals.WEIGHTS, signals.DATA_DRIVEN_WEIGHTS)
        # Verify no subjective tilt like trend=0.20 vs tail_risk=0.05
        weights_values = set(signals.WEIGHTS.values())
        self.assertEqual(len(weights_values), 1, "Subjective asymmetric weights still present!")

    def test_signal_lab_route_renders_table_and_gates(self):
        """The /signal-lab route serves HTTP 200 and renders the matrix and survivorship caveat."""
        res = self.client.get("/signal-lab")
        self.assertEqual(res.status_code, 200)
        html = res.data.decode("utf-8")

        self.assertIn("Signal Lab", html)
        self.assertIn("Survivorship Bias Disclosure", html)
        self.assertIn(signal_lab.SURVIVORSHIP_CAVEAT, html)
        self.assertIn("Rank IC |t-stat|", html)
        self.assertIn("Deflated Sharpe Ratio (DSR)", html)
        self.assertIn("12-1 Month Momentum", html)
        self.assertIn("Short-Term Reversal", html)
        self.assertIn("Post-Earnings Announcement Drift", html)
        self.assertIn("Variance Risk Premium", html)

    def test_signal_lab_api_returns_structured_json(self):
        """The /api/signal-lab endpoint returns complete JSON metadata."""
        res = self.client.get("/api/signal-lab?cost_bps=10")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        self.assertIn("signals", data)
        self.assertIn("validation_gates", data)
        self.assertIn("survivorship_caveat", data)
        self.assertIn("total_trials", data)
        self.assertEqual(data["cost_bps"], 10.0)

        signals_by_key = {s["key"]: s for s in data["signals"]}
        self.assertIn("momentum_12_1", signals_by_key)
        self.assertIn("low_volatility", signals_by_key)
        self.assertIn("vrp", signals_by_key)
        self.assertIn("pead_drift", signals_by_key)

        # PEAD must have descoped status per ADR 0003
        self.assertEqual(signals_by_key["pead_drift"]["status"], "insufficient_history")
        # VRP must have accumulating status per issue 138
        self.assertEqual(signals_by_key["vrp"]["status"], "accumulating")


if __name__ == "__main__":
    unittest.main()
