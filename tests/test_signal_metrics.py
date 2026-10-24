"""Tests for signal_metrics.py and signal_trials.py (issues #133, #134, #135)."""
import math
import os
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy import stats

import db
import signal_metrics
import signal_trials


class TestSignalMetrics(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp_dir.name, "test_signals.db")
        self.patch_db = patch.object(db, "DB_PATH", self.db_path)
        self.patch_db.start()
        db.init_db()

    def tearDown(self):
        self.patch_db.stop()
        self.tmp_dir.cleanup()

    def test_known_ic_recovery(self):
        """Synthetic signal with known cross-sectional correlation is recovered within tolerance."""
        rng = np.random.default_rng(42)
        n_dates = 60
        n_tickers = 50
        dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")

        target_ic = 0.25
        rows = []

        for dt in dates:
            # Generate cross-sectional returns
            r = rng.normal(0, 0.02, size=n_tickers)
            # Generate signal with correlation target_ic to returns
            # s = target_ic * z_r + sqrt(1 - target_ic^2) * z_noise
            z_r = (r - np.mean(r)) / np.std(r)
            noise = rng.normal(0, 1.0, size=n_tickers)
            s = target_ic * z_r + math.sqrt(1.0 - target_ic ** 2) * noise

            for i in range(n_tickers):
                rows.append({
                    "date": dt,
                    "ticker": f"T{i}",
                    "signal": s[i],
                    "forward_ret_1d": r[i],
                    "forward_ret_5d": r[i] * 2.0,
                    "forward_ret_21d": r[i] * 4.0,
                    "is_liquid": True,
                })

        panel = pd.DataFrame(rows).set_index("date")

        res = signal_metrics.compute_rank_ic(panel, signal_col="signal", return_col="forward_ret_1d")
        # Target IC ~ 0.25 within tolerance (0.20 to 0.30)
        self.assertAlmostEqual(res["mean_ic"], target_ic, delta=0.06)
        self.assertGreater(res["t_stat"], 2.0)
        self.assertEqual(res["n_dates"], n_dates)

    def test_noise_signal_t_stat_less_than_two(self):
        """Pure noise signal yields |t| < 2 in a fixed-seed test."""
        rng = np.random.default_rng(999)
        n_dates = 60
        n_tickers = 50
        dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
        rows = []

        for dt in dates:
            r = rng.normal(0, 0.02, size=n_tickers)
            s = rng.normal(0, 1.0, size=n_tickers)  # pure noise
            for i in range(n_tickers):
                rows.append({
                    "date": dt,
                    "ticker": f"T{i}",
                    "signal": s[i],
                    "forward_ret_1d": r[i],
                    "is_liquid": True,
                })

        panel = pd.DataFrame(rows).set_index("date")
        res = signal_metrics.compute_rank_ic(panel, signal_col="signal", return_col="forward_ret_1d")
        self.assertLess(abs(res["t_stat"]), 2.0)

    def test_ic_decay_output_per_horizon(self):
        """Decay output has one IC per horizon."""
        rng = np.random.default_rng(77)
        dates = pd.date_range("2024-01-01", periods=20, freq="B")
        rows = []
        for dt in dates:
            for i in range(30):
                rows.append({
                    "date": dt,
                    "ticker": f"T{i}",
                    "signal": float(i),
                    "forward_ret_1d": 0.01 * (i / 30.0),
                    "forward_ret_5d": 0.03 * (i / 30.0),
                    "forward_ret_21d": 0.05 * (i / 30.0),
                    "is_liquid": True,
                })
        panel = pd.DataFrame(rows).set_index("date")
        decay = signal_metrics.compute_ic_decay(panel, signal_col="signal", horizons=(1, 5, 21))

        self.assertIn(1, decay)
        self.assertIn(5, decay)
        self.assertIn(21, decay)
        self.assertGreater(decay[1]["mean_ic"], 0.9)
        self.assertGreater(decay[5]["mean_ic"], 0.9)
        self.assertGreater(decay[21]["mean_ic"], 0.9)

    def test_known_monotone_signal_yields_monotone_deciles(self):
        """Known monotone signal yields monotonically increasing decile returns."""
        rng = np.random.default_rng(100)
        dates = pd.date_range("2024-01-01", periods=30, freq="B")
        rows = []
        for dt in dates:
            for i in range(100):
                rows.append({
                    "date": dt,
                    "ticker": f"T{i}",
                    "signal": float(i),  # signal 0..99
                    "forward_ret_1d": (i / 100.0) * 0.05 + rng.normal(0, 0.001),  # return strictly increases with i
                    "is_liquid": True,
                })
        panel = pd.DataFrame(rows).set_index("date")

        res = signal_metrics.compute_decile_spreads(panel, signal_col="signal", n_deciles=10)
        self.assertTrue(res["is_strictly_monotonic"])
        self.assertAlmostEqual(res["monotonicity_spearman"], 1.0, delta=0.01)
        self.assertGreater(res["spread_mean_daily"], 0.0)
        self.assertGreater(res["spread_t_stat"], 2.0)

    def test_noise_signal_fails_monotonicity(self):
        """Pure noise signal fails monotonicity check."""
        rng = np.random.default_rng(555)
        dates = pd.date_range("2024-01-01", periods=20, freq="B")
        rows = []
        for dt in dates:
            for i in range(100):
                rows.append({
                    "date": dt,
                    "ticker": f"T{i}",
                    "signal": rng.normal(0, 1),
                    "forward_ret_1d": rng.normal(0, 0.02),
                    "is_liquid": True,
                })
        panel = pd.DataFrame(rows).set_index("date")

        res = signal_metrics.compute_decile_spreads(panel, signal_col="signal", n_deciles=10)
        self.assertFalse(res["is_strictly_monotonic"])

    def test_spread_t_stat_uses_hac_errors(self):
        """Spread t-stat uses HAC standard errors, adjusting for autocorrelation."""
        # Create an autocorrelated spread series
        n = 100
        e = np.random.default_rng(123).normal(0, 0.01, size=n)
        ar_spread = np.zeros(n)
        for t in range(1, n):
            ar_spread[t] = 0.6 * ar_spread[t-1] + e[t] + 0.002

        ols_se = np.std(ar_spread, ddof=1) / math.sqrt(n)
        hac_se = signal_metrics.newey_west_mean_se(ar_spread)

        # Under positive AR(1) persistence, HAC SE > OLS SE
        self.assertGreater(hac_se, ols_se)

    def test_cost_adjusted_sharpe_less_than_gross(self):
        """Cost-adjusted Sharpe is strictly less than gross Sharpe for positive turnover."""
        spread_series = pd.Series([0.002, 0.003, 0.001, -0.001, 0.004] * 20)
        weights_hist = {}
        for idx in range(100):
            # Alternating portfolio -> 100% turnover
            w = {"A": 1.0, "B": -1.0} if idx % 2 == 0 else {"C": 1.0, "D": -1.0}
            weights_hist[pd.Timestamp("2024-01-01") + pd.Timedelta(days=idx)] = w

        cost_res = signal_metrics.compute_turnover_and_costs(
            weights_hist, spread_series, cost_bps=10.0, n_trials=1
        )
        gross_vol = spread_series.std(ddof=1) * math.sqrt(252)
        gross_sharpe = (spread_series.mean() * 252) / gross_vol

        self.assertGreater(cost_res["mean_turnover"], 0.0)
        self.assertLess(cost_res["cost_adjusted_sharpe"], gross_sharpe)

    def test_deflated_sharpe_decreases_as_trials_grow(self):
        """Deflated Sharpe decreases as trial count grows."""
        # A moderately positive returns series with length > 250
        rng = np.random.default_rng(42)
        spread_returns = pd.Series(rng.normal(0.0008, 0.01, size=300))

        dsr_1 = signal_metrics.compute_turnover_and_costs(
            {pd.Timestamp("2024-01-01") + pd.Timedelta(days=i): {"A": 1.0} for i in range(300)},
            spread_returns,
            cost_bps=0.0,
            n_trials=1,
        )["deflated_sharpe"]

        dsr_10 = signal_metrics.compute_turnover_and_costs(
            {pd.Timestamp("2024-01-01") + pd.Timedelta(days=i): {"A": 1.0} for i in range(300)},
            spread_returns,
            cost_bps=0.0,
            n_trials=10,
        )["deflated_sharpe"]

        dsr_100 = signal_metrics.compute_turnover_and_costs(
            {pd.Timestamp("2024-01-01") + pd.Timedelta(days=i): {"A": 1.0} for i in range(300)},
            spread_returns,
            cost_bps=0.0,
            n_trials=100,
        )["deflated_sharpe"]

        self.assertGreater(dsr_1, dsr_10)
        self.assertGreater(dsr_10, dsr_100)

    def test_trial_count_persisted_in_database(self):
        """Trial count is persisted in SQLite and increases on every trial."""
        initial_count = signal_trials.get_total_trial_count()
        self.assertEqual(initial_count, 1)  # Default minimum 1

        t1 = signal_trials.record_signal_trial("momentum_12_1", "v1", {"window": 252})
        self.assertEqual(t1, 1)

        t2 = signal_trials.record_signal_trial("reversal_21d", "v1", {"window": 21})
        self.assertEqual(t2, 2)

        self.assertEqual(signal_trials.get_total_trial_count(), 2)

        ledger = signal_trials.get_trials_ledger()
        self.assertEqual(len(ledger), 2)
        self.assertEqual(ledger[0]["signal_name"], "reversal_21d")
        self.assertEqual(ledger[1]["signal_name"], "momentum_12_1")


if __name__ == "__main__":
    unittest.main()
