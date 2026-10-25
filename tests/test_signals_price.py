"""Tests for signals_price.py (issue #136)."""
import os
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

import db
import signal_metrics
import signal_trials
import signals_price


class TestSignalsPrice(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp_dir.name, "test_signals_price.db")
        self.patch_db = patch.object(db, "DB_PATH", self.db_path)
        self.patch_db.start()
        db.init_db()

    def tearDown(self):
        self.patch_db.stop()
        self.tmp_dir.cleanup()

    def test_momentum_12_1_calculation(self):
        """12-1 momentum correctly skips the most recent 21 days and looks back 252 days."""
        dates = pd.date_range("2023-01-01", periods=300, freq="B")
        # Constant geometric growth: price doubles between t-252 and t-21
        prices = pd.Series(100.0 * (1.002 ** np.arange(300)), index=dates)

        mom = signals_price.compute_momentum_12_1(prices)
        # First 252 values should be NaN
        self.assertTrue(mom.iloc[:252].isna().all())
        # At index 252: p_t21 is price at index 231, p_t252 is price at index 0
        expected = (prices.iloc[231] - prices.iloc[0]) / prices.iloc[0]
        self.assertAlmostEqual(mom.iloc[252], expected)

    def test_short_term_reversal_calculation(self):
        """Short-term reversal negates recent 21-day return (losers get positive signal)."""
        dates = pd.date_range("2024-01-01", periods=50, freq="B")
        # Stock drops from 100 to 80 over 21 days (-20% return)
        prices_drop = pd.Series([100.0] * 20 + list(np.linspace(100, 80, 22)) + [80.0] * 8, index=dates)
        rev = signals_price.compute_short_term_reversal(prices_drop, window=21)

        # 21-day return was -20%, so negated reversal score should be +0.20 (+20%)
        val = rev.iloc[41]
        raw_ret = (prices_drop.iloc[41] - prices_drop.iloc[20]) / prices_drop.iloc[20]
        self.assertAlmostEqual(val, -raw_ret)
        self.assertGreater(val, 0.0)

    def test_residual_reversal_uses_residuals_not_raw_returns(self):
        """Residual reversal must use idiosyncratic residuals, NOT raw returns.

        If a stock's returns are 100% explained by the market (zero residuals),
        residual reversal must be zero, even if raw returns were large.
        """
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        market_ret = pd.Series(np.linspace(0.01, 0.05, n), index=dates)
        # Stock has beta = 2.0, alpha = 0.0, and ZERO residual: stock_ret = 2.0 * market_ret
        stock_ret = 2.0 * market_ret

        res_rev = signals_price.compute_residual_reversal(
            stock_ret, market_ret, window=21, reg_window=50
        )

        # Residuals are zero -> residual reversal must be ~0.0
        valid_res = res_rev.dropna()
        self.assertGreater(len(valid_res), 10)
        self.assertTrue((np.abs(valid_res) < 1e-6).all())

        # If we now add a big idiosyncratic drop in the stock on top of market
        stock_with_shock = stock_ret.copy()
        stock_with_shock.iloc[60:81] -= 0.02  # negative idiosyncratic shock
        res_rev_shock = signals_price.compute_residual_reversal(
            stock_with_shock, market_ret, window=21, reg_window=50
        )
        # Negated negative shock -> positive reversal signal
        self.assertGreater(res_rev_shock.iloc[80], 0.1)

    def test_low_volatility_calculation(self):
        """Low volatility signal negates annualized rolling standard deviation."""
        dates = pd.date_range("2024-01-01", periods=100, freq="B")
        # Steady prices -> ~0 vol
        calm_prices = pd.Series(100.0 + np.sin(np.arange(100)) * 0.1, index=dates)
        low_vol = signals_price.compute_low_volatility(calm_prices, window=63)

        self.assertTrue(low_vol.iloc[:62].isna().all())
        val = low_vol.iloc[63]
        # Should be a negative number close to 0
        self.assertLess(val, 0.0)
        self.assertGreater(val, -0.10)

    def test_all_price_signals_evaluated_in_lab_pipeline(self):
        """All 4 price signals run through the Rank IC, decile spread, and turnover engine."""
        rng = np.random.default_rng(42)
        n_dates = 60
        n_tickers = 30
        dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")

        rows = []
        for dt in dates:
            for i in range(n_tickers):
                rows.append({
                    "date": dt,
                    "ticker": f"STK_{i}",
                    "close": 100.0 + rng.normal(0, 5),
                    "volume": 50000.0,
                    "is_liquid": True,
                    "forward_ret_1d": rng.normal(0.0005, 0.015),
                    "forward_ret_5d": rng.normal(0.002, 0.03),
                    "forward_ret_21d": rng.normal(0.008, 0.06),
                    "momentum_12_1": rng.normal(0.1, 0.2),
                    "short_term_reversal": rng.normal(0, 0.05),
                    "residual_reversal": rng.normal(0, 0.04),
                    "low_volatility": -abs(rng.normal(0.2, 0.05)),
                })

        panel = pd.DataFrame(rows).set_index("date")

        for sig_name in signals_price.PRICE_SIGNAL_NAMES:
            # 1. Rank IC & Decay
            ic_res = signal_metrics.compute_rank_ic(panel, signal_col=sig_name)
            self.assertIn("mean_ic", ic_res)
            self.assertIn("t_stat", ic_res)

            decay = signal_metrics.compute_ic_decay(panel, signal_col=sig_name, horizons=(1, 5, 21))
            self.assertEqual(len(decay), 3)

            # 2. Decile Spreads
            dec_res = signal_metrics.compute_decile_spreads(panel, signal_col=sig_name, n_deciles=10)
            self.assertIn("gross_sharpe", dec_res)
            self.assertIn("spread_t_stat", dec_res)

            # 3. Turnover & Costs
            cost_res = signal_metrics.compute_turnover_and_costs(
                dec_res["weights_history"], dec_res["spread_series"], cost_bps=10.0, n_trials=1
            )
            self.assertIn("cost_adjusted_sharpe", cost_res)
            self.assertIn("deflated_sharpe", cost_res)

            # 4. Record trial
            trial_id = signal_trials.record_signal_trial(
                signal_name=sig_name,
                variant="daily_cross_section",
                metadata={"ic": ic_res["mean_ic"], "sharpe": cost_res["cost_adjusted_sharpe"]},
            )
            self.assertGreater(trial_id, 0)

        total_trials = signal_trials.get_total_trial_count()
        self.assertEqual(total_trials, len(signals_price.PRICE_SIGNAL_NAMES))


if __name__ == "__main__":
    unittest.main()
