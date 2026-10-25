"""Tests for signal_dataset.py (issue #132)."""
import unittest
import numpy as np
import pandas as pd

import signal_dataset


class TestSignalDataset(unittest.TestCase):
    def test_alignment_hand_built_example(self):
        """Verify strict alignment: row t forward return is (price[t+h] - price[t])/price[t]."""
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        # Prices: 100, 105, 110, 99, 120
        prices = pd.Series([100.0, 105.0, 110.0, 99.0, 120.0], index=dates)

        fwd = signal_dataset.compute_forward_returns(prices, horizons=(1, 2))

        # At t=0 (price=100):
        # 1d fwd return = (105 - 100) / 100 = +0.05
        # 2d fwd return = (110 - 100) / 100 = +0.10
        self.assertAlmostEqual(fwd["forward_ret_1d"].iloc[0], 0.05)
        self.assertAlmostEqual(fwd["forward_ret_2d"].iloc[0], 0.10)

        # At t=2 (price=110):
        # 1d fwd return = (99 - 110) / 110 = -0.10
        # 2d fwd return = (120 - 110) / 110 = +0.09090909...
        self.assertAlmostEqual(fwd["forward_ret_1d"].iloc[2], -0.10)
        self.assertAlmostEqual(fwd["forward_ret_2d"].iloc[2], (120.0 - 110.0) / 110.0)

        # At the end of the series, future prices do not exist -> NaN
        self.assertTrue(pd.isna(fwd["forward_ret_1d"].iloc[-1]))
        self.assertTrue(pd.isna(fwd["forward_ret_2d"].iloc[-2]))

    def test_survivorship_caveat_present(self):
        """The survivorship bias caveat must be present in metadata."""
        self.assertIn("survivorship bias applies", signal_dataset.SURVIVORSHIP_CAVEAT)
        panel_meta = signal_dataset.build_research_panel({})
        self.assertEqual(panel_meta["survivorship_caveat"], signal_dataset.SURVIVORSHIP_CAVEAT)

    def test_lookahead_leak_test_shuffling(self):
        """Shuffling future returns destroys any predictive correlation."""
        rng = np.random.default_rng(12345)
        n = 500
        # Construct a strong synthetic predictive signal (e.g. signal ~ future return + noise)
        fwd_returns = pd.Series(rng.normal(0.001, 0.02, size=n))
        signal = fwd_returns * 0.8 + rng.normal(0, 0.01, size=n)

        orig_corr, shuffled_corr = signal_dataset.verify_no_lookahead_leakage(
            signal, fwd_returns, rng_seed=42
        )

        # Original correlation is high (> 0.5)
        self.assertGreater(orig_corr, 0.5)
        # Shuffled future correlation collapses to noise (|corr| < 0.1)
        self.assertLess(abs(shuffled_corr), 0.1)

    def test_build_research_panel_filters(self):
        """Test liquidity and history filters on panel construction."""
        dates = pd.date_range("2024-01-01", periods=100, freq="B")

        # Ticker A: full 100 days, good price and volume
        df_a = pd.DataFrame({
            "close": np.linspace(50, 60, 100),
            "volume": np.full(100, 10000.0),
        }, index=dates)

        # Ticker B: only 30 days (insufficient history)
        df_b = pd.DataFrame({
            "close": np.linspace(20, 25, 30),
            "volume": np.full(30, 5000.0),
        }, index=dates[:30])

        # Ticker C: penny stock ($0.50)
        df_c = pd.DataFrame({
            "close": np.full(100, 0.50),
            "volume": np.full(100, 10000.0),
        }, index=dates)

        prices_map = {"AAA": df_a, "BBB": df_b, "CCC": df_c}
        res = signal_dataset.build_research_panel(prices_map, min_history=63, min_price=1.0)

        # BBB should be excluded due to history
        self.assertIn("BBB", res["excluded"])
        # AAA and CCC are in panel, but CCC has is_liquid=False
        panel = res["panel"]
        self.assertIn("AAA", res["tickers"])
        self.assertIn("CCC", res["tickers"])
        self.assertNotIn("BBB", res["tickers"])

        aaa_liquid = panel[panel["ticker"] == "AAA"]["is_liquid"].all()
        ccc_liquid = panel[panel["ticker"] == "CCC"]["is_liquid"].any()
        self.assertTrue(aaa_liquid)
        self.assertFalse(ccc_liquid)


if __name__ == "__main__":
    unittest.main()
