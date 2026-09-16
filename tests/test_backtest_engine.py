"""Unit tests for backtest_engine.py's statistical-rigor additions.

Each worktree (WT1-WT4) that implements a piece of the backtest-engine spec
(docs/superpowers/specs/2026-09-14-backtest-engine-statistical-rigor-design.md)
adds its tests to this file.
"""
import unittest

import numpy as np
import pandas as pd

from backtest_engine import run_permutation_test, PermutationResult


def _make_ohlcv(n=300, seed=1, trend=0.0006):
    np.random.seed(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    rets = np.random.normal(trend, 0.013, size=n)
    close = 100.0 * np.exp(np.cumsum(rets))
    return pd.DataFrame({
        "open": close * (1 + np.random.normal(0, 0.002, n)),
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": np.random.randint(400_000, 1_500_000, n),
    }, index=dates)


class TestPermutationTest(unittest.TestCase):

    def test_returns_permutation_result_with_expected_shape(self):
        df = _make_ohlcv(n=300)
        result = run_permutation_test(df, n_permutations=20, block_size=20, seed=1)
        self.assertIsInstance(result, PermutationResult)
        self.assertEqual(result.n_permutations, 20)
        self.assertGreaterEqual(result.p_value, 0.0)
        self.assertLessEqual(result.p_value, 1.0)

    def test_p_value_is_not_always_near_zero_on_random_walk(self):
        """Calibration check: on a pure random walk (no real signal-return
        relationship), the p-value should land somewhere plausible under the
        null, not always near 0 -- a p-value that's always ~0 regardless of
        input would indicate a broken null model (e.g. insufficient
        shuffling, or a leak that lets the 'real' run always look best)."""
        np.random.seed(123)
        p_values = []
        for trial_seed in range(5):
            df = _make_ohlcv(n=300, seed=100 + trial_seed, trend=0.0)
            result = run_permutation_test(
                df, n_permutations=30, block_size=20, seed=trial_seed
            )
            p_values.append(result.p_value)
        # Not a strict statistical guarantee with only 5 trials, but a
        # regression guard: if every trial comes back p<0.05, something in
        # the null construction is broken (e.g. real run isn't actually
        # comparable to the shuffled runs).
        self.assertFalse(all(p < 0.05 for p in p_values))

    def test_reproducible_with_same_seed(self):
        df = _make_ohlcv(n=300)
        r1 = run_permutation_test(df, n_permutations=15, block_size=20, seed=7)
        r2 = run_permutation_test(df, n_permutations=15, block_size=20, seed=7)
        self.assertEqual(r1.null_mean, r2.null_mean)
        self.assertEqual(r1.p_value, r2.p_value)

    def test_shuffled_frame_preserves_ohlc_consistency(self):
        """Block shuffling must not break high >= low, high >= close, etc.
        within any row -- rows are moved as whole blocks, never sliced
        across columns."""
        from backtest_engine import _block_shuffle
        df = _make_ohlcv(n=100)
        rng = np.random.default_rng(3)
        shuffled = _block_shuffle(df, block_size=20, rng=rng)
        self.assertEqual(len(shuffled), len(df))
        self.assertTrue((shuffled["high"] >= shuffled["low"]).all())
        self.assertTrue((shuffled["high"] >= shuffled["close"]).all())

    def test_shuffled_frame_covers_all_rows_for_non_divisible_length(self):
        """Regression guard: when len(df) is not an exact multiple of
        block_size, the trailing partial block must still be included --
        not silently dropped. n=253 against block_size=20 leaves a
        remainder of 13 rows that a floor-division n_blocks computation
        would omit."""
        from backtest_engine import _block_shuffle
        df = _make_ohlcv(n=253)
        rng = np.random.default_rng(5)
        shuffled = _block_shuffle(df, block_size=20, rng=rng)
        self.assertEqual(len(shuffled), len(df))
        self.assertEqual(len(shuffled), 253)


if __name__ == "__main__":
    unittest.main()
