"""Unit tests for backtest_engine.py's statistical-rigor additions.

Each worktree (WT1-WT4) that implements a piece of the backtest-engine spec
(docs/superpowers/specs/2026-09-14-backtest-engine-statistical-rigor-design.md)
adds its tests to this file.
"""
import math
import unittest

import numpy as np
import pandas as pd

from backtest_engine import _block_shuffle, run_permutation_test, PermutationResult


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
        shuffling, or a leak that lets the 'real' run always look best).

        This is the headline regression guard for the level-shuffle defect:
        shuffling blocks of raw price levels manufactured huge fake returns
        at block boundaries, which pushed p to ~0.00 on zero-signal data.
        """
        p_values = []
        for trial_seed in range(6):
            df = _make_ohlcv(n=300, seed=100 + trial_seed, trend=0.0)
            result = run_permutation_test(
                df, n_permutations=30, block_size=20, seed=trial_seed
            )
            p_values.append(result.p_value)
        # Under a correct null on zero-signal data these are ~U[0,1]; the
        # seeds are fixed, so these bounds are deterministic, not flaky.
        self.assertFalse(all(p < 0.05 for p in p_values), p_values)
        self.assertGreater(max(p_values), 0.5, p_values)
        self.assertGreater(
            sum(1 for p in p_values if p > 0.10), len(p_values) / 2, p_values
        )
        self.assertGreater(sum(p_values) / len(p_values), 0.15, p_values)

    def test_p_value_is_never_exactly_zero(self):
        """The bias-corrected estimator (1+c)/(1+N) floors at 1/(N+1); a raw
        `(null >= real).mean()` can report exactly 0.0, which is not a valid
        p-value for a Monte Carlo test with finitely many draws."""
        df = _make_ohlcv(n=300, seed=42, trend=0.004)  # strong fake uptrend
        result = run_permutation_test(df, n_permutations=20, block_size=20, seed=3)
        self.assertGreaterEqual(result.p_value, 1.0 / 21.0)

    def test_rejects_degenerate_parameters(self):
        df = _make_ohlcv(n=100)
        for kwargs in ({"block_size": 0}, {"block_size": -5}, {"n_permutations": 0}):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    run_permutation_test(df, **{"n_permutations": 5, **kwargs})

    def test_shuffled_path_preserves_return_volatility(self):
        """Regression guard for the level-shuffle defect: permuting blocks of
        raw price LEVELS splices unrelated price levels together, so every
        block boundary fabricates a large return (measured ~94% annualized
        vol on a shuffled AAPL path vs ~29% real). Permuting the RETURN
        series leaves the return distribution -- and so its volatility --
        intact; only the ordering changes."""
        df = _make_ohlcv(n=500, seed=11)
        shuffled = _block_shuffle(df, block_size=20, rng=np.random.default_rng(3))

        real_vol = df["close"].pct_change().std() * math.sqrt(252)
        shuffled_vol = shuffled["close"].pct_change().std() * math.sqrt(252)
        self.assertLess(
            abs(shuffled_vol - real_vol) / real_vol, 0.15,
            f"shuffled vol {shuffled_vol:.3f} vs real {real_vol:.3f}",
        )

    def test_shuffled_returns_are_a_permutation_of_the_originals(self):
        """Set-equality, not just length: a duplicated-block or dropped-block
        bug keeps len() correct while silently changing the return
        distribution the null is drawn from."""
        df = _make_ohlcv(n=253, seed=9)
        shuffled = _block_shuffle(df, block_size=20, rng=np.random.default_rng(5))

        real_close, shuf_close = df["close"].to_numpy(), shuffled["close"].to_numpy()
        # Bar 0 is the anchor: same starting price, returns after it permuted.
        self.assertAlmostEqual(shuf_close[0], real_close[0], places=10)
        np.testing.assert_allclose(
            np.sort(np.log(shuf_close[1:] / shuf_close[:-1])),
            np.sort(np.log(real_close[1:] / real_close[:-1])),
            rtol=1e-9, atol=1e-12,
        )
        np.testing.assert_array_equal(
            np.sort(shuffled["volume"].to_numpy()), np.sort(df["volume"].to_numpy())
        )
        self.assertTrue(shuffled.index.equals(df.index))
        self.assertEqual(list(shuffled.columns), list(df.columns))

    def test_reproducible_with_same_seed(self):
        df = _make_ohlcv(n=300)
        r1 = run_permutation_test(df, n_permutations=15, block_size=20, seed=7)
        r2 = run_permutation_test(df, n_permutations=15, block_size=20, seed=7)
        self.assertEqual(r1.null_mean, r2.null_mean)
        self.assertEqual(r1.p_value, r2.p_value)

    def test_shuffled_frame_preserves_ohlc_consistency(self):
        """Each row's open/high/low keep their original ratio to their own
        close, re-anchored to the row's new reconstructed close -- so
        high >= low, high >= close, low <= close survive the shuffle."""
        df = _make_ohlcv(n=100)
        shuffled = _block_shuffle(df, block_size=20, rng=np.random.default_rng(3))
        self.assertEqual(len(shuffled), len(df))
        self.assertTrue((shuffled["high"] >= shuffled["low"]).all())
        self.assertTrue((shuffled["high"] >= shuffled["close"]).all())
        self.assertTrue((shuffled["low"] <= shuffled["close"]).all())
        self.assertTrue((shuffled["high"] >= shuffled["open"]).all())
        self.assertTrue((shuffled["low"] <= shuffled["open"]).all())
        self.assertTrue((shuffled[["open", "high", "low", "close"]] > 0).all().all())

    def test_shuffled_frame_covers_all_rows_for_non_divisible_length(self):
        """Regression guard: when the return series' length is not an exact
        multiple of block_size, the trailing partial block must still be
        included -- not silently dropped. n=253 against block_size=20 leaves
        a remainder that a floor-division n_blocks computation would omit."""
        df = _make_ohlcv(n=253)
        shuffled = _block_shuffle(df, block_size=20, rng=np.random.default_rng(5))
        self.assertEqual(len(shuffled), 253)


if __name__ == "__main__":
    unittest.main()
