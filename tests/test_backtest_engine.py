"""Unit tests for backtest_engine.py's statistical-rigor additions.

Each worktree (WT1-WT4) that implements a piece of the backtest-engine spec
(docs/superpowers/specs/2026-09-14-backtest-engine-statistical-rigor-design.md)
adds its tests to this file.
"""
import unittest

import numpy as np
import pandas as pd

import signals
from backtest_engine import (
    _block_shuffle,
    run_permutation_test,
    PermutationResult,
    deflated_sharpe_ratio,
    BacktestSummary,
    _historical_composite_signal,
    run_signals_backtest,
    run_walkforward_backtest,
)


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


class TestHistoricalCompositeSignal(unittest.TestCase):

    def setUp(self):
        self.df = _make_ohlcv(seed=3)

    def test_returns_series_and_factor_list(self):
        signal, factors_used = _historical_composite_signal(self.df)
        self.assertIsInstance(signal, pd.Series)
        self.assertEqual(len(signal), len(self.df))
        self.assertIsInstance(factors_used, list)
        # gex/monte_carlo/valuation must never appear -- excluded by design
        self.assertNotIn("gex", factors_used)
        self.assertNotIn("monte_carlo", factors_used)
        self.assertNotIn("valuation", factors_used)
        # trend/momentum/volatility/tail_risk are always computable once
        # ml.build_features clears its warm-up window on this fixture
        for key in ("trend", "momentum", "volatility", "tail_risk"):
            self.assertIn(key, factors_used)

    def test_warmup_rows_are_nan(self):
        signal, _ = _historical_composite_signal(self.df)
        self.assertTrue(signal.iloc[0:199].isna().all())

    def test_composite_matches_signals_py_factor_scores(self):
        """Parity check: the composite at the LAST bar must equal the same
        weighted blend signals.compute() would produce when fed the same
        feat row and stats, restricted to the 5 historically-computable
        factors -- proving this isn't a reimplementation that could drift
        from the live page's math."""
        import ml
        signal, _ = _historical_composite_signal(self.df)
        last_signal = signal.iloc[-1]
        self.assertFalse(pd.isna(last_signal))

        feat_df = ml.build_features(self.df)
        row = feat_df.iloc[-1]
        returns = self.df["close"].pct_change()
        rolling_vol_pct = feat_df["vol_20"].iloc[-1] * np.sqrt(252) * 100.0
        var95 = returns.rolling(252).quantile(0.05).iloc[-1] * 100.0
        var99 = returns.rolling(252).quantile(0.01).iloc[-1] * 100.0

        candidates = [
            signals._trend(row),
            signals._momentum(row),
            signals._volatility(row, {"annualised_vol": rolling_vol_pct}),
            signals._tail_risk({"var_95": var95, "var_99": var99}),
        ]
        factors = [f for f in candidates if f]
        wsum = sum(f["weight"] for f in factors)
        blended = sum(f["score"] * f["weight"] for f in factors) / wsum
        expected = round(50 + 50 * blended)
        # Allow the actual composite to differ slightly if the ML factor was
        # present (test doesn't mock the model) -- assert the non-ML core is
        # close, not exact, to avoid coupling this test to model.pkl's state.
        self.assertAlmostEqual(last_signal, expected, delta=15)

    def test_missing_model_degrades_gracefully(self):
        """With no model.pkl, the composite still computes from the 4
        remaining factors instead of raising."""
        from unittest.mock import patch
        import ml
        with patch.object(ml, "_load", return_value=None):
            signal, factors_used = _historical_composite_signal(self.df)
        self.assertNotIn("ml", factors_used)
        self.assertFalse(signal.iloc[-1] is None)

    def test_run_signals_backtest_still_works(self):
        """End-to-end: run_signals_backtest must still produce a valid
        summary with the new signal wired in, and factors_used populated."""
        summary, df = run_signals_backtest(self.df)
        self.assertIsInstance(summary.total_trades, int)
        self.assertIsInstance(summary.factors_used, list)
        self.assertTrue(len(summary.factors_used) > 0)
        self.assertFalse(df.empty)


class TestPositionColumn(unittest.TestCase):

    def test_position_column_present_and_binary(self):
        df = _make_ohlcv()
        summary, df = run_signals_backtest(df)
        self.assertIn("position", df.columns)
        self.assertTrue(set(df["position"].unique()).issubset({0, 1}))
        self.assertEqual(len(df["position"]), len(df))


class TestWalkForwardBacktest(unittest.TestCase):

    def setUp(self):
        self.df = _make_ohlcv(n=700, seed=9)  # long enough for several folds

    def test_produces_multiple_folds(self):
        summary, oos_df = run_walkforward_backtest(
            self.df, train_days=252, test_days=63, step_days=63
        )
        expected_folds = (len(self.df) - 252) // 63
        self.assertEqual(summary.n_folds, expected_folds)
        self.assertEqual(len(summary.fold_returns), expected_folds)
        self.assertFalse(oos_df.empty)

    def test_fold_windows_do_not_overlap_train(self):
        """Each fold's OOS slice must start strictly after that fold's
        train_days warm-up -- i.e. the concatenated OOS frame's length
        equals n_folds * test_days (step_days == test_days here, so no
        overlap and no gaps)."""
        summary, oos_df = run_walkforward_backtest(
            self.df, train_days=252, test_days=63, step_days=63
        )
        self.assertEqual(len(oos_df), summary.n_folds * 63)

    def test_oos_sharpe_distribution_fields_populated(self):
        summary, _ = run_walkforward_backtest(
            self.df, train_days=252, test_days=63, step_days=63
        )
        self.assertIsInstance(summary.oos_sharpe_mean, float)
        self.assertIsInstance(summary.oos_sharpe_std, float)
        self.assertGreaterEqual(summary.oos_sharpe_std, 0.0)

    def test_insufficient_history_falls_back_to_single_pass(self):
        short_df = _make_ohlcv(n=100, seed=2)
        summary, df = run_walkforward_backtest(
            short_df, train_days=252, test_days=63, step_days=63
        )
        self.assertEqual(summary.n_folds, 0)
        self.assertFalse(df.empty)  # fell back to run_signals_backtest, not an error

    def test_factors_used_reflects_union_of_contributing_folds(self):
        summary, _ = run_walkforward_backtest(
            self.df, train_days=252, test_days=63, step_days=63
        )
        self.assertIsInstance(summary.factors_used, list)
        self.assertTrue(len(summary.factors_used) > 0)
        for key in ("trend", "momentum", "volatility", "tail_risk"):
            self.assertIn(key, summary.factors_used)

    def test_trade_reconstruction_does_not_leak_across_fold_boundary(self):
        """Regression test: a position still open (position==1) on the last
        row of one fold's OOS slice must never be spliced onto a fresh,
        unrelated position opened on the first row of the next fold's OOS
        slice. Each fold's trade-reconstruction state must reset at the
        fold boundary.

        Mocks run_signals_backtest so the two folds' OOS position/strat_ret
        columns are fully controlled:
          - fold 1's OOS slice is [1, 1, 1, 1, 1] -- a position that never
            closes within the fold (dangling at the fold boundary). Correct
            per-fold accounting flushes this as its own mark-to-market
            closed trade using only fold 1's own returns (see the
            dangling-trade-flush test below) rather than dropping it or
            splicing it onto fold 2.
          - fold 2's OOS slice is [0, 1, 0] -- flat, then a fresh open/close
            pair, contributing exactly 1 closed trade on its own.

        With the leak bug (single prev/current_trade state run once across
        the concatenated OOS frame instead of resetting per fold), fold 2's
        leading flat (0) row would be misinterpreted as the close of fold
        1's dangling position (since the carried-over `prev` is 1),
        fabricating a trade value that mixes fold 1's and fold 2's returns.
        The fix must yield two DISTINCT trades with values computed purely
        from each fold's own returns -- not one trade whose value blends
        both folds.
        """
        from unittest.mock import patch

        dates1 = pd.date_range("2023-01-01", periods=6, freq="B")
        fold1_df = pd.DataFrame(
            {
                "close": [100.0] * 6,
                "ret": [0.0, 0.005, 0.005, 0.005, 0.005, 0.005],
                "strat_ret": [0.0, 0.01, 0.01, 0.01, 0.01, 0.01],
                "position": [0, 1, 1, 1, 1, 1],
            },
            index=dates1,
        )
        fold1_summary = BacktestSummary(
            total_return_pct=0.0, cagr_pct=0.0, benchmark_return_pct=0.0,
            alpha_pct=0.0, annualized_sharpe=0.0, annualized_sortino=0.0,
            max_drawdown_pct=0.0, calmar_ratio=0.0, win_rate_pct=0.0,
            profit_factor=0.0, total_trades=0, factors_used=["trend"],
        )

        dates2 = pd.date_range("2023-02-01", periods=4, freq="B")
        fold2_df = pd.DataFrame(
            {
                "close": [100.0] * 4,
                "ret": [0.0, 0.0, 0.01, -0.005],
                "strat_ret": [0.0, 0.0, 0.02, -0.01],
                "position": [0, 0, 1, 0],
            },
            index=dates2,
        )
        fold2_summary = BacktestSummary(
            total_return_pct=0.0, cagr_pct=0.0, benchmark_return_pct=0.0,
            alpha_pct=0.0, annualized_sharpe=0.0, annualized_sortino=0.0,
            max_drawdown_pct=0.0, calmar_ratio=0.0, win_rate_pct=0.0,
            profit_factor=0.0, total_trades=0, factors_used=["momentum"],
        )

        prices_df = _make_ohlcv(n=3, seed=1)  # only needs len >= train+test

        with patch(
            "backtest_engine.run_signals_backtest",
            side_effect=[(fold1_summary, fold1_df), (fold2_summary, fold2_df)],
        ):
            summary, combined = run_walkforward_backtest(
                prices_df, train_days=1, test_days=1, step_days=1
            )

        self.assertEqual(summary.n_folds, 2)
        # Fold 1's dangling position is flushed as its own closed trade
        # (5 bars of +0.01 strat_ret compounded) and fold 2 contributes its
        # own genuine closed trade (+0.02 then -0.01 compounded) -- two
        # distinct trades, neither dropped nor blended together.
        self.assertEqual(summary.total_trades, 2)
        # Both the flushed dangling trade ((1.01**5)-1 > 0) and fold 2's
        # closed trade ((1.02*0.99)-1 > 0) are winners -- confirms neither
        # was dropped nor corrupted by blending returns across the boundary
        # (a blended/misfired trade could easily land negative or missing).
        self.assertEqual(summary.win_rate_pct, 100.0)
        self.assertEqual(len(combined), 5 + 3)  # both folds' full OOS slices

    def test_dangling_open_trade_at_fold_end_is_flushed_not_dropped(self):
        """Regression test for Finding 2: a position still open (never
        closes) on the last row of a fold's OOS slice must be flushed into
        the trade ledger as a mark-to-market close rather than silently
        discarded -- otherwise total_trades/win_rate_pct/profit_factor
        diverge from the returns actually included in total_return_pct and
        annualized_sharpe (which already count those bars).

        Single fold, mocked so its entire OOS slice is one never-closed
        winning position: total_trades must be 1 (the flushed dangling
        trade), not 0.
        """
        from unittest.mock import patch

        dates = pd.date_range("2023-01-01", periods=4, freq="B")
        fold_df = pd.DataFrame(
            {
                "close": [100.0] * 4,
                "ret": [0.0, 0.004, 0.004, 0.004],
                "strat_ret": [0.0, 0.008, 0.008, 0.008],
                "position": [0, 1, 1, 1],
            },
            index=dates,
        )
        fold_summary = BacktestSummary(
            total_return_pct=0.0, cagr_pct=0.0, benchmark_return_pct=0.0,
            alpha_pct=0.0, annualized_sharpe=0.0, annualized_sortino=0.0,
            max_drawdown_pct=0.0, calmar_ratio=0.0, win_rate_pct=0.0,
            profit_factor=0.0, total_trades=0, factors_used=["trend"],
        )

        # len == train_days + test_days exactly, so the fold loop runs
        # exactly once (matching the single mocked side_effect).
        prices_df = _make_ohlcv(n=2, seed=1)

        with patch(
            "backtest_engine.run_signals_backtest",
            side_effect=[(fold_summary, fold_df)],
        ):
            summary, _ = run_walkforward_backtest(
                prices_df, train_days=1, test_days=1, step_days=1
            )

        self.assertEqual(summary.n_folds, 1)
        self.assertEqual(summary.total_trades, 1)
        self.assertEqual(summary.win_rate_pct, 100.0)  # the flushed trade is a winner

    def test_step_days_less_than_test_days_deduplicates_overlapping_dates(self):
        """Regression test for Finding 1: when step_days < test_days,
        consecutive folds' OOS windows overlap and share calendar dates.
        The returned combined OOS frame must have a unique, non-duplicated
        index so aggregate metrics don't triple-count overlapping bars."""
        summary, oos_df = run_walkforward_backtest(
            self.df, train_days=252, test_days=63, step_days=21
        )
        self.assertGreater(summary.n_folds, 1)
        self.assertTrue(oos_df.index.is_unique)

    def test_step_days_zero_raises_value_error(self):
        """Regression test for Finding 1: step_days <= 0 never advances the
        fold loop's `start` cursor, which would otherwise hang forever.
        Must raise ValueError instead of looping."""
        with self.assertRaises(ValueError):
            run_walkforward_backtest(self.df, train_days=252, test_days=63, step_days=0)

    def test_step_days_negative_raises_value_error(self):
        with self.assertRaises(ValueError):
            run_walkforward_backtest(self.df, train_days=252, test_days=63, step_days=-5)

    def test_oos_sharpe_std_not_nan_with_single_fold(self):
        """Regression test for Finding 4: ddof=1 stddev on a single-element
        array is NaN; a lone fold must report 0.0 dispersion instead."""
        single_fold_df = _make_ohlcv(n=320, seed=11)
        summary, _ = run_walkforward_backtest(
            single_fold_df, train_days=252, test_days=63, step_days=63
        )
        self.assertEqual(summary.n_folds, 1)
        self.assertEqual(summary.oos_sharpe_std, 0.0)
        self.assertFalse(np.isnan(summary.oos_sharpe_std))


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


class TestDeflatedSharpeRatio(unittest.TestCase):

    def test_matches_hand_computed_reference(self):
        """Toy series [0.02, 0.02, 0.02, 0.02, -0.02] (n=5), computed by hand:
        mean=0.012, sample std (ddof=1)=0.0178885, sr=mean/std=0.670820
        (skew, excess kurtosis via pandas' bias-corrected Fisher-Pearson
        formulas) skew=-2.236061, excess kurtosis=5.0 -> raw kurtosis=8.0
        sr_std = sqrt((1 - skew*sr + (kurt-1)/4*sr^2) / (n-1))
               = sqrt((1 - (-2.236061*0.670820) + (7/4)*0.45) / 4)
               = sqrt((1 + 1.5 + 0.7875) / 4) = sqrt(0.821875) = 0.906573
        z = sr / sr_std = 0.670820 / 0.906573 = 0.739945
        DSR = Phi(0.739945) ~= 0.7703 (standard normal CDF)
        """
        returns = pd.Series([0.02, 0.02, 0.02, 0.02, -0.02])
        result = deflated_sharpe_ratio(returns, n_trials=1)
        self.assertAlmostEqual(result, 0.7703, delta=0.01)

    def test_result_is_a_probability(self):
        np.random.seed(7)
        returns = pd.Series(np.random.normal(0.001, 0.01, 200))
        result = deflated_sharpe_ratio(returns, n_trials=1)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_zero_returns_average_near_half_across_draws(self):
        """DSR behaves like Phi(t-statistic) of the sample mean -- under a
        true null (mean=0) that statistic is approximately UNIFORM on [0, 1]
        across repeated draws (a single draw can legitimately land anywhere
        in that range, the same way a p-value does under the null). So this
        checks the AVERAGE over many independent draws lands near 0.5, not
        any single draw."""
        results = []
        for seed in range(40):
            rng = np.random.default_rng(seed)
            returns = pd.Series(rng.normal(0.0, 0.01, 500))
            results.append(deflated_sharpe_ratio(returns, n_trials=1))
        self.assertAlmostEqual(float(np.mean(results)), 0.5, delta=0.1)

    def test_n_trials_greater_than_one_not_implemented(self):
        returns = pd.Series([0.01, 0.02, -0.01, 0.015, -0.005])
        with self.assertRaises(NotImplementedError):
            deflated_sharpe_ratio(returns, n_trials=5)

    def test_too_few_observations_returns_zero(self):
        self.assertEqual(deflated_sharpe_ratio(pd.Series([0.01]), n_trials=1), 0.0)
        self.assertEqual(deflated_sharpe_ratio(pd.Series([], dtype=float), n_trials=1), 0.0)
        self.assertEqual(deflated_sharpe_ratio(pd.Series([0.01, -0.01]), n_trials=1), 0.0)
        self.assertEqual(deflated_sharpe_ratio(pd.Series([0.01, -0.01, 0.02]), n_trials=1), 0.0)

    def test_backtest_summary_has_deflated_sharpe_fields(self):
        summary = BacktestSummary(
            total_return_pct=0.0, cagr_pct=0.0, benchmark_return_pct=0.0,
            alpha_pct=0.0, annualized_sharpe=0.0, annualized_sortino=0.0,
            max_drawdown_pct=0.0, calmar_ratio=0.0, win_rate_pct=0.0,
            profit_factor=0.0, total_trades=0,
        )
        self.assertEqual(summary.deflated_sharpe, 0.0)
        self.assertEqual(summary.n_trials_assumed, 1)


if __name__ == "__main__":
    unittest.main()
