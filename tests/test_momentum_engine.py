"""Synthetic-data tests for momentum_engine.py — no network, no yfinance."""
import math
import unittest

import numpy as np
import pandas as pd

from momentum_engine import (
    absolute_hurdle,
    backtest_rotation,
    backtest_timeseries,
    perf_stats,
    relative_stats,
    rolling_score,
    score_series,
    score_universe,
    sma_score_universe,
    sma_spread,
)


def _bidx(n, start="2015-01-01"):
    return pd.date_range(start, periods=n, freq="B")


def _crossover_series():
    """300 declining bars (death cross) then 300 rising bars (golden cross)."""
    decline = np.linspace(200.0, 100.0, 300)
    rise = np.linspace(100.0, 400.0, 300)
    prices = np.concatenate([decline, rise])
    return pd.Series(prices, index=_bidx(600))


class TestRollingScore(unittest.TestCase):
    def test_offset_correctness_on_linear_series(self):
        close = pd.Series(np.arange(1, 301, dtype=float), index=_bidx(300))
        score = rolling_score(close)
        # close[i] = i + 1 (1-indexed values); at position 252:
        # shift(21) -> close.iloc[231] = 232.0; shift(252) -> close.iloc[0] = 1.0
        self.assertEqual(score.iloc[252], 232.0 / 1.0 - 1.0)
        self.assertTrue(pd.isna(score.iloc[0]))
        self.assertTrue(pd.isna(score.iloc[251]))  # shift(252) still undefined


class TestScoreSeries(unittest.TestCase):
    def test_exact_mom_12_1_and_risk_adj(self):
        close = pd.Series(100.0 + np.arange(273), index=_bidx(273))
        score = score_series(close, symbol="X")
        self.assertIsNotNone(score)

        # p_252 = close.iloc[20] = 120.0 ; p_21 = close.iloc[251] = 351.0
        expected_mom_12_1 = (351.0 - 120.0) / 120.0
        self.assertAlmostEqual(score.mom_12_1, expected_mom_12_1, places=10)
        self.assertAlmostEqual(score.mom_12_1, 1.925, places=10)

        expected_vol = close.pct_change().iloc[-252:].std() * math.sqrt(252)
        self.assertAlmostEqual(score.ann_vol_1y, expected_vol, places=10)
        expected_risk_adj = round(expected_mom_12_1 / expected_vol, 2) if expected_vol > 0 else 0.0
        self.assertEqual(score.risk_adj, expected_risk_adj)

    def test_none_below_minimum_length(self):
        close = pd.Series(np.arange(272, dtype=float), index=_bidx(272))
        self.assertIsNone(score_series(close))

    def test_passes_absolute_hurdle(self):
        close = pd.Series(100.0 + np.arange(273), index=_bidx(273))
        score = score_series(close, hurdle=10.0)  # mom_12_1 (~1.925) below hurdle
        self.assertFalse(score.passes_absolute)
        score2 = score_series(close, hurdle=0.1)
        self.assertTrue(score2.passes_absolute)


class TestScoreUniverseAgreement(unittest.TestCase):
    def test_default_as_of_matches_score_series(self):
        idx = _bidx(400)

        def make_series(drift, vol, seed):
            rng = np.random.default_rng(seed)
            rets = rng.normal(drift, vol, size=399)
            prices = 100.0 * np.cumprod(1 + rets)
            return pd.Series(np.concatenate([[100.0], prices]), index=idx)

        symbols = ["A", "B", "C"]
        price_df = pd.DataFrame({
            "A": make_series(0.0005, 0.01, 1),
            "B": make_series(-0.0003, 0.02, 2),
            "C": make_series(0.0010, 0.015, 3),
        })

        universe_scores = score_universe(price_df, symbols)
        self.assertEqual(set(universe_scores.keys()), set(symbols))

        for sym in symbols:
            direct = score_series(price_df[sym], symbol=sym)
            u = universe_scores[sym]
            self.assertAlmostEqual(u.mom_12_1, direct.mom_12_1, places=9)
            self.assertAlmostEqual(u.ann_vol_1y, direct.ann_vol_1y, places=9)
            self.assertEqual(u.risk_adj, direct.risk_adj)
            self.assertAlmostEqual(u.mom_6m, direct.mom_6m, places=9)
            self.assertAlmostEqual(u.mom_3m, direct.mom_3m, places=9)
            self.assertAlmostEqual(u.mom_1m, direct.mom_1m, places=9)

    def test_historical_as_of_leaves_short_horizons_zero(self):
        idx = _bidx(400)
        rng = np.random.default_rng(11)
        rets = rng.normal(0.0003, 0.01, size=399)
        prices = pd.Series(np.concatenate([[100.0], 100.0 * np.cumprod(1 + rets)]), index=idx)
        price_df = pd.DataFrame({"A": prices})

        scores = score_universe(price_df, ["A"], as_of=300)
        self.assertEqual(scores["A"].mom_6m, 0.0)
        self.assertEqual(scores["A"].mom_3m, 0.0)
        self.assertEqual(scores["A"].mom_1m, 0.0)


class TestSmaSpread(unittest.TestCase):
    def test_sign_crosses_from_death_to_golden(self):
        close = _crossover_series()
        spread = sma_spread(close)
        self.assertLess(spread.iloc[250], 0.0)
        self.assertGreater(spread.iloc[-1], 0.0)


class TestPerfAndRelativeStats(unittest.TestCase):
    def test_perf_stats_constant_return_vs_closed_form(self):
        r = 0.001
        n = 252
        series = pd.Series([r] * n, index=_bidx(n))
        stats = perf_stats(series)

        cum = (1 + r) ** n - 1
        cagr = (1 + cum) ** (252.0 / n) - 1
        self.assertAlmostEqual(stats["total_return"], round(cum * 100, 1), places=6)
        self.assertAlmostEqual(stats["annual_return"], round(cagr * 100, 1), places=6)
        # Sharpe/calmar are division-by-near-zero-std unstable for a perfectly
        # constant series (float noise in std, not a formula bug) — not asserted here.

    def test_relative_stats_identity(self):
        rng = np.random.default_rng(7)
        rets = pd.Series(rng.normal(0.0005, 0.01, 300), index=_bidx(300))
        rel = relative_stats(rets, rets)
        self.assertAlmostEqual(rel["beta"], 1.0, places=6)
        self.assertAlmostEqual(rel["alpha"], 0.0, places=6)
        self.assertAlmostEqual(rel["corr"], 1.0, places=6)

    def test_empty_series_returns_fallback(self):
        stats = perf_stats(pd.Series([], dtype=float))
        self.assertEqual(stats["sharpe"], 0.0)
        self.assertEqual(stats["total_return"], 0.0)


class TestBacktestRotation(unittest.TestCase):
    def _downtrend_universe(self, n_days=400):
        idx = _bidx(n_days)
        decay_rates = [0.999, 0.9988, 0.9993, 0.9985, 0.9990, 0.9980]
        symbols = [f"S{i}" for i in range(6)]
        price_df = pd.DataFrame({
            sym: 100.0 * (rate ** np.arange(n_days))
            for sym, rate in zip(symbols, decay_rates)
        }, index=idx)
        return price_df, symbols

    def test_dual_momentum_de_risks_vs_relative_strength(self):
        price_df, symbols = self._downtrend_universe()

        rel = backtest_rotation(price_df, symbols, strategy_id="relative_strength")
        dual = backtest_rotation(price_df, symbols, strategy_id="dual_momentum")

        rel_total = (1 + rel.strategy_returns).prod() - 1
        dual_total = (1 + dual.strategy_returns).prod() - 1

        self.assertLess(rel_total, -0.01)
        self.assertAlmostEqual(dual_total, 0.0, places=6)
        self.assertGreater(dual_total, rel_total)

    def test_universe_smaller_than_top_n(self):
        idx = _bidx(400)
        symbols = ["A", "B", "C"]  # fewer than TOP_N (5)
        price_df = pd.DataFrame({
            sym: 100.0 * (1.0002 ** np.arange(400)) for sym in symbols
        }, index=idx)

        result = backtest_rotation(price_df, symbols, strategy_id="relative_strength", top_n=5)
        self.assertEqual(len(result.strategy_returns), len(price_df) - 253)
        self.assertFalse(result.strategy_returns.isna().any())

    def test_unscorable_symbol_excluded_from_universe(self):
        idx = _bidx(400)
        rng = np.random.default_rng(5)
        rets = rng.normal(0.0004, 0.01, size=399)
        good = pd.Series(np.concatenate([[100.0], 100.0 * np.cumprod(1 + rets)]), index=idx)
        all_nan = pd.Series([np.nan] * 400, index=idx)

        price_df = pd.DataFrame({"GOOD": good, "BAD": all_nan})
        scores = score_universe(price_df, ["GOOD", "BAD"])
        self.assertIn("GOOD", scores)
        self.assertNotIn("BAD", scores)

        spreads = sma_score_universe(price_df, ["GOOD", "BAD"])
        self.assertIn("GOOD", spreads)
        self.assertNotIn("BAD", spreads)

    def test_all_three_strategies_run_and_match_length(self):
        price_df, symbols = self._downtrend_universe()

        rel = backtest_rotation(price_df, symbols, strategy_id="relative_strength")
        dual = backtest_rotation(price_df, symbols, strategy_id="dual_momentum")
        sma = backtest_rotation(price_df, symbols, strategy_id="sma_trend")

        lengths = {len(rel.strategy_returns), len(dual.strategy_returns), len(sma.strategy_returns)}
        self.assertEqual(len(lengths), 1)
        expected_len = len(price_df) - (252 + 1)
        self.assertEqual(lengths.pop(), expected_len)


class TestBacktestTimeseries(unittest.TestCase):
    def test_sma_trend_flat_then_invested(self):
        close = _crossover_series()
        result = backtest_timeseries(close, strategy_id="sma_trend")

        self.assertEqual(result.trade_signal.iloc[0], 0.0)
        self.assertEqual(result.strategy_returns.iloc[0], 0.0)

        self.assertEqual(result.trade_signal.iloc[-1], 1.0)
        self.assertNotEqual(result.strategy_returns.iloc[-1], 0.0)

    def test_absolute_hurdle_dual_momentum_positive(self):
        hurdle = absolute_hurdle("dual_momentum")
        self.assertIsNotNone(hurdle)
        self.assertGreater(hurdle, 0.0)
        self.assertIsNone(absolute_hurdle("relative_strength"))
        self.assertIsNone(absolute_hurdle("sma_trend"))


if __name__ == "__main__":
    unittest.main()
