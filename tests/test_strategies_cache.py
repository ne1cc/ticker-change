"""Universe-score cache for /strategies: one scoring pass per strategy per TTL."""
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import pandas as pd

import db

_fd, _DB_PATH = tempfile.mkstemp(suffix=".db")
os.close(_fd)
db.DB_PATH = _DB_PATH
db.init_db()

import app as app_module  # noqa: E402
import momentum_engine  # noqa: E402


def _score(symbol):
    return momentum_engine.MomentumScore(
        symbol=symbol, mom_12_1=0.12, mom_6m=0.06, mom_3m=0.03, mom_1m=0.01,
        ann_vol_1y=0.2, risk_adj=0.6, passes_absolute=True)


def _price_frame(days=300):
    idx = pd.bdate_range("2024-06-03", periods=days)
    return pd.DataFrame({
        "AAA": 100 + np.arange(days) * 0.05,
        "BBB": 100 - np.arange(days) * 0.02,
    }, index=idx)


import numpy as np  # noqa: E402


class TestCachedUniverseScores(unittest.TestCase):
    def setUp(self):
        with db.get_conn() as conn:
            conn.execute("DELETE FROM api_cache WHERE provider = 'strategies'")
            conn.execute("DELETE FROM api_cache WHERE provider = 'internal' "
                         "AND key LIKE 'strategies_scores_lock%'")

    def test_miss_computes_and_second_call_hits_cache(self):
        df = _price_frame()
        real = momentum_engine.score_universe
        with patch.object(app_module.momentum_engine, "score_universe",
                          side_effect=lambda pdf, syms, hurdle: {
                              "AAA": _score("AAA"), "BBB": _score("BBB")}) as mock:
            first = app_module._cached_universe_scores(
                "relative_strength", ["AAA", "BBB"], df)
            second = app_module._cached_universe_scores(
                "relative_strength", ["AAA", "BBB"], df)
        self.assertEqual(mock.call_count, 1)
        self.assertEqual(first[0]["AAA"].mom_12_1, second[0]["AAA"].mom_12_1)
        self.assertEqual(second[0]["AAA"].symbol, "AAA")
        self.assertEqual(real, momentum_engine.score_universe)

    def test_strategies_use_separate_cache_keys(self):
        df = _price_frame()
        with patch.object(app_module.momentum_engine, "score_universe",
                          side_effect=lambda pdf, syms, hurdle: {"AAA": _score("AAA")}):
            app_module._cached_universe_scores("relative_strength", ["AAA"], df)
            app_module._cached_universe_scores("dual_momentum", ["AAA"], df)
        self.assertIsNotNone(db.cache_get("strategies", "scores:relative_strength", 1))
        self.assertIsNotNone(db.cache_get("strategies", "scores:dual_momentum", 1))

    def test_malformed_payload_recomputes(self):
        df = _price_frame()
        db.cache_set("strategies", "scores:relative_strength", "garbage")
        with patch.object(app_module.momentum_engine, "score_universe",
                          side_effect=lambda pdf, syms, hurdle: {"AAA": _score("AAA")}):
            scores, spreads = app_module._cached_universe_scores(
                "relative_strength", ["AAA"], df)
        self.assertIn("AAA", scores)
        self.assertEqual(spreads, {})

    def test_sma_spreads_cached_and_restored(self):
        df = _price_frame()
        with patch.object(app_module.momentum_engine, "score_universe",
                          side_effect=lambda pdf, syms, hurdle: {"AAA": _score("AAA")}), \
             patch.object(app_module.momentum_engine, "sma_score_universe",
                          side_effect=lambda pdf, syms: {"AAA": 0.02}):
            app_module._cached_universe_scores("sma_trend", ["AAA"], df)
        with patch.object(app_module.momentum_engine, "sma_score_universe",
                          side_effect=lambda pdf, syms: {"AAA": 0.02}) as sma_mock:
            scores, spreads = app_module._cached_universe_scores(
                "sma_trend", ["AAA"], df)
        self.assertEqual(sma_mock.call_count, 0)  # served from cache
        self.assertEqual(spreads, {"AAA": 0.02})

    def test_lock_loser_falls_back_to_computing(self):
        df = _price_frame()
        # hold the lock like another worker would
        self.assertTrue(db.try_claim_lock("strategies_scores_lock:relative_strength",
                                          ttl_hours=1.0))
        with patch.object(app_module.momentum_engine, "score_universe",
                          side_effect=lambda pdf, syms, hurdle: {"AAA": _score("AAA")}):
            scores, spreads = app_module._cached_universe_scores(
                "relative_strength", ["AAA"], df)
        self.assertIn("AAA", scores)
        self.assertEqual(spreads, {})


if __name__ == "__main__":
    unittest.main()
