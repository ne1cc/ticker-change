"""Verify /api/institutional caches its (expensive) backtest+permutation
computation instead of recomputing on every request."""
import unittest
from unittest.mock import patch


class TestInstitutionalCaching(unittest.TestCase):

    def setUp(self):
        # Evict any pre-existing cache entry for this ticker so the test is
        # idempotent across repeated local/CI runs within the same 24h TTL
        # window -- without this, a rerun on the same day would start from
        # a warm cache and never observe the "first request populates the
        # cache" half of the behavior being tested.
        import db

        with db.get_conn() as conn:
            conn.execute(
                "DELETE FROM api_cache WHERE provider = ? AND key = ?",
                ("institutional_backtest", "AAPL"),
            )

    def test_second_call_within_ttl_does_not_recompute(self):
        from app import app
        import backtest_engine

        call_count = {"n": 0}
        real_run = backtest_engine.run_signals_backtest

        def counting_run(*args, **kwargs):
            call_count["n"] += 1
            return real_run(*args, **kwargs)

        client = app.test_client()
        with patch("backtest_engine.run_signals_backtest", side_effect=counting_run):
            resp1 = client.get("/api/institutional/AAPL")
            first_count = call_count["n"]
            resp2 = client.get("/api/institutional/AAPL")
            second_count = call_count["n"]

        self.assertEqual(resp1.status_code, 200)
        self.assertEqual(resp2.status_code, 200)
        self.assertGreater(first_count, 0)
        self.assertEqual(
            second_count, first_count,
            "second request within the cache TTL must not recompute the backtest",
        )

        # The permutation null test must actually run and be surfaced (and
        # served back out of the same cache entry on the second request) --
        # not just cached alongside an unused run_signals_backtest result.
        for resp in (resp1, resp2):
            body = resp.get_json()
            self.assertIn("permutation_test", body)
            permutation_test = body["permutation_test"]
            for field in (
                "real_sharpe", "null_mean", "null_std", "p_value", "n_permutations",
            ):
                self.assertIn(field, permutation_test)


if __name__ == "__main__":
    unittest.main()
