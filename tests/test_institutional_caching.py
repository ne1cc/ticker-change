"""Verify /api/institutional caches its (expensive) backtest+permutation
computation instead of recomputing on every request."""
import unittest
from unittest.mock import patch


class TestInstitutionalCaching(unittest.TestCase):

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


if __name__ == "__main__":
    unittest.main()
