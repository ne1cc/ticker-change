"""Verify /api/institutional caches its (expensive) backtest+permutation
computation instead of recomputing on every request."""
import os
import unittest
from unittest.mock import patch


class TestInstitutionalCaching(unittest.TestCase):

    def setUp(self):
        import db
        # Importing app is what creates the schema in normal operation; call
        # init_db explicitly too so this file passes when run standalone
        # against a fresh DB (otherwise: no such table: api_cache).
        import app as app_module

        db.init_db()

        # No network (or a rate-limited yfinance) means the route 404s on
        # missing prices -- that's an environment problem, not a regression.
        prices = app_module.get_or_fetch_prices("AAPL", period="2y")
        if prices is None or prices.empty:
            self.skipTest("no AAPL price data available (offline or rate-limited)")

        # Evict any pre-existing cache entry for this ticker so the test is
        # idempotent across repeated local/CI runs within the same 24h TTL
        # window -- without this, a rerun on the same day would start from
        # a warm cache and never observe the "first request populates the
        # cache" half of the behavior being tested. The lock row goes too,
        # so a leftover claim can't send this run down the wait-for-peer path.
        with db.get_conn() as conn:
            conn.execute(
                "DELETE FROM api_cache WHERE provider = ? AND key = ?",
                (app_module._INSTITUTIONAL_CACHE_PROVIDER, "AAPL"),
            )
            conn.execute(
                "DELETE FROM api_cache WHERE provider = 'internal' AND key = ?",
                ("institutional_backtest:AAPL",),
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
        with patch.dict(os.environ, {"WARM_CACHE_TOKEN": "test-token"}), \
             patch("backtest_engine.run_signals_backtest", side_effect=counting_run):
            resp1 = client.get(
                "/api/institutional/AAPL",
                headers={"Authorization": "Bearer test-token"})
            first_count = call_count["n"]
            resp2 = client.get(
                "/api/institutional/AAPL",
                headers={"Authorization": "Bearer test-token"})
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


    def test_cache_row_with_unexpected_shape_is_treated_as_a_miss(self):
        """A cached payload missing an expected key (e.g. written by an older
        build before the field existed) must recompute, not KeyError -> 500."""
        import db
        from app import app, _INSTITUTIONAL_CACHE_PROVIDER

        db.cache_set(
            _INSTITUTIONAL_CACHE_PROVIDER, "AAPL", {"signals_backtest": {"stale": True}}
        )

        with patch.dict(os.environ, {"WARM_CACHE_TOKEN": "test-token"}):
            resp = app.test_client().get(
                "/api/institutional/AAPL",
                headers={"Authorization": "Bearer test-token"})
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertIn("permutation_test", body)
        self.assertNotIn("stale", body["signals_backtest"])


class TestInstitutionalAuth(unittest.TestCase):
    """P0: /api/institutional triggers seconds of CPU per cold ticker; it must
    not be an unauthenticated amplification endpoint."""

    def setUp(self):
        import db
        db.init_db()

    def _client(self):
        from app import app
        return app.test_client()

    def test_disabled_when_no_token_configured(self):
        from app import app
        env = {k: v for k, v in os.environ.items() if k != "WARM_CACHE_TOKEN"}
        with patch.dict(os.environ, env, clear=True):
            os.environ["WARM_CACHE_TOKEN"] = ""
            resp = self._client().get("/api/institutional/AAPL")
        self.assertEqual(resp.status_code, 503)
        self.assertIn("error", resp.get_json())

    def test_wrong_token_401(self):
        from app import app
        with patch.dict(os.environ, {"WARM_CACHE_TOKEN": "test-token"}):
            resp = self._client().get("/api/institutional/AAPL")
        self.assertEqual(resp.status_code, 401)

    def test_correct_token_passes_gate(self):
        import app as app_module
        with patch.dict(os.environ, {"WARM_CACHE_TOKEN": "test-token"}), \
             patch.object(app_module, "_institutional_backtest",
                          return_value={"signals_backtest": {}, "permutation_test": {}}):
            resp = self._client().get(
                "/api/institutional/AAPL",
                headers={"Authorization": "Bearer test-token"})
        self.assertEqual(resp.status_code, 200)


if __name__ == "__main__":
    unittest.main()
