"""Factor warmer: runs on boot, re-warms periodically, tolerates failures."""
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
import factors  # noqa: E402
import signal_lab  # noqa: E402


class TestFactorWarmer(unittest.TestCase):
    def setUp(self):
        self._orig_db = db.DB_PATH
        db.DB_PATH = _DB_PATH
        db.init_db()
        with db.get_conn() as conn:
            conn.execute("DELETE FROM api_cache WHERE provider = 'internal' "
                         "AND key = 'factors_warmer_lock'")
            conn.execute("DELETE FROM api_cache WHERE provider = 'sp500' "
                         "AND key = 'constituents'")

    def tearDown(self):
        db.DB_PATH = self._orig_db

    def test_warm_factors_calls_load_factor_returns(self):
        dummy_df = pd.DataFrame({"Mkt-RF": [0.01], "SMB": [0.0], "HML": [0.0],
                                 "RMW": [0.0], "CMA": [0.0], "Mom": [0.0], "RF": [0.0001]})
        with patch.object(factors, "load_factor_returns",
                          return_value=(dummy_df, "2026-10-24")) as mock_load:
            app_module._warm_factors(background=False)
        mock_load.assert_called_once()

    def test_warm_factors_respects_cross_worker_lock(self):
        # Claim lock first
        claimed = db.try_claim_lock("factors_warmer_lock", ttl_hours=6.0)
        self.assertTrue(claimed)

        with patch.object(factors, "load_factor_returns") as mock_load:
            app_module._warm_factors(background=False)
        mock_load.assert_not_called()

    def test_warm_factors_handles_exception_gracefully(self):
        with patch.object(factors, "load_factor_returns",
                          side_effect=RuntimeError("Network timeout")):
            # Should not raise
            app_module._warm_factors(background=False)

    def test_start_factor_warmer_skips_when_env_disabled(self):
        with patch.dict(os.environ, {"FACTORS_WARM_ON_BOOT": "0"}), \
             patch.object(app_module, "_warm_factors") as mock_warm:
            app_module._start_factor_warmer()
        mock_warm.assert_not_called()

    def test_api_warm_cache_triggers_factor_warmer(self):
        with patch.dict(os.environ, {"WARM_CACHE_TOKEN": "secret-token"}), \
             patch.object(app_module, "_warm_options_cache") as mock_options_warm, \
             patch.object(app_module, "_warm_factors") as mock_factors_warm:
            client = app_module.app.test_client()
            res = client.post(
                "/api/warm-cache",
                headers={"Authorization": "Bearer secret-token"},
            )
            self.assertEqual(res.status_code, 202)
            mock_options_warm.assert_called_once_with(force=True)
            mock_factors_warm.assert_called_once_with(background=True)

    def test_cached_universe_prices_uses_stale_fallback(self):
        # Insert a stale constituent row older than 24h
        old_time = "2020-01-01T00:00:00"
        with db.get_conn() as conn:
            conn.execute(
                "INSERT INTO api_cache (provider, key, fetched_at, payload) "
                "VALUES ('sp500', 'constituents', ?, '[{\"symbol\": \"AAPL\"}, {\"symbol\": \"GOOG\"}]')",
                (old_time,)
            )

        with patch.object(db, "get_prices_batch", return_value={}) as mock_get_batch:
            signal_lab._cached_universe_prices(ticker="MSFT")

        # Must include ticker + constituents from stale cache
        mock_get_batch.assert_called_once_with(["AAPL", "GOOG", "MSFT"])
