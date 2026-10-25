"""Tests for universe metric snapshot engine (Issue #116)."""
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import db
import snapshot_store
import universe_metrics


class TestUniverseMetrics(unittest.TestCase):
    def setUp(self):
        self._orig_db = db.DB_PATH
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.temp_db_path = path
        db.DB_PATH = path
        db.init_db()

    def tearDown(self):
        db.DB_PATH = self._orig_db
        try:
            os.remove(self.temp_db_path)
        except OSError:
            pass

    def _seed_ticker_prices(self, ticker: str, n_days: int = 260, base_price: float = 100.0):
        dates = pd.bdate_range(end="2026-10-23", periods=n_days)
        np.random.seed(42 + hash(ticker) % 1000)
        ret = np.random.normal(0.0005, 0.015, size=n_days)
        price_series = base_price * np.exp(np.cumsum(ret))
        df = pd.DataFrame(
            {
                "Open": price_series,
                "High": price_series * 1.01,
                "Low": price_series * 0.99,
                "Close": price_series,
                "Volume": [1_000_000] * n_days,
            },
            index=dates,
        )
        db.store_prices(ticker, df)
        return df

    def test_metric_definitions_contracts(self):
        """Each metric has a declared definition, window, and direction."""
        for m_key, contract in universe_metrics.METRIC_DEFINITIONS.items():
            self.assertIn("name", contract)
            self.assertIn("window_days", contract)
            self.assertIn("formula", contract)
            self.assertIn("direction", contract)
            self.assertIn("min_bars", contract)
            self.assertIn(contract["direction"], ("higher_is_riskier", "higher_is_better"))

    def test_insufficient_history_excluded_with_count_not_zero(self):
        """Constituents with insufficient history are excluded with a count, not silently zero."""
        # Seed ticker with 30 bars (< MIN_HISTORY_BARS = 63)
        self._seed_ticker_prices("SHORT_HIST", n_days=30)
        # Seed ticker with 260 bars (adequate)
        self._seed_ticker_prices("GOOD_HIST", n_days=260)

        gics_map = {
            "SHORT_HIST": {"symbol": "SHORT_HIST", "sector": "Technology"},
            "GOOD_HIST": {"symbol": "GOOD_HIST", "sector": "Technology"},
        }

        payload = universe_metrics.build_universe_metrics_snapshot(
            tickers=["SHORT_HIST", "GOOD_HIST"],
            gics_map=gics_map,
        )

        self.assertEqual(payload["total_constituents"], 2)
        self.assertEqual(payload["valid_count"], 1)
        self.assertEqual(payload["excluded_count"], 1)
        self.assertIn("SHORT_HIST", payload["excluded_tickers"])
        self.assertIn("insufficient history", payload["excluded_tickers"]["SHORT_HIST"].lower())
        self.assertNotIn("SHORT_HIST", payload["tickers"])
        self.assertIn("GOOD_HIST", payload["tickers"])

    def test_failed_run_leaves_last_good_serving(self):
        """Snapshot published atomically; failed run leaves last-good serving."""
        self._seed_ticker_prices("AAA", n_days=260)
        gics_map = {"AAA": {"symbol": "AAA", "sector": "Energy"}}

        # 1. First successful run publishes
        payload1 = universe_metrics.warm_universe_metrics(
            tickers=["AAA"], gics_map=gics_map
        )
        self.assertIsNotNone(payload1)
        first_id = payload1["snapshot_id"]

        current = universe_metrics.current_universe_metrics()
        self.assertIsNotNone(current)
        self.assertEqual(current["snapshot_id"], first_id)
        self.assertEqual(current["valid_count"], 1)

        # 2. Second run fails during computation / publication
        with patch.object(
            universe_metrics,
            "build_universe_metrics_snapshot",
            side_effect=RuntimeError("Upstream failure"),
        ):
            failed_res = universe_metrics.warm_universe_metrics(
                tickers=["AAA"], gics_map=gics_map
            )
            self.assertIsNone(failed_res)

        # 3. Last good snapshot still serves intact
        still_current = universe_metrics.current_universe_metrics()
        self.assertIsNotNone(still_current)
        self.assertEqual(still_current["snapshot_id"], first_id)
        self.assertEqual(still_current["tickers"]["AAA"]["_sector"], "Energy")
