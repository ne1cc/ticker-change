"""Tests for percentile engine with peer-count guard and trend (Issue #118)."""
import os
import tempfile
import unittest

import db
import percentiles
import snapshot_store
import universe_metrics


class TestPercentileEngine(unittest.TestCase):
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

    def test_known_toy_distribution_exact_percentiles(self):
        """Known toy distribution returns exact percentiles, verified by test."""
        # Toy sample of 5 items: [10, 20, 30, 40, 50]
        pool = [10.0, 20.0, 30.0, 40.0, 50.0]

        # Method 'mean' (mid-rank percentile):
        # 10: (0 + 0.5) / 5 = 10.0%
        # 20: (1 + 0.5) / 5 = 30.0%
        # 30: (2 + 0.5) / 5 = 50.0%
        # 40: (3 + 0.5) / 5 = 70.0%
        # 50: (4 + 0.5) / 5 = 90.0%
        self.assertEqual(percentiles.percentile_rank(pool, 10.0, method="mean"), 10.0)
        self.assertEqual(percentiles.percentile_rank(pool, 20.0, method="mean"), 30.0)
        self.assertEqual(percentiles.percentile_rank(pool, 30.0, method="mean"), 50.0)
        self.assertEqual(percentiles.percentile_rank(pool, 40.0, method="mean"), 70.0)
        self.assertEqual(percentiles.percentile_rank(pool, 50.0, method="mean"), 90.0)

        # Method 'rank' (empirical CDF):
        # 10: 1/5 = 20.0%
        # 30: 3/5 = 60.0%
        # 50: 5/5 = 100.0%
        self.assertEqual(percentiles.percentile_rank(pool, 10.0, method="rank"), 20.0)
        self.assertEqual(percentiles.percentile_rank(pool, 30.0, method="rank"), 60.0)
        self.assertEqual(percentiles.percentile_rank(pool, 50.0, method="rank"), 100.0)

        # Target between values
        self.assertEqual(percentiles.percentile_rank(pool, 25.0, method="mean"), 40.0)

        # NaN and None handling
        pool_with_nans = [10.0, None, 20.0, float("nan"), 30.0, 40.0, 50.0]
        self.assertEqual(percentiles.percentile_rank(pool_with_nans, 30.0, method="mean"), 50.0)
        self.assertIsNone(percentiles.percentile_rank(pool, None))
        self.assertIsNone(percentiles.percentile_rank(pool, float("nan")))

    def test_sector_with_fewer_peers_than_guard_returns_not_meaningful(self):
        """Sector with fewer peers than the guard returns 'not meaningful'."""
        snapshot = {
            "snapshot_id": "test-snap-01",
            "as_of": "2026-10-23",
            "by_metric": {
                "annualised_vol": {
                    "all": [15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0],
                    "sectors": {
                        "LargeSector": [20.0, 25.0, 30.0, 35.0, 40.0, 45.0],  # 6 peers >= 5
                        "TinySector": [15.0, 22.0, 35.0],  # 3 peers < MIN_PEERS_GUARD (5)
                    },
                }
            },
        }

        # Query LargeSector (peer count 6 >= 5 guard)
        res_large = percentiles.calculate_metric_percentile(
            "annualised_vol", 30.0, sector="LargeSector", snapshot=snapshot
        )
        self.assertEqual(res_large["sector"]["status"], "ok")
        self.assertIsNotNone(res_large["sector"]["percentile"])
        self.assertEqual(res_large["sector"]["peer_count"], 6)

        # Query TinySector (peer count 3 < 5 guard) -> returns 'not meaningful'
        res_tiny = percentiles.calculate_metric_percentile(
            "annualised_vol", 22.0, sector="TinySector", snapshot=snapshot
        )
        self.assertEqual(res_tiny["sector"]["status"], "not meaningful")
        self.assertIsNone(res_tiny["sector"]["percentile"])
        self.assertEqual(res_tiny["sector"]["peer_count"], 3)

    def test_trend_is_none_not_zero_when_no_prior_snapshot_exists(self):
        """Trend is None, not 0, when no prior snapshot exists."""
        snapshot = {
            "snapshot_id": "test-snap-01",
            "as_of": "2026-10-23",
            "by_metric": {
                "annualised_vol": {
                    "all": [15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0],
                    "sectors": {"Tech": [20.0, 25.0, 30.0, 35.0, 40.0, 45.0]},
                }
            },
        }
        # Publish this single snapshot
        snapshot_store.publish("universe_metrics", "test-snap-01", snapshot)

        # Only one snapshot exists in history
        history = snapshot_store.history("universe_metrics")
        self.assertEqual(len(history), 1)

        res = percentiles.calculate_metric_percentile(
            "annualised_vol", 30.0, sector="Tech", snapshot=snapshot
        )

        # Acceptance criterion: Trend is None, not 0!
        self.assertIsNone(res["universe"]["trend"])
        self.assertNotEqual(res["universe"]["trend"], 0)
        self.assertNotEqual(res["universe"]["trend"], 0.0)

        self.assertIsNone(res["sector"]["trend"])
        self.assertNotEqual(res["sector"]["trend"], 0)
        self.assertNotEqual(res["sector"]["trend"], 0.0)

    def test_direction_semantics(self):
        """Metric direction semantics correctly identified."""
        self.assertEqual(
            percentiles.get_metric_direction("annualised_vol"), "higher_is_riskier"
        )
        self.assertEqual(
            percentiles.get_metric_direction("max_drawdown"), "higher_is_riskier"
        )
        self.assertEqual(
            percentiles.get_metric_direction("sharpe"), "higher_is_better"
        )
        self.assertEqual(
            percentiles.get_metric_direction("momentum_12_1"), "higher_is_better"
        )
        self.assertEqual(
            percentiles.get_metric_direction("price_book"), "higher_is_more_expensive"
        )
