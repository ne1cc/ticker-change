"""Tests for #155 and #156: event CAR percentile vs pooled distribution,
insignificance de-emphasis, and earnings surprise buckets."""
import os
import tempfile
import unittest
from unittest.mock import patch

import db

_fd, _DB_PATH = tempfile.mkstemp(suffix=".db")
os.close(_fd)
db.DB_PATH = _DB_PATH
db.init_db()

import earnings
import event_pool
from app import app
from flask import render_template


class TestEarningsSurpriseBuckets(unittest.TestCase):
    def test_buckets_assigned_by_surprise_rank(self):
        # 10 mock events with varying surprise and CAR
        events = [
            {"date": f"2024-0{i+1}-01", "surprise_pct": float(i * 5 - 20),
             "car": float(i * 0.01 - 0.04), "drift_20d": float(i * 0.02 - 0.08)}
            for i in range(10)
        ]
        res = earnings.earnings_surprise_buckets(events, n_buckets=5)
        self.assertTrue(res["available"])
        self.assertEqual(res["n_events"], 10)
        self.assertEqual(res["n_buckets"], 5)
        # Verify ordering: Q1 has lowest surprise, Q5 has highest
        self.assertLess(res["buckets"][0]["mean_surprise_pct"], res["buckets"][-1]["mean_surprise_pct"])
        self.assertEqual(res["buckets"][0]["drift_window"], (2, 21))
        # Each bucket has 2 events
        for b in res["buckets"]:
            self.assertEqual(b["n"], 2)

    def test_insufficient_events_declared(self):
        res = earnings.earnings_surprise_buckets([{"date": "2024-01-01", "surprise_pct": 5.0, "car": 0.02}])
        self.assertFalse(res["available"])
        self.assertIn("Insufficient", res["state"])


class TestPooledUiRendering(unittest.TestCase):
    def test_earnings_pool_percentile_and_insignificance_rendered(self):
        mock_pool = {
            "types": {
                "earnings": {"n": 50, "cars": [-0.05, 0.0, 0.02, 0.05, 0.10]}
            }
        }
        mock_events = [
            {"date": "2024-04-24", "surprise_pct": 5.0, "car": 0.03,
             "car_t_stat": 2.5, "car_p_value": 0.01, "is_significant_95": True,
             "drift_10d": 0.01, "drift_20d": 0.02,
             "pool": {"percentile": 60.0, "n": 50, "event_type": "earnings"}},
            {"date": "2024-01-24", "surprise_pct": -2.0, "car": -0.005,
             "car_t_stat": -0.4, "car_p_value": 0.65, "is_significant_95": False,
             "drift_10d": -0.01, "drift_20d": -0.01,
             "pool": {"percentile": 20.0, "n": 50, "event_type": "earnings"}}
        ]
        ern_data = {
            "events": mock_events,
            "summary": earnings.summarize_earnings_drift(mock_events),
            "next_print": None,
            "implied_move": None,
            "realized_move": 0.03,
            "event_vol": "fair",
            "surprise_buckets": earnings.earnings_surprise_buckets(mock_events, n_buckets=2),
        }
        with app.test_request_context("/analytics?ticker=TEST"):
            html = render_template("_earnings.html", data={"earnings": ern_data})

        # Base rate percentile rendered
        self.assertIn("P60", html)
        self.assertIn("(n=50)", html)
        # Insignificant event de-emphasized
        self.assertIn("opacity-50", html)
        # Surprise bucket rendered
        self.assertIn("PEAD", html)
        self.assertIn("Q1", html)

    def test_missing_pool_renders_dash(self):
        mock_events = [
            {"date": "2024-04-24", "surprise_pct": 5.0, "car": 0.03,
             "car_t_stat": 2.5, "car_p_value": 0.01, "is_significant_95": True,
             "drift_10d": 0.01, "drift_20d": 0.02,
             "pool": None}
        ]
        ern_data = {
            "events": mock_events,
            "summary": earnings.summarize_earnings_drift(mock_events),
            "next_print": None,
            "implied_move": None,
            "realized_move": 0.03,
            "event_vol": "fair",
            "surprise_buckets": {"available": False, "state": "Insufficient"},
        }
        with app.test_request_context("/analytics?ticker=TEST"):
            html = render_template("_earnings.html", data={"earnings": ern_data})
        self.assertIn("—", html)
        self.assertNotIn("P60", html)


if __name__ == "__main__":
    unittest.main()
