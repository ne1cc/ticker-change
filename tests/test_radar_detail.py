"""Radar drawer endpoint: cache-only reads, membership, contracts, 30-min cache."""
import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

import db

_fd, _DB_PATH = tempfile.mkstemp(suffix=".db")
os.close(_fd)
db.DB_PATH = _DB_PATH
db.init_db()

import app as app_module  # noqa: E402
import radar  # noqa: E402

SNAP = {
    "meta": {"snapshot_id": "S1", "published_at": "2026-10-04T10:00",
             "source_as_of": {"prices": "2026-10-04T09:00"},
             "metric_contract_version": 3, "value_score_version": 1},
    "drift": {"rows": [{"symbol": "AAA", "print_date": "2024-09-06",
                        "surprise_pct": 20.0, "car_30": 0.05}],
              "upcoming": [], "excluded": []},
    "value": {"rows": [{"symbol": "AAA", "fwd_pe": 10.0, "value_score": 88.0,
                        "fwd_pe_pct": 95.0}],
              "excluded": [{"symbol": "ZZZ", "reason": "no-fundamentals"}]},
    "squeeze": {"rows": [], "excluded": []},
}


class TestRadarDetail(unittest.TestCase):
    def setUp(self):
        with db.get_conn() as conn:
            conn.execute("DELETE FROM api_cache WHERE provider = 'radar'")

    def test_membership_metrics_contracts(self):
        idx = pd.bdate_range("2025-01-01", periods=60)
        close = 100 + (idx - idx[0]).days * 0.1
        db.store_prices("AAA", pd.DataFrame(
            {"Open": close, "High": close, "Low": close, "Close": close,
             "Volume": pd.Series([1_000_000] * 60, index=idx)}, index=idx))
        db.cache_set("yfinance", "earnings:AAA", [
            {"date": "2025-02-03", "eps_estimate": 1.0, "eps_actual": 1.2,
             "surprise_pct": 20.0, "is_upcoming": False}])
        with patch.object(radar, "current_snapshot", return_value=SNAP):
            resp = app_module.app.test_client().get("/api/radar/detail/AAA")
        body = resp.get_json()
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(body["memberships"],
                         [{"tab": "drift", "rank": 1}, {"tab": "value", "rank": 1}])
        self.assertIn("surprise_pct", body["contracts"])
        self.assertEqual(body["contracts"]["surprise_pct"]["term"],
                         "Surprise %")
        self.assertEqual(len(body["price"]["dates"]), 60)
        self.assertEqual(body["earnings_events"][0]["date"], "2025-02-03")

    def test_zero_network(self):
        db.store_prices("AAA", _mk_frame())
        with patch.object(radar, "current_snapshot", return_value=SNAP), \
             patch("yfinance.Ticker", side_effect=AssertionError("network!")), \
             patch.object(app_module.providers, "_finnhub_get",
                          side_effect=AssertionError("network!")), \
             patch.object(app_module.providers, "_fmp_get",
                          side_effect=AssertionError("network!")):
            resp = app_module.app.test_client().get("/api/radar/detail/AAA")
        self.assertEqual(resp.status_code, 200)


def _mk_frame():
    idx = pd.bdate_range("2025-01-01", periods=60)
    close = 100 + (idx - idx[0]).days * 0.1
    return pd.DataFrame({"Open": close, "High": close, "Low": close,
                         "Close": close,
                         "Volume": pd.Series([1_000_000] * 60, index=idx)},
                        index=idx)


class TestDetailCacheAnd404(unittest.TestCase):
    def test_unknown_symbol_404(self):
        with patch.object(radar, "current_snapshot", return_value=SNAP):
            resp = app_module.app.test_client().get("/api/radar/detail/NOPE")
        self.assertEqual(resp.status_code, 404)

    def test_second_hit_served_from_cache(self):
        db.store_prices("AAA", _mk_frame())
        with patch.object(radar, "current_snapshot", return_value=SNAP):
            app_module.app.test_client().get("/api/radar/detail/AAA")
            with patch("db.get_prices",
                       side_effect=AssertionError("should not re-read")) as gp:
                resp = app_module.app.test_client().get("/api/radar/detail/AAA")
        self.assertEqual(resp.status_code, 200)
        gp.assert_not_called()
