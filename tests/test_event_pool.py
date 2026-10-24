"""Pooled universe CAR by event type: pooled stats, injected-AR detection,
atomic publication through snapshot_store, SEC fair-access pause."""
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import db

_fd, _DB_PATH = tempfile.mkstemp(suffix=".db")
os.close(_fd)
db.DB_PATH = _DB_PATH
db.init_db()

import event_pool  # noqa: E402
import event_study  # noqa: E402
from sec_8k import MaterialEvent8K  # noqa: E402

BENCH = None


def _bench(days=400, base=500.0, seed=3):
    global BENCH
    if BENCH is None:
        rng = np.random.default_rng(seed)
        idx = pd.bdate_range("2024-01-01", periods=days)
        BENCH = pd.DataFrame(
            {"close": base * np.exp(np.cumsum(rng.normal(0.0005, 0.012, days)))},
            index=idx)
    return BENCH


def _neutral_stock(ticker_seed, jump_at=None, jump=0.0):
    """Stock that is a scaled copy of the benchmark: abnormal return ≡ 0
    everywhere except an optional injected jump."""
    bench = _bench()
    close = bench["close"] * (0.2 + 0.01 * ticker_seed)
    if jump_at is not None:
        close = close.copy()
        close.iloc[jump_at:] *= (1.0 + jump)
    return pd.DataFrame({"close": close}, index=bench.index)


def _event(ticker, date, items=("1.01",), etype="m&a"):
    return MaterialEvent8K(
        accession_number=f"{ticker}-{date}", cik="1", ticker=ticker,
        filing_date=date, report_date=date, items=list(items),
        item_descriptions=[], primary_doc_url="", event_type=etype)


# Event dates late enough in the 400-session frame for the estimation
# window and both study legs (frame spans 2024-01-01 .. ~2025-07).
EVENT_DATE = "2024-09-03"


class TestComputePooledStats(unittest.TestCase):
    def test_cross_sectional_t_math(self):
        st = event_pool.compute_pooled_stats([0.01, 0.02, 0.03])
        self.assertEqual(st["n"], 3)
        self.assertAlmostEqual(st["mean_car"], 0.02, places=6)
        expected_t = 0.02 / (np.std([0.01, 0.02, 0.03], ddof=1) / np.sqrt(3))
        self.assertAlmostEqual(st["t_stat"], expected_t, places=3)
        self.assertLess(st["p_value"], 0.05)
        self.assertEqual(st["cars"], [0.01, 0.02, 0.03])

    def test_needs_at_least_two_events(self):
        self.assertIsNone(event_pool.compute_pooled_stats([0.01]))
        self.assertIsNone(event_pool.compute_pooled_stats([]))
        self.assertIsNone(event_pool.compute_pooled_stats(None))


class TestPercentileRank(unittest.TestCase):
    def test_empirical_cdf_with_ties(self):
        cars = [0.01, 0.02, 0.02, 0.03]
        self.assertEqual(event_pool.percentile_rank(cars, 0.02), 75.0)
        self.assertEqual(event_pool.percentile_rank(cars, 0.005), 0.0)
        self.assertEqual(event_pool.percentile_rank(cars, 0.05), 100.0)
        self.assertEqual(event_pool.percentile_rank(cars, 0.01), 25.0)
        self.assertEqual(event_pool.percentile_rank([], 0.01), 50.0)


class TestInjectedReturnDetected(unittest.TestCase):
    def test_pool_detects_injected_abnormal_returns(self):
        # 8 events, half carrying an injected +8% abnormal return on the
        # event day (neutral stocks are scaled benchmark copies, so their
        # abnormal return is identically ~0). A sparse single outlier can
        # never clear a cross-sectional t (it supplies its own variance);
        # a 50% injected fraction gives t = sqrt(N*f/(1-f)) ≈ 2.83.
        tickers = [f"NEUT{i}" for i in range(4)] + [f"INJTD{i}" for i in range(4)]
        ev_idx = _bench().index.get_loc(pd.to_datetime(EVENT_DATE))
        fetchers = {t: _neutral_stock(i) for i, t in enumerate(tickers)}
        for t in [t for t in tickers if t.startswith("INJ")]:
            fetchers[t] = _neutral_stock(9, jump_at=ev_idx, jump=0.08)
        events = {t: [_event(t, EVENT_DATE)] for t in tickers}

        payload = event_pool.pool_universe(
            tickers,
            stock_fetcher=lambda t: fetchers[t],
            benchmark_df=_bench(),
            events_fetcher=lambda t: events[t],
            pause=0.0)
        ma = payload["types"]["m&a"]
        self.assertEqual(ma["n"], 8)
        self.assertEqual(payload["meta"]["n_events"], 8)
        self.assertAlmostEqual(ma["mean_car"], 0.080 / 2, delta=0.005)
        self.assertGreater(ma["t_stat"], 2.0)
        self.assertLess(ma["p_value"], 0.05)
        # injected CARs occupy the top of the pooled distribution: each
        # sits at or above P50, the maximum reads P100
        injected_cars = [ma["cars"][-1], ma["cars"][-2], ma["cars"][-3], ma["cars"][-4]]
        for c in injected_cars:
            self.assertGreaterEqual(
                event_pool.pool_percentile(payload, "m&a", c)["percentile"], 50.0)
        self.assertEqual(
            event_pool.pool_percentile(payload, "m&a", ma["cars"][-1])["percentile"],
            100.0)

    def test_routine_and_missing_data_skipped(self):
        tickers = ["NEUT1", "NODATA"]
        fetchers = {"NEUT1": _neutral_stock(0), "NODATA": None}
        events = {
            "NEUT1": [_event("NEUT1", EVENT_DATE),
                      _event("NEUT1", "2024-09-04", items=("5.07",), etype=None)],
            "NODATA": [_event("NODATA", EVENT_DATE)],
        }
        payload = event_pool.pool_universe(
            tickers,
            stock_fetcher=lambda t: fetchers[t],
            benchmark_df=_bench(),
            events_fetcher=lambda t: events[t],
            pause=0.0)
        self.assertEqual(payload["meta"]["n_events"], 1)
        self.assertNotIn("NODATA", payload["meta"]["windows"])

    def test_sec_pause_slept_between_tickers(self):
        tickers = ["A1", "A2", "A3"]
        fetchers = {t: _neutral_stock(i) for i, t in enumerate(tickers)}
        events = {t: [_event(t, EVENT_DATE)] for t in tickers}
        with patch.object(event_pool.time, "sleep") as snooze:
            event_pool.pool_universe(
                tickers,
                stock_fetcher=lambda t: fetchers[t],
                benchmark_df=_bench(),
                events_fetcher=lambda t: events[t],
                pause=0.25)
        self.assertEqual(snooze.call_count, 2)  # between, not after the last
        snooze.assert_called_with(0.25)


class TestPublication(unittest.TestCase):
    def test_roundtrip_and_pointer_atomicity(self):
        payload = {"meta": {"built_at": "t", "n_events": 2},
                   "types": {"m&a": event_pool.compute_pooled_stats([0.01, 0.02])}}
        event_pool.publish_pool(payload)
        cur = event_pool.current_pool()
        self.assertEqual(cur["meta"]["n_events"], 2)
        # building a new pool does not disturb the pointer until publish
        with patch.object(event_pool, "publish_pool") as pub:
            event_pool.pool_universe(
                ["NEUT1"], stock_fetcher=lambda t: _neutral_stock(0),
                benchmark_df=_bench(),
                events_fetcher=lambda t: [_event("NEUT1", EVENT_DATE)],
                pause=0.0)
            pub.assert_not_called()
        self.assertEqual(event_pool.current_pool()["meta"]["n_events"], 2)
        # and a second publish supersedes atomically, with retention pruning
        event_pool.publish_pool({"meta": {"n_events": 3}, "types": {}})
        self.assertEqual(event_pool.current_pool()["meta"]["n_events"], 3)

    def test_pool_percentile_missing_states(self):
        self.assertIsNone(event_pool.pool_percentile(None, "m&a", 0.01))
        self.assertIsNone(event_pool.pool_percentile({"types": {}}, "m&a", 0.01))
        pool = {"types": {"m&a": {"n": 2, "cars": [0.0, 0.04]}}}
        self.assertIsNone(event_pool.pool_percentile(pool, "m&a", None))
        self.assertIsNone(event_pool.pool_percentile(pool, "earnings", 0.01))
