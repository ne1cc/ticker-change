"""Earnings history normalization and caching."""
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

import earnings  # noqa: E402


def _frame():
    idx = pd.DatetimeIndex(["2025-01-30", "2025-04-24", "2026-07-30"])
    return pd.DataFrame(
        {"EPS Estimate": [1.5, 1.6, 1.7], "Reported EPS": [1.7, 1.4, None],
         "Surprise(%)": [13.3, -12.5, None]}, index=idx)


class TestEarningsHistory(unittest.TestCase):
    def test_normalization_and_upcoming_flag(self):
        with patch.object(earnings.yf, "Ticker", MagicMock()) as mock:
            mock.return_value.earnings_dates = _frame()
            rows = earnings.fetch_earnings_history("aapl")
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0], {"date": "2025-01-30", "eps_estimate": 1.5,
                                   "eps_actual": 1.7, "surprise_pct": 13.3,
                                   "is_upcoming": False})
        self.assertTrue(rows[2]["is_upcoming"])
        self.assertIsNone(rows[2]["eps_actual"])
        self.assertEqual([r["date"] for r in rows], sorted(r["date"] for r in rows))

    def test_malformed_rows_skipped(self):
        df = pd.concat([_frame(), pd.DataFrame(
            {"EPS Estimate": [None, None], "Reported EPS": [None, 1.0],
             "Surprise(%)": [None, None]},
            index=pd.DatetimeIndex(["2026-10-15", pd.NaT]))])
        with patch.object(earnings.yf, "Ticker", MagicMock()) as mock:
            mock.return_value.earnings_dates = df
            rows = earnings.fetch_earnings_history("AAPL")
        self.assertEqual([r["date"] for r in rows],
                         ["2025-01-30", "2025-04-24", "2026-07-30", "2026-10-15"])

    def test_none_on_empty_or_error(self):
        with patch.object(earnings.yf, "Ticker", MagicMock()) as mock:
            mock.return_value.earnings_dates = None
            self.assertIsNone(earnings.fetch_earnings_history("AAPL"))
        with patch.object(earnings.yf, "Ticker", side_effect=RuntimeError("x")):
            self.assertIsNone(earnings.fetch_earnings_history("AAPL"))

    def test_get_earnings_history_caches(self):
        with patch.object(earnings.yf, "Ticker", MagicMock()) as mock:
            mock.return_value.earnings_dates = _frame()
            first = earnings.get_earnings_history("NVDA")
            second = earnings.get_earnings_history("NVDA")
        self.assertEqual(first, second)
        self.assertEqual(mock.call_count, 1)


import numpy as np


def _price_frame(days=400, base=100.0, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=days)
    close = base * np.exp(np.cumsum(rng.normal(0.0005, 0.012, days)))
    return pd.DataFrame({"close": close}, index=idx)


def _session_frame():
    # 40 business days; no holidays — sessions == rows
    idx = pd.bdate_range("2025-01-01", periods=40)
    close = pd.Series(100.0 + np.arange(40), index=idx, name="close")
    df = pd.DataFrame({"close": close,
                       "open": close - 1.0,
                       "volume": pd.Series([1_000_000] * 40, index=idx)})
    return df


class TestReactionSession(unittest.TestCase):
    def test_maps_weekend_announcement_to_next_session(self):
        df = _session_frame()
        # 2025-01-04 is a Saturday → reaction is Monday 2025-01-06
        t0, alignment = earnings.reaction_session(df, "2025-01-04")
        self.assertEqual(t0, df.index[3])
        self.assertEqual(alignment, "unverified")

    def test_on_session_date_maps_to_same_session(self):
        df = _session_frame()
        t0, _ = earnings.reaction_session(df, df.index[10].date().isoformat())
        self.assertEqual(t0, df.index[10])

    def test_no_data_when_date_precedes_history(self):
        df = _session_frame()
        t0, alignment = earnings.reaction_session(df, "2020-01-01")
        self.assertIsNone(t0)
        self.assertEqual(alignment, "no-data")


class TestEventMetrics(unittest.TestCase):
    def test_sessions_not_calendar_days(self):
        df = _session_frame()
        t0 = df.index[10].date().isoformat()
        m = earnings.event_metrics(df, t0)
        # bdate range: 10 sessions ahead exist, ~14 calendar days
        self.assertEqual(m["sessions_since"], 29)
        self.assertIsNotNone(m["drift_5d"])      # 5 sessions after t0 exist
        self.assertIsNotNone(m["drift_10d"])      # 10 sessions after t0 exist
        self.assertIsNotNone(m["drift_20d"])      # 29 sessions after t0: 20 exist

    def test_incomplete_window_stays_none(self):
        df = _session_frame().iloc[:15]  # t0 at 10 → only 4 sessions after
        m = earnings.event_metrics(df, df.index[10].date().isoformat())
        self.assertIsNone(m["drift_5d"])
        self.assertIsNone(m["drift_20d"])
        self.assertIsNotNone(m["drift_to_date"])
        self.assertEqual(m["drift_to_date_sessions"], 4)

    def test_reaction_and_gap_exclude_prior_day_from_drift(self):
        df = _session_frame()
        t0_iso = df.index[10].date().isoformat()
        m = earnings.event_metrics(df, t0_iso)
        # closes: 100+i; reaction day close=110, prior 109
        self.assertAlmostEqual(m["reaction_ret"], 110.0 / 109.0 - 1, places=6)
        self.assertAlmostEqual(m["gap_pct"], 109.0 / 109.0 - 1, places=6)  # open = close-1
        self.assertAlmostEqual(m["drift_5d"], 115.0 / 110.0 - 1, places=6)

    def test_rel_volume_excludes_reaction_day(self):
        df = _session_frame()
        t0_iso = df.index[10].date().isoformat()
        m = earnings.event_metrics(df, t0_iso)
        self.assertAlmostEqual(m["rel_volume"], 1.0, places=6)


class TestOwnEventHistory(unittest.TestCase):
    def test_completed_windows_only_and_separate_beats_misses(self):
        df = _session_frame()
        # events at session 5 (beat) and 15 (miss); both have 10 completed
        # sessions after within the 40-row frame? 15+10=25 <= 39 yes; 5+10=15 yes
        events = [
            {"date": df.index[5].date().isoformat(), "surprise_pct": 4.0, "is_upcoming": False},
            {"date": df.index[15].date().isoformat(), "surprise_pct": -3.0, "is_upcoming": False},
            {"date": df.index[35].date().isoformat(), "surprise_pct": 9.0, "is_upcoming": False},  # unfinished
            {"date": "2026-01-01", "surprise_pct": 1.0, "is_upcoming": True},                       # future
        ]
        hist = earnings.own_event_history(df, events, horizon=10)
        self.assertEqual(hist["beat"]["n"], 1)
        self.assertEqual(hist["miss"]["n"], 1)
        self.assertNotIn("2026", str(hist))

    def test_limited_history_state(self):
        df = _session_frame()
        events = [{"date": df.index[5].date().isoformat(),
                   "surprise_pct": 4.0, "is_upcoming": False}]
        hist = earnings.own_event_history(df, events, horizon=10)
        self.assertEqual(hist["beat"]["n"], 1)
        self.assertEqual(hist["state"], "limited history")


class TestComputeEarningsEvents(unittest.TestCase):
    def test_car_attached_and_upcoming_skipped(self):
        df = _price_frame()
        bench = _price_frame(days=400, base=500.0, seed=3)
        events = [
            {"date": "2024-06-03", "surprise_pct": 5.0, "is_upcoming": False},
            {"date": "2030-01-15", "surprise_pct": None, "is_upcoming": True},
        ]
        rows = earnings.compute_earnings_events(df, bench, events, ticker="TEST")
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertIn("car", row)
        self.assertIn("car_p_value", row)
        self.assertIn("drift_5d", row)
        self.assertIn("drift_20d", row)

    def test_summary_means(self):
        events = [
            {"surprise_pct": 5.0, "car": 0.10, "is_significant_95": True},
            {"surprise_pct": -4.0, "car": -0.05, "is_significant_95": False},
        ]
        s = earnings.summarize_earnings_drift(events)
        self.assertEqual(s["n_events"], 2)
        self.assertEqual(s["n_significant"], 1)
        self.assertAlmostEqual(s["avg_car_beat"], 0.10, places=6)
        self.assertAlmostEqual(s["avg_car_miss"], -0.05, places=6)

    def test_summary_empty_is_safe(self):
        s = earnings.summarize_earnings_drift([])
        self.assertEqual(s["n_events"], 0)
        self.assertIsNone(s["avg_car_beat"])


class TestImpliedMove(unittest.TestCase):
    def test_scales_straddle_to_earnings_dte(self):
        oa = {"expected_move_straddle": 10.0, "days_to_expiration": 40}
        out = earnings.implied_event_move(
            oa, (pd.Timestamp.today() + pd.Timedelta(days=10)).date().isoformat())
        self.assertAlmostEqual(out["implied_move"], 5.0, places=6)
        self.assertEqual(out["days_to_earnings"], 10)
        self.assertEqual(out["basis_days_to_exp"], 40)

    def test_none_when_missing_inputs(self):
        self.assertIsNone(earnings.implied_event_move(None, "2026-10-15"))
        self.assertIsNone(earnings.implied_event_move(
            {"expected_move_straddle": None, "days_to_expiration": 30}, "2026-10-15"))
        past = (pd.Timestamp.today() - pd.Timedelta(days=3)).date().isoformat()
        self.assertIsNone(earnings.implied_event_move(
            {"expected_move_straddle": 5, "days_to_expiration": 30}, past))


class TestEventVolFlag(unittest.TestCase):
    def test_rich_cheap_fair(self):
        self.assertEqual(earnings.event_vol_flag(2.0, 1.0), "rich")
        self.assertEqual(earnings.event_vol_flag(0.5, 1.0), "cheap")
        self.assertEqual(earnings.event_vol_flag(1.0, 1.0), "fair")
        self.assertIsNone(earnings.event_vol_flag(None, 1.0))
        self.assertIsNone(earnings.event_vol_flag(1.0, None))


class TestRealizedMoves(unittest.TestCase):
    def test_median_absolute_print_day_move(self):
        idx = pd.bdate_range("2024-01-01", periods=10)
        close = pd.Series([100, 100, 104, 104, 104, 104, 99, 99, 99, 99],
                          index=idx, name="close")
        df = pd.DataFrame({"close": close})
        events = [{"date": idx[1].date().isoformat()},   # +4.0%
                  {"date": idx[5].date().isoformat()}]   # -4.8%
        med = earnings.realized_earnings_moves(df, events)
        self.assertIsNotNone(med)
        self.assertGreater(med, 0.03)
        self.assertLess(med, 0.05)

    def test_median_uses_most_recent_prints(self):
        idx = pd.bdate_range("2024-01-01", periods=32)
        close = [100.0]
        for i in range(1, 32):
            if i % 2 == 1 and i < 17:  # reactions to the 8 oldest prints: huge
                close.append(close[-1] * (1.4 if (i // 2) % 2 == 0 else 0.6))
            else:                      # reactions to the 8 newest prints: +1%
                close.append(close[-1] * 1.01)
        df = pd.DataFrame({"close": close}, index=idx)
        events = [{"date": idx[2 * j].date().isoformat()} for j in range(16)]
        events.append({"date": idx[31].date().isoformat(), "is_upcoming": True})
        med = earnings.realized_earnings_moves(df, events)
        self.assertAlmostEqual(med, 0.01, places=4)
        self.assertLess(med, 0.05)


import decide as decide_module


class TestChecklistSurpriseContext(unittest.TestCase):
    def test_last_surprise_read_from_cache(self):
        db.cache_set("yfinance", "earnings:TICKR", [
            {"date": "2025-01-30", "eps_estimate": 1.5, "eps_actual": 1.7,
             "surprise_pct": 13.3, "is_upcoming": False},
            {"date": "2025-04-24", "eps_estimate": 1.6, "eps_actual": 1.4,
             "surprise_pct": -12.5, "is_upcoming": False},
        ])
        out = decide_module._last_surprise("TICKR")
        self.assertEqual(out["surprise_pct"], -12.5)
        self.assertEqual(out["date"], "2025-04-24")

    def test_check_reason_carries_beat_context(self):
        db.cache_set("yfinance", "earnings:TICKR2", [
            {"date": "2025-01-30", "eps_estimate": 1.0, "eps_actual": 1.2,
             "surprise_pct": 20.0, "is_upcoming": False},
        ])
        with patch("app.get_fundamentals", return_value=None), \
             patch("app.compute_options_analysis", return_value=None), \
             patch.object(decide_module, "days_until_earnings", return_value=30):
            checks = decide_module.build_checklist("TICKR2", {}, "long_stock")
        row = next(c for c in checks["checks"] if c["key"] == "earnings")
        self.assertEqual(row["status"], "pass")
        self.assertIn("last print beat by 20.0%", row["reason"])

    def test_check_without_history_unchanged(self):
        with patch("app.get_fundamentals", return_value=None), \
             patch("app.compute_options_analysis", return_value=None), \
             patch.object(decide_module, "days_until_earnings", return_value=10):
            checks = decide_module.build_checklist("TICKR3", {}, "long_stock")
        row = next(c for c in checks["checks"] if c["key"] == "earnings")
        self.assertEqual(row["reason"], "Earnings in 10d")
