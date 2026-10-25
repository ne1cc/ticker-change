"""Type-specific windows, after-close event timing, and overlap collapse."""
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

import event_study  # noqa: E402
import sec_8k  # noqa: E402


def _frame(days=400, base=100.0, seed=7, jump_at=None, jump=0.0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=days)
    close = base * np.exp(np.cumsum(rng.normal(0.0005, 0.012, days)))
    if jump_at is not None:
        close[jump_at:] *= (1.0 + jump)
    return pd.DataFrame({"close": close}, index=idx)


class TestWindowsFor(unittest.TestCase):
    def test_known_types_have_short_announcement_plus_drift(self):
        for t, (ev, drift) in ((t, event_study.windows_for(t))
                               for t in ("earnings", "m&a", "restatement",
                                         "reg_fd", "management")):
            self.assertLess(ev[1] - ev[0], 10, t)
            self.assertIsNotNone(drift, t)
            self.assertGreater(drift[0], ev[1], t)  # drift starts after

    def test_unknown_type_keeps_legacy_window(self):
        ev, drift = event_study.windows_for("CORPORATE_ACTION")
        self.assertEqual(ev, (-10, 30))
        self.assertIsNone(drift)

    def test_case_insensitive(self):
        self.assertEqual(event_study.windows_for("Earnings"),
                         event_study.windows_for("earnings"))


class TestRunEventStudyWindows(unittest.TestCase):
    def test_explicit_window_beats_type_config(self):
        df = _frame()
        res = event_study.run_event_study(
            df, _frame(days=400, base=500.0, seed=3), "2024-09-03",
            event_type="earnings", event_window=(-10, 30))
        self.assertEqual(res.event_window, (-10, 30))
        self.assertIsNone(res.drift_window)
        self.assertIsNone(res.drift_car)

    def test_type_windows_applied_and_drift_separate(self):
        # stock is a scaled copy of the benchmark (AR ≡ 0), then jumps +8%
        # on the event day: the announcement window captures the whole jump,
        # the drift window (which starts after it) sees ~zero abnormal return.
        bench = _frame(days=400, base=500.0, seed=3)
        ev_idx = bench.index.get_loc(pd.to_datetime("2024-09-03"))
        stock = pd.DataFrame({"close": bench["close"] * 0.2},
                             index=bench.index)
        stock.iloc[ev_idx:, 0] *= 1.08
        ev_win, drift_win = event_study.windows_for("earnings")
        res = event_study.run_event_study(
            stock, bench, "2024-09-03", event_type="earnings",
            event_window=ev_win, drift_window=drift_win)
        self.assertEqual(res.event_window, (-1, 1))
        self.assertEqual(res.drift_window, (2, 21))
        self.assertGreater(res.car, 0.05)
        self.assertIsNotNone(res.drift_car)
        self.assertLess(abs(res.drift_car), 1e-6)

    def test_incomplete_drift_window_is_none(self):
        days = 200
        df = _frame(days=days)
        late = df.index[-13].date().isoformat()  # +21 drift sessions run off the end
        ev_win, drift_win = event_study.windows_for("earnings")
        res = event_study.run_event_study(
            df, _frame(days=days, base=500.0, seed=3), late,
            event_type="earnings", event_window=ev_win, drift_window=drift_win)
        self.assertIsNotNone(res)
        self.assertIsNone(res.drift_car)


class TestAfterCloseTiming(unittest.TestCase):
    def test_after_close_filing_shifts_to_next_session(self):
        stock = _frame()
        bench = _frame(days=400, base=500.0, seed=3)
        # 2024-09-06 is a Friday session; after-close reaction lands Monday
        res = event_study.run_event_study(
            stock, bench, "2024-09-06", event_type="earnings",
            after_close=True)
        monday = stock.index[stock.index.get_loc(
            pd.to_datetime("2024-09-06")) + 1]
        self.assertEqual(res.event_date, monday.date().isoformat())

    def test_without_flag_event_day_is_the_filing_session(self):
        stock = _frame()
        bench = _frame(days=400, base=500.0, seed=3)
        res = event_study.run_event_study(
            stock, bench, "2024-09-06", event_type="earnings")
        self.assertEqual(res.event_date, "2024-09-06")

    def test_after_close_with_no_later_session_is_none(self):
        stock = _frame(days=200)
        last = stock.index[-1].date().isoformat()
        res = event_study.run_event_study(
            stock, _frame(days=200, base=500.0, seed=3), last,
            event_type="earnings", after_close=True)
        self.assertIsNone(res)


class TestSec8KAfterCloseFlag(unittest.TestCase):
    _SUBMISSIONS = {
        "filings": {"recent": {
            "form": ["8-K", "8-K"],
            "filingDate": ["2025-03-03", "2025-03-04"],
            "reportDate": ["2025-02-28", "2025-03-03"],
            "accessionNumber": ["0001-01", "0002-02"],
            "primaryDocument": ["a.htm", "b.htm"],
            "items": ["2.02", "2.02"],
            "primaryDocDescription": ["8-K", "8-K"],
            "acceptanceDateTime": [
                "2025-03-03T16:05:00-05:00",   # after the close
                "2025-03-04T10:30:00-05:00"],  # during hours
        }},
    }

    def test_acceptance_time_sets_after_close(self):
        with patch.object(sec_8k.providers, "sec_cik_for_ticker",
                          return_value="0000320193"), \
             patch.object(sec_8k.providers, "_get_json",
                          return_value=self._SUBMISSIONS):
            events = sec_8k.fetch_and_parse_8k_filings("ZZAFTCL")
        by_date = {e.filing_date: e for e in events}
        self.assertTrue(by_date["2025-03-03"].after_close)
        self.assertFalse(by_date["2025-03-04"].after_close)

    def test_malformed_acceptance_is_not_after_close(self):
        subs = {"filings": {"recent": {
            "form": ["8-K"], "filingDate": ["2025-03-03"],
            "reportDate": ["2025-02-28"], "accessionNumber": ["0001-01"],
            "primaryDocument": ["a.htm"], "items": ["2.02"],
            "primaryDocDescription": ["8-K"],
            "acceptanceDateTime": ["not-a-date"]}}}
        with patch.object(sec_8k.providers, "sec_cik_for_ticker",
                          return_value="0000320193"), \
             patch.object(sec_8k.providers, "_get_json", return_value=subs):
            events = sec_8k.fetch_and_parse_8k_filings("ZZAFTC2")
        self.assertFalse(events[0].after_close)


class TestCollapseOverlapping(unittest.TestCase):
    def _ev(self, date, ticker="AAPL", etype="earnings", end=None):
        d = {"event_date": date, "ticker": ticker, "event_type": etype}
        if end:
            d["end_date"] = end
        return d

    def test_same_type_cluster_collapses_to_first(self):
        events = [self._ev("2024-03-04"), self._ev("2024-03-06"),
                  self._ev("2024-03-08")]
        out = event_study.collapse_overlapping(events, span_days=60)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["event_date"], "2024-03-04")

    def test_resolved_end_date_shrinks_the_absorb_span(self):
        events = [self._ev("2024-03-04", end="2024-03-05"),
                  self._ev("2024-03-08")]
        out = event_study.collapse_overlapping(events)
        self.assertEqual(len(out), 2)  # past the kept event's window end

    def test_different_types_or_tickers_both_survive(self):
        events = [self._ev("2024-03-04"),
                  self._ev("2024-03-05", etype="restatement"),
                  self._ev("2024-03-05", ticker="MSFT")]
        out = event_study.collapse_overlapping(events, span_days=60)
        self.assertEqual(len(out), 3)

    def test_deterministic_regardless_of_input_order_or_ties(self):
        events = [self._ev("2024-03-08", ticker="B"), self._ev("2024-03-04"),
                  self._ev("2024-03-08", ticker="A"), self._ev("2024-03-06")]
        runs = [event_study.collapse_overlapping(list(reversed(events)))
                for _ in range(2)]
        self.assertEqual(runs[0], runs[1])
        # default span absorbs AAPL 03-06 into the kept 03-04; per-ticker
        # clusters keep each ticker's first event, ties broken by ticker
        self.assertEqual([(e["event_date"], e["ticker"]) for e in runs[0]],
                         [("2024-03-04", "AAPL"), ("2024-03-08", "A"),
                          ("2024-03-08", "B")])


class TestAppHelperPlumbing(unittest.TestCase):
    def test_helper_passes_type_windows_and_after_close(self):
        import app as app_module
        from sec_8k import MaterialEvent8K
        event = MaterialEvent8K(
            accession_number="x", cik="1", ticker="TEST",
            filing_date="2024-09-03", report_date="2024-09-03",
            items=["2.02"], item_descriptions=[], primary_doc_url="",
            event_type="earnings", after_close=True)
        with patch.object(event_study, "run_event_study",
                          wraps=event_study.run_event_study) as spy:
            app_module._attach_car_to_8k_events(
                [event], _frame(), _frame(days=400, base=500.0, seed=3), "TEST")
        call = spy.call_args_list[0]
        self.assertEqual(call.kwargs["event_window"], (-1, 1))
        self.assertEqual(call.kwargs["drift_window"], (2, 21))
        self.assertTrue(call.kwargs["after_close"])
