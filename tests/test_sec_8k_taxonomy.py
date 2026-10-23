"""8-K item taxonomy: every known item maps to a type or 'routine';
routine-only filings never trigger an event study."""
import os
import tempfile
import unittest
from unittest.mock import patch

import db

_fd, _DB_PATH = tempfile.mkstemp(suffix=".db")
os.close(_fd)
db.DB_PATH = _DB_PATH
db.init_db()

import sec_8k  # noqa: E402
import event_study  # noqa: E402


class TestTaxonomyCoverage(unittest.TestCase):
    def test_every_known_item_maps_to_a_type_or_routine(self):
        self.assertEqual(set(sec_8k.ITEM_DEFINITIONS),
                         set(sec_8k.ITEM_EVENT_TYPES))
        mapped = set(sec_8k.ITEM_EVENT_TYPES.values())
        self.assertTrue(mapped <= set(sec_8k.EVENT_TYPE_PRIORITY) | {"routine"})

    def test_shareholder_vote_and_other_events_are_routine(self):
        self.assertEqual(sec_8k.ITEM_EVENT_TYPES["5.07"], "routine")
        self.assertEqual(sec_8k.ITEM_EVENT_TYPES["8.01"], "routine")

    def test_core_items_map_to_their_type(self):
        self.assertEqual(sec_8k.ITEM_EVENT_TYPES["2.02"], "earnings")
        self.assertEqual(sec_8k.ITEM_EVENT_TYPES["4.02"], "restatement")
        self.assertEqual(sec_8k.ITEM_EVENT_TYPES["1.01"], "m&a")
        self.assertEqual(sec_8k.ITEM_EVENT_TYPES["7.01"], "reg_fd")


class TestResolveEventType(unittest.TestCase):
    def test_routine_only_resolves_to_none(self):
        self.assertIsNone(sec_8k.resolve_event_type(["5.07"]))
        self.assertIsNone(sec_8k.resolve_event_type(["8.01", "5.03"]))

    def test_mixed_items_resolve_by_priority_not_item_order(self):
        # restatement outranks everything; item order must not matter
        self.assertEqual(sec_8k.resolve_event_type(["5.07", "4.02"]), "restatement")
        self.assertEqual(sec_8k.resolve_event_type(["4.02", "5.07"]), "restatement")
        # earnings outranks reg_fd (release + slides pair)
        self.assertEqual(sec_8k.resolve_event_type(["7.01", "2.02"]), "earnings")

    def test_mixed_resolution_is_deterministic(self):
        items = ["2.01", "5.07", "1.01", "8.01"]
        results = {sec_8k.resolve_event_type(items) for _ in range(5)}
        self.assertEqual(results, {"m&a"})

    def test_empty_or_unknown_items_resolves_to_none(self):
        self.assertIsNone(sec_8k.resolve_event_type([]))
        self.assertIsNone(sec_8k.resolve_event_type(None))


class TestParseAssignsEventType(unittest.TestCase):
    _SUBMISSIONS = {
        "filings": {"recent": {
            "form": ["8-K", "8-K", "8-K"],
            "filingDate": ["2025-03-03", "2025-03-04", "2025-03-05"],
            "reportDate": ["2025-02-28", "2025-03-03", "2025-03-04"],
            "accessionNumber": ["0001-01", "0002-02", "0003-03"],
            "primaryDocument": ["a.htm", "b.htm", "c.htm"],
            "items": ["2.02,7.01", "5.07", "4.02,5.07"],
            "primaryDocDescription": ["8-K", "8-K", "8-K"],
        }},
    }

    def test_parsed_filings_carry_taxonomy_type(self):
        with patch.object(sec_8k.providers, "sec_cik_for_ticker",
                          return_value="0000320193"), \
             patch.object(sec_8k.providers, "_get_json",
                          return_value=self._SUBMISSIONS):
            events = sec_8k.fetch_and_parse_8k_filings("ZZTAXN")
        by_date = {e.filing_date: e for e in events}
        self.assertEqual(by_date["2025-03-03"].event_type, "earnings")
        self.assertIsNone(by_date["2025-03-04"].event_type)  # 5.07 vote
        self.assertEqual(by_date["2025-03-05"].event_type, "restatement")


class TestRoutineNeverStudies(unittest.TestCase):
    """Gate at the app call site: routine-only filings keep car_result None
    and run_event_study is never invoked for them."""

    def test_routine_filings_skipped_informative_studied(self):
        import app as app_module
        from sec_8k import MaterialEvent8K

        def ev(date, items, etype):
            return MaterialEvent8K(
                accession_number="x", cik="1", ticker="TEST",
                filing_date=date, report_date=date, items=items,
                item_descriptions=[], primary_doc_url="",
                event_type=etype)

        events = [ev("2024-09-03", ["2.02"], "earnings"),
                  ev("2024-09-05", ["5.07"], None),
                  ev("2024-09-10", ["4.02"], "restatement")]
        with patch.object(event_study, "run_event_study",
                          wraps=event_study.run_event_study) as spy:
            out = app_module._attach_car_to_8k_events(
                events, _frame(), _frame(), "TEST")
        studied = [e for e in out if e["car_result"] is not None]
        self.assertEqual([e["filing_date"] for e in studied],
                         ["2024-09-03", "2024-09-10"])
        routine = next(e for e in out if e["filing_date"] == "2024-09-05")
        self.assertIsNone(routine["car_result"])
        for call in spy.call_args_list:
            self.assertTrue(call.kwargs.get("event_type"))


import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _frame(days=400, base=100.0, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=days)
    close = base * np.exp(np.cumsum(rng.normal(0.0005, 0.012, days)))
    return pd.DataFrame({"close": close}, index=idx)
