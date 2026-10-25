"""Audit carry-over tests for earnings and glossary integrity (Issue #160)."""
import unittest
import numpy as np
import pandas as pd
import earnings
import glossary


class TestEarningsLookAheadAndSorting(unittest.TestCase):
    def test_realized_earnings_moves_order_invariant(self):
        idx = pd.bdate_range("2024-01-01", periods=32)
        close = [100.0]
        for i in range(1, 32):
            if i % 2 == 1 and i < 17:
                close.append(close[-1] * (1.4 if (i // 2) % 2 == 0 else 0.6))
            else:
                close.append(close[-1] * 1.01)
        df = pd.DataFrame({"close": close}, index=idx)
        events_asc = [{"date": idx[2 * j].date().isoformat(), "is_upcoming": False} for j in range(16)]
        events_desc = list(reversed(events_asc))

        med_asc = earnings.realized_earnings_moves(df, events_asc)
        med_desc = earnings.realized_earnings_moves(df, events_desc)

        self.assertIsNotNone(med_asc)
        self.assertEqual(med_asc, med_desc)
        self.assertAlmostEqual(med_asc, 0.01, places=4)

    def test_incomplete_windows_return_none_preventing_lookahead(self):
        idx = pd.bdate_range("2024-01-01", periods=10)
        df = pd.DataFrame({"close": [100.0 + i for i in range(10)]}, index=idx)
        # Event on session 7 has only 2 subsequent sessions (sessions 8, 9)
        ev_date = idx[7].date().isoformat()
        m = earnings.event_metrics(df, ev_date)
        self.assertIsNotNone(m["t0"])
        self.assertIsNone(m["drift_5d"], "Horizon 5d must be None when fewer than 5 future sessions exist")
        self.assertIsNone(m["drift_10d"], "Horizon 10d must be None when fewer than 10 future sessions exist")
        self.assertIsNone(m["drift_20d"], "Horizon 20d must be None when fewer than 20 future sessions exist")

    def test_compute_earnings_events_takes_most_recent_prints(self):
        idx = pd.bdate_range("2023-01-01", periods=500)
        df = pd.DataFrame({"close": np.linspace(100, 200, 500)}, index=idx)
        bench = pd.DataFrame({"close": np.linspace(300, 400, 500)}, index=idx)

        events_asc = [
            {"date": idx[30 * j].date().isoformat(), "is_upcoming": False, "surprise_pct": 1.0}
            for j in range(1, 15)
        ]
        events_desc = list(reversed(events_asc))

        evs_from_asc = earnings.compute_earnings_events(df, bench, events_asc, ticker="TEST", max_events=4)
        evs_from_desc = earnings.compute_earnings_events(df, bench, events_desc, ticker="TEST", max_events=4)

        self.assertEqual(len(evs_from_asc), 4)
        self.assertEqual(len(evs_from_desc), 4)
        self.assertEqual([e["date"] for e in evs_from_asc], [e["date"] for e in evs_from_desc])
        # Assert dates returned are the 4 most recent dates
        expected_dates = sorted([e["date"] for e in events_asc], reverse=True)[:4]
        self.assertEqual([e["date"] for e in evs_from_asc], expected_dates)


class TestGlossaryAuditedMetrics(unittest.TestCase):
    def test_audited_metrics_have_no_placeholders_and_accurate_text(self):
        audited_keys = ["upside_capture", "downside_capture", "dual_betas", "regime_betas", "vrp"]
        for key in audited_keys:
            self.assertIn(key, glossary.GLOSSARY)
            entry = glossary.GLOSSARY[key]
            for field in ["term", "short", "long"]:
                val = entry.get(field, "")
                self.assertTrue(len(val) > 10, f"{key} {field} is too short")
                for bad in ["TODO", "TBD", "placeholder", "fixme", "xxx"]:
                    self.assertNotIn(bad, val.lower(), f"{key} contains bad placeholder '{bad}'")

        # Check domain-specific honest details:
        self.assertIn("geometric", glossary.GLOSSARY["upside_capture"]["long"].lower())
        self.assertIn("geometric", glossary.GLOSSARY["downside_capture"]["long"].lower())
        self.assertIn("market-direction", glossary.GLOSSARY["dual_betas"]["long"].lower())
        self.assertIn("fred", glossary.GLOSSARY["regime_betas"]["long"].lower())
        self.assertIn("yang–zhang", glossary.GLOSSARY["vrp"]["long"].lower())
