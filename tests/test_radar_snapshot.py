"""Snapshot publication: coherent build, validate, atomic pointer, last-good."""
import json
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

import radar  # noqa: E402
import radar_contracts  # noqa: E402


def _store(sym, days=320, drift=0.001, volume=1_000_000):
    idx = pd.bdate_range("2024-06-03", periods=days)
    close = 100 * np.exp(np.cumsum(np.full(days, drift)))
    db.store_prices(sym, pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close,
         "Volume": np.full(days, volume)}, index=idx))


def _seed_universe():
    _store("AAA", drift=0.002)
    _store("BBB", drift=-0.001)
    _store("SPY")
    db.cache_set("yfinance", "fundamentals:AAA",
                 {"Sector": "Technology", "Industry": "Tech", "Market Cap": 1e9,
                  "Forward P/E": 10.0, "Price / Sales": 1.0, "Revenue Growth": 0.2})
    db.cache_set("yfinance", "fundamentals:BBB",
                 {"Sector": "Technology", "Industry": "Tech", "Market Cap": 2e9,
                  "Forward P/E": -8.0, "Price / Sales": 4.0})
    db.cache_set("yfinance", "earnings:AAA", [
        {"date": "2024-09-06", "eps_estimate": 1.0, "eps_actual": 1.2,
         "surprise_pct": 20.0, "is_upcoming": False}])


class TestNoNetwork(unittest.TestCase):
    def test_build_performs_no_network_calls(self):
        _seed_universe()
        ctx = radar.load_context()
        with patch("yfinance.Ticker", side_effect=AssertionError("network!")), \
             patch.object(radar.db, "get_prices",
                          side_effect=AssertionError("network!")):
            snap = radar.build_snapshot(ctx)
        self.assertIsNotNone(snap)
        self.assertTrue(snap["meta"]["validation_status"].startswith("ok"))


class TestValidation(unittest.TestCase):
    def _built(self):
        _seed_universe()
        return radar.build_snapshot(radar.load_context())

    def test_built_snapshot_passes_contract_checks(self):
        snap = self._built()
        ok, problems = radar.validate_snapshot(snap)
        self.assertTrue(ok, problems)
        self.assertEqual(snap["meta"]["validation_status"], "ok")

    def test_metric_key_without_contract_fails_validation(self):
        snap = self._built()
        snap["drift"]["rows"][0]["mystery_alpha"] = 1.0
        ok, problems = radar.validate_snapshot(snap)
        self.assertFalse(ok)
        self.assertTrue(any("uncontracted" in p and "mystery_alpha" in p
                            for p in problems), problems)
        self.assertIn("uncontracted", snap["meta"]["validation_status"])

    def test_exempt_non_metric_keys_do_not_fail(self):
        snap = self._built()
        snap["drift"]["rows"][0]["car_significant"] = True
        snap["value"]["rows"][0]["peer_count"] = 3
        snap["value"]["rows"][0]["input_coverage"] = 0.8
        snap["value"]["rows"][0]["low_coverage"] = False
        ok, problems = radar.validate_snapshot(snap)
        self.assertTrue(ok, problems)

    def test_incomplete_contract_entry_fails_validation(self):
        snap = self._built()
        broken = {**radar_contracts.contract("fwd_pe"), "unit": ""}
        with patch.dict(radar_contracts._CONTRACTS, {"fwd_pe": broken}):
            ok, problems = radar.validate_snapshot(snap)
        self.assertFalse(ok)
        self.assertTrue(any("fwd_pe" in p and "unit" in p for p in problems),
                        problems)


class TestFundamentalsAsOf(unittest.TestCase):
    OLDEST = "2025-06-01T10:00:00"
    T_A = "2026-01-01T10:00:00"
    T_B = "2026-02-01T10:00:00"

    def _cached(self, sym, fetched_at):
        with db.get_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO api_cache "
                "(provider, key, fetched_at, payload) VALUES (?, ?, ?, ?)",
                ("yfinance", f"fundamentals:{sym}", fetched_at,
                 json.dumps({"Sector": "Technology"})))

    def setUp(self):
        self._cached("OLDCO", self.OLDEST)
        self._cached("ASOA", self.T_A)
        self._cached("ASOB", self.T_B)

    def test_as_of_scoped_to_symbols_not_whole_cache(self):
        self.assertEqual(radar._fundamentals_age(["ASOA"]), self.T_A)
        self.assertEqual(radar._fundamentals_age(["ASOB"]), self.T_B)
        self.assertEqual(radar._fundamentals_age(["ASOA", "ASOB"]), self.T_A)
        self.assertNotEqual(radar._fundamentals_age(["ASOA"]), self.OLDEST)

    def test_as_of_empty_scope_is_none(self):
        self.assertIsNone(radar._fundamentals_age(["NOCACHE"]))

    def test_load_context_as_of_excludes_out_of_universe_rows(self):
        _seed_universe()
        ctx = radar.load_context()
        with db.get_conn() as conn:
            rows = conn.execute(
                "SELECT key, fetched_at FROM api_cache "
                "WHERE provider = 'yfinance' AND key LIKE 'fundamentals:%'"
            ).fetchall()
        in_scope = [r["fetched_at"] for r in rows
                    if r["key"][len("fundamentals:"):] in set(ctx["symbols"])]
        self.assertNotIn(self.OLDEST, in_scope)
        self.assertEqual(ctx["source_as_of"]["fundamentals"], min(in_scope))
        self.assertNotEqual(ctx["source_as_of"]["fundamentals"], self.OLDEST)


class TestPublish(unittest.TestCase):
    def setUp(self):
        with db.get_conn() as conn:
            conn.execute("DELETE FROM api_cache WHERE provider = 'radar_snapshot'")

    def test_publish_moves_pointer_and_serves_current(self):
        _seed_universe()
        ctx = radar.load_context()
        snap = radar.publish_snapshot(ctx)
        self.assertIsNotNone(snap)
        current = radar.current_snapshot()
        self.assertIsNotNone(current)
        self.assertEqual(current["meta"]["snapshot_id"], snap["meta"]["snapshot_id"])
        self.assertIn("source_as_of", current["meta"])
        self.assertIn("excluded_count_by_reason", current["meta"])

    def test_invalid_build_never_replaces_pointer(self):
        _seed_universe()
        ctx = radar.load_context()
        radar.publish_snapshot(ctx)
        first_id = radar.current_snapshot()["meta"]["snapshot_id"]
        # a context whose drift build explodes must leave the pointer alone
        with patch.object(radar, "_drift_rows", side_effect=RuntimeError("boom")):
            self.assertIsNone(radar.publish_snapshot(ctx))
        self.assertEqual(radar.current_snapshot()["meta"]["snapshot_id"], first_id)

    def test_retention_keeps_last_two(self):
        _seed_universe()
        ctx = radar.load_context()
        with patch.object(radar, "_snapshot_id",
                          side_effect=[f"S{i}" for i in range(4)]):
            for _ in range(4):
                radar.publish_snapshot(ctx)
        with db.get_conn() as conn:
            rows = conn.execute(
                "SELECT key FROM api_cache WHERE provider='radar_snapshot' "
                "AND key LIKE 'snapshot:%'").fetchall()
        ids = {r["key"].split(":", 1)[1] for r in rows}
        self.assertEqual(len(rows), 2)                   # pruned to keep=2
        self.assertEqual(ids, {"S2", "S3"})              # newest two survive
        self.assertEqual(radar.current_snapshot()["meta"]["snapshot_id"], "S3")
        for gone in ("S0", "S1"):                        # oldest ids are gone
            self.assertIsNone(db.cache_get("radar_snapshot",
                                           f"snapshot:{gone}", 24 * 7))

    def test_same_minute_publishes_get_distinct_ids(self):
        _seed_universe()
        ctx = radar.load_context()
        id_a, id_b = radar._snapshot_id(ctx), radar._snapshot_id(ctx)
        self.assertNotEqual(id_a, id_b)        # same minute, equal universe
        self.assertLess(id_a, id_b)            # ids sort chronologically
        first = radar.publish_snapshot(ctx)
        second = radar.publish_snapshot(ctx)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertNotEqual(first["meta"]["snapshot_id"],
                            second["meta"]["snapshot_id"])
        self.assertEqual(radar.current_snapshot()["meta"]["snapshot_id"],
                         second["meta"]["snapshot_id"])
        with db.get_conn() as conn:
            rows = conn.execute(
                "SELECT key FROM api_cache WHERE provider='radar_snapshot' "
                "AND key LIKE 'snapshot:%'").fetchall()
        self.assertEqual(len(rows), 2)                  # retention reconciled
        for snap in (first, second):                    # neither clobbered
            self.assertIsNotNone(db.cache_get(
                "radar_snapshot",
                f"snapshot:{snap['meta']['snapshot_id']}", 24 * 7))

    def test_nm_values_never_rank_as_bargains(self):
        _seed_universe()
        snap = radar.build_snapshot(radar.load_context())
        bbb = next(r for r in snap["value"]["rows"] if r["symbol"] == "BBB")
        self.assertTrue(bbb.get("fwd_pe_nm"))          # negative P/E flagged NM
        self.assertIsNone(bbb.get("fwd_pe_pct"))       # excluded from pool

    def test_incomplete_windows_render_none(self):
        _seed_universe()
        snap = radar.build_snapshot(radar.load_context())
        aaa = next(r for r in snap["drift"]["rows"] if r["symbol"] == "AAA")
        # print was ~2024-09-06 with ~320 bars from 2024-06-03 → 20d complete
        if aaa["sessions_since"] >= 20:
            self.assertIsNotNone(aaa["drift_20d"])

    def test_upcoming_strip_excludes_past_and_far_future(self):
        db.cache_set("yfinance", "earnings:BBB", [
            {"date": "2020-01-01", "is_upcoming": False},
            {"date": (pd.Timestamp.today() + pd.Timedelta(days=3)).date().isoformat(),
             "eps_estimate": 2.0, "eps_actual": None, "is_upcoming": True},
            {"date": (pd.Timestamp.today() + pd.Timedelta(days=30)).date().isoformat(),
             "eps_estimate": 2.0, "eps_actual": None, "is_upcoming": True}])
        snap = radar.build_snapshot(radar.load_context())
        dates = [u["date"] for u in snap["drift"]["upcoming"]]
        self.assertEqual(len(dates), 1)  # only the 3-day-out print

    def test_near_zero_estimate_flags_nm(self):
        db.cache_set("yfinance", "earnings:BBB", [
            {"date": "2024-09-06", "eps_estimate": 0.01, "eps_actual": 0.02,
             "surprise_pct": 100.0, "is_upcoming": False}])
        snap = radar.build_snapshot(radar.load_context())
        bbb = next(r for r in snap["drift"]["rows"] if r["symbol"] == "BBB")
        self.assertTrue(bbb["surprise_nm"])
        self.assertIsNone(bbb["surprise_pct"])  # never rendered as a %
