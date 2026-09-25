import os
import tempfile
import unittest
from unittest import mock

import pandas as pd

import db
import heatmap
import sp500
from app import app


def _frame(closes, volume=1_000_000):
    """A daily_prices-shaped frame from a list of closes."""
    idx = pd.date_range("2026-09-01", periods=len(closes), freq="D")
    return pd.DataFrame({
        "Open": closes, "High": closes, "Low": closes,
        "Close": closes, "Volume": [volume] * len(closes),
    }, index=idx)


CONSTITUENTS = [
    {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology"},
    {"symbol": "MSFT", "name": "Microsoft Corp.", "sector": "Technology"},
    {"symbol": "XOM", "name": "Exxon Mobil Corp.", "sector": "Energy"},
    {"symbol": "ONE", "name": "One Session Inc.", "sector": "Energy"},
    {"symbol": "MISSING", "name": "Not Cached Inc.", "sector": "Energy"},
]


class TempDBCase(unittest.TestCase):
    def setUp(self):
        self.orig_db_path = db.DB_PATH
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        db.DB_PATH = self.temp_db.name
        db.init_db()
        self.addCleanup(self._restore_db)

    def _restore_db(self):
        db.DB_PATH = self.orig_db_path
        try:
            os.remove(self.temp_db.name)
        except OSError:
            pass


class TestBuildHeatmapData(TempDBCase):
    def setUp(self):
        super().setUp()
        db.store_prices("AAPL", _frame([100.0, 101.0]))
        db.store_prices("MSFT", _frame([200.0, 198.0]))
        db.store_prices("XOM", _frame([50.0, 50.0]))
        db.store_prices("ONE", _frame([77.0]))
        patcher = mock.patch.object(
            heatmap, "get_constituents", return_value=CONSTITUENTS
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_change_sector_and_breadth(self):
        data = heatmap.build_heatmap_data()
        self.assertIsNotNone(data)
        self.assertEqual(3, data["coverage"])          # ONE and MISSING drop out
        self.assertEqual(len(CONSTITUENTS), data["universe"])
        self.assertEqual(["Energy", "Technology"], data["sectors"])
        self.assertEqual(1, data["advancers"])
        self.assertEqual(1, data["decliners"])
        self.assertEqual(1, data["unchanged"])
        self.assertEqual("2026-09-02", data["as_of"])

        by_symbol = {t["symbol"]: t for t in data["tiles"]}
        self.assertAlmostEqual(1.0, by_symbol["AAPL"]["change_pct"])
        self.assertAlmostEqual(-1.0, by_symbol["MSFT"]["change_pct"])
        self.assertAlmostEqual(0.0, by_symbol["XOM"]["change_pct"])
        self.assertEqual("Technology", by_symbol["AAPL"]["sector"])
        self.assertEqual("Apple Inc.", by_symbol["AAPL"]["name"])

    def test_tile_size_is_trailing_dollar_volume(self):
        data = heatmap.build_heatmap_data()
        aapl = next(t for t in data["tiles"] if t["symbol"] == "AAPL")
        expected = (100.0 * 1_000_000 + 101.0 * 1_000_000) / 2
        self.assertAlmostEqual(expected, aapl["dollar_volume"])

    def test_returns_none_with_no_data(self):
        # Fresh read with constituents present but SQLite returning no rows.
        with mock.patch.object(heatmap, "_recent_rows", return_value=pd.DataFrame()):
            self.assertIsNone(heatmap.build_heatmap_data())


class TestRecentRows(TempDBCase):
    def test_window_function_pulls_only_the_tail(self):
        db.store_prices("AAPL", _frame([100.0 + i for i in range(30)]))
        rows = heatmap._recent_rows(["AAPL"], lookback=25)
        self.assertEqual(25, len(rows))
        self.assertEqual(129.0, float(rows["close"].iloc[-1]))
        self.assertEqual(105.0, float(rows["close"].iloc[0]))

    def test_unknown_symbols_yield_empty_frame(self):
        self.assertTrue(heatmap._recent_rows(["NOPE"]).empty)


class TestPayloadCache(TempDBCase):
    def test_second_call_is_served_from_api_cache(self):
        seeded = {"coverage": 1, "universe": 1, "tiles": [], "sectors": [],
                  "advancers": 0, "decliners": 1, "unchanged": 0, "as_of": "2026-09-02"}
        with mock.patch.object(heatmap, "build_heatmap_data", return_value=seeded) as build:
            first = heatmap.get_heatmap_payload()
            second = heatmap.get_heatmap_payload()
        self.assertEqual(seeded, first)
        self.assertEqual(seeded, second)
        self.assertEqual(1, build.call_count)

    def test_no_data_is_not_cached(self):
        with mock.patch.object(heatmap, "build_heatmap_data", return_value=None) as build:
            self.assertIsNone(heatmap.get_heatmap_payload())
            self.assertIsNone(heatmap.get_heatmap_payload())
        self.assertEqual(2, build.call_count)

    def test_invalidate_payload_forces_a_rebuild(self):
        seeded = {"coverage": 1, "universe": 1, "tiles": [], "sectors": [],
                  "advancers": 0, "decliners": 1, "unchanged": 0, "as_of": "2026-09-02"}
        with mock.patch.object(heatmap, "build_heatmap_data", return_value=seeded) as build:
            self.assertIsNotNone(heatmap.get_heatmap_payload())
            heatmap.invalidate_payload()
            self.assertIsNotNone(heatmap.get_heatmap_payload())
        self.assertEqual(2, build.call_count)


class TestRenderTreemap(TempDBCase):
    def _data(self):
        return {
            "tiles": [
                {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology",
                 "price": 101.0, "change_pct": 1.0, "dollar_volume": 100_500_000,
                 "as_of": "2026-09-02"},
                {"symbol": "MSFT", "name": "Microsoft Corp.", "sector": "Technology",
                 "price": 198.0, "change_pct": -1.0, "dollar_volume": 99_000_000,
                 "as_of": "2026-09-02"},
            ],
            "sectors": ["Technology"], "coverage": 2, "universe": 2,
            "advancers": 1, "decliners": 1, "unchanged": 0, "as_of": "2026-09-02",
        }

    def test_renders_embedded_plotly_div(self):
        html = heatmap.render_treemap(self._data())
        self.assertIn('id="heatmap-treemap"', html)
        self.assertIn("plotly", html.lower())
        self.assertIn("AAPL", html)
        self.assertIn("sector::Technology", html)
        # Change text rides the `text` property (customdata isn't substituted
        # in treemap texttemplate) and must appear in the embedded figure.
        self.assertIn("+1.00%", html)

    def test_dark_and_light_variants_render(self):
        self.assertIsNotNone(heatmap.render_treemap(self._data(), dark=True))
        self.assertIsNotNone(heatmap.render_treemap(self._data(), dark=False))


class TestRefreshUniverse(TempDBCase):
    def _download_payload(self, symbols, closes_by_symbol):
        """MultiIndex frame shaped like yf.download(group_by='ticker')."""
        fields = ["Open", "High", "Low", "Close", "Volume"]
        cols = pd.MultiIndex.from_product([symbols, fields])
        idx = pd.date_range("2026-09-01", periods=2, freq="D")
        data = {}
        for sym in symbols:
            closes = closes_by_symbol.get(sym, [float("nan")] * 2)
            for f in fields:
                data[(sym, f)] = closes if f == "Close" else closes
        return pd.DataFrame(data, index=idx)

    def test_good_symbols_stored_and_nan_symbols_fail(self):
        raw = self._download_payload(["AAPL", "BADT"], {"AAPL": [100.0, 101.0]})
        with mock.patch("yfinance.download", return_value=raw) as dl, \
             mock.patch.object(db, "store_prices", wraps=db.store_prices) as store:
            result = heatmap.refresh_universe(["AAPL", "BADT"])
        self.assertEqual(1, result["fetched"])
        self.assertEqual(["BADT"], result["failed"])
        self.assertEqual(1, store.call_count)
        # The universe goes through in 100-ticker chunks; this pair fits one.
        self.assertEqual(["AAPL", "BADT"], dl.call_args.args[0])

    def test_download_failure_fails_the_whole_chunk_without_raising(self):
        with mock.patch("yfinance.download", side_effect=ConnectionError("boom")):
            result = heatmap.refresh_universe(["AAPL", "MSFT"])
        self.assertEqual(0, result["fetched"])
        self.assertEqual(["AAPL", "MSFT"], result["failed"])


FAKE_CSV = (
    "Symbol,Security,GICS Sector,GICS Sub-Industry,Headquarters Location,Date added,CIK,Founded\n"
    "AAPL,Apple Inc.,Information Technology,Technology Hardware,Cupertino,1982-11-30,320193,1976\n"
    "BRK.B,Berkshire Hathaway Inc.,Financials,Multi-Sector Holdings,Omaha,2010-02-16,1067983,1839\n"
    ",No Symbol Row,Energy,,Houston,2020-01-01,1,2000\n"
    "NOSEC,No Sector Name,,Sub-Industry,Denver,2021-01-01,2,2001\n"
)


class TestFetchConstituents(unittest.TestCase):
    def test_parse_normalises_symbols_and_defaults_sector(self):
        rows = sp500._parse_constituents_csv(FAKE_CSV)
        self.assertEqual(3, len(rows))
        self.assertEqual("BRK-B", rows[1]["symbol"])
        self.assertEqual("Berkshire Hathaway Inc.", rows[1]["name"])
        self.assertEqual("Other", rows[2]["sector"])

    def test_fetch_rejects_implausibly_short_list(self):
        # Same guard as fetch_sp500_tickers: a 3-row CSV means the source's
        # format changed, not that the index shrank.
        resp = mock.Mock()
        resp.text = FAKE_CSV
        resp.raise_for_status = mock.Mock()
        with mock.patch.object(sp500.requests, "get", return_value=resp):
            with self.assertRaises(RuntimeError):
                sp500.fetch_sp500_constituents()

    def test_fetch_failure_raises_runtime_error(self):
        with mock.patch.object(sp500.requests, "get",
                               side_effect=ConnectionError("boom")):
            with self.assertRaises(RuntimeError):
                sp500.fetch_sp500_constituents()


class TestHeatmapRoutes(TempDBCase):
    def setUp(self):
        super().setUp()
        self.client = app.test_client()
        self.client.testing = True

    def test_empty_db_renders_warming_banner(self):
        with mock.patch.object(heatmap, "get_constituents", return_value=CONSTITUENTS):
            resp = self.client.get("/heatmap")
        self.assertEqual(200, resp.status_code)
        html = resp.get_data(as_text=True)
        self.assertIn('id="heatmap-warming"', html)
        self.assertIn('id="heatmap-refresh-btn"', html)

    def test_empty_db_api_returns_warming_status(self):
        with mock.patch.object(heatmap, "get_constituents", return_value=CONSTITUENTS):
            resp = self.client.get("/api/heatmap")
        self.assertEqual(200, resp.status_code)
        payload = resp.get_json()
        self.assertEqual("warming", payload["status"])
        self.assertEqual([], payload["tiles"])

    def test_seeded_db_renders_treemap_and_api_payload(self):
        db.store_prices("AAPL", _frame([100.0, 101.0]))
        db.store_prices("MSFT", _frame([200.0, 198.0]))
        with mock.patch.object(heatmap, "get_constituents", return_value=CONSTITUENTS):
            page = self.client.get("/heatmap")
            api = self.client.get("/api/heatmap")
        self.assertEqual(200, page.status_code)
        html = page.get_data(as_text=True)
        self.assertIn('id="heatmap-treemap"', html)
        self.assertEqual(200, api.status_code)
        payload = api.get_json()
        self.assertEqual("ok", payload["status"])
        self.assertEqual(2, payload["coverage"])
        self.assertEqual(1, payload["advancers"])


if __name__ == "__main__":
    unittest.main()
