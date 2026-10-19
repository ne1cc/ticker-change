import json
import os
import tempfile
import unittest
from unittest import mock

import pandas as pd

import db
import heatmap
import indices
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

    def test_period_changes_need_enough_history(self):
        # 2 closes: 1D exists, 5D and 1M have no base close and are None.
        data = heatmap.build_heatmap_data()
        aapl = next(t for t in data["tiles"] if t["symbol"] == "AAPL")
        self.assertIsNone(aapl["change_5d"])
        self.assertIsNone(aapl["change_1m"])

    def test_period_changes_measure_back_5_and_21_sessions(self):
        closes = [100.0 + i for i in range(30)]   # 100..129
        db.store_prices("AAPL", _frame(closes))
        data = heatmap.build_heatmap_data()
        aapl = next(t for t in data["tiles"] if t["symbol"] == "AAPL")
        self.assertAlmostEqual((129 / 124 - 1) * 100, aapl["change_5d"], places=2)
        self.assertAlmostEqual((129 / 108 - 1) * 100, aapl["change_1m"], places=2)

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

    def test_renders_native_treemap_data(self):
        html = heatmap.render_treemap(self._data())
        self.assertIn('id="heatmap-treemap"', html)
        self.assertIn('"type":"treemap"', html)
        self.assertNotIn("plotly", html.lower())
        self.assertIn("AAPL", html)
        self.assertIn("sector::Technology", html)
        # Percentage labels remain available in the inert chart payload.
        self.assertIn("+1.00%", html)

    def test_dark_and_light_variants_render(self):
        self.assertIsNotNone(heatmap.render_treemap(self._data(), dark=True))
        self.assertIsNotNone(heatmap.render_treemap(self._data(), dark=False))

    def test_embedded_trace_supplies_tooltip_and_click_data(self):
        html = heatmap.render_treemap(self._data())
        option = json.loads(html.split('type="application/json"', 1)[1].split('>', 1)[1].split('</script>', 1)[0])
        sector = option['series'][0]['data'][0]
        leaves = {node['symbol']: node for node in sector['children']}
        self.assertEqual('+1.00%', leaves['AAPL']['change'])
        self.assertEqual('Apple Inc.', leaves['AAPL']['company'])
        self.assertEqual(101.0, leaves['AAPL']['price'])
        self.assertTrue(leaves['AAPL']['leaf'])
        self.assertEqual('-1.00%', leaves['MSFT']['change'])
        self.assertEqual('+0.01%', sector['change'])

    def test_sector_drill_down_is_enabled(self):
        html = heatmap.render_treemap(self._data())
        option = json.loads(html.split('type="application/json"', 1)[1].split('>', 1)[1].split('</script>', 1)[0])
        series = option['series'][0]
        # Sector clicks re-root via charts.js (treemapRootToNode); the breadcrumb is the way back.
        self.assertFalse(series['nodeClick'])
        self.assertTrue(series['breadcrumb']['show'])
        self.assertEqual('sector::Technology', series['data'][0]['id'])

    def test_sector_tooltip_omits_ticker_and_price(self):
        html = heatmap.render_treemap(self._data())
        option = json.loads(html.split('type="application/json"', 1)[1].split('>', 1)[1].split('</script>', 1)[0])
        sector = option['series'][0]['data'][0]
        self.assertNotIn('symbol', sector)
        self.assertNotIn('price', sector)
        self.assertIn('price', sector['children'][0])


def _tile(symbol, sector, change, volume=1_000_000, **extra):
    return {"symbol": symbol, "name": symbol + " Inc.", "sector": sector, "price": 10.0,
            "change_pct": change, "change_5d": extra.get("change_5d"),
            "change_1m": extra.get("change_1m"), "dollar_volume": volume, "as_of": "2026-09-02"}


class TestSectorSummaryAndMovers(unittest.TestCase):
    def test_sector_change_is_dollar_volume_weighted(self):
        tiles = [_tile("A", "Tech", 2.0, volume=300), _tile("B", "Tech", -2.0, volume=100)]
        (row,) = heatmap.sector_summary(tiles)
        self.assertAlmostEqual(1.0, row["change"])    # (2*300 - 2*100) / 400
        self.assertEqual((1, 1, 2), (row["advancers"], row["decliners"], row["count"]))

    def test_tiles_missing_the_period_are_left_out(self):
        tiles = [_tile("A", "Tech", 1.0, change_5d=4.0), _tile("B", "Tech", 1.0)]
        (row,) = heatmap.sector_summary(tiles, "change_5d")
        self.assertEqual(1, row["count"])
        self.assertAlmostEqual(4.0, row["change"])
        self.assertEqual([], heatmap.sector_summary([_tile("B", "Tech", 1.0)], "change_5d"))

    def test_movers_are_ranked_and_exclude_flat_and_missing(self):
        tiles = [_tile("UP2", "X", 2.0), _tile("UP1", "X", 1.0), _tile("FLAT", "X", 0.0),
                 _tile("DN1", "X", -1.0), _tile("DN3", "X", -3.0)]
        m = heatmap.movers(tiles, n=2)
        self.assertEqual(["UP2", "UP1"], [t["symbol"] for t in m["gainers"]])
        self.assertEqual(["DN3", "DN1"], [t["symbol"] for t in m["losers"]])
        self.assertEqual({"gainers": [], "losers": []}, heatmap.movers(tiles, "change_5d"))

    def test_period_views_cover_each_period_in_order(self):
        views = heatmap.period_views({"tiles": [_tile("A", "Tech", 1.0)]})
        self.assertEqual(["1D", "5D", "1M"], [v["key"] for v in views])
        self.assertEqual([3.0, 6.0, 10.0], [v["span"] for v in views])


class TestPeriodTreemap(unittest.TestCase):
    def _option(self, tiles):
        html = heatmap.render_treemap({"tiles": tiles})
        return json.loads(html.split('type="application/json"', 1)[1].split('>', 1)[1].split('</script>', 1)[0])

    def test_option_carries_every_period_and_widens_the_scale(self):
        option = self._option([_tile("A", "Tech", 1.0, change_5d=5.0, change_1m=-12.0)])
        self.assertEqual(["1D", "5D", "1M"], [p["key"] for p in option["terminalPeriods"]])
        self.assertEqual(3.0, option["series"][0]["visualMax"])
        leaves = {p["key"]: p["data"][0]["children"][0] for p in option["terminalPeriods"]}
        self.assertEqual("+1.00%", leaves["1D"]["change"])
        self.assertEqual("+5.00%", leaves["5D"]["change"])
        self.assertEqual("-12.00%", leaves["1M"]["change"])
        self.assertEqual(-10.0, leaves["1M"]["value"][1])   # clamped to the 1M span

    def test_missing_period_is_neutral_and_labelled_na(self):
        option = self._option([_tile("A", "Tech", 1.0)])
        by_key = {p["key"]: p for p in option["terminalPeriods"]}
        leaf = by_key["5D"]["data"][0]["children"][0]
        self.assertEqual("n/a", leaf["change"])
        self.assertEqual(0, leaf["value"][1])
        self.assertEqual("n/a", by_key["5D"]["data"][0]["change"])


class TestIndices(TempDBCase):
    SP = [{"symbol": "AAPL", "name": "Apple", "sector": "Information Technology"}]
    NDX_ROWS = [{"symbol": "AAPL", "name": "Apple", "sector": ""},        # S&P member -> GICS
                {"symbol": "ARM", "name": "Arm", "sector": "Technology"},   # not in S&P -> source text
                {"symbol": "ZZZ", "name": "Zed", "sector": ""}]             # nothing known -> Other

    def _fetchers(self, ndx=None):
        return mock.patch.dict(indices.INDICES, {
            "sp500": {"label": "S&P 500", "fetch": mock.Mock(return_value=self.SP)},
            "ndx": {"label": "Nasdaq-100", "fetch": mock.Mock(return_value=ndx or self.NDX_ROWS)},
            "dow": {"label": "Dow 30", "fetch": mock.Mock(return_value=[])},
        })

    def test_non_sp500_sectors_prefer_gics_then_source_then_other(self):
        with self._fetchers():
            heatmap.get_constituents("sp500")
            rows = heatmap.get_constituents("ndx")
        self.assertEqual(["Information Technology", "Technology", "Other"], [r["sector"] for r in rows])

    def test_each_index_caches_under_its_own_key(self):
        with self._fetchers():
            heatmap.get_constituents("sp500")
            heatmap.get_constituents("ndx")
            self.assertEqual(1, indices.INDICES["ndx"]["fetch"].call_count)
            heatmap.get_constituents("ndx")                      # cache hit
            self.assertEqual(1, indices.INDICES["ndx"]["fetch"].call_count)
        self.assertEqual(self.SP, db.cache_get("sp500", "constituents", ttl_hours=1))
        self.assertEqual(3, len(db.cache_get("ndx", "constituents", ttl_hours=1)))

    def test_request_path_never_fetches(self):
        with self._fetchers():
            self.assertEqual([], heatmap.get_constituents("ndx", fetch=False))
            indices.INDICES["ndx"]["fetch"].assert_not_called()
            heatmap.get_constituents("sp500")
            heatmap.get_constituents("ndx")
            self.assertEqual(3, len(heatmap.get_constituents("ndx", fetch=False)))

    def test_fetch_failure_serves_the_stale_list(self):
        with self._fetchers():
            heatmap.get_constituents("sp500")
            heatmap.get_constituents("ndx")
            with mock.patch.object(db, "cache_get", return_value=None):   # expired
                indices.INDICES["ndx"]["fetch"].side_effect = RuntimeError("down")
                rows = heatmap.get_constituents("ndx")
        self.assertEqual(3, len(rows))

    def test_universe_symbols_is_the_union(self):
        with self._fetchers():
            self.assertEqual(["AAPL", "ARM", "ZZZ"], heatmap.universe_symbols())

    def test_payloads_are_per_index_and_invalidate_clears_all(self):
        with mock.patch.object(heatmap, "build_heatmap_data", side_effect=lambda i: {"index": i}) as build:
            self.assertEqual({"index": "ndx"}, heatmap.get_heatmap_payload("ndx"))
            self.assertEqual({"index": "sp500"}, heatmap.get_heatmap_payload())
            heatmap.get_heatmap_payload("ndx")
            self.assertEqual(2, build.call_count)
            heatmap.invalidate_payload()
            heatmap.get_heatmap_payload("ndx")
            heatmap.get_heatmap_payload()
            self.assertEqual(4, build.call_count)

    def test_build_labels_the_index(self):
        db.store_prices("AAPL", _frame([100.0, 101.0]))
        with self._fetchers():
            heatmap.get_constituents("sp500")
            heatmap.get_constituents("ndx")
            data = heatmap.build_heatmap_data("ndx")
        self.assertEqual(("ndx", "Nasdaq-100", 3), (data["index"], data["label"], data["universe"]))
        self.assertEqual(["AAPL"], [t["symbol"] for t in data["tiles"]])


class TestIndexRoutes(TempDBCase):
    def setUp(self):
        super().setUp()
        self.client = app.test_client()
        self.client.testing = True

    def test_default_is_sp500_and_the_picker_lists_every_index(self):
        html = self.client.get("/heatmap").get_data(as_text=True)
        self.assertIn("S&amp;P 500 Heatmap", html)
        for label in ("S&amp;P 500", "Nasdaq-100", "Dow 30"):
            self.assertIn(label, html)
        self.assertIn('href="/heatmap?index=ndx"', html)
        self.assertEqual(1, html.count('aria-current="page"'))

    def test_index_param_selects_the_universe(self):
        html = self.client.get("/heatmap?index=ndx").get_data(as_text=True)
        self.assertIn("Nasdaq-100 Heatmap", html)
        self.assertIn("Warming the Nasdaq-100 universe", html)   # nothing cached yet
        self.assertIn("/api/heatmap?index=ndx", html)

    def test_unknown_index_is_404_on_page_and_api(self):
        self.assertEqual(404, self.client.get("/heatmap?index=bogus").status_code)
        self.assertEqual(404, self.client.get("/api/heatmap?index=bogus").status_code)

    def test_api_serves_the_requested_index(self):
        db.store_prices("AAPL", _frame([100.0, 101.0]))
        cons = [{"symbol": "AAPL", "name": "Apple", "sector": "Technology"}]
        with mock.patch.object(heatmap, "get_constituents", return_value=cons):
            payload = self.client.get("/api/heatmap?index=dow").get_json()
        self.assertEqual(("dow", "Dow 30"), (payload["index"], payload["label"]))


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

    def test_page_renders_movers_and_sector_panels_for_each_period(self):
        db.store_prices("AAPL", _frame([100.0, 101.0]))
        db.store_prices("MSFT", _frame([200.0, 198.0]))
        with mock.patch.object(heatmap, "get_constituents", return_value=CONSTITUENTS):
            html = self.client.get("/heatmap").get_data(as_text=True)
        for key in ("1D", "5D", "1M"):
            self.assertIn(f'data-period-panel="{key}"', html)
        self.assertIn('/stock?ticker=AAPL', html)                  # gainer links to its price table
        self.assertIn('hidden', html.split('data-period-panel="5D"', 1)[1].split('>', 1)[0])  # only 1D shows first
        self.assertIn('id="heatmap-legend-max"', html)


if __name__ == "__main__":
    unittest.main()
