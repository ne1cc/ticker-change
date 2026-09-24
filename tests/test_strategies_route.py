"""Route-level tests for the /strategies multi-strategy page.

Covers the 3-tab x 3-strategy matrix, the `strategy=bogus` fallback, the
`period=` variants, the two error paths (uncached ticker / insufficient
history), the screener's `abs_only` filter, and the dual-momentum
absolute-hurdle pass/fail split -- plus a couple of direct-builder
regressions for F6 (see `_strategies_ticker_data`'s `score is None` guard).

The fixture seeds a synthetic universe of 8 tickers (plus SPY/QQQ) over a
single shared 400-business-day date range ending on the same day for every
symbol. Sharing the end date matters: the real `stocks.db` is ragged (one
symbol refreshed later than the rest), and a wide date-aligned frame with
even one stale column collapses `score_universe`'s positional lookups to a
single scorable row for the whole universe. Seeding everyone -- SPY and QQQ
included -- through the same last bar is what keeps that bug out of the
fixture.

Drift is chosen so the three strategies genuinely diverge rather than all
picking the same top-5 (see `_DRIFT` and `_reversal_series` below), and so a
real (not degenerate) passes_absolute split exists for `dual_momentum`.
"""
import os
import re
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import db

# `app.py` starts a background options-chain warmer at import time, reading
# db.DB_PATH exactly once (see `_start_options_cache_warmer`). Point it at a
# private, throwaway file before the very first import so a standalone run
# of this module never scans or touches a real stocks.db.
_warmer_fd, _WARMER_DB_PATH = tempfile.mkstemp(suffix=".db")
os.close(_warmer_fd)
db.DB_PATH = _WARMER_DB_PATH
db.init_db()

import app as app_module  # noqa: E402  (module itself, vs the Flask object below)
from app import (  # noqa: E402  (must follow the DB_PATH setup above)
    app as flask_app,
    _strategies_ticker_data,
    _strategies_universe_data,
    _load_price_frame,
)
import momentum_engine  # noqa: E402

END_DATE = "2024-06-28"
N_BARS = 400

# Annualized (drift, vol, seed) per symbol. AAAA-DDDD clear the
# dual-momentum absolute hurdle (~3.66% over the 11-month scoring window,
# see momentum_engine.absolute_hurdle); EEEE/FFFF/GGGG do not -- a real,
# non-trivial passes_absolute split, not a degenerate all-pass/all-fail one.
# GGGG's steady decline also gives a genuinely negative 50/200 SMA spread.
_DRIFT = {
    "AAAA": (0.40, 0.05, 1),
    "BBBB": (0.28, 0.05, 2),
    "CCCC": (0.15, 0.05, 3),
    "DDDD": (0.06, 0.05, 4),
    "EEEE": (-0.02, 0.05, 5),
    "FFFF": (-0.15, 0.05, 6),
    "GGGG": (-0.30, 0.05, 7),
}
STOCK_SYMBOLS = list(_DRIFT) + ["HHHH"]


def _drift_series(mu_annual, sigma_annual, n=N_BARS, seed=0):
    """Deterministic (fixed-seed) synthetic close series with the given
    annualized drift and volatility."""
    rng = np.random.RandomState(seed)
    daily_mu = mu_annual / 252
    daily_sigma = sigma_annual / np.sqrt(252)
    rets = rng.normal(daily_mu, daily_sigma, size=n)
    return 100.0 * np.exp(np.cumsum(rets))


def _reversal_series(n=N_BARS, seed=8, pre_mu=-0.10, pre_sigma=0.05,
                      rally_ret=0.35, rally_days=21):
    """Flat/declining for most of the window, then a sharp rally confined to
    the FINAL `rally_days` bars.

    12-1 momentum excludes the most recent `MOMENTUM_EXCLUDE` (21) days, so
    it never sees this rally and HHHH scores poorly on it; the 50-day SMA
    does see it. This is the one deliberately-divergent name in the
    fixture, so the SMA-trend strategy's top-5 doesn't just coincide with
    the momentum strategies' top-5.
    """
    rng = np.random.RandomState(seed)
    pre_n = n - rally_days
    pre_rets = rng.normal(pre_mu / 252, pre_sigma / np.sqrt(252), size=pre_n)
    rally_daily = np.log(1 + rally_ret) / rally_days
    rets = np.concatenate([pre_rets, np.full(rally_days, rally_daily)])
    return 100.0 * np.exp(np.cumsum(rets))


def _make_universe():
    """{symbol: close_ndarray} for every symbol sharing the common
    400-bar/END_DATE index, including the SPY/QQQ benchmarks."""
    data = {sym: _drift_series(mu, sigma, seed=seed)
            for sym, (mu, sigma, seed) in _DRIFT.items()}
    data["HHHH"] = _reversal_series()
    data["SPY"] = _drift_series(0.08, 0.10, seed=100)
    data["QQQ"] = _drift_series(0.12, 0.12, seed=101)
    return data


def _ohlcv_rows(symbol, dates, closes):
    """(symbol, date, o, h, l, c, v) tuples for a straight INSERT into
    daily_prices -- open == high == low == close since nothing under test
    reads intrabar range."""
    return [
        (symbol, d.strftime("%Y-%m-%d"), float(c), float(c), float(c), float(c), 1_000_000)
        for d, c in zip(dates, closes)
    ]


class TestStrategiesRoute(unittest.TestCase):

    def setUp(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.temp_db_path = path
        db.DB_PATH = path
        db.init_db()

        self.dates = pd.bdate_range(end=END_DATE, periods=N_BARS)
        self._seed(self.dates, _make_universe())

        # A ticker with real, uncontaminated rows but well under the
        # 273-bar (MOMENTUM_LOOKBACK + MOMENTUM_EXCLUDE) scoring minimum --
        # exercises the ticker tab's "insufficient history" error path.
        short_dates = pd.bdate_range(end=END_DATE, periods=100)
        self._seed(short_dates, {"SHORT1": _drift_series(0.10, 0.05, n=100, seed=50)})

        self.client = flask_app.test_client()
        self.client.testing = True

        # Keep the suite offline: the ticker tab now reads through
        # get_or_fetch_prices, so without this a cache miss or stale symbol
        # would hit yfinance. Returning None exercises the stale-while-error
        # fallback (seeded rows are served); the auto-download test patches
        # the fetcher per-test with a successful response instead.
        fetch_patcher = mock.patch.object(
            app_module, "_fetch_yfinance_with_retry", return_value=None
        )
        fetch_patcher.start()
        self.addCleanup(fetch_patcher.stop)
        app_module._PRICE_MEMO.clear()
        self.addCleanup(app_module._PRICE_MEMO.clear)

    def tearDown(self):
        try:
            os.remove(self.temp_db_path)
        except OSError:
            pass

    @staticmethod
    def _seed(dates, data):
        rows = []
        for sym, closes in data.items():
            rows.extend(_ohlcv_rows(sym, dates, closes))
        with db.get_conn() as conn:
            conn.executemany(
                "INSERT INTO daily_prices (symbol, date, open, high, low, close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )

    # -- 1. 3 tabs x 3 strategies -----------------------------------------

    def test_all_tabs_and_strategies_return_200_with_correct_label(self):
        for tab in ("universe", "ticker", "screener"):
            for strategy_id, meta in momentum_engine.STRATEGIES.items():
                with self.subTest(tab=tab, strategy=strategy_id):
                    qs = f"tab={tab}&strategy={strategy_id}"
                    if tab == "ticker":
                        qs += "&symbol=AAAA"
                    resp = self.client.get(f"/strategies?{qs}")
                    self.assertEqual(resp.status_code, 200)
                    # Not just 200 -- the page actually reflects the
                    # strategy that was requested (rendered unconditionally
                    # in the header, regardless of tab or error state).
                    self.assertIn(meta["label"].encode(), resp.data)

    # -- 2. strategy=bogus falls back, doesn't 500 -------------------------

    def test_bogus_strategy_falls_back_to_relative_strength(self):
        resp = self.client.get("/strategies?tab=universe&strategy=bogus")
        self.assertEqual(resp.status_code, 200)
        default_label = momentum_engine.STRATEGIES["relative_strength"]["label"]
        self.assertIn(default_label.encode(), resp.data)

    # -- 3. period= variants on the universe tab ---------------------------

    def test_universe_period_variants_return_200(self):
        for period in ("all", "3y", "1y", "6m", "3m"):
            with self.subTest(period=period):
                resp = self.client.get(f"/strategies?tab=universe&period={period}")
                self.assertEqual(resp.status_code, 200)

    # -- 4. error paths render, don't raise --------------------------------

    def test_ticker_not_in_cached_universe_renders_error(self):
        """A symbol with no cached rows AND a failed download renders the
        error page rather than raising."""
        resp = self.client.get("/strategies?tab=ticker&symbol=NOTREAL")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"is not cached and automatic download failed", resp.data)

    def test_uncached_ticker_auto_downloads_history(self):
        """Scanning a ticker with no cached rows downloads its history from
        yfinance and stores it, instead of failing with the cache error."""
        idx = pd.bdate_range(end=END_DATE, periods=N_BARS)
        closes = pd.Series(_drift_series(0.20, 0.05, seed=77), index=idx)
        fetched = pd.DataFrame({
            "Open": closes, "High": closes, "Low": closes,
            "Close": closes, "Volume": 1_000_000,
        })
        with mock.patch.object(
            app_module, "_fetch_yfinance_with_retry", return_value=fetched
        ):
            resp = self.client.get("/strategies?tab=ticker&symbol=NEW1")
        self.assertEqual(resp.status_code, 200)
        self.assertNotIn(b"Scan Failed", resp.data)
        self.assertIn(b"NEW1", resp.data)
        # The download is persisted, so later scans/score runs see it.
        stored = db.get_prices("NEW1")
        self.assertIsNotNone(stored)
        min_bars = momentum_engine.MOMENTUM_LOOKBACK + momentum_engine.MOMENTUM_EXCLUDE
        self.assertGreaterEqual(len(stored), min_bars)

    def test_benchmark_etf_scan_succeeds_with_genuine_rank(self):
        """SPY is excluded from the cached universe list (benchmark ETF), so
        the old available_symbols membership check rejected it even when its
        history was cached. A direct scan must serve the cached rows and rank
        it against the universe + itself."""
        resp = self.client.get("/strategies?tab=ticker&symbol=SPY")
        self.assertEqual(resp.status_code, 200)
        self.assertNotIn(b"Scan Failed", resp.data)
        m = re.search(rb"Rank #(\d+) of (\d+)", resp.data)
        self.assertIsNotNone(m, "expected the header badge to carry a universe rank")
        self.assertEqual(int(m.group(2)), len(STOCK_SYMBOLS) + 1)

    def test_benchmark_only_cache_still_scans(self):
        """A cache holding only benchmark ETFs yields an EMPTY universe list
        (benchmarks are filtered out); the old empty-cache guard rejected any
        request before the ticker tab could run. A direct SPY scan against
        such a cache must still render with itself as the whole ranking."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        orig_db_path = db.DB_PATH
        db.DB_PATH = path
        db.init_db()

        def _restore():
            db.DB_PATH = orig_db_path
            try:
                os.remove(path)
            except OSError:
                pass

        self.addCleanup(_restore)
        self._seed(self.dates, {"SPY": _drift_series(0.08, 0.10, seed=100)})

        resp = self.client.get("/strategies?tab=ticker&symbol=SPY")
        self.assertEqual(resp.status_code, 200)
        self.assertNotIn(b"Scan Failed", resp.data)
        self.assertIn(b"Rank #1 of 1", resp.data)

    def test_ticker_with_insufficient_history_renders_error(self):
        resp = self.client.get("/strategies?tab=ticker&symbol=SHORT1")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"insufficient price history", resp.data)

    # -- 5. screener abs_only=1 is a genuine subset ------------------------

    def test_screener_abs_only_yields_genuine_subset(self):
        # Wide-open filters so abs_only is the ONLY thing distinguishing
        # the two requests -- min_mom/max_vol/min_risk_adj all disabled,
        # trend set to a value the route treats as "no trend filter".
        common = (
            "tab=screener&strategy=dual_momentum&trend=all"
            "&min_mom=-1000&max_vol=100000&min_risk_adj=-1000"
        )
        resp_off = self.client.get(f"/strategies?{common}&abs_only=0")
        resp_on = self.client.get(f"/strategies?{common}&abs_only=1")
        self.assertEqual(resp_off.status_code, 200)
        self.assertEqual(resp_on.status_code, 200)

        def screened_count(resp):
            m = re.search(rb"Screened (\d+) Tickers", resp.data)
            self.assertIsNotNone(m, "expected a 'Screened N Tickers' badge in the response")
            return int(m.group(1))

        count_off = screened_count(resp_off)
        count_on = screened_count(resp_on)

        # Every one of the 8 scorable symbols clears the wide-open filters
        # (SHORT1 has too little history to be scored at all and drops out
        # of both counts the same way).
        self.assertEqual(count_off, len(STOCK_SYMBOLS))
        self.assertGreater(count_on, 0)
        self.assertLess(count_on, count_off)

    def test_screener_non_numeric_filters_do_not_500(self):
        self.client.get("/strategies?tab=screener&min_mom=abc&max_vol=xyz&min_risk_adj=1.5e999")
        # assert on the LAST response
        resp = self.client.get("/strategies?tab=screener&min_mom=abc&max_vol=xyz")
        self.assertEqual(resp.status_code, 200)

    # -- 6. dual_momentum exposes a real passes_absolute split -------------

    def test_dual_momentum_universe_leaderboard_has_real_pass_fail_split(self):
        price_df = _load_price_frame(STOCK_SYMBOLS + ["SPY", "QQQ"])
        data = _strategies_universe_data(STOCK_SYMBOLS, price_df, "dual_momentum", "all")
        self.assertIsNotNone(data)

        passes = [row["passes_absolute"] for row in data["leaderboard"]]
        self.assertTrue(any(passes), "expected at least one symbol to clear the absolute hurdle")
        self.assertFalse(all(passes), "expected at least one symbol to fail the absolute hurdle")

    # -- F6 regressions: _strategies_ticker_data honors the None contract --

    def test_ticker_data_returns_none_below_scoring_minimum(self):
        """Previously raised AttributeError ('NoneType' object has no
        attribute 'mom_12_1') once `score_series` returned None for a frame
        at least warmup-long (254+ bars) but short of the 273-bar scoring
        minimum."""
        dates = pd.bdate_range(end=END_DATE, periods=260)
        df = pd.DataFrame(
            {"close": _drift_series(0.10, 0.05, n=260, seed=42)}, index=dates
        )
        result = _strategies_ticker_data(
            "SHORT260", df, {}, ["SHORT260"], "relative_strength", "all"
        )
        self.assertIsNone(result)

    def test_ticker_data_returns_none_below_backtest_warmup(self):
        """Previously raised IndexError (`backtest_dates[0]` on an empty
        DatetimeIndex) for a frame shorter than the 253-bar backtest
        warmup."""
        dates = pd.bdate_range(end=END_DATE, periods=100)
        df = pd.DataFrame(
            {"close": _drift_series(0.10, 0.05, n=100, seed=43)}, index=dates
        )
        result = _strategies_ticker_data(
            "SHORT100", df, {}, ["SHORT100"], "relative_strength", "all"
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
