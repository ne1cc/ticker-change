"""ml.py's `_dataset()` only reads price history already cached in SQLite --
it never fetches from yfinance -- so training on the full S&P 500 needs an
explicit backfill step, and a way to get the constituent list without pasting
~500 tickers by hand. This tests the parsing/normalisation logic in isolation,
mocking the network call rather than hitting GitHub in the suite.
"""
import unittest
from unittest.mock import patch, Mock

import sp500


FAKE_CSV = (
    "Symbol,Security,GICS Sector,GICS Sub-Industry,Headquarters Location,Date added,CIK,Founded\n"
    "AAPL,Apple Inc.,Information Technology,Technology Hardware,\"Cupertino, CA\",1982-11-30,320193,1976\n"
    "BRK.B,Berkshire Hathaway,Financials,Multi-Sector Holdings,\"Omaha, NE\",2010-02-16,1067983,1839\n"
    "BF.B,Brown-Forman,Consumer Staples,Distillers & Vintners,\"Louisville, KY\",1999-07-01,14693,1870\n"
    "MSFT,Microsoft Corp.,Information Technology,Systems Software,\"Redmond, WA\",1994-06-01,789019,1975\n"
)


class TestParseTickersCsv(unittest.TestCase):
    def test_dot_share_classes_are_normalised_for_yfinance(self):
        """The official list uses '.' (BRK.B); yfinance requires '-' (BRK-B)."""
        tickers = sp500._parse_tickers_csv(FAKE_CSV)
        self.assertIn("BRK-B", tickers)
        self.assertIn("BF-B", tickers)
        self.assertNotIn("BRK.B", tickers)
        self.assertNotIn("BF.B", tickers)

    def test_result_is_sorted_and_deduplicated(self):
        csv_with_dupe = FAKE_CSV + "AAPL,Apple Inc. (dupe row),,,,,,\n"
        tickers = sp500._parse_tickers_csv(csv_with_dupe)
        self.assertEqual(tickers, sorted(tickers))
        self.assertEqual(1, tickers.count("AAPL"))

    def test_rows_missing_a_symbol_are_skipped(self):
        csv_with_blank = FAKE_CSV + ",No Symbol Here,,,,,,\n"
        tickers = sp500._parse_tickers_csv(csv_with_blank)
        self.assertEqual({"AAPL", "BRK-B", "BF-B", "MSFT"}, set(tickers))


class TestFetchSp500Tickers(unittest.TestCase):
    def test_network_failure_raises_a_clear_error_instead_of_hanging(self):
        with patch("sp500.requests.get", side_effect=ConnectionError("boom")):
            with self.assertRaisesRegex(RuntimeError, "Could not fetch"):
                sp500.fetch_sp500_tickers()

    def test_implausibly_short_list_raises_rather_than_silently_undertraining(self):
        """Guards against the upstream CSV's column name or format changing."""
        with patch("sp500.requests.get") as mock_get:
            mock_get.return_value = Mock(text=FAKE_CSV, raise_for_status=Mock())
            with self.assertRaisesRegex(RuntimeError, "implausibly short"):
                sp500.fetch_sp500_tickers()

    def test_a_plausibly_full_list_passes_through(self):
        full_csv = "Symbol\n" + "\n".join(f"TICK{i}" for i in range(450))
        with patch("sp500.requests.get") as mock_get:
            mock_get.return_value = Mock(text=full_csv, raise_for_status=Mock())
            tickers = sp500.fetch_sp500_tickers()
        self.assertEqual(450, len(tickers))


class TestBackfill(unittest.TestCase):
    def test_already_fresh_tickers_are_skipped_without_a_network_call(self):
        with patch("sp500.db.is_fresh", return_value=True), \
             patch("sp500._fetch_with_retry") as mock_fetch:
            sp500.backfill(["AAPL"])
        mock_fetch.assert_not_called()

    def test_stale_tickers_are_fetched_and_stored(self):
        fake_df = Mock(empty=False)
        fake_df.__len__ = Mock(return_value=1250)
        with patch("sp500.db.is_fresh", return_value=False), \
             patch("sp500._fetch_with_retry", return_value=fake_df) as mock_fetch, \
             patch("sp500.db.store_prices") as mock_store, \
             patch("sp500.time.sleep"):
            sp500.backfill(["AAPL"])
        mock_fetch.assert_called_once()
        mock_store.assert_called_once_with("AAPL", fake_df)

    def test_a_failed_fetch_does_not_stop_the_rest_of_the_list(self):
        good_df = Mock(empty=False)
        good_df.__len__ = Mock(return_value=1250)
        with patch("sp500.db.is_fresh", return_value=False), \
             patch("sp500._fetch_with_retry", side_effect=[None, good_df]), \
             patch("sp500.db.store_prices") as mock_store, \
             patch("sp500.time.sleep"):
            sp500.backfill(["BADTICKER", "GOODTICKER"])
        mock_store.assert_called_once_with("GOODTICKER", good_df)


if __name__ == "__main__":
    unittest.main()
