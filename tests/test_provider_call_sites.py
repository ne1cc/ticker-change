"""Regression tests for provider helpers that were called but never defined.

Three same-class bugs shipped to main: `providers._sec_get` (called from
`sec_8k.py`), `finnhub_insider_transactions` (called from
`providers.get_insider_transactions`), and a bare `Optional` annotation in
`app.py`. All three are name errors that only surface when the call site is
actually reached, so nothing in the suite caught them.

`TestNoUndefinedCrossModuleAttributes` closes that gap statically for the
`module.attr` shape; the behavioural tests below cover the call paths
themselves. Running `pyflakes *.py` catches the wider bare-name shape.
"""
import ast
import os
import pathlib
import tempfile
import unittest
from unittest import mock

import db
import providers
import sec_8k


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _module_toplevel_names(tree: ast.Module) -> set:
    """Names a module binds at import time: defs, classes, assignments, imports."""
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(a.asname or a.name.split(".")[0] for a in node.names)
    return names


class TestNoUndefinedCrossModuleAttributes(unittest.TestCase):
    def test_every_local_module_attribute_reference_resolves(self):
        files = sorted(REPO_ROOT.glob("*.py"))
        local = {p.stem for p in files}
        trees = {p.stem: ast.parse(p.read_text()) for p in files}
        defined = {stem: _module_toplevel_names(t) for stem, t in trees.items()}

        misses = []
        for stem, tree in trees.items():
            imported = {
                alias.asname or alias.name
                for node in ast.walk(tree) if isinstance(node, ast.Import)
                for alias in node.names if alias.name in local
            }
            for node in ast.walk(tree):
                if (isinstance(node, ast.Attribute)
                        and isinstance(node.value, ast.Name)
                        and node.value.id in imported
                        and node.attr not in defined.get(node.value.id, set())):
                    misses.append(
                        f"{stem}.py:{node.lineno}: {node.value.id}.{node.attr} "
                        f"is not defined in {node.value.id}.py"
                    )

        self.assertEqual([], misses, "\n".join(misses))


class ProviderTestCase(unittest.TestCase):
    """Base case giving each test a throwaway SQLite db for the `_cached` layer."""

    def setUp(self):
        self.orig_db_path = db.DB_PATH
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        db.DB_PATH = self.temp_db.name
        db.init_db()

    def tearDown(self):
        db.DB_PATH = self.orig_db_path
        try:
            os.remove(self.temp_db.name)
        except OSError:
            pass


FINNHUB_TX_PAYLOAD = {
    "symbol": "AAPL",
    "data": [
        {
            "name": "COOK TIMOTHY", "share": 3280, "change": -511000,
            "filingDate": "2024-04-03", "transactionDate": "2024-04-01",
            "transactionCode": "S", "transactionPrice": 169.65,
        },
        {
            "name": "ADAMS KATHERINE", "share": 447000, "change": 22000,
            "filingDate": "2024-10-05", "transactionDate": "2024-10-03",
            "transactionCode": "A", "transactionPrice": 0,
        },
    ],
}


class TestFinnhubInsiderTransactions(ProviderTestCase):
    def test_normalizes_finnhub_rows_to_the_shared_shape(self):
        with mock.patch.object(providers, "finnhub_keys", return_value=["k"]), \
             mock.patch.object(providers, "_finnhub_get", return_value=FINNHUB_TX_PAYLOAD):
            rows = providers.finnhub_insider_transactions("AAPL")

        self.assertEqual(2, len(rows))
        # Sorted newest filing first, regardless of payload order.
        self.assertEqual("2024-10-05", rows[0]["filing_date"])

        sale = next(r for r in rows if r["name"] == "COOK TIMOTHY")
        # `change` (signed delta) is the display figure, not `share` (holdings).
        self.assertEqual(-511000, sale["shares"])
        self.assertFalse(sale["is_buy"])
        self.assertEqual("S", sale["code"])
        self.assertAlmostEqual(169.65, sale["price"])
        self.assertEqual("finnhub", sale["source"])

        grant = next(r for r in rows if r["name"] == "ADAMS KATHERINE")
        self.assertTrue(grant["is_buy"])
        self.assertIsNone(grant["price"], "a zero grant price should render as '-', not $0.00")

        # Every row carries the keys templates/positioning.html reads.
        for row in rows:
            self.assertEqual(
                {"name", "shares", "is_buy", "price", "code",
                 "filing_date", "transaction_date", "source"},
                set(row),
            )

    def test_returns_none_without_a_key_instead_of_raising(self):
        with mock.patch.object(providers, "finnhub_keys", return_value=[]):
            self.assertIsNone(providers.finnhub_insider_transactions("AAPL"))

    def test_respects_the_limit(self):
        with mock.patch.object(providers, "finnhub_keys", return_value=["k"]), \
             mock.patch.object(providers, "_finnhub_get", return_value=FINNHUB_TX_PAYLOAD):
            self.assertEqual(1, len(providers.finnhub_insider_transactions("AAPL", limit=1)))


class TestInsiderCascade(ProviderTestCase):
    def test_finnhub_answers_first(self):
        with mock.patch.object(providers, "finnhub_keys", return_value=["k"]), \
             mock.patch.object(providers, "_finnhub_get", return_value=FINNHUB_TX_PAYLOAD), \
             mock.patch.object(providers, "yfinance_insider_transactions") as yf_tx:
            rows = providers.get_insider_transactions("AAPL")

        self.assertEqual("finnhub", rows[0]["source"])
        yf_tx.assert_not_called()

    def test_falls_through_to_yfinance_when_finnhub_is_unconfigured(self):
        """The NameError this covers meant the working yfinance feed was never reached."""
        yf_rows = [{"name": "Insider", "shares": 100, "is_buy": True, "price": None,
                    "code": "P", "filing_date": "2024-01-02",
                    "transaction_date": "2024-01-01", "source": "yfinance"}]
        with mock.patch.object(providers, "finnhub_keys", return_value=[]), \
             mock.patch.object(providers, "yfinance_insider_transactions", return_value=yf_rows):
            rows = providers.get_insider_transactions("AAPL")

        self.assertEqual(yf_rows, rows)

    def test_returns_none_when_every_feed_is_empty(self):
        with mock.patch.object(providers, "finnhub_keys", return_value=[]), \
             mock.patch.object(providers, "yfinance_insider_transactions", return_value=None):
            self.assertIsNone(providers.get_insider_transactions("AAPL"))


SUBMISSIONS_PAYLOAD = {
    "filings": {
        "recent": {
            "form": ["8-K", "10-Q", "8-K"],
            "filingDate": ["2024-05-02", "2024-04-30", "2024-02-01"],
            "reportDate": ["2024-05-01", "2024-03-31", ""],
            "accessionNumber": ["0000320193-24-000069", "0000320193-24-000068",
                                "0000320193-24-000010"],
            "primaryDocument": ["aapl-20240501.htm", "aapl-20240331.htm",
                                "aapl-20240201.htm"],
            "items": ["2.02,7.01", "", "4.02"],
            "primaryDocDescription": ["8-K", "10-Q", "8-K"],
        }
    }
}


class TestSec8kFetch(ProviderTestCase):
    def test_parses_submissions_without_hitting_the_missing_helper(self):
        """`providers._sec_get` never existed; the AttributeError nuked all of
        /analytics' institutional block and 500'd /api/institutional."""
        with mock.patch.object(providers, "sec_cik_for_ticker", return_value=320193), \
             mock.patch.object(providers, "_get_json",
                               return_value=SUBMISSIONS_PAYLOAD) as get_json:
            events = sec_8k.fetch_and_parse_8k_filings("AAPL", limit=5)

        get_json.assert_called_once()
        _, kwargs = get_json.call_args
        self.assertIn("User-Agent", kwargs["headers"])

        # Only the two 8-Ks, not the 10-Q.
        self.assertEqual(2, len(events))
        self.assertEqual(["2.02", "7.01"], events[0].items)
        self.assertFalse(events[0].is_high_impact)

        restatement = events[1]
        self.assertEqual(["4.02"], restatement.items)
        self.assertTrue(restatement.is_high_impact, "Item 4.02 restatements are high impact")
        # reportDate is blank for this filing, so it falls back to filingDate.
        self.assertEqual("2024-02-01", restatement.report_date)
        self.assertIsInstance(restatement.cik, str)
        self.assertIn("/Archives/edgar/data/320193/", restatement.primary_doc_url)

    def test_round_trips_through_the_cache(self):
        """Cached rows rebuild into dataclasses - `cik` must survive as declared."""
        with mock.patch.object(providers, "sec_cik_for_ticker", return_value=320193), \
             mock.patch.object(providers, "_get_json", return_value=SUBMISSIONS_PAYLOAD):
            first = sec_8k.fetch_and_parse_8k_filings("AAPL", limit=5)
            with mock.patch.object(providers, "_get_json") as second_fetch:
                second = sec_8k.fetch_and_parse_8k_filings("AAPL", limit=5)
                second_fetch.assert_not_called()

        self.assertEqual([e.__dict__ for e in first], [e.__dict__ for e in second])

    def test_unknown_ticker_returns_empty(self):
        with mock.patch.object(providers, "sec_cik_for_ticker", return_value=None):
            self.assertEqual([], sec_8k.fetch_and_parse_8k_filings("NOPE"))


if __name__ == "__main__":
    unittest.main()
