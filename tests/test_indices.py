import unittest
from unittest import mock

import indices


def _ndx_payload(n=101, **row):
    rows = [{"symbol": f"T{i}", "companyName": f"Tick {i} Inc. Common Stock", "sector": ""} for i in range(n)]
    rows[0] = {"symbol": "BRK.B", "companyName": "Berkshire Common Stock", "sector": "Finance", **row}
    return {"data": {"data": {"rows": rows}}}


def _dow_html(n=30, symbol_col="Symbol"):
    head = f"<tr><th>Company</th><th>Exchange</th><th>{symbol_col}</th><th>Sector</th></tr>"
    body = "".join(f"<tr><td>Co {i}</td><td>NYSE</td><td>S{i}</td><td>{'Industrials' if i else ''}</td></tr>" for i in range(n))
    # A decoy table without a Symbol column must be skipped.
    return f"<table><tr><th>Year</th></tr><tr><td>1</td></tr></table><table>{head}{body}</table>"


class TestParseNdx(unittest.TestCase):
    def test_normalises_symbols_and_names(self):
        rows = indices.parse_ndx(_ndx_payload())
        self.assertEqual(101, len(rows))
        self.assertEqual({"symbol": "BRK-B", "name": "Berkshire", "sector": "Finance"}, rows[0])
        self.assertEqual("Tick 1 Inc.", rows[1]["name"])

    def test_shape_change_raises(self):
        with self.assertRaises(RuntimeError):
            indices.parse_ndx({"data": None})

    def test_implausible_size_raises(self):
        with self.assertRaises(RuntimeError):
            indices.parse_ndx(_ndx_payload(n=10))


class TestFetchHeaders(unittest.TestCase):
    def test_ndx_request_merges_accept_into_the_default_headers(self):
        resp = mock.Mock(**{"json.return_value": _ndx_payload(), "raise_for_status.return_value": None})
        with mock.patch("indices.requests.get", return_value=resp) as get:
            self.assertEqual(101, len(indices.fetch_ndx()))
        sent = get.call_args.kwargs["headers"]
        self.assertEqual("application/json", sent["Accept"])
        self.assertIn("User-Agent", sent)


class TestParseDow(unittest.TestCase):
    def test_picks_the_symbol_table(self):
        rows = indices.parse_dow(_dow_html())
        self.assertEqual(30, len(rows))
        self.assertEqual({"symbol": "S1", "name": "Co 1", "sector": "Industrials"}, rows[1])
        self.assertEqual("", rows[0]["sector"])   # blank (NaN) sector stays a string

    def test_missing_or_wrong_size_table_raises(self):
        for html in (_dow_html(symbol_col="Ticker"), _dow_html(n=5), "<p>no tables</p>"):
            with self.assertRaises(RuntimeError):
                indices.parse_dow(html)


if __name__ == "__main__":
    unittest.main()
