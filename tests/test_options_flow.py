import unittest
from unittest.mock import patch

import pandas as pd

import app as app_module
import options_flow
import providers


def _chain():
    base = dict(expiration="2026-11-20", dte=30, bid=0, ask=0, iv=0.3)
    return pd.DataFrame([
        dict(base, strike=100.0, cp="C", last_price=2.0, volume=5000, open_interest=1000),  # unusual
        dict(base, strike=95.0, cp="P", last_price=1.0, volume=3000, open_interest=0),      # new, smaller premium
        dict(base, strike=110.0, cp="C", last_price=2.0, volume=500, open_interest=9000),   # vol < OI
        dict(base, strike=120.0, cp="C", last_price=0.05, volume=200, open_interest=10),    # premium too small
    ])


class TestUnusualActivity(unittest.TestCase):
    def test_occ_symbol(self):
        self.assertEqual(options_flow.occ_symbol("aapl", "2024-06-21", "C", 190.0), "AAPL240621C00190000")
        self.assertEqual(options_flow.occ_symbol("SPY", "2026-11-20", "P", 412.5), "SPY261120P00412500")

    def test_filters_and_ranks_by_premium(self):
        rows = options_flow.unusual_activity(_chain(), "XYZ")
        self.assertEqual([r["strike"] for r in rows], [100.0, 95.0])
        self.assertEqual(rows[0]["premium"], 5000 * 2.0 * 100)
        self.assertEqual(rows[0]["vol_oi"], 5.0)
        self.assertIsNone(rows[1]["vol_oi"])  # zero OI -> "new"
        self.assertEqual(rows[0]["symbol"], "XYZ261120C00100000")

    def test_empty(self):
        self.assertEqual(options_flow.unusual_activity(pd.DataFrame(), "XYZ"), [])

    def test_largest_prints(self):
        trades = {"A": [{"t": "t1", "p": 1.0, "s": 10, "x": "C"}, {"t": "t2", "p": 0, "s": 5}],
                  "B": [{"t": "t3", "p": 2.0, "s": 100, "x": "C"}]}
        rows = options_flow.largest_prints(trades)
        self.assertEqual([r["symbol"] for r in rows], ["B", "A"])
        self.assertEqual(rows[0]["premium"], 2.0 * 100 * 100)


class TestAlpacaTrades(unittest.TestCase):
    def test_none_without_credentials(self):
        with patch.object(providers, "alpaca_credentials", return_value=None):
            self.assertIsNone(providers.alpaca_option_trades("XYZ", ["XYZ261120C00100000"]))

    def test_request_shape_and_parse(self):
        payload = {"trades": {"XYZ261120C00100000": [{"t": "t", "p": 1.0, "s": 1}]}}
        with patch.object(providers, "alpaca_credentials", return_value=("id", "secret")), \
             patch.object(providers.db, "cache_get", return_value=None), \
             patch.object(providers.db, "cache_set"), \
             patch.object(providers, "_get_json", return_value=payload) as get:
            out = providers.alpaca_option_trades("XYZ", ["XYZ261120C00100000"])
        self.assertEqual(out, payload["trades"])
        kwargs = get.call_args.kwargs
        self.assertEqual(kwargs["headers"]["APCA-API-KEY-ID"], "id")
        self.assertEqual(kwargs["params"]["feed"], "indicative")

    def test_failure_returns_none(self):
        with patch.object(providers, "alpaca_credentials", return_value=("id", "secret")), \
             patch.object(providers.db, "cache_get", return_value=None), \
             patch.object(providers, "_get_json", return_value=None):
            self.assertIsNone(providers.alpaca_option_trades("XYZ", ["S"]))


class TestOptionsPageFlow(unittest.TestCase):
    def _render(self, trades):
        client = app_module.app.test_client()
        with patch.object(app_module, "get_current_price_yfinance", return_value=100.0), \
             patch.object(app_module, "get_full_option_chain_df", return_value=_chain()), \
             patch.object(app_module, "get_or_fetch_prices", return_value=pd.DataFrame()), \
             patch.object(providers, "alpaca_option_trades", return_value=trades), \
             patch.object(providers, "alpaca_credentials", return_value=("i", "s") if trades is not None else None):
            resp = client.get("/options?ticker=XYZ")
        self.assertEqual(resp.status_code, 200)
        return resp.get_data(as_text=True)

    def test_card_renders_with_prints(self):
        body = self._render({"XYZ261120C00100000": [{"t": "2026-10-16T15:00:00Z", "p": 2.0, "s": 50, "x": "C"}]})
        self.assertIn("Unusual activity", body)
        self.assertIn("XYZ261120C00100000", body)

    def test_card_prompts_for_key_when_unconfigured(self):
        self.assertIn("Add a free Alpaca key", self._render(None))


if __name__ == "__main__":
    unittest.main()
