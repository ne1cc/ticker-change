"""API Schema Contract Tests (Issue #161).

Enforces strict parity between runtime API response payloads and OpenAPI 3.0 schemas.
Fails CI if:
1. An endpoint returns response keys not documented in OpenAPI schema properties.
2. OpenAPI schema requires keys not present in the runtime response.
3. Key names drift (e.g., trade_count vs total_trades, spread vs spread_pct).
"""
import os
import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np

import api_docs
from app import app
import microstructure
import macro_engine
from backtest_engine import BacktestSummary, PermutationResult


def _mock_prices(n=300):
    idx = pd.bdate_range(end="2026-10-20", periods=n)
    close = pd.Series(100.0 + np.cumsum(np.random.normal(0.05, 1.0, n)), index=idx, name="close")
    return pd.DataFrame({
        "open": close * 0.99,
        "high": close * 1.02,
        "low": close * 0.98,
        "close": close,
        "volume": np.full(n, 1_000_000),
    }, index=idx)


class TestApiContractParity(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.spec = api_docs.get_openapi_spec()
        self.schemas = self.spec["components"]["schemas"]

    def test_analytics_api_contract(self):
        """Verify /api/analytics/{ticker} strictly matches AnalyticsResponse schema."""
        schema = self.schemas["AnalyticsResponse"]
        schema_props = schema["properties"]
        stats_props = schema_props["stats"]["properties"]

        with patch("app.get_or_fetch_prices", return_value=_mock_prices()):
            resp = self.client.get("/api/analytics/AAPL")

        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()

        # Check required fields
        for req in schema.get("required", []):
            self.assertIn(req, data, f"Required field '{req}' missing from /api/analytics payload")

        # Check top-level keys parity
        for key in data:
            self.assertIn(key, schema_props, f"Undocumented response key '{key}' in /api/analytics response (schema drift!)")

        # Check stats object keys parity
        stats_data = data.get("stats", {})
        for key in stats_data:
            self.assertIn(key, stats_props, f"Undocumented stat '{key}' in /api/analytics stats payload (schema drift!)")

        # Verify seasonality_obs format
        if data.get("seasonality_obs") is not None:
            self.assertIsInstance(data["seasonality_obs"], list)
            self.assertEqual(len(data["seasonality_obs"]), 2)

    def test_institutional_api_contract(self):
        """Verify /api/institutional/{ticker} strictly matches InstitutionalAnalyticsResponse schema."""
        schema = self.schemas["InstitutionalAnalyticsResponse"]
        schema_props = schema["properties"]

        dummy_micro = microstructure.MicrostructureAnalytics(
            corwin_schultz_spread_pct=0.42,
            roll_effective_spread_pct=0.38,
            amihud_illiquidity=0.012,
            squeeze_risk_score=45.0,
            squeeze_risk_level="MODERATE",
            squeeze_inputs_assumed=["SI % float", "DTC"]
        )
        dummy_macro = macro_engine.MacroRegimeReport(
            fed_funds_rate=4.5,
            yield_curve_2s10s_spread=0.35,
            cpi_inflation_yoy=2.8,
            is_yield_curve_inverted=False,
            current_regime="NEUTRAL_EXPANSION",
            regime_conditional_betas={"overall": 1.15, "bull_beta": 1.25, "bear_beta": 0.95, "bull_days": 310, "bear_days": 254},
            macro_regime_betas={},
            upside_capture_ratio=112.0,
            downside_capture_ratio=88.0,
            cross_asset_correlations={},
        )
        dummy_sim = BacktestSummary(
            total_return_pct=28.5,
            cagr_pct=14.2,
            benchmark_return_pct=15.2,
            alpha_pct=13.3,
            annualized_sharpe=1.65,
            annualized_sortino=2.15,
            max_drawdown_pct=-8.4,
            calmar_ratio=1.69,
            win_rate_pct=58.2,
            profit_factor=1.45,
            total_trades=42,
            factors_used=["MOM_12_1"],
            n_folds=5,
            oos_sharpe_mean=1.25,
            oos_sharpe_std=0.35,
            fold_returns=[0.05, 0.04],
            deflated_sharpe=0.92,
            n_trials_assumed=10,
        )
        dummy_perm = PermutationResult(
            real_sharpe=1.65,
            null_mean=0.12,
            null_std=0.78,
            p_value=0.0396,
            n_permutations=100,
        )

        with patch.dict(os.environ, {"WARM_CACHE_TOKEN": "test-token"}), \
             patch("app.get_or_fetch_prices", return_value=_mock_prices()), \
             patch("microstructure.get_microstructure_analytics", return_value=dummy_micro), \
             patch("macro_engine.get_macro_financial_report", return_value=dummy_macro), \
             patch("app._institutional_backtest", return_value={"signals_backtest": dummy_sim.__dict__, "permutation_test": dummy_perm.__dict__}), \
             patch("sec_8k.fetch_and_parse_8k_filings", return_value=[]), \
             patch("providers.get_short_interest", return_value={"shares_short": 1000, "short_pct_float": 0.02, "days_to_cover": 1.5}):

            resp = self.client.get("/api/institutional/AAPL", headers={"Authorization": "Bearer test-token"})

        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()

        # Check required fields
        for req in schema.get("required", []):
            self.assertIn(req, data, f"Required field '{req}' missing from /api/institutional payload")

        # Check top-level keys parity
        for key in data:
            self.assertIn(key, schema_props, f"Undocumented response key '{key}' in /api/institutional response (schema drift!)")

        # Check microstructure keys
        micro_props = schema_props["microstructure"]["properties"]
        for key in data.get("microstructure", {}):
            self.assertIn(key, micro_props, f"Undocumented key '{key}' in microstructure schema")

        # Check short_interest keys
        si_props = schema_props["short_interest"]["properties"]
        for key in data.get("short_interest", {}):
            self.assertIn(key, si_props, f"Undocumented key '{key}' in short_interest schema")

        # Check macro_conditioning keys
        macro_props = schema_props["macro_conditioning"]["properties"]
        for key in data.get("macro_conditioning", {}):
            self.assertIn(key, macro_props, f"Undocumented key '{key}' in macro_conditioning schema")

        # Check signals_backtest keys
        bt_props = schema_props["signals_backtest"]["properties"]
        for key in data.get("signals_backtest", {}):
            self.assertIn(key, bt_props, f"Undocumented key '{key}' in signals_backtest schema")

        # Ensure total_trades is used and trade_count is NOT present
        self.assertIn("total_trades", data["signals_backtest"])
        self.assertNotIn("trade_count", data["signals_backtest"])

    def test_schema_drift_fails_when_unexpected_key_present(self):
        """Verify that unmapped runtime keys explicitly raise an assertion error."""
        schema = self.schemas["AnalyticsResponse"]
        schema_props = schema["properties"]
        mock_payload = {"ticker": "AAPL", "current_price": 100.0, "stats": {}, "unmapped_drift_key": 123}

        with self.assertRaises(AssertionError) as ctx:
            for key in mock_payload:
                self.assertIn(key, schema_props, f"Undocumented response key '{key}' in /api/analytics response (schema drift!)")
        self.assertIn("unmapped_drift_key", str(ctx.exception))
