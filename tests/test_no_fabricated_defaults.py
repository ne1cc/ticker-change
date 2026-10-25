"""E10.5 regression guard (#164): with EVERY provider returning None/empty,
the /analytics pipeline must leave provider-derived fields None or carry a
declared state -- never a fabricated number.

Mocks sit at the provider boundary (established in test_analytics_honesty.py):
prices are supplied because the pipeline refuses to run without them; FRED,
short interest, option chain, earnings, insider, Finnhub price target and
yfinance (ticker handle for fundamentals/insider) are all emptied. SEC 8-K is
emptied too: a live failure there would blank the whole institutional block
and make the asserts vacuous.
"""
import contextlib
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import app as app_module
import db as db_module
import earnings as earnings_module
import macro_engine
import sec_8k as sec_8k_module


def _prices(n=400, seed=3):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, n)))
    idx = pd.bdate_range(end="2026-10-20", periods=n)
    return pd.DataFrame({"open": close * 0.999, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "volume": rng.integers(1_000_000, 2_000_000, n)}, index=idx)


@pytest.fixture
def empty_providers(monkeypatch, tmp_path):
    """Run the real /analytics route with every provider empty; yield the data
    dict captured just before render_template."""
    # Hermetic DB: the repo's stocks.db could hold cached provider payloads.
    monkeypatch.setattr(db_module, "DB_PATH", str(tmp_path / "guard.db"))
    db_module.init_db()
    captured = {}
    full_data = {}

    def _capture(template, **kw):
        captured.update(kw)
        if "data" in kw and isinstance(kw["data"], dict):
            for k, v in kw["data"].items():
                if k == "charts" and isinstance(v, dict):
                    full_data.setdefault("charts", {}).update(v)
                else:
                    full_data[k] = v
        return ""

    with contextlib.ExitStack() as stack:
        for p in (
            patch.object(app_module, "get_or_fetch_prices", return_value=_prices()),
            patch.object(app_module, "_get_yf_ticker", return_value=object()),
            patch.object(app_module, "get_cached_expirations", return_value=[]),
            patch.object(app_module, "get_cached_chain", return_value=None),
            patch.object(app_module.providers, "get_short_interest", return_value=None),
            patch.object(app_module.providers, "finnhub_price_target", return_value=None),
            patch.object(macro_engine, "fetch_fred_series", return_value=[]),
            patch.object(earnings_module, "fetch_earnings_history", return_value=None),
            patch.object(sec_8k_module, "fetch_and_parse_8k_filings", return_value=[]),
            patch.object(app_module, "render_template", side_effect=_capture),
        ):
            stack.enter_context(p)
        client = app_module.app.test_client()
        resp = client.get("/analytics?ticker=TEST")
        assert resp.status_code == 200
        assert "error" not in captured, f"pipeline failed: {captured.get('error')}"
        client.get("/analytics/fragment/company?ticker=TEST")
        client.get("/analytics/fragment/options?ticker=TEST")
    assert full_data.get("institutional") is not None, \
        "institutional block crashed; asserts would be vacuous"
    yield full_data


def test_fred_fields_are_none_without_fred(empty_providers):
    macro = empty_providers["institutional"]["macro_conditioning"]
    assert macro["fed_funds_rate"] is None
    assert macro["yield_curve_2s10s"] is None
    assert macro["is_inverted"] is None
    assert macro["cpi_yoy"] is None
    assert macro["regime"] is None


def test_short_interest_declares_assumed_not_numbers(empty_providers):
    inst = empty_providers["institutional"]
    assert inst["short_interest"] == {"si_source": "assumed"}
    # The squeeze index may run on signature defaults only if declared.
    assert inst["microstructure"]["squeeze_inputs_assumed"] == ["SI % float", "DTC"]


def test_option_chain_fields_hidden_without_chains(empty_providers):
    data = empty_providers
    assert data["charts"]["options_smile"] is None
    assert data["charts"]["gex"] is None
    assert data["gex"] is None
    inst = data["institutional"]
    assert "atm_contract" not in inst
    assert "vrp" not in inst
    assert "higher_order_greeks" not in inst


def test_earnings_section_is_none_without_history(empty_providers):
    assert empty_providers["earnings"] is None


def test_insider_fields_are_none_without_filings(empty_providers):
    data = empty_providers
    assert data["insider_summary"] is None
    assert data["charts"]["insider"] is None


def test_price_target_and_fundamentals_are_none(empty_providers):
    data = empty_providers
    assert data["charts"]["price_target"] is None
    assert data["fundamentals"] is None
