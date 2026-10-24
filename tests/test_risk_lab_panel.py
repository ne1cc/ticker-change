"""Epic 6 core wiring: credible-risk block on /analytics — declared states,
template rendering, position scaling."""
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import app as app_module
import risk_backtests


def _price_df(n=800, seed=6):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2026-10-01", periods=n)
    ret = pd.Series(rng.normal(0.0003, 0.012, n), index=idx)
    close = 100 * (1 + ret).cumprod()
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "volume": 1_000_000}, index=idx)


def test_risk_lab_block_structure():
    px = _price_df()
    out = app_module._build_risk_lab_block(px["close"].pct_change().dropna())
    assert out["available"] is True and out["n"] == 799
    assert set(out["tail"]) == {"var_95", "var_99"}
    assert out["tail"]["var_99"]["low_confidence"] is True  # 799d -> ~8 tail obs: correctly flagged
    assert "ewma_annualised_pct" in out["vol_forecast"]
    assert out["vol_forecast"]["garch_annualised_pct"] > 0
    assert "evaluation" in out["vol_forecast"]
    for bt in out["backtests"].values():
        assert bt["sufficient"] is True and bt["traffic"] in {"green", "amber", "red"}
    assert out["horizons"]["h10_var_95"]["windows"] == 790  # 799 - 10 + 1


def test_risk_lab_insufficient_history_declared():
    px = _price_df(120)
    out = app_module._build_risk_lab_block(px["close"].pct_change().dropna())
    assert out["available"] is False and "Insufficient" in out["state"]


def test_risk_lab_page_renders_with_position_scaling():
    px = _price_df()
    html = _render(px, position=100_000)
    assert "Credible Risk" in html
    assert "low confidence" not in html or "tail n=" in html
    assert "QLIKE" in html and "Kupiec" in html
    assert "$" in html and "10d 95% VaR on $100000" in html
    assert "overlapping" in html  # sqrt-time caveat present


def test_risk_lab_page_renders_insufficient_state():
    px = _price_df(150)
    html = _render(px)
    assert "Insufficient history" in html


def _render(px, position=None):
    with patch.object(app_module, "get_or_fetch_prices", return_value=px), \
         patch.object(app_module, "get_options_smile", return_value=None), \
         patch.object(app_module, "get_gex_profile", return_value=None), \
         patch.object(app_module.ai, "generate_report", return_value=None), \
         patch.object(app_module.providers, "get_short_interest", return_value=None):
        qs = f"/analytics?ticker=TEST" + (f"&position={position}" if position else "")
        return app_module.app.test_client().get(qs).get_data(as_text=True)
