"""Regressions for the /analytics honesty pass: no fabricated or mislabelled metrics."""
import dataclasses
from unittest.mock import patch

import numpy as np
import pandas as pd

import app as app_module
import glossary
import microstructure


def _prices(n=400, seed=1):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, n)))
    idx = pd.bdate_range(end="2026-10-20", periods=n)
    return pd.DataFrame({"open": close * 0.999, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "volume": rng.integers(1_000_000, 2_000_000, n)}, index=idx)


# --- Removed: VPIN (daily bars cannot produce it) -----------------------------
def test_vpin_is_gone():
    fields = {f.name for f in dataclasses.fields(microstructure.MicrostructureAnalytics)}
    assert "vpin_toxicity" not in fields and "toxicity_regime" not in fields
    assert not hasattr(microstructure, "calculate_vpin")
    assert "vpin" not in glossary.GLOSSARY


# --- Removed: GBM Monte Carlo forecast ----------------------------------------
def test_monte_carlo_forecast_is_gone():
    with patch.object(app_module, "get_or_fetch_prices", return_value=_prices()):
        data = app_module.compute_analytics("TEST")
    assert "forward_estimates" not in data
    assert "monte_carlo" not in data["charts"]
    assert not [k for k in data["stats"] if k.startswith("mc_")]
    assert "monte_carlo" not in glossary.GLOSSARY
