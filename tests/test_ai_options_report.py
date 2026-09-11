import pytest
import re
from ai import _preprocess_options_markdown, _to_html

SAMPLE_RAW_REPORT = """Options Market Regime & Volatility Analysis:
Volatility Risk Premium (VRP): The current ATM IV of 47.46% sits significantly below both the HV30 (76.12%) and HV90 (102.53%). This indicates a deeply negative VRP, where the market is currently underpricing the realized volatility of MU. Selling premium in this regime is statistically disadvantageous, as the realized variance is likely to exceed the implied variance, leading to consistent losses for net sellers.
IV Rank/Percentile: With an IV Rank of 1.9 and an IV Percentile of 4.8, MU is trading at the extreme low end of its historical volatility range. Option premiums are objectively "cheap" relative to the stock's recent history. This environment suggests that the market is currently complacent, potentially underestimating the magnitude of future price swings.
Expected Move & Boundary Levels:
Expected Move Analysis: The market-implied move via the straddle ($27.55) is significantly higher than the Black-Scholes theoretical move ($20.76). This discrepancy suggests that market participants are pricing in a "fat-tail" event or a potential gap risk, despite the low headline IV.
Structural Boundaries:
Gamma Flip Point: 982.77. With the spot at 983.03, MU is hovering exactly at the flip point. This is a critical juncture where dealer hedging behavior shifts from "buy-the-dip/sell-the-rip" (positive gamma) to "sell-the-dip/buy-the-rip" (negative gamma).
Call Wall (Resistance): 1000.0. This represents the primary dealer resistance level. Expect significant selling pressure from market makers if the spot approaches this level.
Put Wall (Support): 950.0. This is the primary support level. A breach below this level could trigger accelerated selling as dealers are forced to hedge their short put positions by selling underlying shares.
Max Pain: 995.0. The proximity of the spot to Max Pain suggests that the market may experience "pinning" pressure toward this level as we approach the 1 DTE expiration.
Quantitative Strategy Recommendations:
Given the compressed IV environment and the proximity to the Gamma Flip, strategies should focus on long-gamma or long-vega exposure to capitalize on a potential volatility expansion.

Bull Call Debit Spread (Vertical):

Structure: Buy 990 Call / Sell 1010 Call.
Rationale: Capitalizes on the expectation of a move toward the 1000 Call Wall while mitigating the cost of low IV through the short leg.
Target: Exit at 50% of max profit.
PoP: Target ~40-45% delta on the long leg.
Long Calendar Spread (Time Spread):

Structure: Sell 1 DTE 980 Put / Buy 30-45 DTE 980 Put (or Call equivalent).
Rationale: Since IV is at historical lows, buying longer-dated volatility while selling the immediate expiration allows the trader to capture potential IV expansion in the back month while benefiting from the rapid theta decay of the 1 DTE option.
Trade Management: Close the position if the spot moves outside the 950–1000 range.
Tail Risk & Risk Management:
Worst-Case Scenario: A rapid move below the 950 Put Wall. In a negative gamma regime (spot < 982.77), dealer hedging will exacerbate downward momentum, leading to a liquidity vacuum.
Defensive Adjustments:
Stop-Loss: Implement a hard stop if the spot price closes below 945.0, as this invalidates the bullish thesis and signals a breakdown of the Put Wall.
Dynamic Delta Hedging: If holding the calendar spread, hedge the delta exposure if the spot price deviates more than 2% from the strike price to neutralize directional risk.
Gamma Risk: Given the 1 DTE expiration, ensure no naked short positions are held into the final hour of trading to avoid assignment risk and uncontrolled gamma exposure."""


def test_preprocess_normalizes_headings():
    cleaned = _preprocess_options_markdown(SAMPLE_RAW_REPORT)
    assert "### 1. Options Market Regime & Volatility Analysis" in cleaned
    assert "### 2. Expected Move & Boundary Levels" in cleaned or "### 2. Expected Move & Key Boundary Levels" in cleaned
    assert "### 3. Quantitative Strategy Recommendations" in cleaned
    assert "### 4. Tail Risk & Risk Management" in cleaned
    assert "#### Structural Boundaries" in cleaned
    assert "#### Bull Call Debit Spread (Vertical)" in cleaned
    assert "#### Long Calendar Spread (Time Spread)" in cleaned
    assert "#### Defensive Adjustments" in cleaned


def test_preprocess_bulletizes_and_spaces_items():
    cleaned = _preprocess_options_markdown(SAMPLE_RAW_REPORT)
    assert "- **Volatility Risk Premium (VRP):**" in cleaned
    assert "- **IV Rank/Percentile:**" in cleaned
    assert "- **Gamma Flip Point:**" in cleaned
    assert "- **Call Wall (Resistance):**" in cleaned
    assert "- **Put Wall (Support):**" in cleaned
    assert "- **Max Pain:**" in cleaned
    assert "- **Worst-Case Scenario:**" in cleaned
    assert "- **Structure:**" in cleaned
    assert "- **Rationale:**" in cleaned
    assert "- **Stop-Loss:**" in cleaned


def test_to_html_renders_headings_and_lists():
    cleaned = _preprocess_options_markdown(SAMPLE_RAW_REPORT)
    html = _to_html(cleaned)
    assert "<h3>" in html
    assert "<h4>" in html
    assert "<ul>" in html
    assert "<li>" in html
    assert "<strong>Volatility Risk Premium (VRP):</strong>" in html
    assert "<strong>Structure:</strong>" in html
    # Ensure paragraphs are not all fused into a single huge block
    p_tags = re.findall(r"<p>.*?</p>", html, re.DOTALL)
    assert len(p_tags) >= 5


def test_preprocess_preserves_already_formatted_markdown():
    proper_md = """### 1. Options Market Regime & Volatility Analysis

| Metric | Value | Comparison |
| --- | --- | --- |
| ATM IV | 47.46% | Low |

- **VRP:** Deeply negative.
- **IV Rank:** 1.9.

### 2. Expected Move & Boundary Levels

- **Expected Move:** ±$27.55.
"""
    cleaned = _preprocess_options_markdown(proper_md)
    assert "### 1. Options Market Regime & Volatility Analysis" in cleaned
    assert "| Metric | Value | Comparison |" in cleaned
    html = _to_html(cleaned)
    assert "<table>" in html
    assert "<th>Metric</th>" in html


def test_generate_options_report_with_cached_markdown(monkeypatch):
    import db
    import ai

    monkeypatch.setattr(ai.providers, "ai_providers", lambda: [{"id": "mock", "label": "Mock", "model": "mock-model", "base_url": "mock", "key": "mock"}])
    monkeypatch.setattr(db, "cache_get", lambda table, key, ttl: {"markdown": SAMPLE_RAW_REPORT, "provider": "mock"})

    html, err = ai.generate_options_report("MU", {"selected_expiration": "2026-09-11"})
    assert err is None
    assert html is not None
    assert "<h3>1. Options Market Regime &amp; Volatility Analysis</h3>" in html
    assert "<strong>Volatility Risk Premium (VRP):</strong>" in html


def test_api_options_ai_report_force_param(monkeypatch):
    import app as flask_app
    import ai

    called = {}

    def mock_generate(ticker, data, force=False):
        called["force"] = force
        return "<h3>Report</h3>", None

    monkeypatch.setattr(flask_app, "compute_options_analysis", lambda ticker, exp, rf: {"spot": 100})
    monkeypatch.setattr(ai, "generate_options_report", mock_generate)

    client = flask_app.app.test_client()

    resp = client.get("/api/options-ai-report/MU?force=1")
    assert resp.status_code == 200
    assert called.get("force") is True
    data = resp.get_json()
    assert data["html"] == "<h3>Report</h3>"

    resp2 = client.get("/api/options-ai-report/MU")
    assert resp2.status_code == 200
    assert called.get("force") is False

