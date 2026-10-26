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
    """Cached Markdown renders correctly and retains every stored report field."""
    import db
    import ai

    seen = {}

    def fake_cache_get(table, key, ttl):
        seen.update(table=table, key=key)
        return {"markdown": SAMPLE_RAW_REPORT, "provider": "mock",
                "generated_at": "2026-10-24T15:00:00",
                "vintage": "2026-10-23T16:14:00+00:00",
                "payload": {"spot": 100.0}}

    monkeypatch.setattr(ai.providers, "ai_providers", lambda: [{"id": "mock", "label": "Mock", "model": "mock-model", "base_url": "mock", "key": "mock"}])
    monkeypatch.setattr(db, "cache_get", fake_cache_get)

    report, err = ai.generate_options_report("MU", {"selected_expiration": "2026-09-11"})
    assert err is None
    assert seen["table"] == "ai_options_report"
    assert seen["key"] == f"options_report:v{ai.OPTIONS_REPORT_VERSION}:MU:2026-09-11:0.045"
    html = report["html"]
    assert "<h3>1. Options Market Regime &amp; Volatility Analysis</h3>" in html
    assert "<strong>Volatility Risk Premium (VRP):</strong>" in html
    assert report["markdown"] == SAMPLE_RAW_REPORT
    assert report["provider"] == "mock"
    assert report["generated_at"] == "2026-10-24T15:00:00"
    assert report["payload"] == {"spot": 100.0}
    assert report["vintage"] == "2026-10-23T16:14:00+00:00"


def test_generate_options_report_cache_round_trip_preserves_full_report(monkeypatch):
    """Require identical six-field reports on fresh/cache paths and four API fields offline."""
    import ai
    from unittest.mock import Mock

    # 1. Isolate the cache and provider so the round trip cannot touch the network.
    cache = {}
    generate = Mock(return_value=(FALLBACK_REPORT, []))
    monkeypatch.setattr(ai.providers, "ai_providers", lambda: [{"id": "mock"}])
    monkeypatch.setattr(ai, "_generate_report_from_provider", generate)
    monkeypatch.setattr(ai.db, "cache_get", lambda table, key, ttl: cache.get((table, key)))
    monkeypatch.setattr(ai.db, "cache_set",
                        lambda table, key, report: cache.update({(table, key): dict(report)}))
    payload = {"selected_expiration": "2026-11-20", "rf_rate": 0.05,
               "spot": 100.0, "vintage": "unknown"}

    # 2. A cache hit must rebuild stale HTML without discarding the stored report fields.
    fresh_report, error = ai.generate_options_report("SPY", payload)
    assert error is None
    assert set(fresh_report) == {"html", "markdown", "provider", "generated_at", "vintage", "payload"}
    cache[("ai_options_report", "options_report:v2:SPY:2026-11-20:0.05")]["html"] = "<p>stale rendering</p>"
    cached_report, error = ai.generate_options_report("SPY", payload)
    assert error is None
    assert cached_report == fresh_report
    generate.assert_called_once()

    # 3. Only the route's dedicated cache reader projects the four public API fields.
    api_report = ai.cached_options_report("SPY", "2026-11-20", 0.05)
    api_fields = {"html", "generated_at", "vintage", "payload"}
    assert api_report == {key: fresh_report[key] for key in api_fields}


def _stub_report_flow(monkeypatch, expirations=("2026-11-20", "2026-12-18")):
    """Route-level stubs: expirations resolved offline, builder and generator recorded."""
    import app as flask_app
    import ai

    calls = {"builder": 0, "generate": []}

    def fake_payload(ticker, exp, rf):
        calls["builder"] += 1
        return {"ticker": ticker, "selected_expiration": exp, "rf_rate": rf,
                "spot": 100.0, "chain_digest": [], "_provenance": {},
                "vintage": "2026-10-23T16:14:00+00:00"}

    def fake_generate(ticker, payload, force=False):
        calls["generate"].append({"ticker": ticker, "payload": payload, "force": force})
        return {"html": "<h3>Report</h3>", "markdown": "Report", "provider": "mock",
                "generated_at": "2026-10-25T13:05:00",
                "vintage": payload.get("vintage"),
                "payload": payload}, None

    monkeypatch.setattr(flask_app, "get_cached_expirations",
                        lambda t, stock=None, force=False: list(expirations))
    monkeypatch.setattr(flask_app, "build_options_report_payload", fake_payload)
    monkeypatch.setattr(ai, "cached_options_report", lambda t, e, r: None)
    monkeypatch.setattr(ai, "generate_options_report", fake_generate)
    return flask_app, ai, calls


def test_api_options_ai_report_force_param(monkeypatch):
    flask_app, ai, calls = _stub_report_flow(monkeypatch)
    client = flask_app.app.test_client()

    resp = client.get("/api/options-ai-report/MU?force=1")
    assert resp.status_code == 200
    assert calls["generate"][0]["force"] is True
    assert calls["generate"][0]["payload"]["selected_expiration"] == "2026-11-20"
    data = resp.get_json()
    assert data["html"] == "<h3>Report</h3>"

    resp2 = client.get("/api/options-ai-report/MU")
    assert resp2.status_code == 200
    assert calls["generate"][1]["force"] is False


def test_api_options_ai_report_cache_hit_skips_builder(monkeypatch):
    flask_app, ai, calls = _stub_report_flow(monkeypatch)

    def boom(*args, **kwargs):
        raise AssertionError("build_options_report_payload must not run on a cache hit")

    monkeypatch.setattr(flask_app, "build_options_report_payload", boom)
    monkeypatch.setattr(ai, "cached_options_report",
                        lambda t, e, r: {"html": "<p>cached</p>", "generated_at": "2026-10-25T13:05:00",
                                         "vintage": "unknown", "payload": {"spot": 100.0}})

    client = flask_app.app.test_client()
    resp = client.get("/api/options-ai-report/SPY")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["html"] == "<p>cached</p>"
    assert data["payload"] == {"spot": 100.0}
    assert calls["generate"] == []


def test_api_options_ai_report_resolves_default_expiration(monkeypatch):
    flask_app, ai, calls = _stub_report_flow(monkeypatch)
    lookup = {}
    monkeypatch.setattr(ai, "cached_options_report",
                        lambda t, e, r: lookup.update(ticker=t, exp=e, rf=r))

    client = flask_app.app.test_client()
    resp = client.get("/api/options-ai-report/SPY?rf_rate=0.05")
    assert resp.status_code == 200

    # Cache lookup and payload both use get_cached_expirations(...)[0] ...
    assert lookup["exp"] == "2026-11-20"
    assert lookup["rf"] == 0.05
    assert calls["generate"][0]["payload"]["selected_expiration"] == "2026-11-20"
    # ... and the versioned report cache key carries v2 and the resolved expiry.
    key = ai._options_cache_key("SPY", calls["generate"][0]["payload"]["selected_expiration"],
                                calls["generate"][0]["payload"]["rf_rate"])
    assert key == f"options_report:v{ai.OPTIONS_REPORT_VERSION}:SPY:2026-11-20:0.05"


def test_api_options_ai_report_force_ignores_cache_and_builds_once(monkeypatch):
    flask_app, ai, calls = _stub_report_flow(monkeypatch)
    monkeypatch.setattr(ai, "cached_options_report",
                        lambda t, e, r: {"html": "<p>stale</p>", "generated_at": "x",
                                         "vintage": "unknown", "payload": {}})

    client = flask_app.app.test_client()
    resp = client.get("/api/options-ai-report/MU?force=1")
    assert resp.status_code == 200
    assert calls["builder"] == 1
    assert calls["generate"][0]["force"] is True
    assert resp.get_json()["html"] == "<h3>Report</h3>"


def test_api_options_ai_report_response_carries_metadata(monkeypatch):
    flask_app, ai, calls = _stub_report_flow(monkeypatch)

    client = flask_app.app.test_client()
    resp = client.get("/api/options-ai-report/MU")
    assert resp.status_code == 200
    data = resp.get_json()
    assert set(data) == {"html", "generated_at", "vintage", "payload"}
    from datetime import datetime
    datetime.fromisoformat(data["generated_at"])
    assert data["vintage"] == "2026-10-23T16:14:00+00:00"


def test_options_report_user_message_is_compact_and_grounded(monkeypatch):
    import ai

    captured = {}

    def mock_openai_report(base_url, key, model, user_msg, system, **kwargs):
        captured["user_msg"] = user_msg
        captured["system"] = system
        return FALLBACK_REPORT

    monkeypatch.setattr(ai.db, "cache_get", lambda *args, **kwargs: None)
    monkeypatch.setattr(ai.db, "cache_set", lambda *args, **kwargs: None)
    monkeypatch.setattr(ai, "_openai_compatible_report", mock_openai_report)
    monkeypatch.setattr(ai.providers, "ai_providers", lambda: _options_providers("m", "small"))

    payload = {"ticker": "MU", "spot": 100.0, "chain_digest": [{"strike": 100}],
               "_provenance": {"skew": "25-delta skew not computable from this chain"},
               "vintage": "2026-10-23T16:14:00+00:00"}
    report, err = ai.generate_options_report("MU", payload, force=True)
    assert err is None

    user_msg = captured["user_msg"]
    assert "chain_digest" in user_msg
    assert "_provenance" in user_msg
    assert "\n  " not in user_msg
    assert "_provenance" in captured["system"]
    assert report["payload"] is payload
    assert report["vintage"] == "2026-10-23T16:14:00+00:00"
    assert report["generated_at"]


def test_options_system_has_no_generic_anchors():
    import ai

    assert "40-45 delta" not in ai.OPTIONS_SYSTEM
    assert "50% max profit" not in ai.OPTIONS_SYSTEM


def test_options_system_greeks_caveats_match_model_limits():
    """Keep dividend/early-exercise limits without claiming rates are ignored."""
    import ai

    assert "Black-Scholes with no dividends" in ai.OPTIONS_SYSTEM
    assert "early exercise is ignored" in ai.OPTIONS_SYSTEM
    assert "rates effects are ignored" not in ai.OPTIONS_SYSTEM


def test_generate_options_report_falls_back_to_secondary_model_on_503(monkeypatch):
    import ai

    models_called = []

    def mock_openai_report(base_url, key, model, user_msg, system, **kwargs):
        models_called.append(model)
        if model == "gemini-3.8-flash":
            raise Exception("HTTP 503: This model is currently experiencing high demand.")
        return "### 1. Options Market Regime & Volatility Analysis\n\n- **IV Rank:** 10.0"

    monkeypatch.setattr(ai.db, "cache_get", lambda *args, **kwargs: None)
    monkeypatch.setattr(ai.db, "cache_set", lambda *args, **kwargs: None)
    monkeypatch.setattr(ai, "_openai_compatible_report", mock_openai_report)

    provider_config = [{
        "id": "gemini",
        "label": "Google Gemini",
        "base_url": "https://example.com",
        "key": "test_key",
        "model": "gemini-3.8-flash",
        "fallback_models": ("gemini-3.1-flash-lite",),
    }]
    monkeypatch.setattr(ai.providers, "ai_providers", lambda: provider_config)

    html, err = ai.generate_options_report("AAPL", {"selected_expiration": "2026-11-13"}, force=True)
    assert err is None
    assert html is not None
    assert models_called == ["gemini-3.8-flash", "gemini-3.1-flash-lite"]


def test_openai_compatible_retries_on_transient_error(monkeypatch):
    import ai
    from unittest.mock import MagicMock

    call_count = 0

    def mock_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        resp = MagicMock()
        if call_count == 1:
            resp.status_code = 503
            resp.text = '{"error": "high demand"}'
        else:
            resp.status_code = 200
            resp.json.return_value = {
                "choices": [{"message": {"content": "Generated report text"}}]
            }
        return resp

    monkeypatch.setattr(ai.requests, "post", mock_post)
    monkeypatch.setattr(ai.time, "sleep", lambda s: None)

    res = ai._openai_compatible_report("https://fake", "key", "model", "msg", max_retries=2)
    assert res == "Generated report text"
    assert call_count == 2


# --- Truncation at the output cap is a failure, never a cached report (#194) ---

FALLBACK_REPORT = "### 1. Options Market Regime & Volatility Analysis\n\n- **IV Rank:** 10.0"


def _fake_anthropic_client(monkeypatch, stop_reason, sent):
    import sys
    from types import SimpleNamespace

    msg = SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="text", text=FALLBACK_REPORT)],
    )

    class _Stream:
        def __enter__(self):
            return SimpleNamespace(get_final_message=lambda: msg)

        def __exit__(self, *exc):
            return False

    def stream(**kwargs):
        sent.update(kwargs)
        return _Stream()

    client = SimpleNamespace(messages=SimpleNamespace(stream=stream))
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=lambda **kw: client))


def test_anthropic_max_tokens_stop_is_a_failure(monkeypatch):
    import ai

    sent = {}
    _fake_anthropic_client(monkeypatch, "max_tokens", sent)
    with pytest.raises(Exception, match="truncated"):
        ai._anthropic_report("k", "claude-x", "msg")
    assert sent["max_tokens"] == 16000


def test_anthropic_end_turn_is_returned(monkeypatch):
    import ai

    _fake_anthropic_client(monkeypatch, "end_turn", {})
    assert ai._anthropic_report("k", "claude-x", "msg") == FALLBACK_REPORT


def _post_returning(finish_reason, sent):
    from unittest.mock import MagicMock

    def post(url, **kwargs):
        sent.update(kwargs["json"])
        resp = MagicMock(status_code=200)
        choice = {"message": {"content": FALLBACK_REPORT}}
        if finish_reason is not None:
            choice["finish_reason"] = finish_reason
        resp.json.return_value = {"choices": [choice]}
        return resp

    return post


def test_openai_length_finish_is_a_failure(monkeypatch):
    import ai

    sent = {}
    monkeypatch.setattr(ai.requests, "post", _post_returning("length", sent))
    with pytest.raises(Exception, match="truncated"):
        ai._openai_compatible_report("https://x", "k", "m", "msg")
    assert sent["max_tokens"] == 16000


@pytest.mark.parametrize("finish_reason", ["stop", None])
def test_openai_missing_or_normal_finish_is_not_truncation(monkeypatch, finish_reason):
    import ai

    monkeypatch.setattr(ai.requests, "post", _post_returning(finish_reason, {}))
    assert ai._openai_compatible_report("https://x", "k", "m", "msg") == FALLBACK_REPORT


def _options_providers(primary_model, fallback_model):
    return [{
        "id": "openrouter",
        "label": "OpenRouter",
        "base_url": "https://example.com",
        "key": "test_key",
        "model": primary_model,
        "fallback_models": (fallback_model,),
    }]


def test_generate_options_report_truncated_primary_caches_fallback_report(monkeypatch):
    import ai

    models_called = []

    def mock_openai_report(base_url, key, model, user_msg, system, **kwargs):
        models_called.append(model)
        if model == "big":
            raise Exception("response truncated at the output token cap (finish_reason=length)")
        return FALLBACK_REPORT

    cache_set_calls = []
    monkeypatch.setattr(ai.db, "cache_get", lambda *args, **kwargs: None)
    monkeypatch.setattr(ai.db, "cache_set", lambda *args, **kwargs: cache_set_calls.append(args))
    monkeypatch.setattr(ai, "_openai_compatible_report", mock_openai_report)
    monkeypatch.setattr(ai.providers, "ai_providers", lambda: _options_providers("big", "small"))

    html, err = ai.generate_options_report("AAPL", {"selected_expiration": "2026-11-13"}, force=True)
    assert err is None
    assert html is not None
    assert models_called == ["big", "small"]
    assert len(cache_set_calls) == 1
    table, cache_key, payload = cache_set_calls[0]
    assert table == "ai_options_report"
    assert payload["markdown"] == FALLBACK_REPORT


def test_generate_options_report_all_truncated_returns_error_without_caching(monkeypatch):
    import ai

    def mock_openai_report(base_url, key, model, user_msg, system, **kwargs):
        raise Exception("response truncated at the output token cap (finish_reason=length)")

    cache_set_calls = []
    monkeypatch.setattr(ai.db, "cache_get", lambda *args, **kwargs: None)
    monkeypatch.setattr(ai.db, "cache_set", lambda *args, **kwargs: cache_set_calls.append(1))
    monkeypatch.setattr(ai, "_openai_compatible_report", mock_openai_report)
    monkeypatch.setattr(ai.providers, "ai_providers", lambda: _options_providers("big", "small"))

    html, err = ai.generate_options_report("AAPL", {"selected_expiration": "2026-11-13"}, force=True)
    assert html is None
    assert "truncated" in err
    assert cache_set_calls == []
