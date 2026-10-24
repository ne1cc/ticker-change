"""Issue 1.6: the LLM payload marks assumed/unavailable values and can never
contain fields removed from the product."""
import json
from unittest.mock import patch

import ai


def _capturing_provider(captured):
    def fake_provider(base_url_or_key, *args, **kwargs):
        captured.append(kwargs.get("user_msg") or args[2] if len(args) > 2 else args[-1])
        return "### Snapshot\nok"
    return fake_provider


def _data_with_assumed_and_unavailable():
    return {
        "ticker": "TEST",
        "current_price": 100.0,
        "stats": {"annualised_vol": 24.0, "mc_prob_gain": 0.85},
        "forward_estimates": {"prob_gain_1y": 0.85},
        "institutional": {
            "short_interest": {"si_source": "assumed"},
            "macro_conditioning": {
                "regime": None, "fed_funds_rate": None, "yield_curve_2s10s": None,
                "is_inverted": None, "cpi_yoy": None,
                "betas": {"bull_beta": 1.1, "bear_beta": 0.9},
                "upside_capture": None, "downside_capture": None,
            },
        },
    }


def test_standard_report_payload_marks_provenance(monkeypatch):
    monkeypatch.setattr(ai.providers, "ai_providers",
                        lambda: [{"id": "openai-compatible", "base_url": "x", "key": "k", "model": "m", "label": "X"}])
    captured = []
    with patch.object(ai.db, "cache_get", return_value=None), \
         patch.object(ai, "_openai_compatible_report",
                      side_effect=lambda base_url, key, model, user_msg, system: captured.append((user_msg, system)) or "ok"):
        ai.generate_report("TEST", _data_with_assumed_and_unavailable())

    user_msg, system = captured[0]
    payload = json.loads(user_msg.split("snapshot:", 1)[1])
    assert payload["institutional"]["_provenance"]["short_interest"].startswith("assumed defaults")
    prov = payload["institutional"]["_provenance"]["macro_conditioning"]
    assert "fed_funds_rate" in prov and "cpi_yoy" in prov
    assert "unavailable" in prov
    # Removed fields can never reappear in the payload
    assert "forward_estimates" not in payload and "monte_carlo" not in payload
    assert "mc_prob_gain" not in payload["statistics"]
    # The model is told the rule
    assert "_provenance" in system


def test_comprehensive_report_payload_is_sanitized(monkeypatch):
    monkeypatch.setattr(ai.providers, "ai_providers",
                        lambda: [{"id": "anthropic", "key": "k", "model": "m", "label": "X"}])
    captured = []
    with patch.object(ai.db, "cache_get", return_value=None), \
         patch.object(ai, "_anthropic_report",
                      side_effect=lambda key, model, user_msg, system: captured.append((user_msg, system)) or "ok"):
        ai.generate_comprehensive_report("TEST", _data_with_assumed_and_unavailable())

    user_msg, system = captured[0]
    payload = json.loads(user_msg.split("dataset:", 1)[1])
    assert "forward_estimates" not in payload
    assert payload["institutional"]["_provenance"]["short_interest"].startswith("assumed defaults")
    assert "never state numbers" in system


def test_report_with_none_macro_carries_no_macro_numbers():
    """The payload a model would narrate from contains no macro numbers at all."""
    data = _data_with_assumed_and_unavailable()
    clean = ai._sanitize_payload(data)
    macro = clean["institutional"]["macro_conditioning"]
    numbers = [v for v in macro.values() if isinstance(v, (int, float))]
    assert numbers == []  # only nulls, the betas dict, and provenance live elsewhere
    assert macro["fed_funds_rate"] is None
    assert "5.33" not in json.dumps(clean) and "3.1" not in json.dumps(clean)
