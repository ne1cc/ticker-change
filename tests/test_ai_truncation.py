"""A response cut off at the output cap is a failure for every report variant."""
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import ai


def _fake_anthropic(monkeypatch, stop_reason, sent):
    msg = SimpleNamespace(stop_reason=stop_reason,
                          content=[SimpleNamespace(type="text", text="### Report")])

    class _Stream:
        def __enter__(self):
            return SimpleNamespace(get_final_message=lambda: msg)

        def __exit__(self, *exc):
            return False

    def stream(**kwargs):
        sent.update(kwargs)
        return _Stream()

    client = SimpleNamespace(messages=SimpleNamespace(stream=stream))
    monkeypatch.setitem(sys.modules, "anthropic",
                        SimpleNamespace(Anthropic=lambda **kw: client))


def test_anthropic_max_tokens_stop_is_a_failure(monkeypatch):
    sent = {}
    _fake_anthropic(monkeypatch, "max_tokens", sent)
    with pytest.raises(Exception, match="truncated"):
        ai._anthropic_report("k", "claude-x", "msg")
    assert sent["max_tokens"] == 16000


def test_anthropic_end_turn_is_returned(monkeypatch):
    _fake_anthropic(monkeypatch, "end_turn", {})
    assert ai._anthropic_report("k", "claude-x", "msg") == "### Report"


def _post_returning(finish_reason, sent):
    def post(url, **kwargs):
        sent.update(kwargs["json"])
        resp = MagicMock(status_code=200)
        choice = {"message": {"content": "text"}}
        if finish_reason is not None:
            choice["finish_reason"] = finish_reason
        resp.json.return_value = {"choices": [choice]}
        return resp
    return post


def test_openai_length_finish_is_a_failure(monkeypatch):
    sent = {}
    monkeypatch.setattr(ai.requests, "post", _post_returning("length", sent))
    with pytest.raises(Exception, match="truncated"):
        ai._openai_compatible_report("https://x", "k", "m", "msg")
    assert sent["max_tokens"] == 16000


@pytest.mark.parametrize("finish_reason", ["stop", None])
def test_openai_missing_or_normal_finish_is_not_truncation(monkeypatch, finish_reason):
    monkeypatch.setattr(ai.requests, "post", _post_returning(finish_reason, {}))
    assert ai._openai_compatible_report("https://x", "k", "m", "msg") == "text"


def test_truncated_primary_falls_through_to_next_model(monkeypatch):
    calls = []

    def report(base_url, key, model, user_msg, system, **kw):
        calls.append(model)
        if model == "big":
            raise Exception("response truncated at the output token cap (finish_reason=length)")
        return "ok"

    monkeypatch.setattr(ai, "_openai_compatible_report", report)
    prov = {"id": "openrouter", "base_url": "https://x", "key": "k",
            "model": "big", "fallback_models": ("small",)}
    text, errors = ai._generate_report_from_provider(prov, "msg", "sys")
    assert text == "ok"
    assert calls == ["big", "small"]
    assert "truncated" in errors[0]
