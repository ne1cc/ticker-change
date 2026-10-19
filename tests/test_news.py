"""Exercise the news service with real SQLite and fake upstream HTTP only."""
import importlib
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

import db


@pytest.fixture
def service(tmp_path, monkeypatch):
    assert importlib.util.find_spec("news") is not None, "news service has not been implemented"
    news = importlib.import_module("news")
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "feed.db"))
    db.init_db()
    monkeypatch.setenv("MARKETAUX_API_KEY", "private-news-token")
    now = datetime.now(timezone.utc)
    monkeypatch.setattr(news, "_now", lambda: now)
    return news


def entity(symbol="AAPL", exchange=None, country="us", name="Apple Inc."):
    return dict(symbol=symbol, exchange=exchange, country=country, name=name, type="equity")


def article(number=0, **overrides):
    item = dict(uuid=f"id-{number}", title=f"Headline {number}",
                url=f"https://publisher.example/story/{number}", source="publisher.example",
                published_at=(datetime.now(timezone.utc) - timedelta(minutes=number + 1)).isoformat(),
                language="en", entities=[entity()], similar=[])
    item.update(overrides)
    return item


def upstream(monkeypatch, service, pages=None, entities=None, page_limit=3):
    calls = []
    pages = pages if pages is not None else [[article()]]

    def get(url, *, params, timeout, allow_redirects):
        calls.append((url, dict(params)))
        if url.endswith("/entity/search"):
            data = [entity()] if entities is None else entities
            if isinstance(data, list):
                data = [item for item in data if item["type"] in params["types"].split(",")]
        else:
            page = params["page"] - 1
            data = pages[page] if page < len(pages) else []
            if isinstance(data, list):
                data = data[:params.get("limit", page_limit)]
        if isinstance(data, Exception):
            raise data
        if isinstance(data, int):
            return SimpleNamespace(status_code=data, headers={"Retry-After": "600"}, json=lambda: {})
        return SimpleNamespace(status_code=200, headers={},
                               json=lambda: {"data": data, "meta": {"limit": page_limit}})

    monkeypatch.setattr(service._SESSION, "get", get)
    return calls


def expire_cache(hours=2):
    with db.get_conn() as conn:
        conn.execute("UPDATE api_cache SET fetched_at=? WHERE provider='marketaux_news'",
                     ((datetime.now(timezone.utc) - timedelta(hours=hours)).replace(tzinfo=None).isoformat(),))
        conn.execute("DELETE FROM api_cache WHERE provider='internal'")


def test_five_pages_supply_fifteen_sorted_articles_and_cache(service, monkeypatch):
    pages = [[article(i) for i in range(start, start + 3)] for start in range(0, 15, 3)]
    calls = upstream(monkeypatch, service, pages)
    feed = service.get_feed("aapl")
    assert feed["status"] == "ok"
    assert len(feed["articles"]) == 15
    assert feed["articles"][0]["title"] == "Headline 0"
    assert feed["articles"][-1]["title"] == "Headline 14"
    assert feed["ticker"] == "AAPL"
    assert len(calls) == 6  # one entity lookup, five news pages
    assert calls[0][1]["countries"] == "us"
    params = calls[1][1]
    assert params["language"] == "en"
    assert "limit" not in params
    assert params["symbols"] == "AAPL"
    assert "private-news-token" not in str(feed)
    assert service.get_feed("AAPL") == feed
    assert len(calls) == 6


@pytest.mark.parametrize("page_limit", [20, 50, 100])
def test_uses_full_provider_plan_pages_and_caches_all_articles(service, monkeypatch, page_limit):
    pages = [[article(i) for i in range(start, start + page_limit)]
             for start in range(0, page_limit * 5, page_limit)]
    calls = upstream(monkeypatch, service, pages=pages, page_limit=page_limit)

    feed = service.get_feed("AAPL")

    assert feed["status"] == "ok"
    assert len(feed["articles"]) == page_limit * 5
    assert feed["articles"][-1]["id"] == f"id-{page_limit * 5 - 1}"
    assert len(calls) == 6
    assert service.get_feed("AAPL") == feed
    assert len(calls) == 6


def test_stops_at_short_provider_plan_page(service, monkeypatch):
    calls = upstream(monkeypatch, service, pages=[[article(i) for i in range(3)]], page_limit=20)

    feed = service.get_feed("AAPL")

    assert len(feed["articles"]) == 3
    assert len(calls) == 2  # entity lookup and a short news page


def test_plan_max_feed_bypasses_legacy_fifteen_article_snapshot(service, monkeypatch):
    listing = service._listing(entity())
    legacy = service._result("AAPL", "ok", listing)
    legacy.update(articles=service._articles([article(i) for i in range(15)], listing),
                  fetched_at=service._iso(service._now()), limit=15)
    db.cache_set("marketaux_news", "feed:" + listing["id"] + ":en:7d", legacy)
    upstream(monkeypatch, service, pages=[[article(i) for i in range(20)]], page_limit=20)

    feed = service.get_feed("AAPL")

    assert len(feed["articles"]) == 20


def test_bare_us_ticker_resolves_exact_marketaux_listing(service, monkeypatch):
    listing = entity("MU", name="Micron Technology, Inc.")
    calls = upstream(monkeypatch, service, pages=[[]], entities=[listing])

    feed = service.get_feed("MU")

    assert feed["status"] == "empty"
    assert feed["listing"]["symbol"] == "MU"
    assert feed["listing"]["name"] == "Micron Technology, Inc."
    assert calls[0][1]["symbols"] == "MU"
    assert calls[0][1]["countries"] == "us"


def test_us_etf_returns_matching_headlines(service, monkeypatch):
    listing = {**entity("SPY", name="SPDR S&P 500 ETF Trust"), "type": "etf"}
    calls = upstream(monkeypatch, service, entities=[listing],
                     pages=[[article(entities=[listing])]])

    feed = service.get_feed("SPY")

    assert feed["status"] == "ok"
    assert feed["listing"]["symbol"] == "SPY"
    assert [item["id"] for item in feed["articles"]] == ["id-0"]
    assert calls[0][1]["countries"] == "us"


def test_legacy_negative_lookup_is_bypassed(service, monkeypatch):
    db.cache_set("marketaux_news", "entity:AAPL", {"candidates": []})
    upstream(monkeypatch, service)

    feed = service.get_feed("AAPL")

    assert feed["status"] == "ok"
    assert [item["id"] for item in feed["articles"]] == ["id-0"]


def test_negative_lookup_retries_after_one_hour(service, monkeypatch):
    entities = []
    calls = upstream(monkeypatch, service, entities=entities)
    assert service.get_feed("AAPL")["status"] == "unsupported"

    entities.append(entity())
    expire_cache(hours=59 / 60)
    assert service.get_feed("AAPL")["status"] == "unsupported"
    assert len(calls) == 1

    expire_cache(hours=61 / 60)
    feed = service.get_feed("AAPL")

    assert feed["status"] == "ok"
    assert [item["id"] for item in feed["articles"]] == ["id-0"]
    assert len(calls) == 3  # original lookup, new lookup, headlines


@pytest.mark.parametrize("days,expected_calls", [(6, 1), (8, 2)])
def test_positive_listing_cache_lasts_seven_days(service, monkeypatch, days, expected_calls):
    calls = upstream(monkeypatch, service)
    listing, _, error = service._resolve("AAPL", None, float("inf"))
    assert error is None

    expire_cache(hours=days * 24)
    refreshed, _, error = service._resolve("AAPL", None, float("inf"))

    assert error is None
    assert refreshed == listing
    assert len(calls) == expected_calls


def test_filters_unsafe_wrong_language_old_future_and_wrong_entity(service, monkeypatch):
    bad = [article(1, language="fr"), article(2, url="javascript:alert(1)"),
           article(3, published_at=(service._now() - timedelta(days=8)).isoformat()),
           article(4, published_at=(service._now() + timedelta(days=1)).isoformat()),
           article(5, entities=[entity("AAP")] ), article(6, published_at="broken"),
           article(7, entities=[entity(country="gb")]), article(8, title=None)]
    items = [article(), *bad]
    upstream(monkeypatch, service, [items[i:i+3] for i in range(0, len(items), 3)])
    feed = service.get_feed("AAPL")
    assert [a["id"] for a in feed["articles"]] == ["id-0"]


def test_deduplicates_ids_and_urls_including_tracking_and_fragments(service, monkeypatch):
    upstream(monkeypatch, service, [[article(), article(1, uuid="id-0"),
                                   article(2, url="https://publisher.example/story/0?utm_source=feed#top")]])
    assert len(service.get_feed("AAPL")["articles"]) == 1


def test_ambiguous_listing_requires_validated_selection(service, monkeypatch):
    candidates = [entity(exchange="NASDAQ"), entity(exchange="LSE", country="gb")]
    calls = upstream(monkeypatch, service, entities=candidates,
                     pages=[[article(entities=[candidates[1]])]])
    feed = service.get_feed("AAPL")
    assert feed["status"] == "needs_selection"
    assert len(calls) == 1
    assert service.get_feed("AAPL", "forged-listing")["status"] == "invalid"
    chosen = service.get_feed("AAPL", feed["candidates"][1]["id"])
    assert chosen["listing"]["country"] == "gb"
    assert chosen["status"] == "ok"


def test_preserves_international_suffix_and_verifies_exchange(service, monkeypatch):
    listing = entity("VOD.L", "LSE", "gb", "Vodafone Group")
    wrong = entity("VOD.L", "NYSE", "gb", "Other listing")
    calls = upstream(monkeypatch, service, entities=[listing],
                     pages=[[article(entities=[listing]), article(1, entities=[wrong])]])
    feed = service.get_feed("vod.l")
    assert feed["listing"]["symbol"] == "VOD.L"
    assert len(feed["articles"]) == 1
    assert calls[0][1]["symbols"] == "VOD.L"
    assert "countries" not in calls[0][1]
    assert calls[1][1]["symbols"] == "VOD.L"


def test_no_silent_symbol_substitution(service, monkeypatch):
    calls = upstream(monkeypatch, service, entities=[entity("VOD")])
    assert service.get_feed("VOD.L")["status"] == "unsupported"
    assert len(calls) == 1


def test_successful_empty_is_cached_separately_from_unsupported(service, monkeypatch):
    calls = upstream(monkeypatch, service, pages=[[]])
    assert service.get_feed("AAPL")["status"] == "empty"
    assert service.get_feed("AAPL")["status"] == "empty"
    assert len(calls) == 2


@pytest.mark.parametrize("failure", [401, 429, 500, requests.Timeout("secret-token-url")])
def test_failed_requests_cool_down_and_never_leak_secrets(service, monkeypatch, failure, capsys):
    calls = upstream(monkeypatch, service, pages=[failure])
    feed = service.get_feed("AAPL")
    assert feed["status"] == "unavailable"
    assert service.get_feed("AAPL")["status"] == "unavailable"
    assert len(calls) == 2
    assert "secret-token-url" not in str(feed) + capsys.readouterr().out


def test_stale_success_keeps_timestamp_on_refresh_failure(service, monkeypatch):
    upstream(monkeypatch, service)
    original = service.get_feed("AAPL")
    expire_cache()
    calls = upstream(monkeypatch, service, pages=[500])
    stale = service.get_feed("AAPL")
    assert stale["stale"] is True
    assert stale["fetched_at"] == original["fetched_at"]
    assert stale["articles"] == original["articles"]
    assert len(calls) == 1  # entity cache still valid
    assert service.get_feed("AAPL")["stale"] is True
    assert len(calls) == 1


def test_does_not_serve_snapshot_older_than_24_hours(service, monkeypatch):
    upstream(monkeypatch, service)
    service.get_feed("AAPL")
    expire_cache(hours=25)
    upstream(monkeypatch, service, pages=[500])
    assert service.get_feed("AAPL")["articles"] == []


def test_exhausted_budget_prevents_upstream_calls(service, monkeypatch):
    for _ in range(90):
        assert db.reserve_news_request(service._now().date().isoformat())
    calls = upstream(monkeypatch, service)
    assert service.get_feed("AAPL")["status"] == "quota_exhausted"
    assert calls == []


def test_partial_pagination_returns_obtained_articles(service, monkeypatch):
    upstream(monkeypatch, service, pages=[[article(i) for i in range(3)], 500])
    feed = service.get_feed("AAPL")
    assert feed["status"] == "ok" and feed["partial"] is True
    assert len(feed["articles"]) == 3


def test_weaker_partial_does_not_replace_fuller_cached_success(service, monkeypatch):
    upstream(monkeypatch, service, pages=[[article(i) for i in range(3)], [article(3)]])
    original = service.get_feed("AAPL")
    expire_cache()
    upstream(monkeypatch, service, pages=[[article(i) for i in range(3)], 500])
    feed = service.get_feed("AAPL")
    assert len(feed["articles"]) == 4
    assert feed["fetched_at"] == original["fetched_at"] and feed["stale"] is True


def test_missing_key_and_invalid_inputs_never_make_requests(service, monkeypatch):
    get = Mock(side_effect=AssertionError("unexpected network"))
    monkeypatch.setattr(service._SESSION, "get", get)
    assert service.get_feed("aapl&symbols=TSLA")["status"] == "invalid"
    monkeypatch.delenv("MARKETAUX_API_KEY")
    assert service.get_feed("AAPL")["status"] == "unconfigured"


def test_competing_worker_does_not_refresh_same_listing(service, monkeypatch):
    calls = upstream(monkeypatch, service)
    # Resolve/cache the listing, then simulate a worker already refreshing it.
    feed = service.get_feed("AAPL")
    expire_cache()
    assert db.try_claim_lock("news:feed:" + feed["listing"]["id"], 2 / 60)
    assert service.get_feed("AAPL")["stale"] is True
    assert len(calls) == 2


def test_cached_articles_are_pruned_again_on_read(service, monkeypatch):
    upstream(monkeypatch, service, pages=[[article(), article(1,
        published_at=(service._now() - timedelta(days=7, minutes=-10)).isoformat())]])
    assert len(service.get_feed("AAPL")["articles"]) == 2
    later = service._now() + timedelta(minutes=20)
    monkeypatch.setattr(service, "_now", lambda: later)
    assert len(service.get_feed("AAPL")["articles"]) == 1


def test_failure_is_charged_and_retry_after_is_respected(service, monkeypatch):
    calls = upstream(monkeypatch, service, pages=[429])
    assert service.get_feed("AAPL")["status"] == "unavailable"
    with db.get_conn() as conn:
        assert conn.execute("SELECT attempts FROM news_request_usage").fetchone()[0] == 2
    later = service._now() + timedelta(minutes=6)
    monkeypatch.setattr(service, "_now", lambda: later)
    assert service.get_feed("AAPL")["status"] == "unavailable"
    assert len(calls) == 2  # Retry-After is ten minutes, longer than default cooldown


def test_global_budget_stops_mid_pagination_and_returns_partial(service, monkeypatch):
    for _ in range(88):
        db.reserve_news_request(service._now().date().isoformat())
    calls = upstream(monkeypatch, service, pages=[[article(i) for i in range(3)]])
    feed = service.get_feed("AAPL")
    assert feed["status"] == "ok" and feed["partial"] is True
    assert len(feed["articles"]) == 3
    assert len(calls) == 2


def test_malformed_upstream_json_returns_availability_state(service, monkeypatch):
    get = Mock(return_value=SimpleNamespace(status_code=200, headers={},
        json=Mock(side_effect=ValueError("url-with-token"))))
    monkeypatch.setattr(service._SESSION, "get", get)
    assert service.get_feed("AAPL")["status"] == "unavailable"


def test_redirects_cannot_issue_uncharged_requests(service, monkeypatch):
    def get(url, **kwargs):
        assert kwargs.get("allow_redirects") is False
        return SimpleNamespace(status_code=302, headers={}, json=lambda: {})
    monkeypatch.setattr(service._SESSION, "get", get)
    assert service.get_feed("AAPL")["status"] == "unavailable"
