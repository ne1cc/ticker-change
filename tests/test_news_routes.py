"""News routes, settings, and stock page isolation."""
from unittest.mock import patch

import pytest

import app as app_module
import db
import news


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "routes.db"))
    db.init_db()
    monkeypatch.delenv("MARKETAUX_API_KEY", raising=False)
    return app_module.app.test_client()


def test_news_page_without_ticker_does_not_fetch(client):
    with patch("news.get_feed", side_effect=AssertionError("unexpected fetch")):
        response = client.get("/news")
    assert response.status_code == 200
    assert b'name="ticker"' in response.data


@pytest.mark.parametrize("status,code", [("ok", 200), ("empty", 200),
    ("needs_selection", 200), ("unsupported", 200), ("invalid", 400),
    ("unconfigured", 503), ("quota_exhausted", 503), ("unavailable", 503)])
def test_feed_status_codes_and_useful_body(client, status, code):
    feed = news._result("AAPL", status)
    with patch("news.get_feed", return_value=feed):
        response = client.get("/api/news?ticker=AAPL")
        page = client.get("/news?ticker=AAPL")
    assert response.status_code == page.status_code == code
    if code == 400:
        assert response.content_type == "application/problem+json"
        assert response.json["status"] == 400
    else:
        assert response.json["status"] == status
    if feed["message"]:
        assert feed["message"].encode() in page.data


def test_page_escapes_headlines_and_shows_stale_timestamp(client):
    feed = news._result("VOD.L", "ok")
    feed.update(stale=True, fetched_at="2026-10-14T12:00:00Z",
                listing=dict(id="VOD.L|LSE|gb", symbol="VOD.L", exchange="LSE", country="gb", name="Vodafone"),
                articles=[dict(title="<script>unsafe()</script>", source="publisher.example",
                               url="https://publisher.example/story", published_at="2026-10-14T11:00:00Z")])
    with patch("news.get_feed", return_value=feed):
        response = client.get("/news?ticker=VOD.L")
    assert b"<script>unsafe()" not in response.data
    assert b"&lt;script&gt;unsafe()" in response.data
    assert b"2026-10-14T12:00:00Z" in response.data
    assert b"Saved headlines" in response.data


def test_stock_document_does_not_fetch_news(client):
    with patch.object(app_module, "get_stock_data", return_value={"ticker": "AAPL", "error": "Prices unavailable"}), \
         patch("news.get_feed", side_effect=AssertionError("blocking news fetch")):
        response = client.get("/stock?ticker=AAPL")
    assert response.status_code == 200
    assert b'id="news-preview"' not in response.data
    assert b"/static/news.js" not in response.data


@pytest.mark.parametrize("ui_mode", ["excel", "light"])
def test_news_page_renders_every_returned_article(client, ui_mode):
    client.set_cookie("ui_mode", ui_mode)
    feed = news._result("SPY", "ok")
    feed["articles"] = [dict(title=f"Headline {i}", source="publisher.example",
                             url=f"https://publisher.example/story/{i}",
                             published_at="2026-10-19T08:00:00Z") for i in range(100)]
    with patch("news.get_feed", return_value=feed):
        response = client.get("/news?ticker=SPY")
    assert response.status_code == 200
    assert response.data.count(b'https://publisher.example/story/') == 100


def test_news_is_same_origin_but_existing_api_cors_is_preserved(client):
    response = client.get("/api/news?ticker=AAPL", headers={"Origin": "https://foreign.example"})
    assert "Access-Control-Allow-Origin" not in response.headers
    with patch.object(app_module, "get_stock_data", return_value={"error": "no prices"}):
        response = client.get("/api/stock/AAPL", headers={"Origin": "https://foreign.example"})
    assert response.headers.get("Access-Control-Allow-Origin") == "https://foreign.example"


def test_marketaux_key_settings_remain_authenticated(client, monkeypatch):
    monkeypatch.setenv("SETTINGS_PASSWORD", "settings-password")
    assert client.post("/settings", data={"marketaux_api_key": "private-news-key"}).status_code == 401
    assert db.get_setting("marketaux_api_key") == ""
    response = client.post("/settings", data={"marketaux_api_key": "private-news-key"},
                           auth=("admin", "settings-password"))
    assert response.status_code == 200
    assert db.get_setting("marketaux_api_key") == "private-news-key"
    assert b"private-news-key" not in client.get("/api/news?ticker=bad%26query").data
