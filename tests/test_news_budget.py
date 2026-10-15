"""News quota reservations must survive connections and competing workers."""
from concurrent.futures import ThreadPoolExecutor

import pytest

import db
import providers


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "news.db"))
    db.init_db()


def test_budget_stops_at_limit_and_resets_on_new_day():
    assert db.reserve_news_request("2026-10-14", limit=2)
    assert db.reserve_news_request("2026-10-14", limit=2)
    assert not db.reserve_news_request("2026-10-14", limit=2)
    db.init_db()  # reopening/initializing does not reset usage
    assert not db.reserve_news_request("2026-10-14", limit=2)
    assert db.reserve_news_request("2026-10-15", limit=2)


def test_budget_reserves_atomically_across_connections():
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: db.reserve_news_request("2026-10-14"), range(110)))
    assert sum(results) == 90


def test_zero_budget_never_reserves():
    assert not db.reserve_news_request("2026-10-14", limit=0)


def test_marketaux_settings_override_environment(monkeypatch):
    monkeypatch.delenv("MARKETAUX_API_KEY", raising=False)
    assert providers.marketaux_key() == ""
    assert providers.configured()["marketaux"] is False
    monkeypatch.setenv("MARKETAUX_API_KEY", "environment-secret")
    assert providers.marketaux_key() == "environment-secret"
    db.set_setting("marketaux_api_key", "settings-secret")
    assert providers.marketaux_key() == "settings-secret"
    status = providers.configured()
    assert status["marketaux"] is True
    assert "settings-secret" not in str(status)
