"""Marketaux ticker headlines, with shared cache and a bounded daily budget."""
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import re
import time
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests

import db
import providers

_SESSION = requests.Session()  # requests defaults to no automatic retries
_PROVIDER = "marketaux_news"
_BASE = "https://api.marketaux.com/v1"


def _now():
    return datetime.now(timezone.utc)


def _date(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso(value):
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _result(ticker, status, listing=None):
    messages = {
        "empty": "No English-language headlines found in the last seven days.",
        "needs_selection": "Choose the listing you are researching.",
        "unsupported": "News coverage is unavailable for this listing.",
        "unconfigured": "News is not configured yet. Please check back later.",
        "quota_exhausted": "News updates are paused for today. Please check back later.",
        "unavailable": "News is temporarily unavailable. Please try again later.",
        "invalid": "Enter a valid ticker and select a listing from the available choices.",
    }
    return dict(ticker=ticker, status=status, listing=listing,
                candidates=[], articles=[], fetched_at=None, stale=False,
                partial=False, message=messages.get(status, ""), limit=15)


def _cooldown():
    hit = db.cache_get(_PROVIDER, "cooldown", 24)
    return hit if hit and _date(hit["until"]) > _now() else None


def _request(path, params, deadline):
    """One charged attempt; never expose authenticated URLs or exceptions."""
    if _cooldown() or time.monotonic() >= deadline:
        return None, "unavailable"
    if not db.reserve_news_request(_now().date().isoformat()):
        return None, "quota_exhausted"
    try:
        response = _SESSION.get(
            _BASE + path, params={**params, "api_token": providers.marketaux_key()},
            timeout=(3, min(5, max(0.1, deadline - time.monotonic()))), allow_redirects=False,
        )
        if response.status_code == 200:
            payload = response.json()
            if isinstance(payload, dict) and isinstance(payload.get("data"), list):
                return payload["data"], None
        retry = response.headers.get("Retry-After", "")
        try:
            delay = max(300, int(retry))
        except (ValueError, TypeError):
            try:
                delay = max(300, (parsedate_to_datetime(retry) - _now()).total_seconds())
            except (ValueError, TypeError, OverflowError):
                delay = 300
    except (requests.RequestException, ValueError):
        delay = 300
    db.cache_set(_PROVIDER, "cooldown", {"until": _iso(_now() + timedelta(seconds=min(delay, 86400)))})
    return None, "unavailable"


def _listing(item):
    symbol = item.get("symbol")
    if not isinstance(symbol, str) or not symbol:
        return None
    country = str(item.get("country") or "").lower()
    exchange = str(item.get("exchange") or "")
    symbol = symbol.upper()
    return dict(id="|".join((symbol, exchange, country)), symbol=symbol,
                exchange=exchange, country=country, name=str(item.get("name") or symbol))


def _resolve(ticker, listing_id, deadline):
    key = "entity:" + ticker
    cached = db.cache_get(_PROVIDER, key, 24 * 7)
    if cached is None:
        if not db.try_claim_lock("news:" + key, 2 / 60):
            return None, [], "unavailable"
        params = {"symbols": ticker, "types": "equity"}
        if "." not in ticker:
            params["countries"] = "us"
        items, error = _request("/entity/search", params, deadline)
        if error:
            return None, [], error
        candidates = {}
        for item in items:
            candidate = _listing(item) if isinstance(item, dict) else None
            if candidate and candidate["symbol"] == ticker:
                candidates[candidate["id"]] = candidate
        cached = {"candidates": list(candidates.values())}
        db.cache_set(_PROVIDER, key, cached)
    candidates = cached["candidates"]
    if listing_id:
        selected = next((c for c in candidates if c["id"] == listing_id), None)
        return selected, candidates, None if selected else "invalid"
    if len(candidates) == 1:
        return candidates[0], candidates, None
    return None, candidates, "needs_selection" if candidates else "unsupported"


def _url(value):
    if not isinstance(value, str) or any(c.isspace() for c in value) or "\\" in value:
        return None
    try:
        parts = urlsplit(value)
        if parts.scheme not in ("https", "http") or not parts.hostname or parts.username or parts.password:
            return None
        query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                 if not k.lower().startswith("utm_") and k.lower() not in ("fbclid", "gclid")]
        return urlunsplit((parts.scheme, parts.netloc.lower(), parts.path, urlencode(query), ""))
    except ValueError:
        return None


def _articles(items, listing):
    valid, ids, urls = [], set(), set()
    for item in items:
        if not isinstance(item, dict) or item.get("language") != "en":
            continue
        title, identity = item.get("title"), item.get("uuid") or item.get("id")
        url = _url(item.get("url"))
        if not isinstance(title, str) or not title.strip() or not isinstance(identity, str) or not url:
            continue
        try:
            published = _date(item.get("published_at", ""))
        except (ValueError, TypeError, AttributeError):
            continue
        if not _now() - timedelta(days=7) <= published <= _now():
            continue
        entities = item.get("entities")
        if not isinstance(entities, list) or not any(
            isinstance(e, dict) and str(e.get("symbol", "")).upper() == listing["symbol"]
            and (not listing["country"] or str(e.get("country") or "").lower() == listing["country"])
            and (not listing["exchange"] or e.get("exchange") == listing["exchange"])
            for e in entities
        ):
            continue
        if identity in ids or url in urls:
            continue
        ids.add(identity)
        urls.add(url)
        valid.append(dict(id=identity, title=title.strip(), url=url,
                          source=urlsplit(url).hostname, published_at=_iso(published),
                          language="en", entities=[{k: listing[k] for k in ("symbol", "exchange", "country")}]))
    return sorted(valid, key=lambda a: a["published_at"], reverse=True)[:15]


def _snapshot(key, listing):
    cached, stored_at = db.cache_get_with_age(_PROVIDER, key, 24)
    if cached is None:
        return None, False
    cached["articles"] = _articles(cached["articles"], listing)
    # Do not serve an old populated snapshot whose articles have all aged out.
    if cached["status"] == "ok" and not cached["articles"]:
        return None, False
    fresh = _now() - _date(stored_at) < timedelta(hours=1)
    return cached, fresh


def _fallback(ticker, listing, cached, error):
    if cached is not None:
        return {**cached, "stale": True,
                "message": "Showing saved headlines; a news update is temporarily unavailable."}
    return _result(ticker, error, listing=listing)


def get_feed(ticker: str, listing_id: str | None = None) -> dict:
    """Resolve one listing and fetch up to five pages, or serve its saved feed."""
    ticker = ticker.strip().upper()
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9.^=-]{0,31}", ticker) or (listing_id and len(listing_id) > 200):
        return _result(ticker, "invalid")
    if not providers.marketaux_key():
        return _result(ticker, "unconfigured")
    deadline = time.monotonic() + 50
    listing, candidates, error = _resolve(ticker, listing_id, deadline)
    if error:
        result = _result(ticker, error)
        result["candidates"] = candidates
        return result
    key = "feed:" + listing["id"] + ":en:7d"
    cached, fresh = _snapshot(key, listing)
    if fresh:
        return cached
    if _cooldown() or not db.try_claim_lock("news:feed:" + listing["id"], 2 / 60):
        return _fallback(ticker, listing, cached, "unavailable")
    items, error = [], None
    for page in range(1, 6):
        params = dict(symbols=listing["symbol"], language="en", filter_entities="true",
                      group_similar="true", limit=3, page=page,
                      published_after=(_now() - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S"))
        batch, error = _request("/news/all", params, deadline)
        if error:
            break
        items.extend(batch)
        if len(batch) < 3 or len(_articles(items, listing)) >= 15:
            break
    articles = _articles(items, listing)
    if error and (not articles or cached and len(cached["articles"]) >= len(articles)):
        return _fallback(ticker, listing, cached, error)
    feed = _result(ticker, "ok" if articles else "empty", listing=listing)
    feed.update(articles=articles, fetched_at=_iso(_now()), partial=bool(error))
    if error:
        feed["message"] = "Some headlines could not be loaded. Showing the available results."
    db.cache_set(_PROVIDER, key, feed)
    return feed
