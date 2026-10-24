"""Source / as-of receipts for /analytics panels.

Intent: every analytics panel states where its numbers came from and how old
they are, derived from the cache rows and data actually used rather than typed
into the template.

Contract:
- receipt() returns {panel, label, source, as_of, age_text, state, note}.
- state is one of: live, stale (older than the fetcher's TTL), assumed (inputs
  were defaults, not reported), no_asof (source known, timestamp not recorded),
  unavailable (the panel's data is absent).

Limits: as_of reads api_cache.fetched_at, i.e. when we fetched, not the
provider's own observation time; panels whose fetcher does not write
api_cache report no_asof rather than a guessed time.
"""
from __future__ import annotations

from datetime import date, datetime

import ai
import db
import earnings
import ml

# panel -> (label, source, (api_cache provider, key template, prefix?) | None, ttl_hours)
# TTLs mirror each fetcher's own freshness window (app.CHAIN_TTL_HOURS = 4).
PANELS = {
    "prices": ("Price history", "Yahoo Finance via SQLite daily_prices", None, 96),
    "fundamentals": ("Fundamentals", "Yahoo Finance (yfinance)", ("yfinance", "fundamentals:{T}", False), 24),
    "price_target": ("Analyst price target", "Yahoo Finance (yfinance)", ("yfinance", "fundamentals:{T}", False), 24),
    "insider": ("Insider transactions", "Yahoo Finance (yfinance)", ("yfinance", "insider:{T}", False), 24),
    "earnings": ("Earnings event study", "Yahoo Finance earnings dates + price history",
                 ("yfinance", "earnings:{T}", False), earnings.EARNINGS_TTL_HOURS),
    "options_chain": ("Options chain (IV smile, GEX)", "Option-chain cache (S3 / SQLite, Yahoo, Cboe)",
                      ("yfinance", "optchain:{T}:", True), 4),
    "ml": ("ML signal", "Local gradient-boosted model", ("ml", "{T}", False), ml._CACHE_TTL_HOURS),
    "macro": ("Macro conditioning", "FRED", None, 24),
    "short_interest": ("Short interest", "Short-interest provider", None, 24),
    "ai": ("AI analyst report", "LLM provider", ("ai_report", "{T}", False), ai.CACHE_TTL_HOURS),
}


def _naive(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if isinstance(value, date) and not isinstance(value, datetime):
        value = datetime(value.year, value.month, value.day)
    return value.replace(tzinfo=None) if value.tzinfo else value


def _age_text(hours: float) -> str:
    if hours < 1:
        return "<1h old"
    return f"{hours:.0f}h old" if hours < 48 else f"{hours / 24:.0f}d old"


def receipt(panel: str, ticker: str, *, present: bool, as_of=None, assumed=(), now: datetime | None = None) -> dict:
    """Receipt for one panel. `present` says whether the panel has data;
    `as_of` overrides the cache lookup (e.g. the last price bar); `assumed`
    names inputs that fell back to defaults."""
    label, source, cache, ttl = PANELS[panel]
    out = {"panel": panel, "label": label, "source": source, "as_of": None,
           "age_text": "", "state": "unavailable", "note": ""}
    if not present:
        return out
    if as_of is None and cache is not None:
        provider, key, prefix = cache
        try:
            as_of = db.cache_fetched_at(provider, key.format(T=ticker.upper()), prefix)
        except Exception:  # an unreadable cache must not take the panel down; declare as-of unknown
            as_of = None
    as_of = _naive(as_of)
    if as_of is None:
        out["state"] = "no_asof"
    else:
        age = ((now or datetime.utcnow()) - as_of).total_seconds() / 3600.0
        out.update(as_of=as_of.isoformat(timespec="minutes"), age_text=_age_text(max(age, 0.0)),
                   state="stale" if age > ttl else "live")
    if assumed:
        out.update(state="assumed", note="assumed inputs: " + ", ".join(assumed))
    return out
