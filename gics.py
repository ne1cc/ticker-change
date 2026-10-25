"""GICS Sector and Sub-Industry taxonomy mapping source for S&P 500 constituents.

See ADR 0002 (docs/adr/0002-gics-sector-mapping-source.md):
- Sourced from frictionlessdata/datasets public domain CSV (PDDL / CC0 1.0)
- Cached 7 days in SQLite (api_cache, provider="gics_taxonomy")
- Ticker symbol normalisation: '.' -> '-' (e.g. BRK.B -> BRK-B)
- Unmapped tickers return None; sector percentiles degrade gracefully to universe-only.
"""
from __future__ import annotations

import csv
import io
import os
import requests

import db

_SP500_CSV_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/"
    "master/data/constituents.csv"
)

_CACHE_PROVIDER = "gics_taxonomy"
_CACHE_KEY = "constituents"
_CACHE_TTL_HOURS = 24 * 7  # 7 days

# Bundled fallback fixture path for deterministic offline operation and test isolation
_FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "tests", "fixtures", "sp500_constituents.csv"
)


def normalize_ticker(ticker: str) -> str:
    """Normalise ticker symbol (uppercase and '.' to '-')."""
    return (ticker or "").strip().upper().replace(".", "-")


def parse_gics_csv(csv_text: str) -> dict[str, dict[str, str]]:
    """Parse CSV text into a mapping of normalized symbol -> constituent info."""
    reader = csv.DictReader(io.StringIO(csv_text))
    mapping: dict[str, dict[str, str]] = {}
    for row in reader:
        sym = (row.get("Symbol") or "").strip()
        if not sym:
            continue
        norm = normalize_ticker(sym)
        sector = (
            row.get("GICS Sector") or row.get("Sector") or ""
        ).strip() or None
        sub_industry = (
            row.get("GICS Sub-Industry") or row.get("Sub-Industry") or ""
        ).strip() or None
        name = (row.get("Security") or row.get("Name") or "").strip()

        mapping[norm] = {
            "symbol": norm,
            "raw_symbol": sym,
            "name": name,
            "sector": sector,
            "sub_industry": sub_industry,
        }
    return mapping


def fetch_gics_mapping(force_refresh: bool = False) -> dict[str, dict[str, str]]:
    """Fetch and return the complete GICS constituent taxonomy mapping.

    Cached in SQLite with a 7-day TTL. On network failure, falls back to
    expired cache, local fixture, or empty dict without crashing.
    """
    if not force_refresh:
        cached = db.cache_get(_CACHE_PROVIDER, _CACHE_KEY, _CACHE_TTL_HOURS)
        if isinstance(cached, dict) and cached:
            return cached

    # Attempt network fetch
    try:
        resp = requests.get(_SP500_CSV_URL, timeout=15)
        resp.raise_for_status()
        mapping = parse_gics_csv(resp.text)
        if len(mapping) >= 400:
            db.cache_set(_CACHE_PROVIDER, _CACHE_KEY, mapping)
            return mapping
    except Exception as e:
        # Fall back to stale cache
        stale = db.cache_get(_CACHE_PROVIDER, _CACHE_KEY, ttl_hours=24 * 365)
        if isinstance(stale, dict) and stale:
            return stale

    # Fall back to local fixture if available
    if os.path.exists(_FIXTURE_PATH):
        try:
            with open(_FIXTURE_PATH, "r", encoding="utf-8") as f:
                mapping = parse_gics_csv(f.read())
                if mapping:
                    return mapping
        except Exception:
            pass

    return {}


def sector_of(ticker: str) -> str | None:
    """Return the GICS sector of the ticker, or None if unmapped."""
    if not ticker:
        return None
    mapping = fetch_gics_mapping()
    norm = normalize_ticker(ticker)
    info = mapping.get(norm)
    if info and info.get("sector"):
        return info["sector"]
    return None


def sub_industry_of(ticker: str) -> str | None:
    """Return the GICS sub-industry of the ticker, or None if unmapped."""
    if not ticker:
        return None
    mapping = fetch_gics_mapping()
    norm = normalize_ticker(ticker)
    info = mapping.get(norm)
    if info and info.get("sub_industry"):
        return info["sub_industry"]
    return None


def constituents_by_sector(sector: str) -> list[str]:
    """Return list of normalized tickers belonging to the given GICS sector."""
    if not sector:
        return []
    target = sector.strip().lower()
    mapping = fetch_gics_mapping()
    return sorted(
        t for t, data in mapping.items()
        if (data.get("sector") or "").strip().lower() == target
    )


def all_sectors() -> list[str]:
    """Return sorted unique list of GICS sectors in the mapped universe."""
    mapping = fetch_gics_mapping()
    sectors = {
        data["sector"]
        for data in mapping.values()
        if data.get("sector")
    }
    return sorted(sectors)
