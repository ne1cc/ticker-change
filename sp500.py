"""S&P 500 ticker universe: fetch the constituent list, and backfill price
history for the whole index so ml.py's triple-barrier training has more than
a handful of names to learn from.

ml.py's `_dataset()` reads only from SQLite (`db.get_prices`) -- it never
fetches from yfinance -- so training on tickers the dashboard has never been
pointed at needs an explicit backfill step first. This module does both:
fetching the current constituent list, and pulling + caching daily price
history for every one of them.

Kept independent of app.py deliberately: importing app.py runs
`_start_options_cache_warmer()` at import time, which is the wrong side
effect to trigger from an offline training script.

Usage (see ml.py's CLI, which drives this):
    python ml.py backfill --sp500
    python ml.py train --sp500
"""
from __future__ import annotations

import csv
import io
import time

import requests
import yfinance as yf

import db

# A widely-used, freely-licensed mirror of the S&P 500 constituent table
# (frictionlessdata/datasets project on GitHub). Fetched live rather than
# hardcoded here so the list doesn't silently drift as the index
# reconstitutes a few times a year.
_SP500_CSV_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/"
    "master/data/constituents.csv"
)


def _parse_tickers_csv(csv_text: str) -> list[str]:
    """Extract, yfinance-normalise, sort and dedupe tickers from the raw CSV.

    The official list uses '.' for share classes (BRK.B); yfinance expects
    '-' (BRK-B).
    """
    reader = csv.DictReader(io.StringIO(csv_text))
    tickers = [
        row["Symbol"].strip().replace(".", "-")
        for row in reader
        if row.get("Symbol")
    ]
    return sorted(set(tickers))


def _parse_constituents_csv(csv_text: str) -> list[dict]:
    """Full constituent rows ({symbol, name, sector}) from the raw CSV.

    Same yfinance symbol normalisation as `_parse_tickers_csv` ('.' -> '-');
    a missing/blank sector falls back to "Other" so heatmap grouping never
    drops a name. Column names have fallbacks in case the source renames
    them again.
    """
    reader = csv.DictReader(io.StringIO(csv_text))
    rows = []
    for row in reader:
        symbol = (row.get("Symbol") or "").strip().replace(".", "-")
        if not symbol:
            continue
        rows.append({
            "symbol": symbol,
            "name": (row.get("Security") or row.get("Name") or "").strip(),
            "sector": (row.get("GICS Sector") or row.get("Sector") or "").strip() or "Other",
        })
    return rows


def fetch_sp500_constituents() -> list[dict]:
    """Return the S&P 500 as [{symbol, name, sector}], yfinance-normalised.

    Raises RuntimeError on fetch/parse failure, same contract as
    `fetch_sp500_tickers`.
    """
    try:
        resp = requests.get(_SP500_CSV_URL, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(
            f"Could not fetch the S&P 500 list from {_SP500_CSV_URL}: {e}."
        ) from e

    rows = _parse_constituents_csv(resp.text)
    if len(rows) < 400:
        raise RuntimeError(
            f"Fetched an implausibly short S&P 500 list ({len(rows)} "
            "rows) -- the source's CSV format may have changed."
        )
    return rows


def fetch_sp500_tickers() -> list[str]:
    """Return the current S&P 500 constituent tickers, yfinance-normalised.

    Raises RuntimeError on any fetch/parse failure rather than silently
    returning a partial list.
    """
    try:
        resp = requests.get(_SP500_CSV_URL, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(
            f"Could not fetch the S&P 500 list from {_SP500_CSV_URL}: {e}. "
            "Pass tickers explicitly instead of --sp500."
        ) from e

    tickers = _parse_tickers_csv(resp.text)
    if len(tickers) < 400:
        raise RuntimeError(
            f"Fetched an implausibly short S&P 500 list ({len(tickers)} "
            "tickers) -- the source's CSV format may have changed."
        )
    return tickers


def _fetch_with_retry(symbol: str, period: str = "5y", retries: int = 3):
    """Same backoff shape as app.py's _fetch_yfinance_with_retry, reimplemented
    here rather than imported so this module never pulls in app.py."""
    delay = 0.6
    for attempt in range(retries):
        try:
            df = yf.Ticker(symbol).history(period=period, auto_adjust=True)
            if not df.empty:
                return df
        except Exception as e:
            print(f"  {symbol}: fetch failed (attempt {attempt + 1}/{retries}): {e}")
        if attempt < retries - 1:
            time.sleep(delay)
            delay *= 2
    return None


def backfill(tickers: list[str], period: str = "5y", pause: float = 0.3) -> None:
    """Ensure each ticker has cached price history, fetching whatever is
    missing or stale. Safe to re-run -- already-fresh tickers are skipped."""
    total = len(tickers)
    fetched = skipped = failed = 0
    for i, t in enumerate(tickers, 1):
        t = t.upper()
        if db.is_fresh(t):
            skipped += 1
            continue
        df = _fetch_with_retry(t, period)
        if df is None or df.empty:
            print(f"[{i}/{total}] {t}: FAILED (no data)")
            failed += 1
            continue
        db.store_prices(t, df)
        fetched += 1
        print(f"[{i}/{total}] {t}: cached {len(df)} rows")
        time.sleep(pause)  # be polite to yfinance's free tier across ~500 names
    print(f"\nBackfill complete: {fetched} fetched, {skipped} already fresh, {failed} failed.")
