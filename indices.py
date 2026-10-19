"""Index registry for the heatmap: label + constituent fetcher per index.

Each fetcher returns [{symbol, name, sector}] with yfinance-normalised symbols
('.' -> '-') and raises RuntimeError on a fetch/parse failure or an
implausible size, so a broken source can never replace a good cached list.
`sector` is the source's own text and may be blank; heatmap.py overrides it
with the S&P 500 GICS sector whenever the symbol is an S&P 500 member.
"""
from __future__ import annotations

import io
import re

import pandas as pd
import requests

import sp500

_HEADERS = {"User-Agent": "Mozilla/5.0 ticker-change/1.0 (research dashboard)"}
_NDX_URL = "https://api.nasdaq.com/api/quote/list-type/nasdaq100"
_DOW_URL = "https://en.wikipedia.org/wiki/List_of_Dow_Jones_Industrial_Average_companies"


def _get(url: str, **headers) -> requests.Response:
    try:
        resp = requests.get(url, headers={**_HEADERS, **headers}, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Could not fetch {url}: {e}") from e
    return resp


def _symbol(raw) -> str:
    return str(raw).strip().replace(".", "-")


def parse_ndx(payload: dict) -> list[dict]:
    """Rows from Nasdaq's list-type/nasdaq100 JSON (101: GOOGL and GOOG both count)."""
    try:
        raw = payload["data"]["data"]["rows"]
    except (KeyError, TypeError) as e:
        raise RuntimeError("Nasdaq-100 response shape changed") from e
    rows = [{
        "symbol": _symbol(r["symbol"]),
        # "Apple Inc. Common Stock" -> "Apple Inc."
        "name": re.sub(r"\s+Common Stock$", "", r.get("companyName") or "").strip(),
        "sector": (r.get("sector") or "").strip(),
    } for r in raw if r.get("symbol")]
    if not 90 <= len(rows) <= 110:
        raise RuntimeError(f"Implausible Nasdaq-100 size ({len(rows)} rows)")
    return rows


def fetch_ndx() -> list[dict]:
    return parse_ndx(_get(_NDX_URL, Accept="application/json").json())


def parse_dow(html: str) -> list[dict]:
    """Rows from the Wikipedia components table (the one with a Symbol column)."""
    try:
        tables = pd.read_html(io.StringIO(html), flavor="lxml")
    except ValueError as e:  # pandas raises when it finds no tables
        raise RuntimeError("No tables on the Dow components page") from e
    for table in tables:
        if "Symbol" in table.columns and 25 <= len(table) <= 35:
            return [{
                "symbol": _symbol(r["Symbol"]),
                "name": str(r.get("Company", "")).strip(),
                "sector": "" if pd.isna(r.get("Sector")) else str(r["Sector"]).strip(),
            } for _, r in table.iterrows()]
    raise RuntimeError("Dow components table not found -- the page layout may have changed")


def fetch_dow() -> list[dict]:
    return parse_dow(_get(_DOW_URL).text)


# Fetchers are looked up at call time (lambda) so tests can patch the module-level functions.
INDICES = {
    "sp500": {"label": "S&P 500", "fetch": lambda: sp500.fetch_sp500_constituents()},
    "ndx": {"label": "Nasdaq-100", "fetch": lambda: fetch_ndx()},
    "dow": {"label": "Dow 30", "fetch": lambda: fetch_dow()},
}
DEFAULT_INDEX = "sp500"
