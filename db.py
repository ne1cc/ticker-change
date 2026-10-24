from __future__ import annotations
import os
import sqlite3
import json
import pandas as pd
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager

DB_PATH = os.environ.get("DB_PATH", "stocks.db")
FRESHNESS_HOURS = 1


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS tickers (\n                symbol       TEXT PRIMARY KEY,\n                last_fetched DATETIME NOT NULL\n            );\n\n            CREATE TABLE IF NOT EXISTS daily_prices (\n                symbol  TEXT    NOT NULL,\n                date    DATE    NOT NULL,\n                open    REAL    NOT NULL,\n                high    REAL    NOT NULL,\n                low     REAL    NOT NULL,\n                close   REAL    NOT NULL,\n                volume  INTEGER NOT NULL,\n                PRIMARY KEY (symbol, date)\n            );\n\n            CREATE TABLE IF NOT EXISTS api_cache (\n                provider   TEXT     NOT NULL,\n                key        TEXT     NOT NULL,\n                fetched_at DATETIME NOT NULL,\n                payload    TEXT     NOT NULL,\n                PRIMARY KEY (provider, key)\n            );\n\n            CREATE TABLE IF NOT EXISTS app_settings (\n                key   TEXT PRIMARY KEY,\n                value TEXT NOT NULL\n            );\n\n            CREATE TABLE IF NOT EXISTS news_request_usage (\n                day      TEXT PRIMARY KEY,\n                attempts INTEGER NOT NULL\n            );\n\n            CREATE TABLE IF NOT EXISTS gex_snapshots (\n                ticker      TEXT NOT NULL,\n                date        TEXT NOT NULL,\n                gamma_flip  REAL,\n                call_wall   REAL,\n                put_wall    REAL,\n                spot        REAL,\n                created_at  TEXT NOT NULL,\n                PRIMARY KEY (ticker, date)\n            );\n\n            CREATE TABLE IF NOT EXISTS short_interest_snapshots (\n                ticker           TEXT NOT NULL,\n                date             TEXT NOT NULL,\n                short_pct_float  REAL,\n                short_ratio      REAL,\n                shares_short     INTEGER,\n                shares_float     INTEGER,\n                created_at       TEXT NOT NULL,\n                PRIMARY KEY (ticker, date)\n            );\n        """)


def reserve_news_request(day: str, limit: int = 90) -> bool:
    """Reserve one upstream news attempt atomically across workers/restarts."""
    if limit <= 0:
        return False
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO news_request_usage(day, attempts) VALUES (?, 1) "
            "ON CONFLICT(day) DO UPDATE SET attempts = attempts + 1 "
            "WHERE attempts < ?", (day, limit),
        )
        return cur.rowcount == 1


def get_setting(key: str, default: str = "") -> str:
    """Return a stored app setting (e.g. a user-supplied API key), or default."""
    try:
        with get_conn() as conn:
            row = conn.execute(
                "SELECT value FROM app_settings WHERE key = ?", (key,)
            ).fetchone()
    except sqlite3.OperationalError:
        return default  # table not created yet
    return row["value"] if row is not None else default


def set_setting(key: str, value: str):
    """Insert or update an app setting."""
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO app_settings (key, value) VALUES (?, ?)",
            (key, value),
        )


def get_all_settings() -> dict:
    """Return every stored app setting as a {key: value} dict."""
    try:
        with get_conn() as conn:
            rows = conn.execute("SELECT key, value FROM app_settings").fetchall()
    except sqlite3.OperationalError:
        return {}
    return {r["key"]: r["value"] for r in rows}


def cache_get(provider: str, key: str, ttl_hours: float):
    """Return cached JSON payload if present and younger than ttl_hours, else None."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT payload, fetched_at FROM api_cache WHERE provider = ? AND key = ?",
            (provider, key),
        ).fetchone()
    if row is None:
        return None
    fetched_at = datetime.fromisoformat(row["fetched_at"])
    if datetime.utcnow() - fetched_at > timedelta(hours=ttl_hours):
        return None
    try:
        return json.loads(row["payload"])
    except (ValueError, TypeError):
        return None


def cache_get_with_age(provider: str, key: str, ttl_hours: float):
    """(payload, fetched_at_iso) for a live cached row; (None, None) on miss,
    expiry, or unparsable payload. Source-age plumbing for snapshot builds."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT payload, fetched_at FROM api_cache WHERE provider = ? AND key = ?",
            (provider, key),
        ).fetchone()
    if row is None:
        return None, None
    fetched_at = datetime.fromisoformat(row["fetched_at"])
    if datetime.utcnow() - fetched_at > timedelta(hours=ttl_hours):
        return None, None
    try:
        return json.loads(row["payload"]), row["fetched_at"]
    except (ValueError, TypeError):
        return None, None


def cache_set(provider: str, key: str, payload):
    """Store a JSON-serialisable payload under (provider, key)."""
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO api_cache (provider, key, fetched_at, payload) "
            "VALUES (?, ?, ?, ?)",
            (provider, key, datetime.utcnow().isoformat(), json.dumps(payload)),
        )


def cache_scan(provider: str, key_prefix: str) -> dict:
    """All cached payloads whose key starts with prefix, keyed by the key suffix.

    Unlike cache_get, no TTL filter: scan consumers (universe screens) prefer
    any cached payload over none; the warmer keeps payloads fresh.
    """
    try:
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT key, payload FROM api_cache WHERE provider = ? AND key LIKE ?",
                (provider, key_prefix + "%"),
            ).fetchall()
    except sqlite3.OperationalError:
        return {}
    out = {}
    for row in rows:
        try:
            out[row["key"][len(key_prefix):]] = json.loads(row["payload"])
        except (ValueError, TypeError):
            continue
    return out


def try_claim_lock(name: str, ttl_hours: float) -> bool:
    """Atomically claim a cross-process lock row in api_cache.

    Returns True if this process won the claim. A lock older than ttl_hours
    is expired and can be re-claimed. INSERT OR IGNORE handles the
    first-ever claim; re-claims use an UPDATE keyed on the previous
    timestamp so two gunicorn workers racing for an expired lock can't
    both win (only one UPDATE matches the old value).
    """
    now = datetime.utcnow().isoformat()
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO api_cache (provider, key, fetched_at, payload) "
            "VALUES ('internal', ?, ?, '{}')",
            (name, now),
        )
        if cur.rowcount == 1:
            return True
        row = conn.execute(
            "SELECT fetched_at FROM api_cache WHERE provider = 'internal' AND key = ?",
            (name,),
        ).fetchone()
        if row is None:
            return False
        if datetime.utcnow() - datetime.fromisoformat(row["fetched_at"]) < timedelta(hours=ttl_hours):
            return False
        cur = conn.execute(
            "UPDATE api_cache SET fetched_at = ? "
            "WHERE provider = 'internal' AND key = ? AND fetched_at = ?",
            (now, name, row["fetched_at"]),
        )
        return cur.rowcount == 1


def is_fresh(symbol: str) -> bool:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT last_fetched FROM tickers WHERE symbol = ?",
            (symbol.upper(),)
        ).fetchone()
    if row is None:
        return False
    last_fetched = datetime.fromisoformat(row["last_fetched"])
    return datetime.utcnow() - last_fetched < timedelta(hours=FRESHNESS_HOURS)


def get_prices(symbol: str) -> pd.DataFrame | None:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT date, open, high, low, close, volume "
            "FROM daily_prices WHERE symbol = ? ORDER BY date ASC",
            (symbol.upper(),)
        ).fetchall()
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return df


def get_prices_batch(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """One query for many symbols. Returns {symbol: DataFrame} with the same
    per-symbol shape get_prices returns. Symbols with no rows are ABSENT from
    the dict (not None values). Empty input -> empty dict."""
    if not symbols:
        return {}
    upper_symbols = [s.upper() for s in symbols]
    CHUNK_SIZE = 900
    rows = []
    with get_conn() as conn:
        for i in range(0, len(upper_symbols), CHUNK_SIZE):
            chunk = upper_symbols[i:i + CHUNK_SIZE]
            placeholders = ",".join("?" * len(chunk))
            rows.extend(conn.execute(
                "SELECT symbol, date, open, high, low, close, volume "
                f"FROM daily_prices WHERE symbol IN ({placeholders}) "
                "ORDER BY symbol, date ASC",
                chunk,
            ).fetchall())

    rows_by_symbol: dict[str, list] = {}
    for row in rows:
        rows_by_symbol.setdefault(row["symbol"], []).append(row)

    result: dict[str, pd.DataFrame] = {}
    for symbol, symbol_rows in rows_by_symbol.items():
        df = pd.DataFrame(
            symbol_rows,
            columns=["symbol", "date", "open", "high", "low", "close", "volume"],
        )
        df = df.drop(columns=["symbol"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        result[symbol] = df
    return result


def store_prices(symbol: str, df: pd.DataFrame):
    if df is None or df.empty:
        return
    symbol = symbol.upper()
    df = df.copy()
    if df is None or df.empty:
        return

    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    df.index = pd.to_datetime(df.index).tz_localize(None)

    rows = [
        (
            symbol,
            idx.strftime("%Y-%m-%d"),
            float(row["Open"]),
            float(row["High"]),
            float(row["Low"]),
            float(row["Close"]),
            int(row["Volume"]),
        )
        for idx, row in df.iterrows()
    ]

    with get_conn() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO daily_prices (symbol, date, open, high, low, close, volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.execute(
            "INSERT OR REPLACE INTO tickers (symbol, last_fetched) VALUES (?, ?)",
            (symbol, datetime.utcnow().isoformat()),
        )

save_prices = store_prices


def insert_gex_snapshot(
    ticker: str,
    date: str,
    gamma_flip: float | None,
    call_wall: float | None,
    put_wall: float | None,
    spot: float | None,
):
    """Insert-or-ignore one daily GEX snapshot row. Idempotent by design --
    called multiple times a day by app.py's options-cache warmer."""
    with get_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO gex_snapshots "
            "(ticker, date, gamma_flip, call_wall, put_wall, spot, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (ticker.upper(), date, gamma_flip, call_wall, put_wall, spot,
             datetime.utcnow().isoformat()),
        )


def insert_short_interest_snapshot(
    ticker: str,
    date: str,
    short_pct_float: float | None = None,
    short_ratio: float | None = None,
    shares_short: int | None = None,
    shares_float: int | None = None,
):
    """Insert-or-ignore one daily Short Interest snapshot row. Idempotent by design."""
    with get_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO short_interest_snapshots "
            "(ticker, date, short_pct_float, short_ratio, shares_short, shares_float, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                ticker.upper(),
                date,
                short_pct_float,
                short_ratio,
                shares_short,
                shares_float,
                datetime.now(timezone.utc).isoformat(),
            ),
        )


def get_short_interest_history(ticker: str) -> list[dict]:
    """Retrieve historical short interest snapshots for ticker."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT ticker, date, short_pct_float, short_ratio, shares_short, shares_float, created_at "
            "FROM short_interest_snapshots WHERE ticker = ? ORDER BY date ASC",
            (ticker.upper(),),
        ).fetchall()
    return [dict(r) for r in rows]
