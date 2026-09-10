"""SQLite snapshot persistence and historical time-series storage for options analytics."""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def get_db_path() -> str:
    return os.environ.get("DB_PATH", "stocks.db")


def init_options_db(db_path: Optional[str] = None):
    """Ensure options_iv_history table exists in SQLite database."""
    path = db_path or get_db_path()
    # Ensure parent directory exists
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    conn = sqlite3.connect(path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS options_iv_history (
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                date TEXT NOT NULL,
                spot REAL,
                atm_iv REAL,
                hv30 REAL,
                vrp REAL,
                call_wall REAL,
                put_wall REAL,
                gamma_flip REAL,
                total_gex REAL,
                skew_25d REAL,
                convention TEXT DEFAULT 'naive',
                as_of TEXT,
                PRIMARY KEY (symbol, date, convention)
            );
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_iv_hist_sym_date ON options_iv_history(symbol, date);"
        )
        conn.commit()
    finally:
        conn.close()


def record_snapshot(
    symbol: str,
    spot: float,
    atm_iv: float,
    hv30: float,
    vrp: float,
    call_wall: Optional[float],
    put_wall: Optional[float],
    gamma_flip: Optional[float],
    total_gex: float,
    skew_25d: float,
    convention: str = "naive",
    db_path: Optional[str] = None,
) -> bool:
    """Idempotently record an options positioning snapshot into SQLite."""
    path = db_path or get_db_path()
    now_utc = datetime.now(timezone.utc)
    now_ts = now_utc.isoformat()
    today_date = now_utc.strftime("%Y-%m-%d")

    init_options_db(path)
    conn = sqlite3.connect(path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO options_iv_history (
                symbol, timestamp, date, spot, atm_iv, hv30, vrp,
                call_wall, put_wall, gamma_flip, total_gex, skew_25d,
                convention, as_of
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                symbol.upper(),
                now_ts,
                today_date,
                float(spot),
                float(atm_iv),
                float(hv30),
                float(vrp),
                float(call_wall) if call_wall is not None else None,
                float(put_wall) if put_wall is not None else None,
                float(gamma_flip) if gamma_flip is not None else None,
                float(total_gex),
                float(skew_25d),
                str(convention),
                now_ts,
            ),
        )
        conn.commit()
        return True
    except Exception as e:
        print(f"[options-storage] Error recording snapshot for {symbol}: {e}")
        return False
    finally:
        conn.close()


def get_snapshot_history(
    symbol: str,
    convention: str = "naive",
    limit: int = 252,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieve historical daily positioning snapshots for a symbol."""
    path = db_path or get_db_path()
    init_options_db(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT * FROM options_iv_history
            WHERE symbol = ? AND convention = ?
            ORDER BY date ASC
            LIMIT ?
            """,
            (symbol.upper(), convention, limit),
        )
        rows = cur.fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        print(f"[options-storage] Error fetching history for {symbol}: {e}")
        return []
    finally:
        conn.close()


def get_history_status(symbol: str, db_path: Optional[str] = None) -> Dict[str, Any]:
    """Check number of historical days recorded for cold-start reporting."""
    history = get_snapshot_history(symbol, limit=252, db_path=db_path)
    count = len(history)
    return {
        "days_recorded": count,
        "is_accumulating": count < 60,
        "latest": history[-1] if history else None,
    }
