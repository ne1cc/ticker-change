"""SQLite snapshot persistence and historical time-series storage for options analytics.

Part of Epic 4 (issues #124, #125, #126, #130, #131):
- Nightly snapshot of constant-maturity 30d ATM IV, term-structure slope, 25-delta skew, and GEX levels
- Idempotent per (symbol, date, convention)
- Stores source and as-of timestamps
- Computes real IV rank/percentile on stored IV history (accumulating n/252 when below 63 days)
- Computes VRP history with percentile bands and dynamic regime labeling
- Tracks term-structure slope and skew history percentiles
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np


def get_db_path() -> str:
    return os.environ.get("DB_PATH", "stocks.db")


def init_options_db(db_path: Optional[str] = None):
    """Ensure options_iv_history table exists in SQLite database with all required columns."""
    path = db_path or get_db_path()
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
                term_slope REAL DEFAULT 0.0,
                source TEXT DEFAULT 'yfinance',
                convention TEXT DEFAULT 'naive',
                as_of TEXT,
                PRIMARY KEY (symbol, date, convention)
            );
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_iv_hist_sym_date ON options_iv_history(symbol, date);"
        )

        # Defensive schema migrations for existing tables
        cur.execute("PRAGMA table_info(options_iv_history);")
        existing_cols = {row[1] for row in cur.fetchall()}
        if "term_slope" not in existing_cols:
            try:
                cur.execute("ALTER TABLE options_iv_history ADD COLUMN term_slope REAL DEFAULT 0.0;")
            except Exception:
                pass
        if "source" not in existing_cols:
            try:
                cur.execute("ALTER TABLE options_iv_history ADD COLUMN source TEXT DEFAULT 'yfinance';")
            except Exception:
                pass

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
    term_slope: float = 0.0,
    source: str = "yfinance",
    convention: str = "naive",
    as_of: Optional[str] = None,
    db_path: Optional[str] = None,
) -> bool:
    """Idempotently record an options positioning snapshot into SQLite.
    
    Re-running on the same date replaces the row without duplicating records.
    """
    path = db_path or get_db_path()
    now_utc = datetime.now(timezone.utc)
    now_ts = as_of or now_utc.isoformat()
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
                term_slope, source, convention, as_of
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                symbol.upper(),
                now_ts,
                today_date,
                float(spot) if spot is not None else None,
                float(atm_iv) if atm_iv is not None else None,
                float(hv30) if hv30 is not None else None,
                float(vrp) if vrp is not None else None,
                float(call_wall) if call_wall is not None else None,
                float(put_wall) if put_wall is not None else None,
                float(gamma_flip) if gamma_flip is not None else None,
                float(total_gex) if total_gex is not None else 0.0,
                float(skew_25d) if skew_25d is not None else 1.0,
                float(term_slope) if term_slope is not None else 0.0,
                str(source),
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


def get_history_count(symbol: str, convention: str = "naive", db_path: Optional[str] = None) -> int:
    """Return count of historical days recorded for a symbol."""
    path = db_path or get_db_path()
    init_options_db(path)
    conn = sqlite3.connect(path)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM options_iv_history WHERE symbol = ? AND convention = ?",
            (symbol.upper(), convention),
        )
        row = cur.fetchone()
        return int(row[0]) if row else 0
    except Exception as e:
        print(f"[options-storage] Error counting history for {symbol}: {e}")
        return 0
    finally:
        conn.close()


def get_history_status(
    symbol: str,
    target: int = 252,
    convention: str = "naive",
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Check number of historical days recorded for cold-start reporting."""
    count = get_history_count(symbol, convention=convention, db_path=db_path)
    history = get_snapshot_history(symbol, convention=convention, limit=target, db_path=db_path)
    is_acc = count < 63
    return {
        "days_recorded": count,
        "is_accumulating": is_acc,
        "status": f"accumulating ({count}/{target})" if is_acc else "ready",
        "latest": history[-1] if history else None,
    }


def compute_iv_rank_and_percentile(
    symbol: str,
    current_iv: Optional[float],
    min_obs: int = 63,
    convention: str = "naive",
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute 52W IV Rank and Percentile from stored IV history.
    
    If fewer than min_obs (default 63) exist, returns accumulating state and
    never substitutes a rolling realized-vol proxy.
    """
    history = get_snapshot_history(symbol, convention=convention, limit=252, db_path=db_path)
    iv_vals = [float(r["atm_iv"]) for r in history if r.get("atm_iv") is not None and float(r["atm_iv"]) > 0]
    n = len(iv_vals)

    if n < min_obs or current_iv is None or current_iv <= 0:
        return {
            "available": False,
            "status": f"accumulating ({n}/252)",
            "n_obs": n,
            "iv_rank": None,
            "iv_percentile": None,
            "min_iv": None,
            "max_iv": None,
        }

    min_iv = float(min(iv_vals))
    max_iv = float(max(iv_vals))
    iv_range = max_iv - min_iv

    if iv_range > 0:
        rank = ((current_iv - min_iv) / iv_range) * 100.0
        rank = max(0.0, min(100.0, rank))
    else:
        rank = 50.0

    percentile = float(sum(1 for v in iv_vals if v < current_iv) / n * 100.0)

    return {
        "available": True,
        "status": "ready",
        "n_obs": n,
        "iv_rank": round(rank, 1),
        "iv_percentile": round(percentile, 1),
        "min_iv": round(min_iv, 2),
        "max_iv": round(max_iv, 2),
    }


def get_vrp_history_with_bands(
    symbol: str,
    min_obs: int = 21,
    convention: str = "naive",
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Calculate VRP history (IV30 - RV21) with historical percentile bands.
    
    Provides percentile-based regime classification (high/low/normal) rather than
    arbitrary fixed spread cutoffs.
    """
    history = get_snapshot_history(symbol, convention=convention, limit=252, db_path=db_path)
    valid_rows = [r for r in history if r.get("vrp") is not None and r.get("date")]
    n = len(valid_rows)

    if n < min_obs:
        return {
            "available": False,
            "status": f"accumulating ({n}/252)",
            "n_obs": n,
            "vrp_history": [],
            "regime": "accumulating",
            "regime_label": f"accumulating ({n}/252)",
        }

    dates = [str(r["date"]) for r in valid_rows]
    vrps = [float(r["vrp"]) for r in valid_rows]

    p10 = float(np.percentile(vrps, 10))
    p25 = float(np.percentile(vrps, 25))
    p50 = float(np.percentile(vrps, 50))
    p75 = float(np.percentile(vrps, 75))
    p90 = float(np.percentile(vrps, 90))

    current_vrp = vrps[-1]
    pct_rank = float(sum(1 for v in vrps if v < current_vrp) / n * 100.0)

    if pct_rank >= 80.0:
        regime = "high"
        regime_label = f"High VRP (Expensive Vol \u00b7 {pct_rank:.0f}th percentile)"
    elif pct_rank <= 20.0:
        regime = "low"
        regime_label = f"Low / Inverted VRP (Cheap Vol \u00b7 {pct_rank:.0f}th percentile)"
    else:
        regime = "normal"
        regime_label = f"Normal VRP ({pct_rank:.0f}th percentile)"

    return {
        "available": True,
        "status": "ready",
        "n_obs": n,
        "dates": dates,
        "vrp_series": [round(v, 2) for v in vrps],
        "current_vrp": round(current_vrp, 2),
        "current_percentile": round(pct_rank, 1),
        "regime": regime,
        "regime_label": regime_label,
        "p10": round(p10, 2),
        "p25": round(p25, 2),
        "median": round(p50, 2),
        "p75": round(p75, 2),
        "p90": round(p90, 2),
    }


def get_term_structure_and_skew_history(
    symbol: str,
    current_slope: Optional[float],
    current_skew: Optional[float],
    min_obs: int = 63,
    convention: str = "naive",
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Calculate historical percentiles for term structure slope and 25-delta skew."""
    history = get_snapshot_history(symbol, convention=convention, limit=252, db_path=db_path)
    slopes = [float(r["term_slope"]) for r in history if r.get("term_slope") is not None]
    skews = [float(r["skew_25d"]) for r in history if r.get("skew_25d") is not None]
    n = min(len(slopes), len(skews))

    if n < min_obs or current_slope is None or current_skew is None:
        return {
            "available": False,
            "status": f"accumulating ({n}/252)",
            "n_obs": n,
            "slope_percentile": None,
            "skew_percentile": None,
        }

    slope_pct = float(sum(1 for s in slopes if s < current_slope) / len(slopes) * 100.0)
    skew_pct = float(sum(1 for k in skews if k < current_skew) / len(skews) * 100.0)

    return {
        "available": True,
        "status": "ready",
        "n_obs": n,
        "slope_percentile": round(slope_pct, 1),
        "skew_percentile": round(skew_pct, 1),
    }
