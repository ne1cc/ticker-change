"""Radar: snapshot-published universe screens (drift / value / squeeze).

Context is loaded in one stable read over SQLite (prices + warmed payload
families, zero network); contract-driven builders assemble rows per
radar_contracts.METRIC_KEYS_BY_TAB; validate_snapshot scrubs NaN/inf and
checks row contracts; publish_snapshot stores atomically (pointer write
last) and prunes to the last two snapshots. Requests read
current_snapshot() and never build.

Tunables are persisted display thresholds edited inline on /radar; they are
not secrets, so /api/radar/settings follows the /api/active-tickers auth
posture: Bearer-checked only when WARM_CACHE_TOKEN is configured.
"""
from __future__ import annotations

from datetime import date, datetime

import math

import pandas as pd  # noqa: E402

import db
import earnings  # noqa: E402
import event_study  # noqa: E402
import microstructure  # noqa: E402
import momentum_engine  # noqa: E402
import peers  # noqa: E402
import radar_contracts  # noqa: E402

TUNABLES: dict[str, dict] = {
    "radar_peer_cap": {"default": 8, "min": 3, "max": 20, "type": int},
    "radar_liquidity_floor_usd": {
        "default": 5_000_000.0, "min": 100_000.0,
        "max": 100_000_000.0, "type": float},
    "radar_min_peers": {"default": 3, "min": 2, "max": 10, "type": int},
}


def get_tunable(key: str):
    """Persisted tunable value; default when unset or unparseable; clamped."""
    spec = TUNABLES.get(key)
    if spec is None:
        raise KeyError(key)
    try:
        val = spec["type"](db.get_setting(key, ""))
    except (TypeError, ValueError):
        return spec["default"]
    if isinstance(val, float) and not math.isfinite(val):
        return spec["default"]
    return spec["type"](min(max(val, spec["min"]), spec["max"]))


def clamp_tunable(key: str, value):
    """Coerce, reject non-finite, and clamp; no persistence."""
    spec = TUNABLES.get(key)
    if spec is None:
        raise ValueError(f"unknown tunable: {key}")
    val = spec["type"](value)
    if isinstance(val, float) and not math.isfinite(val):
        raise ValueError(f"non-finite value for {key}: {val}")
    return spec["type"](min(max(val, spec["min"]), spec["max"]))


def set_tunable(key: str, value):
    """Clamp, persist, and return the stored value. ValueError on unknown key."""
    val = clamp_tunable(key, value)
    db.set_setting(key, str(val))
    return val


SNAPSHOT_PROVIDER = "radar_snapshot"
SOURCE_TTLS = {"earnings": earnings.EARNINGS_TTL_HOURS,
               "fundamentals": peers.FUNDAMENTALS_TTL_HOURS,
               "short": 24, "prices": 1}
VALUE_ROW_CAP = 50


def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def _older(a: str | None, b: str | None) -> str | None:
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


def _price_age(symbols: list[str]) -> str | None:
    """Oldest per-symbol price refresh in the universe (worst-case staleness)."""
    oldest = None
    syms = sorted(set(symbols))
    for i in range(0, len(syms), 500):
        chunk = syms[i:i + 500]
        q = ",".join("?" * len(chunk))
        with db.get_conn() as conn:
            row = conn.execute(
                f"SELECT MIN(last_fetched) m FROM tickers WHERE symbol IN ({q})",
                chunk).fetchone()
        if row and row["m"]:
            oldest = row["m"] if oldest is None else min(oldest, row["m"])
    return oldest


def _fundamentals_age() -> str | None:
    with db.get_conn() as conn:
        row = conn.execute(
            "SELECT MIN(fetched_at) m FROM api_cache "
            "WHERE provider = 'yfinance' AND key LIKE 'fundamentals:%'").fetchone()
    return row["m"] if row else None


def _f(v):
    """Coerce to a plain JSON-safe float, or None when not numeric."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_context() -> dict:
    """One stable read of every radar input. Zero network."""
    symbols = universe_symbols(min_bars=200)
    wanted = symbols + radar_contracts.SECTOR_ETFS + ["SPY"]
    prices = db.get_prices_batch(sorted(set(wanted)))
    ctx = {"symbols": symbols, "prices": prices, "built_at": now_iso(),
           "fundamentals": {}, "earnings": {}, "short": {},
           "source_as_of": {}}
    fund = db.cache_scan("yfinance", "fundamentals:")
    earnings_map, earnings_age = {}, None
    for sym in symbols:
        payload, fetched = db.cache_get_with_age(
            "yfinance", f"earnings:{sym}", SOURCE_TTLS["earnings"])
        if payload:
            earnings_map[sym] = payload
            earnings_age = _older(earnings_age, fetched)
    short_map, short_age = {}, None
    for sym, payload in db.cache_scan("yfinance", "short:").items():
        short_map[sym] = payload
        _, fetched = db.cache_get_with_age("yfinance", f"short:{sym}",
                                           SOURCE_TTLS["short"])
        short_age = _older(short_age, fetched)
    ctx["fundamentals"] = {s: fund.get(s) for s in symbols if fund.get(s)}
    ctx["earnings"] = earnings_map
    ctx["short"] = short_map
    ctx["source_as_of"] = {"prices": _price_age(symbols),
                           "earnings": earnings_age,
                           "fundamentals": _fundamentals_age(),
                           "short": short_age}
    return ctx


def universe_symbols(min_bars: int = 253) -> list[str]:
    """Symbols with enough cached price history to score."""
    try:
        with db.get_conn() as conn:
            rows = conn.execute(
                "SELECT symbol FROM daily_prices GROUP BY symbol HAVING COUNT(*) >= ?",
                (min_bars,),
            ).fetchall()
        return sorted(r["symbol"] for r in rows)
    except Exception:
        return []


def _clean(obj):
    """Recursive NaN/inf -> None scrub (rows must be JSON-safe before store)."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def _next_print(events) -> str | None:
    dates = [e.get("date") for e in events or []
             if e.get("is_upcoming") and e.get("date")]
    return min(dates) if dates else None


def _sector_return(ctx: dict, sector, t0, horizon: int):
    """Sector-ETF return over the same session window as a drift metric."""
    if not sector or t0 is None:
        return None
    etf = (ctx.get("prices") or {}).get(
        radar_contracts.SECTOR_ETF_MAP.get(sector, ""))
    if etf is None or getattr(etf, "empty", True):
        return None
    try:
        idx = etf.index
        start = int(idx.get_loc(t0)) if t0 in idx else int(idx.searchsorted(t0))
        end = start + horizon
        if start >= len(idx) or end >= len(idx):
            return None
        close = etf["close"] if "close" in etf.columns else etf["Close"]
        base, final = float(close.iloc[start]), float(close.iloc[end])
        return round(final / base - 1, 6) if base else None
    except Exception:
        return None


def _drift_rows(ctx: dict) -> tuple[list[dict], list[dict]]:
    """Latest past print per earnings-carrying symbol, per the drift contract.

    Reads only ctx (prices + payloads). Rows rank by remaining 20-session
    drift window; symbols whose prices/alignment are missing land in the
    exclusion ledger instead.
    """
    fundamentals = ctx.get("fundamentals") or {}
    spy = (ctx.get("prices") or {}).get("SPY")
    rows: list[dict] = []
    excluded: list[dict] = []
    for symbol, events in sorted((ctx.get("earnings") or {}).items()):
        past = [e for e in events or [] if not e.get("is_upcoming")]
        if not past:
            continue
        try:
            last = max(past, key=lambda e: e["date"])
        except (KeyError, TypeError, ValueError):
            excluded.append({"symbol": symbol, "reason": "bad-date"})
            continue
        stock_df = (ctx.get("prices") or {}).get(symbol)
        if stock_df is None or getattr(stock_df, "empty", True):
            excluded.append({"symbol": symbol, "reason": "no-prices"})
            continue
        m = earnings.event_metrics(stock_df, last["date"])
        if m.get("sessions_since") is None:
            excluded.append({"symbol": symbol, "reason": "no-aligned-session"})
            continue
        payload = fundamentals.get(symbol) or {}
        est = _f(last.get("eps_estimate"))
        surprise_nm = est is None or abs(est) < 0.05
        hist = earnings.own_event_history(stock_df, events)
        res = None
        if spy is not None:
            try:
                res = event_study.run_event_study(
                    stock_df, spy, last["date"],
                    event_type="EARNINGS", ticker=symbol)
            except Exception:
                res = None
        sessions_since = int(m["sessions_since"])
        rows.append({
            "symbol": symbol,
            "sector": payload.get("Sector"),
            "mkt_cap": _f(payload.get("Market Cap")),
            "print_date": last["date"],
            "alignment": m.get("alignment"),
            "sessions_since": sessions_since,
            "sessions_to_20d": max(0, 20 - sessions_since),
            "eps_estimate": est,
            "eps_actual": _f(last.get("eps_actual")),
            "surprise_pct": None if surprise_nm else _f(last.get("surprise_pct")),
            "surprise_nm": surprise_nm,
            "gap_pct": _f(m.get("gap_pct")),
            "reaction_ret": _f(m.get("reaction_ret")),
            "runup_5d": _f(m.get("runup_5d")),
            "rel_volume": _f(m.get("rel_volume")),
            "drift_5d": _f(m.get("drift_5d")),
            "drift_10d": _f(m.get("drift_10d")),
            "drift_20d": _f(m.get("drift_20d")),
            "drift_to_date": _f(m.get("drift_to_date")),
            "drift_to_date_sessions": m.get("drift_to_date_sessions"),
            "excess_10d_sector": (_sector_return(ctx, payload.get("Sector"),
                                                 m.get("t0"), 10)
                                  if m.get("drift_10d") is not None else None),
            "car_30": round(float(res.car), 4) if res else None,
            "car_significant": bool(res.is_significant_95) if res else False,
            "hist_beat_median": _f(hist.get("beat", {}).get("median")),
            "hist_miss_median": _f(hist.get("miss", {}).get("median")),
            "next_print": _next_print(events),
        })
    rows.sort(key=lambda r: (-r["sessions_to_20d"], r["symbol"]))
    return rows, excluded


def _upcoming_strip(ctx: dict) -> list[dict]:
    """Prints scheduled 0-7 calendar days out, soonest first."""
    today = date.today()
    out: list[dict] = []
    for symbol, events in sorted((ctx.get("earnings") or {}).items()):
        for e in events or []:
            if not e.get("is_upcoming") or not e.get("date"):
                continue
            try:
                days_until = (pd.to_datetime(e["date"]).date() - today).days
            except (TypeError, ValueError):
                continue
            if not 0 <= days_until <= 7:
                continue
            out.append({"symbol": symbol, "date": e["date"],
                        "days_until": days_until,
                        "eps_estimate": _f(e.get("eps_estimate"))})
    out.sort(key=lambda u: (u["days_until"], u["symbol"]))
    return out


_VALUE_MULTIPLES = {"fwd_pe": "Forward P/E", "trail_pe": "Trailing P/E",
                    "ps": "Price / Sales", "peg": "PEG Ratio",
                    "ev_ebitda": "EV / EBITDA"}
_VALUE_SCORE_WEIGHTS = (("cheapness", 0.5), ("growth", 0.25),
                        ("momentum", 0.25))


def _pool_pct(pool: list, x, invert: bool):
    if x is None:
        return None
    return peers.percentile_rank(pool, x, invert=invert)


def _value_rows(ctx: dict) -> tuple[list[dict], list[dict]]:
    """Contract-driven value screen over warmed fundamentals payloads.

    Percentile pools hold valid values only (NM multiples excluded via
    peers.nm_state); peer pools need radar_min_peers valid scores, else rows
    carry the insufficient-peers state instead of a percentile.
    """
    min_peers = int(get_tunable("radar_min_peers"))
    prices = ctx.get("prices") or {}
    shorts = ctx.get("short") or {}
    rows: list[dict] = []
    excluded: list[dict] = []
    for symbol in ctx.get("symbols") or []:
        payload = (ctx.get("fundamentals") or {}).get(symbol)
        if not payload:
            excluded.append({"symbol": symbol, "reason": "no-fundamentals"})
            continue
        px = prices.get(symbol)
        if px is None or getattr(px, "empty", True) or len(px) < 20:
            excluded.append({"symbol": symbol, "reason": "insufficient-history"})
            continue
        ms = None
        try:
            ms = momentum_engine.score_series(px["close"], symbol=symbol)
        except Exception:
            ms = None
        vals = {k: _f(payload.get(src)) for k, src in _VALUE_MULTIPLES.items()}
        nm = {k: peers.nm_state(k, v) for k, v in vals.items()}
        fcf = _f(payload.get("Free Cash Flow"))
        mkt_cap = _f(payload.get("Market Cap"))
        fcf_yield = (round(100.0 * fcf / mkt_cap, 2)
                     if fcf is not None and mkt_cap else None)
        rows.append({
            "symbol": symbol,
            "sector": payload.get("Sector"),
            "industry": payload.get("Industry"),
            "mkt_cap": mkt_cap,
            "fwd_pe": vals["fwd_pe"],
            "trail_pe": vals["trail_pe"],
            "ps": vals["ps"],
            "peg": vals["peg"],
            "ev_ebitda": vals["ev_ebitda"],
            "div_yield": _f(payload.get("Dividend Yield")),
            "fcf_yield": fcf_yield,
            "rev_growth": _f(payload.get("Revenue Growth")),
            "eps_growth": _f(payload.get("Earnings Growth")),
            "mom_12_1": _f(round(ms.mom_12_1, 4)) if ms else None,
            "mom_6m": _f(round(ms.mom_6m, 4)) if ms else None,
            "mom_3m": _f(round(ms.mom_3m, 4)) if ms else None,
            "mom_1m": _f(round(ms.mom_1m, 4)) if ms else None,
            "ann_vol": _f(round(ms.ann_vol_1y, 4)) if ms else None,
            "short_pct_float": _f((shorts.get(symbol) or {}).get("short_pct_float")),
        })
        for k in _VALUE_MULTIPLES:
            rows[-1][f"{k}_nm"] = nm[k]

    def _valid_pool(members, key, inverted: bool) -> list:
        if inverted:
            return [m[key] for m in members
                    if m[key] is not None and not peers.nm_state(key, m[key])]
        return [m[key] for m in members if m[key] is not None]

    industry_groups: dict[str, list[dict]] = {}
    sector_groups: dict[str, list[dict]] = {}
    for row in rows:
        industry_groups.setdefault(
            row["industry"] or row["sector"] or "Unknown", []).append(row)
        sector_groups.setdefault(row["sector"] or "Unknown", []).append(row)

    for row in rows:
        for k in _VALUE_MULTIPLES:
            pool = _valid_pool(sector_groups.get(row["sector"] or "Unknown", []),
                               k, inverted=True)
            row[f"{k}_pct"] = None if row[f"{k}_nm"] else _pool_pct(
                pool, row[k], invert=True)
        for k, out in (("rev_growth", "rev_growth_pct"),
                       ("eps_growth", "eps_growth_pct")):
            pool = _valid_pool(sector_groups.get(row["sector"] or "Unknown", []),
                               k, inverted=False)
            row[out] = _pool_pct(pool, row[k], invert=False)
        mom_pool = _valid_pool(sector_groups.get(row["sector"] or "Unknown", []),
                               "mom_12_1", inverted=False)
        row["mom_pct"] = _pool_pct(mom_pool, row["mom_12_1"], invert=False)

        cheapness = [row[f"{k}_pct"] for k in ("fwd_pe", "ps")]
        growth = [row["rev_growth_pct"], row["eps_growth_pct"]]
        components = {"cheapness": ([v for v in cheapness if v is not None] or None),
                      "growth": ([v for v in growth if v is not None] or None),
                      "momentum": ([row["mom_pct"]] if row["mom_pct"] is not None
                                   else None)}
        weighted = [(w, sum(v) / len(v))
                    for name, w in _VALUE_SCORE_WEIGHTS
                    for v in [components[name]] if v is not None]
        row["value_score"] = (round(sum(w * v for w, v in weighted)
                                    / sum(w for w, _ in weighted), 1)
                              if weighted else None)
        row["_nm"] = any(row[f"{k}_nm"] for k in _VALUE_MULTIPLES)
        available = sum(1 for v in (vals_ok(row, "fwd_pe"), vals_ok(row, "ps"),
                                    row["rev_growth"], row["eps_growth"],
                                    row["mom_12_1"]) if v is not None)
        row["input_coverage"] = round(available / 5.0, 2)
        row["low_coverage"] = available / 5.0 < 0.5

    for row in rows:
        ind_pool = [m["value_score"] for m in industry_groups.get(
            row["industry"] or row["sector"] or "Unknown", [])
            if m["value_score"] is not None]
        sec_pool = [m["value_score"] for m in sector_groups.get(
            row["sector"] or "Unknown", []) if m["value_score"] is not None]
        pool = sec_pool if len(sec_pool) > len(ind_pool) else ind_pool
        row["peer_count"] = len(pool)
        row["_state"] = None if len(pool) >= min_peers else "insufficient-peers"
        row["peer_percentile"] = (
            peers.percentile_rank(pool, row["value_score"])
            if row["value_score"] is not None and len(pool) >= min_peers
            else None)
        row["quadrant"] = peers.quadrant(row["peer_percentile"], row["mom_12_1"])

    rows.sort(key=lambda r: ((-r["value_score"]) if r["value_score"] is not None
                             else float("inf"), r["symbol"]))
    return rows[:VALUE_ROW_CAP], excluded


def vals_ok(row: dict, key: str):
    """Raw multiple value when present and meaningful, else None."""
    v = row.get(key)
    if v is None or peers.nm_state(key, v):
        return None
    return v


def _squeeze_rows(ctx: dict) -> tuple[list[dict], list[dict]]:
    """Squeeze screen over short payloads with both SI fields present.

    Rows carry vendor + calculated days to cover, report/report SI change,
    52w distance, 5d/20d session moves, median dollar volume, and the
    crowding/activation/tradability components behind the composite score.
    """
    floor = float(get_tunable("radar_liquidity_floor_usd"))
    fundamentals = ctx.get("fundamentals") or {}
    rows: list[dict] = []
    excluded: list[dict] = []
    for symbol, si in sorted((ctx.get("short") or {}).items()):
        payload = fundamentals.get(symbol) or {}
        if not si or si.get("short_pct_float") is None \
                or si.get("days_to_cover") is None:
            excluded.append({"symbol": symbol, "reason": "assumed-si"})
            continue
        px = (ctx.get("prices") or {}).get(symbol)
        if px is None or getattr(px, "empty", True) or len(px) < 21:
            excluded.append({"symbol": symbol, "reason": "no-prices"})
            continue
        try:
            recent = px.tail(20)
            dollar_volume = float((recent["close"] * recent["volume"]).median())
        except Exception:
            excluded.append({"symbol": symbol, "reason": "no-prices"})
            continue
        if dollar_volume < floor:
            excluded.append({"symbol": symbol, "reason": "below-floor"})
            continue
        pct = _f(si.get("short_pct_float"))
        dtc_vendor = _f(si.get("days_to_cover"))
        score, level = microstructure.compute_squeeze_risk_index(
            pct * 100.0, dtc_vendor)
        close = px["close"]
        avg_vol = float(recent["volume"].mean())
        shares_short = _f(si.get("shares_short"))
        dtc_calc = (round(shares_short / avg_vol, 2)
                    if shares_short is not None and avg_vol else None)
        prior = _f(si.get("shares_short_prior_month"))
        si_change = (round((shares_short - prior) / abs(prior) * 100.0, 1)
                     if shares_short is not None and prior else None)
        high_52 = float(close.tail(252).max())
        price = float(close.iloc[-1])
        si_settlement = si.get("as_of")
        si_age_days = None
        try:
            if isinstance(si_settlement, (int, float)) \
                    and not isinstance(si_settlement, bool):
                si_age_days = (date.today()
                               - date.fromtimestamp(si_settlement)).days
            elif si_settlement:
                si_age_days = (date.today()
                               - pd.to_datetime(si_settlement).date()).days
        except (TypeError, ValueError, OSError, OverflowError):
            si_age_days = None
        rows.append({
            "symbol": symbol,
            "sector": payload.get("Sector"),
            "mkt_cap": _f(payload.get("Market Cap")),
            "short_pct_float": pct,
            "si_settlement": si_settlement,
            "si_age_days": si_age_days,
            "shares_short": shares_short,
            "prior_shares_short": prior,
            "si_change": si_change,
            "dtc_vendor": dtc_vendor,
            "dtc_calc": dtc_calc,
            "squeeze_score": score,
            "squeeze_level": level,
            "dist_52w_high": round((high_52 - price) / high_52, 6)
            if high_52 else None,
            "price_5d": (round(price / float(close.iloc[-6]) - 1, 6)
                         if float(close.iloc[-6] or 0) else None),
            "price_20d": (round(price / float(close.iloc[-21]) - 1, 6)
                          if float(close.iloc[-21] or 0) else None),
            "dollar_volume": round(dollar_volume),
            "crowding_score": round(min(pct * 100.0 / 25.0, 1.0) * 40.0, 1),
            "activation_score": round(min(dtc_vendor / 7.0, 1.0) * 30.0, 1),
            "tradability_score": round(
                min(dollar_volume / 25_000_000.0, 1.0) * 30.0, 1),
        })
    rows.sort(key=lambda r: (-r["squeeze_score"], r["symbol"]))
    return rows, excluded


def _snapshot_id(context: dict) -> str:
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M")
    return f"{stamp}-{len(context.get('symbols') or [])}"


def _warmed_count(context: dict) -> int:
    families = (context.get("fundamentals") or {}, context.get("earnings") or {},
                context.get("short") or {})
    return len(set(families[0]) | set(families[1]) | set(families[2]))


def build_snapshot(context: dict) -> dict:
    """Contract-driven snapshot from a loaded context. CPU only, no network."""
    drift_rows, drift_excluded = _drift_rows(context)
    upcoming = _upcoming_strip(context)
    value_rows, value_excluded = _value_rows(context)
    squeeze_rows, squeeze_excluded = _squeeze_rows(context)
    counts: dict[str, int] = {}
    for entry in drift_excluded + value_excluded + squeeze_excluded:
        counts[entry["reason"]] = counts.get(entry["reason"], 0) + 1
    snap = {
        "meta": {
            "snapshot_id": _snapshot_id(context),
            "built_at": context.get("built_at"),
            "source_as_of": dict(context.get("source_as_of") or {}),
            "universe_count": len(context.get("symbols") or []),
            "warmed_count": _warmed_count(context),
            "scored_count": (len(drift_rows) + len(value_rows)
                             + len(squeeze_rows)),
            "excluded_count_by_reason": counts,
            "metric_contract_version": radar_contracts.METRIC_CONTRACT_VERSION,
            "value_score_version": radar_contracts.VALUE_SCORE_VERSION,
            "validation_status": "pending",
        },
        "drift": {"rows": drift_rows, "upcoming": upcoming,
                  "excluded": drift_excluded},
        "value": {"rows": value_rows, "excluded": value_excluded},
        "squeeze": {"rows": squeeze_rows, "excluded": squeeze_excluded},
    }
    validate_snapshot(snap)
    return snap


def validate_snapshot(snap: dict) -> tuple[bool, list[str]]:
    """Scrub NaN/inf -> None in place, check row contracts + counts."""
    problems: list[str] = []
    try:
        for tab, keys in radar_contracts.METRIC_KEYS_BY_TAB.items():
            section = snap.get(tab)
            if not isinstance(section, dict) or "rows" not in section \
                    or "excluded" not in section:
                problems.append(f"{tab}: missing section")
                continue
            for row in section["rows"]:
                missing = [k for k in keys if k not in row]
                if missing:
                    problems.append(
                        f"{tab}/{row.get('symbol')}: missing {','.join(missing)}")
        if not isinstance(snap.get("drift", {}).get("upcoming"), list):
            problems.append("drift: missing upcoming")
        total_rows = sum(len(snap.get(t, {}).get("rows", []))
                         for t in radar_contracts.METRIC_KEYS_BY_TAB)
        if snap.get("meta", {}).get("scored_count") != total_rows:
            problems.append("meta: scored_count mismatch")
        recomputed: dict[str, int] = {}
        for t in radar_contracts.METRIC_KEYS_BY_TAB:
            for entry in snap.get(t, {}).get("excluded", []):
                recomputed[entry["reason"]] = recomputed.get(entry["reason"], 0) + 1
        if snap.get("meta", {}).get("excluded_count_by_reason") != recomputed:
            problems.append("meta: excluded_count_by_reason mismatch")
        cleaned = _clean(snap)
        snap.clear()
        snap.update(cleaned)
    except Exception as e:
        problems.append(f"validator error: {e}")
    if problems:
        snap.setdefault("meta", {})["validation_status"] = \
            "failed: " + "; ".join(problems[:5])
        return False, problems
    snap.setdefault("meta", {})["validation_status"] = "ok"
    return True, []


def publish_snapshot(context: dict) -> dict | None:
    """Build -> validate -> store -> move pointer (last) -> prune. On any
    failure the pointer stays untouched and the last-good snapshot serves."""
    started = now_iso()
    try:
        snap = build_snapshot(context)
        ok, problems = validate_snapshot(snap)
        if not ok:
            print(f"[radar] snapshot validation failed: {problems}")
            return None
        snap["meta"]["build_started_at"] = started
        snap["meta"]["build_completed_at"] = now_iso()
        snap["meta"]["published_at"] = started
        snap_id = snap["meta"]["snapshot_id"]
        db.cache_set(SNAPSHOT_PROVIDER, f"snapshot:{snap_id}", snap)
        db.cache_set(SNAPSHOT_PROVIDER, "current", {"snapshot_id": snap_id})
        _prune_snapshots(keep=2)
        return snap
    except Exception as e:
        print(f"[radar] publish failed (pointer untouched): {e}")
        return None


def current_snapshot() -> dict | None:
    """The published snapshot payload, or None when absent/invalid."""
    pointer = db.cache_get(SNAPSHOT_PROVIDER, "current", 24 * 7)
    if not isinstance(pointer, dict) or not pointer.get("snapshot_id"):
        return None
    payload = db.cache_get(SNAPSHOT_PROVIDER,
                           f"snapshot:{pointer['snapshot_id']}", 24 * 7)
    return payload if isinstance(payload, dict) else None


def _prune_snapshots(keep: int = 2):
    """Keep only the newest `keep` snapshot payloads; 'current' is a separate
    key and never matches the snapshot: prefix."""
    with db.get_conn() as conn:
        stored = conn.execute(
            "SELECT key FROM api_cache WHERE provider = ? AND key LIKE 'snapshot:%' "
            "ORDER BY fetched_at DESC, rowid DESC", (SNAPSHOT_PROVIDER,)).fetchall()
        for row in stored[keep:]:
            conn.execute(
                "DELETE FROM api_cache WHERE provider = ? AND key = ?",
                (SNAPSHOT_PROVIDER, row["key"]))
