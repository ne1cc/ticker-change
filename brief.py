"""Daily brief: context bundle in, contract-validated brief dict out.

load_brief_context is one stable read of every brief input for one ticker
from SQLite (prices via the daily_prices table, earnings/fundamentals/short
payloads via the api_cache families radar warms) — zero network. build_brief
turns a context into delta-first sections: what changed in the last
completed session, with significance measured against the ticker's own
trailing completed 20-session return window. Static profile data appears
only as the demoted context section. Session math is earnings.event_metrics,
shared with radar; no re-implementation. Publication is not this module's
job (snapshot_store belongs to the publisher).
"""
from __future__ import annotations

import math

from datetime import date, datetime

import pandas as pd  # noqa: E402

import brief_contracts  # noqa: E402
import db  # noqa: E402
import earnings  # noqa: E402
import peers  # noqa: E402

SOURCE_TTLS = {"prices": db.FRESHNESS_HOURS,
               "earnings": earnings.EARNINGS_TTL_HOURS,
               "fundamentals": peers.FUNDAMENTALS_TTL_HOURS,
               "short": 24}

SIGNIFICANCE_Z = {"notable": 2.0, "elevated": 1.0}

NM_ESTIMATE_FLOOR = 0.05


def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def _f(v):
    """Coerce to a plain float, or None when not numeric."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_brief_context(ticker: str) -> dict:
    """One stable read of every brief input for one ticker. Zero network."""
    ticker = (ticker or "").upper()
    prices_as_of = _price_as_of(ticker)
    fund, fund_age = db.cache_get_with_age(
        "yfinance", f"fundamentals:{ticker}", SOURCE_TTLS["fundamentals"])
    events, earnings_age = db.cache_get_with_age(
        "yfinance", f"earnings:{ticker}", SOURCE_TTLS["earnings"])
    short, short_age = db.cache_get_with_age(
        "yfinance", f"short:{ticker}", SOURCE_TTLS["short"])
    return {
        "ticker": ticker,
        "prices": db.get_prices(ticker),
        "earnings": events,
        "fundamentals": fund,
        "short": short,
        "built_at": now_iso(),
        "source_as_of": {"prices": prices_as_of, "earnings": earnings_age,
                         "fundamentals": fund_age, "short": short_age},
    }


def _price_as_of(ticker: str) -> str | None:
    """The ticker's own last price refresh — not a global cache minimum."""
    with db.get_conn() as conn:
        row = conn.execute(
            "SELECT last_fetched FROM tickers WHERE symbol = ?", (ticker,)
        ).fetchone()
    return row["last_fetched"] if row else None


def _significance(z: float) -> str:
    if z >= SIGNIFICANCE_Z["notable"]:
        return "notable"
    if z >= SIGNIFICANCE_Z["elevated"]:
        return "elevated"
    return "within noise"


def _price_section(context: dict) -> dict:
    """The last completed session and whether the move mattered."""
    out = {k: None for k in brief_contracts.METRIC_KEYS_BY_SECTION["price"]}
    out["state"] = "missing"
    px = context.get("prices")
    if px is None or getattr(px, "empty", True):
        return out
    close = px["close"] if "close" in px.columns else px["Close"]
    if len(close) < 2:
        return out
    rets = close.pct_change()
    last_ret = float(rets.iloc[-1])
    if not math.isfinite(last_ret):
        return out
    out["state"] = "ok"
    out["session_date"] = px.index[-1].date().isoformat()
    out["session_return"] = round(last_ret, 6)
    prev = float(close.iloc[-2])
    if "open" in px.columns and prev:
        out["gap_pct"] = round(float(px["open"].iloc[-1]) / prev - 1, 6)
    n = len(close)
    if "volume" in px.columns and n >= 21:
        base = float(px["volume"].iloc[-21:-1].mean())
        if base:
            out["volume_ratio"] = round(float(px["volume"].iloc[-1]) / base, 2)
    window = rets.iloc[max(1, n - 21):n - 1].dropna()
    out["vol_window_sessions"] = int(len(window))
    if len(window) < 20:
        out["significance_state"] = "window incomplete"
        return out
    mu, sd = float(window.mean()), float(window.std(ddof=1))
    if not sd or not math.isfinite(sd) or sd <= 0 \
            or not math.isfinite(mu) or not math.isfinite(last_ret):
        out["significance_state"] = "degenerate window"
        return out
    z = abs((last_ret - mu) / sd)
    out["z_score"] = round(z, 2)
    out["significance"] = _significance(z)
    out["significance_state"] = "ok"
    return out


def _earnings_section(context: dict) -> dict:
    """Upcoming print plus the most recent print, session-aligned."""
    out = {k: None for k in brief_contracts.METRIC_KEYS_BY_SECTION["earnings"]}
    out["state"] = "missing"
    events = context.get("earnings")
    if not events:
        return out
    out["state"] = "ok"
    upcoming = sorted(e["date"] for e in events
                      if e.get("is_upcoming") and e.get("date"))
    if upcoming:
        out["next_print_date"] = upcoming[0]
        try:
            out["next_print_in_days"] = (
                pd.to_datetime(upcoming[0]).date() - date.today()).days
        except (TypeError, ValueError):
            pass
    past = [e for e in events if not e.get("is_upcoming") and e.get("date")]
    if not past:
        return out
    try:
        last = max(past, key=lambda e: e["date"])
    except (TypeError, ValueError):
        return out
    out["last_print_date"] = last["date"]
    est = _f(last.get("eps_estimate"))
    nm = est is None or abs(est) < NM_ESTIMATE_FLOOR
    out["eps_surprise_nm"] = nm
    out["eps_surprise_pct"] = None if nm else _f(last.get("surprise_pct"))
    px = context.get("prices")
    if px is None or getattr(px, "empty", True):
        return out
    m = earnings.event_metrics(px, last["date"])
    out["sessions_since_print"] = m.get("sessions_since")
    out["reaction_ret"] = m.get("reaction_ret")
    out["print_gap_pct"] = m.get("gap_pct")
    out["print_rel_volume"] = m.get("rel_volume")
    out["post_drift_5d"] = m.get("drift_5d")
    out["post_drift_10d"] = m.get("drift_10d")
    out["post_drift_20d"] = m.get("drift_20d")
    return out


def _context_section(context: dict) -> dict:
    """Static profile, demoted: cached fundamentals + short interest."""
    out = {k: None for k in brief_contracts.METRIC_KEYS_BY_SECTION["context"]}
    out["state"] = "missing"
    fund = context.get("fundamentals") or {}
    si = context.get("short") or {}
    out["sector"] = fund.get("Sector")
    out["mkt_cap"] = _f(fund.get("Market Cap"))
    for key, src in (("fwd_pe", "Forward P/E"), ("ps", "Price / Sales")):
        val = _f(fund.get(src))
        out[key] = val if val is not None and val > 0 else None
    out["div_yield"] = _f(fund.get("Dividend Yield"))
    out["short_pct_float"] = _f(si.get("short_pct_float"))
    out["days_to_cover"] = _f(si.get("days_to_cover"))
    shares, prior = _f(si.get("shares_short")), _f(
        si.get("shares_short_prior_month"))
    out["si_change_pct"] = (round((shares - prior) / abs(prior) * 100.0, 1)
                            if shares is not None and prior else None)
    if fund:
        out["state"] = "ok"
    return out


def _reconciled_receipts(context: dict) -> dict:
    """Per-source as-of, kept honest: a receipt certifies data the brief
    actually used, so an absent payload family always reads None."""
    receipts = dict(context.get("source_as_of") or {})
    if context.get("prices") is None:
        receipts["prices"] = None
    for family in ("earnings", "fundamentals", "short"):
        if not context.get(family):
            receipts[family] = None
    return receipts


def build_brief(ticker: str, context: dict) -> dict:
    """Contract-validated brief from a loaded context. CPU only, no network."""
    brief = {
        "meta": {
            "ticker": (ticker or "").upper(),
            "built_at": context.get("built_at") or now_iso(),
            "receipts": _reconciled_receipts(context),
            "contract_version": brief_contracts.BRIEF_CONTRACT_VERSION,
            "validation_status": "pending",
        },
        "sections": {
            "price": _price_section(context),
            "earnings": _earnings_section(context),
            "context": _context_section(context),
        },
    }
    validate_brief(brief)
    return brief


def _clean(obj):
    """Recursive NaN/inf -> None scrub (the brief must be JSON-safe)."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def validate_brief(brief: dict) -> tuple[bool, list[str]]:
    """Scrub NaN/inf -> None in place, check contracts, keys, and receipts."""
    problems: list[str] = []
    try:
        problems.extend(brief_contracts.incomplete_contracts())
        sections = brief.get("sections") or {}
        for name, keys in brief_contracts.METRIC_KEYS_BY_SECTION.items():
            section = sections.get(name)
            if not isinstance(section, dict):
                problems.append(f"{name}: missing section")
                continue
            missing = [k for k in keys if k not in section]
            if missing:
                problems.append(
                    f"{name}: missing {','.join(missing)}")
            uncontracted = brief_contracts.uncontracted_section_keys(section)
            if uncontracted:
                problems.append(
                    f"{name}: uncontracted {','.join(uncontracted)}")
        meta = brief.get("meta") or {}
        for key in ("ticker", "built_at"):
            if not meta.get(key):
                problems.append(f"meta: missing {key}")
        receipts = meta.get("receipts")
        if not isinstance(receipts, dict) \
                or set(receipts) != set(brief_contracts.SOURCE_KEYS):
            problems.append("meta: receipts must cover all sources")
        cleaned = _clean(brief)
        brief.clear()
        brief.update(cleaned)
    except Exception as e:
        problems.append(f"validator error: {e}")
    if problems:
        brief.setdefault("meta", {})["validation_status"] = \
            "failed: " + "; ".join(problems[:5])
        return False, problems
    brief.setdefault("meta", {})["validation_status"] = "ok"
    return True, []
