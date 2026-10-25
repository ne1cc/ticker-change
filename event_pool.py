"""Pooled universe event studies by 8-K type: base rates for CARs.

Batch job (same shape as warm_s3_cache.py — a standalone main(), never
wired into the request path): walks a ticker universe, studies every
non-routine 8-K with its type's windows, and publishes pooled per-type
CAR distributions plus cross-sectional t-stats (BMP-style) through
snapshot_store, so /analytics reads the last-good snapshot as its base
rate.

SEC fair access: cold-cache runs make one submissions GET per ticker;
`pause` seconds are slept between tickers so the run stays far below
EDGAR's documented 10 requests/second limit (0.15s ⇒ ~6.6 req/s ceiling
even with zero caching).
"""
from __future__ import annotations

import math
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import event_study
import sec_8k
import snapshot_store

PROVIDER = "event_pool"
MAX_DRIFT_SPAN_DAYS = 60  # collapse span ≥ longest drift window (42 sessions)


def compute_pooled_stats(cars: List[float]) -> Optional[Dict[str, Any]]:
    """Cross-sectional t-test (BMP-style) over per-event CARs.

    Returns None when fewer than two events — no distribution, no t-stat.
    """
    vals = [c for c in (cars or []) if c is not None]
    n = len(vals)
    if n < 2:
        return None
    mean = sum(vals) / n
    var = sum((c - mean) ** 2 for c in vals) / (n - 1)
    std = math.sqrt(var)
    t_stat = mean / (std / math.sqrt(n)) if std > 0 else 0.0
    try:
        from scipy import stats
        p_value = float(2 * (1 - stats.norm.cdf(abs(t_stat))))
    except Exception:
        p_value = float(math.erfc(abs(t_stat) / math.sqrt(2)))
    return {
        "n": n,
        "mean_car": round(mean, 6),
        "std_car": round(std, 6),
        "t_stat": round(t_stat, 4),
        "p_value": round(p_value, 5),
        "cars": [round(c, 6) for c in sorted(vals)],
    }


def percentile_rank(cars: List[float], value: float) -> float:
    """Empirical-CDF percentile (0-100): share of pooled CARs at or below
    `value`, so the pooled max reads P100 and the min reads P0."""
    if not cars:
        return 50.0
    at_or_below = sum(1 for c in cars if c <= value)
    return round(100.0 * at_or_below / len(cars), 1)


def pool_percentile(pool: Optional[dict], event_type: Optional[str],
                    car: Optional[float]) -> Optional[Dict[str, Any]]:
    """{"percentile", "n", "event_type"} for one CAR against its type's
    pooled distribution, or None when the pool, the type, or the CAR is
    missing (renderers must show the declared missing state then)."""
    if not pool or not event_type or car is None:
        return None
    dist = (pool.get("types") or {}).get(event_type) or {}
    cars = dist.get("cars") or []
    if not cars:
        return None
    return {"percentile": percentile_rank(cars, car), "n": dist.get("n", len(cars)),
            "event_type": event_type}


def pool_universe(
    tickers: List[str],
    *,
    stock_fetcher: Callable[[str], Any],
    benchmark_df,
    events_fetcher: Optional[Callable[[str], List]] = None,
    pause: float = 0.15,
    max_tickers: Optional[int] = None,
    universe: str = "custom",
    progress: Callable[[str], None] = lambda msg: None,
) -> Dict[str, Any]:
    """Build the pooled payload. Pure with respect to its arguments — all
    fetching is injected, so tests run offline. Publication is the
    caller's job (publish_pool) so a failed build never moves a pointer."""
    events_fetcher = events_fetcher or (lambda t: sec_8k.fetch_and_parse_8k_filings(t, limit=20))
    if max_tickers:
        tickers = tickers[:max_tickers]

    per_type: Dict[str, List[float]] = {}
    n_tickers = n_events = 0
    for i, ticker in enumerate(tickers, 1):
        progress(f"[{i}/{len(tickers)}] {ticker}")
        try:
            stock_df = stock_fetcher(ticker)
            events = events_fetcher(ticker)
        except Exception as e:
            progress(f"  {ticker}: skipped ({e})")
            events = []
            stock_df = None
        if pause and i < len(tickers):
            time.sleep(pause)  # SEC fair access: see module docstring
        if stock_df is None or getattr(stock_df, "empty", True):
            continue
        studied = [e for e in (events or []) if e.event_type is not None]
        clusters = event_study.collapse_overlapping(
            [{"event_date": e.filing_date, "ticker": e.ticker,
              "event_type": e.event_type, "event": e} for e in studied],
            span_days=MAX_DRIFT_SPAN_DAYS)
        ticker_events = 0
        for cl in clusters:
            event = cl["event"]
            ev_win, drift_win = event_study.windows_for(event.event_type)
            res = event_study.run_event_study(
                stock_df, benchmark_df, event.filing_date,
                event_type=event.event_type, ticker=event.ticker,
                event_window=ev_win, drift_window=drift_win,
                after_close=event.after_close)
            if res is None:
                continue
            per_type.setdefault(event.event_type, []).append(res.car)
            ticker_events += 1
        if ticker_events:
            n_tickers += 1
            n_events += ticker_events

    types = {t: compute_pooled_stats(per_type[t]) for t in sorted(per_type)}
    types = {t: st for t, st in types.items() if st}
    return {
        "meta": {
            "built_at": datetime.utcnow().isoformat(),
            "universe": universe,
            "n_tickers": n_tickers,
            "n_tickers_requested": len(tickers),
            "n_events": n_events,
            "stat": "cross-sectional t (BMP-style)",
            "windows": {t: {"event": list(w[0]), "drift": list(w[1]) if w[1] else None}
                        for t, w in ((t, event_study.windows_for(t))
                                     for t in sorted(types))},
        },
        "types": types,
    }


def publish_pool(payload: Dict[str, Any]) -> str:
    """Atomically publish: payload stored, pointer moved last, old editions
    pruned — a failed build before this call never moves the pointer."""
    snapshot_id = snapshot_store.new_snapshot_id()
    snapshot_store.publish(PROVIDER, snapshot_id, payload)
    return snapshot_id


def current_pool() -> Optional[Dict[str, Any]]:
    """The last-good published pool, or None (declared missing state)."""
    return snapshot_store.current(PROVIDER)


def main() -> None:
    import sp500  # lazy: keeps the module importable without yfinance

    tickers = sp500.fetch_sp500_tickers()
    payload = pool_universe(
        tickers,
        stock_fetcher=lambda t: sp500._fetch_with_retry(t, period="2y"),
        benchmark_df=sp500._fetch_with_retry("SPY", period="2y"),
        universe="sp500",
        progress=lambda msg: print(msg, flush=True))
    if not payload["types"]:
        print("[event-pool] FAILED: no events produced a usable CAR.")
        raise SystemExit(1)
    publish_pool(payload)
    meta = payload["meta"]
    print(f"[event-pool] published: {meta['n_events']} events across "
          f"{meta['n_tickers']} tickers, types: {', '.join(payload['types'])}")


if __name__ == "__main__":
    main()
