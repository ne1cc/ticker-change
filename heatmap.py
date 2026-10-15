"""S&P 500 heatmap: sector-grouped treemap data assembly + rendering.

Latency contract: the request path (`get_heatmap_payload` -> `render_treemap`)
only ever reads SQLite and the api_cache -- it never talks to the network.
The bulk price refresh runs in app.py's background warmer thread via
`refresh_universe()`, so a first-ever page load renders instantly with
partial coverage and fills in as the warmer downloads history.
"""
from __future__ import annotations

import pandas as pd
import charts

import db
import sp500

# Assembled-tile snapshot freshness (api_cache). The warmer refreshes the
# underlying prices about every 45 min; this caps re-assembly work per hour.
PAYLOAD_TTL_HOURS = 0.5
# Constituent CSV freshness (names barely move; the list reconstitutes a few
# times a year).
CONSTITUENTS_TTL_HOURS = 24
# Per-symbol recent rows pulled from SQLite: enough for today's change plus a
# 20-session dollar-volume sizing window, without dragging 5y of history.
RECENT_ROWS = 25
# A tile's area is its trailing dollar volume (close x volume) -- a free,
# already-cached liquidity proxy that puts mega-caps in the middle like a
# cap-weighted Finviz map without needing a market-cap provider.
SIZING_WINDOW = 20
# % change drives colour, clamped to +/-MOVE_SPAN so one wild ticker can't
# wash out the scale for everyone else.
MOVE_SPAN = 3.0


def get_constituents() -> list[dict]:
    """[{symbol, name, sector}] from the cached constituent table.

    Falls back to a stale cached copy (TTL ignored) when the live CSV fetch
    fails, preserving the "always render something" contract.
    """
    cached = db.cache_get("sp500", "constituents", ttl_hours=CONSTITUENTS_TTL_HOURS)
    if cached is not None:
        return cached
    try:
        rows = sp500.fetch_sp500_constituents()
    except Exception as e:
        print(f"[heatmap] constituents fetch failed: {e}")
        return _cache_get_stale("sp500", "constituents") or []
    db.cache_set("sp500", "constituents", rows)
    return rows


def _cache_get_stale(provider: str, key: str):
    """Last cached payload for (provider, key) regardless of age, or None."""
    try:
        with db.get_conn() as conn:
            row = conn.execute(
                "SELECT payload FROM api_cache WHERE provider = ? AND key = ?",
                (provider, key),
            ).fetchone()
    except Exception:
        return None
    if row is None:
        return None
    try:
        import json
        return json.loads(row["payload"])
    except (ValueError, TypeError):
        return None


def _recent_rows(symbols: list[str], lookback: int = RECENT_ROWS) -> pd.DataFrame:
    """Per-symbol last `lookback` daily rows in one chunked query.

    A window function keeps this cheap: the full daily_prices table holds ~5y
    per symbol, and the heatmap needs only the freshest handful of rows.
    """
    if not symbols:
        return pd.DataFrame()
    records = []
    with db.get_conn() as conn:
        for i in range(0, len(symbols), 900):
            chunk = [s.upper() for s in symbols[i:i + 900]]
            placeholders = ",".join("?" * len(chunk))
            records.extend(conn.execute(
                f"""
                SELECT symbol, date, close, volume FROM (
                    SELECT symbol, date, close, volume,
                           ROW_NUMBER() OVER (
                               PARTITION BY symbol ORDER BY date DESC
                           ) AS rn
                    FROM daily_prices
                    WHERE symbol IN ({placeholders})
                )
                WHERE rn <= ?
                ORDER BY symbol, date ASC
                """,
                [*chunk, lookback],
            ).fetchall())
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records, columns=["symbol", "date", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_heatmap_data() -> dict | None:
    """Assemble heatmap tiles from SQLite. None when there's nothing to show.

    Tiles carry the latest close-to-close % change (colour) and trailing
    average dollar volume (area), grouped by GICS sector from the constituent
    table.
    """
    constituents = get_constituents()
    if not constituents:
        return None
    sym_meta = {c["symbol"]: c for c in constituents}
    rows = _recent_rows(list(sym_meta))
    if rows.empty:
        return None

    tiles = []
    for symbol, g in rows.groupby("symbol"):
        if len(g) < 2:
            continue
        g = g.sort_values("date")
        last_close = float(g["close"].iloc[-1])
        prev_close = float(g["close"].iloc[-2])
        if last_close <= 0 or prev_close <= 0:
            continue
        meta = sym_meta.get(symbol, {})
        dollar_volume = float(
            (g["close"] * g["volume"]).tail(SIZING_WINDOW).mean()
        )
        tiles.append({
            "symbol": symbol,
            "name": meta.get("name") or symbol,
            "sector": meta.get("sector") or "Other",
            "price": round(last_close, 2),
            "change_pct": round((last_close / prev_close - 1.0) * 100.0, 2),
            "dollar_volume": round(dollar_volume),
            "as_of": g["date"].iloc[-1].strftime("%Y-%m-%d"),
        })
    if not tiles:
        return None

    advancers = sum(1 for t in tiles if t["change_pct"] > 0)
    decliners = sum(1 for t in tiles if t["change_pct"] < 0)
    return {
        "tiles": tiles,
        "sectors": sorted({t["sector"] for t in tiles}),
        "coverage": len(tiles),
        "universe": len(constituents),
        "advancers": advancers,
        "decliners": decliners,
        "unchanged": len(tiles) - advancers - decliners,
        "as_of": max(t["as_of"] for t in tiles),
    }


def get_heatmap_payload() -> dict | None:
    """Cached assembled payload (api_cache), rebuilt only when stale."""
    cached = db.cache_get("heatmap", "sp500_payload", ttl_hours=PAYLOAD_TTL_HOURS)
    if cached is not None:
        return cached
    data = build_heatmap_data()
    if data is None:
        return None
    db.cache_set("heatmap", "sp500_payload", data)
    return data


def invalidate_payload() -> None:
    """Drop the assembled-payload snapshot so the next read rebuilds it.

    Called by the warmer after a successful universe refresh -- otherwise a
    page could keep seeing pre-warm coverage until the TTL runs out.
    """
    try:
        with db.get_conn() as conn:
            conn.execute(
                "DELETE FROM api_cache WHERE provider = 'heatmap' AND key = 'sp500_payload'"
            )
    except Exception as e:
        print(f"[heatmap] payload invalidation failed: {e}")


def refresh_universe(symbols: list[str], period: str = "10d") -> dict:
    """Bulk-refresh daily bars for `symbols` via threaded yf.download.

    Chunks the universe so one bad request can't lose everything, writes each
    ticker into SQLite, and never raises -- failures are reported per symbol.
    Called only from the background warmer, never the request path.
    """
    import yfinance as yf  # lazy so importing heatmap.py stays cheap

    fetched = 0
    failed: list[str] = []
    CHUNK = 100
    for i in range(0, len(symbols), CHUNK):
        chunk = [s.upper() for s in symbols[i:i + CHUNK]]
        try:
            raw = yf.download(
                chunk, period=period, interval="1d", auto_adjust=True,
                group_by="ticker", threads=True, progress=False,
            )
        except Exception as e:
            print(f"[heatmap] download failed ({chunk[0]}..{chunk[-1]}): {e}")
            failed.extend(chunk)
            continue
        multi = isinstance(raw.columns, pd.MultiIndex)
        level0 = raw.columns.get_level_values(0) if multi else []
        for symbol in chunk:
            try:
                sub = raw[symbol] if (multi and symbol in level0) else raw
                sub = sub.dropna(subset=["Close"])
                if sub.empty:
                    failed.append(symbol)
                    continue
                db.store_prices(symbol, sub)
                fetched += 1
            except Exception:
                failed.append(symbol)
    return {"fetched": fetched, "failed": failed}


# Finviz-style diverging scale: deep red through pale neutral to deep green,
# centred on 0. The neutral mid-point tracks the theme so flat tiles don't
# glow against the page background.
def _colorscale(mid: str) -> list:
    return [
        [0.0, "#7a0c22"],
        [0.35, "#d9534f"],
        [0.5, mid],
        [0.65, "#3f9d63"],
        [1.0, "#0b6e4f"],
    ]


def render_treemap(data: dict, dark: bool = False) -> str:
    """Native sector treemap; request path remains entirely cache-only."""
    by_sector = {}
    for tile in data['tiles']:
        by_sector.setdefault(tile['sector'], []).append(tile)
    sectors = []
    for sector, members in sorted(by_sector.items()):
        total = sum(max(m['dollar_volume'], 1) for m in members)
        change = sum(m['change_pct'] * max(m['dollar_volume'], 1) for m in members) / total
        leaves = [{'name': t['symbol'], 'symbol': t['symbol'], 'leaf': True,
                   'value': [max(t['dollar_volume'], 1), max(-MOVE_SPAN, min(MOVE_SPAN, t['change_pct']))],
                   'change': f"{t['change_pct']:+.2f}%", 'company': t['name'], 'price': t['price']}
                  for t in members]
        sectors.append({'id': 'sector::' + sector, 'name': sector, 'change': f'{change:+.2f}%', 'value': [total, max(-MOVE_SPAN, min(MOVE_SPAN, change))], 'children': leaves})
    option = {'series': [{'type': 'treemap', 'name': 'S&P 500', 'data': sectors,
                         'roam': False, 'nodeClick': False, 'leafDepth': 2,
                         'breadcrumb': {'show': False}, 'visualDimension': 1,
                         'visualMin': -MOVE_SPAN, 'visualMax': MOVE_SPAN,
                         'color': [stop[1] for stop in _colorscale('#3f3f46' if dark else '#e4e4e7')],
                         'colorMappingBy': 'value', 'upperLabel': {'show': True, 'height': 24},
                         'label': {'show': True}, 'levels': [{'itemStyle': {'borderWidth': 0, 'gapWidth': 3}},
                             {'itemStyle': {'borderWidth': 2, 'gapWidth': 1}, 'colorMappingBy': 'value', 'visualDimension': 1}]}]}
    return charts.render_chart(option, chart_id='heatmap-treemap', height=600)
