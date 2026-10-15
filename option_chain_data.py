"""Shared option-chain validation and Cboe delayed-data normalization."""
from __future__ import annotations

import math
import re
import threading
import time

import requests


def finite_number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def valid_iv(value):
    number = finite_number(value)
    # Exclude Yahoo's near-zero placeholder IVs; inputs are decimal volatility.
    return number is not None and 0.01 < number <= 5.0


def valid_quote(bid, ask):
    bid, ask = finite_number(bid), finite_number(ask)
    return bid is not None and ask is not None and 0 <= bid <= ask and ask > 0


def chain_quality(payload):
    """Available field groups, with individual zero-OI contracts allowed."""
    available = set()
    spot = finite_number(payload.get('spot_price'))
    for side in ('calls', 'puts'):
        rows = payload.get(side) or []
        if any((finite_number(r.get('openInterest')) or 0) > 0 for r in rows):
            available.add(side + '_oi')
        if any(valid_quote(r.get('bid'), r.get('ask')) for r in rows):
            available.add(side + '_quotes')
        if any(valid_iv(r.get('impliedVolatility')) and
               (not spot or abs((finite_number(r.get('strike')) or 0) - spot) <= spot * 0.05)
               for r in rows):
            available.add(side + '_iv')
    return available


_CBOE_SNAPSHOTS = {}
_CBOE_LOCK = threading.Lock()


def get_cboe_snapshot(ticker):
    """Fetch all expirations once per ticker; bound memory and failed retries."""
    symbol = ticker.upper().replace('^', '_')
    if not re.fullmatch(r'[A-Z0-9_.-]{1,15}', symbol):
        return None
    with _CBOE_LOCK:
        now = time.monotonic()
        cached = _CBOE_SNAPSHOTS.get(symbol)
        if cached and now < cached[0]:
            return cached[1]
        snapshot = None
        try:
            response = requests.get(
                f'https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json',
                timeout=(3, 12),
            )
            response.raise_for_status()
            candidate = response.json()
            if (isinstance(candidate, dict) and candidate.get('timestamp') and
                    isinstance(candidate.get('data'), dict) and
                    isinstance(candidate['data'].get('options'), list) and candidate['data']['options']):
                snapshot = candidate
        except (requests.RequestException, ValueError) as exc:
            print(f'[chain-cache] Cboe fetch failed for {ticker}: {exc}')
        if len(_CBOE_SNAPSHOTS) >= 16:
            _CBOE_SNAPSHOTS.pop(next(iter(_CBOE_SNAPSHOTS)))
        _CBOE_SNAPSHOTS[symbol] = (time.monotonic() + (900 if snapshot else 60), snapshot)
        return snapshot


def cboe_chain(snapshot, ticker, expiration):
    """Map the exact OCC expiration/root to the existing Yahoo chain fields."""
    if not snapshot:
        return None
    data = snapshot['data']
    payload = {'calls': [], 'puts': [], 'spot_price': data.get('current_price'),
               'source': 'Cboe delayed', 'as_of': snapshot['timestamp']}
    exp_code = expiration.replace('-', '')[2:]
    for row in data['options']:
        match = re.fullmatch(r'(.+?)(\d{6})([CP])(\d{8})', row.get('option', ''))
        if not match or match[1] != ticker.upper().lstrip('^') or match[2] != exp_code:
            continue
        payload['calls' if match[3] == 'C' else 'puts'].append({
            'contractSymbol': row['option'], 'strike': int(match[4]) / 1000,
            'bid': row.get('bid'), 'ask': row.get('ask'),
            'impliedVolatility': row.get('iv'), 'openInterest': row.get('open_interest'),
            'volume': row.get('volume'), 'lastPrice': row.get('last_trade_price'),
            'change': row.get('change'), 'percentChange': row.get('percent_change'),
        })
    return payload if payload['calls'] or payload['puts'] else None
