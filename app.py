from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, render_template, request, jsonify, redirect, url_for, Response
from flask_cors import CORS
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np
import charts
import time
import threading
import math
import os
import hmac
import traceback
from concurrent.futures import ThreadPoolExecutor
from db import init_db, is_fresh, get_prices, store_prices
import db
import s3_cache
import providers
import news
import ml
import ai
import corporate_actions
import event_study
import sec_8k
import earnings
import radar
import peers
import microstructure
import macro_engine
import options
from options import compute_options_terminal, calculate_greeks
import derivatives_alpha
import backtest_engine
import api_docs
import decide
import momentum_engine
import heatmap
from glossary import GLOSSARY

def _get_yf_ticker(symbol: str) -> yf.Ticker:
    """Return a yfinance Ticker instance.

    yfinance manages its own curl_cffi session (with browser TLS/header
    impersonation) internally as of the 1.x line; passing a plain
    requests.Session raises "Yahoo API requires curl_cffi session".
    """
    return yf.Ticker(symbol.upper())

app = Flask(__name__)
# Only the JSON API surface needs cross-origin access; HTML pages (notably
# /settings, which manages secrets) must not be fetchable from other origins.
CORS(app, resources={r"/api/(?!news(?:$|/)).*": {"origins": "*"}})

# Expose metric definitions to every template so the `metric` tooltip macro and
# the /glossary page share a single source of truth.
app.jinja_env.globals['GLOSSARY'] = GLOSSARY


@app.context_processor
def inject_ui_mode():
    """Excel mode flag, mirrored from localStorage into the ui_mode cookie by
    base.html so structural rendering can differ server-side."""
    ui_mode = request.cookies.get('ui_mode')
    if ui_mode is None:
        return {'excel_mode': True}
    return {'excel_mode': ui_mode == 'excel'}


@app.template_filter('xlfmt')
def xlfmt(val):
    """Excel accounting style: negatives render in parentheses, e.g. (3.42%).
    Coloring comes from the .xl-neg class, applied by the _excel.html macros."""
    s = str(val)
    if s[:1] in ('-', '−'):
        return f'({s[1:]})'
    return s

with app.app_context():
    init_db()


# Process-level memo: collapses duplicate get_or_fetch_prices() calls within and
# across nearby requests (e.g. SPY is needed by beta, cumulative-return, etc.) so
# we don't re-query SQLite and rebuild the DataFrame several times per page load.
_PRICE_MEMO: dict = {}
_PRICE_MEMO_TTL = 45  # seconds
_PRICE_MEMO_LOCK = threading.Lock()


def _fetch_yfinance_with_retry(symbol: str, period: str, retries: int = 3):
    """Fetch history from yfinance with exponential backoff. Returns a df or None.

    yfinance frequently fails or returns empty on the first attempt (Yahoo rate
    limiting / transient errors); retrying is what eliminates most of the
    intermittent "data could not be retrieved" failures.
    """
    delay = 0.6
    for attempt in range(retries):
        try:
            df = _get_yf_ticker(symbol).history(period=period, auto_adjust=True)
            if not df.empty:
                return df
            print(f"yfinance empty for {symbol} (attempt {attempt + 1}/{retries})")
        except Exception as e:
            print(f"yfinance fetch failed for {symbol} (attempt {attempt + 1}/{retries}): {e}")
        if attempt < retries - 1:
            time.sleep(delay)
            delay *= 2
    return None


def get_or_fetch_prices(symbol: str, period: str = "5y") -> pd.DataFrame | None:
    """Return cached prices from the DB, refreshing from yfinance when stale.

    Layered for speed and resilience:
      1. In-process memo (skips redundant DB reads within a request).
      2. SQLite cache (skips the network when data is < 1h old).
      3. yfinance fetch with retry/backoff on a miss.
      4. Stale-while-error: if the refresh fails but we hold older cached rows,
         serve those rather than returning nothing.
    """
    symbol = symbol.upper()

    now = time.time()
    with _PRICE_MEMO_LOCK:
        cached = _PRICE_MEMO.get(symbol)
        if cached is not None and now - cached[0] < _PRICE_MEMO_TTL:
            return cached[1]

    if not is_fresh(symbol):
        df = _fetch_yfinance_with_retry(symbol, period)
        if df is not None and not df.empty:
            store_prices(symbol, df)
        else:
            # Refresh failed — fall through and serve whatever we already have.
            print(f"Serving stale/cached data for {symbol} (refresh unavailable)")

    result = get_prices(symbol)

    # Only memoise real data so a transient failure isn't cached as "no data".
    if result is not None and not result.empty:
        with _PRICE_MEMO_LOCK:
            _PRICE_MEMO[symbol] = (now, result)

    return result

SHORT_DAY_PERIODS = {'1D': 1, '2D': 2, '3D': 3, '4D': 4, '5D': 5}


def _is_weekend(d=None):
    d = d or datetime.now()
    return d.weekday() >= 5


def _close_n_sessions_ago(df, n_sessions, *, live_mode=False):
    """Return the close from n trading sessions before the current reference point.

    When live_mode is False the current price is the latest daily bar, so 1D uses
    iloc[-2] (prior session). When live_mode is True the current price may be
    ahead of the latest bar (e.g. Monday pre-market with Friday's last bar), so
    1D uses iloc[-1] as yesterday's close.
    """
    if df is None or df.empty or n_sessions <= 0:
        return None

    latest_bar = pd.to_datetime(df.index[-1]).normalize()
    today = pd.Timestamp.now().normalize()

    if live_mode and latest_bar < today:
        hist_idx = -n_sessions
    else:
        hist_idx = -(n_sessions + 1)

    if len(df) < abs(hist_idx):
        return None
    return float(df['close'].iloc[hist_idx])


def calculate_date_periods():
    """Calculate calendar-based comparison dates for 1W through 5Y and YTD."""
    today = datetime.now()
    return {
        '1W':  today - timedelta(weeks=1),
        '2W':  today - timedelta(weeks=2),
        '3W':  today - timedelta(weeks=3),
        '1M':  today - relativedelta(months=1),
        '2M':  today - relativedelta(months=2),
        '3M':  today - relativedelta(months=3),
        '6M':  today - relativedelta(months=6),
        '1Y':  today - relativedelta(years=1),
        '2Y':  today - relativedelta(years=2),
        '3Y':  today - relativedelta(years=3),
        '4Y':  today - relativedelta(years=4),
        '5Y':  today - relativedelta(years=5),
        'YTD': datetime(today.year, 1, 1),
    }


def _find_closest_trading_price(
    df: pd.DataFrame, target_date, max_tolerance_days: int = 30
) -> tuple[float | None, str | None]:
    """Find the historical close price on the trading day closest to target_date.

    Handles weekends, holidays, and boundaries at the start of the historical series.
    Returns (historical_price, matched_date_str) or (None, None) if outside tolerance.
    """
    if df is None or df.empty:
        return None, None

    target_ts = pd.to_datetime(target_date).tz_localize(None).normalize()
    df_idx = pd.to_datetime(df.index).tz_localize(None).normalize()

    sec_diff = np.abs((df_idx - target_ts).total_seconds())
    closest_pos = int(sec_diff.argmin())
    closest_date = df_idx[closest_pos]
    diff_days = abs((closest_date - target_ts).days)

    if diff_days <= max_tolerance_days:
        price = float(df['close'].iloc[closest_pos])
        return price, closest_date.strftime('%Y-%m-%d')
    return None, None

def get_current_price_yfinance(ticker, retries=3):
    """Get current price using yfinance with retry logic and extended hours (pre-market 4am / after-hours)."""
    for attempt in range(retries):
        try:
            stock = _get_yf_ticker(ticker)
            # Try fetching with prepost=True to capture extended hours prices (4:00 AM - 8:00 PM EST)
            hist = stock.history(period="1d", interval="1m", prepost=True)
            if not hist.empty:
                return float(hist['Close'].iloc[-1])

            # Fallback to standard 1d/5d history
            hist = stock.history(period="1d")
            if hist.empty and attempt < retries - 1:
                time.sleep(0.5)
                hist = stock.history(period="5d")
            
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
            
            if attempt < retries - 1:
                time.sleep(1)
                
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {ticker}: {str(e)}")
            if attempt < retries - 1:
                time.sleep(1)
            continue
    
    return None

def get_historical_price_yfinance(ticker, target_date, retries=2):
    """Get historical price using yfinance with fixed datetime handling and retry logic"""
    for attempt in range(retries):
        try:
            stock = _get_yf_ticker(ticker)
            
            # Determine appropriate period based on target date
            days_diff = (datetime.now() - target_date).days

            # Short windows: use 5d for targets up to 5 days ago
            if days_diff <= 5:
                period = "5d"
            # 2-week window
            elif days_diff <= 14:
                period = "1mo"
            # 3-week window
            elif days_diff <= 21:
                period = "1mo"
            # ~4 weeks / 1 month
            elif days_diff <= 30:
                period = "3mo"
            # up to ~3 months
            elif days_diff <= 90:
                period = "6mo"
            # up to ~1 year
            elif days_diff <= 365:
                period = "1y"
            elif days_diff <= 365 * 2:
                period = "2y"
            elif days_diff <= 365 * 5:
                period = "5y"
            else:
                period = "max"
            
            hist = stock.history(period=period)
            
            if hist.empty:
                if attempt < retries - 1:
                    time.sleep(0.5)
                    continue
                return None
                
            # Handle timezone-aware datetime comparison properly
            hist.index = pd.to_datetime(hist.index).tz_localize(None)
            target_date = pd.to_datetime(target_date).tz_localize(None)
            
            # Find the closest date that's before or equal to target
            valid_dates = hist.index[hist.index <= target_date]
            
            if len(valid_dates) > 0:
                closest_date = valid_dates[-1]
                return float(hist.loc[closest_date]['Close'])
            else:
                # If no previous dates, use the first available
                return float(hist['Close'].iloc[0])
                
        except Exception as e:
            print(f"Historical price attempt {attempt + 1} failed for {ticker}: {str(e)}")
            if attempt < retries - 1:
                time.sleep(0.5)
            continue
    
    return None

def calculate_weekdays_ago(num_weekdays):
    """Calculate the date N weekdays (trading days) ago"""
    if num_weekdays <= 0:
        return datetime.now()
    
    current_date = datetime.now()
    weekdays_counted = 0
    
    while weekdays_counted < num_weekdays:
        current_date -= timedelta(days=1)
        # Monday=0, Sunday=6, so weekdays are 0-4
        if current_date.weekday() < 5:
            weekdays_counted += 1
    
    return current_date

def get_price_ranges(ticker, current_price=None):
    """Compute all-time and 52-week high/low and 30-day ATR from cached DB prices."""
    df = get_or_fetch_prices(ticker)
    if df is None or df.empty:
        return None

    one_year_ago = pd.Timestamp.utcnow().tz_localize(None) - pd.DateOffset(years=1)
    year_df = df[df.index >= one_year_ago]

    all_time_high = float(df['high'].max())
    all_time_low  = float(df['low'].min())
    week52_high   = float(year_df['high'].max()) if not year_df.empty else all_time_high
    week52_low    = float(year_df['low'].min())  if not year_df.empty else all_time_low

    result = {
        'all_time_high': round(all_time_high, 2),
        'all_time_low':  round(all_time_low,  2),
        '52_week_high':  round(week52_high,   2),
        '52_week_low':   round(week52_low,    2),
    }

    month_df = df.tail(30).copy()
    if len(month_df) > 1:
        month_df['prev_close'] = month_df['close'].shift(1)
        tr = pd.concat([
            month_df['high'] - month_df['low'],
            (month_df['high'] - month_df['prev_close']).abs(),
            (month_df['low']  - month_df['prev_close']).abs(),
        ], axis=1).max(axis=1)
        atr_value = float(tr.mean())
        result['atr_30d'] = round(atr_value, 2)
        if current_price and current_price > 0:
            result['atr_30d_percent'] = round((atr_value / current_price) * 100, 2)

    return result

def generate_stock_chart(ticker, period="5y"):
    """Return one price-chart mount and ordered OHLCV records from cached prices."""
    df = get_or_fetch_prices(ticker)
    if df is None or df.empty:
        return None
    df = df.sort_index(kind='stable')
    df = df[~df.index.duplicated(keep='last')]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['open', 'high', 'low', 'close'])
    if df.empty:
        return None
    records = [{'time': idx.strftime('%Y-%m-%d'), 'open': float(row['open']),
                'high': float(row['high']), 'low': float(row['low']),
                'close': float(row['close']), 'volume': int(row['volume']) if pd.notna(row['volume']) else 0}
               for idx, row in df.iterrows()]
    return {'html': '<div id="stock-chart" class="stock-chart" role="img" aria-label="Price and volume chart"></div>',
            'date_range': {'start': records[0]['time'], 'end': records[-1]['time']}, 'records': records}

def get_chart_data_json(ticker, start_date=None, end_date=None):
    """Return OHLCV records from DB, optionally filtered by date range."""
    df = get_or_fetch_prices(ticker)
    if df is None or df.empty:
        return None

    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]
    if df.empty:
        return None

    data = [
        {
            'date':   idx.strftime('%Y-%m-%d'),
            'open':   row['open'],
            'high':   row['high'],
            'low':    row['low'],
            'close':  row['close'],
            'volume': int(row['volume']),
        }
        for idx, row in df.iterrows()
    ]
    return {
        'data': data,
        'date_range': {
            'start': df.index[0].strftime('%Y-%m-%d'),
            'end':   df.index[-1].strftime('%Y-%m-%d'),
        }
    }

def _append_period_change(
    percentage_data,
    net_change_data,
    period_name,
    current_price,
    historical_price,
    matched_date=None,
):
    if historical_price is not None and not np.isnan(historical_price) and historical_price != 0:
        net_change = current_price - historical_price
        pct_change = ((current_price - historical_price) / historical_price) * 100
        percentage_data.append({
            "period": period_name,
            "value": round(pct_change, 2),
            "raw_value": pct_change,
            "is_positive": pct_change > 0,
            "matched_date": matched_date,
        })
        net_change_data.append({
            "period": period_name,
            "value": round(net_change, 2),
            "raw_value": net_change,
            "is_positive": net_change > 0,
            "matched_date": matched_date,
        })
    else:
        percentage_data.append({"period": period_name, "value": "N/A", "matched_date": None})
        net_change_data.append({"period": period_name, "value": "N/A", "matched_date": None})


def _append_weekend_placeholder(percentage_data, net_change_data, period_name):
    row = {"period": period_name, "value": "-", "is_weekend": True}
    percentage_data.append(row)
    net_change_data.append(row.copy())


def get_stock_data(ticker):
    """Get all stock data for web display, using the DB as the single data source."""
    if not ticker:
        return {"error": "No ticker provided"}

    df = get_or_fetch_prices(ticker)
    if df is None or df.empty:
        return {"error": f"Could not retrieve data for {ticker}"}

    db_latest_close = float(df['close'].iloc[-1])
    live_price = get_current_price_yfinance(ticker)
    live_mode = live_price is not None and not _is_weekend()
    current_price = live_price if live_mode else db_latest_close
    periods = calculate_date_periods()

    percentage_data = [{"period": "Current Price", "value": f"${current_price:.2f}"}]
    net_change_data = [{"period": "Current Price", "value": f"${current_price:.2f}"}]

    for period_name in list(SHORT_DAY_PERIODS.keys()) + list(periods.keys()):
        if period_name in SHORT_DAY_PERIODS:
            n_sessions = SHORT_DAY_PERIODS[period_name]
            historical_price = _close_n_sessions_ago(df, n_sessions, live_mode=live_mode)
            session_date = None
            if len(df) >= n_sessions:
                latest_bar = pd.to_datetime(df.index[-1]).normalize()
                today_norm = pd.Timestamp.now().normalize()
                hist_idx = -n_sessions if (live_mode and latest_bar < today_norm) else -(n_sessions + 1)
                if len(df) >= abs(hist_idx):
                    session_date = pd.to_datetime(df.index[hist_idx]).strftime('%Y-%m-%d')
            _append_period_change(
                percentage_data, net_change_data, period_name, current_price, historical_price, session_date
            )
            continue

        target_date = periods[period_name]
        tolerance = 30 if ('Y' in period_name and period_name != 'YTD') else 14
        historical_price, matched_date = _find_closest_trading_price(df, target_date, max_tolerance_days=tolerance)
        _append_period_change(
            percentage_data, net_change_data, period_name, current_price, historical_price, matched_date
        )

    chart_result = generate_stock_chart(ticker)
    return {
        "ticker": ticker.upper(),
        "current_price": current_price,
        "percentage_data": percentage_data,
        "net_change_data": net_change_data,
        "chart_html": chart_result['html'] if chart_result else None,
        "chart_date_range": chart_result['date_range'] if chart_result else None,
        "chart_records": chart_result.get('records', []) if chart_result else [],
    }

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    try:
        with db.get_conn() as conn:
            conn.execute("SELECT 1")
        db_status = "ok"
    except Exception:
        db_status = "error"

    body = {
        "status": "ok" if db_status == "ok" else "degraded",
        "db": db_status,
        "s3_cache": "enabled" if s3_cache.enabled() else "disabled",
    }
    return jsonify(body), 200 if db_status == "ok" else 503


@app.route('/api/openapi.json')
def openapi_json():
    """Return raw OpenAPI 3.0.3 schema specification."""
    return jsonify(api_docs.get_openapi_spec())


@app.route('/api/docs')
def swagger_ui():
    """Interactive Swagger UI API Documentation explorer."""
    return api_docs.get_swagger_ui_html(openapi_json_url="/api/openapi.json")


@app.route('/glossary')
def glossary_page():
    """Full metric reference, grouped by section (preserving GLOSSARY order)."""
    sections: dict[str, list] = {}
    for key, g in GLOSSARY.items():
        sections.setdefault(g['section'], []).append({**g, 'key': key})
    return render_template('glossary.html', sections=sections)

@app.route('/api/stock/<ticker>')
def stock_api(ticker):
    data = get_stock_data(ticker)
    return jsonify(data)

@app.route('/api/chart-data/<ticker>')
def chart_data_api(ticker):
    """API endpoint to fetch chart data for a specific date range"""
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    result = get_chart_data_json(ticker, start_date=start_date, end_date=end_date)
    
    if result is None:
        return jsonify({"error": f"Could not retrieve chart data for {ticker}"}), 404
    
    return jsonify(result)

def _news_status(feed):
    if feed['status'] == 'invalid':
        return 400
    return 503 if feed['status'] in ('unconfigured', 'quota_exhausted', 'unavailable') else 200


@app.route('/news')
def news_page():
    ticker = request.args.get('ticker', '').strip().upper()
    feed = news.get_feed(ticker, request.args.get('listing')) if ticker else None
    return render_template('news.html', ticker=ticker, feed=feed), _news_status(feed) if feed else 200


@app.route('/api/news')
def news_api():
    feed = news.get_feed(request.args.get('ticker', ''), request.args.get('listing'))
    status = _news_status(feed)
    if status == 400:
        return Response(jsonify({
            'type': '/api/docs#news-invalid-request', 'title': 'Invalid news request',
            'status': 400, 'detail': feed['message'],
        }).get_data(), status=400, mimetype='application/problem+json')
    return jsonify(feed), status


@app.route('/stock')
def stock_page():
    ticker = request.args.get('ticker', '')
    weekdays = request.args.get('weekdays', '')
    days = request.args.get('days', '')
    
    if ticker:
        data = get_stock_data(ticker)

        # get_stock_data can return {"error": ...} when yfinance is unavailable
        # or the ticker has no cached data. Guard against KeyError on
        # current_price so the page renders an error instead of 500ing.
        if 'error' in data or not data.get('current_price'):
            return render_template('stock.html', data=data, ticker=ticker.upper())

        # Get price ranges (all-time and 52-week) - always fetch for display
        price_ranges = get_price_ranges(ticker, current_price=data.get('current_price'))
        if price_ranges:
            data['price_ranges'] = price_ranges
        
        # Handle custom lookback (weekdays or days)
        custom_result = None
        df_cached = get_or_fetch_prices(ticker)

        def _lookback_result(target_date, label_type, label_count):
            if df_cached is None or df_cached.empty:
                return {'error': 'No price data available'}
            tolerance = 30 if (label_count >= 365) else 14
            historical_price, matched_dt = _find_closest_trading_price(
                df_cached, target_date, max_tolerance_days=tolerance
            )
            if historical_price is None:
                return {'error': f'Could not retrieve price for {label_count} {label_type} ago'}
            current_price = data['current_price']
            net_change = current_price - historical_price
            pct_change = ((current_price - historical_price) / historical_price) * 100
            atr_percent = None
            if price_ranges and 'atr_30d' in price_ranges and current_price > 0:
                atr_percent = round((price_ranges['atr_30d'] / current_price) * 100, 2)
            return {
                'type': label_type,
                'days': label_count,
                'target_date': matched_dt or target_date.strftime('%Y-%m-%d'),
                'historical_price': round(historical_price, 2),
                'current_price': round(current_price, 2),
                'net_change': round(net_change, 2),
                'percentage_change': round(pct_change, 2),
                'is_positive': net_change > 0,
                'atr_percent': atr_percent,
            }

        if weekdays:
            try:
                num_weekdays = int(weekdays)
                if num_weekdays > 0:
                    custom_result = _lookback_result(
                        calculate_weekdays_ago(num_weekdays), 'weekdays', num_weekdays
                    )
            except ValueError:
                custom_result = {'error': 'Please enter a valid number'}
        elif days:
            try:
                num_days = int(days)
                if num_days > 0:
                    custom_result = _lookback_result(
                        datetime.now() - timedelta(days=num_days), 'days', num_days
                    )
            except ValueError:
                custom_result = {'error': 'Please enter a valid number'}
        
        data['custom_result'] = custom_result
        data['fundamentals'] = get_fundamentals(ticker)
        return render_template('stock.html', data=data)
    return render_template('index.html')

def get_fundamentals(ticker: str) -> dict | None:
    """Fetch key valuation multiples, short interest, consensus forecasts, and upcoming events from yfinance.

    Cached 24h in api_cache (provider "yfinance") — the radar warmer and the
    peers module read the same key, and /stock used to hit yfinance live on
    every load.
    """
    cache_key = f"fundamentals:{ticker.upper()}"
    cached = db.cache_get("yfinance", cache_key, 24)
    if cached is not None:
        return cached
    result = _get_fundamentals_uncached(ticker)
    if result is not None:
        db.cache_set("yfinance", cache_key, result)
    return result


def _short_interest_for(ticker: str) -> tuple[dict | None, str]:
    """Real SI payload + honesty flag. "assumed" means the microstructure
    squeeze index ran on its 3.0%/2.0d signature defaults, not reported data."""
    try:
        si = providers.get_short_interest(ticker)
    except Exception:
        si = None
    if si and si.get("short_pct_float") is not None and si.get("days_to_cover") is not None:
        return si, "reported"
    return si, "assumed"


def _si_kwargs(ticker: str) -> tuple[float, float]:
    """(short_pct_float, days_to_cover) kwargs for get_microstructure_analytics,
    real values when reported (fraction x100 — the squeeze index is scaled in
    percent), signature defaults otherwise."""
    si, _source = _short_interest_for(ticker)
    pct = (si or {}).get("short_pct_float")
    dtc = (si or {}).get("days_to_cover")
    return (pct * 100.0 if pct is not None else 3.0,
            dtc if dtc is not None else 2.0)


def _get_fundamentals_uncached(ticker: str) -> dict | None:
    try:
        t = _get_yf_ticker(ticker)
        info = t.info or {}
        
        # Extract upcoming earnings dates and consensus forecasts if available
        cal = None
        try:
            cal = t.calendar
        except Exception:
            pass  # Some tickers (e.g., indices) might fail to return calendar data
            
        next_earnings = None
        consensus_eps = None
        consensus_rev = None
        
        if cal and isinstance(cal, dict):
            dates = cal.get('Earnings Date')
            if dates and len(dates) > 0:
                next_earnings = dates[0].strftime('%Y-%m-%d')
            consensus_eps = cal.get('Earnings Average')
            consensus_rev = cal.get('Revenue Average')

        raw = {
            'Name':              info.get('longName') or info.get('shortName'),
            'Sector':            info.get('sector'),
            'Industry':          info.get('industry'),
            # ETFs have no sector/industry; Yahoo classifies them this way
            # instead (consumed as a fallback by the /ai-summary VAL sheet).
            'Category':          info.get('category'),
            'Fund Family':       info.get('fundFamily'),
            'Market Cap':        info.get('marketCap'),
            'Trailing P/E':      info.get('trailingPE'),
            'Forward P/E':       info.get('forwardPE'),
            'PEG Ratio':         info.get('trailingPegRatio') or info.get('pegRatio'),
            'EV / EBITDA':       info.get('enterpriseToEbitda'),
            'EV / Revenue':      info.get('enterpriseToRevenue'),
            'Price / Book':      info.get('priceToBook'),
            'Price / Sales':     info.get('priceToSalesTrailing12Months'),
            'Free Cash Flow':    info.get('freeCashflow'),
            'EPS (TTM)':         info.get('trailingEps'),
            'Forward EPS':       info.get('forwardEps'),
            'Revenue Growth':    info.get('revenueGrowth'),
            'Earnings Growth':   info.get('earningsGrowth'),
            'Target Mean':       info.get('targetMeanPrice'),
            'Analyst Rating':    info.get('recommendationKey'),
            'Short % Float':     info.get('shortPercentOfFloat'),
            'Short Ratio':       info.get('shortRatio'),
            'Dividend Yield':    info.get('dividendYield'),
            
            # --- Forward Estimates & Guidance ---
            'Next Earnings':     next_earnings,
            'Consensus EPS':     consensus_eps,
            'Consensus Revenue': consensus_rev,
        }
        return {k: v for k, v in raw.items() if v is not None}
    except Exception as e:
        print(f"Fundamentals fetch failed for {ticker}: {e}")
        return None


# --- Shared option-chain cache (S3 primary, SQLite api_cache fallback) ---
# Chains are written to both stores and read S3-first: a configured
# S3_CACHE_BUCKET survives restarts and is shared across instances, while the
# local SQLite copy keeps a single-node deployment working with no AWS setup.
# Stale entries are served when yfinance fails — stale-but-present beats a
# blank options table after hours or under rate limiting.

CHAIN_TTL_HOURS = 4
_STALE_TTL_HOURS = 24 * 365  # "any age" TTL for last-resort SQLite reads


def _chain_records(df) -> list:
    """DataFrame -> JSON-safe records (Timestamps become ISO strings)."""
    if df is None or df.empty:
        return []
    out = df.copy()
    for col in out.columns:
        if str(out[col].dtype).startswith('datetime'):
            out[col] = out[col].apply(lambda x: x.isoformat() if pd.notna(x) else None)
    return out.to_dict('records')


def _chain_from_payload(payload) -> tuple:
    calls = pd.DataFrame(payload.get('calls', []))
    puts = pd.DataFrame(payload.get('puts', []))
    return calls, puts, payload.get('spot_price')


def _spot_price(ticker: str, stock=None) -> float | None:
    try:
        hist = (stock or _get_yf_ticker(ticker)).history(period="1d")
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
    except Exception:
        pass
    df = get_or_fetch_prices(ticker)
    if df is not None and not df.empty:
        return float(df['close'].iloc[-1])
    return None


def get_cached_expirations(ticker: str, stock=None, force: bool = False) -> list:
    """Expiration dates for a ticker: fresh cache first, then live yfinance."""
    ticker = ticker.upper()
    stale = None
    from zoneinfo import ZoneInfo
    today_str = datetime.now(ZoneInfo("America/New_York")).strftime('%Y-%m-%d')

    if not force:
        hit = s3_cache.get_json(f"{ticker}/meta.json")
        if hit is not None:
            payload, age = hit
            if age <= CHAIN_TTL_HOURS and payload.get('expirations'):
                expirations = [d for d in payload['expirations'] if d >= today_str]
                if expirations:
                    return expirations
            stale = payload
        cached = db.cache_get("yfinance", f"optexp:{ticker}", CHAIN_TTL_HOURS)
        if cached:
            expirations = [d for d in cached if d >= today_str]
            if expirations:
                return expirations

    try:
        expirations = list((stock or _get_yf_ticker(ticker)).options or [])
    except Exception as e:
        print(f"[chain-cache] expirations fetch failed for {ticker}: {type(e).__name__}: {e}")
        expirations = []

    if expirations:
        expirations = [d for d in expirations if d >= today_str]
        s3_cache.put_json(f"{ticker}/meta.json", {'expirations': expirations})
        db.cache_set("yfinance", f"optexp:{ticker}", expirations)
        return expirations

    if stale is not None and stale.get('expirations'):
        expirations = [d for d in stale['expirations'] if d >= today_str]
        if expirations:
            return expirations
    stale_sqlite = db.cache_get("yfinance", f"optexp:{ticker}", _STALE_TTL_HOURS) or []
    return [d for d in stale_sqlite if d >= today_str]


def get_cached_chain(ticker: str, expiration: str, stock=None, spot_hint=None, force: bool = False) -> tuple | None:
    """One expiration's chain as (calls_df, puts_df, spot_price), read-through cached.

    Read order: fresh S3 -> fresh SQLite -> live yfinance (written back to both
    stores) -> stale S3 -> stale SQLite -> None.
    """
    ticker = ticker.upper()
    s3_key = f"{ticker}/{expiration}.json"
    sql_key = f"optchain:{ticker}:{expiration}"

    stale = None
    if not force:
        hit = s3_cache.get_json(s3_key)
        if hit is not None:
            payload, age = hit
            if age <= CHAIN_TTL_HOURS:
                return _chain_from_payload(payload)
            stale = payload
        cached = db.cache_get("yfinance", sql_key, CHAIN_TTL_HOURS)
        if cached is not None:
            return _chain_from_payload(cached)

    try:
        chain = (stock or _get_yf_ticker(ticker)).option_chain(expiration)
        calls = chain.calls.copy() if hasattr(chain, 'calls') else pd.DataFrame()
        puts = chain.puts.copy() if hasattr(chain, 'puts') else pd.DataFrame()
        if calls.empty and puts.empty:
            raise ValueError("retrieved option chain is empty")
        spot = spot_hint or _spot_price(ticker, stock)
        payload = {
            'calls': _chain_records(calls),
            'puts': _chain_records(puts),
            'spot_price': spot,
        }
        s3_cache.put_json(s3_key, payload)
        db.cache_set("yfinance", sql_key, payload)
        return calls, puts, spot
    except Exception as e:
        print(f"[chain-cache] live fetch failed for {ticker} {expiration}: {type(e).__name__}: {e}")

    if stale is not None:
        return _chain_from_payload(stale)
    cached = db.cache_get("yfinance", sql_key, _STALE_TTL_HOURS)
    return _chain_from_payload(cached) if cached is not None else None


def get_options_smile(ticker: str, current_price: float) -> str | None:
    """Build an IV smile chart from cached option chains across 3-4 expirations."""
    try:
        stock = _get_yf_ticker(ticker)
        expirations = get_cached_expirations(ticker, stock)
        if not expirations:
            return None

        selected = expirations[:min(4, len(expirations))]
        palette = ['#f59e0b', '#6366f1', '#10b981', '#f43f5e']

        chart_series = []
        plotted = 0
        for exp in selected:
            cached = get_cached_chain(ticker, exp, stock=stock, spot_hint=current_price)
            if cached is None:
                continue
            calls, _, _ = cached
            if calls.empty or 'impliedVolatility' not in calls.columns:
                continue
            calls = calls[
                (calls['strike'] >= current_price * 0.70) &
                (calls['strike'] <= current_price * 1.50) &
                (calls['impliedVolatility'] > 0.01)
            ].sort_values('strike')
            if len(calls) < 3:
                continue
            chart_series.append(charts.line(exp, calls['strike'] / current_price,
                                            calls['impliedVolatility'] * 100, palette[plotted % len(palette)]))
            plotted += 1

        if plotted == 0:
            return None

        return charts.render_chart(charts.cartesian(chart_series, xtype='value',
            xname='Moneyness (Strike / Spot)', yname='Implied Volatility (%)',
            refs=[charts.reference('x', 1, 'ATM')]), height=350)
    except Exception as e:
        print(f"Options smile failed for {ticker}: {e}")
        return None


def get_gex_profile(ticker: str, current_price: float, rf_rate: float = 0.045) -> dict | None:
    """Build a dealer Gamma Exposure (GEX) profile from yfinance option chains.

    Per strike:  GEX = Γ × OpenInterest × 100 × Spot² × 0.01, signed by the
    standard dealer-positioning convention (long calls → +, short puts → −).
    yfinance does not return Γ, so it is computed with the in-house Black-Scholes
    engine (`calculate_greeks`) from the strike, spot, time-to-expiry and the
    chain's implied volatility. Aggregated across the nearest expirations this
    yields the structural levels traders watch: the gamma-flip strike, the call
    wall (resistance) and the put wall (support). Values are in $M per 1% move.

    This is a naive end-of-day model (delayed yfinance OI, uniform dealer
    assumptions) — directional, not an institutional low-latency feed.
    """
    try:
        stock = _get_yf_ticker(ticker)
        expirations = get_cached_expirations(ticker, stock)
        if not expirations or not current_price:
            return None

        today = datetime.now()
        # Near-term expirations dominate dealer gamma; cap the chain count so the
        # page stays responsive (each option_chain call is a separate request).
        selected = []
        for exp in expirations:
            try:
                exp_dt = datetime.strptime(exp, '%Y-%m-%d')
            except ValueError:
                continue
            dte = (exp_dt - today).days
            if dte < 0:
                continue
            selected.append((exp, exp_dt, dte))
            if len(selected) >= 6:
                break
        if not selected:
            return None

        # Independent lookups (S3/SQLite hit or live fetch) — run them concurrently.
        def _fetch(exp_tuple):
            try:
                return exp_tuple, get_cached_chain(
                    ticker, exp_tuple[0], stock=stock, spot_hint=current_price)
            except Exception:
                return exp_tuple, None

        with ThreadPoolExecutor(max_workers=len(selected)) as pool:
            fetched = list(pool.map(_fetch, selected))

        lo, hi = current_price * 0.80, current_price * 1.20
        options_data = []

        for (exp, exp_dt, dte), chain in fetched:
            if chain is None:
                continue
            calls_df, puts_df, _ = chain
            t_years = max(1e-5, (dte + 1) / 365.0)
            for df_side, side in ((calls_df, 'call'), (puts_df, 'put')):
                if df_side is None or df_side.empty:
                    continue
                for _, row in df_side.iterrows():
                    k = float(row.get('strike', 0) or 0)
                    if not (lo <= k <= hi):
                        continue
                    oi = row.get('openInterest', 0)
                    iv = row.get('impliedVolatility', 0)
                    if oi is None or iv is None or np.isnan(oi) or np.isnan(iv):
                        continue
                    oi, iv = float(oi), float(iv)
                    if oi <= 0 or iv <= 0.01:
                        continue
                    options_data.append((k, t_years, side, oi, iv))

        if not options_data:
            return None

        agg: dict = {}  # strike -> {'call': gex, 'put': gex}
        for k, t_years, side, oi, iv in options_data:
            greeks = calculate_greeks(current_price, k, t_years, iv, rf_rate)
            if not greeks:
                continue
            # Dollar gamma per 1% move, scaled to millions of $.
            gex = greeks['gamma'] * oi * 100 * (current_price ** 2) * 0.01 / 1e6
            bucket = agg.setdefault(k, {'call': 0.0, 'put': 0.0})
            bucket[side] += gex

        if not agg:
            return None

        strikes  = sorted(agg.keys())
        call_gex = [agg[k]['call'] for k in strikes]    # dealers long calls  → +
        put_gex  = [-agg[k]['put'] for k in strikes]    # dealers short puts  → −
        net_gex  = [c + p for c, p in zip(call_gex, put_gex)]

        # $5-binned aggregation: collapse minor/weekly strikes into the round
        # institutional levels so a single noisy strike can't masquerade as a
        # wall. Purely a visual view — stats below stay on the precise strikes.
        bin_agg: dict = {}
        for k in strikes:
            slot = bin_agg.setdefault(round(k / 5.0) * 5, {'call': 0.0, 'put': 0.0})
            slot['call'] += agg[k]['call']
            slot['put']  += agg[k]['put']
        bin_strikes = sorted(bin_agg.keys())
        bin_call = [bin_agg[k]['call'] for k in bin_strikes]
        bin_put  = [-bin_agg[k]['put'] for k in bin_strikes]
        bin_net  = [c + p for c, p in zip(bin_call, bin_put)]

        # Cumulative net GEX from the lowest strike up.
        cum     = list(np.cumsum(net_gex))
        bin_cum = list(np.cumsum(bin_net))

        # Walls = largest gamma concentration on the side of spot where it can
        # actually act as resistance/support. A call wall below spot (or put wall
        # above) is meaningless, so constrain by side, falling back to the global
        # extreme only if one side is empty.
        def _wall(vals, want_above, pick):
            sided = [(v, k) for v, k in zip(vals, strikes)
                     if (k >= current_price) == want_above]
            pool  = [(v, k) for v, k in (sided or list(zip(vals, strikes)))
                     if (v > 0 if pick is max else v < 0)]
            return pick(pool)[1] if pool else None

        call_wall = _wall(call_gex, want_above=True,  pick=max)
        put_wall  = _wall(put_gex,  want_above=False, pick=min)

        # Gamma flip: spot price where total net GEX crosses zero.
        # Below it dealers are net-short gamma (vol-expanding);
        # above it net-long gamma (vol-dampening).
        def calc_total_gex(s):
            total = 0.0
            for k, t_years, side, oi, iv in options_data:
                if s <= 0.01:
                    continue
                try:
                    d1 = (math.log(s / k) + (rf_rate + 0.5 * iv * iv) * t_years) / (iv * math.sqrt(t_years))
                    pdf_d1 = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
                    gamma = pdf_d1 / (s * iv * math.sqrt(t_years))
                    sign = 1.0 if side == 'call' else -1.0
                    gex = sign * gamma * oi * 100 * (s ** 2) * 0.01 / 1e6
                    total += gex
                except Exception:
                    continue
            return total

        gamma_flip = None
        if strikes:
            y_lo, y_hi = strikes[0], strikes[-1]
            grid_spots = np.linspace(y_lo, y_hi, 100)
            grid_gex = [calc_total_gex(s) for s in grid_spots]
            
            crossings = []
            for i in range(1, len(grid_gex)):
                if (grid_gex[i - 1] <= 0 < grid_gex[i]) or (grid_gex[i - 1] >= 0 > grid_gex[i]):
                    s0, s1 = grid_spots[i - 1], grid_spots[i]
                    y0, y1 = grid_gex[i - 1], grid_gex[i]
                    val = float(s0 + (s1 - s0) * (-y0) / (y1 - y0)) if y1 != y0 else float(s1)
                    crossings.append(val)
            
            if crossings:
                gamma_flip = round(min(crossings, key=lambda c: abs(c - current_price)), 2)

        total_gex = float(np.sum(net_gex))

        # Top concentration nodes: the strikes carrying the most gamma (by |net|),
        # their share of total gross gamma, and the hedging behaviour they impose.
        # The leaderboard a trader reads instead of eyeballing a 100-point axis.
        gross  = sum(abs(v) for v in net_gex) or 1.0
        ranked = sorted(zip(strikes, net_gex, call_gex, put_gex),
                        key=lambda t: abs(t[1]), reverse=True)
        top_nodes = [{
            'strike': round(k, 2),
            'net':    round(n, 1),
            'pct':    round(abs(n) / gross * 100, 1),
            'kind':   'Vol-dampening' if n >= 0 else 'Trend-amplifying',
            'side':   'Call' if abs(c) >= abs(p) else 'Put',
        } for k, n, c, p in ranked[:5]]

        # --- Key levels & their magnitudes -----------------------------------
        # The handful of strikes a desk actually trades off: the call wall
        # (largest + gamma above spot → resistance), the put wall (largest −
        # gamma below spot → support), and the absolute gamma magnet (HVL — the
        # single strike carrying the most |γ|, the strongest pin / hedge wall).
        net_by_strike = dict(zip(strikes, net_gex))
        call_by_strike = dict(zip(strikes, call_gex))
        put_by_strike = dict(zip(strikes, put_gex))

        def _mag(level, table):
            return round(float(table.get(level, 0.0)), 1) if level is not None else None

        call_wall_val = _mag(call_wall, call_by_strike)
        put_wall_val = _mag(put_wall, put_by_strike)
        hvl = max(strikes, key=lambda k: abs(net_by_strike[k])) if strikes else None
        hvl_val = _mag(hvl, net_by_strike)

        # Symmetric, locked axes so bar proportions reflect real positioning
        # shifts rather than chart rendering auto-scaling one side independently.
        bar_max = max([abs(v) for v in call_gex + put_gex + net_gex
                       + bin_call + bin_put + bin_net] or [1.0]) * 1.12
        cum_max = max([abs(v) for v in cum + bin_cum] or [1.0]) * 1.08
        levels = [charts.reference('y', current_price, f'Spot ${current_price:.2f}', '#f59e0b')]
        for value, label, color in [(gamma_flip, 'Zero-Gamma', '#7c3aed'),
                                    (call_wall, 'Call Wall', '#10b981'), (put_wall, 'Put Wall', '#f43f5e'), (hvl, 'HVL', '#71717a')]:
            if value is not None:
                levels.append(charts.reference('y', value, f'{label} ${value:g}', color))
        def _gex_view(seq, calls, puts, nets, cumulative, split):
            bars = ([charts.bar('Call GEX', calls, seq, '#10b981'), charts.bar('Put GEX', puts, seq, '#f43f5e')]
                    if split else [charts.bar('Net GEX', nets, seq, ['#10b981' if v >= 0 else '#f43f5e' for v in nets])])
            for bar in bars:
                bar['stack'] = 'exposure'
                bar['encode'] = {'x': 0, 'y': 1}
            cumulative_line = charts.line('Cumulative net γ', cumulative, seq, '#6366f1', xAxisIndex=1)
            opt = charts.cartesian(bars + [cumulative_line], xtype='value', yname='Strike ($)', refs=levels)
            opt['xAxis'] = [{'type': 'value', 'name': 'Gamma exposure ($M / 1% move)', 'min': -bar_max, 'max': bar_max},
                            {'type': 'value', 'name': 'Cumulative ($M)', 'min': -cum_max, 'max': cum_max, 'position': 'top', 'splitLine': {'show': False}}]
            opt['terminalHorizontalBars'] = True
            if gamma_flip is not None:
                bars[0]['markArea'] = {'silent': True, 'data': [
                    [{'yAxis': min(seq), 'itemStyle': {'color': 'rgba(244,63,94,.07)'}}, {'yAxis': gamma_flip}],
                    [{'yAxis': gamma_flip, 'itemStyle': {'color': 'rgba(16,185,129,.07)'}}, {'yAxis': max(seq)}]]}
            return opt
        views = [_gex_view(strikes, call_gex, put_gex, net_gex, cum, split) for split in [False, True]]
        views += [_gex_view(bin_strikes, bin_call, bin_put, bin_net, bin_cum, split) for split in [False, True]]
        gex_option = {**views[0]}
        gex_option['terminalViews'] = {'labels': ['Net', 'Call / Put', 'Net · $5', 'Call / Put · $5'], 'options': views.copy()}
        return {
            'chart': charts.render_chart(gex_option, height=520),
            'stats': {
                'total_gex': round(total_gex, 1),
                'regime': 'positive' if total_gex >= 0 else 'negative',
                'call_wall': call_wall,
                'call_wall_val': call_wall_val,
                'put_wall': put_wall,
                'put_wall_val': put_wall_val,
                'gamma_flip': gamma_flip,
                'hvl': hvl,
                'hvl_val': hvl_val,
                'spot': round(float(current_price), 2),
                'top_nodes': top_nodes,
                'dte_range': f"{selected[0][2]}–{selected[-1][2]}d",
                'n_expirations': len(selected),
            },
        }
    except Exception as e:
        print(f"GEX profile failed for {ticker}: {e}")
        return None


def _normalize_insider_df(raw):
    """Clean yfinance insider_transactions into a signed DataFrame.

    Returns (idf, None) on success or (None, reason) if data is unusable.
    Signs shares: Purchase → positive, Sale → negative, Gift/other → 0 (excluded
    from net activity but kept for the raw table).
    """
    if raw is None or (hasattr(raw, 'empty') and raw.empty):
        return None, 'no data'

    idf = raw.copy()
    if hasattr(idf.index, 'tz') and idf.index.tz is not None:
        idf.index = pd.to_datetime(idf.index).tz_localize(None)
    else:
        idf.index = pd.to_datetime(idf.index, errors='coerce')

    idf = idf[idf.index.notna()]
    idf = idf[idf.index >= '2000-01-01']

    shares_col = next((c for c in idf.columns if c.lower() == 'shares'), None)
    text_col   = next((c for c in idf.columns if c.lower() == 'text'), None)
    name_col   = next((c for c in idf.columns if c.lower() == 'insider'), None)
    pos_col    = next((c for c in idf.columns if c.lower() == 'position'), None)
    val_col    = next((c for c in idf.columns if c.lower() == 'value'), None)
    if shares_col is None:
        return None, 'no shares column'

    idf['_shares'] = pd.Series(pd.to_numeric(idf[shares_col], errors='coerce')).fillna(0)

    def _sign(text):
        t = str(text).lower()
        if 'purchase' in t or 'buy' in t or 'acquire' in t:
            return 1
        if 'sale' in t or 'sell' in t:
            return -1
        if 'gift' in t or 'grant' in t:
            return 0
        return -1  # blank/unknown — treat as disposition

    if text_col:
        idf['_signed'] = idf.apply(lambda r: r['_shares'] * _sign(r[text_col]), axis=1)
    else:
        idf['_signed'] = -idf['_shares']  # yfinance defaults to sales

    # Keep metadata for the transactions table
    idf['_name'] = idf[name_col].astype(str) if name_col else ''
    idf['_position'] = idf[pos_col].astype(str) if pos_col else ''
    idf['_value'] = pd.Series(pd.to_numeric(idf[val_col], errors='coerce')).fillna(0) if val_col else 0
    idf['_text'] = idf[text_col].astype(str) if text_col else ''
    return idf, None


def _insider_rows_cached(ticker: str) -> list[dict] | None:
    """Normalized insider rows, 24h-cached (JSON-safe: dates as ISO strings).

    The summary card and the monthly chart each used to hit yfinance on every
    request; filings are slow-moving data, so one fetch per ticker per day is
    plenty. None is never cached (a failed fetch retries next request).
    """
    cached = db.cache_get("yfinance", f"insider:{ticker.upper()}", 24)
    if cached is not None:
        return cached
    try:
        raw = _get_yf_ticker(ticker).insider_transactions
        idf, _err = _normalize_insider_df(raw)
    except Exception as e:
        print(f"Insider fetch failed for {ticker}: {e}")
        return None
    if idf is None or idf.empty:
        return None
    rows = [{
        'date':      idx.strftime('%Y-%m-%d'),
        '_signed':   int(r['_signed']),
        '_name':     str(r['_name']),
        '_position': str(r['_position']),
        '_value':    float(r['_value']),
        '_text':     str(r['_text']),
    } for idx, r in idf.iterrows()]
    db.cache_set("yfinance", f"insider:{ticker.upper()}", rows)
    return rows


def _insider_df_from_rows(rows: list[dict]) -> pd.DataFrame | None:
    """Rebuild the normalized insider DataFrame (date-indexed) from cached rows."""
    if not rows:
        return None
    idf = pd.DataFrame(rows)
    idf.index = pd.DatetimeIndex(pd.to_datetime(idf.pop('date')))
    return idf


def get_insider_summary(ticker: str) -> dict | None:
    """Recent insider transactions + aggregate stats for the analytics card."""
    try:
        idf = _insider_df_from_rows(_insider_rows_cached(ticker))
        if idf is None:
            return None

        buys   = idf[idf['_signed'] > 0]
        sells  = idf[idf['_signed'] < 0]
        total_buy  = int(buys['_signed'].sum())
        total_sell = int(sells['_signed'].sum())
        net        = total_buy + total_sell  # sells are already negative

        txns = []
        for _, r in idf.head(12).iterrows():
            signed = int(r['_signed'])
            txns.append({
                'name':      r['_name'],
                'position':  r['_position'],
                'shares':    signed,
                'is_buy':    signed > 0,
                'value':     float(r['_value']),
                'text':      r['_text'],
                'date':      r.name.strftime('%Y-%m-%d') if r.name is not pd.NaT else '',
            })

        return {
            'n_buys':    len(buys),
            'n_sells':   len(sells),
            'total_buy': total_buy,
            'total_sell': abs(total_sell),
            'net':       net,
            'n_insiders': idf['_name'].nunique(),
            'transactions': txns,
        }
    except Exception as e:
        print(f"Insider summary failed for {ticker}: {e}")
        return None


def get_insider_chart(ticker: str, price_df: pd.DataFrame) -> str | None:
    """Dual-axis chart: net monthly insider share activity vs. stock price."""
    try:
        idf = _insider_df_from_rows(_insider_rows_cached(ticker))
        if idf is None or idf.empty:
            return None

        # Bound insider transactions to price_df time window if price_df is available
        if price_df is not None and not price_df.empty:
            min_date = price_df.index.min()
            idf = idf[idf.index >= min_date]

        # Exclude gifts/grants (sign 0) from the net monthly bar
        signed = idf[idf['_signed'] != 0]
        if signed.empty:
            return None

        monthly = signed['_signed'].resample('ME').sum()
        monthly = monthly[monthly != 0]
        if monthly.empty:
            return None

        bar_colors = ['#10b981' if v > 0 else '#f43f5e' for v in monthly.values]

        chart_series = [charts.bar('Net Insider Shares', monthly.index, monthly.values, bar_colors)]
        if price_df is not None and not price_df.empty:
            chart_series.append(charts.line('Price', price_df.index, price_df['close'], yAxisIndex=1))
        option = charts.cartesian(chart_series, yname='Net Insider Shares')
        option['terminalRanges'] = ['6M', '1Y', '3Y', 'All']
        option['yAxis'] = [{'type': 'value', 'name': 'Net shares'}, {'type': 'value', 'name': 'Price ($)', 'scale': True, 'position': 'right', 'splitLine': {'show': False}}]
        return charts.render_chart(option, height=380)
    except Exception as e:
        print(f"Insider chart failed for {ticker}: {e}")
        return None


def get_cumulative_return_chart(ticker: str, df: pd.DataFrame) -> str | None:
    """Normalised cumulative return: ticker vs. SPY vs. QQQ from the stock's earliest date."""
    try:
        spy_df = get_or_fetch_prices('SPY')
        qqq_df = get_or_fetch_prices('QQQ')

        series = {'ticker': df['close'].copy()}
        if spy_df is not None and not spy_df.empty:
            series['SPY'] = spy_df['close'].copy()
        if qqq_df is not None and not qqq_df.empty:
            series['QQQ'] = qqq_df['close'].copy()

        # Align on common dates, start from the earliest date present in all series
        combined = pd.DataFrame(series).dropna()
        if len(combined) < 2:
            return None

        # Normalise to 100 at day-0
        normalised = (combined / combined.iloc[0]) * 100

        palette = {
            'ticker': '#f59e0b',
            'SPY':    '#a1a1aa',
            'QQQ':    '#6366f1',
        }
        names = {
            'ticker': ticker.upper(),
            'SPY':    'SPY',
            'QQQ':    'QQQ',
        }

        chart_series = []
        for key in ['SPY', 'QQQ', 'ticker']:
            if key in normalised.columns:
                final_val = round(float(normalised[key].iloc[-1]), 1)
                chart_series.append(charts.line(f"{names[key]}  {final_val}", normalised.index, normalised[key], palette[key]))
        return charts.render_chart(charts.cartesian(chart_series, yname='Growth of $100', refs=[charts.reference('y', 100)]), height=350)
    except Exception as e:
        print(f"Cumulative return chart failed for {ticker}: {e}")
        return None


def get_price_target_chart(ticker: str, current_price: float) -> str | None:
    """Analyst consensus price target gauge via Finnhub."""
    try:
        pt = providers.finnhub_price_target(ticker)
        if not pt:
            return None

        t_low  = pt.get('low')
        t_mean = pt.get('mean')
        t_med  = pt.get('median')
        t_high = pt.get('high')
        if not all(isinstance(v, (int, float)) for v in [t_low, t_mean, t_high]):
            return None

        upside_pct = ((t_mean - current_price) / current_price) * 100
        is_upside  = upside_pct >= 0
        upside_color = '#10b981' if is_upside else '#f43f5e'

        chart_series = [charts.line(f'Target range ${t_low:.0f} – ${t_high:.0f}', [t_low, t_high], [.5, .5], '#6366f1', width=4)]
        for value, label, color in [(t_mean, 'Mean target', '#f59e0b'), (t_med, 'Median target', '#a78bfa'), (current_price, 'Current', upside_color)]:
            if value is not None:
                chart_series.append({'name': f'{label} ${value:.2f}', 'type': 'scatter', 'symbol': 'diamond' if label != 'Current' else 'circle', 'symbolSize': 14,
                                     'itemStyle': {'color': color}, 'data': [[value, .5]], 'label': {'show': True, 'position': 'top', 'formatter': f'${value:.0f}'}})
        option = charts.cartesian(chart_series, xtype='value', xname='Price ($)', title=f'{upside_pct:+.1f}% to mean target')
        option['yAxis'].update({'min': 0, 'max': 1, 'show': False})
        option['xAxis'].update({'min': min(t_low, current_price) * .88, 'max': max(t_high, current_price) * 1.08})
        option['graphic'] = {'type': 'text', 'bottom': 2, 'left': 'center', 'style': {'text': f"Updated {pt.get('updated', '')}"}}
        return charts.render_chart(option, height=280)
    except Exception as e:
        print(f"Price target chart failed for {ticker}: {e}")
        return None


def _guard_section(label, fn, default=None):
    """Run one analytics attachment; on failure log and return default so one
    broken section never takes down the whole page."""
    try:
        return fn()
    except Exception:
        print(f"[analytics:{label}] section failed:")
        traceback.print_exc()
        return default


def compute_analytics(ticker):
    try:
        return _compute_analytics_impl(ticker)
    except Exception:
        print(f"[analytics] {ticker}: compute failed:")
        traceback.print_exc()
        return None


def _compute_analytics_impl(ticker: str) -> dict | None:
    """Compute all analytics metrics from cached DB prices."""
    df = get_or_fetch_prices(ticker)
    if df is None or len(df) < 2:
        return None

    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df = df.dropna(subset=['returns'])
    if df.empty:
        return None

    current_price = float(df['close'].iloc[-1])

    # --- Descriptive stats ---
    mean_return  = float(df['returns'].mean())
    std_return   = float(df['returns'].std())
    skewness     = float(df['returns'].skew())
    kurtosis     = float(df['returns'].kurt())

    # --- Return distribution histogram ---
    counts, edges = np.histogram(df['returns'] * 100, bins=80)
    dist_option = charts.cartesian([charts.bar('Daily Returns', (edges[:-1] + edges[1:]) / 2, counts)], xtype='value', xname='Daily Return (%)', yname='Frequency')
    dist_chart = charts.render_chart(dist_option, height=350)

    # --- Rolling 30-day volatility (annualised) ---
    df['rolling_vol'] = df['returns'].rolling(30).std() * np.sqrt(252) * 100
    vol_chart = charts.render_chart(charts.cartesian([charts.line('30d Vol', df.index, df['rolling_vol'], area='rgba(245,158,11,.1)')], yname='Volatility (%)'), height=300)

    # --- Drawdown ---
    df['cummax'] = df['close'].cummax()
    df['drawdown'] = (df['close'] - df['cummax']) / df['cummax'] * 100
    max_drawdown = float(df['drawdown'].min())
    dd_chart = charts.render_chart(charts.cartesian([charts.line('Drawdown', df.index, df['drawdown'], '#ef4444', area='rgba(239,68,68,.15)')], yname='Drawdown (%)'), height=300)

    # --- Rolling Sharpe (252-day, risk-free ≈ 0) ---
    df['rolling_sharpe'] = (
        df['returns'].rolling(252).mean() /
        df['returns'].rolling(252).std()
    ) * np.sqrt(252)
    sharpe_chart = charts.render_chart(charts.cartesian([charts.line('Sharpe', df.index, df['rolling_sharpe'], '#10b981')], yname='Sharpe Ratio', refs=[charts.reference('y', 1, 'Sharpe = 1')]), height=300)

    # --- Seasonality: average monthly return ---
    df['month'] = df.index.month
    monthly_avg = df.groupby('month')['returns'].mean() * 100
    month_labels = ['Jan','Feb','Mar','Apr','May','Jun',
                    'Jul','Aug','Sep','Oct','Nov','Dec']
    colors = ['#10b981' if v >= 0 else '#ef4444' for v in monthly_avg.values]
    season_chart = charts.render_chart(charts.cartesian([{'name': 'Avg Monthly Return', 'type': 'bar', 'data': [{'value': v, 'itemStyle': {'color': c}} for v, c in zip(monthly_avg.values, colors)]}], xtype='category', categories=[month_labels[m - 1] for m in monthly_avg.index], yname='Avg Return (%)'), height=300)

    # --- Beta vs SPY ---
    spy_df = get_or_fetch_prices('SPY')
    beta = None
    beta_chart = None
    if spy_df is not None and not spy_df.empty:
        spy_returns = spy_df['close'].pct_change().dropna()
        merged = pd.concat([df['returns'], spy_returns], axis=1, join='inner')
        merged.columns = ['stock', 'spy']
        if len(merged) > 30:
            cov   = merged['stock'].cov(merged['spy'])
            var   = merged['spy'].var()
            beta  = round(cov / var, 3) if var != 0 else None

            beta_series = [{'name': 'Daily returns', 'type': 'scatter', 'symbolSize': 3, 'itemStyle': {'color': '#6366f1', 'opacity': .5}, 'data': charts.pairs(merged['spy'] * 100, merged['stock'] * 100)}]
            if beta is not None:
                x_vals = np.linspace(merged['spy'].min(), merged['spy'].max(), 100)
                y_vals = beta * x_vals + merged['stock'].mean() - beta * merged['spy'].mean()
                beta_series.append(charts.line(f'β = {beta}', x_vals * 100, y_vals * 100))
            beta_chart = charts.render_chart(charts.cartesian(beta_series, xtype='value', xname='SPY Daily Return (%)', yname=f'{ticker.upper()} Daily Return (%)'), height=350)

    # --- Rolling VaR (95 % & 99 %) + Expected Shortfall ---
    roll_win = 252
    df['var_95'] = df['returns'].rolling(roll_win).quantile(0.05) * 100
    df['var_99'] = df['returns'].rolling(roll_win).quantile(0.01) * 100

    def _rolling_es(series, window, q):
        def _es(arr):
            t = np.percentile(arr, q * 100)
            tail = arr[arr <= t]
            return tail.mean() if len(tail) > 0 else np.nan
        return series.rolling(window).apply(_es, raw=True)

    df['es_95'] = _rolling_es(df['returns'], roll_win, 0.05) * 100

    var_chart = charts.render_chart(charts.cartesian([
        charts.line('VaR 95 %', df.index, df['var_95'].abs(), area='rgba(245,158,11,.08)'),
        charts.line('VaR 99 %', df.index, df['var_99'].abs(), '#f43f5e', area='rgba(244,63,94,.08)'),
        charts.line('ES 95 %', df.index, df['es_95'].abs(), '#6366f1', dash=True)], yname='Potential Daily Loss (%)'))

    # Current VaR figures (latest rolling window)
    latest_var_95 = round(float(df['var_95'].dropna().iloc[-1]), 3) if not df['var_95'].dropna().empty else None
    latest_var_99 = round(float(df['var_99'].dropna().iloc[-1]), 3) if not df['var_99'].dropna().empty else None

    # --- Volume Profile (last 2 years) ---
    vp_cutoff = df.index[-1] - pd.DateOffset(years=2)
    vp_df = df[df.index >= vp_cutoff].copy()
    vp_chart = None
    if len(vp_df) > 30:
        price_min, price_max = vp_df['low'].min(), vp_df['high'].max()
        if price_min == price_max:
            poc_price = None
        else:
            n_bins = 50
            bins = np.linspace(price_min, price_max, n_bins + 1)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            vp_df['_bin'] = pd.cut(vp_df['close'], bins=bins, labels=False)
            vol_by_price = vp_df.groupby('_bin')['volume'].sum().reindex(range(n_bins), fill_value=0)
            vp_colors = [
                '#f59e0b' if bc >= current_price else '#6366f1'
                for bc in bin_centers
            ]
            poc_bin = int(vol_by_price.idxmax())
            poc_price = round(float(bin_centers[poc_bin]), 2)

            vp_option = charts.cartesian([charts.bar('Volume at Price', vol_by_price.values, bin_centers, vp_colors)], xtype='value', xname='Cumulative Volume (2 yr)', yname='Price ($)', refs=[charts.reference('y', current_price, 'Current', '#f59e0b'), charts.reference('y', poc_price, f'POC ${poc_price}', '#6366f1')])
            vp_option['terminalHorizontalBars'] = True
            vp_chart = charts.render_chart(vp_option, height=420)
    else:
        poc_price = None

    # --- Forward estimates (Monte Carlo at multiple horizons) ---
    # Projects prices at 3-month, 6-month, and 1-year horizons using GBM
    # calibrated to the stock's own historical drift and volatility.
    def _mc_forward_estimate(current, mu, sigma, horizon_days, n_sims=1000, seed=42):
        """Compute forward price estimate at horizon_days using Monte Carlo GBM.
        Returns: (median, p5, p25, p75, p95, prob_gain, prob_up20, prob_dn20)"""
        rng = np.random.default_rng(seed)
        shocks = rng.normal(mu, sigma, size=(n_sims, horizon_days))
        paths = current * np.exp(np.cumsum(shocks, axis=1))
        terminal = paths[:, -1]
        return {
            'median': round(float(np.median(terminal)), 2),
            'p5': round(float(np.percentile(terminal, 5)), 2),
            'p25': round(float(np.percentile(terminal, 25)), 2),
            'p75': round(float(np.percentile(terminal, 75)), 2),
            'p95': round(float(np.percentile(terminal, 95)), 2),
            'prob_gain': round(float((terminal > current).mean() * 100), 1),
            'prob_up20': round(float((terminal > current * 1.2).mean() * 100), 1),
            'prob_dn20': round(float((terminal < current * 0.8).mean() * 100), 1),
        }

    forward_estimates = {}
    mc_chart = None
    mc_stats = {}
    log_ret = np.log(df['close'] / df['close'].shift(1)).dropna()
    if len(log_ret) > 30:
        mu = float(log_ret.mean())
        sigma = float(log_ret.std())

        # Calculate estimates at multiple horizons
        forward_estimates['3m'] = _mc_forward_estimate(current_price, mu, sigma, 63)   # ~3 months
        forward_estimates['6m'] = _mc_forward_estimate(current_price, mu, sigma, 126)  # ~6 months
        forward_estimates['1y'] = _mc_forward_estimate(current_price, mu, sigma, 252)  # ~1 year
        mc_stats = forward_estimates['1y'].copy()

        # Full 1-year projection chart
        horizon = 252
        n_sims = 1000
        rng = np.random.default_rng(42)
        shocks = rng.normal(mu, sigma, size=(n_sims, horizon))
        paths = current_price * np.exp(np.cumsum(shocks, axis=1))
        paths = np.hstack([np.full((n_sims, 1), current_price), paths])

        pct = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
        future_dates = pd.bdate_range(df.index[-1], periods=horizon + 1)

        mc_series = []
        # Stack lower bound plus interval width to preserve percentile bands.
        for low, high, label, opacity in [(0, 4, '5–95%', .12), (1, 3, '25–75%', .28)]:
            mc_series.append(charts.line(label + ' lower', future_dates, pct[low], '#6366f1', width=0, stack=label, silent=True))
            mc_series.append(charts.line(label, future_dates, pct[high] - pct[low], '#6366f1', width=0, stack=label, area=f'rgba(99,102,241,{opacity})'))
        for band in mc_series:
            band['tooltip'] = {'show': False, 'trigger': 'none'}
        mc_series.append(charts.line('Median', future_dates, pct[2]))
        refs = [charts.reference('y', current_price, 'Today')]
        for days, label in [(64, '3m'), (127, '6m')]:
            refs.append(charts.reference('x', pd.bdate_range(df.index[-1], periods=days)[-1].strftime('%Y-%m-%d'), label, '#6366f1'))
        mc_option = charts.cartesian(mc_series, yname='Simulated Price ($)', refs=refs)
        mc_option['legend'] = {'data': ['5–95%', '25–75%', 'Median']}
        mc_chart = charts.render_chart(mc_option, height=350)

    return {
        'ticker': ticker.upper(),
        'current_price': current_price,
        'stats': {
            'mean_daily_return': round(mean_return * 100, 4),
            'daily_std': round(std_return * 100, 4),
            'annualised_vol': round(std_return * np.sqrt(252) * 100, 2),
            'skewness': round(skewness, 3),
            'kurtosis': round(kurtosis, 3),
            'max_drawdown': round(max_drawdown, 2),
            'beta': beta,
            'data_points': len(df),
            'var_95': latest_var_95,
            'var_99': latest_var_99,
            'poc_price': poc_price,
            'mc_median':    mc_stats.get('median'),
            'mc_p5':        mc_stats.get('p5'),
            'mc_p95':       mc_stats.get('p95'),
            'mc_prob_gain': mc_stats.get('prob_gain'),
            'mc_prob_up20': mc_stats.get('prob_up20'),
            'mc_prob_dn20': mc_stats.get('prob_dn20'),
        },
        'forward_estimates': forward_estimates,
        'charts': {
            'distribution': dist_chart,
            'volatility': vol_chart,
            'drawdown': dd_chart,
            'sharpe': sharpe_chart,
            'seasonality': season_chart,
            'beta': beta_chart,
            'var': var_chart,
            'volume_profile': vp_chart,
            'monte_carlo': mc_chart,
        }
    }


@app.route('/analytics')
def analytics_page():
    ticker = request.args.get('ticker', '').strip().upper()
    if not ticker:
        return render_template('analytics.html', data=None, ticker='')
    data = compute_analytics(ticker)
    if data is None:
        return render_template('analytics.html', data=None, ticker=ticker,
                               error=f"Could not retrieve data for {ticker}")

    # Attach fundamentals
    data['fundamentals'] = _guard_section(
        "fundamentals", lambda: get_fundamentals(ticker))

    # Attach options smile
    data['charts']['options_smile'] = _guard_section(
        "options_smile",
        lambda: get_options_smile(ticker, data['current_price']))

    # Attach dealer Gamma Exposure (GEX) profile
    gex = _guard_section("gex", lambda: get_gex_profile(ticker, data['current_price']))
    data['charts']['gex'] = gex['chart'] if gex else None
    data['gex'] = gex['stats'] if gex else None

    trade_type = request.args.get('trade_type', 'long_stock')
    if trade_type not in decide.TRADE_TYPES:
        trade_type = 'long_stock'
    data['checklist'] = _guard_section(
        "checklist", lambda: decide.build_checklist(ticker, data, trade_type))

    # Attach insider chart (needs price df)
    price_df = get_or_fetch_prices(ticker)
    if price_df is not None:
        data['charts']['insider'] = _guard_section(
            "insider", lambda: get_insider_chart(ticker, price_df))
        data['insider_summary'] = _guard_section(
            "insider_summary", lambda: get_insider_summary(ticker))
        data['charts']['cumulative_return'] = _guard_section(
            "cumulative_return", lambda: get_cumulative_return_chart(ticker, price_df))

    # Attach analyst price target
    data['charts']['price_target'] = _guard_section(
        "price_target", lambda: get_price_target_chart(ticker, data['current_price']))

    # Machine-learning Buy/Hold/Sell signal (None until a model is trained)
    data['ml'] = _guard_section("ml", lambda: ml.predict(ticker))

    # Attach Institutional Analytics Suite (Microstructure, Macro Conditioning, Higher-Order Greeks, 8-K + Event Study)
    try:
        spy_df = get_or_fetch_prices("SPY", period="2y")
        stock_inst_df = price_df if price_df is not None else get_or_fetch_prices(ticker, period="2y")
        if stock_inst_df is not None and not stock_inst_df.empty:
            benchmark_df = spy_df if spy_df is not None else stock_inst_df
            si_pct, si_dtc = _si_kwargs(ticker)
            si_payload, si_source = _short_interest_for(ticker)
            micro_res = microstructure.get_microstructure_analytics(
                stock_inst_df, short_pct_float=si_pct, days_to_cover=si_dtc)
            macro_res = macro_engine.get_macro_financial_report(stock_inst_df, benchmark_df)
            events_8k = sec_8k.fetch_and_parse_8k_filings(ticker, limit=5)

            # Cumulative Abnormal Return (Market Model) around each 8-K filing date.
            # None is expected (not an error) when the filing is too recent to have
            # a full post-event window, or too close to the start of price history
            # to have a full pre-event estimation window.
            sec_8k_events_with_car = []
            for event in events_8k:
                car_result = event_study.run_event_study(
                    stock_inst_df, benchmark_df, event.filing_date,
                    event_type="SEC_8K", ticker=ticker,
                )
                event_dict = event.__dict__.copy()
                event_dict['car_result'] = car_result.__dict__ if car_result else None
                sec_8k_events_with_car.append(event_dict)

            # Higher-order Greeks & VRP
            atm_iv = (data['gex']['spot'] * 0.01) if (data.get('gex') and data['gex'].get('spot')) else 0.25
            spot_p = data.get('current_price', 100.0)
            ann_vol = (data.get('stats', {}).get('annualised_vol') or 25.0) / 100.0
            hog = derivatives_alpha.compute_higher_order_greeks(
                spot=spot_p,
                strike=spot_p,
                time_to_exp=30.0 / 365.0,
                volatility=max(ann_vol, 0.05),
                risk_free_rate=0.045,
                is_call=True
            )
            vrp_res = derivatives_alpha.calculate_variance_risk_premium(
                atm_implied_vol=max(ann_vol * 1.05, 0.05),
                realized_vol_30d=max(ann_vol, 0.05)
            )

            data['institutional'] = {
                'microstructure': micro_res.__dict__,
                'short_interest': {**(si_payload or {}), 'si_source': si_source},
                'macro_conditioning': {
                    'regime': macro_res.current_regime,
                    'fed_funds_rate': macro_res.fed_funds_rate,
                    'yield_curve_2s10s': macro_res.yield_curve_2s10s_spread,
                    'is_inverted': macro_res.is_yield_curve_inverted,
                    'cpi_yoy': macro_res.cpi_inflation_yoy,
                    'betas': macro_res.regime_conditional_betas,
                    'upside_capture': macro_res.upside_capture_ratio,
                    'downside_capture': macro_res.downside_capture_ratio,
                },
                'higher_order_greeks': hog.__dict__,
                'vrp': vrp_res,
                'sec_8k_events': sec_8k_events_with_car,
            }
    except Exception as e:
        print(f"Error computing institutional analytics for {ticker}: {e}")
        data['institutional'] = None

    # Earnings Event Study — CAR per past print + upcoming-print block.
    data['earnings'] = _guard_section(
        "earnings", lambda: earnings.build_earnings_section(ticker))

    # AI analyst report — reads everything above, including the ML signal
    ai_report_html = ai.generate_report(ticker, data)
    data['ai_report'] = ai_report_html
    if ai_report_html:
        data['ai_sections'] = ai.parse_html_sections(ai_report_html)
    else:
        data['ai_sections'] = None

    return render_template('analytics.html', data=data, ticker=ticker)


@app.route('/api/checklist/<ticker>')
def checklist_api(ticker):
    ticker = ticker.strip().upper()
    trade_type = request.args.get('trade_type', 'long_stock')
    if trade_type not in decide.TRADE_TYPES:
        trade_type = 'long_stock'
    data = compute_analytics(ticker)
    if data is None:
        return jsonify({"error": f"Could not retrieve data for {ticker}"}), 404
    data['fundamentals'] = get_fundamentals(ticker)
    gex = get_gex_profile(ticker, data['current_price'])
    data['gex'] = gex['stats'] if gex else None
    return jsonify(decide.build_checklist(ticker, data, trade_type))


@app.route('/api/analytics/<ticker>')
def analytics_api(ticker):
    data = compute_analytics(ticker)
    if data is None:
        return jsonify({"error": f"Could not retrieve data for {ticker}"}), 404
    # strip chart HTML from JSON response — charts are for the template only
    data.pop('charts', None)
    return jsonify(data)


def _insider_sentiment_chart(sentiment):
    points = [s for s in sentiment if isinstance(s.get('mspr'), (int, float))][-18:]
    if not points:
        return None
    labels = [f"{s['year']}-{str(s['month']).zfill(2)}" for s in points]
    values = [s['mspr'] for s in points]
    return charts.render_chart(charts.cartesian([charts.bar('MSPR', labels, values, ['#10b981' if v >= 0 else '#ef4444' for v in values])], xtype='category', categories=labels, yname='MSPR'), height=300)


def _institutional_holders_chart(holders):
    if not holders:
        return None
    top = holders[:10][::-1]
    option = {'xAxis': {'type': 'value', 'name': 'Shares'}, 'yAxis': {'type': 'category', 'data': [h['holder'] for h in top]},
              'series': [{'name': 'Shares', 'type': 'bar', 'data': [h['shares'] for h in top], 'itemStyle': {'color': '#6366f1'}}]}
    return charts.render_chart(option, height=380)


def _peers_quadrant_chart(peers_data: dict) -> str | None:
    rows = [peers_data['focus'], *peers_data['peers']]
    pts = [r for r in rows if r.get('valuation_pct') is not None and r.get('mom_12_1') is not None]
    if len(pts) < 2:
        return None
    series = [{'name': r['symbol'], 'type': 'scatter', 'data': [[r['valuation_pct'], r['mom_12_1'] * 100]],
               'symbolSize': 14 if r['symbol'] == peers_data['focus']['symbol'] else 9,
               'itemStyle': {'color': '#f59e0b' if r['symbol'] == peers_data['focus']['symbol'] else '#71717a'},
               'label': {'show': True, 'position': 'top', 'formatter': r['symbol']}} for r in pts]
    return charts.render_chart(charts.cartesian(series, xtype='value', xname='Valuation percentile (higher = cheaper)', yname='12-1 momentum %', refs=[charts.reference('y', 0), charts.reference('x', 50)]))


def compute_positioning(ticker: str) -> dict:
    """Assemble market-positioning data from all available free providers.

    Always returns a dict (never None) so the page renders even when every
    provider is unconfigured — each panel reports its own availability.
    """
    symbol = ticker.upper()
    cfg = providers.configured()

    # These provider calls are independent network requests; run them concurrently
    # so a cold-cache positioning load is bounded by the slowest call, not their sum.
    tasks = {
        'valuation':       lambda: providers.finnhub_metrics(symbol),
        'recommendations': lambda: providers.finnhub_recommendations(symbol),
        'sentiment':       lambda: providers.finnhub_insider_sentiment(symbol),
        'transactions':    lambda: providers.get_insider_transactions(symbol),
        'sec_filings':     lambda: providers.sec_recent_filings(symbol, forms=("3", "4", "5")),
        'holders':         lambda: providers.fmp_institutional_holders(symbol),
        'sec_13f':         lambda: providers.sec_recent_filings(symbol, forms=("13F-HR", "13F-HR/A")),
    }
    results = {}
    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        futures = {pool.submit(fn): name for name, fn in tasks.items()}
        for future in futures:
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as e:
                print(f"positioning task {name} failed for {symbol}: {e}")
                results[name] = None

    valuation = results['valuation']
    recommendations = results['recommendations']
    sentiment = results['sentiment']
    transactions = results['transactions']
    sec_filings = results['sec_filings']
    insider = {
        'sentiment': sentiment,
        'transactions': transactions,
        'sec_filings': sec_filings,
        'chart': _insider_sentiment_chart(sentiment) if sentiment else None,
    }

    holders = results['holders']
    sec_13f = results['sec_13f']
    institutional = {
        'holders': holders,
        'sec_filings': sec_13f,
        'chart': _institutional_holders_chart(holders) if holders else None,
    }

    data = {
        'ticker': symbol,
        'configured': cfg,
        'valuation': valuation,
        'recommendations': recommendations,
        'insider': insider,
        'institutional': institutional,
    }

    # Peer Comparison — cache-only percentile table + quadrant scatter.
    try:
        data['peers'] = peers.build_peer_comparison(
            ticker, cap=radar.get_tunable("radar_peer_cap"))
        data['peers_chart'] = _peers_quadrant_chart(data['peers']) if data['peers'] else None
    except Exception as e:
        print(f"peers panel failed for {ticker}: {e}")
        data['peers'] = None
        data['peers_chart'] = None

    return data


@app.route('/positioning')
def positioning_page():
    ticker = request.args.get('ticker', '').strip().upper()
    if not ticker:
        return render_template('positioning.html', data=None, ticker='')
    data = compute_positioning(ticker)
    return render_template('positioning.html', data=data, ticker=ticker)


@app.route('/api/positioning/<ticker>')
def positioning_api(ticker):
    data = compute_positioning(ticker)
    data['insider'].pop('chart', None)
    data['institutional'].pop('chart', None)
    data.pop('peers_chart', None)
    return jsonify(data)


def normal_cdf(x):
    """Cumulative distribution function of standard normal distribution."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_pdf(x):
    """Probability density function of standard normal distribution."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def calculate_greeks(s, k, t, v, r=0.045):
    """Calculate Black-Scholes option pricing and greeks."""
    if t <= 0:
        t = 1e-5
    if v <= 0:
        v = 1e-5
    try:
        d1 = (math.log(s / k) + (r + 0.5 * v * v) * t) / (v * math.sqrt(t))
        d2 = d1 - v * math.sqrt(t)
        
        pdf_d1 = normal_pdf(d1)
        cdf_d1 = normal_cdf(d1)
        cdf_d2 = normal_cdf(d2)
        
        cdf_minus_d1 = normal_cdf(-d1)
        cdf_minus_d2 = normal_cdf(-d2)
        
        # Call Greeks
        call_delta = cdf_d1
        call_theta = (-(s * pdf_d1 * v) / (2 * math.sqrt(t)) - r * k * math.exp(-r * t) * cdf_d2) / 365.0
        call_rho = (k * t * math.exp(-r * t) * cdf_d2) / 100.0
        
        # Put Greeks
        put_delta = cdf_d1 - 1.0
        put_theta = (-(s * pdf_d1 * v) / (2 * math.sqrt(t)) + r * k * math.exp(-r * t) * cdf_minus_d2) / 365.0
        put_rho = (-k * t * math.exp(-r * t) * cdf_minus_d2) / 100.0
        
        # Common Greeks
        gamma = pdf_d1 / (s * v * math.sqrt(t))
        vega = (s * math.sqrt(t) * pdf_d1) / 100.0
        
        return {
            'call_delta': call_delta,
            'call_theta': call_theta,
            'call_rho': call_rho,
            'put_delta': put_delta,
            'put_theta': put_theta,
            'put_rho': put_rho,
            'gamma': gamma,
            'vega': vega,
        }
    except Exception as e:
        print(f"Error calculating greeks: {e}")
        return None


def get_options_greeks_data(ticker, expiration_date=None, rf_rate=0.045, strike_count=None):
    """Retrieve option chain and compute Black-Scholes Greeks for strikes around the spot price.

    Persists the raw option chain (calls/puts records + spot price) through the
    shared chain cache — S3 when S3_CACHE_BUCKET is configured, SQLite api_cache
    otherwise — so the app still works after hours or when yfinance is
    rate-limited. Stale-but-present beats a blank table.

    strike_count controls how much of the chain comes back:
      None    the default +/-30% moneyness band
      'all'   every strike the expiration lists
      int N   the N strikes nearest spot, so the window stays centred on the
              money rather than drifting to one wing on a skewed chain
    """
    try:
        stock = _get_yf_ticker(ticker)
        expirations = get_cached_expirations(ticker, stock)
        if not expirations:
            return None

        if not expiration_date or expiration_date not in expirations:
            expiration_date = expirations[0]

        cached = get_cached_chain(ticker, expiration_date, stock=stock)
        if cached is None:
            return None
        calls, puts, spot_price = cached
        if not spot_price:
            spot_price = _spot_price(ticker.upper(), stock)
        if not spot_price:
            return None
        exp_dt = datetime.strptime(expiration_date, '%Y-%m-%d')
        from zoneinfo import ZoneInfo
        today = datetime.now(ZoneInfo("America/New_York")).replace(tzinfo=None)
        days_to_exp = (exp_dt - today).days + 1
        t_years = max(1e-5, days_to_exp / 365.0)
        
        call_strikes = calls['strike'].tolist() if not calls.empty else []
        put_strikes = puts['strike'].tolist() if not puts.empty else []
        all_strikes = sorted(list(set(call_strikes + put_strikes)))

        if strike_count == 'all':
            filtered_strikes = all_strikes
        elif isinstance(strike_count, int) and strike_count > 0:
            # Nearest-to-spot, then back into ascending order for display.
            nearest = sorted(all_strikes, key=lambda s: abs(s - spot_price))[:strike_count]
            filtered_strikes = sorted(nearest)
        else:
            lower_bound = spot_price * 0.70
            upper_bound = spot_price * 1.30
            filtered_strikes = [s for s in all_strikes if lower_bound <= s <= upper_bound]
        
        call_dict = calls.set_index('strike').to_dict('index') if not calls.empty else {}
        put_dict = puts.set_index('strike').to_dict('index') if not puts.empty else {}
        
        rows = []
        for strike in filtered_strikes:
            c_opt = call_dict.get(strike, {})
            p_opt = put_dict.get(strike, {})
            
            c_iv = c_opt.get('impliedVolatility', 0)
            p_iv = p_opt.get('impliedVolatility', 0)
            
            # Avoid invalid values
            c_iv = c_iv if (c_iv and not np.isnan(c_iv)) else 0
            p_iv = p_iv if (p_iv and not np.isnan(p_iv)) else 0
            
            # Cross-IV fallback: if one side is missing IV but the other side has it,
            # use the other side's IV for Greeks computation (Put-Call parity / arbitrage alignment)
            c_iv_calc = c_iv
            p_iv_calc = p_iv
            if c_iv <= 0.01 and p_iv > 0.01:
                c_iv_calc = p_iv
            if p_iv <= 0.01 and c_iv > 0.01:
                p_iv_calc = c_iv
            
            c_greeks = calculate_greeks(spot_price, strike, t_years, c_iv_calc, rf_rate) if c_iv_calc > 0.01 else None
            p_greeks = calculate_greeks(spot_price, strike, t_years, p_iv_calc, rf_rate) if p_iv_calc > 0.01 else None
            
            c_bid = c_opt.get('bid', 0)
            c_ask = c_opt.get('ask', 0)
            c_last = c_opt.get('lastPrice', 0)
            c_change = c_opt.get('change', 0)
            c_pct_change = c_opt.get('percentChange', 0)

            p_bid = p_opt.get('bid', 0)
            p_ask = p_opt.get('ask', 0)
            p_last = p_opt.get('lastPrice', 0)
            p_change = p_opt.get('change', 0)
            p_pct_change = p_opt.get('percentChange', 0)

            def _safe_num(v):
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return 0
                return v

            rows.append({
                'strike': strike,
                'call_bid': c_bid if not np.isnan(c_bid) else 0,
                'call_ask': c_ask if not np.isnan(c_ask) else 0,
                'call_last': c_last if not np.isnan(c_last) else 0,
                'call_change': round(_safe_num(c_change), 2),
                'call_pct_change': round(_safe_num(c_pct_change), 1),
                'call_volume': int(_safe_num(c_opt.get('volume', 0))),
                'call_oi': int(_safe_num(c_opt.get('openInterest', 0))),
                'call_iv': round(c_iv * 100, 2),
                'call_delta': c_greeks['call_delta'] if c_greeks else 'N/A',
                'call_gamma': c_greeks['gamma'] if c_greeks else 'N/A',
                'call_theta': c_greeks['call_theta'] if c_greeks else 'N/A',
                'call_vega': c_greeks['vega'] if c_greeks else 'N/A',
                'call_rho': c_greeks['call_rho'] if c_greeks else 'N/A',
                'put_bid': p_bid if not np.isnan(p_bid) else 0,
                'put_ask': p_ask if not np.isnan(p_ask) else 0,
                'put_last': p_last if not np.isnan(p_last) else 0,
                'put_change': round(_safe_num(p_change), 2),
                'put_pct_change': round(_safe_num(p_pct_change), 1),
                'put_volume': int(_safe_num(p_opt.get('volume', 0))),
                'put_oi': int(_safe_num(p_opt.get('openInterest', 0))),
                'put_iv': round(p_iv * 100, 2),
                'put_delta': p_greeks['put_delta'] if p_greeks else 'N/A',
                'put_gamma': p_greeks['gamma'] if p_greeks else 'N/A',
                'put_theta': p_greeks['put_theta'] if p_greeks else 'N/A',
                'put_vega': p_greeks['vega'] if p_greeks else 'N/A',
                'put_rho': p_greeks['put_rho'] if p_greeks else 'N/A',
            })
            
        return {
            'ticker': ticker.upper(),
            'expirations': expirations,
            'selected_expiration': expiration_date,
            'spot_price': spot_price,
            'days_to_expiration': days_to_exp,
            'options': rows
        }
    except Exception as e:
        import traceback
        print(f"Error compiling options greeks for {ticker}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None


def get_full_option_chain_df(
    ticker: str,
    stock=None,
    current_price: float | None = None,
    max_expirations: int = 8,
) -> pd.DataFrame:
    """Aggregate cached option chains across expirations into a single normalized DataFrame."""
    try:
        stock = stock or _get_yf_ticker(ticker)
        expirations = get_cached_expirations(ticker, stock)
        if not expirations:
            return pd.DataFrame()

        today = datetime.now()
        selected = []
        for exp in expirations:
            try:
                exp_dt = datetime.strptime(exp, '%Y-%m-%d')
            except ValueError:
                continue
            dte = (exp_dt - today).days
            if dte < 0:
                continue
            selected.append((exp, exp_dt, dte))
            if len(selected) >= max_expirations:
                break

        if not selected:
            return pd.DataFrame()

        def _safe_float(val, default=0.0) -> float:
            try:
                if val is None or pd.isna(val):
                    return default
                return float(val)
            except Exception:
                return default

        def _safe_int(val, default=0) -> int:
            try:
                if val is None or pd.isna(val):
                    return default
                return int(val)
            except Exception:
                return default

        def _fetch(exp_tuple):
            try:
                return exp_tuple, get_cached_chain(ticker, exp_tuple[0], stock=stock, spot_hint=current_price)
            except Exception:
                return exp_tuple, None

        with ThreadPoolExecutor(max_workers=min(len(selected), 8)) as pool:
            fetched = list(pool.map(_fetch, selected))

        rows = []
        for (exp, exp_dt, dte), chain in fetched:
            if chain is None:
                continue
            calls_df, puts_df, _ = chain
            for df_side, cp in ((calls_df, 'C'), (puts_df, 'P')):
                if df_side is None or df_side.empty:
                    continue
                for _, r in df_side.iterrows():
                    strike = r.get('strike')
                    iv = r.get('impliedVolatility')
                    if strike is None or iv is None or pd.isna(strike) or pd.isna(iv):
                        continue
                    rows.append({
                        'strike': float(strike),
                        'cp': cp,
                        'dte': int(dte),
                        'expiration': exp,
                        'bid': _safe_float(r.get('bid', 0.0)),
                        'ask': _safe_float(r.get('ask', 0.0)),
                        'last_price': _safe_float(r.get('lastPrice', 0.0)),
                        'iv': float(iv),
                        'open_interest': _safe_int(r.get('openInterest', 0)),
                        'volume': _safe_int(r.get('volume', 0)),
                        'underlying_price': float(current_price) if current_price else None,
                    })

        return pd.DataFrame(rows) if rows else pd.DataFrame()
    except Exception as e:
        print(f"[options-terminal] Error assembling full chain for {ticker}: {e}")
        return pd.DataFrame()


@app.route('/options')
def options_page():
    ticker = request.args.get('ticker', '').strip().upper() or 'SPY'

    convention = request.args.get('convention', 'naive').strip().lower()
    if convention not in ('naive', 'short_wings'):
        convention = 'naive'

    current_price = get_current_price_yfinance(ticker)
    if not current_price:
        current_price = _spot_price(ticker, _get_yf_ticker(ticker))
    if not current_price:
        return render_template(
            'options.html', ticker=ticker, data=None,
            error=f"Could not determine a current price for {ticker}. Options analytics are unavailable right now.")

    stock = _get_yf_ticker(ticker)
    chains_df = get_full_option_chain_df(ticker, stock=stock, current_price=current_price)
    daily_df = get_or_fetch_prices(ticker)

    terminal_res = compute_options_terminal(
        ticker=ticker,
        spot=current_price,
        chain_df=chains_df,
        daily_df=daily_df,
        convention=convention,
    )

    import dataclasses
    data_dict = dataclasses.asdict(terminal_res)

    return render_template('options.html', ticker=ticker, data=data_dict)


@app.route('/live')
def live_page():
    ticker = request.args.get('ticker', '').strip().upper()
    return render_template('live.html', ticker=ticker)


@app.route('/heatmap')
def heatmap_page():
    """S&P 500 sector treemap. Reads only cached data -- never fetches, so a
    first-ever load renders instantly with partial coverage and fills in as
    the background warmer downloads history."""
    data = heatmap.get_heatmap_payload()
    chart = None
    if data:
        chart = heatmap.render_treemap(data, dark=request.cookies.get('ui_mode') == 'dark')
    return render_template('heatmap.html', data=data, heatmap_chart=chart)


@app.route('/api/heatmap')
def heatmap_api():
    """JSON twin of the heatmap page (same payload, no chart HTML).

    Returns a 200 with `status: "warming"` while the universe is still
    downloading, so the page's poll loop can watch coverage grow."""
    data = heatmap.get_heatmap_payload()
    if data is None:
        return jsonify({
            "status": "warming",
            "tiles": [],
            "sectors": [],
            "coverage": 0,
            "universe": 0,
            "advancers": 0,
            "decliners": 0,
            "unchanged": 0,
            "as_of": None,
        })
    return jsonify({**data, "status": "ok"})


@app.route('/api/config')
def api_config():
    # P0 fix: the raw key used to be served here for the browser-direct
    # Finnhub websocket; any origin could read it via the open CORS policy.
    # Only the boolean ships now -- server-side streaming proxy comes later.
    return jsonify({"has_finnhub": bool(providers.active_finnhub_key())})


# Benchmark / index ETFs — used as comparison series, never ranked as alpha names.
MOMENTUM_BENCHMARK_ETFS = {"SPY", "QQQ", "DIA", "IWM"}
# Risk-free assumption used for Sharpe / Sortino (annualised). Roughly the
# average front-end T-bill yield over the sample; keeps ratios honest rather
# than treating cash as zero-cost.
RF_ANNUAL = momentum_engine.RF_ANNUAL

# Ported to momentum_engine.py verbatim; kept as module-level aliases so
# every existing call site (`_perf_stats(...)`, `_relative_stats(...)`)
# keeps working unchanged.
_perf_stats = momentum_engine.perf_stats
_relative_stats = momentum_engine.relative_stats


def compute_momentum(ticker: str) -> dict:
    try:
        return _compute_momentum_impl(ticker)
    except Exception:
        print(f"[momentum] {ticker}: compute failed:")
        traceback.print_exc()
        return {}


def _compute_momentum_impl(ticker: str) -> dict:
    """Compute momentum scores and backtest statistics for a ticker."""
    symbol = ticker.upper()
    df = db.get_prices(symbol)
    if df is None or len(df) < 273:
        return {}

    close = df['close']
    daily_rets = close.pct_change(fill_method=None)

    score = momentum_engine.score_series(close, symbol=symbol)
    if score is None:
        return {}

    # Backtest stats — 12-1 momentum long/cash trend-following. `relative_strength`
    # has no absolute-return hurdle, so this is "long whenever 12-1 momentum > 0",
    # matching what this function always computed inline.
    bt = momentum_engine.backtest_timeseries(close, strategy_id="relative_strength")
    strat_series = bt.strategy_returns
    hold_series = daily_rets.iloc[253:]
    trade_signals = bt.trade_signal

    strat_stats = _perf_stats(strat_series)
    hold_stats = _perf_stats(hold_series)

    pct_invested = round(float(trade_signals.mean()) * 100, 1) if len(trade_signals) else 0.0

    # Trade records
    trade_records = []
    in_trade = False
    entry_idx = 0
    cost_bps = momentum_engine.DEFAULT_COST_BPS / 1e4
    backtest_start_idx = df.index.get_loc(df.index[253])

    for idx in range(len(trade_signals)):
        sig = trade_signals.iloc[idx]
        if sig == 1 and not in_trade:
            in_trade = True
            entry_idx = idx
        elif sig == 0 and in_trade:
            in_trade = False
            ret_val = close.iloc[backtest_start_idx + idx] / close.iloc[backtest_start_idx + entry_idx] - 1 - cost_bps * 2
            trade_records.append(ret_val)

    if in_trade:
        ret_val = close.iloc[-1] / close.iloc[backtest_start_idx + entry_idx] - 1 - cost_bps
        trade_records.append(ret_val)

    trade_count = len(trade_records)
    wins = [r for r in trade_records if r > 0]
    losses = [r for r in trade_records if r <= 0]

    win_rate = round(len(wins) / trade_count * 100, 1) if trade_count > 0 else 0.0
    profit_factor = round(sum(wins) / abs(sum(losses)), 2) if losses and sum(losses) != 0.0 else (99.0 if wins else 0.0)

    # Relative stats
    spy_df = get_or_fetch_prices("SPY")
    rel = {"alpha": 0.0, "beta": 0.0, "info_ratio": 0.0, "corr": 0.0}
    if spy_df is not None:
        spy_rets = spy_df["close"].pct_change(fill_method=None).dropna()
        merged = pd.concat([strat_series, spy_rets], axis=1, join="inner")
        if len(merged) > 30:
            rel = _relative_stats(merged.iloc[:, 0], merged.iloc[:, 1])

    return {
        "mom_12_1": round(score.mom_12_1 * 100, 2),
        "mom_6m": round(score.mom_6m * 100, 2),
        "mom_3m": round(score.mom_3m * 100, 2),
        "mom_1m": round(score.mom_1m * 100, 2),
        "ann_vol_1y": round(score.ann_vol_1y * 100, 2),
        "risk_adj_mom": score.risk_adj,
        "strat_stats": strat_stats,
        "hold_stats": hold_stats,
        "pct_invested": pct_invested,
        "trade_count": trade_count,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "relative_spy": rel,
    }


# --------------------------------------------------------------------------
# /strategies — shared helpers
#
# The page is three tabs over one universe of cached prices. Each tab splits
# into a data builder (plain data in, plain dict out — no Flask globals, no
# a chart renderer, so it can be exercised without a request context) and a chart
# builder that turns the already-computed series into native chart mounts.
# All scoring and backtest math lives in `momentum_engine`; nothing here
# reimplements it.
# --------------------------------------------------------------------------

DEFAULT_STRATEGY = "relative_strength"


def _resolve_strategy(raw: str) -> str:
    """Validate a `strategy=` query value.

    Anything unrecognized falls back to the default rather than 500ing.
    """
    return raw if raw in momentum_engine.STRATEGIES else DEFAULT_STRATEGY


def _strategy_family(strategy_id: str) -> str:
    """"momentum" or "sma" — the axis every scoring branch dispatches on."""
    return momentum_engine.STRATEGIES.get(strategy_id, {}).get("family", "momentum")


def _strategy_label(strategy_id: str) -> str:
    return momentum_engine.STRATEGIES.get(strategy_id, {}).get("label", strategy_id)


def _strategy_choices() -> dict:
    """Ordered {id: label} mapping backing the strategy <select>."""
    return {sid: meta["label"] for sid, meta in momentum_engine.STRATEGIES.items()}


_PERIOD_OFFSETS = {
    "3y": pd.DateOffset(years=3),
    "1y": pd.DateOffset(years=1),
    "6m": pd.DateOffset(months=6),
    "3m": pd.DateOffset(months=3),
}


def _period_start(latest_date, period, fallback):
    """Cutoff date for a `period=` filter; unknown values fall back."""
    offset = _PERIOD_OFFSETS.get(period)
    return latest_date - offset if offset is not None else fallback


def _load_price_frame(symbols):
    """Wide close-price frame for `symbols` from a single batched query.

    Symbols with no cached rows are absent from the batch — and so from the
    frame's columns. Column order follows `symbols`.

    Callers must pass DB-cased (uppercase) symbols: the lookup is
    case-sensitive and a lowercase symbol silently drops out rather than
    raising, so an all-lowercase list yields an empty (0, 0) frame.
    """
    batch = db.get_prices_batch(symbols)
    return pd.DataFrame({s: batch[s]["close"] for s in symbols if s in batch})


def _universe_rank_scores(price_batch, symbols, strategy_id) -> dict:
    """Latest ranking score per symbol, each from its OWN price history.

    Deliberately not `score_universe`: that indexes a shared date-aligned
    wide frame, so a symbol whose cache lags the universe scores as NaN and
    drops out. The ticker tab's rank has always been computed against each
    symbol's own trailing bars, and stays that way.
    """
    fam = _strategy_family(strategy_id)
    scores = {}
    for sym in symbols:
        sym_df = price_batch.get(sym)
        if sym_df is None:
            continue
        close = sym_df["close"]
        if not len(close):
            continue
        if fam == "sma":
            series = momentum_engine.sma_spread(close)
            if len(series):
                val = series.iloc[-1]
                if pd.notna(val):
                    scores[sym] = float(val)
        elif fam == "rsi":
            sma200 = close.rolling(200).mean()
            r2 = momentum_engine.calc_rsi(close, 2)
            c = close.iloc[-1]
            s = sma200.iloc[-1]
            r = r2.iloc[-1]
            val = (100.0 - r) if (pd.notna(c) and pd.notna(s) and c > s) else -r
            if pd.notna(val):
                scores[sym] = float(val)
        elif fam == "composite":
            series = momentum_engine.composite_score_series(close)
            if len(series):
                val = series.iloc[-1]
                if pd.notna(val):
                    scores[sym] = float(val)
        else:
            series = momentum_engine.rolling_score(close)
            if len(series):
                val = series.iloc[-1]
                if pd.notna(val):
                    scores[sym] = float(val)
    return scores


# --------------------------------------------------------------------------
# /strategies — universe tab
# --------------------------------------------------------------------------

def _deserialize_universe_scores(cached):
    """Rebuild (scores, spreads) from an api_cache payload, or None if malformed."""
    if not isinstance(cached, dict) or not isinstance(cached.get("scores"), dict):
        return None
    try:
        scores = {sym: momentum_engine.MomentumScore(**row)
                  for sym, row in cached["scores"].items()}
    except (TypeError, ValueError):
        return None
    spreads = cached.get("spreads", {})
    return scores, spreads if isinstance(spreads, dict) else {}


def _cached_universe_scores(strategy_id, symbols, price_df):
    """Universe momentum scores, 1h-cached (TTL = daily_prices freshness).

    `score_universe` over ~500 symbols dominates /strategies load time on a
    small shared CPU; scores only change when cached closes do. Returns
    (scores, spreads) — scores as {sym: MomentumScore}, spreads as {sym: float}
    (sma/rsi/composite family only, else {}). Stampede-protected: lock losers wait briefly
    for the winner's payload, then compute anyway (slow beats failing).
    """
    key = f"scores:{strategy_id}"

    def _compute():
        hurdle = momentum_engine.absolute_hurdle(strategy_id)
        scores = momentum_engine.score_universe(price_df, symbols, hurdle=hurdle)
        fam = _strategy_family(strategy_id)
        spreads = {}
        if fam == "sma":
            spreads = momentum_engine.sma_score_universe(price_df, symbols)
        elif fam == "rsi":
            spreads = momentum_engine.rsi_score_universe(price_df, symbols)
        elif fam == "composite":
            spreads = momentum_engine.composite_score_universe(price_df, symbols)
        payload = {
            "scores": {sym: {"symbol": s.symbol, "mom_12_1": s.mom_12_1,
                             "mom_6m": s.mom_6m, "mom_3m": s.mom_3m,
                             "mom_1m": s.mom_1m, "ann_vol_1y": s.ann_vol_1y,
                             "risk_adj": s.risk_adj,
                             "passes_absolute": bool(s.passes_absolute)}
                       for sym, s in scores.items()},
            "spreads": spreads,
        }
        db.cache_set("strategies", key, payload)
        return scores, spreads

    hit = _deserialize_universe_scores(db.cache_get("strategies", key, 1))
    if hit is not None:
        return hit

    if db.try_claim_lock(f"strategies_scores_lock:{strategy_id}", ttl_hours=1.0):
        return _compute()

    for _ in range(6):
        time.sleep(0.5)
        hit = _deserialize_universe_scores(db.cache_get("strategies", key, 1))
        if hit is not None:
            return hit
    return _compute()


def _strategies_universe_data(symbols, price_df, strategy_id, period):
    """Leaderboard + top-N rotation backtest against SPY/QQQ.

    Returns None when `price_df` is too short to score, so the caller can
    render the error page. The returned dict carries the raw return series
    under `_series` for the chart builder; the route pops it before
    rendering.
    """
    if len(price_df) < momentum_engine.MOMENTUM_LOOKBACK + 2:
        return None

    scores, spreads = _cached_universe_scores(strategy_id, symbols, price_df)
    fam = _strategy_family(strategy_id)

    if fam == "sma":
        ranked = sorted(spreads.items(), key=lambda kv: kv[1], reverse=True)
        leaderboard = []
        for idx, (sym, spread) in enumerate(ranked):
            vol = scores[sym].ann_vol_1y if sym in scores else 0.0
            leaderboard.append({
                "symbol": sym,
                "score": round(spread * 100, 2),
                "sma_spread": round(spread * 100, 2),
                "vol": round(vol * 100, 1),
                "risk_adj": round(spread / vol, 2) if vol > 0 else 0.0,
                "passes_absolute": spread > 0,
                "rank": idx + 1,
            })
    elif fam == "rsi":
        ranked = sorted(spreads.items(), key=lambda kv: kv[1], reverse=True)
        leaderboard = []
        for idx, (sym, spread) in enumerate(ranked):
            vol = scores[sym].ann_vol_1y if sym in scores else 0.0
            leaderboard.append({
                "symbol": sym,
                "score": round(spread, 1),
                "vol": round(vol * 100, 1),
                "risk_adj": round(spread / (vol * 100), 2) if vol > 0 else 0.0,
                "passes_absolute": spread > 0,
                "rank": idx + 1,
            })
    elif fam == "composite":
        ranked = sorted(spreads.items(), key=lambda kv: kv[1], reverse=True)
        leaderboard = []
        for idx, (sym, c_score) in enumerate(ranked):
            vol = scores[sym].ann_vol_1y if sym in scores else 0.0
            leaderboard.append({
                "symbol": sym,
                "score": round(c_score, 1),
                "vol": round(vol * 100, 1),
                "risk_adj": round(c_score / (vol * 100), 2) if vol > 0 else 0.0,
                "passes_absolute": c_score >= 50.0,
                "rank": idx + 1,
            })
    else:
        ranked = sorted(scores.items(), key=lambda kv: kv[1].mom_12_1, reverse=True)
        leaderboard = [{
            "symbol": sym,
            "score": round(score.mom_12_1 * 100, 2),
            "vol": round(score.ann_vol_1y * 100, 1),
            "risk_adj": score.risk_adj,
            "passes_absolute": bool(score.passes_absolute),
            "rank": idx + 1,
        } for idx, (sym, score) in enumerate(ranked)]

    top_5 = [sym for sym, _ in ranked[:momentum_engine.TOP_N]]

    result = momentum_engine.backtest_rotation(price_df, symbols, strategy_id)
    backtest_dates = result.dates
    strat_series = result.strategy_returns

    daily_rets = price_df.pct_change(fill_method=None)
    has_spy = "SPY" in daily_rets.columns
    has_qqq = "QQQ" in daily_rets.columns
    spy_series = daily_rets["SPY"].loc[backtest_dates] if has_spy else pd.Series(0.0, index=backtest_dates)
    qqq_series = daily_rets["QQQ"].loc[backtest_dates] if has_qqq else pd.Series(0.0, index=backtest_dates)

    if period != "all":
        start_cutoff = _period_start(price_df.index[-1], period, backtest_dates[0])
        mask = backtest_dates >= start_cutoff
        if mask.any() and mask.sum() >= 10:
            backtest_dates = backtest_dates[mask]
            strat_series = strat_series[mask]
            if has_spy:
                spy_series = spy_series[mask]
            if has_qqq:
                qqq_series = qqq_series[mask]

    strat_stats = _perf_stats(strat_series)
    spy_stats = _perf_stats(spy_series)
    qqq_stats = _perf_stats(qqq_series)
    rel_spy = _relative_stats(strat_series, spy_series)

    excess_spy = round(strat_stats["total_return"] - spy_stats["total_return"], 1)
    beat_spy = strat_stats["total_return"] > spy_stats["total_return"]
    beat_qqq = strat_stats["total_return"] > qqq_stats["total_return"]
    if beat_spy and beat_qqq:
        verdict = "outperformed both benchmarks"
    elif beat_spy or beat_qqq:
        verdict = "beat the S&P 500 but trailed the Nasdaq 100" if beat_spy else "beat the Nasdaq 100 but trailed the S&P 500"
    else:
        verdict = "underperformed both benchmarks"

    return {
        "leaderboard": leaderboard,
        "top_5": top_5,
        "strat_stats": strat_stats,
        "spy_stats": spy_stats,
        "qqq_stats": qqq_stats,
        "rel_spy": rel_spy,
        "excess_spy": excess_spy,
        "verdict": verdict,
        "avg_turnover": result.avg_turnover_pct,
        "rf_annual": round(RF_ANNUAL * 100, 1),
        "cost_bps": momentum_engine.DEFAULT_COST_BPS,
        "start_date": backtest_dates[0].strftime('%Y-%m-%d'),
        "end_date": backtest_dates[-1].strftime('%Y-%m-%d'),
        "_series": {
            "dates": backtest_dates,
            "strat": strat_series,
            "spy": spy_series if has_spy else None,
            "qqq": qqq_series if has_qqq else None,
        },
    }


def _strategies_universe_chart(dates, strat_series, spy_series, qqq_series, strategy_label) -> str:
    series = [charts.line(strategy_label, dates, (1 + strat_series).cumprod() * 10000)]
    for values, label, color in [(spy_series, 'SPY (S&P 500) Benchmark', '#64748b'), (qqq_series, 'QQQ (Nasdaq 100) Benchmark', '#818cf8')]:
        if values is not None:
            series.append(charts.line(label, dates, (1 + values).cumprod() * 10000, color, dash=True))
    return charts.render_chart(charts.cartesian(series, yname='Portfolio Value ($)', title='Growth of $10,000 Investment'), height=400)


# --------------------------------------------------------------------------
# /strategies — ticker tab
# --------------------------------------------------------------------------

def _strategies_ticker_data(symbol, df, price_batch, symbols, strategy_id, period):
    """Single-ticker long/cash trend or mean-reversion backtest plus its universe rank."""
    close = df['close']
    daily_rets = close.pct_change(fill_method=None)
    fam = _strategy_family(strategy_id)
    is_sma = fam == "sma"
    is_rsi = fam == "rsi"
    is_composite = fam == "composite"
    hurdle = momentum_engine.absolute_hurdle(strategy_id)

    score = momentum_engine.score_series(close, symbol=symbol)
    if score is None:
        return None

    rank_scores = _universe_rank_scores(price_batch, symbols, strategy_id)
    ranked = sorted(rank_scores.items(), key=lambda kv: kv[1], reverse=True)
    univ_ranks = {sym: idx + 1 for idx, (sym, _) in enumerate(ranked)}
    rank = univ_ranks.get(symbol, len(symbols))

    # The signal series the strategy trades on, and the threshold it crosses.
    # Keep threshold units stable in chart payloads.
    if is_sma:
        signal_series = momentum_engine.sma_spread(close)
        threshold = 0
    elif is_rsi:
        signal_series = momentum_engine.calc_rsi(close, 2)
        threshold = 15.0
    elif is_composite:
        signal_series = momentum_engine.composite_score_series(close)
        threshold = 55.0
    else:
        signal_series = momentum_engine.rolling_score(close)
        threshold = 0 if hurdle is None else hurdle

    result = momentum_engine.backtest_timeseries(close, strategy_id)
    backtest_dates = result.dates
    strat_series = result.strategy_returns
    trade_signals = result.trade_signal
    warmup = momentum_engine.MOMENTUM_LOOKBACK + 1
    hold_series = daily_rets.iloc[warmup:]

    start_cutoff = None
    if period != 'all':
        start_cutoff = _period_start(df.index[-1], period, backtest_dates[0])
        mask = backtest_dates >= start_cutoff
        if mask.any() and mask.sum() >= 10:
            backtest_dates = backtest_dates[mask]
            strat_series = strat_series[mask]
            hold_series = hold_series[mask]
            trade_signals = trade_signals[mask]

    strat_stats = _perf_stats(strat_series)
    hold_stats = _perf_stats(hold_series)

    pct_invested = round(float(trade_signals.mean()) * 100, 1) if len(trade_signals) else 0.0
    cum_strat = (1 + strat_series).cumprod() * 10000
    cum_hold = (1 + hold_series).cumprod() * 10000

    hold_full_dates = df.index
    hold_full_rets = daily_rets.fillna(0.0)
    if start_cutoff is not None:
        mask_hold_full = hold_full_dates >= start_cutoff
        if mask_hold_full.any() and mask_hold_full.sum() >= 10:
            hold_full_dates = hold_full_dates[mask_hold_full]
            hold_full_rets = hold_full_rets[mask_hold_full]

    sig_diff = trade_signals.diff().fillna(0.0)
    buy_dates = backtest_dates[sig_diff == 1]
    sell_dates = backtest_dates[sig_diff == -1]

    # Individual round-trip trade returns, charged entry + exit cost.
    cost_bps = momentum_engine.DEFAULT_COST_BPS / 1e4
    trade_records = []
    trade_history = []
    in_trade = False
    entry_idx = 0
    backtest_start_idx = df.index.get_loc(backtest_dates[0])
    for idx in range(len(trade_signals)):
        sig = trade_signals.iloc[idx]
        if sig == 1 and not in_trade:
            in_trade = True
            entry_idx = idx
        elif sig == 0 and in_trade:
            in_trade = False
            e_price = float(close.iloc[backtest_start_idx + entry_idx])
            x_price = float(close.iloc[backtest_start_idx + idx])
            t_ret = (x_price / e_price - 1 - cost_bps * 2)
            trade_records.append(t_ret)
            trade_history.append({
                "entry_date": backtest_dates[entry_idx].strftime('%Y-%m-%d'),
                "exit_date": backtest_dates[idx].strftime('%Y-%m-%d'),
                "entry_price": round(e_price, 2),
                "exit_price": round(x_price, 2),
                "return_pct": round(t_ret * 100, 2),
                "duration_days": int((backtest_dates[idx] - backtest_dates[entry_idx]).days),
                "is_win": t_ret > 0,
            })
    if in_trade:
        e_price = float(close.iloc[backtest_start_idx + entry_idx])
        curr_price = float(close.iloc[-1])
        t_ret = (curr_price / e_price - 1 - cost_bps)
        trade_records.append(t_ret)

    trade_count = len(trade_records)
    wins = [r for r in trade_records if r > 0]
    losses = [r for r in trade_records if r <= 0]
    win_rate = round(len(wins) / trade_count * 100, 1) if trade_count > 0 else 0.0
    profit_factor = round(sum(wins) / abs(sum(losses)), 2) if losses and sum(losses) != 0.0 else (99.0 if wins else 0.0)

    # Current position & tactical cockpit details
    curr_sig = trade_signals.iloc[-1] if len(trade_signals) else 0.0
    in_position = bool(curr_sig > 0)
    position_days = 0
    for i in range(len(trade_signals) - 1, -1, -1):
        if trade_signals.iloc[i] == curr_sig:
            position_days += 1
        else:
            break

    entry_price = None
    entry_date_str = None
    unrealized_pct = None
    if in_position and len(trade_signals):
        e_idx = max(0, len(trade_signals) - position_days)
        entry_date_str = backtest_dates[e_idx].strftime('%Y-%m-%d')
        entry_price = round(float(close.loc[backtest_dates[e_idx]]), 2)
        unrealized_pct = round(((float(close.iloc[-1]) / entry_price) - 1) * 100, 2)

    # Rolling signal series for the score chart
    roll_dates = df.index[warmup:]
    roll_series = signal_series.iloc[warmup:]
    if fam in ("sma", "momentum"):
        roll_series = roll_series * 100
    if start_cutoff is not None:
        mask_roll = roll_dates >= start_cutoff
        if mask_roll.any():
            roll_dates = roll_dates[mask_roll]
            roll_series = roll_series[mask_roll]

    # Engine status details
    p_last = float(close.iloc[-1])
    if is_sma:
        sp_now = float(momentum_engine.sma_spread(close).iloc[-1]) * 100
        engine_metric_label = "50/200 SMA Spread"
        engine_metric_val = f"{sp_now:+.2f}%"
        engine_metric_status = "Golden Cross Active (Long)" if sp_now > 0 else "Death Cross Active (Cash)"
        trend_state = "GOLDEN CROSS (Long)" if in_position else "DEATH CROSS (Cash)"
    elif is_rsi:
        r2_now = float(momentum_engine.calc_rsi(close, 2).iloc[-1])
        s200_now = float(close.rolling(200).mean().iloc[-1])
        engine_metric_label = "Connors RSI(2)"
        engine_metric_val = f"{r2_now:.1f}"
        engine_metric_status = (
            f"Oversold Entry (<15) · 200 SMA: ${s200_now:.2f}"
            if in_position else f"Neutral Pullback Zone · 200 SMA: ${s200_now:.2f}"
        )
        trend_state = "OVERSOLD PULLBACK (Long)" if in_position else "WAITING FOR DIP (Cash)"
    elif is_composite:
        c_now = float(momentum_engine.composite_score_series(close).iloc[-1])
        engine_metric_label = "Composite Confluence"
        engine_metric_val = f"{c_now:.0f} / 100"
        engine_metric_status = (
            "Bullish Confluence Active (≥55)" if in_position else "Defensive Regimes (<45)"
        )
        trend_state = "CONFLUENCE BUY (Long)" if in_position else "DEFENSIVE CASH (Flat)"
    else:
        m_now = score.mom_12_1 * 100
        engine_metric_label = "12-1 Momentum"
        engine_metric_val = f"{m_now:+.2f}%"
        engine_metric_status = (
            f"Risk-Adj: {score.risk_adj} · Vol: {score.ann_vol_1y * 100:.1f}%"
        )
        trend_state = "BULLISH (Long)" if in_position else "BEARISH (Flat/Cash)"

    trend_color = "text-emerald-500" if in_position else "text-amber-500"

    # Compute SPY benchmark stats for comparison
    spy_df = price_batch.get("SPY")
    if spy_df is not None and len(spy_df):
        spy_close = spy_df["close"]
        spy_sub = spy_close.pct_change(fill_method=None).reindex(backtest_dates).fillna(0.0)
        rel_spy = momentum_engine.relative_stats(strat_series, spy_sub)
    else:
        rel_spy = {"alpha": 0.0, "beta": 1.0, "info_ratio": 0.0, "corr": 1.0}

    # Moving averages for the chart
    sma50 = close.rolling(50).mean().reindex(backtest_dates)
    sma200 = close.rolling(200).mean().reindex(backtest_dates)
    sma5 = close.rolling(5).mean().reindex(backtest_dates)
    close_sub = close.reindex(backtest_dates)

    return {
        "symbol": symbol,
        "latest_price": p_last,
        "mom_12_1": round(score.mom_12_1 * 100, 2),
        "mom_6m": round(score.mom_6m * 100, 2),
        "mom_3m": round(score.mom_3m * 100, 2),
        "mom_1m": round(score.mom_1m * 100, 2),
        "rank": rank,
        "total_rank_count": len(rank_scores),
        "trend_state": trend_state,
        "trend_color": trend_color,
        "in_position": in_position,
        "position_days": position_days,
        "entry_date": entry_date_str,
        "entry_price": entry_price,
        "unrealized_pct": unrealized_pct,
        "engine_metric_label": engine_metric_label,
        "engine_metric_val": engine_metric_val,
        "engine_metric_status": engine_metric_status,
        "strat_stats": strat_stats,
        "hold_stats": hold_stats,
        "trade_count": trade_count,
        "wins_count": len(wins),
        "losses_count": len(losses),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "risk_adj_mom": score.risk_adj,
        "ann_vol_1y": round(score.ann_vol_1y * 100, 1),
        "pct_invested": pct_invested,
        "rel_spy": rel_spy,
        "trade_history": trade_history[-10:],
        "start_date": backtest_dates[0].strftime('%Y-%m-%d'),
        "end_date": backtest_dates[-1].strftime('%Y-%m-%d'),
        "_series": {
            "dates": backtest_dates,
            "cum_strat": cum_strat,
            "cum_hold": cum_hold,
            "hold_full_dates": hold_full_dates,
            "cum_hold_full": (1 + hold_full_rets).cumprod() * 10000,
            "buy_dates": buy_dates,
            "sell_dates": sell_dates,
            "roll_dates": roll_dates,
            "roll_series": roll_series,
            "threshold": threshold,
            "close": close_sub,
            "sma50": sma50,
            "sma200": sma200,
            "sma5": sma5,
            "trade_signals": trade_signals,
        },
    }


def _strategies_ticker_charts(symbol, series, strategy_id) -> dict:
    """Native synchronized price/equity/drawdown canvas and indicator diagnostic."""
    dates = series['dates']
    cum_strat, cum_hold, close = series['cum_strat'], series['cum_hold'], series['close']
    fam = _strategy_family(strategy_id)
    strat_label = momentum_engine.STRATEGIES.get(strategy_id, {}).get('label', strategy_id)
    performance = [charts.line(strat_label, dates, cum_strat),
                   charts.line(f'Buy & Hold {symbol}', series['hold_full_dates'], series['cum_hold_full'], '#64748b', dash=True)]
    price = [charts.line(f'{symbol} Price', dates, close, '#38bdf8')]
    for key, label, color in ([('sma200', '200-day SMA (Regime)', '#a855f7'), ('sma5', '5-day SMA (Exit)', '#34d399')] if fam == 'rsi' else
                              [('sma50', '50-day SMA', '#fbbf24'), ('sma200', '200-day SMA', '#a855f7')]):
        price.append(charts.line(label, dates, series[key], color, width=1.3))
    for key, label, color, rotate in [('buy_dates', 'Buy Entry', '#10b981', 0), ('sell_dates', 'Exit to Cash', '#ef4444', 180)]:
        if not series[key].empty:
            marker = {'name': label, 'type': 'scatter', 'symbol': 'triangle', 'symbolRotate': rotate, 'symbolSize': 10, 'itemStyle': {'color': color}}
            performance.append({**marker, 'data': charts.pairs(series[key], cum_strat.loc[series[key]])})
            price.append({**marker, 'data': charts.pairs(series[key], close.reindex(series[key]))})
    dd_strat = (cum_strat - cum_strat.cummax()) / cum_strat.cummax() * 100
    dd_hold = (cum_hold - cum_hold.cummax()) / cum_hold.cummax() * 100
    drawdown = [charts.line('Strategy Drawdown (%)', dates, dd_strat, area='rgba(251,191,36,.1)'),
                charts.line(f'{symbol} Drawdown (%)', dates, dd_hold, '#ef4444', dash=True)]
    canvas_series = price + [charts.line('Strategy ($10k)', dates, cum_strat, xAxisIndex=1, yAxisIndex=1),
                             charts.line(f'Buy & Hold {symbol} ($10k)', dates, cum_hold, '#94a3b8', dash=True, xAxisIndex=1, yAxisIndex=1)]
    canvas_series += [{**line, 'xAxisIndex': 1, 'yAxisIndex': 2} for line in drawdown]
    canvas = {'series': canvas_series,
              'grid': [{'left': 55, 'right': 65, 'top': 60, 'height': '43%', 'containLabel': True},
                       {'left': 55, 'right': 65, 'top': '61%', 'height': '27%', 'containLabel': True}],
              'xAxis': [{'type': 'time', 'gridIndex': 0}, {'type': 'time', 'gridIndex': 1}],
              'yAxis': [{'type': 'value', 'name': f'{symbol} Price ($)', 'scale': True, 'gridIndex': 0},
                        {'type': 'value', 'name': 'Portfolio ($10k)', 'scale': True, 'gridIndex': 1},
                        {'type': 'value', 'name': 'Drawdown (%)', 'min': -100, 'max': 5, 'position': 'right', 'gridIndex': 1, 'splitLine': {'show': False}}],
              'axisPointer': {'link': [{'xAxisIndex': 'all'}]},
              'dataZoom': [{'type': 'inside', 'xAxisIndex': [0, 1], 'filterMode': 'none'},
                           {'type': 'slider', 'xAxisIndex': [0, 1], 'height': 15, 'bottom': 10}]}
    if fam == 'sma':
        score_name, threshold_text = 'SMA 50/200 Spread %', 'Golden Cross Line (0%)'
    elif fam == 'rsi':
        score_name, threshold_text = 'Connors RSI(2)', 'Oversold Entry (15)'
    elif fam == 'composite':
        score_name, threshold_text = 'Composite Score', 'Bullish Entry (55)'
    else:
        score_name = '12-1 Momentum %'
        threshold_text = 'Zero Threshold (Trend Switch)' if series['threshold'] == 0 else 'Absolute Hurdle (Trend Switch)'
    threshold = series['threshold'] * (100 if fam in ('sma', 'momentum') else 1)
    refs = [charts.reference('y', threshold, threshold_text, '#ef4444')]
    if fam == 'rsi':
        refs.append(charts.reference('y', 70, 'Overbought Exit (70)', '#10b981'))
    elif fam == 'composite':
        refs.append(charts.reference('y', 45, 'Defensive Exit (45)', '#ef4444'))
    rolling = [charts.line(score_name, series['roll_dates'], series['roll_series'], '#818cf8')]
    return {'canvas_chart_html': charts.render_chart(canvas, height=620),
            'perf_chart_html': charts.render_chart(charts.cartesian(performance, yname='Portfolio Value ($)'), height=350),
            'dd_chart_html': charts.render_chart(charts.cartesian(drawdown, yname='Drawdown (%)'), height=250),
            'roll_chart_html': charts.render_chart(charts.cartesian(rolling, yname=score_name, refs=refs), height=280)}


# --------------------------------------------------------------------------
# /strategies — screener tab
# --------------------------------------------------------------------------

def _strategies_screener_data(symbols, price_df, filters, strategy_id):
    """Filtered, risk-adjusted-ranked table of the cached universe.

    Returns None when `price_df` is too short to score.
    """
    if len(price_df) < momentum_engine.MOMENTUM_LOOKBACK + momentum_engine.MOMENTUM_EXCLUDE:
        return None

    fam = _strategy_family(strategy_id)
    is_sma = fam == "sma"
    is_rsi = fam == "rsi"
    is_composite = fam == "composite"
    scores, spreads = _cached_universe_scores(strategy_id, symbols, price_df)

    results = []
    for sym in symbols:
        score = scores.get(sym)
        if score is None:
            continue

        score_pct = score.mom_12_1 * 100
        vol_pct = score.ann_vol_1y * 100
        risk_adj = score.mom_12_1 / score.ann_vol_1y if score.ann_vol_1y > 0 else 0.0

        if score_pct < filters["min_mom"]:
            continue
        if vol_pct > filters["max_vol"]:
            continue
        if risk_adj < filters["min_risk_adj"]:
            continue

        if is_sma:
            spread = spreads.get(sym)
            if spread is None:
                continue
            passes_absolute = spread > 0
            trend = "BULLISH" if spread > 0 else "BEARISH"
        elif is_rsi:
            spread = spreads.get(sym)
            if spread is None:
                continue
            passes_absolute = spread > 0
            trend = "BULLISH" if spread > 0 else "BEARISH"
        elif is_composite:
            spread = spreads.get(sym)
            if spread is None:
                continue
            passes_absolute = spread >= 50.0
            trend = "BULLISH" if spread >= 50.0 else "BEARISH"
        else:
            passes_absolute = bool(score.passes_absolute)
            trend = "BULLISH" if score_pct > 0 else "BEARISH"

        if filters["trend_filter"] == 'bullish' and trend != 'BULLISH':
            continue
        if filters["trend_filter"] == 'bearish' and trend != 'BEARISH':
            continue

        if filters["abs_only"] and not passes_absolute:
            continue

        row = {
            "symbol": sym,
            "score": round(score_pct, 2),
            "vol": round(vol_pct, 1),
            "risk_adj": round(risk_adj, 2),
            "trend": trend,
            "price": round(float(price_df[sym].iloc[-1]), 2),
            "passes_absolute": passes_absolute,
        }
        if is_sma:
            row["sma_spread"] = round(spread * 100, 2)
        elif is_rsi or is_composite:
            row["strategy_score"] = round(spread, 1)
        results.append(row)

    results = sorted(results, key=lambda r: r["risk_adj"], reverse=True)
    for idx, row in enumerate(results):
        row["rank"] = idx + 1

    return {
        "results": results,
        "min_mom": filters["min_mom"],
        "max_vol": filters["max_vol"],
        "min_risk_adj": filters["min_risk_adj"],
        "trend_filter": filters["trend_filter"],
        "abs_only": filters["abs_only"],
        "total_screened": len(results),
    }


def _safe_float(raw, default):
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


@app.route('/strategies')
@app.route('/momentum')
def strategies_page():
    # `ticker` is accepted alongside `symbol` so links from the other pages
    # (which all carry ?ticker=) land here directly; arriving with a symbol
    # implies the single-ticker view rather than the universe leaderboard.
    raw_symbol = request.args.get('symbol') or request.args.get('ticker') or ''
    symbol = raw_symbol.upper().strip()
    tab = request.args.get('tab', 'ticker' if symbol else 'universe')
    period = request.args.get('period', 'all')
    strategy_id = _resolve_strategy(request.args.get('strategy', DEFAULT_STRATEGY))

    shared = {
        "strategy_id": strategy_id,
        "strategy_label": _strategy_label(strategy_id),
        "strategies": _strategy_choices(),
        "period": period,
    }

    with db.get_conn() as conn:
        rows = conn.execute("SELECT DISTINCT symbol FROM daily_prices").fetchall()
    symbols = [r["symbol"] for r in rows if r["symbol"] not in MOMENTUM_BENCHMARK_ETFS]
    # An empty cache only blocks the universe-derived tabs; the ticker tab
    # can bootstrap itself by downloading the requested symbol below.
    if not symbols and tab != 'ticker':
        return render_template('strategies.html', data=None, error="No stock price data available in the database. Please visit the homepage and search for tickers first.", **shared)

    available_symbols = sorted(symbols)

    if tab == 'ticker':
        if not symbol:
            symbol = available_symbols[0] if available_symbols else ""
        if not symbol:
            return render_template('strategies.html', data=None, tab='ticker', available_symbols=[], searched_symbol='', error="No stock price data available in the database. Please scan a ticker to download its history.", **shared)

        # Auto-download on cache miss: get_or_fetch_prices pulls the ticker's
        # history from yfinance and stores it in daily_prices, so scanning an
        # uncached symbol (benchmark ETFs included — they never enter the
        # universe list below) works without visiting the homepage first.
        # On fetch failure it falls back to stale rows, keeping the errors
        # below as the last resort.
        df = get_or_fetch_prices(symbol, period="5y")
        min_bars = momentum_engine.MOMENTUM_LOOKBACK + momentum_engine.MOMENTUM_EXCLUDE
        if df is None or df.empty:
            return render_template(
                'strategies.html',
                data=None,
                tab='ticker',
                available_symbols=available_symbols,
                searched_symbol=symbol,
                error=f"Ticker '{symbol}' is not cached and automatic download failed. Double-check the symbol and try again.",
                **shared
            )
        if len(df) < min_bars:
            return render_template(
                'strategies.html',
                data=None,
                tab='ticker',
                available_symbols=available_symbols,
                searched_symbol=symbol,
                error=f"Ticker '{symbol}' has insufficient price history (need at least {min_bars} trading days).",
                **shared
            )

        # Benchmark ETFs are excluded from `symbols`, so add the scanned
        # ticker back for the rank — a SPY scan should be ranked against the
        # universe + itself, not pinned to the unranked floor value.
        rank_symbols = symbols if symbol in symbols else symbols + [symbol]
        price_batch = db.get_prices_batch(rank_symbols)
        data = _strategies_ticker_data(symbol, df, price_batch, rank_symbols, strategy_id, period)
        if data is None:
            return render_template(
                'strategies.html',
                data=None,
                tab='ticker',
                available_symbols=available_symbols,
                searched_symbol=symbol,
                error=f"Ticker '{symbol}' could not be scored for this strategy.",
                **shared
            )
        data.update(_strategies_ticker_charts(symbol, data.pop("_series"), strategy_id))

        return render_template(
            'strategies.html',
            data=data,
            tab='ticker',
            header_badge=f"{symbol} · Rank #{data['rank']} of {data['total_rank_count']}",
            available_symbols=available_symbols,
            searched_symbol=symbol,
            error=None,
            **shared
        )

    if tab == 'screener':
        filters = {
            "min_mom": _safe_float(request.args.get('min_mom'), 0.0),
            "max_vol": _safe_float(request.args.get('max_vol'), 60.0),
            "min_risk_adj": _safe_float(request.args.get('min_risk_adj'), 0.5),
            "trend_filter": request.args.get('trend', 'bullish'),
            "abs_only": request.args.get('abs_only', '') == '1',
        }

        price_df = _load_price_frame(symbols)
        if price_df.empty:
            return render_template(
                'strategies.html',
                data=None,
                tab='screener',
                error="No stock data available in database.",
                available_symbols=available_symbols,
                searched_symbol='',
                **shared
            )

        data = _strategies_screener_data(symbols, price_df, filters, strategy_id)
        if data is None:
            return render_template(
                'strategies.html',
                data=None,
                tab='screener',
                error="Insufficient price history in database to run screener.",
                available_symbols=available_symbols,
                searched_symbol='',
                **shared
            )

        return render_template(
            'strategies.html',
            data=data,
            tab='screener',
            header_badge=f"Screened {data['total_screened']} Tickers",
            available_symbols=available_symbols,
            searched_symbol='',
            error=None,
            **shared
        )

    # ELSE: tab == 'universe'
    price_df = _load_price_frame(symbols + ["SPY", "QQQ"])
    if price_df.empty:
        return render_template('strategies.html', data=None, error="Failed to load price data.", **shared)

    data = _strategies_universe_data(symbols, price_df, strategy_id, period)
    if data is None:
        return render_template('strategies.html', data=None, error=f"Insufficient history in database. Need at least {momentum_engine.MOMENTUM_LOOKBACK + 2} daily bars.", **shared)

    series = data.pop("_series")
    data["chart_html"] = _strategies_universe_chart(
        series["dates"], series["strat"], series["spy"], series["qqq"],
        shared["strategy_label"],
    )

    return render_template(
        'strategies.html',
        data=data,
        tab='universe',
        header_badge=f"Sharpe {data['strat_stats']['sharpe']} · IR {data['rel_spy']['info_ratio']} vs SPY",
        available_symbols=available_symbols,
        searched_symbol='',
        error=None,
        **shared
    )


@app.route('/radar')
def radar_page():
    """Universe scanners over warmed caches: drift / value / squeeze."""
    tab = request.args.get('tab', 'drift')
    if tab not in {'drift', 'value', 'squeeze'}:
        tab = 'drift'
    snap = radar.current_snapshot()
    meta = (snap or {}).get("meta") or {}
    warmed_count = int(meta.get("warmed_count") or 0)
    payload = snap if warmed_count else None
    warming = snap is None or warmed_count == 0
    tunables = {key: radar.get_tunable(key) for key in radar.TUNABLES}
    updated_pt = None
    excluded_summary = []
    if payload:
        try:
            from zoneinfo import ZoneInfo
            dt = datetime.fromisoformat(meta.get("published_at") or "").replace(
                tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo("America/Los_Angeles"))
            updated_pt = f"{dt:%b} {dt.day}, {dt.hour % 12 or 12}:{dt:%M %p %Z}"
        except ValueError:
            updated_pt = None
        section = payload.get(tab) or {}
        counts = {}
        for entry in section.get("excluded") or []:
            counts[entry["reason"]] = counts.get(entry["reason"], 0) + 1
        # Drift/squeeze only see symbols that have an earnings/SI payload;
        # the rest of the universe never reaches their exclusion ledgers.
        if tab in _RADAR_UNCOVERED_REASON:
            uncovered = (int(meta.get("universe_count") or 0)
                         - len(section.get("rows") or []) - sum(counts.values()))
            if uncovered > 0:
                counts[_RADAR_UNCOVERED_REASON[tab]] = uncovered
        excluded_summary = [(n, _RADAR_EXCLUDED_LABELS.get(r, r))
                            for r, n in sorted(counts.items(),
                                               key=lambda kv: -kv[1])]
    return render_template('radar.html', tab=tab, payload=payload,
                           warming=warming, tunables=tunables,
                           updated_pt=updated_pt,
                           excluded_summary=excluded_summary)


_RADAR_UNCOVERED_REASON = {"drift": "no-earnings", "squeeze": "assumed-si"}
_RADAR_EXCLUDED_LABELS = {
    "no-earnings": "no earnings data",
    "bad-date": "unreadable earnings date",
    "no-prices": "no price history",
    "no-aligned-session": "print not yet traded",
    "no-fundamentals": "no fundamentals data",
    "insufficient-history": "too little price history",
    "below-value-cap": "outside the top 50",
    "assumed-si": "no short-interest data",
    "below-floor": "below liquidity floor",
}


# User-configurable provider keys. Saved server-side (SQLite) so they apply to the
# backend Finnhub/FMP/LLM calls. Stored keys act as quota fallbacks behind the
# built-in dev key — see providers._ordered_keys / _finnhub_get / _fmp_get and the
# AI provider fallback in ai.py.
SETTINGS_FIELDS = (
    "finnhub_api_key", "fmp_api_key", "sec_user_agent", "marketaux_api_key",
) + providers.AI_SETTING_KEYS


def _settings_auth_required():
    """Gate /settings behind SETTINGS_PASSWORD (HTTP Basic Auth).

    Every stored key is global to the app instance (see app_settings in
    db.py), so an unauthenticated /settings is a full key-read/overwrite
    vulnerability on any publicly reachable deployment. Fails closed: with
    no SETTINGS_PASSWORD configured, the page refuses to serve rather than
    falling back to the old open behaviour.
    """
    expected = os.environ.get('SETTINGS_PASSWORD', '')
    if not expected:
        return jsonify({
            "error": "SETTINGS_PASSWORD is not configured on the server; "
                     "/settings is disabled until it is set.",
        }), 503
    auth = request.authorization
    if auth is None or not hmac.compare_digest(auth.password or '', expected):
        return Response(
            "Authentication required.", 401,
            {"WWW-Authenticate": 'Basic realm="Settings"'},
        )
    return None


@app.route('/settings', methods=['GET', 'POST'])
def settings_page():
    auth_error = _settings_auth_required()
    if auth_error is not None:
        return auth_error

    saved = False
    if request.method == 'POST':
        for field in SETTINGS_FIELDS:
            db.set_setting(field, request.form.get(field, '').strip())
        saved = True

    current = {field: db.get_setting(field) for field in SETTINGS_FIELDS}
    return render_template(
        'settings.html',
        current=current,
        status=providers.configured(),
        ai_providers=providers.AI_PROVIDERS,
        saved=saved,
    )


@app.route('/api/options-greeks/<ticker>')
def options_greeks_api(ticker):
    expiration = request.args.get('expiration', '')
    rf_rate_raw = request.args.get('rf_rate', '0.045')
    try:
        rf_rate = float(rf_rate_raw)
    except ValueError:
        rf_rate = 0.045

    # ?strikes=all | <positive int>; anything else falls back to the default band.
    strikes_raw = request.args.get('strikes', '').strip().lower()
    if strikes_raw == 'all':
        strike_count = 'all'
    else:
        try:
            strike_count = int(strikes_raw) if strikes_raw else None
            if strike_count is not None and strike_count <= 0:
                strike_count = None
        except ValueError:
            strike_count = None

    data = get_options_greeks_data(ticker, expiration, rf_rate, strike_count=strike_count)
    if not data:
        return jsonify({"error": f"Could not retrieve options data for {ticker}"}), 404
        
    return jsonify(data)


def compute_options_analysis(ticker, expiration_date=None, rf_rate=0.045):
    """Compile options statistics and quantitative posture metrics."""
    try:
        greeks_data = get_options_greeks_data(ticker, expiration_date, rf_rate)
        if not greeks_data:
            return None
        
        spot_price = greeks_data['spot_price']
        rows = greeks_data['options']
        selected_expiration = greeks_data['selected_expiration']
        days_to_exp = greeks_data['days_to_expiration']
        
        # Spot price history for HV30/90
        df = get_or_fetch_prices(ticker)
        hv30, hv90, iv_rank, iv_percentile = 0.0, 0.0, 0.0, 0.0
        vol_history = pd.Series(dtype=float)
        if df is not None and not df.empty and len(df) > 30:
            returns = df['close'].pct_change()
            hv30 = float(returns.tail(30).std() * np.sqrt(252)) * 100
            hv90 = float(returns.tail(90).std() * np.sqrt(252)) * 100
            
            rolling_30d = returns.rolling(30).std() * np.sqrt(252) * 100
            vol_history = rolling_30d.dropna().tail(252)
        
        # Calculate ATM IV
        strikes = [r['strike'] for r in rows]
        closest_strike = min(strikes, key=lambda s: abs(s - spot_price)) if strikes else spot_price
        
        atm_row = next((r for r in rows if r['strike'] == closest_strike), None)
        c_iv = atm_row['call_iv'] if atm_row else 0.0
        p_iv = atm_row['put_iv'] if atm_row else 0.0
        
        ivs = [v for v in [c_iv, p_iv] if v > 1.0]
        atm_iv = float(np.mean(ivs)) if ivs else 0.0
        
        # Volatility Rank and Percentile compared to historical realized volatility
        if not vol_history.empty and atm_iv > 0:
            min_vol = float(vol_history.min())
            max_vol = float(vol_history.max())
            iv_rank = ((atm_iv - min_vol) / (max_vol - min_vol)) * 100 if max_vol > min_vol else 50.0
            iv_rank = max(0.0, min(100.0, iv_rank))
            
            below_count = (vol_history < atm_iv).sum()
            iv_percentile = (below_count / len(vol_history)) * 100
        else:
            iv_rank, iv_percentile = 50.0, 50.0
            
        # Expected Move
        expected_move_bs = 0.85 * spot_price * (atm_iv / 100.0) * np.sqrt(days_to_exp / 365.0) if atm_iv > 0 else 0.0
        
        c_bid = atm_row['call_bid'] if atm_row else 0
        c_ask = atm_row['call_ask'] if atm_row else 0
        p_bid = atm_row['put_bid'] if atm_row else 0
        p_ask = atm_row['put_ask'] if atm_row else 0
        
        c_mid = (c_bid + c_ask) / 2.0
        p_mid = (p_bid + p_ask) / 2.0
        
        expected_move_straddle = c_mid + p_mid if (c_mid > 0 and p_mid > 0) else expected_move_bs
        
        # Put-Call Ratio (PCR)
        total_call_vol = sum(r.get('call_volume', 0) for r in rows)
        total_put_vol = sum(r.get('put_volume', 0) for r in rows)
        total_call_oi = sum(r.get('call_oi', 0) for r in rows)
        total_put_oi = sum(r.get('put_oi', 0) for r in rows)
        
        vol_pcr = total_put_vol / total_call_vol if total_call_vol > 0 else 0
        oi_pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 0
        
        # Max Pain
        pains = {}
        for strike_i in strikes:
            pain = 0
            for r in rows:
                strike = r['strike']
                call_oi = r.get('call_oi', 0)
                put_oi = r.get('put_oi', 0)
                if strike < strike_i:
                    pain += call_oi * (strike_i - strike)
                elif strike > strike_i:
                    pain += put_oi * (strike - strike_i)
            pains[strike_i] = pain
            
        max_pain = min(pains, key=pains.get) if pains else spot_price
        
        # GEX stats
        gex_profile = get_gex_profile(ticker, spot_price, rf_rate)
        gex_stats = gex_profile['stats'] if gex_profile else None
        
        # Strategy recommendations
        selling_environment = atm_iv > hv30
        strategy_recommendation = ""
        strategy_rationale = ""
        if iv_rank > 50:
            strategy_recommendation = "Sell Premium (Strangle / Iron Condor)"
            strategy_rationale = f"IV Rank is high ({iv_rank:.1f}%), meaning implied volatility is elevated relative to history. Selling premium takes advantage of the volatility risk premium (VRP) contraction."
        elif selling_environment:
            strategy_recommendation = "Sell Premium (Credit Spreads / Covered Call)"
            strategy_rationale = f"ATM Implied Volatility ({atm_iv:.1f}%) is trading at a premium over 30d Realized Volatility ({hv30:.1f}%). Option pricing is rich relative to actual stock movement."
        else:
            strategy_recommendation = "Buy Premium / Defined Risk (Debit Spreads / Calendar Spreads)"
            strategy_rationale = f"Implied Volatility ({atm_iv:.1f}%) is low and trading at a discount to realized volatility. Option premium is cheap, making buying strategies or debit spreads more attractive."
            
        return {
            'ticker': ticker.upper(),
            'spot_price': round(spot_price, 2),
            'selected_expiration': selected_expiration,
            'days_to_expiration': days_to_exp,
            'expirations': greeks_data['expirations'],
            'hv30': round(hv30, 2),
            'hv90': round(hv90, 2),
            'atm_iv': round(atm_iv, 2),
            'iv_rank': round(iv_rank, 1),
            'iv_percentile': round(iv_percentile, 1),
            'expected_move_bs': round(expected_move_bs, 2),
            'expected_move_straddle': round(expected_move_straddle, 2),
            'vol_pcr': round(vol_pcr, 3),
            'oi_pcr': round(oi_pcr, 3),
            'max_pain': round(max_pain, 2),
            'total_call_vol': total_call_vol,
            'total_put_vol': total_put_vol,
            'total_call_oi': total_call_oi,
            'total_put_oi': total_put_oi,
            'gex_stats': gex_stats,
            'strategy_recommendation': strategy_recommendation,
            'strategy_rationale': strategy_rationale,
            'selling_environment': selling_environment
        }
    except Exception as e:
        print(f"Error computing options analysis: {e}")
        return None


@app.route('/api/options-analysis/<ticker>')
def options_analysis_api(ticker):
    expiration = request.args.get('expiration', '')
    rf_rate_raw = request.args.get('rf_rate', '0.045')
    try:
        rf_rate = float(rf_rate_raw)
    except ValueError:
        rf_rate = 0.045
        
    data = compute_options_analysis(ticker, expiration, rf_rate)
    if not data:
        return jsonify({"error": f"Could not compute options analysis for {ticker}"}), 404
        
    return jsonify(data)


@app.route('/api/options-terminal/<ticker>')
def api_options_terminal(ticker):
    ticker = ticker.strip().upper()
    convention = request.args.get('convention', 'naive').strip().lower()
    if convention not in ('naive', 'short_wings'):
        convention = 'naive'

    current_price = get_current_price_yfinance(ticker)
    if not current_price:
        current_price = _spot_price(ticker, _get_yf_ticker(ticker))
    if not current_price:
        return jsonify({"error": f"No price data available for {ticker}; options terminal unavailable"}), 503

    stock = _get_yf_ticker(ticker)
    chains_df = get_full_option_chain_df(ticker, stock=stock, current_price=current_price)
    daily_df = get_or_fetch_prices(ticker)

    terminal_res = compute_options_terminal(
        ticker=ticker,
        spot=current_price,
        chain_df=chains_df,
        daily_df=daily_df,
        convention=convention,
    )

    import dataclasses
    return jsonify(dataclasses.asdict(terminal_res))


@app.route('/api/options-ai-report/<ticker>')
def options_ai_report_api(ticker):
    expiration = request.args.get('expiration', '')
    rf_rate_raw = request.args.get('rf_rate', '0.045')
    force = request.args.get('force', '').lower() in ('1', 'true', 'yes')
    try:
        rf_rate = float(rf_rate_raw)
    except ValueError:
        rf_rate = 0.045
        
    data = compute_options_analysis(ticker, expiration, rf_rate)
    if not data:
        return jsonify({"error": f"Could not retrieve options data for {ticker}"}), 404
        
    report, error = ai.generate_options_report(ticker, data, force=force)
    if error:
        return jsonify({"error": error}), 500
        
    return jsonify({"html": report})


@app.route('/ai-summary')
def ai_summary_page():
    ticker = request.args.get('ticker', '').strip().upper()
    if not ticker:
        return render_template('ai_summary.html', data=None, ticker='')

    # Load all statistics and data points
    analytics_data = compute_analytics(ticker)
    if analytics_data is None:
        return render_template('ai_summary.html', data=None, ticker=ticker,
                               error=f"Could not retrieve data for {ticker}")

    # Gather fundamentals & positioning
    positioning_data = _guard_section(
        "positioning", lambda: compute_positioning(ticker), default={})
    fundamentals = get_fundamentals(ticker) or {}

    # Merge finnhub metrics (list of dicts) and yfinance fundamentals into a single dict
    valuation_dict = {}
    finnhub_val = positioning_data.get("valuation")
    if finnhub_val and isinstance(finnhub_val, list):
        for item in finnhub_val:
            valuation_dict[item["label"]] = item["value"]
    if fundamentals:
        for k, v in fundamentals.items():
            if k not in valuation_dict:
                valuation_dict[k] = v

    # The VAL sheet (and the LLM payload) speaks Finnhub's label names, but
    # Finnhub doesn't cover ETFs and may be unconfigured/rate-limited -- so
    # alias the yfinance fundamentals already fetched by get_fundamentals
    # into those labels instead of rendering N/A for data we have.
    for wb_key, yf_key in (
        ('P/E (TTM)', 'Trailing P/E'),
        ('P/B',       'Price / Book'),
        ('P/S (TTM)', 'Price / Sales'),
    ):
        val = fundamentals.get(yf_key)
        if valuation_dict.get(wb_key) is None and isinstance(val, (int, float)):
            valuation_dict[wb_key] = round(val, 2)
    div_yield = fundamentals.get('Dividend Yield')
    if valuation_dict.get('Div Yield') is None and isinstance(div_yield, (int, float)):
        valuation_dict['Div Yield'] = f"{div_yield:.2f}%"
    # ETF fallbacks: yfinance classifies funds by category / fund family
    # rather than sector / industry.
    if not valuation_dict.get('Sector') and fundamentals.get('Category'):
        valuation_dict['Sector'] = fundamentals['Category']
    if not valuation_dict.get('Industry') and fundamentals.get('Fund Family'):
        valuation_dict['Industry'] = fundamentals['Fund Family']
    # Finnhub-only consensus breakdown; yfinance's single rating is the
    # keyless fallback for the Consensus Reco cell.
    rating = fundamentals.get('Analyst Rating')
    if rating and not valuation_dict.get('Consensus Rating'):
        valuation_dict['Consensus Rating'] = str(rating).replace('_', ' ').title()

    # Gather momentum data
    momentum_data = _guard_section(
        "momentum", lambda: compute_momentum(ticker), default={})

    # Gather ML signal
    ml_signal = ml.predict(ticker)

    # Gather GEX stats
    gex_profile = get_gex_profile(ticker, analytics_data['current_price'])
    gex_stats = gex_profile['stats'] if gex_profile else None

    # Merge all stats into a single context payload
    # Remove charts and figures so we only feed clean numbers to the LLM (and stay within limits / keep it clean)
    payload = {
        "ticker": ticker,
        "current_price": analytics_data.get("current_price"),
        "general_stats": analytics_data.get("stats"),
        "forward_estimates": analytics_data.get("forward_estimates"),
        "valuation": valuation_dict,
        "recommendations": positioning_data.get("recommendations"),
        "insider_sentiment": positioning_data.get("insider", {}).get("sentiment"),
        "insider_transactions": positioning_data.get("insider", {}).get("transactions"),
        "sec_filings": positioning_data.get("insider", {}).get("sec_filings"),
        "institutional_holders": positioning_data.get("institutional", {}).get("holders"),
        "momentum": {
            "mom_12_1_pct": momentum_data.get("mom_12_1"),
            "mom_6m_pct": momentum_data.get("mom_6m"),
            "mom_3m_pct": momentum_data.get("mom_3m"),
            "mom_1m_pct": momentum_data.get("mom_1m"),
            "ann_vol_1y_pct": momentum_data.get("ann_vol_1y"),
            "risk_adjusted_mom_score": momentum_data.get("risk_adj_mom"),
            "strategy_annual_return_pct": momentum_data.get("strat_stats", {}).get("annual_return"),
            "strategy_sharpe": momentum_data.get("strat_stats", {}).get("sharpe"),
            "strategy_max_dd_pct": momentum_data.get("strat_stats", {}).get("max_dd"),
            "hold_annual_return_pct": momentum_data.get("hold_stats", {}).get("annual_return"),
            "hold_sharpe": momentum_data.get("hold_stats", {}).get("sharpe"),
            "hold_max_dd_pct": momentum_data.get("hold_stats", {}).get("max_dd"),
            "percent_invested": momentum_data.get("pct_invested"),
            "trade_count": momentum_data.get("trade_count"),
            "win_rate_pct": momentum_data.get("win_rate"),
            "profit_factor": momentum_data.get("profit_factor"),
            "relative_to_spy": momentum_data.get("relative_spy"),
        },
        "ml_signal": ml_signal,
        "dealer_gex": gex_stats,
    }

    # Generate or fetch the comprehensive strategy report
    comprehensive_report, report_error = ai.generate_comprehensive_report(ticker, payload)

    # We also pass the clean payload to the page so it can render the raw tables as well!
    return render_template(
        'ai_summary.html',
        ticker=ticker,
        data=payload,
        report=comprehensive_report,
        report_error=report_error,
        configured=bool(providers.ai_providers()),
    )


@app.route('/api/raw-sec-filings/<ticker>')
def raw_sec_filings_api(ticker):
    try:
        filings = providers.sec_recent_filings(ticker, forms=None, limit=30)
        return jsonify(filings or [])
    except Exception as e:
        print(f"Error fetching raw SEC filings: {e}")
        return jsonify({"error": "SEC filings temporarily unavailable"}), 502


# --- Automatic EOD options chain cache warmer ---
# Pre-warms the options cache (S3 when configured, plus local SQLite) for every
# ticker stored in the DB across ALL of its expirations, so the options tab,
# GEX profile and IV smile load instantly and keep working after hours or
# during rate-limiting. Runs in a background thread on startup and every
# 4 hours afterward. Started at import time so it also runs under gunicorn on Render.
# A cross-process lock in api_cache ensures only one gunicorn worker does the
# fetching each cycle.

def _warm_options_cache(force=False):
    """Fetch and cache option chains for all tickers in the DB, in the background.

    force=True (the EOD /api/warm-cache endpoint) bypasses cached chains first
    so the re-fetch captures the closing snapshot instead of re-serving a
    still-fresh mid-session cache. Even forced runs claim a short lock so
    scheduler retries can't stampede.
    """
    def _worker():
        try:
            lock_ttl = 0.15 if force else 3.5  # hours; short TTL just dedupes retries
            if not db.try_claim_lock("options_warmer_lock", ttl_hours=lock_ttl):
                return
            with db.get_conn() as conn:
                rows = conn.execute("SELECT DISTINCT symbol FROM daily_prices").fetchall()
            symbols = [r["symbol"] for r in rows]
            if not symbols:
                return

            store = f"s3://{s3_cache.bucket()}" if s3_cache.enabled() else "sqlite"
            print(f"[options-cache] warming {len(symbols)} tickers -> {store}{' (forced EOD refresh)' if force else ''}...")
            chains = 0
            for sym in symbols:
                try:
                    stock = _get_yf_ticker(sym)
                    for exp in get_cached_expirations(sym, stock, force=force):
                        if get_cached_chain(sym, exp, stock=stock, force=force) is not None:
                            chains += 1
                        time.sleep(0.15)  # stay gentle on yfinance rate limits

                    # Persist daily positioning snapshot to options_iv_history
                    try:
                        chain_df = get_full_option_chain_df(sym, stock=stock)
                        daily_df = get_or_fetch_prices(sym)
                        spot = _spot_price(sym, stock)
                        if not chain_df.empty and spot:
                            compute_options_terminal(sym, spot, chain_df, daily_df, record_db=True)
                        elif not spot:
                            print(f"[options-cache] {sym}: no spot, skipping IV snapshot")
                    except Exception as snap_err:
                        print(f"[options-cache] snapshot error for {sym}: {snap_err}")
                except Exception as e:
                    print(f"[options-cache] {sym} failed: {e}")
            print(f"[options-cache] done ({len(symbols)} tickers, {chains} chains)")
        except Exception as e:
            print(f"[options-cache] warmer error: {e}")

    threading.Thread(target=_worker, daemon=True, name="options-cache-warmer").start()


def _start_options_cache_warmer():
    """Warm once now, then re-warm every 4 hours. Called once at import time —
    _warm_options_cache itself must not spawn the recurring thread, or every
    cycle (and every /api/warm-cache call) would leak a new scheduler."""
    _warm_options_cache()

    def _recurring():
        while True:
            time.sleep(4 * 3600)
            _warm_options_cache()

    threading.Thread(target=_recurring, daemon=True, name="options-cache-recurring").start()


def _warm_heatmap():
    """Background-refresh the whole S&P 500 universe into daily_prices.

    Mirrors _warm_options_cache: background thread, cross-worker lock, never
    raises. The /heatmap request path itself only ever reads SQLite, so the
    10-30s bulk download never lands on a page load.
    """
    def _worker():
        try:
            if not db.try_claim_lock("heatmap_warmer_lock", ttl_hours=1.0):
                return
            constituents = heatmap.get_constituents()
            symbols = [c["symbol"] for c in constituents]
            if not symbols:
                return
            print(f"[heatmap] warming {len(symbols)} S&P 500 tickers ...")
            result = heatmap.refresh_universe(symbols)
            print(f"[heatmap] done ({result['fetched']} fetched, "
                  f"{len(result['failed'])} failed)")
            if result["fetched"]:
                heatmap.invalidate_payload()
        except Exception as e:
            print(f"[heatmap] warmer error: {e}")

    threading.Thread(target=_worker, daemon=True, name="heatmap-warmer").start()


def _start_heatmap_warmer():
    """Warm once now, then every 45 minutes (just inside the 1h price
    freshness window, so the map stays a session behind at worst). Import-time
    call, like the options warmer; the test suite sets HEATMAP_WARM_ON_BOOT=0
    to keep pytest offline."""
    if os.environ.get("HEATMAP_WARM_ON_BOOT", "1") != "1":
        return
    _warm_heatmap()

    def _recurring():
        while True:
            time.sleep(45 * 60)
            _warm_heatmap()

    threading.Thread(target=_recurring, daemon=True, name="heatmap-recurring").start()


def _warm_radar_cache(symbols: list[str] | None = None, background: bool = True):
    """Refresh the per-ticker earnings/fundamentals/SI payloads the /radar
    scans read. Mirrors _warm_heatmap: background thread, cross-worker lock,
    never raises. Chunked with sleeps to stay polite on the free tiers.
    `background=False` runs the worker synchronously (used by tests)."""
    def _worker():
        try:
            if not db.try_claim_lock("radar_warmer_lock", ttl_hours=6.0):
                return
            # The heatmap warmer keeps only ~10 sessions per S&P 500 name, so
            # backfill a year (batched, no per-ticker calls) before the
            # 200-bar universe cut. Sector ETF + SPY closes feed the drift
            # excess-vs-sector and CAR math.
            backfill = list(radar.BENCHMARK_SYMBOLS)
            if not symbols:
                backfill += [c["symbol"] for c in heatmap.get_constituents()]
            try:
                heatmap.refresh_universe(sorted(set(backfill)), period="1y")
            except Exception as e:
                print(f"[radar] price backfill failed: {e}")
            syms = symbols or radar.universe_symbols(min_bars=200)
            if not syms:
                return
            print(f"[radar] warming {len(syms)} tickers ...")
            done = 0
            for sym in syms:
                for key, fn in (("earnings", earnings.get_earnings_history),
                                ("fundamentals", get_fundamentals),
                                ("short", providers.get_short_interest)):
                    try:
                        payload = fn(sym)
                        if payload is not None:
                            db.cache_set("yfinance", f"{key}:{sym}", payload)
                    except Exception as e:
                        print(f"[radar] {fn.__name__} failed for {sym}: {e}")
                done += 1
                if done % 25 == 0:
                    db.cache_set("radar", "warm_progress",
                                 {"done": done, "total": len(syms),
                                  "updated": radar.now_iso()})
                    time.sleep(2)  # chunk pause, mirrors heatmap politeness
            db.cache_set("radar", "warm_progress",
                         {"done": done, "total": len(syms),
                          "updated": radar.now_iso()})
            print(f"[radar] done ({done} tickers)")
            radar.publish_snapshot(radar.load_context())
        except Exception as e:
            print(f"[radar] warmer error: {e}")

    if background:
        threading.Thread(target=_worker, daemon=True, name="radar-warmer").start()
    else:
        _worker()


def _start_radar_warmer():
    """Warm once shortly after boot, then every 12h (payload TTL is 24h)."""
    if os.environ.get("RADAR_WARM_ON_BOOT", "1") != "1":
        return

    def _recurring():
        time.sleep(120)  # let the app bind before competing for locks
        _warm_radar_cache()
        while True:
            time.sleep(12 * 3600)
            _warm_radar_cache()

    threading.Thread(target=_recurring, daemon=True, name="radar-recurring").start()


@app.route('/api/warm-cache', methods=['POST'])
def api_warm_cache():
    """Trigger a forced EOD options-chain warm (see .github/workflows/warm-cache.yml).

    Requires WARM_CACHE_TOKEN in the environment, supplied by the caller as
    an Authorization: Bearer header (or ?token= fallback).
    """
    expected = os.environ.get('WARM_CACHE_TOKEN', '')
    if not expected:
        return jsonify({"error": "WARM_CACHE_TOKEN is not configured on the server"}), 503
    provided = request.headers.get('Authorization', '')
    provided = provided[7:] if provided.startswith('Bearer ') else request.args.get('token', '')
    if not hmac.compare_digest(provided, expected):
        return jsonify({"error": "unauthorized"}), 403
    _warm_options_cache(force=True)
    return jsonify({"status": "accepted", "detail": "EOD options cache warm started in background"}), 202


@app.route('/api/corporate-actions/<ticker>')
def api_corporate_actions(ticker):
    """Expose point-in-time identity and corporate action history for a ticker."""
    as_of = request.args.get('as_of')
    entity = corporate_actions.engine.resolve_entity_as_of(ticker, as_of_date=as_of)
    if not entity:
        return jsonify({"ticker": ticker.upper(), "entity": None, "actions": []}), 200

    actions = corporate_actions.engine.get_corporate_actions_for_entity(entity.entity_id)
    history = corporate_actions.engine.get_symbol_history(entity.entity_id)

    return jsonify({
        "ticker": ticker.upper(),
        "entity": {
            "entity_id": entity.entity_id,
            "cik": entity.cik,
            "figi": entity.figi,
            "cusip": entity.cusip,
            "legal_name": entity.legal_name,
        },
        "symbol_timeline": history,
        "actions": [
            {
                "action_id": a.action_id,
                "action_type": a.action_type.value,
                "effective_date": a.effective_date,
                "announcement_date": a.announcement_date,
                "ratio": a.ratio,
                "old_value": a.old_value,
                "new_value": a.new_value,
                "status": a.status,
            }
            for a in actions
        ],
    }), 200


# Provider namespace is versioned: the cached payload's shape is part of the
# contract, so a future field addition (e.g. WT4's Deflated Sharpe) bumps to
# _v2 and self-invalidates instead of serving up to 24h of incompatible rows.
_INSTITUTIONAL_CACHE_PROVIDER = "institutional_backtest_v1"
_INSTITUTIONAL_CACHE_KEYS = ("signals_backtest", "permutation_test")
_INSTITUTIONAL_LOCK_TTL_HOURS = 0.02  # ~72s; longer than one cold computation
_INSTITUTIONAL_LOCK_WAIT_S = 3.0


def _institutional_backtest(ticker, stock_df):
    """Signals backtest + block-bootstrap permutation null test, cached 24h.

    The permutation test reruns the backtest 100x, so a cold computation is
    seconds of CPU (see docs/superpowers/specs/2026-09-14-backtest-engine-
    statistical-rigor-design.md, WT3). Two guards on top of the cache:

    - stampede protection: concurrent cold requests for the same ticker race
      for a cross-process lock (same db.try_claim_lock idiom the options
      warmer uses); the losers wait for the winner's cache write rather than
      each paying the full cost. If that write never lands (winner crashed,
      or is slower than the wait budget) they compute it themselves -- slow
      beats failing, per the "always render something" contract.
    - a cache row missing an expected key is treated as a miss, so a
      shape change can never turn into a KeyError -> 500.
    """
    def _read_cache():
        cached = db.cache_get(_INSTITUTIONAL_CACHE_PROVIDER, ticker, ttl_hours=24)
        if isinstance(cached, dict) and all(k in cached for k in _INSTITUTIONAL_CACHE_KEYS):
            return cached
        return None

    hit = _read_cache()
    if hit is not None:
        return hit

    claimed = db.try_claim_lock(
        f"institutional_backtest:{ticker}", ttl_hours=_INSTITUTIONAL_LOCK_TTL_HOURS
    )
    if not claimed:
        deadline = time.time() + _INSTITUTIONAL_LOCK_WAIT_S
        while time.time() < deadline:
            time.sleep(0.5)
            hit = _read_cache()
            if hit is not None:
                return hit

    sim_res, _ = backtest_engine.run_walkforward_backtest(stock_df)
    perm_res = backtest_engine.run_permutation_test(stock_df)
    payload = {
        "signals_backtest": sim_res.__dict__,
        "permutation_test": perm_res.__dict__,
    }
    db.cache_set(_INSTITUTIONAL_CACHE_PROVIDER, ticker, payload)
    return payload


def _institutional_authorized():
    expected = os.environ.get("WARM_CACHE_TOKEN", "").strip()
    if not expected:
        return False
    header = request.headers.get("Authorization", "")
    return header.startswith("Bearer ") and hmac.compare_digest(header[len("Bearer "):], expected)


@app.route('/api/institutional/<ticker>')
def api_institutional(ticker):
    """Institutional quantitative analytics suite: Microstructure, Macro, CAR, and Greeks."""
    if not _institutional_authorized():
        if not os.environ.get("WARM_CACHE_TOKEN", "").strip():
            return jsonify({"error": "institutional endpoint disabled: WARM_CACHE_TOKEN not configured"}), 503
        return jsonify({"error": "unauthorized"}), 401
    ticker = ticker.upper()
    stock_df = get_or_fetch_prices(ticker, period="2y")
    spy_df = get_or_fetch_prices("SPY", period="2y")

    if stock_df is None or stock_df.empty:
        return jsonify({"error": f"No pricing data for {ticker}"}), 404

    # 1. Microstructure & Squeeze
    si_pct, si_dtc = _si_kwargs(ticker)
    si_payload, si_source = _short_interest_for(ticker)
    micro_res = microstructure.get_microstructure_analytics(
        stock_df, short_pct_float=si_pct, days_to_cover=si_dtc)

    # 2. Macro Conditioning
    macro_res = macro_engine.get_macro_financial_report(stock_df, spy_df if spy_df is not None else stock_df)

    # 3. Signals Walk-Forward Simulation + block-bootstrap permutation null test
    backtest_payload = _institutional_backtest(ticker, stock_df)
    sim_res_dict = backtest_payload["signals_backtest"]
    perm_res_dict = backtest_payload["permutation_test"]

    # 4. Recent SEC 8-K Events
    events_8k = sec_8k.fetch_and_parse_8k_filings(ticker, limit=5)

    return jsonify({
        "ticker": ticker,
        "microstructure": micro_res.__dict__,
        "short_interest": {**(si_payload or {}), "si_source": si_source},
        "macro_conditioning": {
            "regime": macro_res.current_regime,
            "fed_funds_rate": macro_res.fed_funds_rate,
            "yield_curve_2s10s": macro_res.yield_curve_2s10s_spread,
            "is_inverted": macro_res.is_yield_curve_inverted,
            "cpi_yoy": macro_res.cpi_inflation_yoy,
            "betas": macro_res.regime_conditional_betas,
            "upside_capture": macro_res.upside_capture_ratio,
            "downside_capture": macro_res.downside_capture_ratio,
        },
        "signals_backtest": sim_res_dict,
        "permutation_test": perm_res_dict,
        "sec_8k_events": [e.__dict__ for e in events_8k],
    }), 200


@app.route('/api/active-tickers')
def api_active_tickers():
    """Serve a list of all active tickers currently tracked in the database daily_prices."""
    expected = os.environ.get('WARM_CACHE_TOKEN', '')
    if expected:
        provided = request.headers.get('Authorization', '')
        provided = provided[7:] if provided.startswith('Bearer ') else request.args.get('token', '')
        if not hmac.compare_digest(provided, expected):
            return jsonify({"error": "unauthorized"}), 403

    try:
        with db.get_conn() as conn:
            rows = conn.execute("SELECT DISTINCT symbol FROM daily_prices").fetchall()
        symbols = [r["symbol"] for r in rows]
        return jsonify(symbols)
    except Exception as e:
        print(f"Error serving active tickers: {e}")
        return jsonify([]), 500


def _optional_token_authorized() -> bool:
    """active-tickers posture: enforce only when WARM_CACHE_TOKEN is configured."""
    expected = os.environ.get("WARM_CACHE_TOKEN", "").strip()
    if not expected:
        return True
    header = request.headers.get("Authorization", "")
    return header.startswith("Bearer ") and hmac.compare_digest(
        header[len("Bearer "):], expected)


@app.route('/api/radar/settings', methods=['POST'])
def api_radar_settings():
    """Persist /radar inline tunables (display thresholds, not secrets)."""
    if not _optional_token_authorized():
        return jsonify({"error": "unauthorized"}), 401
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or not body:
        return jsonify({"error": "expected a JSON object of tunables"}), 400
    try:
        stored = {key: radar.clamp_tunable(key, value)
                  for key, value in body.items()}
    except (ValueError, TypeError, OverflowError) as e:
        return jsonify({"error": str(e)}), 400
    for key, val in stored.items():
        radar.set_tunable(key, val)
    # Tunables apply at snapshot build time; rebuild from cache (no network).
    # The 1-minute lock keeps repeated saves from stacking rebuilds.
    if db.try_claim_lock("radar_republish_lock", ttl_hours=1 / 60):
        threading.Thread(
            target=lambda: radar.publish_snapshot(radar.load_context()),
            daemon=True, name="radar-republish").start()
    return jsonify(stored), 200


# Import-time start: gunicorn imports app:app and never runs __main__, so
# this is what activates the warmer in production (each worker starts the
# threads; the api_cache lock makes only one actually fetch).
_start_options_cache_warmer()
_start_heatmap_warmer()
_start_radar_warmer()


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001, threaded=True)
