"""Chart data must remain numerical, safe to embed, and aligned across renderers."""
import json
import re
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


def payload(html):
    return json.loads(re.search(r'<script[^>]*type="application/json"[^>]*>(.*?)</script>', html, re.S).group(1))


def test_chart_json_cannot_close_its_script_element():
    import charts
    html = charts.render_chart({'series': [{'name': '</script><script>alert(1)</script>', 'data': [np.float64(1), np.nan, np.inf, -2, 0]}]})
    assert html.count('</script>') == 1
    assert payload(html)['series'][0]['data'] == [1, None, None, -2, 0]
    assert payload(html)['series'][0]['name'] == '</script><script>alert(1)</script>'


def test_chart_mount_ids_are_distinct_and_dates_are_iso():
    import charts
    first = charts.render_chart({'data': [pd.Timestamp('2024-01-02')]})
    second = charts.render_chart({})
    assert re.search(r'id="([^"]+)"', first)[1] != re.search(r'id="([^"]+)"', second)[1]
    assert payload(first)['data'] == ['2024-01-02T00:00:00']


def test_cumulative_returns_keep_the_baseline_and_dates():
    import app
    prices = pd.DataFrame({'close': [100., 110., 99.]}, index=pd.bdate_range('2024-01-01', periods=3))
    with patch.object(app, 'get_or_fetch_prices', return_value=prices):
        chart = payload(app.get_cumulative_return_chart('AAA', prices))
    line = chart['series'][-1]
    assert [point[1] for point in line['data']] == pytest.approx([100., 110., 99.])
    assert [point[0][:10] for point in line['data']] == ['2024-01-01', '2024-01-02', '2024-01-03']


def test_stock_records_are_ordered_and_deduplicated():
    import app
    prices = pd.DataFrame({'open': [2, 1, 3], 'high': [3, 2, 4], 'low': [1, .5, 2], 'close': [2.5, 1.5, 3.5], 'volume': [20, 10, 30]}, index=pd.to_datetime(['2024-01-02', '2024-01-01', '2024-01-02']))
    with patch.object(app, 'get_or_fetch_prices', return_value=prices):
        result = app.generate_stock_chart('AAA')
    assert [r['time'] for r in result['records']] == ['2024-01-01', '2024-01-02']
    assert result['records'][-1]['close'] == 3.5
    assert 'plotly' not in result['html'].lower()


def test_panning_before_available_history_returns_unavailable_not_500():
    import app
    prices = pd.DataFrame({'open': [1.], 'high': [2.], 'low': [.5], 'close': [1.5], 'volume': [10]}, index=pd.to_datetime(['2024-01-02']))
    with patch.object(app, 'get_or_fetch_prices', return_value=prices):
        response = app.app.test_client().get('/api/chart-data/AAA?start_date=2020-01-01&end_date=2020-12-31')
    assert response.status_code == 404


def test_gex_views_preserve_sign_and_common_symmetric_scale():
    import app
    from datetime import datetime, timedelta
    calls = pd.DataFrame({'strike': [90., 100., 110.], 'openInterest': [100, 200, 300], 'impliedVolatility': [.25, .25, .25]})
    puts = calls.copy(); puts['openInterest'] = [300, 200, 100]
    expiry = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
    with patch.object(app, '_get_yf_ticker', return_value=object()), patch.object(app, 'get_cached_expirations', return_value=[expiry]), patch.object(app, 'get_cached_chain', return_value=(calls, puts, 100.)):
        result = app.get_gex_profile('AAA', 100.)
    assert result is not None
    option = payload(result['chart'])
    views = option['terminalViews']['options']
    assert len(views) == 4
    assert all(v['xAxis'][0]['min'] == -v['xAxis'][0]['max'] for v in views)
    assert len({v['xAxis'][0]['max'] for v in views}) == 1
    split = views[1]['series']
    assert all(p[0] >= 0 for p in split[0]['data'])
    assert all(p[0] <= 0 for p in split[1]['data'])
    assert [p[1] for p in split[0]['data']] == [90., 100., 110.]
