"""Analytics page summary and disclosure layout regressions."""
from types import SimpleNamespace

from flask import render_template

from app import app


def _payload():
    stats = SimpleNamespace(
        annualised_vol=24.5, max_drawdown=-18.2, skewness=-0.15, kurtosis=1.42,
        beta=1.15, var_95=-2.1, var_99=-3.4, mean_daily_return=0.08,
        daily_std=1.54, data_points=504,
    )
    charts = SimpleNamespace(
        distribution='<div id="distribution-chart"></div>',
        beta='<div id="beta-chart"></div>',
        cumulative_return='<div id="cumulative-chart"></div>',
        var='<div id="var-chart"></div>', volatility='<div id="volatility-chart"></div>',
        drawdown='<div id="drawdown-chart"></div>', forward_estimates=None,
        volume_profile=None, monte_carlo=None,
        options_smile='<div id="options-chart"></div>', gex=None,
        price_target=None, sharpe='<div id="sharpe-chart"></div>',
        seasonality='<div id="seasonality-chart"></div>', insider=None,
    )
    fundamentals = {
        'Name': 'Example Semiconductor Company', 'Sector': 'Technology',
        'Industry': 'Semiconductors', 'Next Earnings': '2026-12-23',
        'Consensus EPS': 38.21, 'Consensus Revenue': 61_670_000_000,
        'PEG Ratio': 0.16, 'Earnings Growth': 10.595, 'Revenue Growth': 3.793,
    }
    return SimpleNamespace(
        ticker='MU', current_price=762.35, stats=stats, charts=charts,
        fundamentals=fundamentals, checklist=None, ai_sections=None,
        forward_estimates=None, ml=None, gex=None, ai_report=None,
        insider_summary=None, institutional=None, earnings=None,
    )


def test_analytics_summary_disclosures_and_forecast_content_render_once():
    with app.test_request_context('/analytics?ticker=MU'):
        html = render_template('analytics.html', data=_payload(), ticker='MU', excel_mode=True)

    first_disclosure = html.index('<details class="analytics-section')
    assert html.index('id="cumulative-chart"') < first_disclosure
    for section in ('risk', 'company', 'options'):
        start = html.index(f'data-analytics-section="{section}"')
        assert '<summary' in html[start:]
        assert ' open' not in html[start:html.index('</details>', start)]

    assert '2026-12-23' in html
    assert '$38.21' in html
    assert '$61.67B' in html
    assert '+1059.5%' in html
    assert '+379.3%' in html
    assert html.count('id="cumulative-chart"') == 1
    assert html.count('id="distribution-chart"') == 1
    assert html.count('id="sharpe-chart"') == 1
    assert html.count('id="options-chart"') == 1
