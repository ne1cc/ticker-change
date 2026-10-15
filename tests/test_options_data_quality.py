"""Regressions for incomplete option chains masquerading as real zero metrics."""
from copy import deepcopy
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

import app as app_module


@pytest.fixture
def payload():
    return {
        'spot_price': 100.0,
        'calls': [{'strike': 100.0, 'bid': 2.0, 'ask': 2.2, 'lastPrice': 2.1,
                   'impliedVolatility': 0.25, 'openInterest': 100, 'volume': 20}],
        'puts': [{'strike': 100.0, 'bid': 1.8, 'ask': 2.0, 'lastPrice': 1.9,
                  'impliedVolatility': 0.27, 'openInterest': 200, 'volume': 30}],
    }


def degraded(payload):
    result = deepcopy(payload)
    for side in ('calls', 'puts'):
        for row in result[side]:
            row.update(bid=0.0, ask=0.0, impliedVolatility=0.00001, openInterest=0)
    return result


@pytest.fixture
def chain_cache(monkeypatch):
    """Replace only network/storage boundaries; run real cache selection."""
    store = {}
    monkeypatch.setattr(app_module, 'get_cboe_snapshot', lambda ticker: None)
    monkeypatch.setattr(app_module.s3_cache, 'get_json', lambda key: None)
    monkeypatch.setattr(app_module.s3_cache, 'put_json', lambda key, value: True)
    monkeypatch.setattr(app_module.db, 'cache_get', lambda provider, key, ttl: store.get(key))
    monkeypatch.setattr(app_module.db, 'cache_set', lambda provider, key, value: store.update({key: value}))
    return store


def test_incomplete_fresh_s3_chain_fetches_real_oi_and_iv(monkeypatch, payload, chain_cache):
    bad = degraded(payload)
    monkeypatch.setattr(app_module.s3_cache, 'get_json', lambda key: (bad, 0.1))
    stock = SimpleNamespace(option_chain=lambda exp: SimpleNamespace(
        calls=pd.DataFrame(payload['calls']), puts=pd.DataFrame(payload['puts'])))
    calls, puts, spot = app_module.get_cached_chain('SPY', '2099-01-16', stock, spot_hint=100)
    assert calls.iloc[0]['openInterest'] == 100
    assert puts.iloc[0]['impliedVolatility'] == 0.27
    assert chain_cache['optchain:SPY:2099-01-16']['calls'][0]['openInterest'] == 100


def test_fresh_sqlite_beats_incomplete_s3(monkeypatch, payload, chain_cache):
    monkeypatch.setattr(app_module.s3_cache, 'get_json', lambda key: (degraded(payload), 0.1))
    chain_cache['optchain:SPY:2099-01-16'] = payload
    stock = SimpleNamespace(option_chain=Mock(side_effect=AssertionError('fresh complete cache needs no fetch')))
    calls, _, _ = app_module.get_cached_chain('SPY', '2099-01-16', stock)
    assert calls.iloc[0]['openInterest'] == 100


@pytest.mark.parametrize('force', [False, True])
def test_partial_live_fetch_does_not_replace_better_stale_data(monkeypatch, payload, chain_cache, force):
    monkeypatch.setattr(app_module.s3_cache, 'get_json', lambda key: (payload, 8.0))
    bad = degraded(payload)
    stock = SimpleNamespace(option_chain=lambda exp: SimpleNamespace(
        calls=pd.DataFrame(bad['calls']), puts=pd.DataFrame(bad['puts'])))
    calls, puts, _ = app_module.get_cached_chain('SPY', '2099-01-16', stock, spot_hint=100, force=force)
    assert calls.iloc[0]['openInterest'] == 100
    assert puts.iloc[0]['impliedVolatility'] == 0.27
    assert 'optchain:SPY:2099-01-16' not in chain_cache


def test_partial_data_keeps_volume_and_limits_repeated_live_requests(payload, chain_cache):
    bad = degraded(payload)
    fetch = Mock(return_value=SimpleNamespace(calls=pd.DataFrame(bad['calls']), puts=pd.DataFrame(bad['puts'])))
    stock = SimpleNamespace(option_chain=fetch)
    for _ in range(2):
        calls, puts, _ = app_module.get_cached_chain('SPY', '2099-01-16', stock, spot_hint=100)
        assert calls.iloc[0]['volume'] == 20
        assert puts.iloc[0]['lastPrice'] == 1.9
    assert fetch.call_count == 1


@pytest.fixture
def analysis(monkeypatch, payload):
    expiration = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
    monkeypatch.setattr(app_module, '_get_yf_ticker', lambda symbol: object())
    monkeypatch.setattr(app_module, 'get_cached_expirations', lambda *args: [expiration])
    monkeypatch.setattr(app_module, 'get_cached_chain', lambda *args, **kwargs:
                        (pd.DataFrame(payload['calls']), pd.DataFrame(payload['puts']), 100.0))
    monkeypatch.setattr(app_module, 'get_or_fetch_prices', lambda symbol: pd.DataFrame())
    monkeypatch.setattr(app_module, 'get_gex_profile', lambda *args: None)
    return lambda: app_module.app.test_client().get('/api/options-analysis/SPY').get_json()


def test_analysis_sums_whole_expiration_including_far_strikes(payload, analysis):
    payload['calls'].append(dict(payload['calls'][0], strike=160.0, openInterest=300, volume=80))
    result = analysis()
    assert result['total_call_oi'] == 400
    assert result['total_put_oi'] == 200
    assert result['oi_pcr'] == 0.5
    assert result['vol_pcr'] == 0.3
    assert result['atm_iv'] == 26.0
    assert result['expected_move_straddle'] == 4.0


def test_analysis_uses_nearest_usable_iv_on_each_side(payload, analysis):
    payload['calls'][0]['impliedVolatility'] = 0.00001
    payload['puts'][0]['impliedVolatility'] = np.nan
    payload['calls'].append(dict(payload['calls'][0], strike=101.0, impliedVolatility=0.24))
    payload['puts'].append(dict(payload['puts'][0], strike=99.0, impliedVolatility=0.28))
    assert analysis()['atm_iv'] == 26.0


@pytest.mark.parametrize('iv', [0, 0.00001, float('inf'), float('nan')])
def test_missing_iv_does_not_recommend_cheap_premium(payload, analysis, iv):
    payload['calls'][0]['impliedVolatility'] = iv
    payload['puts'][0]['impliedVolatility'] = iv
    result = analysis()
    assert result['atm_iv'] is None
    assert result['iv_rank'] is None
    assert result['iv_percentile'] is None
    assert result['expected_move_bs'] is None
    assert result['expected_move_straddle'] == 4.0  # Valid quotes remain useful.
    assert result['selling_environment'] is None
    assert 'unavailable' in result['strategy_rationale'].lower()


def test_far_wing_iv_is_not_mislabeled_atm(payload, analysis):
    payload['calls'][0]['impliedVolatility'] = 0.00001
    payload['puts'][0]['impliedVolatility'] = 0.00001
    payload['calls'].append(dict(payload['calls'][0], strike=160.0, impliedVolatility=0.50))
    assert analysis()['atm_iv'] is None


def test_no_oi_or_quotes_returns_unavailable_instead_of_zero(payload, analysis):
    for side in ('calls', 'puts'):
        payload[side][0].update(openInterest=0, bid=0.0, ask=0.0, impliedVolatility=0.00001)
    result = analysis()
    assert result['oi_pcr'] is None
    assert result['total_call_oi'] is None
    assert result['total_put_oi'] is None
    assert result['max_pain'] is None
    assert result['expected_move_straddle'] is None
    assert result['total_call_vol'] == 20
    assert result['vol_pcr'] == 1.5


def test_zero_calls_with_positive_put_oi_is_undefined_ratio(payload, analysis):
    payload['calls'][0]['openInterest'] = 0
    result = analysis()
    assert result['total_call_oi'] == 0
    assert result['total_put_oi'] == 200
    assert result['oi_pcr'] is None
    assert result['max_pain'] == 100.0


def test_missing_oi_on_one_side_does_not_become_zero(payload, analysis):
    payload['calls'][0]['openInterest'] = np.nan
    result = analysis()
    assert result['total_call_oi'] is None
    assert result['total_put_oi'] == 200
    assert result['oi_pcr'] is None
    assert result['max_pain'] is None


def test_invalid_crossed_quotes_do_not_price_straddle(payload, analysis):
    payload['calls'][0].update(bid=3.0, ask=2.0)
    result = analysis()
    assert result['expected_move_straddle'] is None


def test_cboe_fallback_recovers_same_expiration(monkeypatch, payload, chain_cache):
    bad = degraded(payload)
    stock = SimpleNamespace(option_chain=lambda exp: SimpleNamespace(
        calls=pd.DataFrame(bad['calls']), puts=pd.DataFrame(bad['puts'])))
    snapshot = {
        'timestamp': '2099-01-15 16:00:00',
        'data': {'current_price': 100, 'options': [
            {'option': 'SPY990116C00100000', 'bid': 2.0, 'ask': 2.2,
             'iv': 0.25, 'open_interest': 100, 'volume': 20, 'last_trade_price': 2.1},
            {'option': 'SPY990116P00100000', 'bid': 1.8, 'ask': 2.0,
             'iv': 0.27, 'open_interest': 200, 'volume': 30, 'last_trade_price': 1.9},
            {'option': 'SPY990117C00100000', 'iv': 0.50, 'open_interest': 999},
        ]}}
    monkeypatch.setattr(app_module, 'get_cboe_snapshot', lambda ticker: snapshot, raising=False)
    calls, puts, _ = app_module.get_cached_chain('SPY', '2099-01-16', stock, spot_hint=100)
    assert len(calls) == 1
    assert calls.iloc[0]['openInterest'] == 100
    assert puts.iloc[0]['impliedVolatility'] == 0.27
    assert calls.attrs['source'] == 'Cboe delayed'
    assert calls.attrs['as_of'] == '2099-01-15 16:00:00'


@pytest.mark.parametrize('live_complete', [False, True])
@pytest.mark.parametrize('yahoo_state', ['partial', 'empty', 'error'])
def test_warmer_preserves_complete_snapshot_and_uploads_recovery(monkeypatch, payload, live_complete, yahoo_state):
    import io
    import json
    import warm_s3_cache as warmer

    saved = {'optchains/SPY/2099-01-16.json': deepcopy(payload)}
    class FakeS3:
        def get_object(self, Bucket, Key):
            return {'Body': io.BytesIO(json.dumps(saved[Key]).encode())}
        def put_object(self, Bucket, Key, Body, **kwargs):
            saved[Key] = json.loads(Body)
    bad = degraded(payload)
    monkeypatch.setenv('S3_CACHE_BUCKET', 'test-bucket')
    monkeypatch.setenv('APP_URL', 'https://app.invalid')
    monkeypatch.setattr(warmer.requests, 'get', lambda *args, **kwargs:
                        SimpleNamespace(status_code=200, json=lambda: ['SPY']))
    monkeypatch.setattr(warmer.boto3, 'client', lambda *args, **kwargs: FakeS3())
    def yahoo_chain(exp):
        if yahoo_state == 'error':
            raise RuntimeError('Yahoo unavailable')
        return SimpleNamespace(calls=pd.DataFrame(bad['calls']) if yahoo_state == 'partial' else pd.DataFrame(),
                               puts=pd.DataFrame(bad['puts']) if yahoo_state == 'partial' else pd.DataFrame())
    monkeypatch.setattr(warmer, '_get_yf_ticker', lambda ticker: SimpleNamespace(
        options=['2099-01-16'], option_chain=yahoo_chain))
    monkeypatch.setattr(warmer, '_spot_price', lambda *args: 100)
    monkeypatch.setattr(warmer.time, 'sleep', lambda seconds: None)
    recovery = dict(deepcopy(payload), source='Cboe delayed', as_of='2099-01-15 16:00:00') if live_complete else None
    monkeypatch.setattr(warmer, 'get_cboe_snapshot', lambda ticker: None, raising=False)
    monkeypatch.setattr(warmer, 'cboe_chain', lambda *args: recovery, raising=False)
    warmer.main()
    result = saved['optchains/SPY/2099-01-16.json']
    assert result['calls'][0]['openInterest'] == 100
    if live_complete:
        assert result['source'] == 'Cboe delayed'


@pytest.mark.parametrize('iv,oi,expected_iv,expected_oi', [(None, None, 'N/A', 'N/A'), (26.0, 0.0, '26.0%', '0.00')])
def test_analyst_displays_missing_metrics_without_breaking(iv, oi, expected_iv, expected_oi):
    import json
    import shutil
    import subprocess
    from pathlib import Path

    if not shutil.which('node'):
        pytest.skip('Node is required to exercise the analyst display function')
    template = (Path(__file__).resolve().parents[1] / 'templates/options.html').read_text()
    function = template[template.index('  async function loadOptionsAnalysis()'):template.index('  async function generateOptionsAiReport()')]
    data = dict(atm_iv=iv, hv30=10.0, hv90=12.0, iv_rank=None, iv_percentile=None,
                expected_move_bs=None, expected_move_straddle=None, gex_stats=None,
                max_pain=None, vol_pcr=1.5, oi_pcr=oi, total_call_vol=20, total_put_vol=30,
                total_call_oi=None, total_put_oi=None, strategy_recommendation='Insufficient data',
                strategy_rationale='Implied volatility is unavailable.', source='Cboe delayed',
                source_as_of='2026-10-14 16:00:00')
    script = '''const nodes = new Map();
    const document = {getElementById(id) {
      if (!nodes.has(id)) nodes.set(id, {value: id === 'expiration-select' ? '2026-10-15' : '0.045', innerText: '', className: ''});
      return nodes.get(id);
    }};
    const ticker = 'SPY'; let optionsAnalysisData;
    const fetch = async () => ({ok: true, json: async () => (DATA)});
    FUNCTION
    loadOptionsAnalysis().then(() => process.stdout.write(JSON.stringify(Object.fromEntries(
      [...nodes].map(([id, el]) => [id, el.innerText])))));
    '''.replace('DATA', json.dumps(data)).replace('FUNCTION', function)
    result = subprocess.run(['node', '-e', script], text=True, capture_output=True, check=True)
    displayed = json.loads(result.stdout)
    assert displayed['opt-ai-atm-iv'] == expected_iv
    assert displayed['opt-ai-oi-pcr'] == expected_oi
    assert displayed['opt-ai-max-pain'] == 'N/A'
    assert displayed['opt-ai-em-straddle'] == 'N/A'
    assert displayed['opt-ai-strategy-rec'] == 'Insufficient data'
    assert 'Cboe delayed' in displayed['opt-ai-data-source']
    assert '2026-10-14 16:00:00' in displayed['opt-ai-data-source']


def test_standalone_warmer_does_not_require_app_analytics_dependencies():
    import subprocess
    import sys
    result = subprocess.run([
        sys.executable, '-c',
        "import sys; sys.modules['scipy'] = None; sys.modules['flask'] = None; import warm_s3_cache"
    ], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_same_day_expected_move_uses_remaining_time_to_close(monkeypatch, analysis):
    from zoneinfo import ZoneInfo
    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            value = cls(2099, 1, 16, 10, tzinfo=ZoneInfo('America/New_York'))
            return value.astimezone(tz) if tz else value.replace(tzinfo=None)
    monkeypatch.setattr(app_module, 'datetime', FixedDateTime)
    monkeypatch.setattr(app_module, 'get_cached_expirations', lambda *args: ['2099-01-16'])
    result = analysis()
    assert result['days_to_expiration'] == 0
    assert result['expected_move_bs'] == 0.58


def test_cboe_snapshot_reuses_one_download_across_expirations(monkeypatch):
    import option_chain_data as chain_data
    snapshot = {'timestamp': '2026-10-15 06:54:04', 'data': {'options': [{'option': 'SPY261016C00100000'}]}}
    monkeypatch.setattr(chain_data, '_CBOE_SNAPSHOTS', {})
    fetch = Mock(return_value=SimpleNamespace(raise_for_status=lambda: None, json=lambda: snapshot))
    monkeypatch.setattr(chain_data.requests, 'get', fetch)
    assert chain_data.get_cboe_snapshot('SPY') == snapshot
    assert chain_data.get_cboe_snapshot('SPY') == snapshot
    assert fetch.call_count == 1


@pytest.mark.parametrize('body', [[], {'data': []}, {'data': {'options': []}}])
def test_invalid_cboe_response_is_a_cache_miss(monkeypatch, body):
    import option_chain_data as chain_data
    monkeypatch.setattr(chain_data, '_CBOE_SNAPSHOTS', {})
    monkeypatch.setattr(chain_data.requests, 'get', lambda *args, **kwargs:
                        SimpleNamespace(raise_for_status=lambda: None, json=lambda: body))
    assert chain_data.get_cboe_snapshot('SPY') is None


@pytest.mark.parametrize('bid', [None, float('nan'), float('inf')])
def test_missing_bid_does_not_become_a_valid_zero_bid(payload, analysis, bid):
    payload['calls'][0]['bid'] = bid
    assert analysis()['expected_move_straddle'] is None


def test_genuine_zero_bid_can_price_a_straddle(payload, analysis):
    payload['calls'][0]['bid'] = 0
    assert analysis()['expected_move_straddle'] == 3.0


def test_absent_bid_field_does_not_become_a_valid_zero_bid(payload, analysis):
    payload['calls'][0].pop('bid')
    assert analysis()['expected_move_straddle'] is None
