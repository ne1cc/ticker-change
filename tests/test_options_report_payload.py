"""Report-only AI options payload builder (#197): real chain digest plus declared gaps."""
import json
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

import app as app_module
from options.models import (GexProfile, OptionsTerminalResult, RealizedVolCones,
                            VolTermStructure)
from options.vol import compute_25d_skew

EXP1 = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
EXP2 = (datetime.now() + timedelta(days=60)).strftime('%Y-%m-%d')
EXP3 = (datetime.now() + timedelta(days=90)).strftime('%Y-%m-%d')
EXP4 = (datetime.now() + timedelta(days=120)).strftime('%Y-%m-%d')
EXP5 = (datetime.now() + timedelta(days=150)).strftime('%Y-%m-%d')


def _chain_frames(strikes, iv_call=0.20, iv_put=0.20, last_trade_call=None, last_trade_put=None):
    calls = [{'strike': s, 'bid': 2.0, 'ask': 2.2, 'lastPrice': 2.1,
              'impliedVolatility': iv_call, 'openInterest': 100, 'volume': 10}
             for s in strikes]
    puts = [{'strike': s, 'bid': 1.8, 'ask': 2.0, 'lastPrice': 1.9,
             'impliedVolatility': iv_put, 'openInterest': 200, 'volume': 20}
            for s in strikes]
    if last_trade_call:
        for row in calls:
            row['lastTradeDate'] = last_trade_call
    if last_trade_put:
        for row in puts:
            row['lastTradeDate'] = last_trade_put
    return pd.DataFrame(calls), pd.DataFrame(puts), 100.0


def _terminal_stub(error=None):
    gex = GexProfile(total_gex=12.34, call_gex=20.0, put_gex=-7.66, regime='positive',
                     gamma_flip=97.5, call_wall=105.0, call_wall_val=8.2, put_wall=95.0,
                     put_wall_val=-6.1, hvl=100.0, hvl_val=3.0, convention='naive',
                     strikes=[95.0, 105.0], net_gex_series=[-6.1, 8.2])
    vol = VolTermStructure(expirations=[EXP1, EXP2], dtes=[30, 60], atm_ivs=[20.0, 21.0],
                           slope=1.0, regime='contango')
    cones = RealizedVolCones(current_iv=20.0, hv10=10.0, hv20=11.0, hv30=12.0, hv60=13.0,
                             hv90=14.0, yang_zhang_30=12.5, yang_zhang_90=13.5, vrp=8.0,
                             vrp_regime='Negative VRP', iv_rank=None, iv_percentile=None,
                             iv_min_52w=10.0, iv_max_52w=30.0, expected_move_1d=1.0,
                             expected_move_straddle=4.0, coverage_68_pct=0.6)
    return OptionsTerminalResult(ticker='SPY', spot_price=101.5, as_of='x', convention='naive',
                                 gex=gex, vol=vol, cones=cones, error=error)


@pytest.fixture
def env(monkeypatch):
    """Every network and storage boundary mocked; only pure math runs real."""
    prices = pd.DataFrame({'close': 100 + np.sin(np.arange(60))})
    monkeypatch.setattr(app_module, '_get_yf_ticker', lambda symbol: object())
    monkeypatch.setattr(app_module, 'get_cached_expirations', lambda *args: [EXP1, EXP2])
    monkeypatch.setattr(app_module, 'get_cached_chain', lambda *args, **kwargs: _chain_frames(
        [round(70 + i * 0.5, 1) for i in range(120)]))
    monkeypatch.setattr(app_module, 'get_or_fetch_prices', lambda symbol: prices)
    monkeypatch.setattr(app_module, 'get_current_price_yfinance', lambda symbol: 101.5)
    monkeypatch.setattr(app_module, 'get_full_option_chain_df', lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(app_module, 'compute_options_terminal', lambda **kwargs: _terminal_stub())
    monkeypatch.setattr(app_module, 'compute_iv_rank_and_percentile',
                        lambda *args, **kwargs: {'iv_rank': None, 'iv_percentile': None,
                                                 'status': 'insufficient'})
    monkeypatch.setattr(app_module, 'get_atm_contract', lambda *args, **kwargs: None)
    monkeypatch.setattr(app_module.earnings, 'get_earnings_history', lambda ticker: [])
    return app_module


def build(env, expiration=EXP1, **kwargs):
    payload = env.build_options_report_payload('SPY', expiration, 0.045, **kwargs)
    json.dumps(payload, separators=(',', ':'), default=str)
    return payload


def test_digest_band_size_atm_and_shape(env):
    payload = build(env)
    assert payload['spot'] == 101.5
    assert payload['snapshot_spot'] == 100.0
    digest = payload['chain_digest']
    sigma = 20.0 / 100 * math.sqrt(30 / 365) * 100
    assert len(digest) <= 43
    assert all(abs(row['strike'] - 100.0) <= 2 * sigma + 1e-9 for row in digest)
    assert 100.0 in [row['strike'] for row in digest]
    row = digest[0]
    assert set(row) == {'strike'} | {f'{side}_{f}'
                                     for side in ('call', 'put')
                                     for f in ('bid', 'ask', 'spread_pct', 'iv', 'delta',
                                               'gamma', 'theta', 'vega', 'oi', 'volume')}
    assert payload['page_heuristic']['recommendation']
    assert 'strategy_recommendation' not in payload
    assert payload['term_structure']['window'] == 'first 8 listed expiries'
    assert set(payload['vol_cones']) == {'hv10', 'hv20', 'hv30', 'hv60', 'hv90',
                                         'yang_zhang_30', 'yang_zhang_90'}
    assert payload['liquidity']['median_spread_pct'] == 10.0
    assert 'earnings' in payload['_provenance']


def test_digest_falls_back_to_20_nearest_strikes(env, monkeypatch):
    strikes = [float(s) for s in range(91, 111)]
    monkeypatch.setattr(env, 'get_cached_chain', lambda *args, **kwargs:
                        _chain_frames(strikes, iv_call=0.02, iv_put=0.02))
    payload = build(env)
    digest = payload['chain_digest']
    assert len(digest) == 20
    assert sorted(row['strike'] for row in digest) == strikes


def test_invalid_quotes_and_na_greeks_serialize_as_null(env, monkeypatch):
    calls, puts, _ = _chain_frames([round(70 + i * 0.5, 1) for i in range(120)])
    for frame in (calls, puts):
        row = frame.loc[frame['strike'] == 105.0].index[0]
        frame.loc[row, 'impliedVolatility'] = 0.0
    calls.loc[calls['strike'] == 105.0, ['bid', 'ask']] = 0.0
    monkeypatch.setattr(env, 'get_cached_chain', lambda *args, **kwargs: (calls, puts, 100.0))
    payload = build(env)
    row = next(r for r in payload['chain_digest'] if r['strike'] == 105.0)
    assert row['call_bid'] is None
    assert row['call_ask'] is None
    assert row['call_spread_pct'] is None
    assert row['call_delta'] is None
    assert row['put_delta'] is None
    healthy = next(r for r in payload['chain_digest'] if r['strike'] == 100.0)
    assert healthy['call_bid'] == 2.0
    assert healthy['call_spread_pct'] == 9.5


def test_skew_matches_compute_25d_skew(env, monkeypatch):
    monkeypatch.setattr(env, 'get_cached_chain', lambda *args, **kwargs:
                        _chain_frames([round(70 + i * 0.5, 1) for i in range(120)],
                                      iv_call=0.20, iv_put=0.24))
    payload = build(env)
    rows = env.get_options_greeks_data('SPY', EXP1, 0.045, strike_count='all')['options']
    frame = env._options_skew_frame(rows)
    ratio, risk_reversal = compute_25d_skew(frame, 100.0)
    assert payload['skew'] == {'ratio': ratio, 'risk_reversal': risk_reversal}
    assert payload['skew']['ratio'] == 1.2
    assert payload['skew']['risk_reversal'] == -4.0


def test_skew_null_when_not_computable(env):
    payload = build(env)
    assert payload['skew'] is None
    assert 'skew' in payload['_provenance']


@pytest.mark.parametrize('mode', ['error', 'raise'])
def test_terminal_failure_nulls_terminal_fields(env, monkeypatch, mode):
    def terminal(**kwargs):
        stub = _terminal_stub(error='boom')
        if mode == 'raise':
            raise RuntimeError('boom')
        return stub
    monkeypatch.setattr(env, 'compute_options_terminal', terminal)
    payload = build(env)
    assert payload['gex_stats'] is None
    assert payload['term_structure'] is None
    assert payload['vol_cones'] is None
    assert {'gex_stats', 'term_structure', 'vol_cones'} <= set(payload['_provenance'])


def test_vix_failure_keeps_digest(env, monkeypatch):
    prices = pd.DataFrame({'close': 100 + np.sin(np.arange(60))})

    def fetch(symbol):
        if symbol == '^VIX':
            raise RuntimeError('vix down')
        return prices
    monkeypatch.setattr(env, 'get_or_fetch_prices', fetch)
    payload = build(env)
    assert payload['market_context']['vix_close'] is None
    assert any(k.startswith('market_context.vix') for k in payload['_provenance'])
    assert payload['chain_digest']


def test_market_context_reports_vix_and_spy(env):
    context = build(env)['market_context']
    expected_close = round(float((100 + np.sin(np.arange(60)))[-1]), 2)
    assert context['vix_close'] == expected_close
    assert isinstance(context['vix_change_1m_pct'], float)
    assert context['spy_trend'] in ('above', 'below')
    assert isinstance(context['spy_vs_50d_pct'], float)


def test_vintage_unknown_without_chain_timestamps(env):
    assert build(env)['vintage'] == 'unknown'


def test_vintage_uses_latest_last_trade_date(env, monkeypatch):
    monkeypatch.setattr(env, 'get_cached_chain', lambda *args, **kwargs: _chain_frames(
        [round(70 + i * 0.5, 1) for i in range(120)],
        last_trade_call='2026-10-19 15:59:00', last_trade_put='2026-10-20 16:00:00'))
    payload = build(env)
    assert payload['vintage'] == '2026-10-20T16:00:00+00:00'


def test_iv_rank_provenance_notes_non_front_expiry(env, monkeypatch):
    monkeypatch.setattr(env, 'get_cached_expirations', lambda *args: [EXP1, EXP2, EXP3])
    payload = build(env, expiration=EXP2)
    assert 'front-expiry history' in payload['_provenance']['iv_rank']


def test_earnings_receives_first_four_expiry_frames(env, monkeypatch):
    expirations = [EXP1, EXP2, EXP3, EXP4, EXP5]
    monkeypatch.setattr(env, 'get_cached_expirations', lambda *args: expirations)
    seen = {}

    def fake_earnings(ticker, spot, chains_by_exp, prices_df=None):
        seen.update(chains_by_exp)
        return {'available': True, 'implied_move_pct': 2.0}
    monkeypatch.setattr(env, 'get_earnings_implied_vs_realized', fake_earnings)
    payload = build(env)
    assert set(seen) == set(expirations[:4])
    assert all(set(df['cp'].unique()) == {'C', 'P'} for df in seen.values())
    assert payload['earnings']['available'] is True


def test_unavailable_earnings_declares_reason(env):
    payload = build(env)
    assert payload['earnings']['available'] is False
    assert payload['_provenance']['earnings'] == payload['earnings']['reason']


def test_zero_oi_side_declares_missing_concentration(env, monkeypatch):
    calls, puts, _ = _chain_frames([round(70 + i * 0.5, 1) for i in range(120)])
    puts['openInterest'] = np.nan
    monkeypatch.setattr(env, 'get_cached_chain', lambda *args, **kwargs: (calls, puts, 100.0))
    payload = build(env)
    assert payload['concentrations']['call_oi']
    assert payload['concentrations']['put_oi'] is None
    assert 'put open interest' in payload['_provenance']['concentrations.put_oi']
