"""Comprehensive unit tests for the institutional Options Terminal suite."""
import math
import tempfile
import numpy as np
import pandas as pd
import pytest

from options.greeks import calculate_greeks, vectorized_greeks
from options.gex import clean_chain, compute_gex_profile, find_gamma_flip, _hedge_weight, _dealer_sign
from options.vol import compute_vol_term_structure, compute_25d_skew, build_surface_grid
from options.cones import compute_close_to_close_hv, compute_yang_zhang_vol, compute_expected_move_coverage, compute_vol_cones_and_vrp
from options.storage import init_options_db, record_snapshot, get_snapshot_history, get_history_status
from options import compute_options_terminal


class TestGreeksCalculus:
    def test_greeks_boundary_floors(self):
        # Even with zero T or zero IV, calculations should not crash or produce inf/NaN
        res = calculate_greeks(s=100.0, k=100.0, t=0.0, v=0.0)
        assert math.isfinite(res['delta'])
        assert math.isfinite(res['gamma'])
        assert math.isfinite(res['vanna'])
        assert math.isfinite(res['charm'])

    def test_vanna_finite_difference_cross_validation(self):
        """Cross-validate analytical Vanna = dDelta / dSigma using central finite differences."""
        s, k, t, sigma, r = 150.0, 150.0, 0.25, 0.30, 0.045
        analytical = calculate_greeks(s, k, t, sigma, r, cp='call')['vanna']

        h = 1e-4
        delta_plus = calculate_greeks(s, k, t, sigma + h, r, cp='call')['delta']
        delta_minus = calculate_greeks(s, k, t, sigma - h, r, cp='call')['delta']
        finite_diff = (delta_plus - delta_minus) / (2.0 * h)

        assert abs(analytical - finite_diff) < 1e-4, f"Analytical Vanna {analytical} vs FD {finite_diff}"

    def test_charm_finite_difference_cross_validation(self):
        """Cross-validate analytical Charm = -dDelta / dt using central finite differences."""
        s, k, t, sigma, r = 100.0, 105.0, 0.50, 0.25, 0.045
        analytical = calculate_greeks(s, k, t, sigma, r, cp='call')['charm']

        # Delta decay per calendar day: -(Delta(t + dt) - Delta(t - dt)) / (2 * dt * 365)
        dt = 1e-4
        delta_t_plus = calculate_greeks(s, k, t + dt, sigma, r, cp='call')['delta']
        delta_t_minus = calculate_greeks(s, k, t - dt, sigma, r, cp='call')['delta']
        finite_diff = - (delta_t_plus - delta_t_minus) / (2.0 * dt * 365.0)

        assert abs(analytical - finite_diff) < 1e-3, f"Analytical Charm {analytical} vs FD {finite_diff}"

    def test_vectorized_greeks_matches_scalar(self):
        s = 100.0
        strikes = np.array([90.0, 100.0, 110.0])
        t_years = np.array([0.1, 0.2, 0.3])
        sigmas = np.array([0.2, 0.25, 0.3])

        vec = vectorized_greeks(s, strikes, t_years, sigmas, cp='C')
        for i in range(len(strikes)):
            scal = calculate_greeks(s, strikes[i], t_years[i], sigmas[i], cp='call')
            assert abs(vec['delta'][i] - scal['delta']) < 1e-5
            assert abs(vec['gamma'][i] - scal['gamma']) < 1e-5
            assert abs(vec['vanna'][i] - scal['vanna']) < 1e-5
            assert abs(vec['charm'][i] - scal['charm']) < 1e-5


class TestGexRealismAndConventions:
    @pytest.fixture
    def synthetic_chain(self):
        """Create a synthetic options chain with 0DTE, monthly OPEX, and mixed open interest."""
        rows = [
            # Strike, DTE, Expiration, CP, Bid, Ask, IV, OI, Volume
            {"strike": 90.0, "dte": 0, "expiration": "2026-09-10", "cp": "P", "bid": 0.2, "ask": 0.3, "iv": 0.35, "open_interest": 0, "volume": 5000},
            {"strike": 100.0, "dte": 0, "expiration": "2026-09-10", "cp": "C", "bid": 1.5, "ask": 1.6, "iv": 0.25, "open_interest": 0, "volume": 12000},
            {"strike": 110.0, "dte": 0, "expiration": "2026-09-10", "cp": "C", "bid": 0.1, "ask": 0.2, "iv": 0.30, "open_interest": 0, "volume": 8000},
            # 30D Monthly OPEX
            {"strike": 85.0, "dte": 30, "expiration": "2026-10-10", "cp": "P", "bid": 0.8, "ask": 0.9, "iv": 0.32, "open_interest": 15000, "volume": 200},
            {"strike": 95.0, "dte": 30, "expiration": "2026-10-10", "cp": "P", "bid": 2.2, "ask": 2.3, "iv": 0.28, "open_interest": 20000, "volume": 500},
            {"strike": 100.0, "dte": 30, "expiration": "2026-10-10", "cp": "C", "bid": 4.1, "ask": 4.2, "iv": 0.25, "open_interest": 25000, "volume": 1200},
            {"strike": 105.0, "dte": 30, "expiration": "2026-10-10", "cp": "C", "bid": 2.0, "ask": 2.1, "iv": 0.26, "open_interest": 18000, "volume": 800},
            {"strike": 115.0, "dte": 30, "expiration": "2026-10-10", "cp": "C", "bid": 0.5, "ask": 0.6, "iv": 0.29, "open_interest": 12000, "volume": 400},
        ]
        return pd.DataFrame(rows)

    def test_0dte_weighting_uses_volume_when_oi_zero(self, synthetic_chain):
        weights = _hedge_weight(synthetic_chain)
        # 0DTE strikes have OI=0 and Volume=5000, 12000, 8000
        assert weights[0] == 5000.0
        assert weights[1] == 12000.0
        assert weights[2] == 8000.0
        # 30D strikes have OI=15000, 20000, etc.
        assert weights[3] == 15000.0

    def test_dual_dealer_sign_conventions(self, synthetic_chain):
        spot = 100.0
        naive_signs = _dealer_sign(synthetic_chain, convention="naive", spot=spot)
        wings_signs = _dealer_sign(synthetic_chain, convention="short_wings", spot=spot)

        # Naive: all calls +1, all puts -1
        assert naive_signs[0] == -1.0  # Put
        assert naive_signs[1] == 1.0   # Call

        # Short wings: OTM put (strike 90 < 100) -> -1.0; OTM call (strike 110 > 100) -> -1.0
        assert wings_signs[0] == -1.0  # OTM put -> -1.0 (dealers short OTM put)
        assert wings_signs[2] == -1.0  # OTM call strike 110 > 100 -> -1.0 (dealers short OTM call)

    def test_gex_profile_computation_and_walls(self, synthetic_chain):
        spot = 100.0
        gex_naive = compute_gex_profile(synthetic_chain, spot=spot, convention="naive")
        assert gex_naive.total_gex != 0.0
        assert gex_naive.call_wall is not None
        assert gex_naive.put_wall is not None
        assert gex_naive.call_wall >= spot
        assert gex_naive.put_wall <= spot
        assert len(gex_naive.top_nodes) > 0

    def test_gamma_flip_finds_zero_crossing(self, synthetic_chain):
        spot = 100.0
        flip = find_gamma_flip(synthetic_chain, spot=spot, convention="naive")
        # Flip should be a valid positive float within reasonable grid bounds
        assert flip is not None
        assert 50.0 <= flip <= 150.0


class TestVolatilityAndCones:
    @pytest.fixture
    def synthetic_daily_prices(self):
        np.random.seed(42)
        n = 100
        returns = np.random.normal(0.0005, 0.015, n)
        prices = [100.0]
        for r in returns:
            prices.append(prices[-1] * math.exp(r))
        
        dates = pd.date_range(end="2026-09-10", periods=n + 1, freq="B")
        df = pd.DataFrame({
            "close": prices,
            "open": [p * 0.998 for p in prices],
            "high": [p * 1.012 for p in prices],
            "low": [p * 0.988 for p in prices],
            "volume": [1000000] * (n + 1),
        }, index=dates)
        return df

    def test_yang_zhang_vol_computation(self, synthetic_daily_prices):
        yz = compute_yang_zhang_vol(synthetic_daily_prices, window=30)
        assert yz > 0.0
        assert 10.0 <= yz <= 50.0  # Reasonable annualized % volatility

    def test_expected_move_coverage(self, synthetic_daily_prices):
        coverage = compute_expected_move_coverage(synthetic_daily_prices, current_iv=25.0, window=30)
        assert 0.0 <= coverage <= 100.0
        assert coverage > 40.0  # Synthetic normal returns should comfortably stay within 1-sigma coverage

    def test_vol_cones_and_vrp(self, synthetic_daily_prices):
        cones = compute_vol_cones_and_vrp(synthetic_daily_prices, current_iv=30.0)
        assert cones.hv30 > 0.0
        assert cones.vrp == round(30.0 - cones.hv30, 2)
        assert cones.expected_move_1d > 0.0
        assert cones.expected_move_straddle > 0.0


class TestStorageAndPersistence:
    def test_record_and_retrieve_snapshots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_db = f"{tmpdir}/test_stocks.db"
            init_options_db(test_db)

            # Record two snapshots
            ok1 = record_snapshot(
                symbol="TEST", spot=100.0, atm_iv=25.0, hv30=20.0, vrp=5.0,
                call_wall=110.0, put_wall=90.0, gamma_flip=98.5, total_gex=12.5,
                skew_25d=1.1, convention="naive", db_path=test_db,
            )
            assert ok1 is True

            history = get_snapshot_history("TEST", convention="naive", db_path=test_db)
            assert len(history) == 1
            assert history[0]["spot"] == 100.0
            assert history[0]["gamma_flip"] == 98.5

            status = get_history_status("TEST", db_path=test_db)
            assert status["days_recorded"] == 1
            assert status["is_accumulating"] is True


class TestOptionsTerminalOrchestrator:
    def test_compute_options_terminal_with_valid_chain(self):
        spot = 100.0
        chain_df = pd.DataFrame([
            {"strike": 95.0, "dte": 7, "expiration": "2026-09-17", "cp": "P", "bid": 1.0, "ask": 1.1, "iv": 0.28, "open_interest": 5000, "volume": 100},
            {"strike": 100.0, "dte": 7, "expiration": "2026-09-17", "cp": "C", "bid": 2.0, "ask": 2.1, "iv": 0.25, "open_interest": 8000, "volume": 200},
            {"strike": 105.0, "dte": 7, "expiration": "2026-09-17", "cp": "C", "bid": 0.8, "ask": 0.9, "iv": 0.27, "open_interest": 4000, "volume": 50},
        ])
        dates = pd.date_range(end="2026-09-10", periods=40, freq="B")
        daily_df = pd.DataFrame({
            "close": [100.0 + i * 0.1 for i in range(40)],
            "open": [99.5 + i * 0.1 for i in range(40)],
            "high": [101.0 + i * 0.1 for i in range(40)],
            "low": [99.0 + i * 0.1 for i in range(40)],
        }, index=dates)

        res = compute_options_terminal("TEST", spot, chain_df, daily_df, convention="naive")
        assert res.ticker == "TEST"
        assert res.spot_price == 100.0
        assert res.error is None
        assert res.gex.total_gex != 0.0
        assert len(res.contracts) > 0

    def test_compute_options_terminal_graceful_empty_chain_fallback(self):
        # When chain is None/empty (e.g. non-optionable ticker), it falls back without crashing
        res = compute_options_terminal("UNKNOWN", 50.0, None, None)
        assert res.ticker == "UNKNOWN"
        assert res.error is not None
        assert "unavailable" in res.error.lower()
        assert res.gex.total_gex == 0.0
        assert res.contracts == []


class TestOptionsTerminalFlaskRoutes:
    @pytest.fixture
    def client(self):
        from app import app
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client

    def test_options_terminal_page_returns_200(self, client, monkeypatch):
        # Mock external network lookups
        from unittest.mock import patch
        with patch('app.get_current_price_yfinance', return_value=150.0), \
             patch('app.get_full_option_chain_df', return_value=pd.DataFrame()):
            resp = client.get('/options?ticker=AAPL')
            assert resp.status_code == 200
            html = resp.get_data(as_text=True)
            assert 'Options Terminal' in html
            assert 'Dealer Convention' in html
            assert 'AAPL' in html

    def test_api_options_terminal_json_endpoint(self, client, monkeypatch):
        from unittest.mock import patch
        with patch('app.get_current_price_yfinance', return_value=150.0), \
             patch('app.get_full_option_chain_df', return_value=pd.DataFrame()):
            resp = client.get('/api/options-terminal/AAPL?convention=short_wings')
            assert resp.status_code == 200
            data = resp.get_json()
            assert data['ticker'] == 'AAPL'
            assert data['spot_price'] == 150.0
            assert data['convention'] == 'short_wings'
            assert 'gex' in data
            assert 'vol' in data
            assert 'cones' in data
