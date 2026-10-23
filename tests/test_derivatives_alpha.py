"""Issue 1.1: put charm equals call charm (q=0), finite-difference check, no divergence
between the two BS implementations in the repo."""
import math

import derivatives_alpha
from options.greeks import calculate_greeks


def test_put_charm_equals_call_charm_when_q_is_zero():
    kwargs = dict(spot=100.0, strike=105.0, time_to_exp=30 / 365, volatility=0.35, risk_free_rate=0.04)
    call = derivatives_alpha.compute_higher_order_greeks(is_call=True, **kwargs)
    put = derivatives_alpha.compute_higher_order_greeks(is_call=False, **kwargs)
    assert put.charm == call.charm


def test_charm_matches_finite_difference_of_delta_over_calendar_time():
    s, k, v, r = 100.0, 100.0, 0.30, 0.05
    t_years = 45 / 365
    h = 1 / 365  # one calendar day, in years
    for is_call in (True, False):
        greeks = derivatives_alpha.compute_higher_order_greeks(s, k, t_years, v, r, is_call)
        d_before = derivatives_alpha.compute_higher_order_greeks(s, k, t_years - h, v, r, is_call).delta
        d_after = derivatives_alpha.compute_higher_order_greeks(s, k, t_years + h, v, r, is_call).delta
        fd_charm = -(d_after - d_before) / (2 * h) / 365  # -dDelta/dT, per calendar day
        assert abs(fd_charm - greeks.charm) < 1e-4, (is_call, fd_charm, greeks.charm)


def test_charm_agrees_with_options_greeks_implementation():
    s, k, t, v, r = 100.0, 105.0, 30 / 365, 0.35, 0.04
    for side in ("call", "put"):
        a = derivatives_alpha.compute_higher_order_greeks(s, k, t, v, r, side == "call")
        b = calculate_greeks(s, k, t, v, r, side)
        assert abs(a.charm - b["charm"]) < 1e-6
