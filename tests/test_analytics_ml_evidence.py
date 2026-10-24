"""ML evidence panel on /analytics: OOS stats beside the call, hidden when weak (#151).

The gate lives in ml.predict (show = valid card AND |IC t-stat| >= 2). The
template must render the OOS hit rate / IC / n beside the call and must hide
the directional call (action text, probability bars, "Why" explainer) when
the gate is closed.
"""
import unittest
from types import SimpleNamespace

from flask import render_template

from app import app


def _payload(ml):
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
        drawdown='<div id="drawdown-chart"></div>',
        volume_profile=None,
        options_smile='<div id="options-chart"></div>', gex=None,
        price_target=None, sharpe='<div id="sharpe-chart"></div>',
        seasonality='<div id="seasonality-chart"></div>', insider=None,
        ml_reliability=None,
    )
    return SimpleNamespace(
        ticker='MU', current_price=762.35, stats=stats, charts=charts,
        fundamentals={'Name': 'Example Semiconductor Company'}, checklist=None,
        ai_sections=None, ml=ml, gex=None, ai_report=None,
        insider_summary=None, institutional=None, earnings=None,
    )


def _ml(show, card_status='ok'):
    return SimpleNamespace(
        action='Buy', confidence=0.61, calibrated=True,
        proba={'buy': 0.61, 'hold': 0.24, 'sell': 0.15},
        horizon_days=10, show=show, card_status=card_status,
        validated=card_status == 'ok', trained_at='2026-10-20T14:00:00+00:00',
        oos={'n': 853, 'hit_rate': 0.3716, 'ic': 0.0312, 'ic_tstat': 2.71,
             'turnover': 0.31, 'brier': 0.61, 'log_loss': 1.02,
             'confusion_matrix': [[10, 20, 30], [10, 200, 100], [30, 100, 353]],
             'precision': {'Buy': 0.46, 'Hold': 0.63, 'Sell': 0.21}},
        drivers=None, rationale='The model leans Buy at 61% calibrated probability.',
    )


def _render(ml):
    with app.test_request_context('/analytics?ticker=MU'):
        return render_template('analytics.html', data=_payload(ml), ticker='MU', excel_mode=True)


class TestMlEvidencePanel(unittest.TestCase):
    def test_weak_evidence_hides_the_call_but_shows_oos_stats(self):
        html = _render(_ml(show=False, card_status='ok'))
        self.assertIn('Call withheld', html)
        self.assertNotIn('significance gate (ok model card)', html)
        self.assertNotIn('>Buy</span>', html)              # directional call hidden
        self.assertNotIn('calibrated probability', html)   # call line hidden
        self.assertNotIn('The model leans Buy', html)      # "Why" explainer hidden
        # OOS evidence stays visible, with n
        self.assertIn('37.2% hit rate', html)
        self.assertIn('IC 0.0312 (t 2.71)', html)
        self.assertIn('n 853', html)
        self.assertIn('trained 2026-10-20', html)

    def test_below_gate_with_stale_card_names_the_card_status(self):
        html = _render(_ml(show=False, card_status='stale'))
        self.assertIn('significance gate (stale model card)', html)

    def test_strong_evidence_shows_call_calibrated_probability_and_stats(self):
        html = _render(_ml(show=True))
        self.assertNotIn('Call withheld', html)
        self.assertIn('>Buy</span>', html)
        self.assertIn('61% calibrated probability', html)
        self.assertIn('The model leans Buy', html)
        self.assertIn('OOS: 37.2% hit rate', html)
        self.assertIn('n 853', html)

    def test_missing_oos_renders_without_evidence_line(self):
        ml = _ml(show=True)
        ml.oos = None
        html = _render(ml)
        self.assertIn('>Buy</span>', html)
        self.assertNotIn('OOS:', html)


if __name__ == "__main__":
    unittest.main()
