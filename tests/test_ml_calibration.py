"""Probability calibration: per-class Platt on OOF, ECE before/after (#150).

Raw 'confidence' was an uncalibrated max class probability. A calibrator fit
on the purged OOF predictions must measurably reduce calibration error on a
held-out set, and predict() must serve calibrated probabilities.
"""
import pickle
import unittest
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd

import ml


class _ShiftCal:
    """Picklable 1-D Platt stand-in: P(class) = raw^0.8 (non-uniform distort)."""

    def predict_proba(self, X):
        x = np.clip(np.asarray(X, dtype=float).ravel(), 0, 1) ** 0.8
        return np.column_stack([1 - x, x])


def _miscalibrated_oof(n=6000, seed=9):
    """OOF whose raw probabilities are systematically distorted (p^1.5):
    predicted probabilities are too low in the mid-high range -> real ECE.
    Only classes -1 and 1 occur, so class 0 has a single-valued label and its
    per-class fit must be skipped (raw fallback)."""
    rng = np.random.default_rng(seed)
    half = n // 2
    true_p = rng.beta(2, 2, n)                      # well spread over (0, 1)
    y = np.where(rng.random(n) < true_p, 1.0, -1.0)
    raw = true_p ** 1.5
    # 3-column layout in CLASS_ORDER [-1, 0, 1]; the two live classes mirror.
    proba = np.column_stack([1 - raw, np.zeros(n), raw])
    return proba, y, half


class TestCalibrationReducesError(unittest.TestCase):
    def test_platt_reduces_ece_on_held_out_set(self):
        proba, y, half = _miscalibrated_oof()
        oof_fit = {"proba": proba[:half], "y": y[:half]}
        calibrator = ml.fit_calibrator(oof_fit)
        self.assertIn(1, calibrator)
        self.assertNotIn(0, calibrator)  # single-valued label -> no fit -> raw fallback

        held = slice(half, None)
        ece_before = ml.expected_calibration_error(proba[held], y[held])
        ece_after = ml.expected_calibration_error(
            ml.apply_calibrator(proba[held], calibrator), y[held])
        self.assertGreater(ece_before - ece_after, 0.02)
        self.assertLess(ece_after, ece_before)

    def test_calibrated_rows_sum_to_one(self):
        proba, y, half = _miscalibrated_oof()
        calibrator = ml.fit_calibrator({"proba": proba[:half], "y": y[:half]})
        out = ml.apply_calibrator(proba, calibrator)
        self.assertEqual(out.shape, proba.shape)
        np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-9)


class TestReliabilityBins(unittest.TestCase):
    def test_bins_track_the_diagonal_for_well_calibrated_input(self):
        rng = np.random.default_rng(2)
        n = 4000
        p = rng.beta(2, 2, n)
        y = np.where(rng.random(n) < p, 1.0, -1.0)
        proba = np.column_stack([1 - p, np.zeros(n), p])
        curve = ml.reliability_bins(proba, y)
        self.assertEqual([c["class"] for c in curve], [-1, 0, 1])
        live = curve[2]["points"]
        self.assertGreaterEqual(len(live), 8)
        for pt in live:
            self.assertGreater(pt["n"], 0)
            self.assertLess(abs(pt["x"] - pt["y"]), 0.1)  # calibrated -> near diagonal


class TestPredictAppliesCalibrator(unittest.TestCase):
    def tearDown(self):
        import importlib
        importlib.reload(ml)

    def test_confidence_comes_from_calibrated_probabilities(self):
        ml.db = SimpleNamespace(
            cache_get=lambda *a: None,
            cache_set=lambda *a: None,
            get_prices=lambda t: _price_frame(),
        )
        path = "/tmp/test_ml_cal_%d.pkl" % datetime.now().timestamp()
        bundle = {"model": _StubModel(), "features": ml.FEATURES,
                  "horizon": ml.HORIZON, "card": None,
                  "calibrator": {c: _ShiftCal() for c in (-1, 0, 1)}}
        with open(path, "wb") as f:
            pickle.dump(bundle, f)
        ml.MODEL_PATH = path

        result = ml.predict("AAPL")
        self.assertTrue(result["calibrated"])
        # raw stub probs are (0.2, 0.3, 0.5); the calibrator shrinks the max
        # class prob to 0.8*0.5 before renormalising
        self.assertAlmostEqual(sum(result["proba"].values()), 1.0, places=6)
        self.assertLess(result["proba"]["buy"], 0.5)
        self.assertAlmostEqual(result["confidence"], result["proba"]["buy"], places=6)


# -- shared fixture shims (duplicated from test_ml_model_card to stay standalone)

class _StubModel:
    classes_ = [-1, 0, 1]

    def predict(self, X):
        return np.array([1])

    def predict_proba(self, X):
        return np.array([[0.2, 0.3, 0.5]])


def _price_frame(n=260):
    rng = np.random.default_rng(5)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n)))
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "open": close * (1 + rng.normal(0, 0.002, n)),
        "high": close * (1 + np.abs(rng.normal(0, 0.004, n))),
        "low": close * (1 - np.abs(rng.normal(0, 0.004, n))),
        "close": close,
        "volume": np.abs(rng.normal(1e6, 1e5, n)),
    }, index=idx)


if __name__ == "__main__":
    unittest.main()
