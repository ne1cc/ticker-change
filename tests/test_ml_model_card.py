"""Model card: OOS evaluation persisted in the bundle; predict flags staleness (#149).

train() used to print walk-forward results and discard them. The bundle must
carry a card (trained_at, universe, horizon, confusion matrix, per-class
precision, Brier, log-loss, IC, hit rate, turnover) and predict() must flag a
missing or stale card as unvalidated so the UI gate (#151) can hide the call.
"""
import pickle
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd

import ml


def _oof(n=900, seed=11, skill=0.5):
    """Synthetic OOF with a model that is right more often than not."""
    rng = np.random.default_rng(seed)
    y = rng.choice([-1, 0, 1], size=n, p=[0.3, 0.4, 0.3])
    pred = np.where(rng.random(n) < skill, y, rng.choice([-1, 0, 1], size=n))
    proba = np.full((n, 3), 0.2)
    for i in range(n):
        proba[i, {c: j for j, c in enumerate([-1, 0, 1])}[pred[i]]] = 0.6
    return {"y": y, "pred": pred, "proba": proba,
            "ret": rng.normal(0.001, 0.01, n) * (pred == 1)}


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


class _StubModel:
    """Picklable stand-in for the trained classifier."""
    classes_ = [-1, 0, 1]

    def predict(self, X):
        return np.array([1])

    def predict_proba(self, X):
        return np.array([[0.2, 0.3, 0.5]])


def _bundle(card=None):
    return {"model": _StubModel(), "features": ml.FEATURES, "horizon": ml.HORIZON,
            "card": card}


class TestBuildModelCard(unittest.TestCase):
    def test_card_contains_provenance_and_all_oos_metrics(self):
        import json
        oof = _oof()
        card = ml.build_model_card(oof, ["AAPL", "MSFT"], n_samples=1234)

        self.assertEqual(card["universe"], ["AAPL", "MSFT"])
        self.assertEqual(card["horizon"], ml.HORIZON)
        self.assertEqual(card["n_samples"], 1234)
        datetime.fromisoformat(card["trained_at"])  # parseable

        oos = card["oos"]
        for key in ("n", "confusion_matrix", "precision", "brier", "log_loss",
                    "ic", "ic_tstat", "hit_rate", "turnover"):
            self.assertIn(key, oos)
        self.assertEqual(oos["n"], len(oof["y"]))
        self.assertEqual(sum(sum(row) for row in oos["confusion_matrix"]), oos["n"])
        for v in oos["precision"].values():
            self.assertGreaterEqual(v, 0.0)
        self.assertLess(oos["brier"], 0.5)          # skilled predictions -> low Brier
        self.assertLess(oos["log_loss"], 1.0)
        self.assertGreater(oos["hit_rate"], 0.45)   # skill=0.5 on top of chance
        self.assertGreater(oos["ic_tstat"], 0)      # edge scores track realised returns
        self.assertTrue(0.0 <= oos["turnover"] <= 1.0)
        json.dumps(card)  # JSON-safe (goes into api_cache and the JSON page twin)

    def test_degenerate_oof_gives_none_ic_instead_of_nan(self):
        oof = _oof()
        oof["ret"] = np.zeros(len(oof["y"]))        # flat returns -> IC undefined
        card = ml.build_model_card(oof, ["X"], n_samples=len(oof["y"]))
        self.assertIsNone(card["oos"]["ic"])
        self.assertIsNone(card["oos"]["ic_tstat"])


class TestCardStatus(unittest.TestCase):
    def test_missing_stale_and_fresh_cards(self):
        self.assertEqual(ml.card_status(_bundle(card=None)), "missing")
        self.assertEqual(ml.card_status({"card": {}}), "missing")
        old = {"trained_at": (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()}
        self.assertEqual(ml.card_status(_bundle(card=old)), "stale")
        fresh = {"trained_at": datetime.now(timezone.utc).isoformat()}
        self.assertEqual(ml.card_status(_bundle(card=fresh)), "ok")
        self.assertEqual(ml.card_status({"card": {"trained_at": "not a date"}}), "stale")


class TestPredictFlagsUnvalidated(unittest.TestCase):
    def tearDown(self):
        # reload restores the real db module and env-derived MODEL_PATH
        import importlib
        importlib.reload(ml)

    def _predict_with(self, card):
        ml.db = SimpleNamespace(
            cache_get=lambda *a: None,
            cache_set=lambda *a: None,
            get_prices=lambda t: _price_frame(),
        )
        path = "/tmp/test_ml_card_%d.pkl" % datetime.now().timestamp()
        with open(path, "wb") as f:
            pickle.dump(_bundle(card), f)
        ml.MODEL_PATH = path
        return ml.predict("AAPL")

    def test_missing_card_flags_unvalidated(self):
        result = self._predict_with(card=None)
        self.assertEqual(result["card_status"], "missing")
        self.assertFalse(result["validated"])
        self.assertIsNone(result["oos"])

    def test_stale_card_flags_unvalidated(self):
        stale = {"trained_at": (datetime.now(timezone.utc) - timedelta(days=40)).isoformat(),
                 "oos": ml.build_model_card(_oof(), ["AAPL"], 100)["oos"]}
        result = self._predict_with(card=stale)
        self.assertEqual(result["card_status"], "stale")
        self.assertFalse(result["validated"])
        self.assertEqual(result["oos"]["n"], len(_oof()["y"]))

    def test_fresh_card_is_validated_and_carries_oos(self):
        fresh = ml.build_model_card(_oof(), ["AAPL", "MSFT"], 900)
        result = self._predict_with(card=fresh)
        self.assertEqual(result["card_status"], "ok")
        self.assertTrue(result["validated"])
        self.assertEqual(result["oos"]["n"], 900)
        self.assertEqual(result["trained_at"], fresh["trained_at"])


if __name__ == "__main__":
    unittest.main()
