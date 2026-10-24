"""Purged k-fold with embargo must remove the overlap-label leak (#148).

Triple-barrier labels overlap their HORIZON-day window, so in a plain date
k-fold a training row adjacent to the test block carries nearly the same
label (its window shares HORIZON-1 days). A model that can read training
labels therefore scores inflated OOS accuracy. After purging + embargo the
nearest training window is disjoint from every test window and the leak is
gone.
"""
import unittest

import numpy as np
import pandas as pd

import ml


def _overlapping_label_fixture(n_days=1200, seed=7):
    """Prices + heavily-overlapping labels: label[i] = sign of mean return
    over the next HORIZON days (adjacent labels share HORIZON-1 days)."""
    rng = np.random.default_rng(seed)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, n_days + ml.HORIZON + 1)))
    fwd = close.shift(-1).rolling(ml.HORIZON).mean().shift(-ml.HORIZON)
    ret = fwd / close - 1
    y = pd.Series(np.sign(ret), index=pd.RangeIndex(n_days))
    dates = pd.Series(pd.date_range("2024-01-01", periods=n_days, freq="B"))
    return y, dates


def _plain_kfold(dates, n_splits=5):
    """Same date-block split as purged_kfold but with NO purge and NO embargo."""
    unique_dates = pd.Series(sorted(dates.unique()))
    pos = dates.map({d: i for i, d in enumerate(unique_dates)}).to_numpy()
    edges = np.linspace(0, len(unique_dates), n_splits + 1).astype(int)
    return [(np.where(~((pos >= edges[k]) & (pos < edges[k + 1])))[0],
             np.where((pos >= edges[k]) & (pos < edges[k + 1]))[0])
            for k in range(n_splits)]


def _edge_rows(plain_splits, horizon):
    """Test rows within `horizon` of a block edge — where the overlap leak lives."""
    rows = []
    for _, test_idx in plain_splits:
        lo, hi = test_idx.min(), test_idx.max()
        rows.append(test_idx[(test_idx - lo < horizon) | (hi - test_idx < horizon)])
    return rows


def _nearest_label_accuracy_on(splits, y, rows_by_fold):
    """1-NN over the time axis on the given rows: predict each test row with
    the label of the temporally nearest training row — pure label leakage."""
    correct = total = 0
    for (train_idx, _), rows in zip(splits, rows_by_fold):
        tr = np.sort(train_idx)
        for i in rows:
            near = tr[np.argmin(np.abs(tr - i))]
            if not (np.isfinite(y.iloc[i]) and np.isfinite(y.iloc[near])):
                continue
            correct += int(y.iloc[near] == y.iloc[i])
            total += 1
    return correct / total


class TestPurgedKfold(unittest.TestCase):
    def setUp(self):
        self.y, self.dates = _overlapping_label_fixture()

    def test_plain_kfold_leaks_overlapping_labels_and_purging_removes_it(self):
        plain = _plain_kfold(self.dates)
        purged = ml.purged_kfold(self.dates)

        # Structural: plain folds contain training rows whose HORIZON-day label
        # window overlaps the test block; purged folds contain none.
        for train_idx, test_idx in plain:
            lo, hi = test_idx.min(), test_idx.max()
            self.assertTrue(np.any((train_idx >= lo - ml.HORIZON) & (train_idx <= hi + ml.HORIZON)))
        for train_idx, test_idx in purged:
            lo, hi = test_idx.min(), test_idx.max()
            self.assertFalse(np.any((train_idx >= lo - ml.HORIZON) & (train_idx <= hi + ml.HORIZON)))
            self.assertFalse(np.any((train_idx >= lo) & (train_idx <= hi)))

        # Scored at the fold boundary, where the leak lives: reading the label
        # of the nearest training row is predictive in plain CV (windows
        # overlap) and collapses to chance once purged (windows disjoint).
        edge = _edge_rows(plain, ml.HORIZON)
        acc_plain = _nearest_label_accuracy_on(plain, self.y, edge)
        acc_purged = _nearest_label_accuracy_on(purged, self.y, edge)
        self.assertGreater(acc_plain, 0.7)
        self.assertLess(acc_purged, 0.62)
        self.assertGreater(acc_plain - acc_purged, 0.15)

    def test_embargo_and_purge_derive_from_horizon(self):
        ml.HORIZON = 5
        try:
            train_idx, test_idx = ml.purged_kfold(self.dates, n_splits=5)[2]
            lo, hi = test_idx.min(), test_idx.max()
            # Purge band before the block and purge+embargo band after both
            # widen/shrink with HORIZON (embargo defaults to HORIZON).
            self.assertLess(train_idx[train_idx < lo].max(), lo - 5)
            after = train_idx[train_idx > hi]
            self.assertGreater(after.min(), hi + 5)
            self.assertLessEqual(after.min(), hi + 2 * 5 + 1)
        finally:
            ml.HORIZON = 10

    def test_explicit_embargo_overrides_the_default(self):
        wide = ml.purged_kfold(self.dates, n_splits=5, embargo=30)
        narrow = ml.purged_kfold(self.dates, n_splits=5, embargo=0)
        train_wide, test_wide = wide[0]
        train_narrow, _ = narrow[0]
        hi = test_wide.max()
        self.assertGreater(train_wide[train_wide > hi].min(),
                           train_narrow[train_narrow > hi].min())


class TestPurgedOof(unittest.TestCase):
    def test_oof_covers_the_sample_and_is_class_aligned(self):
        rng = np.random.default_rng(3)
        n = 600
        X = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"))
        y = pd.Series(rng.choice([-1, 0, 1], size=n, p=[0.3, 0.4, 0.3]))
        dates = pd.Series(pd.date_range("2024-01-01", periods=n, freq="B"))
        returns = pd.Series(rng.normal(0, 0.01, n))
        oof = ml.purged_oof(X, y, dates, returns, n_splits=4)
        self.assertIsNotNone(oof)
        self.assertEqual(oof["proba"].shape[1], 3)
        self.assertTrue(np.isfinite(oof["proba"]).all())
        np.testing.assert_allclose(oof["proba"].sum(axis=1), 1.0, atol=1e-6)
        self.assertEqual(len(oof["y"]), len(oof["pred"]))
        self.assertTrue(np.isfinite(oof["ret"]).all())


if __name__ == "__main__":
    unittest.main()
