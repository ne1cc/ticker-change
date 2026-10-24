"""Real SI plumbed into microstructure squeeze inputs, flagged honestly."""
import os
import tempfile
import unittest
from unittest.mock import patch

import db
import pandas as pd

_fd, _DB_PATH = tempfile.mkstemp(suffix=".db")
os.close(_fd)
db.DB_PATH = _DB_PATH
db.init_db()

import app as app_module  # noqa: E402

SI = {"short_pct_float": 0.18, "days_to_cover": 6.0, "si_mom_change": 0.1,
      "shares_short": 1, "shares_short_prior_month": 1, "as_of": 1,
      "source": "reported"}


class TestShortInterestWiring(unittest.TestCase):
    def test_reported_when_payload_complete(self):
        with patch.object(app_module.providers, "get_short_interest", return_value=dict(SI)):
            payload, source = app_module._short_interest_for("AAPL")
        self.assertEqual(source, "reported")
        self.assertEqual(payload["short_pct_float"], 0.18)

    def test_assumed_when_missing(self):
        with patch.object(app_module.providers, "get_short_interest", return_value=None):
            payload, source = app_module._short_interest_for("AAPL")
        self.assertEqual(source, "assumed")
        self.assertIsNone(payload)

    def test_assumed_when_fields_incomplete(self):
        partial = {"short_pct_float": None, "days_to_cover": None}
        with patch.object(app_module.providers, "get_short_interest", return_value=partial):
            payload, source = app_module._short_interest_for("AAPL")
        self.assertEqual(source, "assumed")

    def test_micro_call_receives_real_kwargs(self):
        captured = {}
        real = app_module.microstructure.get_microstructure_analytics

        def spy(df, **kwargs):
            captured.update(kwargs)
            return real(df, **kwargs)

        with patch.object(app_module.providers, "get_short_interest", return_value=dict(SI)), \
             patch.object(app_module.microstructure, "get_microstructure_analytics",
                          side_effect=spy):
            pct, dtc, assumed = app_module._si_inputs("AAPL")
            app_module.microstructure.get_microstructure_analytics(
                pd.DataFrame(), short_pct_float=pct, days_to_cover=dtc,
                assumed_si_inputs=assumed)
        self.assertEqual(captured["short_pct_float"], 18.0)
        self.assertEqual(captured["days_to_cover"], 6.0)
        self.assertEqual(captured["assumed_si_inputs"], [])

    def test_zero_valued_payload_is_reported_and_preserved(self):
        zeros = {"short_pct_float": 0.0, "days_to_cover": 0.0}
        with patch.object(app_module.providers, "get_short_interest", return_value=dict(zeros)):
            payload, source = app_module._short_interest_for("AAPL")
            inputs = app_module._si_inputs("AAPL")
        self.assertEqual(source, "reported")
        self.assertEqual(inputs, (0.0, 0.0, []))

    def test_assumed_inputs_are_listed_not_hidden(self):
        with patch.object(app_module.providers, "get_short_interest", return_value=None):
            inputs = app_module._si_inputs("AAPL")
        self.assertEqual(inputs, (3.0, 2.0, ["SI % float", "DTC"]))

    def test_partially_reported_payload_lists_only_the_gaps(self):
        partial = {"short_pct_float": 0.12, "days_to_cover": None}
        with patch.object(app_module.providers, "get_short_interest", return_value=partial):
            inputs = app_module._si_inputs("AAPL")
        self.assertEqual(inputs, (12.0, 2.0, ["DTC"]))
