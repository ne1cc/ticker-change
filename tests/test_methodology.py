"""Unit tests for Methodology & Limitations page and enriched glossary fields (Issue #162)."""
import unittest
from app import app
import glossary


class TestMethodologyPageAndGlossary(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    def test_every_glossary_entry_has_required_methodology_fields(self):
        """Verify all metrics define estimator, window, uncertainty, limits, and source."""
        required_fields = ("estimator", "window", "uncertainty", "limits", "source")
        self.assertGreater(len(glossary.GLOSSARY), 70, "GLOSSARY has fewer entries than expected")

        for key, entry in glossary.GLOSSARY.items():
            for field in required_fields:
                self.assertIn(
                    field, entry,
                    f"Metric '{key}' missing required methodology field '{field}'"
                )
                val = entry[field]
                self.assertIsInstance(val, str, f"Metric '{key}' field '{field}' is not a string")
                self.assertGreater(
                    len(val.strip()), 10,
                    f"Metric '{key}' field '{field}' is too short or empty"
                )
                for placeholder in ("TODO", "TBD", "placeholder", "fixme", "xxx"):
                    self.assertNotIn(
                        placeholder, val.lower(),
                        f"Metric '{key}' field '{field}' contains placeholder '{placeholder}'"
                    )

    def test_validation_fails_on_missing_limits_or_source(self):
        """Verify failure test triggers if limits or source is omitted."""
        incomplete_entry = {
            "term": "Incomplete Metric",
            "section": "Risk &amp; Return",
            "short": "A short description",
            "long": "A long description",
            "estimator": "Sample mean",
            "window": "252 days",
            "uncertainty": "Standard error",
            # 'limits' and 'source' deliberately missing
        }
        with self.assertRaises(AssertionError):
            self.assertIn("limits", incomplete_entry)
        with self.assertRaises(AssertionError):
            self.assertIn("source", incomplete_entry)

    def test_methodology_routes_serve_200_and_contain_disclaimers(self):
        """Verify /methodology and /analytics/methodology serve 200 with required disclosures."""
        for path in ("/methodology", "/analytics/methodology"):
            resp = self.client.get(path)
            self.assertEqual(resp.status_code, 200, f"{path} failed to serve 200")
            html = resp.get_data(as_text=True)

            # Check Mandatory Disclaimers
            self.assertIn("Not Investment Advice", html)
            self.assertIn("Delayed Market Data", html)

            # Check core sections are rendered
            self.assertIn("Risk &amp; Return", html)
            self.assertIn("Institutional Analytics", html)
            self.assertIn("Options &amp; Volatility", html)
            self.assertIn("Market Microstructure", html)
            self.assertIn("Fundamentals &amp; Positioning", html)

            # Check deep-link anchor IDs exist for sample metrics
            for key in ("annualised_vol", "var", "gex", "corwin_schultz", "upside_capture", "vrp"):
                self.assertIn(f'id="{key}"', html, f"{path} missing anchor id='{key}'")

            # Check search input exists
            self.assertIn('id="methodology-search"', html)
