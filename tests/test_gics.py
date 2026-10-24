"""Tests for GICS sector and sub-industry taxonomy mapping (ADR 0002)."""
import os
import unittest
from unittest.mock import patch, MagicMock

import gics
import db


class TestGicsTaxonomy(unittest.TestCase):
    def setUp(self):
        self._orig_db = db.DB_PATH

    def tearDown(self):
        db.DB_PATH = self._orig_db

    def test_adr_recorded_with_licensing_note(self):
        adr_path = os.path.join(
            os.path.dirname(__file__), "..", "docs", "adr", "0002-gics-sector-mapping-source.md"
        )
        self.assertTrue(os.path.exists(adr_path), f"ADR not found at {adr_path}")
        with open(adr_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("ADR 0002", content)
        self.assertIn("PDDL", content)
        self.assertIn("CC0", content)
        self.assertIn("frictionlessdata", content)

    def test_fixture_constituents_map_to_gics_sector_and_sub_industry(self):
        # Using the fixture
        mapping = gics.fetch_gics_mapping(force_refresh=True)
        self.assertGreater(len(mapping), 15)

        # Key tickers
        self.assertEqual(gics.sector_of("AAPL"), "Information Technology")
        sub_ind = gics.sub_industry_of("AAPL")
        self.assertTrue("Technology Hardware" in sub_ind and "Peripherals" in sub_ind)

        self.assertEqual(gics.sector_of("XOM"), "Energy")
        self.assertEqual(gics.sub_industry_of("XOM"), "Integrated Oil & Gas")

        self.assertEqual(gics.sector_of("JPM"), "Financials")
        self.assertEqual(gics.sub_industry_of("JPM"), "Diversified Banks")

        # Symbol normalization with dot notation
        self.assertEqual(gics.sector_of("BRK.B"), "Financials")
        self.assertEqual(gics.sector_of("BRK-B"), "Financials")

        # Lowercase normalization
        self.assertEqual(gics.sector_of("msft"), "Information Technology")

    def test_unmapped_ticker_returns_none(self):
        self.assertIsNone(gics.sector_of("UNKNOWN_TICKER_XYZ"))
        self.assertIsNone(gics.sub_industry_of("UNKNOWN_TICKER_XYZ"))
        self.assertIsNone(gics.sector_of(""))
        self.assertIsNone(gics.sector_of(None))

    def test_all_sectors_and_constituents_by_sector(self):
        sectors = gics.all_sectors()
        self.assertIn("Information Technology", sectors)
        self.assertIn("Energy", sectors)
        self.assertIn("Financials", sectors)

        tech = gics.constituents_by_sector("Information Technology")
        self.assertIn("AAPL", tech)
        self.assertIn("MSFT", tech)
        self.assertIn("NVDA", tech)
        self.assertNotIn("XOM", tech)

    @patch("requests.get")
    def test_fetch_caches_in_sqlite(self, mock_get):
        csv_sample = (
            "Symbol,Security,GICS Sector,GICS Sub-Industry\n"
            + "\n".join(f"T{i},Co {i},Tech,Software" for i in range(450))
        )
        mock_resp = MagicMock()
        mock_resp.text = csv_sample
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        mapping = gics.fetch_gics_mapping(force_refresh=True)
        self.assertEqual(len(mapping), 450)
        self.assertEqual(mapping["T0"]["sector"], "Tech")

        # Second call without force_refresh should hit cache, not requests.get
        mock_get.reset_mock()
        cached = gics.fetch_gics_mapping(force_refresh=False)
        self.assertEqual(len(cached), 450)
        mock_get.assert_not_called()
