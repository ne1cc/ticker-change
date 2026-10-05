"""Every radar metric has a complete contract; sector map covers GICS set."""
import unittest

import radar_contracts as rc


class TestContracts(unittest.TestCase):
    def test_versions(self):
        self.assertEqual(rc.METRIC_CONTRACT_VERSION, 2)
        self.assertEqual(rc.VALUE_SCORE_VERSION, 1)

    def test_sector_map_has_eleven_gics_sectors(self):
        expected = {"Technology", "Financial Services", "Healthcare", "Consumer Cyclical",
                    "Consumer Defensive", "Energy", "Industrials", "Basic Materials",
                    "Real Estate", "Utilities", "Communication Services"}
        self.assertEqual(set(rc.SECTOR_ETF_MAP), expected)
        for etf in rc.SECTOR_ETF_MAP.values():
            self.assertRegex(etf, r"^[A-Z]{2,4}$")
        self.assertEqual(rc.SECTOR_ETFS, sorted(set(rc.SECTOR_ETF_MAP.values())))

    def test_every_tab_metric_has_complete_contract(self):
        for tab, keys in rc.METRIC_KEYS_BY_TAB.items():
            for key in keys:
                c = rc.contract(key)  # raises KeyError if missing
                for field in ("term", "definition", "unit", "window", "source", "missing_state"):
                    self.assertTrue(str(c[field]).strip(), f"{key}.{field} empty")

    def test_unknown_key_raises(self):
        with self.assertRaises(KeyError):
            rc.contract("not_a_metric")


if __name__ == "__main__":
    unittest.main()
