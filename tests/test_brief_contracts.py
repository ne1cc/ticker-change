"""Every brief metric has a complete contract; sources map to receipts."""
import unittest

import brief_contracts as bc


class TestBriefContracts(unittest.TestCase):
    def test_version(self):
        self.assertEqual(bc.BRIEF_CONTRACT_VERSION, 1)

    def test_every_section_metric_has_complete_contract(self):
        for section, keys in bc.METRIC_KEYS_BY_SECTION.items():
            for key in keys:
                c = bc.contract(key)  # raises KeyError if missing
                for field in ("term", "definition", "unit", "window",
                              "source", "missing_state"):
                    self.assertTrue(str(c[field]).strip(),
                                    f"{key}.{field} empty")

    def test_section_sources_map_to_receipt_keys(self):
        for section, keys in bc.METRIC_KEYS_BY_SECTION.items():
            for key in keys:
                self.assertIn(bc.contract(key)["source"], bc.SOURCE_KEYS,
                              f"{section}/{key}")

    def test_unknown_key_raises(self):
        with self.assertRaises(KeyError):
            bc.contract("not_a_metric")

    def test_sections_are_exactly_the_built_ones(self):
        self.assertEqual(set(bc.METRIC_KEYS_BY_SECTION),
                         {"price", "earnings", "context"})


if __name__ == "__main__":
    unittest.main()
